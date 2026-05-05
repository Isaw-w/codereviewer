"""
Build the embedding cache required by find_unique_criteria.py.

Reads rubric files from --rubric_dir (same layout expected by find_unique_criteria.py),
normalizes criterion text, then calls the OpenAI embeddings API to embed every unique
normalized text. Outputs two files to --output_dir:

  texts_cache.json      JSON list of normalized text strings, in registry order
  embeddings_cache.npy  float32 array of shape (n_texts, embedding_dim), L2-normalized

Both files are required inputs for find_unique_criteria.py (--embed_cache flag).

The normalization function strips opening verbs and leading adverbs so that embeddings
focus on propositional content rather than presentation style. The paper used
text-embedding-3-large via the OpenAI API.

Usage:
    python3 build_embedding_cache.py \\
        --rubric_dir /path/to/rubrics \\
        --output_dir /path/to/embed_cache \\
        --ak $OPENAI_API_KEY

    # Or via OpenRouter:
    python3 build_embedding_cache.py \\
        --rubric_dir /path/to/rubrics \\
        --output_dir /path/to/embed_cache \\
        --ak $OPENROUTER_API_KEY \\
        --base_url https://openrouter.ai/api/v1

The rubric_dir must contain one subdirectory per model, each holding:
  human_rubric_*.jsonl   (one file)
  ai_rubric_*.jsonl      (one file)

Each JSONL row: {"TASK_ID": "case_001", "RUBRIC": [...]}
Each criterion in RUBRIC: {"id": "crit_001", "title": "criterion text", ...}
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

MODELS = [
    "claude_sonnet4_openrouter",
    "deepseek_r1_0528_openrouter",
    "deepseekv32exp_openrouter",
    "gemini25_pro_openrouter",
    "gemini31_openrouter",
    "gemini3_flash_openrouter",
    "gpt54_openrouter",
    "kimi_k2_5_openrouter",
    "mimo_v2_pro_openrouter",
    "opus46_openrouter",
    "qwen35_397b_a17b_openrouter",
]

_NEGATION_RE = re.compile(
    r"^(does not|do not|don'?t|doesn'?t|avoids?|never|fails?|ignores?|omits?|overlooks?|dismisses?|penaliz"
    r"|guilt-|over-center|over-focus|over-empha"
    r"|assumes?\b|blames?\b|downplays?\b|endorses?\b|minimizes?\b|mocks?\b|shames?\b|trivializes?\b"
    r"|validates?\b|encourages?\b)",
    re.I,
)
_ADVERBS_RE = re.compile(
    r"^(briefly|clearly|explicitly|carefully|accurately|thoroughly|directly|fully|properly|appropriately)\s+",
    re.I,
)


def normalize_for_embed(title: str) -> str:
    """Strip opening verb/adverb so embeddings focus on propositional content."""
    title = title.strip().strip("'\"").strip()
    if not title:
        return title
    if re.match(r"^(the )?response\b", title, re.I):
        return title
    if re.match(r"^advice\s*,", title, re.I):
        return title
    if re.match(r"^suggests?\s*,.*\bfails?\b", title, re.I):
        return title
    if _NEGATION_RE.match(title):
        return title
    if re.match(r"^detrimental\s*:", title, re.I):
        return title
    first_token = re.match(r"^(\S+)", title)
    if first_token and re.search(r"[-/]", first_token.group(1)):
        m_hyph = re.match(r"^\S+(?:\s*,\s*\w+)*(?:\s+or\s+\w+)?\s+(.+)$", title, re.DOTALL)
        if m_hyph:
            return f"Does the response consider {m_hyph.group(1)}"
    stripped = _ADVERBS_RE.sub("", title)
    m_comma = re.match(r"^(\w+),\s*[^,]+,\s*(.+)$", stripped, re.DOTALL)
    if m_comma:
        return f"Does the response consider {m_comma.group(2)}"
    m = re.match(r"^\w+\s+(.+)$", stripped, re.DOTALL)
    if m:
        return f"Does the response consider {m.group(1)}"
    return title


def _parse_rubric(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise ValueError(f"Unsupported RUBRIC type: {type(raw).__name__}")


def load_rubric_file(path: Path) -> dict[str, list[dict]]:
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            result[row["TASK_ID"]] = _parse_rubric(row["RUBRIC"])
    return result


def load_all_rubrics(model: str, rubric_dir: Path):
    model_dir = rubric_dir / model
    human_files = sorted(model_dir.glob("human_rubric_*.jsonl"))
    ai_files = sorted(model_dir.glob("ai_rubric_*.jsonl"))
    if len(human_files) != 1 or len(ai_files) != 1:
        raise FileNotFoundError(
            f"Expected 1 human + 1 ai rubric for {model}, "
            f"found {len(human_files)} human, {len(ai_files)} ai"
        )
    return load_rubric_file(human_files[0]), load_rubric_file(ai_files[0])


def build_text_list(rubric_dir: Path) -> list[str]:
    """
    Reconstruct the ordered list of normalized texts that find_unique_criteria.py uses.
    Returns texts in registry insertion order (human side first, then each model).
    """
    first_human, _ = load_all_rubrics(MODELS[0], rubric_dir)
    task_ids = sorted(first_human.keys())

    registry: dict[str, dict] = {}

    def ensure(text: str):
        norm = normalize_for_embed(text)
        if norm not in registry:
            registry[norm] = {}
        return norm

    for task_id in task_ids:
        for c in first_human[task_id]:
            ensure(c["title"])

    for model in MODELS:
        _, ai = load_all_rubrics(model, rubric_dir)
        for task_id in task_ids:
            for c in ai[task_id]:
                ensure(c["title"])

    return list(registry.keys())


def embed_texts(texts: list[str], client, model: str, batch_size: int = 100,
                max_retries: int = 5) -> np.ndarray:
    """Embed all texts in batches, returning an L2-normalized (n, dim) float32 array."""
    all_vecs = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start:start + batch_size]
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                vecs = np.array([d.embedding for d in sorted(resp.data, key=lambda x: x.index)],
                                dtype=np.float32)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                all_vecs.append(vecs / norms)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"  batch {start}–{start + len(batch)}: retry {attempt + 1} ({e}), sleeping {wait}s",
                          flush=True)
                    time.sleep(wait)
                else:
                    raise
        done = min(start + batch_size, n)
        if done % 10000 == 0 or done == n:
            print(f"  embedded {done}/{n}", flush=True)
    return np.concatenate(all_vecs, axis=0)


def main():
    parser = argparse.ArgumentParser(
        description="Build embedding cache (texts_cache.json + embeddings_cache.npy) for find_unique_criteria.py."
    )
    parser.add_argument("--rubric_dir", required=True,
                        help="Directory with per-model rubric subdirs (same as --rubric_dir for find_unique_criteria.py)")
    parser.add_argument("--output_dir", "-o", required=True,
                        help="Output directory for texts_cache.json and embeddings_cache.npy")
    parser.add_argument("--ak", required=True,
                        help="API key (OpenAI or OpenRouter)")
    parser.add_argument("--base_url", default=None,
                        help="API base URL (omit for OpenAI; set to https://openrouter.ai/api/v1 for OpenRouter)")
    parser.add_argument("--embedding_model", default="text-embedding-3-large",
                        help="Embedding model identifier (default: text-embedding-3-large)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Texts per embedding API call (default: 100)")
    args = parser.parse_args()

    from openai import OpenAI
    client_kwargs = {"api_key": args.ak}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts_path = out_dir / "texts_cache.json"
    emb_path = out_dir / "embeddings_cache.npy"

    if texts_path.exists() and emb_path.exists():
        print(f"Cache already exists at {out_dir}. Delete to rebuild.", flush=True)
        return

    print(f"Building text registry from {args.rubric_dir} ...", flush=True)
    texts = build_text_list(Path(args.rubric_dir))
    print(f"  unique normalized texts: {len(texts)}", flush=True)

    texts_path.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote texts_cache.json", flush=True)

    print(f"Embedding {len(texts)} texts with {args.embedding_model} ...", flush=True)
    vecs = embed_texts(texts, client, args.embedding_model, batch_size=args.batch_size)
    print(f"  embedding shape: {tuple(vecs.shape)}", flush=True)

    np.save(emb_path, vecs)
    print(f"  wrote embeddings_cache.npy", flush=True)
    print(f"\nDone. Pass --embed_cache {out_dir} to find_unique_criteria.py.", flush=True)


if __name__ == "__main__":
    main()
