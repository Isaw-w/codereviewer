#!/usr/bin/env python3
"""
Embed all 500-case human rubric criteria and per-model AI rubric criteria,
then for each (model, task_id) pair identify:
  - model-unique criteria: max cosine to any human criterion < threshold
  - human-unique criteria: max cosine to any model criterion < threshold

Output: one JSONL per model under --output_dir/<model>/unique_criteria.jsonl
Summary: --output_dir/summary.json

Usage:
    python3 build_human_model_unique_criteria.py \
        -ak "$LAB_OPENROUTER_KEY" \
        --output_dir data/paper_release/finding3/coverage/human_model_unique_t70_all \
        --threshold 0.75 \
        --n 5   # test with 5 cases first
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import re

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = RELEASE_ROOT / "data" / "canonical_full" / "rubrics"
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
EMBED_MODEL = "openai/text-embedding-3-large"
API_URL = "https://openrouter.ai/api/v1/embeddings"
BATCH_SIZE = 100
MAX_RETRIES = 5


def resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    release_candidate = RELEASE_ROOT / p
    repo_candidate = REPO_ROOT / p
    if release_candidate.exists():
        return release_candidate
    if repo_candidate.exists():
        return repo_candidate
    if p.parts and p.parts[0] in {"code", "data", "docs", "manifests"}:
        return release_candidate
    return repo_candidate


# ─── Text normalization ───────────────────────────────────────────────────────

_NEGATION_RE = re.compile(
    r"^(does not|do not|don'?t|doesn'?t|avoids?|never|fails?|ignores?|omits?|overlooks?|dismisses?|penaliz"
    r"|guilt-|over-center|over-focus|over-empha"
    r"|assumes?\b|blames?\b|downplays?\b|endorses?\b|minimizes?\b|mocks?\b|shames?\b|trivializes?\b"
    r"|validates?\b|encourages?\b)",
    re.I,
)

_ADVERBS_RE = re.compile(
    r'^(briefly|clearly|explicitly|carefully|accurately|thoroughly|directly|fully|properly|appropriately)\s+',
    re.I,
)

def normalize_for_embed(title: str) -> str:
    """
    Strip the opening verb (and any leading adverb) and reframe as
    'Does the response consider ...' so that embedding comparison focuses
    on propositional content, not verb framing.
    Negation-framed and special-form criteria are left unchanged.
    Original data is never modified — this is only used for the embed text.
    """
    title = title.strip().strip("'\"").strip()
    if not title:
        return title
    # "The response..." / "Response, if..." — already in a good form, keep as-is
    if re.match(r'^(the )?response\b', title, re.I):
        return title
    # "Advice, if given, does not..." — negation buried in clause, keep as-is
    if re.match(r'^advice\s*,', title, re.I):
        return title
    # "Suggests, endorses, or fails to warn against..." — negative coordinated verb list
    if re.match(r'^suggests?\s*,.*\bfails?\b', title, re.I):
        return title
    if _NEGATION_RE.match(title):
        return title
    # "Detrimental: Verb rest" — negative/penalty criterion, keep as-is
    if re.match(r'^detrimental\s*:', title, re.I):
        return title
    # "Guilt-trips X", "Over-centers X", "Re-frames X", "Calculates/compars X"
    # Only applies when the FIRST token actually contains - or /
    first_token = re.match(r'^(\S+)', title)
    if first_token and re.search(r'[-/]', first_token.group(1)):
        # Strip first hyphenated/slash token + any coordinated verbs (", shames, or coerces")
        m_hyph = re.match(r'^\S+(?:\s*,\s*\w+)*(?:\s+or\s+\w+)?\s+(.+)$', title, re.DOTALL)
        if m_hyph:
            return f"Does the response consider {m_hyph.group(1)}"
    # Strip leading adverb if present (e.g. "Briefly summarizes" -> "summarizes")
    stripped = _ADVERBS_RE.sub('', title)
    # Handle "Verb, parenthetical clause, rest" -> strip up to second comma
    # e.g. "Considers, when coming to a decision, the consequences..."
    #   -> "Does the response consider the consequences..."
    m_comma = re.match(r'^(\w+),\s*[^,]+,\s*(.+)$', stripped, re.DOTALL)
    if m_comma:
        return f"Does the response consider {m_comma.group(2)}"
    # Normal case: strip opening verb
    m = re.match(r'^\w+\s+(.+)$', stripped, re.DOTALL)
    if m:
        return f"Does the response consider {m.group(1)}"
    return title


# ─── I/O helpers ─────────────────────────────────────────────────────────────

def _parse_rubric(raw):
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise ValueError(f"Unsupported RUBRIC type: {type(raw).__name__}")


def load_rubric_file(path: Path) -> dict[str, list[dict]]:
    """Return {task_id: [criterion, ...]} from a rubric jsonl."""
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            result[row["TASK_ID"]] = _parse_rubric(row["RUBRIC"])
    return result


def load_all_rubrics(model: str):
    model_dir = CANONICAL_DIR / model
    human_files = sorted(model_dir.glob("human_rubric_*.jsonl"))
    ai_files = sorted(model_dir.glob("ai_rubric_*.jsonl"))
    if len(human_files) != 1 or len(ai_files) != 1:
        raise FileNotFoundError(
            f"Expected 1 human + 1 ai rubric for {model}, "
            f"found {len(human_files)} human, {len(ai_files)} ai"
        )
    return load_rubric_file(human_files[0]), load_rubric_file(ai_files[0])


# ─── Embedding ────────────────────────────────────────────────────────────────

def embed_batch(texts: list[str], api_key: str, retries: int = MAX_RETRIES) -> list[list[float]]:
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(retries):
        try:
            req = urllib.request.Request(API_URL, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            items = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in items]
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [embed retry {attempt+1}/{retries}] {e}, waiting {wait}s", flush=True)
            time.sleep(wait)


def embed_all_texts(
    texts: list[str], api_key: str, workers: int = 8
) -> np.ndarray:
    """Embed all texts in batches, return normalized (N, D) float32 array."""
    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    results: dict[int, list[list[float]]] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(embed_batch, b, api_key): i for i, b in enumerate(batches)}
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            exc = future.exception()
            if exc:
                raise RuntimeError(f"Embedding batch {idx} failed: {exc}") from exc
            results[idx] = future.result()
            done += 1
            if done % 5 == 0 or done == len(batches):
                print(f"  embedded {done}/{len(batches)} batches", flush=True)

    all_vecs = []
    for i in range(len(batches)):
        all_vecs.extend(results[i])
    arr = np.array(all_vecs, dtype=np.float32)
    # normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


# ─── Analysis ────────────────────────────────────────────────────────────────

def pool_deduplicate(
    human_criteria: list[dict],
    human_vecs: np.ndarray,
    model_criteria: list[dict],
    model_vecs: np.ndarray,
    threshold: float,
) -> list[dict]:
    """
    Pool human + model criteria together, then greedily deduplicate.

    Processing order: human first, then model.
    - A new criterion is added as a representative if its cosine to every
      existing representative is < threshold.
    - If it is >= threshold to an existing representative, it is merged into
      that representative (its side is added to the representative's side set).

    Each surviving representative has:
      title, id, weight, representative_side ("human"|"model"),
      side ("human"|"model"|"shared"), max_cos_merged
    """
    all_criteria = (
        [{"_side": "human", **c} for c in human_criteria]
        + [{"_side": "model", **c} for c in model_criteria]
    )
    if not all_criteria:
        return []

    parts = [v for v in [human_vecs, model_vecs] if len(v) > 0]
    all_vecs = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]  # (N, D)

    kept: list[dict] = []
    kept_vecs: list[np.ndarray] = []
    kept_sides: list[set] = []

    for i, crit in enumerate(all_criteria):
        v = all_vecs[i]
        side = crit["_side"]

        if kept_vecs:
            mat = np.stack(kept_vecs)              # (K, D)
            sims = (mat @ v).astype(float)         # (K,)
            best_idx = int(sims.argmax())
            best_cos = float(sims[best_idx])
            if best_cos >= threshold:
                # merge: just record the other side touched this cluster
                kept_sides[best_idx].add(side)
                kept[best_idx]["sides"] = sorted(kept_sides[best_idx])
                kept[best_idx]["max_cos_merged"] = max(
                    kept[best_idx].get("max_cos_merged", 0.0), best_cos
                )
                continue

        # new representative
        row = {
            "title": crit["title"],
            "id": crit["id"],
            "weight": crit.get("weight"),
            "representative_side": side,
            "sides": [side],
            "max_cos_merged": 0.0,
        }
        kept.append(row)
        kept_vecs.append(v)
        kept_sides.append({side})

    # Assign final side label
    for row in kept:
        s = set(row["sides"])
        row["side"] = "shared" if len(s) > 1 else list(s)[0]

    return kept


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ak", "--api_key", required=True)
    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--n", type=int, default=None, help="Limit to first N cases (debug)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model tags to run (default: all 11)")
    args = parser.parse_args()

    run_models = MODELS
    if args.models:
        run_models = [m.strip() for m in args.models.split(",")]
        unknown = [m for m in run_models if m not in MODELS]
        if unknown:
            print(f"WARNING: unknown models: {unknown}", flush=True)

    out = resolve_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Collect all unique criterion texts ──────────────────────────
    print("Loading rubrics...", flush=True)

    # human criteria: deduplicate across models by criterion_id
    # (all models share the same human rubric per task_id)
    human_by_task: dict[str, list[dict]] = {}   # task_id -> [criterion]
    model_rubrics: dict[str, dict[str, list[dict]]] = {}  # model -> task_id -> [criterion]

    # Use first model to load human rubrics (all models share same human)
    first_human, _ = load_all_rubrics(MODELS[0])
    task_ids = sorted(first_human.keys())
    if args.n:
        task_ids = task_ids[: args.n]
    for tid in task_ids:
        human_by_task[tid] = first_human[tid]
    print(f"  {len(task_ids)} cases, {sum(len(v) for v in human_by_task.values())} human criteria", flush=True)

    for model in run_models:
        _, ai = load_all_rubrics(model)
        model_rubrics[model] = {tid: ai[tid] for tid in task_ids if tid in ai}
        mc = sum(len(v) for v in model_rubrics[model].values())
        print(f"  {model}: {mc} model criteria", flush=True)

    # ── Step 2: Collect all unique texts for embedding ───────────────────────
    print("\nCollecting unique texts...", flush=True)

    text_to_idx: dict[str, int] = {}
    all_texts: list[str] = []

    def register(text: str) -> int:
        normed = normalize_for_embed(text)
        if normed not in text_to_idx:
            text_to_idx[normed] = len(all_texts)
            all_texts.append(normed)
        return text_to_idx[normed]

    # Human texts
    human_idx_map: dict[str, dict[str, int]] = {}  # task_id -> criterion_id -> idx
    for tid, criteria in human_by_task.items():
        human_idx_map[tid] = {}
        for c in criteria:
            idx = register(c["title"])
            human_idx_map[tid][c["id"]] = idx

    # Model texts
    model_idx_map: dict[str, dict[str, dict[str, int]]] = {}  # model -> task_id -> crit_id -> idx
    for model in run_models:
        model_idx_map[model] = {}
        for tid, criteria in model_rubrics[model].items():
            model_idx_map[model][tid] = {}
            for c in criteria:
                idx = register(c["title"])
                model_idx_map[model][tid][c["id"]] = idx

    print(f"  {len(all_texts)} unique texts to embed", flush=True)

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    embed_cache_path = out / "embeddings_cache.npy"
    texts_cache_path = out / "texts_cache.json"

    if embed_cache_path.exists() and texts_cache_path.exists():
        print("\nLoading cached embeddings...", flush=True)
        cached_texts = json.loads(texts_cache_path.read_text())
        if cached_texts == all_texts:
            all_vecs = np.load(embed_cache_path)
            print(f"  loaded {all_vecs.shape} from cache", flush=True)
        else:
            print("  cache mismatch, re-embedding...", flush=True)
            all_vecs = None
    else:
        all_vecs = None

    if all_vecs is None:
        print(f"\nEmbedding {len(all_texts)} texts in batches of {BATCH_SIZE}...", flush=True)
        all_vecs = embed_all_texts(all_texts, args.api_key, workers=args.workers)
        np.save(embed_cache_path, all_vecs)
        texts_cache_path.write_text(json.dumps(all_texts))
        print(f"  saved embeddings {all_vecs.shape}", flush=True)

    # ── Step 4: Per-model, per-case analysis ──────────────────────────────────
    print("\nAnalyzing unique criteria per model...", flush=True)

    summary = {}

    for model in run_models:
        model_out_dir = out / model
        model_out_dir.mkdir(exist_ok=True)
        out_path = model_out_dir / "unique_criteria.jsonl"

        all_rows = []
        n_cases = 0
        counts = {"human": 0, "model": 0, "shared": 0}

        for tid in task_ids:
            if tid not in model_rubrics[model]:
                continue

            h_criteria = human_by_task[tid]
            m_criteria = model_rubrics[model][tid]

            h_vecs = all_vecs[[human_idx_map[tid][c["id"]] for c in h_criteria]]
            m_vecs = all_vecs[[model_idx_map[model][tid][c["id"]] for c in m_criteria]]

            pooled = pool_deduplicate(h_criteria, h_vecs, m_criteria, m_vecs, args.threshold)

            for row in pooled:
                all_rows.append({"task_id": tid, "model": model, **row})
                counts[row["side"]] += 1
            n_cases += 1

        # Write output
        with out_path.open("w", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        n_total = sum(counts.values())
        summary[model] = {
            "cases": n_cases,
            "total_unique_concepts": n_total,
            "human_only": counts["human"],
            "model_only": counts["model"],
            "shared": counts["shared"],
            "model_only_pct": round(100 * counts["model"] / n_total, 1) if n_total else 0,
            "human_only_pct": round(100 * counts["human"] / n_total, 1) if n_total else 0,
            "shared_pct": round(100 * counts["shared"] / n_total, 1) if n_total else 0,
        }
        print(
            f"  {model}: total={n_total} "
            f"shared={counts['shared']}({summary[model]['shared_pct']}%) "
            f"human_only={counts['human']}({summary[model]['human_only_pct']}%) "
            f"model_only={counts['model']}({summary[model]['model_only_pct']}%)",
            flush=True,
        )

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    summary_path = out / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"threshold": args.threshold, "n_cases": len(task_ids), "models": summary}, f, indent=2)
    print(f"\nDone. Summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
