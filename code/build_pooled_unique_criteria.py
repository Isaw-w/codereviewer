#!/usr/bin/env python3
"""
Pool all 11-model AI rubric criteria together (per case), deduplicate, then
compare against human rubric criteria to identify:
  - human_only: concepts only in human rubric
  - model_only: concepts found by at least one model but not in human rubric
  - shared:     concepts present on both sides

Within model_only, each concept also records which models raised it (model_coverage).

Output:
  --output_dir/pooled_unique_criteria.jsonl   (one row per concept per task_id)
  --output_dir/summary.json

Usage:
    python3 build_pooled_unique_criteria.py \\
        -ak "$LAB_OPENROUTER_KEY" \\
        -o data/paper_release/finding2/coverage/pooled_unique_t70 \\
        --threshold 0.70

    # reuse embeddings from a previous run:
    python3 build_pooled_unique_criteria.py \\
        -ak "$LAB_OPENROUTER_KEY" \\
        -o data/paper_release/finding2/coverage/pooled_unique_t70 \\
        --threshold 0.70 \\
        --embed_cache data/paper_release/finding2/coverage/human_model_unique_t70_all
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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


# ─── Text normalization (same as build_human_model_unique_criteria.py) ────────

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
    title = title.strip().strip("'\"").strip()
    if not title:
        return title
    if re.match(r'^(the )?response\b', title, re.I):
        return title
    if re.match(r'^advice\s*,', title, re.I):
        return title
    if re.match(r'^suggests?\s*,.*\bfails?\b', title, re.I):
        return title
    if _NEGATION_RE.match(title):
        return title
    if re.match(r'^detrimental\s*:', title, re.I):
        return title
    first_token = re.match(r'^(\S+)', title)
    if first_token and re.search(r'[-/]', first_token.group(1)):
        m_hyph = re.match(r'^\S+(?:\s*,\s*\w+)*(?:\s+or\s+\w+)?\s+(.+)$', title, re.DOTALL)
        if m_hyph:
            return f"Does the response consider {m_hyph.group(1)}"
    stripped = _ADVERBS_RE.sub('', title)
    m_comma = re.match(r'^(\w+),\s*[^,]+,\s*(.+)$', stripped, re.DOTALL)
    if m_comma:
        return f"Does the response consider {m_comma.group(2)}"
    m = re.match(r'^\w+\s+(.+)$', stripped, re.DOTALL)
    if m:
        return f"Does the response consider {m.group(1)}"
    return title


# ─── I/O ─────────────────────────────────────────────────────────────────────

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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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


def embed_all_texts(texts: list[str], api_key: str, workers: int = 8) -> np.ndarray:
    batches = [texts[i: i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    results: dict[int, list] = {}
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
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


# ─── Core: two-stage dedup ────────────────────────────────────────────────────

def deduplicate_side(
    criteria: list[dict],
    vecs: np.ndarray,
    threshold: float,
    side_label: str,
    model_tag: str | None = None,
) -> tuple[list[dict], list[np.ndarray]]:
    """
    Greedily deduplicate one side's criteria.
    Returns (kept_criteria_list, kept_vecs_list).
    Each kept criterion has 'side', 'models' (list of model tags that contributed).
    """
    kept: list[dict] = []
    kept_vecs: list[np.ndarray] = []

    for i, crit in enumerate(criteria):
        v = vecs[i]
        if kept_vecs:
            mat = np.stack(kept_vecs)
            sims = (mat @ v).astype(float)
            best_idx = int(sims.argmax())
            best_cos = float(sims[best_idx])
            if best_cos >= threshold:
                # merge: add model tag if not already present
                if model_tag and model_tag not in kept[best_idx]["models"]:
                    kept[best_idx]["models"].append(model_tag)
                kept[best_idx]["max_cos_merged"] = max(
                    kept[best_idx].get("max_cos_merged", 0.0), best_cos
                )
                continue
        row = {
            "title": crit["title"],
            "id": crit["id"],
            "weight": crit.get("weight"),
            "side": side_label,
            "models": [model_tag] if model_tag else [],
            "max_cos_merged": 0.0,
        }
        kept.append(row)
        kept_vecs.append(v)

    return kept, kept_vecs


def merge_model_into_pool(
    pool_criteria: list[dict],
    pool_vecs: list[np.ndarray],
    new_criteria: list[dict],
    new_vecs: np.ndarray,
    threshold: float,
    model_tag: str,
):
    """
    Merge new model's criteria into the existing model pool (in-place).
    Near-duplicates update the existing entry; novel criteria are appended.
    """
    for i, crit in enumerate(new_criteria):
        v = new_vecs[i]
        if pool_vecs:
            mat = np.stack(pool_vecs)
            sims = (mat @ v).astype(float)
            best_idx = int(sims.argmax())
            best_cos = float(sims[best_idx])
            if best_cos >= threshold:
                if model_tag not in pool_criteria[best_idx]["models"]:
                    pool_criteria[best_idx]["models"].append(model_tag)
                pool_criteria[best_idx]["max_cos_merged"] = max(
                    pool_criteria[best_idx].get("max_cos_merged", 0.0), best_cos
                )
                continue
        row = {
            "title": crit["title"],
            "id": crit["id"],
            "weight": crit.get("weight"),
            "side": "model",
            "models": [model_tag],
            "max_cos_merged": 0.0,
        }
        pool_criteria.append(row)
        pool_vecs.append(v)


def compare_pool_to_human(
    human_criteria: list[dict],
    human_vecs: np.ndarray,
    model_pool: list[dict],
    model_pool_vecs: list[np.ndarray],
    threshold: float,
) -> list[dict]:
    """
    Given a deduplicated human set and a deduplicated model pool,
    determine final side labels:
      - human criteria: shared if max cosine to any model concept >= threshold, else human_only
      - model criteria: shared if max cosine to any human concept >= threshold, else model_only

    Returns combined list with 'side' updated.
    """
    # Build arrays
    h_mat = human_vecs  # (H, D)
    m_mat = np.stack(model_pool_vecs) if model_pool_vecs else None  # (M, D)

    result = []

    # Human side: check against model pool
    for i, crit in enumerate(human_criteria):
        row = dict(crit)
        if m_mat is not None and len(m_mat) > 0:
            sims = (m_mat @ h_mat[i]).astype(float)
            best_cos = float(sims.max())
            row["max_cos_to_other"] = best_cos
            row["side"] = "shared" if best_cos >= threshold else "human_only"
        else:
            row["max_cos_to_other"] = 0.0
            row["side"] = "human_only"
        result.append(row)

    # Model side: check against human
    h_has_data = h_mat is not None and h_mat.shape[0] > 0
    for i, crit in enumerate(model_pool):
        row = dict(crit)
        v = model_pool_vecs[i]
        if h_has_data:
            sims = (h_mat @ v).astype(float)
            best_cos = float(sims.max())
            row["max_cos_to_other"] = best_cos
            row["side"] = "shared" if best_cos >= threshold else "model_only"
        else:
            row["max_cos_to_other"] = 0.0
            row["side"] = "model_only"
        result.append(row)

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ak", "--api_key", required=True)
    parser.add_argument("--output_dir", "-o", required=True)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--n", type=int, default=None, help="Limit to first N cases (debug)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--embed_cache", default=None,
        help="Directory containing embeddings_cache.npy + texts_cache.json from a previous run "
             "(e.g. human_model_unique_t70_all). Texts must match exactly to use cache."
    )
    args = parser.parse_args()

    out = resolve_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load rubrics ──────────────────────────────────────────────────
    print("Loading rubrics...", flush=True)
    first_human, _ = load_all_rubrics(MODELS[0])
    task_ids = sorted(first_human.keys())
    if args.n:
        task_ids = task_ids[: args.n]

    human_by_task: dict[str, list[dict]] = {tid: first_human[tid] for tid in task_ids}
    model_rubrics: dict[str, dict[str, list[dict]]] = {}
    for model in MODELS:
        _, ai = load_all_rubrics(model)
        model_rubrics[model] = {tid: ai[tid] for tid in task_ids if tid in ai}
        mc = sum(len(v) for v in model_rubrics[model].values())
        print(f"  {model}: {mc} criteria", flush=True)

    hc_total = sum(len(v) for v in human_by_task.values())
    print(f"  human: {hc_total} criteria across {len(task_ids)} cases", flush=True)

    # ── Step 2: Collect unique embed texts ────────────────────────────────────
    print("\nCollecting unique texts for embedding...", flush=True)
    text_to_idx: dict[str, int] = {}
    all_texts: list[str] = []

    def register(text: str) -> int:
        normed = normalize_for_embed(text)
        if normed not in text_to_idx:
            text_to_idx[normed] = len(all_texts)
            all_texts.append(normed)
        return text_to_idx[normed]

    human_idx_map: dict[str, dict[str, int]] = {}
    for tid, criteria in human_by_task.items():
        human_idx_map[tid] = {c["id"]: register(c["title"]) for c in criteria}

    model_idx_map: dict[str, dict[str, dict[str, int]]] = {}
    for model in MODELS:
        model_idx_map[model] = {}
        for tid, criteria in model_rubrics[model].items():
            model_idx_map[model][tid] = {c["id"]: register(c["title"]) for c in criteria}

    print(f"  {len(all_texts)} unique texts to embed", flush=True)

    # ── Step 3: Embed (or load cache) ─────────────────────────────────────────
    embed_cache_path = out / "embeddings_cache.npy"
    texts_cache_path = out / "texts_cache.json"

    # Try external cache first
    if args.embed_cache:
        embed_cache_dir = resolve_path(args.embed_cache)
        ext_npy = embed_cache_dir / "embeddings_cache.npy"
        ext_txt = embed_cache_dir / "texts_cache.json"
        if ext_npy.exists() and ext_txt.exists():
            cached_texts = json.loads(ext_txt.read_text())
            if cached_texts == all_texts:
                print(f"\nLoading embeddings from external cache: {embed_cache_dir}", flush=True)
                all_vecs = np.load(ext_npy)
                print(f"  loaded {all_vecs.shape}", flush=True)
                # save to local cache too
                shutil.copy(ext_npy, embed_cache_path)
                shutil.copy(ext_txt, texts_cache_path)
            else:
                print(f"  external cache texts mismatch — will re-embed", flush=True)
                all_vecs = None
        else:
            print(f"  external cache not found at {args.embed_cache}", flush=True)
            all_vecs = None
    elif embed_cache_path.exists() and texts_cache_path.exists():
        cached_texts = json.loads(texts_cache_path.read_text())
        if cached_texts == all_texts:
            print("\nLoading cached embeddings...", flush=True)
            all_vecs = np.load(embed_cache_path)
            print(f"  loaded {all_vecs.shape}", flush=True)
        else:
            print("  cache mismatch, re-embedding...", flush=True)
            all_vecs = None
    else:
        all_vecs = None

    if all_vecs is None:
        print(f"\nEmbedding {len(all_texts)} texts...", flush=True)
        all_vecs = embed_all_texts(all_texts, args.api_key, workers=args.workers)
        np.save(embed_cache_path, all_vecs)
        texts_cache_path.write_text(json.dumps(all_texts))
        print(f"  saved {all_vecs.shape}", flush=True)

    # ── Step 4: Per-case pooled analysis ──────────────────────────────────────
    print("\nRunning pooled deduplication per case...", flush=True)

    all_rows = []
    counts = {"human_only": 0, "model_only": 0, "shared": 0}
    # model_only coverage histogram: how many models raised it
    coverage_hist: dict[int, int] = {}  # n_models -> count

    for case_i, tid in enumerate(task_ids):
        h_criteria = human_by_task[tid]
        h_vecs = all_vecs[[human_idx_map[tid][c["id"]] for c in h_criteria]]  # (H, D)

        # Step A: use human criteria as-is (no internal dedup needed — each criterion has a unique id)
        h_kept_mat = h_vecs  # (H, D)
        h_kept = h_criteria

        # Step B: build model pool by merging all 11 models sequentially
        m_pool: list[dict] = []
        m_pool_vecs: list[np.ndarray] = []
        for model in MODELS:
            if tid not in model_rubrics[model]:
                continue
            m_criteria = model_rubrics[model][tid]
            m_vecs = all_vecs[[model_idx_map[model][tid][c["id"]] for c in m_criteria]]
            merge_model_into_pool(m_pool, m_pool_vecs, m_criteria, m_vecs, args.threshold, model)

        # Step C: compare human pool vs model pool
        final = compare_pool_to_human(
            h_kept, h_kept_mat, m_pool, m_pool_vecs, args.threshold
        )

        for row in final:
            all_rows.append({"task_id": tid, **row})
            counts[row["side"]] += 1
            if row["side"] == "model_only":
                n_models = len(row.get("models", []))
                coverage_hist[n_models] = coverage_hist.get(n_models, 0) + 1

        if (case_i + 1) % 50 == 0 or (case_i + 1) == len(task_ids):
            print(f"  {case_i + 1}/{len(task_ids)} cases done", flush=True)

    # ── Step 5: Write outputs ─────────────────────────────────────────────────
    out_jsonl = out / "pooled_unique_criteria.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(all_rows)} rows -> {out_jsonl}", flush=True)

    n_total = sum(counts.values())
    summary = {
        "threshold": args.threshold,
        "n_cases": len(task_ids),
        "n_models": len(MODELS),
        "total_unique_concepts": n_total,
        "human_only": counts["human_only"],
        "model_only": counts["model_only"],
        "shared": counts["shared"],
        "human_only_pct": round(100 * counts["human_only"] / n_total, 1) if n_total else 0,
        "model_only_pct": round(100 * counts["model_only"] / n_total, 1) if n_total else 0,
        "shared_pct": round(100 * counts["shared"] / n_total, 1) if n_total else 0,
        "model_only_coverage_histogram": {
            str(k): v for k, v in sorted(coverage_hist.items())
        },
    }
    summary_path = out / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS (threshold={args.threshold}, {len(task_ids)} cases, {len(MODELS)} models)", flush=True)
    print(f"  total unique concepts : {n_total}", flush=True)
    print(f"  human_only            : {counts['human_only']} ({summary['human_only_pct']}%)", flush=True)
    print(f"  model_only            : {counts['model_only']} ({summary['model_only_pct']}%)", flush=True)
    print(f"  shared                : {counts['shared']} ({summary['shared_pct']}%)", flush=True)
    print(f"\n  model_only by coverage (# models that raised it):", flush=True)
    for k, v in sorted(coverage_hist.items()):
        print(f"    {k} model(s): {v} concepts", flush=True)
    print(f"\nSummary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
