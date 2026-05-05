#!/usr/bin/env python3
"""
Global deduplication of human vs model rubric criteria across all 500 cases.

Step 1: Collect all human criteria across 500 cases → greedy global dedup → human concept set
Step 2: Collect all model criteria across 11 models × 500 cases → greedy global dedup → model concept set
Step 3: Cross-compare: human_only / model_only / shared

Reuses embedding cache from a previous run (--embed_cache).

Usage:
    python3 build_global_unique_criteria.py \\
        -ak "$LAB_OPENROUTER_KEY" \\
        -o data/paper_release/finding3/coverage/global_unique_t70 \\
        --threshold 0.70 \\
        --embed_cache data/paper_release/finding3/coverage/human_model_unique_t70_all
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

def embed_batch(texts: list[str], api_key: str) -> list[list[float]]:
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(API_URL, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f"  [embed retry {attempt+1}] {e}, waiting {wait}s", flush=True)
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
            if done % 10 == 0 or done == len(batches):
                print(f"  embedded {done}/{len(batches)} batches", flush=True)
    flat = []
    for i in range(len(batches)):
        flat.extend(results[i])
    arr = np.array(flat, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms == 0, 1.0, norms)


# ─── Global greedy dedup ──────────────────────────────────────────────────────

def global_greedy_dedup(
    items: list[dict],
    all_vecs: np.ndarray,
    threshold: float,
    label: str,
    batch_size: int = 512,
) -> tuple[list[dict], np.ndarray]:
    """
    Batch greedy dedup for speed.

    Per batch:
      Phase 1 — compare all B items against pre-batch kept_mat at once: (B,D)@(D,K)→(B,K)
                items that match an existing representative are merged immediately.
      Phase 2 — unmatched items within the batch are sequentially deduped against each other.

    Precision vs pure sequential: negligible (only batch-boundary effects).
    Returns (kept_list, kept_mat ndarray).
    """
    D = all_vecs.shape[1]
    kept: list[dict] = []
    kept_vecs: list[np.ndarray] = []
    kept_mat: np.ndarray = np.empty((0, D), dtype=np.float32)

    total = len(items)
    for batch_start in range(0, total, batch_size):
        batch_items = items[batch_start: batch_start + batch_size]
        B = len(batch_items)
        batch_vecs = all_vecs[[it["embed_idx"] for it in batch_items]]  # (B, D)

        # Phase 1: compare batch against existing kept
        assigned = [-1] * B
        if len(kept_mat) > 0:
            sim = batch_vecs @ kept_mat.T           # (B, K)
            best_k = sim.argmax(axis=1)             # (B,)
            best_cos = sim[np.arange(B), best_k]   # (B,)
            for i in range(B):
                if best_cos[i] >= threshold:
                    assigned[i] = int(best_k[i])

        for i, item in enumerate(batch_items):
            if assigned[i] >= 0:
                k = assigned[i]
                if label == "model" and item.get("model") and item["model"] not in kept[k]["models"]:
                    kept[k]["models"].append(item["model"])
                kept[k]["n_merged"] = kept[k].get("n_merged", 0) + 1
                w = item.get("weight")
                if w is not None and (kept[k]["weight"] is None or abs(w) > abs(kept[k]["weight"])):
                    kept[k]["weight"] = w
                d = item.get("dimension")
                if d and d not in kept[k]["dimensions"]:
                    kept[k]["dimensions"].append(d)

        # Phase 2: sequential dedup among unmatched items within this batch
        new_kept: list[dict] = []
        new_vecs: list[np.ndarray] = []
        for i in range(B):
            if assigned[i] >= 0:
                continue
            item = batch_items[i]
            v = batch_vecs[i]
            if new_vecs:
                new_mat = np.stack(new_vecs)
                sims = (new_mat @ v).astype(float)
                best_idx = int(sims.argmax())
                if float(sims[best_idx]) >= threshold:
                    if label == "model" and item.get("model") and item["model"] not in new_kept[best_idx]["models"]:
                        new_kept[best_idx]["models"].append(item["model"])
                    new_kept[best_idx]["n_merged"] = new_kept[best_idx].get("n_merged", 0) + 1
                    w = item.get("weight")
                    if w is not None and (new_kept[best_idx]["weight"] is None or abs(w) > abs(new_kept[best_idx]["weight"])):
                        new_kept[best_idx]["weight"] = w
                    d = item.get("dimension")
                    if d and d not in new_kept[best_idx]["dimensions"]:
                        new_kept[best_idx]["dimensions"].append(d)
                    continue
            row: dict = {
                "title": item["title"], "side": label,
                "task_id": item.get("task_id"), "weight": item.get("weight"),
                "dimensions": [item["dimension"]] if item.get("dimension") else [],
                "n_merged": 0,
            }
            if label == "model":
                row["models"] = [item["model"]] if item.get("model") else []
            new_kept.append(row)
            new_vecs.append(v)

        if new_kept:
            kept.extend(new_kept)
            kept_vecs.extend(new_vecs)
            kept_mat = np.stack(kept_vecs)  # rebuild once per batch, not per item

        done = batch_start + B
        if done % 5000 == 0 or done == total:
            print(f"    {done}/{total} processed, {len(kept)} kept", flush=True)

    return kept, kept_mat


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
        help="Dir with embeddings_cache.npy + texts_cache.json from a previous run"
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

    h_total = sum(len(v) for v in human_by_task.values())
    m_total = sum(len(v) for vv in model_rubrics.values() for v in vv.values())
    print(f"  human: {h_total} criteria, model (11×): {m_total} criteria", flush=True)

    # ── Step 2: Build text registry for embedding ─────────────────────────────
    print("\nCollecting unique texts...", flush=True)
    text_to_idx: dict[str, int] = {}
    all_texts: list[str] = []

    def register(text: str) -> int:
        normed = normalize_for_embed(text)
        if normed not in text_to_idx:
            text_to_idx[normed] = len(all_texts)
            all_texts.append(normed)
        return text_to_idx[normed]

    def get_dimension(c: dict) -> str | None:
        ann = c.get("annotations")
        if isinstance(ann, dict):
            return ann.get("rubric_dimension")
        return None

    # Human items (flat list, preserving task_id + weight + dimension)
    human_items: list[dict] = []
    for tid in task_ids:
        for c in human_by_task[tid]:
            human_items.append({
                "title": c["title"],
                "task_id": tid,
                "weight": c.get("weight"),
                "dimension": get_dimension(c),
                "embed_idx": register(c["title"]),
            })

    # Model items (flat list, preserving model + task_id + weight + dimension)
    model_items: list[dict] = []
    for model in MODELS:
        for tid in task_ids:
            for c in model_rubrics[model].get(tid, []):
                model_items.append({
                    "title": c["title"],
                    "task_id": tid,
                    "model": model,
                    "weight": c.get("weight"),
                    "dimension": get_dimension(c),
                    "embed_idx": register(c["title"]),
                })

    print(f"  {len(all_texts)} unique normalized texts", flush=True)

    # ── Step 3: Load or compute embeddings ────────────────────────────────────
    embed_cache_path = out / "embeddings_cache.npy"
    texts_cache_path = out / "texts_cache.json"
    all_vecs = None

    if args.embed_cache:
        embed_cache_dir = resolve_path(args.embed_cache)
        ext_npy = embed_cache_dir / "embeddings_cache.npy"
        ext_txt = embed_cache_dir / "texts_cache.json"
        if ext_npy.exists() and ext_txt.exists():
            cached_texts = json.loads(ext_txt.read_text())
            if cached_texts == all_texts:
                print(f"\nLoading embeddings from external cache: {embed_cache_dir}", flush=True)
                all_vecs = np.load(ext_npy)
                shutil.copy(ext_npy, embed_cache_path)
                shutil.copy(ext_txt, texts_cache_path)
                print(f"  loaded {all_vecs.shape}", flush=True)
            else:
                print("  external cache texts mismatch — will re-embed", flush=True)

    if all_vecs is None and embed_cache_path.exists() and texts_cache_path.exists():
        cached_texts = json.loads(texts_cache_path.read_text())
        if cached_texts == all_texts:
            print("\nLoading local cached embeddings...", flush=True)
            all_vecs = np.load(embed_cache_path)
            print(f"  loaded {all_vecs.shape}", flush=True)

    if all_vecs is None:
        print(f"\nEmbedding {len(all_texts)} texts...", flush=True)
        all_vecs = embed_all_texts(all_texts, args.api_key, workers=args.workers)
        np.save(embed_cache_path, all_vecs)
        texts_cache_path.write_text(json.dumps(all_texts))
        print(f"  saved {all_vecs.shape}", flush=True)

    # ── Step 4: Global greedy dedup — human side ──────────────────────────────
    print(f"\nDeduplicating human side ({len(human_items)} criteria)...", flush=True)
    h_kept, h_mat = global_greedy_dedup(human_items, all_vecs, args.threshold, "human")
    print(f"  → {len(h_kept)} unique human concepts", flush=True)

    # ── Step 5: Global greedy dedup — model side ──────────────────────────────
    print(f"\nDeduplicating model side ({len(model_items)} criteria, 11 models × 500 cases)...", flush=True)
    m_kept, m_mat = global_greedy_dedup(model_items, all_vecs, args.threshold, "model")
    print(f"  → {len(m_kept)} unique model concepts", flush=True)

    # ── Step 6: Cross-compare ─────────────────────────────────────────────────
    print("\nCross-comparing human vs model concept sets...", flush=True)

    # For each human concept: check if any model concept is within threshold
    h_max_cos = np.zeros(len(h_kept), dtype=np.float32)
    m_max_cos = np.zeros(len(m_kept), dtype=np.float32)

    if len(h_mat) > 0 and len(m_mat) > 0:
        chunk = 500
        for i in range(0, len(h_kept), chunk):
            block = h_mat[i: i + chunk] @ m_mat.T   # (chunk, M)
            h_max_cos[i: i + chunk] = block.max(axis=1)
            m_max_cos = np.maximum(m_max_cos, block.max(axis=0))

    h_has_match = h_max_cos >= args.threshold
    m_has_match = m_max_cos >= args.threshold

    # Assign final side labels
    # shared is counted separately per side (h_shared = human concepts covered by model,
    # m_shared = model concepts that overlap with human — these are different sets)
    all_rows = []
    h_only = h_shared = m_only = m_shared = 0
    coverage_hist: dict[int, int] = {}

    for i, row in enumerate(h_kept):
        r = dict(row)
        r["max_cos_to_other"] = round(float(h_max_cos[i]), 4)
        if h_has_match[i]:
            r["side"] = "shared"
            h_shared += 1
        else:
            r["side"] = "human_only"
            h_only += 1
        all_rows.append(r)

    for i, row in enumerate(m_kept):
        r = dict(row)
        r["max_cos_to_other"] = round(float(m_max_cos[i]), 4)
        if m_has_match[i]:
            r["side"] = "shared"
            m_shared += 1
        else:
            r["side"] = "model_only"
            m_only += 1
            n_models = len(r.get("models", []))
            coverage_hist[n_models] = coverage_hist.get(n_models, 0) + 1
        all_rows.append(r)

    H = len(h_kept)
    M = len(m_kept)

    # ── Step 7: Write outputs ─────────────────────────────────────────────────
    out_jsonl = out / "global_unique_criteria.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def weight_dist(rows: list[dict]) -> dict:
        freq: dict[str, int] = {}
        for r in rows:
            w = r.get("weight")
            if w is None:
                continue
            key = str(int(w)) if isinstance(w, float) and w == int(w) else str(w)
            freq[key] = freq.get(key, 0) + 1
        return {k: v for k, v in sorted(freq.items(), key=lambda x: float(x[0]))}

    def dimension_dist(rows: list[dict]) -> dict:
        freq: dict[str, int] = {}
        for r in rows:
            for d in r.get("dimensions", []):
                freq[d] = freq.get(d, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: -x[1]))

    by_side: dict[str, list[dict]] = {"human_only": [], "human_shared": [], "model_only": [], "model_shared": []}
    for r in all_rows:
        if r["side"] == "human_only":
            by_side["human_only"].append(r)
        elif r["side"] == "model_only":
            by_side["model_only"].append(r)
        elif r["side"] == "shared" and r.get("models") is None:  # human shared rows have no "models" key
            by_side["human_shared"].append(r)
        else:
            by_side["model_shared"].append(r)

    summary = {
        "threshold": args.threshold,
        "n_cases": len(task_ids),
        "n_models": len(MODELS),
        "human_raw_criteria": h_total,
        "model_raw_criteria": m_total,
        "human_unique_concepts": H,
        "model_unique_concepts": M,
        # Human perspective
        "human_only": h_only,
        "human_covered_by_model": h_shared,
        "human_only_pct": round(100 * h_only / H, 1) if H else 0,
        "human_covered_pct": round(100 * h_shared / H, 1) if H else 0,
        # Model perspective
        "model_only": m_only,
        "model_overlaps_human": m_shared,
        "model_only_pct": round(100 * m_only / M, 1) if M else 0,
        "model_overlap_pct": round(100 * m_shared / M, 1) if M else 0,
        "model_only_coverage_histogram": {str(k): v for k, v in sorted(coverage_hist.items())},
        # Weight distributions per cluster
        "weight_distribution": {
            "human_only":   weight_dist(by_side["human_only"]),
            "human_shared": weight_dist(by_side["human_shared"]),
            "model_only":   weight_dist(by_side["model_only"]),
            "model_shared": weight_dist(by_side["model_shared"]),
        },
        # Dimension distributions per cluster
        "dimension_distribution": {
            "human_only":   dimension_dist(by_side["human_only"]),
            "human_shared": dimension_dist(by_side["human_shared"]),
            "model_only":   dimension_dist(by_side["model_only"]),
            "model_shared": dimension_dist(by_side["model_shared"]),
        },
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}", flush=True)
    print(f"GLOBAL DEDUP RESULTS (threshold={args.threshold}, {len(task_ids)} cases, {len(MODELS)} models)", flush=True)
    print(f"  human raw → unique : {h_total} → {H}", flush=True)
    print(f"  model raw → unique : {m_total} → {M}", flush=True)
    print(f"", flush=True)
    print(f"  Human perspective (of {H} human concepts):", flush=True)
    print(f"    covered by model : {h_shared} ({summary['human_covered_pct']}%)", flush=True)
    print(f"    human_only       : {h_only} ({summary['human_only_pct']}%)", flush=True)
    print(f"", flush=True)
    print(f"  Model perspective (of {M} model concepts):", flush=True)
    print(f"    overlaps human   : {m_shared} ({summary['model_overlap_pct']}%)", flush=True)
    print(f"    model_only       : {m_only} ({summary['model_only_pct']}%)", flush=True)
    print(f"", flush=True)
    print(f"  model_only by coverage (# models that raised it):", flush=True)
    for k, v in sorted(coverage_hist.items()):
        print(f"    {k:2d} model(s): {v}", flush=True)
    print(f"\nSummary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
