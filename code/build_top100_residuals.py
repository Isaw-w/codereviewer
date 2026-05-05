#!/usr/bin/env python3
"""
Build a CLEAN residuals file for Finding 3 dual-method validation.

Strategy: pick the 100 densest cases (most human-only concepts per case),
which covers ~50% of all 2,227 human-only concepts. For each selected case,
include ALL human-only criterion IDs (not just a sample).

Output: a single residuals_top100.jsonl with one row per (case, concept).
No merging with old data. No mixing of sources.

Usage:
    python3 build_top100_residuals.py [--n_cases 100] [--dry_run]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(__file__).resolve().parents[1]


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


GLOBAL_UNIQUE_PATH = resolve_path("data/paper_release/finding3/coverage/global_unique_t70/global_unique_criteria.jsonl")
HUMAN_RUBRIC_PATH = resolve_path("data/paper_release/rubrics/original/human_rubric_500cases.jsonl")
OUT_PATH = resolve_path("data/paper_release/finding3/direct_check/residuals_top100.jsonl")


# ── Same normalization as build_global_unique_criteria.py ────────────────────

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


# ── Load data ─────────────────────────────────────────────────────────────────

def load_human_rubric() -> dict[str, list[dict]]:
    result = {}
    with HUMAN_RUBRIC_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rubric = row["RUBRIC"]
            if isinstance(rubric, str):
                rubric = json.loads(rubric)
            result[row["TASK_ID"]] = rubric
    return result


def load_human_only_by_case() -> dict[str, list[dict]]:
    by_case: dict[str, list[dict]] = {}
    with GLOBAL_UNIQUE_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["side"] == "human_only":
                by_case.setdefault(row["task_id"], []).append(row)
    return by_case


def resolve_criterion_id(
    concept_title: str,
    rubric: list[dict],
) -> str | None:
    # Build lookup maps
    exact = {c["title"]: c["id"] for c in rubric}
    normed = {normalize_for_embed(c["title"]): c["id"] for c in rubric}

    # 1. Exact match
    if concept_title in exact:
        return exact[concept_title]

    # 2. Normalized match
    concept_normed = normalize_for_embed(concept_title)
    if concept_normed in normed:
        return normed[concept_normed]

    # 3. Substring match (last resort)
    for title, cid in exact.items():
        if concept_title.lower() in title.lower() or title.lower() in concept_title.lower():
            return cid

    return None


def get_band(cos: float) -> str:
    if cos < 0.60:
        return "<0.60"
    if cos < 0.65:
        return "0.60-0.65"
    return "0.65-0.70"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cases", type=int, default=100)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    print("Loading human-only concepts by case...")
    by_case = load_human_only_by_case()
    total_concepts = sum(len(v) for v in by_case.values())
    print(f"  {total_concepts} concepts across {len(by_case)} cases")

    print("\nLoading human rubric...")
    human_rubric = load_human_rubric()
    print(f"  {len(human_rubric)} cases")

    # Select top N densest cases
    sorted_cases = sorted(by_case.items(), key=lambda x: -len(x[1]))
    selected = sorted_cases[:args.n_cases]
    selected_ids = set(tid for tid, _ in selected)

    n_selected_concepts = sum(len(concepts) for _, concepts in selected)
    print(f"\nSelected {len(selected)} cases, covering {n_selected_concepts} concepts ({100*n_selected_concepts/total_concepts:.1f}%)")

    # Band distribution
    band_counts = {"<0.60": 0, "0.60-0.65": 0, "0.65-0.70": 0}
    for _, concepts in selected:
        for c in concepts:
            band_counts[get_band(c["max_cos_to_other"])] += 1
    print(f"\nBand distribution in selected cases:")
    for b, n in band_counts.items():
        print(f"  {b}: {n} ({100*n/n_selected_concepts:.1f}%)")

    # Resolve all criterion IDs
    print(f"\nResolving criterion IDs...")
    residuals = []
    resolved = unresolved = 0
    for tid, concepts in selected:
        rubric = human_rubric.get(tid, [])
        for concept in concepts:
            cid = resolve_criterion_id(concept["title"], rubric)
            if cid is None:
                print(f"  [WARN] Unresolved: task={tid} title={concept['title'][:60]!r}")
                unresolved += 1
                continue
            resolved += 1
            residuals.append({
                "normalized_text": concept["title"],
                "representative_title": concept["title"],
                "max_cos_to_other": concept["max_cos_to_other"],
                "cosine_band": get_band(concept["max_cos_to_other"]),
                "side": "human_only",
                "weight": concept.get("weight"),
                "source": "top100_density",
                "rows": [{
                    "task_id": tid,
                    "criterion_id": cid,
                    "final_text_source": "top100_density",
                    "source_criterion": concept["title"],
                    "final_text": concept["title"],
                }]
            })

    print(f"  Resolved: {resolved}, Unresolved: {unresolved}")

    if args.dry_run:
        print("\n[DRY RUN] Not writing output file.")
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in residuals:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(residuals)} entries to: {OUT_PATH}")

    # Summary
    cases_in_file = set()
    for row in residuals:
        for occ in row["rows"]:
            cases_in_file.add(occ["task_id"])
    print(f"  Unique cases: {len(cases_in_file)}")
    print(f"  Concepts: {len(residuals)}")
    print(f"  Coverage: {100*len(residuals)/total_concepts:.1f}% of {total_concepts}")


if __name__ == "__main__":
    main()
