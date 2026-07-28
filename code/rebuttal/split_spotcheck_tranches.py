#!/usr/bin/env python3
"""
Split the seed-42 spot-check sample into two nested tranches of 50.

The full 100-row sheet (build_spotcheck_sample.py) is kept as-is. This script
reproduces the same seed-42 draw, keeps it in DRAW order rather than sorted
order, and cuts it in half:

  tranche_a  first 50 drawn  — annotate this one
  tranche_b  last 50 drawn   — hold in reserve

Tranche A is itself a valid uniform random sample of the changed criteria, and
A + B is exactly the documented seed-42 sample of 100, so extending later needs
no re-sampling and no change to the reported sampling procedure.

Local only — no API calls.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "release_staging/data/paper_release/rubrics/rewrite/cascade_rewrite_audit.jsonl"
RUBRIC = REPO / "release_staging/data/paper_release/rubrics/rewrite/human_rubric_cascade_rewritten.jsonl"
OUT_DIR = REPO / "data/rebuttal/outputs/spotcheck"

N_SAMPLE = 100
SEED = 42
HEADER = [
    "task_id", "criterion_id", "dilemma_excerpt",
    "original_criterion", "rewritten_criterion",
    "intent_preserved (Y/N)", "satisfies_generality (Y/N)", "notes",
]


def main() -> None:
    changed = []
    with AUDIT.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["changed"]:
                    changed.append(row)

    dilemmas = {}
    with RUBRIC.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                dilemmas[row["TASK_ID"]] = row["DILEMMA"]

    # Same draw as build_spotcheck_sample.py, but kept in draw order.
    sample = random.Random(SEED).sample(changed, N_SAMPLE)

    for name, part in (("a", sample[:50]), ("b", sample[50:])):
        part = sorted(part, key=lambda r: (r["task_id"], r["criterion_id"]))
        out = OUT_DIR / f"rewrite_spotcheck_seed42_tranche_{name}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            for r in part:
                w.writerow([
                    r["task_id"], r["criterion_id"],
                    dilemmas[r["task_id"]][:400],
                    r["original_title"], r["final_title"],
                    "", "", "",
                ])
        n_dil = len({r["task_id"] for r in part})
        print(f"tranche {name}: {len(part)} criteria over {n_dil} dilemmas -> {out.name}")

    keys = lambda rs: {(r["task_id"], r["criterion_id"]) for r in rs}
    assert not keys(sample[:50]) & keys(sample[50:]), "tranches overlap"
    assert len(keys(sample)) == N_SAMPLE, "duplicate keys in draw"
    print("tranches are disjoint and together reproduce the seed-42 sample of 100")


if __name__ == "__main__":
    main()
