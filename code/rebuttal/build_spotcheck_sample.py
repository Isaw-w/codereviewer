#!/usr/bin/env python3
"""
Rebuttal run 2: build the human-expert spot-check sheet.

Samples 100 changed criteria (seed 42) from the cascade rewrite audit and writes
a CSV for an ethics-trained annotator with two blank judgment columns:
  intent_preserved      — does the rewrite preserve the moral intent of the original?
  satisfies_generality  — does the rewrite satisfy MoReBench's generality requirement?

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
OUT = REPO / "data/rebuttal/outputs/spotcheck/rewrite_spotcheck_sample_seed42.csv"

N_SAMPLE = 100
SEED = 42


def main() -> None:
    changed = []
    with AUDIT.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["changed"]:
                changed.append(row)
    print(f"changed criteria: {len(changed)}")

    dilemmas = {}
    with RUBRIC.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            dilemmas[row["TASK_ID"]] = row["DILEMMA"]

    rng = random.Random(SEED)
    sample = rng.sample(changed, N_SAMPLE)
    sample.sort(key=lambda r: (r["task_id"], r["criterion_id"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "task_id", "criterion_id", "dilemma_excerpt",
            "original_criterion", "rewritten_criterion",
            "intent_preserved (Y/N)", "satisfies_generality (Y/N)", "notes",
        ])
        for r in sample:
            w.writerow([
                r["task_id"], r["criterion_id"],
                dilemmas[r["task_id"]][:400],
                r["original_title"], r["final_title"],
                "", "", "",
            ])
    print(f"wrote {N_SAMPLE} rows -> {OUT}")


if __name__ == "__main__":
    main()
