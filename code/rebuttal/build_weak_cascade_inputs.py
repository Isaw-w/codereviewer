#!/usr/bin/env python3
"""
Rebuttal run 3 (weak-response discriminative test), input builder.

Takes the weak-baseline 100-case response files (already merged with the
ORIGINAL human rubric for Finding 1) and swaps RUBRIC for the cascade-rewritten
human rubric, producing judge inputs in the standard model_resp format.

Local only — no API calls. The judge command is in run_weak_cascade_judge.sh.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CASCADE_RUBRIC = (
    REPO / "release_staging/data/paper_release/rubrics/rewrite/human_rubric_cascade_rewritten.jsonl"
)
WEAK_EVAL_DIR = (
    REPO / "data/rebuttal/outputs/finding1_weak_baseline_response_eval"
)
OUT_DIR = REPO / "data/rebuttal/outputs/weak_cascade_test/inputs"

WEAK_MODELS = [
    "llama31_8b_openrouter",
    "llama32_3b_openrouter",
    "mistral7b_v01_openrouter",
    "qwen25_7b_openrouter",
]


def load_cascade_rubrics() -> tuple[dict[str, object], dict[str, str]]:
    rubrics, dilemmas = {}, {}
    with CASCADE_RUBRIC.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rubrics[row["TASK_ID"]] = row["RUBRIC"]
            dilemmas[row["TASK_ID"]] = row["DILEMMA"]
    return rubrics, dilemmas


def main() -> None:
    cascade, dilemmas = load_cascade_rubrics()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model in WEAK_MODELS:
        src = WEAK_EVAL_DIR / model / "inputs" / f"{model}_response_under_human_rubric.jsonl"
        if not src.exists():
            raise FileNotFoundError(src)
        out_path = OUT_DIR / f"{model}_response_under_cascade_rubric.jsonl"
        n = 0
        with src.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                # Weak inputs carry the 500-case ORIG_IDX as TASK_ID; the cascade
                # rubric file keys cases as "case_{ORIG_IDX:03d}".
                tid = f"case_{int(row['TASK_ID']):03d}"
                if tid not in cascade:
                    raise KeyError(f"{model}: no cascade rubric for {tid}")
                if row["DILEMMA"][:200] != dilemmas[tid][:200]:
                    raise ValueError(f"{model}: DILEMMA mismatch for {tid}")
                row["TASK_ID"] = tid
                row["RUBRIC"] = cascade[tid]
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
        print(f"{model}: {n} rows -> {out_path}")


if __name__ == "__main__":
    main()
