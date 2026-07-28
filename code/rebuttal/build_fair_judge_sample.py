#!/usr/bin/env python3
"""
Build a representative sample for judge selection (Step A).

For each of the 7 Finding-1 capture models, pick 1 case per DILEMMA_TYPE
(expert_case / long_case / short_case), seeded -> 3 cases x 7 models = 21
capture rows in ONE combined input. Each row keeps its own model's rubric and
'__rubricasresp' marker, so run_best_judge_on_responses.py handles the mix.

This sample stands in for the whole (7 models x 100 cases x 3 case types) so we
can score candidate judges against the Opus+GPT-5.6-sol reference on it.

Local only.
"""
from __future__ import annotations
import json, random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CAP = REPO / "release_staging/data/paper_release/finding1/rubric_as_response_capture"
OUT = REPO / "data/rebuttal/outputs/judge_select/inputs"
SEED = 42

MODELS = {
    "gemini25_pro": CAP / "gemini25_pro/inputs/gemini25_pro_finding1_rubricasresp_nointro.jsonl",
    "gpt54":        CAP / "gpt54/inputs/gpt54_finding1_rubricasresp_nointro.jsonl",
    "opus46":       CAP / "opus46/inputs/opus46_finding1_rubricasresp_nointro.jsonl",
    "llama31_8b":   CAP / "small_model_baselines/llama31_8b_openrouter/inputs/llama31_8b_openrouter_finding1_rubricasresp_nointro.jsonl",
    "llama32_3b":   CAP / "small_model_baselines/llama32_3b_openrouter/inputs/llama32_3b_openrouter_finding1_rubricasresp_nointro.jsonl",
    "mistral7b":    CAP / "small_model_baselines/mistral7b_v01_openrouter/inputs/mistral7b_v01_openrouter_finding1_rubricasresp_nointro.jsonl",
    "qwen25_7b":    CAP / "small_model_baselines/qwen25_7b_openrouter/inputs/qwen25_7b_openrouter_finding1_rubricasresp_nointro.jsonl",
}
TYPES = ["expert_case", "long_case", "short_case"]


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    combined, manifest = [], []
    for i, (tag, path) in enumerate(MODELS.items()):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        by_type = {t: [r for r in rows if r.get("DILEMMA_TYPE") == t] for t in TYPES}
        # per-model rng offset so different models draw different cases
        mrng = random.Random(SEED + i)
        for t in TYPES:
            pool = by_type[t]
            if not pool:
                continue
            pick = mrng.choice(pool)
            combined.append(pick)
            manifest.append({"model_tag": tag, "task_id": pick["TASK_ID"],
                             "dilemma_type": t, "n_criteria": len(pick["RUBRIC"])})

    out_path = OUT / "fair_judge_sample.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "fair_judge_sample_manifest.json").write_text(json.dumps(manifest, indent=1))

    ncrit = sum(m["n_criteria"] for m in manifest)
    print(f"wrote {len(combined)} rows ({ncrit} criteria) -> {out_path}")
    from collections import Counter
    print("by model:", dict(Counter(m["model_tag"] for m in manifest)))
    print("by type :", dict(Counter(m["dilemma_type"] for m in manifest)))
    print("distinct cases:", len({m["task_id"] for m in manifest}))


if __name__ == "__main__":
    main()
