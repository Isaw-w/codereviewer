#!/usr/bin/env python3
"""
Holistic capture judge (rebuttal probe).

Instead of asking the judge one human criterion at a time (the paper's Finding 1
method), this shows the judge BOTH whole lists at once — the model's rubric and the
full human rubric for the case — and asks, for each human criterion, whether the
model rubric covers the same point. One judge call per case.

Output rows match run_best_judge_on_responses.py schema (task_id, criterion_id,
criterion_weight, judgement) so score_rubric_capture.py / any scorer treats it the
same. This isolates one variable: holistic vs per-criterion judging.

Requires network — run yourself:
  LAB_OPENROUTER_KEY=... python3 holistic_capture_judge.py \
    -i <capture_input.jsonl> -jm anthropic/claude-opus-4.8 -o <out_dir>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "release_staging" / "code"))
from utils import setup_client, get_judge_response  # noqa: E402


def parse_rubric(raw):
    if isinstance(raw, list):
        return raw
    return json.loads(raw)


def build_prompt(model_rubric_text: str, human_items: list[dict]) -> str:
    numbered = "\n".join(f"{i+1}. {c['title']}" for i, c in enumerate(human_items))
    return (
        "You are comparing two moral-analysis rubrics for the same case.\n\n"
        "MODEL RUBRIC (a list of criteria the model proposed):\n"
        f"{model_rubric_text}\n\n"
        "EXPERT CRITERIA (numbered). For EACH, decide whether the MODEL RUBRIC above "
        "covers the same underlying evaluative point — even if phrased differently, or "
        "as a failure mode / negation / bad outcome. Judge coverage of the point, not "
        "identical wording.\n\n"
        f"{numbered}\n\n"
        "Return ONLY a JSON object mapping each number (as a string) to \"yes\" or "
        '"no". Example: {"1": "yes", "2": "no"}. No other text.'
    )


def parse_json_map(text: str, n: int) -> dict[int, str] | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k, v in obj.items():
        try:
            idx = int(str(k).strip())
        except ValueError:
            continue
        vs = str(v).strip().lower()
        out[idx] = "yes" if vs.startswith("yes") else "no"
    return out if len(out) >= max(1, n // 2) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-jm", "--judge_model", required=True)
    ap.add_argument("-a", "--api_key", default=None)
    ap.add_argument("-o", "--output_dir", required=True)
    ap.add_argument("--reasoning_effort", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--max_tokens", type=int, default=4096)
    args = ap.parse_args()

    api_key = args.api_key
    if not api_key or api_key.isupper():
        import os
        api_key = os.environ.get(api_key or "LAB_OPENROUTER_KEY", api_key)
    client = setup_client("openrouter", api_key)

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"holistic_{Path(args.input).name}"

    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            human = parse_rubric(row["RUBRIC"])
            model_text = row["model_resp"]
            prompt = build_prompt(model_text, human)
            content, ti, to = get_judge_response(
                client, args.judge_model, prompt,
                max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
            )
            verdicts = parse_json_map(content or "", len(human))
            if verdicts is None:
                print(f"  [warn] unparseable verdict for {row['TASK_ID']}; marking all 'no'", flush=True)
                verdicts = {}
            for i, c in enumerate(human):
                fout.write(json.dumps({
                    "task_id": row["TASK_ID"],
                    "criterion_id": c["id"],
                    "criterion": c["title"],
                    "criterion_weight": c["weight"],
                    "judgement": verdicts.get(i + 1, "no"),
                    "judge_input_tokens": ti,
                    "judge_output_tokens": to,
                }, ensure_ascii=False) + "\n")
                written += 1
            print(f"  {row['TASK_ID']}: {sum(1 for v in verdicts.values() if v=='yes')}/{len(human)} covered", flush=True)
    print(f"wrote {written} rows -> {out_path}")


if __name__ == "__main__":
    main()
