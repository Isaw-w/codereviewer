#!/usr/bin/env python3
"""
Batch, blinded capture judge (rebuttal).

One judge call PER CASE (not per criterion), so a reasoning judge does one pass per
case instead of ~24 -> ~9x cheaper. The judge sees BOTH rubrics for the case,
neutrally labelled "Rubric A" and "Rubric B" with NO indication of which is the
expert (human) rubric and which is the model's. It returns, for each criterion in
the expert rubric, the number of the matching criterion in the other rubric (or
"none"). Matched -> yes (covered), none -> no.

Provenance mapping used internally (never shown to the judge):
  Rubric A = expert/human criteria (the ones we score)
  Rubric B = model rubric (model_resp text)
Optionally --swap_labels flips which letter the expert rubric gets, per case by
hash, to remove any positional cue.

Output rows match the per-criterion schema (task_id, criterion_id, criterion,
criterion_weight, judgement) plus matched_other_item for auditing.

Run yourself (network):
  LAB_OPENROUTER_KEY=... python3 batch_blind_capture.py \
    -i <capture_input.jsonl> -jm anthropic/claude-opus-4.8 -o <out_dir>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "release_staging" / "code"))
from utils import setup_client, get_judge_response  # noqa: E402

def parse_rubric(raw):
    return raw if isinstance(raw, list) else json.loads(raw)


def build_prompt(expert_items: list[dict], other_text: str, expert_is_A: bool) -> str:
    """Two neutrally-labelled rubrics. No word reveals which is expert vs model.
    We ask about coverage of the expert list, referring to it only by its letter."""
    expert_block = "\n".join(f"{i+1}. {c['title']}" for i, c in enumerate(expert_items))
    e = "A" if expert_is_A else "B"   # letter the expert rubric is shown under
    o = "B" if expert_is_A else "A"   # the other (model) rubric's letter
    if expert_is_A:
        blocks = f"Rubric A:\n{expert_block}\n\nRubric B:\n{other_text}\n\n"
    else:
        blocks = f"Rubric A:\n{other_text}\n\nRubric B:\n{expert_block}\n\n"
    instr = (
        "Below are two rubrics, Rubric A and Rubric B, each a list of criteria for "
        "evaluating responses to the same moral dilemma. "
        f"For each numbered criterion in Rubric {e}, decide whether any criterion in "
        f"Rubric {o} engages or captures the same underlying moral consideration, "
        "regardless of the specific actions, examples, and particular details. Judge "
        "whether the moral consideration is addressed, not whether the specific action, "
        "example, or wording matches.\n\n"
        f"Return ONLY a JSON object mapping each Rubric {e} criterion number (as a "
        f"string) to a list of the single best-matching criterion in Rubric {o} "
        "(include a second only if another clearly captures the same consideration "
        "equally well - at most two), or \"none\" if no criterion there captures it. "
        'Example: {"1": [4], "2": "none", "3": [2, 5]}. No other text.'
    )
    return blocks + instr


def _json_candidates(text: str):
    """Yield candidate JSON-object substrings, most-likely first.
    Handles reasoning models that emit prose (with stray braces) then the JSON."""
    if not text:
        return
    # 1) fenced ```json ... ``` block
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        yield m.group(1)
    # 2) every balanced {...} object, scanned last-to-first (answer usually trails)
    stack = []
    spans = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            if not stack:
                spans.append(text[start:i + 1])
    for s in reversed(spans):
        yield s
    # 3) greedy fallback
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        yield m.group(0)


def parse_map(text: str, n: int) -> dict[int, object] | None:
    for cand in _json_candidates(text):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        out = {}
        for k, v in obj.items():
            try:
                ki = int(str(k).strip())
            except ValueError:
                continue
            # Normalise value to a sorted list of ints, or "none".
            if isinstance(v, bool):
                out[ki] = "none"
            elif isinstance(v, list):
                nums = sorted({int(x) for x in v if str(x).strip().lstrip("-").isdigit()})
                out[ki] = nums[:2] if nums else "none"   # cap at 2 best
            elif isinstance(v, (int, float)):
                out[ki] = [int(v)]
            elif str(v).strip().lower() in {"none", "no", "null", ""}:
                out[ki] = "none"
            else:
                nums = sorted({int(x) for x in re.findall(r"\d+", str(v))})
                out[ki] = nums[:2] if nums else "none"   # cap at 2 best
        if len(out) >= max(1, n // 2):
            return out
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-jm", "--judge_model", required=True)
    ap.add_argument("-a", "--api_key", default="LAB_OPENROUTER_KEY")
    ap.add_argument("-o", "--output_dir", required=True)
    ap.add_argument("--reasoning_effort", default="high", choices=["low", "medium", "high"])
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--swap_labels", action="store_true",
                    help="Per-case, give the expert rubric label A or B by hash (removes positional cue)")
    args = ap.parse_args()

    key = os.environ.get(args.api_key, args.api_key)
    client = setup_client("openrouter", key)

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"blind_{Path(args.input).name}"
    raw_path = out_dir / f"raw_{Path(args.input).name}"
    fail_path = out_dir / f"failed_{Path(args.input).name}"

    written = 0
    with out_path.open("w", encoding="utf-8") as fout, \
         raw_path.open("w", encoding="utf-8") as fraw, \
         fail_path.open("w", encoding="utf-8") as ffail:
        for row in rows:
            expert = parse_rubric(row["RUBRIC"])
            other_text = row["model_resp"]
            expert_is_A = True
            if args.swap_labels:
                expert_is_A = (hash(str(row["TASK_ID"])) % 2 == 0)
            prompt = build_prompt(expert, other_text, expert_is_A)
            vmap = None
            for attempt in range(2):
                content, ti, to = get_judge_response(
                    client, args.judge_model,
                    prompt if attempt == 0 else prompt + "\n\nReturn the JSON object only, nothing else.",
                    max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort,
                )
                fraw.write(json.dumps({"task_id": row["TASK_ID"], "attempt": attempt,
                                       "content": content}, ensure_ascii=False) + "\n")
                vmap = parse_map(content, len(expert))
                if vmap is not None:
                    break
            if vmap is None:
                # Do NOT silently score all-none (that corrupts the number). Skip + log.
                print(f"  [FAIL] {row['TASK_ID']} unparseable after retry; SKIPPED (not scored)", flush=True)
                ffail.write(json.dumps({"task_id": row["TASK_ID"], "n_criteria": len(expert)}) + "\n")
                continue
            covered = 0
            for i, c in enumerate(expert):
                matched = vmap.get(i + 1, "none")
                is_yes = isinstance(matched, list) and len(matched) > 0
                covered += is_yes
                fout.write(json.dumps({
                    "task_id": row["TASK_ID"],
                    "criterion_id": c["id"],
                    "criterion": c["title"],
                    "criterion_weight": c["weight"],
                    "judgement": "yes" if is_yes else "no",
                    "matched_other_items": matched,
                    "expert_label": "A" if expert_is_A else "B",
                    "judge_input_tokens": ti,
                    "judge_output_tokens": to,
                }, ensure_ascii=False) + "\n")
                written += 1
            print(f"  {row['TASK_ID']}: {covered}/{len(expert)} covered", flush=True)
    print(f"wrote {written} rows -> {out_path}")


if __name__ == "__main__":
    main()
