"""
Transform H=No/M=Yes matched pairs into run_best_judge_on_responses.py input format.

Reads the paired criteria file and the opus46 response file, produces a JSONL
where each row is one case with a RUBRIC list containing both human and model
criteria.  Criterion IDs encode source (h/m) so results can be split later.

Usage:
    python prepare_judge_input_from_pairs.py \
        --pairs  .../opus46_h_no_m_yes.jsonl \
        --responses .../claude-opus-4.6_reasoning_high_seed_0.jsonl \
        --output .../opus46_h_no_m_yes_judge_input.jsonl \
        --judgement_type model_resp
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", "-p", required=True,
                        help="Path to opus46_h_no_m_yes.jsonl")
    parser.add_argument("--responses", "-r", required=True,
                        help="Path to opus46 response file (generations)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output JSONL for run_best_judge_on_responses.py")
    parser.add_argument("--judgement_type", "-jt", default="model_resp",
                        choices=["model_resp", "thinking_trace"])
    args = parser.parse_args()

    # --- Load responses, index by idx ---
    resp_by_idx = {}
    with open(args.responses) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            resp_by_idx[d["idx"]] = d
    print(f"Loaded {len(resp_by_idx)} responses")

    # --- Load pairs, group by case_id ---
    pairs = [json.loads(l) for l in Path(args.pairs).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(pairs)} pairs")

    grouped = defaultdict(list)
    for p in pairs:
        grouped[p["case_id"]].append(p)
    print(f"Grouped into {len(grouped)} cases")

    # --- Build judge input rows ---
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0
    with open(out_path, "w") as fout:
        for case_id in sorted(grouped.keys()):
            idx = int(case_id.replace("case_", ""))
            resp = resp_by_idx.get(idx)
            if resp is None:
                print(f"WARNING: no response for {case_id} (idx={idx}), skipping")
                skipped += 1
                continue

            case_pairs = grouped[case_id]
            rubric = []
            for p in case_pairs:
                # Human criterion
                rubric.append({
                    "id": f"h_{p['h_atom']}_{case_id}",
                    "title": p["h_text"],
                    "weight": p["h_weight"],
                    "annotations": {"rubric_dimension": "human_criterion"}
                })
                # Model criterion
                rubric.append({
                    "id": f"m_{p['m_atom']}_{case_id}",
                    "title": p["m_text"],
                    "weight": p["m_weight"],
                    "annotations": {"rubric_dimension": "model_criterion"}
                })

            row = {
                "TASK_ID": case_id,
                "DILEMMA": resp.get("DILEMMA", ""),
                "DILEMMA_SOURCE": resp.get("DILEMMA_SOURCE", ""),
                "DILEMMA_TYPE": resp.get("DILEMMA_TYPE", ""),
                "RUBRIC": rubric,
                "model_resp": resp.get("model_resp", ""),
                "thinking_trace": resp.get("thinking_trace", ""),
                "input_tokens": resp.get("input_tokens", 0),
                "output_tokens": resp.get("output_tokens", 0),
                "reasoning_tokens": resp.get("reasoning_tokens", 0),
                "model": resp.get("model", "claude-opus-4.6"),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"\nDone: {written} cases written to {out_path}")
    if skipped:
        print(f"WARNING: {skipped} cases skipped (no matching response)")

    # --- Quick validation ---
    total_criteria = sum(len(grouped[c]) * 2 for c in grouped
                         if int(c.replace("case_", "")) in resp_by_idx)
    print(f"Total criteria in output: {total_criteria} ({total_criteria//2} h + {total_criteria//2} m)")
    print(f"\nRun judge with:")
    print(f"  python run_best_judge_on_responses.py \\")
    print(f"    -i {out_path} \\")
    print(f"    -jt {args.judgement_type} \\")
    print(f"    -es {written} \\")
    print(f"    -ak $LAB_OPENROUTER_KEY")


if __name__ == "__main__":
    main()
