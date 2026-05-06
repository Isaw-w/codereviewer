"""
Convert AI-generated rubric files into a generations-style file.

This lets run_best_judge_on_responses.py score an AI rubric, treated as a
numbered response, against the human/expert rubric.

Inputs:
  --ai_rubric_file: AI rubric JSONL with TASK_ID, RUBRIC = AI-generated criteria
  --responses_file: model responses JSONL with idx, RUBRIC = expert rubric, metadata
  --case_filter_csv: optional CSV with ORIG_IDX column; if given, only those cases are kept
  --output_file: output JSONL path

Each output row keeps the expert RUBRIC and metadata from the responses file,
but replaces model_resp and thinking_trace with the AI rubric serialized as a
numbered list of criterion titles. TASK_ID is set equal to idx.
"""

import argparse
import csv
import json
import os
import re


def parse_task_id(tid):
    """Normalize TASK_ID to int. Accepts 19, '19', or 'case_019'."""
    if isinstance(tid, int):
        return tid
    s = str(tid).strip()
    m = re.fullmatch(r"case_(\d+)", s)
    if m:
        return int(m.group(1))
    return int(s)


def serialize_ai_rubric(rubric_items):
    """Serialize as numbered criterion titles only."""
    lines = []
    for i, item in enumerate(rubric_items, 1):
        lines.append(f"{i}. {item.get('title', '').strip()}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ai_rubric_file", "-r", required=True)
    parser.add_argument(
        "--responses_file",
        "-g",
        required=True,
        help="Model responses JSONL skeleton that provides expert RUBRIC and metadata",
    )
    parser.add_argument(
        "--case_filter_csv",
        "-c",
        default=None,
        help="Optional CSV with ORIG_IDX column",
    )
    parser.add_argument("--output_file", "-o", required=True)
    args = parser.parse_args()

    with open(args.ai_rubric_file) as f:
        ai_rubrics = [json.loads(l) for l in f if l.strip()]
    with open(args.responses_file) as f:
        responses = [json.loads(l) for l in f if l.strip()]

    ai_by_id = {parse_task_id(r["TASK_ID"]): r for r in ai_rubrics}
    resp_by_id = {parse_task_id(r["idx"]): r for r in responses}

    if args.case_filter_csv:
        with open(args.case_filter_csv) as f:
            target_ids = {int(row["ORIG_IDX"]) for row in csv.DictReader(f)}
        print(f"Filter CSV: {args.case_filter_csv} -> {len(target_ids)} target cases")
    else:
        target_ids = set(ai_by_id) & set(resp_by_id)

    keep = sorted(target_ids & set(ai_by_id) & set(resp_by_id))
    drop_no_ai = sorted(target_ids - set(ai_by_id))
    drop_no_resp = sorted(target_ids - set(resp_by_id))
    print(f"AI rubric file: {len(ai_by_id)} cases")
    print(f"Responses file: {len(resp_by_id)} cases")
    print(f"Keeping: {len(keep)}")
    if drop_no_ai:
        suffix = "..." if len(drop_no_ai) > 10 else ""
        print(f"  dropped (no AI rubric): {drop_no_ai[:10]}{suffix}")
    if drop_no_resp:
        suffix = "..." if len(drop_no_resp) > 10 else ""
        print(f"  dropped (no response): {drop_no_resp[:10]}{suffix}")

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    written = 0
    sample_lens = []
    with open(args.output_file, "w") as fw:
        for tid in keep:
            resp_row = resp_by_id[tid]
            ai_row = ai_by_id[tid]
            ai_text = serialize_ai_rubric(ai_row["RUBRIC"])

            new_row = dict(resp_row)
            new_row["TASK_ID"] = tid
            new_row["model_resp"] = ai_text
            new_row["thinking_trace"] = ai_text
            new_row["model"] = resp_row.get("model", "") + "__rubricasresp"
            new_row["input_tokens"] = 0
            new_row["output_tokens"] = 0
            new_row["reasoning_tokens"] = 0

            fw.write(json.dumps(new_row, ensure_ascii=False) + "\n")
            sample_lens.append(len(ai_text))
            written += 1
            if written <= 2:
                print(f"\n--- Sample row idx={tid} ---")
                print(f"  AI rubric items: {len(ai_row['RUBRIC'])}")
                print(f"  Expert rubric items: {len(resp_row['RUBRIC'])}")
                print(f"  model_resp len chars: {len(ai_text)}")
                print(f"  preview:\n{ai_text[:400]}")

    if sample_lens:
        median = sorted(sample_lens)[len(sample_lens) // 2]
        print(f"\nResponse text length stats over {written} rows:")
        print(f"  min/median/max: {min(sample_lens)}/{median}/{max(sample_lens)}")
    print(f"Wrote {written} rows to {args.output_file}")


if __name__ == "__main__":
    main()
