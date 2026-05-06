"""Score rubric-as-response capture judgements.

The input is a judgement JSONL produced by run_best_judge_on_responses.py on
rows created by convert_rubric_to_response.py. The scoring rule is:

  yes -> abs(criterion_weight) credit
  no  -> 0 credit

This measures whether a model-written rubric captures the same underlying
evaluative points as the human rubric criteria. It is not the original
MoReBench fulfillment score for an answer.
"""

import argparse
import json
import os
from collections import Counter, defaultdict


def norm_judgement(value):
    text = str(value).strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return text


def task_id(row):
    for key in ("task_id", "TASK_ID", "idx"):
        if key in row:
            return str(row[key])
    return "unknown"


def weight(row):
    try:
        return int(row.get("criterion_weight", 0))
    except (TypeError, ValueError):
        return 0


def dimension(row):
    return str(row.get("criterion_dimension", "unknown")).lower()


def add_score(bucket, row):
    w = abs(weight(row))
    if w == 0:
        return
    bucket["total_weight"] += w
    bucket["total"] += 1
    if norm_judgement(row.get("judgement", "")) == "yes":
        bucket["yes_weight"] += w
        bucket["yes"] += 1


def pct(bucket):
    if bucket["total_weight"] == 0:
        return None
    return round(100 * bucket["yes_weight"] / bucket["total_weight"], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--model_name", default=None)
    args = parser.parse_args()

    with open(args.judgements) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    by_task = defaultdict(lambda: {"yes_weight": 0, "total_weight": 0, "yes": 0, "total": 0})
    by_dim = defaultdict(lambda: {"yes_weight": 0, "total_weight": 0, "yes": 0, "total": 0})
    by_weight = defaultdict(lambda: {"yes_weight": 0, "total_weight": 0, "yes": 0, "total": 0})
    overall = {"yes_weight": 0, "total_weight": 0, "yes": 0, "total": 0}
    judgements = Counter()

    for row in rows:
        add_score(overall, row)
        add_score(by_task[task_id(row)], row)
        add_score(by_dim[dimension(row)], row)
        add_score(by_weight[str(weight(row))], row)
        judgements[norm_judgement(row.get("judgement", ""))] += 1

    task_scores = [pct(v) for v in by_task.values() if pct(v) is not None]
    mean_task_score = round(sum(task_scores) / len(task_scores), 1) if task_scores else None

    summary = {
        "model": args.model_name,
        "scoring_rule": "rubric-list capture scoring: yes receives abs(weight) credit for positive and negative criteria",
        "rows": len(rows),
        "tasks": len(by_task),
        "overall_weighted_by_criterion": pct(overall),
        "overall_mean_of_task_scores": mean_task_score,
        "judgement_counts": dict(judgements),
        "criterion_dimension": {k: pct(v) for k, v in sorted(by_dim.items())},
        "criterion_dimension_counts": {k: dict(v) for k, v in sorted(by_dim.items())},
        "criterion_weight": {k: pct(v) for k, v in sorted(by_weight.items(), key=lambda kv: int(kv[0]))},
        "criterion_weight_counts": {k: dict(v) for k, v in sorted(by_weight.items(), key=lambda kv: int(kv[0]))},
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
