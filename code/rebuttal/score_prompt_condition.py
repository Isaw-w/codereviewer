#!/usr/bin/env python3
"""E10, prompting-condition robustness. Offline, no API calls.

The AC asks for "a robustness analysis across varying experimental conditions". One
condition already varies in the released data and is not reported anywhere in the paper:
the elicitation prompt. MoReBench ships judgements for two prompt types, `minimal` and
`richer`, for three models. This scores both under the official formula
(utils.calculate_score_for_a_task) on the tasks the two conditions share, so the
comparison is paired: same cases, same criteria, same weights, same judge.

    python3 code/rebuttal/score_prompt_condition.py

Writes results to data/rebuttal/outputs/prompt_condition/.
"""
import json, pathlib, statistics, sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils import calculate_score_for_a_task  # noqa: E402

JUDGEMENTS = ROOT / "model_resp_judgements"
OUT = ROOT / "data/rebuttal/outputs/prompt_condition"

MODELS = ["gpt-5.2", "gpt-5", "deepseek-chat-v3-0324"]
TMPL = "{model}_reasoning_high_{cond}_seed_0.jsonl"


def load(path):
    """task_id -> list of criterion rows."""
    by_task = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_task[r["task_id"]].append(r)
    return by_task


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["E10, prompting-condition robustness (minimal vs richer)",
             "Paired over shared task_ids. Official scoring formula, same judge, same rubric.", ""]
    rows = []

    for model in MODELS:
        paths = {c: JUDGEMENTS / TMPL.format(model=model, cond=c) for c in ("minimal", "richer")}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            lines.append(f"{model}: skipped, missing {missing}")
            continue

        data = {c: load(p) for c, p in paths.items()}
        shared = sorted(set(data["minimal"]) & set(data["richer"]))
        if not shared:
            lines.append(f"{model}: skipped, no shared task_ids")
            continue

        scores = {c: [calculate_score_for_a_task(data[c][t]) for t in shared] for c in data}
        deltas = [r - m for m, r in zip(scores["minimal"], scores["richer"])]
        mean_m, mean_r = statistics.mean(scores["minimal"]), statistics.mean(scores["richer"])
        d_mean = statistics.mean(deltas)
        # paired 95% CI on the difference
        se = statistics.stdev(deltas) / len(deltas) ** 0.5 if len(deltas) > 1 else 0.0
        lo, hi = d_mean - 1.96 * se, d_mean + 1.96 * se
        up = sum(1 for d in deltas if d > 0)
        down = sum(1 for d in deltas if d < 0)

        rows.append((model, len(shared), mean_m, mean_r, d_mean, lo, hi, up, down))
        lines += [
            f"{model}",
            f"  shared tasks           {len(shared)}"
            f"   (minimal only {len(set(data['minimal']) - set(shared))},"
            f" richer only {len(set(data['richer']) - set(shared))})",
            f"  minimal mean score     {mean_m:.2f}",
            f"  richer mean score      {mean_r:.2f}",
            f"  paired difference      {d_mean:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]",
            f"  cases richer is higher {up}, lower {down}, tied {len(shared) - up - down}",
            "",
        ]

    if rows:
        pooled = statistics.mean(r[4] for r in rows)
        lines += [f"Mean paired difference across {len(rows)} models: {pooled:+.2f}", ""]
        lines += ["| model | n | minimal | richer | diff | 95% CI |",
                  "| :-- | --: | --: | --: | --: | :-- |"]
        lines += [f"| {m} | {n} | {a:.1f} | {b:.1f} | {d:+.1f} | [{lo:+.1f}, {hi:+.1f}] |"
                  for m, n, a, b, d, lo, hi, _, _ in rows]

    text = "\n".join(lines) + "\n"
    (OUT / "results.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
