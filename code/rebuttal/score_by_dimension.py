#!/usr/bin/env python3
"""E11, fulfilment by criterion dimension. Offline, no API calls.

MoReBench labels every criterion with a `criterion_dimension`. Those labels map onto the
four abilities the paper distinguishes:

    identifying                      -> (1) identify morally relevant facts
                                        (2) convert them into moral considerations
    logical process, clear process   -> (3) organise considerations into a coherent analysis
    helpful outcome, harmless outcome-> (4) issue an action recommendation

So the released judgements already say whether models are competent at (3) and (4), not
only at (1) and (2). This scores each dimension separately, under the original human rubric
and under the rewritten one, using the official weighting restricted to that dimension:

    100 * sum(w if judged yes and w > 0, |w| if judged no and w < 0) / sum(|w|)

    python3 code/rebuttal/score_by_dimension.py

Writes results to data/rebuttal/outputs/by_dimension/.
"""
import json, pathlib, statistics
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVAL = ROOT / "outputs/canonical/answer_eval"
OUT = ROOT / "data/rebuttal/outputs/by_dimension"

ABILITY = {
    "identifying": "(1)+(2) identify and convert",
    "logical process": "(3) organise into an analysis",
    "clear process": "(3) organise into an analysis",
    "helpful outcome": "(4) action recommendation",
    "harmless outcome": "(4) action recommendation",
}
ORDER = ["(1)+(2) identify and convert", "(3) organise into an analysis",
         "(4) action recommendation"]
RUBRICS = {"human": "original human rubric", "cascade": "rewritten human rubric"}


def score(rows):
    """Official formula restricted to a set of criterion rows."""
    hi = ach = 0
    for r in rows:
        w, j = r["criterion_weight"], r["judgement"].strip().lower()
        hi += abs(w)
        if "yes" in j and w > 0:
            ach += w
        elif "no" in j and w < 0:
            ach -= w
    return 100 * ach / hi if hi else None


def load(model_dir, kind):
    d = model_dir / "full/judgements" / kind
    if not d.is_dir():
        return None
    files = [f for f in sorted(d.glob("*.jsonl")) if not f.name.endswith((".bak", ".errors.jsonl"))]
    files = [f for f in files if ".bak_" not in f.name]
    if not files:
        return None
    by_ability = defaultdict(list)
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                a = ABILITY.get(r.get("criterion_dimension"))
                if a:
                    by_ability[a].append(r)
    return by_ability


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = sorted(p for p in EVAL.iterdir() if p.is_dir())
    lines = ["E11, fulfilment by criterion dimension",
             "MoReBench's own dimension labels, grouped by the ability each one tests.", ""]
    table = {}

    for kind, label in RUBRICS.items():
        rows = []
        for m in models:
            by_a = load(m, kind)
            if not by_a:
                continue
            vals = {a: score(by_a[a]) for a in ORDER if by_a.get(a)}
            if len(vals) == len(ORDER):
                rows.append((m.name, vals, {a: len(by_a[a]) for a in ORDER}))
        if not rows:
            continue
        table[kind] = rows
        lines += [f"### Under the {label} ({len(rows)} models)", "",
                  "| model | " + " | ".join(ORDER) + " |",
                  "| :-- | --: | --: | --: |"]
        lines += [f"| {n} | " + " | ".join(f"{v[a]:.1f}" for a in ORDER) + " |" for n, v, _ in rows]
        means = {a: statistics.mean(v[a] for _, v, _ in rows) for a in ORDER}
        lines += ["| **mean** | " + " | ".join(f"**{means[a]:.1f}**" for a in ORDER) + " |", "",
                  "criteria per ability: " +
                  ", ".join(f"{a} {sum(c[a] for _, _, c in rows):,}" for a in ORDER), ""]

    if "human" in table and "cascade" in table:
        h = {n: v for n, v, _ in table["human"]}
        c = {n: v for n, v, _ in table["cascade"]}
        both = sorted(set(h) & set(c))
        lines += [f"### Change from the original to the rewritten rubric ({len(both)} models)", "",
                  "| ability | original | rewritten | change |", "| :-- | --: | --: | --: |"]
        for a in ORDER:
            mh = statistics.mean(h[n][a] for n in both)
            mc = statistics.mean(c[n][a] for n in both)
            lines.append(f"| {a} | {mh:.1f} | {mc:.1f} | {mc - mh:+.1f} |")
        lines.append("")

    text = "\n".join(lines) + "\n"
    (OUT / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
