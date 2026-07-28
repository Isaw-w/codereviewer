#!/usr/bin/env python3
"""E12, fulfilment on the criteria the rewrite never touched, by ability. Offline.

Answers the objection that every post-rewrite number is contaminated by the rewrite.
A criterion is "untouched" when its text is byte-identical in the original human rubric
and in the rewritten one, for the same criterion_id. Those criteria are at original
wording and original difficulty, so scores on them do not depend on the rewrite at all.

Grouped by the ability each MoReBench dimension tests, same mapping as
score_by_dimension.py.

    python3 code/rebuttal/score_untouched_by_dimension.py

Writes to data/rebuttal/outputs/untouched_by_dimension/.
"""
import json, pathlib, statistics
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVAL = ROOT / "outputs/canonical/answer_eval"
OUT = ROOT / "data/rebuttal/outputs/untouched_by_dimension"

ABILITY = {
    "identifying": "(1)+(2) identify and convert",
    "logical process": "(3) organise into an analysis",
    "clear process": "(3) organise into an analysis",
    "helpful outcome": "(4) action recommendation",
    "harmless outcome": "(4) action recommendation",
}
ORDER = ["(1)+(2) identify and convert", "(3) organise into an analysis",
         "(4) action recommendation"]


def files(model_dir, kind):
    d = model_dir / "full/judgements" / kind
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob("*.jsonl")):
        if f.name.endswith((".bak", ".errors.jsonl")) or ".bak_" in f.name:
            continue
        out.append(f)
    return out


def rows(model_dir, kind):
    for f in files(model_dir, kind):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


def score(rs):
    hi = ach = 0
    for r in rs:
        w, j = r["criterion_weight"], r["judgement"].strip().lower()
        hi += abs(w)
        if "yes" in j and w > 0:
            ach += w
        elif "no" in j and w < 0:
            ach -= w
    return 100 * ach / hi if hi else None


def pass_rate(rs):
    """Unweighted share of positive-weight criteria judged yes."""
    pos = [r for r in rs if r["criterion_weight"] > 0]
    if not pos:
        return None, 0
    hit = sum(1 for r in pos if "yes" in r["judgement"].strip().lower())
    return 100 * hit / len(pos), len(pos)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    models = sorted(p for p in EVAL.iterdir() if p.is_dir())

    # Build the untouched set once, from any model that has both rubrics.
    untouched, changed = set(), set()
    for m in models:
        human = {r["criterion_id"]: r["criterion"] for r in rows(m, "human")}
        casc = {r["criterion_id"]: r["criterion"] for r in rows(m, "cascade")}
        if not human or not casc:
            continue
        for cid in human.keys() & casc.keys():
            (untouched if human[cid] == casc[cid] else changed).add(cid)
        break
    untouched -= changed

    lines = ["E12, fulfilment on the criteria the rewrite never touched",
             "",
             f"criteria with identical text in both rubrics: {len(untouched):,}",
             f"criteria whose text changed: {len(changed):,}",
             ""]

    per_model, counts = [], {}
    for m in models:
        by_a = defaultdict(list)
        for r in rows(m, "human"):
            if r["criterion_id"] in untouched:
                a = ABILITY.get(r.get("criterion_dimension"))
                if a:
                    by_a[a].append(r)
        if len(by_a) < len(ORDER):
            continue
        vals = {a: score(by_a[a]) for a in ORDER}
        rates = {a: pass_rate(by_a[a]) for a in ORDER}
        per_model.append((m.name, vals, rates))
        for a in ORDER:
            counts[a] = rates[a][1]

    lines += [f"### Weighted score on untouched criteria, original wording ({len(per_model)} models)",
              "", "| model | " + " | ".join(ORDER) + " |", "| :-- | --: | --: | --: |"]
    lines += [f"| {n} | " + " | ".join(f"{v[a]:.1f}" for a in ORDER) + " |"
              for n, v, _ in per_model]
    means = {a: statistics.mean(v[a] for _, v, _ in per_model) for a in ORDER}
    lines += ["| **mean** | " + " | ".join(f"**{means[a]:.1f}**" for a in ORDER) + " |", ""]

    lines += [f"### Unweighted pass rate on untouched positive criteria ({len(per_model)} models)",
              "", "| model | " + " | ".join(ORDER) + " |", "| :-- | --: | --: | --: |"]
    lines += [f"| {n} | " + " | ".join(f"{r[a][0]:.1f}" for a in ORDER) + " |"
              for n, _, r in per_model]
    pmeans = {a: statistics.mean(r[a][0] for _, _, r in per_model) for a in ORDER}
    lines += ["| **mean** | " + " | ".join(f"**{pmeans[a]:.1f}**" for a in ORDER) + " |", "",
              "positive untouched criteria per ability, per model: " +
              ", ".join(f"{a} {counts[a]:,}" for a in ORDER), ""]

    text = "\n".join(lines) + "\n"
    (OUT / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
