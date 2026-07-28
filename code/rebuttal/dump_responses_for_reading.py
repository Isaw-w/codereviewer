#!/usr/bin/env python3
"""Dump side-by-side model answers for a few cases, so they can be read rather than scored.

Offline, no API. For each selected case it writes one markdown file containing the dilemma,
then every chosen model's full answer with the score that answer received under the
rewritten human rubric, ordered worst to best. The point is to see what a score of 89 on a
4B model actually looks like next to a frontier model on the same case.

    python3 code/rebuttal/dump_responses_for_reading.py
    python3 ... --n 8 --models gemma3_4b_openrouter opus46 claude_sonnet4

Writes to data/rebuttal/outputs/read_responses/.
"""
import argparse, json, pathlib, random
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVAL = ROOT / "outputs/canonical/answer_eval"
OUT = ROOT / "data/rebuttal/outputs/read_responses"

DEFAULT = ["gemma3_4b_openrouter", "qwen35_9b_openrouter", "opus46",
           "gpt54_openrouter", "claude_sonnet4"]


def judgement_file(model, kind="cascade"):
    d = EVAL / model / "full/judgements" / kind
    for f in sorted(d.glob("*.jsonl")):
        if ".bak_" not in f.name and not f.name.endswith(".errors.jsonl"):
            return f
    return None


def load_scored(model):
    """task_id -> (score, response, [(criterion, weight, judgement), ...])"""
    f = judgement_file(model)
    if f is None:
        return {}
    by_task = defaultdict(list)
    resp = {}
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_task[r["task_id"]].append(r)
            resp[r["task_id"]] = r["response"]
    out = {}
    for t, rs in by_task.items():
        hi = ach = 0
        for r in rs:
            w, j = r["criterion_weight"], r["judgement"].strip().lower()
            hi += abs(w)
            if "yes" in j and w > 0:
                ach += w
            elif "no" in j and w < 0:
                ach -= w
        out[t] = (100 * ach / hi if hi else 0.0, resp[t], rs)
    return out


def load_dilemmas(model):
    f = EVAL / model / "full/inputs" / f"{model}_full_under_cascade_human_rubric.jsonl"
    if not f.exists():
        return {}
    d = {}
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                d[r["TASK_ID"]] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="how many cases to dump")
    ap.add_argument("--models", nargs="+", default=DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    scored = {m: load_scored(m) for m in args.models}
    scored = {m: s for m, s in scored.items() if s}
    if not scored:
        raise SystemExit("no judgement files found")

    shared = set.intersection(*(set(s) for s in scored.values()))
    cases = sorted(shared)
    random.Random(args.seed).shuffle(cases)
    cases = cases[:args.n]

    dilemmas = load_dilemmas(args.models[0])
    index = ["# Responses to read", "",
             f"{len(cases)} cases, seed {args.seed}, models: {', '.join(scored)}.",
             "Scores are under the rewritten human rubric. Each file lists answers worst first.", ""]

    for t in cases:
        d = dilemmas.get(t, {})
        parts = [f"# {t}", ""]
        if d.get("DILEMMA"):
            parts += [f"Source: {d.get('DILEMMA_SOURCE','?')} · type: {d.get('DILEMMA_TYPE','?')}"
                      f" · role: {d.get('ROLE_DOMAIN','?')}", "", "## The dilemma", "",
                      d["DILEMMA"].strip(), ""]
        ranked = sorted(scored, key=lambda m: scored[m][t][0])
        parts += ["## Scores on this case", ""]
        parts += [f"- {m}: {scored[m][t][0]:.1f}" for m in ranked] + [""]
        for m in ranked:
            s, resp, rs = scored[m][t]
            missed = [r["criterion"] for r in rs
                      if r["criterion_weight"] > 0 and "yes" not in r["judgement"].strip().lower()]
            parts += [f"## {m} — {s:.1f}", "", resp.strip(), ""]
            if missed:
                parts += [f"*Missed {len(missed)} of the positive criteria, including:*", ""]
                parts += [f"- {c}" for c in missed[:5]] + [""]
        (OUT / f"{t}.md").write_text("\n".join(parts))
        index.append(f"- [{t}]({t}.md) — " +
                     ", ".join(f"{m.split('_')[0]} {scored[m][t][0]:.0f}" for m in ranked))

    (OUT / "README.md").write_text("\n".join(index) + "\n")
    print(f"wrote {len(cases)} case files to {OUT}")
    for line in index[4:]:
        print(" ", line)


if __name__ == "__main__":
    main()
