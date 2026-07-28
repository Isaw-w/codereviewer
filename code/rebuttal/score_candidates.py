#!/usr/bin/env python3
"""
Score candidate judges against the Opus+GPT-5.6-sol frontier consensus.

frontier_consensus.json: {criterion_id: bool}  (True = both frontier judges said
yes/covered, False = both said no). Only criteria where the two frontier judges
AGREE are in it (463 of 502) - the clean reference.

For each candidate we report, on the shared criteria:
  n            criteria scored that are in the consensus
  acc          fraction the candidate matches the consensus label   <- the number
  capture      candidate's own yes-rate on those criteria
  ref_capture  consensus yes-rate on those criteria (fixed, ~95.7%)
  FN/FP        misses (consensus yes, cand no) / false hits (consensus no, cand yes)
  bias         +lenient (says yes more than ref) / -conservative
GPT-OSS is the incumbent; a good second judge beats its acc while staying disjoint.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve()
SEL = HERE.parents[2] / "data/rebuttal/outputs/judge_select"
CONS = json.load(open(SEL / "frontier_consensus.json"))  # criterion_id -> bool


def yn(j: str) -> bool | None:
    s = str(j).strip().lower().lstrip("*# ")
    if s.startswith("yes"):
        return True
    if s.startswith("no"):
        return False
    return None


def score(path: Path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    n = tp = tn = fp = fn = unparsed = 0
    cand_yes = 0
    for r in rows:
        cid = r["criterion_id"]
        if cid not in CONS:
            continue
        v = yn(r["judgement"])
        if v is None:
            unparsed += 1
            continue
        ref = CONS[cid]
        n += 1
        cand_yes += v
        if ref and v:      tp += 1
        elif ref and not v: fn += 1
        elif not ref and v: fp += 1
        else:               tn += 1
    acc = (tp + tn) / n if n else 0.0
    return dict(file=path.parent.name, n=n, acc=acc,
                capture=cand_yes / n if n else 0.0,
                fn=fn, fp=fp, unparsed=unparsed)


def main(argv):
    ref_yes = sum(1 for v in CONS.values() if v) / len(CONS)
    print(f"consensus: {len(CONS)} criteria, ref yes-rate {ref_yes:.1%}\n")
    hdr = f"{'candidate':<20}{'n':>5}{'acc':>8}{'capture':>9}{'FN':>5}{'FP':>5}{'unparsed':>10}"
    print(hdr); print("-" * len(hdr))
    for p in argv:
        p = Path(p)
        f = p if p.is_file() else next(p.glob("*fair_judge_sample.jsonl"))
        s = score(f)
        print(f"{s['file']:<20}{s['n']:>5}{s['acc']:>8.1%}{s['capture']:>9.1%}"
              f"{s['fn']:>5}{s['fp']:>5}{s['unparsed']:>10}")


if __name__ == "__main__":
    main(sys.argv[1:] or [str(SEL / "cand_gptoss120b")])
