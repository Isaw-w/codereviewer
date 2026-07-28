#!/usr/bin/env python3
"""
Pairwise ITEM-LEVEL agreement between judges (not raw accuracy).

Each judgement is "yes, N" (covered, matched rubric item N) or "no" (not covered).
With a ~96% yes-rate, raw accuracy is dominated by the yes-majority and says little.
The meaningful question is whether two judges agree on the SAME covered items:

  cov_jaccard   |both say yes| / |either says yes|   agreement on WHICH items covered
  both_yes      count of items both call covered
  same_item     of both-yes items where both name a match number,
                fraction pointing to the SAME rubric item N   <- anti-rubber-stamp
  a>b / b>a     items one calls covered and the other does not (who is stricter)

Same-item at chance ~ 1/(avg rubric length) ~ 5-10%. Well above that = the two
judges independently found the same correspondence, not rubber-stamping.

Usage:
  python3 item_agreement.py tag1=dir1 tag2=dir2 ...
"""
from __future__ import annotations
import json, re, sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve()
SEL = HERE.parents[2] / "data/rebuttal/outputs/judge_select"


def parse(j: str):
    s = str(j).strip().lower().lstrip("*# ")
    if s.startswith("no"):
        return (False, None)
    if s.startswith("yes"):
        m = re.search(r"\d+", s)
        return (True, int(m.group()) if m else None)
    return (None, None)


def load(spec: str):
    tag, _, loc = spec.partition("=")
    p = Path(loc) if loc else SEL / f"cand_{tag}"
    f = p if p.is_file() else next(p.glob("*fair_judge_sample.jsonl"))
    d = {}
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        yes, num = parse(r["judgement"])
        if yes is not None:
            d[r["criterion_id"]] = (yes, num)
    return tag, d


def pair(A, B):
    shared = A.keys() & B.keys()
    both_yes = either_yes = same = numbered = a_only = b_only = 0
    for k in shared:
        ay, an = A[k]; by, bn = B[k]
        if ay or by:
            either_yes += 1
        if ay and by:
            both_yes += 1
            if an is not None and bn is not None:
                numbered += 1
                same += (an == bn)
        elif ay and not by:
            a_only += 1
        elif by and not ay:
            b_only += 1
    return dict(n=len(shared), both_yes=both_yes,
                cov_jac=both_yes / either_yes if either_yes else 0.0,
                same_item=same / numbered if numbered else 0.0,
                numbered=numbered, a_only=a_only, b_only=b_only)


def main(argv):
    if not argv:
        argv = ["opus48=" + str(SEL / "opus48"),
                "gpt56sol=" + str(SEL / "gpt56sol"),
                "gptoss120b", "deepseek_v4pro"]
    judges = [load(s) for s in argv]
    print(f"{'pair':<26}{'n':>5}{'cov_jac':>9}{'both_yes':>9}{'same_item':>10}"
          f"{'(k)':>7}{'a>b':>5}{'b>a':>5}")
    print("-" * 76)
    for (ta, A), (tb, B) in combinations(judges, 2):
        r = pair(A, B)
        print(f"{ta+' vs '+tb:<26}{r['n']:>5}{r['cov_jac']:>9.1%}{r['both_yes']:>9}"
              f"{r['same_item']:>10.1%}{r['numbered']:>7}{r['a_only']:>5}{r['b_only']:>5}")


if __name__ == "__main__":
    main(sys.argv[1:])
