#!/usr/bin/env python3
"""
Score returned criterion-comparison annotations against the hidden key.

  python3 score_spotcheck.py returned/*.json        (or a directory)

Two questions per trial. Q1: do the two target the same underlying moral
consideration? Q2: which of the two criteria is more general, in the sense of
the benchmark's own instruction (reflected in most good responses to the case, as
opposed to being part of only one line of argument). The annotator never saw which
version was the original human criterion and which was the cascade rewrite, and the
left/right order was randomised.

Q2 is asked only when the annotator answered "same" to Q1, so the generality rates
below are CONDITIONAL on the pair being judged to preserve the consideration. Report
them that way: "among pairs judged to preserve the consideration, the rewrite was
preferred X% of the time". n_eligible records how many pairs reached Q2.

Two strata, reported separately:

  REAL          representative sample; the rewrite is the LONGER text in 39 of 40
  REAL_CONTROL  length-inverted; the rewrite is the SHORTER text in all 12

The control stratum is the bias check. Three patterns are distinguishable:

  genuine judgement   rewrite preferred in BOTH strata
  length heuristic    rewrite preferred in REAL, near zero in REAL_CONTROL
  random clicking     near 50% in both

Chance between the two directional options is 50%. "About the same" is reported
separately and excluded from the directional rate.

Q1 is reported as intent preservation on the REAL strata, checked against foil
rejection. A high intent rate is only meaningful if foil rejection is also high;
otherwise the annotator is simply answering "same" throughout.
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve()


def _locate(fname, legacy):
    """Find a support file. Works both in the development tree and in the
    standalone release, where the key sits beside the returned answers."""
    cands = []
    for a in sys.argv[1:]:
        p = Path(a)
        cands.append((p if p.is_dir() else p.parent) / fname)
    cands.append(HERE.parent / fname)
    cands.append(HERE.parents[2] / legacy / fname)
    for c in cands:
        if c.is_file():
            return c
    raise SystemExit(
        f"{fname} not found. Pass the directory holding it, or place it "
        f"next to this script. Looked in: " + ", ".join(str(c) for c in cands))

KEY_PATH = _locate("spotcheck_key.json", "data/rebuttal/outputs/spotcheck/annotation")
KEY = json.load(open(KEY_PATH))
ANN = KEY_PATH.parent
STRATA = ("REAL", "REAL_CONTROL")


def load_files(argv):
    paths = []
    for a in argv:
        p = Path(a)
        paths += sorted(p.glob("*.json")) if p.is_dir() else [p]
    subs = []
    for p in paths:
        if p.name in ("spotcheck_key.json", "spotcheck_items.json"):
            continue
        try:
            o = json.load(open(p))
        except Exception:
            continue
        if isinstance(o, dict) and "answers" in o:
            subs.append(o)
    return subs


def pct(a, b):
    return f"{a / b:5.1%}" if b else "  n/a"


def tally(sub):
    out = {s: Counter() for s in STRATA}
    q1 = Counter()
    opened = answered = trials = 0
    for tid, k in KEY.items():
        a = sub["answers"].get(tid, {})
        g, same = a.get("more_general"), a.get("same")
        if a.get("opened_case"):
            opened += 1
        if same in ("yes", "no"):
            trials += 1
            real = k["type"].startswith("REAL")
            bucket = "real" if real else ("hard" if k["type"] == "FOIL_HARD" else "easy")
            q1[bucket + "_n"] += 1
            if (same == "yes") == real:
                q1[bucket + "_ok"] += 1
        if g not in ("v1", "v2", "same"):
            continue
        answered += 1
        if k["type"] not in out:
            continue
        if g == "same":
            out[k["type"]]["same"] += 1
        else:
            pos = 1 if g == "v1" else 2
            out[k["type"]]["rewrite" if pos == k["rewrite_position"] else "original"] += 1
    return out, q1, trials, answered, opened


def verdict(t):
    def rate(s):
        c = t[s]
        d = c["rewrite"] + c["original"]
        return (c["rewrite"] / d) if d else None
    r, cc = rate("REAL"), rate("REAL_CONTROL")
    if r is None or cc is None:
        return "insufficient data"
    if r >= 0.6 and cc >= 0.6:
        return "PREFERS REWRITE IN BOTH STRATA (length ruled out)"
    if r >= 0.6 and cc <= 0.4:
        return "LENGTH HEURISTIC (preference vanishes when the rewrite is shorter)"
    if 0.4 < r < 0.6 and 0.4 < cc < 0.6:
        return "NO CLEAR PREFERENCE (consistent with random responding)"
    return "MIXED - inspect by hand"


def main(argv):
    subs = load_files(argv or [str(ANN / "returned")])
    if not subs:
        print("no annotation files found. Put returned answers_*.json in a folder and "
              "pass it, e.g.:  python3 score_spotcheck.py path/to/returned/")
        return
    tc = Counter(k["type"] for k in KEY.values())
    print(f"key: {dict(tc)}   chance = 50% of directional choices\n")

    pooled = {s: Counter() for s in STRATA}
    pq1 = Counter()
    for sub in subs:
        t, q1, trials, answered, opened = tally(sub)
        pq1.update(q1)
        name = sub.get("annotator", "?")
        elig = {st: sum(1 for tid, k in KEY.items() if k["type"] == st
                        and sub["answers"].get(tid, {}).get("same") == "yes")
                for st in STRATA}
        print(f"{name}   {trials}/{len(KEY)} trials completed  "
              f"({answered} reached Q2, {trials - answered} stopped at 'not the same')  "
              f"case opened on {opened}")
        print(f"   Q1 intent preserved (REAL) {pct(q1['real_ok'], q1['real_n'])}"
              f"   foil rejection {pct(q1['hard_ok']+q1['easy_ok'], q1['hard_n']+q1['easy_n'])}"
              f"  (hard {pct(q1['hard_ok'], q1['hard_n'])}, easy {pct(q1['easy_ok'], q1['easy_n'])})")
        for s in STRATA:
            c = t[s]
            d = c["rewrite"] + c["original"]
            print(f"   {s:<13} rewrite {c['rewrite']:3d}  original {c['original']:3d}  "
                  f"same {c['same']:3d}   -> rewrite {pct(c['rewrite'], d)} of directional"
                  f"   [reached Q2: {elig[s]}]")
            pooled[s].update(c)
        print(f"   verdict: {verdict(t)}\n")

    if len(subs) > 1:
        print("pooled across annotators")
        print(f"   Q1 intent preserved (REAL) {pct(pq1['real_ok'], pq1['real_n'])}"
              f"   foil rejection {pct(pq1['hard_ok']+pq1['easy_ok'], pq1['hard_n']+pq1['easy_n'])}")
        for s in STRATA:
            c = pooled[s]
            d = c["rewrite"] + c["original"]
            print(f"   {s:<13} rewrite {c['rewrite']:3d}  original {c['original']:3d}  "
                  f"same {c['same']:3d}   -> rewrite {pct(c['rewrite'], d)} of directional")
        print(f"   verdict: {verdict(pooled)}")

        # Item-level aggregation. The claim is about criteria, not about judgements, so
        # the criterion is the natural unit: n = number of rewrites, not n = judgements.
        # Pooling judgements treats 3 ratings of one criterion as 3 independent facts,
        # which understates the interval because ratings of the same item are correlated.
        print("\nitem-level, using the criterion as the unit (n = rewrites judged by >=2 people)")
        for stratum in ("REAL", "REAL_CONTROL", "FOIL"):
            n = maj = una = 0
            for tid, k in KEY.items():
                grp = "FOIL" if k["type"].startswith("FOIL") else k["type"]
                if grp != stratum:
                    continue
                votes = [x["answers"].get(tid, {}).get("same") for x in subs]
                votes = [v for v in votes if v in ("yes", "no")]
                if len(votes) < 2:
                    continue
                n += 1
                want = "yes" if stratum.startswith("REAL") else "no"
                if votes.count(want) * 2 > len(votes):
                    maj += 1
                if votes.count(want) == len(votes):
                    una += 1
            label = "preserved" if stratum.startswith("REAL") else "rejected"
            print(f"   {stratum:<13} {label} by a majority {pct(maj, n)}   "
                  f"unanimously {pct(una, n)}   (n={n})")

        print("\nannotator agreement (trials both answered):")
        for a, b in combinations(subs, 2):
            n = ag = 0
            ns = ags = 0
            for t in KEY:
                xa = a["answers"].get(t, {}).get("more_general")
                xb = b["answers"].get(t, {}).get("more_general")
                if xa in ("v1", "v2", "same") and xb in ("v1", "v2", "same"):
                    n += 1
                    ag += (xa == xb)
                ya = a["answers"].get(t, {}).get("same")
                yb = b["answers"].get(t, {}).get("same")
                if ya in ("yes", "no") and yb in ("yes", "no"):
                    ns += 1
                    ags += (ya == yb)
            print(f"   {a.get('annotator','?'):12s} vs {b.get('annotator','?'):12s}  "
                  f"Q1 {pct(ags, ns)} (n={ns})   Q2 {pct(ag, n)} (n={n})")


if __name__ == "__main__":
    main(sys.argv[1:])
