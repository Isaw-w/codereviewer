#!/usr/bin/env python3
"""
Score returned foil annotations against the hidden ground-truth key.

  python3 score_foil.py returned/*.json        (or a directory)

foil_key.json holds, per trial: type (POS / FOIL_HARD / FOIL_EASY) and the correct
label (yes for a true match, no for a foil). Each returned file is one annotator's
answers. We report, against a 50% chance baseline:

  accuracy       fraction of trials the annotator labels correctly
  hit rate       yes on true matches   (sensitivity; validates the judges' matches)
  correct-rej    no on foils           (specificity; shows it is not yes-bias)
  by foil type   correct-rej split into HARD (borderline, frontier vs lenient) and
                 EASY (other-dilemma) so hard-case discrimination is visible
If >=2 annotators: pairwise human-human agreement (the honest task ceiling,
parallel to the 80.4% frontier-frontier same-item number).
"""
from __future__ import annotations
import json, sys
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

KEY_PATH = _locate("foil_key.json", "data/rebuttal/outputs/judge_select/annotation")
KEY = json.load(open(KEY_PATH))
ANN = KEY_PATH.parent


def load_files(argv):
    paths = []
    for a in argv:
        p = Path(a)
        paths += sorted(p.glob("*.json")) if p.is_dir() else [p]
    subs = []
    for p in paths:
        if p.name in ("foil_key.json", "foil_items.json"):
            continue
        try:
            o = json.load(open(p))
        except Exception:
            continue
        if isinstance(o, dict) and "answers" in o:
            subs.append(o)
    return subs


def score_one(sub):
    ok = hit = hit_n = cr = cr_n = 0
    crh = crh_n = cre = cre_n = 0
    for tid, k in KEY.items():
        a = sub["answers"].get(tid, {})
        m = a.get("match")
        if m not in ("yes", "no"):
            continue
        correct = (m == k["label"])
        ok += correct
        if k["label"] == "yes":
            hit_n += 1; hit += correct
        else:
            cr_n += 1; cr += correct
            if k["type"] == "FOIL_HARD":
                crh_n += 1; crh += correct
            else:
                cre_n += 1; cre += correct
    answered = hit_n + cr_n
    def r(a, b): return f"{a/b:5.1%}" if b else "  n/a"
    return dict(name=sub.get("annotator", "?"), answered=answered,
                acc=r(ok, answered), hit=r(hit, hit_n), cr=r(cr, cr_n),
                crh=r(crh, crh_n), cre=r(cre, cre_n))


def human_human(subs):
    ids = [t for t in KEY]
    print("\nhuman-human agreement (shared answered trials):")
    for a, b in combinations(subs, 2):
        n = agree = 0
        for t in ids:
            ma = a["answers"].get(t, {}).get("match")
            mb = b["answers"].get(t, {}).get("match")
            if ma in ("yes", "no") and mb in ("yes", "no"):
                n += 1; agree += (ma == mb)
        pct = f"{agree/n:.1%}" if n else "n/a"
        print(f"  {a.get('annotator','?'):12s} vs {b.get('annotator','?'):12s}  {pct}  (n={n})")


def main(argv):
    subs = load_files(argv or [str(ANN / 'returned')])
    if not subs:
        print("no annotation files found. Put returned foil_answers_*.json in a folder "
              "and pass it, e.g.:  python3 score_foil.py path/to/returned/")
        return
    from collections import Counter
    tc = Counter(k["type"] for k in KEY.values())
    print(f"key: {dict(tc)}  (chance accuracy = 50%)\n")
    hdr = f"{'annotator':<14}{'n':>4}{'accuracy':>10}{'hit(POS)':>10}{'corr-rej':>10}{'  hard':>8}{'  easy':>8}"
    print(hdr); print("-" * len(hdr))
    for s in subs:
        r = score_one(s)
        print(f"{r['name']:<14}{r['answered']:>4}{r['acc']:>10}{r['hit']:>10}{r['cr']:>10}{r['crh']:>8}{r['cre']:>8}")
    if len(subs) >= 2:
        human_human(subs)


if __name__ == "__main__":
    main(sys.argv[1:])
