#!/usr/bin/env python3
"""
Build the BLINDED rewrite spot-check item set (E3), mirroring the foil package.

Each trial shows two statements, Version 1 and Version 2, in randomised order, so
the annotator cannot tell which is the original human criterion and which is the
cascade rewrite. They answer two questions:

  same_consideration  do the two target the same underlying moral consideration?
  more_general        which better reflects what most good responses would include,
                      rather than one specific line of argument?

Trial types (annotator never sees these):
  REAL          first 40 of the seed-42 draw: a genuine pair
  REAL_CONTROL  12 pairs where the rewrite is SHORTER than the
                original, drawn from the 457 such cases

REAL_CONTROL addresses a length confound: across all 5,043 changed criteria the
rewrite is longer than the original 91% of the time (median 169 vs 111 chars), so
in a natural sample an annotator who simply picks the longer text scores ~98%
"prefers the rewrite" with no judgement at all. The control stratum inverts that
relation, so if the rewrite is preferred in BOTH strata, length is ruled out.
Report the two strata separately; REAL is the representative sample, REAL_CONTROL
is deliberately non-representative and exists only as the check.

It is also the only bias check in the design, so it carries the load the foils used
to: a genuine annotator prefers the rewrite in both strata, one following length
alone prefers it in REAL and never in REAL_CONTROL, and one answering at random
sits near half in both.

Writes spotcheck_items.json (goes to the annotator) and spotcheck_key.json (kept).
Local only - no API calls.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "release_staging/data/paper_release/rubrics/rewrite/cascade_rewrite_audit.jsonl"
RUBRIC = REPO / "release_staging/data/paper_release/rubrics/rewrite/human_rubric_cascade_rewritten.jsonl"
OUT_DIR = REPO / "data/rebuttal/outputs/spotcheck/annotation"

SAMPLE_SEED = 42      # must match build_spotcheck_sample.py so tranche A is unchanged
BLIND_SEED = 4242     # controls version order and trial order only
N_SAMPLE = 100        # documented seed-42 draw; tranche A is its first half
N_REAL = 40           # first 40 of that draw - still a nested random sample
N_CONTROL = 12        # length control: rewrite shorter than original
N_FOIL_HARD = 8       # mismatched pair, same case  -> correct answer to Q1 is "no"
N_FOIL_EASY = 2       # mismatched pair, other case -> correct answer to Q1 is "no"


def main() -> None:
    changed = []
    with AUDIT.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if row["changed"]:
                    changed.append(row)

    dilemmas = {}
    with RUBRIC.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                dilemmas[row["TASK_ID"]] = row["DILEMMA"]

    real = random.Random(SAMPLE_SEED).sample(changed, N_SAMPLE)[:N_REAL]
    used = {(r["task_id"], r["criterion_id"]) for r in real}
    used_tasks = {r["task_id"] for r in real}

    rng = random.Random(BLIND_SEED)

    # Length control: rewrite strictly shorter than the original, disjoint from REAL.
    shorter = [r for r in changed
               if len(r["final_title"]) < len(r["original_title"])
               and (r["task_id"], r["criterion_id"]) not in used
               and r["task_id"] not in used_tasks]
    control = rng.sample(shorter, N_CONTROL)
    used_tasks |= {r["task_id"] for r in control}

    # Foils: the only check on a yes-bias in Q1. Without them an annotator who
    # answers "same consideration" throughout is indistinguishable from one who
    # genuinely finds every rewrite faithful.
    by_task = defaultdict(list)
    for r in changed:
        if r["task_id"] not in used_tasks:
            by_task[r["task_id"]].append(r)

    trials = []
    hard_pool = sorted(t for t, rs in by_task.items() if len(rs) >= 2)
    for t in rng.sample(hard_pool, N_FOIL_HARD):
        a, b = rng.sample(by_task[t], 2)
        trials.append(("FOIL_HARD", t, a["original_title"], b["final_title"], b["criterion_id"]))

    easy_pool = sorted(set(by_task) - {t for _, t, *_ in trials})
    for ta, tb in zip(rng.sample(easy_pool, N_FOIL_EASY), rng.sample(easy_pool, N_FOIL_EASY)):
        if ta == tb:
            continue
        a, b = rng.choice(by_task[ta]), rng.choice(by_task[tb])
        trials.append(("FOIL_EASY", ta, a["original_title"], b["final_title"], b["criterion_id"]))

    for kind, group in (("REAL", real), ("REAL_CONTROL", control)):
        for r in group:
            trials.append((kind, r["task_id"], r["original_title"],
                           r["final_title"], r["criterion_id"]))

    # Stratified interleave rather than a plain shuffle: spread each stratum evenly
    # through the running order so that ANY prefix is roughly proportional. An
    # annotator who runs out of time and stops at item 35 still leaves usable counts
    # in all four strata. Then break up adjacent foils so the structure is not visible.
    groups = defaultdict(list)
    for t in trials:
        groups[t[0]].append(t)
    ranked = []
    for kind, gs in groups.items():
        rng.shuffle(gs)
        for i, t in enumerate(gs):
            ranked.append(((i + 0.5) / len(gs) + rng.uniform(-0.01, 0.01), t))
    ranked.sort(key=lambda x: x[0])
    trials = [t for _, t in ranked]

    is_foil = lambda t: t[0].startswith("FOIL")
    for i in range(1, len(trials)):
        if is_foil(trials[i]) and is_foil(trials[i - 1]):
            for j in range(i + 1, len(trials)):
                if not is_foil(trials[j]) and not is_foil(trials[j - 1]):
                    trials[i], trials[j] = trials[j], trials[i]
                    break


    items, key = [], {}
    for i, (kind, task_id, original, rewrite, crit_id) in enumerate(trials, 1):
        tid = f"sc_{i:03d}"
        rewrite_first = rng.random() < 0.5
        v1, v2 = (rewrite, original) if rewrite_first else (original, rewrite)
        items.append({
            "id": tid,
            "dilemma": dilemmas[task_id],
            "v1": v1,
            "v2": v2,
        })
        key[tid] = {
            "type": kind,
            "rewrite_position": 1 if rewrite_first else 2,
            "expected_same": "yes" if kind.startswith("REAL") else "no",
            "rewrite_longer": len(rewrite) > len(original),
            "task_id": task_id,
            "criterion_id": crit_id,
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "spotcheck_items.json").write_text(
        json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT_DIR / "spotcheck_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")

    from collections import Counter
    tc = Counter(v["type"] for v in key.values())
    n_rewrite_first = sum(1 for v in key.values() if v["rewrite_position"] == 1)
    print(f"trials: {len(items)}   {dict(tc)}")
    print(f"rewrite shown as Version 1 in {n_rewrite_first}/{len(items)} trials")
    print(f"distinct dilemmas: {len({v['task_id'] for v in key.values()})}")
    for t in ("REAL", "REAL_CONTROL"):
        g = [v for v in key.values() if v["type"] == t]
        print(f"  {t:<13} n={len(g):3d}  rewrite longer in "
              f"{sum(1 for v in g if v['rewrite_longer'])}/{len(g)}")
    assert len({(i['v1'], i['v2']) for i in items}) == len(items), "duplicate trial content"
    assert all(not v["rewrite_longer"] for v in key.values()
               if v["type"] == "REAL_CONTROL"), "control stratum is not length-inverted"
    print(f"wrote {OUT_DIR}/spotcheck_items.json and spotcheck_key.json")


if __name__ == "__main__":
    main()
