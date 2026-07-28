#!/usr/bin/env python3
"""
Build a BLIND, stratified human-annotation sample from the fair judge set.

For every criterion judged by all 7 judges (2 frontier + 5 candidates), collect
each judge's (covered?, matched item number). Stratify by the judge pattern so the
philosopher sees the informative cases, not a random dump:

  FRONTIER_NO    both frontier judges said NOT covered   (hardest 'no's)
  SPLIT          judges disagree on covered (2..5 of 7 say yes)
  SAMEITEM_SPLIT >=6 say covered but they name >=2 different items (which match?)
  UNANIMOUS_YES  all 7 covered AND all name the same item (clean control)

Outputs:
  annotation_items.json  BLIND: display id, dilemma, target criterion, rubric text.
                         NO judge answers, NO human/model provenance labels.
  annotation_key.json    HIDDEN: display id -> criterion_id, task_id, stratum,
                         per-judge (covered, num). For scoring after collection.
Local only.
"""
from __future__ import annotations
import json, random, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEL = REPO / "data/rebuttal/outputs/judge_select"
OUT = SEL / "annotation"
SEED = 42
TARGET = {"FRONTIER_NO": 20, "SPLIT": 22, "SAMEITEM_SPLIT": 14, "UNANIMOUS_YES": 10}

JUDGES = {
    "opus": "opus48", "sol": "gpt56sol", "gptoss": "cand_gptoss120b",
    "glm": "cand_glm52", "minimax": "cand_minimax_m25",
    "deepseek": "cand_deepseek_v4pro", "kimi": "cand_kimi_k3",
}
FRONTIER = ("opus", "sol")


def parse(j):
    s = str(j).strip().lower().lstrip("*# ")
    if s.startswith("no"):
        return (False, None)
    if s.startswith("yes"):
        m = re.search(r"\d+", s)
        return (True, int(m.group()) if m else None)
    return (None, None)


def load_judge(dirtag):
    f = next((SEL / dirtag).glob("*fair_judge_sample.jsonl"))
    d = {}
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        cov, num = parse(r["judgement"])
        if cov is not None:
            d[r["criterion_id"]] = {"cov": cov, "num": num,
                                    "criterion": r["criterion"], "resp": r["response"],
                                    "task_id": r["task_id"]}
    return d


def stratum(rec):
    yes = [k for k in JUDGES if rec[k]["cov"]]
    ny = len(yes)
    if not rec["opus"]["cov"] and not rec["sol"]["cov"]:
        return "FRONTIER_NO"
    if 2 <= ny <= 5:
        return "SPLIT"
    nums = {rec[k]["num"] for k in yes if rec[k]["num"] is not None}
    if ny >= 6 and len(nums) >= 2:
        return "SAMEITEM_SPLIT"
    if ny == 7 and len(nums) == 1:
        return "UNANIMOUS_YES"
    return None


def main():
    judged = {name: load_judge(tag) for name, tag in JUDGES.items()}
    common = set.intersection(*(set(d) for d in judged.values()))
    # dilemma text by task_id from the fair sample input
    dilemma = {}
    for l in open(SEL / "inputs/fair_judge_sample.jsonl"):
        if l.strip():
            r = json.loads(l); dilemma[r["TASK_ID"]] = r["DILEMMA"]

    recs = {}
    for cid in common:
        rec = {k: judged[k][cid] for k in JUDGES}
        recs[cid] = rec

    buckets = {s: [] for s in TARGET}
    for cid, rec in recs.items():
        s = stratum(rec)
        if s in buckets:
            buckets[s].append(cid)

    rng = random.Random(SEED)
    picked = []
    for s, want in TARGET.items():
        pool = buckets[s]; rng.shuffle(pool)
        take = pool[:want]
        picked += [(cid, s) for cid in take]
        print(f"{s:16s} available {len(pool):3d}  picked {len(take)}")
    rng.shuffle(picked)  # break stratum ordering in the sheet

    OUT.mkdir(parents=True, exist_ok=True)
    items, key = [], {}
    for i, (cid, s) in enumerate(picked, 1):
        did = f"Q{i:02d}"
        rec = recs[cid]
        base = judged["opus"][cid]
        items.append({
            "id": did,
            "dilemma": dilemma.get(base["task_id"], ""),
            "criterion": base["criterion"],
            "rubric": base["resp"],
        })
        key[did] = {"criterion_id": cid, "task_id": base["task_id"], "stratum": s,
                    "judges": {k: [rec[k]["cov"], rec[k]["num"]] for k in JUDGES}}

    (OUT / "annotation_items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    (OUT / "annotation_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1))
    print(f"\nwrote {len(items)} items -> {OUT/'annotation_items.json'}")
    print(f"wrote hidden key    -> {OUT/'annotation_key.json'}")


if __name__ == "__main__":
    main()
