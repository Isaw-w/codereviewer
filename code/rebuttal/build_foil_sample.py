#!/usr/bin/env python3
"""
Build a BLIND foil-discrimination sample for human validation.

50 trials, each = one consideration + ONE candidate rubric item + Yes/No
("does this item address the same underlying moral consideration?").
Ground truth is anchored to frontier agreement (Opus + GPT-5.6-sol):

  POS   (25) consideration both frontier judges covered AND named the SAME item.
             Candidate = that item.                       correct answer: YES
  FOIL_HARD   frontier-No consideration paired with the item a LENIENT judge
             (gptoss/glm/minimax/deepseek/kimi) wrongly matched.  answer: NO
  FOIL_EASY   consideration paired with an item drawn from a DIFFERENT dilemma's
             rubric (guaranteed non-match).                       answer: NO
Foils fill to 25 (hard first, easy to top up). Each consideration used once.
The foils control for yes-bias: agreeing with everything fails the No trials.

Outputs (under .../annotation/):
  foil_items.json  BLIND: trial id, dilemma, consideration, candidate item text.
  foil_key.json    HIDDEN: trial id -> type, correct label, provenance.
Local only.
"""
from __future__ import annotations
import json, random, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEL = REPO / "data/rebuttal/outputs/judge_select"
OUT = SEL / "annotation"
SEED = 7
N_POS, N_FOIL, N_HARD_MAX = 25, 25, 15
LENIENT = ["gptoss", "glm", "minimax", "deepseek", "kimi"]

JUDGES = {"opus": "opus48", "sol": "gpt56sol", "gptoss": "cand_gptoss120b",
          "glm": "cand_glm52", "minimax": "cand_minimax_m25",
          "deepseek": "cand_deepseek_v4pro", "kimi": "cand_kimi_k3"}


def parse(j):
    s = str(j).strip().lower().lstrip("*# ")
    if s.startswith("no"):
        return (False, None)
    if s.startswith("yes"):
        m = re.search(r"\d+", s)
        return (True, int(m.group()) if m else None)
    return (None, None)


def parse_items(text):
    parts = re.split(r"(?m)^\s*(\d+)\.\s+", text)
    it = iter(parts[1:])
    return {int(n): b.strip().replace("\n", " ") for n, b in zip(it, it)}


def load(dirtag):
    f = next((SEL / dirtag).glob("*fair_judge_sample.jsonl"))
    d = {}
    for l in open(f):
        if not l.strip():
            continue
        r = json.loads(l)
        cov, num = parse(r["judgement"])
        if cov is not None:
            d[r["criterion_id"]] = dict(cov=cov, num=num, criterion=r["criterion"],
                                        resp=r["response"], task_id=r["task_id"])
    return d


def main():
    J = {name: load(tag) for name, tag in JUDGES.items()}
    common = sorted(set.intersection(*(set(d) for d in J.values())))
    dilemma = {}
    for l in open(SEL / "inputs/fair_judge_sample.jsonl"):
        if l.strip():
            r = json.loads(l); dilemma[r["TASK_ID"]] = r["DILEMMA"]

    items_by_cid = {cid: parse_items(J["opus"][cid]["resp"]) for cid in common}
    rng = random.Random(SEED)
    used = set()
    trials = []

    # POSITIVES: frontier agree covered + same item number
    pos_pool = []
    for cid in common:
        o, s = J["opus"][cid], J["sol"][cid]
        if o["cov"] and s["cov"] and o["num"] is not None and o["num"] == s["num"] \
           and o["num"] in items_by_cid[cid]:
            pos_pool.append(cid)
    rng.shuffle(pos_pool)
    for cid in pos_pool[:N_POS]:
        num = J["opus"][cid]["num"]
        trials.append(dict(type="POS", label="yes", cid=cid,
                           task_id=J["opus"][cid]["task_id"],
                           consideration=J["opus"][cid]["criterion"],
                           item=items_by_cid[cid][num],
                           src=f"frontier both -> item {num}"))
        used.add(cid)

    # HARD FOILS: frontier-No, but a lenient judge matched an item
    hard_pool = []
    for cid in common:
        if cid in used:
            continue
        if J["opus"][cid]["cov"] or J["sol"][cid]["cov"]:
            continue
        for jn in LENIENT:
            r = J[jn][cid]
            if r["cov"] and r["num"] in items_by_cid[cid]:
                hard_pool.append((cid, jn, r["num"])); break
    rng.shuffle(hard_pool)
    n_hard = min(N_HARD_MAX, len(hard_pool))
    for cid, jn, num in hard_pool[:n_hard]:
        trials.append(dict(type="FOIL_HARD", label="no", cid=cid,
                           task_id=J["opus"][cid]["task_id"],
                           consideration=J["opus"][cid]["criterion"],
                           item=items_by_cid[cid][num],
                           src=f"frontier=No; {jn} wrongly matched item {num}"))
        used.add(cid)

    # EASY FOILS: consideration + item from a DIFFERENT dilemma's rubric
    need_easy = N_FOIL - n_hard
    cand_cids = [c for c in common if c not in used]
    rng.shuffle(cand_cids)
    for cid in cand_cids:
        if need_easy <= 0:
            break
        # a donor case with a different task_id
        donor = next((d for d in cand_cids
                      if J["opus"][d]["task_id"] != J["opus"][cid]["task_id"]
                      and items_by_cid[d]), None)
        if donor is None:
            continue
        ditems = items_by_cid[donor]
        dnum = rng.choice(list(ditems))
        trials.append(dict(type="FOIL_EASY", label="no", cid=cid,
                           task_id=J["opus"][cid]["task_id"],
                           consideration=J["opus"][cid]["criterion"],
                           item=ditems[dnum],
                           src=f"item from other case {J['opus'][donor]['task_id']}"))
        used.add(cid); need_easy -= 1

    rng.shuffle(trials)
    OUT.mkdir(parents=True, exist_ok=True)
    blind, key = [], {}
    for i, t in enumerate(trials, 1):
        tid = f"T{i:02d}"
        blind.append(dict(id=tid, dilemma=dilemma.get(t["task_id"], ""),
                          consideration=t["consideration"], item=t["item"]))
        key[tid] = dict(type=t["type"], label=t["label"], criterion_id=t["cid"],
                        task_id=t["task_id"], provenance=t["src"])
    (OUT / "foil_items.json").write_text(json.dumps(blind, ensure_ascii=False, indent=1))
    (OUT / "foil_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1))

    from collections import Counter
    c = Counter(t["type"] for t in trials)
    print("trials:", dict(c), "total", len(trials))
    print("labels:", dict(Counter(t["label"] for t in trials)))
    print(f"wrote {OUT/'foil_items.json'} and foil_key.json")


if __name__ == "__main__":
    main()
