#!/usr/bin/env python3
"""Does the Finding 3 gap depend on how loosely criteria were matched? (YMmq Q)

The 5,181 confirmed pairs are stored after filtering, without the matched/nearby label the
matching judge assigned. Two independent one-to-one matching runs
(outputs/canonical/criterion_match/{allow_repeat,online_pool_remove}) do record a per-pair
label of matched / partial / none, so we attach those labels to the confirmed pairs by
joining on criterion text within each (model, case).

Coverage is partial by construction: the confirmed pairs come from candidate-pair labelling,
where one criterion may pair with several, while these runs assign at most one match per
criterion. Only the overlap can be labelled, about 44% of the 5,181. The check is therefore
whether the gap holds within each match-quality stratum, not a full recomputation.

No API calls. Run from the repository root.
"""
import json, os, math, collections
from statistics import NormalDist
PAIRS="release_staging/data/paper_release/finding3/criterion_pairs/finding1_confirmed_pairs.json"
pairs=json.load(open(PAIRS))["pairs"]
norm=lambda s:" ".join(str(s).split()).strip().lower()
def build(mode):
    CM=f"data/canonical_full/criterion_match/{mode}"
    st={}
    for md,c in {(p["md"],p["c"]) for p in pairs}:
        f=f"{CM}/{md}/case_runs/{c}/cases/{c}.json"
        if not os.path.exists(f): continue
        d=json.load(open(f))
        h={x["criterion_id"]:norm(x["title"]) for x in d.get("human_criteria",[])}
        m={x["criterion_id"]:norm(x["title"]) for x in d.get("model_criteria",[])}
        def ids(r):
            v=r.get("matched_atom_ids")
            if isinstance(v,str):
                try: v=json.loads(v.replace("'",'"'))
                except Exception: v=[]
            if not v and str(r.get("matched_atom_id")) not in ("None",""): v=[r["matched_atom_id"]]
            return v or []
        for r in d.get("human_to_model_simple",[]):
            for o in ids(r):
                if r["atom_id"] in h and o in m: st.setdefault((md,c,h[r["atom_id"]],m[o]), r["status"])
        for r in d.get("model_to_human_simple",[]):
            for o in ids(r):
                if o in h and r["atom_id"] in m: st.setdefault((md,c,h[o],m[r["atom_id"]]), r["status"])
    return st
def rates(sub):
    n=len(sub)
    hy=sum(1 for p in sub if p["hj"].upper().startswith("Y")); my=sum(1 for p in sub if p["mj"].upper().startswith("Y"))
    a=sum(1 for p in sub if p["hj"].upper().startswith("Y") and not p["mj"].upper().startswith("Y"))
    b=sum(1 for p in sub if p["mj"].upper().startswith("Y") and not p["hj"].upper().startswith("Y"))
    return n,100*hy/n,100*my/n,a,b
def sign_p(a,b):
    n=a+b; k=min(a,b)
    if n==0: return float("nan")
    if n<1100: return 2*sum(math.comb(n,i) for i in range(k+1))/2**n
    return 2*(1-NormalDist().cdf(abs(a-b)/math.sqrt(n)))
for mode in ("allow_repeat","online_pool_remove"):
    st=build(mode)
    hit=[p for p in pairs if (p["md"],p["c"],norm(p["ht"]),norm(p["mt"])) in st]
    lab=collections.Counter(st[(p["md"],p["c"],norm(p["ht"]),norm(p["mt"]))] for p in hit)
    print(f"\n=== {mode} ===  joined {len(hit):,}/{len(pairs):,}   labels {dict(lab)}")
    r=rates(pairs); print(f"{'all confirmed (paper)':24s} {r[0]:6d} pairs  human {r[1]:.1f}  model {r[2]:.1f}  gap {r[2]-r[1]:+.1f}  {r[3]} vs {r[4]}  p={sign_p(r[3],r[4]):.1e}")
    for L in ("matched","partial"):
        sub=[p for p in hit if st[(p["md"],p["c"],norm(p["ht"]),norm(p["mt"]))]==L]
        if not sub: continue
        r=rates(sub); print(f"{L+' only':24s} {r[0]:6d} pairs  human {r[1]:.1f}  model {r[2]:.1f}  gap {r[2]-r[1]:+.1f}  {r[3]} vs {r[4]}  p={sign_p(r[3],r[4]):.1e}")
