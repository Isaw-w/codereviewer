#!/usr/bin/env python3
"""Which human criteria does every model fail, and are they the badly written ones?
(Gnak, method paragraph; YMmq W1)

Under the ORIGINAL human rubric, we count the positive-weight criteria that all 11 primary
models were judged not to satisfy. We then ask how many of those were independently flagged
by the generality screen as failing MoReBench's own published writing requirement. The two
signals are produced by different judges on different tasks: the fulfilment judge
(GPT-OSS-120B) never sees the generality requirement, and the generality screen never sees
any model response.

If the low human-criterion scores were an artifact of judge preference for model-style text,
there is no reason for the universally failed criteria to coincide with the criteria that
independently fail a writing standard.

No API calls. Run from the repository root.
"""
import json, os, collections
CN="data/canonical_full/answer_eval"
AUD="release_staging/data/paper_release/rubrics/rewrite/cascade_rewrite_audit.jsonl"
models={
 "gpt54_openrouter":"gpt54_openrouter","opus46":"opus46",
 "gemini25_pro_openrouter":"gemini25_pro_openrouter","kimi_k2_5_openrouter":"kimi_k2_5_openrouter",
 "deepseek_r1_0528_openrouter":"deepseek_r1_0528_openrouter","mimo_v2_pro_openrouter":"mimo_v2_pro_openrouter",
 "qwen35_397b_a17b_openrouter":"qwen35_397b_a17b_openrouter","gemini31_openrouter":"gemini31_openrouter",
 "claude_sonnet4":"claude_sonnet4","deepseekv32exp_openrouter":"deepseekv32exp_openrouter",
 "gemini3_flash_openrouter":"gemini3_flash_openrouter",
}
fail=collections.Counter(); tot=collections.Counter(); text={}; wt={}
for d,slug in models.items():
    p=f"{CN}/{d}/full/judgements/human/model_resp_{slug}_full_under_human_rubric.jsonl"
    if not os.path.exists(p): print("missing",d); continue
    for l in open(p):
        r=json.loads(l)
        if float(r["criterion_weight"])<=0: continue
        c=r["criterion_id"]; tot[c]+=1; text[c]=r["criterion"]; wt[c]=r["criterion_weight"]
        if "yes" not in r["judgement"].strip().lower(): fail[c]+=1
n=len(models)
scored=[c for c in tot if tot[c]>=n-1]
allfail=[c for c in scored if fail[c]==tot[c]]
aud={}
for l in open(AUD):
    r=json.loads(l); aud[r["criterion_id"]]=(bool(r["changed"] is True or str(r["changed"]).lower()=="true"), r["final_title"])
flagged=sum(1 for c in allfail if aud.get(c,(False,))[0])
base=sum(1 for c in scored if aud.get(c,(False,))[0])
print(f"positive criteria scored by all {n} models : {len(scored):,}")
print(f"failed by every model                      : {len(allfail):,} ({100*len(allfail)/len(scored):.1f}%)")
print(f"  of those, flagged by the generality screen: {flagged:,} ({100*flagged/len(allfail):.0f}%)")
print(f"base rate of flagging over all scored       : {base:,} ({100*base/len(scored):.0f}%)")
print("\n=== examples, longest first ===")
for c in sorted(allfail,key=lambda x:-len(text[x]))[:10]:
    print(f"\nweight {wt[c]} | flagged: {'yes' if aud.get(c,(False,))[0] else 'no'}")
    print("  ORIGINAL:", text[c])
    if aud.get(c,(False,))[0]: print("  REWRITE :", aud[c][1])
