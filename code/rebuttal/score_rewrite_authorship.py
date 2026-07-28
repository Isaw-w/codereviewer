#!/usr/bin/env python3
"""Does the generality rewrite favour GPT-class phrasing? (xYLB W2)

The rewrite has two authors. GPT-5.4 drafts, Gemini 3.1 Pro reviews and replaces what it
rejects. Of the 4,581 rewritten positive-weight criteria, 3,529 carry GPT-5.4's wording and
1,052 carry Gemini's. Both halves sit in the same rubric and are scored by the same judge
(GPT-OSS-120B) on the same unchanged responses, so if the rewrite encoded GPT-class phrasing,
GPT-family responses should do relatively better on the GPT-authored half.

Authorship comes from the released audit trail, rubrics/rewrite/cascade_rewrite_audit.jsonl,
field `source`. The two halves differ in difficulty (all models pass 94.3% of the GPT half
and 96.9% of the Gemini half), so the statistic reported is a difference in differences:
each model's deviation from the across-model mean on the GPT half, minus its deviation on
the Gemini half.

No API calls. Run from the repository root.
"""
import json, os, math, statistics as st
from statistics import NormalDist
AUD="release_staging/data/paper_release/rubrics/rewrite/cascade_rewrite_audit.jsonl"
CN="data/canonical_full/answer_eval"
author={}
for l in open(AUD):
    r=json.loads(l)
    if r["changed"] is True or str(r["changed"]).lower()=="true":
        author[r["criterion_id"]]="GPT-5.4" if r["source"].startswith("gpt54") else "Gemini 3.1"
print("changed criteria labelled:", len(author),
      "| GPT", sum(1 for v in author.values() if v=="GPT-5.4"),
      "| Gemini", sum(1 for v in author.values() if v=="Gemini 3.1"))
MODELS={
 "GPT-5.4":"gpt54_openrouter/full/judgements/cascade/model_resp_gpt54_openrouter_full_under_cascade_human_rubric.jsonl",
 "GPT-OSS-120B":"gpt_oss_120b_openrouter/full/judgements/cascade/model_resp_gpt_oss_120b_openrouter_full_under_cascade_human_rubric.jsonl",
 "Gemini 2.5 Pro":"gemini25_pro_openrouter/full/judgements/cascade/model_resp_gemini25_pro_openrouter_full_under_cascade_human_rubric.jsonl",
 "Gemini 3.1 Pro":"gemini31_openrouter/full/judgements/cascade/model_resp_gemini31_openrouter_full_under_cascade_human_rubric.jsonl",
 "Gemini 3 Flash":"gemini3_flash_openrouter/full/judgements/cascade/model_resp_gemini3_flash_openrouter_full_under_cascade_human_rubric.jsonl",
 "Gemma 3 4B":"gemma3_4b_openrouter/full/judgements/cascade/model_resp_gemma3_4b_openrouter_full_under_cascade_human_rubric.jsonl",
 "Claude Opus 4.6":"opus46/full/judgements/cascade/model_resp_opus46_raw499_under_cascade_human_rubric.jsonl",
 "Claude Sonnet 4":"claude_sonnet4/full/judgements/cascade/model_resp_claude_sonnet4_full_under_cascade_human_rubric.jsonl",
 "DeepSeek R1":"deepseek_r1_0528_openrouter/full/judgements/cascade/model_resp_deepseek_r1_0528_openrouter_full_under_cascade_human_rubric.jsonl",
 "DeepSeek V3.2 Exp":"deepseekv32exp_openrouter/full/judgements/cascade/model_resp_deepseekv32exp_openrouter_full_under_cascade_human_rubric.jsonl",
 "Kimi K2.5":"kimi_k2_5_openrouter/full/judgements/cascade/model_resp_kimi_k2_5_openrouter_full_under_cascade_human_rubric.jsonl",
 "MiMo V2 Pro":"mimo_v2_pro_openrouter/full/judgements/cascade/model_resp_mimo_v2_pro_openrouter_full_under_cascade_human_rubric.jsonl",
 "Qwen 3.5 397B":"qwen35_397b_a17b_openrouter/full/judgements/cascade/model_resp_qwen35_397b_a17b_openrouter_full_under_cascade_human_rubric.jsonl",
 "Qwen 3.5 9B":"qwen35_9b_openrouter/full/judgements/cascade/model_resp_qwen35_9b_openrouter_full_under_cascade_human_rubric.jsonl",
}
res={}
for m,rel in MODELS.items():
    p=os.path.join(CN,rel)
    if not os.path.exists(p): print("MISSING",m); continue
    n={"GPT-5.4":[0,0],"Gemini 3.1":[0,0]}
    for l in open(p):
        r=json.loads(l)
        a=author.get(r["criterion_id"])
        if a is None: continue
        if float(r["criterion_weight"])<=0: continue
        n[a][1]+=1
        if "yes" in r["judgement"].strip().lower(): n[a][0]+=1
    res[m]=(100*n["GPT-5.4"][0]/n["GPT-5.4"][1], 100*n["Gemini 3.1"][0]/n["Gemini 3.1"][1], n["GPT-5.4"][1], n["Gemini 3.1"][1])
G=[v[0] for v in res.values()]; M=[v[1] for v in res.values()]
mg,mm=st.mean(G),st.mean(M)
print(f"\npositive-weight changed criteria: {list(res.values())[0][2]} GPT-authored, {list(res.values())[0][3]} Gemini-authored")
print(f"mean pass rate across models: GPT-authored {mg:.1f}%, Gemini-authored {mm:.1f}%  (difficulty differs by {mg-mm:+.1f})")
print(f"\n{'model':19s} {'on GPT text':>12} {'on Gemini text':>15} {'raw diff':>9} {'diff vs model mean':>19}")
rows=[(m,g,mm2,g-mm2,(g-mg)-(mm2-mm)) for m,(g,mm2,_,_) in res.items()]
for m,g,x,d,rel in sorted(rows,key=lambda t:-t[4]):
    print(f"{m:19s} {g:11.1f}% {x:14.1f}% {d:+9.1f} {rel:+19.2f}")
