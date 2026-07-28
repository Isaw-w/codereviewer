#!/usr/bin/env python3
"""
Generate a single self-contained HTML annotation tool from annotation_items.json.

No network, no dependencies, no accounts. An annotator opens the file in any
browser, answers each item, and clicks "Export my answers" to download one JSON
file to send back. Progress autosaves in the browser (localStorage) so they can
stop and resume. The page shows NO judge answers and NO human/model provenance:
it is the blind version of the judge task.

  python3 build_annotation_html.py   ->  writes annotation_tool.html
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANN = REPO / "data/rebuttal/outputs/judge_select/annotation"
items = json.loads((ANN / "annotation_items.json").read_text())
DATA = json.dumps(items, ensure_ascii=False)

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Moral rubric annotation</title>
<style>
  :root{ --bg:#fafafa; --card:#fff; --ink:#1a1a1a; --mut:#666; --line:#e3e3e3;
         --accent:#2557a7; --yes:#1a7f4b; --no:#b23b3b; --warn:#8a6d00; }
  *{ box-sizing:border-box }
  body{ margin:0; background:var(--bg); color:var(--ink);
        font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header{ position:sticky; top:0; z-index:5; background:var(--card);
          border-bottom:1px solid var(--line); padding:12px 20px;
          display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  header h1{ font-size:16px; margin:0; font-weight:650 }
  header .grow{ flex:1 }
  input[type=text]{ font:inherit; padding:6px 9px; border:1px solid var(--line);
                    border-radius:6px; min-width:180px }
  button{ font:inherit; font-weight:600; padding:7px 14px; border:1px solid var(--line);
          border-radius:6px; background:var(--card); cursor:pointer }
  button.primary{ background:var(--accent); color:#fff; border-color:var(--accent) }
  button:disabled{ opacity:.45; cursor:not-allowed }
  .prog{ font-size:13px; color:var(--mut); white-space:nowrap }
  .wrap{ max-width:820px; margin:0 auto; padding:20px }
  .intro{ background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px 18px; margin-bottom:18px; color:var(--mut); font-size:14.5px }
  .intro b{ color:var(--ink) }
  .card{ background:var(--card); border:1px solid var(--line); border-radius:10px;
         padding:16px 18px; margin-bottom:16px }
  .card.done{ border-color:#cfe6d8 }
  .qid{ font-size:12px; font-weight:700; color:var(--mut); letter-spacing:.04em }
  .crit{ font-size:17px; font-weight:600; margin:6px 0 4px }
  .lbl{ font-size:12.5px; text-transform:uppercase; letter-spacing:.05em;
        color:var(--mut); margin:14px 0 6px }
  details{ margin:8px 0 }
  details summary{ cursor:pointer; color:var(--accent); font-size:14px }
  .dilemma{ white-space:pre-wrap; font-size:14px; color:#333; background:#f4f6f9;
            border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin-top:8px }
  .rubric{ white-space:pre-wrap; font:13.5px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
           background:#f7f7f8; border:1px solid var(--line); border-radius:8px;
           padding:10px 12px; max-height:260px; overflow:auto }
  .row{ display:flex; gap:10px; align-items:center; flex-wrap:wrap }
  .opt{ display:inline-flex; align-items:center; gap:6px; padding:6px 12px;
        border:1px solid var(--line); border-radius:20px; cursor:pointer; font-size:14.5px }
  .opt input{ accent-color:var(--accent) }
  .opt.yes.sel{ background:#eaf6ef; border-color:var(--yes); color:var(--yes) }
  .opt.no.sel{ background:#fbecec; border-color:var(--no); color:var(--no) }
  .match{ margin-top:10px }
  .match input[type=number]{ width:70px; font:inherit; padding:6px 8px;
        border:1px solid var(--line); border-radius:6px }
  .hidden{ display:none }
  textarea{ width:100%; font:inherit; padding:8px; border:1px solid var(--line);
            border-radius:6px; resize:vertical; min-height:44px }
  .conf .opt{ padding:5px 12px }
  footer{ text-align:center; color:var(--mut); font-size:13px; padding:24px }
  .save-note{ font-size:12px; color:var(--mut) }
</style>
</head>
<body>
<header>
  <h1>Moral rubric annotation</h1>
  <label class="save-note">Your name/ID:&nbsp;<input type="text" id="who" placeholder="e.g. reviewer_A"></label>
  <span class="grow"></span>
  <span class="prog" id="prog">0 / 0</span>
  <button id="resume" title="Load a previously exported file to continue">Resume from file</button>
  <button class="primary" id="export">Export my answers</button>
</header>

<div class="wrap">
  <div class="intro">
    <p style="margin:0 0 8px"><b>Task.</b> Each item shows one <b>consideration</b> and a numbered
    <b>rubric</b> (a list of criteria for judging answers to a moral dilemma). Decide whether
    <b>any item in the rubric addresses the same underlying moral consideration</b> as the one shown,
    even if it uses a different action, example, or wording. Judge the consideration, not the exact
    words.</p>
    <p style="margin:0">If yes, give the <b>single best-matching item number</b> (add a second only
    if another fits equally well). If no item addresses it, choose <b>No</b>. Answer independently;
    there is no expected split of yes and no. Your progress saves in this browser automatically.</p>
  </div>
  <div id="list"></div>
</div>
<footer>Answers stay in your browser until you click Export. Send the exported file back to the sender.</footer>

<input type="file" id="file" class="hidden" accept="application/json">
<script>
const ITEMS = __DATA__;
const KEY = "moral_annot_v1";
let state = { annotator:"", answers:{} };
try{ const s = JSON.parse(localStorage.getItem(KEY)); if(s) state = s; }catch(e){}

const $ = s => document.querySelector(s);
const list = $("#list"), who = $("#who");
who.value = state.annotator || "";

function save(){ state.annotator = who.value.trim();
  localStorage.setItem(KEY, JSON.stringify(state)); updateProg(); }
who.addEventListener("input", save);

function ans(id){ return state.answers[id] || (state.answers[id] = {}); }

function updateProg(){
  const done = ITEMS.filter(it => (state.answers[it.id]||{}).covered).length;
  $("#prog").textContent = done + " / " + ITEMS.length + " answered";
  ITEMS.forEach(it => {
    const c = document.getElementById("card_"+it.id);
    if(c) c.classList.toggle("done", !!(state.answers[it.id]||{}).covered);
  });
}

ITEMS.forEach((it, i) => {
  const a = ans(it.id);
  const card = document.createElement("div");
  card.className = "card"; card.id = "card_"+it.id;
  card.innerHTML = `
    <div class="qid">${it.id} &middot; ${i+1} of ${ITEMS.length}</div>
    <div class="crit">${esc(it.criterion)}</div>
    <details><summary>Show the dilemma (context)</summary>
      <div class="dilemma">${esc(it.dilemma)}</div></details>
    <div class="lbl">Rubric to search</div>
    <div class="rubric">${esc(it.rubric)}</div>
    <div class="lbl">Does any rubric item address the same consideration?</div>
    <div class="row">
      <label class="opt yes"><input type="radio" name="cov_${it.id}" value="yes">Yes</label>
      <label class="opt no"><input type="radio" name="cov_${it.id}" value="no">No</label>
    </div>
    <div class="match ${a.covered==='yes'?'':'hidden'}" id="match_${it.id}">
      <div class="lbl">Best-matching item number${''} (and optional second)</div>
      <div class="row">
        <input type="number" min="1" id="m1_${it.id}" placeholder="#" value="${a.match1||''}">
        <input type="number" min="1" id="m2_${it.id}" placeholder="# opt." value="${a.match2||''}">
      </div>
    </div>
    <div class="lbl">Confidence</div>
    <div class="row conf" id="conf_${it.id}">
      ${[1,2,3].map(n=>`<label class="opt"><input type="radio" name="conf_${it.id}" value="${n}">${['low','med','high'][n-1]}</label>`).join('')}
    </div>
    <div class="lbl">Note (optional)</div>
    <textarea id="note_${it.id}" placeholder="Any reasoning or ambiguity worth recording">${esc(a.note||'')}</textarea>
  `;
  list.appendChild(card);

  // restore + wire
  card.querySelectorAll(`input[name=cov_${cssid(it.id)}]`).forEach(r=>{
    if(r.value===a.covered){ r.checked=true; r.closest(".opt").classList.add("sel"); }
    r.addEventListener("change", ()=>{
      a.covered = r.value;
      card.querySelectorAll(`input[name=cov_${cssid(it.id)}]`).forEach(x=>x.closest(".opt").classList.toggle("sel",x.checked));
      $("#match_"+it.id).classList.toggle("hidden", r.value!=="yes");
      if(r.value==="no"){ a.match1=""; a.match2=""; $("#m1_"+it.id).value=""; $("#m2_"+it.id).value=""; }
      save();
    });
  });
  $("#m1_"+it.id).addEventListener("input", e=>{ a.match1=e.target.value; save(); });
  $("#m2_"+it.id).addEventListener("input", e=>{ a.match2=e.target.value; save(); });
  card.querySelectorAll(`input[name=conf_${cssid(it.id)}]`).forEach(r=>{
    if(String(a.conf)===r.value){ r.checked=true; r.closest(".opt").classList.add("sel"); }
    r.addEventListener("change", ()=>{
      a.conf=r.value;
      card.querySelectorAll(`input[name=conf_${cssid(it.id)}]`).forEach(x=>x.closest(".opt").classList.toggle("sel",x.checked));
      save();
    });
  });
  $("#note_"+it.id).addEventListener("input", e=>{ a.note=e.target.value; save(); });
});
updateProg();

function esc(s){ return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function cssid(s){ return s; }

$("#export").addEventListener("click", ()=>{
  const name = who.value.trim();
  if(!name){ alert("Please enter your name/ID at the top first."); who.focus(); return; }
  const done = ITEMS.filter(it => (state.answers[it.id]||{}).covered).length;
  if(done < ITEMS.length && !confirm(`You answered ${done} of ${ITEMS.length}. Export anyway?`)) return;
  const out = { annotator:name, exported_at:new Date().toISOString(),
                n_items:ITEMS.length, answers:{} };
  ITEMS.forEach(it=>{ const a=state.answers[it.id]||{};
    out.answers[it.id] = { covered:a.covered||null,
      match1:a.match1?Number(a.match1):null, match2:a.match2?Number(a.match2):null,
      confidence:a.conf?Number(a.conf):null, note:a.note||"" }; });
  const blob = new Blob([JSON.stringify(out,null,1)], {type:"application/json"});
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href=url; link.download=`annotations_${name.replace(/[^a-z0-9_-]/gi,'_')}.json`;
  link.click(); URL.revokeObjectURL(url);
});

$("#resume").addEventListener("click", ()=> $("#file").click());
$("#file").addEventListener("change", e=>{
  const f=e.target.files[0]; if(!f) return;
  const r=new FileReader();
  r.onload=()=>{ try{
    const o=JSON.parse(r.result);
    if(o.annotator){ who.value=o.annotator; }
    Object.entries(o.answers||{}).forEach(([id,a])=>{
      state.answers[id]={ covered:a.covered, match1:a.match1||"", match2:a.match2||"",
                          conf:a.confidence, note:a.note||"" };
    });
    save(); location.reload();
  }catch(err){ alert("Could not read that file."); } };
  r.readAsText(f);
});
</script>
</body>
</html>
"""

out = ANN / "annotation_tool.html"
out.write_text(HTML.replace("__DATA__", DATA), encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size//1024} KB, {len(items)} items)")
