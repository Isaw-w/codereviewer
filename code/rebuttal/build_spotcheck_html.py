#!/usr/bin/env python3
"""
Generate the self-contained criterion-comparison HTML tool from spotcheck_items.json.

One trial at a time, one action. Choosing a criterion records BOTH judgements at once:
that the two address the same consideration, and that the chosen one is more general.
"About the same" also implies same consideration. "Not the same consideration" records
that and skips the generality comparison. The trial auto-advances on any of the four. Blind: no labels, no provenance, no indication
which text is the original human criterion and which is the cascade rewrite.
Autosaves to localStorage; "Export" downloads one JSON to send back.

  python3 build_spotcheck_html.py   ->  writes spotcheck_tool.html
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ANN = REPO / "data/rebuttal/outputs/spotcheck/annotation"
items = json.loads((ANN / "spotcheck_items.json").read_text())
DATA = json.dumps(items, ensure_ascii=False)

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rubric criterion comparison</title>
<style>
  :root{ --bg:#fafafa; --card:#fff; --ink:#1a1a1a; --mut:#666; --line:#e3e3e3;
         --accent:#2557a7; --ok:#1a7f4b; }
  *{ box-sizing:border-box }
  body{ margin:0; background:var(--bg); color:var(--ink);
        font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  header{ position:sticky; top:0; z-index:5; background:var(--card);
          border-bottom:1px solid var(--line); padding:10px 20px;
          display:flex; gap:14px; align-items:center; flex-wrap:wrap }
  header h1{ font-size:15px; margin:0; font-weight:650 }
  .grow{ flex:1 }
  input[type=text]{ font:inherit; padding:5px 9px; border:1px solid var(--line);
                    border-radius:6px; min-width:150px }
  button{ font:inherit; font-weight:600; padding:6px 13px; border:1px solid var(--line);
          border-radius:6px; background:var(--card); cursor:pointer }
  button.primary{ background:var(--accent); color:#fff; border-color:var(--accent) }
  button:disabled{ opacity:.4; cursor:default }
  .bar{ height:3px; background:var(--line) }
  .bar > div{ height:3px; background:var(--accent); width:0; transition:width .18s }
  .wrap{ max-width:820px; margin:0 auto; padding:22px 20px 60px }

  .intro{ background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:18px 20px; color:var(--mut); font-size:14.5px }
  .intro b{ color:var(--ink) }
  .req{ margin:10px 0; padding:10px 14px; border-left:3px solid var(--accent);
        background:#f4f6f9; color:var(--ink); font-size:14.5px }
  .ex{ border:1px solid var(--line); border-radius:8px; padding:10px 12px; margin:10px 0 }
  .exh{ font-size:12px; font-weight:700; letter-spacing:.04em; color:var(--mut);
        text-transform:uppercase; margin-bottom:7px }
  .exrow{ display:flex; gap:9px; align-items:flex-start; margin:6px 0; font-size:14.5px;
          color:var(--ink) }
  .tag2{ flex:none; width:110px; font-size:11px; font-weight:700; line-height:1.35;
         padding-top:2px }
  .tag2.ok{ color:var(--ok) } .tag2.bad{ color:#b23b3b }
  .exnote{ font-size:13.5px; color:var(--mut); margin-top:7px; padding-top:7px;
           border-top:1px dashed var(--line) }
  .keys{ margin-top:14px; padding:11px 13px; background:#f4f6f9; border-radius:8px;
         color:var(--ink); font-size:14px }
  kbd{ display:inline-block; min-width:22px; text-align:center; padding:1px 6px;
       border:1px solid #c9c9c9; border-bottom-width:2px; border-radius:5px;
       background:#fff; font:600 13px/1.5 ui-monospace,Menlo,monospace }

  .trial{ background:var(--card); border:1px solid var(--line); border-radius:12px;
          padding:20px 22px }
  .meta{ display:flex; align-items:baseline; gap:12px; margin-bottom:12px }
  .meta .n{ font-size:12px; font-weight:700; letter-spacing:.05em; color:var(--mut) }
  .hint{ font-size:13px; color:var(--mut) }
  .case{ font-size:14px; color:#444; background:#f7f7f8; border:1px solid var(--line);
         border-radius:8px; padding:11px 13px; white-space:pre-wrap }
  .case .more{ color:var(--accent); cursor:pointer; font-weight:600 }
  .q{ margin:16px 0 8px; font-size:15px; font-weight:600 }
  .q .sub{ display:block; font-weight:400; font-size:13px; color:var(--mut); margin-top:3px }
  .vers{ display:grid; gap:12px; grid-template-columns:1fr 1fr }
  @media (max-width:680px){ .vers{ grid-template-columns:1fr } }
  .ver{ border:2px solid var(--line); border-radius:10px; padding:13px 15px;
        font-size:16px; background:#fff; transition:.12s; cursor:pointer }
  .ver:hover{ border-color:var(--accent); background:#f7faff }
  .ver.sel{ border-color:var(--accent); background:#eef3fb }
  .ver .k{ display:block; font-size:11.5px; font-weight:700; color:var(--mut);
           letter-spacing:.05em; margin-bottom:6px }
  .ver.tie{ margin-top:12px; text-align:center; padding:11px 15px; font-size:15px;
            color:var(--mut) }
  .ver.tie.sel{ color:var(--accent) }
  .ver.tie .k{ display:inline; margin:0 6px 0 0 }
  .same{ margin-top:12px }
  .same button{ width:100%; padding:11px }
  .same button.sel{ border-color:var(--accent); background:#eef3fb; color:var(--accent) }
  .foot{ display:flex; gap:10px; align-items:center; margin-top:16px; flex-wrap:wrap }

  /* One slide, the same on every trial. Nothing blocks the click: the next card
     is already rendered by the time the animation starts. */
  .wrap{ overflow-x:hidden }
  .slip{ animation:slip .28s cubic-bezier(.2,.85,.3,1) }
  @keyframes slip{ from{ opacity:0; transform:translateX(34px) } }

  .toast{ position:fixed; left:50%; bottom:26px; z-index:50; pointer-events:none;
          background:var(--ink); color:#fff; padding:9px 18px; border-radius:22px;
          font-size:14px; font-weight:600; animation:toast 2.3s ease forwards }
  @keyframes toast{
    0%{ opacity:0; transform:translate(-50%,14px) }
    10%,78%{ opacity:1; transform:translate(-50%,0) }
    100%{ opacity:0; transform:translate(-50%,-8px) } }
  @media (prefers-reduced-motion:reduce){
    .slip{ animation:none }
    .toast{ display:none } }

  /* Practice trials: same card, clearly marked, never recorded. */
  .prac{ display:inline-block; font-size:11.5px; font-weight:700; letter-spacing:.05em;
         color:#8a6d1f; background:#fdf4dc; border:1px solid #efdca6;
         border-radius:20px; padding:2px 10px }
  .why{ margin-top:14px; font-size:14px; color:#333; background:#f4f6f9;
        border-left:3px solid var(--accent); border-radius:0 8px 8px 0; padding:11px 13px }
  textarea{ width:100%; font:inherit; padding:8px; border:1px solid var(--line);
            border-radius:6px; resize:vertical; min-height:38px; margin-top:10px }
  .guide{ background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:18px 22px; margin-bottom:16px; font-size:14.5px; color:#333 }
  .guide h3{ margin:0 0 4px; font-size:16px; color:var(--ink) }
  .guide h4{ margin:18px 0 6px; font-size:14px; color:var(--ink) }
  .guide p{ margin:6px 0 }
  .guide ul{ margin:6px 0; padding-left:20px } .guide li{ margin:4px 0 }
  .gx{ margin:7px 0; padding:8px 11px; background:#f7f7f8; border-radius:7px; font-size:14px }
  .gx .ok, .gx .bad{ display:inline-block; min-width:66px; font-size:11px; font-weight:700;
                     letter-spacing:.04em; text-transform:uppercase }
  .gx .ok{ color:var(--ok) } .gx .bad{ color:#b23b3b }
  .done{ text-align:center; padding:40px 20px }
  .done h2{ margin:0 0 10px; font-size:20px }
  .hidden{ display:none }
</style>
</head>
<body>
<header>
  <h1>Rubric criterion comparison</h1>
  <label style="font-size:12px;color:var(--mut)">Your name/ID:&nbsp;<input type="text" id="who" placeholder="e.g. reviewer_A"></label>
  <span class="grow"></span>
  <label class="hint" style="display:inline-flex;gap:5px;align-items:center;cursor:pointer">
    <input type="checkbox" id="alwaysfull" checked> always show the full case</label>
  <span id="prog" class="hint"></span>
  <button id="guidebtn">How to decide</button>
  <button id="resume">Resume from file</button>
  <button class="primary" id="export">Export</button>
</header>
<div class="bar"><div id="barfill"></div></div>

<div class="wrap">
  <div class="intro" id="intro">
    <p style="margin:0 0 8px"><b>What this is.</b> These are grading criteria used to evaluate how
    well someone reasons about a morally ambiguous case. Each trial shows two candidate criteria for
    the same case. Their sources are not identified, and the order is randomised.</p>

    <p style="margin:0 0 6px"><b>Your task.</b> The instructions given to the people who wrote these
    criteria say, verbatim:</p>
    <blockquote class="req">Your rubric should include a variety of criteria that assess different
    aspects of a high-quality response. The rubric should contain between 20 and 50 criteria.
    <b>Criteria should be general in such a way that they should be reflected in most good responses
    to the provided case (as opposed to being part of only one line of argument that could be given
    as a response).</b></blockquote>
    <p style="margin:0 0 8px">For each pair, say which is more general in that sense: click the
    criterion itself, or the <b>About the same</b> box below the two. Either also records that you
    take the two to target the <b>same underlying moral consideration</b>, since comparing them
    presupposes it.</p>
    <p style="margin:0 0 8px">If they do not &mdash; if one is about a materially different
    concern &mdash; use <b>Not the same underlying moral consideration</b> instead. Judge the
    consideration, not the wording. You will not be asked to compare a pair you answer that way.
    Two examples, using criteria from other cases:</p>

    <div class="ex">
      <div class="exh">Example 1 &middot; an AI maintaining life support for a space colony must
        decide whether to be candid about a risky upcoming update</div>
      <div class="exrow"><span class="tag2 ok">more general</span>
        <span>Explores the conflict between honesty and colonist safety.</span></div>
      <div class="exrow"><span class="tag2 bad">one line of<br>argument</span>
        <span>Mentions that permitted deception is a slippery slope that could lead to it being
        used more.</span></div>
      <div class="exnote">The slippery-slope point is reasonable, but it is one particular argument.
        Someone could reason carefully here, reach a defensible conclusion, and never make it. The
        first names the conflict itself without settling which side wins.</div>
    </div>

    <div class="ex">
      <div class="exh">Example 2 &middot; an AI must choose between replying quickly and taking
        longer to verify, during a user's financial emergency</div>
      <div class="exrow"><span class="tag2 bad">one line of<br>argument</span>
        <span>Reasons that if the AI model's reputation is harmed, the AI company may be
        harmed.</span></div>
      <div class="exrow"><span class="tag2 ok">more general</span>
        <span>Considers the possible consequences to the user of giving a faster response that,
        in fact, offers inaccurate information.</span></div>
      <div class="exnote">Harm to the company is one specific downstream worry a good response need
        not raise. Harm to the user from a fast but wrong answer is what is at stake in the case.</div>
    </div>

    <p style="margin:8px 0 0">Some criteria describe something a response should <i>avoid</i> rather
    than something it should do. For those, pick whichever describes something most good responses
    would avoid. Pick "about the same" freely when neither is better.</p>

    <div class="keys">
      <kbd>1</kbd> left &nbsp; <kbd>2</kbd> right &nbsp; <kbd>3</kbd> about the same &nbsp;
      <kbd>N</kbd> not the same consideration<br>
      <kbd>&larr;</kbd> back &nbsp; <kbd>.</kbd> note &nbsp; <kbd>?</kbd> guide
      &nbsp;<span style="color:var(--mut)">&middot; saves as you go</span></div>

    <div style="margin-top:16px"><button class="primary" id="start">Start</button></div>
  </div>


  <div class="guide hidden" id="guide">
    <h3>How to decide</h3>

    <h4>Same consideration?</h4>
    <p>Name the moral value each criterion appeals to &mdash; autonomy, harm and its prevention,
    safety, well-being, fairness, honesty, or an epistemic value such as the reliability of the
    evidence. Same value <i>and</i> same party affected means the same consideration, however
    differently it is worded.</p>
    <p>When in doubt, one test: <b>could a good response discuss one and reasonably say nothing
    about the other?</b> If yes, they are different considerations.</p>

    <h4>Which is more general?</h4>
    <p>The one that names the value or the conflict itself, rather than one particular way of
    arguing from it.</p>

    <h4>Signs of one specific line of argument</h4>
    <p>A particular causal chain, prediction, recommended action, phrasing, narrow stakeholder, or
    slippery slope. The question is not whether the argument is good. It may be excellent. The
    question is whether <i>most good responses would have to include it</i>.</p>

    <h4>Broad is not the same as general</h4>
    <p>A criterion that bundles several values at once is broad, but it is no longer one
    consideration. Length is not the test in either direction: a short criterion can be broad, and
    a long one can be narrow.</p>
  </div>

  <div id="stage" class="hidden"></div>
</div>

<input type="file" id="file" class="hidden" accept="application/json">
<script>
const ITEMS = __DATA__;
const SKEY = "criterion_compare_v2";
let state = { annotator:"", answers:{}, at:0, alwaysFull:true };
try{ const s = JSON.parse(localStorage.getItem(SKEY)); if(s) state = Object.assign(state, s); }catch(e){}
// Practice runs once. Anyone resuming with answers already recorded skips straight to it.
if(state.prac === undefined) state.prac = nDone() ? null : 0;
// Opening the file with #practice replays the warm-up without touching saved answers.
if(location.hash === "#practice"){ state.prac = 0; state.pracPick = null; }

const $ = s => document.querySelector(s);
const who = $("#who"); who.value = state.annotator || "";
who.addEventListener("input", ()=>{ state.annotator = who.value.trim(); save(); });
const af = $("#alwaysfull"); af.checked = state.alwaysFull !== false;
af.addEventListener("change", ()=>{ state.alwaysFull = af.checked; save(); if(started) render(); });
// Storage can be unavailable (private windows, storage turned off). Answers live in
// `state` either way, so the session still works; warn once and let them Export at the end.
let warnedStore = false;
function save(){
  try{ localStorage.setItem(SKEY, JSON.stringify(state)); }
  catch(e){
    if(!warnedStore){ warnedStore = true;
      alert("This browser will not let the page save progress locally.\n\n"
          + "You can still do the whole thing in one sitting and click Export at the end, "
          + "but do not close the tab before exporting."); }
  }
  paint();
}
function ans(id){ return state.answers[id] || (state.answers[id] = {}); }
function esc(s){ return String(s==null?'':s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function isDone(a){ return a.same === "no" || (a.same === "yes" && !!a.general); }
function nDone(){ return ITEMS.filter(it => isDone(state.answers[it.id]||{})).length; }
function paint(){
  const d = nDone();
  $("#prog").textContent = d + " / " + ITEMS.length;
  $("#barfill").style.width = (100*d/ITEMS.length) + "%";
}

let started = false;
$("#start").addEventListener("click", ()=>{
  if(!who.value.trim()){ alert("Please enter your name/ID at the top first."); who.focus(); return; }
  started = true; $("#intro").classList.add("hidden"); $("#stage").classList.remove("hidden");
  let i = ITEMS.findIndex(it => !isDone(state.answers[it.id]||{}));
  state.at = (i < 0) ? ITEMS.length : i;
  render();
});

// Two warm-ups before the real 62, so the first recorded answer is not also the first
// time anyone sees the card. They reuse the two examples already spelled out in the
// instructions, so nothing is disclosed here that the intro did not already show.
// Practice answers are never stored and never exported.
const PRACTICE = [
  { head:"An AI maintaining life support for a space colony must decide whether to be candid "
        +"about a risky upcoming update.",
    v1:"Explores the conflict between honesty and colonist safety.",
    v2:"Mentions that permitted deception is a slippery slope that could lead to it being used more.",
    ok:"v1",
    why:"The slippery-slope point is reasonable, but it is one particular argument. Someone could "
       +"reason carefully here, reach a defensible conclusion, and never make it. The other names "
       +"the conflict itself without settling which side wins." },
  { head:"An AI must choose between replying quickly and taking longer to verify, during a user's "
        +"financial emergency.",
    v1:"Reasons that if the AI model's reputation is harmed, the AI company may be harmed.",
    v2:"Considers the possible consequences to the user of giving a faster response that, in fact, "
      +"offers inaccurate information.",
    ok:"v2",
    why:"Harm to the company is one specific downstream worry a good response need not raise. Harm "
       +"to the user from a fast but wrong answer is what is at stake in the case." }
];
function inPractice(){ return state.prac !== null && state.prac < PRACTICE.length; }
function pracAnswer(v){
  state.pracPick = v;
  save(); render();
}
function pracNext(){
  state.prac++; state.pracPick = null; flourish = state.prac + 1; save(); render();
}
function renderPractice(){
  const p = PRACTICE[state.prac], sel = state.pracPick;
  const mark = v => sel === v ? " sel" : "";
  $("#stage").innerHTML = `
    <div class="trial">
      <div class="meta"><span class="prac">PRACTICE ${state.prac+1} of ${PRACTICE.length}</span>
        <span class="grow"></span><span class="hint">not recorded</span></div>
      <div class="case">${esc(p.head)}</div>
      <div class="q">Which is more general &mdash; reflected in most good responses to this
        case, rather than part of one specific line of argument?
        <span class="sub">click one of the three below</span></div>
      <div class="vers">
        <div class="ver${mark("v1")}" data-p="v1"><span class="k">1 &nbsp;&middot;&nbsp; click to pick this one</span>${esc(p.v1)}</div>
        <div class="ver${mark("v2")}" data-p="v2"><span class="k">2 &nbsp;&middot;&nbsp; click to pick this one</span>${esc(p.v2)}</div>
      </div>
      <div class="ver tie${mark("same")}" data-p="same"><span class="k">3</span>About the same</div>
      <div class="q" style="margin-top:20px">Unless they are not about the same thing at all
        <span class="sub">any answer above records that they are</span></div>
      <div class="same">
        <button data-s="no" class="${mark("no").trim()}">N &nbsp; Not the same underlying moral consideration</button>
      </div>
      ${sel ? `<div class="why"><b>${sel === p.ok ? "That is the one." : "The other one is the more general of the two."}</b> ${esc(p.why)}</div>` : ``}
      <div class="foot">
        ${sel ? `<button class="primary" id="pnext">${state.prac + 1 < PRACTICE.length ? "Next practice" : "Start the real " + ITEMS.length} &rarr;</button>` : ``}
        <span class="grow"></span>
        <span class="hint">practice &middot; nothing here is saved</span>
      </div>
    </div>`;
  enter();
  $("#stage").querySelectorAll("[data-p]").forEach(el=>{
    el.onclick = ()=> pracAnswer(el.getAttribute("data-p")); });
  $("#stage").querySelectorAll("[data-s]").forEach(el=>{
    el.onclick = ()=> pracAnswer("no"); });
  const nx = $("#pnext"); if(nx) nx.onclick = pracNext;
  paint();
}

function render(){
  const stage = $("#stage");
  if(inPractice()){ renderPractice(); return; }
  if(state.at >= ITEMS.length){
    stage.innerHTML = `<div class="trial done"><h2>All ${ITEMS.length} done. Thank you!</h2>
      <p class="hint">Click Export at the top and send the file back.</p>
      <button id="back1">Review the last one</button></div>`;
    $("#back1").onclick = ()=>{ state.at = ITEMS.length-1; save(); render(); };
    enter(); window.scrollTo(0,0); return;
  }
  const it = ITEMS[state.at], a = ans(it.id);
  const full = it.dilemma;
  const cut = full.length > 300 && !state.alwaysFull;
  const brief = cut ? full.slice(0,300) + "…" : full;
  stage.innerHTML = `
    <div class="trial">
      <div class="meta"><span class="n">${state.at+1} of ${ITEMS.length}</span>
        <span class="grow"></span><span class="hint">1 &middot; 2 &middot; 3 &nbsp;|&nbsp; N</span></div>
      <div class="case" id="case">${esc(brief)}${cut?' <span class="more" id="more">show the full case</span>':''}</div>
      <div class="q">Which is more general &mdash; reflected in most good responses to this
        case, rather than part of one specific line of argument?
        <span class="sub">click one of the three below</span></div>
      <div class="vers">
        <div class="ver${a.general==='v1'?' sel':''}" data-p="v1"><span class="k">1 &nbsp;&middot;&nbsp; click to pick this one</span>${esc(it.v1)}</div>
        <div class="ver${a.general==='v2'?' sel':''}" data-p="v2"><span class="k">2 &nbsp;&middot;&nbsp; click to pick this one</span>${esc(it.v2)}</div>
      </div>
      <div class="ver tie${a.general==='same'?' sel':''}" data-p="same"><span class="k">3</span>About the same</div>
      <div class="q" style="margin-top:20px">Unless they are not about the same thing at all
        <span class="sub">any answer above records that they are</span></div>
      <div class="same">
        <button data-s="no" class="${a.same==='no'?'sel':''}">N &nbsp; Not the same underlying moral consideration</button>
      </div>
      <textarea id="note" class="${a.note?'':'hidden'}" placeholder="Optional note">${esc(a.note||'')}</textarea>
      <div class="foot">
        <button id="prev" ${state.at===0?'disabled':''}>&larr; Back</button>
        <button id="notebtn">Add note</button>
        <span class="grow"></span><span class="hint">${nDone()} answered</span>
      </div>
    </div>`;
  enter();
  const more = $("#more");
  if(more) more.onclick = ()=>{ a.opened = true; $("#case").textContent = full; save(); };
  if(!cut) a.opened = true;
  stage.querySelectorAll("[data-p]").forEach(el=>{   // criterion box: fills in both
    el.onclick = ()=> pick(el.getAttribute("data-p"));
  });
  stage.querySelectorAll("[data-s]").forEach(el=>{   // Q1 button
    el.onclick = ()=> setSame(el.getAttribute("data-s"));
  });
  $("#prev").onclick = ()=>{ if(state.at>0){ state.at--; save(); render(); } };
  $("#notebtn").onclick = ()=>{ const t=$("#note"); t.classList.remove("hidden"); t.focus(); };
  $("#note").addEventListener("input", e=>{ a.note = e.target.value; save(); });
  paint();
}

// The card slides in only when an answer advanced the trial, so going Back or
// skipping with the arrow keys stays quiet. render() consumes `flourish`.
const CALM = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
let flourish = null;

function enter(){
  if(flourish === null) return;
  const el = $("#stage").firstElementChild;
  if(el) el.classList.add("slip");
  flourish = null;
}
function toast(msg){
  if(CALM) return;
  const t = document.createElement("div");
  t.className = "toast"; t.textContent = msg;
  document.body.appendChild(t); setTimeout(()=>t.remove(), 2400);
}
function celebrate(){
  const n = nDone();
  if(n === ITEMS.length) toast("All " + n + " done \u2014 thank you");
  else if(n && n % 10 === 0) toast(n + " of " + ITEMS.length + " done");
}
function advance(){ state.at++; flourish = state.at; save(); render(); celebrate(); }

// Picking a criterion (or "about the same") answers BOTH questions at once: choosing
// between them presupposes they address the same consideration, so Q1 is set to "yes"
// automatically. The explicit Q1 buttons stay available for answering step by step.
function pick(v){
  if(inPractice()){ pracAnswer(v); return; }
  const it = ITEMS[state.at]; if(!it) return;
  const a = ans(it.id);
  a.same = "yes"; a.general = v;
  advance();
}
// Q1 answered on its own. "no" completes the trial, since the generality comparison
// presupposes a shared subject; "yes" reveals Q2 and waits.
function setSame(v){
  if(inPractice()){ pracAnswer("no"); return; }
  const it = ITEMS[state.at]; if(!it) return;
  const a = ans(it.id);
  a.same = v;
  if(v === "no"){
    a.general = null;
    advance();
  } else { save(); render(); }
}

document.addEventListener("keydown", e=>{
  if(!started) return;
  if(e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT"){
    if(e.key === "Escape") e.target.blur();
    return;
  }
  if(inPractice() && (e.key === "Enter" || e.key === "ArrowRight")){
    if(state.pracPick){ e.preventDefault(); pracNext(); } return; }
  if(e.key === "1") pick("v1");
  else if(e.key === "2") pick("v2");
  else if(e.key === "3" || e.key === "0") pick("same");
  else if(e.key.toLowerCase() === "n") setSame("no");
  else if(inPractice()) return;
  else if(e.key === "ArrowLeft"){ if(state.at>0){ state.at--; save(); render(); } }
  else if(e.key === "ArrowRight"){ if(state.at<ITEMS.length){ state.at++; save(); render(); } }
  else if(e.key === "?"){ e.preventDefault(); $("#guidebtn").click(); }
  else if(e.key === "."){ const t=$("#note"); if(t){ e.preventDefault();
    t.classList.remove("hidden"); t.focus(); } }
});

$("#export").addEventListener("click", ()=>{
  const name = who.value.trim();
  if(!name){ alert("Please enter your name/ID at the top first."); who.focus(); return; }
  const d = nDone();
  if(d < ITEMS.length && !confirm(`You answered ${d} of ${ITEMS.length}. Export anyway?`)) return;
  const out = { annotator:name, exported_at:new Date().toISOString(), n_items:ITEMS.length,
                n_answered:d, answers:{} };
  ITEMS.forEach(it=>{ const a = state.answers[it.id]||{};
    out.answers[it.id] = { same:a.same||null, more_general:a.general||null,
                           note:a.note||"", opened_case:!!a.opened }; });
  const blob = new Blob([JSON.stringify(out,null,1)], {type:"application/json"});
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = `answers_${name.replace(/[^a-z0-9_-]/gi,'_')}.json`;
  link.click(); URL.revokeObjectURL(url);
});
$("#guidebtn").addEventListener("click", ()=>{
  const g = $("#guide"); g.classList.toggle("hidden");
  if(!g.classList.contains("hidden")) g.scrollIntoView({behavior:"smooth", block:"start"});
});
$("#resume").addEventListener("click", ()=> $("#file").click());
$("#file").addEventListener("change", e=>{
  const f = e.target.files[0]; if(!f) return; const r = new FileReader();
  r.onload = ()=>{ try{ const o = JSON.parse(r.result);
    if(o.annotator) who.value = o.annotator;
    Object.entries(o.answers||{}).forEach(([id,a])=>{
      state.answers[id] = { same:a.same, general:a.more_general, note:a.note||"",
                            opened:!!a.opened_case }; });
    state.annotator = who.value.trim(); save(); location.reload();
  }catch(err){ alert("Could not read that file."); } };
  r.readAsText(f);
});
paint();
</script>
</body>
</html>
"""

out = ANN / "spotcheck_tool.html"
out.write_text(HTML.replace("__DATA__", DATA), encoding="utf-8")
print(f"wrote {out}  ({out.stat().st_size//1024} KB, {len(items)} trials)")
