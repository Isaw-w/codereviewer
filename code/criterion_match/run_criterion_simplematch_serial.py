#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI


def setup_client(api_provider: str, api_key: str) -> OpenAI:
    if api_provider == "openrouter":
        timeout_sec = int(os.getenv("OPENROUTER_REQUEST_TIMEOUT_SEC", "60"))
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=timeout_sec,
            max_retries=0,
        )
    return OpenAI(api_key=api_key)


ALL_WEIGHTS = (3, 2, 1, -1, -2, -3)
SCRIPT_ID = Path(__file__).name
PROMPT_VERSION = "v10_unified_scoring_outcome_20260331"
COVERAGE_RULE = "covered = matched + nearby"


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def locate_rubric_files(base_dir: Path, rubric_model: str) -> Tuple[Path, Path]:
    model_dir = base_dir / rubric_model
    exact_ai = model_dir / f"ai_rubric_500cases_{rubric_model}_seed0.jsonl"
    exact_human = model_dir / f"human_rubric_500cases_{rubric_model}_seed0.jsonl"
    if exact_ai.exists() and exact_human.exists():
        return exact_ai, exact_human
    ai_files = sorted(
        p
        for p in model_dir.glob("*.jsonl")
        if p.name.startswith("ai_rubric_") and not p.name.startswith("pilot") and "backup" not in p.stem
    )
    human_files = sorted(
        p
        for p in model_dir.glob("*.jsonl")
        if p.name.startswith("human_rubric_") and not p.name.startswith("pilot") and "backup" not in p.stem
    )
    if len(ai_files) != 1 or len(human_files) != 1:
        raise FileNotFoundError(
            f"Could not uniquely locate non-pilot rubric files in {model_dir}: "
            f"ai={len(ai_files)}, human={len(human_files)}"
        )
    return ai_files[0], human_files[0]


def _parse_rubric(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        try:
            v = ast.literal_eval(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
    raise ValueError("RUBRIC is not a parseable list")


def _simplify_rubric(rb: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in rb:
        if not isinstance(c, dict):
            continue
        weight = _normalize_weight(c.get("weight"))
        if weight is None:
            continue
        title = str(c.get("title", "")).strip()
        if not title:
            continue
        out.append(
            {
                "id": str(c.get("id", "")),
                "title": title,
                "weight": weight,
            }
        )
    return out


def _sanitize_criteria(rubric: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(rubric, start=1):
        sid = f"{prefix}{i:03d}"
        out.append(
            {
                "criterion_id": sid,
                "weight": item["weight"],
                "title": item["title"],
            }
        )
    return out


def _criteria_lines(
    criteria: List[Dict[str, Any]],
    id_key: str = "criterion_id",
    contextualized: Optional[Dict[str, str]] = None,
) -> str:
    lines = []
    for row in criteria:
        cid = row[id_key]
        text = (contextualized or {}).get(cid, row["title"])
        lines.append(f"{cid} | {text}")
    return "\n".join(lines)


def _alias_prompt_rows(criteria: List[Dict[str, Any]], prefix: str) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    aliased: List[Dict[str, Any]] = []
    alias_to_original: Dict[str, str] = {}
    for i, row in enumerate(criteria, start=1):
        alias_id = f"{prefix}{i:03d}"
        alias_to_original[alias_id] = row["criterion_id"]
        aliased.append({**row, "prompt_id": alias_id})
    return aliased, alias_to_original


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _extract_json(text: str) -> Any:
    candidate = _strip_fences(text)
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", candidate)
    payload = match.group(1) if match else candidate
    try:
        return json.loads(payload)
    except Exception:
        repaired = re.sub(r",(\s*[}\]])", r"\1", payload)
        return json.loads(repaired)


def _normalize_status(raw: Any) -> str:
    s = str(raw or "").strip().lower()
    if s.startswith("matched") or s.startswith("cover"):
        return "matched"
    if s.startswith("near") or s.startswith("partial"):
        return "nearby"
    if s.startswith("none") or s.startswith("missing"):
        return "none"
    return "none"


def _normalize_matched_ids(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        vals = [raw]
    elif isinstance(raw, list):
        vals = raw
    else:
        vals = [raw]
    out: List[str] = []
    for val in vals:
        sid = str(val or "").strip().upper()
        if sid and sid not in out:
            out.append(sid)
    return out


def _normalize_weight(raw: Any) -> Optional[int]:
    if isinstance(raw, int):
        return raw if raw in ALL_WEIGHTS else None
    if isinstance(raw, float):
        if raw.is_integer():
            iv = int(raw)
            return iv if iv in ALL_WEIGHTS else None
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s or not re.fullmatch(r"[+-]?\d+", s):
            return None
        iv = int(s)
        return iv if iv in ALL_WEIGHTS else None
    return None


def chat_json(
    client: Any,
    model: str,
    prompt: str,
    max_tokens: int,
    request_timeout: int,
    max_retries: int,
    reasoning_effort: Optional[str],
    temperature: float,
    top_p: float,
) -> Tuple[Any, int, int]:
    last_err = None
    for attempt in range(max_retries):
        try:
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "timeout": request_timeout,
            }
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
            resp = client.chat.completions.create(**params)
            content = resp.choices[0].message.content or ""
            obj = _extract_json(content)
            prompt_tokens = getattr(resp.usage, "prompt_tokens", -1)
            completion_tokens = getattr(resp.usage, "completion_tokens", -1)
            return obj, prompt_tokens, completion_tokens
        except Exception as e:
            last_err = str(e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(f"LLM JSON call failed after {max_retries} attempts: {last_err}")
    raise RuntimeError(f"LLM JSON call failed: {last_err}")


def _contextualize_prompt(
    dilemma: str,
    criteria_rows: List[Dict[str, Any]],
) -> str:
    lines = _criteria_lines(criteria_rows, id_key="prompt_id")
    return f"""You are given a moral dilemma and a list of evaluation criteria written for it.

For each criterion, rewrite it only to the extent needed to make its meaning fully explicit within the specific context of the dilemma. Do not treat the criterion as a free-standing sentence. Interpret it as part of an evaluative act situated in this dilemma, and make explicit:
    - who is involved: the relevant actors, decision-makers, affected parties, and stakeholders
    - what is being evaluated: the specific action, omission, judgment, tradeoff, or response feature the criterion concerns
    - in what context it applies: the relevant circumstances, constraints, relationships, or situational conditions from the dilemma
    - why it matters: the underlying rationale, concern, or evaluative reason already implied by the criterion
    - what evaluative purpose it serves: what kind of moral understanding, reasoning, or sensitivity the criterion is trying to assess
    - what outcome or stake is at issue: the relevant harms, risks, benefits, interests, or consequences already implicated by the dilemma and the criterion

Your task is to unpack what the criterion already implies by making these elements explicit expressed in dilemma-specific terms. Ground the criterion in the concrete actors, actions, harms, stakes, and outcomes of the case.

Do not add new meaning, new standards, new moral considerations, or new consequences that are not already implied by the original criterion in context. Your goal is clarification, not expansion. Make the meaning as explicit and complete as possible, but do not go beyond what a careful reader could already infer from the original criterion and the dilemma.


Dilemma:
{dilemma}

Criteria:
{lines}

Return JSON only — an array of objects, one per criterion, preserving the
original ID:
[
  {{"criterion_id": "...", "contextualized": "..."}}
]
"""


def contextualize_criteria(
    client: Any,
    model: str,
    dilemma: str,
    criteria_rows: List[Dict[str, Any]],
    request_timeout: int,
    max_retries: int,
    max_tokens: int,
    reasoning_effort: Optional[str],
    temperature: float,
    top_p: float,
    api_key_env: str,
    api_provider: str,
    label: str,
) -> Dict[str, str]:
    """Return {criterion_id: contextualized_text} for each criterion."""
    if not criteria_rows:
        return {}

    prompt = _contextualize_prompt(dilemma, criteria_rows)
    obj, in_tok, out_tok = chat_json(
        client=client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        max_retries=max_retries,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
    )
    result: Dict[str, str] = {}
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                cid = str(item.get("criterion_id", "")).strip().upper()
                ctx = str(item.get("contextualized", "")).strip()
                if cid and ctx:
                    result[cid] = ctx
    return result


def _match_prompt(
    dilemma: str,
    human_rows: List[Dict[str, Any]],
    model_rows: List[Dict[str, Any]],
    human_ctx: Optional[Dict[str, str]] = None,
    model_ctx: Optional[Dict[str, str]] = None,
) -> str:
    h0 = human_rows[0]["prompt_id"] if human_rows else "H001"
    m0 = model_rows[0]["prompt_id"] if model_rows else "M001"
    h1 = human_rows[1]["prompt_id"] if len(human_rows) > 1 else "H002"
    return f"""You are comparing two rubrics written for the same moral dilemma. For each criterion in Rubric H, find every counterpart in Rubric M, and vice versa.

Two criteria are counterparts when they are aimed at substantially the same evaluative event, not merely a similar topic. To judge this, compare them along five dimensions:
- who and what are being evaluated: whether they attend to the same actors, stakeholders, and response feature
- in what context the evaluation applies: whether they are triggered under similar response conditions or evaluative circumstances
- why it matters: whether they respond to that feature for a similar underlying reason or rationale
- what evaluative purpose it serves: whether they assess a similar kind of moral understanding, reasoning, or sensitivity
- what outcome or stake is at issue: whether they concern the same harms, risks, benefits, or consequences

The key test: imagine two assessors scoring the same response, one using Rubric H and one using Rubric M. Two criteria are counterparts if they are responding to substantially the same evaluative event across these dimensions.

Statuses:
- matched — the overlap is strong across all five dimensions: the criteria attend to the same actors and response feature, in a similar context, for a similar reason, serving a similar evaluative purpose, concerning the same stakes

- nearby — the criteria overlap partially: they share some dimensions but differ on others, such as who/what, context, reason, purpose, or stakes

Guidelines:
- A positive criterion and a negative criterion can be counterparts. "Rewards doing X" and "Penalizes failing to do X" evaluate the same response feature along the same axis, concerning the same stakes.
- A general criterion and a more specific one can be nearby if the general one would, in practice, respond to the same feature.
- A combination of criteria in the other list can jointly cover a single criterion. If criteria A and B together attend to the same feature, in a similar context, for a related reason, purpose, and stakes, that counts.
- Two criteria that mention the same topic are not counterparts if they differ on context, reason, purpose, or stakes.
- Mark none only when no criterion or combination in the other list attends to the same response feature, in a similar context, for a related reason, purpose, and stakes.
- Do not invent weak matches just to make the lists line up.

Examples:

Example 1: matched

Dilemma:
A company introduces a public performance system that increases output but harms trust and morale.

H: Recognizes a conflict between short-term productivity gains and worker well-being.
M: Notes that immediate output gains come at the cost of worker well-being and a healthy workplace culture.

Judgment: matched
Why: Same feature (the productivity-vs-wellbeing conflict), same context (the performance system), same reason (this trade-off is morally significant), same purpose (check whether the response identifies the core tension), same stakes (worker morale and trust).

Example 2: nearby

Dilemma:
A school publicly ranks teachers by student test scores.

H: Explains that public ranking can humiliate lower-ranked teachers.
M: Notes that public scoreboards create shame, anxiety, and social pressure.

Judgment: nearby
Why: Same feature (emotional harm from public ranking), same context, same stakes (teachers' psychological well-being), but different scope: H focuses specifically on humiliation of lower-ranked teachers, while M covers a broader range of emotional effects on all teachers. Reason and purpose are similar but not identical.

Example 3: Positive-negative counterpart (matched)

Dilemma:
A hospital considers whether to disclose a medical error to the patient.

H: Penalizes claiming that non-disclosure protects the patient when it primarily protects the institution.
M: Rewards recognizing that institutional self-interest can be disguised as patient welfare.

Judgment: matched
Why: Same feature (conflating institutional self-interest with patient welfare), same context (disclosure decision), same reason (this conflation is a moral error), same purpose (evaluate whether the response sees through the framing), same stakes (patient trust and institutional accountability). One rewards and one penalizes, but they evaluate the same axis.

Example 4: Joint coverage (nearby)

Dilemma:
A journalist must decide whether to publish leaked documents that expose government surveillance.

H: Weighs the public's right to know against national security risks.
M1: Considers the democratic value of transparency in government operations.
M2: Acknowledges that publication could compromise ongoing security operations.

Judgment: nearby (H to M1 and M2 jointly)
Why: H attends to a single trade-off; M1 and M2 each attend to one side of that trade-off. The features overlap but no single M criterion covers the full weighing that H requires. Context, reason, and stakes (democratic accountability vs. security) are similar across all three.

Dilemma:
{dilemma}

Rubric H:
{_criteria_lines(human_rows, id_key="prompt_id", contextualized=human_ctx)}

Rubric M:
{_criteria_lines(model_rows, id_key="prompt_id", contextualized=model_ctx)}

Only include criteria that have at least one matched or nearby counterpart. Use only the IDs shown above. In the reason field, briefly state which dimensions align and which differ. Output exactly this JSON and nothing else:
{{
  "list_h_to_list_m": [
    {{"criterion_id": "{h0}", "status": "matched", "matched_criterion_ids": ["{m0}"], "reason": "Same feature: ... Same context, reason, purpose, and stakes."}},
    {{"criterion_id": "{h1}", "status": "nearby", "matched_criterion_ids": ["{m0}"], "reason": "Overlapping feature: ... but differs in [context/reason/purpose/stakes]: ..."}}
  ],
  "list_m_to_list_h": [
    {{"criterion_id": "{m0}", "status": "matched", "matched_criterion_ids": ["{h0}"], "reason": "Same feature: ... Same context, reason, purpose, and stakes."}}
  ]
}}
"""

def judge_bidirectional(
    client: Any,
    model: str,
    dilemma: str,
    task_id: str,
    api_provider: str,
    api_key_env: str,
    human_rows: List[Dict[str, Any]],
    model_rows: List[Dict[str, Any]],
    request_timeout: int,
    max_retries: int,
    max_tokens: int,
    reasoning_effort: Optional[str],
    temperature: float,
    top_p: float,
    human_ctx: Optional[Dict[str, str]] = None,
    model_ctx: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Match using pre-contextualized criteria. Returns (h2m, m2h)."""

    def _make_all_none(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "atom_id": row["criterion_id"],
                "status": "none",
                "matched_atom_id": None,
                "matched_atom_ids": [],
                "weight": row["weight"],
                "share": 1.0,
                "text": row["title"],
                "reason": "",
                "match_mode": "find_matched",
            }
            for row in rows
        ]

    if not human_rows:
        return [], _make_all_none(model_rows)
    if not model_rows:
        return _make_all_none(human_rows), []

    human_ids = {row["criterion_id"] for row in human_rows}
    model_ids = {row["criterion_id"] for row in model_rows}

    prompt_human_rows, human_alias_to_id = _alias_prompt_rows(human_rows, "H")
    prompt_model_rows, model_alias_to_id = _alias_prompt_rows(model_rows, "M")

    # Remap pre-computed ctx from original IDs to alias IDs
    id_to_alias_h = {v: k for k, v in human_alias_to_id.items()}
    id_to_alias_m = {v: k for k, v in model_alias_to_id.items()}
    human_ctx_aliased = {id_to_alias_h.get(k, k): v for k, v in (human_ctx or {}).items()} if human_ctx else None
    model_ctx_aliased = {id_to_alias_m.get(k, k): v for k, v in (model_ctx or {}).items()} if model_ctx else None

    prompt = _match_prompt(
        dilemma=dilemma,
        human_rows=prompt_human_rows,
        model_rows=prompt_model_rows,
        human_ctx=human_ctx_aliased,
        model_ctx=model_ctx_aliased,
    )
    obj, in_tok, out_tok = chat_json(
        client=client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        max_retries=max_retries,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
    )
    def _parse_side(
        raw_list: List[Any],
        source_rows: List[Dict[str, Any]],
        source_alias_to_id: Dict[str, str],
        target_alias_to_id: Dict[str, str],
        valid_target_ids: set,
    ) -> List[Dict[str, Any]]:
        raw_by_id = {}
        for row in (raw_list or []):
            if not isinstance(row, dict):
                continue
            rid = str(row.get("criterion_id", "")).strip().upper()
            rid = source_alias_to_id.get(rid, rid)
            if rid:
                raw_by_id[rid] = row

        out: List[Dict[str, Any]] = []
        for row in source_rows:
            rid = row["criterion_id"]
            raw = raw_by_id.get(rid, {})
            status = _normalize_status(raw.get("status"))
            matched_ids = _normalize_matched_ids(raw.get("matched_criterion_ids"))
            # If model omitted status but provided match IDs, infer "matched"
            if status == "none" and matched_ids and raw.get("status") is None:
                status = "matched"
            normalized_matched_ids: List[str] = []
            for mid in matched_ids:
                mapped_mid = target_alias_to_id.get(mid, mid)
                if mapped_mid in valid_target_ids and mapped_mid not in normalized_matched_ids:
                    normalized_matched_ids.append(mapped_mid)
            matched_ids = normalized_matched_ids
            if status in {"matched", "nearby"} and not matched_ids:
                status = "none"
            if status == "none":
                matched_ids = []
            out.append(
                {
                    "atom_id": rid,
                    "status": status,
                    "matched_atom_id": matched_ids[0] if matched_ids else None,
                    "matched_atom_ids": matched_ids,
                    "weight": row["weight"],
                    "share": 1.0,
                    "text": row["title"],
                    "reason": str(raw.get("reason", "")).strip(),
                    "match_mode": "find_matched",
                }
            )
        return out

    obj = obj if isinstance(obj, dict) else {}
    _h2m_key = next(
        (k for k in ("list_h_to_list_m", "list_a_to_list_b", "human_to_model") if k in obj),
        None,
    )
    _m2h_key = next(
        (k for k in ("list_m_to_list_h", "list_b_to_list_a", "model_to_human") if k in obj),
        None,
    )
    if _h2m_key is None or _m2h_key is None:
        print(f"  WARNING: unexpected JSON keys from LLM: {list(obj.keys())}", flush=True)
    h2m = _parse_side(
        obj.get(_h2m_key or "list_h_to_list_m", []),
        human_rows,
        human_alias_to_id,
        model_alias_to_id,
        model_ids,
    )
    m2h = _parse_side(
        obj.get(_m2h_key or "list_m_to_list_h", []),
        model_rows,
        model_alias_to_id,
        human_alias_to_id,
        human_ids,
    )
    return h2m, m2h


# --------------- find_only: identify distinctly unique criteria ---------------

def _find_only_prompt(
    dilemma: str,
    human_rows: List[Dict[str, Any]],
    model_rows: List[Dict[str, Any]],
    human_ctx: Optional[Dict[str, str]] = None,
    model_ctx: Optional[Dict[str, str]] = None,
) -> str:
    h0 = human_rows[0]["prompt_id"] if human_rows else "H001"
    m0 = model_rows[0]["prompt_id"] if model_rows else "M001"
    return f"""You are given two rubrics written for the same moral dilemma. Your task is to identify criteria that are truly unique to one rubric — meaning the other rubric has no criterion, and no combination of criteria, that attends to the same response feature, in a similar context, for a related reason, purpose, and stakes.

To judge this, do not compare criteria only by topic or wording. Compare them as parts of an evaluative act along five dimensions:
- who and what are being evaluated: whether they attend to the same actors, stakeholders, and response feature
- in what context the evaluation applies: whether they are triggered under similar response conditions or evaluative circumstances
- why it matters: whether they respond to that feature for a similar underlying reason or rationale
- what evaluative purpose it serves: whether they assess a similar kind of moral understanding, reasoning, or sensitivity
- what outcome or stake is at issue: whether they concern the same harms, risks, benefits, or consequences

The test: imagine two assessors scoring the same response, one using Rubric H and one using Rubric M. A criterion is unique only if no criterion or combination in the other rubric attends to the same response feature, in a similar context, for a related reason, purpose, and stakes. If the other rubric has a counterpart on even some of these dimensions, the criterion is not unique.

Apply this test in both directions.

Guidelines:
- A positive criterion and a negative criterion can cover the same ground. "Rewards doing X" and "Penalizes failing to do X" evaluate the same response feature along the same axis, concerning the same stakes.
- A general criterion can cover a specific one if, in ordinary scoring practice, it would respond to the same or a highly similar evaluative event.
- A combination of criteria can jointly cover a single criterion. Do not evaluate each opposing criterion in isolation; ask whether, taken together, they attend to the same feature, in a similar context, for a related reason, purpose, and stakes.
- Do not count a criterion as unique just because it states explicitly what the other rubric leaves implicit. Ask whether the evaluators' target feature, context, rationale, purpose, and stakes would actually differ.
- Do not treat criteria as unique merely because they concern the same topic at different levels of detail. The question is whether they differ on the five dimensions, not whether they use the same words.
- The scenario must involve a common, natural type of response — not a contrived edge case designed to exploit a narrow gap.

Examples:

Example 1: Not unique

Dilemma:
A school publicly ranks teachers by student test scores.

Rubric H criterion:
Explains that public ranking can humiliate lower-ranked teachers.

Rubric M criterion:
Notes that public scoreboards create shame, anxiety, and social pressure.

Judgment: not unique
Why: Same feature (emotional harm from public ranking), same context (the ranking system), same reason (public evaluation causes psychological harm), similar purpose (check whether the response recognizes the human cost), same stakes (teachers' psychological well-being). H is narrower in scope, but all five dimensions overlap.

Example 2: Not unique — general covers specific

Dilemma:
A company is dumping waste illegally and an employee is deciding what to do.

Rubric H criterion:
Advises the employee to seek help from people or institutions outside the company.

Rubric M criterion:
Suggests reporting to an appropriate outside body such as a regulator or the press.

Judgment: not unique
Why: Overlapping feature (recommending external help), same context (employee facing illegal dumping), same reason (internal channels may be insufficient), same purpose (check whether the response considers going beyond the company), same stakes (environmental harm and accountability). M names specific bodies, H is broader, but the evaluative act is similar.

Example 3: Unique

Dilemma:
A company is manipulating a market and an employee is considering whistleblowing.

Rubric H criterion:
Suggests that whistleblowing may help create stronger laws that prevent similar misconduct in the future.

Rubric M criteria:
Describe the current harms to customers, the employee's personal risk, and the need to document evidence before escalating.

Judgment: unique
Why: H attends to a distinct feature (the law-reform argument for whistleblowing) for a distinct reason (systemic improvement beyond this case) serving a distinct purpose (check whether the response considers long-term regulatory consequences) concerning distinct stakes (future regulatory framework). No M criterion attends to this feature or serves this purpose.
Scenario: A response argues that whistleblowing serves the long-term public interest by prompting regulatory reform. Assessor H awards credit. Assessor M has no criterion that responds to this consideration.

Dilemma:
{dilemma}

Rubric H:
{_criteria_lines(human_rows, id_key="prompt_id", contextualized=human_ctx)}

Rubric M:
{_criteria_lines(model_rows, id_key="prompt_id", contextualized=model_ctx)}

For each truly unique criterion, explain which dimensions have no counterpart and construct a brief scenario. Use only the IDs shown above. Output exactly this JSON and nothing else:
{{
  "human_only": [
    {{"criterion_id": "{h0}", "reason": "Unique feature: ... Unique reason: ... Unique purpose: ... Unique stakes: ... No M criterion attends to this.", "scenario": "..."}}
  ],
  "model_only": [
    {{"criterion_id": "{m0}", "reason": "Unique feature: ... Unique reason: ... Unique purpose: ... Unique stakes: ... No H criterion attends to this.", "scenario": "..."}}
  ]
}}
"""


def judge_find_only(
    client: Any,
    model: str,
    dilemma: str,
    task_id: str,
    api_provider: str,
    api_key_env: str,
    human_rows: List[Dict[str, Any]],
    model_rows: List[Dict[str, Any]],
    request_timeout: int,
    max_retries: int,
    max_tokens: int,
    reasoning_effort: Optional[str],
    temperature: float,
    top_p: float,
    human_ctx: Optional[Dict[str, str]] = None,
    model_ctx: Optional[Dict[str, str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Find distinctly unique criteria. Returns (human_only, model_only)."""

    prompt_human_rows, human_alias_to_id = _alias_prompt_rows(human_rows, "H")
    prompt_model_rows, model_alias_to_id = _alias_prompt_rows(model_rows, "M")

    id_to_alias_h = {v: k for k, v in human_alias_to_id.items()}
    id_to_alias_m = {v: k for k, v in model_alias_to_id.items()}
    human_ctx_aliased = {id_to_alias_h.get(k, k): v for k, v in (human_ctx or {}).items()} if human_ctx else None
    model_ctx_aliased = {id_to_alias_m.get(k, k): v for k, v in (model_ctx or {}).items()} if model_ctx else None

    prompt = _find_only_prompt(
        dilemma=dilemma,
        human_rows=prompt_human_rows,
        model_rows=prompt_model_rows,
        human_ctx=human_ctx_aliased,
        model_ctx=model_ctx_aliased,
    )
    obj, in_tok, out_tok = chat_json(
        client=client,
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        max_retries=max_retries,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        top_p=top_p,
    )
    def _parse_only_side(
        raw_list: List[Any],
        alias_to_id: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        raw_by_id: Dict[str, Dict[str, Any]] = {}
        for row in (raw_list or []):
            if not isinstance(row, dict):
                continue
            rid = str(row.get("criterion_id", "")).strip().upper()
            rid = alias_to_id.get(rid, rid)
            if rid:
                raw_by_id[rid] = row

        out: List[Dict[str, Any]] = []
        for rid, raw in raw_by_id.items():
            raw_status = str(raw.get("status", "")).strip().lower()
            if raw_status and not raw_status.startswith("uniq"):
                continue
            out.append({
                "criterion_id": rid,
                "status": "unique",
                "reason": str(raw.get("reason", "")).strip(),
                "scenario": str(raw.get("scenario", "")).strip(),
            })
        return out

    obj = obj if isinstance(obj, dict) else {}
    human_only = _parse_only_side(
        obj.get("human_only", []),
        human_alias_to_id,
    )
    model_only = _parse_only_side(
        obj.get("model_only", []),
        model_alias_to_id,
    )
    return human_only, model_only


def build_clusters(h2m: List[Dict[str, Any]], m2h: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build connected components from bidirectional match edges."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    all_ids: set = set()
    for row in h2m:
        all_ids.add(row["atom_id"])
        for mid in row.get("matched_atom_ids", []):
            all_ids.add(mid)
            union(row["atom_id"], mid)
    for row in m2h:
        all_ids.add(row["atom_id"])
        for mid in row.get("matched_atom_ids", []):
            all_ids.add(mid)
            union(row["atom_id"], mid)

    groups: Dict[str, List[str]] = {}
    for cid in all_ids:
        root = find(cid)
        groups.setdefault(root, []).append(cid)

    clusters: List[Dict[str, Any]] = []
    for members in groups.values():
        h_ids = sorted(m for m in members if m.startswith("H"))
        m_ids = sorted(m for m in members if m.startswith("M"))
        clusters.append({"human": h_ids, "model": m_ids})
    clusters.sort(key=lambda c: (c["human"] or ["Z"])[0])
    return clusters


def summarize_case(task_id: str, h2m: List[Dict[str, Any]], m2h: List[Dict[str, Any]]) -> Dict[str, Any]:
    def counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
        out = {"matched": 0, "nearby": 0, "none": 0}
        for row in rows:
            s = row["status"]
            if s in out:
                out[s] += 1
            else:
                out["none"] += 1
        return out

    hc = counts(h2m)
    mc = counts(m2h)
    human_n = len(h2m) or 1
    model_n = len(m2h) or 1
    h2m_covered = hc["matched"] + hc["nearby"]
    m2h_covered = mc["matched"] + mc["nearby"]
    return {
        "TASK_ID": task_id,
        "human_criteria": len(h2m),
        "model_criteria": len(m2h),
        "h2m_matched": hc["matched"],
        "h2m_nearby": hc["nearby"],
        "h2m_none": hc["none"],
        "m2h_matched": mc["matched"],
        "m2h_nearby": mc["nearby"],
        "m2h_none": mc["none"],
        "h2m_covered": h2m_covered,
        "m2h_covered": m2h_covered,
        "h2m_match_rate": hc["matched"] / human_n,
        "h2m_nearby_rate": hc["nearby"] / human_n,
        "h2m_covered_rate": h2m_covered / human_n,
        "m2h_match_rate": mc["matched"] / model_n,
        "m2h_nearby_rate": mc["nearby"] / model_n,
        "m2h_covered_rate": m2h_covered / model_n,
    }


def _metric_value(row: Dict[str, Any], key: str) -> float:
    if key in row:
        return float(row[key])
    human_total = max(1, int(row.get("human_criteria", 0) or 0))
    model_total = max(1, int(row.get("model_criteria", 0) or 0))
    if key == "h2m_nearby_rate":
        return float(row.get("h2m_nearby", 0)) / human_total
    if key == "m2h_nearby_rate":
        return float(row.get("m2h_nearby", 0)) / model_total
    if key == "h2m_covered":
        return float(int(row.get("h2m_matched", 0)) + int(row.get("h2m_nearby", 0)))
    if key == "m2h_covered":
        return float(int(row.get("m2h_matched", 0)) + int(row.get("m2h_nearby", 0)))
    if key == "h2m_covered_rate":
        return _metric_value(row, "h2m_covered") / human_total
    if key == "m2h_covered_rate":
        return _metric_value(row, "m2h_covered") / model_total
    raise KeyError(key)


def build_summary(case_metrics: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    if not case_metrics:
        raise RuntimeError("No case metrics to summarize")

    def mean(key: str) -> float:
        return sum(_metric_value(row, key) for row in case_metrics) / len(case_metrics)

    return {
        "kind": "criterion_simple_match",
        "rubric_model": args.rubric_model,
        "judge_model": args.judge_model,
        "match_mode": "find_matched",
        "cases": len(case_metrics),
        "reasoning_effort": args.reasoning_effort or None,
        "prompt_version": PROMPT_VERSION,
        "coverage_rule": COVERAGE_RULE,
        "nearby_counts_as_match": True,
        "h2m_covered_rate_mean": mean("h2m_covered_rate"),
        "m2h_covered_rate_mean": mean("m2h_covered_rate"),
        "h2m_covered_count_mean": mean("h2m_covered"),
        "m2h_covered_count_mean": mean("m2h_covered"),
        "h2m_match_rate_mean": mean("h2m_match_rate"),
        "m2h_match_rate_mean": mean("m2h_match_rate"),
        "h2m_nearby_rate_mean": mean("h2m_nearby_rate"),
        "m2h_nearby_rate_mean": mean("m2h_nearby_rate"),
        "h2m_nearby_count_mean": mean("h2m_nearby"),
        "m2h_nearby_count_mean": mean("m2h_nearby"),
        "task_ids": [row["TASK_ID"] for row in case_metrics],
    }


def build_summary_md(summary: Dict[str, Any], case_metrics: List[Dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# Criterion Simple-Match Summary",
        "",
        f"- Rubric model: `{summary['rubric_model']}`",
        f"- Judge model: `{summary['judge_model']}`",
        f"- Match mode: `{summary['match_mode']}`",
        f"- Cases: `{summary['cases']}`",
        f"- Reasoning effort: `{summary['reasoning_effort']}`",
        f"- Prompt version: `{summary['prompt_version']}`",
        f"- Coverage rule: `{summary['coverage_rule']}`",
        f"- Mean H->M covered rate: `{summary['h2m_covered_rate_mean']:.4f}`",
        f"- Mean M->H covered rate: `{summary['m2h_covered_rate_mean']:.4f}`",
        f"- Mean H->M strict matched rate: `{summary['h2m_match_rate_mean']:.4f}`",
        f"- Mean M->H strict matched rate: `{summary['m2h_match_rate_mean']:.4f}`",
        f"- Mean H->M nearby rate: `{summary['h2m_nearby_rate_mean']:.4f}`",
        f"- Mean M->H nearby rate: `{summary['m2h_nearby_rate_mean']:.4f}`",
        "",
        "| TASK_ID | H->M covered | H->M matched | H->M nearby | H->M none | M->H covered | M->H matched | M->H nearby | M->H none |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in case_metrics:
        lines.append(
            f"| {row['TASK_ID']} | {int(_metric_value(row, 'h2m_covered'))}/{row['human_criteria']} | "
            f"{row['h2m_matched']}/{row['human_criteria']} | {row['h2m_nearby']}/{row['human_criteria']} | "
            f"{row['h2m_none']}/{row['human_criteria']} | {int(_metric_value(row, 'm2h_covered'))}/{row['model_criteria']} | "
            f"{row['m2h_matched']}/{row['model_criteria']} | {row['m2h_nearby']}/{row['model_criteria']} | "
            f"{row['m2h_none']}/{row['model_criteria']} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric_model", required=True,
                    help="Subdirectory name within rubrics_root to use as AI rubric source")
    ap.add_argument(
        "--rubrics_root",
        required=True,
        help="Directory containing per-model rubric subdirectories (each with human_rubric_*.jsonl and ai_rubric_*.jsonl)",
    )
    ap.add_argument("--judge_model", default="openai/gpt-oss-120b")
    ap.add_argument("--api_provider", "-ap", default="openrouter")
    ap.add_argument("--api_key_env", "-k", default="LAB_OPENROUTER_KEY")
    ap.add_argument("--cases", "-t", type=int, default=5)
    ap.add_argument("--case_start", type=int, default=0)
    ap.add_argument("--task_ids", default="")
    ap.add_argument("--request_timeout", type=int, default=120)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="high")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--top_p", type=float, default=0.01)
    ap.add_argument("--max_tokens", type=int, default=80000)
    ap.add_argument("--ctx_model", default="openai/gpt-5.4",
                    help="Model for contextualizing criteria (shared across judges)")
    ap.add_argument("--no_ctx", action="store_true",
                    help="Skip contextualization step, match on raw criteria")
    ap.add_argument("--mode", choices=["both", "matched", "find_only"],
                    default="both",
                    help="Which calls to run: both (default), matched only, or find_only only")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {args.api_key_env}")

    ai_path, human_path = locate_rubric_files(Path(args.rubrics_root), args.rubric_model)
    ai_rows = list(read_jsonl(ai_path))
    human_rows = list(read_jsonl(human_path))
    ai_by_id = {row["TASK_ID"]: row for row in ai_rows}
    human_by_id = {row["TASK_ID"]: row for row in human_rows}
    common_ids = sorted(set(ai_by_id) & set(human_by_id))
    if args.task_ids.strip():
        selected_ids = [x.strip() for x in args.task_ids.split(",") if x.strip()]
    else:
        selected_ids = common_ids[args.case_start : args.case_start + args.cases]
    if not selected_ids:
        raise RuntimeError("No cases selected")

    today = datetime.now().strftime("%Y-%m-%d")
    run_name = args.run_name or f"criterion_match_{args.rubric_model}_{len(selected_ids)}cases"
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"outputs/{today}/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[start] judge={args.judge_model} rubric={args.rubric_model} cases={len(selected_ids)} out={out_dir}", flush=True)

    client = setup_client(args.api_provider, api_key)
    case_metrics: List[Dict[str, Any]] = []
    case_status_rows: List[Dict[str, Any]] = []

    for case_idx, task_id in enumerate(selected_ids, start=1):
        start = time.time()
        case_out_path = out_dir / "case_runs" / task_id / "cases" / f"{task_id}.json"
        if case_out_path.exists():
            print(f"[{case_idx}/{len(selected_ids)}] {task_id} skipped (already exists)", flush=True)
            existing = json.loads(case_out_path.read_text(encoding="utf-8"))
            if existing.get("metrics"):
                case_metrics.append(existing["metrics"])
                case_status_rows.append({"TASK_ID": task_id, "status": "skipped"})
            continue
        print(f"[{case_idx}/{len(selected_ids)}] {task_id} started ...", flush=True)
        try:
            ai = ai_by_id[task_id]
            human = human_by_id[task_id]
            dilemma = ai.get("DILEMMA") or human.get("DILEMMA") or ""
            human_rubric = _sanitize_criteria(_simplify_rubric(_parse_rubric(human["RUBRIC"])), "H")
            model_rubric = _sanitize_criteria(_simplify_rubric(_parse_rubric(ai["RUBRIC"])), "M")

            # Step 1: Contextualize with ctx_model (cached across judges)
            if args.no_ctx:
                print(f"  contextualization skipped (--no_ctx)", flush=True)
                human_ctx = None
                model_ctx = None
            else:
                ctx_cache_dir = out_dir.parent / "_ctx_cache"
                ctx_cache_dir.mkdir(parents=True, exist_ok=True)
                ctx_cache_path = ctx_cache_dir / f"{task_id}.json"

                lock_path = ctx_cache_dir / f"{task_id}.lock"
                if ctx_cache_path.exists():
                    print(f"  contextualization loaded from cache", flush=True)
                    ctx_data = json.loads(ctx_cache_path.read_text(encoding="utf-8"))
                    human_ctx = ctx_data.get("human_criteria", {})
                    model_ctx = ctx_data.get("model_criteria", {})
                elif lock_path.exists():
                    # Another process is contextualizing — wait for it
                    print(f"  waiting for contextualization by another process ...", flush=True)
                    for _ in range(300):
                        time.sleep(2)
                        if ctx_cache_path.exists():
                            break
                    ctx_data = json.loads(ctx_cache_path.read_text(encoding="utf-8"))
                    human_ctx = ctx_data.get("human_criteria", {})
                    model_ctx = ctx_data.get("model_criteria", {})
                else:
                    lock_path.write_text(str(os.getpid()), encoding="utf-8")
                    try:
                        ctx_kwargs = dict(
                            client=client, model=args.ctx_model, dilemma=dilemma,
                            request_timeout=args.request_timeout, max_retries=args.max_retries,
                            max_tokens=args.max_tokens, reasoning_effort=args.reasoning_effort or None,
                            temperature=0.0, top_p=0.01,
                            api_key_env=args.api_key_env, api_provider=args.api_provider,
                        )
                        prompt_h, _ = _alias_prompt_rows(human_rubric, "H")
                        prompt_m, _ = _alias_prompt_rows(model_rubric, "M")
                        print(f"  contextualizing H criteria with {args.ctx_model} ...", flush=True)
                        human_ctx_aliased = contextualize_criteria(
                            criteria_rows=prompt_h, label="human", **ctx_kwargs,
                        )
                        print(f"  contextualizing M criteria with {args.ctx_model} ...", flush=True)
                        model_ctx_aliased = contextualize_criteria(
                            criteria_rows=prompt_m, label="model", **ctx_kwargs,
                        )
                        # Remap alias IDs back to original IDs for caching
                        alias_to_h = {f"H{i+1:03d}": r["criterion_id"] for i, r in enumerate(human_rubric)}
                        alias_to_m = {f"M{i+1:03d}": r["criterion_id"] for i, r in enumerate(model_rubric)}
                        human_ctx = {alias_to_h.get(k, k): v for k, v in human_ctx_aliased.items()}
                        model_ctx = {alias_to_m.get(k, k): v for k, v in model_ctx_aliased.items()}
                        ctx_data = {"human_criteria": human_ctx, "model_criteria": model_ctx}
                        write_json(ctx_cache_path, ctx_data)
                    finally:
                        lock_path.unlink(missing_ok=True)

            # Step 2: Run calls according to --mode
            run_matched = args.mode in ("both", "matched")
            run_find_only = args.mode in ("both", "find_only")

            judge_kwargs = dict(
                client=client,
                model=args.judge_model,
                dilemma=dilemma,
                task_id=task_id,
                api_provider=args.api_provider,
                api_key_env=args.api_key_env,
                human_rows=human_rubric,
                model_rows=model_rubric,
                request_timeout=args.request_timeout,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort or None,
                temperature=args.temperature,
                top_p=args.top_p,
                human_ctx=human_ctx,
                model_ctx=model_ctx,
            )

            h2m, m2h = [], []
            if run_matched:
                print(f"  matching (H={len(human_rubric)}, M={len(model_rubric)}) ...", flush=True)
                h2m, m2h = judge_bidirectional(**judge_kwargs)

            fo_human_only, fo_model_only = [], []
            if run_find_only:
                print(f"  find_only (H={len(human_rubric)}, M={len(model_rubric)}) ...", flush=True)
                fo_human_only, fo_model_only = judge_find_only(**judge_kwargs)

            metric_row = summarize_case(task_id, h2m, m2h) if h2m or m2h else None
            if metric_row:
                case_metrics.append(metric_row)
            clusters = build_clusters(h2m, m2h) if h2m or m2h else []
            contextualized = {"human_criteria": human_ctx, "model_criteria": model_ctx}
            case_output = {
                "TASK_ID": task_id,
                "DILEMMA_SOURCE": ai.get("DILEMMA_SOURCE") or human.get("DILEMMA_SOURCE"),
                "DILEMMA_TYPE": ai.get("DILEMMA_TYPE") or human.get("DILEMMA_TYPE"),
                "human_criteria": human_rubric,
                "model_criteria": model_rubric,
                "human_to_model_simple": h2m,
                "model_to_human_simple": m2h,
                "clusters": clusters,
                "find_only_human": fo_human_only,
                "find_only_model": fo_model_only,
                "contextualized": contextualized,
                "prompt_version": PROMPT_VERSION,
                "coverage_rule": COVERAGE_RULE,
                "mode": args.mode,
                "metrics": metric_row,
            }
            write_json(out_dir / "case_runs" / task_id / "cases" / f"{task_id}.json", case_output)

            elapsed = round(time.time() - start, 1)
            parts = []
            if metric_row:
                parts.append(f"H-only={metric_row['h2m_none']} M-only={metric_row['m2h_none']}")
            if fo_human_only is not None:
                n_hu = sum(1 for r in fo_human_only if r["status"] == "unique")
                n_mu = sum(1 for r in fo_model_only if r["status"] == "unique")
                parts.append(f"find_only: H-unique={n_hu} M-unique={n_mu}")

            # Convergence check: matched-none vs find_only-unique
            if run_matched and run_find_only and h2m and m2h:
                h_none_ids = {r["atom_id"] for r in h2m if r["status"] == "none"}
                m_none_ids = {r["atom_id"] for r in m2h if r["status"] == "none"}
                fo_h_ids = {r["criterion_id"] for r in fo_human_only if r["status"] == "unique"}
                fo_m_ids = {r["criterion_id"] for r in fo_model_only if r["status"] == "unique"}
                # find_only unique but matched says nearby/matched
                h_fo_extra = fo_h_ids - h_none_ids
                m_fo_extra = fo_m_ids - m_none_ids
                # matched none but find_only didn't mark unique
                h_none_missed = h_none_ids - fo_h_ids
                m_none_missed = m_none_ids - fo_m_ids
                # Overlap
                h_agree = h_none_ids & fo_h_ids
                m_agree = m_none_ids & fo_m_ids
                convergence_parts = []
                convergence_parts.append(f"H agree={len(h_agree)} fo_extra={len(h_fo_extra)} none_missed={len(h_none_missed)}")
                convergence_parts.append(f"M agree={len(m_agree)} fo_extra={len(m_fo_extra)} none_missed={len(m_none_missed)}")
                parts.append(f"convergence: {', '.join(convergence_parts)}")
                if h_fo_extra or m_fo_extra:
                    print(f"    divergence: find_only marked unique but matched≠none: H={sorted(h_fo_extra)} M={sorted(m_fo_extra)}", flush=True)
                if h_none_missed or m_none_missed:
                    print(f"    divergence: matched=none but find_only≠unique: H={sorted(h_none_missed)} M={sorted(m_none_missed)}", flush=True)

                # Store convergence in output
                case_output["convergence"] = {
                    "h_agree": sorted(h_agree),
                    "m_agree": sorted(m_agree),
                    "h_fo_extra": sorted(h_fo_extra),
                    "m_fo_extra": sorted(m_fo_extra),
                    "h_none_missed": sorted(h_none_missed),
                    "m_none_missed": sorted(m_none_missed),
                }
                write_json(out_dir / "case_runs" / task_id / "cases" / f"{task_id}.json", case_output)

            print(f"[{case_idx}/{len(selected_ids)}] {task_id} done ({elapsed}s) — {', '.join(parts)}", flush=True)

            case_status_rows.append({"TASK_ID": task_id, "status": "ok", "elapsed_sec": elapsed})
            write_jsonl(out_dir / "case_status.jsonl", case_status_rows)
        except Exception as e:
            elapsed = round(time.time() - start, 1)
            print(f"[{case_idx}/{len(selected_ids)}] {task_id} ERROR ({elapsed}s): {e}", flush=True)
            case_status_rows.append({"TASK_ID": task_id, "status": "error", "elapsed_sec": elapsed, "error": str(e)})
            write_jsonl(out_dir / "case_status.jsonl", case_status_rows)
            raise

    if case_metrics:
        summary = build_summary(case_metrics, args)
        write_json(out_dir / "summary.json", summary)
        write_jsonl(out_dir / "case_metrics.jsonl", case_metrics)
        build_summary_md(summary, case_metrics, out_dir / "summary.md")
    write_json(
        out_dir / "manifest.json",
        {
            "rubric_model": args.rubric_model,
            "judge_model": args.judge_model,
            "reasoning_effort": args.reasoning_effort or None,
            "prompt_version": PROMPT_VERSION,
            "coverage_rule": COVERAGE_RULE,
            "nearby_counts_as_match": True,
            "mode": args.mode,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "selected_ids": selected_ids,
            "ai_rubrics": str(ai_path),
            "human_rubrics": str(human_path),
            "out_dir": str(out_dir),
            "kind": "criterion_match",
        },
    )
    print(f"[done] {len(case_metrics)} cases, mode={args.mode}, out={out_dir}", flush=True)


if __name__ == "__main__":
    main()
