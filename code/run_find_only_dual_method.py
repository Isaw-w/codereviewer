#!/usr/bin/env python3
"""
Dual-method LLM confirmation for human-unique residual criteria (Finding 3).

For each of the 99 cases that contain >=1 human residual criterion (cosine
top1<0.80, top5_mean<0.78), compares the full human rubric against each of 11
canonical model rubrics using the find_only judgment task. The human rubric
sent to the judge always uses the original human-rubric titles; residual IDs
from `residuals_t80.jsonl` are used only to define which criteria count as
cosine residual candidates for confirmation. A human criterion is LLM-
confirmed as unique if it appears in the human_only list for a given
(case, model) comparison.

Rubrics:  release_staging/data/paper_release/rubrics/original/human_rubric_500cases.jsonl
          release_staging/data/paper_release/rubrics/ai_rubrics/{model}.jsonl
Residuals: release_staging/data/paper_release/finding2/direct_check/residuals_top100.jsonl
Output:    release_staging/data/paper_release/finding2/direct_check/raw_llm/{rubric_model}/{judge_slug}/results.jsonl

Usage:
    python3 run_find_only_dual_method.py --ak $OPENROUTER_API_KEY
    python3 run_find_only_dual_method.py --ak $OPENROUTER_API_KEY --rubric_model gpt54 --workers 5
    python3 run_find_only_dual_method.py --ak $OPENROUTER_API_KEY --debug   # 1 case only
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    release_candidate = RELEASE_ROOT / p
    repo_candidate = REPO_ROOT / p
    if release_candidate.exists():
        return release_candidate
    if repo_candidate.exists():
        return repo_candidate
    if p.parts and p.parts[0] in {"code", "data", "docs", "manifests"}:
        return release_candidate
    return repo_candidate


FIND2_DIR = resolve_path("data/paper_release/finding2/direct_check")
RESIDUALS_PATH = FIND2_DIR / "residuals_top100.jsonl"
HUMAN_RUBRIC_PATH = resolve_path("data/paper_release/rubrics/original/human_rubric_500cases.jsonl")
AI_RUBRIC_DIR = resolve_path("data/paper_release/rubrics/ai_rubrics")
OUT_BASE = FIND2_DIR / "raw_llm"

RUBRIC_MODELS = {
    "gpt54":        "gpt54.jsonl",
    "gemini31":     "gemini31_pro.jsonl",
    "opus46":       "claude_opus46.jsonl",
    "sonnet4":      "claude_sonnet4.jsonl",
    "ds_r1":        "deepseek_r1.jsonl",
    "ds_v32":       "deepseek_v32_exp.jsonl",
    "gemini25":     "gemini25_pro.jsonl",
    "gemini3flash": "gemini3_flash.jsonl",
    "kimi":         "kimi_k25.jsonl",
    "mimo":         "mimo_v2_pro.jsonl",
    "qwen35":       "qwen35_397b.jsonl",
}

JUDGE_MODEL = "openai/gpt-5.4"
MAX_RETRIES = 5
DEFAULT_WORKERS = 8
PROMPT_VERSION = "v2_high"


# ── prompt ───────────────────────────────────────────────────────────────────

def _fmt_criteria(crit_list: list[dict]) -> str:
    return "\n".join(f"{c['alias']} | {c['title']}" for c in crit_list)


SYSTEM = (
    "You are given two rubrics for the same moral dilemma. "
    "Return the IDs of criteria in Rubric H that have no counterpart in Rubric M, "
    "and the IDs of criteria in Rubric M that have no counterpart in Rubric H. "
    "If any criterion in the other rubric evaluates the same feature in the dilemma, that criterion counts as a counterpart even though the wording is different. "
    "It is completely ok if you are unable to find any unique criteria. "
    "Output exactly this JSON and nothing else:\n"
    '{"human_only": ["H001", ...], "model_only": ["M001", ...]}'
)


def build_prompt(dilemma: str, human: list[dict], model: list[dict]) -> str:
    return (
        f"Dilemma:\n{dilemma}\n\n"
        f"Rubric H:\n{_fmt_criteria(human)}\n\n"
        f"Rubric M:\n{_fmt_criteria(model)}"
    )


# ── data loading ──────────────────────────────────────────────────────────────

def load_rubric(path: Path) -> dict[str, dict]:
    cases = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rubric = row["RUBRIC"]
            if isinstance(rubric, str):
                rubric = json.loads(rubric)
            cases[row["TASK_ID"]] = {
                "dilemma": row.get("DILEMMA", ""),
                "criteria": [
                    {"id": c["id"], "title": c["title"], "weight": c.get("weight", c.get("WEIGHT", 0))}
                    for c in rubric
                ],
            }
    return cases


def load_residuals() -> dict[str, set[str]]:
    """Returns {task_id: {criterion_id, ...}} for the t80 residual pool."""
    result: dict[str, set[str]] = {}
    with RESIDUALS_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            for occ in r["rows"]:
                tid = occ["task_id"]
                cid = occ["criterion_id"]
                result.setdefault(tid, set()).add(cid)
    return result


def alias_criteria(criteria: list[dict], prefix: str) -> tuple[list[dict], dict[str, str]]:
    """Assign short aliases (H001, M001, ...) to avoid UUID fragility in LLM parsing."""
    aliased = []
    alias_to_id: dict[str, str] = {}
    for i, c in enumerate(criteria):
        alias = f"{prefix}{i+1:03d}"
        aliased.append({"alias": alias, "title": c["title"], "id": c["id"]})
        alias_to_id[alias] = c["id"]
    return aliased, alias_to_id


# ── API call ──────────────────────────────────────────────────────────────────

def call_llm(client, prompt: str, judge_model: str, max_tokens: int) -> tuple[str, int, int]:
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                extra_body={"reasoning": {"effort": "high"}},
            )
            content = resp.choices[0].message.content or ""
            in_tok = resp.usage.prompt_tokens if resp.usage else 0
            out_tok = resp.usage.completion_tokens if resp.usage else 0
            if not content.strip():
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** attempt
                    print(f"  empty content, retry {attempt+1}, sleeping {wait}s", flush=True)
                    time.sleep(wait)
                    continue
            return content, in_tok, out_tok
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  retry {attempt+1} ({e}), sleeping {wait}s", flush=True)
                time.sleep(wait)
            else:
                raise


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.find("\n") + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _extract_id(item) -> str | None:
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict):
        for key in ("alias", "criterion_id", "id"):
            if key in item and isinstance(item[key], str) and item[key].strip():
                return item[key].strip()
    return None


def parse_response(raw: str) -> tuple[list[str], list[str]]:
    candidate = _strip_fences(raw)
    m = re.search(r"\{[\s\S]*\}", candidate)
    payload = m.group(0) if m else candidate
    try:
        obj = json.loads(payload)
    except Exception:
        repaired = re.sub(r",(\s*[}\]])", r"\1", payload)
        obj = json.loads(repaired)
    human_only = [_id for item in obj.get("human_only", []) if (_id := _extract_id(item)) is not None]
    model_only = [_id for item in obj.get("model_only", []) if (_id := _extract_id(item)) is not None]
    return human_only, model_only


# ── resume ────────────────────────────────────────────────────────────────────

def load_done(out_file: Path, judge_model: str, rubric_model: str) -> set[str]:
    done = set()
    if not out_file.exists():
        return done
    with out_file.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                if (row.get("judge_model") == judge_model
                        and row.get("rubric_model") == rubric_model
                        and row.get("prompt_version") == PROMPT_VERSION):
                    done.add(row["task_id"])
            except Exception:
                pass
    return done


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ak", required=True, help="OpenRouter API key")
    parser.add_argument("--rubric_model", default="all",
                        choices=list(RUBRIC_MODELS) + ["all"],
                        help="Which model rubric to compare against (default: all 3)")
    parser.add_argument("--judge_model", default=JUDGE_MODEL)
    parser.add_argument("--max_tokens", type=int, default=32000)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--num_cases", type=int, default=None, help="Limit to first N cases")
    parser.add_argument("--debug", action="store_true", help="Run 1 case only")
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=args.ak, base_url="https://openrouter.ai/api/v1")

    residuals = load_residuals()
    target_cases = sorted(residuals.keys())
    if args.num_cases is not None:
        target_cases = target_cases[:args.num_cases]
    if args.debug:
        target_cases = target_cases[:1]

    human_rubric = load_rubric(HUMAN_RUBRIC_PATH)
    models_to_run = list(RUBRIC_MODELS) if args.rubric_model == "all" else [args.rubric_model]

    judge_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", args.judge_model)

    for rmodel in models_to_run:
        print(f"\n{'='*60}")
        print(f"Rubric model: {rmodel}  Judge: {args.judge_model}")
        print(f"Cases: {len(target_cases)}  Workers: {args.workers}")

        ai_rubric = load_rubric(AI_RUBRIC_DIR / RUBRIC_MODELS[rmodel])

        out_dir = OUT_BASE / rmodel / judge_slug / PROMPT_VERSION
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "results.jsonl"

        done = load_done(out_file, args.judge_model, rmodel)
        todo = [c for c in target_cases if c not in done]
        print(f"Already done: {len(done)}  Remaining: {len(todo)}")
        if not todo:
            print("Nothing to do.")
            continue

        def process(task_id: str):
            hcase = human_rubric[task_id]
            acase = ai_rubric[task_id]

            residual_ids = residuals.get(task_id, set())
            human_crits_raw = hcase["criteria"]

            human_aliased, h_alias_to_id = alias_criteria(human_crits_raw, "H")
            model_aliased, m_alias_to_id = alias_criteria(acase["criteria"], "M")

            prompt = build_prompt(hcase["dilemma"], human_aliased, model_aliased)
            raw, in_tok, out_tok = call_llm(client, prompt, args.judge_model, args.max_tokens)

            try:
                human_only_aliases, model_only_aliases = parse_response(raw)
            except Exception as e:
                human_only_aliases, model_only_aliases = [], []
                print(f"  parse error {task_id}: {e}", flush=True)

            human_only_ids = [h_alias_to_id[a] for a in human_only_aliases if a in h_alias_to_id]
            model_only_ids = [m_alias_to_id[a] for a in model_only_aliases if a in m_alias_to_id]

            human_ids_set = set(human_only_ids)
            confirmed = list(residual_ids & human_ids_set)
            missed = list(residual_ids - human_ids_set)

            return {
                "task_id": task_id,
                "rubric_model": rmodel,
                "judge_model": args.judge_model,
                "prompt_version": PROMPT_VERSION,
                "n_human_criteria": len(human_crits_raw),
                "n_model_criteria": len(acase["criteria"]),
                "human_only_aliases": human_only_aliases,
                "human_only": human_only_ids,
                "model_only_aliases": model_only_aliases,
                "model_only": model_only_ids,
                "residual_ids_in_case": list(residual_ids),
                "residual_confirmed": confirmed,
                "residual_missed": missed,
                "raw_response": raw,
                "in_tokens": in_tok,
                "out_tokens": out_tok,
            }, in_tok, out_tok

        success = fail = total_in = total_out = 0
        with out_file.open("a", encoding="utf-8") as fw:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(process, tid): tid for tid in todo}
                for fut in as_completed(futures):
                    tid = futures[fut]
                    try:
                        rec, in_tok, out_tok = fut.result()
                        fw.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fw.flush()
                        success += 1
                        total_in += in_tok
                        total_out += out_tok
                        conf = len(rec["residual_confirmed"])
                        miss = len(rec["residual_missed"])
                        print(f"  {tid}  confirmed={conf}  missed={miss}", flush=True)
                    except Exception as e:
                        fail += 1
                        print(f"  FAIL {tid}: {e}", flush=True)

        print(f"\nDone: success={success}, fail={fail}")
        print(f"Tokens: {total_in:,} in / {total_out:,} out")
        print(f"Output: {out_file}")


if __name__ == "__main__":
    main()
