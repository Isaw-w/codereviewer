#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


SCRIPT_PATH = Path(__file__).parent / "run_criterion_simplematch_serial.py"
PROMPT_VERSION = "v2_find_matched_items_besttarget_20260325"
COVERAGE_RULE = "covered = matched + partial"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_task_id(raw: object) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"none", "null", "nan"}:
        return None
    return s


def locate_rubric_files(base_dir: Path, rubric_model: str) -> tuple[Path, Path]:
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


def selected_task_ids_from_rubrics(
    rubrics_root: Path,
    rubric_model: str,
    case_start: int,
    cases: int,
) -> List[str]:
    ai_path, human_path = locate_rubric_files(rubrics_root, rubric_model)
    ai_rows = read_jsonl(ai_path)
    human_ids = {
        tid
        for tid in (_normalize_task_id(row.get("TASK_ID", row.get("idx"))) for row in read_jsonl(human_path))
        if tid is not None
    }
    ordered_common: List[str] = []
    seen = set()
    for row in ai_rows:
        tid = _normalize_task_id(row.get("TASK_ID", row.get("idx")))
        if tid is None or tid in seen or tid not in human_ids:
            continue
        ordered_common.append(tid)
        seen.add(tid)
    return ordered_common[case_start : case_start + cases]


def mean(rows: List[Dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r[key]) for r in rows) / len(rows)


def _metric_value(row: Dict[str, Any], key: str) -> float:
    if key in row:
        return float(row[key])
    human_total = max(1, int(row.get("human_criteria", 0) or 0))
    model_total = max(1, int(row.get("model_criteria", 0) or 0))
    if key == "h2m_partial_rate":
        return float(row.get("h2m_partial", 0)) / human_total
    if key == "m2h_partial_rate":
        return float(row.get("m2h_partial", 0)) / model_total
    if key == "h2m_covered":
        return float(int(row.get("h2m_matched", 0)) + int(row.get("h2m_partial", 0)))
    if key == "m2h_covered":
        return float(int(row.get("m2h_matched", 0)) + int(row.get("m2h_partial", 0)))
    if key == "h2m_covered_rate":
        return _metric_value(row, "h2m_covered") / human_total
    if key == "m2h_covered_rate":
        return _metric_value(row, "m2h_covered") / model_total
    raise KeyError(key)


def ensure_coverage_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    out.setdefault("h2m_covered", int(_metric_value(out, "h2m_covered")))
    out.setdefault("m2h_covered", int(_metric_value(out, "m2h_covered")))
    out.setdefault("h2m_covered_rate", _metric_value(out, "h2m_covered_rate"))
    out.setdefault("m2h_covered_rate", _metric_value(out, "m2h_covered_rate"))
    out.setdefault("h2m_partial_rate", _metric_value(out, "h2m_partial_rate"))
    out.setdefault("m2h_partial_rate", _metric_value(out, "m2h_partial_rate"))
    return out


def load_existing_ok_cases(out_dir: Path, task_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    for task_id in task_ids:
        job_dir = out_dir / "case_jobs" / task_id
        status_path = job_dir / "case_status.jsonl"
        root_case_json = out_dir / "case_runs" / task_id / "cases" / f"{task_id}.json"
        job_case_json = job_dir / "case_runs" / task_id / "cases" / f"{task_id}.json"
        if not status_path.exists():
            continue
        status_rows = read_jsonl(status_path)
        if not status_rows:
            continue
        last_status = status_rows[-1]
        if last_status.get("status") != "ok":
            continue
        case_json = root_case_json if root_case_json.exists() else job_case_json
        if not case_json.exists():
            continue
        try:
            obj = read_json(case_json)
            metrics = ensure_coverage_metrics(obj["metrics"])
            obj["metrics"] = metrics
        except Exception:
            continue
        existing[task_id] = {
            "TASK_ID": task_id,
            "returncode": 0,
            "stdout": "[resumed from existing case output]",
            "stderr": "",
            "job_dir": str(job_dir),
            "resumed": True,
            "metrics": metrics,
            "status_rows": status_rows,
            "case_obj": obj,
        }
    return existing


def build_summary(case_metrics: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    def summary_mean(key: str) -> float:
        return sum(_metric_value(row, key) for row in case_metrics) / len(case_metrics)

    return {
        "kind": "criterion_simple_match",
        "rubric_model": args.rubric_model,
        "judge_model": args.judge_model,
        "match_mode": args.match_mode,
        "cases": len(case_metrics),
        "reasoning_effort": args.reasoning_effort or None,
        "prompt_version": PROMPT_VERSION,
        "coverage_rule": COVERAGE_RULE,
        "partial_counts_as_match": True,
        "h2m_covered_rate_mean": summary_mean("h2m_covered_rate"),
        "m2h_covered_rate_mean": summary_mean("m2h_covered_rate"),
        "h2m_covered_count_mean": summary_mean("h2m_covered"),
        "m2h_covered_count_mean": summary_mean("m2h_covered"),
        "h2m_match_rate_mean": summary_mean("h2m_match_rate"),
        "m2h_match_rate_mean": summary_mean("m2h_match_rate"),
        "h2m_partial_rate_mean": summary_mean("h2m_partial_rate"),
        "m2h_partial_rate_mean": summary_mean("m2h_partial_rate"),
        "h2m_partial_count_mean": summary_mean("h2m_partial"),
        "m2h_partial_count_mean": summary_mean("m2h_partial"),
        "task_ids": [row["TASK_ID"] for row in case_metrics],
    }

def build_progress_summary(
    results: List[Dict[str, Any]], task_ids: List[str], args: argparse.Namespace
) -> Dict[str, Any]:
    completed = len(results)
    resumed = sum(1 for r in results if r.get("resumed"))
    ok = 0
    error = 0
    for r in results:
        if r.get("resumed"):
            ok += 1
            continue
        if r.get("returncode", 1) == 0:
            ok += 1
        else:
            error += 1
    pending = max(0, len(task_ids) - completed)
    return {
        "kind": "criterion_simple_match_progress",
        "rubric_model": args.rubric_model,
        "judge_model": args.judge_model,
        "match_mode": args.match_mode,
        "reasoning_effort": args.reasoning_effort or None,
        "prompt_version": PROMPT_VERSION,
        "coverage_rule": COVERAGE_RULE,
        "partial_counts_as_match": True,
        "total_cases": len(task_ids),
        "completed_cases": completed,
        "pending_cases": pending,
        "ok_cases": ok,
        "error_cases": error,
        "resumed_cases": resumed,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
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
        f"- Mean H->M partial rate: `{summary['h2m_partial_rate_mean']:.4f}`",
        f"- Mean M->H partial rate: `{summary['m2h_partial_rate_mean']:.4f}`",
        "",
        "| TASK_ID | H->M covered | H->M matched | H->M partial | H->M none | M->H covered | M->H matched | M->H partial | M->H none |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in case_metrics:
        lines.append(
            f"| {row['TASK_ID']} | {int(_metric_value(row, 'h2m_covered'))}/{row['human_criteria']} | "
            f"{row['h2m_matched']}/{row['human_criteria']} | {row['h2m_partial']}/{row['human_criteria']} | "
            f"{row['h2m_none']}/{row['human_criteria']} | {int(_metric_value(row, 'm2h_covered'))}/{row['model_criteria']} | "
            f"{row['m2h_matched']}/{row['model_criteria']} | {row['m2h_partial']}/{row['model_criteria']} | "
            f"{row['m2h_none']}/{row['model_criteria']} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def run_one(task_id: str, args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    job_dir = out_dir / "case_jobs" / task_id
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--rubric_model",
        args.rubric_model,
        "--rubrics_root",
        args.rubrics_root,
        "--judge_model",
        args.judge_model,
        "--api_provider",
        args.api_provider,
        "--api_key_env",
        args.api_key_env,
        "--task_ids",
        task_id,
        "--match_mode",
        args.match_mode,
        "--request_timeout",
        str(args.request_timeout),
        "--max_retries",
        str(args.max_retries),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--max_tokens",
        str(args.max_tokens),
        "--out_dir",
        str(job_dir),
    ]
    if args.reasoning_effort:
        cmd += ["--reasoning_effort", args.reasoning_effort]
    if args.weights:
        cmd += ["--weights", args.weights]
    env = dict(**os.environ)
    env["PYTHONPATH"] = "."
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, cwd=args.repo_root, env=env, capture_output=True, text=True)
    return {
        "TASK_ID": task_id,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-2000:],
        "stderr": proc.stderr[-2000:],
        "job_dir": str(job_dir),
        "resumed": False,
    }


def materialize_progress(
    out_dir: Path,
    results: List[Dict[str, Any]],
    task_ids: List[str],
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    results_sorted = sorted(results, key=lambda r: r["TASK_ID"])
    write_jsonl(out_dir / "runner_results.jsonl", results_sorted)

    case_status_rows: List[Dict[str, Any]] = []
    case_metrics: List[Dict[str, Any]] = []
    for r in results_sorted:
        task_id = r["TASK_ID"]
        if r.get("resumed"):
            status_rows = r["status_rows"]
            obj = r["case_obj"]
            case_status_rows.extend(status_rows)
            write_json(out_dir / "case_runs" / task_id / "cases" / f"{task_id}.json", obj)
            case_metrics.append(ensure_coverage_metrics(r["metrics"]))
            continue

        job_dir = Path(r["job_dir"])
        status_rows = read_jsonl(job_dir / "case_status.jsonl")
        if status_rows:
            case_status_rows.extend(status_rows)

        case_json = job_dir / "case_runs" / task_id / "cases" / f"{task_id}.json"
        if case_json.exists():
            obj = read_json(case_json)
            obj["metrics"] = ensure_coverage_metrics(obj["metrics"])
            write_json(out_dir / "case_runs" / task_id / "cases" / f"{task_id}.json", obj)
            case_metrics.append(obj["metrics"])
        elif not status_rows:
            case_status_rows.append(
                {
                    "TASK_ID": task_id,
                    "status": "error",
                    "elapsed_sec": 0.0,
                    "error": f"subprocess rc={r['returncode']}",
                }
            )

    case_status_rows.sort(key=lambda r: r["TASK_ID"])
    case_metrics.sort(key=lambda r: r["TASK_ID"])
    write_jsonl(out_dir / "case_status.jsonl", case_status_rows)
    write_jsonl(out_dir / "case_metrics.jsonl", case_metrics)
    write_json(out_dir / "progress_summary.json", build_progress_summary(results_sorted, task_ids, args))
    return case_status_rows, case_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rubric_model", default="gpt52_openrouter")
    ap.add_argument(
        "--rubrics_root",
        required=True,
        help="Directory containing per-model rubric subdirectories (each with human_rubric_*.jsonl and ai_rubric_*.jsonl)",
    )
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--api_provider", "-ap", default="openrouter")
    ap.add_argument("--api_key_env", "-k", default="LAB_OPENROUTER_KEY")
    ap.add_argument("--case_start", type=int, default=0)
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--max_parallel_cases", type=int, default=10)
    ap.add_argument("--request_timeout", type=int, default=120)
    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--match_mode", choices=["allow_repeat", "online_pool_remove"], default="allow_repeat")
    ap.add_argument("--reasoning_effort", choices=["low", "medium", "high"], default="")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=0.01)
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--run_name", default="")
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--weights", default="3,2,1,-1,-2,-3",
                    help="Comma-separated weight levels to match (default: all)")
    ap.add_argument("--repo_root", default=".")
    args = ap.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
        run_name = out_dir.name
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        run_name = (
            args.run_name
            or f"criterion_{args.match_mode}_parallel_{args.rubric_model}_{args.cases}cases"
        )
        out_dir = Path(f"outputs/{today}/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    task_ids = selected_task_ids_from_rubrics(Path(args.rubrics_root), args.rubric_model, args.case_start, args.cases)
    if not task_ids:
        raise SystemExit(
            f"No task ids selected from {args.rubrics_root}/{args.rubric_model} "
            f"for case_start={args.case_start}, cases={args.cases}"
        )
    existing_ok = load_existing_ok_cases(out_dir, task_ids)
    pending_task_ids = [tid for tid in task_ids if tid not in existing_ok]

    results: List[Dict[str, Any]] = list(existing_ok.values())
    case_status_rows, case_metrics = materialize_progress(out_dir, results, task_ids, args)
    if pending_task_ids:
        with ThreadPoolExecutor(max_workers=args.max_parallel_cases) as ex:
            futs = {ex.submit(run_one, tid, args, out_dir): tid for tid in pending_task_ids}
            for fut in as_completed(futs):
                result = fut.result()
                results.append(result)
                case_status_rows, case_metrics = materialize_progress(out_dir, results, task_ids, args)
                status_label = "ok" if result.get("returncode", 1) == 0 else "error"
                print(
                    f"[progress] rubric_model={args.rubric_model} judge_model={args.judge_model} "
                    f"match_mode={args.match_mode} task_id={result['TASK_ID']} "
                    f"status={status_label} completed={len(results)}/{len(task_ids)}",
                    flush=True,
                )

    ok_cases = [r for r in case_status_rows if r.get("status") == "ok"]
    if len(ok_cases) != len(task_ids):
        raise SystemExit(f"Only {len(ok_cases)}/{len(task_ids)} cases succeeded; see {out_dir}")

    summary = build_summary(case_metrics, args)
    write_json(out_dir / "summary.json", summary)
    build_summary_md(summary, case_metrics, out_dir / "summary.md")
    write_json(
        out_dir / "manifest.json",
        {
            "rubric_model": args.rubric_model,
            "judge_model": args.judge_model,
            "match_mode": args.match_mode,
            "reasoning_effort": args.reasoning_effort or None,
            "prompt_version": PROMPT_VERSION,
            "coverage_rule": COVERAGE_RULE,
            "partial_counts_as_match": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "cases": args.cases,
            "case_start": args.case_start,
            "max_parallel_cases": args.max_parallel_cases,
            "selected_ids": task_ids,
            "kind": f"criterion_simple_match_parallel_{args.match_mode}",
            "out_dir": str(out_dir),
        },
    )
    print(json.dumps({"out_dir": str(out_dir), "selected_ids": task_ids}, ensure_ascii=False))


if __name__ == "__main__":
    main()
