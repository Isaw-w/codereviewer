#!/usr/bin/env python3
"""
Incremental rubric-judge runner for the refreshed OpenRouter-high line.

Supports:
- non-Claude refreshed rubric roots
- pilot-first (5 cases by default)
- two formal match modes:
  - allow_repeat
  - online_pool_remove
- single-judge config (currently OSS120B-high)
- resume-safe reruns using one persistent out_dir per (match_mode, rubric_model)
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent
RUNNER = CODE_ROOT / "criterion_match" / "run_criterion_simplematch_parallel_cases.py"


@dataclass
class RubricJudgeTarget:
    tag: str
    enabled: bool = True


@dataclass
class JudgeSpec:
    tag: str
    api_provider: str
    model: str
    api_key_env: str
    reasoning_effort: str = "high"
    max_workers: int = 6
    max_tokens: int = 10500
    max_retries: int = 3
    request_timeout: int = 180
    enabled: bool = True


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def bootstrap_env() -> None:
    for candidate in [RELEASE_ROOT / ".env", REPO_ROOT / ".env"]:
        load_env_file(candidate)


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


def build_output_base(outputs_root: Path, date_value: str, outputs_subdir: str) -> Path:
    if date_value.strip():
        return outputs_root / date_value / outputs_subdir if outputs_subdir else outputs_root / date_value
    return outputs_root / outputs_subdir if outputs_subdir else outputs_root


def parse_targets(raw_models: Iterable[Dict[str, object]]) -> List[RubricJudgeTarget]:
    out: List[RubricJudgeTarget] = []
    for i, item in enumerate(raw_models):
        try:
            out.append(
                RubricJudgeTarget(
                    tag=str(item["tag"]),
                    enabled=bool(item.get("enabled", True)),
                )
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing required key in rubric judge target #{i}: {exc}") from exc
    return out


def parse_judges(raw_judges: Iterable[Dict[str, object]]) -> List[JudgeSpec]:
    out: List[JudgeSpec] = []
    for i, item in enumerate(raw_judges):
        try:
            out.append(
                JudgeSpec(
                    tag=str(item["tag"]),
                    api_provider=str(item["api_provider"]),
                    model=str(item["model"]),
                    api_key_env=str(item["api_key_env"]),
                    reasoning_effort=str(item.get("reasoning_effort", "high")),
                    max_workers=int(item.get("max_workers", 6)),
                    max_tokens=int(item.get("max_tokens", 10500)),
                    max_retries=int(item.get("max_retries", 3)),
                    request_timeout=int(item.get("request_timeout", 180)),
                    enabled=bool(item.get("enabled", True)),
                )
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing required key in judge config entry #{i}: {exc}") from exc
    return out


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def count_ok_cases(run_dir: Path) -> int:
    status_path = run_dir / "case_status.jsonl"
    if status_path.exists():
        last_by_task: Dict[str, Dict] = {}
        for row in read_jsonl(status_path):
            task_id = row.get("TASK_ID")
            if task_id:
                last_by_task[task_id] = row
        return sum(1 for row in last_by_task.values() if row.get("status") == "ok")

    case_jobs = run_dir / "case_jobs"
    if not case_jobs.exists():
        return 0

    ok = 0
    for status_file in case_jobs.glob("*/case_status.jsonl"):
        rows = read_jsonl(status_file)
        if rows and rows[-1].get("status") == "ok":
            ok += 1
    return ok


def compute_parallelism(shared_workers: int, job_count: int) -> Tuple[int, int]:
    parallel_jobs = min(job_count, max(1, shared_workers))
    per_job_workers = max(1, shared_workers // parallel_jobs)
    return parallel_jobs, per_job_workers


def run_is_complete(run_dir: Path, expected_cases: int, match_mode: str) -> bool:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if summary.get("match_mode") != match_mode:
        return False
    return count_ok_cases(run_dir) >= expected_cases


def run_rubric_judge(
    *,
    python_bin: Path,
    run_dir: Path,
    rubrics_root: Path,
    rubric_model: str,
    judge: JudgeSpec,
    match_mode: str,
    workers: int,
    cases: int,
    max_tokens: int,
    max_retries: int,
    request_timeout: int,
    weights: str,
    dry_run: bool,
) -> None:
    cmd = [
        str(python_bin),
        str(RUNNER),
        "--rubric_model",
        rubric_model,
        "--rubrics_root",
        str(rubrics_root),
        "--judge_model",
        judge.model,
        "--api_provider",
        judge.api_provider,
        "--api_key_env",
        judge.api_key_env,
        "--match_mode",
        match_mode,
        "--cases",
        str(cases),
        "--case_start",
        "0",
        "--max_parallel_cases",
        str(workers),
        "--request_timeout",
        str(request_timeout),
        "--max_retries",
        str(max_retries),
        "--reasoning_effort",
        judge.reasoning_effort,
        "--temperature",
        "0.0",
        "--top_p",
        "0.01",
        "--max_tokens",
        str(max_tokens),
        "--out_dir",
        str(run_dir),
        "--repo_root",
        str(RELEASE_ROOT),
    ]
    if weights:
        cmd += ["--weights", weights]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CODE_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(CODE_ROOT)
    env["PYTHONUNBUFFERED"] = "1"

    print("CMD:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=RELEASE_ROOT, env=env, check=True)


def run_one_judge_task(
    *,
    target: RubricJudgeTarget,
    match_mode: str,
    python_bin: Path,
    run_dir: Path,
    rubrics_root: Path,
    judge: JudgeSpec,
    cases: int,
    pilot_cases: int,
    pilot_only: bool,
    skip_pilot: bool,
    workers: int,
    request_timeout: int,
    max_retries: int,
    max_tokens: int,
    weights: str,
    dry_run: bool,
) -> Tuple[str, str, str]:
    print(f"\n=== [{target.tag}] mode={match_mode} ===")
    if run_is_complete(run_dir, cases, match_mode):
        print(f"Already complete: {cases}/{cases}. Skip.")
        return (target.tag, match_mode, "done_already")

    existing_ok = count_ok_cases(run_dir)
    if not skip_pilot and existing_ok < pilot_cases:
        print(f"Running pilot {pilot_cases} cases...")
        run_rubric_judge(
            python_bin=python_bin,
            run_dir=run_dir,
            rubrics_root=rubrics_root,
            rubric_model=target.tag,
            judge=judge,
            match_mode=match_mode,
            workers=workers,
            cases=pilot_cases,
            max_tokens=max_tokens,
            max_retries=max_retries,
            request_timeout=request_timeout,
            weights=weights,
            dry_run=dry_run,
        )
        pilot_ok = count_ok_cases(run_dir)
        if pilot_ok < pilot_cases and not dry_run:
            raise RuntimeError(
                f"Pilot failed for {target.tag}/{match_mode}: expected >= {pilot_cases} ok, got {pilot_ok}"
            )
    else:
        print(f"Pilot already satisfied: {existing_ok}/{pilot_cases}")

    if pilot_only:
        return (target.tag, match_mode, "pilot_only_done")

    print(f"Running full increment (resume): existing {count_ok_cases(run_dir)}/{cases}")
    run_rubric_judge(
        python_bin=python_bin,
        run_dir=run_dir,
        rubrics_root=rubrics_root,
        rubric_model=target.tag,
        judge=judge,
        match_mode=match_mode,
        workers=workers,
        cases=cases,
        max_tokens=max_tokens,
        max_retries=max_retries,
        request_timeout=request_timeout,
        weights=weights,
        dry_run=dry_run,
    )
    status = "done" if run_is_complete(run_dir, cases, match_mode) or dry_run else "partial"
    print(f"Done count: {count_ok_cases(run_dir)}/{cases}")
    return (target.tag, match_mode, status)


def selected_match_modes(config_modes: Sequence[str], cli_mode: str) -> List[str]:
    if cli_mode == "both":
        return list(config_modes)
    if cli_mode not in {"allow_repeat", "online_pool_remove"}:
        raise RuntimeError(f"Unsupported match mode: {cli_mode}")
    return [cli_mode]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrix",
        default="code/configs/rubric_judge_matrix_refresh_openrouter_high.json",
        help="Rubric judge matrix JSON path.",
    )
    ap.add_argument(
        "--judge_matrix",
        default="code/configs/judge_matrix_oss120b_medium.json",
        help="Single-judge config JSON path.",
    )
    ap.add_argument("--date", default="", help="Optional date folder. Leave empty to write directly into the configured release path.")
    ap.add_argument(
        "--rubrics_date",
        default="",
        help="Optional date folder that contains rubric-generation outputs. Leave empty for the configured release path.",
    )
    ap.add_argument("--rubrics_root", default="", help="Override refreshed rubrics root directly.")
    ap.add_argument("--pilot_cases", type=int, default=5)
    ap.add_argument("--skip_pilot", action="store_true")
    ap.add_argument("--pilot_only", action="store_true")
    ap.add_argument("--models", default="", help="Comma-separated rubric model tags to run.")
    ap.add_argument(
        "--match_mode",
        choices=["allow_repeat", "online_pool_remove", "both"],
        default="both",
        help="Which rubric-judge line(s) to run.",
    )
    ap.add_argument("--workers", type=int, default=0, help="Override judge max_parallel_cases.")
    ap.add_argument("--request_timeout", type=int, default=0)
    ap.add_argument("--max_retries", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=0)
    ap.add_argument("--python_bin", default=".venv/bin/python")
    ap.add_argument("--weights", default="", help="Weight levels to match, e.g. '-1,-2,-3'. Passed to parallel_cases runner.")
    ap.add_argument("--output_subdir", default="", help="Override output_subdir from matrix config.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stop_on_error", action="store_true")
    args = ap.parse_args()

    if args.skip_pilot and args.pilot_only:
        raise RuntimeError("--skip_pilot and --pilot_only cannot be used together.")

    bootstrap_env()

    matrix_path = resolve_path(args.matrix)
    judge_matrix_path = resolve_path(args.judge_matrix)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    judge_matrix = json.loads(judge_matrix_path.read_text(encoding="utf-8"))

    targets = [t for t in parse_targets(matrix["models"]) if t.enabled]
    if args.models.strip():
        selected = {x.strip() for x in args.models.split(",") if x.strip()}
        known = {t.tag for t in targets}
        unknown = sorted(selected - known)
        if unknown:
            raise RuntimeError(f"Unknown rubric tags: {unknown}. Known enabled tags: {sorted(known)}")
        targets = [t for t in targets if t.tag in selected]
    if not targets:
        raise RuntimeError("No rubric models selected to run.")

    judges = [j for j in parse_judges(judge_matrix["judges"]) if j.enabled]
    if len(judges) != 1:
        raise RuntimeError("This runner currently expects exactly one enabled judge.")
    judge = judges[0]

    python_bin = resolve_path(args.python_bin)
    if not python_bin.exists():
        raise FileNotFoundError(f"Python binary not found: {python_bin}")
    if not RUNNER.exists():
        raise FileNotFoundError(f"Parallel rubric-judge runner not found: {RUNNER}")
    if not os.environ.get(judge.api_key_env) and not args.dry_run:
        raise RuntimeError(f"Missing API key in env var {judge.api_key_env}")

    outputs_root = resolve_path(str(matrix.get("outputs_root", "data/canonical_full")))
    rubrics_root = (
        resolve_path(args.rubrics_root)
        if args.rubrics_root
        else build_output_base(outputs_root, args.rubrics_date, str(matrix.get("rubrics_subdir", "rubrics")))
    )
    if not rubrics_root.exists():
        raise FileNotFoundError(f"Rubrics root not found: {rubrics_root}")

    output_subdir = args.output_subdir or str(matrix.get("output_subdir", "criterion_match"))
    target_cases = int(matrix.get("target_cases", 500))
    match_modes = selected_match_modes(matrix.get("match_modes", ["allow_repeat"]), args.match_mode)
    out_base = build_output_base(outputs_root, args.date, output_subdir)
    out_base.mkdir(parents=True, exist_ok=True)

    workers = args.workers or judge.max_workers
    request_timeout = args.request_timeout or judge.request_timeout
    max_retries = args.max_retries or judge.max_retries
    max_tokens = args.max_tokens or judge.max_tokens

    print(f"Loaded {len(targets)} rubric targets from: {matrix_path}")
    print(f"Judge config: {judge_matrix_path}")
    print(f"Rubrics root: {rubrics_root}")
    print(f"Judge output root: {out_base}")
    print(f"Match modes: {match_modes}")

    shared_workers = args.workers or judge.max_workers
    tasks = [(target, match_mode) for target in targets for match_mode in match_modes]
    parallel_jobs, per_job_workers = compute_parallelism(shared_workers, len(tasks))
    print(
        f"Cross-task parallelism: {parallel_jobs} job(s); shared workers: {shared_workers}; "
        f"per-task workers: {per_job_workers}"
    )

    results_by_task: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
        futures = {}
        for target, match_mode in tasks:
            run_dir = out_base / match_mode / target.tag
            run_dir.mkdir(parents=True, exist_ok=True)
            futures[
                executor.submit(
                    run_one_judge_task,
                    target=target,
                    match_mode=match_mode,
                    python_bin=python_bin,
                    run_dir=run_dir,
                    rubrics_root=rubrics_root,
                    judge=judge,
                    cases=target_cases,
                    pilot_cases=args.pilot_cases,
                    pilot_only=args.pilot_only,
                    skip_pilot=args.skip_pilot,
                    workers=per_job_workers,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                    max_tokens=max_tokens,
                    weights=args.weights,
                    dry_run=args.dry_run,
                )
            ] = (target.tag, match_mode)
        for future in as_completed(futures):
            tag, match_mode = futures[future]
            try:
                results_by_task[(tag, match_mode)] = future.result()
            except Exception as exc:  # noqa: BLE001
                if args.stop_on_error:
                    raise
                print(f"ERROR [{tag}::{match_mode}]: {exc}")
                results_by_task[(tag, match_mode)] = (tag, match_mode, "error")

    summary = [
        results_by_task[(target.tag, match_mode)]
        for target in targets
        for match_mode in match_modes
        if (target.tag, match_mode) in results_by_task
    ]

    print("\n=== Summary ===")
    for tag, mode, status in summary:
        print(f"{tag:32s}  {mode:20s}  {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
