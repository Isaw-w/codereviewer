#!/usr/bin/env python3
"""
Incremental answer-under-rubric evaluation runner.

Supports:
- reused Claude answers/rubrics plus refreshed non-Claude roots
- pilot-first (5 cases by default)
- resume-safe judging via run_best_judge_on_responses.py
- per-model summary outputs for human-rubric and model-rubric branches
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent
JUDGE_SCRIPT = CODE_ROOT / "run_best_judge_on_responses.py"
SCORE_SCRIPT = CODE_ROOT / "calculate_morebench.py"


@dataclass
class AnswerEvalSpec:
    label: str
    response_tag: str
    rubric_tag: str
    reuse_response: bool = False
    reuse_rubric: bool = False
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


def path_for_summary(path: Path) -> str:
    for base in (RELEASE_ROOT, REPO_ROOT):
        try:
            return str(path.relative_to(base))
        except ValueError:
            continue
    return str(path)


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_response_rows(rows: List[Dict], *, model_tag: str, response_file: Path) -> None:
    bad_rows = []
    for row in rows:
        resp = row.get("model_resp")
        if resp is None or (isinstance(resp, str) and not resp.strip()):
            bad_rows.append(
                {
                    "idx": row.get("idx"),
                    "task_id": row.get("TASK_ID") or row.get("task_id"),
                }
            )
    if not bad_rows:
        return
    preview = ", ".join(
        f"idx={r['idx']} task_id={r['task_id']}" for r in bad_rows[:10]
    )
    more = "" if len(bad_rows) <= 10 else f" ... (+{len(bad_rows) - 10} more)"
    raise RuntimeError(
        f"Source response file has {len(bad_rows)} blank/None model_resp row(s) for {model_tag}: "
        f"{preview}{more}. Source: {response_file}"
    )


def parse_models(raw_models: Iterable[Dict[str, object]]) -> List[AnswerEvalSpec]:
    out: List[AnswerEvalSpec] = []
    for i, m in enumerate(raw_models):
        try:
            out.append(
                AnswerEvalSpec(
                    label=str(m["label"]),
                    response_tag=str(m["response_tag"]),
                    rubric_tag=str(m["rubric_tag"]),
                    reuse_response=bool(m.get("reuse_response", False)),
                    reuse_rubric=bool(m.get("reuse_rubric", False)),
                    enabled=bool(m.get("enabled", True)),
                )
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing required key in answer eval matrix entry #{i}: {exc}") from exc
    return out


def run_slug_for_spec(spec: AnswerEvalSpec, response_tag_counts: Dict[str, int]) -> str:
    if response_tag_counts.get(spec.response_tag, 0) <= 1:
        return spec.response_tag
    return f"{spec.response_tag}__rubric__{spec.rubric_tag}"


def display_label_for_spec(spec: AnswerEvalSpec, response_tag_counts: Dict[str, int]) -> str:
    if response_tag_counts.get(spec.response_tag, 0) <= 1:
        return spec.label
    return f"{spec.label} | response={spec.response_tag} rubric={spec.rubric_tag}"


def selector_keys_for_spec(spec: AnswerEvalSpec, response_tag_counts: Dict[str, int]) -> set[str]:
    run_slug = run_slug_for_spec(spec, response_tag_counts)
    return {spec.response_tag, run_slug}


def normalize_rubric_field(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list):
            raise ValueError("Parsed RUBRIC is not a list")
        return parsed
    raise ValueError(f"Unsupported RUBRIC type: {type(value).__name__}")


def build_runtime_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CODE_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(CODE_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def find_single_jsonl(directory: Path) -> Path:
    files = sorted(directory.glob("*.jsonl"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one jsonl in {directory}, found {len(files)}")
    return files[0]


def find_rubric_files(directory: Path) -> Tuple[Path, Path]:
    exact_ai = directory / f"ai_rubric_500cases_{directory.name}_seed0.jsonl"
    exact_human = directory / f"human_rubric_500cases_{directory.name}_seed0.jsonl"
    if exact_ai.exists() and exact_human.exists():
        return exact_ai, exact_human
    ai = sorted(
        p
        for p in directory.glob("*.jsonl")
        if p.name.startswith("ai_rubric_") and not p.name.startswith("pilot") and "backup" not in p.stem
    )
    human = sorted(
        p
        for p in directory.glob("*.jsonl")
        if p.name.startswith("human_rubric_") and not p.name.startswith("pilot") and "backup" not in p.stem
    )
    if len(ai) != 1 or len(human) != 1:
        raise ValueError(
            f"Expected one non-pilot ai and one non-pilot human rubric file in {directory}; "
            f"found ai={len(ai)}, human={len(human)}"
        )
    return ai[0], human[0]


def parse_score_stdout(stdout: str):
    out = {"overall": None, "normalized": None, "avg_len": None}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("Overall Score:"):
            out["overall"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("Average Response Length:"):
            match = re.search(r"[-+]?[0-9]*\.?[0-9]+", line)
            if not match:
                raise ValueError(f"Could not parse average response length from line: {line}")
            out["avg_len"] = float(match.group(0))
        elif line.startswith("Normalized Score"):
            out["normalized"] = float(line.split(":", 1)[1].strip())
    return out


def redact_cmd(cmd: List[str]) -> List[str]:
    redacted = list(cmd)
    for i, token in enumerate(redacted[:-1]):
        if token in {"-ak", "--api_key"}:
            redacted[i + 1] = "***REDACTED***"
    return redacted


def run(cmd, env=None, dry_run: bool = False):
    print("CMD:", " ".join(redact_cmd(cmd)))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")
    runtime_env = dict(env or build_runtime_env())
    runtime_env.setdefault("PYTHONPATH", str(CODE_ROOT))
    return subprocess.run(
        cmd,
        cwd=RELEASE_ROOT,
        env=runtime_env,
        text=True,
        capture_output=True,
    )


def format_progress(done: int, expected: int) -> str:
    if expected <= 0:
        return f"{done}/0"
    pct = done / expected * 100
    return f"{done}/{expected} ({pct:.1f}%)"


def run_judge_with_progress(
    *,
    cmd: List[str],
    env: Dict[str, str],
    dry_run: bool,
    model_tag: str,
    branch_name: str,
    output_judgement: Path,
    expected_samples: int,
    expected_input_rows: int,
) -> subprocess.CompletedProcess:
    print(
        f"[branch] {model_tag} {branch_name} start: "
        f"{format_progress(count_jsonl_rows(output_judgement), expected_samples)} criteria, "
        f"{expected_input_rows} cases"
    )
    print("CMD:", " ".join(redact_cmd(cmd)))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, "", "")

    proc = subprocess.Popen(
        cmd,
        cwd=RELEASE_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    last_reported = -1
    last_report_ts = 0.0
    poll_sec = 5.0

    while proc.poll() is None:
        current_rows = count_jsonl_rows(output_judgement)
        now = time.time()
        if current_rows != last_reported or now - last_report_ts >= poll_sec:
            print(
                f"[branch] {model_tag} {branch_name} progress: "
                f"{format_progress(current_rows, expected_samples)} criteria"
            )
            last_reported = current_rows
            last_report_ts = now
        time.sleep(2.0)

    stdout, stderr = proc.communicate()
    final_rows = count_jsonl_rows(output_judgement)
    print(
        f"[branch] {model_tag} {branch_name} judged: "
        f"{format_progress(final_rows, expected_samples)} criteria"
    )
    return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)


def judgement_output_path(branch_out_dir: Path, input_path: Path) -> Path:
    return branch_out_dir / f"model_resp_{input_path.name}"


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A concurrent writer may leave the trailing line incomplete briefly.
                continue
            if "judgement" in row:
                judgement = row.get("judgement")
                if judgement is None:
                    continue
                if isinstance(judgement, str) and not judgement.strip():
                    continue
            elif "model_resp" in row:
                resp = row.get("model_resp")
                if resp is None:
                    continue
                if isinstance(resp, str) and not resp.strip():
                    continue
            count += 1
    return count


def select_source_dir(
    spec: AnswerEvalSpec,
    refreshed_root: Optional[Path],
    reused_root: Optional[Path],
    which: str,
) -> Path:
    if which == "response":
        use_reused = spec.reuse_response
        tag = spec.response_tag
    else:
        use_reused = spec.reuse_rubric
        tag = spec.rubric_tag
    root = reused_root if use_reused else refreshed_root
    if root is None:
        raise RuntimeError(f"Missing configured source root for {which} tag {tag}")
    return root / tag


def build_merged_rows(
    response_rows: List[Dict],
    ai_rubrics: Dict[str, Dict],
    human_rubrics: Dict[str, Dict],
) -> Tuple[List[Dict], List[Dict]]:
    merged_human = []
    merged_model = []
    for row in response_rows:
        idx = row["idx"]
        task_id = row.get("TASK_ID") or f"case_{idx:03d}"
        if task_id not in ai_rubrics or task_id not in human_rubrics:
            raise ValueError(f"Missing rubric row for {task_id}")

        human_row = dict(row)
        human_row["TASK_ID"] = task_id
        human_row["RUBRIC"] = normalize_rubric_field(human_rubrics[task_id]["RUBRIC"])
        merged_human.append(human_row)

        model_row = dict(row)
        model_row["TASK_ID"] = task_id
        model_row["RUBRIC"] = normalize_rubric_field(ai_rubrics[task_id]["RUBRIC"])
        merged_model.append(model_row)

    return merged_human, merged_model


def evaluate_branch(
    *,
    model_tag: str,
    python_bin: Path,
    branch_name: str,
    rows: List[Dict],
    input_path: Path,
    branch_out_dir: Path,
    judge_model: str,
    api_key: str,
    judge_temperature: float,
    judge_top_p: float,
    judge_max_tokens: int,
    judge_reasoning_effort: str,
    judge_parallel: int,
    dry_run: bool,
) -> Dict[str, object]:
    expected_input_rows = len(rows)
    expected_samples = sum(len(row["RUBRIC"]) for row in rows)
    output_judgement = judgement_output_path(branch_out_dir, input_path)

    judge_cmd = [
        str(python_bin),
        str(JUDGE_SCRIPT),
        "-i",
        str(input_path),
        "-ak",
        api_key,
        "-jt",
        "model_resp",
        "-jm",
        judge_model,
        "--temperature",
        str(judge_temperature),
        "--top_p",
        str(judge_top_p),
        "--max_tokens",
        str(judge_max_tokens),
        "--reasoning_effort",
        judge_reasoning_effort,
        "-n",
        str(judge_parallel),
        "-es",
        str(expected_input_rows),
        "-o",
        str(branch_out_dir),
    ]
    existing_rows = count_jsonl_rows(output_judgement)
    if existing_rows < expected_samples:
        runtime_env = build_runtime_env()
        judge_proc = run_judge_with_progress(
            cmd=judge_cmd,
            env=runtime_env,
            dry_run=dry_run,
            model_tag=model_tag,
            branch_name=branch_name,
            output_judgement=output_judgement,
            expected_samples=expected_samples,
            expected_input_rows=expected_input_rows,
        )
        if judge_proc.returncode != 0:
            raise RuntimeError(
                f"{branch_name} judge failed rc={judge_proc.returncode}\nSTDOUT:\n{judge_proc.stdout}\nSTDERR:\n{judge_proc.stderr}"
            )
    else:
        print(
            f"[branch] {model_tag} {branch_name} skip_judge: "
            f"{format_progress(existing_rows, expected_samples)} criteria already present"
        )

    print(f"[branch] {model_tag} {branch_name} scoring: expected {expected_samples} criteria")
    score_cmd = [
        str(python_bin),
        str(SCORE_SCRIPT),
        "-i",
        str(output_judgement),
        "-f",
        "human",
        "-es",
        str(expected_samples),
    ]
    score_proc = run(score_cmd, env=build_runtime_env(), dry_run=dry_run)
    if score_proc.returncode != 0:
        raise RuntimeError(
            f"{branch_name} scoring failed rc={score_proc.returncode}\nSTDOUT:\n{score_proc.stdout}\nSTDERR:\n{score_proc.stderr}"
        )
    return {
        "expected_samples": expected_samples,
        "judgement_rows": count_jsonl_rows(output_judgement),
        "judgement_file": path_for_summary(output_judgement),
        "scores": parse_score_stdout(score_proc.stdout),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrix",
        default="code/configs/answer_eval_matrix_refresh_openrouter_high.json",
    )
    ap.add_argument("--date", default="", help="Optional date folder. Leave empty to write directly into the configured release path.")
    ap.add_argument("--output_subdir", default="")
    ap.add_argument("--pilot_cases", type=int, default=5)
    ap.add_argument("--skip_pilot", action="store_true")
    ap.add_argument("--pilot_only", action="store_true")
    ap.add_argument("--models", default="", help="Comma-separated response tags to run.")
    ap.add_argument("--judge_model", default="openai/gpt-oss-120b")
    ap.add_argument("--judge_temperature", type=float, default=1.0)
    ap.add_argument("--judge_top_p", type=float, default=1.0)
    ap.add_argument("--judge_max_tokens", type=int, default=10500)
    ap.add_argument("--judge_reasoning_effort", choices=["low", "medium", "high"], default="medium")
    ap.add_argument("--judge_parallel", type=int, default=10)
    ap.add_argument("--api_key_env", default="LAB_OPENROUTER_KEY")
    ap.add_argument("--python_bin", default=".venv/bin/python")
    ap.add_argument(
        "--refresh_responses_root",
        default="",
        help="Root with refreshed non-Claude answers. Required for non-reused rows.",
    )
    ap.add_argument(
        "--refresh_rubrics_root",
        default="",
        help="Root with refreshed non-Claude rubrics. Required for non-reused rows.",
    )
    ap.add_argument(
        "--reused_responses_root",
        default="",
    )
    ap.add_argument(
        "--reused_rubrics_root",
        default="",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stop_on_error", action="store_true")
    args = ap.parse_args()

    if args.skip_pilot and args.pilot_only:
        raise RuntimeError("--skip_pilot and --pilot_only cannot be used together.")

    bootstrap_env()
    matrix_path = resolve_path(args.matrix)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

    specs = [m for m in parse_models(matrix["models"]) if m.enabled]
    full_response_tag_counts = Counter(spec.response_tag for spec in specs)
    if args.models.strip():
        selected = {x.strip() for x in args.models.split(",") if x.strip()}
        known = set()
        for spec in specs:
            known.update(selector_keys_for_spec(spec, full_response_tag_counts))
        unknown = sorted(selected - known)
        if unknown:
            raise RuntimeError(f"Unknown response tag(s): {unknown}. Known enabled tags: {sorted(known)}")
        specs = [
            spec
            for spec in specs
            if selector_keys_for_spec(spec, full_response_tag_counts) & selected
        ]
    if not specs:
        raise RuntimeError("No models selected to run.")

    python_bin = resolve_path(args.python_bin)
    if not python_bin.exists():
        raise FileNotFoundError(f"Python binary not found: {python_bin}")
    api_key = os.environ.get(args.api_key_env) or ("RELEASE_PLACEHOLDER_KEY" if args.dry_run else "")
    if not api_key:
        raise RuntimeError(f"Missing API key in env var {args.api_key_env}")

    outputs_root = resolve_path(str(matrix.get("outputs_root", "data/canonical_full")))
    output_subdir = args.output_subdir or str(matrix.get("output_subdir", "answer_eval"))
    matrix_refresh_responses_root = str(matrix.get("refresh_responses_root", "") or "").strip()
    matrix_refresh_rubrics_root = str(matrix.get("refresh_rubrics_root", "") or "").strip()
    refreshed_responses_root_raw = args.refresh_responses_root or matrix_refresh_responses_root
    refreshed_rubrics_root_raw = args.refresh_rubrics_root or matrix_refresh_rubrics_root
    matrix_reused_responses_root = str(matrix.get("reused_responses_root", "") or "").strip()
    matrix_reused_rubrics_root = str(matrix.get("reused_rubrics_root", "") or "").strip()
    reused_responses_root_raw = args.reused_responses_root or matrix_reused_responses_root
    reused_rubrics_root_raw = args.reused_rubrics_root or matrix_reused_rubrics_root

    refreshed_responses_root = resolve_path(refreshed_responses_root_raw) if refreshed_responses_root_raw else None
    refreshed_rubrics_root = resolve_path(refreshed_rubrics_root_raw) if refreshed_rubrics_root_raw else None
    reused_responses_root = resolve_path(reused_responses_root_raw) if reused_responses_root_raw else None
    reused_rubrics_root = resolve_path(reused_rubrics_root_raw) if reused_rubrics_root_raw else None
    out_base = build_output_base(outputs_root, args.date, output_subdir)
    out_base.mkdir(parents=True, exist_ok=True)

    summary: List[Tuple[str, str]] = []

    for spec in specs:
        run_slug = run_slug_for_spec(spec, full_response_tag_counts)
        display_label = display_label_for_spec(spec, full_response_tag_counts)
        print(f"\n=== [{run_slug}] {display_label} ===")
        try:
            if spec.reuse_response and reused_responses_root is None:
                raise RuntimeError("Missing reused_responses_root for reused response rows.")
            if not spec.reuse_response and refreshed_responses_root is None:
                raise RuntimeError("Missing --refresh_responses_root for refreshed non-Claude evaluation.")
            if spec.reuse_rubric and reused_rubrics_root is None:
                raise RuntimeError("Missing reused_rubrics_root for reused rubric rows.")
            if not spec.reuse_rubric and refreshed_rubrics_root is None:
                raise RuntimeError("Missing --refresh_rubrics_root for refreshed non-Claude evaluation.")

            responses_dir = select_source_dir(spec, refreshed_responses_root, reused_responses_root, "response")
            rubrics_dir = select_source_dir(spec, refreshed_rubrics_root, reused_rubrics_root, "rubric")
            response_file = find_single_jsonl(responses_dir)
            ai_rubric_file, human_rubric_file = find_rubric_files(rubrics_dir)

            all_response_rows = sorted(load_jsonl(response_file), key=lambda r: r["idx"])
            if not all_response_rows:
                raise RuntimeError(f"No response rows found in {response_file}")

            ai_rubrics = {row["TASK_ID"]: row for row in load_jsonl(ai_rubric_file)}
            human_rubrics = {row["TASK_ID"]: row for row in load_jsonl(human_rubric_file)}

            model_out = out_base / run_slug
            branches_summary = {}

            if not args.skip_pilot:
                pilot_rows = all_response_rows[: args.pilot_cases]
                validate_response_rows(pilot_rows, model_tag=run_slug, response_file=response_file)
                merged_human, merged_model = build_merged_rows(pilot_rows, ai_rubrics, human_rubrics)
                pilot_dir = model_out / "pilot"
                human_input = pilot_dir / "inputs" / f"{run_slug}_pilot_under_human_rubric.jsonl"
                model_input = pilot_dir / "inputs" / f"{run_slug}_pilot_under_model_rubric.jsonl"
                write_jsonl(human_input, merged_human)
                write_jsonl(model_input, merged_model)
                branches_summary["pilot_human"] = evaluate_branch(
                    model_tag=run_slug,
                    python_bin=python_bin,
                    branch_name="pilot_human",
                    rows=merged_human,
                    input_path=human_input,
                    branch_out_dir=pilot_dir / "judgements" / "human",
                    judge_model=args.judge_model,
                    api_key=api_key,
                    judge_temperature=args.judge_temperature,
                    judge_top_p=args.judge_top_p,
                    judge_max_tokens=args.judge_max_tokens,
                    judge_reasoning_effort=args.judge_reasoning_effort,
                    judge_parallel=args.judge_parallel,
                    dry_run=args.dry_run,
                )
                branches_summary["pilot_model"] = evaluate_branch(
                    model_tag=run_slug,
                    python_bin=python_bin,
                    branch_name="pilot_model",
                    rows=merged_model,
                    input_path=model_input,
                    branch_out_dir=pilot_dir / "judgements" / "model",
                    judge_model=args.judge_model,
                    api_key=api_key,
                    judge_temperature=args.judge_temperature,
                    judge_top_p=args.judge_top_p,
                    judge_max_tokens=args.judge_max_tokens,
                    judge_reasoning_effort=args.judge_reasoning_effort,
                    judge_parallel=args.judge_parallel,
                    dry_run=args.dry_run,
                )

                if args.pilot_only:
                    write_json(
                        model_out / "summary.json",
                        {
                            "label": spec.label,
                            "run_slug": run_slug,
                            "response_tag": spec.response_tag,
                            "rubric_tag": spec.rubric_tag,
                            "reuse_response": spec.reuse_response,
                            "reuse_rubric": spec.reuse_rubric,
                            "response_file": path_for_summary(response_file),
                            "ai_rubric_file": path_for_summary(ai_rubric_file),
                            "human_rubric_file": path_for_summary(human_rubric_file),
                            "judge_model": args.judge_model,
                            "judge_reasoning_effort": args.judge_reasoning_effort,
                            "branches": branches_summary,
                            "pilot_only": True,
                        },
                    )
                    summary.append((run_slug, "pilot_only_done"))
                    continue

            validate_response_rows(all_response_rows, model_tag=run_slug, response_file=response_file)
            merged_human_full, merged_model_full = build_merged_rows(all_response_rows, ai_rubrics, human_rubrics)
            full_dir = model_out / "full"
            human_input = full_dir / "inputs" / f"{run_slug}_full_under_human_rubric.jsonl"
            model_input = full_dir / "inputs" / f"{run_slug}_full_under_model_rubric.jsonl"
            write_jsonl(human_input, merged_human_full)
            write_jsonl(model_input, merged_model_full)
            branches_summary["full_human"] = evaluate_branch(
                model_tag=run_slug,
                python_bin=python_bin,
                branch_name="full_human",
                rows=merged_human_full,
                input_path=human_input,
                branch_out_dir=full_dir / "judgements" / "human",
                judge_model=args.judge_model,
                api_key=api_key,
                judge_temperature=args.judge_temperature,
                judge_top_p=args.judge_top_p,
                judge_max_tokens=args.judge_max_tokens,
                judge_reasoning_effort=args.judge_reasoning_effort,
                judge_parallel=args.judge_parallel,
                dry_run=args.dry_run,
            )
            branches_summary["full_model"] = evaluate_branch(
                model_tag=run_slug,
                python_bin=python_bin,
                branch_name="full_model",
                rows=merged_model_full,
                input_path=model_input,
                branch_out_dir=full_dir / "judgements" / "model",
                judge_model=args.judge_model,
                api_key=api_key,
                judge_temperature=args.judge_temperature,
                judge_top_p=args.judge_top_p,
                judge_max_tokens=args.judge_max_tokens,
                judge_reasoning_effort=args.judge_reasoning_effort,
                judge_parallel=args.judge_parallel,
                dry_run=args.dry_run,
            )

            write_json(
                model_out / "summary.json",
                {
                    "label": spec.label,
                    "run_slug": run_slug,
                    "response_tag": spec.response_tag,
                    "rubric_tag": spec.rubric_tag,
                    "reuse_response": spec.reuse_response,
                    "reuse_rubric": spec.reuse_rubric,
                    "response_file": path_for_summary(response_file),
                    "ai_rubric_file": path_for_summary(ai_rubric_file),
                    "human_rubric_file": path_for_summary(human_rubric_file),
                    "judge_model": args.judge_model,
                    "judge_reasoning_effort": args.judge_reasoning_effort,
                    "branches": branches_summary,
                },
            )
            summary.append((run_slug, "done"))
        except Exception as exc:  # noqa: BLE001
            if args.stop_on_error:
                raise
            print(f"ERROR [{run_slug}]: {exc}")
            summary.append((run_slug, "error"))

    print("\n=== Summary ===")
    for tag, status in summary:
        print(f"{tag:32s}  {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
