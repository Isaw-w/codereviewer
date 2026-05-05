#!/usr/bin/env python3
"""
Incremental rubric-generation runner for experiment A1.

What it does:
1) Reads a model matrix JSON.
2) Reuses existing outputs by seeding model files (if provided).
3) Enforces pilot-first (5-case by default) unless skipped.
4) Runs full generation with resume behavior (skip existing TASK_ID rows).
5) Continues on model failures by default, so one bad model ID does not stop all runs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent
GEN_SCRIPT = CODE_ROOT / "generate_ai_rubrics_exp2.py"


@dataclass
class ModelSpec:
    tag: str
    api_provider: str
    model: str
    api_key_env: str
    enabled: bool = True
    seed_ai: Optional[str] = None
    seed_human: Optional[str] = None
    reasoning_effort: Optional[str] = None
    max_tokens: int = 8000


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


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


def read_task_ids(path: Path) -> Tuple[int, Set[str]]:
    if not path.exists():
        return 0, set()
    total = 0
    ids: Set[str] = set()

    def normalize_tid(raw: object) -> Optional[str]:
        if raw is None:
            return None
        s = str(raw).strip()
        if not s or s.lower() in {"none", "null", "nan"}:
            return None
        return s

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                row = json.loads(s)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"JSON parse error in {path} line {i}: {exc}") from exc
            tid = normalize_tid(row.get("TASK_ID", row.get("idx")))
            if tid is not None:
                ids.add(tid)
    return total, ids


def read_paired_task_ids(ai_path: Path, human_path: Path) -> Dict[str, object]:
    ai_total, ai_ids = read_task_ids(ai_path)
    human_total, human_ids = read_task_ids(human_path)
    return {
        "ai_total": ai_total,
        "human_total": human_total,
        "ai_ids": ai_ids,
        "human_ids": human_ids,
        "shared_ids": ai_ids & human_ids,
    }


def seed_if_needed(src_raw: Optional[str], dst: Path, dry_run: bool = False) -> bool:
    if not src_raw or dst.exists():
        return False
    src = resolve_path(src_raw)
    if not src.exists():
        raise FileNotFoundError(f"Seed file not found: {src}")
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def compute_parallelism(shared_workers: int, job_count: int) -> Tuple[int, int]:
    parallel_jobs = min(job_count, max(1, shared_workers))
    per_job_workers = max(1, shared_workers // parallel_jobs)
    return parallel_jobs, per_job_workers


def run_generator(
    python_bin: Path,
    input_file: Path,
    output_ai: Path,
    output_human: Path,
    api_provider: str,
    model: str,
    api_key_env: str,
    workers: int,
    test_n: int = 0,
    reasoning_effort: Optional[str] = None,
    max_tokens: int = 8000,
    dry_run: bool = False,
) -> None:
    cmd = [
        str(python_bin),
        str(GEN_SCRIPT),
        "-i",
        str(input_file),
        "-o",
        str(output_ai),
        "-ho",
        str(output_human),
        "-ap",
        api_provider,
        "-m",
        model,
        "-k",
        api_key_env,
        "-w",
        str(workers),
        "--max_tokens",
        str(max_tokens),
    ]
    if test_n > 0:
        cmd += ["-t", str(test_n)]
    if reasoning_effort:
        cmd += ["--reasoning_effort", reasoning_effort]

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CODE_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(CODE_ROOT)

    print("CMD:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=RELEASE_ROOT, env=env, check=True)


def run_one_model(
    *,
    model: ModelSpec,
    input_file: Path,
    target_cases: int,
    out_base: Path,
    python_bin: Path,
    per_job_workers: int,
    pilot_cases: int,
    skip_pilot: bool,
    pilot_only: bool,
    dry_run: bool,
) -> Tuple[str, str, int, int]:
    print(f"\n=== [{model.tag}] {model.model} ({model.api_provider}) ===")
    if not os.environ.get(model.api_key_env):
        print("WARN:", f"Missing env key `{model.api_key_env}` for model `{model.tag}`. Skip.")
        return (model.tag, "skipped_missing_key", 0, 0)

    model_dir = out_base / model.tag
    model_dir.mkdir(parents=True, exist_ok=True)
    ai_out = model_dir / f"ai_rubric_{target_cases}cases_{model.tag}_seed0.jsonl"
    human_out = model_dir / f"human_rubric_{target_cases}cases_{model.tag}_seed0.jsonl"
    pilot_ai = ai_out
    pilot_human = human_out

    seeded_ai = seed_if_needed(model.seed_ai, ai_out, dry_run=dry_run)
    seeded_human = seed_if_needed(model.seed_human, human_out, dry_run=dry_run)
    if seeded_ai or seeded_human:
        print(f"Seeded existing files: ai={seeded_ai}, human={seeded_human}")

    existing_state = read_paired_task_ids(ai_out, human_out)
    existing_ids = existing_state["shared_ids"]
    if existing_state["ai_ids"] != existing_state["human_ids"]:
        print(
            "WARN:",
            f"{model.tag} has mismatched ai/human task coverage; "
            f"ai={len(existing_state['ai_ids'])}, human={len(existing_state['human_ids'])}, "
            f"shared={len(existing_ids)}. Will continue to repair.",
        )
    if len(existing_ids) >= target_cases:
        print(f"Already complete: {len(existing_ids)}/{target_cases}. Skip full run.")
        return (model.tag, "done_already", len(existing_ids), target_cases)

    if not skip_pilot:
        pilot_state = read_paired_task_ids(pilot_ai, pilot_human)
        pilot_ids = pilot_state["shared_ids"]
        if len(pilot_ids) < pilot_cases:
            print(f"Running pilot {pilot_cases} cases...")
            run_generator(
                python_bin=python_bin,
                input_file=input_file,
                output_ai=pilot_ai,
                output_human=pilot_human,
                api_provider=model.api_provider,
                model=model.model,
                api_key_env=model.api_key_env,
                workers=per_job_workers,
                test_n=pilot_cases,
                reasoning_effort=model.reasoning_effort,
                max_tokens=model.max_tokens,
                dry_run=dry_run,
            )
            pilot_state_after = read_paired_task_ids(pilot_ai, pilot_human)
            pilot_ids_after = pilot_state_after["shared_ids"]
            if len(pilot_ids_after) < pilot_cases and not dry_run:
                raise RuntimeError(
                    f"Pilot failed for {model.tag}: expected {pilot_cases}, got {len(pilot_ids_after)}"
                )
        else:
            print(f"Pilot already exists: {len(pilot_ids)}/{pilot_cases}")

    if pilot_only:
        done_pilot_ids = read_paired_task_ids(pilot_ai, pilot_human)["shared_ids"]
        return (model.tag, "pilot_only_done", len(done_pilot_ids), pilot_cases)

    print(f"Running full increment (resume): existing {len(existing_ids)}/{target_cases}")
    run_generator(
        python_bin=python_bin,
        input_file=input_file,
        output_ai=ai_out,
        output_human=human_out,
        api_provider=model.api_provider,
        model=model.model,
        api_key_env=model.api_key_env,
        workers=per_job_workers,
        test_n=0,
        reasoning_effort=model.reasoning_effort,
        max_tokens=model.max_tokens,
        dry_run=dry_run,
    )

    done_ids = read_paired_task_ids(ai_out, human_out)["shared_ids"]
    print(f"Done count: {len(done_ids)}/{target_cases}")
    status = "done" if len(done_ids) >= target_cases else "partial"
    return (model.tag, status, len(done_ids), target_cases)


def parse_models(raw_models: Iterable[Dict[str, object]]) -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for i, m in enumerate(raw_models):
        try:
            out.append(
                ModelSpec(
                    tag=str(m["tag"]),
                    api_provider=str(m["api_provider"]),
                    model=str(m["model"]),
                    api_key_env=str(m["api_key_env"]),
                    enabled=bool(m.get("enabled", True)),
                    seed_ai=str(m["seed_ai"]) if m.get("seed_ai") else None,
                    seed_human=str(m["seed_human"]) if m.get("seed_human") else None,
                    reasoning_effort=str(m["reasoning_effort"]) if m.get("reasoning_effort") else None,
                    max_tokens=int(m.get("max_tokens", 8000)),
                )
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing required key in matrix model entry #{i}: {exc}") from exc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrix",
        default="code/configs/model_matrix_refresh_openrouter_high.json",
        help="Model matrix JSON path.",
    )
    ap.add_argument("--input_file", default="", help="Override matrix input_file.")
    ap.add_argument("--date", default="", help="Optional date folder. Leave empty to write directly into the configured release path.")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--pilot_cases", type=int, default=5)
    ap.add_argument("--skip_pilot", action="store_true")
    ap.add_argument("--pilot_only", action="store_true")
    ap.add_argument("--models", default="", help="Comma-separated model tags to run.")
    ap.add_argument("--python_bin", default=".venv/bin/python")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--stop_on_error", action="store_true")
    args = ap.parse_args()

    if args.skip_pilot and args.pilot_only:
        raise RuntimeError("--skip_pilot and --pilot_only cannot be used together.")

    bootstrap_env()

    matrix_path = resolve_path(args.matrix)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

    input_file_raw = args.input_file or matrix.get("input_file", "outputs/datasets/morebench_test.jsonl")
    input_file = resolve_path(str(input_file_raw))
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    target_cases = int(matrix.get("target_cases", 500))
    outputs_root = resolve_path(str(matrix.get("outputs_root", "data/canonical_full")))
    outputs_subdir = str(matrix.get("outputs_subdir", "rubrics"))
    out_base = build_output_base(outputs_root, args.date, outputs_subdir)
    out_base.mkdir(parents=True, exist_ok=True)

    python_bin = resolve_path(args.python_bin)
    if not python_bin.exists():
        raise FileNotFoundError(f"Python binary not found: {python_bin}")
    if not GEN_SCRIPT.exists():
        raise FileNotFoundError(f"Generator script not found: {GEN_SCRIPT}")

    models = [m for m in parse_models(matrix["models"]) if m.enabled]
    if args.models.strip():
        selected = {x.strip() for x in args.models.split(",") if x.strip()}
        known = {m.tag for m in models}
        unknown = sorted(selected - known)
        if unknown:
            raise RuntimeError(
                f"Unknown model tag(s): {unknown}. "
                f"Known enabled tags: {sorted(known)}"
            )
        models = [m for m in models if m.tag in selected]
    if not models:
        raise RuntimeError("No models selected to run.")

    print(f"Loaded {len(models)} enabled models from: {matrix_path}")
    print(f"Input: {input_file}")
    print(f"Output root: {out_base}")
    print(f"Target cases/model: {target_cases}")

    parallel_jobs, per_job_workers = compute_parallelism(args.workers, len(models))
    print(
        f"Cross-model parallelism: {parallel_jobs} job(s); shared workers: {args.workers}; "
        f"per-model workers: {per_job_workers}"
    )

    results_by_tag: Dict[str, Tuple[str, str, int, int]] = {}
    with ThreadPoolExecutor(max_workers=parallel_jobs) as executor:
        futures = {
            executor.submit(
                run_one_model,
                model=m,
                input_file=input_file,
                target_cases=target_cases,
                out_base=out_base,
                python_bin=python_bin,
                per_job_workers=per_job_workers,
                pilot_cases=args.pilot_cases,
                skip_pilot=args.skip_pilot,
                pilot_only=args.pilot_only,
                dry_run=args.dry_run,
            ): m
            for m in models
        }
        for future in as_completed(futures):
            model = futures[future]
            try:
                results_by_tag[model.tag] = future.result()
            except Exception as exc:  # noqa: BLE001
                if args.stop_on_error:
                    raise
                print(f"ERROR [{model.tag}]: {exc}")
                ai_out = out_base / model.tag / f"ai_rubric_{target_cases}cases_{model.tag}_seed0.jsonl"
                _, done_ids = read_task_ids(ai_out)
                results_by_tag[model.tag] = (model.tag, "error", len(done_ids), target_cases)

    summary = [results_by_tag[m.tag] for m in models if m.tag in results_by_tag]

    print("\n=== Summary ===")
    for tag, status, done, target in summary:
        print(f"{tag:28s}  {status:18s}  {done:4d}/{target}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
