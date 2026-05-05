#!/usr/bin/env python3
"""
Incremental answer-generation runner for the OpenRouter high-refresh line.

What it does:
1) Reads a response model matrix JSON.
2) Enforces pilot-first (5 cases by default) using the same output file the full run resumes from.
3) Runs full generation with resume behavior through `run_inferences_on_dilemmas.py`.
4) Continues on model failures by default so one model does not stop the whole refresh line.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from utils import get_model_filename


REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(__file__).resolve().parent
RESP_SCRIPT = CODE_ROOT / "run_inferences_on_dilemmas.py"
DEFAULT_RELEASE_INPUT = RELEASE_ROOT / "data" / "paper_release" / "inputs" / "morebench_test.jsonl"


@dataclass
class ModelSpec:
    tag: str
    api_provider: str
    model: str
    api_key_env: str
    enabled: bool = True
    reasoning_effort: str = "high"


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
                    reasoning_effort=str(m.get("reasoning_effort", "high")),
                )
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing required key in response matrix entry #{i}: {exc}") from exc
    return out


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                resp = row.get("model_resp", "__MISSING__")
                if resp == "__MISSING__":
                    count += 1
                elif resp is not None and (not isinstance(resp, str) or resp.strip()):
                    count += 1
    return count


def output_file_for(model: ModelSpec, generations_dir: Path, seed: int) -> Path:
    model_for_filename = get_model_filename(model.model, model.api_provider)
    return generations_dir / f"{model_for_filename}_reasoning_{model.reasoning_effort}_seed_{seed}.jsonl"


def load_source_rows(source_input: Optional[Path], hf_token: Optional[str]) -> List[Dict[str, object]]:
    if source_input is not None and source_input.exists():
        if source_input.suffix.lower() == ".jsonl":
            rows: List[Dict[str, object]] = []
            with source_input.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows
        if source_input.suffix.lower() == ".csv":
            with source_input.open("r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
        raise RuntimeError(f"Unsupported local input format for response generation: {source_input}")

    if not hf_token:
        raise RuntimeError("Missing HuggingFace token and no local input source is available.")

    from datasets import load_dataset

    ds = load_dataset("morebench/morebench", token=hf_token, data_files="morebench_public.csv", split="train")
    return ds.to_pandas().to_dict(orient="records")


def normalize_source_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    normalized: List[Dict[str, object]] = []
    for idx, row in enumerate(rows):
        theory = str(row.get("THEORY", "")).strip()
        if theory and theory != "neutral":
            continue
        normalized_row = dict(row)
        normalized_row.setdefault("ORIG_IDX", idx)
        rubric = normalized_row.get("RUBRIC")
        if isinstance(rubric, (list, dict)):
            normalized_row["RUBRIC"] = repr(rubric)
        normalized.append(normalized_row)
    return normalized


def write_rows_to_csv(path: Path, rows: List[Dict[str, object]]) -> Path:
    if not rows:
        raise RuntimeError(f"Cannot write empty input CSV to {path}")
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def build_prepared_input_csv(
    out_base: Path,
    source_input: Optional[Path],
    hf_token: Optional[str],
    *,
    limit_cases: Optional[int] = None,
) -> Path:
    rows = normalize_source_rows(load_source_rows(source_input, hf_token))
    if limit_cases is not None:
        rows = rows[:limit_cases]
    if not rows:
        raise RuntimeError("No neutral rows available to build response-generation input.")
    pilot_input_dir = out_base / "_pilot_inputs"
    pilot_input_dir.mkdir(parents=True, exist_ok=True)
    if limit_cases is None:
        out_path = pilot_input_dir / "neutral_full_with_orig_idx.csv"
    else:
        out_path = pilot_input_dir / f"neutral_first_{limit_cases}_with_orig_idx.csv"
    return write_rows_to_csv(out_path, rows)


def redact_cmd(cmd: List[str]) -> List[str]:
    redacted = list(cmd)
    for i, token in enumerate(redacted[:-1]):
        if token in {"-ak", "--api_key", "-ht", "--hf_token"}:
            redacted[i + 1] = "***REDACTED***"
    return redacted


def run_generator(
    python_bin: Path,
    generations_dir: Path,
    hf_token: str,
    model: ModelSpec,
    workers: int,
    seed: int,
    debug: bool,
    input_file: Optional[Path],
    dry_run: bool,
) -> None:
    api_key = os.environ.get(model.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env key `{model.api_key_env}` for model `{model.tag}`")

    cmd = [
        str(python_bin),
        str(RESP_SCRIPT),
        "-ap",
        model.api_provider,
        "-ak",
        api_key,
        "-m",
        model.model,
        "-n",
        str(workers),
        "-g",
        str(generations_dir),
        "-s",
        str(seed),
        "-ht",
        hf_token,
        "-r",
        "-re",
        model.reasoning_effort,
    ]
    if input_file is not None:
        cmd.extend(["-i", str(input_file)])
    if debug:
        cmd.append("-d")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{CODE_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(CODE_ROOT)

    print("CMD:", " ".join(redact_cmd(cmd)))
    if dry_run:
        return
    subprocess.run(cmd, cwd=RELEASE_ROOT, env=env, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--matrix",
        default="code/configs/response_matrix_refresh_openrouter_high.json",
        help="Response matrix JSON path.",
    )
    ap.add_argument("--date", default="", help="Optional date folder. Leave empty to write directly into the configured release path.")
    ap.add_argument("--workers", type=int, default=1, help="Per-model request concurrency.")
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
    models = [m for m in parse_models(matrix["models"]) if m.enabled]
    if args.models.strip():
        selected = {x.strip() for x in args.models.split(",") if x.strip()}
        known = {m.tag for m in models}
        unknown = sorted(selected - known)
        if unknown:
            raise RuntimeError(f"Unknown model tag(s): {unknown}. Known enabled tags: {sorted(known)}")
        models = [m for m in models if m.tag in selected]
    if not models:
        raise RuntimeError("No models selected to run.")

    outputs_root = resolve_path(str(matrix.get("outputs_root", "data/canonical_full")))
    outputs_subdir = str(matrix.get("outputs_subdir", "responses"))
    target_cases = int(matrix.get("target_cases", 500))
    seed = int(matrix.get("seed", 0))
    out_base = build_output_base(outputs_root, args.date, outputs_subdir)
    out_base.mkdir(parents=True, exist_ok=True)

    python_bin = resolve_path(args.python_bin)
    if not python_bin.exists():
        raise FileNotFoundError(f"Python binary not found: {python_bin}")
    if not RESP_SCRIPT.exists():
        raise FileNotFoundError(f"Response generation script not found: {RESP_SCRIPT}")

    hf_token = os.environ.get("HuggingFace") or os.environ.get("HF_TOKEN")
    source_input_raw = str(matrix.get("input_file", "") or "").strip()
    if source_input_raw:
        source_input = resolve_path(source_input_raw)
    elif DEFAULT_RELEASE_INPUT.exists():
        source_input = DEFAULT_RELEASE_INPUT
    else:
        source_input = None
    if not hf_token and not (source_input and source_input.exists()):
        raise RuntimeError("Missing HuggingFace token and no local input_file was provided in the matrix.")

    command_hf_token = hf_token or "RELEASE_PLACEHOLDER_TOKEN"

    pilot_input_path: Optional[Path] = None
    full_input_path: Optional[Path] = None
    if not args.skip_pilot:
        pilot_input_path = build_prepared_input_csv(out_base, source_input, hf_token, limit_cases=args.pilot_cases)
    if source_input is not None or hf_token:
        full_input_path = build_prepared_input_csv(out_base, source_input, hf_token, limit_cases=None)

    print(f"Loaded {len(models)} enabled models from: {matrix_path}")
    print(f"Output root: {out_base}")
    print(f"Target cases/model: {target_cases}")
    if pilot_input_path is not None:
        print(f"Exact pilot input: {pilot_input_path}")
    if full_input_path is not None:
        print(f"Prepared full input: {full_input_path}")

    summary: List[Tuple[str, str, int, int]] = []

    for model in models:
        print(f"\n=== [{model.tag}] {model.model} ({model.api_provider}) ===")
        model_dir = out_base / model.tag
        model_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_file_for(model, model_dir, seed)
        existing_rows = count_rows(output_file)

        try:
            if existing_rows >= target_cases:
                print(f"Already complete: {existing_rows}/{target_cases}. Skip full run.")
                summary.append((model.tag, "done_already", existing_rows, target_cases))
                continue

            if not args.skip_pilot and existing_rows < args.pilot_cases:
                print(f"Running pilot {args.pilot_cases} cases...")
                run_generator(
                    python_bin=python_bin,
                    generations_dir=model_dir,
                    hf_token=command_hf_token,
                    model=model,
                    workers=args.workers,
                    seed=seed,
                    debug=False,
                    input_file=pilot_input_path,
                    dry_run=args.dry_run,
                )
                pilot_rows = count_rows(output_file)
                if pilot_rows < args.pilot_cases and not args.dry_run:
                    raise RuntimeError(
                        f"Pilot failed for {model.tag}: expected at least {args.pilot_cases}, got {pilot_rows}"
                    )
                existing_rows = pilot_rows
            else:
                print(f"Pilot already satisfied: {existing_rows}/{args.pilot_cases}")

            if args.pilot_only:
                pilot_rows = count_rows(output_file)
                summary.append((model.tag, "pilot_only_done", pilot_rows, args.pilot_cases))
                continue

            print(f"Running full increment (resume): existing {existing_rows}/{target_cases}")
            run_generator(
                python_bin=python_bin,
                generations_dir=model_dir,
                hf_token=command_hf_token,
                model=model,
                workers=args.workers,
                seed=seed,
                debug=False,
                input_file=full_input_path,
                dry_run=args.dry_run,
            )
            done_rows = count_rows(output_file)
            status = "done" if done_rows >= target_cases else "partial"
            print(f"Done count: {done_rows}/{target_cases}")
            summary.append((model.tag, status, done_rows, target_cases))
        except Exception as exc:  # noqa: BLE001
            if args.stop_on_error:
                raise
            done_rows = count_rows(output_file)
            print(f"ERROR [{model.tag}]: {exc}")
            summary.append((model.tag, "error", done_rows, target_cases))

    print("\n=== Summary ===")
    for tag, status, done, target in summary:
        print(f"{tag:32s}  {status:18s}  {done:4d}/{target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
