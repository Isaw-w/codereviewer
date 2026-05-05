#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

RELEASE_ROOT = Path(__file__).resolve().parents[2]
PAPER = RELEASE_ROOT / "data" / "paper_release"
CANON = RELEASE_ROOT / "data" / "canonical_full"
MANIFESTS = RELEASE_ROOT / "manifests"
VALIDATION_DIR = MANIFESTS / "validation"
EXPECTED_PATH = MANIFESTS / "expected_metrics.json"
REPORT_PATH = VALIDATION_DIR / "metrics_report.json"

PRIMARY_MODELS = [
    "claude_sonnet4",
    "opus46",
    "deepseek_r1_0528_openrouter",
    "deepseekv32exp_openrouter",
    "gemini25_pro_openrouter",
    "gemini31_openrouter",
    "gemini3_flash_openrouter",
    "gpt54_openrouter",
    "kimi_k2_5_openrouter",
    "mimo_v2_pro_openrouter",
    "qwen35_397b_a17b_openrouter",
]
AUX_MODELS = [
    "gemma3_4b_openrouter",
    "qwen35_9b_openrouter",
    "gpt_oss_120b_openrouter",
]
ALL_SCORED_MODELS = PRIMARY_MODELS + AUX_MODELS
COMPARABLE_SCORE_MODELS = PRIMARY_MODELS + ["gemma3_4b_openrouter", "qwen35_9b_openrouter"]
SELF_JUDGE_MODEL = "gpt_oss_120b_openrouter"
FINDING1_BINS = [
    (">=0.85", lambda c: c >= 0.85),
    ("0.80-0.85", lambda c: 0.80 <= c < 0.85),
    ("0.70-0.80", lambda c: 0.70 <= c < 0.80),
    ("0.60-0.70", lambda c: 0.60 <= c < 0.70),
    ("<0.60", lambda c: c < 0.60),
]


def round_float(x: float, digits: int = 4) -> float:
    return round(float(x), digits)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_single_data_jsonl(directory: Path) -> Path:
    files = sorted(
        p for p in directory.glob("*.jsonl")
        if ".errors." not in p.name and ".bak" not in p.name
    )
    if len(files) != 1:
        raise RuntimeError(f"Expected one data jsonl in {directory}, found {len(files)}")
    return files[0]


def score_from_judgements(path: Path) -> float:
    rows = load_jsonl(path)
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    task_scores = []
    for task_id, criteria in by_task.items():
        max_score = 0
        achieved = 0
        for crit in criteria:
            weight = crit["criterion_weight"]
            judgement = str(crit["judgement"]).strip().lower()
            max_score += abs(weight)
            if "yes" in judgement and weight > 0:
                achieved += weight
            elif "no" in judgement and weight < 0:
                achieved -= weight
        task_scores.append(100.0 * achieved / max_score if max_score else 0.0)
    return round_float(sum(task_scores) / len(task_scores), 2)


def compare(a: Any, b: Any, path: str = "") -> list[str]:
    diffs = []
    if type(a) != type(b):
        return [f"{path}: type mismatch {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, dict):
        ak = set(a)
        bk = set(b)
        for key in sorted(ak - bk):
            diffs.append(f"{path}.{key}: missing from expected")
        for key in sorted(bk - ak):
            diffs.append(f"{path}.{key}: missing from actual")
        for key in sorted(ak & bk):
            diffs.extend(compare(a[key], b[key], f"{path}.{key}" if path else str(key)))
        return diffs
    if isinstance(a, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length mismatch {len(a)} != {len(b)}")
            return diffs
        for idx, (av, bv) in enumerate(zip(a, b)):
            diffs.extend(compare(av, bv, f"{path}[{idx}]"))
        return diffs
    if isinstance(a, float):
        if abs(a - b) > 1e-6:
            diffs.append(f"{path}: {a} != {b}")
        return diffs
    if a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def compute_finding1() -> dict:
    obj = json.loads((PAPER / "finding1" / "criterion_pairs" / "finding1_confirmed_pairs.json").read_text())
    pairs = obj["pairs"]
    human_yes = sum(1 for p in pairs if p["hj"] == "YES")
    model_yes = sum(1 for p in pairs if p["mj"] == "YES")
    bins = []
    for label, pred in FINDING1_BINS:
        rows = [p for p in pairs if pred(float(p["cos"]))]
        h_yes = sum(1 for p in rows if p["hj"] == "YES")
        m_yes = sum(1 for p in rows if p["mj"] == "YES")
        bins.append({
            "bin": label,
            "n": len(rows),
            "human_yes_pct": round_float(100.0 * h_yes / len(rows), 1),
            "model_yes_pct": round_float(100.0 * m_yes / len(rows), 1),
            "gap_pp": round_float(100.0 * (m_yes - h_yes) / len(rows), 1),
            "h_no_m_yes": sum(1 for p in rows if p["hj"] == "NO" and p["mj"] == "YES"),
            "h_yes_m_no": sum(1 for p in rows if p["hj"] == "YES" and p["mj"] == "NO"),
        })
    return {
        "models": len(obj["models"]),
        "dilemmas": len(obj["dilemmas"]),
        "pairs": len(pairs),
        "human_yes_pct": round_float(100.0 * human_yes / len(pairs), 2),
        "model_yes_pct": round_float(100.0 * model_yes / len(pairs), 2),
        "gap_pp": round_float(100.0 * (model_yes - human_yes) / len(pairs), 2),
        "bins": bins,
    }


def load_weight_map() -> dict[tuple[str, str], int]:
    mapping = {}
    for row in load_jsonl(PAPER / "rubrics" / "original" / "human_rubric_500cases.jsonl"):
        rubric = row["RUBRIC"]
        if isinstance(rubric, str):
            rubric = json.loads(rubric)
        for crit in rubric:
            mapping[(row["TASK_ID"], crit["id"])] = int(crit["weight"])
    return mapping


def compute_finding2() -> dict:
    audit_rows = load_jsonl(PAPER / "rubrics" / "rewrite" / "cascade_rewrite_audit.jsonl")
    weight_map = load_weight_map()
    changed = [r for r in audit_rows if r["changed"]]
    unchanged = [r for r in audit_rows if not r["changed"]]
    pos_changed = sum(1 for r in changed if weight_map[(r["task_id"], r["criterion_id"])] > 0)
    neg_changed = sum(1 for r in changed if weight_map[(r["task_id"], r["criterion_id"])] < 0)

    model_scores = {}
    for tag in ALL_SCORED_MODELS:
        base = CANON / "answer_eval" / tag / "full" / "judgements"
        human_file = find_single_data_jsonl(base / "human")
        cascade_file = find_single_data_jsonl(base / "cascade")
        model_scores[tag] = {
            "original_human_score": score_from_judgements(human_file),
            "cascade_rewritten_score": score_from_judgements(cascade_file),
        }
        model_scores[tag]["uplift"] = round_float(
            model_scores[tag]["cascade_rewritten_score"] - model_scores[tag]["original_human_score"], 2
        )

    orig_scores = [model_scores[tag]["original_human_score"] for tag in COMPARABLE_SCORE_MODELS]
    casc_scores = [model_scores[tag]["cascade_rewritten_score"] for tag in COMPARABLE_SCORE_MODELS]
    return {
        "criteria_total": len(audit_rows),
        "changed": len(changed),
        "unchanged": len(unchanged),
        "changed_positive_weight": pos_changed,
        "changed_negative_weight": neg_changed,
        "mean_original_human_score_13": round_float(statistics.mean(orig_scores), 2),
        "mean_cascade_rewritten_score_13": round_float(statistics.mean(casc_scores), 2),
        "mean_uplift_13": round_float(statistics.mean(c - o for o, c in zip(orig_scores, casc_scores)), 2),
        "scored_models": model_scores,
    }


def compute_pairwise_primary_means(summary: dict) -> dict:
    raw = summary["primary_by_pair_side"]
    human = defaultdict(list)
    model = defaultdict(list)
    for pair_side, counts in raw.items():
        total = sum(counts.values())
        if not total:
            continue
        target = human if pair_side.endswith("::human_only") else model
        for label, count in counts.items():
            target[label].append(100.0 * count / total)
    labels = sorted(set(human) | set(model))
    return {
        label: {
            "human_mean": round_float(statistics.mean(human[label]) if human[label] else 0.0, 1),
            "model_mean": round_float(statistics.mean(model[label]) if model[label] else 0.0, 1),
        }
        for label in labels
    }


def compute_finding3() -> dict:
    global_summary = json.loads((PAPER / "finding3" / "coverage" / "global_unique_t70" / "summary.json").read_text())
    direct_check = json.loads((PAPER / "finding3" / "direct_check" / "top100_v2_high_pooled_summary.json").read_text())
    samebranch_input = json.loads((PAPER / "finding3" / "normative_tendencies" / "finding3_samebranch_intersection_input_summary.json").read_text())
    samebranch_labels = json.loads((PAPER / "finding3" / "normative_tendencies" / "finding3_samebranch_intersection_label_summary.complete_all.json").read_text())
    pairwise_means = compute_pairwise_primary_means(samebranch_labels)
    selected = [
        "consequences, harm, or benefit",
        "practical wisdom or framing",
        "epistemic humility",
        "duties, rights, or autonomy",
        "role obligations or boundaries",
    ]
    return {
        "global_unique": {
            "human_only": int(global_summary["human_only"]),
            "model_only": int(global_summary["model_only"]),
            "human_covered_by_model": int(global_summary["human_covered_by_model"]),
            "model_overlaps_human": int(global_summary["model_overlaps_human"]),
        },
        "direct_check": {
            "selected_cases": int(direct_check["n_selected_cases"]),
            "selected_concepts": int(direct_check["n_selected_concepts"]),
            "coverage_pct": round_float(direct_check["coverage_pct_of_pooled_human_only_concepts"], 1),
            "pooled_human_only_total": int(direct_check["pooled_human_only_concepts_total"]),
            "confirmed_human_only": int(direct_check["pooled_human_confirmation"]["confirmed_human_only"]),
            "concept_model_evaluations": int(direct_check["pooled_human_confirmation"]["concept_model_evaluations"]),
            "confirmation_rate_pct": round_float(direct_check["pooled_human_confirmation"]["confirmation_rate_pct"], 1),
            "human_only_instances": int(direct_check["subset_raw_instance_asymmetry"]["human_only_instances"]),
            "model_only_instances": int(direct_check["subset_raw_instance_asymmetry"]["model_only_instances"]),
        },
        "samebranch_intersection": {
            "rows": int(samebranch_input["total_rows"]),
            "human_only_rows": int(samebranch_input["human_only_rows"]),
            "model_only_rows": int(samebranch_input["model_only_rows"]),
            "pair_count": len(samebranch_labels["primary_by_pair_side"]) // 2,
        },
        "pairwise_primary_means": {label: pairwise_means[label] for label in selected},
    }


def build_metrics() -> dict:
    return {
        "finding1": compute_finding1(),
        "finding2": compute_finding2(),
        "finding3": compute_finding3(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", choices=["all", "finding1", "finding2", "finding3"], default="all")
    parser.add_argument("--write-expected", action="store_true")
    args = parser.parse_args()

    metrics = build_metrics()
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_expected or not EXPECTED_PATH.exists():
        EXPECTED_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    expected = json.loads(EXPECTED_PATH.read_text())
    actual = metrics if args.section == "all" else {args.section: metrics[args.section]}
    baseline = expected if args.section == "all" else {args.section: expected[args.section]}
    diffs = compare(actual, baseline)
    report = {
        "section": args.section,
        "status": "ok" if not diffs else "failed",
        "differences": diffs,
        "actual": actual,
        "expected_path": str(EXPECTED_PATH.relative_to(RELEASE_ROOT)),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"metrics validation [{args.section}]: {'ok' if not diffs else 'failed'}")
    print(f"report -> {REPORT_PATH}")
    if diffs:
        for diff in diffs[:20]:
            print(diff)
        sys.exit(1)

if __name__ == "__main__":
    main()
