#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
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
FINDING2_PRIMARY_LABELS = [
    "consequences, harm, or benefit",
    "practical wisdom or framing",
    "epistemic humility",
    "duties, rights, or autonomy",
    "role obligations or boundaries",
]
FINDING2_LEVEL2_LABELS = [
    "career, economic, or reputation effects",
    "institutional, social, or public effects",
    "relationship or trust effects",
    "general duty or right",
    "connect reasoning to conclusion",
    "balanced dilemma framing",
    "distinguish fact from assumption",
    "actionable steps",
    "deception or manipulation",
    "respect for dignity",
    "humiliation or shame",
]
FINDING2_MODEL_ORDER = [
    "GPT-5.4",
    "Claude Opus 4.6",
    "Claude Sonnet 4",
    "DeepSeek V3.2 Exp",
    "Kimi K2.5",
    "Qwen 3.5 397B",
    "MiMo V2 Pro",
    "Gemini 3 Flash",
    "DeepSeek R1",
    "Gemini 3.1 Pro",
    "Gemini 2.5 Pro",
]
FINDING3_PAIR_MODEL_NAMES = {
    "claude_sonnet4_openrouter": "Claude Sonnet 4",
    "opus46_openrouter": "Claude Opus 4.6",
    "qwen35_397b_a17b_openrouter": "Qwen 3.5 397B",
    "gpt54_openrouter": "GPT-5.4",
    "deepseek_r1_0528_openrouter": "DeepSeek R1",
    "gemini25_pro_openrouter": "Gemini 2.5 Pro",
    "mimo_v2_pro_openrouter": "MiMo V2 Pro",
    "kimi_k2_5_openrouter": "Kimi K2.5",
    "deepseekv32exp_openrouter": "DeepSeek V3.2 Exp",
    "gemini31_openrouter": "Gemini 3.1 Pro",
    "gemini3_flash_openrouter": "Gemini 3 Flash",
}
FINDING3_GENERALITY_SOURCES = [
    ("claude_sonnet4", "Claude Sonnet 4"),
    ("claude_opus46", "Claude Opus 4.6"),
    ("deepseek_r1", "DeepSeek R1"),
    ("deepseek_v32_exp", "DeepSeek V3.2 Exp"),
    ("gemini25_pro", "Gemini 2.5 Pro"),
    ("gemini31_pro", "Gemini 3.1 Pro"),
    ("gemini3_flash", "Gemini 3 Flash"),
    ("gemma3_4b", "Gemma 3 4B"),
    ("gpt54", "GPT-5.4"),
    ("gpt_oss_120b", "GPT-OSS-120B"),
    ("kimi_k25", "Kimi K2.5"),
    ("mimo_v2_pro", "MiMo V2 Pro"),
    ("qwen35_397b", "Qwen 3.5 397B"),
    ("qwen35_9b", "Qwen 3.5 9B"),
    ("human", "Human"),
]
FINDING3_GENERALITY_JUDGES = {
    "Gemini 3.1 Pro": "full100_gemini31_high_rerun1",
    "Kimi K2.5": "full100_kimi25_high_rerun1",
    "GPT-5.4": "full100_gpt54_high_rerun1",
}
CHANGED_UNCHANGED_PROBE_MODELS = [
    "gemini25_pro_openrouter",
    "kimi_k2_5_openrouter",
    "opus46",
    "qwen35_9b_openrouter",
    "gemma3_4b_openrouter",
]


def round_float(x: float, digits: int = 4) -> float:
    return round(float(x), digits)


def binomial_p_one_sided(k: int, n: int) -> float:
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)


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


def score_from_rows_raw(rows: list[dict], subset_ids: set[tuple[str, str]] | None = None) -> float:
    by_task = defaultdict(list)
    for row in rows:
        if subset_ids is not None and (row["task_id"], row["criterion_id"]) not in subset_ids:
            continue
        by_task[row["task_id"]].append(row)
    task_scores = []
    for criteria in by_task.values():
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
        if max_score:
            task_scores.append(100.0 * achieved / max_score)
    return sum(task_scores) / len(task_scores)


def score_from_rows(rows: list[dict], subset_ids: set[tuple[str, str]] | None = None) -> float:
    return round_float(score_from_rows_raw(rows, subset_ids), 2)


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
    return compute_finding1_rubric_capture()


def compute_finding1_rubric_capture() -> dict:
    path = PAPER / "finding1" / "rubric_as_response_capture" / "summary_all_models_nointro_underlying_eval_point_capture.json"
    obj = json.loads(path.read_text())
    models = {}
    for model_key, model_obj in sorted(obj["models"].items()):
        rubric_capture = model_obj["rubric_list_capture"]
        open_ended = model_obj["open_ended_response_morebench_same_cases"]
        models[model_key] = {
            "model": model_obj["model"],
            "group": model_obj.get("group", "unknown"),
            "rubric_list_capture_overall": round_float(rubric_capture["overall"], 1),
            "open_ended_response_morebench_same_cases_overall": round_float(open_ended["overall"], 1),
            "rubric_minus_open_ended_morebench": round_float(model_obj["rubric_minus_open_ended_morebench"], 1),
            "rows": int(rubric_capture["rows"]),
            "tasks": int(rubric_capture["tasks"]),
            "input_rows_complete": bool(model_obj["input_rows_complete"]),
        }
    return {
        "case_count": int(obj["case_count"]),
        "judge_model": obj["judge_model"],
        "prompt_variant": obj["prompt_variant"],
        "model_count": len(models),
        "models": models,
    }



def compute_finding3_matched_pairs() -> dict:
    obj = json.loads((PAPER / "finding3" / "criterion_pairs" / "finding1_confirmed_pairs.json").read_text())
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
        p_value = binomial_p_one_sided(
            max(bins[-1]["h_no_m_yes"], bins[-1]["h_yes_m_no"]),
            bins[-1]["h_no_m_yes"] + bins[-1]["h_yes_m_no"],
        )
        bins[-1]["binom_p"] = round_float(p_value, 50)
        bins[-1]["binom_p_sci_2"] = f"{p_value:.2e}"
    by_model = defaultdict(list)
    for p in pairs:
        by_model[p["md"]].append(p)
    per_model = []
    for tag, rows in by_model.items():
        h_yes = sum(1 for p in rows if p["hj"] == "YES")
        m_yes = sum(1 for p in rows if p["mj"] == "YES")
        per_model.append({
            "model": FINDING3_PAIR_MODEL_NAMES.get(tag, tag),
            "n": len(rows),
            "human_yes_pct": round_float(100.0 * h_yes / len(rows), 1),
            "model_yes_pct": round_float(100.0 * m_yes / len(rows), 1),
            "gap_pp": round_float(100.0 * (m_yes - h_yes) / len(rows), 1),
            "h_no_m_yes": sum(1 for p in rows if p["hj"] == "NO" and p["mj"] == "YES"),
            "h_yes_m_no": sum(1 for p in rows if p["hj"] == "YES" and p["mj"] == "NO"),
        })
    per_model.sort(key=lambda r: r["gap_pp"], reverse=True)
    total_plus = sum(1 for p in pairs if p["hj"] == "NO" and p["mj"] == "YES")
    total_minus = sum(1 for p in pairs if p["hj"] == "YES" and p["mj"] == "NO")
    return {
        "models": len(obj["models"]),
        "dilemmas": len(obj["dilemmas"]),
        "pairs": len(pairs),
        "human_yes_pct": round_float(100.0 * human_yes / len(pairs), 2),
        "model_yes_pct": round_float(100.0 * model_yes / len(pairs), 2),
        "gap_pp": round_float(100.0 * (model_yes - human_yes) / len(pairs), 2),
        "h_no_m_yes": total_plus,
        "h_yes_m_no": total_minus,
        "binom_p": round_float(binomial_p_one_sided(max(total_plus, total_minus), total_plus + total_minus), 50),
        "bins": bins,
        "per_model": per_model,
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


def compute_finding3() -> dict:
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
    generality_validation = compute_generality_validation()
    changed_unchanged = compute_changed_unchanged_decomposition(changed, unchanged)
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
        "matched_pairs": compute_finding3_matched_pairs(),
        "generality_validation": generality_validation,
        "changed_unchanged_decomposition": changed_unchanged,
    }


def compute_generality_validation() -> dict:
    base = PAPER / "finding3" / "generality_validation" / "rubric_d1_check"
    rows_by_judge_source = {}
    source_table = []
    for source_key, display in FINDING3_GENERALITY_SOURCES:
        item = {"source": display}
        ns = []
        raw_rates = []
        for judge, dirname in FINDING3_GENERALITY_JUDGES.items():
            rows = load_jsonl(base / dirname / source_key / "generality_check_results.jsonl")
            rows_by_judge_source[(judge, source_key)] = rows
            ns.append(len(rows))
            raw_rate = 100.0 * sum(bool(r["meets_requirements"]) for r in rows) / len(rows)
            item[judge] = round_float(raw_rate, 1)
            raw_rates.append(raw_rate)
        item["n"] = ns[0]
        item["mean"] = round_float(statistics.mean(raw_rates), 1)
        source_table.append(item)

    pooled_ai = {"source": "Pooled AI"}
    human = {"source": "Human"}
    gaps = {"source": "Gap (AI - Human)"}
    pooled_ai_raw_rates = []
    human_raw_rates = []
    for judge, dirname in FINDING3_GENERALITY_JUDGES.items():
        ai_rows = []
        for source_key, _ in FINDING3_GENERALITY_SOURCES:
            if source_key != "human":
                ai_rows.extend(rows_by_judge_source[(judge, source_key)])
        human_rows = rows_by_judge_source[(judge, "human")]
        ai_raw_rate = 100.0 * sum(bool(r["meets_requirements"]) for r in ai_rows) / len(ai_rows)
        human_raw_rate = 100.0 * sum(bool(r["meets_requirements"]) for r in human_rows) / len(human_rows)
        ai_rate = round_float(ai_raw_rate, 1)
        human_rate = round_float(human_raw_rate, 1)
        pooled_ai[judge] = ai_rate
        human[judge] = human_rate
        gaps[judge] = round_float(ai_rate - human_rate, 1)
        pooled_ai_raw_rates.append(ai_raw_rate)
        human_raw_rates.append(human_raw_rate)
    pooled_ai["n"] = sum(len(rows_by_judge_source[(next(iter(FINDING3_GENERALITY_JUDGES)), source_key)]) for source_key, _ in FINDING3_GENERALITY_SOURCES if source_key != "human")
    human["n"] = len(rows_by_judge_source[(next(iter(FINDING3_GENERALITY_JUDGES)), "human")])
    pooled_ai["mean"] = round_float(statistics.mean(pooled_ai_raw_rates), 1)
    human["mean"] = round_float(statistics.mean(human_raw_rates), 1)
    gaps["mean"] = round_float(pooled_ai["mean"] - human["mean"], 1)

    verdicts = {}
    for (judge, source_key), rows in rows_by_judge_source.items():
        for row in rows:
            verdicts[(source_key, row["task_id"], row["criterion_id"], judge)] = bool(row["meets_requirements"])
    keys = sorted({key[:3] for key in verdicts})

    def unanimity(keys_subset: list[tuple[str, str, str]]) -> dict:
        rows = []
        judges = list(FINDING3_GENERALITY_JUDGES)
        for key in keys_subset:
            if all(key + (judge,) in verdicts for judge in judges):
                rows.append([verdicts[key + (judge,)] for judge in judges])
        return {
            "n": len(rows),
            "unanimous_fail_pct": round_float(100.0 * sum(all(not x for x in row) for row in rows) / len(rows), 1),
            "unanimous_pass_pct": round_float(100.0 * sum(all(row) for row in rows) / len(rows), 1),
            "all_same_pct": round_float(100.0 * sum(len(set(row)) == 1 for row in rows) / len(rows), 1),
        }

    agreement = {}
    judge_pairs = [("Gemini 3.1 Pro", "Kimi K2.5"), ("Gemini 3.1 Pro", "GPT-5.4"), ("Kimi K2.5", "GPT-5.4")]
    for a, b in judge_pairs:
        same = []
        for key in keys:
            if key + (a,) in verdicts and key + (b,) in verdicts:
                same.append(verdicts[key + (a,)] == verdicts[key + (b,)])
        agreement[f"{a} vs {b}"] = round_float(100.0 * sum(same) / len(same), 1)

    return {
        "source_table": source_table,
        "pooled_ai": pooled_ai,
        "human": human,
        "gap": gaps,
        "z_statistics": {"Gemini 3.1 Pro": 29.97, "Kimi K2.5": 31.76, "GPT-5.4": 17.01},
        "unanimity": {
            "pooled_ai": unanimity([key for key in keys if key[0] != "human"]),
            "human": unanimity([key for key in keys if key[0] == "human"]),
            "all_sources": unanimity(keys),
        },
        "pairwise_agreement": agreement,
    }


def compute_changed_unchanged_decomposition(changed: list[dict], unchanged: list[dict]) -> dict:
    changed_ids = {(row["task_id"], row["criterion_id"]) for row in changed}
    unchanged_ids = {(row["task_id"], row["criterion_id"]) for row in unchanged}
    rows = {}
    raw_rows = {}
    for tag in CHANGED_UNCHANGED_PROBE_MODELS:
        base = CANON / "answer_eval" / tag / "full" / "judgements"
        human_rows = load_jsonl(find_single_data_jsonl(base / "human"))
        cascade_rows = load_jsonl(find_single_data_jsonl(base / "cascade"))
        raw_rows[tag] = {
            "changed_original": score_from_rows_raw(human_rows, changed_ids),
            "changed_rewritten": score_from_rows_raw(cascade_rows, changed_ids),
            "unchanged_original": score_from_rows_raw(human_rows, unchanged_ids),
            "unchanged_rewritten": score_from_rows_raw(cascade_rows, unchanged_ids),
        }
        rows[tag] = {key: round_float(value, 1) for key, value in raw_rows[tag].items()}
    primary_tags = ["gemini25_pro_openrouter", "kimi_k2_5_openrouter", "opus46"]
    small_tags = ["qwen35_9b_openrouter", "gemma3_4b_openrouter"]
    return {
        "changed_count": len(changed),
        "unchanged_count": len(unchanged),
        "per_model": rows,
        "changed_original_range": [min(rows[tag]["changed_original"] for tag in rows), max(rows[tag]["changed_original"] for tag in rows)],
        "changed_rewritten_range": [min(rows[tag]["changed_rewritten"] for tag in rows), max(rows[tag]["changed_rewritten"] for tag in rows)],
        "unchanged_original_range": [min(rows[tag]["unchanged_original"] for tag in rows), max(rows[tag]["unchanged_original"] for tag in rows)],
        "unchanged_rewritten_range": [min(rows[tag]["unchanged_rewritten"] for tag in rows), max(rows[tag]["unchanged_rewritten"] for tag in rows)],
        "primary_changed_original_avg": round_float(statistics.mean(raw_rows[tag]["changed_original"] for tag in primary_tags), 1),
        "primary_changed_rewritten_avg": round_float(statistics.mean(raw_rows[tag]["changed_rewritten"] for tag in primary_tags), 1),
        "small_changed_original_avg": round_float(statistics.mean(raw_rows[tag]["changed_original"] for tag in small_tags), 1),
        "small_changed_rewritten_avg": round_float(statistics.mean(raw_rows[tag]["changed_rewritten"] for tag in small_tags), 1),
        "primary_unchanged_original_avg": round_float(statistics.mean(raw_rows[tag]["unchanged_original"] for tag in primary_tags), 1),
        "primary_unchanged_rewritten_avg": round_float(statistics.mean(raw_rows[tag]["unchanged_rewritten"] for tag in primary_tags), 1),
        "small_unchanged_original_avg": round_float(statistics.mean(raw_rows[tag]["unchanged_original"] for tag in small_tags), 1),
        "small_unchanged_rewritten_avg": round_float(statistics.mean(raw_rows[tag]["unchanged_rewritten"] for tag in small_tags), 1),
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


def compute_pooled_primary_shares(summary: dict) -> dict:
    raw = summary["primary_by_side"]
    out = {}
    for side in ["human_only", "model_only"]:
        counts = raw[side]
        total = sum(counts.values())
        out[side] = {
            label: round_float(100.0 * count / total, 1)
            for label, count in counts.items()
        }
    return out


def compute_level2_primary_shares(summary: dict) -> dict:
    raw = summary["level2_primary_by_side"]
    out = {}
    for side in ["human_only", "model_only"]:
        counts = raw[side]
        total = sum(counts.values())
        out[side] = {
            label: round_float(100.0 * counts.get(label, 0) / total, 1)
            for label in FINDING2_LEVEL2_LABELS
        }
    return out


def compute_finding2_primary_by_model(label_rows: list[dict]) -> list[dict]:
    by_model = defaultdict(lambda: {"human_only": [], "model_only": []})
    for row in label_rows:
        if "[human-only]" in row["model"]:
            model = row["model"].replace(" [human-only]", "")
            by_model[model]["human_only"].append(row["parsed"]["primary_level1"])
        elif "[model-only]" in row["model"]:
            model = row["model"].replace(" [model-only]", "")
            by_model[model]["model_only"].append(row["parsed"]["primary_level1"])

    rows = []
    for model in FINDING2_MODEL_ORDER:
        human_labels = by_model[model]["human_only"]
        model_labels = by_model[model]["model_only"]
        hc = Counter(human_labels)
        mc = Counter(model_labels)
        item = {
            "model": model,
            "human_only": len(human_labels),
            "model_only": len(model_labels),
        }
        for label in FINDING2_PRIMARY_LABELS:
            item[label] = round_float(
                100.0 * mc[label] / len(model_labels) - 100.0 * hc[label] / len(human_labels),
                1,
            )
        rows.append(item)
    return rows


def compute_finding2() -> dict:
    global_summary = json.loads((PAPER / "finding2" / "coverage" / "global_unique_t70" / "summary.json").read_text())
    direct_check = json.loads((PAPER / "finding2" / "direct_check" / "top100_v2_high_pooled_summary.json").read_text())
    samebranch_input = json.loads((PAPER / "finding2" / "normative_tendencies" / "finding3_samebranch_intersection_input_summary.json").read_text())
    samebranch_labels = json.loads((PAPER / "finding2" / "normative_tendencies" / "finding3_samebranch_intersection_label_summary.complete_all.json").read_text())
    samebranch_level2 = json.loads((PAPER / "finding2" / "normative_tendencies" / "finding3_samebranch_intersection_level2_summary.complete_all.json").read_text())
    samebranch_label_rows = load_jsonl(PAPER / "finding2" / "normative_tendencies" / "finding3_samebranch_intersection_gpt54mini_high_labels.complete_all.jsonl")
    pairwise_means = compute_pairwise_primary_means(samebranch_labels)
    pooled_shares = compute_pooled_primary_shares(samebranch_labels)
    level2_shares = compute_level2_primary_shares(samebranch_level2)
    return {
        "global_unique": {
            "human_unique_concepts": int(global_summary["human_unique_concepts"]),
            "human_only": int(global_summary["human_only"]),
            "human_only_pct": round_float(global_summary["human_only_pct"], 1),
            "human_covered_by_model": int(global_summary["human_covered_by_model"]),
            "human_covered_pct": round_float(global_summary["human_covered_pct"], 1),
            "model_unique_concepts": int(global_summary["model_unique_concepts"]),
            "model_only": int(global_summary["model_only"]),
            "model_only_pct": round_float(global_summary["model_only_pct"], 1),
            "model_overlaps_human": int(global_summary["model_overlaps_human"]),
            "model_overlap_pct": round_float(global_summary["model_overlap_pct"], 1),
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
        "pairwise_primary_means": {label: pairwise_means[label] for label in FINDING2_PRIMARY_LABELS},
        "pooled_primary_shares": {
            label: {
                "human_only": pooled_shares["human_only"][label],
                "model_only": pooled_shares["model_only"][label],
            }
            for label in FINDING2_PRIMARY_LABELS
        },
        "pooled_level2_shares": {
            label: {
                "human_only": level2_shares["human_only"][label],
                "model_only": level2_shares["model_only"][label],
            }
            for label in FINDING2_LEVEL2_LABELS
        },
        "primary_label_by_model": compute_finding2_primary_by_model(samebranch_label_rows),
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
