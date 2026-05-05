#!/usr/bin/env python3
"""Build judge-ready matched-pair feature ablation inputs.

This script takes dual-method matched human/model rubric pairs and builds
single-factor ablation variants on the human criterion so we can re-judge
them under the original response.

The design is intentionally narrow. It automates only the feature edits that
are reasonably safe to do mechanically:

1. opening-verb replacement
2. degree / intensity stripping
3. explicit enumeration / example stripping

The script supports two rewrite modes:

1. `rules`
   narrow mechanical edits only
2. `llm`
   constrained single-feature rewrites with structured JSON output

The LLM mode is the path intended for semantic ablations like abstraction and
proposition-depth reduction.

Usage example:
    python3 release_staging/code/build_paired_feature_ablation.py \
        --output_dir release_staging/data/paper_release/finding1/paired_feature_ablation_smoke \
        --models opus46_openrouter \
        --subset h_no_m_yes \
        --min_cosine 0.80 \
        --per_model_limit 5
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import os

from openai import OpenAI


def setup_client(api_key: str) -> OpenAI:
    """Create an OpenRouter-backed OpenAI client."""
    timeout_sec = int(os.getenv("OPENROUTER_REQUEST_TIMEOUT_SEC", "60"))
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=timeout_sec,
        max_retries=0,
    )

PAIR_LABELS = {
    ("no", "yes"): "h_no_m_yes",
    ("yes", "no"): "h_yes_m_no",
    ("yes", "yes"): "h_yes_m_yes",
    ("no", "no"): "h_no_m_no",
}

MODEL_DIR_OVERRIDES = {
    "claude_sonnet4_openrouter": "claude_sonnet4",
    "opus46_openrouter": "opus46",
}

SHELL_PREFIXES = [
    "The response should ",
    "The response must ",
    "The response needs to ",
    "The response ought to ",
    "The response ",
    "Response should ",
    "Response must ",
    "Response needs to ",
    "Response ",
]

OPENING_ADVERBS = {
    "briefly",
    "clearly",
    "carefully",
    "explicitly",
    "directly",
    "systematically",
    "honestly",
    "accurately",
    "logically",
    "flatly",
    "gently",
    "completely",
    "enthusiastically",
    "transparently",
    "critically",
    "appropriately",
    "uncritically",
    "dogmatically",
}

DEGREE_WORDS = [
    "massive",
    "significant",
    "substantial",
    "major",
    "serious",
    "severe",
    "grave",
    "critical",
    "catastrophic",
    "considerable",
    "extreme",
    "dramatic",
]

ENUM_MARKERS = [
    "including",
    "such as",
    "e.g.",
    "for example",
    "i.e.",
]

FACTOR_ORDER = [
    "verb_model",
    "degree_strip",
    "enum_strip",
    "verb_degree",
    "verb_enum",
    "degree_enum",
    "verb_degree_enum",
]

LLM_FACTOR_ORDER = [
    "verb_only",
    "verb_format_neutral",
    "degree_only",
    "specificity_only",
    "proposition_depth_only",
]

# Bidirectional: harden model criterion by inserting human surface features
LLM_HARDEN_FACTORS = ["verb_harden", "degree_harden"]

LLM_SYSTEM_PROMPT = (
    "You rewrite one rubric criterion for causal ablation. "
    "Change exactly one requested feature and preserve everything else as much as possible. "
    "Be conservative. If you cannot make a clean one-feature rewrite, return the original unchanged. "
    "Do not optimize against any response. Do not make the criterion generally better. "
    "Do not copy the matched model criterion unless the requested factor is the opening verb and that verb is the only needed change. "
    "Return JSON only."
)


def normalize_judgement(value: str) -> str:
    text = (value or "").strip().lower()
    if "yes" in text:
        return "yes"
    if "no" in text:
        return "no"
    return "other"


def normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def strip_shell_prefix(text: str) -> Tuple[str, str]:
    for prefix in SHELL_PREFIXES:
        if text.startswith(prefix):
            return prefix, text[len(prefix):]
    return "", text


def parse_opening(text: str) -> Dict[str, object]:
    shell, remainder = strip_shell_prefix(text)
    working = remainder.lstrip()
    leading = remainder[: len(remainder) - len(working)]
    if not working:
        return {
            "shell": shell + leading,
            "adverbs": [],
            "verb": None,
            "rest": "",
            "verb_token": None,
        }

    tokens = working.split()
    adverbs: List[str] = []
    idx = 0
    while idx < len(tokens) and tokens[idx].strip(" ,.;:!?").lower() in OPENING_ADVERBS:
        adverbs.append(tokens[idx])
        idx += 1

    if idx >= len(tokens):
        return {
            "shell": shell + leading,
            "adverbs": adverbs,
            "verb": None,
            "rest": "",
            "verb_token": None,
        }

    verb_token = tokens[idx]
    consumed_tokens = tokens[: idx + 1]
    consumed_text = " ".join(consumed_tokens)
    rest = working[len(consumed_text):]
    return {
        "shell": shell + leading,
        "adverbs": adverbs,
        "verb": verb_token.strip(" ,.;:!?").lower(),
        "rest": rest,
        "verb_token": verb_token,
    }


def replace_opening_verb(text: str, replacement_token: str) -> Optional[str]:
    parsed = parse_opening(text)
    if not parsed["verb_token"]:
        return None
    rebuilt = f"{parsed['shell']}{replacement_token}{parsed['rest']}"
    return normalize_space(rebuilt)


def find_degree_terms(text: str) -> List[str]:
    found = []
    for word in DEGREE_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE):
            found.append(word)
    return found


def strip_degree_terms(text: str) -> Optional[str]:
    original = text
    for word in DEGREE_WORDS:
        text = re.sub(rf"\b{re.escape(word)}\b\s*", "", text, flags=re.IGNORECASE)
    text = normalize_space(text)
    if text == normalize_space(original):
        return None
    return text


def find_enum_markers(text: str) -> List[str]:
    found = []
    lower = text.lower()
    for marker in ENUM_MARKERS:
        if marker in lower:
            found.append(marker)
    if re.search(r"\([^)]*,[^)]*,[^)]*\)", text):
        found.append("parenthetical_list")
    return found


def strip_parenthetical_lists(text: str) -> str:
    return re.sub(r"\s*\((?:[^)]*,){2,}[^)]*\)", "", text)


def strip_marker_to_boundary(text: str, marker: str) -> str:
    pattern = rf"(,?\s*{re.escape(marker)}\s*.*?)(?=[.;]|$)"
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


def strip_enumeration(text: str) -> Optional[str]:
    original = text
    text = strip_parenthetical_lists(text)
    for marker in ENUM_MARKERS:
        text = strip_marker_to_boundary(text, marker)
    text = normalize_space(text)
    if text == normalize_space(original):
        return None
    return text


def capitalized_noninitial_count(text: str) -> int:
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9'’.-]*\b", text)
    if not tokens:
        return 0
    first = re.findall(r"\b[A-Z][A-Za-z0-9'’.-]*\b", text[:80])
    initial = first[0] if first else None
    return sum(1 for tok in tokens if tok != initial)


def approximate_info_units(text: str) -> int:
    units = 1
    units += text.count(",")
    units += len(re.findall(r"\bif\b", text, flags=re.IGNORECASE))
    units += len(re.findall(r"\bwhether\b", text, flags=re.IGNORECASE))
    units += len(re.findall(r"\bincluding\b", text, flags=re.IGNORECASE))
    units += len(re.findall(r"\bsuch as\b", text, flags=re.IGNORECASE))
    return units


def load_repo_env() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def extract_json_obj(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {cleaned[:300]}")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object")
    return obj


def shared_metadata(human_title: str, model_title: str) -> Dict[str, object]:
    human_open = parse_opening(human_title)
    model_open = parse_opening(model_title)
    return {
        "human_main_verb": human_open["verb"],
        "model_main_verb": model_open["verb"],
        "degree_terms": find_degree_terms(human_title),
        "enum_markers": find_enum_markers(human_title),
        "comma_count": human_title.count(","),
        "if_count": len(re.findall(r"\bif\b", human_title, flags=re.IGNORECASE)),
        "whether_count": len(re.findall(r"\bwhether\b", human_title, flags=re.IGNORECASE)),
        "capitalized_noninitial_count": capitalized_noninitial_count(human_title),
        "approx_info_units": approximate_info_units(human_title),
    }


def llm_factor_instructions(factor: str) -> str:
    mapping = {
        "verb_only": (
            "Change only the opening response-act / mode-of-achievement verb or nearest equivalent phrasing. "
            "Keep the rest of the proposition, examples, conditionals, and degree words intact."
        ),
        "verb_format_neutral": (
            "Change only the opening verb. "
            "Replace it with a verb that preserves exactly the same level of content demandingness — "
            "the response must still engage with the topic at the same depth and specificity — "
            "but removes any implicit format or presentation requirement the original verb carried. "
            "For example: 'analyzes' → 'examines', 'outlines' → 'describes', "
            "'explains' → 'addresses', "
            "'considers' → 'addresses', 'summarizes' → 'examines', 'evaluates' → 'examines', 'assesses' → 'examines', 'asks' → 'advises'. "
            "If the opening verb is any of the following, you MUST return it unchanged: "
            "notes, states, suggests, acknowledges, identifies, discusses, mentions, "
            "points out, concedes, weighs, indicates, describes, recognizes, highlights. "
            "These verbs already carry no format implication. "
            "If the criterion begins with a negation ('Does not', 'Fails to', 'Ignores', 'Never', "
            "'Avoids', 'Lacks'), it is a negation-framed criterion with no format implication — "
            "return it unchanged. "
            "Do NOT choose a verb that is generally easier or broader than the original. "
            "Keep all other wording, examples, conditionals, and degree words intact."
        ),
        "degree_only": (
            "Change only intensity or severity wording. "
            "Remove or soften degree terms like massive, significant, severe, major, catastrophic. "
            "Do not remove examples, lists, actors, scope, or conditionals."
        ),
        "specificity_only": (
            "Change only specificity level. "
            "Replace explicit examples, subdomains, or concrete listed realizations with a shorter abstract category. "
            "Keep the same verb and keep the same overall concern."
        ),
        "proposition_depth_only": (
            "Change only proposition depth. "
            "Compress a multi-layer or over-packed criterion into its core proposition. "
            "Keep the same general concern, polarity, and actor. "
            "Do not also deliberately soften the opening verb unless grammar makes a tiny adjustment unavoidable."
        ),
        "verb_harden": (
            "Replace the opening response-act verb in the MODEL criterion with the opening verb from the HUMAN criterion. "
            "Keep the model criterion's content, scope, examples, and degree words exactly as-is. Only swap the verb."
        ),
        "degree_harden": (
            "Insert one degree or intensity word from the HUMAN criterion into the MODEL criterion at a grammatically natural position. "
            "Do not change the model criterion's verb, scope, or examples. "
            "If no clean single-word insertion is possible, return the original unchanged."
        ),
    }
    return mapping[factor]


def llm_user_prompt(row: dict, factor: str) -> str:
    return f"""You are doing a single-feature ablation on a human rubric criterion.

Target feature to change: {factor}

Human criterion:
{row['ht']}

Matched model criterion for semantic reference only:
{row['mt']}

Pair context:
- pair_label: {row['pair_label']}
- cosine: {row['cos']}
- human_judgement: {row['hj']}
- model_judgement: {row['mj']}
- case_id: {row['c']}
- model_id: {row['md']}

Instructions:
- {llm_factor_instructions(factor)}
- Change one feature only.
- Preserve polarity, actor, scope, moral concern, and substantive target.
- Do not rewrite toward the response.
- Do not make the sentence more elegant than necessary.
- If a clean one-feature rewrite is not possible, return the original unchanged.

Return JSON only:
{{
  "rewritten": "<criterion text>",
  "changed": true or false,
  "changed_feature": "{factor}" or "none",
  "rationale": "<one short sentence>"
}}"""


def llm_harden_user_prompt(row: dict, factor: str) -> str:
    return f"""You are doing a single-feature ablation to harden a model rubric criterion.

Target feature to add: {factor}

Model criterion (rewrite this):
{row['mt']}

Human criterion (take the feature from here):
{row['ht']}

Pair context:
- cosine: {row['cos']}
- human_judgement: {row['hj']}
- model_judgement: {row['mj']}
- case_id: {row['c']}

Instructions:
- {llm_factor_instructions(factor)}
- Change one feature only in the model criterion.
- Preserve the model criterion's polarity, actor, scope, and substantive content exactly.
- Do not rewrite toward or against any response.
- Do not make the criterion more elegant or correct otherwise.
- If a clean single-feature change is not possible, return the original unchanged.

Return JSON only:
{{
  "rewritten": "<rewritten model criterion>",
  "changed": true or false,
  "changed_feature": "{factor}" or "none",
  "rationale": "<one short sentence>"
}}"""


def chat_completion(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 800,
    reasoning_effort: Optional[str] = None,
    retries: int = 4,
) -> tuple[str, int, int]:
    for attempt in range(retries):
        try:
            extra: dict = {}
            if reasoning_effort:
                extra["extra_body"] = {"reasoning": {"effort": reasoning_effort}}
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                top_p=0.01,
                max_tokens=max_tokens,
                **extra,
            )
            content = resp.choices[0].message.content or ""
            prompt_tokens = getattr(resp.usage, "prompt_tokens", -1)
            completion_tokens = getattr(resp.usage, "completion_tokens", -1)
            return content, prompt_tokens, completion_tokens
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def llm_factorize_title(
    client: OpenAI,
    rewrite_model: str,
    row: dict,
    factor: str,
    api_key_env: str,
) -> Tuple[Optional[str], Dict[str, object]]:
    metadata = shared_metadata(row["ht"], row["mt"])
    user = llm_user_prompt(row, factor)
    content, prompt_tokens, completion_tokens = chat_completion(
        client,
        rewrite_model,
        system=LLM_SYSTEM_PROMPT,
        user=user,
    )
    payload = extract_json_obj(content)
    rewritten = normalize_space(payload.get("rewritten", row["ht"]))
    changed = bool(payload.get("changed", False))
    if rewritten == normalize_space(row["ht"]):
        changed = False
    metadata.update(
        {
            "rewrite_mode": "llm",
            "rewrite_model": rewrite_model,
            "llm_changed_feature": str(payload.get("changed_feature", "")).strip(),
            "llm_rationale": str(payload.get("rationale", "")).strip(),
        }
    )
    if not changed:
        return None, metadata
    return rewritten, metadata


def llm_harden_title(
    client: OpenAI,
    rewrite_model: str,
    row: dict,
    factor: str,
    api_key_env: str,
) -> Tuple[Optional[str], Dict[str, object]]:
    """Harden model criterion by inserting a surface feature from human criterion."""
    metadata = shared_metadata(row["ht"], row["mt"])
    metadata["ablation_direction"] = "model_hardened"
    user = llm_harden_user_prompt(row, factor)
    content, prompt_tokens, completion_tokens = chat_completion(
        client,
        rewrite_model,
        system=LLM_SYSTEM_PROMPT,
        user=user,
    )
    payload = extract_json_obj(content)
    rewritten = normalize_space(payload.get("rewritten", row["mt"]))
    changed = bool(payload.get("changed", False))
    if rewritten == normalize_space(row["mt"]):
        changed = False
    metadata.update(
        {
            "rewrite_mode": "llm_harden",
            "rewrite_model": rewrite_model,
            "llm_changed_feature": str(payload.get("changed_feature", "")).strip(),
            "llm_rationale": str(payload.get("rationale", "")).strip(),
        }
    )
    if not changed:
        return None, metadata
    return rewritten, metadata


def factorize_title(
    human_title: str,
    model_title: str,
    factor: str,
) -> Tuple[Optional[str], Dict[str, object]]:
    human_open = parse_opening(human_title)
    model_open = parse_opening(model_title)
    metadata: Dict[str, object] = shared_metadata(human_title, model_title)
    metadata["rewrite_mode"] = "rules"

    text = human_title
    changed = False

    steps = {
        "verb_model": ["verb_model"],
        "degree_strip": ["degree_strip"],
        "enum_strip": ["enum_strip"],
        "verb_degree": ["verb_model", "degree_strip"],
        "verb_enum": ["verb_model", "enum_strip"],
        "degree_enum": ["degree_strip", "enum_strip"],
        "verb_degree_enum": ["verb_model", "degree_strip", "enum_strip"],
    }[factor]

    for step in steps:
        if step == "verb_model":
            replacement = model_open["verb_token"]
            if not replacement:
                return None, metadata
            updated = replace_opening_verb(text, replacement)
        elif step == "degree_strip":
            updated = strip_degree_terms(text)
        elif step == "enum_strip":
            updated = strip_enumeration(text)
        else:
            raise ValueError(f"Unknown step: {step}")

        if updated is not None and normalize_space(updated) != normalize_space(text):
            text = updated
            changed = True

    if not changed:
        return None, metadata
    return normalize_space(text), metadata


def model_dir(model_id: str) -> str:
    return MODEL_DIR_OVERRIDES.get(model_id, model_id)


def canonical_input_path(model_id: str, eval_dir: Path) -> Path:
    folder = model_dir(model_id)
    return eval_dir / folder / "full" / "inputs" / f"{folder}_full_under_human_rubric.jsonl"


def load_canonical_inputs(model_ids: Iterable[str], eval_dir: Path) -> Dict[str, Dict[str, dict]]:
    out: Dict[str, Dict[str, dict]] = {}
    for model_id in sorted(set(model_ids)):
        path = canonical_input_path(model_id, eval_dir)
        task_map: Dict[str, dict] = {}
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                task_map[row["TASK_ID"]] = row
        out[model_id] = task_map
    return out


def select_pairs(
    rows: Sequence[dict],
    subset: str,
    min_cosine: float,
    max_cosine: float,
    models: Optional[set[str]],
    per_model_limit: Optional[int],
) -> List[dict]:
    selected = []
    by_model_count: Counter[str] = Counter()
    for row in rows:
        hj = normalize_judgement(row["hj"])
        mj = normalize_judgement(row["mj"])
        label = PAIR_LABELS.get((hj, mj))
        if label != subset:
            continue
        cos = float(row["cos"])
        if cos < min_cosine or cos >= max_cosine:
            continue
        if models and row["md"] not in models:
            continue
        if per_model_limit is not None and by_model_count[row["md"]] >= per_model_limit:
            continue
        row = dict(row)
        row["pair_label"] = label
        selected.append(row)
        by_model_count[row["md"]] += 1
    return selected


def build_lookup(task_record: dict) -> Dict[str, dict]:
    return {normalize_space(r["title"]): r for r in task_record["RUBRIC"]}


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_cases_file(path: Path) -> List[dict]:
    """Load pre-selected cases file (from annotate_select_ablation_cases.py) and
    normalise field names to the internal pair row format."""
    rows = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                "c":          obj["case_id"],
                "ha":         obj["human_alias"],
                "ma":         obj["model_alias"],
                "md":         obj["model_id"],
                "cos":        float(obj["cosine"]),
                "ht":         obj["human_text"],
                "mt":         obj["model_text"],
                "hj":         obj["human_judgement"],
                "mj":         obj["model_judgement"],
                "pair_label": "h_no_m_yes",
            })
    return rows


def build_outputs(args: argparse.Namespace) -> Dict[str, Dict[str, int]]:
    rewrite_mode = getattr(args, "rewrite_mode", "rules")

    # ── load pairs ──────────────────────────────────────────────────────────
    if getattr(args, "cases_file", None):
        selected = load_cases_file(Path(args.cases_file))
        print(f"Loaded {len(selected)} pre-selected cases from {args.cases_file}")
    else:
        with Path(args.pairs_json).open() as f:
            data = json.load(f)
        pairs = data["pairs"]

        models = None
        if args.models:
            models = {m.strip() for m in args.models.split(",") if m.strip()}

        selected = select_pairs(
            pairs,
            subset=args.subset,
            min_cosine=args.min_cosine,
            max_cosine=args.max_cosine,
            models=models,
            per_model_limit=args.per_model_limit,
        )
        print(
            f"Selected {len(selected)} pairs "
            f"(subset={args.subset}, cosine in [{args.min_cosine:.2f}, {args.max_cosine:.2f}))"
        )

    print("Pairs by model:")
    counts = Counter(row["md"] for row in selected)
    for model_id, n in sorted(counts.items()):
        print(f"  {model_id}: {n}")

    # ── LLM client (only for llm mode) ──────────────────────────────────────
    client: Optional[OpenAI] = None
    rewrite_model_name: str = ""
    if rewrite_mode == "llm":
        load_repo_env()
        api_key_env = getattr(args, "api_key_env", "OPENROUTER_API_KEY")
        rewrite_model_name = getattr(args, "rewrite_model", "openai/gpt-4o")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise SystemExit(f"Missing API key: env var '{api_key_env}' is not set")
        client = setup_client(api_key)

    eval_dir = Path(args.eval_dir)
    canonical = load_canonical_inputs((row["md"] for row in selected), eval_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    factor_rows: Dict[Tuple[str, str], Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    skipped: Counter[str] = Counter()
    skipped_lock = __import__("threading").Lock()

    factors_list = [f.strip() for f in args.factors.split(",") if f.strip()]
    max_workers = getattr(args, "max_workers", 20)

    # Build work items: each (row, factor, original_criterion) tuple
    work_items = []
    for row in selected:
        model_id = row["md"]
        task_id = row["c"]
        task_record = canonical[model_id].get(task_id)
        if task_record is None:
            with skipped_lock:
                skipped["missing_task"] += 1
            continue
        title_lookup = build_lookup(task_record)
        original_criterion = title_lookup.get(normalize_space(row["ht"]))
        if original_criterion is None:
            with skipped_lock:
                skipped["missing_criterion"] += 1
            continue
        for factor in factors_list:
            work_items.append((row, factor, original_criterion))

    total_work = len(work_items)
    done_count = __import__("itertools").count(1)
    print(f"Submitting {total_work} rewrite tasks (max_workers={max_workers})")

    api_key_env_val = getattr(args, "api_key_env", "OPENROUTER_API_KEY")

    def process_item(item):
        row, factor, original_criterion = item
        model_id = row["md"]
        task_id = row["c"]
        if rewrite_mode == "llm":
            if factor in LLM_HARDEN_FACTORS:
                ablated_title, metadata = llm_harden_title(
                    client, rewrite_model_name, row, factor, api_key_env_val
                )
            else:
                ablated_title, metadata = llm_factorize_title(
                    client, rewrite_model_name, row, factor, api_key_env_val
                )
        else:
            ablated_title, metadata = factorize_title(row["ht"], row["mt"], factor)
        return row, factor, original_criterion, model_id, task_id, ablated_title, metadata

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item): item for item in work_items}
        for future in as_completed(futures):
            idx = next(done_count)
            try:
                row, factor, original_criterion, model_id, task_id, ablated_title, metadata = future.result()
            except Exception as exc:
                item = futures[future]
                print(f"  [{idx}/{total_work}] ERROR {item[0]['c']} {item[0]['ha']} +{item[1]}: {exc}")
                with skipped_lock:
                    skipped[f"error_{item[1]}"] += 1
                continue
            if not ablated_title:
                with skipped_lock:
                    skipped[f"unchanged_{factor}"] += 1
                continue
            print(f"  [{idx}/{total_work}] {task_id} {row['ha']} +{factor}: {ablated_title[:80]}")
            results.append((model_id, factor, task_id, original_criterion, row, ablated_title, metadata))

    for model_id, factor, task_id, original_criterion, row, ablated_title, metadata in results:
        is_harden = factor in LLM_HARDEN_FACTORS
        original_text = row["mt"] if is_harden else row["ht"]
        matched_text = row["ht"] if is_harden else row["mt"]
        new_criterion = {
            "id": f"{original_criterion['id']}__{factor}",
            "title": ablated_title,
            "weight": original_criterion["weight"],
            "annotations": {
                "rubric_dimension": original_criterion.get("annotations", {}).get("rubric_dimension"),
                "ablation_factor": factor,
                "ablation_direction": "model_hardened" if is_harden else "human_softened",
                "pair_label": row["pair_label"],
                "original_title": original_text,
                "matched_criterion_title": matched_text,
                "human_alias": row["ha"],
                "model_alias": row["ma"],
                "cosine": row["cos"],
                "human_judgement": row["hj"],
                "model_judgement": row["mj"],
                **metadata,
            },
        }
        factor_rows[(model_id, factor)][task_id].append(new_criterion)

    summary: Dict[str, Dict[str, int]] = defaultdict(dict)
    for (model_id, factor), by_task in sorted(factor_rows.items()):
        out_path = output_dir / f"{model_id}_{factor}.jsonl"
        judge_rows = []
        for task_id in sorted(by_task):
            task_record = dict(canonical[model_id][task_id])
            task_record["RUBRIC"] = by_task[task_id]
            task_record["_ablation_factor"] = factor
            judge_rows.append(task_record)
        n_tasks = write_jsonl(out_path, judge_rows)
        n_criteria = sum(len(v) for v in by_task.values())
        summary[model_id][factor] = n_criteria
        print(f"Wrote {out_path} :: tasks={n_tasks} criteria={n_criteria}")

    if skipped:
        print("Skipped rows:")
        for key, n in sorted(skipped.items()):
            print(f"  {key}: {n}")

    return summary


# ── canonical H_no mode helpers ─────────────────────────────────────────────

def llm_user_prompt_standalone(criterion_title: str, task_id: str, factor: str) -> str:
    return f"""You are doing a single-feature ablation on a human rubric criterion.

Target feature to change: {factor}

Human criterion:
{criterion_title}

Case context:
- task_id: {task_id}

Instructions:
- {llm_factor_instructions(factor)}
- Change one feature only.
- Preserve polarity, actor, scope, moral concern, and substantive target.
- Do not rewrite toward any response.
- Do not make the sentence more elegant than necessary.
- If a clean one-feature rewrite is not possible, return the original unchanged.

Return JSON only:
{{
  "rewritten": "<criterion text>",
  "changed": true or false,
  "changed_feature": "{factor}" or "none",
  "rationale": "<one short sentence>"
}}"""


def llm_factorize_standalone(
    client: OpenAI,
    rewrite_model: str,
    criterion_id: str,
    criterion_title: str,
    task_id: str,
    factor: str,
    api_key_env: str,
    *,
    max_tokens: int = 800,
    reasoning_effort: Optional[str] = None,
) -> Tuple[Optional[str], Dict[str, object]]:
    user = llm_user_prompt_standalone(criterion_title, task_id, factor)
    content, prompt_tokens, completion_tokens = chat_completion(
        client,
        rewrite_model,
        system=LLM_SYSTEM_PROMPT,
        user=user,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
    )
    payload = extract_json_obj(content)
    rewritten = normalize_space(payload.get("rewritten", criterion_title))
    changed = bool(payload.get("changed", False))
    if rewritten == normalize_space(criterion_title):
        changed = False
    metadata = {
        "rewrite_mode": "llm_standalone",
        "rewrite_model": rewrite_model,
        "llm_changed_feature": str(payload.get("changed_feature", "")).strip(),
        "llm_rationale": str(payload.get("rationale", "")).strip(),
    }
    if not changed:
        return None, metadata
    return rewritten, metadata


def load_canonical_rubric_lookup(rubric_dir: Path) -> Dict[str, dict]:
    """Load all criteria from one canonical human rubric file (all models share identical rubrics).
    Returns {criterion_id: {id, title, weight, annotations, task_id}}.
    rubric_dir must contain per-model subdirectories each with a human_rubric_*.jsonl file.
    """
    rubric_files = list(rubric_dir.glob("*/human_rubric_*.jsonl"))
    if not rubric_files:
        raise FileNotFoundError(f"No human rubric files found under {rubric_dir}")
    lookup: Dict[str, dict] = {}
    with rubric_files[0].open() as f:
        for line in f:
            record = json.loads(line)
            task_id = record["TASK_ID"]
            for crit in record["RUBRIC"]:
                lookup[crit["id"]] = {
                    "id": crit["id"],
                    "title": crit["title"],
                    "weight": crit["weight"],
                    "annotations": crit.get("annotations", {}),
                    "task_id": task_id,
                }
    return lookup


def load_canonical_hno_task_map(
    models: List[str],
    eval_dir: Path,
) -> Dict[str, Dict[str, List[str]]]:
    """For each model, collect {task_id: [criterion_ids]} where H judgement is NO.
    Returns {model_id: {task_id: [criterion_id, ...]}}.
    """
    result: Dict[str, Dict[str, List[str]]] = {}
    for m in models:
        folder = MODEL_DIR_OVERRIDES.get(m, m)
        jfile = eval_dir / folder / "full" / "judgements" / "human" / f"model_resp_{folder}_full_under_human_rubric.jsonl"
        by_task: Dict[str, List[str]] = defaultdict(list)
        with jfile.open() as f:
            for line in f:
                d = json.loads(line)
                if str(d.get("judgement", "")).strip().upper() == "NO":
                    by_task[d["task_id"]].append(d["criterion_id"])
        result[m] = dict(by_task)
    return result


def build_outputs_canonical_hno(args: argparse.Namespace) -> None:
    """Rewrite human criteria that got H_no in any model, then emit per-model judge inputs."""
    load_repo_env()
    api_key_env = getattr(args, "api_key_env", "OPENROUTER_API_KEY")
    rewrite_model_name = getattr(args, "rewrite_model", "openai/gpt-4o")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key: env var '{api_key_env}' is not set")
    client = setup_client(api_key)

    eval_dir = Path(args.eval_dir)
    rubric_dir = Path(args.rubric_dir)
    factor = args.factors.strip().split(",")[0].strip()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── discover models ──────────────────────────────────────────────────────
    all_models = [
        "opus46", "claude_sonnet4", "deepseekv32exp_openrouter",
        "qwen35_397b_a17b_openrouter", "gpt54_openrouter",
        "deepseek_r1_0528_openrouter", "gemini31_openrouter",
        "mimo_v2_pro_openrouter", "gemini3_flash_openrouter",
        "kimi_k2_5_openrouter", "gemini25_pro_openrouter",
    ]
    # reverse-map so model dir names -> canonical model id
    dir_to_model = {MODEL_DIR_OVERRIDES.get(m, m): m for m in all_models}

    print("Loading H_no judgements from canonical ...")
    hno_map = load_canonical_hno_task_map(all_models, eval_dir)
    all_hno_criterion_ids: set = set()
    for by_task in hno_map.values():
        for cids in by_task.values():
            all_hno_criterion_ids.update(cids)
    print(f"Unique H_no criteria: {len(all_hno_criterion_ids)}")
    total_pairs = sum(len(cids) for by_task in hno_map.values() for cids in by_task.values())
    print(f"Total (criterion x model) H_no pairs to re-judge: {total_pairs}")

    print("Loading canonical rubric lookup ...")
    rubric_lookup = load_canonical_rubric_lookup(rubric_dir)
    missing = all_hno_criterion_ids - set(rubric_lookup)
    if missing:
        print(f"WARNING: {len(missing)} criterion IDs not found in rubric lookup")

    # ── load/resume rewrite cache ────────────────────────────────────────────
    cache_path = Path(args.rewrite_cache) if getattr(args, "rewrite_cache", None) else \
        output_dir / f"rewrite_cache_{factor}.jsonl"
    rewrite_cache: Dict[str, Optional[str]] = {}  # criterion_id -> rewritten title (None = unchanged)
    if cache_path.exists():
        with cache_path.open() as f:
            for line in f:
                entry = json.loads(line)
                rewrite_cache[entry["criterion_id"]] = entry["rewritten"]
        print(f"Resumed: {len(rewrite_cache)} cached rewrites from {cache_path}")

    to_rewrite = [
        cid for cid in sorted(all_hno_criterion_ids)
        if cid not in rewrite_cache and cid in rubric_lookup
    ]
    smoke_n = getattr(args, "smoke_n", None)
    if smoke_n:
        import random as _random
        rng = _random.Random(42)
        to_rewrite = rng.sample(to_rewrite, min(smoke_n, len(to_rewrite)))
        print(f"SMOKE TEST: sampled {len(to_rewrite)} criteria for review")
    print(f"Criteria to rewrite: {len(to_rewrite)} (skipping {len(rewrite_cache)} cached)")

    # ── parallel rewrite ─────────────────────────────────────────────────────
    cache_lock = __import__("threading").Lock()
    max_workers = getattr(args, "max_workers", 3000)
    rewrite_max_tokens = getattr(args, "rewrite_max_tokens", 800)
    reasoning_effort_val = getattr(args, "reasoning_effort", None) or None
    done_count = __import__("itertools").count(1)
    total_rewrite = len(to_rewrite)

    def rewrite_one(criterion_id: str):
        crit = rubric_lookup[criterion_id]
        rewritten, metadata = llm_factorize_standalone(
            client, rewrite_model_name, criterion_id, crit["title"], crit["task_id"], factor, api_key_env,
            max_tokens=rewrite_max_tokens, reasoning_effort=reasoning_effort_val,
        )
        with cache_lock:
            rewrite_cache[criterion_id] = rewritten
            with cache_path.open("a") as cf:
                cf.write(json.dumps({"criterion_id": criterion_id, "rewritten": rewritten,
                                     "original": crit["title"], **metadata}) + "\n")
        return criterion_id, rewritten

    errors = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(rewrite_one, cid): cid for cid in to_rewrite}
        for future in as_completed(futures):
            idx = next(done_count)
            cid = futures[future]
            try:
                cid, rewritten = future.result()
                if idx % 200 == 0 or idx <= 5:
                    title = rubric_lookup[cid]["title"][:60]
                    result_str = rewritten[:60] if rewritten else "(unchanged)"
                    print(f"  [{idx}/{total_rewrite}] {cid[:8]}  {title!r} -> {result_str!r}")
            except Exception as exc:
                errors += 1
                print(f"  [{idx}/{total_rewrite}] ERROR {cid[:8]}: {exc}")

    changed_count = sum(1 for v in rewrite_cache.values() if v is not None)
    print(f"\nRewrite done: {changed_count}/{len(rewrite_cache)} changed, {errors} errors")

    # ── load canonical inputs for all models ────────────────────────────────
    print("Loading canonical input files ...")
    canonical = load_canonical_inputs(all_models, eval_dir)

    # ── emit per-model judge input files ─────────────────────────────────────
    for model_id in all_models:
        by_task = hno_map.get(model_id, {})
        if not by_task:
            continue
        judge_rows = []
        skipped_tasks = 0
        for task_id in sorted(by_task):
            criterion_ids = by_task[task_id]
            new_rubric = []
            for cid in criterion_ids:
                if cid not in rubric_lookup:
                    continue
                crit = rubric_lookup[cid]
                rewritten = rewrite_cache.get(cid)
                title = rewritten if rewritten is not None else crit["title"]
                new_rubric.append({
                    "id": f"{cid}__{factor}",
                    "title": title,
                    "weight": crit["weight"],
                    "annotations": {
                        **crit["annotations"],
                        "ablation_factor": factor,
                        "ablation_direction": "human_softened",
                        "original_title": crit["title"],
                        "rewritten": rewritten is not None,
                    },
                })
            if not new_rubric:
                skipped_tasks += 1
                continue
            task_dir = MODEL_DIR_OVERRIDES.get(model_id, model_id)
            task_record = canonical[model_id].get(task_id)
            if task_record is None:
                skipped_tasks += 1
                continue
            row = dict(task_record)
            row["RUBRIC"] = new_rubric
            row["_ablation_factor"] = factor
            judge_rows.append(row)

        out_path = output_dir / f"{MODEL_DIR_OVERRIDES.get(model_id, model_id)}_{factor}_hno.jsonl"
        n = write_jsonl(out_path, judge_rows)
        n_crit = sum(len(r["RUBRIC"]) for r in judge_rows)
        print(f"Wrote {out_path} :: tasks={n} criteria={n_crit} skipped_tasks={skipped_tasks}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs_json", default=None,
                        help="Path to all_bidir_pairs_data.json (paired mode; required unless --cases_file is set)")
    parser.add_argument("--eval_dir", required=True,
                        help="Root answer_eval directory (per-model subdirs with full/inputs and full/judgements)")
    parser.add_argument("--rubric_dir", default=None,
                        help="Rubric directory with per-model subdirs containing human_rubric_*.jsonl "
                             "(required for canonical_hno mode)")
    parser.add_argument(
        "--cases_file",
        default=None,
        help="Pre-selected cases JSONL (from annotate_select_ablation_cases.py). "
             "If set, overrides --pairs_json selection args.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--subset", default="h_no_m_yes", choices=sorted(set(PAIR_LABELS.values())))
    parser.add_argument("--min_cosine", type=float, default=0.70)
    parser.add_argument("--max_cosine", type=float, default=1.01)
    parser.add_argument("--models", default="")
    parser.add_argument("--per_model_limit", type=int, default=None)
    parser.add_argument(
        "--rewrite_mode",
        default="rules",
        choices=["rules", "llm"],
        help="'rules' = mechanical edits; 'llm' = LLM single-feature rewrites",
    )
    parser.add_argument(
        "--rewrite_model",
        default="openai/gpt-4o",
        help="Model to use for LLM rewrites (OpenRouter model string)",
    )
    parser.add_argument(
        "--api_key_env",
        default="LAB_OPENROUTER_KEY",
        help="Env var name for API key",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=20,
        help="Concurrent LLM rewrite workers",
    )
    parser.add_argument(
        "--mode",
        default="paired",
        choices=["paired", "canonical_hno"],
        help="'paired' = existing matched-pair ablation; 'canonical_hno' = rewrite all H_no criteria from canonical rubrics",
    )
    parser.add_argument(
        "--rewrite_cache",
        default=None,
        help="Path to resume/save rewrite cache JSONL (canonical_hno mode only). Defaults to output_dir/rewrite_cache_{factor}.jsonl",
    )
    parser.add_argument(
        "--factors",
        default=None,
        help="Comma-separated factors. Rules factors: " + ", ".join(FACTOR_ORDER) +
             ". LLM factors: " + ", ".join(LLM_FACTOR_ORDER),
    )
    parser.add_argument(
        "--smoke_n",
        type=int,
        default=None,
        help="If set, randomly sample this many criteria for a smoke test (canonical_hno mode only)",
    )
    parser.add_argument(
        "--rewrite_max_tokens",
        type=int,
        default=800,
        help="max_tokens for LLM rewrite calls (increase for verbose reasoning models)",
    )
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort for rewrite model (passed via OpenRouter extra_body)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "canonical_hno":
        if args.factors is None:
            args.factors = "verb_format_neutral"
        requested = [f.strip() for f in args.factors.split(",") if f.strip()]
        invalid = [f for f in requested if f not in LLM_FACTOR_ORDER]
        if invalid:
            raise SystemExit(f"Invalid factor(s) for canonical_hno mode: {invalid}. Valid: {LLM_FACTOR_ORDER}")
        if len(requested) != 1:
            raise SystemExit("canonical_hno mode supports exactly one factor at a time")
        if args.rewrite_mode != "llm":
            raise SystemExit("canonical_hno mode requires --rewrite_mode llm")
        build_outputs_canonical_hno(args)
        return

    rewrite_mode = getattr(args, "rewrite_mode", "rules")
    valid_factors = (LLM_FACTOR_ORDER + LLM_HARDEN_FACTORS) if rewrite_mode == "llm" else FACTOR_ORDER
    default_factors = ",".join(LLM_FACTOR_ORDER + LLM_HARDEN_FACTORS) if rewrite_mode == "llm" else \
        "verb_model,degree_strip,enum_strip,verb_degree,verb_enum,degree_enum,verb_degree_enum"
    if args.factors is None:
        args.factors = default_factors
    requested = [f.strip() for f in args.factors.split(",") if f.strip()]
    invalid = [f for f in requested if f not in valid_factors]
    if invalid:
        raise SystemExit(f"Invalid factor(s) for mode '{rewrite_mode}': {invalid}. Valid: {valid_factors}")
    build_outputs(args)


if __name__ == "__main__":
    main()
