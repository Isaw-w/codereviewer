#!/usr/bin/env bash
# Finding 1 consideration-level capture, FULL 100 cases, all 7 capture models,
# judged by MoReBench's own judge (GPT-OSS-120B). Validated on 15 cases to match
# the frontier reference (Opus 4.8 + GPT-5.6-sol) at 92.2%, and ~4pt conservative.
#
#   LAB_OPENROUTER_KEY=... bash run_consid_full100_gptoss.sh
# Resumable: rerun to repair any timed-out criteria.
set -uo pipefail
: "${LAB_OPENROUTER_KEY:?set LAB_OPENROUTER_KEY (e.g. =\$OPENROUTER_API_KEY)}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../../../.." && pwd)
JUDGE="$REPO/release_staging/code/run_best_judge_on_responses.py"
CAP="$REPO/release_staging/data/paper_release/finding1/rubric_as_response_capture"
OUT="$REPO/data/rebuttal/outputs/finding1_consid_full100"
JM="openai/gpt-oss-120b"

run() {  # $1 tag  $2 input
  echo "=== $1 ==="
  python3 "$JUDGE" -i "$2" -jt model_resp -jm "$JM" -a "$LAB_OPENROUTER_KEY" \
    -es 100 -n 40 -o "$OUT/$1" \
    --reasoning_effort high --request_timeout_sec 300 \
    --rubricasresp_prompt_variant consideration_level_num \
    || echo "!! $1 incomplete (see $OUT/$1/*.errors.jsonl); rerun to resume." >&2
}

# 3 frontier
run gemini25_pro "$CAP/gemini25_pro/inputs/gemini25_pro_finding1_rubricasresp_nointro.jsonl"
run gpt54        "$CAP/gpt54/inputs/gpt54_finding1_rubricasresp_nointro.jsonl"
run opus46       "$CAP/opus46/inputs/opus46_finding1_rubricasresp_nointro.jsonl"
# 4 small baselines
SB="$CAP/small_model_baselines"
run llama31_8b   "$SB/llama31_8b_openrouter/inputs/llama31_8b_openrouter_finding1_rubricasresp_nointro.jsonl"
run llama32_3b   "$SB/llama32_3b_openrouter/inputs/llama32_3b_openrouter_finding1_rubricasresp_nointro.jsonl"
run mistral7b    "$SB/mistral7b_v01_openrouter/inputs/mistral7b_v01_openrouter_finding1_rubricasresp_nointro.jsonl"
run qwen25_7b    "$SB/qwen25_7b_openrouter/inputs/qwen25_7b_openrouter_finding1_rubricasresp_nointro.jsonl"

echo "Done. Tell Claude to score $OUT/*"
