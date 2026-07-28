#!/usr/bin/env bash
# 15-case consideration-level probe: confirm the 3-case result scales.
# Runs 3 judge passes on the same 15-case gemini capture input:
#   1. Opus  literal            (underlying_eval_point)  -> for the literal->consideration gap
#   2. Opus  consideration_level                          -> the new number
#   3. GPT-OSS consideration_level                        -> cross-judge agreement at same prompt
#
#   LAB_OPENROUTER_KEY=... bash run_consid15_probe.sh
# Resumable: rerun to repair any timed-out criteria.
set -uo pipefail

: "${LAB_OPENROUTER_KEY:?set LAB_OPENROUTER_KEY (e.g. LAB_OPENROUTER_KEY=\$OPENROUTER_API_KEY)}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../../../.." && pwd)
JUDGE="$REPO/release_staging/code/run_best_judge_on_responses.py"
IN="$REPO/data/rebuttal/outputs/judge_probe/inputs/gemini25_pro_capture_15case.jsonl"
OUT="$REPO/data/rebuttal/outputs/judge_probe"

run() {  # $1 out-subdir  $2 judge-model  $3 variant  $4 workers
  echo "=== $3 | $2 ==="
  python3 "$JUDGE" -i "$IN" -jt model_resp -jm "$2" -a "$LAB_OPENROUTER_KEY" \
    -es 15 -n "$4" -o "$OUT/$1" \
    --reasoning_effort high --request_timeout_sec 300 \
    --rubricasresp_prompt_variant "$3" \
    || echo "!! $1 incomplete (see $OUT/$1/*.errors.jsonl); rerun to resume." >&2
}

run opus48_lit15     anthropic/claude-opus-4.8 underlying_eval_point 12
run opus48_consid15  anthropic/claude-opus-4.8 consideration_level   12
run gptoss_consid15  openai/gpt-oss-120b       consideration_level   30

echo "Done. Tell Claude to score: opus48_lit15 / opus48_consid15 / gptoss_consid15"
