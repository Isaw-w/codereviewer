#!/usr/bin/env bash
# Validate the SIMPLE consideration prompt across judges (15-case gemini capture).
# Always runs Opus + GPT-OSS. Runs GPT-5.6 too IF you set its slug:
#
#   LAB_OPENROUTER_KEY=...  bash run_consid_simple15.sh                 # Opus + GPT-OSS
#   LAB_OPENROUTER_KEY=...  GPT56_MODEL=openai/gpt-5.6-sol  bash run_consid_simple15.sh
#
# Resumable: rerun to repair any timed-out criteria.
set -uo pipefail
: "${LAB_OPENROUTER_KEY:?set LAB_OPENROUTER_KEY (e.g. =\$OPENROUTER_API_KEY)}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../../../.." && pwd)
JUDGE="$REPO/release_staging/code/run_best_judge_on_responses.py"
IN="$REPO/data/rebuttal/outputs/judge_probe/inputs/gemini25_pro_capture_15case.jsonl"
OUT="$REPO/data/rebuttal/outputs/judge_probe"
GPT56_MODEL="${GPT56_MODEL:-}"

run() {  # $1 out-subdir  $2 judge  $3 variant  $4 workers
  echo "=== $3 | $2 ==="
  python3 "$JUDGE" -i "$IN" -jt model_resp -jm "$2" -a "$LAB_OPENROUTER_KEY" \
    -es 15 -n "$4" -o "$OUT/$1" \
    --reasoning_effort high --request_timeout_sec 300 \
    --rubricasresp_prompt_variant "$3" \
    || echo "!! $1 incomplete (see $OUT/$1/*.errors.jsonl); rerun to resume." >&2
}

run opus48_simple15   anthropic/claude-opus-4.8 consideration_simple 12
run gptoss_simple15    openai/gpt-oss-120b       consideration_simple 30

if [ -n "$GPT56_MODEL" ]; then
  run gpt56_simple15   "$GPT56_MODEL" consideration_simple 12
  run gpt56_lit15      "$GPT56_MODEL" underlying_eval_point 12
else
  echo ">> GPT56_MODEL not set — skipped GPT-5.6 runs. Set it and rerun to add them." >&2
fi

echo "Done. Tell Claude to score the *_simple15 (and gpt56_*) dirs."
