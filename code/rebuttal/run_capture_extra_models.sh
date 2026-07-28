#!/usr/bin/env bash
# Finding 1 consideration-level capture for the 8 primary models that were never
# capture-judged. Same 100 cases, same human rubrics, same judge (GPT-OSS-120B),
# same prompt variant as run_consid_full100_gptoss.sh, so results are poolable
# with the finding1 consideration run described in the release notes.
#
# Inputs were rebuilt locally from outputs/canonical/rubrics/<model>/ai_rubric_500cases_*.jsonl
# and verified to reproduce the released gpt54 input file exactly.
#
#   LAB_OPENROUTER_KEY=... bash run_capture_extra_models.sh
# Resumable: rerun to repair any timed-out criteria.
set -uo pipefail
: "${LAB_OPENROUTER_KEY:?set LAB_OPENROUTER_KEY (e.g. =\$OPENROUTER_API_KEY)}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../../../.." && pwd)
JUDGE="$REPO/release_staging/code/run_best_judge_on_responses.py"
BASE="$REPO/data/rebuttal/outputs/capture_extra_models"
IN="$BASE/inputs"
OUT="$BASE"
JM="openai/gpt-oss-120b"

run() {
  echo "=== $1 ==="
  python3 "$JUDGE" -i "$IN/$1_finding1_rubricasresp_nointro.jsonl" -jt model_resp -jm "$JM" \
    -a "$LAB_OPENROUTER_KEY" -es 100 -n 100 -o "$OUT/$1" \
    --reasoning_effort high --request_timeout_sec 300 \
    --rubricasresp_prompt_variant consideration_level_num \
    || echo "!! $1 incomplete (see $OUT/$1/*.errors.jsonl); rerun to resume." >&2
}

# All 8 primary models that were never capture-judged. Together with the existing
# 7 runs this gives 15 models on both axes, spanning 74.9 to 91.0 on the open-ended
# side with no gap. Table 6 rewritten scores, for reference:
#   kimi_k2_5 91.04, deepseek_r1 90.17, mimo_v2_pro 89.87, qwen35_397b 89.57,
#   gemini31 88.85, deepseekv32exp 88.78, gemini3_flash 87.50, claude_sonnet4 84.22
# The judged model contributes only its already-written rubric text; every call goes
# to GPT-OSS-120B, so model choice does not change throughput.
run claude_sonnet4
run deepseek_r1
run deepseekv32exp
run gemini31
run gemini3_flash
run kimi_k2_5
run mimo_v2_pro
run qwen35_397b

echo "Done. 8 models x 2,254 human criteria = 18,032 judgements."
echo "Then: python3 scripts/rebuttal/score_capture_vs_analysis.py"
