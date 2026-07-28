#!/usr/bin/env bash
# Rebuttal run 3: score weak-model open-ended responses under the cascade-rewritten
# human rubric. Requires network — run this yourself.
#
#   LAB_OPENROUTER_KEY=... bash run_weak_cascade_judge.sh
#
# Inputs were built locally by build_weak_cascade_inputs.py (100 cases per model).
# Judge is GPT-OSS-120B, same as the paper's cascade rescoring.
# Afterwards run: python3 score_weak_cascade.py
set -euo pipefail

: "${LAB_OPENROUTER_KEY:?LAB_OPENROUTER_KEY must be set}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../../../.." && pwd)
CODE="$REPO/release_staging/code"
BASE="$REPO/data/rebuttal/outputs/weak_cascade_test"

JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}
WORKERS=${WORKERS:-50}

for model in llama31_8b_openrouter llama32_3b_openrouter mistral7b_v01_openrouter qwen25_7b_openrouter; do
  echo "=== $model ==="
  python3 "$CODE/run_best_judge_on_responses.py" \
    -i "$BASE/inputs/${model}_response_under_cascade_rubric.jsonl" \
    -jt model_resp \
    -jm "$JUDGE_MODEL" \
    -a "$LAB_OPENROUTER_KEY" \
    -es 100 \
    -n "$WORKERS" \
    -o "$BASE/judgements/$model"
done
echo "Done. Now run: python3 $SCRIPT_DIR/score_weak_cascade.py"
