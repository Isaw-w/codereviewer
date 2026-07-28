#!/usr/bin/env bash
# Step B: score cheap candidate judges on the fair sample (502 criteria, 7 models),
# to find one that beats GPT-OSS at matching the Opus+GPT-5.6-sol consensus.
# All candidates are family-disjoint from the 7 judged models (no Qwen/Google/
# OpenAI-frontier/Anthropic/Meta/Mistral). Chinese families: DeepSeek, GLM, MiniMax, MiMo.
#
#   LAB_OPENROUTER_KEY=... bash run_judge_candidates.sh
# Resumable. Cheap (~$5 total on the sample).
set -uo pipefail
: "${LAB_OPENROUTER_KEY:?set LAB_OPENROUTER_KEY}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../../../.." && pwd)
JUDGE="$REPO/release_staging/code/run_best_judge_on_responses.py"
IN="$REPO/data/rebuttal/outputs/judge_select/inputs/fair_judge_sample.jsonl"
OUT="$REPO/data/rebuttal/outputs/judge_select"

# tag|slug   (all family-disjoint from the 7 judged models)
CANDIDATES=(
  "gptoss120b|openai/gpt-oss-120b"
  "deepseek_v4pro|deepseek/deepseek-v4-pro"
  "glm52|z-ai/glm-5.2"
  "minimax_m25|minimax/minimax-m2.5"
  "kimi_k3|moonshotai/kimi-k3"
)

for c in "${CANDIDATES[@]}"; do
  IFS='|' read -r tag slug <<< "$c"
  echo "=== $tag ($slug) ==="
  python3 "$JUDGE" -i "$IN" -jt model_resp -jm "$slug" -a "$LAB_OPENROUTER_KEY" \
    -es 21 -n 12 -o "$OUT/cand_$tag" \
    --reasoning_effort high --request_timeout_sec 300 \
    --rubricasresp_prompt_variant consideration_level_num \
    || echo "!! $tag incomplete (see $OUT/cand_$tag/*.errors.jsonl); rerun to resume." >&2
done
echo "Done. Tell Claude to score the candidates against frontier_consensus.json"
