#!/usr/bin/env bash
# Rebuttal run 1: cross-judge check of the Finding 1 rubric-as-response capture.
# Re-judges the existing capture inputs (100 cases, 7 models) with a second judge
# from a different model family. Requires network — run this yourself.
#
#   LAB_OPENROUTER_KEY=... bash run_crossjudge_capture.sh
#   # optionally: JUDGE_MODEL=deepseek/deepseek-v3.2-exp bash run_crossjudge_capture.sh
# Default judge is Kimi K2.5: already one of the paper's three generality judges
# (Appendix B.2.2), and family-disjoint from all 7 evaluated models.
#
# Outputs go to a judge-specific directory; nothing in release_staging is touched.
set -euo pipefail

: "${LAB_OPENROUTER_KEY:?LAB_OPENROUTER_KEY must be set}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../../../.." && pwd)
CODE="$REPO/release_staging/code"
CAP="$REPO/release_staging/data/paper_release/finding1/rubric_as_response_capture"

JUDGE_MODEL=${JUDGE_MODEL:-moonshotai/kimi-k2.5:nitro}
JUDGE_TAG=$(echo "$JUDGE_MODEL" | tr '/.:' '___')
OUT_BASE="$REPO/data/rebuttal/outputs/crossjudge_capture/$JUDGE_TAG"
# Kimi K2.5 is a reasoning model. Keep reasoning HIGH for judgement quality, but the
# default 60s per-request cap times out under a long trace + high concurrency. Fix is:
# generous per-request deadline, generous max_tokens (never truncate a high trace, or
# the judgement comes back empty), and lower concurrency to avoid provider queueing.
# Override any of these via env.
WORKERS=${WORKERS:-16}
REASONING=${REASONING:-high}
REQ_TIMEOUT=${REQ_TIMEOUT:-300}
MAX_TOKENS=${MAX_TOKENS:-10500}

declare -a JOBS=(
  "gemini25_pro|$CAP/gemini25_pro/inputs/gemini25_pro_finding1_rubricasresp_nointro.jsonl|Gemini 2.5 Pro"
  "gpt54|$CAP/gpt54/inputs/gpt54_finding1_rubricasresp_nointro.jsonl|GPT-5.4"
  "opus46|$CAP/opus46/inputs/opus46_finding1_rubricasresp_nointro.jsonl|Opus 4.6"
  "llama31_8b|$CAP/small_model_baselines/llama31_8b_openrouter/inputs/llama31_8b_openrouter_finding1_rubricasresp_nointro.jsonl|LLaMA 3.1 8B"
  "llama32_3b|$CAP/small_model_baselines/llama32_3b_openrouter/inputs/llama32_3b_openrouter_finding1_rubricasresp_nointro.jsonl|LLaMA 3.2 3B"
  "mistral7b|$CAP/small_model_baselines/mistral7b_v01_openrouter/inputs/mistral7b_v01_openrouter_finding1_rubricasresp_nointro.jsonl|Mistral 7B v0.1"
  "qwen25_7b|$CAP/small_model_baselines/qwen25_7b_openrouter/inputs/qwen25_7b_openrouter_finding1_rubricasresp_nointro.jsonl|Qwen 2.5 7B"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r key input name <<< "$job"
  echo "=== $name (judge: $JUDGE_MODEL) ==="
  outdir="$OUT_BASE/$key/judgements"
  # Do not let one model's stragglers abort the loop; the judge is resumable, so a
  # second pass repairs any timed-out idx. Score only once judging returns cleanly.
  if python3 "$CODE/run_best_judge_on_responses.py" \
    -i "$input" \
    -jt model_resp \
    -jm "$JUDGE_MODEL" \
    -a "$LAB_OPENROUTER_KEY" \
    -es 100 \
    -n "$WORKERS" \
    -o "$outdir" \
    --reasoning_effort "$REASONING" \
    --request_timeout_sec "$REQ_TIMEOUT" \
    --max_tokens "$MAX_TOKENS" \
    --rubricasresp_prompt_variant underlying_eval_point; then
    python3 "$CODE/score_rubric_capture.py" \
      --judgements "$outdir/model_resp_$(basename "$input")" \
      --output "$OUT_BASE/$key/summary_capture.json" \
      --model_name "$name"
  else
    echo "!! $name judging incomplete (see $outdir/*.errors.jsonl); rerun to resume." >&2
  fi
done
echo "Done. Summaries under $OUT_BASE/*/summary_capture.json"
