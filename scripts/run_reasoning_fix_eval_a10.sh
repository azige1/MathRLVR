#!/usr/bin/env bash
set -euo pipefail

# Run from the EasyR1 repository root inside the EasyR1 Docker container.
# This evaluates whether concise prompting + stop postprocessing reduces
# overlong/truncation and verifier false-negative issues.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="${EASYR1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${EASYR1_ROOT}"
mkdir -p logs results

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
GRPO_LORA="${GRPO_LORA:-checkpoints/easyr1_reasoning/qwen2_5_1_5b_math_grpo_a10/global_step_150/actor/lora_adapter}"
LIMIT="${LIMIT:-200}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SINGLE_GPU_ID="${SINGLE_GPU_ID:-0}"
PROMPT_FILE="${PROMPT_FILE:-examples/format_prompt/math_concise_final.jinja}"

run_eval() {
  local name="$1"
  local lora="$2"
  local lora_args=()
  if [ -n "${lora}" ]; then
    lora_args=(--lora-path "${lora}")
  fi

  env CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" python scripts/eval_math_best_of_n.py \
    --model-path "${MODEL_PATH}" \
    "${lora_args[@]}" \
    --dataset-name hiyouga/math12k \
    --dataset-split test \
    --prompt-key problem \
    --answer-key answer \
    --format-prompt "${PROMPT_FILE}" \
    --output-file "results/${name}_math12k_concise_stop_${LIMIT}.jsonl" \
    --report-file "results/${name}_math12k_concise_stop_${LIMIT}_report.json" \
    --limit "${LIMIT}" \
    --batch-size "${BATCH_SIZE}" \
    --n 1 \
    --temperature 0.0 \
    --top-p 1.0 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --postprocess-stop-markers \
    --stop-after-boxed
}

echo "== Concise prompt + stop postprocessing eval =="
echo "MODEL_PATH=${MODEL_PATH}"
echo "LIMIT=${LIMIT}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "PROMPT_FILE=${PROMPT_FILE}"

run_eval base ""
run_eval grpo_composite "${GRPO_LORA}"

python scripts/collect_rlvr_eval_reports.py \
  --best-of-n-output results/best_of_n_eval_report.json \
  --cross-benchmark-output results/cross_benchmark_eval_report.json

echo "Fix eval finished."
