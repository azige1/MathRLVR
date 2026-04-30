#!/usr/bin/env bash

set -euo pipefail

# Convenience launcher for the EasyR1 reasoning RL project.
# Run from the EasyR1 repository root.
#
# Usage:
#   SUITE_MODE=base bash examples/run_reasoning_rl_suite_a10.sh
#   SUITE_MODE=grpo_smoke bash examples/run_reasoning_rl_suite_a10.sh
#   SUITE_MODE=grpo bash examples/run_reasoning_rl_suite_a10.sh
#   SUITE_MODE=grpo_filtered bash examples/run_reasoning_rl_suite_a10.sh
#   SUITE_MODE=answer_only bash examples/run_reasoning_rl_suite_a10.sh
#   SUITE_MODE=format_only bash examples/run_reasoning_rl_suite_a10.sh
#   SUITE_MODE=clean_data bash examples/run_reasoning_rl_suite_a10.sh
#   SUITE_MODE=clean_concise bash examples/run_reasoning_rl_suite_a10.sh
#   SUITE_MODE=gspo bash examples/run_reasoning_rl_suite_a10.sh
#   SUITE_MODE=print bash examples/run_reasoning_rl_suite_a10.sh

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-easyr1-reasoning}"

SUITE_MODE="${SUITE_MODE:-print}"
GPU_IDS="${GPU_IDS:-0,1}"
SINGLE_GPU_ID="${SINGLE_GPU_ID:-0}"

mkdir -p logs results

print_plan() {
  cat <<'EOF'
Recommended order:
  1. SUITE_MODE=base
  2. SUITE_MODE=grpo_smoke
  3. SUITE_MODE=grpo
  4. SUITE_MODE=grpo_filtered
  5. SUITE_MODE=answer_only
  6. SUITE_MODE=format_only
  7. SUITE_MODE=clean_data
  8. SUITE_MODE=clean_concise
  9. SUITE_MODE=gspo_smoke
  10. SUITE_MODE=gspo

Examples:
  SUITE_MODE=base bash examples/run_reasoning_rl_suite_a10.sh
  SUITE_MODE=grpo_smoke bash examples/run_reasoning_rl_suite_a10.sh
  SUITE_MODE=grpo bash examples/run_reasoning_rl_suite_a10.sh
EOF
}

case "${SUITE_MODE}" in
  print)
    print_plan
    ;;

  base)
    CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" \
    OUTPUT_FILE=results/qwen2_5_1_5b_math_base_eval.jsonl \
    SUMMARY_FILE=results/qwen2_5_1_5b_math_base_eval_report.json \
    bash examples/qwen2_5_1_5b_math_base_eval_a10.sh \
      | tee logs/qwen2_5_1_5b_math_base_eval.log
    ;;

  grpo_smoke)
    CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" \
    N_GPUS=1 \
    MAX_STEPS=2 \
    VAL_FREQ=-1 \
    SAVE_FREQ=1 \
    VAL_BEFORE_TRAIN=false \
    FORMAT_WEIGHT=0.1 \
    EXPERIMENT_NAME=qwen2_5_1_5b_math_grpo_smoke \
    bash examples/qwen2_5_1_5b_math_grpo_a10.sh \
      | tee logs/qwen2_5_1_5b_math_grpo_smoke.log
    ;;

  gspo_smoke)
    CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" \
    N_GPUS=1 \
    MAX_STEPS=2 \
    VAL_FREQ=-1 \
    SAVE_FREQ=1 \
    VAL_BEFORE_TRAIN=false \
    FORMAT_WEIGHT=0.1 \
    EXPERIMENT_NAME=qwen2_5_1_5b_math_gspo_smoke \
    bash examples/qwen2_5_1_5b_math_gspo_a10.sh \
      | tee logs/qwen2_5_1_5b_math_gspo_smoke.log
    ;;

  grpo)
    nohup env CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    MAX_STEPS="${MAX_STEPS:-300}" \
    VAL_FREQ="${VAL_FREQ:-50}" \
    SAVE_FREQ="${SAVE_FREQ:-50}" \
    VAL_BEFORE_TRAIN=false \
    FORMAT_WEIGHT=0.1 \
    EXPERIMENT_NAME=qwen2_5_1_5b_math_grpo_a10 \
    bash examples/qwen2_5_1_5b_math_grpo_a10.sh \
      > logs/qwen2_5_1_5b_math_grpo_a10.log 2>&1 &
    echo "Started GRPO composite. Log: logs/qwen2_5_1_5b_math_grpo_a10.log"
    ;;

  grpo_filtered)
    nohup env CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    MAX_STEPS="${MAX_STEPS:-300}" \
    VAL_FREQ="${VAL_FREQ:-50}" \
    SAVE_FREQ="${SAVE_FREQ:-50}" \
    VAL_BEFORE_TRAIN=false \
    FORMAT_WEIGHT=0.1 \
    ONLINE_FILTERING=true \
    FILTER_LOW=0.01 \
    FILTER_HIGH=0.99 \
    EXPERIMENT_NAME=qwen2_5_1_5b_math_grpo_filtered_a10 \
    bash examples/qwen2_5_1_5b_math_grpo_a10.sh \
      > logs/qwen2_5_1_5b_math_grpo_filtered_a10.log 2>&1 &
    echo "Started GRPO filtered. Log: logs/qwen2_5_1_5b_math_grpo_filtered_a10.log"
    ;;

  answer_only)
    nohup env CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    MAX_STEPS="${MAX_STEPS:-150}" \
    VAL_FREQ="${VAL_FREQ:-50}" \
    SAVE_FREQ="${SAVE_FREQ:-50}" \
    VAL_BEFORE_TRAIN=false \
    FORMAT_WEIGHT=0.0 \
    EXPERIMENT_NAME=qwen2_5_1_5b_math_grpo_answer_only_a10 \
    bash examples/qwen2_5_1_5b_math_grpo_a10.sh \
      > logs/qwen2_5_1_5b_math_grpo_answer_only_a10.log 2>&1 &
    echo "Started answer-only ablation. Log: logs/qwen2_5_1_5b_math_grpo_answer_only_a10.log"
    ;;

  format_only)
    nohup env CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    MAX_STEPS="${MAX_STEPS:-150}" \
    VAL_FREQ="${VAL_FREQ:-50}" \
    SAVE_FREQ="${SAVE_FREQ:-50}" \
    VAL_BEFORE_TRAIN=false \
    FORMAT_WEIGHT=1.0 \
    EXPERIMENT_NAME=qwen2_5_1_5b_math_grpo_format_only_a10 \
    bash examples/qwen2_5_1_5b_math_grpo_a10.sh \
      > logs/qwen2_5_1_5b_math_grpo_format_only_a10.log 2>&1 &
    echo "Started format-only ablation. Log: logs/qwen2_5_1_5b_math_grpo_format_only_a10.log"
    ;;

  clean_data)
    CLEAN_DATA_DIR="${CLEAN_DATA_DIR:-data/math12k_verifier_clean}"
    nohup env CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    TRAIN_FILES="${CLEAN_DATA_DIR}@train" \
    VAL_FILES="${VAL_FILES:-hiyouga/math12k@test}" \
    FORMAT_PROMPT="${FORMAT_PROMPT:-./examples/format_prompt/math.jinja}" \
    MAX_STEPS="${MAX_STEPS:-150}" \
    VAL_FREQ="${VAL_FREQ:-50}" \
    SAVE_FREQ="${SAVE_FREQ:-50}" \
    VAL_BEFORE_TRAIN=false \
    FORMAT_WEIGHT=0.1 \
    EXPERIMENT_NAME=qwen2_5_3b_math_grpo_clean_data_a10 \
    bash examples/qwen2_5_1_5b_math_grpo_a10.sh \
      > logs/qwen2_5_3b_math_grpo_clean_data_a10.log 2>&1 &
    echo "Started verifier-clean GRPO. Log: logs/qwen2_5_3b_math_grpo_clean_data_a10.log"
    ;;

  clean_concise)
    CLEAN_DATA_DIR="${CLEAN_DATA_DIR:-data/math12k_verifier_clean}"
    nohup env CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    TRAIN_FILES="${CLEAN_DATA_DIR}@train" \
    VAL_FILES="${VAL_FILES:-hiyouga/math12k@test}" \
    FORMAT_PROMPT="${FORMAT_PROMPT:-./examples/format_prompt/math_concise_final.jinja}" \
    MAX_STEPS="${MAX_STEPS:-150}" \
    VAL_FREQ="${VAL_FREQ:-50}" \
    SAVE_FREQ="${SAVE_FREQ:-50}" \
    VAL_BEFORE_TRAIN=false \
    FORMAT_WEIGHT=0.1 \
    EXPERIMENT_NAME=qwen2_5_3b_math_grpo_clean_concise_a10 \
    bash examples/qwen2_5_1_5b_math_grpo_a10.sh \
      > logs/qwen2_5_3b_math_grpo_clean_concise_a10.log 2>&1 &
    echo "Started verifier-clean + concise GRPO. Log: logs/qwen2_5_3b_math_grpo_clean_concise_a10.log"
    ;;

  gspo)
    nohup env CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    MAX_STEPS="${MAX_STEPS:-300}" \
    VAL_FREQ="${VAL_FREQ:-50}" \
    SAVE_FREQ="${SAVE_FREQ:-50}" \
    VAL_BEFORE_TRAIN=false \
    FORMAT_WEIGHT=0.1 \
    EXPERIMENT_NAME=qwen2_5_1_5b_math_gspo_a10 \
    bash examples/qwen2_5_1_5b_math_gspo_a10.sh \
      > logs/qwen2_5_1_5b_math_gspo_a10.log 2>&1 &
    echo "Started GSPO composite. Log: logs/qwen2_5_1_5b_math_gspo_a10.log"
    ;;

  *)
    echo "Unknown SUITE_MODE: ${SUITE_MODE}" >&2
    print_plan >&2
    exit 2
    ;;
esac
