#!/usr/bin/env bash

set -euo pipefail
set -x

# Run from the EasyR1 repository root.
# Base evaluation for Qwen2.5-3B-Instruct on math12k test.
# The filename keeps an early historical 1_5b label; MODEL_PATH is the source of truth.
# This script uses plain Hugging Face generation, not RL training.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
DATASET_NAME="${DATASET_NAME:-hiyouga/math12k}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
OUTPUT_FILE="${OUTPUT_FILE:-results/qwen2_5_1_5b_math_base_eval.jsonl}"
SUMMARY_FILE="${SUMMARY_FILE:-results/qwen2_5_1_5b_math_base_eval_report.json}"
FORMAT_PROMPT="${FORMAT_PROMPT:-examples/format_prompt/math.jinja}"
REWARD_FILE="${REWARD_FILE:-examples/reward_function/math.py}"

LIMIT="${LIMIT:-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-false}"

mkdir -p "$(dirname "${OUTPUT_FILE}")" "$(dirname "${SUMMARY_FILE}")"

python3 scripts/eval_math_base.py \
    --model-path "${MODEL_PATH}" \
    --dataset-name "${DATASET_NAME}" \
    --dataset-split "${DATASET_SPLIT}" \
    --output-file "${OUTPUT_FILE}" \
    --summary-file "${SUMMARY_FILE}" \
    --format-prompt "${FORMAT_PROMPT}" \
    --reward-file "${REWARD_FILE}" \
    --limit "${LIMIT}" \
    --batch-size "${BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --trust-remote-code "${TRUST_REMOTE_CODE}"
