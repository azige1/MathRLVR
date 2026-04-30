#!/usr/bin/env bash
set -euo pipefail

# Run from the EasyR1 repository root inside the EasyR1 Docker container.
# B -> A upgrade:
#   B: verifier-aware clean data GRPO
#   A: verifier-aware clean data + concise protocol GRPO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="${EASYR1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${EASYR1_ROOT}"
mkdir -p logs results data

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-easyr1-reasoning}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1}"
SINGLE_GPU_ID="${SINGLE_GPU_ID:-0}"
CLEAN_DATA_DIR="${CLEAN_DATA_DIR:-data/math12k_verifier_clean}"
MAX_STEPS="${MAX_STEPS:-150}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
N_GENERATIONS="${N_GENERATIONS:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.40}"
POLL_SECONDS="${POLL_SECONDS:-60}"
FORCE_REBUILD_CLEAN="${FORCE_REBUILD_CLEAN:-false}"
FORCE_RERUN="${FORCE_RERUN:-false}"
RUN_CLEAN_DATA="${RUN_CLEAN_DATA:-true}"
RUN_CLEAN_CONCISE="${RUN_CLEAN_CONCISE:-true}"
RUN_CLEAN_EVAL="${RUN_CLEAN_EVAL:-false}"

active_training_count() {
  pgrep -f "verl.trainer.main" >/dev/null && pgrep -fc "verl.trainer.main" || true
}

checkpoint_done() {
  local ckpt_dir="$1"
  local expected_step="$2"
  if [ ! -f "${ckpt_dir}/checkpoint_tracker.json" ]; then
    return 1
  fi
  python - "$ckpt_dir" "$expected_step" <<'PY'
import json
import sys
from pathlib import Path

tracker = Path(sys.argv[1]) / "checkpoint_tracker.json"
expected = int(sys.argv[2])
data = json.loads(tracker.read_text())
last = int(data.get("last_global_step") or data.get("best_global_step") or 0)
raise SystemExit(0 if last >= expected else 1)
PY
}

wait_for_checkpoint() {
  local exp_name="$1"
  local expected_step="$2"
  local log_file="$3"
  local ckpt_dir="checkpoints/easyr1_reasoning/${exp_name}"

  echo "[WAIT] ${exp_name}, expected last_global_step >= ${expected_step}"
  while ! checkpoint_done "${ckpt_dir}" "${expected_step}"; do
    sleep "${POLL_SECONDS}"
    if ! pgrep -f "verl.trainer.main" >/dev/null; then
      echo "[ERROR] No verl.trainer.main process is running, but ${exp_name} is not complete."
      echo "[ERROR] Log tail:"
      tail -n 120 "${log_file}" || true
      exit 1
    fi
    tail -n 5 "${log_file}" || true
  done
  echo "[DONE] ${exp_name}"
}

build_clean_dataset_if_needed() {
  if [ "${FORCE_REBUILD_CLEAN}" != "true" ] && [ -f "${CLEAN_DATA_DIR}/train.jsonl" ] && [ -f "${CLEAN_DATA_DIR}/test.jsonl" ]; then
    echo "[SKIP] clean dataset exists: ${CLEAN_DATA_DIR}"
    return
  fi

  echo "== Build verifier-aware clean dataset =="
  python scripts/build_rlvr_clean_dataset.py \
    --dataset-name hiyouga/math12k \
    --train-split train \
    --val-split test \
    --format-prompt examples/format_prompt/math_concise_final.jinja \
    --tokenizer-path "${MODEL_PATH}" \
    --max-prompt-length 1024 \
    --policy strict_clean \
    --output-dir "${CLEAN_DATA_DIR}" \
    --report-file results/math12k_verifier_clean_dataset_report.json \
    --manifest-file results/math12k_verifier_clean_manifest.jsonl \
    | tee logs/build_rlvr_clean_dataset.log
}

run_suite_if_needed() {
  local suite_mode="$1"
  local exp_name="$2"
  local log_file="logs/${exp_name}.log"

  if [ "${FORCE_RERUN}" != "true" ] && checkpoint_done "checkpoints/easyr1_reasoning/${exp_name}" "${MAX_STEPS}"; then
    echo "[SKIP] ${exp_name} already complete."
    return
  fi

  echo "[RUN] SUITE_MODE=${suite_mode}, EXP=${exp_name}, EXPECTED_STEP=${MAX_STEPS}"
  MODEL_PATH="${MODEL_PATH}" \
  GPU_IDS="${GPU_IDS}" \
  SINGLE_GPU_ID="${SINGLE_GPU_ID}" \
  CLEAN_DATA_DIR="${CLEAN_DATA_DIR}" \
  MAX_STEPS="${MAX_STEPS}" \
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
  N_GENERATIONS="${N_GENERATIONS}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  VAL_FREQ=50 \
  SAVE_FREQ=50 \
  SUITE_MODE="${suite_mode}" \
  bash examples/run_reasoning_rl_suite_a10.sh

  wait_for_checkpoint "${exp_name}" "${MAX_STEPS}" "${log_file}"
}

if [ "$(active_training_count)" != "" ]; then
  echo "[ERROR] Existing verl.trainer.main process detected. Stop it before running this sequential pipeline."
  ps -ef | grep -E "verl.trainer.main|ray" | grep -v grep || true
  exit 1
fi

build_clean_dataset_if_needed

if [ "${RUN_CLEAN_DATA}" = "true" ]; then
  echo "== B. Verifier-clean data GRPO =="
  run_suite_if_needed clean_data qwen2_5_3b_math_grpo_clean_data_a10
fi

if [ "${RUN_CLEAN_CONCISE}" = "true" ]; then
  echo "== A. Verifier-clean data + concise protocol GRPO =="
  run_suite_if_needed clean_concise qwen2_5_3b_math_grpo_clean_concise_a10
fi

echo "== Summarize updated runs =="
python scripts/summarize_rlvr_runs.py \
  --output-json results/checkpoint_stability_report_with_clean_protocol.json \
  --output-text results/reasoning_rl_all_runs_compare_with_clean_protocol.txt \
  | tee logs/summarize_rlvr_runs_with_clean_protocol.log

if [ "${RUN_CLEAN_EVAL}" = "true" ]; then
  echo "== Run clean protocol eval =="
  MODEL_PATH="${MODEL_PATH}" \
  SINGLE_GPU_ID="${SINGLE_GPU_ID}" \
  bash scripts/run_clean_protocol_eval_a10.sh \
    | tee logs/run_clean_protocol_eval_a10.log
fi

echo "Clean data -> concise protocol GRPO pipeline finished."
