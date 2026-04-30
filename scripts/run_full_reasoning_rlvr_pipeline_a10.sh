#!/usr/bin/env bash
set -euo pipefail

# Run from the EasyR1 repository root inside the EasyR1 Docker container.
# This script skips completed runs, launches missing P0 training runs, waits for
# completion, then runs the P0 analysis/evaluation suite.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="${EASYR1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${EASYR1_ROOT}"
mkdir -p logs results

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_PROJECT="${WANDB_PROJECT:-easyr1-reasoning}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1}"
SINGLE_GPU_ID="${SINGLE_GPU_ID:-0}"

RUN_TRAINING="${RUN_TRAINING:-true}"
RUN_P0_ANALYSIS="${RUN_P0_ANALYSIS:-true}"
FORCE_RERUN="${FORCE_RERUN:-false}"
ALLOW_CONCURRENT="${ALLOW_CONCURRENT:-false}"

GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-300}"
GSPO_MAX_STEPS="${GSPO_MAX_STEPS:-150}"
FILTERED_MAX_STEPS="${FILTERED_MAX_STEPS:-150}"
ANSWER_ONLY_MAX_STEPS="${ANSWER_ONLY_MAX_STEPS:-150}"
FORMAT_ONLY_MAX_STEPS="${FORMAT_ONLY_MAX_STEPS:-150}"

POLL_SECONDS="${POLL_SECONDS:-60}"

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

run_suite_if_needed() {
  local suite_mode="$1"
  local exp_name="$2"
  local expected_step="$3"
  local log_file="logs/${exp_name}.log"

  if [ "${FORCE_RERUN}" != "true" ] && checkpoint_done "checkpoints/easyr1_reasoning/${exp_name}" "${expected_step}"; then
    echo "[SKIP] ${exp_name} already complete."
    return
  fi

  echo "[RUN] SUITE_MODE=${suite_mode}, EXP=${exp_name}, EXPECTED_STEP=${expected_step}"
  MODEL_PATH="${MODEL_PATH}" \
  GPU_IDS="${GPU_IDS}" \
  SINGLE_GPU_ID="${SINGLE_GPU_ID}" \
  MAX_STEPS="${expected_step}" \
  VAL_FREQ=50 \
  SAVE_FREQ=50 \
  SUITE_MODE="${suite_mode}" \
  bash examples/run_reasoning_rl_suite_a10.sh

  wait_for_checkpoint "${exp_name}" "${expected_step}" "${log_file}"
}

run_base_if_needed() {
  if [ "${FORCE_RERUN}" != "true" ] && {
    [ -f results/qwen2_5_3b_math_base_eval_report.json ] || [ -f results/qwen2_5_1_5b_math_base_eval_report.json ];
  }; then
    echo "[SKIP] base eval already exists."
    return
  fi

  echo "[RUN] base eval"
  MODEL_PATH="${MODEL_PATH}" \
  SINGLE_GPU_ID="${SINGLE_GPU_ID}" \
  SUITE_MODE=base \
  bash examples/run_reasoning_rl_suite_a10.sh
}

resolve_lora_path() {
  local exp_name="$1"
  local fallback_step="$2"
  python - "$exp_name" "$fallback_step" <<'PY'
import json
import sys
from pathlib import Path

exp = sys.argv[1]
fallback = int(sys.argv[2])
base = Path("checkpoints/easyr1_reasoning") / exp
tracker = base / "checkpoint_tracker.json"
step = fallback
if tracker.exists():
    data = json.loads(tracker.read_text())
    step = int(data.get("best_global_step") or data.get("last_global_step") or fallback)
print(str(base / f"global_step_{step}" / "actor" / "lora_adapter"))
PY
}

if [ "${ALLOW_CONCURRENT}" != "true" ] && [ "$(active_training_count)" != "" ]; then
  echo "[ERROR] Existing verl.trainer.main process detected. Stop it or set ALLOW_CONCURRENT=true."
  ps -ef | grep -E "verl.trainer.main|ray" | grep -v grep || true
  exit 1
fi

if [ "${RUN_TRAINING}" = "true" ]; then
  echo "== Training / skip-completed phase =="
  run_base_if_needed
  run_suite_if_needed grpo qwen2_5_1_5b_math_grpo_a10 "${GRPO_MAX_STEPS}"
  run_suite_if_needed gspo qwen2_5_1_5b_math_gspo_a10 "${GSPO_MAX_STEPS}"
  run_suite_if_needed grpo_filtered qwen2_5_1_5b_math_grpo_filtered_a10 "${FILTERED_MAX_STEPS}"
  run_suite_if_needed answer_only qwen2_5_1_5b_math_grpo_answer_only_a10 "${ANSWER_ONLY_MAX_STEPS}"
  run_suite_if_needed format_only qwen2_5_1_5b_math_grpo_format_only_a10 "${FORMAT_ONLY_MAX_STEPS}"
fi

if [ "${RUN_P0_ANALYSIS}" = "true" ]; then
  echo "== P0 analysis phase =="
  export MODEL_PATH
  export SINGLE_GPU_ID
  export GRPO_LORA="${GRPO_LORA:-$(resolve_lora_path qwen2_5_1_5b_math_grpo_a10 150)}"
  export GSPO_LORA="${GSPO_LORA:-$(resolve_lora_path qwen2_5_1_5b_math_gspo_a10 150)}"
  export ANSWER_ONLY_LORA="${ANSWER_ONLY_LORA:-$(resolve_lora_path qwen2_5_1_5b_math_grpo_answer_only_a10 150)}"
  bash scripts/run_p0_reasoning_analysis_a10.sh
fi

echo "Full EasyR1 Math RLVR pipeline finished."
