#!/usr/bin/env bash
set -euo pipefail

# Single large-GPU handoff runner for the EasyR1 Math RLVR project.
# The filename keeps the original A100 handoff label; it also applies to A800.
# Goal: fill the current project gap by running the clean-data + concise-protocol
# GRPO experiment, then evaluate it with the same Math12K/GSM8K diagnostics.
#
# Run inside the EasyR1 Docker/container environment:
#
#   cd /path/to/EasyR1
#   MODEL_PATH=/path/to/Qwen2.5-3B-Instruct \
#   GPU_ID=0 \
#   bash scripts/run_a100_clean_concise_project.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="${EASYR1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${EASYR1_ROOT}"

mkdir -p logs results data

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-easyr1-reasoning}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
GPU_ID="${GPU_ID:-0}"
SINGLE_GPU_ID="${SINGLE_GPU_ID:-${GPU_ID}}"
CLEAN_DATA_DIR="${CLEAN_DATA_DIR:-data/math12k_verifier_clean}"

PROJECT_NAME="${PROJECT_NAME:-easyr1_reasoning}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2_5_3b_math_grpo_clean_concise_a100}"
LOGGER="${LOGGER:-[file,wandb]}"

MAX_STEPS="${MAX_STEPS:-150}"
VAL_FREQ="${VAL_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
SAVE_LIMIT="${SAVE_LIMIT:-3}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
N_GENERATIONS="${N_GENERATIONS:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.60}"

LR="${LR:-5e-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.1}"

RUN_TRAIN="${RUN_TRAIN:-true}"
RUN_EVAL="${RUN_EVAL:-true}"
RUN_SUMMARY="${RUN_SUMMARY:-true}"
FORCE_REBUILD_CLEAN="${FORCE_REBUILD_CLEAN:-false}"
FORCE_RERUN="${FORCE_RERUN:-false}"

MATH_LIMIT="${MATH_LIMIT:-500}"
GSM8K_LIMIT="${GSM8K_LIMIT:-500}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
BEST_OF_N="${BEST_OF_N:-8}"
PROMPT_FILE="${PROMPT_FILE:-examples/format_prompt/math_concise_final.jinja}"

checkpoint_done() {
  local ckpt_dir="$1"
  local expected_step="$2"
  if [ ! -f "${ckpt_dir}/checkpoint_tracker.json" ]; then
    return 1
  fi
  python3 - "$ckpt_dir" "$expected_step" <<'PY'
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

resolve_lora_path() {
  local exp_name="$1"
  python3 - "$exp_name" <<'PY'
import json
import sys
from pathlib import Path

exp = sys.argv[1]
base = Path("checkpoints/easyr1_reasoning") / exp
tracker = base / "checkpoint_tracker.json"
if not tracker.exists():
    raise SystemExit(f"Missing checkpoint tracker: {tracker}")
data = json.loads(tracker.read_text())
step = int(data.get("best_global_step") or data.get("last_global_step") or 0)
path = base / f"global_step_{step}" / "actor" / "lora_adapter"
if not path.exists():
    raise SystemExit(f"Missing LoRA adapter: {path}")
print(path)
PY
}

run_if_missing() {
  local report="$1"
  shift
  if [ "${FORCE_RERUN}" != "true" ] && [ -f "${report}" ]; then
    echo "[SKIP] ${report} exists"
  else
    echo "[RUN] $*"
    "$@"
  fi
}

build_clean_dataset_if_needed() {
  if [ "${FORCE_REBUILD_CLEAN}" != "true" ] && [ -f "${CLEAN_DATA_DIR}/train.jsonl" ]; then
    echo "[SKIP] clean dataset exists: ${CLEAN_DATA_DIR}"
    return
  fi

  echo "== Build verifier-aware clean dataset =="
  python3 scripts/build_rlvr_clean_dataset.py \
    --dataset-name hiyouga/math12k \
    --train-split train \
    --val-split test \
    --format-prompt "${PROMPT_FILE}" \
    --tokenizer-path "${MODEL_PATH}" \
    --max-prompt-length "${MAX_PROMPT_LENGTH}" \
    --policy strict_clean \
    --output-dir "${CLEAN_DATA_DIR}" \
    --report-file results/math12k_verifier_clean_dataset_report.json \
    --manifest-file results/math12k_verifier_clean_manifest.jsonl \
    | tee logs/build_rlvr_clean_dataset_a100.log
}

run_clean_concise_training() {
  local ckpt_dir="checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
  if [ "${FORCE_RERUN}" != "true" ] && checkpoint_done "${ckpt_dir}" "${MAX_STEPS}"; then
    echo "[SKIP] ${EXPERIMENT_NAME} already reached step ${MAX_STEPS}"
    return
  fi

  echo "== Train clean-data + concise-protocol GRPO on A100 =="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  MODEL_PATH="${MODEL_PATH}" \
  TRAIN_FILES="${CLEAN_DATA_DIR}@train" \
  VAL_FILES="hiyouga/math12k@test" \
  FORMAT_PROMPT="${PROMPT_FILE}" \
  PROJECT_NAME="${PROJECT_NAME}" \
  EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  LOGGER="${LOGGER}" \
  N_GPUS=1 \
  MAX_STEPS="${MAX_STEPS}" \
  VAL_FREQ="${VAL_FREQ}" \
  SAVE_FREQ="${SAVE_FREQ}" \
  SAVE_LIMIT="${SAVE_LIMIT}" \
  VAL_BEFORE_TRAIN=false \
  ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE}" \
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE}" \
  N_GENERATIONS="${N_GENERATIONS}" \
  MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
  MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
  TENSOR_PARALLEL_SIZE=1 \
  LR="${LR}" \
  WARMUP_RATIO="${WARMUP_RATIO}" \
  LORA_RANK="${LORA_RANK}" \
  LORA_ALPHA="${LORA_ALPHA}" \
  FORMAT_WEIGHT="${FORMAT_WEIGHT}" \
  bash examples/qwen2_5_1_5b_math_grpo_a10.sh \
    2>&1 | tee "logs/${EXPERIMENT_NAME}.log"
}

eval_math12k_greedy() {
  local name="$1"
  local lora="$2"
  run_if_missing "results/${name}_math12k_greedy_report.json" \
    env CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" python3 scripts/eval_math_best_of_n.py \
      --model-path "${MODEL_PATH}" \
      --lora-path "${lora}" \
      --dataset-name hiyouga/math12k \
      --dataset-split test \
      --prompt-key problem \
      --answer-key answer \
      --format-prompt "${PROMPT_FILE}" \
      --output-file "results/${name}_math12k_greedy.jsonl" \
      --report-file "results/${name}_math12k_greedy_report.json" \
      --limit "${MATH_LIMIT}" \
      --batch-size "${EVAL_BATCH_SIZE}" \
      --n 1 \
      --temperature 0.0 \
      --top-p 1.0 \
      --max-new-tokens "${MAX_RESPONSE_LENGTH}" \
      --postprocess-stop-markers \
      --stop-after-boxed
}

eval_math12k_best_of_n() {
  local name="$1"
  local lora="$2"
  run_if_missing "results/${name}_math12k_best_of_${BEST_OF_N}_report.json" \
    env CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" python3 scripts/eval_math_best_of_n.py \
      --model-path "${MODEL_PATH}" \
      --lora-path "${lora}" \
      --dataset-name hiyouga/math12k \
      --dataset-split test \
      --prompt-key problem \
      --answer-key answer \
      --format-prompt "${PROMPT_FILE}" \
      --output-file "results/${name}_math12k_best_of_${BEST_OF_N}.jsonl" \
      --report-file "results/${name}_math12k_best_of_${BEST_OF_N}_report.json" \
      --limit "${MATH_LIMIT}" \
      --batch-size "${EVAL_BATCH_SIZE}" \
      --n "${BEST_OF_N}" \
      --temperature 0.7 \
      --top-p 0.95 \
      --max-new-tokens "${MAX_RESPONSE_LENGTH}" \
      --postprocess-stop-markers \
      --stop-after-boxed
}

eval_gsm8k_greedy() {
  local name="$1"
  local lora="$2"
  run_if_missing "results/${name}_gsm8k_greedy_report.json" \
    env CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" python3 scripts/eval_math_best_of_n.py \
      --model-path "${MODEL_PATH}" \
      --lora-path "${lora}" \
      --dataset-name openai/gsm8k \
      --dataset-config main \
      --dataset-split test \
      --prompt-key question \
      --answer-key answer \
      --answer-extractor gsm8k \
      --format-prompt "${PROMPT_FILE}" \
      --output-file "results/${name}_gsm8k_greedy.jsonl" \
      --report-file "results/${name}_gsm8k_greedy_report.json" \
      --limit "${GSM8K_LIMIT}" \
      --batch-size "${EVAL_BATCH_SIZE}" \
      --n 1 \
      --temperature 0.0 \
      --top-p 1.0 \
      --max-new-tokens "${MAX_RESPONSE_LENGTH}" \
      --postprocess-stop-markers \
      --stop-after-boxed
}

run_clean_concise_eval() {
  local lora_path
  lora_path="$(resolve_lora_path "${EXPERIMENT_NAME}")"
  local run_name="${RUN_NAME:-grpo_clean_concise_a100}"

  echo "== Evaluate ${run_name} =="
  echo "LoRA: ${lora_path}"
  eval_math12k_greedy "${run_name}" "${lora_path}"
  eval_math12k_best_of_n "${run_name}" "${lora_path}"
  eval_gsm8k_greedy "${run_name}" "${lora_path}"

  echo "== Collect reports and robust diagnostics =="
  python3 scripts/collect_rlvr_eval_reports.py \
    --best-of-n-output results/best_of_n_eval_report_a100.json \
    --cross-benchmark-output results/cross_benchmark_eval_report_a100.json \
    | tee logs/collect_rlvr_eval_reports_a100.log

  python3 scripts/rescore_rlvr_outputs_v2_lite.py \
    "${run_name}=results/${run_name}_math12k_greedy.jsonl" \
    --output-jsonl results/reward_v2_rescored_outputs_a100.jsonl \
    --report-file results/reward_v2_rescoring_report_a100.json \
    | tee logs/reward_v2_rescoring_a100.log
}

echo "== A100 project config =="
echo "EASYR1_ROOT=${EASYR1_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "GPU_ID=${GPU_ID}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}"
echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "N_GENERATIONS=${N_GENERATIONS}"
echo "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"

build_clean_dataset_if_needed

if [ "${RUN_TRAIN}" = "true" ]; then
  run_clean_concise_training
fi

if [ "${RUN_EVAL}" = "true" ]; then
  run_clean_concise_eval
fi

if [ "${RUN_SUMMARY}" = "true" ]; then
  python3 scripts/summarize_rlvr_runs.py \
    --run "grpo_clean_concise_a100=checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/experiment_log.jsonl" \
    --output-json results/checkpoint_stability_report_a100.json \
    --output-text results/reasoning_rl_all_runs_compare_a100.txt \
    | tee logs/summarize_rlvr_runs_a100.log
fi

echo "A100 clean-concise project run finished."
