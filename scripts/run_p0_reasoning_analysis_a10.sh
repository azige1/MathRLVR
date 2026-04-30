#!/usr/bin/env bash
set -euo pipefail

# Run from the EasyR1 repository root inside the EasyR1 Docker container.
# This script performs P0 analysis/evaluation after the RL training runs finish.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EASYR1_ROOT="${EASYR1_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${EASYR1_ROOT}"
mkdir -p logs results

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

ENABLE_WANDB_MONITOR="${ENABLE_WANDB_MONITOR:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-easyr1-reasoning}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MONITOR_RUN_NAME="${WANDB_MONITOR_RUN_NAME:-p0_reasoning_analysis_monitor}"
WANDB_MONITOR_POLL_SECONDS="${WANDB_MONITOR_POLL_SECONDS:-60}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
MATH_LIMIT="${MATH_LIMIT:-500}"
GSM8K_LIMIT="${GSM8K_LIMIT:-500}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SINGLE_GPU_ID="${SINGLE_GPU_ID:-0}"

GRPO_LORA="${GRPO_LORA:-checkpoints/easyr1_reasoning/qwen2_5_1_5b_math_grpo_a10/global_step_150/actor/lora_adapter}"
GSPO_LORA="${GSPO_LORA:-checkpoints/easyr1_reasoning/qwen2_5_1_5b_math_gspo_a10/global_step_150/actor/lora_adapter}"
ANSWER_ONLY_LORA="${ANSWER_ONLY_LORA:-checkpoints/easyr1_reasoning/qwen2_5_1_5b_math_grpo_answer_only_a10/global_step_150/actor/lora_adapter}"

start_wandb_monitor() {
  if [[ "${ENABLE_WANDB_MONITOR}" != "1" ]]; then
    echo "[W&B] P0 monitor disabled: ENABLE_WANDB_MONITOR=${ENABLE_WANDB_MONITOR}"
    return
  fi
  if ! python - <<'PY' >/dev/null 2>&1
import wandb
PY
  then
    echo "[W&B] wandb is not importable; skip P0 monitor."
    return
  fi

  monitor_args=(
    --project "${WANDB_PROJECT}"
    --run-name "${WANDB_MONITOR_RUN_NAME}"
    --log-file logs/run_p0_reasoning_analysis_a10.log
    --poll-seconds "${WANDB_MONITOR_POLL_SECONDS}"
    --finish-when-complete
  )
  if [[ -n "${WANDB_ENTITY}" ]]; then
    monitor_args+=(--entity "${WANDB_ENTITY}")
  fi

  nohup python scripts/watch_p0_wandb.py "${monitor_args[@]}" \
    > logs/watch_p0_wandb.log 2>&1 &
  echo "$!" > logs/watch_p0_wandb.pid
  echo "[W&B] Started P0 monitor pid=$(cat logs/watch_p0_wandb.pid), run=${WANDB_MONITOR_RUN_NAME}"
}

run_if_missing() {
  local report="$1"
  shift
  if [ -f "${report}" ]; then
    echo "[SKIP] ${report} exists"
  else
    echo "[RUN] $*"
    "$@"
  fi
}

start_wandb_monitor

echo "== 1. Summarize RL runs and checkpoint stability =="
python scripts/summarize_rlvr_runs.py \
  --output-json results/checkpoint_stability_report.json \
  --output-text results/reasoning_rl_all_runs_compare.txt \
  | tee logs/summarize_rlvr_runs.log

echo "== 2. Data inspection with tokenizer =="
run_if_missing results/math12k_data_inspection_report_tokenized.json \
  python scripts/inspect_math12k_data.py \
    --dataset-name hiyouga/math12k \
    --train-split train \
    --val-split test \
    --format-prompt examples/format_prompt/math.jinja \
    --tokenizer-path "${MODEL_PATH}" \
    --max-prompt-length 1024 \
    --output-report results/math12k_data_inspection_report_tokenized.json \
    --sample-output results/math12k_data_samples_tokenized.jsonl

run_if_missing results/math12k_data_profile_report.json \
  python scripts/profile_rlvr_dataset.py \
    --dataset-name hiyouga/math12k \
    --train-split train \
    --val-split test \
    --format-prompt examples/format_prompt/math.jinja \
    --tokenizer-path "${MODEL_PATH}" \
    --max-prompt-length 1024 \
    --eval-limit "${MATH_LIMIT}" \
    --output-report results/math12k_data_profile_report.json \
    --manifest-output results/math12k_data_manifest.jsonl \
    --eval-subset-output results/math12k_eval_subset_manifest.jsonl \
    --sample-output results/math12k_stratified_data_samples.jsonl

eval_math12k_greedy() {
  local name="$1"
  local lora="$2"
  local lora_args=()
  if [ -n "${lora}" ]; then
    lora_args=(--lora-path "${lora}")
  fi
  run_if_missing "results/${name}_math12k_greedy_report.json" \
    env CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" python scripts/eval_math_best_of_n.py \
      --model-path "${MODEL_PATH}" \
      "${lora_args[@]}" \
      --dataset-name hiyouga/math12k \
      --dataset-split test \
      --prompt-key problem \
      --answer-key answer \
      --output-file "results/${name}_math12k_greedy.jsonl" \
      --report-file "results/${name}_math12k_greedy_report.json" \
      --limit "${MATH_LIMIT}" \
      --batch-size "${BATCH_SIZE}" \
      --n 1 \
      --temperature 0.0 \
      --top-p 1.0 \
      --max-new-tokens "${MAX_NEW_TOKENS}"
}

eval_best_of_n() {
  local name="$1"
  local lora="$2"
  local n="$3"
  local lora_args=()
  if [ -n "${lora}" ]; then
    lora_args=(--lora-path "${lora}")
  fi
  run_if_missing "results/${name}_math12k_best_of_${n}_report.json" \
    env CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" python scripts/eval_math_best_of_n.py \
      --model-path "${MODEL_PATH}" \
      "${lora_args[@]}" \
      --dataset-name hiyouga/math12k \
      --dataset-split test \
      --prompt-key problem \
      --answer-key answer \
      --output-file "results/${name}_math12k_best_of_${n}.jsonl" \
      --report-file "results/${name}_math12k_best_of_${n}_report.json" \
      --limit "${MATH_LIMIT}" \
      --batch-size "${BATCH_SIZE}" \
      --n "${n}" \
      --temperature 0.7 \
      --top-p 0.95 \
      --max-new-tokens "${MAX_NEW_TOKENS}"
}

eval_gsm8k_greedy() {
  local name="$1"
  local lora="$2"
  local lora_args=()
  if [ -n "${lora}" ]; then
    lora_args=(--lora-path "${lora}")
  fi
  run_if_missing "results/${name}_gsm8k_greedy_report.json" \
    env CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" python scripts/eval_math_best_of_n.py \
      --model-path "${MODEL_PATH}" \
      "${lora_args[@]}" \
      --dataset-name openai/gsm8k \
      --dataset-config main \
      --dataset-split test \
      --prompt-key question \
      --answer-key answer \
      --answer-extractor gsm8k \
      --output-file "results/${name}_gsm8k_greedy.jsonl" \
      --report-file "results/${name}_gsm8k_greedy_report.json" \
      --limit "${GSM8K_LIMIT}" \
      --batch-size "${BATCH_SIZE}" \
      --n 1 \
      --temperature 0.0 \
      --top-p 1.0 \
      --max-new-tokens "${MAX_NEW_TOKENS}"
}

echo "== 3. Math12K greedy eval for P0 models =="
eval_math12k_greedy base ""
eval_math12k_greedy grpo_composite "${GRPO_LORA}"
eval_math12k_greedy gspo "${GSPO_LORA}"
eval_math12k_greedy grpo_answer_only "${ANSWER_ONLY_LORA}"

echo "== 4. Math12K best-of-N eval =="
for n in 4 8; do
  eval_best_of_n base "" "${n}"
  eval_best_of_n grpo_composite "${GRPO_LORA}" "${n}"
  eval_best_of_n gspo "${GSPO_LORA}" "${n}"
  eval_best_of_n grpo_answer_only "${ANSWER_ONLY_LORA}" "${n}"
done

echo "== 5. GSM8K greedy cross-benchmark eval =="
eval_gsm8k_greedy base ""
eval_gsm8k_greedy grpo_composite "${GRPO_LORA}"
eval_gsm8k_greedy gspo "${GSPO_LORA}"
eval_gsm8k_greedy grpo_answer_only "${ANSWER_ONLY_LORA}"

echo "== 6. Collect best-of-N and cross-benchmark reports =="
python scripts/collect_rlvr_eval_reports.py \
  --best-of-n-output results/best_of_n_eval_report.json \
  --cross-benchmark-output results/cross_benchmark_eval_report.json \
  | tee logs/collect_rlvr_eval_reports.log

echo "== 7. V2-lite offline rescoring on Math12K greedy outputs =="
python scripts/rescore_rlvr_outputs_v2_lite.py \
  base=results/base_math12k_greedy.jsonl \
  grpo_composite=results/grpo_composite_math12k_greedy.jsonl \
  gspo=results/gspo_math12k_greedy.jsonl \
  grpo_answer_only=results/grpo_answer_only_math12k_greedy.jsonl \
  --output-jsonl results/reward_v2_rescored_outputs.jsonl \
  --report-file results/reward_v2_rescoring_report.json \
  | tee logs/reward_v2_rescoring.log

echo "== 8. Build verifier audit and error analysis samples =="
python scripts/build_verifier_audit_and_error_samples.py \
  base=results/base_math12k_greedy.jsonl \
  grpo_composite=results/grpo_composite_math12k_greedy.jsonl \
  gspo=results/gspo_math12k_greedy.jsonl \
  grpo_answer_only=results/grpo_answer_only_math12k_greedy.jsonl \
  --audit-output results/verifier_audit.jsonl \
  --error-output results/error_analysis_50.jsonl \
  | tee logs/build_verifier_audit_and_error_samples.log

echo "P0 analysis finished."
