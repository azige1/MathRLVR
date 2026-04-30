#!/usr/bin/env bash
set -euo pipefail

# Evaluate B/A clean protocol runs after training completes.
# Runs:
#   - Math12K greedy
#   - Math12K best-of-8
#   - GSM8K greedy
#   - V2-lite / robust diagnostics on Math12K greedy outputs

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
SINGLE_GPU_ID="${SINGLE_GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MATH_LIMIT="${MATH_LIMIT:-500}"
GSM8K_LIMIT="${GSM8K_LIMIT:-300}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BEST_OF_N="${BEST_OF_N:-8}"
PROMPT_FILE="${PROMPT_FILE:-examples/format_prompt/math_concise_final.jinja}"
USE_STOP_POSTPROCESS="${USE_STOP_POSTPROCESS:-1}"

CLEAN_DATA_EXP="${CLEAN_DATA_EXP:-qwen2_5_3b_math_grpo_clean_data_a10}"
CLEAN_CONCISE_EXP="${CLEAN_CONCISE_EXP:-qwen2_5_3b_math_grpo_clean_concise_a10}"

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

resolve_lora_path() {
  local exp_name="$1"
  python - "$exp_name" <<'PY'
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

postprocess_args=()
if [ "${USE_STOP_POSTPROCESS}" = "1" ]; then
  postprocess_args=(--postprocess-stop-markers --stop-after-boxed)
fi

CLEAN_DATA_LORA="${CLEAN_DATA_LORA:-$(resolve_lora_path "${CLEAN_DATA_EXP}")}"
CLEAN_CONCISE_LORA="${CLEAN_CONCISE_LORA:-$(resolve_lora_path "${CLEAN_CONCISE_EXP}")}"

eval_math12k_greedy() {
  local name="$1"
  local lora="$2"
  run_if_missing "results/${name}_math12k_greedy_report.json" \
    env CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" python scripts/eval_math_best_of_n.py \
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
      --batch-size "${BATCH_SIZE}" \
      --n 1 \
      --temperature 0.0 \
      --top-p 1.0 \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      "${postprocess_args[@]}"
}

eval_math12k_best_of_n() {
  local name="$1"
  local lora="$2"
  run_if_missing "results/${name}_math12k_best_of_${BEST_OF_N}_report.json" \
    env CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" python scripts/eval_math_best_of_n.py \
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
      --batch-size "${BATCH_SIZE}" \
      --n "${BEST_OF_N}" \
      --temperature 0.7 \
      --top-p 0.95 \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      "${postprocess_args[@]}"
}

eval_gsm8k_greedy() {
  local name="$1"
  local lora="$2"
  run_if_missing "results/${name}_gsm8k_greedy_report.json" \
    env CUDA_VISIBLE_DEVICES="${SINGLE_GPU_ID}" python scripts/eval_math_best_of_n.py \
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
      --batch-size "${BATCH_SIZE}" \
      --n 1 \
      --temperature 0.0 \
      --top-p 1.0 \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      "${postprocess_args[@]}"
}

echo "== Clean protocol eval config =="
echo "MODEL_PATH=${MODEL_PATH}"
echo "CLEAN_DATA_LORA=${CLEAN_DATA_LORA}"
echo "CLEAN_CONCISE_LORA=${CLEAN_CONCISE_LORA}"
echo "MATH_LIMIT=${MATH_LIMIT}"
echo "GSM8K_LIMIT=${GSM8K_LIMIT}"
echo "BEST_OF_N=${BEST_OF_N}"
echo "PROMPT_FILE=${PROMPT_FILE}"
echo "USE_STOP_POSTPROCESS=${USE_STOP_POSTPROCESS}"

echo "== 1. Math12K greedy =="
eval_math12k_greedy grpo_clean_data "${CLEAN_DATA_LORA}"
eval_math12k_greedy grpo_clean_concise "${CLEAN_CONCISE_LORA}"

echo "== 2. Math12K best-of-${BEST_OF_N} =="
eval_math12k_best_of_n grpo_clean_data "${CLEAN_DATA_LORA}"
eval_math12k_best_of_n grpo_clean_concise "${CLEAN_CONCISE_LORA}"

echo "== 3. GSM8K greedy =="
eval_gsm8k_greedy grpo_clean_data "${CLEAN_DATA_LORA}"
eval_gsm8k_greedy grpo_clean_concise "${CLEAN_CONCISE_LORA}"

echo "== 4. Collect reports =="
python scripts/collect_rlvr_eval_reports.py \
  --best-of-n-output results/best_of_n_eval_report_with_clean_protocol.json \
  --cross-benchmark-output results/cross_benchmark_eval_report_with_clean_protocol.json \
  | tee logs/collect_rlvr_eval_reports_with_clean_protocol.log

echo "== 5. V2-lite / robust diagnostics =="
python scripts/rescore_rlvr_outputs_v2_lite.py \
  grpo_clean_data=results/grpo_clean_data_math12k_greedy.jsonl \
  grpo_clean_concise=results/grpo_clean_concise_math12k_greedy.jsonl \
  --output-jsonl results/reward_v2_rescored_outputs_clean_protocol.jsonl \
  --report-file results/reward_v2_rescoring_report_clean_protocol.json \
  | tee logs/reward_v2_rescoring_clean_protocol.log

echo "Clean protocol eval finished."
