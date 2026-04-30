#!/usr/bin/env bash

set -euo pipefail
set -x

# Run from the EasyR1 repository root.
# This is the low-risk 2xA10 text-only GSPO comparison for math12k.
# The filename keeps an early historical 1_5b label; MODEL_PATH is the source of truth.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_FILES="${TRAIN_FILES:-hiyouga/math12k@train}"
VAL_FILES="${VAL_FILES:-hiyouga/math12k@test}"

PROJECT_NAME="${PROJECT_NAME:-easyr1_reasoning}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen2_5_1_5b_math_gspo_a10}"
LOGGER="${LOGGER:-[file,wandb]}"

N_GPUS="${N_GPUS:-2}"
MAX_STEPS="${MAX_STEPS:-300}"
VAL_FREQ="${VAL_FREQ:-50}"
SAVE_FREQ="${SAVE_FREQ:-50}"
SAVE_LIMIT="${SAVE_LIMIT:-3}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MICRO_UPDATE_BATCH_SIZE="${MICRO_UPDATE_BATCH_SIZE:-1}"
MICRO_EXPERIENCE_BATCH_SIZE="${MICRO_EXPERIENCE_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"

N_GENERATIONS="${N_GENERATIONS:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"

LR="${LR:-5e-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.50}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
FORMAT_WEIGHT="${FORMAT_WEIGHT:-0.1}"
ONLINE_FILTERING="${ONLINE_FILTERING:-false}"
FILTER_LOW="${FILTER_LOW:-0.01}"
FILTER_HIGH="${FILTER_HIGH:-0.99}"

# EasyR1's official GSPO-style example uses sequence averaging, tiny ratio clips,
# and no reference KL. Keep those defaults for v1, then add ablations only if needed.
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-3e-4}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-4e-4}"
DISABLE_KL="${DISABLE_KL:-true}"

mkdir -p logs

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.rollout_batch_size="${ROLLOUT_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.format_prompt=./examples/format_prompt/math.jinja \
    algorithm.adv_estimator=grpo \
    algorithm.disable_kl="${DISABLE_KL}" \
    algorithm.use_kl_loss=false \
    algorithm.kl_coef=0.0 \
    algorithm.online_filtering="${ONLINE_FILTERING}" \
    algorithm.filter_key=overall \
    algorithm.filter_low="${FILTER_LOW}" \
    algorithm.filter_high="${FILTER_HIGH}" \
    worker.actor.global_batch_size="${GLOBAL_BATCH_SIZE}" \
    worker.actor.micro_batch_size_per_device_for_update="${MICRO_UPDATE_BATCH_SIZE}" \
    worker.actor.micro_batch_size_per_device_for_experience="${MICRO_EXPERIENCE_BATCH_SIZE}" \
    worker.actor.loss_type=gspo_token \
    worker.actor.loss_avg_mode=seq \
    worker.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    worker.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.trust_remote_code=false \
    worker.actor.model.enable_gradient_checkpointing=true \
    worker.actor.model.lora.rank="${LORA_RANK}" \
    worker.actor.model.lora.alpha="${LORA_ALPHA}" \
    worker.actor.model.lora.target_modules=all-linear \
    worker.actor.model.lora.exclude_modules=null \
    worker.actor.optim.lr="${LR}" \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.optim.lr_scheduler_type=cosine \
    worker.actor.optim.lr_warmup_ratio="${WARMUP_RATIO}" \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.fsdp.enable_full_shard=true \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.rollout.n="${N_GENERATIONS}" \
    worker.rollout.temperature=1.0 \
    worker.rollout.top_p=1.0 \
    worker.rollout.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    worker.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    worker.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    worker.rollout.val_override_config.temperature=0.6 \
    worker.rollout.val_override_config.top_p=0.95 \
    worker.rollout.val_override_config.n=1 \
    worker.reward.reward_function=./examples/reward_function/math.py:compute_score \
    worker.reward.reward_function_kwargs.format_weight="${FORMAT_WEIGHT}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="${LOGGER}" \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.max_steps="${MAX_STEPS}" \
    trainer.val_freq="${VAL_FREQ}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.val_generations_to_log=5 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.save_limit="${SAVE_LIMIT}" \
    trainer.find_last_checkpoint=true
