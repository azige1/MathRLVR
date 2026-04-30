# EasyR1 Math RLVR A100 接手机划

## 当前状态

现有服务器 `/home/wzy/migrate/EasyR1` 已完成一轮 A10 项目结果，核心结论可以支撑项目叙事：

```text
Base Math12K greedy: 0.426
GRPO-composite Math12K greedy: 0.458
GSPO Math12K greedy: 0.426
GRPO-answer-only Math12K greedy: 0.416

Base Math12K best-of-8: 0.602
GRPO-composite Math12K best-of-8: 0.636

Base GSM8K greedy: 0.790
GRPO-composite GSM8K greedy: 0.768
```

已完成的训练 checkpoint：

```text
qwen2_5_1_5b_math_grpo_a10: step300, best step150
qwen2_5_1_5b_math_gspo_a10: step150
qwen2_5_1_5b_math_grpo_filtered_a10: step150
qwen2_5_1_5b_math_grpo_answer_only_a10: step150
qwen2_5_3b_math_grpo_clean_data_a10: step150
```

注意：部分历史文件名包含 `1_5b`，但实际模型路径是 `Qwen2.5-3B-Instruct`。最终报告中应统一称为 `Qwen2.5-3B-Instruct`。

## 当前缺口

最重要的缺口是：

```text
clean data + concise final-answer protocol 的完整 GRPO 训练尚未完成。
```

已有：

```text
clean_data GRPO step150
concise prompt + stop postprocess 的 200 条评估
```

缺少：

```text
qwen2_5_3b_math_grpo_clean_concise_a100 step150
Math12K greedy / best-of-8 / GSM8K greedy / robust diagnostic
```

这个实验是项目闭环的关键：从 failure analysis 发现 overlong reasoning、boxed 缺失、prompt contamination 和 verifier false negative，再用 clean data 与 concise protocol 做改进验证。

## A100 运行目标

硬件假设：

```text
1x A100 40GB
Qwen2.5-3B-Instruct
LoRA rank 16
Math12K
GRPO
```

默认训练配置：

```text
MAX_STEPS=150
N_GENERATIONS=4
ROLLOUT_BATCH_SIZE=32
GLOBAL_BATCH_SIZE=16
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=512
GPU_MEMORY_UTILIZATION=0.60
```

如果显存或速度不理想，优先降低：

```text
N_GENERATIONS=2
ROLLOUT_BATCH_SIZE=16
GPU_MEMORY_UTILIZATION=0.50
```

如果显存稳定且速度可接受，再考虑：

```text
ROLLOUT_BATCH_SIZE=64
GLOBAL_BATCH_SIZE=32
```

## 一键脚本

新增脚本：

```text
scripts/run_a100_clean_concise_project.sh
```

它会执行：

```text
1. 构建 verifier-aware clean dataset
2. 训练 clean-data + concise-protocol GRPO
3. 评估 Math12K greedy
4. 评估 Math12K best-of-8
5. 评估 GSM8K greedy
6. 汇总 best-of-N / cross-benchmark 报告
7. 生成 V2-lite robust diagnostic
8. 生成 checkpoint stability summary
```

推荐命令：

```bash
cd /path/to/EasyR1

MODEL_PATH=/path/to/Qwen2.5-3B-Instruct \
GPU_ID=0 \
MAX_STEPS=150 \
N_GENERATIONS=4 \
ROLLOUT_BATCH_SIZE=32 \
GLOBAL_BATCH_SIZE=16 \
MAX_RESPONSE_LENGTH=512 \
GPU_MEMORY_UTILIZATION=0.60 \
bash scripts/run_a100_clean_concise_project.sh
```

只训练不评估：

```bash
RUN_EVAL=false bash scripts/run_a100_clean_concise_project.sh
```

只评估已有 checkpoint：

```bash
RUN_TRAIN=false bash scripts/run_a100_clean_concise_project.sh
```

强制重跑：

```bash
FORCE_RERUN=true bash scripts/run_a100_clean_concise_project.sh
```

## 验收指标

训练完成后至少应存在：

```text
checkpoints/easyr1_reasoning/qwen2_5_3b_math_grpo_clean_concise_a100/checkpoint_tracker.json
results/grpo_clean_concise_a100_math12k_greedy_report.json
results/grpo_clean_concise_a100_math12k_best_of_8_report.json
results/grpo_clean_concise_a100_gsm8k_greedy_report.json
results/reward_v2_rescoring_report_a100.json
results/checkpoint_stability_report_a100.json
```

重点比较：

```text
GRPO-composite old
GRPO-clean-data
GRPO-clean-concise-a100
```

指标：

```text
Math12K greedy accuracy
Math12K best-of-8 accuracy
GSM8K greedy accuracy
mean_response_length
format_at_k / format_rate
hit_length_cap_rate
missing_boxed_answer_rate
prompt_contamination_rate
verifier_false_negative_candidate_rate
```

## 报告口径

推荐最终项目表述：

```text
基于 EasyR1/veRL，在 Qwen2.5-3B + LoRA + 小算力条件下，
系统分析数学 RLVR 的收益来源、checkpoint 稳定性、verifier 噪声和输出截断问题，
并通过 verifier-aware clean data 与 concise final-answer protocol 做闭环改进。
```

避免表述：

```text
复现 DeepSeek-R1
训练出 SOTA 数学模型
证明 GRPO 普遍优于 GSPO
```
