# EasyR1 Math RLVR: 小算力数学后训练机制分析

本项目基于 [EasyR1](https://github.com/hiyouga/EasyR1) / veRL 构建数学推理 RLVR 后训练实验链路，目标是在真实小算力约束下分析 RLVR 收益来源，而不是复现 DeepSeek-R1 或训练 SOTA 数学模型。

核心问题：

```text
在 2xA10 + Qwen2.5-3B-Instruct + LoRA 条件下，
数学 RLVR 的收益来自算法选择、reward 设计、format 约束、数据过滤、
采样策略，还是 checkpoint selection？
```

实际使用模型统一表述为 `Qwen2.5-3B-Instruct`。部分历史脚本文件名包含 `1_5b`，但实验运行时通过 `MODEL_PATH` 指向 3B Instruct 模型。

## 技术栈

```text
EasyR1 / veRL
Ray
vLLM
LoRA
Docker
W&B
Qwen2.5-3B-Instruct
Math12K / GSM8K
```

训练与评测流程：

```text
Math12K problem
-> prompt template
-> vLLM rollout
-> rule-based verifier reward
-> GRPO / GSPO advantage
-> actor policy update
-> validation and checkpoint selection
-> best-of-N / cross-benchmark / verifier audit
```

## 数据与 Reward

主数据集：

```text
hiyouga/math12k
train: 12000
test: 500
```

本项目使用 RLVR 数据契约，而不是 SFT 数据契约：

```text
problem -> actor rollout prompt
answer  -> verifier-only gold label
```

模型不会把 gold answer 当作监督目标看到。gold answer 只用于 rule-based verifier 给采样结果打分。

训练 reward 保持简单：

```text
R_train = 0.9 * answer_accuracy + 0.1 * format
```

`V2-lite` 只用于离线诊断，不参与训练：

```text
diagnostic_score_v2_lite =
  0.85 * answer_accuracy
+ 0.10 * format
+ 0.05 * answer_extractable
- 0.10 * invalid_output_penalty
```

Math12K verifier-aware profiling 结果：

| item | value |
|---|---:|
| train rows | 12000 |
| test rows | 500 |
| train/test problem overlap | 0 |
| train duplicate pair rows | 2 |
| train empty gold answers | 2 |
| train over-max-prompt examples | 9 |
| test over-max-prompt examples | 0 |
| train verifier-ready rate | 84.91% |
| test verifier-ready rate | 84.20% |

结论：Math12K 是可验证数据集，但不是 reward-noise-free 数据集。约 15% 样本存在 symbolic/text gold、possible multi-answer、complex LaTeX、empty gold、long gold 或 overlong prompt 等 verifier 风险。

## 实验矩阵

已完成 A10 主实验：

| run | purpose |
|---|---|
| Base | Qwen2.5-3B-Instruct 原始能力基线 |
| GRPO-composite | answer accuracy + format 主线 RLVR |
| GSPO | 算法对比 |
| GRPO-filtered | online filtering / 数据过滤对比 |
| GRPO-answer-only | 去掉 format shaping 的 reward ablation |
| best-of-N | 评估采样候选分布质量 |
| GSM8K greedy | 跨基准 sanity check |
| V2-lite rescoring | 离线诊断 answer extraction / truncation / repetition |
| verifier audit | 人工审计 verifier 可靠性 |

## 核心结果

训练期验证集结果：

| model/run | accuracy | format rate | train_reward_v1 | notes |
|---|---:|---:|---:|---|
| Base | 0.404 | 0.020 | 0.3656 | Qwen2.5-3B-Instruct |
| GRPO-composite step150 | 0.520 | 0.632 | 0.5312 | best checkpoint |
| GRPO-composite step300 | 0.502 | 0.664 | 0.5182 | final checkpoint |
| GSPO step150 | 0.484 | 0.518 | 0.4874 | algorithm comparison |
| GRPO-filtered step150 | 0.504 | 0.702 | 0.5238 | higher format, lower accuracy than GRPO-composite |
| GRPO-answer-only step150 | 0.506 | 0.530 | 0.5060 | answer-only reward |

独立 Math12K test-500 P0 评测：

| run | greedy | best-of-4 | best-of-8 |
|---|---:|---:|---:|
| Base | 0.426 | 0.560 | 0.602 |
| GRPO-composite | 0.458 | 0.578 | 0.636 |
| GSPO | 0.426 | 0.554 | 0.604 |
| GRPO-answer-only | 0.416 | 0.560 | 0.600 |

GSM8K cross-benchmark sanity check：

| run | GSM8K greedy |
|---|---:|
| Base | 0.790 |
| GRPO-composite | 0.768 |

结论必须保持克制：GRPO-composite 提升了 Math12K in-domain 表现和 best-of-N 候选分布，但 GSM8K 有轻微下降，不能声称泛化能力全面提升。

## Checkpoint Stability

GRPO-composite 的最佳 checkpoint 是 `step150`，不是最终 `step300`。

```text
step150:
accuracy: 0.520
format_rate: 0.632
train_reward_v1: 0.5312
mean_response_length: 367.01

step300:
accuracy: 0.502
format_rate: 0.664
train_reward_v1: 0.5182
mean_response_length: 353.92
```

这说明小算力 LoRA RLVR 中 checkpoint selection 是核心评估环节。继续训练不一定更好，format rate 提升也可能伴随 answer accuracy 回落。

## Verifier Audit 与失败模式

人工审计：

```text
audited examples: 100
verifier-human agreement: 93.0%
false positive count: 0
false negative count: 7
```

人工错误分析 50 条样本：

| category | count |
|---|---:|
| overlong reasoning or truncation | 21 |
| verifier false negative | 13 |
| wrong problem understanding | 5 |
| algebra transformation error | 5 |
| arithmetic error | 3 |
| reasoning shortcut | 2 |
| repetition | 1 |

主要失败链路：

```text
overlong reasoning
-> truncation
-> missing or malformed boxed final answer
-> answer extraction failure
-> zero reward / verifier false negative
```

因此后续闭环改进方向不是盲目堆训练，而是 verifier-aware clean data、concise final-answer prompt、stop-after-boxed postprocess，以及更稳健的离线 verifier 诊断。

## 复现环境

推荐使用与 A10 实验一致的 Docker 镜像：

```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0

docker run -it --ipc=host --gpus=all \
  -v /path/to/EasyR1:/workspace/EasyR1 \
  -v /path/to/models:/models \
  -w /workspace/EasyR1 \
  hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

容器内检查：

```bash
cd /workspace/EasyR1
python3 - <<'PY'
import torch, transformers, ray, vllm
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("ray", ray.__version__)
print("vllm", vllm.__version__)
PY
```

## 最小复现命令

所有命令都应从仓库根目录执行，并通过 `MODEL_PATH` 指向本地 Qwen2.5-3B-Instruct。

Base eval：

```bash
MODEL_PATH=/path/to/Qwen2.5-3B-Instruct \
SINGLE_GPU_ID=0 \
SUITE_MODE=base \
bash examples/run_reasoning_rl_suite_a10.sh
```

GRPO smoke：

```bash
MODEL_PATH=/path/to/Qwen2.5-3B-Instruct \
SINGLE_GPU_ID=0 \
SUITE_MODE=grpo_smoke \
bash examples/run_reasoning_rl_suite_a10.sh
```

GRPO-composite full run：

```bash
MODEL_PATH=/path/to/Qwen2.5-3B-Instruct \
GPU_IDS=0,1 \
MAX_STEPS=300 \
SUITE_MODE=grpo \
bash examples/run_reasoning_rl_suite_a10.sh
```

P0 analysis：

```bash
MODEL_PATH=/path/to/Qwen2.5-3B-Instruct \
SINGLE_GPU_ID=0 \
GRPO_LORA=checkpoints/easyr1_reasoning/qwen2_5_1_5b_math_grpo_a10/global_step_150/actor/lora_adapter \
GSPO_LORA=checkpoints/easyr1_reasoning/qwen2_5_1_5b_math_gspo_a10/global_step_150/actor/lora_adapter \
ANSWER_ONLY_LORA=checkpoints/easyr1_reasoning/qwen2_5_1_5b_math_grpo_answer_only_a10/global_step_150/actor/lora_adapter \
bash scripts/run_p0_reasoning_analysis_a10.sh
```

Clean-concise 闭环实验入口：

```bash
MODEL_PATH=/path/to/Qwen2.5-3B-Instruct \
GPU_ID=0 \
MAX_STEPS=150 \
N_GENERATIONS=4 \
ROLLOUT_BATCH_SIZE=32 \
GLOBAL_BATCH_SIZE=16 \
MAX_RESPONSE_LENGTH=512 \
bash scripts/run_a100_clean_concise_project.sh
```

## 结果与产物

小型 summary/report 可随代码提交，例如：

```text
imported_results_20260425/results/reasoning_rl_all_runs_compare.txt
imported_results_20260425/results/checkpoint_stability_report.json
imported_results_20260425/results/best_of_n_eval_report.json
imported_results_20260425/results/cross_benchmark_eval_report.json
imported_results_20260425/results/reward_v2_rescoring_report.json
imported_results_20260425/results/human_audit_summary.json
```

大体积原始产物不进入 git，包括：

```text
logs/
checkpoints/
results/*.jsonl
imported_results_20260425/results/*.jsonl
easyr1_results_20260425.tar.gz
```

文档资产：

```text
docs/reasoning_rlvr_report.md              # 完整项目报告
docs/reasoning_rlvr_resume_package_zh.md   # 简历描述与 2 分钟讲稿
docs/reasoning_rlvr_qa_bank_zh.md          # 高频面试问答
docs/reasoning_rlvr_interview_script.md    # 英文面试讲稿
docs/a100_handoff_plan_zh.md               # 后续大卡实验交接
docs/reasoning_rlvr_a100_extension_plan_zh.md
```

## 局限性

```text
小模型：Qwen2.5-3B-Instruct
小算力：2xA10
LoRA 而非全参 RL
训练步数有限
rule-based outcome verifier only
没有 process reward
没有把 SFT vs RLVR 作为问题定义
没有声称泛化全面提升
```

本项目的正确简历口径：

```text
基于 EasyR1/veRL 构建小算力数学 RLVR 后训练系统，
系统分析 GRPO/GSPO、reward ablation、checkpoint stability、best-of-N、
verifier audit 和 failure analysis。
```

不要包装成：

```text
复现 DeepSeek-R1
训练出 SOTA 数学模型
证明 GRPO 普遍优于 GSPO
证明模型数学能力全面提升
```
