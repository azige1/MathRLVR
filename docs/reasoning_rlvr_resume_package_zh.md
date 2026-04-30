# EasyR1 Math RLVR 简历与面试包装

## 简历短版

```text
基于 EasyR1/veRL 搭建小算力数学推理 RLVR 后训练系统，在 2xA10 环境下对 Qwen2.5-3B-Instruct 进行 LoRA-GRPO/GSPO 训练；完成 Math12K 数据画像、verifier reward 设计、reward ablation、checkpoint stability、best-of-N 评测、GSM8K 跨基准评测和 100 条 verifier audit。GRPO-composite 在 Math12K test-500 上将 greedy accuracy 从 42.6% 提升到 45.8%，best-of-8 accuracy 从 60.2% 提升到 63.6%。
```

## 简历长版

```text
基于 EasyR1/veRL 构建小算力数学推理 RLVR 后训练实验系统，使用 Qwen2.5-3B-Instruct + LoRA 在 2xA10 上对比 GRPO/GSPO 及 reward ablation。设计 answer accuracy + format 的 rule-based verifier reward，并完成 Math12K verifier-aware data profiling、train/test leakage check、answer type/risk tagging、checkpoint stability analysis、best-of-N/pass@k/maj@k 推理评测、GSM8K cross-benchmark 和 V2-lite offline rescoring。实验显示 GRPO-composite 在 Math12K test-500 上将 greedy accuracy 从 42.6% 提升到 45.8%，best-of-8 accuracy 从 60.2% 提升到 63.6%；人工审计 100 条样本得到 verifier-human agreement 93.0%，定位主要瓶颈为 overlong reasoning 导致 truncation 和 final-answer extraction failure。
```

## 中文项目经历版

```text
小算力数学推理 RLVR 后训练系统
- 基于 EasyR1/veRL、Ray、vLLM、LoRA 和 W&B 搭建数学 RLVR 实验链路，使用 Qwen2.5-3B-Instruct 在 2xA10 上完成 GRPO/GSPO 训练与分析。
- 设计训练 reward：R_train = 0.9 * answer_accuracy + 0.1 * format，并将复杂的 V2-lite score 仅用于离线诊断，避免过度 reward shaping。
- 对 Math12K 做 verifier-aware data profiling，检查 12000 train / 500 test 的字段、token length、answer type、verifier-risk tags、duplicates 和 train/test leakage，发现 test overlap 为 0，verifier-ready rate 约 84%。
- 完成 Base、GRPO-composite、GSPO、GRPO-filtered、GRPO-answer-only 等实验矩阵，分析 checkpoint stability、reward ablation、best-of-N 和 GSM8K cross-benchmark。
- GRPO-composite 在 Math12K 独立 test-500 上将 greedy accuracy 从 42.6% 提升到 45.8%，best-of-8 accuracy 从 60.2% 提升到 63.6%；训练期验证集从 40.4% 提升到 52.0%。
- 人工审计 100 条 verifier 样本和 50 条错误样本，统计 verifier-human agreement 为 93.0%，发现主要失败链路为 overlong reasoning -> truncation -> missing boxed final answer -> answer extraction failure -> verifier false negative。
```

## 30 秒面试介绍

我做了一个基于 EasyR1/veRL 的小算力数学推理 RLVR 项目。这个项目不是复现 DeepSeek-R1，而是在 `2xA10 + Qwen2.5-3B + LoRA` 条件下分析 RLVR 的收益来源。

我完成了 GRPO、GSPO、filtered data、answer-only reward 等实验，对 Math12K 做了数据画像和 verifier 风险分析，并做了 best-of-N、GSM8K 跨基准、V2-lite 离线诊断、100 条 verifier audit 和 50 条 error analysis。核心结果是 GRPO-composite 在 Math12K test-500 上把 greedy accuracy 从 `42.6%` 提升到 `45.8%`，best-of-8 从 `60.2%` 提升到 `63.6%`。同时我也发现 GSM8K 有轻微下降，主要失败模式是推理太长导致截断和最终答案抽取失败。

## 2 分钟面试介绍

这个项目的目标是研究小算力条件下数学 RLVR 的真实收益来源。我的算力限制是 `2xA10`，模型是 `Qwen2.5-3B-Instruct`，训练方式是 LoRA，所以我没有把它包装成强模型训练，而是做成机制分析。

数据上，我使用 `hiyouga/math12k`，它有 `12000` 条 train 和 `500` 条 test。我不是直接把数据拿来训练，而是做了 verifier-aware profiling：检查 prompt length、answer type、symbolic answer、multi-answer、complex LaTeX、empty gold、duplicate 和 train/test leakage。最后发现 train/test overlap 是 `0`，但 verifier-ready rate 只有约 `84%`，说明数学数据虽然可验证，但 verifier 不是无噪声的。

训练上，我用的是简单 reward：

```text
R_train = 0.9 * answer_accuracy + 0.1 * format
```

然后对比了 Base、GRPO-composite、GSPO、GRPO-filtered、GRPO-answer-only。训练期验证集上，Base accuracy 是 `0.404`，GRPO-composite best checkpoint 是 `0.520`。独立 P0 评测上，GRPO-composite 把 Math12K greedy 从 `0.426` 提升到 `0.458`，best-of-8 从 `0.602` 提升到 `0.636`。

但我没有只报提升。GSM8K 上 Base 是 `0.790`，GRPO 是 `0.768`，说明这个 RLVR 主要提升 in-domain Math12K，不应该夸大泛化。最后我做了 100 条 verifier audit 和 50 条 error analysis，发现 verifier-human agreement 是 `93%`，主要失败是 overlong reasoning 导致 truncation，最终没有 boxed answer，所以 verifier 抽不到答案。

## 10 分钟项目讲法

### 1. 背景

大模型后训练里，数学推理是典型的 RLVR 场景。RLVR 是 reinforcement learning with verifiable reward，即用可验证规则而不是人工偏好模型来给回答打分。

我想研究的问题是：

```text
小算力条件下，RLVR 到底有没有收益？
收益来自算法、reward、格式、采样，还是 checkpoint selection？
```

### 2. 框架

项目基于：

```text
EasyR1 / veRL / Ray / vLLM / LoRA / W&B / Docker
```

流程是：

```text
dataset -> prompt template -> rollout -> verifier reward -> advantage -> policy update -> validation
```

### 3. 数据

数据是 Math12K。

我强调这不是 SFT 数据格式，而是 RLVR 数据格式：

```text
problem 给模型生成
answer 只给 verifier 打分
```

我做了数据画像，发现：

```text
train: 12000
test: 500
train/test overlap: 0
verifier-ready rate: train 84.91%, test 84.20%
```

这说明数据不是完全干净，有 symbolic/text gold、multi-answer、complex LaTeX、empty gold 等 verifier 风险。

### 4. Reward

训练 reward 保持简单：

```text
0.9 * answer_accuracy + 0.1 * format
```

复杂的 V2-lite 只做离线诊断，不参与训练。这样做是为了避免模型优化 length 或 template，而不是数学正确性。

### 5. 实验

实验包括：

```text
Base
GRPO-composite
GSPO
GRPO-filtered
GRPO-answer-only
```

训练期结果：

```text
Base 0.404
GRPO-composite best 0.520
```

独立 P0 评测：

```text
Math12K greedy: 0.426 -> 0.458
Math12K best-of-8: 0.602 -> 0.636
GSM8K: 0.790 -> 0.768
```

### 6. 分析

我发现三个重要现象。

第一，best checkpoint 是 step150，不是 step300。step300 format 更高，但 accuracy 回落。

第二，best-of-N 提升比 greedy 明显，说明 GRPO 改善了候选答案分布。

第三，GSM8K 下降，说明不能说泛化全面提升。

### 7. 审计

我做了 100 条 verifier audit：

```text
agreement: 93%
false positive: 0
false negative: 7
truncation: 45
```

50 条错误分析显示：

```text
overlong reasoning/truncation: 21
verifier false negative: 13
```

最终定位：

```text
overlong reasoning -> truncation -> no boxed final answer -> extraction failure -> zero reward
```

### 8. 结论

这个项目的结论是：当前小算力设置下，GRPO 对 Math12K 有 in-domain 收益，但主要瓶颈已经从“会不会算”转向“能不能及时输出可验证最终答案”。这也是后续如果做 A100 增强或改 reward/prompt 时最应该优先优化的方向。

## 项目亮点

### 亮点 1：不是只跑 demo，有完整实验矩阵

```text
Base / GRPO / GSPO / filtered / answer-only / best-of-N / GSM8K / audit
```

### 亮点 2：数据处理不是空话

做了 verifier-aware data profiling，而不是只写“使用 Math12K”。

### 亮点 3：结论克制

没有声称模型全面变强，而是明确指出 GSM8K 下降和 verifier 噪声。

### 亮点 4：有人工审计

100 条 verifier audit + 50 条 error analysis，能支撑 failure analysis。

### 亮点 5：能讲清 RLVR 工程栈

EasyR1/veRL/Ray/vLLM/LoRA/W&B 都能和具体训练流程对应起来。

## 不能乱说的话

不要说：

```text
我复现了 DeepSeek-R1。
GRPO 一定比 GSPO 好。
模型数学能力全面提升。
训练出了强数学模型。
GSM8K 也提升了。
```

应该说：

```text
在当前 Qwen2.5-3B、LoRA、2xA10、Math12K 和当前 reward 设计下，
GRPO-composite 比 GSPO 更稳，并提升了 Math12K in-domain 表现。
但 GSM8K 有轻微下降，说明泛化需要进一步验证。
```

## 推荐简历关键词

```text
LLM post-training
RLVR
GRPO
GSPO
EasyR1
veRL
Ray
vLLM
LoRA
rule-based verifier
verifier-aware data profiling
reward ablation
checkpoint stability
best-of-N
pass@k
maj@k
verifier audit
error analysis
answer extraction failure
truncation
```

## 一句话项目定位

```text
MedicalGPT 展示 non-verifiable medical alignment；
EasyR1 Math RLVR 展示 verifiable reasoning RL。
```

这两个项目组合起来，能覆盖后训练岗位里两类核心场景：

- 医疗：安全、保守、不可完全规则验证。
- 数学：可验证 reward、RLVR、GRPO/GSPO、推理模型训练。

## 最终项目口径

推荐口径：

```text
这是一个基于已有 Instruct 模型的 post-instruction RLVR 分析项目。
Base baseline 是 Qwen2.5-3B-Instruct 原模型；
实验重点是 verifiable reward、GRPO/GSPO、checkpoint selection、best-of-N 和 verifier failure analysis。
```

不要把它讲成：

```text
SFT vs RLVR 对比项目
base model 指令微调项目
复现 DeepSeek-R1 项目
强数学模型训练项目
```

如果被问到为什么没有 SFT baseline，可以回答：

```text
当前起点已经是 Qwen2.5-3B-Instruct，不是 base model。
我的问题定义是已有 Instruct 模型上的 RLVR 后训练增益和失败模式，
所以 Base-Instruct 是合理基线。SFT baseline 可以作为扩展，但不是当前 P0 的必要对照。
```
