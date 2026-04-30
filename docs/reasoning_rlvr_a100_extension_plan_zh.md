# EasyR1 Math RLVR A100 增强实验计划

## 目标

A100 不用于重新铺一整套 A10 ablation，而是做一个增强验证：

```text
同一套 EasyR1/veRL RLVR pipeline 在更强 base model 上是否仍然成立？
```

最推荐目标：

```text
Qwen3-4B-Instruct
```

备选目标：

```text
Qwen2.5-7B-Instruct
```

## 为什么 A100 不要重跑全部实验

A10 已经完成了机制分析：

```text
Base
GRPO-composite
GSPO
GRPO-filtered
GRPO-answer-only
best-of-N
GSM8K
V2-lite
verifier audit
error analysis
```

A100 如果再完整复刻这套实验，会花很多钱，但信息增量不大。

更合理的是：

```text
只验证更强模型上的主线结论。
```

## 最小花费方案

### 方案 A：Qwen3-4B GRPO-composite

适合：

```text
如果能下载到 Qwen3-4B-Instruct。
```

实验：

```text
1. Base greedy eval
2. GRPO-composite 150-300 steps
3. Math12K greedy eval
4. Math12K best-of-4 / best-of-8
5. GSM8K greedy eval
6. V2-lite rescoring
7. 抽样 30 条 verifier audit
```

预期价值：

```text
验证较新/较强 base model 下，GRPO 是否还能提升 Math12K in-domain 表现。
```

### 方案 B：Qwen2.5-7B GRPO-composite

适合：

```text
如果没有 Qwen3-4B，或者希望展示 7B scaling。
```

训练方式：

```text
LoRA 或 QLoRA
```

推荐优先：

```text
A100 40GB: QLoRA 更稳
A100 80GB: LoRA 更稳，batch 可以更宽松
```

实验：

```text
1. 7B Base greedy eval
2. 7B GRPO-composite smoke 100-150 steps
3. Math12K greedy eval
4. Math12K best-of-4
5. GSM8K greedy eval
```

如果时间足够再补：

```text
best-of-8
V2-lite rescoring
30 条 audit
```

## A100 40GB 推荐配置

### Qwen3-4B / Qwen2.5-7B 通用

```text
LoRA/QLoRA
bf16
gradient_checkpointing=true
max_prompt_length=1024
max_response_length=512 或 768
rollout_batch_size=32 起步
global_batch_size=16 起步
n_generations=2
temperature=1.0
top_p=1.0
```

如果 OOM：

```text
1. 降 max_response_length
2. 降 rollout_batch_size
3. 降 global_batch_size
4. 改 QLoRA
5. 降 LoRA rank
```

## 时间预算

A100 费用按 `8 元/小时` 估算。

### 最小验证

```text
Base eval: 1-2 小时
GRPO 100-150 steps: 3-6 小时
Math12K greedy + best-of-4: 2-4 小时
GSM8K greedy: 1-2 小时
整理报告: 本地完成
```

总计：

```text
7-14 小时
约 56-112 元
```

### 稳妥验证

```text
GRPO 300 steps
best-of-8
V2-lite
audit subset
```

总计：

```text
15-24 小时
约 120-192 元
```

## 推荐执行顺序

### Step 1：先做 Base Eval

目的：

```text
确认模型加载、数据、prompt、verifier 全部正常。
```

如果 Base eval 明显异常，不要开始训练。

### Step 2：跑 GRPO-composite smoke

先跑：

```text
MAX_STEPS=50
```

确认：

```text
loss 正常
reward 正常
显存稳定
checkpoint 能保存
W&B 正常同步
```

### Step 3：跑主训练

如果 smoke 正常，再跑：

```text
MAX_STEPS=150 或 300
```

不建议一上来跑很久。

### Step 4：统一评测

至少评：

```text
Math12K greedy
Math12K best-of-4
GSM8K greedy
```

如果时间足够：

```text
Math12K best-of-8
V2-lite rescoring
30 条 audit
```

## 成功标准

A100 增强实验不要求一定超过 A10 很多。

成功标准是：

```text
1. 更强 base 上 RLVR pipeline 能稳定跑通。
2. Math12K in-domain 有正向趋势。
3. best-of-N 候选分布不下降。
4. GSM8K 不出现明显灾难性下降。
5. failure mode 和 A10 是否一致能被分析。
```

如果结果不好，也可以包装：

```text
更强模型下，小步 GRPO 未必直接带来稳定收益；
需要重新调 response length、group size、KL、rollout batch 和 prompt。
```

## 不建议做的事

不要在 A100 上做：

```text
完整重跑 GRPO / GSPO / filtered / answer-only / format-only
重新做 100 条 audit
process reward
LLM-as-Judge 主评测
全量 MATH500
长时间 7B GSPO ablation
```

原因：

```text
这些不直接服务于当前项目叙事，花钱多，边际收益低。
```

## 最推荐 A100 结论包装

如果增强实验成功，可以写：

```text
在 A10 上完成完整机制分析后，进一步在 A100 上验证同一 RLVR pipeline 对更强 base model 的可迁移性。结果显示 GRPO-composite 在更强模型上仍能稳定训练，并维持/提升 Math12K in-domain 表现，说明该系统不是针对单一 3B 模型的偶然 demo。
```

如果增强实验不成功，可以写：

```text
A100 scaling 实验显示，小算力 3B 上有效的 GRPO 配置不能直接迁移到更强模型；主要瓶颈转向 response length、KL 控制、group size 和 prompt finalization。这说明 RLVR 配置需要随模型规模重新调参，而不是简单扩大模型。
```

两种结果都能讲，只要实验设计清楚。
