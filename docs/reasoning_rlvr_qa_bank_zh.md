# EasyR1 Math RLVR 高频面试 Q&A

## 1. 这个项目一句话是什么？

这是一个基于 `EasyR1/veRL` 的小算力数学推理 RLVR 后训练机制分析项目。我在 `2xA10 + Qwen2.5-3B-Instruct + LoRA` 条件下，对比 GRPO、GSPO、reward ablation、best-of-N、checkpoint selection 和 verifier audit，分析数学 RLVR 的收益和失败原因。

## 2. 这个项目为什么不是 toy？

不是 toy 的原因有四点。

第一，它不是只跑一个 demo，而是有完整实验矩阵：Base、GRPO-composite、GSPO、GRPO-filtered、answer-only、best-of-N、GSM8K 和 verifier audit。

第二，它有数据处理和 verifier-aware profiling，不只是写了一个数据集名字。

第三，它有正负结果：Math12K 提升，但 GSM8K 下降，没有只报喜。

第四，它有人工审计和错误分析，定位到 overlong reasoning、truncation 和 answer extraction failure。

但我也不会把它包装成强模型训练，因为模型只有 3B，算力是 2xA10，主要价值是机制分析。

## 3. RLVR 是什么？

RLVR 是 `Reinforcement Learning with Verifiable Reward`，中文可以理解为“带可验证奖励的强化学习”。

和 RLHF 不同，RLHF 通常需要 reward model 或人类偏好；RLVR 的 reward 可以由规则直接判断，比如数学最终答案是否正确。

在这个项目里：

```text
problem -> 模型采样 response -> rule-based verifier 判断答案和格式 -> reward
```

## 4. 为什么数学适合 RLVR？

数学题通常有客观标准答案，可以用 rule-based verifier 检查最终答案是否等价。相比开放问答或医疗咨询，数学 reward 更可重复、更便宜，也更适合小算力实验。

但它不是完全无噪声。比如 `15` 和 `15 cm^2`、矩阵、区间、符号答案、多答案题，都可能让 verifier 出错。

## 5. EasyR1 和 veRL 分别是什么？

`EasyR1` 是面向 reasoning RL 的训练项目，封装了数据、reward、rollout、训练脚本和示例。

`veRL` 是更底层的 RLHF/RLVR 训练框架，负责分布式 worker、rollout、logprob、advantage、policy update、checkpoint 等流程。

在本项目里，EasyR1 更像实验入口和配置层，veRL 是实际训练引擎。

## 6. 项目用了什么模型？

实际使用的是：

```text
Qwen2.5-3B-Instruct
```

虽然部分文件名里有 `1_5b`，但实际 `MODEL_PATH` 指向的是 Qwen2.5-3B-Instruct。

## 7. 为什么用 3B，不直接用 7B？

因为项目目标是小算力机制分析，A10 显存有限。3B + LoRA 可以在 `2xA10` 上完成多组实验矩阵。

如果直接上 7B，单次实验成本更高，反而不利于做 GRPO/GSPO、ablation、audit、best-of-N 这些机制分析。7B 更适合作为 A100 增强实验。

## 8. 数据集是什么？

主数据集是：

```text
hiyouga/math12k
train: 12000
test: 500
```

跨基准评测使用：

```text
GSM8K test subset
```

## 9. 数据怎么处理？

我做了 verifier-aware data profiling，包括字段检查、prompt 渲染、token length、answer type、topic bucket、difficulty proxy、verifier-risk tags、duplicate check 和 train/test leakage check。

关键结果：

```text
train rows: 12000
test rows: 500
train/test overlap: 0
train duplicate pair rows: 2
train empty gold answers: 2
train verifier-ready rate: 84.91%
test verifier-ready rate: 84.20%
```

## 10. 为什么说数据处理是项目核心？

因为 RLVR 的 reward 质量依赖数据和 verifier。如果 gold answer 有复杂 LaTeX、区间、单位、多答案、空答案，rule-based verifier 可能误判。

所以数据处理不是简单清洗，而是判断哪些样本更适合 rule reward，哪些样本容易产生 reward noise。

## 11. 什么是 verifier-ready rate？

`verifier-ready rate` 指样本是否适合被当前 rule-based verifier 稳定判断。

比如整数、简单分数通常 verifier-ready；复杂矩阵、区间、多答案、空答案、复杂符号表达式就有风险。

本项目里 test verifier-ready rate 约 `84.20%`，说明数据可验证但不是无噪声。

## 12. Reward 是怎么设计的？

训练 reward 是：

```text
R_train = 0.9 * answer_accuracy + 0.1 * format
```

`answer_accuracy` 是答案正确性，`format` 是格式是否满足要求。

这样设计是为了让最终答案正确性占主导，format 只是辅助 verifier 抽取。

## 13. 为什么不把 length/repetition 直接放进训练 reward？

因为 reward shaping 太复杂会引入非目标优化。模型可能学会短回答、模板化回答或避免重复，但数学准确率不一定提高。

所以训练 reward 保持简单，复杂指标只做离线诊断。

## 14. V2-lite 是什么？

V2-lite 是离线诊断分数，不参与训练：

```text
diagnostic_score_v2_lite =
  0.85 * answer_accuracy
+ 0.10 * format
+ 0.05 * answer_extractable
- 0.10 * invalid_output_penalty
```

它用来分析输出是否可抽取、是否截断、是否复读，而不是拿来更新模型。

## 15. GRPO 是什么？

GRPO 是 Group Relative Policy Optimization，中文可以理解为“组内相对优势策略优化”。

它对同一个 prompt 采样多个 response，用组内 reward 均值和方差计算相对 advantage。这样不需要单独训练 critic/value model。

如果同一题采样 4 个答案，reward 是：

```text
[1, 0, 0, 1]
```

组均值是 `0.5`，正确答案 advantage 为正，错误答案 advantage 为负。

## 16. GRPO 和 PPO 的区别是什么？

PPO 通常需要 value model 估计 baseline，并用 GAE 计算 advantage。

GRPO 不训练 critic，而是用同一 prompt 下多个 sampled responses 的组内统计作为 baseline。

对数学 RLVR 来说，GRPO 更省显存、更适合小算力。

## 17. GSPO 是什么？

GSPO 使用 sequence-level importance ratio。

GRPO 主要是 token-level ratio：

```text
ratio_t = exp(log pi_theta(y_t) - log pi_old(y_t))
```

GSPO 使用序列级平均：

```text
seq_log_ratio = mean_t(log pi_theta(y_t) - log pi_old(y_t))
```

可以理解为 GSPO 在前向目标里更关注整条 response 的平均变化，而不是每个 token 独立变化。

## 18. 为什么当前实验里 GRPO 比 GSPO 好？

只能说在当前设置下 GRPO 更稳，不能泛化。

当前设置是：

```text
Qwen2.5-3B
LoRA
2xA10
Math12K
small rollout
current reward
```

结果上，GRPO-composite best accuracy 是 `0.520`，GSPO 是 `0.484`。可能原因包括小 batch、小 group size、sequence-level ratio 更敏感、reward sparse、response 过长等。

## 19. 为什么不能说 GRPO 全面优于 GSPO？

因为算法表现依赖模型规模、batch size、group size、reward、KL、clip、响应长度和实现细节。

本项目只能证明：

```text
在当前小算力配置下，GRPO-composite 比 GSPO 更稳。
```

不能说：

```text
GRPO 一定比 GSPO 好。
```

## 20. 什么是 checkpoint stability？

Checkpoint stability 是看不同训练步数的模型表现是否稳定。

本项目里 GRPO step150 最好：

```text
step150 accuracy: 0.520
step300 accuracy: 0.502
```

step300 format 更高，但 accuracy 回落，说明继续训练不一定带来更高准确率。

## 21. 为什么 step150 比 step300 好？

可能是 reward 逐渐偏向格式优化，或者小算力 LoRA 下训练后期出现轻微 over-optimization。

具体表现是：

```text
step150 accuracy 更高
step300 format_rate 更高
```

所以最终选择 best checkpoint，而不是 last checkpoint。

## 22. Greedy 输出是什么？

Greedy 输出是 `temperature=0, n=1` 的确定性生成。

也就是每题只生成一条最可能的 response，不做随机采样。

## 23. Best-of-N 是什么？

Best-of-N 是每道题采样 N 条 response，然后用 verifier 从中选 reward 最高的一条。

它衡量的是模型候选答案分布里是否包含正确答案。

本项目里：

```text
Base best-of-8: 0.602
GRPO best-of-8: 0.636
```

说明 GRPO 让模型更容易采样出正确候选。

## 24. Pass@k 和 maj@k 是什么？

`pass@k`：k 个候选里至少一个正确就算通过。

`maj@k`：多数投票，取出现最多的答案作为最终答案。

数学推理里 best-of-N 和 pass@k 可以反映模型的潜在推理能力，greedy 反映默认输出能力。

## 25. GSM8K 为什么下降？

GSM8K 从 `0.790` 到 `0.768`，下降 `2.2` 个百分点。

这说明 Math12K 上的 RLVR 主要提升 in-domain 表现，不保证跨基准泛化。可能原因是训练数据分布、prompt 格式、reward 偏好和输出长度策略不同。

这也是项目里需要如实讲的负结果。

## 26. 什么是 verifier audit？

Verifier audit 是人工检查自动 verifier 判分是否正确。

我抽了 100 条样本，人工判断模型答案和 verifier 判断是否一致。

结果：

```text
verifier-human agreement: 93.0%
false positive: 0
false negative: 7
```

## 27. wrong_reject 是什么？

`wrong_reject` 是 verifier false negative。

意思是：

```text
模型其实答对了，但 verifier 判错。
```

典型原因：

- 答案在推理中出现，但没有 boxed。
- 输出被截断，没有最终答案。
- `15` 和 `15 cm^2` 这种单位等价没有识别。
- ordered pair 或 LaTeX 格式没被抽取。

## 28. false positive 是什么？

false positive 是模型答错，但 verifier 判对。

本项目 100 条审计里 false positive 是 `0`。这说明当前 verifier 更偏保守，主要问题是漏判正确答案，而不是误收错误答案。

## 29. 主要失败模式是什么？

主要失败链路是：

```text
overlong reasoning
-> truncation
-> missing boxed final answer
-> answer extraction failure
-> verifier false negative
```

中文解释：

模型推理太长，打满 token 上限，被截断，没有输出标准 boxed final answer，所以 verifier 抽不到答案。

## 30. 为什么说主要问题是 overlong？

证据有三层。

第一，V2-lite 诊断显示：

```text
truncation ≈ 1.0
mean_response_length ≈ 511 / 512
```

第二，1024-token subset eval 仍然：

```text
mean_response_length ≈ 1023 / 1024
```

第三，人工错误分析里：

```text
overlong reasoning or truncation: 21/50
verifier false negative: 13/50
```

所以不是简单 token budget 太短，而是模型不会及时收束。

## 31. 为什么 1024 token 没解决问题？

因为模型不是只差一点空间，而是倾向于持续解释。把上限从 512 增到 1024 后，平均长度仍接近 1024。

这说明应该优化 prompt、stop criteria、answer-finalization，而不是单纯增加 max_new_tokens。

## 32. 这个项目的最大贡献是什么？

最大贡献不是最终 accuracy 多高，而是把小算力 RLVR 的收益和失败原因拆开了：

- GRPO 在当前设置下确实提升 Math12K。
- Best-of-N 显示候选分布变好。
- GSM8K 显示泛化不能夸大。
- Verifier audit 显示主要问题是漏判和截断。
- 数据画像显示 Math12K 可验证但不是无噪声。

## 33. 如果面试官说这个项目有点 toy，怎么回答？

我会承认它不是大规模模型训练，但强调它不是 toy demo。

回答：

```text
如果把项目目标定义成训练强数学模型，那 3B + 2xA10 确实不够。
但我的目标是小算力 RLVR 机制分析。
我不仅跑了 GRPO，还做了 GRPO/GSPO 对比、reward ablation、best-of-N、GSM8K、数据画像、verifier audit 和 error analysis。
所以它的价值在于后训练机制和 verifier 诊断，而不是模型最终能力。
```

## 34. 如果给 A100，下一步怎么做？

我不会盲目扩大所有实验，而是做一个增强主线：

```text
Qwen3-4B 或 Qwen2.5-7B
GRPO-composite
Math12K greedy / best-of-N
GSM8K cross-benchmark
verifier audit subset
```

目标是验证同一套 pipeline 在更强 base model 上是否仍成立。

## 35. 为什么不用 process reward？

当前项目使用 outcome reward，也就是只看最终答案。Process reward 需要步骤级标注或过程 verifier，成本更高，而且评估复杂。

在小算力项目里，我先把 outcome reward 的机制和失败模式分析清楚。Process reward 可以作为后续方向。

## 36. 为什么不用 LLM-as-Judge？

数学题有客观答案，rule-based verifier 更可重复。

LLM-as-Judge 可以用于辅助错误分析，但不适合作为主评测指标，因为它会引入额外 judge bias。

## 37. 为什么不做 SFT baseline？

这个项目的起点已经是 `Qwen2.5-3B-Instruct`，不是 base model。它已经具备基本指令跟随和数学问答能力。

所以当前问题定义不是：

```text
SFT 和 RLVR 谁更强？
```

而是：

```text
在已有 Instruct 模型上，verifiable reward 的 RL 后训练能带来什么变化？
```

因此 `Base-Instruct` 是合理基线。SFT baseline 是合理扩展，但不是当前 P0 的必要对照。P0 重点是 Base-Instruct、GRPO、GSPO、reward ablation、checkpoint selection、best-of-N 和 verifier analysis。

同时，MedicalGPT 项目已经覆盖 SFT/DPO/RM/RLHF 这条 non-verifiable alignment 链路。

面试回答可以说：

```text
如果研究 base-to-instruct 或 SFT-vs-RLVR，我会补 SFT baseline。
但当前项目研究 post-instruction RLVR，所以我用 Qwen2.5-3B-Instruct 原模型作为 Base，对比 RLVR 后训练前后的变化。
```

## 38. 项目有哪些局限？

局限包括：

```text
Qwen2.5-3B 模型较小
2xA10 算力有限
LoRA 而非全参 RL
训练步数不长
只用 outcome verifier
没有 process reward
没有 SFT baseline
GSM8K 没有提升
输出控制问题严重
```

## 39. 这个项目和 MedicalGPT 怎么互补？

MedicalGPT 是 non-verifiable medical alignment，重点是保守医疗回答、SFT/DPO/RM/RLHF、安全评测。

EasyR1 Math RLVR 是 verifiable reasoning RL，重点是 rule reward、GRPO/GSPO、rollout、verifier audit、best-of-N。

两者覆盖了后训练岗位两个核心方向：

- 不可完全规则验证的领域对齐。
- 可验证推理任务的 RLVR。

## 40. 最终应该怎么总结？

推荐总结：

```text
这个项目证明了在小算力约束下，RLVR 仍然可以通过 GRPO 改善 Math12K in-domain 表现和候选答案分布。
但项目更重要的发现是：当前瓶颈不只是数学能力，而是过长推理导致的截断、最终答案缺失和 verifier false negative。
因此后续优化应该优先改 prompt、stop criteria、answer extraction 和 verifier robustness，再考虑扩大模型规模。
```
