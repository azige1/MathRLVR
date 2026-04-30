# Reasoning RLVR Interview Script

## 1. 30-Second Version

I built a small-compute math reasoning RLVR analysis system on EasyR1/veRL. The project uses Qwen2.5-3B-Instruct with LoRA on 2xA10 and studies where RLVR gains come from: GRPO vs GSPO, reward ablation, format constraint, checkpoint selection, best-of-N inference, and verifier reliability.

The core result is that, on training-time Math12K validation, the base accuracy is 40.4%, while the best GRPO-composite checkpoint reaches 52.0%, an absolute improvement of 11.6 percentage points. In the independent P0 evaluation, GRPO-composite improves Math12K greedy accuracy from 42.6% to 45.8% and best-of-8 accuracy from 60.2% to 63.6%. The conclusion is scoped to the current small-compute setup; I do not claim GRPO is universally better than GSPO.

## 2. 10-Minute Project Walkthrough

### Problem

The project asks:

```text
Under 2xA10 + Qwen2.5-3B LoRA, where do math RLVR gains come from?
```

I intentionally frame this as a mechanism-analysis project, not an R1 reproduction.

### Framework

The stack is:

```text
EasyR1 / veRL / Ray / vLLM / LoRA / Docker / W&B
```

Training flow:

```text
dataset -> prompt template -> vLLM rollout -> rule verifier reward
-> GRPO or GSPO advantage -> actor update -> validation -> checkpoint selection
```

### Data Pipeline

The data contract is not SFT-style input-output imitation. It is RLVR-style:

```text
problem -> actor prompt
answer  -> verifier-only gold label
```

The model only receives the rendered problem prompt. The gold answer is never
used as a teacher-forcing target; it is only used by the rule verifier.

I treat the prompt template as part of the data contract:

```text
math problem
-> require <think>...</think>
-> require final answer in \boxed{}
```

This matters because verifier reliability depends on final-answer extraction.
If the prompt contract is unstable, the reward label becomes noisy.

The dataset processing has three layers:

```text
static data profiling:
  field checks, duplicates, train/test leakage, token length, answer type,
  topic bucket, difficulty proxy, verifier-risk tags

dynamic rollout data:
  prompt, sampled responses, verifier reward, old/ref logprobs, advantage

offline diagnostic data:
  best-of-N candidates, V2-lite rescoring, verifier audit, error taxonomy
```

Main artifacts:

```text
math12k_data_profile_report.json
math12k_data_manifest.jsonl
math12k_eval_subset_manifest.jsonl
math12k_stratified_data_samples.jsonl
```

The final data profile found:

```text
train rows: 12000
test rows: 500
train/test overlap: 0
train duplicate pairs: 2
train empty gold answers: 2
train verifier-ready rate: 84.91%
test verifier-ready rate: 84.20%
```

The important observation is that Math12K is verifiable but not reward-noise-free.
Roughly 15% of samples have potential verifier risk, such as symbolic/text
answers, possible multi-answer labels, complex LaTeX, long answers, empty gold
answers, or overlong prompts.

This is why I frame data as part of the RL system, not just as a dataset name.

### Reward

Training reward is deliberately simple:

```text
R_train = 0.9 * answer_accuracy + 0.1 * format
```

I call this metric:

```text
train_reward_v1
```

The complex score is not used for training. It is only an offline diagnostic:

```text
diagnostic_score_v2_lite =
  0.85 * answer_accuracy
+ 0.10 * format
+ 0.05 * answer_extractable
- 0.10 * invalid_output_penalty
```

This avoids excessive reward shaping during RL, while still allowing verifier noise analysis.

### Experiments

Experiment matrix:

```text
Base greedy
GRPO-composite
GSPO
GRPO-filtered
GRPO-answer-only
GRPO-format-only
```

Main result:

```text
Training-time validation:
Base accuracy: 0.404
GRPO best accuracy: 0.520
Absolute improvement: +11.6 percentage points

Independent P0 Math12K:
Base greedy: 0.426
GRPO greedy: 0.458
Base best-of-8: 0.602
GRPO best-of-8: 0.636

GSM8K cross-benchmark:
Base greedy: 0.790
GRPO greedy: 0.768
```

Important caveat:

```text
In this current Qwen2.5-3B LoRA, Math12K, 2xA10 setup, GRPO is more stable than GSPO.
This is not a general claim that GRPO is always better.
```

### Diagnostics

I separate four kinds of metrics:

```text
result metrics: accuracy / format_rate / train_reward_v1 / mean_response_length
inference metrics: best-of-N / pass@k / maj@k
stability metrics: KL / approx_KL / clip_ratio / grad_norm / checkpoint curve
diagnostic metrics: diagnostic_score_v2_lite / extract failure / truncation / verifier FP/FN
```

The strongest failure-analysis result is that nearly all 512-token greedy outputs
hit the generation cap. A 1024-token subset eval showed the same behavior:

```text
Base 1024 subset-200 mean length: 1023.09
GRPO 1024 subset-200 mean length: 1023.06
```

So the problem is not only a small token budget. The current prompt and model
tend to over-generate and fail to finalize answers cleanly.

## 3. GRPO Explanation

GRPO samples multiple responses for the same prompt and computes relative advantage within that group.

If a prompt has rewards:

```text
[1.0, 0.0, 0.0, 1.0]
```

then the group mean is `0.5`. Good responses get positive advantage and bad responses get negative advantage.

The key difference from PPO:

```text
PPO uses a critic/value model as baseline.
GRPO uses group statistics as baseline.
```

This makes GRPO cheaper and more suitable for rule-verifiable math tasks under limited compute.

## 4. GSPO Explanation

GSPO uses a sequence-level importance ratio instead of token-level ratio.

GRPO uses:

```text
ratio_t = exp(log pi_theta(y_t) - log pi_old(y_t))
```

GSPO uses a sequence-level log ratio:

```text
seq_log_ratio = mean_t(log pi_theta(y_t) - log pi_old(y_t))
```

In our current small-group, small-LoRA setup, GSPO underperformed GRPO. I treat this as a configuration-specific result, not a universal algorithm conclusion.

## 5. Why Step150 Instead of Step300

The GRPO run reached its best validation point around step150. Later checkpoints improved or maintained format behavior but did not improve accuracy.

This suggests:

```text
small-compute LoRA RLVR needs checkpoint selection;
longer training is not automatically better.
```

## 6. Why Not SFT Baseline

This project is not a full alignment pipeline. It studies fixed-base RLVR mechanisms. SFT baseline is a reasonable extension, but not part of this project's P0 scope.

MedicalGPT covers non-verifiable alignment with SFT/DPO/RM/RLHF. This project covers verifiable reasoning RL.

## 7. Why Not LLM-as-Judge

Math has objective answers. A rule-based verifier is more reproducible than LLM-as-Judge for the main metric.

LLM-as-Judge can be useful as an auxiliary analysis tool, but it is not used as the primary score.

## 8. Why V2-lite Is Not Used for Training

More reward shaping is not always better. If length, repetition, and extractability are weighted too heavily, the model may optimize formatting or safe templates rather than mathematical correctness.

So the training reward stays simple:

```text
answer accuracy dominant, format as small auxiliary term
```

V2-lite is used offline to diagnose reward noise and bad outputs.

## 9. Failure Analysis Talking Points

Expected failure classes:

```text
arithmetic error
algebra transformation error
wrong problem understanding
final answer extraction error
format violation
truncation
repetition
verifier false positive
verifier false negative
```

The key point is that verifier-based RL is only as good as verifier reliability. That is why the project includes a 100-example verifier audit.

The human audit result is:

```text
100 audited examples
verifier-human agreement: 93.0%
false positives: 0
false negatives: 7

label distribution:
correct_accept: 34
correct_reject: 11
truncation: 45
repetition: 3
wrong_reject: 7
```

I labeled conservatively. A case is counted as `wrong_reject`, meaning verifier
false negative, only when the correct answer is explicitly present or directly
equivalent to the boxed answer but the verifier gives zero reward.

The 50-case error analysis gives:

```text
overlong reasoning or truncation: 21
verifier false negative: 13
wrong problem understanding: 5
algebra transformation error: 5
arithmetic error: 3
reasoning shortcut: 2
repetition: 1
```

The dominant failure chain I observed is:

```text
overlong reasoning
-> truncation
-> missing boxed final answer
-> answer extraction failure
-> verifier false negative
```

This means some responses contain a correct intermediate answer but receive zero
reward because the verifier cannot extract a valid final answer.

## 10. Final Resume Version

```text
Built a small-compute math reasoning RLVR post-training system on EasyR1/veRL. Trained Qwen2.5-3B-Instruct with LoRA-GRPO/GSPO on 2xA10, designed an answer-accuracy + format verifier reward, and performed verifier-aware data profiling, reward ablation, checkpoint stability analysis, best-of-N/pass@k/maj@k evaluation, GSM8K cross-benchmark testing, V2-lite offline rescoring, and verifier audit. On independent Math12K test-500 evaluation, GRPO-composite improved greedy accuracy from 42.6% to 45.8% and best-of-8 accuracy from 60.2% to 63.6%; error analysis showed the dominant limitation is overlong reasoning causing truncation and final-answer extraction failure.
```
