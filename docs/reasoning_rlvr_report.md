# EasyR1 Math RLVR Report

## 1. Project Positioning

This project is a small-compute math reasoning RLVR analysis system built on EasyR1/veRL.

It is not a DeepSeek-R1 reproduction and not a claim of training a state-of-the-art math model. The goal is to study where RLVR gains come from under a constrained setup:

```text
2xA10 + Qwen2.5-3B-Instruct + LoRA + EasyR1/veRL
```

Core question:

```text
Under a small-compute LoRA setting, do math RLVR gains come from algorithm choice,
verifier reward, format constraint, filtering, sampling strategy, or checkpoint selection?
```

## 2. Framework

Training stack:

```text
EasyR1 / veRL / Ray / vLLM / LoRA / W&B / Docker
```

Pipeline:

```text
Math dataset
-> prompt template
-> vLLM rollout
-> rule-based verifier reward
-> GRPO or GSPO advantage
-> actor update
-> validation
-> checkpoint selection
-> offline diagnostics
```

## 3. Data Pipeline and Verifier Contract

Primary data:

```text
hiyouga/math12k
train: 12000
test: 500
```

This project uses an RLVR data contract rather than an SFT data contract:

```text
problem -> actor rollout prompt
answer  -> verifier-only gold label
```

The model never sees the gold answer as a supervised target. It only sees the
rendered problem prompt. The gold answer is used by the rule-based verifier to
score sampled responses.

Prompt contract:

```text
problem
-> math.jinja
-> require <think>...</think>
-> require final answer in \boxed{}
```

This prompt contract is treated as part of the data pipeline because it controls
whether the verifier can reliably extract final answers.

Data processing and profiling:

```text
field validation: problem / answer presence
prompt rendering: math.jinja
prompt length profiling: chars and tokenizer length
overlong prompt filtering: max_prompt_length = 1024
answer type classification: integer / fraction / decimal / symbolic / sqrt / power / none
topic classification: algebra / geometry / number theory / combinatorics / probability / sequence / arithmetic / other
difficulty proxy: prompt length + answer complexity + symbolic patterns
verifier-risk tags: empty gold, none/no-solution, symbolic gold, multi-answer, long gold, complex LaTeX
duplicate checks: problem-level and problem-answer pair-level
train/test leakage check: exact normalized problem hash overlap
```

Artifacts:

```text
results/math12k_data_inspection_report_tokenized.json
results/math12k_data_profile_report.json
results/math12k_data_manifest.jsonl
results/math12k_eval_subset_manifest.jsonl
results/math12k_stratified_data_samples.jsonl
```

Known inspection summary from the first local profile:

```text
train/test problem overlap: 0
train empty answer: 2
train duplicate problem-answer rows: 2
train answer mix: mostly integer, with fraction/symbolic/sqrt/power cases
```

Final data profile:

```text
train rows: 12000
test rows: 500
train/test problem overlap: 0
train duplicate pair rows: 2
train empty gold answers: 2
train over-max-prompt examples: 9
test over-max-prompt examples: 0
train verifier-ready rate: 84.91%
test verifier-ready rate: 84.20%
```

Verifier-risk buckets include symbolic/text gold answers, possible multi-answer
answers, complex LaTeX, long gold answers, empty gold answers, and overlong
prompts. This means Math12K is verifiable but not reward-noise-free.

The eval subset manifest records the deterministic test subset used for P0
evaluation. This makes the reported Math12K test-500 results reproducible.

Rollout data:

```text
static data: problem + verifier-only gold answer
dynamic data: sampled responses + verifier rewards + logprobs + GRPO/GSPO advantages
```

The dynamic rollout data is the real training signal in RLVR. For each prompt,
multiple responses are sampled and scored; group-level reward differences drive
the policy update.

P0 cross-benchmark:

```text
GSM8K
```

P1 optional benchmark:

```text
MATH500
```

## 4. Reward Design

Training reward V1:

```text
R_train = 0.9 * answer_accuracy + 0.1 * format
```

Metric name:

```text
train_reward_v1
```

V2-lite is not used for training. It is an offline diagnostic score:

```text
diagnostic_score_v2_lite =
  0.85 * answer_accuracy
+ 0.10 * format
+ 0.05 * answer_extractable
- 0.10 * invalid_output_penalty
```

`invalid_output_penalty = 1` only for likely truncation or severe repetition.

Metric name:

```text
diagnostic_score_v2_lite
```

Do not mix `train_reward_v1` and `diagnostic_score_v2_lite`.

## 5. Experiment Matrix

P0 runs:

```text
Base greedy
GRPO-composite
GSPO
GRPO-filtered
GRPO-answer-only
GRPO-format-only
```

Training-time validation result:

```text
Base accuracy: 0.404
GRPO-composite best checkpoint accuracy: 0.520
Absolute improvement: +11.6 percentage points
```

Conclusion wording must remain scoped:

```text
Under the current Qwen2.5-3B LoRA, Math12K, 2xA10, small-rollout, and current reward setup,
GRPO is more stable than GSPO.
```

Do not claim GRPO is generally better than GSPO.

## 6. Results

### 6.1 Checkpoint Stability

GRPO-composite peaked at step 150 rather than the final step.

```text
GRPO-composite step150:
accuracy: 0.520
format_rate: 0.632
train_reward_v1: 0.5312
mean_response_length: 367.01

GRPO-composite step300:
accuracy: 0.502
format_rate: 0.664
train_reward_v1: 0.5182
mean_response_length: 353.92
```

The final checkpoint improved format rate but lost answer accuracy. This is why
the project reports the best checkpoint and treats checkpoint selection as part
of RLVR evaluation.

### 6.2 Math12K Greedy and Best-of-N

P0 evaluation uses the same Math12K test-500 subset for all models.

```text
Math12K greedy:
Base             0.426
GRPO-composite   0.458
GSPO             0.426
GRPO-answer-only 0.416

Math12K best-of-4:
Base             0.560
GRPO-composite   0.578
GSPO             0.554
GRPO-answer-only 0.560

Math12K best-of-8:
Base             0.602
GRPO-composite   0.636
GSPO             0.604
GRPO-answer-only 0.600
```

GRPO-composite improves both greedy output quality and the candidate answer
distribution. The best-of-8 result shows that GRPO-composite more often samples
a verifier-selected correct answer than the base model.

### 6.3 Cross-Benchmark GSM8K

GSM8K is used as a cross-benchmark sanity check, not as the training target.

```text
GSM8K greedy:
Base             0.790
GRPO-composite   0.768
delta           -2.2 percentage points
```

GRPO-composite does not catastrophically degrade GSM8K, but it shows a small
cross-benchmark drop. The scoped conclusion is that Math12K RLVR improves
in-domain behavior while cross-benchmark generalization needs monitoring.

### 6.4 V2-Lite Offline Rescoring

V2-lite is an offline diagnostic, not a training reward. It re-scores Math12K
greedy outputs to measure answer extractability, truncation, repetition, and
invalid output penalties.

```text
Base:
accuracy: 0.426
format_rate: 0.116
answer_extractable: 0.524
extract_failure: 0.476
truncation: 1.000
diagnostic_score_v2_lite: 0.2999

GRPO-composite:
accuracy: 0.458
format_rate: 0.170
answer_extractable: 0.572
extract_failure: 0.428
truncation: 0.998
diagnostic_score_v2_lite: 0.3351
```

GRPO-composite improves answer extractability and reduces extraction failures,
but almost all greedy outputs still hit the 512-token cap.

### 6.5 1024-Token Sanity Eval

To test whether the 512-token cap is the only problem, a smaller Math12K
subset-200 greedy eval was run with `max_new_tokens=1024`.

```text
Base 1024-token subset-200:
accuracy: 0.340
format_rate: 0.215
diagnostic_score_v2_lite: 0.2567
mean_response_length: 1023.09

GRPO-composite 1024-token subset-200:
accuracy: 0.350
format_rate: 0.265
diagnostic_score_v2_lite: 0.2707
mean_response_length: 1023.06
```

Increasing the response budget alone does not solve the failure mode. Both
models still generate near the length cap. The main issue is not just a short
token budget, but overlong reasoning and weak answer-finalization behavior under
the current prompt.

## 7. Result Metrics

Result metrics:

```text
accuracy
format_rate
train_reward_v1
mean_response_length
```

Inference enhancement metrics:

```text
best_of_N_accuracy
pass@k
maj@k
```

Training stability metrics:

```text
KL / approx_KL
clip_ratio
grad_norm
reward_curve
checkpoint_curve
length_curve
```

Offline diagnostic metrics:

```text
diagnostic_score_v2_lite
extract_failure_rate
truncation_rate
repetition_rate
verifier-human agreement
false_positive_rate
false_negative_rate
```

## 8. Required Artifacts

P0 artifacts:

```text
results/reasoning_rl_all_runs_compare.txt
results/checkpoint_stability_report.json
results/best_of_n_eval_report.json
results/cross_benchmark_eval_report.json
results/reward_v2_rescoring_report.json
results/verifier_audit.jsonl
results/error_analysis_50.jsonl
```

All P0 artifacts have been generated locally under:

```text
imported_results_20260425/results
```

## 9. Verifier Audit

Audit size:

```text
100 examples
```

Sampling strategy:

```text
50 disagreement / suspicious cases
30 incorrect cases
20 random cases
```

Allowed labels:

```text
correct_accept
correct_reject
wrong_accept
wrong_reject
format_fail
extract_fail
parse_fail
truncation
repetition
ambiguous_gold
```

Report:

```text
verifier-human agreement
false positive rate
false negative rate
extract failure rate
parse failure rate
```

Manual audit result:

```text
audited examples: 100
verifier-human agreement: 93.0%
false positive count: 0
false negative count: 7
```

Human audit label distribution:

```text
correct_accept: 34
correct_reject: 11
truncation: 45
repetition: 3
wrong_reject: 7
```

I used a conservative labeling rule: a sample is counted as `wrong_reject`
only when the correct answer is explicitly present or directly equivalent to
the boxed answer, but the verifier still assigns zero reward. Most rejected
samples are not verifier mistakes; they are incomplete outputs caused by
truncation or repetition.

## 10. Error Analysis

Analyze 50 examples with this taxonomy:

```text
arithmetic error
algebra transformation error
wrong problem understanding
reasoning shortcut
final answer extraction error
format violation
overlong reasoning or truncation
repetition
verifier false positive
verifier false negative
other
```

Observed dominant error chain:

```text
overlong reasoning
-> output truncation
-> missing or malformed boxed final answer
-> answer extraction failure
-> verifier false negative
```

Some cases contain the correct intermediate answer but receive zero reward
because the response never finalizes into a valid `\boxed{}` answer.

Manual error analysis result:

```text
overlong reasoning or truncation: 21
verifier false negative: 13
wrong problem understanding: 5
algebra transformation error: 5
arithmetic error: 3
reasoning shortcut: 2
repetition: 1
```

The main conclusion is that the dominant bottleneck is not only mathematical
ability. A large part of the loss comes from answer-finalization failure:
responses often keep reasoning until they hit the generation cap, so the
verifier cannot extract a final boxed answer even when the correct answer has
already appeared in the reasoning.

## 11. Limitations

Known limits:

```text
small model: Qwen2.5-3B
small compute: 2xA10
LoRA instead of full-parameter RL
short training budget
rule-based outcome verifier only
no process reward
no LLM-as-Judge main metric
no SFT baseline in this project
```

## 12. Follow-Up Work

P1:

```text
prompt shortening / stricter answer-finalization instruction
stop criteria or post-answer stopping
length penalty or overlong penalty in reward
MATH500 eval
GRPO-format-only training and eval
V2-lite best-of-N rerank
```

P2:

```text
Qwen3-4B or 7B scaling on A100
GRPO robust-v2 training after verifier audit
SFT baseline
process reward
LLM-as-Judge auxiliary analysis
```
