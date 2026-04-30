#!/usr/bin/env python3
"""Evaluate math models with greedy or best-of-N sampling."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer

from rlvr_eval_utils import (
    diagnostic_score_v2_lite,
    extract_boxed_content_fallback,
    grade_prediction,
    normalize_text,
    postprocess_generation,
)


def load_dataset_split(dataset_name: str, dataset_config: str | None, split: str):
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split)
    return load_dataset(dataset_name, split=split)


def extract_ground_truth(raw_answer: Any, mode: str, dataset_name: str) -> str:
    text = str(raw_answer or "")
    if mode == "gsm8k" or (mode == "auto" and "gsm8k" in dataset_name.lower()):
        if "####" in text:
            return text.split("####")[-1].strip()
    return text.strip()


def render_prompt(problem: str, template: Template | None) -> str:
    if template is None:
        return problem
    return template.render(content=problem)


def load_model_and_tokenizer(args: argparse.Namespace):
    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    if args.lora_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_path)
    model.eval()
    return model, tokenizer


def batched(records: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def majority_accuracy(candidates: list[dict[str, Any]], ground_truth: str) -> float:
    raw_answers = [str(candidate.get("boxed_answer", "") or "").strip() for candidate in candidates]
    raw_answers = [answer for answer in raw_answers if answer]
    if not raw_answers:
        return 0.0
    normalized_to_raw = {}
    normalized_answers = []
    for answer in raw_answers:
        normalized = normalize_text(answer)
        normalized_answers.append(normalized)
        normalized_to_raw.setdefault(normalized, answer)
    counts = Counter(normalized_answers)
    majority = normalized_to_raw[counts.most_common(1)[0][0]]
    return 1.0 if grade_prediction(majority, ground_truth) else 0.0


def select_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        candidates,
        key=lambda item: (
            float(item["diagnostics"].get("train_reward_v1", 0.0) or 0.0),
            float(item["diagnostics"].get("accuracy", 0.0) or 0.0),
            -float(item.get("response_length", 0.0) or 0.0),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-path", default="")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--dataset-name", default="hiyouga/math12k")
    parser.add_argument("--dataset-config", default="")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--prompt-key", default="problem")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--answer-extractor", choices=["auto", "none", "gsm8k"], default="auto")
    parser.add_argument("--format-prompt", default="examples/format_prompt/math.jinja")
    parser.add_argument("--reward-file", default="examples/reward_function/math.py")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--report-file", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--postprocess-stop-markers",
        action="store_true",
        help="Cut generated text at leaked chat turn markers such as Human:/Assistant:.",
    )
    parser.add_argument(
        "--stop-after-boxed",
        action="store_true",
        help="Cut generated text immediately after the final complete \\boxed{...} answer.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    template = None
    if args.format_prompt:
        template = Template(Path(args.format_prompt).read_text(encoding="utf-8"))

    dataset = load_dataset_split(args.dataset_name, args.dataset_config or None, args.dataset_split)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    records = [dict(row) for row in dataset]

    model, tokenizer = load_model_and_tokenizer(args)
    output_path = Path(args.output_file)
    report_path = Path(args.report_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    totals = {
        "count": 0,
        "greedy_accuracy": 0.0,
        "best_of_N_accuracy": 0.0,
        "pass_at_k": 0.0,
        "maj_at_k": 0.0,
        "format_at_k": 0.0,
        "selected_train_reward_v1": 0.0,
        "selected_diagnostic_score_v2_lite": 0.0,
        "greedy_robust_accuracy": 0.0,
        "best_of_N_robust_accuracy": 0.0,
        "robust_pass_at_k": 0.0,
        "verifier_false_negative_candidate_rate": 0.0,
        "robust_extract_failure_rate": 0.0,
        "hit_length_cap_rate": 0.0,
        "unclosed_think_rate": 0.0,
        "missing_boxed_answer_rate": 0.0,
        "prompt_contamination_rate": 0.0,
        "mean_response_length": 0.0,
    }

    do_sample = args.n > 1 or args.temperature > 0.0
    with output_path.open("w", encoding="utf-8") as f:
        for batch in batched(records, args.batch_size):
            prompts = [render_prompt(str(item[args.prompt_key]), template) for item in batch]
            ground_truths = [
                extract_ground_truth(item[args.answer_key], args.answer_extractor, args.dataset_name) for item in batch
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            input_width = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=args.top_p if do_sample else None,
                    num_return_sequences=args.n,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            decoded = []
            for output_ids in outputs:
                response_ids = output_ids[input_width:]
                raw_response = tokenizer.decode(response_ids, skip_special_tokens=True)
                response = postprocess_generation(
                    raw_response,
                    cut_turn_markers=args.postprocess_stop_markers,
                    stop_after_boxed=args.stop_after_boxed,
                )
                decoded.append((response, raw_response))

            for item_idx, item in enumerate(batch):
                start = item_idx * args.n
                responses = decoded[start : start + args.n]
                ground_truth = ground_truths[item_idx]
                candidates = []
                for candidate_idx, (response, raw_response) in enumerate(responses):
                    response_length = len(tokenizer.encode(response, add_special_tokens=False))
                    diagnostics = diagnostic_score_v2_lite(
                        response=response,
                        ground_truth=ground_truth,
                        reward_file=args.reward_file,
                        response_length=response_length,
                        max_response_length=args.max_new_tokens,
                    )
                    candidates.append(
                        {
                            "candidate_index": candidate_idx,
                            "response": response,
                            **({"raw_response": raw_response} if raw_response != response else {}),
                            "response_length": response_length,
                            "boxed_answer": extract_boxed_content_fallback(response),
                            "diagnostics": diagnostics,
                        }
                    )

                selected = select_best_candidate(candidates)
                greedy = candidates[0]
                pass_at_k = 1.0 if any(float(c["diagnostics"]["accuracy"]) > 0.0 for c in candidates) else 0.0
                robust_pass_at_k = (
                    1.0 if any(float(c["diagnostics"].get("robust_accuracy", 0.0) or 0.0) > 0.0 for c in candidates) else 0.0
                )
                maj_at_k = majority_accuracy(candidates, ground_truth)
                format_at_k = sum(float(c["diagnostics"]["format_rate"]) for c in candidates) / max(len(candidates), 1)
                mean_len = sum(float(c["response_length"]) for c in candidates) / max(len(candidates), 1)

                totals["count"] += 1
                totals["greedy_accuracy"] += float(greedy["diagnostics"]["accuracy"])
                totals["best_of_N_accuracy"] += float(selected["diagnostics"]["accuracy"])
                totals["pass_at_k"] += pass_at_k
                totals["maj_at_k"] += maj_at_k
                totals["format_at_k"] += format_at_k
                totals["selected_train_reward_v1"] += float(selected["diagnostics"]["train_reward_v1"])
                totals["selected_diagnostic_score_v2_lite"] += float(
                    selected["diagnostics"]["diagnostic_score_v2_lite"]
                )
                totals["greedy_robust_accuracy"] += float(greedy["diagnostics"].get("robust_accuracy", 0.0) or 0.0)
                totals["best_of_N_robust_accuracy"] += float(
                    selected["diagnostics"].get("robust_accuracy", 0.0) or 0.0
                )
                totals["robust_pass_at_k"] += robust_pass_at_k
                totals["verifier_false_negative_candidate_rate"] += float(
                    greedy["diagnostics"].get("verifier_false_negative_candidate", 0.0) or 0.0
                )
                totals["robust_extract_failure_rate"] += float(
                    greedy["diagnostics"].get("robust_extract_failure", 0.0) or 0.0
                )
                totals["hit_length_cap_rate"] += float(greedy["diagnostics"].get("hit_length_cap", 0.0) or 0.0)
                totals["unclosed_think_rate"] += float(greedy["diagnostics"].get("unclosed_think", 0.0) or 0.0)
                totals["missing_boxed_answer_rate"] += float(
                    greedy["diagnostics"].get("missing_boxed_answer", 0.0) or 0.0
                )
                totals["prompt_contamination_rate"] += float(
                    greedy["diagnostics"].get("prompt_contamination", 0.0) or 0.0
                )
                totals["mean_response_length"] += mean_len

                f.write(
                    json.dumps(
                        {
                            "problem": item[args.prompt_key],
                            "answer": ground_truth,
                            "prompt": prompts[item_idx],
                            "candidates": candidates,
                            "selected_candidate_index": selected["candidate_index"],
                            "metrics": {
                                "greedy_accuracy": float(greedy["diagnostics"]["accuracy"]),
                                "best_of_N_accuracy": float(selected["diagnostics"]["accuracy"]),
                                "pass_at_k": pass_at_k,
                                "robust_pass_at_k": robust_pass_at_k,
                                "maj_at_k": maj_at_k,
                                "format_at_k": format_at_k,
                                "mean_response_length": mean_len,
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    count = max(int(totals["count"]), 1)
    report = {
        "model_path": args.model_path,
        "lora_path": args.lora_path or None,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config or None,
        "dataset_split": args.dataset_split,
        "limit": args.limit,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "format_prompt": args.format_prompt,
        "postprocess_stop_markers": args.postprocess_stop_markers,
        "stop_after_boxed": args.stop_after_boxed,
        "count": totals["count"],
        "greedy_accuracy": totals["greedy_accuracy"] / count,
        "best_of_N_accuracy": totals["best_of_N_accuracy"] / count,
        "pass_at_k": totals["pass_at_k"] / count,
        "maj_at_k": totals["maj_at_k"] / count,
        "format_at_k": totals["format_at_k"] / count,
        "selected_train_reward_v1": totals["selected_train_reward_v1"] / count,
        "selected_diagnostic_score_v2_lite": totals["selected_diagnostic_score_v2_lite"] / count,
        "greedy_robust_accuracy": totals["greedy_robust_accuracy"] / count,
        "best_of_N_robust_accuracy": totals["best_of_N_robust_accuracy"] / count,
        "robust_pass_at_k": totals["robust_pass_at_k"] / count,
        "verifier_false_negative_candidate_rate": totals["verifier_false_negative_candidate_rate"] / count,
        "robust_extract_failure_rate": totals["robust_extract_failure_rate"] / count,
        "hit_length_cap_rate": totals["hit_length_cap_rate"] / count,
        "unclosed_think_rate": totals["unclosed_think_rate"] / count,
        "missing_boxed_answer_rate": totals["missing_boxed_answer_rate"] / count,
        "prompt_contamination_rate": totals["prompt_contamination_rate"] / count,
        "mean_response_length": totals["mean_response_length"] / count,
        "output_file": str(output_path),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
