#!/usr/bin/env python3
"""Evaluate a base/instruct model on math12k-style data with the EasyR1 math reward."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from jinja2 import Template
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_reward_function(reward_file: str):
    reward_path = Path(reward_file).resolve()
    spec = importlib.util.spec_from_file_location("easy_r1_math_reward", reward_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reward file: {reward_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def batched(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def render_prompt(problem: str, format_prompt: Template | None) -> str:
    if format_prompt is None:
        return problem
    return format_prompt.render(content=problem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-name", default="hiyouga/math12k")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--prompt-key", default="problem")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--summary-file", required=True)
    parser.add_argument("--format-prompt", default="examples/format_prompt/math.jinja")
    parser.add_argument("--reward-file", default="examples/reward_function/math.py")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", default="false")
    args = parser.parse_args()

    trust_remote_code = str(args.trust_remote_code).lower() in {"1", "true", "yes"}
    output_path = Path(args.output_file)
    summary_path = Path(args.summary_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    format_template = None
    if args.format_prompt:
        format_template = Template(Path(args.format_prompt).read_text(encoding="utf-8"))

    reward_fn = load_reward_function(args.reward_file)

    dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    records = [dict(row) for row in dataset]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
    )
    model.eval()

    do_sample = args.temperature > 0
    totals = {"overall": 0.0, "format": 0.0, "accuracy": 0.0, "count": 0}

    with output_path.open("w", encoding="utf-8") as f:
        for batch in batched(records, args.batch_size):
            prompts = [render_prompt(str(item[args.prompt_key]), format_template) for item in batch]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            input_width = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=args.top_p if do_sample else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            responses = []
            for output_ids in outputs:
                response_ids = output_ids[input_width:]
                responses.append(tokenizer.decode(response_ids, skip_special_tokens=True))

            reward_inputs = [
                {
                    "response": response,
                    "response_length": len(tokenizer.encode(response)),
                    "ground_truth": str(item[args.answer_key]),
                }
                for item, response in zip(batch, responses)
            ]
            scores = reward_fn(reward_inputs)

            for item, prompt, response, score in zip(batch, prompts, responses, scores):
                for key in ["overall", "format", "accuracy"]:
                    totals[key] += float(score.get(key, 0.0) or 0.0)
                totals["count"] += 1
                f.write(
                    json.dumps(
                        {
                            "problem": item[args.prompt_key],
                            "answer": item[args.answer_key],
                            "prompt": prompt,
                            "response": response,
                            "score": score,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    count = max(int(totals["count"]), 1)
    summary = {
        "model_path": args.model_path,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "limit": args.limit,
        "count": totals["count"],
        "overall": totals["overall"] / count,
        "format": totals["format"] / count,
        "accuracy": totals["accuracy"] / count,
        "output_file": str(output_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
