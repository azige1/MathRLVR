#!/usr/bin/env python3
"""Inspect Math12K-style data used by the EasyR1 reasoning RL experiments.

This script is intentionally lightweight: it does not load a model. It checks
field validity, prompt rendering, duplicates/leakage, length distribution, and
answer surface types so the RLVR setup is auditable before training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from jinja2 import Template


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return float(ordered[lo] * (1 - weight) + ordered[hi] * weight)


def length_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def classify_answer(answer: Any) -> str:
    text = str(answer or "").strip()
    compact = text.replace(" ", "")
    if not text:
        return "empty"
    if re.fullmatch(r"[-+]?\d+", compact):
        return "integer"
    if re.fullmatch(r"[-+]?\d+\.\d+", compact):
        return "decimal"
    if re.fullmatch(r"[-+]?\d+/\d+", compact):
        return "plain_fraction"
    if "\\frac" in compact:
        return "latex_fraction"
    if "\\sqrt" in compact or "sqrt" in compact:
        return "sqrt_expression"
    if "^" in compact or "**" in compact:
        return "power_expression"
    if any(sym in compact for sym in ["\\pi", "pi", "°", "^\\circ"]):
        return "symbolic_constant"
    if re.search(r"[a-zA-Z]", compact):
        return "text_or_symbolic"
    return "other_expression"


def load_split(dataset_name: str, split: str, limit: int) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in ds]


def render_prompt(problem: Any, template: Template | None) -> str:
    content = str(problem or "")
    if template is None:
        return content
    return template.render(content=content)


def tokenize_lengths(prompts: list[str], tokenizer_path: str | None) -> tuple[list[int] | None, str | None]:
    if not tokenizer_path:
        return None, None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=False)
        return [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts], None
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, repr(exc)


def inspect_records(
    records: list[dict[str, Any]],
    split_name: str,
    prompt_key: str,
    answer_key: str,
    template: Template | None,
    max_prompt_length: int,
    tokenizer_path: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    missing_prompt = 0
    missing_answer = 0
    empty_prompt = 0
    empty_answer = 0
    normalized_problem_counter: Counter[str] = Counter()
    normalized_pair_counter: Counter[tuple[str, str]] = Counter()
    answer_types: Counter[str] = Counter()
    problem_char_lengths: list[int] = []
    prompt_char_lengths: list[int] = []
    answer_char_lengths: list[int] = []
    rendered_prompts: list[str] = []

    for row in records:
        if prompt_key not in row:
            missing_prompt += 1
        if answer_key not in row:
            missing_answer += 1

        problem = row.get(prompt_key, "")
        answer = row.get(answer_key, "")
        problem_norm = normalize_text(problem)
        answer_norm = normalize_text(answer)

        if not problem_norm:
            empty_prompt += 1
        if not answer_norm:
            empty_answer += 1

        prompt = render_prompt(problem, template)
        rendered_prompts.append(prompt)

        normalized_problem_counter[problem_norm] += 1
        normalized_pair_counter[(problem_norm, answer_norm)] += 1
        answer_types[classify_answer(answer)] += 1
        problem_char_lengths.append(len(str(problem or "")))
        prompt_char_lengths.append(len(prompt))
        answer_char_lengths.append(len(str(answer or "")))

    prompt_token_lengths, tokenizer_error = tokenize_lengths(rendered_prompts, tokenizer_path)
    overlong_prompt_count = None
    if prompt_token_lengths is not None:
        overlong_prompt_count = sum(1 for length in prompt_token_lengths if length > max_prompt_length)

    duplicate_problems = sum(count - 1 for count in normalized_problem_counter.values() if count > 1)
    duplicate_pairs = sum(count - 1 for count in normalized_pair_counter.values() if count > 1)

    duplicate_examples: list[dict[str, Any]] = []
    seen_duplicate_hashes: set[str] = set()
    for row in records:
        problem_norm = normalize_text(row.get(prompt_key, ""))
        answer_norm = normalize_text(row.get(answer_key, ""))
        pair = (problem_norm, answer_norm)
        if normalized_pair_counter[pair] <= 1:
            continue
        h = stable_hash(problem_norm + "\n" + answer_norm)
        if h in seen_duplicate_hashes:
            continue
        seen_duplicate_hashes.add(h)
        duplicate_examples.append(
            {
                "split": split_name,
                "hash": h,
                "count": normalized_pair_counter[pair],
                "problem": row.get(prompt_key, ""),
                "answer": row.get(answer_key, ""),
            }
        )
        if len(duplicate_examples) >= 10:
            break

    report = {
        "split": split_name,
        "num_rows": len(records),
        "field_checks": {
            "missing_prompt_key": missing_prompt,
            "missing_answer_key": missing_answer,
            "empty_prompt": empty_prompt,
            "empty_answer": empty_answer,
        },
        "duplicates": {
            "duplicate_problem_rows": duplicate_problems,
            "duplicate_problem_answer_rows": duplicate_pairs,
        },
        "lengths": {
            "problem_chars": length_stats(problem_char_lengths),
            "rendered_prompt_chars": length_stats(prompt_char_lengths),
            "answer_chars": length_stats(answer_char_lengths),
            "rendered_prompt_tokens": length_stats(prompt_token_lengths or []),
            "over_max_prompt_length": overlong_prompt_count,
            "max_prompt_length": max_prompt_length,
            "tokenizer_error": tokenizer_error,
        },
        "answer_types": dict(answer_types.most_common()),
    }
    return report, duplicate_examples


def make_samples(
    records: list[dict[str, Any]],
    split_name: str,
    prompt_key: str,
    answer_key: str,
    template: Template | None,
    rng: random.Random,
    n: int,
) -> list[dict[str, Any]]:
    if not records or n <= 0:
        return []
    picked = rng.sample(records, k=min(n, len(records)))
    samples = []
    for row in picked:
        problem = row.get(prompt_key, "")
        answer = row.get(answer_key, "")
        samples.append(
            {
                "split": split_name,
                "problem": problem,
                "answer": answer,
                "answer_type": classify_answer(answer),
                "rendered_prompt": render_prompt(problem, template),
            }
        )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="hiyouga/math12k")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--prompt-key", default="problem")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--format-prompt", default="examples/format_prompt/math.jinja")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--limit-train", type=int, default=-1)
    parser.add_argument("--limit-val", type=int, default=-1)
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-report", default="results/math12k_data_inspection_report.json")
    parser.add_argument("--sample-output", default="results/math12k_data_samples.jsonl")
    args = parser.parse_args()

    template = None
    if args.format_prompt:
        template = Template(Path(args.format_prompt).read_text(encoding="utf-8"))

    train_records = load_split(args.dataset_name, args.train_split, args.limit_train)
    val_records = load_split(args.dataset_name, args.val_split, args.limit_val)

    train_report, train_duplicate_examples = inspect_records(
        train_records,
        args.train_split,
        args.prompt_key,
        args.answer_key,
        template,
        args.max_prompt_length,
        args.tokenizer_path or None,
    )
    val_report, val_duplicate_examples = inspect_records(
        val_records,
        args.val_split,
        args.prompt_key,
        args.answer_key,
        template,
        args.max_prompt_length,
        args.tokenizer_path or None,
    )

    train_problem_hashes = {stable_hash(normalize_text(row.get(args.prompt_key, ""))) for row in train_records}
    val_problem_hashes = {stable_hash(normalize_text(row.get(args.prompt_key, ""))) for row in val_records}
    overlap_hashes = train_problem_hashes & val_problem_hashes

    overlap_examples = []
    if overlap_hashes:
        val_by_hash = {}
        for row in val_records:
            h = stable_hash(normalize_text(row.get(args.prompt_key, "")))
            val_by_hash.setdefault(h, row)
        for h in sorted(overlap_hashes)[:20]:
            row = val_by_hash[h]
            overlap_examples.append(
                {
                    "hash": h,
                    "val_problem": row.get(args.prompt_key, ""),
                    "val_answer": row.get(args.answer_key, ""),
                }
            )

    rng = random.Random(args.seed)
    samples = (
        make_samples(train_records, args.train_split, args.prompt_key, args.answer_key, template, rng, args.sample_size)
        + make_samples(val_records, args.val_split, args.prompt_key, args.answer_key, template, rng, args.sample_size)
    )

    report = {
        "dataset_name": args.dataset_name,
        "prompt_key": args.prompt_key,
        "answer_key": args.answer_key,
        "format_prompt": args.format_prompt,
        "tokenizer_path": args.tokenizer_path or None,
        "train": train_report,
        "val": val_report,
        "train_val_leakage": {
            "train_unique_problem_hashes": len(train_problem_hashes),
            "val_unique_problem_hashes": len(val_problem_hashes),
            "overlap_problem_count": len(overlap_hashes),
            "overlap_examples": overlap_examples,
        },
        "duplicate_examples": train_duplicate_examples + val_duplicate_examples,
    }

    report_path = Path(args.output_report)
    sample_path = Path(args.sample_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with sample_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {report_path}")
    print(f"Saved samples: {sample_path}")


if __name__ == "__main__":
    main()
