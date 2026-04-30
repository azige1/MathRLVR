#!/usr/bin/env python3
"""Build an auditable RLVR dataset profile for math reasoning experiments.

The goal is not to create SFT data. For RLVR, the data contract is:

  problem -> actor rollout prompt
  answer  -> verifier-only gold label

This profiler makes that contract explicit and produces artifacts that are
useful for reports and interviews: topic/difficulty distributions, answer
surface types, prompt length stats, verifier-risk tags, leakage checks,
stratified samples, and deterministic eval manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from datasets import load_dataset
from jinja2 import Template


TOPIC_PATTERNS: list[tuple[str, list[str]]] = [
    ("geometry", ["triangle", "circle", "angle", "area", "perimeter", "radius", "diameter", "polygon", "cube", "sphere"]),
    ("number_theory", ["integer", "prime", "divisible", "modulo", "remainder", "factor", "gcd", "lcm", "congruent"]),
    ("combinatorics", ["how many", "ways", "arrange", "permutation", "combination", "choose", "distinct", "probability"]),
    ("algebra", ["equation", "solve", "polynomial", "quadratic", "root", "system", "value of x", "function"]),
    ("sequence", ["sequence", "recurrence", "arithmetic progression", "geometric progression", "term"]),
    ("probability", ["probability", "random", "expected", "dice", "coin"]),
    ("arithmetic", ["sum", "product", "average", "mean", "ratio", "percent", "fraction"]),
]


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


def numeric_stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p95": None, "p99": None, "max": None, "mean": None}
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
    if compact.lower() in {"none", "no solution", "undefined", "does not exist"}:
        return "none_or_no_solution"
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
    if any(sym in compact for sym in ["\\pi", "pi", "^\\circ", "degree"]):
        return "symbolic_constant"
    if re.search(r"[a-zA-Z]", compact):
        return "text_or_symbolic"
    return "other_expression"


def classify_topic(problem: Any) -> str:
    text = normalize_text(problem)
    hits: Counter[str] = Counter()
    for topic, patterns in TOPIC_PATTERNS:
        for pattern in patterns:
            if pattern in text:
                hits[topic] += 1
    if not hits:
        return "other"
    return hits.most_common(1)[0][0]


def render_prompt(problem: Any, template: Template | None) -> str:
    content = str(problem or "")
    if template is None:
        return content
    return template.render(content=content)


def load_split(dataset_name: str, split: str, limit: int) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split)
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return [dict(row) for row in ds]


def load_tokenizer(tokenizer_path: str | None):
    if not tokenizer_path:
        return None, None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, trust_remote_code=False), None
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, repr(exc)


def token_length(tokenizer: Any, text: str) -> int | None:
    if tokenizer is None:
        return None
    return len(tokenizer.encode(text, add_special_tokens=False))


def classify_difficulty(prompt_chars: int, prompt_tokens: int | None, answer_type: str, problem: Any) -> str:
    length = prompt_tokens if prompt_tokens is not None else prompt_chars
    hard_types = {"sqrt_expression", "power_expression", "text_or_symbolic", "other_expression", "none_or_no_solution"}
    text = normalize_text(problem)
    complexity = 0
    complexity += len(re.findall(r"\\frac|sqrt|\\sqrt|\^", text))
    complexity += len(re.findall(r"\$[^$]+\$", str(problem or "")))
    complexity += 1 if answer_type in hard_types else 0
    complexity += 1 if any(word in text for word in ["prove", "minimum", "maximum", "equilateral", "modulo", "probability"]) else 0

    if length >= 900 or complexity >= 5:
        return "hard"
    if length >= 500 or complexity >= 3:
        return "medium"
    return "easy"


def verifier_risk_tags(answer: Any, prompt_chars: int, prompt_tokens: int | None, max_prompt_length: int) -> list[str]:
    text = str(answer or "").strip()
    compact = text.replace(" ", "")
    lower = compact.lower()
    tags: list[str] = []
    answer_type = classify_answer(text)

    if not text:
        tags.append("empty_gold_answer")
    if lower in {"none", "nosolution", "undefined", "doesnotexist"}:
        tags.append("none_or_no_solution_gold")
    if answer_type in {"text_or_symbolic", "other_expression", "symbolic_constant"}:
        tags.append("symbolic_or_text_gold")
    if "," in text or " or " in text.lower():
        tags.append("possible_multi_answer")
    if len(text) > 50:
        tags.append("long_gold_answer")
    if "\\begin" in text or "\\left" in text or "\\right" in text:
        tags.append("complex_latex_gold")
    if prompt_tokens is not None and prompt_tokens > max_prompt_length:
        tags.append("over_max_prompt_tokens")
    elif prompt_tokens is None and prompt_chars > max_prompt_length * 4:
        tags.append("possibly_overlong_prompt")
    return tags or ["clean"]


def row_manifest(
    row: dict[str, Any],
    split: str,
    index: int,
    prompt_key: str,
    answer_key: str,
    template: Template | None,
    tokenizer: Any,
    max_prompt_length: int,
) -> dict[str, Any]:
    problem = row.get(prompt_key, "")
    answer = row.get(answer_key, "")
    rendered = render_prompt(problem, template)
    prompt_tokens = token_length(tokenizer, rendered)
    answer_type = classify_answer(answer)
    topic = classify_topic(problem)
    difficulty = classify_difficulty(len(rendered), prompt_tokens, answer_type, problem)
    tags = verifier_risk_tags(answer, len(rendered), prompt_tokens, max_prompt_length)
    problem_norm = normalize_text(problem)
    answer_norm = normalize_text(answer)
    return {
        "id": f"{split}-{index:06d}",
        "split": split,
        "index": index,
        "problem_hash": stable_hash(problem_norm),
        "pair_hash": stable_hash(problem_norm + "\n" + answer_norm),
        "topic": topic,
        "difficulty": difficulty,
        "answer_type": answer_type,
        "prompt_chars": len(rendered),
        "prompt_tokens": prompt_tokens,
        "answer_chars": len(str(answer or "")),
        "verifier_risk_tags": tags,
        "problem": problem,
        "answer": answer,
    }


def summarize_manifest(rows: list[dict[str, Any]], max_prompt_length: int) -> dict[str, Any]:
    prompt_chars = [int(r["prompt_chars"]) for r in rows]
    prompt_tokens = [int(r["prompt_tokens"]) for r in rows if r["prompt_tokens"] is not None]
    answer_chars = [int(r["answer_chars"]) for r in rows]
    risk_counter: Counter[str] = Counter(tag for row in rows for tag in row["verifier_risk_tags"])
    clean_count = risk_counter.get("clean", 0)
    return {
        "num_rows": len(rows),
        "topic_distribution": dict(Counter(r["topic"] for r in rows).most_common()),
        "difficulty_distribution": dict(Counter(r["difficulty"] for r in rows).most_common()),
        "answer_type_distribution": dict(Counter(r["answer_type"] for r in rows).most_common()),
        "verifier_risk_distribution": dict(risk_counter.most_common()),
        "verifier_ready_rate": clean_count / len(rows) if rows else 0.0,
        "lengths": {
            "prompt_chars": numeric_stats(prompt_chars),
            "prompt_tokens": numeric_stats(prompt_tokens),
            "answer_chars": numeric_stats(answer_chars),
            "max_prompt_length": max_prompt_length,
            "over_max_prompt_tokens": sum(
                1 for r in rows if r["prompt_tokens"] is not None and int(r["prompt_tokens"]) > max_prompt_length
            ),
        },
    }


def duplicate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    problem_counts = Counter(r["problem_hash"] for r in rows)
    pair_counts = Counter(r["pair_hash"] for r in rows)
    return {
        "duplicate_problem_rows": sum(c - 1 for c in problem_counts.values() if c > 1),
        "duplicate_pair_rows": sum(c - 1 for c in pair_counts.values() if c > 1),
        "duplicate_problem_examples": [
            {"problem_hash": h, "count": c}
            for h, c in problem_counts.most_common()
            if c > 1
        ][:20],
        "duplicate_pair_examples": [
            {"pair_hash": h, "count": c}
            for h, c in pair_counts.most_common()
            if c > 1
        ][:20],
    }


def make_stratified_samples(rows: list[dict[str, Any]], rng: random.Random, per_bucket: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[f"topic:{row['topic']}"].append(row)
        buckets[f"difficulty:{row['difficulty']}"].append(row)
        for tag in row["verifier_risk_tags"]:
            if tag != "clean":
                buckets[f"risk:{tag}"].append(row)

    samples: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bucket_name in sorted(buckets):
        candidates = buckets[bucket_name]
        if not candidates:
            continue
        picked = rng.sample(candidates, k=min(per_bucket, len(candidates)))
        for row in picked:
            key = f"{bucket_name}:{row['id']}"
            if key in seen:
                continue
            seen.add(key)
            sample = dict(row)
            sample["bucket"] = bucket_name
            samples.append(sample)
    return samples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    parser.add_argument("--eval-limit", type=int, default=500)
    parser.add_argument("--sample-per-bucket", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-report", default="results/math12k_data_profile_report.json")
    parser.add_argument("--manifest-output", default="results/math12k_data_manifest.jsonl")
    parser.add_argument("--eval-subset-output", default="results/math12k_eval_subset_manifest.jsonl")
    parser.add_argument("--sample-output", default="results/math12k_stratified_data_samples.jsonl")
    args = parser.parse_args()

    template = None
    if args.format_prompt:
        template = Template(Path(args.format_prompt).read_text(encoding="utf-8"))

    tokenizer, tokenizer_error = load_tokenizer(args.tokenizer_path or None)
    train_records = load_split(args.dataset_name, args.train_split, args.limit_train)
    val_records = load_split(args.dataset_name, args.val_split, args.limit_val)

    train_rows = [
        row_manifest(row, args.train_split, i, args.prompt_key, args.answer_key, template, tokenizer, args.max_prompt_length)
        for i, row in enumerate(train_records)
    ]
    val_rows = [
        row_manifest(row, args.val_split, i, args.prompt_key, args.answer_key, template, tokenizer, args.max_prompt_length)
        for i, row in enumerate(val_records)
    ]

    train_hashes = {r["problem_hash"] for r in train_rows}
    val_hashes = {r["problem_hash"] for r in val_rows}
    overlap = sorted(train_hashes & val_hashes)

    rng = random.Random(args.seed)
    all_rows = train_rows + val_rows
    eval_subset = val_rows[: min(args.eval_limit, len(val_rows))]
    samples = make_stratified_samples(all_rows, rng, args.sample_per_bucket)

    report = {
        "dataset_name": args.dataset_name,
        "data_contract": {
            "actor_input": args.prompt_key,
            "verifier_only_gold": args.answer_key,
            "format_prompt": args.format_prompt,
            "max_prompt_length": args.max_prompt_length,
            "note": "Gold answers are not used for teacher forcing; they are labels for rule-based verifier rewards.",
        },
        "tokenizer_path": args.tokenizer_path or None,
        "tokenizer_error": tokenizer_error,
        "train": summarize_manifest(train_rows, args.max_prompt_length),
        "val": summarize_manifest(val_rows, args.max_prompt_length),
        "duplicates": {
            "train": duplicate_summary(train_rows),
            "val": duplicate_summary(val_rows),
        },
        "train_val_leakage": {
            "train_unique_problem_hashes": len(train_hashes),
            "val_unique_problem_hashes": len(val_hashes),
            "overlap_problem_count": len(overlap),
            "overlap_problem_hashes": overlap[:50],
        },
        "eval_subset": {
            "split": args.val_split,
            "num_rows": len(eval_subset),
            "selection": "deterministic first N rows from the validation split, matching the P0 eval protocol.",
            "topic_distribution": dict(Counter(r["topic"] for r in eval_subset).most_common()),
            "difficulty_distribution": dict(Counter(r["difficulty"] for r in eval_subset).most_common()),
            "answer_type_distribution": dict(Counter(r["answer_type"] for r in eval_subset).most_common()),
            "verifier_risk_distribution": dict(Counter(tag for row in eval_subset for tag in row["verifier_risk_tags"]).most_common()),
        },
        "artifacts": {
            "manifest_output": args.manifest_output,
            "eval_subset_output": args.eval_subset_output,
            "sample_output": args.sample_output,
        },
    }

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(Path(args.manifest_output), all_rows)
    write_jsonl(Path(args.eval_subset_output), eval_subset)
    write_jsonl(Path(args.sample_output), samples)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {args.output_report}")
    print(f"Saved manifest: {args.manifest_output}")
    print(f"Saved eval subset: {args.eval_subset_output}")
    print(f"Saved stratified samples: {args.sample_output}")


if __name__ == "__main__":
    main()
