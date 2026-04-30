#!/usr/bin/env python3
"""Build verifier-aware clean JSONL splits for Math RLVR.

The output is a local Hugging Face JSON dataset directory:

  data/math12k_verifier_clean/train.jsonl
  data/math12k_verifier_clean/test.jsonl

EasyR1 can consume it with:

  TRAIN_FILES=data/math12k_verifier_clean@train
  VAL_FILES=data/math12k_verifier_clean@test
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from jinja2 import Template

from profile_rlvr_dataset import (
    classify_answer,
    row_manifest,
    summarize_manifest,
    load_tokenizer,
)


DEFAULT_DROP_TAGS = {
    "empty_gold_answer",
    "none_or_no_solution_gold",
    "symbolic_or_text_gold",
    "possible_multi_answer",
    "complex_latex_gold",
    "long_gold_answer",
    "over_max_prompt_tokens",
    "possibly_overlong_prompt",
}


def load_split(dataset_name: str, split: str) -> list[dict[str, Any]]:
    return [dict(row) for row in load_dataset(dataset_name, split=split)]


def keep_row(manifest: dict[str, Any], policy: str, drop_tags: set[str]) -> bool:
    tags = set(manifest["verifier_risk_tags"])
    if policy == "strict_clean":
        return tags == {"clean"}
    if policy == "drop_high_risk":
        return not (tags & drop_tags)
    raise ValueError(f"Unknown clean policy: {policy}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_split(
    records: list[dict[str, Any]],
    split: str,
    prompt_key: str,
    answer_key: str,
    template: Template | None,
    tokenizer: Any,
    max_prompt_length: int,
    policy: str,
    drop_tags: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    kept_manifest: list[dict[str, Any]] = []
    dropped_manifest: list[dict[str, Any]] = []

    for index, row in enumerate(records):
        manifest = row_manifest(
            row=row,
            split=split,
            index=index,
            prompt_key=prompt_key,
            answer_key=answer_key,
            template=template,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
        )
        if keep_row(manifest, policy, drop_tags):
            kept.append(
                {
                    prompt_key: row.get(prompt_key, ""),
                    answer_key: row.get(answer_key, ""),
                    "source_split": split,
                    "source_index": index,
                    "topic": manifest["topic"],
                    "difficulty": manifest["difficulty"],
                    "answer_type": manifest["answer_type"],
                    "prompt_tokens": manifest["prompt_tokens"],
                    "verifier_risk_tags": manifest["verifier_risk_tags"],
                }
            )
            kept_manifest.append(manifest)
        else:
            dropped_manifest.append(manifest)

    return kept, kept_manifest, dropped_manifest


def tag_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(tag for row in rows for tag in row["verifier_risk_tags"]).most_common())


def answer_type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(classify_answer(row.get("answer", "")) for row in rows).most_common())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="hiyouga/math12k")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--prompt-key", default="problem")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument("--format-prompt", default="examples/format_prompt/math_concise_final.jinja")
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--policy", choices=["strict_clean", "drop_high_risk"], default="strict_clean")
    parser.add_argument("--output-dir", default="data/math12k_verifier_clean")
    parser.add_argument("--report-file", default="results/math12k_verifier_clean_dataset_report.json")
    parser.add_argument("--manifest-file", default="results/math12k_verifier_clean_manifest.jsonl")
    args = parser.parse_args()

    template = None
    if args.format_prompt:
        template = Template(Path(args.format_prompt).read_text(encoding="utf-8"))

    tokenizer, tokenizer_error = load_tokenizer(args.tokenizer_path or None)
    train_records = load_split(args.dataset_name, args.train_split)
    val_records = load_split(args.dataset_name, args.val_split)

    train_rows, train_manifest, train_dropped = build_split(
        records=train_records,
        split=args.train_split,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
        template=template,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        policy=args.policy,
        drop_tags=DEFAULT_DROP_TAGS,
    )
    val_rows, val_manifest, val_dropped = build_split(
        records=val_records,
        split=args.val_split,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
        template=template,
        tokenizer=tokenizer,
        max_prompt_length=args.max_prompt_length,
        policy=args.policy,
        drop_tags=DEFAULT_DROP_TAGS,
    )

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", val_rows)
    write_jsonl(Path(args.manifest_file), train_manifest + val_manifest)

    report = {
        "dataset_name": args.dataset_name,
        "policy": args.policy,
        "drop_tags": sorted(DEFAULT_DROP_TAGS),
        "format_prompt": args.format_prompt,
        "tokenizer_path": args.tokenizer_path or None,
        "tokenizer_error": tokenizer_error,
        "max_prompt_length": args.max_prompt_length,
        "output_dir": str(output_dir),
        "train": {
            "original_rows": len(train_records),
            "kept_rows": len(train_rows),
            "dropped_rows": len(train_dropped),
            "keep_rate": len(train_rows) / max(len(train_records), 1),
            "kept_summary": summarize_manifest(train_manifest, args.max_prompt_length),
            "dropped_risk_distribution": tag_counts(train_dropped),
            "dropped_answer_type_distribution": answer_type_counts(train_dropped),
        },
        "val": {
            "original_rows": len(val_records),
            "kept_rows": len(val_rows),
            "dropped_rows": len(val_dropped),
            "keep_rate": len(val_rows) / max(len(val_records), 1),
            "kept_summary": summarize_manifest(val_manifest, args.max_prompt_length),
            "dropped_risk_distribution": tag_counts(val_dropped),
            "dropped_answer_type_distribution": answer_type_counts(val_dropped),
        },
        "data_contract": {
            "actor_input": args.prompt_key,
            "verifier_only_gold": args.answer_key,
            "note": "Gold answers remain verifier-only labels; no SFT target is created.",
        },
    }

    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved clean train: {output_dir / 'train.jsonl'}")
    print(f"Saved clean val: {output_dir / 'test.jsonl'}")
    print(f"Saved manifest: {args.manifest_file}")
    print(f"Saved report: {args.report_file}")


if __name__ == "__main__":
    main()
