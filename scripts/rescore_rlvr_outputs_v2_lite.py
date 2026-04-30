#!/usr/bin/env python3
"""Offline rescore math RLVR outputs with V1 training reward and V2-lite diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rlvr_eval_utils import average_numeric, diagnostic_score_v2_lite, read_jsonl, write_jsonl


SUMMARY_KEYS = [
    "accuracy",
    "format_rate",
    "train_reward_v1",
    "diagnostic_score_v2_lite",
    "answer_extractable",
    "extract_failure",
    "robust_answer_extractable",
    "robust_extract_failure",
    "robust_accuracy",
    "verifier_false_negative_candidate",
    "hit_length_cap",
    "unclosed_think",
    "missing_boxed_answer",
    "prompt_contamination",
    "truncation",
    "repetition",
    "invalid_output_penalty",
]


def parse_input_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        return name, Path(path)
    path = Path(spec)
    return path.stem, path


def get_ground_truth(row: dict[str, Any]) -> str:
    for key in ["ground_truth", "answer", "gold", "target"]:
        if key in row:
            return str(row[key])
    return ""


def flatten_rows(name: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if "candidates" not in row:
            item = dict(row)
            item["_source_run"] = name
            item["_source_index"] = idx
            flattened.append(item)
            continue

        for candidate_idx, candidate in enumerate(row.get("candidates", [])):
            item = {
                **{k: v for k, v in row.items() if k != "candidates"},
                **candidate,
                "_source_run": name,
                "_source_index": idx,
                "_candidate_index": candidate_idx,
            }
            flattened.append(item)
    return flattened


def rescore_rows(
    name: str,
    path: Path,
    reward_file: str,
    max_response_length: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return [], {"name": name, "path": str(path), "exists": False, "count": 0}

    raw_rows = read_jsonl(path)
    rows = []
    for row in flatten_rows(name, raw_rows):
        response = str(row.get("response", ""))
        ground_truth = get_ground_truth(row)
        response_length = row.get("response_length")
        if response_length is None:
            response_length = len(response.split())
        scores = diagnostic_score_v2_lite(
            response=response,
            ground_truth=ground_truth,
            reward_file=reward_file,
            response_length=int(response_length),
            max_response_length=max_response_length,
            existing_v1=row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else None,
        )
        rows.append(
            {
                **row,
                "ground_truth": ground_truth,
                "response_length": response_length,
                "diagnostics": scores,
            }
        )

    flat_scores = [row["diagnostics"] for row in rows]
    summary = {
        "name": name,
        "path": str(path),
        "exists": True,
        "count": len(rows),
        **average_numeric(flat_scores, SUMMARY_KEYS),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input JSONL files, optionally name=path.")
    parser.add_argument("--reward-file", default="examples/reward_function/math.py")
    parser.add_argument("--max-response-length", type=int, default=512)
    parser.add_argument("--output-jsonl", default="results/reward_v2_rescored_outputs.jsonl")
    parser.add_argument("--report-file", default="results/reward_v2_rescoring_report.json")
    args = parser.parse_args()

    all_rows: list[dict[str, Any]] = []
    summaries = []
    for spec in args.inputs:
        name, path = parse_input_spec(spec)
        rows, summary = rescore_rows(name, path, args.reward_file, args.max_response_length)
        all_rows.extend(rows)
        summaries.append(summary)

    report = {
        "score_definition": {
            "train_reward_v1": "0.9 * answer_accuracy + 0.1 * format_rate",
            "diagnostic_score_v2_lite": (
                "0.85 * answer_accuracy + 0.10 * format_rate + 0.05 * answer_extractable "
                "- 0.10 * invalid_output_penalty"
            ),
            "invalid_output_penalty": "1 iff output is likely truncated or has severe repetition, else 0",
            "robust_accuracy": "offline-only relaxed answer extraction accuracy for verifier audit, not training reward",
            "verifier_false_negative_candidate": "1 when strict verifier score is 0 but robust offline extraction matches gold",
        },
        "runs": summaries,
    }

    write_jsonl(args.output_jsonl, all_rows)
    report_path = Path(args.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved rescored rows: {args.output_jsonl}")
    print(f"Saved report: {args.report_file}")


if __name__ == "__main__":
    main()
