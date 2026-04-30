#!/usr/bin/env python3
"""Build structured verifier-audit and error-analysis samples from RLVR outputs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from rlvr_eval_utils import diagnostic_score_v2_lite, read_jsonl, write_jsonl


AUDIT_LABELS = [
    "correct_accept",
    "correct_reject",
    "wrong_accept",
    "wrong_reject",
    "format_fail",
    "extract_fail",
    "parse_fail",
    "truncation",
    "repetition",
    "ambiguous_gold",
]

ERROR_TYPES = [
    "arithmetic error",
    "algebra transformation error",
    "wrong problem understanding",
    "reasoning shortcut",
    "final answer extraction error",
    "format violation",
    "overlong reasoning or truncation",
    "repetition",
    "verifier false positive",
    "verifier false negative",
    "other",
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


def flatten_record(run_name: str, row: dict[str, Any], row_idx: int) -> list[dict[str, Any]]:
    if "candidates" not in row:
        return [{**row, "_run": run_name, "_row_index": row_idx, "_candidate_index": None}]

    flattened = []
    for candidate in row.get("candidates", []):
        flattened.append(
            {
                **{k: v for k, v in row.items() if k != "candidates"},
                **candidate,
                "_run": run_name,
                "_row_index": row_idx,
                "_candidate_index": candidate.get("candidate_index"),
            }
        )
    return flattened


def collect_cases(inputs: list[str], reward_file: str, max_response_length: int) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for spec in inputs:
        run_name, path = parse_input_spec(spec)
        if not path.exists():
            continue
        for row_idx, row in enumerate(read_jsonl(path)):
            for item in flatten_record(run_name, row, row_idx):
                response = str(item.get("response", ""))
                ground_truth = get_ground_truth(item)
                response_length = item.get("response_length")
                if response_length is None:
                    response_length = len(response.split())
                diagnostics = item.get("diagnostics")
                if not isinstance(diagnostics, dict):
                    diagnostics = diagnostic_score_v2_lite(
                        response=response,
                        ground_truth=ground_truth,
                        reward_file=reward_file,
                        response_length=int(response_length),
                        max_response_length=max_response_length,
                    )
                case_id = f"{run_name}:{row_idx}:{item.get('_candidate_index')}"
                cases.append(
                    {
                        "id": case_id,
                        "run": run_name,
                        "problem": item.get("problem", ""),
                        "gold": ground_truth,
                        "response": response,
                        "response_length": response_length,
                        "diagnostics": diagnostics,
                    }
                )
    return cases


def is_suspicious(case: dict[str, Any]) -> bool:
    d = case["diagnostics"]
    accuracy = float(d.get("accuracy", 0.0) or 0.0)
    fmt = float(d.get("format_rate", 0.0) or 0.0)
    return any(
        [
            accuracy > 0 and fmt <= 0,
            accuracy <= 0 and fmt > 0,
            float(d.get("extract_failure", 0.0) or 0.0) > 0,
            float(d.get("truncation", 0.0) or 0.0) > 0,
            float(d.get("repetition", 0.0) or 0.0) > 0,
            float(d.get("invalid_output_penalty", 0.0) or 0.0) > 0,
        ]
    )


def unique_by_id(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for case in cases:
        if case["id"] in seen:
            continue
        seen.add(case["id"])
        out.append(case)
    return out


def sample_cases(cases: list[dict[str, Any]], rng: random.Random, audit_size: int, error_size: int):
    suspicious = [case for case in cases if is_suspicious(case)]
    incorrect = [case for case in cases if float(case["diagnostics"].get("accuracy", 0.0) or 0.0) <= 0.0]
    random_pool = list(cases)
    rng.shuffle(suspicious)
    rng.shuffle(incorrect)
    rng.shuffle(random_pool)

    audit = unique_by_id(suspicious[:50] + incorrect[:30] + random_pool[:20])
    if len(audit) < audit_size:
        audit = unique_by_id(audit + random_pool)
    audit = audit[:audit_size]

    error_candidates = unique_by_id(incorrect + suspicious + random_pool)
    error = error_candidates[:error_size]
    return audit, error


def make_audit_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        **case,
        "audit_allowed_labels": AUDIT_LABELS,
        "human_audit_label": "",
        "human_is_verifier_correct": None,
        "human_notes": "",
    }


def make_error_row(case: dict[str, Any]) -> dict[str, Any]:
    return {
        **case,
        "allowed_error_types": ERROR_TYPES,
        "human_error_type": "",
        "human_analysis": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input JSONL files, optionally name=path.")
    parser.add_argument("--reward-file", default="examples/reward_function/math.py")
    parser.add_argument("--max-response-length", type=int, default=512)
    parser.add_argument("--audit-size", type=int, default=100)
    parser.add_argument("--error-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit-output", default="results/verifier_audit.jsonl")
    parser.add_argument("--error-output", default="results/error_analysis_50.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    cases = collect_cases(args.inputs, args.reward_file, args.max_response_length)
    audit, error = sample_cases(cases, rng, args.audit_size, args.error_size)
    write_jsonl(args.audit_output, [make_audit_row(case) for case in audit])
    write_jsonl(args.error_output, [make_error_row(case) for case in error])

    summary = {
        "total_cases": len(cases),
        "audit_output": args.audit_output,
        "audit_count": len(audit),
        "error_output": args.error_output,
        "error_count": len(error),
        "audit_sampling": "50 suspicious/disagreement + 30 incorrect + 20 random, backfilled if needed",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
