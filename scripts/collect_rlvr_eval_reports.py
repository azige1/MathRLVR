#!/usr/bin/env python3
"""Collect individual EasyR1 evaluation reports into project-level reports."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def load_reports(pattern: str) -> list[dict[str, Any]]:
    reports = []
    for path_str in sorted(glob.glob(pattern)):
        path = Path(path_str)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        data["_report_file"] = str(path)
        reports.append(data)
    return reports


def infer_run_name(report: dict[str, Any]) -> str:
    output_file = str(report.get("output_file") or report.get("_report_file") or "")
    name = Path(output_file).stem
    for suffix in [
        "_math12k_greedy",
        "_math12k_best_of_4",
        "_math12k_best_of_8",
        "_gsm8k_greedy",
        "_report",
    ]:
        name = name.replace(suffix, "")
    return name


def normalize_report(report: dict[str, Any]) -> dict[str, Any]:
    out = dict(report)
    out["run_name"] = infer_run_name(report)
    if "best_of_N_accuracy" in out:
        out["metric_family"] = "best_of_n" if int(out.get("n", 1) or 1) > 1 else "greedy"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-of-n-pattern", default="results/*_math12k_best_of_*_report.json")
    parser.add_argument("--cross-benchmark-pattern", default="results/*_gsm8k_greedy_report.json")
    parser.add_argument("--best-of-n-output", default="results/best_of_n_eval_report.json")
    parser.add_argument("--cross-benchmark-output", default="results/cross_benchmark_eval_report.json")
    args = parser.parse_args()

    best_of_n = [normalize_report(report) for report in load_reports(args.best_of_n_pattern)]
    cross = [normalize_report(report) for report in load_reports(args.cross_benchmark_pattern)]

    best_payload = {
        "description": "Math12K best-of-N / pass@k / maj@k reports.",
        "reports": best_of_n,
    }
    cross_payload = {
        "description": "Cross-benchmark reports. GSM8K is P0; MATH500 is P1 optional.",
        "reports": cross,
    }

    best_path = Path(args.best_of_n_output)
    cross_path = Path(args.cross_benchmark_output)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    cross_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cross_path.write_text(json.dumps(cross_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {best_path} with {len(best_of_n)} reports")
    print(f"Saved {cross_path} with {len(cross)} reports")


if __name__ == "__main__":
    main()
