#!/usr/bin/env python3
"""Monitor EasyR1 P0 analysis and mirror progress/results to W&B."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


STAGE_NAMES = [
    "summarize",
    "data_inspection",
    "math12k_greedy",
    "best_of_n",
    "gsm8k",
    "collect_reports",
    "v2_rescore",
    "audit_and_errors",
    "finished",
]


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def count_lines(path: Path) -> int:
    try:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except Exception:
        return 0


def current_stage(log_file: Path) -> tuple[int, str]:
    if not log_file.exists():
        return 0, "log_missing"
    try:
        lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return 0, "log_unreadable"
    stage_lines = [line.strip() for line in lines if line.startswith("== ")]
    if not stage_lines:
        return 0, lines[-1][-180:] if lines else "log_empty"
    latest = stage_lines[-1]
    for index, name in enumerate(STAGE_NAMES, start=1):
        if name.replace("_", "-") in latest.lower() or name.replace("_", " ") in latest.lower():
            return index, latest
    return len(stage_lines), latest


def flatten_checkpoint_report(payload: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    base = payload.get("base") or {}
    if base:
        metrics["base/accuracy"] = base.get("accuracy")
        metrics["base/format_rate"] = base.get("format")
        metrics["base/train_reward_v1"] = base.get("overall")
    for run in payload.get("runs", []):
        name = run.get("name")
        best = run.get("best_by_train_reward_v1") or {}
        stability = run.get("stability") or {}
        if not name or not best:
            continue
        prefix = f"checkpoint/{name}"
        metrics[f"{prefix}/best_step"] = best.get("step")
        metrics[f"{prefix}/accuracy"] = best.get("accuracy")
        metrics[f"{prefix}/format_rate"] = best.get("format_rate")
        metrics[f"{prefix}/train_reward_v1"] = best.get("train_reward_v1")
        metrics[f"{prefix}/mean_response_length"] = best.get("mean_response_length")
        metrics[f"{prefix}/accuracy_drop_from_best"] = stability.get("accuracy_drop_from_best")
    return metrics


def flatten_eval_report(report_file: Path, prefix: str) -> dict[str, Any]:
    payload = load_json(report_file)
    if payload is None:
        return {}
    metrics: dict[str, Any] = {}
    for key in [
        "greedy_accuracy",
        "best_of_N_accuracy",
        "pass_at_k",
        "maj_at_k",
        "format_at_k",
        "selected_train_reward_v1",
        "selected_diagnostic_score_v2_lite",
        "mean_response_length",
        "count",
    ]:
        if key in payload:
            metrics[f"{prefix}/{key}"] = payload[key]
    return metrics


def flatten_collected_reports(path: Path, prefix: str) -> dict[str, Any]:
    payload = load_json(path)
    if payload is None:
        return {}
    metrics: dict[str, Any] = {}
    for report in payload.get("reports", []):
        run_name = report.get("run_name") or Path(str(report.get("output_file", ""))).stem
        n = report.get("n", "na")
        item_prefix = f"{prefix}/{run_name}/n{n}"
        for key in [
            "greedy_accuracy",
            "best_of_N_accuracy",
            "pass_at_k",
            "maj_at_k",
            "format_at_k",
            "mean_response_length",
            "count",
        ]:
            if key in report:
                metrics[f"{item_prefix}/{key}"] = report[key]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="easyr1-reasoning")
    parser.add_argument("--entity", default="")
    parser.add_argument("--run-name", default="p0_reasoning_analysis_monitor")
    parser.add_argument("--log-file", default="logs/run_p0_reasoning_analysis_a10.log")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--finish-when-complete", action="store_true")
    args = parser.parse_args()

    import wandb

    wandb_kwargs = {
        "project": args.project,
        "name": args.run_name,
        "job_type": "p0_eval_monitor",
        "config": {
            "purpose": "Monitor EasyR1 P0 offline eval/report generation",
            "poll_seconds": args.poll_seconds,
        },
    }
    if args.entity:
        wandb_kwargs["entity"] = args.entity
    run = wandb.init(**wandb_kwargs)

    log_file = Path(args.log_file)
    report_paths = {
        "checkpoint": Path("results/checkpoint_stability_report.json"),
        "best_of_n": Path("results/best_of_n_eval_report.json"),
        "cross": Path("results/cross_benchmark_eval_report.json"),
        "reward_v2": Path("results/reward_v2_rescoring_report.json"),
        "audit": Path("results/verifier_audit.jsonl"),
        "errors": Path("results/error_analysis_50.jsonl"),
    }

    uploaded: set[str] = set()
    step = 0
    try:
        while True:
            stage_index, stage_text = current_stage(log_file)
            metrics: dict[str, Any] = {
                "monitor/heartbeat": step,
                "monitor/stage_index": stage_index,
                "monitor/log_exists": int(log_file.exists()),
                "monitor/generated_report_files": sum(int(path.exists()) for path in report_paths.values()),
                "monitor/verifier_audit_rows": count_lines(report_paths["audit"]),
                "monitor/error_analysis_rows": count_lines(report_paths["errors"]),
            }
            run.summary["monitor/stage_text"] = stage_text

            checkpoint_payload = load_json(report_paths["checkpoint"])
            if checkpoint_payload:
                metrics.update(flatten_checkpoint_report(checkpoint_payload))

            metrics.update(flatten_collected_reports(report_paths["best_of_n"], "best_of_n"))
            metrics.update(flatten_collected_reports(report_paths["cross"], "cross_benchmark"))

            for report_file in sorted(Path("results").glob("*_report.json")):
                metrics.update(flatten_eval_report(report_file, f"eval_report/{report_file.stem}"))

            reward_v2 = load_json(report_paths["reward_v2"])
            if reward_v2:
                for run_name, summary in (reward_v2.get("runs") or {}).items():
                    for key, value in summary.items():
                        if isinstance(value, (int, float)):
                            metrics[f"reward_v2/{run_name}/{key}"] = value

            for name, path in report_paths.items():
                if path.exists() and str(path) not in uploaded:
                    wandb.save(str(path), policy="now")
                    uploaded.add(str(path))
                    run.summary[f"artifact/{name}"] = str(path)

            wandb.log(metrics, step=step)

            complete = report_paths["audit"].exists() and report_paths["errors"].exists()
            complete = complete and count_lines(report_paths["audit"]) >= 100 and count_lines(report_paths["errors"]) >= 50
            if complete:
                run.summary["p0_status"] = "finished"
                run.summary["verifier_audit_rows"] = count_lines(report_paths["audit"])
                run.summary["error_analysis_rows"] = count_lines(report_paths["errors"])
                if args.finish_when_complete:
                    break

            step += 1
            time.sleep(args.poll_seconds)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
