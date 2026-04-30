#!/usr/bin/env python3
"""Summarize EasyR1 math RLVR runs and checkpoint stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_RUNS = {
    "grpo_composite": "checkpoints/easyr1_reasoning/qwen2_5_1_5b_math_grpo_a10/experiment_log.jsonl",
    "gspo": "checkpoints/easyr1_reasoning/qwen2_5_1_5b_math_gspo_a10/experiment_log.jsonl",
    "grpo_filtered": "checkpoints/easyr1_reasoning/qwen2_5_1_5b_math_grpo_filtered_a10/experiment_log.jsonl",
    "grpo_answer_only": "checkpoints/easyr1_reasoning/qwen2_5_1_5b_math_grpo_answer_only_a10/experiment_log.jsonl",
    "grpo_format_only": "checkpoints/easyr1_reasoning/qwen2_5_1_5b_math_grpo_format_only_a10/experiment_log.jsonl",
    "grpo_clean_data": "checkpoints/easyr1_reasoning/qwen2_5_3b_math_grpo_clean_data_a10/experiment_log.jsonl",
    "grpo_clean_concise": "checkpoints/easyr1_reasoning/qwen2_5_3b_math_grpo_clean_concise_a10/experiment_log.jsonl",
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Run spec must be name=path, got: {spec}")
    name, path = spec.split("=", 1)
    return name, Path(path)


def validation_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val" not in row:
                continue
            val = row.get("val", {})
            actor = row.get("actor", {})
            response_length = row.get("val_response_length", {})
            rows.append(
                {
                    "step": row.get("step"),
                    "accuracy": val.get("accuracy_reward"),
                    "format_rate": val.get("format_reward"),
                    "train_reward_v1": val.get("reward_score") or val.get("overall_reward"),
                    "mean_response_length": response_length.get("mean"),
                    "clip_ratio": response_length.get("clip_ratio"),
                    "kl_loss": actor.get("kl_loss"),
                    "approx_kl": actor.get("ppo_kl"),
                    "pg_clipfrac_higher": actor.get("pg_clipfrac_higher"),
                    "pg_clipfrac_lower": actor.get("pg_clipfrac_lower"),
                    "grad_norm": actor.get("grad_norm"),
                    "lr": actor.get("lr"),
                }
            )
    return rows


def best_row(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get(key) is not None]
    if not valid:
        return None
    return max(valid, key=lambda row: float(row[key]))


def run_summary(name: str, log_path: Path) -> dict[str, Any]:
    rows = validation_rows(log_path)
    best_by_reward = best_row(rows, "train_reward_v1")
    best_by_accuracy = best_row(rows, "accuracy")
    last = rows[-1] if rows else None
    stability: dict[str, Any] = {}
    if best_by_reward and last:
        stability = {
            "best_step": best_by_reward.get("step"),
            "last_step": last.get("step"),
            "train_reward_v1_drop_from_best": (
                float(best_by_reward["train_reward_v1"]) - float(last["train_reward_v1"])
                if last.get("train_reward_v1") is not None
                else None
            ),
            "accuracy_drop_from_best": (
                float(best_by_reward["accuracy"]) - float(last["accuracy"]) if last.get("accuracy") is not None else None
            ),
            "format_rate_change_from_best": (
                float(last["format_rate"]) - float(best_by_reward["format_rate"])
                if last.get("format_rate") is not None
                else None
            ),
        }
    return {
        "name": name,
        "experiment_log": str(log_path),
        "exists": log_path.exists(),
        "validation_rows": rows,
        "best_by_train_reward_v1": best_by_reward,
        "best_by_accuracy": best_by_accuracy,
        "last": last,
        "stability": stability,
    }


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def table_text(base_summary: dict[str, Any] | None, runs: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("# EasyR1 Math RLVR Run Comparison")
    lines.append("")
    lines.append("## Base")
    if base_summary:
        lines.append(
            "| model | accuracy | format_rate | train_reward_v1 | count |\n"
            "|---|---:|---:|---:|---:|\n"
            f"| base | {format_value(base_summary.get('accuracy'))} | {format_value(base_summary.get('format'))} | "
            f"{format_value(base_summary.get('overall'))} | {format_value(base_summary.get('count'))} |"
        )
    else:
        lines.append("Base report not found.")
    lines.append("")
    lines.append("## RL Runs")
    lines.append(
        "| run | best_step | accuracy | format_rate | train_reward_v1 | mean_response_length | last_step | accuracy_drop_from_best |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for run in runs:
        best = run.get("best_by_train_reward_v1") or {}
        stability = run.get("stability") or {}
        lines.append(
            f"| {run['name']} | {format_value(best.get('step'))} | {format_value(best.get('accuracy'))} | "
            f"{format_value(best.get('format_rate'))} | {format_value(best.get('train_reward_v1'))} | "
            f"{format_value(best.get('mean_response_length'))} | {format_value(stability.get('last_step'))} | "
            f"{format_value(stability.get('accuracy_drop_from_best'))} |"
        )
    lines.append("")
    lines.append(
        "Note: `train_reward_v1` is the training verifier score. It should not be mixed with "
        "`diagnostic_score_v2_lite` from offline rescoring."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], help="Run spec in name=experiment_log.jsonl format.")
    parser.add_argument("--base-report", default="results/qwen2_5_3b_math_base_eval_report.json")
    parser.add_argument("--fallback-base-report", default="results/qwen2_5_1_5b_math_base_eval_report.json")
    parser.add_argument("--output-json", default="results/checkpoint_stability_report.json")
    parser.add_argument("--output-text", default="results/reasoning_rl_all_runs_compare.txt")
    args = parser.parse_args()

    run_specs = dict(DEFAULT_RUNS)
    for spec in args.run:
        name, path = parse_run_spec(spec)
        run_specs[name] = str(path)

    base_report = load_json(Path(args.base_report)) or load_json(Path(args.fallback_base_report))
    runs = [run_summary(name, Path(path)) for name, path in run_specs.items()]

    report = {
        "base": base_report,
        "runs": runs,
        "metric_names": {
            "train_reward_v1": "0.9 * answer_accuracy + 0.1 * format_rate",
            "diagnostic_score_v2_lite": "offline-only diagnostic score, not used for training",
        },
        "recommended_main_result": "Use the best GRPO-composite checkpoint by validation train_reward_v1, not the final checkpoint.",
    }

    output_json = Path(args.output_json)
    output_text = Path(args.output_text)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_text.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_text.write_text(table_text(base_report, runs), encoding="utf-8")

    print(output_text.read_text(encoding="utf-8"))
    print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
