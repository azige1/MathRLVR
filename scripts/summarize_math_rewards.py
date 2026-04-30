#!/usr/bin/env python3
"""Summarize EasyR1 math JSONL predictions that contain score dictionaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_file(path: Path) -> dict:
    totals = {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            score = item.get("score", {})
            for key in totals:
                totals[key] += float(score.get(key, 0.0) or 0.0)
            count += 1
    denom = max(count, 1)
    return {
        "file": str(path),
        "count": count,
        "overall": totals["overall"] / denom,
        "format": totals["format"] / denom,
        "accuracy": totals["accuracy"] / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    rows = [summarize_file(Path(file)) for file in args.files]
    text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
