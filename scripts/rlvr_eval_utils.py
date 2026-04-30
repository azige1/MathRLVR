#!/usr/bin/env python3
"""Shared helpers for the EasyR1 math RLVR analysis scripts."""

from __future__ import annotations

import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)
FORMAT_RE = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
TURN_MARKER_RE = re.compile(
    r"(?:^|\n|(?<=[\]\)\}]))\s*(?:Human|Assistant|User|System)\s*:",
    re.IGNORECASE,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_reward_function(reward_file: str):
    reward_path = Path(reward_file).resolve()
    spec = importlib.util.spec_from_file_location("easy_r1_math_reward", reward_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reward file: {reward_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _balanced_command_matches(text: str, command: str = r"\boxed") -> list[tuple[str, int, int]]:
    """Return balanced-brace command contents plus their span in the original text."""
    matches: list[tuple[str, int, int]] = []
    idx = 0
    while True:
        cmd_start = text.find(command, idx)
        if cmd_start < 0:
            break
        brace_start = text.find("{", cmd_start + len(command))
        if brace_start < 0:
            break

        depth = 0
        for pos in range(brace_start, len(text)):
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    matches.append((text[brace_start + 1 : pos].strip(), cmd_start, pos + 1))
                    idx = pos + 1
                    break
        else:
            break
    return matches


def extract_boxed_content_fallback(response: str) -> str:
    text = response or ""
    balanced = _balanced_command_matches(text)
    if balanced:
        return balanced[-1][0]
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else ""


def postprocess_generation(
    response: str,
    *,
    cut_turn_markers: bool = False,
    stop_after_boxed: bool = False,
) -> str:
    """Cut obvious chat contamination and optional text after the final boxed answer."""
    text = response or ""
    if cut_turn_markers:
        marker = TURN_MARKER_RE.search(text)
        if marker:
            text = text[: marker.start()].rstrip()

    if stop_after_boxed:
        boxed = _balanced_command_matches(text)
        if boxed:
            text = text[: boxed[-1][2]].rstrip()
    return text


def format_reward_fallback(response: str) -> float:
    normalized = re.sub(r"\s*(<|>|/)\s*", r"\1", response or "")
    return 1.0 if re.fullmatch(FORMAT_RE, normalized) else 0.0


def simple_answer_equal(prediction: str, ground_truth: str) -> bool:
    pred = normalize_text(prediction)
    gold = normalize_text(ground_truth)
    if pred == gold:
        return True
    try:
        return math.isclose(float(pred), float(gold), rel_tol=1e-9, abs_tol=1e-9)
    except Exception:
        return False


def normalize_answer_for_relaxed_match(text: Any) -> str:
    """Normalize common verifier false-negative cases for offline diagnostics only."""
    value = str(text or "").strip()
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = re.sub(r"\$+", "", value)
    value = re.sub(r"\\(?:left|right)\s*", "", value)
    value = re.sub(r"\\(?:text|mbox|mathrm)\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\\(?:,|!|;|:)", "", value)
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
    value = re.sub(r"_\{([^{}]+)\}", r"_\1", value)
    value = re.sub(r"\^\{?\\circ\}?", "", value)
    value = value.replace("°", "")
    value = value.replace("\\%", "%")
    value = re.sub(
        r"(?i)\b(?:degrees?|degree|cm|mm|km|m|meters?|inches?|inch|feet|foot|ft|"
        r"seconds?|minutes?|hours?|dollars?|cents?|units?)\b(?:\s*\^\{?\d+\}?)?",
        "",
        value,
    )
    value = value.replace("%", "")
    value = value.strip()
    value = re.sub(r"^\{(.+)\}$", r"\1", value)
    value = re.sub(r"\s+", "", value.lower())
    return value


def relaxed_answer_equal(prediction: str, ground_truth: str) -> bool:
    """A conservative offline-only equality check for audit/diagnostic metrics."""
    if grade_prediction(prediction, ground_truth):
        return True
    pred = normalize_answer_for_relaxed_match(prediction)
    gold = normalize_answer_for_relaxed_match(ground_truth)
    if pred and pred == gold:
        return True
    try:
        return math.isclose(float(pred), float(gold), rel_tol=1e-9, abs_tol=1e-9)
    except Exception:
        return False


def grade_prediction(prediction: str, ground_truth: str) -> bool:
    try:
        from mathruler.grader import grade_answer

        return bool(grade_answer(prediction, ground_truth))
    except Exception:
        return simple_answer_equal(prediction, ground_truth)


def _answer_candidate_spans(response: str) -> list[tuple[str, int, int, str]]:
    text = response or ""
    candidates: list[tuple[str, int, int, str]] = []
    for value, start, end in _balanced_command_matches(text):
        candidates.append((value, start, end, "boxed"))

    tail_start = max(0, len(text) - 1800)
    tail = text[tail_start:]
    patterns: list[tuple[str, str]] = [
        ("pmatrix", r"\\begin\{pmatrix\}.*?\\end\{pmatrix\}"),
        ("fraction", r"-?\\(?:dfrac|tfrac|frac)\{[^{}\n]{1,80}\}\{[^{}\n]{1,80}\}"),
        ("sqrt", r"-?\\sqrt\{[^{}\n]{1,80}\}"),
        ("ordered_pair", r"\([^()\n]{1,80}\)"),
        ("base_number", r"-?\d+(?:\.\d+)?_\{?\d+\}?"),
        ("number", r"-?\d+(?:\.\d+)?(?:\s*(?:\\%|%|\\circ|°|degrees?))?"),
    ]
    for kind, pattern in patterns:
        for match in re.finditer(pattern, tail, flags=re.DOTALL):
            value = match.group(0).strip()
            candidates.append((value, tail_start + match.start(), tail_start + match.end(), kind))
    return candidates


def _has_answer_cue_near(response: str, start: int, end: int) -> bool:
    window = response[max(0, start - 260) : min(len(response), end + 260)].lower()
    cues = [
        "answer",
        "final",
        "therefore",
        "thus",
        "hence",
        "so",
        "equals",
        "is",
        "=",
        "\\boxed",
    ]
    return any(cue in window for cue in cues)


def extract_robust_answer_candidate(response: str, ground_truth: str) -> str:
    """Find a likely final answer missed by strict boxed extraction.

    This is only for offline diagnosis of verifier false negatives. It deliberately
    requires answer-like local context or a tail-position match to avoid turning
    arbitrary intermediate calculations into the reported model answer.
    """
    text = response or ""
    boxed = extract_boxed_content_fallback(text)
    if boxed:
        return boxed

    for candidate, start, end, _kind in reversed(_answer_candidate_spans(text)):
        near_tail = end >= len(text) - 220
        if (near_tail or _has_answer_cue_near(text, start, end)) and relaxed_answer_equal(candidate, ground_truth):
            return candidate.strip()
    return ""


def score_response_v1(response: str, ground_truth: str, reward_file: str, format_weight: float = 0.1) -> dict[str, float]:
    try:
        reward_fn = load_reward_function(reward_file)
        score = reward_fn(
            [
                {
                    "response": response,
                    "response_length": len(response),
                    "ground_truth": ground_truth,
                }
            ],
            format_weight=format_weight,
        )[0]
        accuracy = float(score.get("accuracy", 0.0) or 0.0)
        fmt = float(score.get("format", 0.0) or 0.0)
    except Exception:
        pred = extract_boxed_content_fallback(response)
        accuracy = 1.0 if grade_prediction(pred, ground_truth) else 0.0
        fmt = format_reward_fallback(response)

    return {
        "accuracy": accuracy,
        "format_rate": fmt,
        "train_reward_v1": (1.0 - format_weight) * accuracy + format_weight * fmt,
    }


def has_severe_repetition(text: str, ngram_size: int = 4, threshold: float = 0.34) -> bool:
    tokens = re.findall(r"\S+", text or "")
    if len(tokens) < ngram_size * 3:
        return False

    ngrams = [tuple(tokens[i : i + ngram_size]) for i in range(len(tokens) - ngram_size + 1)]
    if not ngrams:
        return False
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    if repeated / max(len(ngrams), 1) >= threshold:
        return True

    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    line_counts = Counter(lines)
    return any(count >= 3 and len(line) >= 12 for line, count in line_counts.items())


def looks_truncated(response: str, response_length: int | None, max_response_length: int | None) -> bool:
    text = response or ""
    return bool(
        (max_response_length and response_length is not None and response_length >= max_response_length - 1)
        or ("<think>" in text and "</think>" not in text)
    )


def generation_issue_flags(
    response: str,
    response_length: int | None,
    max_response_length: int | None,
) -> dict[str, float]:
    text = response or ""
    hit_length_cap = bool(max_response_length and response_length is not None and response_length >= max_response_length - 1)
    unclosed_think = "<think>" in text and "</think>" not in text
    missing_boxed = not extract_boxed_content_fallback(text)
    prompt_contamination = TURN_MARKER_RE.search(text) is not None
    return {
        "hit_length_cap": 1.0 if hit_length_cap else 0.0,
        "unclosed_think": 1.0 if unclosed_think else 0.0,
        "missing_boxed_answer": 1.0 if missing_boxed else 0.0,
        "prompt_contamination": 1.0 if prompt_contamination else 0.0,
        "truncation": 1.0 if (hit_length_cap or unclosed_think) else 0.0,
    }


def diagnostic_score_v2_lite(
    response: str,
    ground_truth: str,
    reward_file: str,
    response_length: int | None = None,
    max_response_length: int | None = None,
    existing_v1: dict[str, Any] | None = None,
) -> dict[str, float | int | str]:
    if existing_v1 and {"accuracy", "format_rate", "train_reward_v1"}.issubset(existing_v1):
        v1 = {
            "accuracy": float(existing_v1.get("accuracy", 0.0) or 0.0),
            "format_rate": float(existing_v1.get("format_rate", 0.0) or 0.0),
            "train_reward_v1": float(existing_v1.get("train_reward_v1", 0.0) or 0.0),
        }
    else:
        v1 = score_response_v1(response, ground_truth, reward_file=reward_file, format_weight=0.1)
    boxed = extract_boxed_content_fallback(response)
    robust_answer = extract_robust_answer_candidate(response, ground_truth)
    robust_accuracy = 1.0 if robust_answer and relaxed_answer_equal(robust_answer, ground_truth) else 0.0
    answer_extractable = 1.0 if boxed and len(boxed) <= 100 else 0.0
    robust_answer_extractable = 1.0 if robust_answer and len(robust_answer) <= 120 else 0.0
    issue_flags = generation_issue_flags(response, response_length, max_response_length)
    repetition = 1.0 if has_severe_repetition(response) else 0.0
    invalid_output_penalty = 1.0 if issue_flags["truncation"] or repetition or issue_flags["prompt_contamination"] else 0.0
    score = (
        0.85 * float(v1["accuracy"])
        + 0.10 * float(v1["format_rate"])
        + 0.05 * answer_extractable
        - 0.10 * invalid_output_penalty
    )
    return {
        **v1,
        "diagnostic_score_v2_lite": score,
        "answer_extractable": answer_extractable,
        "extract_failure": 1.0 - answer_extractable,
        "robust_answer_extractable": robust_answer_extractable,
        "robust_extract_failure": 1.0 - robust_answer_extractable,
        "robust_accuracy": robust_accuracy,
        "verifier_false_negative_candidate": (
            1.0 if float(v1["accuracy"]) <= 0.0 and robust_accuracy > 0.0 else 0.0
        ),
        **issue_flags,
        "repetition": repetition,
        "invalid_output_penalty": invalid_output_penalty,
        "boxed_answer": boxed,
        "robust_answer": robust_answer,
    }


def average_numeric(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    denom = max(len(rows), 1)
    for key in keys:
        out[key] = sum(float(row.get(key, 0.0) or 0.0) for row in rows) / denom
    return out
