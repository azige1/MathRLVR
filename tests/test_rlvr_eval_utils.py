from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rlvr_eval_utils import extract_boxed_content_fallback, postprocess_generation


def test_extract_boxed_content_uses_last_balanced_boxed_answer() -> None:
    response = r"Try \boxed{12}. Final answer is \boxed{\frac{3}{4}}."

    assert extract_boxed_content_fallback(response) == r"\frac{3}{4}"


def test_stop_after_boxed_cuts_text_after_final_boxed_answer() -> None:
    response = r"<think>done</think> The answer is \boxed{42}. Extra unfinished reasoning"

    assert postprocess_generation(response, stop_after_boxed=True) == r"<think>done</think> The answer is \boxed{42}"


def test_turn_marker_postprocess_removes_chat_contamination() -> None:
    response = "Reasoning complete.\nAssistant: let me continue with another answer"

    assert postprocess_generation(response, cut_turn_markers=True) == "Reasoning complete."
