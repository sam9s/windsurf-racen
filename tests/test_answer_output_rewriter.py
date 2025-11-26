from __future__ import annotations

import os

from src.racen.answer_output_rewriter import rewrite_answer_language


def test_rewrite_disabled_returns_identity(monkeypatch) -> None:
    """When the env toggle is off, the answer should not be rewritten."""
    monkeypatch.setenv("RACEN_ANSWER_OUTPUT_REWRITE_ENABLE", "0")
    # Ensure API key presence does not matter when disabled
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    res = rewrite_answer_language("Some answer", "Original question", user_mode="HI_EN")
    assert res.rewritten_answer == "Some answer"
    assert res.used_rewriter is False


def test_non_hinglish_mode_returns_identity(monkeypatch) -> None:
    """Non-Hinglish modes should bypass the rewriter even when enabled."""
    monkeypatch.setenv("RACEN_ANSWER_OUTPUT_REWRITE_ENABLE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    res = rewrite_answer_language("Some answer", "Original question", user_mode="EN")
    assert res.rewritten_answer == "Some answer"
    assert res.used_rewriter is False


def test_missing_api_key_falls_back_to_identity(monkeypatch) -> None:
    """When enabled but API key is missing, we should not raise and return identity."""
    monkeypatch.setenv("RACEN_ANSWER_OUTPUT_REWRITE_ENABLE", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    res = rewrite_answer_language("Another answer", "Some Hinglish query", user_mode="HI_EN")
    assert res.rewritten_answer == "Another answer"
    assert res.used_rewriter is False
