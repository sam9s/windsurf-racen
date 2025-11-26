from __future__ import annotations

import os

import pytest

from scripts import step4_answer as sa


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set for live LLM test")
def test_cheapest_iphone_messed_up_english_normalized() -> None:
    """End-to-end: messy cheapest-iPhone queries should still hit family price flow.

    This test intentionally hits the live RACEN pipeline (DB + retrieval + LLM)
    with a noisy, typo-heavy query. It verifies that the LLM-based
    normalization layer plus deterministic iPhone family price logic produce
    a grounded, catalog-backed answer instead of a generic rephrase fallback.
    """

    os.environ["RACEN_QUERY_NORMALIZATION_ENABLE"] = "1"

    query = "whts the chipest iphone you have?"
    answer, _ = sa.answer_query(query, top_k=6)

    assert answer
    low = answer.lower()

    # Should not fall back to static "rephrase" style message.
    assert "rephrase your question" not in low
    assert "mujhe thoda clear nahi hua" not in low

    # Deterministic iPhone family answer should include bullets and collection link.
    assert "https://grest.in/collections/iphones" in answer
    assert any(ln.strip().startswith("- [") and "](http" in ln for ln in answer.splitlines())


@pytest.mark.skip(reason="Hinglish normalization will be tested separately later")
def test_cheapest_iphone_hinglish_normalized() -> None:
    """Placeholder for future Hinglish normalization tests.

    The actual Hinglish normalization behaviour will be exercised in a
    dedicated test suite once the Hinglish logic is finalised.
    """

    assert True
