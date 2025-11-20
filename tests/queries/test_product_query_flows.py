from __future__ import annotations

from scripts import step4_answer as sa


def test_product_query_iphone_11_end_to_end() -> None:
    """End-to-end: 'do you have iphone 11?' should surface real specs.

    This hits the live RACEN pipeline (DB + retrieval + LLM) and verifies
    that the iPhone 11 answer contains key spec fields coming from the
    ingested product page rather than generic placeholders.
    """

    answer, citations = sa.answer_query("do you have iphone 11?", top_k=10)

    assert answer
    assert isinstance(citations, list)

    low = answer.lower()
    # Mention the correct model.
    assert "iphone 11" in low
    # Price and storage options should be populated from specs, not "Not specified".
    assert "14,499" in answer  # allow with or without the currency symbol
    assert "storage options" in low
    assert "64 gb" in low
