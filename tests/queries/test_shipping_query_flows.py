from __future__ import annotations

from scripts import step4_answer as sa


def test_shipping_query_end_to_end() -> None:
    """End-to-end: 'what is your shipping policy?' should use shipping page.

    This calls the live RACEN pipeline and verifies that the shipping answer
    cites the shipping policy URL and mentions shipping-related details.
    """

    answer, citations = sa.answer_query("what is your shipping policy?", top_k=10)

    assert answer
    assert isinstance(citations, list)

    # At least one citation should come from a shipping page.
    assert any("/pages/shipping" in c.url or "/policies/shipping/policy" in c.url for c in citations)

    low = answer.lower()
    assert "ship" in low or "delivery" in low
