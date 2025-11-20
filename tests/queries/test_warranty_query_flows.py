from __future__ import annotations

from scripts import step4_answer as sa


def test_warranty_query_end_to_end() -> None:
    """End-to-end: 'what warranty do you offer?' should use warranty page.

    This calls the live RACEN pipeline and verifies that the warranty answer
    cites the warranty URL and mentions warranty details.
    """

    answer, citations = sa.answer_query("what warranty do you offer?", top_k=10)

    assert answer
    assert isinstance(citations, list)

    # At least one citation should come from the warranty page.
    assert any("/pages/warranty" in c.url for c in citations)

    low = answer.lower()
    assert "warranty" in low
