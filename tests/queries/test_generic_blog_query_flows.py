from __future__ import annotations

from scripts import step4_answer as sa


def test_generic_buying_advice_query_hits_blogs_end_to_end() -> None:
    """End-to-end: generic buying advice should cite blog URLs.

    This calls the live RACEN pipeline and verifies that a question like
    'whats the best time to buy refurbished phone?' cites at least one
    blog URL in the answer's citations.
    """

    q = "whats the best time to buy refurbished phone?"
    answer, citations = sa.answer_query(q, top_k=10)

    assert answer
    assert isinstance(citations, list)

    # At least one citation should point to a blog article.
    assert any("/blogs/news/" in c.url for c in citations)
