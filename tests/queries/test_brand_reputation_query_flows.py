from __future__ import annotations

import os

import pytest

from scripts import step4_answer as sa


class DummyChunk:
    """Minimal stand-in for RetrievedChunk for brand-reputation tests.

    Args:
        source: Chunk source URL/path.
        text: Chunk text content.
        start_line: Start line number.
        end_line: End line number.
    """

    def __init__(self, source: str, text: str, start_line: int = 1, end_line: int = 20) -> None:
        self.source = source
        self.text = text
        self.start_line = start_line
        self.end_line = end_line


def test_brand_reputation_query_routes_to_trustpilot_and_mouthshut(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Brand reputation queries should bias retrieval to Trustpilot/Mouthshut.

    This test focuses on routing: for a query like "why should I buy from grest?",
    we expect the domain classifier to tag it as brand_reputation and the
    retrieval allowlist to include both Trustpilot and Mouthshut patterns.
    """

    called = {"retrieve": False}
    captured_allowlist = {"value": ""}

    def fake_domain_classifier(query: str) -> str:  # type: ignore[override]
        # Force brand_reputation classification for this test.
        return "brand_reputation"

    def fake_retrieve(query: str, top_k: int = 6):  # type: ignore[override]
        called["retrieve"] = True
        captured_allowlist["value"] = os.getenv("RETRIEVE_SOURCE_ALLOWLIST", "")
        return [
            DummyChunk(
                source="https://www.trustpilot.com/review/grest.in",
                text="Trustpilot rating and reviews for Grest.",
            )
        ]

    def fake_call_openai(prompt: str, max_retries: int = 3, model: str = "gpt-4o-mini") -> str:  # type: ignore[override]
        # We do not assert on the exact answer text here.
        return "Reputation answer"

    monkeypatch.setattr(sa, "_classify_product_domain", fake_domain_classifier)
    monkeypatch.setattr(sa, "retrieve", fake_retrieve)
    monkeypatch.setattr(sa, "_call_openai", fake_call_openai)

    q = "why should I buy from grest?"
    answer, citations = sa.answer_query(q, top_k=6)

    assert answer
    assert isinstance(citations, list)
    assert called["retrieve"] is True

    allowlist = captured_allowlist["value"]
    assert "trustpilot.com/review" in allowlist
    assert "mouthshut.com/product-reviews/grest-reviews" in allowlist


def test_brand_reputation_trustpilot_rating_end_to_end() -> None:
    """End-to-end: Trustpilot rating query should cite Trustpilot."""

    q = "what is the Trustpilot rating of Grest?"
    answer, citations = sa.answer_query(q, top_k=10)

    assert answer
    assert isinstance(citations, list)
    assert any("https://www.trustpilot.com/review/grest.in" in c.url for c in citations)


def test_brand_reputation_mouthshut_reviews_end_to_end() -> None:
    """End-to-end: Mouthshut review query should cite Mouthshut."""

    q = "what do people say about Grest on Mouthshut?"
    answer, citations = sa.answer_query(q, top_k=10)

    assert answer
    assert isinstance(citations, list)
    assert any(
        "https://www.mouthshut.com/product-reviews/grest-reviews-926180198" in c.url
        for c in citations
    )


def test_brand_reputation_can_i_trust_grest_end_to_end() -> None:
    """End-to-end: trust question should use external review sources."""

    q = "can I trust grest.in for refurbished phones?"
    answer, citations = sa.answer_query(q, top_k=10)

    assert answer
    assert isinstance(citations, list)
    assert any(
        "https://www.trustpilot.com/review/grest.in" in c.url
        or "https://www.mouthshut.com/product-reviews/grest-reviews-926180198" in c.url
        for c in citations
    )


def test_brand_reputation_followup_mentions_reviews_or_links() -> None:
    """End-to-end: brand reputation follow-up should talk about reviews/links."""

    q = "how do you rate grest?"
    answer, citations = sa.answer_query(q, top_k=10)

    assert answer
    assert isinstance(citations, list)

    low = answer.lower()
    assert "review" in low or "rating" in low or "links" in low


def test_brand_reputation_are_customers_happy_end_to_end() -> None:
    """End-to-end: customer satisfaction query should use review sources."""

    q = "are grest customers happy with their experience?"
    answer, citations = sa.answer_query(q, top_k=10)

    assert answer
    assert isinstance(citations, list)
    assert any(
        "https://www.trustpilot.com/review/grest.in" in c.url
        or "https://www.mouthshut.com/product-reviews/grest-reviews-926180198" in c.url
        for c in citations
    )


def test_brand_reputation_grest_rating_short_query_end_to_end() -> None:
    """End-to-end: short 'grest rating' query should still hit reviews."""

    q = "grest rating"
    answer, citations = sa.answer_query(q, top_k=10)

    assert answer
    assert isinstance(citations, list)
    assert any(
        "https://www.trustpilot.com/review/grest.in" in c.url
        or "https://www.mouthshut.com/product-reviews/grest-reviews-926180198" in c.url
        for c in citations
    )
