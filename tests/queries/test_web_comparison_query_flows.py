from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ and scripts/ are importable when running pytest from project root
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts import step4_answer as sa


class DummyChunk:
    """Minimal stand-in for RetrievedChunk for web comparison tests.

    Args:
        source: Chunk source URL/path.
        text: Chunk text content.
    """

    def __init__(self, source: str, text: str) -> None:
        self.chunk_id = "db-1"
        self.document_id = "doc-1"
        self.source = source
        self.text = text
        self.start_line = 1
        self.end_line = 20
        self.score = 1.0
        self.score_vector = 1.0
        self.score_lexical = 0.0


class DummyWebResult:
    """Simple stand-in for WebSearchResult returned by search_comparison."""

    def __init__(self, url: str, title: str, snippet: str) -> None:
        self.url = url
        self.title = title
        self.snippet = snippet


def test_phone_comparison_uses_web_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Comparison query for iPhones should attach external web citations.

    We mock search_comparison, retrieve, and _call_openai so no real
    HTTP, DB, or LLM calls are made.
    """

    monkeypatch.setenv("ENABLE_WEB_COMPARISON", "1")

    def fake_search_comparison(query: str, max_results: int = 5):  # type: ignore[override]
        return [
            DummyWebResult(
                url="https://example.com/iphone-14-vs-iphone-15",
                title="iPhone 14 vs iPhone 15",
                snippet="Comparison of iPhone 14 and iPhone 15.",
            )
        ]

    def fake_retrieve(query_text: str, top_k: int = 6):  # type: ignore[override]
        return [
            DummyChunk(
                source="https://grest.in/pages/faqs",
                text="Internal FAQ snippet.",
            )
        ]

    def fake_call_openai(prompt: str, max_retries: int = 3, model: str = "gpt-4o-mini") -> str:  # type: ignore[override]
        return "Comparison answer"

    monkeypatch.setattr(sa, "search_comparison", fake_search_comparison)
    monkeypatch.setattr(sa, "retrieve", fake_retrieve)
    monkeypatch.setattr(sa, "_call_openai", fake_call_openai)

    q = "iphone 14 vs iphone 15 which is better?"
    answer, citations = sa.answer_query(q, top_k=6)

    assert answer
    assert isinstance(citations, list)
    # Expect at least one external citation from the mocked web results
    assert any(
        "https://example.com/iphone-14-vs-iphone-15" in c.url for c in citations
    )


def test_phone_comparison_which_is_better_uses_web_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'which is better X or Y' should also trigger web comparison for iPhones."""

    monkeypatch.setenv("ENABLE_WEB_COMPARISON", "1")

    def fake_search_comparison(query: str, max_results: int = 5):  # type: ignore[override]
        return [
            DummyWebResult(
                url="https://example.com/iphone-13-vs-iphone-14",
                title="iPhone 13 vs iPhone 14",
                snippet="Comparison of iPhone 13 and iPhone 14.",
            )
        ]

    def fake_retrieve(query_text: str, top_k: int = 6):  # type: ignore[override]
        return [
            DummyChunk(
                source="https://grest.in/pages/faqs",
                text="Internal FAQ snippet.",
            )
        ]

    def fake_call_openai(
        prompt: str,
        max_retries: int = 3,
        model: str = "gpt-4o-mini",
    ) -> str:  # type: ignore[override]
        return "Comparison answer"

    monkeypatch.setattr(sa, "search_comparison", fake_search_comparison)
    monkeypatch.setattr(sa, "retrieve", fake_retrieve)
    monkeypatch.setattr(sa, "_call_openai", fake_call_openai)

    q = "which is better iphone 13 or iphone 14?"
    answer, citations = sa.answer_query(q, top_k=6)

    assert answer
    assert isinstance(citations, list)
    assert any("https://example.com/iphone-13-vs-iphone-14" in c.url for c in citations)


def test_brand_comparison_grest_vs_cashify_uses_web_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Comparison query 'Grest vs Cashify' should use external web sources."""

    monkeypatch.setenv("ENABLE_WEB_COMPARISON", "1")

    def fake_search_comparison(query: str, max_results: int = 5):  # type: ignore[override]
        return [
            DummyWebResult(
                url="https://example.com/grest-vs-cashify",
                title="Grest vs Cashify",
                snippet="External review comparing Grest and Cashify.",
            )
        ]

    def fake_retrieve(query_text: str, top_k: int = 6):  # type: ignore[override]
        return [
            DummyChunk(
                source="https://grest.in/pages/faqs",
                text="Internal FAQ snippet.",
            )
        ]

    def fake_call_openai(prompt: str, max_retries: int = 3, model: str = "gpt-4o-mini") -> str:  # type: ignore[override]
        return "Brand comparison answer"

    monkeypatch.setattr(sa, "search_comparison", fake_search_comparison)
    monkeypatch.setattr(sa, "retrieve", fake_retrieve)
    monkeypatch.setattr(sa, "_call_openai", fake_call_openai)

    q = "Grest vs Cashify which is better for refurbished iPhones?"
    answer, citations = sa.answer_query(q, top_k=6)

    assert answer
    assert isinstance(citations, list)
    assert any("https://example.com/grest-vs-cashify" in c.url for c in citations)


def test_live_phone_comparison_which_is_better_end_to_end() -> None:
    """Live end-to-end test for a comparison query using real LLM + web search.

    This test is intentionally gated behind ENABLE_LIVE_WEB_COMPARISON_TEST so it
    does not run in normal CI. When enabled, it exercises the full RACEN
    pipeline (DB retrieval, LLM classifiers, SerpAPI DuckDuckGo client, and
    answer synthesis) for a natural comparison-style question.
    """

    import os

    if os.getenv("ENABLE_LIVE_WEB_COMPARISON_TEST", "0") not in {"1", "true", "TRUE", "yes"}:
        pytest.skip("live web comparison test disabled via ENABLE_LIVE_WEB_COMPARISON_TEST")

    # Ensure web comparison is enabled for this run; this only affects the
    # current process and does not mock or stub any network calls.
    os.environ["ENABLE_WEB_COMPARISON"] = "1"

    q = "which is better iphone 13 or iphone 14?"
    answer, citations = sa.answer_query(q, top_k=6)

    assert answer
    assert isinstance(citations, list)
    assert citations


def test_live_phone_comparison_difference_between_end_to_end() -> None:
    """Second live end-to-end test for a typical 'difference between' query.

    Like the previous test, this is gated by ENABLE_LIVE_WEB_COMPARISON_TEST and
    uses the real RACEN pipeline without monkeypatching `_call_openai` or
    `search_comparison`.
    """

    import os

    if os.getenv("ENABLE_LIVE_WEB_COMPARISON_TEST", "0") not in {"1", "true", "TRUE", "yes"}:
        pytest.skip("live web comparison test disabled via ENABLE_LIVE_WEB_COMPARISON_TEST")

    os.environ["ENABLE_WEB_COMPARISON"] = "1"

    q = "what is the difference between iphone 14 and iphone 15?"
    answer, citations = sa.answer_query(q, top_k=6)

    assert answer
    assert isinstance(citations, list)
    assert citations
