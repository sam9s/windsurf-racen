from __future__ import annotations

import pytest

from scripts import step4_answer as sa


class DummyChunk:
    """Minimal stand-in for RetrievedChunk used in flow tests.

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


def test_product_query_uses_specs_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Product queries should invoke the catalog-based specs loader.

    This ensures the new design (loading specs from product pages in the
    corpus) is wired for product intents without changing non-product flows.
    """

    called = {"loader": False, "retrieved": False, "llm": False}

    def fake_loader(match):  # type: ignore[override]
        called["loader"] = True
        return {}

    def fake_retrieve(query: str, top_k: int = 6):  # type: ignore[override]
        called["retrieved"] = True
        return [DummyChunk(source="/products/refurbished-iphone-11", text="Refurbished iPhone 11 page")]

    def fake_call_openai(prompt: str, max_retries: int = 3, model: str = "gpt-4o-mini") -> str:  # type: ignore[override]
        called["llm"] = True
        return "Test product answer"

    monkeypatch.setattr(sa, "_load_product_specs_for_candidates", fake_loader)
    monkeypatch.setattr(sa, "retrieve", fake_retrieve)
    monkeypatch.setattr(sa, "_call_openai", fake_call_openai)

    answer, citations = sa.answer_query("do you have iphone 11?", top_k=3)

    assert answer
    assert isinstance(citations, list)
    assert called["loader"] is True
    assert called["retrieved"] is True
    assert called["llm"] is True
