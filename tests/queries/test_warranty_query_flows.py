from __future__ import annotations

import pytest

from scripts import step4_answer as sa


class DummyChunk:
    """Minimal stand-in for RetrievedChunk for warranty-flow tests.

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


def test_warranty_query_does_not_call_product_specs_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warranty queries must not go through the product specs loader.

    This ensures the product-only corpus/specs path is not used for
    generic warranty policy questions.
    """

    called = {"loader": False}

    def fake_loader(match):  # type: ignore[override]
        called["loader"] = True
        raise AssertionError("_load_product_specs_for_candidates should not be called for warranty queries")

    def fake_retrieve(query: str, top_k: int = 6):  # type: ignore[override]
        return [
            DummyChunk(
                source="/pages/warranty",
                text="We offer a 6-month warranty on refurbished devices covering hardware defects.",
            )
        ]

    def fake_call_openai(prompt: str, max_retries: int = 3, model: str = "gpt-4o-mini") -> str:  # type: ignore[override]
        return "Test warranty answer"

    monkeypatch.setattr(sa, "_load_product_specs_for_candidates", fake_loader)
    monkeypatch.setattr(sa, "retrieve", fake_retrieve)
    monkeypatch.setattr(sa, "_call_openai", fake_call_openai)

    answer, citations = sa.answer_query("what warranty do you offer?", top_k=3)

    assert answer
    assert isinstance(citations, list)
    assert called["loader"] is False
