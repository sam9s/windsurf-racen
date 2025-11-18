import types

import pytest

from scripts import step4_answer as sa


class DummyChunk:
    """Simple stand-in for RetrievedChunk for unit tests."""

    def __init__(self, source: str, text: str, start_line: int = 1, end_line: int = 20) -> None:
        self.source = source
        self.text = text
        self.start_line = start_line
        self.end_line = end_line


@pytest.mark.parametrize(
    "query,expected",
    [
        ("do you have macbook air?", "product"),
        ("kya iphone 13 hai?", "product"),
        ("how about macbooks?", "product"),
        ("what is your shipping policy?", "shipping"),
        ("how do I get a refund?", "returns"),
    ],
)
def test_detect_intent_core_paths(query: str, expected: str) -> None:
    """Core queries should map to the expected high-level intents."""
    intent = sa._detect_intent(query)
    assert intent == expected


@pytest.mark.parametrize(
    "prev_ans,expected",
    [
        ("This MacBook Air is currently out of stock.", "product"),
        ("Product page: https://grest.in/products/refurbished-apple-iphone-xs-max", "product"),
        ("Refunds are processed within 5-7 working days.", "returns"),
        ("We ship via Bluedart and delivery takes 3-5 days.", "shipping"),
    ],
)
def test_infer_last_intent_from_previous_answer(prev_ans: str, expected: str) -> None:
    """Last intent should be inferred from previous assistant answer content."""
    last_intent = sa._infer_last_intent(prev_ans)
    assert last_intent == expected


def test_product_followup_biases_retrieval_to_previous_product_family(monkeypatch: pytest.MonkeyPatch) -> None:
    """For product follow-ups, retrieval query should include the previous product noun (e.g., macbook)."""

    captured = {"query": None}

    def fake_retrieve(query: str, top_k: int = 6):  # type: ignore[override]
        captured["query"] = query
        # Minimal chunk list, content doesn't matter for this test
        return [DummyChunk(source="/products/test", text="Test product page")]

    monkeypatch.setattr(sa, "retrieve", fake_retrieve)

    # Simulate a previous MacBook-style product answer
    previous_answer = "Yes, we have the MacBook Air available."
    previous_user = "do you have macbook air?"

    # Follow-up is an ACK / more-details style question
    query = "ok can I atleast have the details for this?"

    # Call answer_query; we only care about the retrieval query that fake_retrieve sees
    sa.answer_query(query=query, top_k=3, previous_answer=previous_answer, previous_user=previous_user)

    assert captured["query"] is not None
    q = captured["query"].lower()
    assert "macbook" in q, "follow-up retrieval should stay anchored on the MacBook product family"


def test_product_fallback_does_not_leak_noisy_catalog_snippets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Product fallback text should not surface random catalog items like vitamins or Sour Apple."""

    noisy_text = (
        "Vitamin D3 30 Tablets, Caffeine 60 Tablets, Magnesium 30 Tablets, Sour Apple 500ml, "
        "quiz, collections, not sure what product is right for you"
    )

    def fake_retrieve(query: str, top_k: int = 6):  # type: ignore[override]
        return [DummyChunk(source="/pages/collections", text=noisy_text)]

    def fake_call_openai(prompt: str, max_retries: int = 3, model: str = "gpt-4o-mini") -> str:  # type: ignore[override]
        # Force the generic fallback trigger so _build_fallback_text is used
        return "Not found in sources provided. [LLM error: test]"

    monkeypatch.setattr(sa, "retrieve", fake_retrieve)
    monkeypatch.setattr(sa, "_call_openai", fake_call_openai)

    # Product-style query that we do not really have (e.g., iphone 10)
    answer, _ = sa.answer_query("do you have iphone 10?", top_k=3)
    low = answer.lower()
    # Ensure noisy catalog words are not present
    assert "vitamin" not in low
    assert "sour apple" not in low
    # And we should instead see a graceful clarification
    assert "exact match" in low or "exact model" in low


def test_category_query_uses_product_family_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Category-style queries like 'macbooks' should be treated as product and bias retrieval to that family."""

    captured = {"query": None}

    def fake_retrieve(query: str, top_k: int = 6):  # type: ignore[override]
        captured["query"] = query
        return [DummyChunk(source="/products/test-macbook", text="Refurbished MacBook listing")]

    # Ensure we don't hit the real LLM
    def fake_call_openai(prompt: str, max_retries: int = 3, model: str = "gpt-4o-mini") -> str:  # type: ignore[override]
        return "Test answer"

    monkeypatch.setattr(sa, "retrieve", fake_retrieve)
    monkeypatch.setattr(sa, "_call_openai", fake_call_openai)

    ans, _ = sa.answer_query("how about macbooks?", top_k=3)
    assert ans  # sanity
    assert captured["query"] is not None
    q = captured["query"].lower()
    # The retrieval query should include a macbook-family keyword from config
    assert "macbook" in q
