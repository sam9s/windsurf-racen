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
    assert (
        "rephrase your question" in low
        or "samajh nahi" in low
        or "fir se likh sakte" in low
    )


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
    monkeypatch.setenv("RACEN_CACHE_ENABLED", "0")

    ans, _ = sa.answer_query("how about macbooks?", top_k=3)
    assert ans  # sanity
    assert captured["query"] is not None
    q = captured["query"].lower()
    # The retrieval query should include a macbook-family keyword from config
    assert "macbook" in q


def test_classify_intent_product_and_returns() -> None:
    """classify_intent should map obvious product and returns queries correctly."""

    intent, last_intent = sa._classify_intent("do you have macbook air?", "")
    assert intent == "product"
    assert last_intent == "general"

    intent2, _ = sa._classify_intent("how do I get a refund?", "")
    assert intent2 == "returns"


def test_unclear_intent_triggers_rephrase_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Very noisy/short queries should use the unclear-intent fallback instead of retrieval."""

    called = {"retrieved": False}

    def fake_retrieve(query: str, top_k: int = 6):  # type: ignore[override]
        called["retrieved"] = True
        return [DummyChunk(source="/pages/test", text="Some text")]

    monkeypatch.setattr(sa, "retrieve", fake_retrieve)

    # Extremely short/noisy input
    answer, cites = sa.answer_query("???", top_k=3)
    assert not called["retrieved"], "unclear intent path should not hit retrieval"
    assert not cites
    low = answer.lower()
    assert "rephrase" in low or "detail" in low or "clear" in low


def test_product_answer_uses_matching_chunk_not_exact_match_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """If retrieval returns a chunk clearly mentioning 'iPhone 14 Plus', do not use the 'no exact match' fallback."""

    def fake_retrieve(query: str, top_k: int = 6):  # type: ignore[override]
        text = "Certified Refurbished Apple iPhone 14 Plus with 128GB storage and 6.7-inch display."
        return [DummyChunk(source="/products/apple-iphone-14-plus-128", text=text)]

    def fake_call_openai(prompt: str, max_retries: int = 3, model: str = "gpt-4o-mini") -> str:  # type: ignore[override]
        # Simulate a normal product answer from the LLM using the retrieved chunk.
        return "Yes, we have the iPhone 14 Plus available. Product page: https://grest.in/products/apple-iphone-14-plus-128"

    monkeypatch.setattr(sa, "retrieve", fake_retrieve)
    monkeypatch.setattr(sa, "_call_openai", fake_call_openai)

    answer, _ = sa.answer_query("do you have iphone 14 plus?", top_k=3)
    low = answer.lower()
    # Ensure the strict 'no exact match' fallback phrase is not used when a matching chunk exists
    assert "couldnt find an exact match" not in low
    assert "iphone 14 plus" in low


def test_extract_family_hints_detects_iphone_aliases() -> None:
    """_extract_family_hints should catch common noisy iPhone spellings in Hinglish."""

    hints = sa._extract_family_hints("25000 ke ander konsa iphne hai?")
    assert "iphone" in hints


def test_preserve_family_hints_appends_missing_family() -> None:
    """_preserve_family_hints should re-attach a missing iPhone family token after normalization."""

    raw = "25000 ke ander konsa iphne hai?"
    normalized = "Which phones are available under 25,000?"
    out = sa._preserve_family_hints(raw, normalized)
    low = out.lower()
    assert "iphone" in low


def test_preserve_family_hints_does_not_invent_family() -> None:
    """_preserve_family_hints must not introduce a family when none was present in the raw query."""

    raw = "cheapest phone batao"
    normalized = "Which is the cheapest phone?"
    out = sa._preserve_family_hints(raw, normalized)
    assert "iphone" not in out.lower()


def test_preserve_brand_hints_reinserts_grest() -> None:
    """_preserve_brand_hints should re-attach the Grest token after normalisation.

    This guards against the LLM turning "GREST rating" into
    "great rating" and losing the brand name altogether.
    """

    raw = "GREST rating"
    normalized = "great rating"
    out = sa._preserve_brand_hints(raw, normalized)
    low = out.lower()
    assert "grest" in low


def test_preserve_brand_hints_does_not_invent_brand() -> None:
    """_preserve_brand_hints must not introduce Grest when it was not present."""

    raw = "best rating"
    normalized = "great rating"
    out = sa._preserve_brand_hints(raw, normalized)
    assert "grest" not in out.lower()
