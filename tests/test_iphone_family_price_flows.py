import pytest

from scripts import step4_answer as sa


def _has_bulleted_links(text: str) -> bool:
    """Return True when answer contains at least one markdown bullet link."""
    lines = (text or "").splitlines()
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("- [") and "](http" in ln:
            return True
    return False


def test_iphone_cheapest_family_fallback() -> None:
    """Cheapest iPhone query should use catalog-backed family answer."""

    q = "what's the cheapest iPhone available"
    answer, _ = sa.answer_query(q, top_k=6)

    assert answer
    low = answer.lower()
    assert "cheapest iphone" in low
    assert _has_bulleted_links(answer)
    assert "https://grest.in/collections/iphones" in answer


def test_iphone_most_expensive_family_fallback() -> None:
    """Query for the most expensive iPhone should use catalog-backed family answer.

    This exercises the deterministic iPhone family fallback instead of the
    generic product fallback when no exact catalog model is detected.
    """

    q = "what's the most expensive iPhone available"
    answer, citations = sa.answer_query(q, top_k=6)

    assert answer
    low = answer.lower()
    # Should clearly talk about a most-expensive style choice
    assert "most expensive iphone" in low or "most expensive iphone we currently have" in low
    # Should surface at least one markdown bullet link for a catalog product
    assert _has_bulleted_links(answer)
    # Family collection link should always be present
    assert "https://grest.in/collections/iphones" in answer
    # Citations should be present but we do not assert on exact URLs here
    assert isinstance(citations, list)


@pytest.mark.parametrize(
    "query",
    [
        "list me iPhones under 50000",
        "list me iPhones under 50,000",
    ],
)
def test_iphone_under_price_range(query: str) -> None:
    """Price-ceiling queries should list iPhones in that range from catalog."""

    answer, _ = sa.answer_query(query, top_k=6)

    assert answer
    low = answer.lower()
    # Header should communicate that we are in a price range context
    assert "that price range" in low
    assert _has_bulleted_links(answer)
    assert "https://grest.in/collections/iphones" in answer


def test_iphone_between_price_range() -> None:
    """Between-range queries should also use the deterministic family listing."""

    q = "list me iPhones between 20000 and 50,000"
    answer, _ = sa.answer_query(q, top_k=6)

    assert answer
    low = answer.lower()
    assert "that price range" in low or "we currently have in that price range" in low
    assert _has_bulleted_links(answer)
    assert "https://grest.in/collections/iphones" in answer


def test_iphone_family_browse_query() -> None:
    """Generic browse-style query should list a small set of catalog iPhones."""

    q = "what all iPhones you have?"
    answer, _ = sa.answer_query(q, top_k=6)

    assert answer
    low = answer.lower()
    # Should mention that several iPhones are available or similar wording
    assert "we have several iphones" in low or "we have these iphones" in low
    assert _has_bulleted_links(answer)
    assert "https://grest.in/collections/iphones" in answer
