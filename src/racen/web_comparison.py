import os
from typing import List

import requests
from pydantic import BaseModel

from racen.log import get_logger


logger = get_logger("racen.web_comparison")


class WebSearchResult(BaseModel):
    """Single web search result from SerpAPI DuckDuckGo.

    Args:
        url: Result URL.
        title: Result title.
        snippet: Short text snippet describing the result.
    """

    url: str
    title: str
    snippet: str


def _web_comparison_enabled() -> bool:
    """Check whether web comparison is enabled via env toggle.

    Returns:
        bool: True if ENABLE_WEB_COMPARISON is enabled.
    """

    val = os.getenv("ENABLE_WEB_COMPARISON", "0")
    return val in {"1", "true", "TRUE", "yes"}


def search_comparison(query: str, max_results: int = 5) -> List[WebSearchResult]:
    """Run a comparison-style web search via SerpAPI DuckDuckGo.

    This helper is intentionally narrow and side-effect free:
    it only talks to SerpAPI and returns structured results. It does
    not call any LLMs and does not mutate global RACEN state.

    The caller is responsible for deciding when a query is a
    comparison query and how to use the returned results as context.

    Web comparison is gated by two environment variables:
    - ENABLE_WEB_COMPARISON: must be truthy ("1", "true", etc).
    - SERPAPI_API_KEY: SerpAPI API key.

    If either is missing/disabled or if the HTTP request fails,
    this function returns an empty list and logs a warning.

    Args:
        query: Natural-language comparison query, e.g.
            "difference between iphone 14 and iphone 15".
        max_results: Maximum number of organic results to return.

    Returns:
        List[WebSearchResult]: Parsed search results, possibly empty.
    """

    q = (query or "").strip()
    if not q:
        return []

    if not _web_comparison_enabled():
        logger.info("Web comparison disabled via ENABLE_WEB_COMPARISON; skipping SerpAPI call.")
        return []

    api_key = (os.getenv("SERPAPI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("SERPAPI_API_KEY not set; skipping SerpAPI web comparison.")
        return []

    params = {
        "engine": "duckduckgo",
        "q": q,
        "api_key": api_key,
        # Keep HTML out of snippets to simplify downstream prompts.
        "no_html": "true",
    }

    try:
        resp = requests.get("https://serpapi.com/search", params=params, timeout=10)
    except Exception as exc:  # pragma: no cover - network failure path
        logger.warning("SerpAPI request failed: %s", exc)
        return []

    if resp.status_code != 200:
        logger.warning("SerpAPI returned non-200 status: %s", resp.status_code)
        return []

    try:
        data = resp.json()
    except Exception as exc:  # pragma: no cover - JSON failure path
        logger.warning("SerpAPI JSON parse failed: %s", exc)
        return []

    organic = data.get("organic_results") or []
    results: List[WebSearchResult] = []
    for item in organic:
        try:
            url = (item.get("link") or "").strip()
            title = (item.get("title") or "").strip()
            snippet = (item.get("snippet") or "").strip()
        except Exception:
            continue
        if not url:
            continue
        results.append(WebSearchResult(url=url, title=title, snippet=snippet))
        if len(results) >= max_results:
            break

    return results
