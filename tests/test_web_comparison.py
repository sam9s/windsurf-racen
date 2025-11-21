from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Ensure src/ is importable so that `racen` modules can be used when
# tests are run via `python -m pytest` from the project root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racen.web_comparison import WebSearchResult, search_comparison


def test_search_comparison_returns_results_when_enabled(monkeypatch) -> None:
    """search_comparison should parse SerpAPI results when enabled.

    This test mocks requests.get so no real HTTP call is made.
    """

    monkeypatch.setenv("ENABLE_WEB_COMPARISON", "1")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

    called = {"value": False}

    class DummyResponse:
        status_code = 200

        @staticmethod
        def json() -> dict[str, Any]:
            return {
                "organic_results": [
                    {
                        "link": "https://example.com/iphone-14",
                        "title": "iPhone 14 Review",
                        "snippet": "Details about iPhone 14.",
                    },
                    {
                        "link": "https://example.com/iphone-15",
                        "title": "iPhone 15 Review",
                        "snippet": "Details about iPhone 15.",
                    },
                ]
            }

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int = 10):  # type: ignore[override]
        called["value"] = True
        assert url == "https://serpapi.com/search"
        assert params is not None
        assert params.get("engine") == "duckduckgo"
        assert params.get("q") == "iphone 14 vs iphone 15"
        assert params.get("api_key") == "test-key"
        return DummyResponse()

    monkeypatch.setattr("racen.web_comparison.requests.get", fake_get)

    results = search_comparison("iphone 14 vs iphone 15", max_results=2)

    assert called["value"] is True
    assert len(results) == 2
    assert all(isinstance(r, WebSearchResult) for r in results)
    assert results[0].url == "https://example.com/iphone-14"
    assert results[1].url == "https://example.com/iphone-15"


def test_search_comparison_returns_empty_when_disabled(monkeypatch) -> None:
    """When feature toggle is off, search_comparison should not call SerpAPI."""

    monkeypatch.setenv("ENABLE_WEB_COMPARISON", "0")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")

    called = {"value": False}

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int = 10):  # type: ignore[override]
        called["value"] = True
        raise AssertionError("requests.get should not be called when web comparison is disabled")

    monkeypatch.setattr("racen.web_comparison.requests.get", fake_get)

    results = search_comparison("iphone 14 vs iphone 15", max_results=2)

    assert results == []
    assert called["value"] is False


def test_search_comparison_returns_empty_when_api_key_missing(monkeypatch) -> None:
    """When SERPAPI_API_KEY is missing, search_comparison should bail out safely."""

    monkeypatch.setenv("ENABLE_WEB_COMPARISON", "1")
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    called = {"value": False}

    def fake_get(url: str, params: dict[str, Any] | None = None, timeout: int = 10):  # type: ignore[override]
        called["value"] = True
        raise AssertionError("requests.get should not be called when SERPAPI_API_KEY is missing")

    monkeypatch.setattr("racen.web_comparison.requests.get", fake_get)

    results = search_comparison("iphone 14 vs iphone 15", max_results=2)

    assert results == []
    assert called["value"] is False


def test_search_comparison_handles_empty_query() -> None:
    """Empty query should return an empty list without calling SerpAPI."""

    os.environ.pop("ENABLE_WEB_COMPARISON", None)
    os.environ.pop("SERPAPI_API_KEY", None)

    results = search_comparison("", max_results=3)

    assert results == []
