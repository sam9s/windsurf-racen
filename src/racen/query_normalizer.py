from __future__ import annotations

"""LLM-based query normalization for typos and Hinglish.

This module provides a narrow, side-effect-light helper that rewrites noisy
user queries (spelling mistakes, shorthand, basic Hinglish) into clean
English. The goal is to make downstream intent and price parsing more
robust without hardcoding every possible misspelling.

The normalizer is controlled via environment variables and is safe to
leave disabled by default in tests or low-resource environments.
"""

import os
from typing import Optional

import requests
from pydantic import BaseModel

from racen.log import get_logger


logger = get_logger("racen.query_normalizer")


class NormalizedQueryResult(BaseModel):
    """Structured result from query normalization.

    Args:
        original_query: Raw user query text as received from the caller.
        normalized_query: Rewritten query text, typically in clean English.
        model: Underlying LLM model used for normalization.
        used_normalizer: True when a remote LLM call was made.
    """

    original_query: str
    normalized_query: str
    model: str
    used_normalizer: bool

    @property
    def changed(self) -> bool:
        """Return True when the normalized query differs from the original.

        Returns:
            bool: True if normalization changed the query text.
        """

        return (self.normalized_query or "").strip() != (self.original_query or "").strip()


def _normalization_enabled() -> bool:
    """Check whether query normalization is enabled via env toggle.

    Returns:
        bool: True when RACEN_QUERY_NORMALIZATION_ENABLE is truthy.
    """

    val = os.getenv("RACEN_QUERY_NORMALIZATION_ENABLE", "0")
    return val in {"1", "true", "TRUE", "yes"}


def normalize_query(query: str, user_locale: Optional[str] = None) -> NormalizedQueryResult:
    """Normalize a noisy user query using a small LLM rewrite.

    This helper is intentionally narrow:

    - Fixes common spelling mistakes and SMS-style shorthand.
    - Translates basic Hinglish/Hindi phrasing to clear English while
      preserving intent, product names, and numbers.
    - Never answers the question; it only rewrites the query text.

    When the normalizer is disabled or misconfigured, this function
    returns an identity mapping (normalized_query == original_query)
    and does not raise.

    Args:
        query: Raw user query text.
        user_locale: Optional locale hint (currently informational only).

    Returns:
        NormalizedQueryResult: Normalization result with original and
        normalized text.
    """

    original = (query or "").strip()
    if not original:
        return NormalizedQueryResult(
            original_query="",
            normalized_query="",
            model="",
            used_normalizer=False,
        )

    if not _normalization_enabled():
        # Fast path: normalization disabled, behave as identity.
        return NormalizedQueryResult(
            original_query=original,
            normalized_query=original,
            model="",
            used_normalizer=False,
        )

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; skipping query normalization.")
        return NormalizedQueryResult(
            original_query=original,
            normalized_query=original,
            model="",
            used_normalizer=False,
        )

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = (
        os.getenv("RACEN_QUERY_NORMALIZATION_MODEL")
        or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    # Reason: keep prompt extremely small and focused to control latency/cost.
    system_prompt = (
        "You rewrite noisy customer queries into clear English. "
        "Fix spelling and grammar and translate any Hindi or Hinglish "
        "(Hindi + English mix) into simple English, but do not change the "
        "meaning, products, prices, or constraints. Do not answer the "
        "question. Return only the rewritten query text."
    )

    user_hint = ""
    if user_locale:
        user_hint = f"\nUser locale hint: {user_locale}"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Original query:{user_hint}\n\n{original}\n\nRewritten query:",
            },
        ],
        "temperature": 0.0,
        "max_tokens": 80,
    }

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()
        if not content:
            # Fallback to identity if the model returned an empty string.
            return NormalizedQueryResult(
                original_query=original,
                normalized_query=original,
                model=model,
                used_normalizer=True,
            )
        return NormalizedQueryResult(
            original_query=original,
            normalized_query=content,
            model=model,
            used_normalizer=True,
        )
    except Exception as exc:
        # Never break the main flow due to normalization issues.
        logger.warning("Query normalization failed: %s", exc)
        return NormalizedQueryResult(
            original_query=original,
            normalized_query=original,
            model=model,
            used_normalizer=False,
        )
