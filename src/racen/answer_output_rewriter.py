from __future__ import annotations

"""LLM-based output language rewriter for final answers.

This module is responsible for rewriting the *final* answer text produced by
`answer_query` into the user's language/mode (for example, Hinglish) while
preserving structured details such as URLs, product names, and numeric
prices.

It is separate from the input-side query normalizer so that we can:

- Normalize noisy queries into clean English for intent/price parsing.
- Optionally localize the final answer back into the user's language without
  altering the underlying logic.
"""

import os
from typing import Optional

import requests
from pydantic import BaseModel

from racen.log import get_logger


logger = get_logger("racen.answer_output_rewriter")


class AnswerRewriteResult(BaseModel):
    """Result of an output language rewrite.

    Args:
        original_answer: The original answer text before rewriting.
        rewritten_answer: The answer text after rewriting.
        model: The underlying LLM model used for rewriting.
        used_rewriter: True when a remote LLM call was attempted.
    """

    original_answer: str
    rewritten_answer: str
    model: str
    used_rewriter: bool


def _rewrite_enabled() -> bool:
    """Return True when output rewriting is enabled via env toggle.

    Returns:
        bool: True if RACEN_ANSWER_OUTPUT_REWRITE_ENABLE is truthy.
    """

    val = os.getenv("RACEN_ANSWER_OUTPUT_REWRITE_ENABLE", "0")
    return val in {"1", "true", "TRUE", "yes"}


def _should_rewrite_for_mode(user_mode: Optional[str]) -> bool:
    """Decide whether we should attempt a rewrite based on user mode.

    Args:
        user_mode: Detected user mode string (for example, "EN" or "HI_EN").

    Returns:
        bool: True when the answer should be localized for this mode.
    """

    if not user_mode:
        return False
    mode = user_mode.strip().upper()
    # Treat Hinglish/Hindi-style modes as candidates for localization.
    return mode in {"HI_EN", "HI", "HIN", "HI-IN"}


def rewrite_answer_language(
    answer: str,
    user_query: str,
    user_mode: Optional[str] = None,
) -> AnswerRewriteResult:
    """Rewrite the final answer into the user's language when appropriate.

    This helper is called at the *end* of the answer pipeline. It takes the
    canonical answer text (usually English) and, when enabled and when the
    user mode suggests Hinglish/Hindi, uses a small LLM call to rewrite the
    answer into a more natural Hinglish/Hindi style.

    The rewriter must preserve URLs, product names, and numeric values
    exactly as they appear.

    Args:
        answer: Final answer text produced by `answer_query`.
        user_query: Original user query text.
        user_mode: Optional detected user mode (for example, "EN", "HI_EN").

    Returns:
        AnswerRewriteResult: Structured result with the rewritten answer.
    """

    original = (answer or "").strip()
    if not original:
        return AnswerRewriteResult(
            original_answer="",
            rewritten_answer="",
            model="",
            used_rewriter=False,
        )

    if not _rewrite_enabled():
        return AnswerRewriteResult(
            original_answer=original,
            rewritten_answer=original,
            model="",
            used_rewriter=False,
        )

    if not _should_rewrite_for_mode(user_mode):
        return AnswerRewriteResult(
            original_answer=original,
            rewritten_answer=original,
            model="",
            used_rewriter=False,
        )

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; skipping answer output rewrite.")
        return AnswerRewriteResult(
            original_answer=original,
            rewritten_answer=original,
            model="",
            used_rewriter=False,
        )

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = (
        os.getenv("RACEN_ANSWER_OUTPUT_REWRITE_MODEL")
        or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )

    # Reason: keep the prompt narrow and instruct the model not to touch URLs
    # or numbers while allowing natural Hinglish/Hindi phrasing.
    system_prompt = (
        "You rewrite finalized support answers into natural Hinglish (Hindi + English) "
        "or Hindi, matching the user's language. Fix minor grammar or phrasing, "
        "but do not change the meaning. Do not add new information. Preserve all "
        "URLs, product names, and numeric values (prices, counts) exactly as they "
        "appear. Keep any bullet points and Markdown formatting. Return only the "
        "rewritten answer text."
    )

    user_content = (
        "Original question (for context):\n"
        f"{(user_query or '').strip()}\n\n"
        "Answer to rewrite (do not change URLs or numbers):\n"
        f"{original}\n\n"
        "Rewritten answer:"
    )

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
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
            return AnswerRewriteResult(
                original_answer=original,
                rewritten_answer=original,
                model=model,
                used_rewriter=True,
            )
        return AnswerRewriteResult(
            original_answer=original,
            rewritten_answer=content,
            model=model,
            used_rewriter=True,
        )
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Answer output rewrite failed: %s", exc)
        return AnswerRewriteResult(
            original_answer=original,
            rewritten_answer=original,
            model=model,
            used_rewriter=False,
        )
