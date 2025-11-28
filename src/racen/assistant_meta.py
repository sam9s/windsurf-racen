"""
Assistant meta Q&A handling for identity/capabilities/privacy style questions.

This module provides a deterministic, retrieval-free response path for "meta"
questions about the assistant itself (e.g., who are you, who built you,
where do you run, what can you do). It is designed to be called early in the
answer flow and should return None when the query appears to be primarily about
products to avoid intercepting real product questions.

All responses are short, Markdown-friendly sentences and avoid leaking
implementation details.
"""
from __future__ import annotations

import re
from typing import Optional

from racen.business_facts import get_business_facts


_META_PATTERNS = {
    "identity": [
        r"\bwho\s+are\s+you\b",
        r"\bwho\s+r\s*u\b",
        r"\bwhat\s+are\s+you\b",
        r"\btell\s+me\s+about\s+yourself\b",
        r"\bintroduce\s+yourself\b",
        r"\bwhat\s+is\s+racen\b",
        r"\bracen\s+stands\s+for\b",
    ],
    "creator": [
        r"\bwho\s+made\s+you\b",
        r"\bwho\s+built\s+you\b",
        r"\bwho\s+created\s+you\b",
        r"\bwho\s+developed\s+you\b",
        r"\bwho\s+is\s+your\s+creator\b",
    ],
    "location": [
        r"\bwhere\s+do\s+you\s+live\b",
        r"\bwhere\s+are\s+you\s+located\b",
        r"\bwhere\s+do\s+you\s+run\b",
    ],
    "capabilities": [
        r"\bwhat\s+can\s+you\s+do\b",
        r"\bhow\s+can\s+you\s+help\b",
        r"\bwhat\s+do\s+you\s+do\b",
        r"\bwhat\s+are\s+your\s+capabilities\b",
    ],
    "privacy": [
        r"\b(do\s+you\s+)?(store|save|keep|retain)\s+(my\s+)?"
        r"(data|chats?|conversation)\b",
        r"\bprivacy\b",
        r"\bhow\s+do\s+you\s+use\s+my\s+data\b",
    ],
    "human": [
        r"\bare\s+you\s+human\b",
        r"\bare\s+you\s+(a\s+)?bot\b",
        r"\bare\s+you\s+ai\b",
    ],
    "language": [
        r"\bwhat\s+languages?\s+do\s+you\s+speak\b",
        r"\bwhat\s+language\s+can\s+you\s+use\b",
    ],
}

# Simple guard to avoid intercepting product queries that contain meta words
_PRODUCT_TOKENS = {
    "iphone",
    "ipad",
    "macbook",
    "laptop",
    "phone",
    "pro ",
    " max",
    " mini",
    " plus",
    "refurbished",
    "price",
    "prices",
}


def _match_any(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    for p in patterns:
        if re.search(p, t):
            return True
    return False


def try_answer_meta_question(query: str) -> Optional[str]:
    """
    Return a deterministic answer for assistant meta questions.

    Args:
        query (str): User input.

    Returns:
        Optional[str]: Answer text if handled, else None.
    """
    q = (query or "").strip()
    if not q:
        return None

    ql = q.lower()
    # Avoid intercept when product tokens are present; let main flow handle it
    if any(tok in ql for tok in _PRODUCT_TOKENS):
        return None

    # Identity
    if _match_any(ql, _META_PATTERNS["identity"]):
        return (
            "I’m RACEN — Rapid Automation "
            "Customer Engagement Network. "
            "I’m a GREST assistant. "
            "I help with products, "
            "orders, policies, "
            "and brand reputation."
        )

    # Creator
    if _match_any(ql, _META_PATTERNS["creator"]):
        return "I was built by the GREST team."

    # Location
    if _match_any(ql, _META_PATTERNS["location"]):
        return (
            "I run in the cloud as part of GREST’s systems; "
            "I don’t have a physical location."
        )

    # Capabilities
    if _match_any(ql, _META_PATTERNS["capabilities"]):
        return (
            "I can help with: product availability, pricing, and specs; "
            "order/returns/warranty/shipping policies; buying advice; "
            "simple comparisons; and brand‑reputation basics. "
            "Ask in English or Hinglish."
        )

    # Privacy & data usage
    if _match_any(ql, _META_PATTERNS["privacy"]):
        base = (
            "I use the text you send here and GREST’s public content to answer. "
            "I don’t store personal data by default. Some responses may be "
            "cached briefly to improve speed. For details, see GREST’s "
            "privacy policy."
        )
        # Append canonical privacy policy URL when available in business facts.
        try:
            biz = get_business_facts()
            urls_cfg = getattr(biz, "urls", None)
            privacy_url = getattr(urls_cfg, "privacy", None) if urls_cfg else None
        except Exception:
            privacy_url = None
        if privacy_url:
            return f"{base} Full privacy policy: {privacy_url}"
        return base

    # Human/bot
    if _match_any(ql, _META_PATTERNS["human"]):
        return "I’m an AI assistant, not a human."

    # Language
    if _match_any(ql, _META_PATTERNS["language"]):
        return "I can respond in English and Hinglish (English + Hindi mix)."

    return None
