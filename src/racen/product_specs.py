from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class ProductSpecs:
    """Structured specs extracted from a single product page snippet.

    This is intentionally minimal and generic so it can evolve with the
    Grest product pages without changing the public interface.
    """

    price_strings: List[str]
    storage_options: List[str]
    conditions: List[str]
    warranty_strings: List[str]
    color_options: List[str]


_PRICE_PATTERNS = [
    # ₹ 14,499 or ₹14,499 or Rs. 14,499
    # Allow optional whitespace between the currency symbol and the digits.
    re.compile(r"(?:₹\s*|Rs\.?\s*)([0-9]{1,3}(?:,[0-9]{2,3})+)"),
    # 14499 (plain number) preceded by 'price' within a short window
    re.compile(r"price[^\n]{0,40}?([0-9]{4,6})", re.IGNORECASE),
]

_STORAGE_PATTERN = re.compile(r"(\d+\s*(?:GB|TB))", re.IGNORECASE)

# Common refurb conditions used on electronics sites
_CONDITION_PATTERNS = [
    re.compile(r"(brand\s*new)", re.IGNORECASE),
    re.compile(r"(like\s*new)", re.IGNORECASE),
    re.compile(r"(open\s*box)", re.IGNORECASE),
    re.compile(r"(excellent\s*condition)", re.IGNORECASE),
    re.compile(r"(very\s*good\s*condition)", re.IGNORECASE),
    re.compile(r"(good\s*condition)", re.IGNORECASE),
]

_WARRANTY_PATTERN = re.compile(
    r"(\d+\s*(?:month|months|year|years)\s*(?:warranty|guarantee))",
    re.IGNORECASE,
)

_COLOR_PATTERN = re.compile(
    r"\b(black|white|blue|red|green|yellow|pink|purple|gold|silver|space\s*grey|space\s*gray|midnight|starlight|product\s*red)\b",
    re.IGNORECASE,
)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def extract_product_specs(text: str) -> ProductSpecs:
    """Best-effort, deterministic extraction of key specs from product text.

    The function returns *all* matches it can reliably see, instead of trying
    to guess a single canonical value. The phrasing layer can then decide
    which price or storage option to surface (e.g., sale vs MRP) using
    explicit rules, but this extractor stays purely pattern-based.
    """

    cleaned = _normalize_whitespace(text)

    # Prices
    price_strings: List[str] = []
    seen_price_spans: List[tuple[int, int]] = []
    for pat in _PRICE_PATTERNS:
        for m in pat.finditer(cleaned):
            span = m.span(0)
            if any(s <= span[0] < e or s < span[1] <= e for s, e in seen_price_spans):
                continue
            seen_price_spans.append(span)
            price_strings.append(cleaned[span[0] : span[1]].strip())

    # Storage: normalise to e.g. "64 GB", "256 GB", "1 TB".
    storage_raw = set()
    for m in _STORAGE_PATTERN.finditer(cleaned):
        val = m.group(1).upper().strip()
        # Collapse any internal whitespace and ensure single space before unit
        val = re.sub(r"\s+", " ", val)
        val = val.replace("GB", " GB").replace("TB", " TB")
        val = re.sub(r"\s+GB", " GB", val)
        val = re.sub(r"\s+TB", " TB", val)
        storage_raw.add(val.strip())
    storage_options = sorted(storage_raw)

    # Conditions
    conditions_set = set()
    for pat in _CONDITION_PATTERNS:
        for m in pat.finditer(cleaned):
            conditions_set.add(_normalize_whitespace(m.group(1)))
    conditions = sorted(conditions_set)

    # Warranty
    warranty_strings = sorted(
        {_normalize_whitespace(m.group(1)) for m in _WARRANTY_PATTERN.finditer(cleaned)}
    )

    # Colors
    color_options = sorted(
        {m.group(1).lower() for m in _COLOR_PATTERN.finditer(cleaned)}
    )

    return ProductSpecs(
        price_strings=price_strings,
        storage_options=storage_options,
        conditions=conditions,
        warranty_strings=warranty_strings,
        color_options=color_options,
    )
