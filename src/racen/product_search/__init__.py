from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, List, Literal, Optional

import yaml  # type: ignore


MatchType = Literal["EXACT", "CLOSE_BUT_DIFFERENT", "NONE"]


@dataclass
class ProductCandidate:
    """Structured representation of a single catalog product.

    This is intentionally generic so it can be reused across brands/sites.

    Args:
        id: Stable internal identifier for the product.
        name: Human-readable product name.
        url: Canonical product page URL.
        family: High-level family/category (e.g., "iphone", "macbook").
        base_model: Base model identifier within the family (e.g., "13", "16").
        variant_tokens: Variant descriptors (e.g., ["pro"], ["pro", "max"], ["mini"]).
        attributes: Optional free-form attributes such as storage, color, condition.
    """

    id: str
    name: str
    url: str
    family: str
    base_model: str
    variant_tokens: List[str]
    attributes: Optional[Dict[str, str]] = None


@dataclass
class ProductSearchResult:
    """Result of a high-level product search.

    Args:
        match_type: One of "EXACT", "CLOSE_BUT_DIFFERENT", or "NONE".
        candidates: Ordered list of product candidates (best-first).
    """

    match_type: MatchType
    candidates: List[ProductCandidate]


def _project_root() -> Path:
    """Return the project root directory.

    Returns:
        Path: Absolute path to the Windsurf_Project root.
    """

    # __file__ = .../Windsurf_Project/src/racen/product_search/__init__.py
    # parents: [0]=product_search, [1]=racen, [2]=src, [3]=Windsurf_Project.
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _load_iphone_catalog() -> List[ProductCandidate]:
    """Load iPhone products from the grest_iphone_products.yaml file.

    The YAML file is expected to live under Grest_Data/grest_iphone_products.yaml
    and contain a top-level key ``iphones`` with a list of URLs.

    Returns:
        List[ProductCandidate]: Parsed catalog entries for iPhone products.
    """

    root = _project_root()
    cfg_path = root / "Grest_Data" / "grest_iphone_products.yaml"
    if not cfg_path.exists():
        return []

    data: Dict[str, object]
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
        data = dict(raw) if isinstance(raw, dict) else {}

    urls = data.get("iphones")
    if not isinstance(urls, list):
        return []

    candidates: List[ProductCandidate] = []
    for idx, raw_url in enumerate(urls):
        if not raw_url:
            continue
        url = str(raw_url).strip()
        if not url:
            continue

        # Strip query parameters to get a stable handle.
        handle = url.split("?", 1)[0].rstrip("/")
        slug = handle.rsplit("/", 1)[-1]

        family, base_model, variants = _parse_iphone_slug(slug)
        if family != "iphone" or not base_model:
            # If we cannot confidently parse this as an iPhone, skip for now.
            continue

        name = slug.replace("-", " ").title()
        candidate = ProductCandidate(
            id=f"iphone:{idx}",
            name=name,
            url=handle,
            family=family,
            base_model=base_model,
            variant_tokens=variants,
            attributes=None,
        )
        candidates.append(candidate)

    return candidates


def _parse_iphone_slug(slug: str) -> tuple[str, str, List[str]]:
    """Parse an iPhone product slug into family, base model and variants.

    Args:
        slug: Last path segment of the product URL (without query string).

    Returns:
        Tuple[str, str, List[str]]: (family, base_model, variant_tokens).
            family will be "iphone" when recognized, otherwise "".
    """

    parts = [p for p in slug.lower().split("-") if p]
    family = ""
    base_model = ""
    variants: List[str] = []

    # Find the position of the "iphone" token in the slug.
    try:
        idx = parts.index("iphone")
    except ValueError:
        return family, base_model, variants

    family = "iphone"
    # Base model is usually the next token (e.g., 11, 12, 13, 14, 15, 16, xr, xs).
    if idx + 1 < len(parts):
        base_candidate = parts[idx + 1]
        # Accept numeric or common non-numeric model identifiers.
        if re.fullmatch(r"\d+", base_candidate) or base_candidate in {
            "x",
            "xr",
            "xs",
            "se",
        }:
            base_model = base_candidate
            variants = parts[idx + 2 :]
        else:
            # If the next token is not a clean model identifier, treat it as variant.
            base_model = base_candidate
            variants = parts[idx + 2 :]
    # Normalize some common combined variants.
    norm_variants: List[str] = []
    for v in variants:
        if not v:
            continue
        if v in {"max", "mini", "plus", "pro"}:
            norm_variants.append(v)
        elif v in {"promax", "pro-max"}:
            norm_variants.extend(["pro", "max"])
        else:
            norm_variants.append(v)

    return family, base_model, norm_variants


def _extract_iphone_query_bits(query: str) -> tuple[str, List[str]]:
    """Extract base model and variant tokens from an iPhone-style query.

    Args:
        query: Raw user query text.

    Returns:
        Tuple[str, List[str]]: (base_model, variant_tokens).
            base_model may be an empty string when not found.
    """

    q = query.lower()
    tokens = re.findall(r"[a-z0-9]+", q)
    base_model = ""
    variants: List[str] = []

    # Find the iphone token and look at the following words.
    try:
        idx = tokens.index("iphone")
    except ValueError:
        return base_model, variants

    following = tokens[idx + 1 :]
    if following:
        first = following[0]
        if re.fullmatch(r"\d+", first) or first in {"x", "xr", "xs", "se"}:
            base_model = first
            tail = following[1:]
        else:
            base_model = first
            tail = following[1:]
    else:
        tail = []

    for t in tail:
        if t in {"pro", "max", "mini", "plus"}:
            variants.append(t)
        elif t in {"promax", "pro-max"}:
            variants.extend(["pro", "max"])

    return base_model, variants


def product_search(
    query: str,
    family_hint: Optional[str] = None,
) -> ProductSearchResult:
    """High-level product search abstraction.

    For now this is implemented only for iPhone products and uses the
    Grest iPhone catalog loaded from ``Grest_Data/grest_iphone_products.yaml``.

    Args:
        query: Raw user query text.
        family_hint: Optional hint about the product family (e.g., "iphone").

    Returns:
        ProductSearchResult: Match type and ordered candidates.
    """

    q = (query or "").strip()
    if not q:
        return ProductSearchResult(match_type="NONE", candidates=[])

    ql = q.lower()
    # Only handle iphone-style queries in this first backend.
    if "iphone" not in ql and (family_hint or "").lower() != "iphone":
        return ProductSearchResult(match_type="NONE", candidates=[])

    catalog = _load_iphone_catalog()
    if not catalog:
        return ProductSearchResult(match_type="NONE", candidates=[])

    base_model_q, variants_q = _extract_iphone_query_bits(ql)
    if not base_model_q:
        # If we cannot even infer the base model, fall back to NONE for now.
        return ProductSearchResult(match_type="NONE", candidates=[])

    variants_q_set = set(variants_q)
    exact: List[ProductCandidate] = []
    close: List[ProductCandidate] = []

    for cand in catalog:
        if cand.family != "iphone":
            continue
        if cand.base_model != base_model_q:
            continue

        cand_var_set = set(cand.variant_tokens)
        if variants_q_set and variants_q_set.issubset(cand_var_set):
            exact.append(cand)
        elif not variants_q_set and not cand_var_set:
            # User did not specify a variant and product is also base model.
            exact.append(cand)
        else:
            close.append(cand)

    if exact:
        # When we have at least one exact match, treat the query as EXACT but
        # still return all same-base candidates so downstream logic can surface
        # sibling variants (e.g., iPhone 13 mini / Pro) alongside the base
        # model. Exact matches come first, followed by close variants.
        ordered = list(exact) + list(close)
        return ProductSearchResult(match_type="EXACT", candidates=ordered)

    if close:
        return ProductSearchResult(match_type="CLOSE_BUT_DIFFERENT", candidates=close)

    return ProductSearchResult(match_type="NONE", candidates=[])
