from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable so that `racen` and scripts can be imported when
# tests are run via `python -m pytest` from the project root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racen.product_search import ProductSearchResult, product_search
from scripts.step4_answer import ProductAnswerPlan, build_product_answer_plan


def _assert_same_base(plan: ProductAnswerPlan, expected_base: str) -> None:
    """Helper to assert that primary and siblings share the same base model."""

    if plan.primary is not None:
        assert plan.primary.base_model == expected_base
    for sib in plan.siblings:
        assert sib.base_model == expected_base


def test_plan_exact_base_iphone_16() -> None:
    """Base query 'iphone 16' should pick base 16 as primary (EXACT match)."""

    res: ProductSearchResult = product_search("do you have iphone 16?")
    plan = build_product_answer_plan("do you have iphone 16?", res)

    assert plan.match_type == "EXACT"
    assert plan.primary is not None
    assert plan.primary.family == "iphone"
    assert plan.primary.base_model == "16"
    _assert_same_base(plan, "16")


def test_plan_base_with_variants_iphone_13_family() -> None:
    """Base query 'iphone 13' should expose 13 and its variants as same-base candidates."""

    res: ProductSearchResult = product_search("do you have iphone 13?")
    plan = build_product_answer_plan("do you have iphone 13?", res)

    # Catalog currently has at least base 13 and 13 mini / pro variants.
    assert plan.match_type in {"EXACT", "CLOSE_BUT_DIFFERENT"}
    assert plan.primary is not None
    assert plan.primary.family == "iphone"
    assert plan.primary.base_model == "13"
    _assert_same_base(plan, "13")

    # At least one sibling should be another 13 variant when present in catalog.
    has_mini_or_other = False
    for sib in plan.siblings:
        if sib.base_model == "13":
            has_mini_or_other = True
            break
    assert has_mini_or_other


def test_plan_variant_exact_iphone_11_pro() -> None:
    """Query 'iphone 11 pro' should prioritise the 11 Pro variant as primary when present."""

    res: ProductSearchResult = product_search("do you have iphone 11 pro?")
    plan = build_product_answer_plan("do you have iphone 11 pro?", res)

    assert plan.primary is not None
    assert plan.primary.family == "iphone"
    assert plan.primary.base_model == "11"
    # Expect the primary to be the Pro variant when it exists.
    assert "pro" in plan.primary.variant_tokens or plan.match_type == "CLOSE_BUT_DIFFERENT"
    _assert_same_base(plan, "11")


def test_plan_close_but_different_iphone_16_pro() -> None:
    """Query 'iphone 16 pro' should at least recognise base 16 when Pro is missing."""

    res: ProductSearchResult = product_search("do you have iphone 16 pro?")
    plan = build_product_answer_plan("do you have iphone 16 pro?", res)

    assert plan.match_type in {"EXACT", "CLOSE_BUT_DIFFERENT"}
    assert plan.primary is not None
    assert plan.primary.family == "iphone"
    assert plan.primary.base_model == "16"
    _assert_same_base(plan, "16")


def test_plan_none_for_unknown_model() -> None:
    """Clearly non-existent models should produce NONE with no primary or siblings."""

    res: ProductSearchResult = product_search("do you have iphone 99 ultra mega?")
    plan = build_product_answer_plan("do you have iphone 99 ultra mega?", res)

    assert plan.match_type == "NONE"
    assert plan.primary is None
    assert plan.siblings == []
