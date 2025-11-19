from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable so that `racen` can be imported when tests
# are run via `python -m pytest` from the project root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racen.product_search import product_search


def test_product_search_exact_match_iphone_13() -> None:
    """Query 'iphone 13' should resolve to an EXACT match in the iPhone catalog."""

    result = product_search("do you have iphone 13?")

    assert result.match_type == "EXACT"
    assert result.candidates
    first = result.candidates[0]
    assert first.family == "iphone"
    assert first.base_model == "13"


def test_product_search_close_but_different_iphone_16_pro() -> None:
    """Query 'iphone 16 pro' should fall back to CLOSE_BUT_DIFFERENT when only base 16 exists."""

    result = product_search("do you have iphone 16 pro?")

    # We expect to at least recognize base model 16. If there is no exact
    # Pro variant, we should surface CLOSE_BUT_DIFFERENT with iphone 16
    # candidates instead of returning NONE.
    assert result.match_type in {"EXACT", "CLOSE_BUT_DIFFERENT"}
    assert result.candidates
    first = result.candidates[0]
    assert first.family == "iphone"
    assert first.base_model == "16"


def test_product_search_none_for_unknown_model() -> None:
    """Clearly non-existent models should return NONE without candidates."""

    result = product_search("do you have iphone 99 ultra mega?")

    assert result.match_type == "NONE"
    assert not result.candidates
