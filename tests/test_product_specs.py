from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is importable so that `racen` can be imported when tests
# are run via `python -m pytest` from the project root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racen.product_specs import extract_product_specs


def test_extract_product_specs_basic_iphone_11_snippet() -> None:
    text = """
    Refurbished Apple iPhone 11 (64 GB)

    Sale price ₹ 14,499
    MRP ₹ 39,999

    Available in Black, White and Product Red.
    Condition: Excellent Condition
    Warranty: 6 months warranty from Grest.
    """

    specs = extract_product_specs(text)

    # We should capture both price lines exactly as they appear.
    assert any("₹ 14,499" in p for p in specs.price_strings)
    assert any("₹ 39,999" in p for p in specs.price_strings)

    # Storage detection
    assert "64 GB" in specs.storage_options

    # Condition
    assert any("Excellent Condition".lower() == c.lower() for c in specs.conditions)

    # Warranty
    assert any("6 months warranty" in w.lower() for w in specs.warranty_strings)

    # Colors
    assert "black" in specs.color_options
    assert "white" in specs.color_options
    assert any("product red" == c for c in specs.color_options)


def test_extract_product_specs_multiple_storage_and_colors() -> None:
    text = """
    Refurbished iPhone 13 mini – 128GB / 256GB

    Starting from Rs. 27,999 depending on storage.
    Available colors: Midnight, Starlight, Blue.
    """

    specs = extract_product_specs(text)

    # Storage options
    assert "128 GB" in specs.storage_options
    assert "256 GB" in specs.storage_options

    # Price pattern with Rs.
    assert any("Rs." in p or "Rs" in p for p in specs.price_strings)

    # Colors
    assert "midnight" in specs.color_options
    assert "starlight" in specs.color_options
    assert "blue" in specs.color_options


def test_extract_product_specs_handles_missing_fields_gracefully() -> None:
    text = """
    Generic product without explicit price or warranty.
    Only description text is present here.
    """

    specs = extract_product_specs(text)

    assert specs.price_strings == []
    assert specs.storage_options == []
    assert specs.conditions == []
    assert specs.warranty_strings == []
    assert specs.color_options == []


def test_extract_product_specs_iphone_14_plus_snippet() -> None:
    """Extractor should handle an iPhone 14 Plus-style snippet with sale/MRP, storage, colors and warranty."""

    text = """
    Refurbished Apple iPhone 14 Plus (128 GB)

    Sale price ₹ 42,999
    MRP ₹ 79,999

    Available in Midnight and Starlight.
    Condition: Like New
    Warranty: 12 months warranty from Grest.
    """

    specs = extract_product_specs(text)

    # Prices
    assert any("₹ 42,999" in p for p in specs.price_strings)
    assert any("₹ 79,999" in p for p in specs.price_strings)

    # Storage
    assert "128 GB" in specs.storage_options

    # Condition
    assert any("like new" == c.lower() for c in specs.conditions)

    # Warranty
    assert any("12 months warranty" in w.lower() for w in specs.warranty_strings)

    # Colors
    assert "midnight" in specs.color_options
    assert "starlight" in specs.color_options
