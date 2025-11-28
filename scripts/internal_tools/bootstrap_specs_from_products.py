"""Bootstrap a product specs CSV from Grest product URLs.

This script is intended as a one-time (or occasional) helper for internal use.
It reads the iPhone catalog URLs from ``Grest_Data/grest_iphone_products.yaml``,
normalises them to base product URLs (stripping any ``?variant=...`` part),
fetches each page from ``grest.in``, extracts simple structured specs using the
existing ``extract_product_specs`` helper, and writes a CSV file that can be
imported into the shared Google Sheet used for the manual product-specs
database.

The generated CSV has the following columns in this order:

- ``S.No.``: Sequential row index starting from 1 (matches the Google Sheet).
- ``Model Details``: Human-readable model name (e.g., "Refurbished Apple iPhone 14").
- ``Superb`` / ``Good`` / ``Fair``: Initial price strings derived from the
  page. For bootstrap purposes, the same first price string is copied into
  all three columns so the sheet has a starting value to refine.
- ``Slug``: Product slug (for example, ``refurbished-apple-iphone-14``) for
  backend mapping.
- ``ProductURL``: Normalised base product URL without variant query.
- ``Details``: Left blank for manual notes or descriptions.

The output CSV path is ``outputs/product_specs_bootstrap.csv`` relative to the
Windsurf_Project root.
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
import yaml

# Ensure the project root (Windsurf_Project) is on sys.path so we can import
# src.racen.* modules when running this file as a standalone script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.racen.product_specs import ProductSpecs, extract_product_specs


LOGGER = logging.getLogger(__name__)


def _project_root() -> Path:
    """Return the Windsurf_Project root directory.

    Returns:
        Path: Absolute path to the project root.
    """

    # scripts/internal_tools/bootstrap_specs_from_products.py
    # parents: [0]=internal_tools, [1]=scripts, [2]=Windsurf_Project
    return Path(__file__).resolve().parents[2]


def _load_catalog_urls() -> List[str]:
    """Load raw product URLs from the iPhone catalog YAML file.

    Returns:
        List[str]: List of raw product URLs as strings. May include ``?variant``
        query parameters, which will be normalised later.
    """

    root = _project_root()
    cfg_path = root / "Grest_Data" / "grest_iphone_products.yaml"
    if not cfg_path.exists():
        LOGGER.warning("Catalog file not found: %s", cfg_path)
        return []

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        LOGGER.warning("Unexpected catalog YAML structure in %s", cfg_path)
        return []

    urls = raw.get("iphones")
    if not isinstance(urls, list):
        LOGGER.warning("Missing or invalid 'iphones' key in %s", cfg_path)
        return []

    out: List[str] = []
    for item in urls:
        if not item:
            continue
        url = str(item).strip()
        if url:
            out.append(url)
    return out


def _normalize_product_url(raw_url: str) -> Tuple[str, str]:
    """Normalise a product URL to its base handle and slug.

    Args:
        raw_url: Raw URL from the catalog, potentially including ``?variant``
            and a trailing slash.

    Returns:
        Tuple[str, str]: ``(base_url, slug)`` where ``base_url`` has no query
        string and no trailing slash, and ``slug`` is the last path segment.
        If the input is empty, both values are empty strings.
    """

    url = (raw_url or "").strip()
    if not url:
        return "", ""
    base = url.split("?", 1)[0].rstrip("/")
    if not base:
        return "", ""
    slug = base.rsplit("/", 1)[-1] if "/" in base else base
    return base, slug


def _fetch_page(url: str, timeout: int = 20) -> str:
    """Fetch a product page from grest.in.

    Args:
        url: Normalised base product URL.
        timeout: Request timeout in seconds.

    Returns:
        str: Response text (HTML) as UTF-8.
    """

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    if not resp.encoding:
        resp.encoding = "utf-8"
    return resp.text or ""


def _extract_primary_price(specs: ProductSpecs) -> str:
    """Return the first price string from extracted specs, if any.

    Args:
        specs: ProductSpecs extracted from the page.

    Returns:
        str: First price-looking substring, or an empty string when not found.
    """

    if not specs or not specs.price_strings:
        return ""
    return specs.price_strings[0].strip() if specs.price_strings[0] else ""


def _build_rows() -> List[Dict[str, str]]:
    """Build bootstrap rows for the product specs CSV.

    Returns:
        List[Dict[str, str]]: List of row dictionaries keyed by column name.
    """

    raw_urls = _load_catalog_urls()
    seen: Dict[str, str] = {}
    rows: List[Dict[str, str]] = []

    for raw in raw_urls:
        base_url, slug = _normalize_product_url(raw)
        if not base_url or not slug:
            continue
        if base_url in seen:
            continue
        seen[base_url] = slug

        try:
            html = _fetch_page(base_url)
        except Exception as exc:  # Reason: network errors should not stop all rows.
            LOGGER.warning("Failed to fetch %s: %s", base_url, exc)
            continue

        try:
            specs = extract_product_specs(html)
        except Exception as exc:
            LOGGER.warning("Failed to extract specs for %s: %s", base_url, exc)
            continue

        price_str = _extract_primary_price(specs)
        model = slug.replace("-", " ").title() if slug else ""

        row: Dict[str, str] = {
            "Model Details": model,
            "Superb": price_str,
            "Good": price_str,
            "Fair": price_str,
            "Slug": slug,
            "ProductURL": base_url,
            "Details": "",
        }
        rows.append(row)

    return rows


def _write_csv(rows: Iterable[Dict[str, str]]) -> Path:
    """Write the bootstrap rows to the outputs CSV file.

    Args:
        rows: Iterable of row dictionaries.

    Returns:
        Path: Absolute path to the written CSV file.
    """

    root = _project_root()
    out_dir = root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "product_specs_bootstrap.csv"

    fieldnames = [
        "S.No.",
        "Model Details",
        "Superb",
        "Good",
        "Fair",
        "Slug",
        "ProductURL",
        "Details",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            record = dict(row)
            record["S.No."] = str(idx)
            writer.writerow(record)

    return out_path


def main() -> None:
    """Entry point for generating the bootstrap product specs CSV.

    This function is safe to run multiple times; it overwrites the output CSV
    on each run.
    """

    logging.basicConfig(level=logging.INFO)
    rows = _build_rows()
    if not rows:
        LOGGER.warning("No rows generated; nothing to write.")
        return
    out_path = _write_csv(rows)
    LOGGER.info("Wrote %d rows to %s", len(rows), out_path)


if __name__ == "__main__":
    main()
