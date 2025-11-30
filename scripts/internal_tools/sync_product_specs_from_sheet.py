"""Sync product specs from a CSV/Sheet into Postgres.

Internal helper script for RACEN.

For now, this script reads a local CSV file (typically the exported Google
Sheet, such as ``outputs/product_specs_bootstrap.csv``) and upserts rows into
``docling.grest_iphone_product_specs`` in the ``racen`` database.

The CSV is expected to have at least the following columns (header row):

- ``S.No.``
- ``Model Details``
- ``Superb``
- ``Good``
- ``Fair``
- ``Slug``
- ``ProductURL``
- ``Details``

In a later phase, this script can be extended to load rows directly from the
Google Sheet via the Sheets API instead of a CSV export, but the DB mapping
will remain the same.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import sys
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from psycopg import Connection
from urllib.request import urlopen

# Ensure the project root (Windsurf_Project) is on sys.path so we can import
# src.racen.* modules when running this file as a standalone script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.racen.step2_write import DBConfig, get_conn  # noqa: E402


LOGGER = logging.getLogger(__name__)
SHEET_CSV_URL_ENV = "IPHONE_SPECS_SHEET_CSV_URL"
ENV_PATH = PROJECT_ROOT / ".env"


def _parse_price(price_str: str) -> Optional[int]:
    """Parse a price string like ``"Rs. 25,999"`` into an integer rupee value.

    Args:
        price_str: Raw price text from the CSV.

    Returns:
        Optional[int]: Parsed integer value, or None when we cannot confidently
        parse it.
    """

    if not price_str:
        return None
    txt = price_str.strip()
    if not txt:
        return None
    # Remove any currency prefixes and commas, keep digits only.
    digits = re.sub(r"[^0-9]", "", txt)
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _load_env_from_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env-style file into os.environ.

    Existing environment variables are not overridden. Lines starting with "#"
    or without an "=" are ignored. Surrounding single/double quotes are
    stripped from values.

    Args:
        path: Path to a .env-style file.

    Returns:
        None
    """

    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _ensure_specs_table(conn: Connection) -> None:
    """Ensure the grest_iphone_product_specs table exists in the current schema.

    Args:
        conn: Open psycopg connection.
    """

    ddl = (
        "CREATE TABLE IF NOT EXISTS docling.grest_iphone_product_specs ("
        " slug TEXT PRIMARY KEY,"
        " model_details TEXT NOT NULL,"
        " price_superb INTEGER,"
        " price_good INTEGER,"
        " price_fair INTEGER,"
        " product_url TEXT NOT NULL,"
        " details TEXT"
        ");"
    )
    with conn.cursor() as cur:
        cur.execute(ddl)


def _load_rows_from_csv(csv_path: Path) -> List[Dict[str, str]]:
    """Load product-spec rows from a CSV file on disk.

    Args:
        csv_path: Path to a CSV file with the expected headers.

    Returns:
        List[Dict[str, str]]: Row dictionaries keyed by column name.
    """

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return _load_rows_from_csv_filehandle(fh)


def _load_rows_from_csv_url(url: str) -> List[Dict[str, str]]:
    """Load product-spec rows from a CSV served at an HTTP(S) URL.

    This is intended for Google Sheets CSV export links specified via the
    SHEET_CSV_URL_ENV environment variable.

    Args:
        url: HTTP(S) URL returning CSV content with the expected headers.

    Returns:
        List[Dict[str, str]]: Row dictionaries keyed by column name.
    """

    LOGGER.info("Fetching CSV from URL %s", url)
    try:
        with urlopen(url) as resp:  # type: ignore[call-arg]
            data = resp.read().decode("utf-8")
    except Exception as exc:  # Reason: explicit error context for ops/debugging
        raise RuntimeError(f"Failed to fetch CSV from URL {url!r}: {exc}") from exc

    fh = StringIO(data)
    return _load_rows_from_csv_filehandle(fh)


def _load_rows_from_csv_filehandle(fh) -> List[Dict[str, str]]:
    """Parse product-spec rows from an open CSV file-like handle.

    Args:
        fh: Text file-like object positioned at the start of CSV data.

    Returns:
        List[Dict[str, str]]: Row dictionaries keyed by column name.
    """

    rows: List[Dict[str, str]] = []
    reader = csv.DictReader(fh)
    for raw_row in reader:
        # Normalise keys to exactly the ones we expect; missing keys default
        # to empty strings.
        row: Dict[str, str] = {
            "S.No.": (raw_row.get("S.No.") or "").strip(),
            "Model Details": (raw_row.get("Model Details") or "").strip(),
            "Superb": (raw_row.get("Superb") or "").strip(),
            "Good": (raw_row.get("Good") or "").strip(),
            "Fair": (raw_row.get("Fair") or "").strip(),
            "Slug": (raw_row.get("Slug") or "").strip(),
            "ProductURL": (raw_row.get("ProductURL") or "").strip(),
            "Details": (raw_row.get("Details") or "").strip(),
        }
        # Skip rows without a slug; they cannot be mapped reliably.
        if not row["Slug"]:
            continue
        rows.append(row)
    return rows


def _find_duplicate_slugs(rows: Iterable[Dict[str, str]]) -> List[str]:
    """Find any duplicate Slug values in the given rows.

    Args:
        rows: Iterable of row dictionaries from the CSV/Sheet.

    Returns:
        List[str]: Slug values that appear more than once.
    """

    counter: Counter[str] = Counter()
    for row in rows:
        slug = (row.get("Slug") or "").strip()
        if slug:
            counter[slug] += 1
    return [slug for slug, count in counter.items() if count > 1]


def _summarise_missing_prices(rows: Iterable[Dict[str, str]]) -> tuple[List[str], List[str]]:
    """Identify rows with missing condition prices.

    Args:
        rows: Iterable of row dictionaries from the CSV/Sheet.

    Returns:
        tuple[List[str], List[str]]: Two lists of slugs:
            - slugs_all_missing: all three prices are missing/unparseable.
            - slugs_some_missing: at least one price is missing/unparseable,
              but not all three.
    """

    slugs_all_missing: List[str] = []
    slugs_some_missing: List[str] = []

    for row in rows:
        slug = (row.get("Slug") or "").strip()
        if not slug:
            continue
        superb = _parse_price(row.get("Superb", ""))
        good = _parse_price(row.get("Good", ""))
        fair = _parse_price(row.get("Fair", ""))
        prices = (superb, good, fair)
        if all(p is None for p in prices):
            slugs_all_missing.append(slug)
        elif any(p is None for p in prices):
            slugs_some_missing.append(slug)

    return slugs_all_missing, slugs_some_missing


def _sync_rows(conn: Connection, rows: Iterable[Dict[str, str]]) -> int:
    """Replace contents of grest_iphone_product_specs with the given rows.

    This function performs a full replace (delete then insert) so that the
    database matches the sheet exactly at the moment of sync.

    Args:
        conn: Open psycopg connection.
        rows: Iterable of row dictionaries from the CSV/Sheet.

    Returns:
        int: Number of rows written.
    """

    count = 0

    # Ensure the target table exists before attempting to delete/insert.
    _ensure_specs_table(conn)

    with conn.cursor() as cur:
        LOGGER.info("Clearing existing rows from docling.grest_iphone_product_specs")
        cur.execute("DELETE FROM docling.grest_iphone_product_specs")

        insert_sql = (
            "INSERT INTO docling.grest_iphone_product_specs "
            "(slug, model_details, price_superb, price_good, price_fair, product_url, details) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )

        for row in rows:
            slug = row["Slug"]
            model_details = row["Model Details"] or slug
            price_superb = _parse_price(row["Superb"])
            price_good = _parse_price(row["Good"])
            price_fair = _parse_price(row["Fair"])
            product_url = row["ProductURL"]
            details = row["Details"] or None

            cur.execute(
                insert_sql,
                (
                    slug,
                    model_details,
                    price_superb,
                    price_good,
                    price_fair,
                    product_url,
                    details,
                ),
            )
            count += 1

    return count


def sync_specs_from_sheet(csv_path: Optional[str] = None) -> Dict[str, object]:
    """Synchronise product specs from the configured Sheet/CSV into Postgres.

    This is the programmatic equivalent of :func:`main`, intended for use by
    admin HTTP endpoints (for example, when triggered from a Slack command).

    Args:
        csv_path: Optional override path to a CSV file. This is only used when
            the :data:`SHEET_CSV_URL_ENV` environment variable is not set.

    Returns:
        Dict[str, object]: Summary dictionary with keys:
            - ``status``: ``"ok"``, ``"skipped"``, or ``"error"``.
            - ``reason``: Short machine-readable reason string.
            - ``rows_written``: Number of rows synced into the table.
            - ``duplicate_slugs``: List of duplicate slugs, if any.
            - ``slugs_all_missing``: Slugs with all prices missing.
            - ``slugs_some_missing``: Slugs with some prices missing.
            - ``source``: String describing the data source (URL or path).
    """

    _load_env_from_file(ENV_PATH)

    sheet_csv_url = os.getenv(SHEET_CSV_URL_ENV)
    rows: List[Dict[str, str]]
    source: str

    if sheet_csv_url:
        LOGGER.info("%s is set; loading rows from %s", SHEET_CSV_URL_ENV, sheet_csv_url)
        rows = _load_rows_from_csv_url(sheet_csv_url)
        source = sheet_csv_url
    else:
        effective_csv = Path(
            csv_path or (PROJECT_ROOT / "outputs" / "product_specs_bootstrap.csv")
        ).resolve()
        LOGGER.info("Loading rows from %s", effective_csv)
        rows = _load_rows_from_csv(effective_csv)
        source = str(effective_csv)

    if not rows:
        LOGGER.warning("No rows loaded from %s; aborting without DB changes.", source)
        return {
            "status": "skipped",
            "reason": "no_rows",
            "rows_written": 0,
            "duplicate_slugs": [],
            "slugs_all_missing": [],
            "slugs_some_missing": [],
            "source": source,
        }

    duplicate_slugs = _find_duplicate_slugs(rows)
    if duplicate_slugs:
        LOGGER.error(
            "Aborting sync: duplicate Slug values found in sheet: %s. "
            "Please fix duplicates in the Google Sheet and rerun.",
            ", ".join(sorted(duplicate_slugs)),
        )
        return {
            "status": "error",
            "reason": "duplicate_slugs",
            "rows_written": 0,
            "duplicate_slugs": sorted(duplicate_slugs),
            "slugs_all_missing": [],
            "slugs_some_missing": [],
            "source": source,
        }

    slugs_all_missing, slugs_some_missing = _summarise_missing_prices(rows)
    if slugs_all_missing:
        LOGGER.warning(
            "No prices set (Superb/Good/Fair) for slugs: %s. These products "
            "may be ignored in price-based answers until prices are filled.",
            ", ".join(sorted(slugs_all_missing)),
        )
    if slugs_some_missing:
        LOGGER.warning(
            "Some condition prices (Superb/Good/Fair) are missing for slugs: %s. "
            "RACEN will use available prices and fall back or skip where "
            "values are missing.",
            ", ".join(sorted(slugs_some_missing)),
        )

    cfg = DBConfig.from_env()
    conn: Optional[Connection] = None
    try:
        conn = get_conn(cfg)
        written = _sync_rows(conn, rows)
        LOGGER.info("Synced %d rows into docling.grest_iphone_product_specs", written)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    return {
        "status": "ok",
        "reason": "",
        "rows_written": written,
        "duplicate_slugs": [],
        "slugs_all_missing": sorted(slugs_all_missing),
        "slugs_some_missing": sorted(slugs_some_missing),
        "source": source,
    }


def main() -> None:
    """CLI entry point for syncing product specs from CSV into Postgres."""

    parser = argparse.ArgumentParser(description="Sync product specs from CSV into Postgres.")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "product_specs_bootstrap.csv"),
        help=(
            "Path to the CSV file exported from the Google Sheet. "
            "Defaults to outputs/product_specs_bootstrap.csv. "
            "Ignored when the IPHONE_SPECS_SHEET_CSV_URL environment "
            "variable is set."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    summary = sync_specs_from_sheet(csv_path=args.csv_path)
    status = str(summary.get("status", ""))
    if status != "ok":
        reason = str(summary.get("reason", ""))
        LOGGER.error("Specs sync did not complete successfully: %s", reason)


if __name__ == "__main__":
    main()
