from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import yaml  # type: ignore

# Ensure local 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env automatically if present, so DB config matches the app
try:
    import dotenv  # type: ignore

    dotenv.load_dotenv()
except Exception:
    pass

from racen.step2_write import DBConfig, get_conn


def _load_iphone_urls(cfg_path: Path, key: str = "iphones") -> List[str]:
    """Load iPhone product URLs from a YAML file.

    Args:
        cfg_path (Path): Path to the YAML configuration file.
        key (str): Top-level key under which URLs are stored (default: "iphones").

    Returns:
        List[str]: Normalized list of non-empty URL strings.
    """

    if not cfg_path.exists():
        raise FileNotFoundError(f"YAML config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    raw_urls = data.get(key)
    if not isinstance(raw_urls, list):
        raise ValueError(f"Expected a list under key '{key}' in {cfg_path}")

    urls: List[str] = []
    for u in raw_urls:
        if not u:
            continue
        s = str(u).strip()
        if s:
            urls.append(s)
    return urls


def main() -> int:
    """Check which iPhone URLs from the YAML are present in the DB.

    This script compares the URLs listed in ``grest_iphone_products.yaml``
    against the ``documents`` table, printing which ones are present and which
    are missing. It is read-only and does not modify the database.

    Returns:
        int: Zero on success.
    """

    cfg_path = ROOT / "Grest_Data" / "grest_iphone_products.yaml"
    urls = _load_iphone_urls(cfg_path, key="iphones")

    conn = get_conn(DBConfig.from_env())
    missing: List[str] = []
    try:
        for url in urls:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM documents WHERE source = %s", (url,))
                row = cur.fetchone()
            if row is None:
                print(f"MISSING: {url}")
                missing.append(url)
            else:
                doc_id = row["id"]
                print(f"OK: {url} -> doc_id={doc_id}")
    finally:
        conn.close()

    print("\nSummary:")
    print(f"  total URLs:   {len(urls)}")
    print(f"  missing docs: {len(missing)}")
    if missing:
        print("  Missing URLs:")
        for u in missing:
            print(f"    - {u}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
