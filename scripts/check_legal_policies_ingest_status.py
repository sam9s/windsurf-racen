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

# Load .env automatically so DB config/search_path match the app
try:
    import dotenv  # type: ignore

    dotenv.load_dotenv()
except Exception:
    pass

from racen.step2_write import DBConfig, get_conn


def _load_urls(cfg_path: Path, key: str = "legal_policies") -> List[str]:
    """Load URLs from a YAML file under the given key.

    Args:
        cfg_path (Path): Path to the YAML config.
        key (str): Top-level key name (default: ``legal_policies``).

    Returns:
        List[str]: Normalized list of URL strings.
    """

    if not cfg_path.exists():
        raise FileNotFoundError(f"YAML config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    raw = data.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list under key '{key}' in {cfg_path}")

    urls: List[str] = []
    for u in raw:
        if not u:
            continue
        s = str(u).strip()
        if s:
            urls.append(s)
    return urls


def main() -> int:
    """Check which legal/policy URLs from the YAML are present in the DB.

    This is read-only: it compares URLs from ``grest_legal_policies.yaml``
    against the ``documents`` table and reports which ones are missing.

    Returns:
        int: Zero on success.
    """

    cfg_path = ROOT / "Grest_Data" / "grest_legal_policies.yaml"
    urls = _load_urls(cfg_path, key="legal_policies")

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
