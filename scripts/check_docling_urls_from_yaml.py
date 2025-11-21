from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Set

# Ensure local 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env automatically if present
try:  # pragma: no cover - defensive
    import dotenv  # type: ignore

    dotenv.load_dotenv()
except Exception:  # pragma: no cover - CLI guard
    pass

import yaml  # type: ignore

from racen.log import get_logger
from racen.step2_write import DBConfig, get_conn


logger = get_logger("racen.check_docling_urls")


def _collect_urls_from_yaml(path: Path) -> List[str]:
    """Collect all string URLs from a YAML file.

    Args:
        path (Path): Path to the YAML file.

    Returns:
        List[str]: List of URLs discovered in the YAML structure.
    """
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    urls: List[str] = []

    def _walk(node: object) -> None:
        """Recursively walk YAML data and collect string leaves.

        Args:
            node (object): Current YAML node.
        """
        if isinstance(node, str):
            s = node.strip()
            if s:
                urls.append(s)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            for v in node.values():
                _walk(v)

    _walk(data)
    return urls


def _load_yaml_urls(directory: Path, pattern: str) -> List[str]:
    """Load and deduplicate URLs from all matching YAML files.

    Args:
        directory (Path): Base directory containing YAML files.
        pattern (str): Glob pattern for YAML files (e.g., "*.yaml").

    Returns:
        List[str]: Sorted list of unique URLs across all files.
    """
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No YAML files matching {pattern!r} in {directory}")

    all_urls: List[str] = []
    for p in files:
        file_urls = _collect_urls_from_yaml(p)
        logger.info("Loaded %d URLs from %s", len(file_urls), p.name)
        all_urls.extend(file_urls)

    cleaned = [u.strip() for u in all_urls if isinstance(u, str) and u.strip()]
    unique = sorted(set(cleaned))
    logger.info("Collected %d unique URLs from %d YAML files", len(unique), len(files))
    return unique


def _fetch_db_sources(table: str, column: str) -> Set[str]:
    """Fetch distinct URL-like sources from the Postgres documents table.

    Args:
        table (str): Qualified table name (e.g., "docling.documents").
        column (str): Column name containing the URL/source (e.g., "source").

    Returns:
        Set[str]: Set of distinct non-empty source values.
    """
    cfg = DBConfig.from_env()
    conn = get_conn(cfg)

    try:
        with conn.cursor() as cur:
            cur.execute("SHOW search_path")
            row = cur.fetchone() or {}
            search_path = row.get("search_path") if isinstance(row, dict) else None
            logger.info("search_path: %s", search_path)

            query = f"SELECT DISTINCT {column} AS source FROM {table}"
            cur.execute(query)
            rows = cur.fetchall() or []
    finally:
        conn.close()

    sources: Set[str] = set()
    for r in rows:
        if not isinstance(r, dict):
            continue
        raw = r.get("source")
        if not raw:
            continue
        s = str(raw).strip()
        if s:
            sources.add(s)

    logger.info("Fetched %d distinct sources from %s", len(sources), table)
    return sources


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point: compare Grest YAML URLs against docling.documents.

    Args:
        argv (Iterable[str] | None): Optional CLI arguments.

    Returns:
        int: Process exit code (0 on success, 1 if any URLs are missing).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check that all URLs referenced in Grest YAML configs "
            "exist in the specified Postgres documents table."
        )
    )
    default_dir = ROOT / "Grest_Data"
    parser.add_argument(
        "--dir",
        type=str,
        default=str(default_dir),
        help=f"Directory containing YAML configs (default: {default_dir})",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.yaml",
        help="Glob pattern for YAML files (default: *.yaml)",
    )
    parser.add_argument(
        "--table",
        type=str,
        default="docling.documents",
        help="Qualified Postgres table to check (default: docling.documents)",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="source",
        help="Column name holding the URL/source (default: source)",
    )
    parser.add_argument(
        "--show-extra",
        action="store_true",
        help=(
            "Also print EXTRA: lines for sources present in the DB but not in the YAML files "
            "(default: only report missing YAML URLs)."
        ),
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    base_dir = Path(args.dir).resolve()
    logger.info("Scanning YAML files in %s (pattern=%s)", base_dir, args.glob)

    try:
        yaml_urls = _load_yaml_urls(base_dir, args.glob)
    except Exception as exc:  # pragma: no cover - CLI guard
        logger.error("Failed to load URLs from YAML files: %s", exc)
        return 1

    print(f"Total YAML URLs (deduped): {len(yaml_urls)}")

    db_sources = _fetch_db_sources(args.table, args.column)

    missing = sorted(u for u in yaml_urls if u not in db_sources)
    extras = sorted(s for s in db_sources if s not in set(yaml_urls))

    print(f"Distinct DB sources from {args.table}: {len(db_sources)}")
    print(f"Missing URLs (YAML -> DB): {len(missing)}")

    for m in missing:
        print(f"MISSING: {m}")

    if args.show_extra and extras:
        print(f"Extra URLs in {args.table} not present in YAML configs: {len(extras)}")
        for e in extras:
            print(f"EXTRA: {e}")

    if missing:
        logger.warning("Found %d missing URLs compared to YAML configs", len(missing))
        return 1

    logger.info("All YAML URLs are present in %s", args.table)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
