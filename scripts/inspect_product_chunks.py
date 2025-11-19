from __future__ import annotations

import argparse
import sys
from pathlib import Path


# Ensure local 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env with the same precedence as other scripts so PGOPTIONS/search_path
# and DB settings match the answer pipeline.
try:
    import dotenv  # type: ignore

    for env_path in [ROOT / "windsurf-racen-local" / ".env", ROOT / ".env"]:
        if env_path.exists():
            dotenv.load_dotenv(dotenv_path=env_path, override=False)
            break
except Exception:
    pass

from racen.step2_write import get_conn, DBConfig  # type: ignore


def main() -> None:
    """Inspect stored chunks for a given product URL pattern.

    This is a read-only helper to see what text (including prices)
    is currently stored in the Postgres corpus for a product URL.
    """
    parser = argparse.ArgumentParser(description="Inspect stored product chunks by URL pattern")
    parser.add_argument(
        "--pattern",
        required=True,
        help="Substring to match in documents.source (e.g., 'refurbished-apple-iphone-13')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of chunks to print",
    )
    parser.add_argument(
        "--contains",
        default="",
        help="Optional substring filter that must appear in chunk text (e.g., '₹' or '27,999')",
    )
    args = parser.parse_args()

    conn = get_conn(DBConfig.from_env())
    try:
        with conn.cursor() as cur:
            sql = (
                "SELECT d.source, "
                "       c.start_line, "
                "       c.end_line, "
                "       LEFT(c.text, 1000) AS snippet "
                "FROM chunks c "
                "JOIN documents d ON c.document_id = d.id "
                "WHERE d.source LIKE %s"
            )
            params = [f"%{args.pattern}%"]
            if args.contains:
                sql += " AND c.text ILIKE %s"
                params.append(f"%{args.contains}%")
            sql += " ORDER BY c.start_line LIMIT %s"
            params.append(args.limit)
            cur.execute(sql, params)
            rows = cur.fetchall()
            if not rows:
                print("No chunks found for pattern:", args.pattern)
                return
            for row in rows:
                source = row["source"]
                start = row["start_line"]
                end = row["end_line"]
                snippet = row["snippet"]
                print(f"{source} ({start}-{end}):")
                print(snippet)
                print("-" * 60)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
