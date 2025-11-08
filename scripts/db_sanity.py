from __future__ import annotations

import sys
from pathlib import Path

# Ensure local 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racen.step2_write import get_conn, DBConfig


def main() -> None:
    conn = get_conn(DBConfig.from_env())
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS c FROM documents")
            docs = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) AS c FROM chunks")
            chunks = cur.fetchone()["c"]
            cur.execute("SELECT COUNT(*) AS c FROM embeddings")
            embs = cur.fetchone()["c"]
            cur.execute(
                """
                SELECT COUNT(*) AS c
                FROM chunks c
                LEFT JOIN embeddings e ON e.chunk_id = c.id
                WHERE e.chunk_id IS NULL
                """
            )
            missing = cur.fetchone()["c"]
        print(f"documents: {docs}")
        print(f"chunks: {chunks}")
        print(f"embeddings: {embs}")
        print(f"chunks_without_embeddings: {missing}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
