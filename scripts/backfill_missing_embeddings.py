from __future__ import annotations

import sys
from pathlib import Path
import argparse

# Ensure local 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env automatically if present
try:
    import dotenv  # type: ignore
    dotenv.load_dotenv()
except Exception:
    pass

from racen.step2_write import get_conn, DBConfig, upsert_embedding  # noqa: E402
from racen.step2_embed import OpenAIEmbedder  # noqa: E402
from racen.log import get_logger  # noqa: E402

logger = get_logger("racen.backfill")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Backfill embeddings for chunks that are missing vectors"
    )
    p.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    p.add_argument(
        "--limit", type=int, default=500, help="Max chunks to process this run"
    )
    args = p.parse_args()

    conn = get_conn(DBConfig.from_env())
    embedder = OpenAIEmbedder()

    # Fetch missing chunk ids and texts
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, c.text
            FROM chunks c
            LEFT JOIN embeddings e ON e.chunk_id = c.id
            WHERE e.chunk_id IS NULL
            LIMIT %s
            """,
            (args.limit,),
        )
        rows = cur.fetchall()

    if not rows:
        logger.info("No missing embeddings found.")
        return 0

    logger.info(f"Backfilling {len(rows)} chunks (dim={args.dim})")

    filled = 0
    for row in rows:
        cid = row["id"]
        text = row["text"]
        try:
            emb = embedder.embed(id=cid, text=text, metadata=None)
            upsert_embedding(
                conn,
                chunk_id=cid,
                vector=emb.vector,
                model=emb.model,
                embedding_dim=args.dim,
            )
            filled += 1
        except Exception as e:  # keep going on errors
            logger.warning(f"Backfill failed for {cid}: {e}")
            continue

    logger.info(f"Backfill complete: filled={filled} of {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
