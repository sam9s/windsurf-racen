from __future__ import annotations

from pathlib import Path
import sys

# Add project src to sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load environment
try:
    from dotenv import load_dotenv

    for env_path in [
        BASE_DIR / "windsurf-racen-local" / ".env",
        BASE_DIR / ".env",
    ]:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            break
except Exception:
    pass

from racen.step2_write import get_conn
from racen.log import get_logger

logger = get_logger("scripts.migrate_embeddings_dim_1536")


def main() -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            logger.info("Dropping existing vector index if present")
            cur.execute(
                "DROP INDEX IF EXISTS idx_embeddings_vec"
            )
            logger.info("Truncating embeddings table to allow type change")
            cur.execute("TRUNCATE TABLE embeddings")
            logger.info("Altering embeddings.embedding to vector(1536)")
            cur.execute(
                "ALTER TABLE embeddings ALTER COLUMN embedding TYPE vector(1536)"
            )
            logger.info("Recreating vector index (ivfflat, cosine)")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_embeddings_vec ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
            logger.info("Migration complete")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
