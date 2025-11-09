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

logger = get_logger("scripts.migrate_add_chunk_lines")


def main() -> None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            logger.info("Adding start_line and end_line to chunks if missing")
            cur.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS start_line INT"
            )
            cur.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS end_line INT"
            )
            logger.info("Done")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
