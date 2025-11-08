from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racen.log import get_logger
from racen.orchestrator import ingest_url

logger = get_logger("racen.ingest")


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest a URL into RACEN (pgvector)")
    p.add_argument("url", help="Web URL to ingest")
    p.add_argument("--doc-id", dest="doc_id", default=None)
    p.add_argument("--dim", dest="dim", type=int, default=256)
    args = p.parse_args()

    logger.info(f"Ingest starting for URL: {args.url}")
    res = ingest_url(args.url, doc_id=args.doc_id, embedding_dim=args.dim)
    logger.info(
        f"Ingest finished: doc_id={res.doc_id}, chunks={res.chunks_inserted}, embeddings={res.embeddings_inserted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
