from __future__ import annotations

import sys
from pathlib import Path

# Ensure local 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racen.log import get_logger
from racen.step3_retrieve import retrieve

logger = get_logger("racen.smoke3")


def main():
    query = "example domain"
    results = retrieve(query_text=query, top_k=5)
    if not results:
        logger.info("No results found")
        return
    logger.info(f"Top {len(results)} results for: {query}")
    for i, r in enumerate(results, 1):
        preview = (r.text[:140] + "...") if len(r.text) > 140 else r.text
        logger.info(
            f"#{i} score={r.score:.3f} (vec={r.score_vector:.3f}, lex={r.score_lexical:.3f}) | "
            f"chunk={r.chunk_id} doc={r.document_id} src={r.source} | {preview}"
        )


if __name__ == "__main__":
    logger.info("Smoke Step 3 (retrieve): start")
    main()
    logger.info("Smoke Step 3 (retrieve): done")
