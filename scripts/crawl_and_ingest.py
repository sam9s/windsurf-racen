from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

from racen.log import get_logger
from racen.crawler import CrawlConfig, crawl
from racen.orchestrator import ingest_url

logger = get_logger("racen.crawl_ingest")


def main() -> int:
    p = argparse.ArgumentParser(description="Crawl a domain and ingest pages into RACEN (pgvector)")
    p.add_argument("start_url", help="Starting URL (e.g., https://grest.in)")
    p.add_argument("--max-pages", type=int, default=500)
    p.add_argument("--same-domain", action="store_true", help="Restrict crawl to the same domain as start_url")
    p.add_argument("--rps", type=float, default=1.0, help="Requests per second (politeness)")
    p.add_argument("--timeout", type=int, default=20)
    p.add_argument("--dim", type=int, default=256, help="Embedding dimension for pgvector column")
    args = p.parse_args()

    cfg = CrawlConfig(
        start_url=args.start_url,
        max_pages=args.max_pages,
        same_domain=args.same_domain or True,
        rate_limit_rps=args.rps,
        timeout=args.timeout,
    )

    logger.info(
        f"Crawl start: url={cfg.start_url} max_pages={cfg.max_pages} same_domain={cfg.same_domain} rps={cfg.rate_limit_rps}"
    )
    urls = crawl(cfg)

    if not urls:
        logger.info("No pages discovered. Exiting.")
        return 0

    logger.info(f"Discovered {len(urls)} pages. Starting ingest...")
    total_chunks = 0
    total_emb = 0
    for i, url in enumerate(urls, 1):
        try:
            res = ingest_url(url, embedding_dim=args.dim)
            logger.info(f"[{i}/{len(urls)}] Ingested {url} -> doc_id={res.doc_id}, chunks={res.chunks_inserted}")
            total_chunks += res.chunks_inserted
            total_emb += res.embeddings_inserted
        except Exception as e:
            logger.warning(f"Ingest failed for {url}: {e}")

    logger.info(f"Crawl+Ingest complete: pages={len(urls)} chunks={total_chunks} embeddings={total_emb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
