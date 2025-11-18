from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

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

import yaml  # type: ignore

from racen.log import get_logger
from racen.orchestrator import ingest_url


logger = get_logger("racen.bulk_ingest")


def _load_urls_from_yaml(path: Path, key: str) -> List[str]:
    """Load a list of URLs from a YAML file.

    Args:
        path (Path): Path to the YAML file.
        key (str): Top-level key under which URLs are stored.

    Returns:
        List[str]: List of URLs.
    """
    if not path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    urls = data.get(key)
    if not isinstance(urls, list):
        raise ValueError(f"Expected a list under key '{key}' in {path}")

    # Normalize to strings and strip whitespace
    out: List[str] = []
    for u in urls:
        if not u:
            continue
        s = str(u).strip()
        if s:
            out.append(s)
    return out


def main() -> int:
    """Bulk-ingest URLs from a YAML config using the RACEN orchestrator.

    This script is intended for one-off bulk ingestion runs (e.g., all iPhone
    product pages). It prints per-URL progress so you can monitor the run in
    the terminal and see any errors.
    """
    default_cfg = ROOT / "Grest_Data" / "grest_iphone_products.yaml"

    p = argparse.ArgumentParser(description="Bulk-ingest URLs from a YAML file")
    p.add_argument(
        "--config",
        type=str,
        default=str(default_cfg),
        help="Path to YAML config (default: Grest_Data/grest_iphone_products.yaml)",
    )
    p.add_argument(
        "--key",
        type=str,
        default="iphones",
        help="Top-level YAML key containing the URL list (default: iphones)",
    )
    p.add_argument(
        "--dim",
        type=int,
        default=1536,
        help="Embedding dimension for pgvector column (default: 1536)",
    )
    args = p.parse_args()

    cfg_path = Path(args.config).resolve()

    try:
        urls = _load_urls_from_yaml(cfg_path, args.key)
    except Exception as exc:  # pragma: no cover - CLI guard
        logger.error(f"Failed to load URLs from {cfg_path}: {exc}")
        return 1

    if not urls:
        logger.info(f"No URLs found under key '{args.key}' in {cfg_path}")
        return 0

    total = len(urls)
    logger.info(f"Starting bulk ingest for {total} URLs from {cfg_path}")

    total_chunks = 0
    total_emb = 0
    failures = 0

    for idx, url in enumerate(urls, start=1):
        # Print progress to both logger and stdout for easy CLI monitoring.
        msg_prefix = f"[{idx}/{total}] Ingesting {url}"
        print(msg_prefix, flush=True)
        logger.info(msg_prefix)

        try:
            res = ingest_url(url, embedding_dim=args.dim)
        except Exception as exc:  # pragma: no cover - CLI guard
            failures += 1
            err_msg = f"[{idx}/{total}] Ingest failed for {url}: {exc}"
            print(err_msg, flush=True)
            logger.warning(err_msg)
            continue

        total_chunks += res.chunks_inserted
        total_emb += res.embeddings_inserted

        ok_msg = (
            f"[{idx}/{total}] Done: doc_id={res.doc_id}, "
            f"chunks={res.chunks_inserted}, embeddings={res.embeddings_inserted}"
        )
        print(ok_msg, flush=True)
        logger.info(ok_msg)

    summary = (
        f"Bulk ingest complete: urls={total}, "
        f"failures={failures}, chunks={total_chunks}, embeddings={total_emb}"
    )
    print(summary, flush=True)
    logger.info(summary)

    return 0 if failures == 0 else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
