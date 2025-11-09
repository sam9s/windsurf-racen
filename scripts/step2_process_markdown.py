from __future__ import annotations

import glob
import hashlib
from pathlib import Path
from typing import Tuple

# Add project src to sys.path for absolute imports per repo conventions
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Load environment variables from .env (prefer local env file)
try:
    from dotenv import load_dotenv

    # Search typical locations
    env_candidates = [
        BASE_DIR / "windsurf-racen-local" / ".env",
        BASE_DIR / ".env",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            break
except Exception:
    # Proceed even if dotenv isn't available
    pass

from racen.step2_ingest import Cleaner, Chunker
from racen.step2_embed import OpenAIEmbedder
from racen.step2_write import (
    ensure_schema,
    get_conn,
    upsert_document,
    upsert_chunk,
    upsert_embedding,
)
from racen.log import get_logger

logger = get_logger("scripts.step2_process_markdown")


def doc_id_for(path: Path) -> Tuple[str, str]:
    """
    Compute a stable document_id and source from a markdown filename.

    Returns:
        tuple[str, str]: (document_id, source)
    """
    name = path.stem
    # If original URL is embedded in filename pattern "host-path.md", reconstruct a plausible source URL.
    parts = name.split("-")
    if parts:
        host = parts[0]
        source = (
            f"https://{host}/" + "/".join(parts[1:])
        )
    else:
        source = name
    # Stable id via sha1 of source
    did = hashlib.sha1(source.encode("utf-8")).hexdigest()
    return did, source


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    md_dir = (
        base
        / "windsurf-racen-local"
        / "outputs"
        / "markdown_crawl"
    )
    files = sorted(glob.glob(str(md_dir / "*.md")))
    if not files:
        logger.error(f"No markdown files found in {md_dir}")
        return

    # Ensure DB schema with OpenAI 1536d embeddings
    # Support DATABASE_URL by mapping to PG* env vars if present
    import os
    from urllib.parse import urlparse

    db_url = os.getenv("DATABASE_URL")
    if db_url and not os.getenv("PGHOST"):
        u = urlparse(db_url)
        if u.scheme.startswith("postgres"):
            if u.hostname:
                os.environ["PGHOST"] = u.hostname
            if u.port:
                os.environ["PGPORT"] = str(u.port)
            if u.path and len(u.path) > 1:
                os.environ["PGDATABASE"] = u.path.lstrip("/")
            if u.username:
                os.environ["PGUSER"] = u.username
            if u.password:
                os.environ["PGPASSWORD"] = u.password

    conn = get_conn()
    try:
        ensure_schema(conn, embedding_dim=1536)

        cleaner = Cleaner()
        chunker = Chunker(max_tokens=500, overlap_tokens=50)
        embedder = OpenAIEmbedder(
            model="text-embedding-3-small"
        )

        for fp in files:
            p = Path(fp)
            logger.info(f"Processing {p.name}")
            text = p.read_text(encoding="utf-8", errors="ignore")

            cleaned = cleaner.clean(text)
            chunks = chunker.chunk(cleaned)

            document_id, source = doc_id_for(p)
            upsert_document(conn, doc_id=document_id, source=source)

            for ch in chunks:
                # Stable chunk_id using document_id + start/end
                chunk_key = f"{document_id}:{ch.start_char}:{ch.end_char}"
                chunk_id = hashlib.sha1(chunk_key.encode("utf-8")).hexdigest()
                upsert_chunk(
                    conn,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    start_char=ch.start_char,
                    end_char=ch.end_char,
                    start_line=int(ch.meta.get("start_line", 0)),
                    end_line=int(ch.meta.get("end_line", 0)),
                    text=ch.text,
                )
                emb = embedder.embed(
                    id=chunk_id,
                    text=ch.text,
                    metadata={"source": source},
                )
                upsert_embedding(
                    conn,
                    chunk_id=chunk_id,
                    vector=emb.vector,
                    model=emb.model,
                    embedding_dim=1536,
                )
            logger.info(f"Wrote {len(chunks)} chunks for {p.name}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
