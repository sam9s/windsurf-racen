from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .log import get_logger
from .step1_fetch_convert import MarkItDownClient
from .step2_ingest import Cleaner, Chunker
from .step2_embed import OpenAIEmbedder
from .step2_write import (
    DBConfig,
    ensure_schema,
    get_conn,
    upsert_document,
    upsert_chunk,
    upsert_embedding,
)

logger = get_logger("racen.orchestrator")


@dataclass
class IngestResult:
    doc_id: str
    chunks_inserted: int
    embeddings_inserted: int


def ingest_url(
    url: str,
    *,
    doc_id: Optional[str] = None,
    embedding_dim: int = 256,
) -> IngestResult:
    """End-to-end ingest: URL -> markdown -> clean -> chunk -> embed -> write."""
    client = MarkItDownClient()
    cleaner = Cleaner()
    chunker = Chunker(max_tokens=400, overlap_tokens=40)
    embedder = OpenAIEmbedder()

    # Step 1: Fetch & Convert
    md = client.convert_to_markdown(url=url)

    # Step 2: Clean + Chunk
    cleaned = cleaner.clean(md)
    chunks = chunker.chunk(cleaned)

    # Step 2b: Write
    did = doc_id or f"doc_{hash(url) & 0xFFFFFFFF:x}"
    conn = get_conn(DBConfig.from_env())
    ensure_schema(conn, embedding_dim=embedding_dim)
    upsert_document(conn, doc_id=did, source=url)

    c_ins = 0
    e_ins = 0
    for ch in chunks:
        upsert_chunk(
            conn,
            chunk_id=ch.id,
            document_id=did,
            start_char=ch.start_char,
            end_char=ch.end_char,
            text=ch.text,
        )
        emb = embedder.embed(
            id=ch.id,
            text=ch.text,
            metadata={"doc_id": did, "source": url},
        )
        upsert_embedding(
            conn,
            chunk_id=ch.id,
            vector=emb.vector,
            model=emb.model,
            embedding_dim=embedding_dim,
        )
        c_ins += 1
        e_ins += 1

    logger.info(
        f"Ingest complete: doc_id={did}, chunks={c_ins}, embeddings={e_ins}"
    )
    conn.close()
    return IngestResult(doc_id=did, chunks_inserted=c_ins, embeddings_inserted=e_ins)
