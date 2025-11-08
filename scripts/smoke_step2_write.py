from __future__ import annotations

import sys
from pathlib import Path

# Ensure local 'src' is importable when running as a script
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_chunks() -> list[tuple[str, str]]:
    # Returns list of (chunk_id, text)
    step2_dir = Path("outputs/step2")
    if not step2_dir.exists():
        raise FileNotFoundError("Run scripts/smoke_step2.py first")
    chunks = []
    for p in sorted(step2_dir.glob("c*.md")):
        chunks.append((p.stem, p.read_text(encoding="utf-8")))
    if not chunks:
        raise FileNotFoundError("No chunks found; run scripts/smoke_step2.py")
    return chunks


def main():
    # Lazy imports to keep sys.path logic first
    from racen.log import get_logger
    from racen.step2_embed import OpenAIEmbedder
    from racen.step2_write import (
        DBConfig,
        ensure_schema,
        get_conn,
        upsert_document,
        upsert_chunk,
        upsert_embedding,
    )

    logger = get_logger("racen.smoke2.write")
    # Load .env if python-dotenv is installed
    try:
        import dotenv  # type: ignore
        dotenv.load_dotenv()
    except Exception:
        pass

    doc_id = "doc_example_com"
    source = str(Path("outputs/step1/example_com.md").resolve())

    embedder = OpenAIEmbedder()
    chunks = load_chunks()

    conn = get_conn(DBConfig.from_env())
    ensure_schema(conn, embedding_dim=256)
    upsert_document(conn, doc_id=doc_id, source=source)

    for cid, text in chunks:
        upsert_chunk(
            conn,
            chunk_id=cid,
            document_id=doc_id,
            start_char=0,
            end_char=len(text),
            text=text,
        )
        emb = embedder.embed(
            id=cid,
            text=text,
            metadata={"doc_id": doc_id},
        )
        upsert_embedding(
            conn,
            chunk_id=cid,
            vector=emb.vector,
            model=emb.model,
            embedding_dim=256,
        )

    # Verify counts
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS c FROM chunks WHERE document_id=%s", (doc_id,))
        c_chunks = cur.fetchone()["c"]
        cur.execute("SELECT COUNT(*) AS c FROM embeddings")
        c_emb = cur.fetchone()["c"]
        logger.info(f"Inserted chunks={c_chunks}, embeddings={c_emb}")


if __name__ == "__main__":
    from racen.log import get_logger

    _logger = get_logger("racen.smoke2.write")
    _logger.info("Smoke Step 2 (write): start")
    main()
    _logger.info("Smoke Step 2 (write): done")
