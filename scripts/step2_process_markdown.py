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

        # Select cleaner/chunker mode
        import os
        doc_cleaner = os.getenv("DOC_CLEANER", "basic").lower()
        try:
            max_tokens = int(os.getenv("DOC_CHUNK_MAX_TOKENS", "500"))
        except Exception:
            max_tokens = 500
        try:
            overlap_tokens = int(os.getenv("DOC_CHUNK_OVERLAP", "50"))
        except Exception:
            overlap_tokens = 50

        if doc_cleaner == "docling":
            # Require actual Docling package; fail fast if missing
            try:
                import importlib  # type: ignore

                spec = getattr(importlib, "util").find_spec("docling")  # type: ignore[attr-defined]
            except Exception:
                spec = None
            if spec is None:
                raise RuntimeError(
                    "DOC_CLEANER=docling requested but 'docling' package is not installed. "
                    "Please install it (e.g., pip install docling) and re-run."
                )
            # Defer imports until needed to avoid hard dependency when not in docling mode
            from docling.document_converter import DocumentConverter  # type: ignore
            from docling.chunking import HybridChunker  # type: ignore
            cleaner = None  # sentinel to branch per-file processing
            chunker = (DocumentConverter, HybridChunker)
        else:
            cleaner = Cleaner()
            chunker = Chunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        embedder = OpenAIEmbedder(
            model="text-embedding-3-small"
        )

        for fp in files:
            p = Path(fp)
            logger.info(f"Processing {p.name}")
            # When in docling mode, use DocumentConverter + HybridChunker
            if doc_cleaner == "docling":
                DocumentConverter, HybridChunker = chunker  # type: ignore[assignment]
                dl_doc = DocumentConverter().convert(source=str(p)).document
                dl_chunker = HybridChunker()

                # Wrap results into objects compatible with downstream usage
                class _C:
                    __slots__ = ("text", "start_char", "end_char")

                    def __init__(self, text: str, start_char: int, end_char: int) -> None:
                        self.text = text
                        self.start_char = start_char
                        self.end_char = end_char

                chunks = []
                cursor = 0
                for dl_chunk in dl_chunker.chunk(dl_doc=dl_doc):
                    enriched = dl_chunker.contextualize(chunk=dl_chunk)
                    start_char = cursor
                    end_char = start_char + len(enriched)
                    chunks.append(_C(enriched, start_char, end_char))
                    # maintain small overlap consistent with configured overlap_tokens
                    # Reason: preserve some continuity for retrieval without exploding chunk count
                    overlap_chars = max(0, overlap_tokens * 4)
                    cursor = end_char - min(overlap_chars, len(enriched))
            else:
                text = p.read_text(encoding="utf-8", errors="ignore")
                cleaned = cleaner.clean(text)  # type: ignore[union-attr]
                chunks = chunker.chunk(cleaned)  # type: ignore[union-attr]

            # Precompute line offsets for reliable line refs (only in non-docling mode)
            if doc_cleaner != "docling":
                line_starts = [0]
                for i, ch_ in enumerate(cleaned):
                    if ch_ == "\n":
                        line_starts.append(i + 1)
                def to_line(pos: int) -> int:
                    # Find largest index where line_starts[idx] <= pos, return idx+1
                    lo, hi = 0, len(line_starts) - 1
                    ans = 0
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if line_starts[mid] <= pos:
                            ans = mid
                            lo = mid + 1
                        else:
                            hi = mid - 1
                    return ans + 1
            else:
                def to_line(pos: int) -> int:  # type: ignore[no-redef]
                    return 0

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
                    start_line=to_line(ch.start_char) if doc_cleaner != "docling" else 0,
                    end_line=to_line(ch.end_char) if doc_cleaner != "docling" else 0,
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
            # Post-ingest per-document stats
            avg_len = int(sum(len(c.text) for c in chunks) / max(1, len(chunks)))
            logger.info(
                f"Wrote {len(chunks)} chunks for {p.name} | avg_chars={avg_len} max_tokens={max_tokens} overlap_tokens={overlap_tokens}"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
