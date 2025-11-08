from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

import psycopg

from .log import get_logger
from .step2_embed import OpenAIEmbedder
from .step2_write import get_conn

logger = get_logger("racen.retrieve")


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    source: str
    text: str
    score: float
    score_vector: float
    score_lexical: float


def _normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mx = max(scores)
    mn = min(scores)
    if mx == mn:
        return [0.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def _vec_literal(vec: List[float]) -> str:
    # pgvector text literal: [v1, v2, ...]
    return "[" + ", ".join(f"{v:.6f}" for v in vec) + "]"


def search_vector(
    conn: psycopg.Connection,
    *,
    query_vec: List[float],
    top_k: int = 5,
) -> List[RetrievedChunk]:
    lit = _vec_literal(query_vec)
    sql = (
        "SELECT c.id AS chunk_id, c.document_id, d.source, c.text, "
        f"(1 - (e.embedding <=> '{lit}'::vector)) AS score_vector "
        "FROM embeddings e "
        "JOIN chunks c ON c.id = e.chunk_id "
        "JOIN documents d ON d.id = c.document_id "
        f"ORDER BY (e.embedding <=> '{lit}'::vector) ASC "
        "LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (top_k,))
        rows = cur.fetchall()
    out: List[RetrievedChunk] = []
    for r in rows:
        out.append(
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                source=r["source"],
                text=r["text"],
                score=r["score_vector"],
                score_vector=r["score_vector"],
                score_lexical=0.0,
            )
        )
    return out


def _lexical_score(text: str, query: str) -> float:
    # simple token overlap score
    q_tokens = {t for t in re.findall(r"\w+", query.lower()) if t}
    t_tokens = {t for t in re.findall(r"\w+", text.lower()) if t}
    if not q_tokens or not t_tokens:
        return 0.0
    inter = len(q_tokens & t_tokens)
    return inter / len(q_tokens)


def search_lexical(conn: psycopg.Connection, *, query_text: str, top_k: int = 5) -> List[RetrievedChunk]:
    like = f"%{query_text}%"
    sql = (
        "SELECT c.id AS chunk_id, c.document_id, d.source, c.text "
        "FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE c.text ILIKE %s "
        "LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (like, top_k * 5))  # wider candidate set
        rows = cur.fetchall()
    scored: List[Tuple[float, RetrievedChunk]] = []
    for r in rows:
        sc = _lexical_score(r["text"], query_text)
        if sc <= 0:
            continue
        scored.append(
            (
                sc,
                RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    document_id=r["document_id"],
                    source=r["source"],
                    text=r["text"],
                    score=sc,
                    score_vector=0.0,
                    score_lexical=sc,
                ),
            )
        )
    scored.sort(key=lambda x: x[0], reverse=True)
    return [rc for _, rc in scored[:top_k]]


def search_hybrid(
    conn: psycopg.Connection,
    *,
    query_text: str,
    query_vec: List[float],
    top_k: int = 5,
    w_vector: float = 0.6,
    w_lex: float = 0.4,
) -> List[RetrievedChunk]:
    vec_results = search_vector(conn, query_vec=query_vec, top_k=top_k)
    lex_results = search_lexical(conn, query_text=query_text, top_k=top_k)

    # index by chunk id
    by_id: dict[str, RetrievedChunk] = {}
    for r in vec_results:
        by_id[r.chunk_id] = r
    for r in lex_results:
        if r.chunk_id in by_id:
            # merge
            a = by_id[r.chunk_id]
            a.score_lexical = r.score_lexical
        else:
            by_id[r.chunk_id] = r

    items = list(by_id.values())
    vec_scores = [x.score_vector for x in items]
    lex_scores = [x.score_lexical for x in items]
    vec_norm = _normalize(vec_scores)
    lex_norm = _normalize(lex_scores)

    for i, x in enumerate(items):
        x.score = w_vector * vec_norm[i] + w_lex * lex_norm[i]

    items.sort(key=lambda x: x.score, reverse=True)
    return items[:top_k]


def retrieve(query_text: str, top_k: int = 5) -> List[RetrievedChunk]:
    """High-level API: embed query, run hybrid search, return results."""
    embedder = OpenAIEmbedder()
    conn = get_conn()
    try:
        # Detect embedding dimension from DB (fallback to 256 if empty)
        expected_dim = 256
        with conn.cursor() as cur:
            try:
                cur.execute("SELECT vector_dims(embedding) AS dim FROM embeddings LIMIT 1")
                row = cur.fetchone()
                if row and row.get("dim"):
                    expected_dim = int(row["dim"])  # type: ignore[arg-type]
            except Exception:
                expected_dim = 256

        # Embed query and fit to expected dimension
        q_emb = embedder.embed(id="q", text=query_text)
        q_vec = list(q_emb.vector)
        if len(q_vec) > expected_dim:
            q_vec = q_vec[:expected_dim]
        elif len(q_vec) < expected_dim:
            q_vec = q_vec + [0.0] * (expected_dim - len(q_vec))

        results = search_hybrid(
            conn,
            query_text=query_text,
            query_vec=q_vec,
            top_k=top_k,
        )
        return results
    finally:
        conn.close()
