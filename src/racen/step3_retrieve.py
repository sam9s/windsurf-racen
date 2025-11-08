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


def search_vector(conn: psycopg.Connection, *, query_vec: List[float], top_k: int = 5) -> List[RetrievedChunk]:
    sql = (
        "SELECT c.id AS chunk_id, c.document_id, d.source, c.text, "
        "(1 - (e.embedding <=> %s::vector)) AS score_vector "
        "FROM embeddings e "
        "JOIN chunks c ON c.id = e.chunk_id "
        "JOIN documents d ON d.id = c.document_id "
        "ORDER BY (e.embedding <=> %s::vector) ASC "
        "LIMIT %s"
    )
    with conn.cursor() as cur:
        cur.execute(sql, (query_vec, query_vec, top_k))
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
    q_emb = embedder.embed(id="q", text=query_text)
    conn = get_conn()
    try:
        results = search_hybrid(
            conn,
            query_text=query_text,
            query_vec=q_emb.vector,
            top_k=top_k,
        )
        return results
    finally:
        conn.close()
