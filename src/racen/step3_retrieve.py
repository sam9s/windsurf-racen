from __future__ import annotations

import os
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
    start_line: int
    end_line: int
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


def _source_filter_clause(patterns: List[str]) -> str:
    if not patterns:
        return ""
    ors = " OR ".join(["d.source ILIKE %s"] * len(patterns))
    return f" AND ({ors})"


def _source_filter_params(patterns: List[str]) -> List[str]:
    return [f"%{p}%" for p in patterns]


def search_vector(
    conn: psycopg.Connection,
    *,
    query_vec: List[float],
    top_k: int = 5,
    allow_sources: List[str] | None = None,
) -> List[RetrievedChunk]:
    lit = _vec_literal(query_vec)
    allow = allow_sources or []
    where = _source_filter_clause(allow)
    sql = (
        "SELECT c.id AS chunk_id, c.document_id, d.source, c.text, "
        "c.start_line, c.end_line, "
        f"(1 - (e.embedding <=> '{lit}'::vector)) AS score_vector "
        "FROM embeddings e "
        "JOIN chunks c ON c.id = e.chunk_id "
        "JOIN documents d ON d.id = c.document_id "
        f"WHERE true{where} "
        f"ORDER BY (e.embedding <=> '{lit}'::vector) ASC "
        "LIMIT %s"
    )
    params: List[object] = []
    if allow:
        params.extend(_source_filter_params(allow))
    params.append(top_k)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    out: List[RetrievedChunk] = []
    for r in rows:
        out.append(
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                source=r["source"],
                text=r["text"],
                start_line=int(r.get("start_line", 0) or 0),
                end_line=int(r.get("end_line", 0) or 0),
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


def search_lexical(
    conn: psycopg.Connection, *, query_text: str, top_k: int = 5
) -> List[RetrievedChunk]:
    allow_env = os.getenv("RETRIEVE_SOURCE_ALLOWLIST", "").strip()
    allow = [p for p in (s.strip() for s in allow_env.split(",")) if p]
    where = _source_filter_clause(allow)
    sql = (
        "SELECT c.id AS chunk_id, c.document_id, d.source, c.text, c.start_line, c.end_line, "
        "ts_rank(c.fts, plainto_tsquery('english', %s)) AS rank "
        "FROM chunks c JOIN documents d ON d.id = c.document_id "
        "WHERE c.fts @@ plainto_tsquery('english', %s) "
        f"{where} "
        "ORDER BY rank DESC "
        "LIMIT %s"
    )
    params: List[object] = [query_text, query_text]
    if allow:
        params.extend(_source_filter_params(allow))
    params.append(top_k)
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    out: List[RetrievedChunk] = []
    for r in rows:
        sc = float(r.get("rank", 0.0) or 0.0)
        out.append(
            RetrievedChunk(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                source=r["source"],
                text=r["text"],
                start_line=int(r.get("start_line", 0) or 0),
                end_line=int(r.get("end_line", 0) or 0),
                score=sc,
                score_vector=0.0,
                score_lexical=sc,
            )
        )
    return out


def _source_bias(source: str) -> float:
    s = source.lower()
    if "/pages/warranty" in s:
        return 0.30
    if "/pages/faqs" in s:
        return 0.25
    if "/policies/" in s:
        return 0.20
    return 0.0


def search_hybrid(
    conn: psycopg.Connection,
    *,
    query_text: str,
    query_vec: List[float],
    top_k: int = 5,
    candidate_k: int = 20,
    w_vector: float = 0.6,
    w_lex: float = 0.4,
) -> List[RetrievedChunk]:
    kk = max(candidate_k, top_k)
    allow_env = os.getenv("RETRIEVE_SOURCE_ALLOWLIST", "").strip()
    allow = [p for p in (s.strip() for s in allow_env.split(",")) if p]
    vec_results = search_vector(
        conn, query_vec=query_vec, top_k=kk, allow_sources=allow
    )
    lex_results = search_lexical(
        conn, query_text=query_text, top_k=kk
    )

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
        base = w_vector * vec_norm[i] + w_lex * lex_norm[i]
        base += 0.15 * _source_bias(x.source)
        x.score = base

    items.sort(key=lambda x: x.score, reverse=True)
    return items[:kk]


def retrieve(query_text: str, top_k: int = 5) -> List[RetrievedChunk]:
    """High-level API: embed query, run hybrid search, return results."""
    embedder = OpenAIEmbedder()
    conn = get_conn()
    try:
        fast_mode = os.getenv("FAST_MODE", "0") in {"1", "true", "TRUE", "yes"}
        # Bound reranker workload to keep latency predictable
        try:
            rerank_top_n = int(os.getenv("RERANK_TOP_N", "12"))
        except Exception:
            rerank_top_n = 12
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

        candidates = search_hybrid(
            conn,
            query_text=query_text,
            query_vec=q_vec,
            top_k=top_k,
            # Use a slightly smaller candidate set to reduce latency even with reranker
            candidate_k=max(12, top_k * 3),
        )
        items = candidates

        if not fast_mode:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                model = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    device="cpu",
                )
                # Only rerank top-N to keep latency bounded
                items = items[:rerank_top_n]
                pairs = [(query_text, it.text) for it in items]
                scores = model.predict(pairs)
                for it, sc in zip(items, scores):
                    it.score = float(sc)
                items.sort(key=lambda x: x.score, reverse=True)
            except Exception:
                pass

        return items[:top_k]
    finally:
        conn.close()
