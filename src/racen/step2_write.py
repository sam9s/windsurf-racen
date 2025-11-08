from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import psycopg
from psycopg.rows import dict_row

from .log import get_logger

logger = get_logger("racen.write")


@dataclass
class DBConfig:
    host: str
    port: int
    dbname: str
    user: str
    password: str

    @classmethod
    def from_env(cls) -> "DBConfig":
        return cls(
            host=os.getenv("PGHOST", "localhost"),
            port=int(os.getenv("PGPORT", "5432")),
            dbname=os.getenv("PGDATABASE", "racen"),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", "postgres"),
        )


def embedding_exists(conn: psycopg.Connection, *, chunk_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM embeddings WHERE chunk_id = %s LIMIT 1",
            (chunk_id,),
        )
        return cur.fetchone() is not None


def get_conn(cfg: Optional[DBConfig] = None) -> psycopg.Connection:
    cfg = cfg or DBConfig.from_env()
    logger.info(
        f"Connecting to Postgres {cfg.host}:{cfg.port}/{cfg.dbname} as {cfg.user}"
    )
    conn = psycopg.connect(
        host=cfg.host,
        port=cfg.port,
        dbname=cfg.dbname,
        user=cfg.user,
        password=cfg.password,
        autocommit=True,
        row_factory=dict_row,
    )
    return conn


def ensure_schema(conn: psycopg.Connection, embedding_dim: int = 256) -> None:
    with conn.cursor() as cur:
        logger.info("Ensuring pgvector extension and tables")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
              id TEXT PRIMARY KEY,
              source TEXT,
              created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              id TEXT PRIMARY KEY,
              document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
              start_char INT,
              end_char INT,
              text TEXT
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS embeddings (
              chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
              embedding VECTOR({embedding_dim}),
              model TEXT
            )
            """
        )
        cur.execute(
            (
                "CREATE INDEX IF NOT EXISTS idx_embeddings_vec "
                "ON embeddings USING ivfflat (embedding vector_cosine_ops) "
                "WITH (lists = 100)"
            )
        )


def upsert_document(conn: psycopg.Connection, *, doc_id: str, source: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (id, source)
            VALUES (%s, %s)
            ON CONFLICT (id) DO UPDATE SET source = EXCLUDED.source
            """,
            (doc_id, source),
        )


def upsert_chunk(
    conn: psycopg.Connection,
    *,
    chunk_id: str,
    document_id: str,
    start_char: int,
    end_char: int,
    text: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO chunks (id, document_id, start_char, end_char, text)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id)
            DO UPDATE SET document_id = EXCLUDED.document_id,
                          start_char = EXCLUDED.start_char,
                          end_char = EXCLUDED.end_char,
                          text = EXCLUDED.text
            """,
            (chunk_id, document_id, start_char, end_char, text),
        )


def upsert_embedding(
    conn: psycopg.Connection,
    *,
    chunk_id: str,
    vector: List[float],
    model: str,
    embedding_dim: int = 256,
) -> None:
    # pad or trim to embedding_dim
    v = list(vector[:embedding_dim])
    if len(v) < embedding_dim:
        v.extend([0.0] * (embedding_dim - len(v)))
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO embeddings (chunk_id, embedding, model)
            VALUES (%s, %s, %s)
            ON CONFLICT (chunk_id)
            DO UPDATE SET embedding = EXCLUDED.embedding,
                          model = EXCLUDED.model
            """,
            (chunk_id, v, model),
        )
