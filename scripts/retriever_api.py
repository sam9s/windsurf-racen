from __future__ import annotations

import os
import time
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.racen.step3_retrieve import retrieve, RetrievedChunk

app = FastAPI(title="RACEN Retriever API", version="0.1.0")


class SearchRequest(BaseModel):
    """
    Request payload for /search.

    Args:
        query (str): The user query text to retrieve against the DB corpus.
        top_k (int): Number of chunks to return.
    """

    query: str = Field(..., min_length=1)
    top_k: int = Field(6, ge=1, le=20)


class Citation(BaseModel):
    """
    A single retrieved chunk citation.

    Args:
        text (str): The chunk text.
        source (str): The source URL of the chunk.
        start_line (int): Start line of the chunk in the source.
        end_line (int): End line of the chunk in the source.
        score (float): Retrieval score (post-fusion or rerank score).
    """

    text: str
    source: str
    start_line: int
    end_line: int
    score: float


class SearchResponse(BaseModel):
    """
    Response for /search.

    Args:
        items (List[Citation]): Top-k citations.
        latency_ms (int): Retrieval latency in milliseconds.
    """

    items: List[Citation]
    latency_ms: int


@app.get("/health")
def health() -> dict:
    """Simple health check endpoint.

    Returns:
        dict: Status payload.
    """
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    """
    Retrieve top-k citations for the given query using RACEN's retrieval pipeline.

    This wraps src.racen.step3_retrieve.retrieve(), respecting env flags like
    FAST_MODE, RERANK_TOP_N, RETRIEVE_SOURCE_ALLOWLIST, and backoff controls.

    Args:
        req (SearchRequest): Query and top_k parameters.

    Returns:
        SearchResponse: List of citations and latency.
    """
    t0 = time.time()
    items: List[RetrievedChunk] = retrieve(req.query, top_k=req.top_k)
    dt_ms = int((time.time() - t0) * 1000)
    out = [
        Citation(
            text=it.text,
            source=it.source,
            start_line=int(getattr(it, "start_line", 0) or 0),
            end_line=int(getattr(it, "end_line", 0) or 0),
            score=float(getattr(it, "score", 0.0) or 0.0),
        )
        for it in items
    ]
    return SearchResponse(items=out, latency_ms=dt_ms)


if __name__ == "__main__":
    # Reason: Allow running directly during local development and in Docker.
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("scripts.retriever_api:app", host="0.0.0.0", port=port, reload=False)
