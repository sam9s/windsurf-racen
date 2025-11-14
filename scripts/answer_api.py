"""
FastAPI service exposing the pipeline's answer_query via HTTP.

Provides a stable contract for the Slack bot and other clients to call
and reuse the exact same retrieval+LLM logic validated in evaluations.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

# Ensure local 'src' is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env
try:
    import dotenv  # type: ignore

    for env_path in [ROOT / "windsurf-racen-local" / ".env", ROOT / ".env"]:
        if env_path.exists():
            dotenv.load_dotenv(dotenv_path=env_path, override=False)
            break
except Exception:
    pass

from racen.step3_retrieve import effective_settings  # noqa: E402
from scripts.step4_answer import answer_query  # noqa: E402


class AnswerRequest(BaseModel):
    """
    HTTP request model for answering a question.

    Args:
        question (str): The user question.
        allowlist (Optional[str]): Comma-separated source allowlist patterns.
        k (Optional[int]): Top-k chunks to retrieve. Defaults to 10 if unset.
        short (Optional[bool]): If true, shorter answers and smaller chunks are used.
        previous_answer (Optional[str]): The last assistant reply in this thread, if any, to help the LLM interpret acknowledgements.
    """

    question: str = Field(..., min_length=1)
    allowlist: Optional[str] = Field(default=None)
    k: Optional[int] = Field(default=None, ge=1, le=50)
    short: Optional[bool] = Field(default=None)
    previous_answer: Optional[str] = Field(default=None)


class CitationOut(BaseModel):
    """
    Outgoing citation structure.

    Args:
        url (str): Source URL.
        start_line (int): Start line in the source.
        end_line (int): End line in the source.
    """

    url: str
    start_line: int
    end_line: int


class AnswerResponse(BaseModel):
    """
    Answer response payload.

    Args:
        answer (str): Final answer text.
        citations (List[CitationOut]): Evidence citations.
        settings_summary (str): Compact settings ribbon for traceability.
    """

    answer: str
    citations: List[CitationOut]
    settings_summary: str


app = FastAPI(title="RACEN Answer API", version="1.0.0")


@app.get("/health")
def health() -> dict:
    """
    Lightweight healthcheck endpoint.

    Returns:
        dict: Simple status flag used by orchestration and the Slack bot
        to verify the service is up.
    """
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    """
    Answer a question using the current retrieval + LLM settings.

    - Applies an in-process allowlist if provided.
    - Sets k and short modes via environment overrides when supplied.

    Returns:
        AnswerResponse: answer text, citations, and a compact settings ribbon.
    """
    # Apply runtime overrides
    if req.allowlist:
        os.environ["RETRIEVE_SOURCE_ALLOWLIST"] = req.allowlist
    if req.k is not None:
        os.environ["TOP_K"] = str(req.k)
    if req.short is not None:
        os.environ["ANSWER_SHORT"] = "1" if req.short else "0"

    k = req.k if req.k is not None else int(os.getenv("TOP_K", "10"))

    text, cits = answer_query(req.question, top_k=k, previous_answer=(req.previous_answer or ""))

    eff = effective_settings()
    ribbon = (
        f"k={k} | FAST_MODE={eff.get('FAST_MODE')} | RERANK_TOP_N={eff.get('RERANK_TOP_N')} "
        f"| allowlist={eff.get('RETRIEVE_SOURCE_ALLOWLIST')} | model={os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}"
        f" | short={os.getenv('ANSWER_SHORT', '')}"
    )

    payload = AnswerResponse(
        answer=text,
        citations=[CitationOut(url=c.url, start_line=c.start_line, end_line=c.end_line) for c in cits],
        settings_summary=ribbon,
    )
    return payload


if __name__ == "__main__":
    # Local run helper: uvicorn scripts.answer_api:app --reload --port 8000
    import uvicorn  # type: ignore

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("scripts.answer_api:app", host="0.0.0.0", port=port, reload=False)
