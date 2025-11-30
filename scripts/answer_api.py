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

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, HttpUrl, ValidationError

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
from scripts.step4_answer import answer_query, get_last_debug_summary  # noqa: E402
from racen.orchestrator import ingest_url as orchestrator_ingest_url  # noqa: E402
from scripts.internal_tools.sync_product_specs_from_sheet import (  # noqa: E402
    sync_specs_from_sheet,
)
import uuid  # noqa: E402
from datetime import datetime  # noqa: E402


class AnswerRequest(BaseModel):
    """
    HTTP request model for answering a question.

    Args:
        question (str): The user question.
        allowlist (Optional[str]): Comma-separated source allowlist patterns.
        k (Optional[int]): Top-k chunks to retrieve. Defaults to 10 if unset.
        short (Optional[bool]): If true, shorter answers and smaller chunks are used.
        previous_answer (Optional[str]): The last assistant reply in this thread, if any, to help the LLM interpret acknowledgements.
        previous_user (Optional[str]): The last user message to preserve language continuity and resolve acknowledgements.
    """

    question: str = Field(..., min_length=1)
    allowlist: Optional[str] = Field(default=None)
    k: Optional[int] = Field(default=None, ge=1, le=50)
    short: Optional[bool] = Field(default=None)
    previous_answer: Optional[str] = Field(default=None)
    previous_user: Optional[str] = Field(default=None)


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


# Simple in-memory job store for ingestion status (sufficient for local/dev and Slack polling)
JOBS: dict[str, dict] = {}

# Shared-secret env var names for admin-only operations (for example, Slack commands).
# Prefer a general RACEN_ADMIN_TOKEN, but honour the legacy IPHONE_SPECS_SYNC_TOKEN
# for backwards compatibility with earlier deployments.
ADMIN_TOKEN_ENV = "RACEN_ADMIN_TOKEN"
LEGACY_SPECS_ADMIN_TOKEN_ENV = "IPHONE_SPECS_SYNC_TOKEN"


def _get_expected_admin_token() -> str:
    """Return the expected admin token from the environment, if configured.

    Checks the general :data:`ADMIN_TOKEN_ENV` first and falls back to the
    legacy :data:`LEGACY_SPECS_ADMIN_TOKEN_ENV` to avoid breaking older
    setups that only configured the iPhone-specific token.
    """

    token = os.getenv(ADMIN_TOKEN_ENV, "")
    if token:
        return token
    return os.getenv(LEGACY_SPECS_ADMIN_TOKEN_ENV, "")


def _enforce_admin_token(token: Optional[str]) -> None:
    """Enforce the shared admin token when configured.

    When no admin token is set in the environment this becomes a no-op so
    local development remains frictionless. When a token *is* configured,
    callers must supply the same value via the request payload.
    """

    expected = _get_expected_admin_token()
    if not expected:
        return
    if not token or token != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


class IngestRequest(BaseModel):
    url: HttpUrl
    requested_by: str = Field(..., min_length=1)
    token: Optional[str] = Field(default=None)


class IngestResponse(BaseModel):
    job_id: str


class IngestStatus(BaseModel):
    job_id: str
    status: str
    stage: str | None = None
    detail: str | None = None
    chunks_inserted: int | None = None
    embeddings_inserted: int | None = None
    updated_at: float


class SpecsSyncRequest(BaseModel):
    """Request payload for triggering an iPhone specs sheet→DB sync.

    Args:
        token: Optional shared-secret token. When an admin token environment
            variable is configured (``RACEN_ADMIN_TOKEN`` or legacy
            ``IPHONE_SPECS_SYNC_TOKEN``), this value must match it.
        csv_path: Optional override CSV path, only used when the sheet URL
            environment variable is not configured.
    """

    token: Optional[str] = Field(default=None)
    csv_path: Optional[str] = Field(default=None)


class SpecsSyncResponse(BaseModel):
    """Response payload for the iPhone specs sheet→DB sync endpoint.

    Mirrors the summary structure returned by ``sync_specs_from_sheet``.

    Args:
        status: Overall status ("ok", "skipped", or "error").
        reason: Machine-readable reason when status is not "ok".
        rows_written: Number of rows written to the specs table.
        duplicate_slugs: List of duplicate slugs detected, if any.
        slugs_all_missing: Slugs with all condition prices missing.
        slugs_some_missing: Slugs with some condition prices missing.
        source: Description of the data source (URL or path).
    """

    status: str
    reason: str
    rows_written: int
    duplicate_slugs: List[str]
    slugs_all_missing: List[str]
    slugs_some_missing: List[str]
    source: str


def _validate_domain(url: str) -> None:
    try:
        from urllib.parse import urlparse
        o = urlparse(url)
        host = (o.hostname or "").lower()
        if not host.endswith("grest.in"):
            raise ValueError("Only grest.in URLs are allowed")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _set_job(job_id: str, **kwargs) -> None:
    now = datetime.utcnow().timestamp()
    state = JOBS.get(job_id, {})
    state.update(kwargs)
    state["updated_at"] = now
    JOBS[job_id] = state


def _run_ingest_job(job_id: str, url: str) -> None:
    # Mark started
    _set_job(job_id, status="started", stage="started", detail="Ingestion started")
    try:
        # Coarse stage: embeddings (covers crawl->parse->embed in orchestrator)
        _set_job(job_id, stage="embedding", detail="Requesting embeddings")
        res = orchestrator_ingest_url(url, embedding_dim=1536)
        _set_job(
            job_id,
            status="done",
            stage="done",
            detail=f"Ingested {url}",
            chunks_inserted=getattr(res, "chunks_inserted", None),
            embeddings_inserted=getattr(res, "embeddings_inserted", None),
        )
    except Exception as e:
        _set_job(job_id, status="error", stage="error", detail=f"{type(e).__name__}: {e}")


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

    text, cits = answer_query(
        req.question,
        top_k=k,
        previous_answer=(req.previous_answer or ""),
        previous_user=(req.previous_user or ""),
    )

    eff = effective_settings()
    ribbon = (
        f"k={k} | FAST_MODE={eff.get('FAST_MODE')} | RERANK_TOP_N={eff.get('RERANK_TOP_N')} "
        f"| allowlist={eff.get('RETRIEVE_SOURCE_ALLOWLIST')} | model={os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}"
        f" | short={os.getenv('ANSWER_SHORT', '')}"
    )
    debug_on = os.getenv("ANSWER_DEBUG_FLAGS", "0") in {"1", "true", "TRUE", "yes"}
    if debug_on:
        try:
            dbg = get_last_debug_summary()
            if dbg:
                ribbon = ribbon + f" | {dbg}"
        except Exception:
            pass
        # In debug mode, surface a compact view of the top citation URLs so
        # downstream clients (e.g., Slack) can display evidence without
        # changing their formatting logic.
        try:
            if cits:
                top_cits = ", ".join(c.url for c in cits[:3])
                ribbon = ribbon + f" | cits={top_cits}"
        except Exception:
            pass

    payload = AnswerResponse(
        answer=text,
        citations=[CitationOut(url=c.url, start_line=c.start_line, end_line=c.end_line) for c in cits],
        settings_summary=ribbon,
    )
    return payload


@app.post("/ingest/url", response_model=IngestResponse)
def ingest_url_api(req: IngestRequest, background: BackgroundTasks) -> IngestResponse:
    """Enqueue a background ingestion job for a single URL.

    Only grest.in domain is allowed. Returns a job_id that can be polled via
    ``/ingest/status/{job_id}``.
    """

    # Enforce shared admin token when configured so only trusted internal
    # callers (for example, the Slack bot) can trigger ingestion even if the
    # HTTP endpoint is reachable from elsewhere.
    _enforce_admin_token(req.token)
    # Enforce optional ingest allowlist based on Slack user IDs
    allow_raw = os.getenv("INGEST_ALLOWED_USERS", "")
    allowlisted: set[str] = set(u.strip() for u in allow_raw.split(",") if u.strip())
    if allowlisted and req.requested_by not in allowlisted:
        raise HTTPException(status_code=403, detail="User not allowed to ingest.")

    _validate_domain(str(req.url))
    job_id = str(uuid.uuid4())
    _set_job(job_id, status="accepted", stage="queued", detail=f"Accepted {req.url}")
    background.add_task(_run_ingest_job, job_id, str(req.url))
    return IngestResponse(job_id=job_id)


@app.get("/ingest/status/{job_id}", response_model=IngestStatus)
def ingest_status_api(job_id: str) -> IngestStatus:
    st = JOBS.get(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="job_id not found")
    return IngestStatus(
        job_id=job_id,
        status=st.get("status", "unknown"),
        stage=st.get("stage"),
        detail=st.get("detail"),
        chunks_inserted=st.get("chunks_inserted"),
        embeddings_inserted=st.get("embeddings_inserted"),
        updated_at=st.get("updated_at", 0.0),
    )


@app.post("/admin/sync/iphone-specs", response_model=SpecsSyncResponse)
def sync_iphone_specs(req: SpecsSyncRequest) -> SpecsSyncResponse:
    """Admin endpoint to sync iPhone specs from the Google Sheet into Postgres.

    This is intended to be called from a Slack admin command in the sibling
    Slack bot project. Authentication is handled via a simple shared-secret
    token configured in the :data:`SPECS_SYNC_ADMIN_TOKEN_ENV` environment
    variable.

    Returns:
        SpecsSyncResponse: Structured summary of the sync operation.
    """

    # Reuse the same admin token enforcement logic as other internal
    # operations so we have a single scalable security pattern.
    _enforce_admin_token(req.token)

    try:
        summary = sync_specs_from_sheet(csv_path=req.csv_path)
    except Exception as exc:
        # Surface a compact error back to the caller; full traceback stays in logs.
        raise HTTPException(
            status_code=500,
            detail=f"Sync failed: {type(exc).__name__}: {exc}",
        ) from exc

    return SpecsSyncResponse(
        status=str(summary.get("status", "")),
        reason=str(summary.get("reason", "")),
        rows_written=int(summary.get("rows_written", 0)),
        duplicate_slugs=list(summary.get("duplicate_slugs", [])),
        slugs_all_missing=list(summary.get("slugs_all_missing", [])),
        slugs_some_missing=list(summary.get("slugs_some_missing", [])),
        source=str(summary.get("source", "")),
    )


if __name__ == "__main__":
    # Local run helper: uvicorn scripts.answer_api:app --reload --port 8000
    import uvicorn  # type: ignore

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("scripts.answer_api:app", host="0.0.0.0", port=port, reload=False)
