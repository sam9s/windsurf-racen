from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

# Ensure src/ is importable so that racen and scripts modules can be imported
# when tests are run via `python -m pytest` from the project root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts import answer_api as answer_api_module  # type: ignore  # noqa: E402
from scripts.answer_api import ADMIN_TOKEN_ENV  # type: ignore  # noqa: E402


def test_ingest_url_rejects_when_admin_token_mismatch(monkeypatch) -> None:
    """Ingest endpoint should reject callers with a wrong shared admin token.

    This ensures that once an admin token is configured, external HTTP callers
    cannot trigger ingestion jobs without knowing it, even if they can guess a
    Slack user ID.
    """

    monkeypatch.setenv(ADMIN_TOKEN_ENV, "correct-token")
    # Do not restrict by INGEST_ALLOWED_USERS so we only test token behaviour.
    monkeypatch.delenv("INGEST_ALLOWED_USERS", raising=False)

    client = TestClient(answer_api_module.app)

    resp = client.post(
        "/ingest/url",
        json={
            "url": "https://grest.in/products/test-product",
            "requested_by": "U_TEST",
            "token": "wrong-token",
        },
    )

    assert resp.status_code == 403


def test_ingest_url_accepts_with_correct_admin_token(monkeypatch) -> None:
    """Ingest endpoint should accept requests with the correct admin token.

    The actual ingestion job runner is stubbed so tests do not hit external
    dependencies.
    """

    monkeypatch.setenv(ADMIN_TOKEN_ENV, "correct-token")
    monkeypatch.delenv("INGEST_ALLOWED_USERS", raising=False)

    captured: dict[str, Any] = {}

    def fake_run_ingest_job(job_id: str, url: str) -> None:
        captured["job_id"] = job_id
        captured["url"] = url

    monkeypatch.setattr(
        answer_api_module,
        "_run_ingest_job",
        fake_run_ingest_job,
        raising=True,
    )

    client = TestClient(answer_api_module.app)

    resp = client.post(
        "/ingest/url",
        json={
            "url": "https://grest.in/products/test-product",
            "requested_by": "U_TEST",
            "token": "correct-token",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    # Ensure the background task wiring used our fake runner.
    assert captured["url"] == "https://grest.in/products/test-product"
