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


def test_sync_iphone_specs_forbidden_when_token_mismatch(monkeypatch) -> None:
    """Admin sync endpoint should reject requests with a wrong shared-secret token."""
    monkeypatch.setenv(ADMIN_TOKEN_ENV, "correct-secret")
    client = TestClient(answer_api_module.app)

    resp = client.post(
        "/admin/sync/iphone-specs",
        json={"token": "wrong-secret"},
    )

    assert resp.status_code == 403


def test_sync_iphone_specs_success_uses_helper(monkeypatch) -> None:
    """Admin sync endpoint should call sync_specs_from_sheet and return its summary."""
    monkeypatch.setenv(ADMIN_TOKEN_ENV, "admin-token")

    called: dict[str, Any] = {}

    def fake_sync_specs_from_sheet(csv_path: str | None = None) -> dict[str, Any]:
        """Fake implementation capturing arguments and returning a fixed summary."""

        called["csv_path"] = csv_path
        return {
            "status": "ok",
            "reason": "",
            "rows_written": 5,
            "duplicate_slugs": ["dup-slug"],
            "slugs_all_missing": ["missing-all"],
            "slugs_some_missing": ["missing-some"],
            "source": "dummy-source",
        }

    monkeypatch.setattr(
        answer_api_module,
        "sync_specs_from_sheet",
        fake_sync_specs_from_sheet,
        raising=True,
    )

    client = TestClient(answer_api_module.app)
    resp = client.post(
        "/admin/sync/iphone-specs",
        json={"token": "admin-token", "csv_path": "custom.csv"},
    )

    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert data["rows_written"] == 5
    assert data["duplicate_slugs"] == ["dup-slug"]
    assert data["slugs_all_missing"] == ["missing-all"]
    assert data["slugs_some_missing"] == ["missing-some"]
    assert data["source"] == "dummy-source"

    # Ensure the helper was called with the CSV path provided by the caller.
    assert called["csv_path"] == "custom.csv"
