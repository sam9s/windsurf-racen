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


def test_save_iphone_specs_forbidden_without_correct_token(monkeypatch) -> None:
    """Saving specs should be forbidden when the admin token does not match."""

    monkeypatch.setenv(ADMIN_TOKEN_ENV, "correct-token")
    client = TestClient(answer_api_module.app)

    payload = {
        "rows": [
            {
                "s_no": "1",
                "model_details": "Refurbished Apple iPhone 13",
                "superb": "25,999",
                "good": "23,999",
                "fair": "21,999",
                "slug": "apple-iphone-13",
                "product_url": "https://grest.in/products/apple-iphone-13",
                "details": "",
                "is_new": False,
            }
        ]
    }

    resp = client.post("/admin/iphone-specs/save?token=wrong-token", json=payload)

    assert resp.status_code == 403


def test_save_iphone_specs_returns_summary(monkeypatch) -> None:
    """Saving specs should surface the summary from the underlying helper."""

    monkeypatch.setenv(ADMIN_TOKEN_ENV, "admin-token")

    captured: dict[str, Any] = {}

    def fake_save(rows):
        captured["rows"] = rows
        return {
            "status": "ok",
            "message": "updated 2 rows, created 1",
            "updated_slugs": ["slug-1", "slug-2"],
            "created_slugs": ["slug-3"],
        }

    monkeypatch.setattr(
        answer_api_module,
        "_save_iphone_specs_rows_to_sheet",
        fake_save,
        raising=True,
    )

    client = TestClient(answer_api_module.app)

    payload = {
        "rows": [
            {
                "s_no": "1",
                "model_details": "Model 1",
                "superb": "25,999",
                "good": "23,999",
                "fair": "21,999",
                "slug": "slug-1",
                "product_url": "https://grest.in/products/slug-1",
                "details": "",
                "is_new": False,
            },
            {
                "s_no": "2",
                "model_details": "Model 2",
                "superb": "30,999",
                "good": "28,999",
                "fair": "26,999",
                "slug": "slug-3",
                "product_url": "https://grest.in/products/slug-3",
                "details": "",
                "is_new": True,
            },
        ]
    }

    resp = client.post("/admin/iphone-specs/save?token=admin-token", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert data["message"] == "updated 2 rows, created 1"
    assert data["updated_slugs"] == ["slug-1", "slug-2"]
    assert data["created_slugs"] == ["slug-3"]

    # Ensure the helper saw the expected rows
    assert "rows" in captured
    assert len(captured["rows"]) == 2
