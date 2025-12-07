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


def test_list_iphone_specs_requires_admin_token(monkeypatch) -> None:
    """Listing iPhone specs should be forbidden when token is wrong."""

    monkeypatch.setenv(ADMIN_TOKEN_ENV, "correct-token")
    client = TestClient(answer_api_module.app)

    resp = client.get("/admin/iphone-specs/list", params={"token": "wrong"})

    assert resp.status_code == 403


def test_list_iphone_specs_returns_rows(monkeypatch) -> None:
    """Listing iPhone specs should return normalised rows from the sheet."""

    monkeypatch.setenv(ADMIN_TOKEN_ENV, "admin-token")

    sample_rows = [
        {
            "S.No.": "1",
            "Model Details": "iPhone 13 128GB",
            "Superb": "25,999",
            "Good": "23,999",
            "Fair": "21,999",
            "Slug": "apple-iphone-13-128gb",
            "ProductURL": "https://grest.in/products/apple-iphone-13-128gb",
            "Details": "Sample row",
        },
        {
            "S.No.": "2",
            "Model Details": "iPhone 14 128GB",
            "Superb": "30,999",
            "Good": "28,999",
            "Fair": "26,999",
            "Slug": "apple-iphone-14-128gb",
            "ProductURL": "https://grest.in/products/apple-iphone-14-128gb",
            "Details": "",
        },
    ]

    def fake_loader() -> list[dict[str, Any]]:
        return sample_rows

    monkeypatch.setattr(
        answer_api_module,
        "_load_iphone_specs_rows_from_sheet",
        fake_loader,
        raising=True,
    )

    client = TestClient(answer_api_module.app)
    resp = client.get(
        "/admin/iphone-specs/list",
        params={"token": "admin-token"},
    )

    assert resp.status_code == 200
    data = resp.json()

    assert isinstance(data, list)
    assert len(data) == 2

    first = data[0]
    assert first["slug"] == "apple-iphone-13-128gb"
    assert first["model_details"] == "iPhone 13 128GB"
    assert first["superb"] == "25,999"
    assert first["product_url"].endswith("apple-iphone-13-128gb")
