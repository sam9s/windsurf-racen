from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def _load_runner_module() -> ModuleType:
    """Import the messed-up English normalization runner module.

    This ensures the tests/queries directory is on sys.path so the runner
    script can be imported as a normal module.

    Returns:
        ModuleType: Imported runner module.
    """
    root = Path(__file__).resolve().parents[2]
    queries_dir = root / "tests" / "queries"
    if str(queries_dir) not in sys.path:
        sys.path.insert(0, str(queries_dir))
    import grest_messed_up_english_normalizer_report as runner_mod  # type: ignore

    return runner_mod


def test_load_queries_parses_counts_and_defaults(tmp_path) -> None:
    """_load_queries should parse lines with and without explicit top_k."""
    mod = _load_runner_module()
    path = tmp_path / "sample_queries.txt"
    path.write_text(
        "# comment line\n"
        "\n"
        "first query||5\n"
        "second query\n",
        encoding="utf-8",
    )

    items = mod._load_queries(path)
    assert items == [("first query", 5), ("second query", 10)]


def test_status_for_answer_basic_cases() -> None:
    """_status_for_answer should classify empty, fallback, and normal answers."""
    mod = _load_runner_module()
    assert mod._status_for_answer("") == "error"
    assert mod._status_for_answer("Not found in sources provided ...") == "fallback"
    assert mod._status_for_answer("Some normal answer") == "ok"
