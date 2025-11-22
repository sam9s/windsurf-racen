"""
Unified test runner for the full test suite.

Run directly with:
    python tests/queries/grest_holy_grail_test_file.py

This auto-discovers and runs ALL tests under the tests/ folder recursively.
"""
from __future__ import annotations

from pathlib import Path
import sys
import pytest


def run(quiet: bool = True) -> int:
    """Execute the entire tests/ tree via pytest.

    Args:
        quiet (bool): If True, run pytest with -q.

    Returns:
        int: Pytest exit code (0 = success, nonzero = failures).
    """
    # Ensure project root is importable so tests can `from scripts import ...`.
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    tests_dir = root / "tests"
    args: list[str] = [str(tests_dir)]
    if quiet:
        args.insert(0, "-q")
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(run(quiet=True))
