from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure local 'src' is importable (mirrors step4_answer style)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Allow importing scripts.step4_answer when running from project root
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import yaml  # type: ignore
except Exception as e:
    print(f"Missing dependency pyyaml: {e}")
    sys.exit(1)

from step4_answer import answer_query  # type: ignore


def load_suite(path: Path) -> List[Dict[str, Any]]:
    """
    Load a YAML test suite file.

    Args:
        path (Path): Path to YAML file with queries.

    Returns:
        list[dict]: Parsed list of test cases.
    """
    data: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
            if not isinstance(data, list):
                data = []
    except Exception as e:
        print(f"Error reading suite {path}: {e}")
    return data


def run_suite(cases: List[Dict[str, Any]], k: int = 6) -> List[Dict[str, Any]]:
    """
    Execute a list of query cases using in-process answer_query.

    Args:
        cases (list[dict]): Items with keys 'q' and optional 'expect_any' (list[str]).
        k (int): Top-k for retrieval.

    Returns:
        list[dict]: Result rows with query, pass/fail, and notes.
    """
    rows: List[Dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        q = (case.get("q") or "").strip()
        expect_any: List[str] = case.get("expect_any") or []
        if not q:
            rows.append({"q": q, "pass": False, "notes": "empty query"})
            continue
        ans, _ = answer_query(q, top_k=k)
        a_low = (ans or "").lower()
        ok = True
        note_parts: List[str] = []
        if expect_any:
            if not any(tok.lower() in a_low for tok in expect_any):
                ok = False
                note_parts.append(f"no expected tokens present: {expect_any}")
        if a_low.startswith("not found in sources provided"):
            ok = False
            note_parts.append("fallback triggered")
        rows.append({
            "q": q,
            "pass": ok,
            "notes": "; ".join(note_parts),
            "answer": ans[:300] if ans else "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query regression suites")
    parser.add_argument("--suites", nargs="+", help="Paths to YAML suites under tests/queries")
    parser.add_argument("--k", type=int, default=6, help="Top-k retrieval")
    parser.add_argument("--out_csv", default=str(ROOT / "tests" / "query_report.csv"))
    parser.add_argument("--out_json", default=str(ROOT / "tests" / "query_report.json"))
    args = parser.parse_args()

    all_rows: List[Dict[str, Any]] = []
    for suite_path in args.suites:
        p = Path(suite_path)
        if not p.is_absolute():
            p = ROOT / suite_path
        cases = load_suite(p)
        rows = run_suite(cases, k=args.k)
        all_rows.extend(rows)

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["q", "pass", "notes", "answer"])
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    # Write JSON
    out_json = Path(args.out_json)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)

    total = len(all_rows)
    passed = sum(1 for r in all_rows if r.get("pass"))
    print(f"Total: {total} | Passed: {passed} | Failed: {total - passed}")


if __name__ == "__main__":
    main()
