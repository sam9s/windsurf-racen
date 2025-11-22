from __future__ import annotations

import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Iterable, Tuple

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import step4_answer as sa  # type: ignore

TESTS_DIR = ROOT / "tests"
OUT_MD = TESTS_DIR / "queries" / "grest_holy_grail_results.md"

# Regexes to extract queries from tests
PAT_LITERAL = re.compile(
    r"(?P<prefix>\b(?:sa\.)?answer_query\()\s*(?P<qquote>['\"]) (?P<query>.+?) (?P=qquote) (?P<rest>[^)]*)\)",
    re.VERBOSE | re.DOTALL,
)
# For calls with variable argument, try to resolve from nearby assignment
PAT_CALL = re.compile(r"\b(?:sa\.)?answer_query\(\s*(?P<arg>[^,\s)]+)")
PAT_ASSIGN = re.compile(r"^\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(['\"])(?P<val>.+?)\2\s*$")
PAT_TOPK = re.compile(r"top_k\s*=\s*(?P<k>\d+)")


def _scan_file_for_queries(path: Path) -> Iterable[Tuple[str, int]]:
    """Yield (query, top_k) pairs discovered in a test file.

    - Captures literal string queries in answer_query(...)
    - Also resolves simple variable assignments like q = "..." followed by answer_query(q, ...)
    """
    text = path.read_text(encoding="utf-8", errors="ignore")

    # First, literal string calls
    for m in PAT_LITERAL.finditer(text):
        query = m.group("query").strip()
        rest = m.group("rest") or ""
        km = PAT_TOPK.search(rest)
        top_k = int(km.group("k")) if km else 10
        if query:
            yield (query, top_k)

    # Next, variable-based calls with nearby assignment
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        cm = PAT_CALL.search(ln)
        if not cm:
            continue
        arg = cm.group("arg").strip()
        # If arg starts with quote, it was already captured above
        if arg.startswith(("'", '"')):
            continue
        # Walk back up to 5 lines to find simple assignment
        assign_val: str | None = None
        for j in range(max(0, i - 5), i):
            am = PAT_ASSIGN.match(lines[j])
            if am and am.group("name") == arg:
                assign_val = am.group("val").strip()
        if assign_val:
            km2 = PAT_TOPK.search(ln)
            top_k2 = int(km2.group("k")) if km2 else 10
            yield (assign_val, top_k2)


def collect_queries() -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for py in TESTS_DIR.rglob("test_*.py"):
        # Skip our own runners/results
        if py.name in {"grest_holy_grail_test_file.py", "grest_holy_grail_output_runner.py", "grest_holy_grail_output_all_tests.py"}:
            continue
        for q, k in _scan_file_for_queries(py):
            pairs.append((q, k))
    # Deduplicate while preserving order
    seen = set()
    uniq: list[tuple[str, int]] = []
    for q, k in pairs:
        key = (q, k)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((q, k))
    return uniq


def run() -> int:
    queries = collect_queries()
    lines: list[str] = []
    lines.append("# GREST Holy Grail Live Outputs (All Tests)\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Total queries discovered: {len(queries)}\n")

    for idx, (query, k) in enumerate(queries, 1):
        try:
            answer, citations = sa.answer_query(query, top_k=k)
        except Exception as e:
            answer, citations = f"<error: {e}>", []
        lines.append(f"\n## [{idx}] Query\n")
        lines.append(f"**Text:** {query}\n")
        lines.append(f"**Top-K:** {k}\n")
        lines.append("**Answer:**\n")
        lines.append("```")
        lines.append(answer or "<empty>")
        lines.append("```")
        lines.append("**Citations:**\n")
        if citations:
            for i, c in enumerate(citations, 1):
                url = getattr(c, "url", "")
                s = getattr(c, "start_line", 1)
                e = getattr(c, "end_line", 1)
                lines.append(f"- [{i}] {url} (lines {s}-{e})")
        else:
            lines.append("- <none>")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote results to {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
