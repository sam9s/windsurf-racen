"""
Generate a Markdown report for assistant meta Q&A queries.

Reads queries from tests/queries/assistant_meta_queries.txt, calls the
integrated answer flow (answer_query), and writes a report to
tests/queries/assistant_meta_results.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Tuple

# Ensure src/ and scripts/ are importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.step4_answer import answer_query  # type: ignore  # noqa: E402
from racen.assistant_meta import try_answer_meta_question  # type: ignore  # noqa: E402


def load_queries(path: Path) -> List[str]:
    """
    Load newline-delimited queries.

    Args:
        path (Path): Path to text file with one query per line.

    Returns:
        List[str]: Queries.
    """
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run_meta_suite(queries: List[str]) -> Tuple[str, int, int, float]:
    """
    Run the meta queries via answer_query and build a Markdown report.

    Args:
        queries (List[str]): Input queries.

    Returns:
        Tuple[str, int, int, float]: (markdown, ok_count, total, avg_ms)
    """
    lines: List[str] = []
    lines.append("# RACEN Assistant Meta Q&A Report")
    lines.append("")

    total = 0
    ok = 0
    latencies: List[float] = []

    for q in queries:
        total += 1
        t0 = time.perf_counter()
        ans, cits = answer_query(q, top_k=1)
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1000.0
        latencies.append(ms)

        expected = try_answer_meta_question(q)
        status = "OK" if (expected is not None and ans.strip() == expected.strip()) else "NotMeta/Mismatch"
        if status == "OK":
            ok += 1

        lines.append(f"## Q{total}. {q}")
        lines.append("")
        lines.append(f"- Status: {status}")
        lines.append(f"- Latency: {ms:.1f} ms")
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append("```")
        lines.append(ans)
        lines.append("```")
        lines.append("")

    avg_ms = sum(latencies) / len(latencies) if latencies else 0.0
    lines.insert(1, f"- Total: {total}")
    lines.insert(2, f"- OK: {ok}")
    lines.insert(3, f"- Avg latency: {avg_ms:.1f} ms")

    md = "\n".join(lines)
    return md, ok, total, avg_ms


def main() -> None:
    qpath = ROOT / "tests" / "queries" / "assistant_meta_queries.txt"
    out = ROOT / "tests" / "queries" / "assistant_meta_results.md"
    queries = load_queries(qpath)
    md, ok, total, avg_ms = run_meta_suite(queries)
    out.write_text(md, encoding="utf-8")
    print(f"Wrote {out} | OK={ok}/{total}, avg={avg_ms:.1f} ms")


if __name__ == "__main__":
    main()
