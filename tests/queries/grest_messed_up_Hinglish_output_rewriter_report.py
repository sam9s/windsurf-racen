from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import step4_answer as sa  # type: ignore

QUERIES_TXT = ROOT / "tests" / "queries" / "grest_messed_up_Hinglish_queries.txt"
OUT_MD = ROOT / "tests" / "queries" / "grest_messed_up_Hinglish_output_rewriter_report.md"


def _load_queries(path: Path) -> List[Tuple[str, int]]:
    """Load messed-up Hinglish queries from a text file.

    Each non-empty, non-comment line is either:
    - "query" (implicitly top_k=10), or
    - "query||k" where k is an integer top_k value.

    Args:
        path: Path to the queries text file.

    Returns:
        List of (query, top_k) tuples.
    """

    items: List[Tuple[str, int]] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if "||" in s:
            left, right = s.split("||", 1)
            q = left.strip()
            try:
                k = int(right.strip())
            except Exception:
                k = 10
            if q:
                items.append((q, k))
        else:
            items.append((s, 10))
    return items


def _status_for_answer(ans: str) -> str:
    """Classify the status of an answer for reporting.

    Args:
        ans: Answer text returned by answer_query.

    Returns:
        "ok" for normal answers,
        "fallback" when the static no-answer message is used,
        "error" when the answer is empty.
    """

    t = (ans or "").strip().lower()
    if not t:
        return "error"
    if t.startswith("not found in sources provided"):
        return "fallback"
    return "ok"


def run() -> int:
    """Run the messed-up Hinglish benchmark and write a Markdown report.

    Returns:
        Process exit code (0 on success).
    """

    pairs = _load_queries(QUERIES_TXT)
    n = len(pairs)

    answers: List[str] = [""] * n
    citations: List[list] = [[] for _ in range(n)]
    statuses: List[str] = ["error"] * n
    cold_ms: List[float] = [0.0] * n
    hot_ms: List[float] = [0.0] * n

    # Cold pass: capture outputs and latency
    t0_cold = time.perf_counter()
    for i, (q, k) in enumerate(pairs):
        start = time.perf_counter()
        try:
            ans, cits = sa.answer_query(q, top_k=k)
            answers[i] = ans
            citations[i] = cits
            statuses[i] = _status_for_answer(ans)
        except Exception as exc:  # pragma: no cover - defensive logging path
            answers[i] = f"<error: {exc}>"
            citations[i] = []
            statuses[i] = "error"
        cold_ms[i] = (time.perf_counter() - start) * 1000.0
    cold_total_s = time.perf_counter() - t0_cold

    # Hot pass: latency only
    t0_hot = time.perf_counter()
    for i, (q, k) in enumerate(pairs):
        start = time.perf_counter()
        try:
            _ = sa.answer_query(q, top_k=k)
        except Exception:
            # Errors are already captured in the cold pass
            pass
        hot_ms[i] = (time.perf_counter() - start) * 1000.0
    hot_total_s = time.perf_counter() - t0_hot

    ok = sum(1 for st in statuses if st == "ok")
    fallback = sum(1 for st in statuses if st == "fallback")
    err = sum(1 for st in statuses if st == "error")

    avg_cold = (sum(cold_ms) / n) if n else 0.0
    avg_hot = (sum(hot_ms) / n) if n else 0.0

    # Compose Markdown
    lines: List[str] = []
    lines.append("# GREST Messed-Up Hinglish Output Rewrite Report\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Total queries: {n}\n")

    lines.append("## Summary\n")
    lines.append(f"OK: {ok}")
    lines.append(f"Fallback: {fallback}")
    lines.append(f"Error: {err}")
    lines.append(f"Cold run total: {cold_total_s:.3f} s")
    lines.append(f"Hot run total:  {hot_total_s:.3f} s")
    if hot_total_s > 0:
        lines.append(f"Overall speedup: x{cold_total_s / hot_total_s:.2f}")
    lines.append(f"Avg latency (cold): {avg_cold:.1f} ms")
    lines.append(f"Avg latency (hot):  {avg_hot:.1f} ms")
    lines.append("")

    for idx, ((q, k), st) in enumerate(zip(pairs, statuses), 1):
        ans = answers[idx - 1]
        cits = citations[idx - 1]
        c_time = cold_ms[idx - 1]
        h_time = hot_ms[idx - 1]
        lines.append(f"\n## [{idx}] Query\n")
        lines.append(f"**Original (Hinglish) text:** {q}\n")
        lines.append(f"**Top-K:** {k}\n")
        lines.append(f"**Status:** {st}\n")
        lines.append(f"**Latency (cold/hot):** {c_time:.1f} ms / {h_time:.1f} ms\n")
        lines.append("**Answer:**\n")
        lines.append("```")
        lines.append(ans or "<empty>")
        lines.append("```")
        lines.append("**Citations:**\n")
        if cits:
            for j, cit in enumerate(cits, 1):
                try:
                    url = getattr(cit, "url", "")
                    sline = getattr(cit, "start_line", 1)
                    eline = getattr(cit, "end_line", 1)
                except Exception:
                    url, sline, eline = "", 1, 1
                lines.append(f"- [{j}] {url} (lines {sline}-{eline})")
        else:
            lines.append("- <none>")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote messed-up Hinglish output rewrite report to {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
