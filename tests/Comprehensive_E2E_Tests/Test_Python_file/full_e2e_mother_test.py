from __future__ import annotations

"""Comprehensive E2E harness for RACEN.

This script reads a query list, runs the full `answer_query` pipeline
(cold + hot) for each entry, and writes a single Markdown report
containing:

- Original text
- Detected mode (EN / HI_EN)
- Normalized LLM text
- Effective query after family/brand preservation
- Top-K
- Status (ok / fallback / error)
- Cold / hot latency
- Final user-visible answer
- Citations

The query file format is documented in:
`tests/Comprehensive_E2E_Tests/final_testing_plan.md`.
"""

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root on sys.path (file is under tests/Comprehensive_E2E_Tests/Test_Python_file)
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import step4_answer as sa  # type: ignore
from racen.query_normalizer import normalize_query  # type: ignore


@dataclass
class QueryItem:
    """Parsed query metadata for a single test case."""

    intent_id: str
    variant: str
    text: str
    top_k: int


def _parse_line(raw: str) -> Optional[QueryItem]:
    """Parse a single line from the query file.

    Supports two formats (see final_testing_plan.md):

    1. Simple:
       - "query"
       - "query||k"

    2. Grouped:
       - "intent_id||variant||query||k"
       - "intent_id||variant||query" (defaults k=10)

    Args:
        raw: Raw line from the query file.

    Returns:
        QueryItem or None when the line is empty/comment.
    """

    s = raw.strip()
    if not s or s.startswith("#"):
        return None

    parts = s.split("||")
    # Grouped format with intent + variant
    if len(parts) >= 3:
        intent_id = parts[0].strip()
        variant = parts[1].strip()
        query = parts[2].strip()
        top_k = 10
        if len(parts) >= 4:
            try:
                top_k = int(parts[3].strip())
            except Exception:
                top_k = 10
        return QueryItem(intent_id=intent_id, variant=variant, text=query, top_k=top_k)

    # Simple formats
    if len(parts) == 2:
        query = parts[0].strip()
        try:
            top_k = int(parts[1].strip())
        except Exception:
            top_k = 10
        return QueryItem(intent_id="", variant="", text=query, top_k=top_k)

    # Only query text
    return QueryItem(intent_id="", variant="", text=s, top_k=10)


def _load_queries(path: Path) -> List[QueryItem]:
    """Load queries from a text file.

    Args:
        path: Path to the queries text file.

    Returns:
        List of QueryItem entries.
    """

    items: List[QueryItem] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for raw in text.splitlines():
        item = _parse_line(raw)
        if item is not None and item.text:
            items.append(item)
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


def _normalize_with_family_preservation(query: str) -> Tuple[str, str, str]:
    """Run the LLM normalizer and family-preservation logic for a query.

    This mirrors the behavior used inside answer_query without modifying
    core logic.

    Args:
        query: Original user text.

    Returns:
        A tuple of (mode, normalized_llm, effective_query).
    """

    mode = sa._detect_mode(query)  # type: ignore[attr-defined]
    locale = "hi-IN" if mode == "HI_EN" else "en-IN"
    try:
        result = normalize_query(query, user_locale=locale)
        normalized_llm = (result.normalized_query or query).strip()
    except Exception:
        normalized_llm = query.strip()
    try:
        effective_query = sa._preserve_family_hints(query, normalized_llm)  # type: ignore[attr-defined]
    except Exception:
        effective_query = normalized_llm or query
    return mode, normalized_llm or query, effective_query or query


def run_full_e2e(query_file: Path, out_md: Path) -> int:
    """Run the full E2E mother test and write a Markdown report.

    Args:
        query_file: Path to the input queries file.
        out_md: Path to the output Markdown report.

    Returns:
        Process exit code (0 on success).
    """

    items = _load_queries(query_file)
    n = len(items)

    modes: List[str] = [""] * n
    norm_texts: List[str] = [""] * n
    eff_texts: List[str] = [""] * n
    answers: List[str] = [""] * n
    citations_all: List[List] = [[] for _ in range(n)]
    statuses: List[str] = ["error"] * n
    cold_ms: List[float] = [0.0] * n
    hot_ms: List[float] = [0.0] * n

    # Cold pass: normalization + answer + citations + latency
    t0_cold = time.perf_counter()
    for i, item in enumerate(items):
        mode, norm_llm, eff_query = _normalize_with_family_preservation(item.text)
        modes[i] = mode
        norm_texts[i] = norm_llm
        eff_texts[i] = eff_query

        start = time.perf_counter()
        try:
            ans, cits = sa.answer_query(item.text, top_k=item.top_k)  # type: ignore[call-arg]
            answers[i] = ans
            citations_all[i] = cits
            statuses[i] = _status_for_answer(ans)
        except Exception as exc:  # pragma: no cover - defensive logging path
            answers[i] = f"<error: {exc}>"
            citations_all[i] = []
            statuses[i] = "error"
        cold_ms[i] = (time.perf_counter() - start) * 1000.0
    cold_total_s = time.perf_counter() - t0_cold

    # Hot pass: answer-only latency
    t0_hot = time.perf_counter()
    for i, item in enumerate(items):
        start = time.perf_counter()
        try:
            _ = sa.answer_query(item.text, top_k=item.top_k)  # type: ignore[call-arg]
        except Exception:
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
    lines.append("# RACEN Full E2E Mother Test Report\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Query file: {query_file}\n")
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

    for idx, item in enumerate(items, 1):
        ans = answers[idx - 1]
        cits = citations_all[idx - 1]
        st = statuses[idx - 1]
        c_time = cold_ms[idx - 1]
        h_time = hot_ms[idx - 1]
        mode = modes[idx - 1]
        norm_llm = norm_texts[idx - 1]
        eff_q = eff_texts[idx - 1]

        lines.append(f"\n## [{idx}] Query\n")
        if item.intent_id:
            lines.append(f"**Intent ID:** {item.intent_id}\n")
        if item.variant:
            lines.append(f"**Variant:** {item.variant}\n")
        lines.append(f"**Original text:** {item.text}\n")
        lines.append(f"**Detected mode:** {mode}\n")
        lines.append(f"**Normalized (LLM) text:** {norm_llm}\n")
        lines.append(f"**Effective query after family preservation:** {eff_q}\n")
        lines.append(f"**Top-K:** {item.top_k}\n")
        lines.append(f"**Status:** {st}\n")
        lines.append(f"**Latency (cold/hot):** {c_time:.1f} ms / {h_time:.1f} ms\n")
        lines.append("**Final answer (user-visible):**\n")
        lines.append("```text")
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

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote full E2E mother test report to {out_md}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for the full E2E mother test.

    Args:
        argv: Optional list of command-line arguments.

    Returns:
        Process exit code.
    """

    parser = argparse.ArgumentParser(description="RACEN full E2E mother test harness")
    parser.add_argument(
        "query_file",
        type=str,
        help=(
            "Path to the query file (see Test_Queries format). "
            "Relative paths are resolved from the project root."
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help=(
            "Optional output Markdown path under Test_Results. "
            "If omitted, a name is derived from the query file and timestamp."
        ),
    )

    args = parser.parse_args(argv)

    query_path = Path(args.query_file)
    if not query_path.is_absolute():
        query_path = ROOT / query_path
    if not query_path.exists():
        raise SystemExit(f"Query file not found: {query_path}")

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
    else:
        # Default under Test_Results
        results_dir = (
            ROOT
            / "tests"
            / "Comprehensive_E2E_Tests"
            / "Test_Results"
        )
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = query_path.stem
        out_path = results_dir / f"full_e2e_{base}_{ts}.md"

    return run_full_e2e(query_path, out_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
