from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import step4_answer as sa  # type: ignore
try:
    from tests.queries.grest_holy_grail_output_all_tests import collect_queries  # type: ignore
except Exception:  # pragma: no cover
    # Fallback: import via path insertion to avoid package requirements
    tq = ROOT / "tests"
    tq_q = tq / "queries"
    if str(tq) not in sys.path:
        sys.path.insert(0, str(tq))
    if str(tq_q) not in sys.path:
        sys.path.insert(0, str(tq_q))
    from grest_holy_grail_output_all_tests import collect_queries  # type: ignore

OUT_MD = ROOT / "tests" / "queries" / "grest_cache_benchmark_results.md"


def _load_queries_from_file(path: Path) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # format: text || top_k
        if "||" in s:
            left, right = s.split("||", 1)
            q = left.strip()
            try:
                k = int(right.strip())
            except Exception:
                k = 10
            if q:
                out.append((q, k))
        else:
            out.append((s, 10))
    return out


def run_two_pass_benchmark() -> int:
    # If a file path is supplied as first arg, use that; else collect from tests
    queries: list[tuple[str, int]]
    arg_path = None
    if len(sys.argv) > 1:
        try:
            maybe = Path(sys.argv[1])
            if maybe.exists() and maybe.is_file():
                arg_path = maybe
        except Exception:
            arg_path = None
    if arg_path is not None:
        queries = _load_queries_from_file(arg_path)
    else:
        queries = collect_queries()

    def run_once() -> tuple[float, list[float]]:
        t0 = time.perf_counter()
        per_q: list[float] = []
        for q, k in queries:
            s = time.perf_counter()
            try:
                sa.answer_query(q, top_k=k)
            except Exception:
                pass
            per_q.append(time.perf_counter() - s)
        total = time.perf_counter() - t0
        return total, per_q

    cold_total, cold_each = run_once()  # warm cache
    hot_total, hot_each = run_once()    # use cache

    def fmt(ms: float) -> str:
        return f"{ms*1000:.1f} ms"

    lines: list[str] = []
    lines.append("# RACEN Cache Benchmark\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Total queries: {len(queries)}\n")

    lines.append("## Summary\n")
    lines.append(f"Cold run total: {cold_total:.3f} s")
    lines.append(f"Hot run total:  {hot_total:.3f} s")
    if cold_total > 0:
        speedup = cold_total / max(hot_total, 1e-6)
        lines.append(f"Overall speedup: x{speedup:.2f}")
    lines.append("")

    lines.append("## Per-query timings (hot)\n")
    for i, (pair, dt) in enumerate(zip(queries, hot_each), 1):
        q, k = pair
        q_disp = q if len(q) <= 80 else q[:77] + "..."
        lines.append(f"- [{i}] {fmt(dt)} | {q_disp}")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote benchmark to {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_two_pass_benchmark())
