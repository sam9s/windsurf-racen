from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import step4_answer as sa  # type: ignore

QUERIES_TXT = ROOT / "tests" / "queries" / "grest_benchmark_queries.txt"
OUT_MD = ROOT / "tests" / "queries" / "grest_holy_grail_results_51.md"


def _load_queries(path: Path) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
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
    t = (ans or "").strip().lower()
    if not t:
        return "error"
    # Same primary check used in step4_answer fallback
    if t.startswith("not found in sources provided"):
        return "fallback"
    return "ok"


def run() -> int:
    pairs = _load_queries(QUERIES_TXT)
    ok = 0
    fallback = 0
    error = 0

    lines: list[str] = []
    lines.append("# GREST Holy Grail Live Outputs (51 curated queries)\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Total queries: {len(pairs)}\n")

    for idx, (q, k) in enumerate(pairs, 1):
        try:
            ans, cits = sa.answer_query(q, top_k=k)
            st = _status_for_answer(ans)
        except Exception as e:
            ans, cits, st = f"<error: {e}>", [], "error"
        if st == "ok":
            ok += 1
        elif st == "fallback":
            fallback += 1
        else:
            error += 1

        lines.append(f"\n## [{idx}] Query\n")
        lines.append(f"**Text:** {q}\n")
        lines.append(f"**Top-K:** {k}\n")
        lines.append(f"**Status:** {st}\n")
        lines.append("**Answer:**\n")
        lines.append("```")
        lines.append(ans or "<empty>")
        lines.append("```")
        lines.append("**Citations:**\n")
        if cits:
            for i, c in enumerate(cits, 1):
                try:
                    url = getattr(c, "url", "")
                    s = getattr(c, "start_line", 1)
                    e = getattr(c, "end_line", 1)
                except Exception:
                    url, s, e = "", 1, 1
                lines.append(f"- [{i}] {url} (lines {s}-{e})")
        else:
            lines.append("- <none>")

    # Prepend summary section
    summary = [
        "\n## Summary\n",
        f"OK: {ok}",
        f"Fallback: {fallback}",
        f"Error: {error}",
        "",
    ]
    final = lines[:2] + summary + lines[2:]
    OUT_MD.write_text("\n".join(final), encoding="utf-8")
    print(f"Wrote 51-query results to {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
