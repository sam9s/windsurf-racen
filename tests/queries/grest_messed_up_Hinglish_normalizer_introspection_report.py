from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import step4_answer as sa  # type: ignore
from racen.query_normalizer import normalize_query  # type: ignore

QUERIES_TXT = ROOT / "tests" / "queries" / "grest_messed_up_Hinglish_queries.txt"
OUT_MD = ROOT / "tests" / "queries" / "grest_messed_up_Hinglish_normalizer_introspection.md"


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


def _safe_normalize(q: str) -> str:
    """Run the LLM normalizer on a Hinglish query and return the normalized text.

    This mirrors the locale logic used in answer_query: we detect the
    language/mode and pass a locale hint, but fall back gracefully if
    anything fails.

    Args:
        q: Original messy Hinglish query.

    Returns:
        Normalized query text (or the original query when normalization
        fails for any reason).
    """

    try:
        try:
            mode_hint = sa._detect_mode(q)  # type: ignore[attr-defined]
            locale_hint = "hi-IN" if mode_hint == "HI_EN" else "en-IN"
        except Exception:
            locale_hint = None
        result = normalize_query(q, user_locale=locale_hint)
        norm = (result.normalized_query or q).strip()
        return norm or q
    except Exception as exc:  # pragma: no cover - defensive path
        return f"<normalization error: {exc}>"


def run() -> int:
    """Generate an introspection report for messed-up Hinglish normalization.

    For each query in grest_messed_up_Hinglish_queries.txt, this script:
    - Records the original Hinglish text.
    - Runs the LLM-based normalizer to capture the normalized text.
    - Calls the full answer_query pipeline to capture the final answer and
      citations.
    - Writes a Markdown report showing original vs normalized text plus the
      generated answer and citations.

    Returns:
        Process exit code (0 on success).
    """

    pairs = _load_queries(QUERIES_TXT)

    lines: List[str] = []
    lines.append("# GREST Messed-Up Hinglish Normalization Introspection\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
    lines.append(f"Total queries: {len(pairs)}\n")

    for idx, (q, k) in enumerate(pairs, 1):
        normalized = _safe_normalize(q)
        try:
            answer, citations = sa.answer_query(q, top_k=k)
        except Exception as exc:  # pragma: no cover - defensive path
            answer, citations = f"<error from answer_query: {exc}>", []

        lines.append(f"\n## [{idx}] Query\n")
        lines.append(f"**Original (Hinglish) text:** {q}\n")
        lines.append(f"**Normalized text:** {normalized}\n")
        lines.append(f"**Top-K:** {k}\n")
        lines.append("**Answer:**\n")
        lines.append("```")
        lines.append(answer or "<empty>")
        lines.append("```")
        lines.append("**Citations:**\n")
        if citations:
            for j, cit in enumerate(citations, 1):
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
    print(f"Wrote messed-up Hinglish normalization introspection report to {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
