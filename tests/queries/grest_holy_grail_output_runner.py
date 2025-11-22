from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

# Ensure project root is on sys.path so 'scripts' can be imported when
# running this file directly from tests/queries.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import step4_answer as sa


def run() -> int:
    """
    Live end-to-end query run (no mocks) and write actual outputs to a Markdown file.

    Returns:
        int: 0 on success, nonzero on failure.
    """
    root = Path(__file__).resolve().parents[2]
    out_path = root / "tests" / "queries" / "grest_holy_grail_results.md"

    scenarios: list[tuple[str, str, int]] = [
        ("Brand reputation: Trustpilot rating", "what is the Trustpilot rating of Grest?", 10),
        ("Generic/blog: best time to buy", "whats the best time to buy refurbished phone?", 10),
        ("Product: iPhone 11 availability/specs", "do you have iphone 11?", 10),
        ("Shipping policy", "what is your shipping policy?", 10),
        ("Warranty policy", "what warranty do you offer?", 10),
        ("Comparison (default settings)", "which is better iphone 13 or iphone 14?", 6),
    ]

    lines: list[str] = []
    lines.append(f"# GREST Holy Grail Live Outputs\n")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")

    for title, query, k in scenarios:
        try:
            answer, citations = sa.answer_query(query, top_k=k)
        except Exception as e:
            answer, citations = f"<error: {e}>", []
        lines.append(f"\n## {title}\n")
        lines.append(f"**Query:** {query}\n")
        lines.append("**Answer:**\n")
        lines.append("```")
        lines.append(answer or "<empty>")
        lines.append("```")
        lines.append("**Citations:**\n")
        if citations:
            for i, c in enumerate(citations, 1):
                try:
                    url = getattr(c, "url", "")
                    s = getattr(c, "start_line", 1)
                    e = getattr(c, "end_line", 1)
                except Exception:
                    url, s, e = "", 1, 1
                lines.append(f"- [{i}] {url} (lines {s}-{e})")
        else:
            lines.append("- <none>")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
