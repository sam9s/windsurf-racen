from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Make local src and scripts importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load .env if present
try:
    import dotenv  # type: ignore

    for env_path in [ROOT / "windsurf-racen-local" / ".env", ROOT / ".env"]:
        if env_path.exists():
            dotenv.load_dotenv(dotenv_path=env_path, override=False)
            break
except Exception:
    pass

from racen.log import get_logger
from step4_answer import answer_query

logger = get_logger("scripts.eval_freeform")


@dataclass
class QAItem:
    question: str
    expected_hint: Optional[str] = None  # e.g., "FAQs", "Warranty", "Returns", "Shipping", "Terms"


ALIASES = {
    "FAQs": ["/pages/faqs"],
    "Warranty": ["/pages/warranty"],
    "Returns": ["/policies/refund/policy", "/pages/returns-refund-cancellation"],
    "Shipping": ["/policies/shipping/policy", "/pages/shipping"],
    "Terms": ["/policies/terms/of/service", "/pages/terms-and-condition"],
}


def parse_markdown(md_path: Path) -> List[QAItem]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    items: List[QAItem] = []
    current_hint: Optional[str] = None

    # Heuristics to extract questions and expected hints from flexible MD
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue

        # Capture a hint line like: **Source:** FAQs/Shipping or Expected: Warranty
        low = s.lower()
        if low.startswith("**source:**") or low.startswith("expected:"):
            hint_val = s.split(":", 1)[1].strip()
            found = None
            for label in ALIASES.keys():
                if label.lower() in hint_val.lower():
                    found = label
                    break
            current_hint = found
            continue

        # Headings as questions: ### what's your warranty?
        if s.startswith("### "):
            q = s[4:].strip()
            if q:
                items.append(QAItem(question=q, expected_hint=current_hint))
                continue

        # Bullet questions like: - what is your return policy?
        if s.startswith("- ") and s.endswith("?") and len(s) > 4:
            q = s[2:].strip()
            items.append(QAItem(question=q, expected_hint=current_hint))
            continue

        # Plain-line question ending with ? but avoid URLs/lists
        if s.endswith("?") and not s.startswith("http") and " " in s:
            if not s.startswith("## ") and not s.startswith("# "):
                items.append(QAItem(question=s, expected_hint=current_hint))
                continue

    logger.info(f"Parsed {len(items)} question(s) from {md_path.name}")
    return items


def match_pass(expected: Optional[str], citation_urls: List[str]) -> Tuple[bool, str]:
    if not expected:
        return True, "no-expected"
    patterns = ALIASES.get(expected, [])
    if not patterns:
        return False, f"unknown-expected:{expected}"
    low_urls = [u.lower() for u in citation_urls]
    for pat in patterns:
        pat_l = pat.lower()
        if any(pat_l in u for u in low_urls):
            return True, pat
    return False, ",".join(patterns)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate free-form questions and generate a markdown report")
    ap.add_argument("--md", required=True, help="Path to Grest_Use_Cases_Freeform.md")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval_freeform_report.md"))
    ap.add_argument("--k", type=int, default=6, help="Top-k chunks to use for answering")
    args = ap.parse_args()

    md_path = Path(args.md)
    if not md_path.exists():
        raise SystemExit(f"Markdown file not found: {md_path}")

    items = parse_markdown(md_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[str] = []
    rows.append("# Free-form Evaluation Report")
    rows.append("")
    rows.append(f"Source: {md_path}")
    rows.append("")
    passed = 0

    for idx, it in enumerate(items, 1):
        ans, cits = answer_query(it.question, top_k=args.k)
        urls = [c.url for c in cits]
        ok, rule = match_pass(it.expected_hint, urls)
        if ok:
            passed += 1
        rows.append(f"## Q{idx}")
        rows.append(it.question)
        if it.expected_hint:
            rows.append(f"Expected: {it.expected_hint}")
        rows.append("")
        rows.append("Answer:")
        rows.append(ans)
        rows.append("")
        rows.append("Citations:")
        for i, u in enumerate(urls, 1):
            rows.append(f"- [{i}] {u}")
        rows.append("")
        rows.append(f"Result: {'PASS' if ok else 'FAIL'} ({rule})")
        rows.append("")

    total = len(items)
    acc = (passed / total * 100.0) if total else 0.0
    rows.insert(0, f"Accuracy: {passed}/{total} ({acc:.1f}%)")
    rows.insert(0, "")

    out_path.write_text("\n".join(rows), encoding="utf-8")
    logger.info(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
