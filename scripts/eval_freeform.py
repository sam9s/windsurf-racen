from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
import os
import time
import statistics
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
from racen.step3_retrieve import effective_settings
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
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only the first N questions (0 = all)")
    ap.add_argument("--offset", type=int, default=0, help="Skip the first N questions before evaluating")
    ap.add_argument(
        "--expected",
        action="append",
        help=(
            "Filter to only questions whose Expected hint matches one of the provided labels "
            "(e.g., FAQs, Warranty, Returns, Shipping, Terms). Can be repeated."
        ),
    )
    ap.add_argument(
        "--allowlist",
        default="",
        help="Comma-separated source allowlist patterns (e.g., /pages/faqs,/policies/shipping/policy)",
    )
    args = ap.parse_args()

    md_path = Path(args.md)
    if not md_path.exists():
        raise SystemExit(f"Markdown file not found: {md_path}")

    items = parse_markdown(md_path)
    # Optional filtering by Expected hint labels
    if args.expected:
        wanted = {s.strip() for s in args.expected if s and s.strip()}
        if wanted:
            items = [it for it in items if it.expected_hint in wanted]
    if args.offset and args.offset > 0:
        items = items[args.offset :]
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    # Force in-process allowlist if provided (ensures it's visible in effective settings)
    if args.allowlist:
        os.environ["RETRIEVE_SOURCE_ALLOWLIST"] = args.allowlist

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[str] = []
    rows.append("# Free-form Evaluation Report")
    rows.append("")
    rows.append(f"Source: {md_path}")
    rows.append("")
    # Settings snapshot for reproducibility
    rows.append("## Settings")
    rows.append(f"k: {args.k}")
    rows.append(f"limit: {args.limit}")
    rows.append(f"offset: {args.offset}")
    rows.append(f"expected filter: {', '.join(args.expected) if args.expected else '[none]'}")
    rows.append("")
    rows.append("### Env (key retrieval/answering flags)")
    def g(name: str, default: str = "") -> str:
        try:
            v = os.getenv(name)
            return v if v is not None and v != "" else default
        except Exception:
            return default
    rows.append(f"FAST_MODE: {g('FAST_MODE','[unset]')}")
    rows.append(f"TOP_K: {g('TOP_K','[unset]')}")
    rows.append(f"RERANK_TOP_N: {g('RERANK_TOP_N','[unset]')}")
    rows.append(f"RETRIEVE_SOURCE_ALLOWLIST: {g('RETRIEVE_SOURCE_ALLOWLIST','[unset]')}")
    rows.append(f"RETRIEVE_BACKOFF_ENABLE: {g('RETRIEVE_BACKOFF_ENABLE','[unset]')}")
    rows.append(f"RETRIEVE_BACKOFF_THRESHOLD: {g('RETRIEVE_BACKOFF_THRESHOLD','[unset]')}")
    rows.append(f"RETRIEVE_BACKOFF_SECONDARY: {g('RETRIEVE_BACKOFF_SECONDARY','[unset]')}")
    rows.append(f"HNSW_EF_SEARCH: {g('HNSW_EF_SEARCH','[unset]')}")
    rows.append(f"EMBED_MODEL: {g('EMBED_MODEL','[unset]')}")
    rows.append(f"EMBED_DIM: {g('EMBED_DIM','[unset]')}")
    rows.append(f"OPENAI_MODEL: {g('OPENAI_MODEL','[unset]')}")
    rows.append(f"OPENAI_BASE_URL: {g('OPENAI_BASE_URL','[unset]')}")
    rows.append(f"PGHOST: {g('PGHOST','[unset]')}")
    rows.append(f"PGPORT: {g('PGPORT','[unset]')}")
    rows.append(f"PGDATABASE: {g('PGDATABASE','[unset]')}")
    # Answer shaping and DB schema hints
    rows.append(f"ANSWER_SHORT: {g('ANSWER_SHORT','[unset]')}")
    rows.append(f"ANSWER_MAX_TOKENS: {g('ANSWER_MAX_TOKENS','[unset]')}")
    rows.append(f"ANSWER_CHUNK_CHAR_BUDGET: {g('ANSWER_CHUNK_CHAR_BUDGET','[unset]')}")
    rows.append(f"PGOPTIONS: {g('PGOPTIONS','[unset]')}")
    rows.append("")
    # Include effective retriever settings with parsed fields
    try:
        eff = effective_settings()
        rows.append("### Effective retriever settings")
        for kname in [
            "FAST_MODE",
            "RERANK_TOP_N",
            "RETRIEVE_SOURCE_ALLOWLIST",
            "parsed_allowlist",
            "RETRIEVE_BACKOFF_ENABLE",
            "RETRIEVE_BACKOFF_THRESHOLD",
            "RETRIEVE_BACKOFF_SECONDARY",
            "parsed_backoff_secondary",
            "PGOPTIONS",
        ]:
            rows.append(f"{kname}: {eff.get(kname, '[unset]')}")
        rows.append("")
    except Exception:
        pass
    passed = 0
    latencies_ms: List[float] = []

    for idx, it in enumerate(items, 1):
        rows.append(f"## Q{idx}")
        rows.append(it.question)
        if it.expected_hint:
            rows.append(f"Expected: {it.expected_hint}")
        rows.append("")
        try:
            t0 = time.perf_counter()
            ans, cits = answer_query(it.question, top_k=args.k)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            urls = [c.url for c in cits]
            ok, rule = match_pass(it.expected_hint, urls)
            if ok:
                passed += 1
            rows.append("Answer:")
            rows.append(ans)
            rows.append("")
            rows.append("Citations:")
            for i, u in enumerate(urls, 1):
                rows.append(f"- [{i}] {u}")
            rows.append("")
            rows.append(f"Latency: {elapsed_ms:.0f} ms")
            rows.append("")
            rows.append(f"Result: {'PASS' if ok else 'FAIL'} ({rule})")
        except Exception as e:
            # Log and continue
            logger.error(f"Error evaluating Q{idx}: {e}")
            rows.append("Answer:")
            rows.append(f"[ERROR] {e}")
            rows.append("")
            rows.append("Citations:")
            rows.append("- [none]")
            rows.append("")
            rows.append("Result: ERROR")
        rows.append("")

    total = len(items)
    acc = (passed / total * 100.0) if total else 0.0
    if latencies_ms:
        avg_ms = statistics.mean(latencies_ms)
        p50_ms = statistics.median(latencies_ms)
        p90_ms = sorted(latencies_ms)[max(0, int(0.9 * len(latencies_ms)) - 1)]
        rows.insert(0, f"Latency: avg={avg_ms:.0f} ms, p50={p50_ms:.0f} ms, p90={p90_ms:.0f} ms")
        rows.insert(0, "")
    rows.insert(0, f"Accuracy: {passed}/{total} ({acc:.1f}%)")
    rows.insert(0, "")

    out_path.write_text("\n".join(rows), encoding="utf-8")
    logger.info(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
