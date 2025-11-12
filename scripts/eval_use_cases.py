from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure local 'src' is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env (prefer project local)
try:
    import dotenv  # type: ignore

    for env_path in [ROOT / "windsurf-racen-local" / ".env", ROOT / ".env"]:
        if env_path.exists():
            dotenv.load_dotenv(dotenv_path=env_path, override=False)
            break
except Exception:
    pass

from racen.log import get_logger
from racen.step3_retrieve import retrieve

logger = get_logger("scripts.eval_use_cases")


def parse_curated_md(md_path: Path) -> List[Tuple[str, str]]:
    """
    Parse curated eval queries from the MD file.

    Returns:
        List of (query, expected_substring) where expected_substring may be ''.
    """
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in text.splitlines()]

    # Collect list items under the curated section or any list items with 'expect:'
    out: List[Tuple[str, str]] = []
    for ln in lines:
        if not ln.startswith("-"):  # only consider bullets we added
            continue
        # Example: - Shipping ... expect: policies/shipping
        m = re.match(r"^-\s*(.+?)(?:\s+expect:\s*(.+))?$", ln)
        if not m:
            continue
        q = m.group(1).strip()
        exp = (m.group(2) or "").strip()
        out.append((q, exp))
    return out


def run_eval(cases: List[Tuple[str, str]], top_k: int = 3) -> None:
    total = 0
    hits = 0
    for q, exp in cases:
        total += 1
        res = retrieve(q, top_k=top_k)
        ok = False
        sources = []
        for r in res:
            sources.append(r.source)
            if exp and exp.lower() in r.source.lower():
                ok = True
        hit_str = "HIT" if ok else "MISS"
        if ok:
            hits += 1
        logger.info(f"Q: {q}")
        logger.info(f"Exp: {exp or '-'} | {hit_str}")
        for i, s in enumerate(sources, 1):
            logger.info(f"  #{i} {s}")
    if total:
        pct = hits / total
        logger.info(f"Summary: {hits}/{total} = {pct:.2%} hit@{top_k}")


def main() -> None:
    md = ROOT / "docs" / "Grest_Use_Cases.md"
    cases = parse_curated_md(md)
    if not cases:
        logger.info("No curated cases found in MD file")
        return
    run_eval(cases, top_k=3)


if __name__ == "__main__":
    main()
