from __future__ import annotations

import sys
from pathlib import Path
import os
import argparse


# Ensure local 'src' and 'scripts' are importable, mirroring other scripts
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (SRC, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Load .env (reuse the same precedence as other scripts)
try:
    import dotenv  # type: ignore

    for env_path in [ROOT / "windsurf-racen-local" / ".env", ROOT / ".env"]:
        if env_path.exists():
            dotenv.load_dotenv(dotenv_path=env_path, override=False)
            break
except Exception:
    pass

from step4_answer import answer_query  # type: ignore


def run_query(q: str, top_k: int = 10) -> None:
    """Run a single query through answer_query and print answer + citations.

    Args:
        q (str): User question to send through the RACEN pipeline.
        top_k (int): Number of chunks to retrieve.
    """
    print("==== QUERY ====")
    print(q)
    print("==============")
    answer, citations = answer_query(q, top_k=top_k)
    print("\n---- ANSWER ----")
    print(answer)
    print("\n---- CITATIONS ----")
    for c in citations:
        try:
            url = getattr(c, "url", "")
            start = getattr(c, "start_line", "")
            end = getattr(c, "end_line", "")
            print(f"- {url} ({start}-{end})")
        except Exception:
            print(str(c))
    print("-----------------\n")


def main() -> None:
    """Simple CLI for manual product debugging via answer_query.

    If no queries are provided, defaults to iPhone 11 and 13 product checks.
    """
    parser = argparse.ArgumentParser(description="Manual product debug CLI for RACEN")
    parser.add_argument(
        "--q",
        "--query",
        dest="queries",
        action="append",
        help="Query string to send. Can be repeated.",
    )
    parser.add_argument(
        "--k",
        dest="top_k",
        type=int,
        default=int(os.getenv("TOP_K", "10")),
        help="Top-k retrieval (defaults from TOP_K env or 10).",
    )

    args = parser.parse_args()
    queries = args.queries or [
        "do you have iphone 11?",
        "do you have iphone 13?",
    ]

    for q in queries:
        run_query(q, top_k=args.top_k)


if __name__ == "__main__":
    main()
