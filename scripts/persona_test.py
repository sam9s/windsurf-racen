from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

# Ensure local 'src' is importable for step4_answer dependencies
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Load .env
try:
    import dotenv  # type: ignore

    for env_path in [ROOT / "windsurf-racen-local" / ".env", ROOT / ".env"]:
        if env_path.exists():
            dotenv.load_dotenv(dotenv_path=env_path, override=False)
            break
except Exception:
    pass

import yaml  # type: ignore
from scripts.step4_answer import answer_query  # noqa: E402


def read_tests_path() -> Path:
    default = (
        ROOT
        / ".."
        / "Grest_RACEN_Slack_Bot"
        / "slack-openai-bot"
        / "Persona"
        / "tone_unit_tests.v1.yaml"
    )
    p = Path(os.getenv("PERSONA_TONE_TESTS_PATH", str(default)))
    return p


def load_tests(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        return yaml.safe_load(f) or {}


def sentences_count(text: str) -> int:
    # naive sentence splitter
    import re as _re

    parts = _re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return len([s for s in parts if s])


def run_test_case(tc: Dict[str, Any]) -> Dict[str, Any]:
    name = tc.get("name", "")
    user_input = tc.get("input", "")
    must_include: List[str] = tc.get("must_include", []) or []
    max_sent = tc.get("max_sentences_first_reply")

    answer, _ = answer_query(user_input, top_k=int(os.getenv("TOP_K", "6")))

    ok = True
    failures: List[str] = []

    # First reply is the first paragraph (before blank line)
    first_para = answer.split("\n\n", 1)[0]

    for req in must_include:
        if req not in answer:
            ok = False
            failures.append(f"missing: {req}")

    if isinstance(max_sent, int):
        n = sentences_count(first_para)
        if n > max_sent:
            ok = False
            failures.append(f"sentence_cap: expected<= {max_sent}, got {n}")

    return {"name": name, "ok": ok, "failures": failures, "answer": first_para}


def main() -> None:
    p = read_tests_path()
    if not p.exists():
        print(f"Tests file not found: {p}")
        sys.exit(2)

    suite = load_tests(p)
    tests = suite.get("tests", []) or []
    results = [run_test_case(tc) for tc in tests]

    passed = sum(1 for r in results if r["ok"]) 
    total = len(results)

    summary = {
        "passed": passed,
        "total": total,
        "details": results,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
