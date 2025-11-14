from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

# Ensure local 'src' is importable
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

from racen.log import get_logger
from racen.step3_retrieve import retrieve, RetrievedChunk

logger = get_logger("scripts.step4_answer")


@dataclass
class Citation:
    url: str
    start_line: int
    end_line: int


def _compose_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    lines: List[str] = []
    short = os.getenv("ANSWER_SHORT", "0") in {"1", "true", "TRUE", "yes"}
    lines.append("You are a support assistant for GREST. Answer ONLY using the provided context.")
    lines.append("If the answer is not present in the context, reply: 'Not found in sources provided.'")
    match_lang = os.getenv("ANSWER_MATCH_INPUT_LANGUAGE", "0") in {"1", "true", "TRUE", "yes"}
    if match_lang:
        lines.append("Mirror the user's language and style. If the input is Hinglish (Hindi + English), respond in Hinglish.")
    if short:
        lines.append("Cite minimally like [1] and add a compact Citations list.")
    else:
        lines.append("Cite evidence inline like [1], [2] and provide a final Citations list.")
    lines.append("")
    lines.append("Question:")
    lines.append(query)
    lines.append("")
    lines.append("Context:")
    # Cap per-chunk context to keep prompt small in short mode
    try:
        char_budget = int(os.getenv("ANSWER_CHUNK_CHAR_BUDGET", "0"))
    except Exception:
        char_budget = 0
    for idx, ch in enumerate(chunks, 1):
        lines.append(f"[{idx}] Source: {ch.source} (lines {ch.start_line}-{ch.end_line})")
        # Keep context blocks short to fit model limits, but we already chunked
        if char_budget and char_budget > 0:
            lines.append(ch.text[:char_budget])
        else:
            lines.append(ch.text)
        lines.append("")
    lines.append("Instructions:")
    if short:
        lines.append("- Provide a short answer (2-4 sentences) with necessary citations like [1].")
        lines.append("- Include a compact 'Citations' section listing [n] with URL and line range.")
    else:
        lines.append("- Provide a concise answer (3-6 sentences).")
        lines.append("- After each claim, add citations like [1], [2] from the Context indices.")
        lines.append("- Include a final 'Citations' section listing each [n] with URL and line range.")
    lines.append("- Do NOT use any external knowledge beyond the provided context.")
    return "\n".join(lines)


def _call_openai(prompt: str, max_retries: int = 3, model: str = "gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        return "Not found in sources provided. [No API key configured]"
    import requests  # type: ignore

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    short = os.getenv("ANSWER_SHORT", "0") in {"1", "true", "TRUE", "yes"}
    try:
        max_tokens = int(os.getenv("ANSWER_MAX_TOKENS", "120" if short else "180"))
    except Exception:
        max_tokens = 120 if short else 180
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=45)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            return content
        except Exception as e:
            if attempt == max_retries:
                return f"Not found in sources provided. [LLM error: {e}]"
            time.sleep(1.5 * attempt)
    return "Not found in sources provided."


def answer_query(query: str, top_k: int = 6) -> tuple[str, List[Citation]]:
    # Retrieve
    items = retrieve(query, top_k=top_k)
    if not items:
        return "Not found in sources provided.", []

    # Build citations list in the same order as chunks appear in prompt
    citations: List[Citation] = []
    for it in items:
        citations.append(Citation(url=it.source, start_line=it.start_line, end_line=it.end_line))

    # Compose and call LLM
    prompt = _compose_prompt(query, items)
    txt = _call_openai(prompt)
    return txt, citations


def main() -> None:
    parser = argparse.ArgumentParser(description="Answer a question with citations from the 5-page curated corpus")
    parser.add_argument("--q", required=True, help="Question text")
    parser.add_argument("--k", type=int, default=6, help="Top-k chunks to use")
    args = parser.parse_args()

    ans, cits = answer_query(args.q, top_k=args.k)
    print("=== Answer ===")
    print(ans)
    print("\n=== Citations ===")
    for i, c in enumerate(cits, 1):
        print(f"[{i}] {c.url} (lines {c.start_line}-{c.end_line})")


if __name__ == "__main__":
    main()
