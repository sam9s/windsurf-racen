from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List


def _read_text_file(path_str: str) -> str:
    try:
        p = Path(path_str)
        if p.exists() and p.is_file():
            return p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        pass
    return ""

def _read_lexicon(path_str: str) -> dict:
    try:
        import yaml  # type: ignore

        p = Path(path_str)
        if p.exists() and p.is_file():
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        return {}
    return {}

def _detect_mode(text: str) -> str:
    t = (text or "").lower()
    # naive signals for Hinglish
    hinges = ["hai", "hain", "hoon", "sakti", "sakun", "karo", "kariye", "aap", "ji", "nahi", "haa"]
    if any(w in t for w in hinges):
        return "HI_EN"
    return "EN"

def _shape_first_paragraph(answer: str, mode: str, lex: dict) -> str:
    if not answer or not lex:
        return answer
    parts = answer.split("\n\n", 1)
    first = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    modes = (lex.get("modes") or {})
    cfg = modes.get(mode) or {}
    reps = cfg.get("replacements") or {}
    forbidden = cfg.get("forbidden") or []
    # apply replacements (simple, whole-phrase case-insensitive)
    for k, v in reps.items():
        try:
            first = re.sub(rf"\b{re.escape(k)}\b", v, first, flags=re.IGNORECASE)
        except re.error:
            # fallback literal replace
            first = first.replace(k, v)
    # remove forbidden phrases
    for phrase in forbidden:
        first = first.replace(phrase, "")
    shaped = first.strip()
    if rest:
        return shaped + "\n\n" + rest
    return shaped
 

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


def _compose_prompt(query: str, chunks: List[RetrievedChunk], intent: str = "", previous_answer: str = "") -> str:
    lines: List[str] = []
    # Persona: prepend system prompt if provided
    persona_path = os.getenv(
        "PERSONA_SYSTEM_PROMPT_PATH",
        str(
            ROOT
            / ".."
            / "Grest_RACEN_Slack_Bot"
            / "slack-openai-bot"
            / "Persona"
            / "system_prompt.md"
        ),
    )
    persona_text = _read_text_file(persona_path)
    if persona_text:
        lines.append("Persona:")
        lines.append(persona_text)
        lines.append("")
    short = os.getenv("ANSWER_SHORT", "0") in {"1", "true", "TRUE", "yes"}
    lines.append("You are a support assistant for GREST. Answer ONLY using the provided context.")
    lines.append("If the answer is not present in the context, reply: 'Not found in sources provided.'")
    match_lang = os.getenv("ANSWER_MATCH_INPUT_LANGUAGE", "0") in {"1", "true", "TRUE", "yes"}
    if match_lang:
        lines.append(
            "Mirror the user's language and style closely. Prefer the user's language; avoid mixing languages unless the user mixes them."
        )
        lines.append(
            "If the input is Hinglish (Hindi + English), respond in Hinglish. Use a female first-person voice (e.g., 'main madad kar sakti hoon')."
        )
    else:
        lines.append("Use a female first-person voice (e.g., 'main madad kar sakti hoon').")
    # Avoid trailing full-English sentences in Hinglish replies
    lines.append(
        "If replying in Hinglish, avoid appending a full English sentence at the end; keep tone consistent."
    )
    # Emoji tone control via env
    try:
        emoji_level = int(os.getenv("PERSONA_EMOJI_LEVEL", "0"))
    except Exception:
        emoji_level = 0
    if emoji_level <= 0:
        lines.append("Use no emojis unless the user uses them explicitly.")
    elif emoji_level == 1:
        lines.append("You may use up to 1 subtle emoji (e.g., 🙂, ✅) only when the topic is informal or cheerful; avoid in serious topics like refunds/escalations.")
    else:
        lines.append("You may use up to 2 subtle emojis when the tone is clearly informal; avoid emojis in serious topics like complaints, denials, or escalations.")
    # Instruct the model to propose helpful follow-ups when enabled
    followups_on = os.getenv("ANSWER_FOLLOWUPS_ENABLE", "1") in {"1", "true", "TRUE", "yes"}
    if followups_on:
        lines.append(
            "After answering, naturally suggest 1-2 next helpful things you can do, phrased conversationally as part of the reply (no headers or bullet lists)."
        )
        lines.append(
            "If contacting support may help, optionally offer to share the support phone/email (do not invent details)."
        )
    # Provide the previous assistant message to help the model interpret short acknowledgements
    if previous_answer:
        lines.append("If your previous reply offered to take an action (e.g., share support details) and the user's current message indicates consent/acknowledgement in any language, proceed and respond naturally.")
    if short:
        lines.append("Cite minimally like [1] and add a compact Citations list.")
    else:
        lines.append("Cite evidence inline like [1], [2] and provide a final Citations list.")
    # Support contact guardrails
    lines.append(
        "For any phone/email/contact details, NEVER invent names or numbers. Only use details present in the provided context or authoritative support facts."
    )
    lines.append(
        "If sharing support details, keep them concise and natural, and ensure they can be traced to on-site sources (e.g., /pages/contact-us)."
    )
    lines.append("")
    # Inject authoritative support facts from env when available
    support_phone = (os.getenv("SUPPORT_PHONE", "") or "").strip()
    support_email = (os.getenv("SUPPORT_EMAIL", "") or "").strip()
    support_address = (os.getenv("SUPPORT_ADDRESS", "") or "").strip()
    if any([support_phone, support_email, support_address]):
        lines.append("Authoritative support facts (from system configuration):")
        if support_phone:
            lines.append(f"- Support phone: {support_phone}")
        if support_email:
            lines.append(f"- Support email: {support_email}")
        if support_address:
            lines.append(f"- Support address: {support_address}")
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


def _detect_intent(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in ["return", "refund", "cancel", "exchange"]):
        return "returns"
    if "warranty" in q or "guarantee" in q:
        return "warranty"
    if "ship" in q or "delivery" in q or "international" in q:
        return "shipping"
    if "privacy" in q or "data" in q or "terms" in q:
        return "policy"
    if "contact" in q or "support" in q or "help" in q:
        return "contact"
    # order/buy intent (broad coverage for EN + Hinglish)
    buy_signals = [
        "order", "buy", "purchase", "checkout", "cart", "place order",
        "kharid", "kharidna", "order kaise", "kaise karun", "kaise karu", "order karu"
    ]
    if any(sig in q for sig in buy_signals):
        return "order_buy"
    return "general"


def _clean_snippet(text: str) -> str:
    t = text or ""
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\bhttps?://\S+", " ", t)
    # Remove tel: links and map-like query crumbs
    t = re.sub(r"tel:\S+", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(entry=ttu|g_ep=|hl=|utm_[a-z]+)=\S+", " ", t, flags=re.IGNORECASE)
    # Drop percent-encoded noise-heavy tokens
    t = re.sub(r"(?:%[0-9A-Fa-f]{2}){4,}", " ", t)
    lines = []
    for ln in t.splitlines():
        if any(k in ln for k in ["{", "}", ": ", ";", "position:", "margin", "padding", "width:", "height:"]):
            continue
        # Skip lines that are mostly URL-ish or numeric codes
        if len(re.findall(r"[/:%&=?]", ln)) > 3 or len(re.findall(r"\d{4,}", ln)) > 0:
            continue
        lines.append(ln)
    t = " ".join(lines)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > 300:
        t = t[:300].rstrip() + " …"
    return t

def _followups_for_intent(intent: str) -> List[str]:
    if intent == "returns":
        return [
            "Return window and eligibility?",
            "Refund processing steps?",
            "Can I exchange instead of return?",
        ]
    if intent == "warranty":
        return [
            "What is covered vs excluded?",
            "Warranty claim process?",
            "Warranty duration and proof needed?",
        ]
    if intent == "shipping":
        return [
            "Domestic vs international shipping times?",
            "Shipping charges and carriers?",
            "How do I track my order?",
        ]
    if intent == "policy":
        return [
            "Key points from privacy policy?",
            "Important terms customers should know?",
            "How to request data deletion?",
        ]
    if intent == "contact":
        return [
            "Support email and phone?",
            "Business hours?",
            "Escalation process?",
        ]
    return [
        "Show related FAQs?",
        "Where can I learn more?",
        "Talk to support?",
    ]


def answer_query(query: str, top_k: int = 6, previous_answer: str = "") -> tuple[str, List[Citation]]:
    # Detect user intent for conversational follow-ups
    intent = _detect_intent(query)

    # Retrieval with simple, intent-based augmentation (language-agnostic keywords)
    aug = ""
    if intent == "returns":
        aug = " return returns refund cancellation policy"
    elif intent == "warranty":
        aug = " warranty 6-month 6 month policy"
    elif intent == "shipping":
        aug = " shipping delivery timeline policy"
    elif intent == "policy":
        aug = " policy terms conditions"
    elif intent == "contact":
        aug = " contact support phone email"
    elif intent == "order_buy":
        aug = " order buy purchase checkout cart payment how to buy place order"
    aug_query = (query + aug).strip()

    # Retrieve
    items = retrieve(aug_query, top_k=top_k)
    if not items:
        # Best-effort: no retrieval, return empty with hint handled by caller
        return "Not found in sources provided.", []

    # Build citations list in the same order as chunks appear in prompt
    citations: List[Citation] = []
    for it in items:
        citations.append(Citation(url=it.source, start_line=it.start_line, end_line=it.end_line))

    # Compose and call LLM
    prompt = _compose_prompt(query, items, intent=intent, previous_answer=previous_answer)
    txt = _call_openai(prompt)

    # Apply best-effort fallback if enabled and the model could not find an answer
    fallback_on = os.getenv("ANSWER_FALLBACK_ENABLE", "1") in {"1", "true", "TRUE", "yes"}
    followups_on = os.getenv("ANSWER_FOLLOWUPS_ENABLE", "1") in {"1", "true", "TRUE", "yes"}

    def _build_fallback_text() -> str:
        blocks: List[str] = []
        blocks.append("I couldn’t find an exact answer in the provided sources. Here’s the closest relevant info:")
        for ch in items[:2]:
            snippet = _clean_snippet(ch.text)
            if snippet:
                blocks.append(snippet)
        return " " .join(blocks)

    out_text = txt
    if fallback_on and out_text.strip().lower().startswith("not found in sources provided"):
        out_text = _build_fallback_text()

    # Append a single, mode-aware follow-up line using lexicon snippet
    if followups_on:
        lower = out_text.lower()
        if "follow-ups:" not in lower and "follow ups:" not in lower:
            # Try to load lexicon snippet for offer_details
            lex_path = os.getenv(
                "PERSONA_LEXICON_PATH",
                str(
                    ROOT
                    / ".."
                    / "Grest_RACEN_Slack_Bot"
                    / "slack-openai-bot"
                    / "Persona"
                    / "lexicon.v1.yaml"
                ),
            )
            lexicon = _read_lexicon(lex_path) if lex_path else {}
            mode = _detect_mode(query)
            offer = None
            if lexicon:
                modes = lexicon.get("modes") or {}
                cfg = modes.get(mode) or {}
                snips = cfg.get("snippets") or {}
                offer = snips.get("offer_details")
            if offer:
                out_text = f"{out_text}\n\n{offer}"

    # Reply shaper using lexicon (first paragraph only)
    lex_path = os.getenv(
        "PERSONA_LEXICON_PATH",
        str(
            ROOT
            / ".."
            / "Grest_RACEN_Slack_Bot"
            / "slack-openai-bot"
            / "Persona"
            / "lexicon.v1.yaml"
        ),
    )
    lexicon = _read_lexicon(lex_path) if lex_path else {}
    if lexicon:
        mode = _detect_mode(query)
        out_text = _shape_first_paragraph(out_text, mode, lexicon)

    return out_text, citations


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
