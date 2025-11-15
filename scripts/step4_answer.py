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
    hinges = [
        # core verbs / pronouns
        "hai", "hain", "hun", "hoon", "mein", "mai", "main", "aap", "ji", "nahi", "haa",
        # ask/intent words
        "kaise", "kese", "kya", "kyu", "kyun", "chahiye", "batao",
        # action variants (spellings)
        "kar", "karo", "kariye", "ker", "kr", "krna", "karna",
        # ability/tense variants
        "sakta", "sakte", "sakti", "sakun", "raha", "rha",
    ]
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


def _limit_first_bubble(text: str, max_sent: int = 2) -> str:
    """Limit first bubble to a small number of sentences without breaking addresses.

    Respects env ANSWER_LIMIT_FIRST_BUBBLE; when not enabled, returns text unchanged.
    Avoids splitting on common abbreviations like 'No.', 'Pvt.', 'Ltd.', 'Dr.', 'St.'.
    """
    if os.getenv("ANSWER_LIMIT_FIRST_BUBBLE", "0") not in {"1", "true", "TRUE", "yes"}:
        return text or ""
    t = text or ""
    parts = t.split("\n\n", 1)
    first = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    import re as _re
    exceptions = {"no.", "pvt.", "ltd.", "dr.", "st.", "mr.", "ms.", "mrs.", "mt.", "rd.", "fl."}
    tokens = _re.split(r"(\s+)", first)
    out = []
    sent_count = 0
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        out.append(tok)
        if tok and tok.strip().endswith(('.', '!', '?')):
            prev_word = ''
            # find previous non-space token's last word
            j = i
            while j > 0:
                w = tokens[j-1].strip()
                if w:
                    prev_word = w.lower()
                    break
                j -= 1
            if prev_word not in exceptions:
                sent_count += 1
                if sent_count >= max_sent:
                    # truncate remainder of first paragraph
                    break
        i += 1
    limited_first = ''.join(out).strip()
    return limited_first if not rest else f"{limited_first}\n\n{rest}"


def _infer_last_intent(prev_ans: str) -> str:
    p = (prev_ans or "").lower()
    if any(k in p for k in ["refund", "cancel", "return", "exchange"]):
        return "returns"
    if any(k in p for k in ["warranty", "guarantee"]):
        return "warranty"
    if any(k in p for k in ["ship", "delivery", "tracking"]):
        return "shipping"
    if any(k in p for k in ["address", "phone", "email", "contact", "head office", "location"]):
        return "contact"
    if any(k in p for k in ["order", "buy", "purchase", "checkout", "payment"]):
        return "order_buy"
    return "general"


def _classify_ack(prev_assistant: str, user_msg: str) -> str:
    prev = (prev_assistant or "").strip()
    usr = (user_msg or "").strip()
    # Heuristic fast path when LLM errors or keys missing
    def heuristic() -> str:
        u = usr.lower()
        ack_signals = [
            "yes", "yeah", "yep", "ok", "okay", "please share", "share", "more info", "more details",
            "haan", "han ji", "haan ji", "theek hai", "kariye", "kar dijiye", "kar do"
        ]
        new_topic_signals = ["warranty", "refund", "returns", "shipping", "price", "warranty kitna"]
        if any(sig in u for sig in ack_signals) and not any(sig in u for sig in new_topic_signals):
            return "ACK_CONTINUE"
        return "NEW_TOPIC"

    try:
        # Use the same OpenAI endpoint as _call_openai with a tiny prompt
        prompt = (
            "Classification: ACK_CONTINUE or NEW_TOPIC.\n\n"  # keep tiny
            f"Previous assistant: {prev}\n"
            f"User: {usr}\n\n"
            "Examples:\n"
            "Q: \"where is grest head office?\"\nA: \"address\"\nU: \"yes please more info\"\n-> ACK_CONTINUE\n\n"
            "U: \"ok and your return policy?\"\n-> NEW_TOPIC\n\n"
            "Now classify only with the token:"
        )
        # Minimal body for classification
        import requests as _rq  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not api_key:
            return heuristic()
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body = {
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0,
            "max_tokens": 2,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = _rq.post(url, headers=headers, json=body, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        label = (data["choices"][0]["message"]["content"] or "").strip()
        return "ACK_CONTINUE" if "ACK_CONTINUE" in label else ("NEW_TOPIC" if "NEW_TOPIC" in label else heuristic())
    except Exception:
        return heuristic()


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

# Module-level debug snapshot for the API to read
_LAST_DEBUG: str = ""


# Facet bundles per intent to support "more details" drill-down without rigid flows
# Reason: Keeps LLM-led conversation while ensuring the right evidence is available when users ask to expand.
FACET_BUNDLES = {
    "contact": {
        "default_facet": "contact_basics",
        "facet_keywords": ["contact", "phone", "email", "business hours", "address"],
        "facet_allowlist": ["/pages/contact-us", "/pages/faqs"],
    },
    "returns": {
        "default_facet": "refund_timeline",
        "facet_keywords": ["refund timeline", "cancel steps", "return window", "exchange"],
        "facet_allowlist": ["/pages/returns-refund-cancellation"],
    },
    "warranty": {
        "default_facet": "coverage_claim",
        "facet_keywords": ["coverage", "exclusions", "claim process", "duration"],
        "facet_allowlist": ["/pages/warranty", "/pages/faqs"],
    },
    "shipping": {
        "default_facet": "timelines_charges",
        "facet_keywords": ["shipping timeline", "charges", "tracking"],
        "facet_allowlist": ["/policies/shipping/policy", "/pages/faqs"],
    },
    "order_buy": {
        "default_facet": "payment_checkout",
        "facet_keywords": ["payment options", "COD", "EMI", "BNPL", "how to order"],
        "facet_allowlist": ["/pages/faqs"],
    },
}


@dataclass
class Citation:
    url: str
    start_line: int
    end_line: int


def _compose_prompt(
    query: str,
    chunks: List[RetrievedChunk],
    intent: str = "",
    previous_answer: str = "",
    previous_user: str = "",
) -> str:
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
    lang_lock_on = os.getenv("ANSWER_LANGUAGE_LOCK", "0") in {"1", "true", "TRUE", "yes"}
    if match_lang:
        lines.append(
            "Mirror the user's language and style closely. Prefer the user's language; avoid mixing languages unless the user mixes them."
        )
        lines.append(
            "If the input is Hinglish (Hindi + English), respond in Hinglish. Use a female first-person voice (e.g., 'main madad kar sakti hoon')."
        )
        if previous_user:
            # Nudge model to stick to the last user message language
            lines.append(
                "Do not switch languages unless the user's most recent message switched languages."
            )
    # Strong lock: when enabled, strictly pin output language to the last user message
    if lang_lock_on:
        _mode_for_lock = _detect_mode(previous_user or query)
        if _mode_for_lock == "HI_EN":
            lines.append("IMPORTANT: Answer strictly in Hinglish (Hindi + English). Do not switch languages.")
        else:
            lines.append("IMPORTANT: Answer strictly in English. Do not switch languages.")
    else:
        lines.append("Use a female first-person voice (e.g., 'main madad kar sakti hoon').")
    # Avoid trailing full-English sentences in Hinglish replies
    lines.append(
        "If replying in Hinglish, avoid appending a full English sentence at the end; keep tone consistent."
    )
    # Friendly-first tone; switch to formal/empathetic on complaints or upset tone
    lines.append("Default to a friendly, conversational tone; avoid corporate phrases; be concise and direct. If the user sounds upset, complaining, or escalates, switch to a calm, professional, empathetic tone and avoid emojis.")
    # When user asks for more details after your previous answer, expand the most relevant facet
    lines.append(
        "If the user asks for 'more details' or acknowledges to continue based on your last answer, expand the most relevant facet (e.g., contact details, refund timeline). If multiple facets could apply, briefly suggest 1–2 options and proceed with the most likely unless the user specifies."
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
        lines.append("Do not include inline numeric citation markers like [1] or [2].")
        lines.append("Do not include a 'Citations' section in your text; the caller will attach citations separately.")
    else:
        lines.append("Do not include inline numeric citation markers like [1] or [2].")
        lines.append("Do not include a 'Citations' section in your text; the caller will attach citations separately.")
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
        lines.append("- Provide a short answer (2-4 sentences).")
        lines.append("- Do not add inline [n] markers or a 'Citations' section; the caller will attach citations separately.")
    else:
        lines.append("- Provide a concise answer (3-6 sentences).")
        lines.append("- Do not add inline [n] markers or a 'Citations' section; the caller will attach citations separately.")
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
    address_signals = [
        "address", "head office", "headoffice", "hq", "headquarter", "headquarters",
        "location", "where is office", "office where", "where is your office"
    ]
    if "contact" in q or "support" in q or "help" in q or any(s in q for s in address_signals):
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
    # Remove CSS/JS-style block comments
    t = re.sub(r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/", " ", t, flags=re.DOTALL)
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

def _strip_inline_citations(text: str) -> str:
    t = text or ""
    # remove inline numeric citation markers like [1], [12]
    t = re.sub(r"\s*\[\d+\]", "", t)
    # remove trivial 'Citations' header lines the model might add
    t = re.sub(r"\n+\s*Citations\s*:\s*\n?", "\n", t, flags=re.IGNORECASE)
    return t.strip()

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


def answer_query(
    query: str,
    top_k: int = 6,
    previous_answer: str = "",
    previous_user: str = "",
) -> tuple[str, List[Citation]]:
    # Detect user intent for conversational follow-ups
    intent = _detect_intent(query)
    last_intent = _infer_last_intent(previous_answer)

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

    # Retrieve (with optional per-intent allowlist boost and facet expansion)
    original_allow = os.getenv("RETRIEVE_SOURCE_ALLOWLIST", "")
    def _ensure_in_allowlist(pattern: str) -> None:
        prim = [p for p in (s.strip() for s in original_allow.split(",")) if p]
        if pattern not in prim:
            os.environ["RETRIEVE_SOURCE_ALLOWLIST"] = ",".join(prim + [pattern])
        else:
            os.environ["RETRIEVE_SOURCE_ALLOWLIST"] = original_allow
    
    # Detect acknowledgement consent (LLM classifier with heuristic fallback)
    ql = (query or "").lower()
    prevl = (previous_answer or "").lower()
    ack_label = _classify_ack(previous_answer, query) if previous_answer else "NEW_TOPIC"
    ack = ack_label == "ACK_CONTINUE"
    prev_offered = any(tok in prevl for tok in ["support", "phone", "email", "contact"])
    # Detect generic "more details" request
    more_details = any(tok in ql for tok in ["more details", "details", "zyada details", "details chahiye"]) and len(ql) <= 60
    # Effective intent for acknowledgements
    effective_intent = intent
    if ack and intent == "general" and last_intent != "general":
        effective_intent = last_intent

    # If user is acknowledging/asking for more details, decide facet by last intent
    facet_cfg = FACET_BUNDLES.get(effective_intent) or {}
    if (ack or more_details) and facet_cfg:
        # Allowlist expansion for facet
        for pat in (facet_cfg.get("facet_allowlist") or []):
            _ensure_in_allowlist(pat)
        # Keyword expansion for facet
        facet_kw = " ".join(facet_cfg.get("facet_keywords") or [])
        if facet_kw:
            aug_query = f"{aug_query} {facet_kw}".strip()
    try:
        if effective_intent == "returns":
            # Temporarily bias retrieval to include the canonical cancellation route
            _ensure_in_allowlist("/pages/returns-refund-cancellation")
        elif effective_intent == "shipping":
            # Bias retrieval to include the canonical shipping policy page
            _ensure_in_allowlist("/policies/shipping/policy")
        # If user acknowledged and previous answer offered sharing support details, include contact page
        if ack and prev_offered:
            _ensure_in_allowlist("/pages/contact-us")
        # Retrieve
        items = retrieve(aug_query, top_k=top_k)
    finally:
        # Restore allowlist regardless of errors
        os.environ["RETRIEVE_SOURCE_ALLOWLIST"] = original_allow
    if not items:
        # Best-effort: no retrieval, return empty with hint handled by caller
        return "Not found in sources provided.", []

    # Build citations list in the same order as chunks appear in prompt
    citations: List[Citation] = []
    for it in items:
        citations.append(Citation(url=it.source, start_line=it.start_line, end_line=it.end_line))

    # Compose and call LLM
    prompt = _compose_prompt(
        query,
        items,
        intent=effective_intent,
        previous_answer=previous_answer,
        previous_user=previous_user,
    )
    txt = _call_openai(prompt)

    # Apply best-effort fallback if enabled and the model could not find an answer
    fallback_on = os.getenv("ANSWER_FALLBACK_ENABLE", "1") in {"1", "true", "TRUE", "yes"}
    followups_on = os.getenv("ANSWER_FOLLOWUPS_ENABLE", "1") in {"1", "true", "TRUE", "yes"}

    def _build_fallback_text() -> str:
        mode = _detect_mode(query)
        intro = (
            "Exact info nahi mila, par yeh closest details hain:" if mode == "HI_EN" else
            "I couldn’t find the exact info, here’s the closest helpful detail:"
        )
        pieces: List[str] = [intro]
        for ch in items[:2]:
            snippet = _clean_snippet(ch.text)
            if snippet:
                pieces.append(snippet)
        # Suggest 2 concrete next options from facet/intent
        opts_map = {
            "contact": ["Phone number", "Email", "Contact link"],
            "returns": ["Cancel steps", "Refund timeline"],
            "warranty": ["Coverage", "Claim process"],
            "shipping": ["Delivery timelines", "Charges"],
            "order_buy": ["Payment options", "How to order"],
            "general": ["Policy link", "Details"]
        }
        opts = opts_map.get(intent) or opts_map["general"]
        if mode == "HI_EN":
            ask = f"Kya main {opts[0]} ya {opts[1]} share karun? Ya aap bata dein kis cheez ki details chahiye, main help kar dungi."
        else:
            ask = f"Want me to share {opts[0]} or {opts[1]}? Or tell me what you need and I’ll help."
        pieces.append(ask)
        return " ".join(pieces)

    out_text = _strip_inline_citations(txt)
    if fallback_on and out_text.strip().lower().startswith("not found in sources provided"):
        out_text = _build_fallback_text()

    # Deterministic contact drill-down: if user asked to continue/more details for contact/address
    if effective_intent == "contact" and (ack or more_details):
        # Prefer authoritative env facts when available
        support_phone = (os.getenv("SUPPORT_PHONE", "") or "").strip()
        support_email = (os.getenv("SUPPORT_EMAIL", "") or "").strip()
        contact_link = "https://grest.in/pages/contact-us"
        mode = _detect_mode(previous_user or query)
        if support_phone or support_email:
            if mode == "HI_EN":
                parts: List[str] = ["Yeh contact details hain:"]
                if support_phone:
                    parts.append(f"Phone: {support_phone}")
                if support_email:
                    parts.append(f"Email: {support_email}")
                parts.append(f"Link: {contact_link}")
                out_text = ". ".join(parts)
            else:
                parts2: List[str] = ["Here are the contact basics:"]
                if support_phone:
                    parts2.append(f"Phone: {support_phone}")
                if support_email:
                    parts2.append(f"Email: {support_email}")
                parts2.append(f"Contact link: {contact_link}")
                out_text = ". ".join(parts2)

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
                # Optionally inline the follow-up into the first paragraph for reliable Slack display
                try:
                    emoji_level = int(os.getenv("PERSONA_EMOJI_LEVEL", "0"))
                except Exception:
                    emoji_level = 0
                suffix = ""
                if emoji_level > 0 and not any(e in offer for e in ["🙂", "✅", "😊", "😉"]):
                    suffix = " 🙂"
                if emoji_level > 0:
                    parts = out_text.split("\n\n", 1)
                    first = parts[0]
                    rest = parts[1] if len(parts) > 1 else ""
                    # join inline to avoid hidden second paragraph in Slack
                    first_inline = f"{first}  {offer}{suffix}"
                    out_text = first_inline if not rest else f"{first_inline}\n\n{rest}"
                else:
                    # Keep previous behavior (as a second paragraph) when emoji_level == 0
                    out_text = f"{out_text}\n\n{offer}{suffix}"

    # Enforce first bubble length before lexicon shaping (gated by env)
    out_text = _limit_first_bubble(out_text, max_sent=2)
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

    # Snapshot debug info for ribbon when enabled
    try:
        top_score = items[0].score if items else 0.0
    except Exception:
        top_score = 0.0
    fallback_used = out_text.startswith("Exact info nahi mila") or out_text.startswith("I couldn’t find the exact info")
    global _LAST_DEBUG
    _LAST_DEBUG = (
        f"intent={intent} | last_intent={last_intent} | eff_intent={effective_intent} | ack={int(ack)} | more_details={int(more_details)} | top_score={top_score:.2f} | fallback={int(fallback_used)}"
    )

    return out_text, citations


def get_last_debug_summary() -> str:
    return _LAST_DEBUG


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
