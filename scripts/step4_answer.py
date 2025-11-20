from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse


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


def _load_product_families() -> dict:
    """Load product families from a YAML config for domain-specific nouns.

    The file domain_product_families.yaml should define:
    families:
      - name: iphone
        keywords: ["iphone", "iphones"]
      - name: macbook
        keywords: ["macbook", "mac book", "macbooks", "mackbooks"]
    """
    # Compute project root locally to avoid relying on module-level ROOT ordering
    project_root = Path(__file__).resolve().parents[1]
    cfg_path = project_root / "domain_product_families.yaml"
    data = _read_lexicon(str(cfg_path)) if cfg_path.exists() else {}
    fams = data.get("families") or []
    out: dict[str, list[str]] = {}
    for fam in fams:
        try:
            name = (fam.get("name") or "").strip().lower()
            kws = [str(k).strip().lower() for k in (fam.get("keywords") or []) if str(k).strip()]
            if name and kws:
                out[name] = kws
        except Exception:
            continue
    return out


PRODUCT_FAMILIES = _load_product_families()

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
    # Product-like answers: device families and product pages
    if any(k in p for k in ["macbook", "mac book", "iphone", "ipad", "laptop", "product page"]):
        return "product"
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
        # If user mentions a different product family than the previous answer, treat as new topic
        try:
            pl = prev.lower()
            # Collect product family keywords from config + generic nouns
            cfg_nouns = [kw for fam in PRODUCT_FAMILIES.values() for kw in fam]
            generic_nouns = [
                "macbook",
                "mac book",
                "mackbook",
                "iphone",
                "ipad",
                "laptop",
                "notebook",
                "phone",
                "mobile",
            ]
            all_nouns = list(dict.fromkeys(cfg_nouns + generic_nouns))
            prev_fams = {n for n in all_nouns if n in pl}
            user_fams = {n for n in all_nouns if n in u}
            if prev_fams and user_fams and prev_fams.isdisjoint(user_fams):
                return "NEW_TOPIC"
        except Exception:
            pass
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
from racen.step2_write import DBConfig, get_conn
from racen.step3_retrieve import retrieve, RetrievedChunk
from racen.product_search import MatchType, ProductCandidate, ProductSearchResult, product_search
from racen.product_specs import ProductSpecs, extract_product_specs

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


@dataclass
class ProductAnswerPlan:
    """Deterministic plan for how to answer a product query.

    Args:
        requested_text: Original user query text.
        match_type: Catalog match type (EXACT, CLOSE_BUT_DIFFERENT, NONE).
        primary: Primary product to talk about in the answer, or None.
        siblings: Same-base sibling products (e.g., other variants) to optionally mention.
    """

    requested_text: str
    match_type: MatchType
    primary: Optional[ProductCandidate]
    siblings: List[ProductCandidate]


def _load_product_specs_for_candidates(match: Optional[ProductSearchResult]) -> Dict[str, ProductSpecs]:
    """Load structured specs for catalog candidates directly from the corpus.

    Args:
        match: ProductSearchResult from product_search, or None.

    Returns:
        Dict[str, ProductSpecs]: Mapping from candidate URL handle to extracted specs.
    """

    specs_by_url: Dict[str, ProductSpecs] = {}
    if match is None or not match.candidates:
        return specs_by_url

    try:
        conn = get_conn(DBConfig.from_env())
    except Exception:
        return specs_by_url

    try:
        for cand in match.candidates:
            url = (cand.url or "").strip()
            if not url or url in specs_by_url:
                continue
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT c.text
                        FROM chunks c
                        JOIN documents d ON c.document_id = d.id
                        WHERE d.source LIKE %s
                        ORDER BY c.start_line
                        """,
                        (f"{url}%",),
                    )
                    rows = cur.fetchall()
            except Exception:
                continue
            if not rows:
                continue
            full_text_parts: List[str] = []
            for row in rows:
                try:
                    part = row.get("text") or ""
                except Exception:
                    part = ""
                if part:
                    full_text_parts.append(str(part))
            if not full_text_parts:
                continue
            full_text = "\n\n".join(full_text_parts)
            try:
                specs = extract_product_specs(full_text)
            except Exception:
                continue
            if (
                specs.price_strings
                or specs.storage_options
                or specs.conditions
                or specs.warranty_strings
                or specs.color_options
            ):
                specs_by_url[url] = specs
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return specs_by_url


def build_product_answer_plan(query: str, match: Optional[ProductSearchResult]) -> ProductAnswerPlan:
    """Build a deterministic answer plan from catalog match results.

    This function encodes the high-level behavior for base vs variant queries
    so that downstream prompt logic does not have to guess. It does not touch
    retrieval or phrasing; it only decides which products are primary vs
    siblings and how to label availability.

    Args:
        query: Raw user query text.
        match: ProductSearchResult from product_search, or None.

    Returns:
        ProductAnswerPlan: Structured plan with primary product and siblings.
    """

    requested = (query or "").strip()
    if match is None or not match.candidates:
        return ProductAnswerPlan(
            requested_text=requested,
            match_type="NONE",
            primary=None,
            siblings=[],
        )

    # Normalise match_type to one of the known literals.
    mt: MatchType
    if match.match_type in {"EXACT", "CLOSE_BUT_DIFFERENT", "NONE"}:
        mt = match.match_type  # type: ignore[assignment]
    else:
        mt = "NONE"

    candidates = list(match.candidates)
    primary: Optional[ProductCandidate] = None
    siblings: List[ProductCandidate] = []

    if candidates:
        primary = candidates[0]
        if len(candidates) > 1:
            siblings = candidates[1:]

    return ProductAnswerPlan(
        requested_text=requested,
        match_type=mt,
        primary=primary,
        siblings=siblings,
    )


def _compose_prompt(
    query: str,
    chunks: List[RetrievedChunk],
    intent: str = "",
    previous_answer: str = "",
    previous_user: str = "",
    product_match: Optional[ProductSearchResult] = None,
    product_plan: Optional[ProductAnswerPlan] = None,
    product_specs_by_url: Optional[Dict[str, ProductSpecs]] = None,
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
    # Inject a small internal catalog match summary for product intents so
    # the model understands whether the requested variant exists or is only
    # close to a catalog item, and which URL is authoritative.
    if (intent or "").lower() == "product" and product_match is not None:
        if product_match.match_type == "EXACT" and product_match.candidates:
            cand = product_match.candidates[0]
            lines.append(
                "Internal catalog match: exact product found in catalog. "
                f"Base model '{cand.base_model}', variants '{' '.join(cand.variant_tokens)}', "
                f"name '{cand.name}', URL '{cand.url}'. Always use this URL as the product page when answering."
            )
        elif product_match.match_type == "CLOSE_BUT_DIFFERENT" and product_match.candidates:
            cand = product_match.candidates[0]
            lines.append(
                "Internal catalog match: requested variant not found in catalog. "
                f"Closest available product is a different variant: name '{cand.name}', "
                f"base model '{cand.base_model}', variants '{' '.join(cand.variant_tokens)}', URL '{cand.url}'. "
                "Clearly state that the requested variant is not available and that you are describing this closest available product instead."
            )
        elif product_match.match_type == "NONE":
            lines.append(
                "Internal catalog match: no product found in catalog for this model. "
                "Be explicit that this product is not available in the catalog and avoid suggesting a specific product URL."
            )
        lines.append("")

    # Provide a deterministic view of the primary product and its sibling
    # variants from the catalog so the model does not have to infer this
    # structure from context alone.
    if (intent or "").lower() == "product" and product_plan is not None:
        lines.append("Internal product answer plan (do not show to user):")
        lines.append(f"- Requested text: {product_plan.requested_text}")
        lines.append(f"- Catalog match type: {product_plan.match_type}")
        if product_plan.primary is not None:
            p = product_plan.primary
            lines.append(
                "- Primary product: "
                f"name='{p.name}', base_model='{p.base_model}', "
                f"variants='{', '.join(p.variant_tokens)}', url='{p.url}'"
            )
        if product_plan.siblings:
            lines.append("- Sibling variants for the same base model:")
            for sib in product_plan.siblings:
                lines.append(
                    f"  * name='{sib.name}', variants='{', '.join(sib.variant_tokens)}', url='{sib.url}'"
                )
        lines.append("")

    if (intent or "").lower() == "product" and product_specs_by_url:
        lines.append("Internal product specs (for model use only, do not show this section to the user):")

        def _append_specs(label: str, cand: ProductCandidate) -> None:
            sp = product_specs_by_url.get(cand.url)
            if sp is None:
                return
            if not (
                sp.price_strings
                or sp.storage_options
                or sp.conditions
                or sp.warranty_strings
                or sp.color_options
            ):
                return
            lines.append(f"- {label} specs for '{cand.name}' (url='{cand.url}'):")
            if sp.price_strings:
                lines.append(f"  * prices: {', '.join(sp.price_strings)}")
            if sp.storage_options:
                lines.append(f"  * storage_options: {', '.join(sp.storage_options)}")
            if sp.conditions:
                lines.append(f"  * conditions: {', '.join(sp.conditions)}")
            if sp.warranty_strings:
                lines.append(f"  * warranty: {', '.join(sp.warranty_strings)}")
            if sp.color_options:
                lines.append(f"  * colors: {', '.join(sp.color_options)}")

        if product_plan is not None:
            if product_plan.primary is not None:
                _append_specs("Primary", product_plan.primary)
            if product_plan.siblings:
                for sib in product_plan.siblings:
                    _append_specs("Sibling", sib)
        else:
            for url, sp in product_specs_by_url.items():
                if not (
                    sp.price_strings
                    or sp.storage_options
                    or sp.conditions
                    or sp.warranty_strings
                    or sp.color_options
                ):
                    continue
                lines.append(f"- Specs for product url='{url}':")
                if sp.price_strings:
                    lines.append(f"  * prices: {', '.join(sp.price_strings)}")
                if sp.storage_options:
                    lines.append(f"  * storage_options: {', '.join(sp.storage_options)}")
                if sp.conditions:
                    lines.append(f"  * conditions: {', '.join(sp.conditions)}")
                if sp.warranty_strings:
                    lines.append(f"  * warranty: {', '.join(sp.warranty_strings)}")
                if sp.color_options:
                    lines.append(f"  * colors: {', '.join(sp.color_options)}")
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
    # Product intent: shape answer towards product summary with key specs and link (all grounded)
    if (intent or "").lower() == "product":
        lines.append(
            "- If the question is about a product, first clearly confirm availability in 1 short sentence, "
            "then provide 3-6 Markdown bullet points for core specs using this pattern when data is present: "
            "'- **Storage Options**: ...', '- **Condition**: ...', '- **Price**: ...', '- **Warranty**: ...', "
            "'- **Battery Health**: ...', '- **Colors**: ...'."
        )
        lines.append("- Include the product page link once. Use only URLs that appear either in the 'Context:' section headers or in the Internal catalog match summary above. Do not invent or guess new URLs or paths.")
        lines.append("- Pay close attention to qualifiers in the user's question like 'retina', year (e.g., 2015), screen size (e.g., 13-inch), storage or RAM. Prefer a product whose title/description matches these qualifiers in the provided context.")
        lines.append("- If at least one product title or description clearly contains the requested model name or variant from the question, answer about that product using only the provided context and/or the Internal catalog match summary.")
        lines.append("- When product specs are listed in the 'Internal product specs' section above, use only those extracted values for price, storage, condition, warranty, and colors. Do not invent or modify any numeric values or capacities.")
        lines.append(
            "- If you see a structured 'Product Details' block in the context (for example with fields like 'Display', 'Rear Camera', 'Front Camera', 'Storage', 'Processor', 'SIM card', 'Connectors', 'Bluetooth', 'Battery', 'Size and weight', 'Operating system', 'Water resistance'), copy those fields into additional Markdown bullets AFTER the core spec bullets, using labels like '**Display**', '**Rear Camera**', etc., and preserve the exact numbers and phrases from the context. Do not add new specs or change any numbers."
        )
        lines.append("- If multiple prices appear for a product, treat a line labelled 'Sale price' as the current selling price when present; otherwise, treat the lowest numeric price as the current selling price and mention higher prices (such as MRP) only as reference.")
        # Base vs variant wording, derived from the raw query text.
        # If the user mentions only the base model (e.g., "iphone 13"), treat that as
        # the primary product when it exists, but still surface same-base variants
        # (e.g., mini / pro / pro max / plus) as alternatives.
        ql_prod = (query or "").lower()
        has_variant_word = any(t in ql_prod for t in [" pro", " max", " mini", " plus"])  # simple heuristic
        if not has_variant_word:
            lines.append(
                "- When the user asks for a base model (for example, 'iPhone 13'), describe that base product "
                "when it exists, and if the Internal product answer plan lists same-base variants (such as "
                "'iPhone 13 mini' or 'iPhone 13 Pro Max'), you MUST add a section like 'Other variants you can "
                "consider:' followed by one Markdown bullet per sibling with its full name and product page URL."
            )
        else:
            lines.append(
                "- When the user asks for a specific variant (for example, 'iPhone 16 Pro' or 'iPhone 13 mini') and "
                "that exact variant exists in the Internal product answer plan, focus the answer on that variant "
                "first (with its own specs and URL), then you may briefly list other same-base variants as "
                "alternatives, clearly labelling them as different options."
            )
        lines.append("- If no product in the context or catalog matches the key qualifiers from the question, clearly say that the exact variant is not available or not found and avoid attaching a specific product URL.")
        lines.append("- If a very similar product is present (for example, the same model family without an extra word like 'Pro', 'Max', or 'Plus'), you may briefly describe that closest product while making it explicit that it is a different variant from what the user asked for. Never present the similar product as if it were the exact requested model.")
        # Use product_match to control how availability is communicated so behavior stays
        # consistent and dynamic with the catalog.
        if product_match is not None:
            if product_match.match_type == "EXACT" and product_match.candidates:
                lines.append(
                    "- Since the internal catalog match is EXACT, clearly confirm that the requested model is available and describe that exact product using details from the context and the catalog URL above."
                )
            elif product_match.match_type == "CLOSE_BUT_DIFFERENT" and product_match.candidates:
                lines.append(
                    "- Since the internal catalog match is CLOSE_BUT_DIFFERENT, explicitly say that the exact requested variant is not available. Then immediately introduce the closest available variant from the catalog (using its name and URL from the Internal catalog match summary) and describe its details (storage, colors, condition, warranty, price) based only on the provided context."
                )
            elif product_match.match_type == "NONE":
                lines.append(
                    "- Since the internal catalog match is NONE, clearly state that the requested product is not available in the catalog and do not attach any specific product URL. If other related products appear in context, you may mention them generically without pretending they are the requested model."
                )
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


def _rewrite_language(text: str, target_mode: str) -> str:
    """Rewrite assistant text to the target language mode (EN or HI_EN).

    Args:
        text: The original assistant output.
        target_mode: "EN" or "HI_EN".

    Returns:
        Rewritten text on success; original text on failure.
    """
    try:
        if not text.strip():
            return text
        if target_mode == "HI_EN":
            instr = (
                "Rewrite the reply strictly in Hinglish (mix Hindi + English naturally). "
                "Keep meaning intact, do not add new content, and keep it concise."
            )
        else:
            instr = (
                "Rewrite the reply strictly in clear, simple English. "
                "Keep meaning intact, do not add new content, and keep it concise."
            )
        prompt = f"{instr}\n\nOriginal:\n{text}\n\nRewrite:"
        return _call_openai(prompt)
    except Exception:
        return text


def _detect_intent(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in ["return", "refund", "cancel", "exchange"]):
        return "returns"
    if "warranty" in q or "guarantee" in q:
        return "warranty"
    # Shipping / delivery detection (avoid location hardcoding; rely on generic phrasing)
    if any(k in q for k in [
        "ship",
        "shipping",
        "ship to",
        "deliver",
        "deliver to",
        "delivery",
        "international"
    ]):
        return "shipping"
    if "privacy" in q or "data" in q or "terms" in q:
        return "policy"
    address_signals = [
        "address", "head office", "headoffice", "hq", "headquarter", "headquarters",
        "location", "where is office", "office where", "where is your office"
    ]
    if "contact" in q or "support" in q or "help" in q or any(s in q for s in address_signals):
        return "contact"
    # Product-style queries (generic, driven by config-based device families):
    # If users ask for details/specs/price OR mention any configured family keyword,
    # treat as product info intent.
    product_cues = [
        "spec",
        "specs",
        "specifications",
        "details",
        "price",
        "prices",
        "features",
    ]
    if any(c in q for c in product_cues):
        return "product"
    # Any product family keyword from config should map to product intent
    for _, kws in PRODUCT_FAMILIES.items():
        if any(k in q for k in kws):
            return "product"
    # Fallback: generic device nouns if config is missing or incomplete
    device_nouns = [
        "macbook",
        "mac book",
        "mackbook",
        "iphone",
        "ipad",
        "laptop",
        "notebook",
        "phone",
        "mobile",
    ]
    if any(n in q for n in device_nouns):
        return "product"
    # order/buy intent (broad coverage for EN + Hinglish)
    buy_signals = [
        "order", "buy", "purchase", "checkout", "cart", "place order",
        "kharid", "kharidna", "order kaise", "kaise karun", "kaise karu", "order karu"
    ]
    if any(sig in q for sig in buy_signals):
        return "order_buy"
    return "general"


def _detect_tone(text: str) -> str:
    t = (text or "").lower().strip()
    if not t:
        return "neutral"
    upset_signals = [
        "late", "delay", "delayed", "not delivered", "still not", "why", "angry", "frustrated",
        "complaint", "issue", "problem", "worst", "bad service", "refund now", "escalate"
    ]
    if any(sig in t for sig in upset_signals):
        return "upset"
    exclaim = t.count("!") >= 2
    allcaps = any(len(w) >= 3 and w.isupper() for w in t.split())
    if exclaim or allcaps:
        return "upset"
    return "neutral"


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


def _classify_intent(query: str, previous_answer: str) -> tuple[str, str]:
    """Classify high-level intent for the current turn.

    Args:
        query: Current user message.
        previous_answer: Last assistant answer in the thread.

    Returns:
        A tuple of (intent, last_intent) where intent is one of
        {"product", "returns", "shipping", "contact", "order_buy", "general", "unclear"}.
    """

    intent = _detect_intent(query)
    last_intent = _infer_last_intent(previous_answer)
    ql = (query or "").lower().strip()

    # Basic heuristic for unclear/noisy queries: no alphabetic characters or obvious junk tokens
    has_alpha = any(ch.isalpha() for ch in ql)
    noise_tokens = {"???", "????", "asdf", "qwerty"}
    if (not ql or not has_alpha) and intent == "general":
        return "unclear", last_intent
    if any(tok in ql for tok in noise_tokens) and intent == "general":
        return "unclear", last_intent

    return intent, last_intent


def _classify_product_domain(query: str) -> str:
    """Classify product-like questions into a finer domain.

    This uses a small LLM prompt to avoid brittle keyword hardcoding while
    staying cheap. It is only called when the heuristic intent is "product"
    but the catalog cannot find a concrete model match.

    Domains:
        - product_specs: asking about availability/price/specs of a specific model.
        - buying_advice: when/why/if to buy, comparisons, pros/cons.
        - brand_reputation: ratings, reviews, whether the brand/site is trusted.
        - generic_support: anything else.
    """

    q = (query or "").strip()
    if not q:
        return "generic_support"

    # If no API key is configured, fall back to treating this as a specs-style
    # question so behaviour degrades gracefully instead of failing.
    if not os.getenv("OPENAI_API_KEY"):
        return "product_specs"

    prompt = (
        "You classify product-related questions into one of four domains.\n\n"
        "Domains:\n"
        "- product_specs: user asks about availability, price, storage, colours, condition, warranty, or specs of a specific model (for example 'do you have iPhone 11', 'what is the price of iPhone 11').\n"
        "- buying_advice: user asks when/why/if to buy, which model to choose, comparisons, pros/cons, best time to buy, or which option gives better value.\n"
        "- brand_reputation: user asks about ratings, reviews, trust, customer satisfaction, or overall reputation of a brand or website (for example 'what is the rating of GREST').\n"
        "- generic_support: other support questions that are not about a specific product's specs, buying advice, or brand reputation.\n\n"
        f"Question: {q}\n\n"
        "Return exactly one token: product_specs, buying_advice, brand_reputation, or generic_support."
    )

    try:
        label_raw = _call_openai(prompt, max_retries=2, model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    except Exception:
        return "product_specs"

    label = (label_raw or "").strip().lower()
    if "buying_advice" in label:
        return "buying_advice"
    if "brand_reputation" in label:
        return "brand_reputation"
    if "generic_support" in label:
        return "generic_support"
    return "product_specs"


def answer_query(
    query: str,
    top_k: int = 6,
    previous_answer: str = "",
    previous_user: str = "",
) -> tuple[str, List[Citation]]:
    # Detect user intent for conversational follow-ups via classifier abstraction
    intent, last_intent = _classify_intent(query, previous_answer)
    domain_tag = ""

    # For product intents, run catalog-aware product_search (currently iPhone-only)
    # to understand whether the requested model is an exact match, a close
    # variant (e.g., 16 vs 16 Pro), or not in the catalog at all. Then build a
    # deterministic answer plan from that match so the prompt does not have to
    # infer the variant relationships.
    product_match: Optional[ProductSearchResult] = None
    product_plan: Optional[ProductAnswerPlan] = None
    if intent == "product":
        # Default domain for clear catalog hits is product_specs; we only
        # refine further when the catalog cannot find a concrete match.
        domain_tag = "product_specs"
        ql_ps = (query or "").lower()
        family_hint = "iphone" if "iphone" in ql_ps else None
        try:
            product_match = product_search(query, family_hint=family_hint)
        except Exception:
            product_match = None
        try:
            product_plan = build_product_answer_plan(query, product_match)
        except Exception:
            product_plan = None

        # If there is no concrete catalog match, treat this as an ambiguous
        # product-like question (for example, generic buying advice) and let a
        # small classifier decide whether it is really about specs or not.
        no_catalog_match = (
            product_match is None
            or not getattr(product_match, "candidates", None)
            or getattr(product_match, "match_type", "NONE") == "NONE"
        )
        if no_catalog_match:
            try:
                domain_tag = _classify_product_domain(query)
            except Exception:
                domain_tag = "product_specs"
            # When the domain is not specs, treat this as a non-product query so
            # we do not run product specs loaders or product-style prompting.
            if domain_tag != "product_specs":
                intent = "general"
                product_match = None
                product_plan = None

    # Early exit for unclear intent: ask user to rephrase instead of guessing
    if intent == "unclear":
        mode = _detect_mode(query or previous_user)
        if mode == "HI_EN":
            msg = "Mujhe thoda clear nahi hua. Please thoda detail mein ya alag tareeke se bataoge?"
        else:
            msg = "I’m not fully sure what you mean. Can you rephrase or add a bit more detail?"
        return msg, []

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
    elif intent == "product":
        # Generic product cues without hardcoding specific product names
        aug = " product details specs specifications features price"
    aug_query = (query + aug).strip()

    specs_by_url: Dict[str, ProductSpecs] = {}
    if intent == "product" and domain_tag == "product_specs":
        try:
            specs_by_url = _load_product_specs_for_candidates(product_match)
        except Exception:
            specs_by_url = {}

    # Retrieve (with optional per-intent allowlist boost and facet expansion)
    original_allow = os.getenv("RETRIEVE_SOURCE_ALLOWLIST", "")

    def _ensure_in_allowlist(pattern: str) -> None:
        """Ensure a retrieval allowlist pattern is present for this call.

        This function accumulates patterns on top of the current
        RETRIEVE_SOURCE_ALLOWLIST value and relies on the outer scope to
        restore the original_allow at the end of answer_query.
        """
        current = os.getenv("RETRIEVE_SOURCE_ALLOWLIST", "")
        prim = [p for p in (s.strip() for s in current.split(",")) if p]
        if pattern not in prim:
            os.environ["RETRIEVE_SOURCE_ALLOWLIST"] = ",".join(prim + [pattern])

    # For product intents, ensure that catalog-backed product pages for the
    # matched base model (and its variants) are explicitly included in the
    # retrieval allowlist so the LLM can see sibling variants like "iPhone 13"
    # and "iPhone 13 mini" together.
    if intent == "product" and product_match is not None and product_match.candidates:
        for cand in product_match.candidates:
            try:
                parsed = urlparse(cand.url)
                path = parsed.path or cand.url
                if path:
                    _ensure_in_allowlist(path)
            except Exception:
                continue
    
    # Detect acknowledgement consent (LLM classifier with heuristic fallback)
    ql = (query or "").lower()
    prevl = (previous_answer or "").lower()
    ack_label = _classify_ack(previous_answer, query) if previous_answer else "NEW_TOPIC"
    ack = ack_label == "ACK_CONTINUE"
    prev_offered = any(tok in prevl for tok in ["support", "phone", "email", "contact"])
    # Detect generic "more details" request from the current turn
    more_details = any(tok in ql for tok in ["more details", "details", "zyada details", "details chahiye"]) and len(ql) <= 60
    # If the previous answer explicitly offered more details and the user acknowledged (e.g., "yes please"),
    # treat this as a dynamic request for more details on the same topic, without hardcoding product names.
    if not more_details and ack:
        more_detail_offer_signals = [
            "would you like to see more details",
            "would you like more details",
            "want more details",
            "would you like to know more",
            "see more details",
            "share more details",
            "zyada details",
        ]
        if any(sig in prevl for sig in more_detail_offer_signals):
            more_details = True
    # Effective intent for acknowledgements
    effective_intent = intent
    if ack and intent == "general" and last_intent != "general":
        effective_intent = last_intent

    # For product follow-ups, keep retrieval biased toward the same product family
    # mentioned in the previous answer using configured domain nouns instead of
    # hardcoded SKUs.
    if effective_intent == "product" and (ack or more_details) and previous_answer:
        matched = False
        for fam_name, kws in PRODUCT_FAMILIES.items():
            for kw in kws:
                if kw in prevl and kw not in aug_query.lower():
                    aug_query = f"{aug_query} {kw}".strip()
                    matched = True
                    break
            if matched:
                break
        # Fallback: if no config keyword matched, use generic device nouns
        if not matched:
            fallback_nouns = [
                "macbook",
                "mac book",
                "mackbook",
                "iphone",
                "ipad",
                "laptop",
                "notebook",
                "phone",
                "mobile",
            ]
            for noun in fallback_nouns:
                if noun in prevl and noun not in aug_query.lower():
                    aug_query = f"{aug_query} {noun}".strip()
                    break

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
            _ensure_in_allowlist("/pages/shipping")
        elif effective_intent == "product":
            # Bias retrieval toward product pages without hardcoding any product names
            _ensure_in_allowlist("/products/")
        # Domain-level routing for non-product informational queries so that
        # buying advice and reputation questions can target the right sources
        # (blogs, FAQs, reviews) without brittle keyword checks.
        if domain_tag == "buying_advice" and effective_intent in {"general", "order_buy"}:
            _ensure_in_allowlist("/blogs/news/")
            _ensure_in_allowlist("/pages/faqs")
        if domain_tag == "brand_reputation" and effective_intent in {"general", "order_buy"}:
            _ensure_in_allowlist("trustpilot.com/review")
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

    if intent == "product" and product_match is not None and product_match.candidates:
        cand_paths: Dict[str, str] = {}
        for cand in product_match.candidates:
            try:
                parsed = urlparse(cand.url)
                path = parsed.path or cand.url
            except Exception:
                path = cand.url
            key = (path or "").split("?", 1)[0].rstrip("/")
            if key:
                cand_paths[key] = cand.url
        if cand_paths:
            for ch in items:
                try:
                    parsed_src = urlparse(ch.source)
                    path_src = parsed_src.path or ch.source
                except Exception:
                    path_src = ch.source
                key_src = (path_src or "").split("?", 1)[0].rstrip("/")
                cand_url = cand_paths.get(key_src)
                if not cand_url or cand_url in specs_by_url:
                    continue
                try:
                    specs = extract_product_specs(ch.text)
                except Exception:
                    continue
                if (
                    specs.price_strings
                    or specs.storage_options
                    or specs.conditions
                    or specs.warranty_strings
                    or specs.color_options
                ):
                    specs_by_url[cand_url] = specs

    # Build citations list in the same order as chunks appear in prompt
    citations: List[Citation] = []
    for it in items:
        citations.append(Citation(url=it.source, start_line=it.start_line, end_line=it.end_line))

    # Compose and call LLM
    prompt = _compose_prompt(
        query=query,
        chunks=items,
        intent=effective_intent,
        previous_answer=previous_answer,
        previous_user=previous_user,
        product_match=product_match if effective_intent == "product" else None,
        product_plan=product_plan if effective_intent == "product" else None,
        product_specs_by_url=specs_by_url if effective_intent == "product" else None,
    )
    txt = _call_openai(prompt)

    # Apply best-effort fallback if enabled and the model could not find an answer
    fallback_on = os.getenv("ANSWER_FALLBACK_ENABLE", "1") in {"1", "true", "TRUE", "yes"}
    followups_on = os.getenv("ANSWER_FOLLOWUPS_ENABLE", "1") in {"1", "true", "TRUE", "yes"}
    tone_on = os.getenv("ANSWER_TONE_AWARE", "0") in {"1", "true", "TRUE", "yes"}
    tone = _detect_tone(previous_user or query) if tone_on else "neutral"

    def _build_fallback_text() -> str:
        mode = _detect_mode(query)
        graceful = os.getenv("ANSWER_FALLBACK_GRACEFUL", "0") in {"1", "true", "TRUE", "yes"}
        use_emoji = False
        try:
            elv = int(os.getenv("PERSONA_EMOJI_LEVEL", "0"))
        except Exception:
            elv = 0
        if tone != "upset" and elv > 0:
            use_emoji = True
        emoji = " 🙂" if use_emoji else ""
        opts_map = {
            "contact": ["Phone number", "Email"],
            "returns": ["Cancel steps", "Refund timeline"],
            "warranty": ["Coverage", "Claim process"],
            # Offer Charges first for shipping queries to feel more relevant
            "shipping": ["Charges", "Delivery timelines"],
            "order_buy": ["Payment options", "How to order"],
            "general": ["Policy link", "Details"],
        }
        opts = opts_map.get(intent) or opts_map["general"]
        # For product queries, avoid leaking noisy catalog snippets (e.g. vitamins/quiz)
        # and instead return a controlled, graceful clarification message.
        if intent == "product":
            if mode == "HI_EN":
                head = "Mujhe is exact iPhone/MacBook model ka product page nahi mila." + emoji
                ask = (
                    "Kya aap exact model (jaise iPhone 11, iPhone 12, MacBook Air 2017) "
                    "bata sakte ho, taaki main sahi details de sakun?"
                )
            else:
                head = "I couldn’t find an exact match for that model in our catalog." + emoji
                ask = (
                    "Could you please rephrase or mention the exact model you’re looking for "
                    "(for example, iPhone 11, iPhone 12, or a specific MacBook variant)?"
                )
            return f"{head}\n\n{ask}"
        if graceful:
            if mode == "HI_EN":
                head = "Exact line nahi mila, par yeh closest info hai." + emoji
                ask = (
                    f"Aap chaho to main {opts[0]} ya {opts[1]} share kar sakti hoon.\n"
                    "Jo exact detail chahiye batao, main turant nikaal dungi."
                )
            else:
                head = (
                    "I couldn’t find an exact line on that yet, but here’s the closest helpful info I do have." 
                    + emoji
                )
                ask = (
                    f"If you want, I can share {opts[0]} or {opts[1]}.\n"
                    "Tell me the exact detail you need and I’ll fetch it."
                )
            pieces: List[str] = [head]
            for ch in items[:2]:
                sn = _clean_snippet(ch.text)
                if sn:
                    pieces.append(sn)
            # Keep ask as a separate line for Slack readability
            pieces.append(ask)
            return "\n\n".join(pieces)
        else:
            intro = (
                "Exact info nahi mila, par yeh closest details hain:" if mode == "HI_EN" else
                "I couldn’t find the exact info, here’s the closest helpful detail:"
            )
            pieces2: List[str] = [intro]
            for ch in items[:2]:
                snippet = _clean_snippet(ch.text)
                if snippet:
                    pieces2.append(snippet)
            if mode == "HI_EN":
                ask2 = f"Kya main {opts[0]} ya {opts[1]} share karun? Ya aap bata dein kis cheez ki details chahiye, main help kar dungi."
            else:
                ask2 = f"Want me to share {opts[0]} or {opts[1]}? Or tell me what you need and I’ll help."
            pieces2.append(ask2)
            return "\n\n".join(pieces2)

    out_text = _strip_inline_citations(txt)
    low = out_text.strip().lower()
    if fallback_on and (low.startswith("not found in sources provided")):
        out_text = _build_fallback_text()
    # If user explicitly asked for shipping charges but none of the retrieved texts contain charge-like tokens,
    # use the graceful fallback even if the model produced a generic shipping answer.
    if fallback_on and effective_intent == "shipping":
        q_has_charges = any(tok in ql for tok in ["charge", "charges", "fee", "fees", "cost", "costs", "pricing", "price"])
        if q_has_charges:
            ctx_join = " ".join((it.text or "") for it in items[:6]).lower()
            ctx_has_charges = any(tok in ctx_join for tok in ["charge", "charges", "fee", "fees", "cost", "costs", "₹", "rs ", "rs."])
            if not ctx_has_charges:
                out_text = _build_fallback_text()

    # Deterministic sibling variants section for product intents so the
    # base-model flow always lists alternatives.
    if effective_intent == "product" and product_plan is not None and product_plan.siblings:
        already_has = "other variants you can consider" in out_text.lower()
        if not already_has:
            ql_prod2 = (query or "").lower()
            has_variant_word2 = any(t in ql_prod2 for t in [" pro", " max", " mini", " plus"])
            header = "Other variants in the same family:" if has_variant_word2 else "Other variants you can consider:"
            sib_lines: List[str] = []
            for sib in product_plan.siblings:
                name = getattr(sib, "name", "") or ""
                url = getattr(sib, "url", "") or ""
                if not name or not url:
                    continue
                sib_lines.append(f"- {name}  {url}")
            if sib_lines:
                section = "\n".join([header] + sib_lines)
                out_text = f"{out_text}\n\n{section}"

    # Deterministic contact drill-down: if user asked to continue/more details for contact/address
    # Observe-only guard: when ANSWER_ACK_OBSERVE_ONLY is enabled, do not change behavior
    ack_observe_only = os.getenv("ANSWER_ACK_OBSERVE_ONLY", "1") in {"1", "true", "TRUE", "yes"}
    if (not ack_observe_only) and effective_intent == "contact" and (ack or more_details):
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
    if followups_on and effective_intent != "product":
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
                if tone == "upset":
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

    # Optional warm suffix for neutral tone (no facts changed)
    try:
        emoji_level2 = int(os.getenv("PERSONA_EMOJI_LEVEL", "0"))
    except Exception:
        emoji_level2 = 0
    warmth_on = os.getenv("ANSWER_PERSONA_WARMTH", "0") in {"1", "true", "TRUE", "yes"}
    if warmth_on and tone != "upset" and emoji_level2 > 0:
        mode = _detect_mode(previous_user or query)
        parts_ws = out_text.split("\n\n", 1)
        first_ws = parts_ws[0]
        rest_ws = parts_ws[1] if len(parts_ws) > 1 else ""
        has_emoji = any(e in first_ws for e in ["🙂", "✅", "😊", "😉"])
        is_greeting = (previous_user or "").strip().lower() in {"hi", "hello", "hey", "hii", "hye"}
        already_warm = any(s in first_ws.lower() for s in ["happy to help", "help kar dungi", "glad to help"]) 
        # Add warmth only for greetings, avoid repetition
        if not has_emoji and not already_warm and is_greeting:
            if mode == "HI_EN":
                warm = "  Batao, main help kar dungi 🙂"
            else:
                warm = "  Happy to help 🙂"
            first_ws = first_ws + warm
            out_text = first_ws if not rest_ws else f"{first_ws}\n\n{rest_ws}"

    # Language force-rewrite guard (optional)
    lang_lock_on = os.getenv("ANSWER_LANGUAGE_LOCK", "0") in {"1", "true", "TRUE", "yes"}
    lang_force_on = os.getenv("ANSWER_LANGUAGE_FORCE_REWRITE", "0") in {"1", "true", "TRUE", "yes"}
    target_mode = _detect_mode(previous_user or query)
    current_mode = _detect_mode(out_text)
    if lang_lock_on and lang_force_on and current_mode != target_mode:
        out_text = _rewrite_language(out_text, target_mode)
        current_mode = _detect_mode(out_text)

    # Snapshot debug info for ribbon when enabled
    try:
        top_score = items[0].score if items else 0.0
    except Exception:
        top_score = 0.0
    def _is_fallback_text(s: str) -> bool:
        t = (s or "").strip()
        return (
            t.startswith("Exact info nahi mila")
            or t.startswith("I couldn’t find the exact info")
            or t.startswith("I couldn’t find a direct line on that")
        )

    fallback_used = (
        out_text.startswith("Exact info nahi mila") 
        or out_text.startswith("I couldn’t find the exact info")
        or out_text.startswith("I couldn’t find a direct line on that")
    )

    # Escalation: if previous answer was a fallback and current is also a fallback, offer human support
    if fallback_on and fallback_used and _is_fallback_text(previous_answer):
        support_phone = (os.getenv("SUPPORT_PHONE", "") or "").strip()
        support_email = (os.getenv("SUPPORT_EMAIL", "") or "").strip()
        contact_link = "https://grest.in/pages/contact-us"
        mode_es = _detect_mode(previous_user or query)
        emoji_es = " 🙂" if (tone != "upset" and int(os.getenv("PERSONA_EMOJI_LEVEL", "0") or 0) > 0) else ""
        if mode_es == "HI_EN":
            esc = [
                "Agar aap chaho to main support se connect kara sakti hoon." + emoji_es,
                f"Phone: {support_phone}" if support_phone else "",
                f"Email: {support_email}" if support_email else "",
                f"Link: {contact_link}",
            ]
        else:
            esc = [
                "If you want, I can connect you to our support team." + emoji_es,
                f"Phone: {support_phone}" if support_phone else "",
                f"Email: {support_email}" if support_email else "",
                f"Contact link: {contact_link}",
            ]
        esc_text = "\n".join([line for line in esc if line])
        out_text = f"{out_text}\n\n{esc_text}"
    global _LAST_DEBUG
    _LAST_DEBUG = (
        f"intent={intent} | last_intent={last_intent} | eff_intent={effective_intent} | ack={int(ack)} | more_details={int(more_details)} | top_score={top_score:.2f} | fallback={int(fallback_used)} | lang_target={target_mode} | lang_out={current_mode} | tone={tone}"
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
