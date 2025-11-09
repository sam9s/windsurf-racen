from fastapi import APIRouter
from pydantic import BaseModel, Field
import os
import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Tuple

# OpenAI SDK used against Mistral's OpenAI-compatible endpoint
from openai import OpenAI

router = APIRouter()

# ----------------------
# Models
# ----------------------
class FaqChatRequest(BaseModel):
    # Backward compatible: accept either a single url or multiple urls
    url: str | None = Field(default=None, description="Public page URL to ground answers from")
    urls: List[str] | None = Field(default=None, description="Multiple public page URLs to ground answers from")
    question: str = Field(..., description="User question")
    k: int | None = Field(default=6, description="Number of top context chunks to use")

class Citation(BaseModel):
    snippet: str
    source: str

class FaqChatResponse(BaseModel):
    answer: str
    citations: List[Citation]

# ----------------------
# Defaults
# ----------------------
DEFAULT_URLS = [
    "https://grest.in/pages/faqs",
    "https://grest.in/policies/refund-policy",       # Returns & Refunds
    "https://grest.in/policies/shipping-policy",     # Shipping & Delivery
    "https://grest.in/policies/terms-of-service",    # Terms & Conditions
    "https://grest.in/pages/warranty",               # Warranty policy (best guess)
]

# ----------------------
# Helpers
# ----------------------

def fetch_and_clean_html(url: str) -> str:
    """Fetches the URL and returns main text content with minimal boilerplate.
    We keep list items and headings since FAQs/policies are often structured that way.
    """
    print(f"[faq_chat] Fetching URL: {url}")
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove scripts/styles/nav/footer
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "form", "iframe"]):
        tag.decompose()

    # Heuristic: find the largest content container
    candidates = soup.find_all(["main", "article", "section", "div"])
    best = max(candidates, key=lambda c: len(c.get_text(" ", strip=True))) if candidates else soup
    text = best.get_text("\n", strip=True)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"\s+", " ", text)
    return text


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 150) -> List[str]:
    """Chunk text into overlapping segments to preserve context."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def simple_keyword_score(query: str, chunk: str) -> float:
    """Very simple scoring: keyword overlap weighted by term frequency, length-normalized."""
    q_terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2]
    if not q_terms:
        return 0.0
    c_terms = re.findall(r"[a-z0-9]+", chunk.lower())
    if not c_terms:
        return 0.0
    tf: Dict[str, int] = {}
    for t in c_terms:
        tf[t] = tf.get(t, 0) + 1
    score = sum(tf.get(t, 0) for t in q_terms) / (len(c_terms) ** 0.5)
    return float(score)


def rank_chunks(question: str, sourced_chunks: List[Tuple[str, str]], k: int) -> List[Tuple[str, str]]:
    # sourced_chunks: list of (chunk_text, source_url)
    scored = [
        (simple_keyword_score(question, chunk_text), chunk_text, source_url)
        for (chunk_text, source_url) in sourced_chunks
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [(c, src) for s, c, src in scored[: max(1, k)]]
    return top


def build_context(top_chunks: List[Tuple[str, str]]) -> str:
    parts = []
    for i, (c, src) in enumerate(top_chunks, 1):
        parts.append(f"[Chunk {i} | {src}]\n{c}")
    return "\n\n".join(parts)


def mistral_client() -> OpenAI:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not configured")
    # Use Mistral's OpenAI-compatible endpoint
    client = OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")
    return client


def ask_mistral(context: str, question: str) -> str:
    client = mistral_client()
    system = (
        "You are RACEN, a customer support AI. Answer ONLY using the provided context. "
        "If the answer is not present in the context, say you cannot find it in the provided sources. "
        "Keep answers concise and policy-accurate."
    )

    prompt = (
        "Context:\n" + context + "\n\n" +
        "Instructions: Use only the context to answer. If insufficient, state that clearly.\n" +
        f"Question: {question}"
    )

    completion = client.chat.completions.create(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return completion.choices[0].message.content or ""


# ----------------------
# Endpoint
# ----------------------
@router.post("/faq-chat")
def faq_chat(body: FaqChatRequest) -> FaqChatResponse:
    """Answer a question using one or more public URLs as the only knowledge source."""
    try:
        # Build target URL list respecting backward compatibility and defaults
        targets: List[str] = []
        if body.urls and len(body.urls) > 0:
            targets.extend([u for u in body.urls if isinstance(u, str) and u.strip()])
        elif body.url and body.url.strip():
            targets.append(body.url.strip())
        else:
            # Fallback to default curated set for GREST
            targets.extend(DEFAULT_URLS)

        print(f"[faq_chat] Targets: {targets}")

        # Fetch and chunk across all sources
        sourced_chunks: List[Tuple[str, str]] = []  # (chunk_text, source_url)
        for u in targets:
            try:
                raw_text = fetch_and_clean_html(u)
                print(f"[faq_chat] Extracted characters from {u}: {len(raw_text)}")
                chunks = chunk_text(raw_text)
                for ch in chunks:
                    sourced_chunks.append((ch, u))
            except Exception as fe:
                print(f"[faq_chat][warn] Skipping {u}: {fe}")
                continue

        if not sourced_chunks:
            raise RuntimeError("No content could be fetched from provided URLs")

        top_pairs = rank_chunks(body.question, sourced_chunks, body.k or 6)
        context = build_context(top_pairs)

        answer = ask_mistral(context, body.question)

        citations = [
            Citation(
                snippet=(c[:300] + ("…" if len(c) > 300 else "")),
                source=src,
            )
            for (c, src) in top_pairs
        ]
        return FaqChatResponse(answer=answer.strip(), citations=citations)
    except Exception as e:
        # Return graceful failure with minimal details
        print(f"[faq_chat][error] {e}")
        return FaqChatResponse(
            answer=(
                "I couldn't retrieve an answer from the sources right now. "
                "Please try again shortly or escalate to a human."
            ),
            citations=[],
        )
