# RACEN Gaps – Holy Grail 51 (Round 1)

Source: 51-query mother test report generated on 2025-11-26.

This file tracks all known gaps discovered from the 51-query "holy grail" harness run. We only mark a gap as **CLOSED** when:

- Behavior is consistent for all relevant queries in the 51 set **and** their obvious variants.
- URLs and business facts are present wherever we decided they must be.
- The behavior is enforced by code and covered by tests / the harness, not just a one-off fix.

When every gap below is marked **CLOSED**, this file should be renamed with a `*_closed.md` suffix and a new gaps file started for the next wave.

---

## Gap 1 – Generic iPhone family browse flows

- **Description:** Natural-language iPhone browse queries ("What all iPhones you have?", "List a few iPhones you have", etc.) should deterministically list real iPhone models from the Grest catalog with prices, key specs, and correct URLs (including the iPhone collection URL).
- **Examples in harness:** Q17–Q21.
- **Current behavior (2025-11-26 run):** All of these now return deterministic, catalog-backed iPhone lists plus the collection URL; no fallbacks.
- **Status:** **CLOSED** (based on current 51 queries).
- **Criteria to keep CLOSED:**
  - New iPhone family phrasings added later should route through the same deterministic path.
  - Future regression tests / harness runs must keep these at `Status: ok` with no generic fallbacks.

---

## Gap 2 – Policy / shipping / returns / cancellation / warranty URLs

- **Description:** Any answer about shipping, returns, cancellation, or warranty must:
  - State the correct policy facts, and
  - Include at least one **canonical policy URL** in the visible answer (not only in citations), e.g.:
    - `https://grest.in/pages/returns-refund-cancellation`
    - `https://grest.in/policies/shipping/policy`
    - `https://grest.in/pages/warranty`
- **Examples in harness:**
  - Q22: "What is your shipping policy?" – has URL in the text.
  - Q23: "What warranty do you provide?" – no warranty URL in the text.
  - Q24–Q26: returns / cancellation related – facts are generally correct but URL usage is inconsistent.
- **Current behavior:**
  - Shipping answer surfaces the URL.
  - Warranty/returns/cancellation answers are factually aligned but **do not consistently surface URLs** in the visible text.
- **Status:** **OPEN**.
- **Closure criteria:**
  - Every shipping / returns / cancellation / warranty style query in the 51 set (and obvious variants) includes at least one canonical URL in the final answer.
  - No answer in this domain is purely free-text without a supporting URL.

---

## Gap 3 – COD availability ("Do you offer cash on delivery?")

- **Description:** Questions explicitly asking whether Cash on Delivery (COD) is available must be answered deterministically from `business_facts` (not via RAG randomness or graceful fallback).
- **Examples in harness:**
  - Q29: "What payment options do you have?" – correctly mentions COD as one of the options.
  - Q30: "Do you offer cash on delivery?" – still returns the static fallback message.
  - Additional variant mentioned by user: "Do you have cash on delivery COD available?" – same symptom.
- **Current behavior:**
  - Payment options answers reference COD correctly.
  - Direct COD yes/no queries still go to the generic fallback.
- **Status:** **OPEN**.
- **Closure criteria:**
  - All explicit COD-availability queries ("Do you offer COD?", "Is cash on delivery available?", etc.) are answered from `business_facts.payment.cod_enabled` with a clear yes/no and, ideally, a link to the relevant FAQ/policy page.
  - No COD query in the 51 set ends up at the generic fallback.

---

## Gap 4 – Support contact answers must include URLs

- **Description:** Answers that provide support phone or email must also give the **relevant support/contact URL** so users can self-serve from the site.
  - For example: `https://grest.in/pages/contact-us`.
- **Examples in harness:**
  - Q33: "What is your contact number?" – returns correct phone, but **no URL**.
  - Q34: "What is your customer support email?" – returns correct email, but **no URL**.
- **Current behavior:**
  - Phone and email values are correct (wired from env / business facts).
  - No contact/support URL is surfaced in these answers.
- **Status:** **OPEN**.
- **Closure criteria:**
  - All support contact answers (phone, email, possibly address) include a stable contact/support URL in the visible text.
  - 51-query harness shows URLs present for all such questions.

---

## Gap 5 – Brand reputation (Trustpilot / MouthShut) URLs

- **Description:** Any answer about Grest's reputation, ratings, or reviews **must** include at least one canonical review URL in the visible text whenever the information comes from our indexed review sources, e.g.:
  - `https://www.trustpilot.com/review/grest.in`
  - `https://www.mouthshut.com/product-reviews/grest-reviews-926180198`
- **Examples in harness:**
  - Several queries ask for Trustpilot rating, MouthShut reviews, or general Grest reputation.
  - Only one flagship answer currently embeds both URLs explicitly; most others talk about reviews but **omit URLs**.
- **Current behavior:**
  - Facts about ratings and sentiment are generally correct.
  - URLs are **missing** from most brand-reputation answers; they appear in only a minority of responses.
- **Status:** **OPEN**.
- **Closure criteria:**
  - Every brand-reputation answer in this domain includes at least one of the canonical URLs in the visible answer.
  - For queries that mention a specific platform (e.g., Trustpilot), that platform's URL is always present.

---

## Gap 6 – Comparison flows (iPhone vs iPhone; internal vs web)

- **Description:** Comparison questions (e.g., "iPhone 11 vs iPhone 12", "Is iPhone 13 worth it over 12?", "Should I buy 14 or wait for 15?") should:
  - Describe both options with correct specs/prices/URLs.
  - End with a clear, non-truncated recommendation.
  - Prefer web comparison when available, with a robust internal fallback when web search is rate-limited.
- **Examples in harness:**
  - Q46, Q47, Q49, Q50, Q51 – look good: full answers, no truncation, clear recommendation.
  - Q45: "What is the difference between iPhone 14 and iPhone 15?" – still the generic fallback.
- **Current behavior:**
  - Most iPhone-vs-iPhone comparisons are now handled well using internal data; no truncation after increasing `ANSWER_MAX_TOKENS`.
  - Q45 remains a gap: falls straight to the static fallback.
- **Status:** **OPEN** (not closed until Q45 is also handled deterministically or via a clear internal fallback).
- **Closure criteria:**
  - Q45 is answered with a structured comparison similar in quality to Q46–Q51.
  - All comparison-style queries in this 51-set are either:
    - answered clearly using internal data, or
    - (for future) backed by web comparison with an internal fallback, never ending in a generic apology.

---

## Gap 7 – Grest vs Cashify (brand vs competitor web comparison)

- **Description:** Queries like "Grest vs Cashify which is better for refurbished iPhones?" require **external web comparison** under guardrails. Internal data alone is not enough; we must:
  - Use web search (DuckDuckGo via SerpAPI) when allowed, and
  - Fall back to a safe, honest internal comparison when web search is unavailable or rate-limited.
- **Examples in harness:**
  - Q48: "Grest vs Cashify which is better for refurbished iPhones?" – currently returns the generic fallback.
- **Current behavior:**
  - Web comparison path is not fully wired; SerpAPI 429s lead to a static fallback instead of an internal best-effort comparison.
- **Status:** **OPEN / PARKED** (intentionally deferred until we design the solid web-comparison layer).
- **Closure criteria:**
  - Web comparison pipeline is implemented with clear guardrails and tests.
  - Q48 and similar queries receive a grounded comparison answer, never the generic fallback.

---

## Gap 8 – Mode detection / output language mismatch for simple English queries

- **Description:** Pure English queries should normally be detected as `EN` and answered in clean English unless we deliberately choose Hinglish. Hinglish answers should only appear when the user actually writes in Hinglish.
- **Examples in harness:**
  - Q34: "What is your customer support email?"
    - Detected mode: `HI_EN` (incorrect for a plain English question).
    - Answer: in Hinglish, and missing a support/contact URL.
- **Current behavior:**
  - This appears to be a rare edge case but shows that our mode detection and/or normalization can misclassify straightforward English questions and then leak that into the answer language.
- **Status:** **OPEN**.
- **Closure criteria:**
  - Q34 is detected as `EN` and answered in English, or we have a clear, robust rule explaining when and why Hinglish would still be correct.
  - We add guardrails/tests so that simple English support queries are not randomly classified as Hinglish.

---

## Gap 9 – Coverage of all COD phrasings

- **Description:** Beyond the exact wording "Do you offer cash on delivery?", other reasonable phrasings like "Do you have cash on delivery COD available?" must route to the same deterministic COD answer path.
- **Relation to Gap 3:** This is effectively the **coverage** dimension of Gap 3; tracking separately here so we do not accidentally overfit to a single phrasing.
- **Current behavior:**
  - The user reports that alternate COD phrasings still hit the generic fallback.
- **Status:** **OPEN** (will be closed together with Gap 3).
- **Closure criteria:**
  - A small, well-tested intent/phrase detector (or classifier feature) maps all reasonable COD questions to the COD business-facts handler.
  - None of these variants hit the generic fallback in the 51-query harness or in targeted unit tests.
