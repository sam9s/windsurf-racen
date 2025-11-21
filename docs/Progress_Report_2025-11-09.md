# RACEN MCP Integration – Progress Report

Date: 2025-11-13
Time: 14:30 IST (UTC+05:30)

## Vision (How RACEN should feel)
- GPT-like conversational experience: mirror user language/style (incl. Hinglish), concise, helpful.
- Grounded-first answers with citations; never fabricate specifics. Clearly note when exact facts aren’t present.
- Best‑effort behavior: provide closest relevant info with disclaimers and strong guardrails.
- Intelligent follow-ups on every answer (success and no‑exact‑answer), focused on user goal and next best actions.
- Minimal friction operations: Slack-first workflows (ingest URL, status), auditable settings ribbon, clear source labeling for External data.

## Guiding Principle
- **No hardcoding of conversational logic.** RACEN must feel like GPT: dynamic, intent-aware, language-mirroring, and context-driven.
- **Slack bot remains thin.** All behavior (follow-ups, acknowledgements, action offers) is LLM-led via backend prompts and retrieval guardrails.
- **Thread awareness via `previous_answer`.** The backend provides the last assistant reply to the LLM so short acknowledgements (e.g., “haan ji”, “yes please”) are interpreted naturally without regex.

## Scope
- High-accuracy, grounded Q&A over GREST core site pages with citations.
- Slack bot wired to local Answer API for internal testing (Socket Mode).
- Incremental ingestion of core policy/info pages, then full-site.
- UX goals: settings ribbon, citations, and intelligent follow-ups (even on no-exact-answer).

## Current Pipeline (Implemented)
- **Crawl:** Local Crawl4AI-based scripts
  - scripts/crawl_and_ingest.py (targeted, single-URL), scripts/crawl_urls.py
- **Convert (HTML→Markdown):** MarkItDown fallback path
  - src/racen/step1 + scripts/convert_markdown.py
- **Clean + Chunk:** Cleaner + Chunker with line bounds
  - src/racen/step2_ingest.py (Cleaner, Chunker; records start_line/end_line)
- **Embeddings (locked):** OpenAI `text-embedding-3-small` (1536d)
  - src/racen/step2_embed.py (OpenAIEmbedder)
  - DB schema vector(1536)
  - Note: Local embeddings MCP disabled to avoid 256-d mismatch
- **Store:** Postgres + pgvector
  - src/racen/step2_write.py (ensure_schema, upserts)
- **Retrieval + Reranker:** Hybrid + bounded reranker
  - src/racen/step3_retrieve.py
  - Hybrid: pgvector cosine + Postgres FTS + fusion; RERANK_TOP_N applied
- **Answer Synthesis API:** FastAPI Answer API with settings ribbon
  - scripts/answer_api.py → POST /answer
  - scripts/step4_answer.py (short mode, token caps, citations)
- **Slack Bot:** Socket Mode bot calls Answer API
  - slack-openai-bot/app.js (env-driven allowlist; citations; ribbon)
- **Evaluation:** Free-form harness with latency
  - scripts/eval_freeform.py (settings snapshot + metrics)

## Current Working Settings (Slack + API)
- Retrieval
  - K (Slack call): 18
  - RERANK_TOP_N: 14
  - FAST_MODE: 1
  - Full-site mode for Slack: SLACK_ALLOWLIST_PRESET=all, RETRIEVE_SOURCE_ALLOWLIST=""
  - Backoff: RETRIEVE_BACKOFF_ENABLE=1, RETRIEVE_BACKOFF_THRESHOLD=0.35, RETRIEVE_BACKOFF_SECONDARY=/pages/faqs
- Answer shaping
  - ANSWER_SHORT=1, ANSWER_MAX_TOKENS=120, ANSWER_CHUNK_CHAR_BUDGET=1200
- Embeddings
  - EMBED_MODEL=text-embedding-3-small, EMBED_DIM=1536
  - EMBEDDINGS_MCP_URL disabled (avoid 256-d fallback)
- API endpoints
  - RACEN_ANSWER_URL=http://127.0.0.1:8000/answer (Python tools)
  - Slack bot uses base URL and appends /answer in code

## Status
DONE/Complete
- Answer API healthy (uvicorn), Slack E2E working with citations + ribbon.
- Core ingestion complete for 9 /pages/* plus 3 /policies/* (see docs/SANITY_COMMANDS.md for queries).
- Hinglish mirroring enabled (ANSWER_MATCH_INPUT_LANGUAGE=1) and verified via Slack.
DONE/Complete
- Best‑effort fallback implemented; intent‑based follow‑ups appended to answers.
DONE/Complete
-\- Early Slack tests: FAQs, warranty, returns policy questions grounded; some “timeline” queries lack explicit facts (by-site), which is correct behavior for grounded mode.
DONE/Complete
- LLM-driven acknowledgements enabled using `previous_answer`; Slack-side regex handler removed to avoid hardcoding.
DONE/Complete
- Persona v1 integrated (env-driven):
  - System prompt loader: PERSONA_SYSTEM_PROMPT_PATH → included at top of prompt.
  - Lexicon reply shaper (first paragraph only): PERSONA_LEXICON_PATH.
  - Tone unit tests runner added: scripts/persona_test.py (PERSONA_TONE_TESTS_PATH).
  - Slack npm script: `yarn persona:test`.

DONE/Complete
- Escalation UX refined: on 3rd repeated fallback, replace fallback body with a concise support block (Phone, Email, Contact link), shown before the ribbon.
- Recent-thread reuse added (per user+channel, 10 minutes) so fallback counts and `previous_answer` persist even if not explicitly replying in thread.
- Fallback detection hardened in Slack bot (uses ribbon + safe answer-prefix check).
- Retrieval confidence surfaced in ribbon via `top_score`/confidence for audit.

Partially Done
- Facet-like options appear in shipping fallback; consistent facet bundles across major intents (Shipping/Returns/Warranty) still to finalize.
- Citation dedup not yet applied (confidence surfaced; dedup pending).

## Plan for next 5 steps
1) Answer behavior (best‑effort + follow‑ups)
   - Best‑effort fallback: if no exact fact is present, reply “I couldn’t find an exact answer; here’s the closest info…” with citations. No fabrication.
   - Slightly expand `ANSWER_CHUNK_CHAR_BUDGET` (e.g., 1800) to include nearby clarifiers.
   - Add 2–3 intent‑based follow‑up suggestions to every answer (success and no‑exact‑answer).
   - Ribbon shows fallback and follow‑ups flags for audit.

   Sub‑plan (micro steps for follow‑ups and contact details)
    - DONE/Complete: Enable language mirroring (Hinglish/English) via `ANSWER_MATCH_INPUT_LANGUAGE=1`.
    - DONE/Complete: Add intent‑based follow‑ups appended to every answer.
    - DONE/Complete: Provide support details from env or `/pages/contact-us` (LLM-led acknowledgements, no Slack regex).
      - `SUPPORT_PHONE`, `SUPPORT_EMAIL`, `SUPPORT_ADDRESS` configured in backend `.env`.
    - DONE/Complete: Persona v1 wired (system prompt + lexicon), tone tests added.
    - Planned (Option C – later, best long‑term):
      - Create `contact_facts` table (key, value, source_url, extracted_at, confidence).
      - Backfill from existing `docling.chunks` (no re‑ingest required).
      - Update facts during future ingests.

2) Ops ergonomics (Slack ingest)
   - Slack “Ingest URL” action + `/racen-ingest <url>` slash command for internal users.
   - Backend enqueues `crawl_and_ingest` with `--dim 1536`, posts progress and completion with chunk counts.
   - Enforce domain allowlist (grest.in) and internal user restrictions.

3) Trustpilot (External, labeled)
   - Minimal ingest of Trustpilot grest.in profile with low RPS; store as External.
   - Use only high‑level stats/quotes with timestamps; clearly label citations as External.
   - Guardrail: prefer on‑site sources; include External only when relevant.

4) Product pages ingestion and UX
   - Ingest product pages at scale; store key attributes and image URLs.
   - Slack: render concise product blocks (title/specs/price/link + small image within limits).
   - Plan a lightweight Web UI (cards/galleries/comparisons) for richer presentation later.

5) Intent‑driven Internet search (guardrailed)
   - RACEN decides to search externally only on specific intents (comparison/reputation/news) and only after on‑site retrieval is low‑confidence.
   - Use domain allowlist/denylist, strict token/time caps, cache results, and clear External citations.
   - Evaluate `ottomator-agents-main/pydantic-ai-advanced-researcher` as a web‑research subagent with our adapter; alternatives possible if simpler.

## Explicit Decisions
- Embeddings: OpenAI `text-embedding-3-small` (1536-d); DB vector(1536); disable local 256-d providers.
- Slack preset: default to full-site for internal testing; scope can be narrowed per intent later.
- Settings tracking: `.env` is the source of truth; changes mirrored when Slack/app settings are adjusted.
- Persona config is env-driven (no hardcoding):
  - PERSONA_SYSTEM_PROMPT_PATH
  - PERSONA_LEXICON_PATH
  - PERSONA_TONE_TESTS_PATH
- Git policy: snapshot branches maintained; commit Answer API, scripts, tests, and README updates before feature jumps.
- Canonical planning/tracking files:
  - `docs/Progress_Report_2025-11-09.md` (this document) for phases, architecture and narrative status.
  - `TASK.md` for a compact checklist of active/completed tasks (IDs aligned with code/tests).

## Ingestion Strategy
- Targeted single-URL runs for clarity; verify chunks>0 after each.
- Keep ingestion local (no Drive mounts). Backup DB after batch completes.
- Scale to full-site after core pages validated via Slack.
- Utility scripts:
  - `scripts/bulk_ingest_from_yaml.py`: bulk-ingest URLs from `Grest_Data` YAML configs into Postgres/docling via `racen.orchestrator.ingest_url`.
  - `scripts/check_docling_urls_from_yaml.py`: sanity-check that all URLs from `Grest_Data` YAML configs exist in `docling.documents.source` and print any missing URLs.

## Operational Notes
- Keep Google Drive sync paused during heavy runs; unpause to sync reports/changes when idle.
- Local caches for HF/Transformers/Torch should stay on local disk (C:\ml-cache) to avoid Drive contention.
- Use uvicorn --reload during local API development; restart Slack bot after env/code changes.
- Run persona tone tests from Slack bot folder: `yarn persona:test`.

## Next Actions Checklist
DONE/Complete
- Ingest core pages (batch 1) one-by-one with `--dim 1536`; verify chunks>0 each.
- Validate Slack answers with citations from newly ingested pages.
DONE/Complete
- Implement best-effort fallback + follow-up suggestions.
- Persona v1 integrated (system prompt + lexicon + tone tests; Slack script).
Pending
- Fix remaining lints/formatting in `scripts/step4_answer.py` (imports/line length).
- Prepare PR: “RACEN Persona v1 — System Prompt + Lexicon + Reply Shaper”.
- Add Slack “Ingest URL” action/slash command; backend enqueues crawl_and_ingest and reports status.
- Optionally ingest Trustpilot page(s) for reputation queries (clearly labeled as External).
- Plan product pages ingestion and Slack-friendly rendering; outline Web UI for richer cards/images.
- Design intent‑driven Internet search with guardrails and integrate the researcher subagent if chosen.
 - Define and surface facet bundles for Shipping/Returns/Warranty (2–3 compact options on low-confidence and fallback turns).
 - Apply citation deduplication (collapse duplicate/near-duplicate sources); confidence remains in ribbon.
 - Add live-agent handoff stub (config + API shape) and wire escalation to call it behind a flag.

## Update – 2025-11-18 (Slack Ingest + Product Q&A)

### Completed Since Last Report
- **Slack ingest allowlist and 403 handling**
  - Backend: `scripts/answer_api.py` exposes `/ingest/url` with `INGEST_ALLOWED_USERS` env-driven allowlist.
  - Slack bot: `slack-openai-bot/app.js` checks for HTTP 403 from backend; on 403, sends a direct "not authorized to ingest" DM and does not start status polling.
  - Behavior verified with two Slack accounts (allowed vs non-allowed) and logs.

- **Product Q&A robustness (MacBook / iPhone flows)**
  - Product intent detection improved in `scripts/step4_answer.py` for device queries (MacBook/iPhone/laptop/iPad) including Hinglish variants.
  - Follow-up logic now biases retrieval toward the same product family mentioned in the previous answer (e.g., MacBook vs iPhone) rather than drifting to unrelated products.
  - Graceful product fallback implemented for non-existent models (e.g., iPhone 10):
    - Returns a clean clarification asking for the exact model, without leaking noisy catalog text (vitamins/quiz etc.).
    - Dedicated fallback path for `intent=product` avoids including low-signal snippets in the reply.

- **Domain config and tests**
  - New YAML config: `domain_product_families.yaml` with product families (iphone, macbook, laptop, ipad) and their keywords.
  - `step4_answer.py` now loads `PRODUCT_FAMILIES` from YAML for:
    - Product intent detection.
    - Product follow-up bias toward the same family.
  - Pytest suite added: `tests/test_step4_answer.py` covering:
    - Intent detection for core queries (MacBook/iPhone/shipping/returns).
    - Last-intent inference from previous answers.
    - Product follow-up anchoring to the previous product family.
    - Product fallback behavior (no noisy snippets; clean clarification).
    - Early category-style handling for queries like "how about macbooks?" (retrieval query includes family keyword).

### Phase Framing (Internal)
- **Phase 1 (by 2025-11-30 – internal demo)**
  - Goal: Solid, grounded catalog Q&A and ingest for internal Slack and a minimal Web UI.
  - Scope (backend):
    - Exact product queries for iPhones/MacBooks/laptops (availability, key specs, stock status).
    - Graceful fallback for non-existent models (no hallucinated catalog items).
    - Ingest allowlist and status polling in Slack.
    - Hinglish mirroring and best-effort fallback retained from earlier work.
  - Scope (frontend):
    - Slack bot as primary interface.
    - A small Web UI on the website that calls the same `/answer` API (single chat box, grounded answers, optional source links).

- **Phase 2 (work-in-progress, not shown in first demo)**
  - ACK refinement: treat "ok how about X" with a different product family as a new topic instead of a pure acknowledgement.
  - Category/browse behavior for product families:
    - Queries like "any macbooks you have?", "iphones?" list a small set of products and ask the user which one they want details on.
  - "Latest model you have" within catalog (rank among ingested products rather than global web knowledge).
  - Intent-driven external comparison (later): use the n8n + Brave web researcher agent for cross-model comparisons (e.g., iPhone 13 vs iPhone 14) under strict guardrails and clear External labeling.

- **Phase 1.5 – MVP refinement in current repo (in progress)**
  - Tighten product behavior without breaking the existing API:
    - Robust product intent detection (e.g., "do you have X", Android/Apple phones, headphones/accessories).
    - Qualifier-aware answers: respect words like "retina", year, size, storage/RAM; when no exact variant exists, say so clearly and suggest alternatives.
    - Define an "iPhone-only" test surface: ingest all iPhone product URLs from grest.in (via YAML allowlist + Slack ingest) and validate product behavior and fallbacks primarily on this family before expanding.
  - Introduce an `intent_classifier` abstraction inside Python while keeping external behavior stable:
    - Wrap existing `_detect_intent` / `_infer_last_intent` logic behind a small classifier interface.
    - Add tests around common flows (product/returns/shipping/contact/general/unclear).
  - Add explicit "unclear intent" fallback:
    - For noisy or ambiguous queries, return a short, honest rephrase request instead of random policy snippets.
  - Round out category/browse flows for current families (MacBook/iPhone/iPad/laptop) using the YAML config.

- **Phase 3 – MNC-grade evolution roadmap (design + later implementation)**
  - Product- and DB-aware retrieval:
    - Define a `product_search` abstraction that can evolve from pure RAG over `/products/*` to DB+RAG (using structured attributes like brand, family, model, RAM/storage, platform).
    - Keep this behind a stable function boundary so the Slack bot and `/answer` API do not change.
  - Stronger intent classification:
    - Evolve from heuristics to a dedicated intent classifier (LLM-based first; fine-tuned later when enough data is available).
    - Support future categories (Android phones, headphones/accessories, etc.) without per-product code changes.
  - Multi-category and multi-brand support:
    - Ensure the design can handle arbitrary new product lines (e.g., mid-range Android, accessories) by relying on DB schema and generic qualifiers, not hardcoded names.
  - Multi-format ingestion:
    - Extend the ingestion pipeline to support non-HTML sources (PDF, Word, TXT) by adding a loader layer that normalizes each format to markdown before chunking/embedding, so RACEN can answer from richer GREST docs beyond web pages.
  - Guardrails and UX:
    - Preserve the current tested behaviors (no hallucinated catalog items, clear fallbacks, citations) while gradually adding structure.

These phase definitions keep the current Slack + Answer API MVP on track while explicitly documenting how we will evolve toward a more DB-aware, classifier-driven, multi-category product without throwing away the existing codebase.

### Session / Context Strategy – 2025-11-19

- Slack (current behavior):
  - The bot sees Slack `user`, `channel`, and `thread_ts` and tracks context per thread in memory (last answer, fallback counts, last intent), but does not persist long-lived per-user profiles.
- Near-term plan (Phase 1.x):
  - Keep context light and thread-focused; avoid heavy persistent memory.
  - Optionally derive a simple `family_hint` (e.g., `iphone`, `macbook`) from the last answer and pass it into `product_search` so follow-ups like "what about 128 GB?" stay on the same product family.
- Web UI + future personalization:
  - When the Web UI is introduced, add a small per-user/session store (e.g., Redis or DB-backed sessions):
    - Key: session ID or logged-in GREST user ID.
    - Values: last product family, last selected product, language preference, etc.
    - Sessions expire on logout or inactivity so data is wiped automatically.
  - `product_search` is designed so it can accept an optional context hint (such as `family_hint`) later, fed from this session layer, without changing its core interface.
- Principle:
  - For Phase 1.x, keep RACEN mostly stateless with minimal, explicit context hints.
  - Defer full per-user personalization and Redis integration until the Web UI phase.

## Update – 2025-11-20 (Product Specs, Intent Routing, and E2E Tests)

### Completed Since Last Update

- **Catalog-backed product specs from Grest pages**
  - Implemented a direct product-specs path that:
    - Uses the iPhone product catalog (YAML) to identify canonical `/products/*` URLs.
    - Fetches corresponding chunks from Postgres and runs `extract_product_specs` to build structured specs (price, storage, condition, warranty, colors).
    - Wires these specs into `answer_query` only for clear product intents so price/spec answers are stable even when retrieval ranking changes.

- **Domain intent and source-bucket routing**
  - Introduced a small domain classifier for ambiguous product-like queries to distinguish:
    - `product_specs` vs `buying_advice` vs `brand_reputation` vs `generic_support`.
  - Updated `answer_query` so:
    - Product-like queries with no exact catalog match can be re-routed to blogs/FAQs (`buying_advice`) or Trust/reputation sources (`brand_reputation`) instead of falling back to noisy product flows.
    - Retrieval allowlists are expanded per-domain (e.g., `/blogs/news/`, `/pages/faqs/`), keeping product specs logic isolated to true product queries.

- **Query-flow end-to-end tests using real corpus**
  - Added pytest-based E2E tests that call `answer_query` end-to-end (DB + retrieval + LLM) for:
    - Product queries (e.g., iPhone 11) ensuring specs (price, storage) are surfaced from real pages.
    - Shipping queries hitting `/pages/shipping` / `/policies/shipping/policy`.
    - Warranty queries hitting `/pages/warranty`.
    - Generic/blog buying-advice queries hitting `/blogs/news/*`.
  - These tests no longer use dummy chunks; they verify real URLs and answer snippets from the ingested Grest corpus.

- **Re-ingestion overwrite semantics**
  - Updated the orchestrator ingest flow so re-ingesting a URL deletes the old document + chunks + embeddings before inserting new ones.
  - Added tests to confirm overwrite behavior so RACEN always answers from the latest page content.

- **Citation visibility in Answer API ribbon (debug)**
  - Extended the Answer API `/answer` response when `ANSWER_DEBUG_FLAGS=1` to include a compact `cits=` field in the settings ribbon:
    - Shows the top 3 citation URLs used in the answer.
    - Allows Slack and CLI to display which sources were used without changing frontend logic.
  - This is enabled only in debug mode and can be turned off later for production UIs.

### Updated Next Actions (Phase 1 – next 4–5 days)

1) **Trustpilot / Brand Reputation (Phase 1.1)**
   - Ingest the Grest Trustpilot profile into Postgres/pgvector as an External source.
   - Extend domain routing so `brand_reputation` queries (e.g., "why should I buy from Grest?", "are you trustworthy?", "what is your rating?"):
     - Prefer Trustpilot content + key on-site pages (e.g., about/why-Grest), with clear External labeling.
   - Add tests to verify:
     - Trustpilot URLs appear in citations for reputation queries.
     - Answers summarize rating, review volume, and a small number of representative quotes.

2) **Guardrailed Web Comparison via Advanced Web Researcher (Phase 1.2)**
   - Integrate the advanced web researcher from `ottomator-agents-main/advanced-web-researcher` (Brave search API) as a narrow, internal web-search tool.
   - Only RACEN decides when to call web search, based on semantic patterns such as:
     - "difference between iphone 14 and iphone 17"
     - "iphone 14 vs iphone 15 which is better"
   - Guardrails:
     - Users cannot directly command a web search; the backend chooses to use it only for specific comparison/reputation-style intents.
     - Answers clearly label web-sourced content as External and include citations.
   - Add tests with the web client mocked to ensure:
     - Comparison intents trigger a single web-search call.
     - Answers mention both models and surface the stubbed External URLs as citations.

3) **Minimal RACEN Web UI Embedded on GREST (Phase 1.3)**
   - Build a minimal web UI (single chat box with history) that calls the existing `/answer` API:
     - Show answers in a simple chat layout.
     - Optionally show the settings ribbon and/or citations in a developer/debug view.
   - Plan how to embed this UI into the Grest website (e.g., dedicated page or embedded widget) for the Phase 1 demo.
   - Smoke tests:
     - Verify the Web UI can run the same core flows as Slack (product Q&A, policy questions, Trustpilot reputation, and one comparison query).
