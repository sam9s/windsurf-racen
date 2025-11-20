# RACEN – Task Tracker

## Canonical Tracking
- This file and `docs/Progress_Report_2025-11-09.md` are the **only canonical trackers** for RACEN work.
- Update this file whenever you add/complete work items.

## Active Tasks

### Product Q&A / Slack
- [ ] Add category/browse behavior for product families (e.g., macbooks/iphones) so RACEN can list products. (ID: answer-category-browse)
- [ ] Tighten product matching so qualifiers like "retina", year, and size are respected and we don’t show a non-matching model as if it were exact. (ID: answer-product-qualifier-matching)
- [ ] Implement Planner → Extractor → Phraser pipeline for product answers, starting with the full Grest iPhone catalog (all current iPhone product URLs) as the template set, so RACEN can deterministically pick primary + variant products and surface exact specs/prices from pages. (ID: answer-product-pipeline-plan)

### Trust & Reputation
- [ ] Ingest the Grest Trustpilot profile and route brand reputation queries to Trustpilot (clearly labeled as External) plus key on-site pages. (ID: answer-trustpilot-brand-reputation)

### External Web Search
- [ ] Integrate the advanced web researcher (Brave search API) from `ottomator-agents-main/advanced-web-researcher` for comparison-style queries (e.g., "iphone 14 vs iphone 17") behind strict guardrails so only RACEN decides when to call web search, never the user. (ID: answer-web-comparison-guardrailed)

### Web UI
- [ ] Build a minimal RACEN Web UI (single chat box using the /answer API) and plan how to embed it on the GREST website for the Phase 1 demo. (ID: answer-racen-web-ui-phase1)

### Architecture / Search
- [ ] Review product DB/schema and plan a future product_search abstraction that can be swapped from pure RAG to DB+RAG later. (ID: answer-product-search-architecture)
- [ ] Design a lightweight per-user/session context strategy (Slack per-thread state now, Redis/DB-backed sessions later with Web UI) and define how it feeds into intent handling and product_search without overcomplicating Phase 1.x. (ID: answer-session-context-design)

### Ingestion & Formats
- [ ] Design multi-format ingestion (PDF, Word, TXT, etc.) for GREST docs, update plan/progress report, and prepare code changes to support it. (ID: answer-multiformat-ingestion-plan)

## Completed (for reference only)
- [x] Externalize product/domain nouns into YAML config for `answer_query`. (ID: answer-domain-config-yaml)
- [x] Add explicit 'unclear intent' graceful fallback path for noisy/ambiguous queries. (ID: answer-unclear-intent-fallback)
- [x] Design and implement an intent_classifier abstraction with tests, without breaking current API. (ID: answer-intent-classifier-abstraction)
- [x] Implement catalog-backed product specs extraction from Grest product pages (via Postgres chunks) and wire it into `answer_query` only for product intents. (ID: answer-product-specs-from-pages)
- [x] Implement domain intent and source-bucket routing so product, policy, blog/FAQ, and brand-reputation style queries retrieve from the right sources without hardcoded phrases. (ID: answer-domain-intent-routing)
- [x] Add end-to-end pytest flows for product, shipping, warranty, and generic/blog buying-advice queries using the real ingested corpus. (ID: answer-e2e-query-flows)
- [x] Ensure re-ingestion of a URL overwrites its previous document, chunks, and embeddings so the latest page content is always used. (ID: ingest-url-overwrite)
## Sample Flow - Finilized (example)
- A. 
    Query for base model, e.g. “iphone 16”
    If base model exists:
    Primary product = base 16 (if present).
    Also mention other variants with same base (16 Pro, 16 Pro Max) as alternatives.
    If base 16 doesn’t exist but variants do (16 Pro, 16 Plus, etc.):
    Say: “We don’t currently have the plain iPhone 16, but we do have these variants…”
    List those variants (names + URLs).
- B. 
    Query for a specific variant, e.g. “iphone 16 pro”
    If that specific variant exists:
    Focus on that exact product (16 Pro), with full details + URL.
    Optionally mention siblings (16, 16 Pro Max) as alternatives.
    If that variant doesn’t exist but same base has others:
    Say: “We don’t have iPhone 16 Pro, but we do have iPhone 16 / 16 Plus…”
Then describe the closest one(s) with real details + URLs.

