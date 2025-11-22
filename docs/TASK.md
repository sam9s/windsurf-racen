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
- [ ] Trustpilot/MouthShut URLs are already injected; ensure indexing/routing and clear External labeling; maintain tests for citations and summaries. (ID: answer-trustpilot-brand-reputation)

### External Web Search
- [ ] Use DuckDuckGo via SerpAPI for comparison-style queries under strict guardrails; only RACEN decides when to call it. (ID: answer-web-comparison-guardrailed)
- [ ] Enable web comparison path in Slack by wiring comparison-intent gating; keep CLI working. (ID: web-compare-slack-enable)
- [ ] Add mocked tests to verify single web-search call and External citations in answers. (ID: web-compare-tests)

### Web UI
- [ ] Build a minimal RACEN Web UI (single chat box using the /answer API) and plan how to embed it on the GREST website for the Phase 1 demo. (ID: answer-racen-web-ui-phase1)

### Mobile (PWA → TWA)
- [ ] Convert Web UI to PWA (manifest, icons, basic offline shell). (ID: mobile-pwa)
- [ ] Package PWA as Android Trusted Web Activity (TWA) and test installability. (ID: mobile-twa)
- [ ] Link back to grest.in and verify deep links/open-in-app behavior. (ID: mobile-deeplinks)

### Architecture / Search
- [ ] Review product DB/schema and plan a future product_search abstraction that can be swapped from pure RAG to DB+RAG later. (ID: answer-product-search-architecture)
- [ ] Design a lightweight per-user/session context strategy (Slack per-thread state now, Redis/DB-backed sessions later with Web UI) and define how it feeds into intent handling and product_search without overcomplicating Phase 1.x. (ID: answer-session-context-design)

### Ingestion & Formats
- [ ] Design multi-format ingestion (PDF, Word, TXT, etc.) for GREST docs, update plan/progress report, and prepare code changes to support it. (ID: answer-multiformat-ingestion-plan)

### Caching Optimization (deferred until after Web UI)
- [ ] Final‑answer TTL cache for safe flows (product, brand‑reputation); exclude unclear intent. (ID: cache-answer-ttl-safe)
- [ ] Cache comparison answers (TTL 12–24h) and add a web‑results subcache; label External sources clearly. (ID: cache-compare-ttl)
- [ ] Precompute and Redis‑cache iPhone family price/browse answers (cheapest/most‑expensive/under/between/all). (ID: cache-precompute-iphone-family)
- [ ] Add Postgres connection pooling (psycopg_pool) and Redis cache for product specs by URL to avoid repeated DB reads. (ID: cache-db-pool-specs)
- [ ] Add cache hit/miss instrumentation and per‑domain breakdown for retrieval and answer caches. (ID: cache-metrics)
- [ ] Tune `ANSWER_CHUNK_CHAR_BUDGET` (e.g., 800) and domain `top_k` (policy/support ≤5; brand ≤5; product ~6). (ID: cache-tune-chunk-topk)
- [ ] Prewarm the curated 51 queries on deploy and nightly to keep cache hot. (ID: cache-prewarm-51)
- Target (hot run): ≥49/51 under 5s; ≤2 between 5–7s. (ID: cache-latency-target)

## Completed (for reference only)
- [x] Externalize product/domain nouns into YAML config for `answer_query`. (ID: answer-domain-config-yaml)
- [x] Add explicit 'unclear intent' graceful fallback path for noisy/ambiguous queries. (ID: answer-unclear-intent-fallback)
- [x] Design and implement an intent_classifier abstraction with tests, without breaking current API. (ID: answer-intent-classifier-abstraction)
- [x] Implement catalog-backed product specs extraction from Grest product pages (via Postgres chunks) and wire it into `answer_query` only for product intents. (ID: answer-product-specs-from-pages)
- [x] Implement domain intent and source-bucket routing so product, policy, blog/FAQ, and brand-reputation style queries retrieve from the right sources without hardcoded phrases. (ID: answer-domain-intent-routing)
- [x] Add end-to-end pytest flows for product, shipping, warranty, and generic/blog buying-advice queries using the real ingested corpus. (ID: answer-e2e-query-flows)
- [x] Ensure re-ingestion of a URL overwrites its previous document, chunks, and embeddings so the latest page content is always used. (ID: ingest-url-overwrite)
- [x] Implement deterministic iPhone family browse and price-range listing flows (cheapest / most expensive / under / between) backed by the catalog and wired into `answer_query` and Slack. (ID: answer-iphone-family-price-flows)
- [x] Add pytest coverage for iPhone family/price queries (cheapest, most expensive, under 50k, between 20k–50k, and family browse) using the real corpus. (ID: answer-iphone-family-price-tests)
- [x] Clean up Slack product link rendering so multi-product family answers no longer append an extra generic "Product page" link when bullets or the collection URL are already present. (ID: slack-product-link-cleanup)
- [x] Update unit tests to expect the static no-answer fallback wording for product queries. (ID: tests-fallback-static-message)
- [x] Relax Mouthshut citation assertion to accept any valid Mouthshut Grest reviews URL. (ID: tests-brand-rep-mouthshut-assert)
- [x] Run 51-query cache benchmark (cold + hot) and write results to tests/queries/grest_cache_benchmark_results.md; overall speedup x1.74 on 51 queries. (ID: cache-benchmark-51-queries)
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

