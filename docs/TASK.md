# RACEN – Task Tracker

## Canonical Tracking
- This file and `docs/Progress_Report_2025-11-09.md` are the **only canonical trackers** for RACEN work.
- Update this file whenever you add/complete work items.

## Active Tasks

### Product Q&A / Slack
- [ ] Add category/browse behavior for product families (e.g., macbooks/iphones) so RACEN can list products. (ID: answer-category-browse)
- [ ] Tighten product matching so qualifiers like "retina", year, and size are respected and we don’t show a non-matching model as if it were exact. (ID: answer-product-qualifier-matching)
- [x] Define and ingest a clean, comprehensive allowlist of all iPhone product pages on grest.in and use that as the primary test surface for product behavior. (ID: answer-iphone-ingest-surface)

### Architecture / Search
- [ ] Review product DB/schema and plan a future product_search abstraction that can be swapped from pure RAG to DB+RAG later. (ID: answer-product-search-architecture)
- [ ] Design a lightweight per-user/session context strategy (Slack per-thread state now, Redis/DB-backed sessions later with Web UI) and define how it feeds into intent handling and product_search without overcomplicating Phase 1.x. (ID: answer-session-context-design)

### Ingestion & Formats
- [ ] Design multi-format ingestion (PDF, Word, TXT, etc.) for GREST docs, update plan/progress report, and prepare code changes to support it. (ID: answer-multiformat-ingestion-plan)

## Completed (for reference only)
- [x] Externalize product/domain nouns into YAML config for `answer_query`. (ID: answer-domain-config-yaml)
- [x] Add explicit 'unclear intent' graceful fallback path for noisy/ambiguous queries. (ID: answer-unclear-intent-fallback)
- [x] Design and implement an intent_classifier abstraction with tests, without breaking current API. (ID: answer-intent-classifier-abstraction)
