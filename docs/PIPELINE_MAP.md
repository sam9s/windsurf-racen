# RACEN Pipeline Mental Map

This document maps the current end-to-end flow, modules, data handoffs, and where MCP services can replace or augment components.

## High-level stages

1) Crawl & Discover (pages list)
- Module: `src/racen/crawler.py` (CLI: `scripts/crawl_and_ingest.py`)
- Responsibility:
  - Start from `start_url`, BFS crawl within domain
  - Apply include/exclude filters, rate limiting
  - Seed via sitemap if available, handle origin URL always
- Output: Ordered list of page URLs

2) Fetch & Convert (HTML -> Markdown)
- Module: `src/racen/step1_fetch_convert.py`
  - Class: `MarkItDownClient`
  - Default: fetch HTML via `requests`, convert via `markdownify`
  - Optional MCP hook: if `MARKITDOWN_HTTP_URL` is set, POST URL/HTML to MCP and use its Markdown; fallback to local conversion on error
- Output: Markdown string for each URL

3) Clean & Chunk (prepare text)
- Module: `src/racen/step2_ingest.py`
  - Class: `Cleaner`
    - Removes boilerplate comments/templates; collapses whitespace
    - (Room to expand: strip Shopify/JSON/script-like noise)
  - Class: `Chunker(max_tokens=400, overlap_tokens=40)`
    - Token-aware (~4 chars/token heuristic)
    - Splits large sections and long lines to stay under budget
- Output: List of `Chunk` objects (id, text, offsets)

4) Embed (vectorize chunks)
- Module: `src/racen/step2_embed.py`
  - Class: `OpenAIEmbedder(model=text-embedding-3-small)`
  - Loads `OPENAI_API_KEY` from env (.env in scripts)
  - Retries with exponential backoff for transient errors
  - Local deterministic fallback if key absent (for testing)
- Output: Embedding vectors per chunk

5) Write to DB (Postgres + pgvector)
- Module: `src/racen/step2_write.py`
  - Tables: `documents`, `chunks`, `embeddings`
  - `ensure_schema()` creates tables and `vector` extension
  - `upsert_document()`, `upsert_chunk()`, `upsert_embedding()`
  - `embedding_exists()` for dedupe
- Output: Persisted documents/chunks/embeddings

6) Orchestration (end-to-end for each URL)
- Module: `src/racen/orchestrator.py`
  - Function: `ingest_url(url, embedding_dim=256)`
  - Pipeline: convert -> clean -> chunk -> write -> embed -> write
  - Stable chunk IDs (`doc_id:start:end:sha1(text)`) and skip-if-exists for embeddings to avoid re-billing

7) Retrieval (query-time)
- Module: `src/racen/step3_retrieve.py`
  - `retrieve(query_text, top_k)`
    - Embed query with OpenAI, detect DB vector dim, pad/trim to match
    - Vector search: pgvector `<=>` over `embeddings`
    - Lexical search: simple ILIKE + token-overlap score (naive)
    - Hybrid merge: normalize and combine vector + lexical
  - Returns: `RetrievedChunk` with scores and citations

8) Scripts (ops/testing)
- `scripts/crawl_and_ingest.py` — crawl then ingest all discovered URLs
- `scripts/ingest_url.py` — ingest a single URL
- `scripts/db_sanity.py` — counts of documents/chunks/embeddings
- `scripts/backfill_missing_embeddings.py` — fill embeddings for missing chunks
- `scripts/smoke_step3.py` — simple retrieval smoke test with example/support queries

## Data model (DB)
- `documents(id, source, created_at)`
- `chunks(id, document_id, start_char, end_char, text)`
- `embeddings(chunk_id, embedding VECTOR(256), model)`

## Environment and config
- `.env`: `OPENAI_API_KEY`, `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`, optional `MARKITDOWN_HTTP_URL`
- CLI flags (crawl): `--max-pages`, `--same-domain`, `--rps`, `--timeout`, `--retries`, `--include`, `--exclude`, `--dim`

## Current strengths
- Deterministic & resumable ingest (skip-if-exists, stable IDs)
- Resilient crawling and embedding with retries
- Simple, local retrieval stack (no external infra required)

## Current gaps (why results can look off)
- Cleaner is basic; Shopify/JSON/JS blobs can leak into text
- Lexical retrieval is naive (token overlap), not BM25/pg_trgm
- No domain/path steering for support-like queries by default

## Where MCP can plug in
- HTML->Markdown conversion (already optional):
  - `MARKITDOWN_HTTP_URL` for better fidelity & boilerplate handling
- Retrieval MCP (recommended for production):
  - Replace `step3_retrieve.retrieve()` call with an MCP client
  - Delegate lexical+vector+rerank to MCP service
  - Keep Postgres/pgvector as your store, or align to MCP’s index

## Options forward

- Option A (incremental):
  - Enhance `Cleaner` to strip Shopify/JSON/script noise
  - Add retrieval filters/boosts: `source_prefix` (e.g., grest.in), path-based boosts for `/policy,/policies,/faq,/page,/support` and penalties for `/products,/collections` on support queries
  - Enable MarkItDown MCP for conversion quality
  - Short top-up crawl over policy/FAQ pages to refresh embeddings
  - Pros: Fast; minimal external dependencies

- Option B (Retriever MCP):
  - Keep ingest as-is (or adapt) but call a professional retriever MCP for query-time search
  - Pros: Better ranking & maintenance offloaded; Cons: requires MCP setup/integration

- Option C (hybrid, recommended):
  - Do Option A tweaks now for immediate quality boost
  - Integrate MarkItDown MCP for conversion
  - Then wire a Retriever MCP and switch `retrieve()` to use it, keeping local retriever as fallback

## Suggested small improvements (if staying local for now)
- Cleaner: heuristic removal of lines with `window.`, `Shopify`, `variant=`, long escaped blobs, high punctuation ratio
- Retrieval: optional `source_prefix` and path boosts for support-like queries
- Lexical: consider upgrading to pg_trgm/BM25 later

## Flow diagram (text)

crawl (crawler.py)
  -> urls[]
for each url in urls:
  fetch (requests) + convert (MarkItDownClient) [MCP optional]
    -> markdown
  clean + chunk (Cleaner, Chunker)
    -> chunks[]
  write (documents, chunks)
  embed chunks (OpenAIEmbedder) [retry/backoff, skip-if-exists]
  write embeddings (pgvector)

retrieve(query)
  -> embed query (OpenAI)
  -> vector search (pgvector) + lexical search (ILIKE)
  -> hybrid merge -> top-k results with citations

MCP injection points:
- MarkItDown MCP (conversion)
- Retriever MCP (replace step3_retrieve)
