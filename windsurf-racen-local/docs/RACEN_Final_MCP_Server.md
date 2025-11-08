# RACEN Final MCP-Server

# **MCP shortlist for RACEN (TradeIndia)**

Grounded in your repo and the TreasureAgents catalog you shared, here’s a practical, production-leaning set of MCPs to replace weak spots in the current pipeline without a rewrite.

## **Step 1 — Crawl & Convert**

- **Crawling: crawl4AI-agent-v2**
    - Why: Purpose-built doc crawler; sitemap ingestion; chunking awareness; better robustness than ad-hoc BFS.
    - Use as the primary page discovery. Keep our include/exclude and rate-limits configurable.
    - Alt: crawl4AI-agent (older variant) or multi-page-scraper-agent if you need lighter flows (n8n).
- **HTML → Markdown conversion: ottomarkdown-agent or Microsoft MarkItDown**
    - ottomarkdown-agent: File→Markdown API + error handling; good for heterogeneous inputs.
    - Microsoft MarkItDown: Best-in-class fidelity; preferred when content is HTML/JS-heavy.
    - Recommendation: Use MarkItDown as primary converter MCP; fall back to ottomarkdown-agent; final fallback = local markdownify (already in repo).
- Optional extraction: docling-rag-agent
    - If TradeIndia pages are semi-structured (product/FAQ/policies), docling provides structure-aware parsing that improves chunk quality dramatically.

## **Step 2 — Clean + Chunk + Embed + Write**

- **Cleaner + Chunker: docling-rag-agent**
    - Why: Better structural segmentation (headings, lists, tables), fewer noise tokens. Preserves hierarchy for metadata-aware retrieval and reranking.
    - Configure windowed sentence chunks with overlaps (e.g., ~350–500 tokens with 10–15% overlap) and persist title/h1–h3/url_path.
- **Embeddings: pydantic-ai-mcp-agent (as an embedding MCP wrapper)**
    - Set provider = Mistral; model = **`mistral-embed`**; dim = 1024.
    - If a direct Mistral-embed MCP is not present, keep embeddings in our code (small adapter) while other steps leverage MCPs. This avoids blocking.
- **Store: keep our Postgres/pgvector writer**
    - Rationale: It’s already stable and wired. MCP for pgvector is optional; not necessary to gain retrieval quality.

## **Step 3 — Retrieve (Hybrid + Rerank)**

- **Hybrid retrieval MCP: foundational-rag-agent or all-rag-strategies**
    - Look for an MCP that:
        - Combines lexical (BM25/FTS/pg_trgm) + vector (pgvector) with rank fusion (RRF) or learned weights.
        - Accepts query text and returns reranked chunks with citations.
    - If none with Postgres backends, keep our DB and:
        - Expose a retrieval MCP that calls:
            - Lexical: Postgres FTS/pg_trgm with weighted fields (title>h1>h2>body).
            - Vector: pgvector with HNSW index.
            - Rank fusion: simple RRF or weighted normalize.
            - Cross-encoder rerank: sentence-transformers MCP on top 50–100.
- **Cross-encoder Reranker MCP: all-rag-strategies (re-ranking section)**
    - Use a small, efficient cross-encoder (e.g., msmarco MiniLM) to rerank top-K candidates.
    - This is the biggest quality win. Make it an MCP sidecar so Slack and future channels can reuse it.

## **Step 4 — Channel agent (Slack) [Parked]**

- Your Slack bot → calls Retriever MCP → calls Reranker MCP → formats answer + citations → applies refusal policy.

# **Rationale and expected improvements**

- **Conversion + cleaning** upgrades (MarkItDown/docling) reduce boilerplate/noise and produce better, self-contained chunks.
- **Metadata-rich chunking** (titles/headers/breadcrumbs) boosts lexical relevance and reranking quality.
- **Hybrid search with proper lexical** (BM25/pg_trgm) massively improves recall vs LIKE/token-overlap.
- **Cross-encoder reranking** improves precision: better top-3 even with noisy corpora.
- **HNSW vector index** increases recall at low latency.

# **Integration plan (minimal code movement, MCP-first)**

- **Step 1**:
    - Orchestrator: call crawl4AI-agent-v2 MCP to get URLs.
    - For each URL: call MarkItDown MCP; fallback ottomarkdown; fallback local.
- **Step 2**:
    - Send Markdown to docling MCP for clean+chunk; persist chunk metadata (title/h1–h3/url).
    - Embed via Mistral (MCP wrapper or local embedder adapter).
    - Write via existing Postgres code (unchanged).
- **Step 3**:
    - Replace **`retrieve()`** with a thin client that calls Retriever MCP.
    - Retriever MCP wraps: Postgres FTS/pg_trgm + pgvector (HNSW) + cross-encoder rerank.
    - Keep current local retriever as a fallback for dev/offline.

# **Configuration notes**

- **Mistral**: **`MISTRAL_API_KEY`**, **`MISTRAL_EMBED_MODEL=mistral-embed`**, **`EMBED_DIM=1024`**.
- **Postgres**: keep **`racen_tradeindia`** and add HNSW index:
    - **`CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);`**
    - Use **`SET hnsw.ef_search = 64;`** at query time or session default.
- **Lexical**: Add FTS/pg_trgm and store **`tsvector`** with weighted fields.
- **Refusal**: Implement threshold after rerank; always cite sources.

# **Validation plan (fast)**

- **Smoke ingest**: 5–10 TradeIndia pages.
- **Hybrid retrieve**: 10–20 handpicked queries (policies/FAQ/product-category).
- **Metrics**: nDCG5, MRR5; manual spot checks of citations.

# **What I’ll read next (to confirm endpoints and configs)**

- crawl4AI-agent-v2 README: features, API routes, sitemap handling, throttling.
- ottomarkdown-agent README: API path and auth; throughput limits.
- docling-rag-agent README: ingest API, chunk configs, metadata outputs.
- foundational-rag-agent and all-rag-strategies README: do they expose a retriever MCP or only examples? If not, I’ll compose a retriever MCP from building blocks (FTS+pgvector+rerank).
- pydantic-ai-mcp-agent README: whether it exposes embedders; how to bind to Mistral.

If you want, I can proceed to deep-dive each of those READMEs and return:

- API contracts and env keys.
- A connection diagram for RACEN → MCPs.
- A short step-by-step to wire our orchestrator to those MCPs, keeping our DB writer intact.

# **Final picks (initial)**

- **Crawler**: crawl4AI-agent-v2
- **Convert**: MarkItDown (Microsoft), fallback ottomarkdown-agent
- **Clean/Chunk**: docling-rag-agent
- **Embeddings**: Mistral via pydantic-ai-mcp-agent (or local adapter)
- **Hybrid/Rerank**: Retriever MCP composed of Postgres FTS + pgvector (HNSW) + cross-encoder reranker (from all-rag-strategies)
- **Slack**: your bot, later.

# **Next step**

Approve me to:

- Extract the exact API endpoints/env for the five agents above and produce a concrete wiring doc for our repo (no code changes yet).
- If you have constraints (GPU availability for reranker, self-host-only), tell me so I can narrow model choices.

Status: Reviewed your repo and TreasureAgents catalog; proposed best-fit MCPs and integration approach. Awaiting approval to finalize API contracts and exact wiring steps.
