# RACEN MCP Integration – Progress Report

Date: 2025-11-09
Time: 19:35 IST (UTC+05:30)

## Scope
- 5 curated GREST pages; hybrid retrieval with citations and latency tracking.
- Goal: ≥95% accuracy with typical p90 latency ≤10s.

## Current Pipeline (Implemented)
- **Crawl:** Local Crawl4AI-based scripts
  - scripts/crawl_urls.py, scripts/fetch_known_urls_with_crawl4ai.py, scripts/sitemap_seed.py, scripts/legacy_bfs.py
- **Convert (HTML→Markdown):** Local MarkItDown path
  - scripts/convert_markdown.py; also reuse Crawl4AI markdown output
- **Clean + Chunk:** Basic cleaner+chunker
  - src/racen/step2_ingest.py (Cleaner, Chunker)
  - step2_process_markdown.py writes chunks with start_line/end_line
- **Embeddings (we will stick to this):** OpenAI text-embedding-3-small (1536d)
  - src/racen/step2_embed.py (OpenAIEmbedder) for both ingest and query
  - DB schema vector(1536)
- **Store:** Postgres + pgvector
  - src/racen/step2_write.py (ensure_schema, upserts)
- **Retrieval + Reranker:** Custom hybrid + bounded reranker
  - src/racen/step3_retrieve.py
  - Hybrid: pgvector cosine + Postgres FTS + fusion
  - Bounded CrossEncoder reranker (msmarco MiniLM) with RERANK_TOP_N
  - FAST_MODE available (disabled in baseline)
- **Answer Synthesis:** OpenAI chat w/ citations
  - scripts/step4_answer.py, short-answer mode with context/token caps
- **Evaluation:** Free-form harness with latency
  - scripts/eval_freeform.py (per-question latency, report avg/p50/p90, batching)

## Config used for Baseline (Balanced+Short)
- Retrieval: k=6, RERANK_TOP_N=12
- Answering: ANSWER_SHORT=1, ANSWER_CHUNK_CHAR_BUDGET=1200, ANSWER_MAX_TOKENS=120

## Results (Balanced+Short, all batches)
- Batch1 (0–49): Accuracy 82.0% | Latency avg=11.2s, p50=7.8s, p90=9.2s
- Batch2 (50–99): Accuracy 92.0% | Latency avg=10.9s, p50=7.9s, p90=9.0s
- Batch3 (100–149): Accuracy 96.0% | Latency avg=10.7s, p50=7.8s, p90=9.8s
- Batch4 (150–199): Accuracy 90.0% | Latency avg=10.7s, p50=7.8s, p90=9.2s
- Batch5 (200–249): Accuracy 94.0% | Latency avg=11.5s, p50=8.4s, p90=10.9s
- Batch6 (250–299): Accuracy 94.0% | Latency avg=11.8s, p50=8.1s, p90=10.4s
- Batch7 (300–315): Accuracy 87.5% | Latency avg=18.4s, p50=8.2s, p90=34.8s (small tail set; one outlier)

- Overall: 288/316 = **91.1% accuracy**
- Latency: Typical p50 ~7.7–8.4s; typical p90 ~9.0–10.9s (outlier in Batch7 only)

## What’s Planned Next
- Integrate **Docling** clean+chunk (structure-aware) into ingest via `DOC_CLEANER=docling` switch.
- Re-ingest the 5 pages using Docling mode, then re-run the same evaluation suite.
- Target: push accuracy toward **≥95%** without exceeding p90 ~10s.

## Explicit Decisions
- **Embeddings:** We will continue using OpenAI `text-embedding-3-small` (1536-d). No switch to pydantic/Mistral embeddings at this time.
- **Git policy:** After Docling integration, do not commit/backup until user reviews results and explicitly approves.

## DB Strategy for Docling Re-ingest
- Options:
  - **New DB (recommended):** Create a new database (e.g., `racen_docling`) or a new schema (e.g., `docling`) to keep old baseline intact. Switch `DATABASE_URL` for Docling runs.
  - **Truncate existing DB:** Simpler, but removes the old baseline—harder to revert.
- Recommendation: Use a **new DB or schema** to avoid mixing chunking strategies and preserve the current 91.1% baseline for comparison/revert.

## Operational Notes
- Keep Google Drive sync paused during heavy runs; unpause to sync reports/changes when idle.
- Local caches for HF/Transformers/Torch should stay on local disk (C:\ml-cache) to avoid Drive contention.

## Next Actions Checklist
- Configure Docling mode via env (`DOC_CLEANER=docling`, `DOC_CHUNK_MAX_TOKENS`, `DOC_CHUNK_OVERLAP`).
- Re-ingest curated 5 pages into a new DB/schema.
- Run full balanced+short evaluation suite and aggregate results.
- Present side-by-side accuracy/latency: Baseline vs Docling.
- Await explicit approval before any git backup.
