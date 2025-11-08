# windsurf-racen-local

Local-first RACEN pipeline orchestrating MCP servers via Docker (Windows). This repo follows the plan in docs/RACEN_Final_MCP_Server.md and treats existing local code as fallback.

## Stack
- Crawl: crawl4AI-agent-v2 (Docker, local)
- Convert: Microsoft MarkItDown MCP (Docker, local)
- Clean/Chunk: docling-rag-agent (Docker, local)
- Embeddings: pydantic-ai-mcp-agent wrapping provider (OpenAI 1536d) (Docker, local)
- Store: Postgres/pgvector (existing DB)
- Retrieve + Rerank: all-rag-strategies (Docker, local)

## Quickstart
1) Copy .env.example to .env and fill values
2) Start MCP services locally
   - docker compose -f compose/docker-compose.local.yml up -d
3) Run smoke tests (to be added under scripts/) to validate each endpoint
4) Run pipeline ingestion and retrieval

## Notes
- All services are bound to localhost so no external ports are exposed beyond the host.
- We keep local fallbacks in code; MCP-first orchestration is the default path.
