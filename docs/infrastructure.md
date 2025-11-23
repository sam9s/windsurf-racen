# RACEN Infrastructure Snapshot (Local Dev)

## 1. Core Services & Ports

- **Answer API (FastAPI / uvicorn)**
  - Module: `scripts.answer_api:app`
  - Default port (when run directly): `8000`
    - Command pattern: `uvicorn scripts.answer_api:app --reload --port 8000`
    - In `answer_api.py`: `port = int(os.getenv("PORT", "8000"))`
  - Current **Slack + Web UI** usage (per env / your setup):
    - You run the Answer API on **port `8011`** for Slack + Web UI.

- **Slack Bot (Socket Mode)**
  - Location: `Grest_RACEN_Slack_Bot/slack-openai-bot/app.js`
  - Starts on an internal port (Socket Mode, no public HTTP required):
    - `app.start(process.env.PORT || 3000)`
  - Connects to Slack using:
    - `SLACK_BOT_TOKEN`
    - `SLACK_APP_TOKEN`
    - `SLACK_SIGNING_SECRET`
  - Loads its env from: `Grest_RACEN_Slack_Bot/.env`.

- **Web UI (Next.js / React)**
  - Location: `racen-webui` (project folder at repo root).
  - Dev server: `npm run dev` → `http://localhost:3000`.
  - Uses Next.js rewrites to proxy API traffic to the Answer API (port aligned with your current Answer API, typically `8011`).

- **MCP / Helper Services (via Docker compose)**
  - Compose file: `windsurf-racen-local/compose/docker-compose.local.yml`.
  - Example enabled service:
    - `markitdown-mcp` at `http://localhost:8082` (host) → container `8080`.
  - Other services (crawl4ai, docling-rag-agent, embeddings-mcp, retriever-mcp) are present but commented out.

## 2. Environment Configuration (Backend)

- **Main backend env file**: `Windsurf_Project/.env`

### 2.1. MCP Endpoints

- `MARKITDOWN_HTTP_URL=http://localhost:8082`
- `CRAWL4AI_HTTP_URL=http://localhost:8081`
- `DOCLING_HTTP_URL=http://localhost:8083`
- `RETRIEVER_MCP_URL=http://localhost:8086`

### 2.2. Embeddings / LLM

- `OPENAI_API_KEY` (your key)
- `EMBED_DIM=1536`
- `EMBED_MODEL=text-embedding-3-small`

### 2.3. Database (Postgres)

- `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/racen`
- `PGHOST=localhost`
- `PGPORT=5432`
- `PGUSER=postgres`
- `PGPASSWORD=postgres`
- `PGDATABASE=racen`
- `PGOPTIONS=-c search_path=docling,public` (search path default schema order)

Code using this config:

- `src/racen/step2_write.py`:
  - `DBConfig.from_env()` reads `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD`.
  - `get_conn()` logs and connects to Postgres using the above.

### 2.4. Retrieval & Answer Shaping

Key knobs in `Windsurf_Project/.env`:

- Retrieval tuning:
  - `HNSW_EF_SEARCH=100`
  - `TOP_K=18`
- Retrieval controls:
  - `RERANK_TOP_N=14`
  - `FAST_MODE=1`
  - `RETRIEVE_BACKOFF_ENABLE=1`
  - `RETRIEVE_BACKOFF_THRESHOLD=0.35`
  - `RETRIEVE_BACKOFF_SECONDARY=/pages/faqs,/pages/returns-refund-cancellation`
- Answer shaping:
  - `ANSWER_SHORT=1`
  - `ANSWER_MAX_TOKENS=120`
  - `ANSWER_CHUNK_CHAR_BUDGET=1200`
  - `ANSWER_MATCH_INPUT_LANGUAGE=1`
  - `ANSWER_FOLLOWUPS_ENABLE=0`
  - `ANSWER_FALLBACK_ENABLE=1`
  - `ANSWER_TONE_AWARE=1`
  - `ANSWER_FALLBACK_GRACEFUL=1`
  - `ANSWER_PERSONA_WARMTH=1`
  - `ANSWER_LIMIT_FIRST_BUBBLE=0`
  - `ANSWER_DEBUG_FLAGS=1` (enables settings ribbon + extra debug in Answer API)
  - `ANSWER_LANGUAGE_LOCK=1`
  - `ANSWER_LANGUAGE_FORCE_REWRITE=1`

- Persona config:
  - `PERSONA_SYSTEM_PROMPT_PATH=...\Grest_RACEN_Slack_Bot\slack-openai-bot\Persona\system_prompt.md`
  - `PERSONA_LEXICON_PATH=...\Grest_RACEN_Slack_Bot\slack-openai-bot\Persona\lexicon.v1.yaml`
  - `PERSONA_EMOJI_LEVEL=2`

### 2.5. Support Contact (Backend)

- `SUPPORT_PHONE=+91 92665 22338`
- `SUPPORT_EMAIL=care@grest.in`
- `SUPPORT_ADDRESS=Radical Aftermarket Services Pvt. Ltd. ... Gurugram, Haryana-122003`

### 2.6. Web Comparison (SerpAPI)

- `SERPAPI_API_KEY=...` (your key)
- `ENABLE_WEB_COMPARISON=1`

### 2.7. Redis Cache (RACEN)

- `RACEN_CACHE_ENABLED=1`
- `RACEN_CACHE_PREFIX=racen:`
- `RACEN_ANSWER_CACHE_TTL_S=7200`
- `RACEN_RETRIEVAL_CACHE_TTL_S=21600`
- `RACEN_CACHE_SALT=dev1`
- `REDIS_URL=redis://127.0.0.1:6379/2`

Code using this config:

- `src/racen/cache.py`:
  - `get_cache()` reads `RACEN_CACHE_ENABLED` and `REDIS_URL`.
  - If Redis is reachable at `redis://127.0.0.1:6379/2`, uses it; otherwise falls back to in-memory cache.
  - Keys are prefixed with `RACEN_CACHE_PREFIX` and salted with `RACEN_CACHE_SALT`.

## 3. Answer API Details

- File: `scripts/answer_api.py`
- App: `app = FastAPI(title="RACEN Answer API", version="1.0.0")`

### 3.1. Endpoints

- `GET /health`
  - Returns `{ "status": "ok" }` for health checks.

- `POST /answer`
  - Request model: `AnswerRequest`
    - `question: str`
    - `allowlist: Optional[str]`
    - `k: Optional[int]`
    - `short: Optional[bool]`
    - `previous_answer: Optional[str]`
    - `previous_user: Optional[str]`
  - Response model: `AnswerResponse`
    - `answer: str`
    - `citations: List[CitationOut]`
    - `settings_summary: str` (debug ribbon)
  - Internally calls `scripts.step4_answer.answer_query`.

- `POST /ingest/url`
  - Enqueues ingestion via `racen.orchestrator.ingest_url`.
  - Enforces `INGEST_ALLOWED_USERS` by Slack user ID.

- `GET /ingest/status/{job_id}`
  - Returns ingestion job status.

### 3.2. How env is loaded

- On startup, `answer_api.py` loads the first available env file from:
  - `windsurf-racen-local/.env`, then
  - `Windsurf_Project/.env`.

## 4. Slack Bot Infrastructure

- Location: `Grest_RACEN_Slack_Bot/slack-openai-bot`
- Env file: `Grest_RACEN_Slack_Bot/.env`

### 4.1. Slack Bot Env (key items)

- Slack auth:
  - `SLACK_BOT_TOKEN=...`
  - `SLACK_APP_TOKEN=...`
  - `SLACK_SIGNING_SECRET=...`
- Behavior knobs:
  - `SLACK_ALLOWLIST_PRESET=all` (Slack-level preset, separate from backend default `faqs`)
  - `SLACK_SHOW_CITATIONS=0`
  - `OPENAI_MODEL=gpt-4o-mini`

- RACEN services (Slack view):
  - `RACEN_RETRIEVER_URL=http://racen-retriever:8000` (for future retriever service)
  - `RACEN_ANSWER_URL=http://127.0.0.1:8011`  
    - **Important**: This is the **base URL**, without `/answer`.
    - `app.js` calls `${ANSWER_URL.replace(/\/$/, "")}/answer` and `/ingest/...`.

- Retrieval controls (Slack perspective):
  - `RERANK_TOP_N=14`
  - `FAST_MODE=1`
  - `RETRIEVE_BACKOFF_ENABLE=1`
  - `RETRIEVE_BACKOFF_THRESHOLD=0.35`
  - `RETRIEVE_BACKOFF_SECONDARY=/pages/faqs`
  - `RETRIEVE_SOURCE_ALLOWLIST=` (empty → use preset logic in bot)

- Answer shaping (Slack perspective):
  - `ANSWER_SHORT=1`
  - `ANSWER_MAX_TOKENS=120`
  - `ANSWER_CHUNK_CHAR_BUDGET=1200`

- Support contact (Slack escalation):
  - `SUPPORT_PHONE=+91 92665 22338`
  - `SUPPORT_EMAIL=care@grest.in`
  - `SUPPORT_ADDRESS=Radical Aftermarket Services Pvt. Ltd. ... Gurugram, Haryana-122003`

- Web comparison comment:
  - Notes that web comparison is configured in the RACEN backend via `Windsurf_Project/.env`.

### 4.2. Slack Bot Behavior Overview

- Uses `@slack/bolt` `App` in Socket Mode (no public HTTP server).
- On `app_mention`:
  - Determines thread id (reuse recent thread per user+channel if fresh).
  - Computes `allowlist` from env or `SLACK_ALLOWLIST_PRESET`.
  - If a `grest.in` URL is present in the message, overrides allowlist to that exact path.
  - When `SLACK_THINKING_ENABLE` (default `1`) is on:
    - Posts `"Thinking…"` in the thread.
    - Updates that message with the final answer or `"Info not found"`.
  - Calls Answer API at `${RACEN_ANSWER_URL}/answer` with `k=18`, `short=true`, and previous answer/user for context.
  - Builds final Slack-formatted text with citations (optional), contact escalation after repeated fallbacks, and a debug ribbon.
  - Ensures product pages are surfaced as clean links for Slack unfurl.

- Ingestion flow (`racen_ingest_url` shortcut + modal):
  - Accepts a grest.in URL.
  - Calls `/ingest/url` on the Answer API.
  - Polls `/ingest/status/{job_id}` using the same `RACEN_ANSWER_URL` base.
  - Sends DM updates to the requesting Slack user.

## 5. Web UI (Next.js) Overview

- Folder: `racen-webui` at repo root.
- Dev server: `npm run dev` → `http://localhost:3000`.
- Uses a client library (`lib/api.ts`) to call the Answer API via a Next.js rewrite rule mapping `/api/answer` (and possibly related paths) to the backend port (currently **8011** in your setup).
- Types: `lib/types.ts` declares the Answer API response shape (`answer`, `citations`, `settings_summary`).
- Main page: `app/page.tsx` wires the chat UI to the backend `answerQuestion()` helper.

## 6. Summary: Canonical Local Setup

- **Postgres**
  - Host: `localhost`
  - Port: `5432`
  - DB: `racen`
  - User: `postgres`
  - Password: `postgres`
  - Search path: `docling,public`

- **Redis**
  - URL: `redis://127.0.0.1:6379/2`
  - Used for RACEN caches when `RACEN_CACHE_ENABLED=1`.

- **Answer API**
  - FastAPI app in `scripts/answer_api.py`.
  - Default run port: `8000` (via `PORT` env), but **your current practice** is to run on **8011** for Slack + Web UI.
  - Endpoints: `/health`, `/answer`, `/ingest/url`, `/ingest/status/{job_id}`.

- **Slack Bot**
  - Socket Mode app in `Grest_RACEN_Slack_Bot/slack-openai-bot/app.js`.
  - Env from `Grest_RACEN_Slack_Bot/.env`.
  - `RACEN_ANSWER_URL=http://127.0.0.1:8011` (base URL, no `/answer`).
  - Uses `OPENAI_MODEL=gpt-4o-mini` for final message shaping.

- **Web UI**
  - Next.js app in `racen-webui`.
  - `npm run dev` → `http://localhost:3000`.
  - Proxies API to Answer API on port **8011** via Next rewrites.

This file is the canonical snapshot of the current local development infrastructure so future work does not rely on assumptions about ports, URLs, or databases.
