# RACEN Dev Environment (Local)

This file captures the canonical environment and commands for working on RACEN locally.

## 1. Project root

```
d:\RAVENs\GREST\RACEN_LOCAL\Windsurf_Project
```

All commands assume this as the current working directory.

## 2. Python virtual environment

Python is run from the local virtualenv:

```powershell
.\.venv\Scripts\python.exe --version
```

Use this for **all** Python commands (tests, scripts, tools):

```powershell
.\.venv\Scripts\python.exe <module or script>
```

## 3. Core commands

### 3.1. Run tests

- All tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

- Query flow tests only:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/queries -q
```

- Brand reputation routing tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/queries/test_brand_reputation_query_flows.py -q
```

### 3.2. Answer API (FastAPI)

Start the HTTP API (used by Slack bot and CLI):

```powershell
.\.venv\Scripts\python.exe -m uvicorn scripts.answer_api:app --host 0.0.0.0 --port 8011 --reload
```

By default the Slack bot and local tools assume:

- Base URL: `http://127.0.0.1:8011`
- Main endpoint: `POST /answer`

### 3.3. Manual CLI test to /answer

PowerShell example (see also `docs/grest_cli_command.txt`):

```powershell
$body = @{
  question = "do you have iPhone 11?"
  k        = 18
  short    = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://127.0.0.1:8011/answer" `
  -Method POST `
  -ContentType "application/json" `
  -Body $body |
  ConvertTo-Json -Depth 6
```

## 4. Ingestion

### 4.1. Bulk ingest from YAML

General pattern:

```powershell
.\.venv\Scripts\python.exe scripts/bulk_ingest_from_yaml.py `
  --config <YAML_PATH> `
  --key <TOP_LEVEL_KEY> `
  --dim 1536
```

Examples:

- Legal policies:

```powershell
.\.venv\Scripts\python.exe scripts/bulk_ingest_from_yaml.py `
  --config Grest_Data/grest_legal_policies.yaml `
  --key legal_policies `
  --dim 1536
```

- Brand reviews (Trustpilot + Mouthshut):

```powershell
.\.venv\Scripts\python.exe scripts/bulk_ingest_from_yaml.py `
  --config Grest_Data/grest_brand_reviews.yaml `
  --key brand_reviews `
  --dim 1536
```

## 5. DB sanity checks

For validating ingestion in the `docling` schema, use the commands in:

- `docs/SANITY_COMMANDS.md`

Those assume a Docker Postgres container named `racen-pg`.

## 6. Slack bot

The Slack bot runs from the sibling project (node):

```powershell
cd d:\RAVENs\GREST\RACEN_LOCAL\Grest_RACEN_Slack_Bot\slack-openai-bot
node app.js
```

It expects the RACEN Answer API to be available at `http://127.0.0.1:8011/answer`.

## 7. Debug flags

To expose extra information (citations, settings) in the Answer API debug ribbon, set:

- `ANSWER_DEBUG_FLAGS=1`

Behaviour:

- Adds retrieval settings (k, FAST_MODE, RERANK_TOP_N) to the response ribbon.
- Appends top citation URLs so they are visible in clients (e.g., Slack).

## 8. Common paths

- Project root: `d:\RAVENs\GREST\RACEN_LOCAL\Windsurf_Project`
- Data YAMLs: `Grest_Data/`
- Docs: `docs/`
- Core code: `src/racen/`
- Scripts: `scripts/`
- Tests: `tests/`
