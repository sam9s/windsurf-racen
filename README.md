# RACEN (Rapid Automation Customer Engagement Network)

Initial Step 1 scaffold: Fetch & Convert (file/URL -> Markdown).

## Quickstart

1. Create and activate a virtual environment (Python 3.10+).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run smoke test:

```bash
python -m scripts.smoke_step1
```

Outputs land in `outputs/step1/`.

## Notes
- `MarkItDownClient` is a placeholder for the MarkItDown MCP `convert_to_markdown` capability, with a local HTML->Markdown fallback.
- Next steps: Ingest (clean -> chunk -> embed -> write to Postgres/pgvector), Retrieve (hybrid + rerank), Slack channel agent.
