# Pipeline Settings and Repro Steps

This document captures the working configuration and commands to run the full RACEN pipeline end-to-end (Step 2 Docling DB build → Retrieval → Answer → Evaluation).

## Models and Defaults
- FAST_MODE=1
- TOP_K=10
- RERANK_TOP_N=14
- RETRIEVE_BACKOFF_ENABLE=1
- RETRIEVE_BACKOFF_THRESHOLD=0.35
- HNSW_EF_SEARCH=100
- EMBED_MODEL=text-embedding-3-small (dim=1536)
- OPENAI_MODEL=gpt-4o-mini
- PGOPTIONS="-c search_path=docling,public"

## Allowlist Guidance
- FAQs: `/pages/faqs`
- Shipping: `/policies/shipping/policy,/policies/refund/policy`

## Environment
Provide the following via environment variables (never commit secrets):
- OPENAI_API_KEY
- PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD

Optional tuning (falls back to defaults above if unset):
- FAST_MODE, TOP_K, RERANK_TOP_N, RETRIEVE_SOURCE_ALLOWLIST,
  RETRIEVE_BACKOFF_ENABLE, RETRIEVE_BACKOFF_THRESHOLD, RETRIEVE_BACKOFF_SECONDARY,
  ANSWER_SHORT, ANSWER_MAX_TOKENS, ANSWER_CHUNK_CHAR_BUDGET, PGOPTIONS.

## Step 2: Build/Refresh Docling DB
Run from repo root (Windows PowerShell example):
```powershell
# Build Docling schema and ingest markdown
python scripts/step2_process_markdown.py
```

The Step 2 modules involved:
- scripts/step2_process_markdown.py
- src/racen/step2_ingest.py
- src/racen/step2_write.py
- src/racen/step2_embed.py

## Retrieval + Answer (Programmatic)
Use `scripts/step4_answer.py` if calling directly, or rely on the evaluator below which calls the same answering logic.

## Evaluation (Traceable Reports)
- The evaluator writes a full settings snapshot into each report, including effective parsed allowlists.

Examples (Windows PowerShell):

- FAQs only:
```powershell
python scripts/eval_freeform.py \
  --md docs\Grest_Use_Cases_Freeform.md \
  --out outputs\step2\test-reports\eval_freeform_report_docling_faqs_latest.md \
  --k 10 \
  --expected FAQs \
  --allowlist /pages/faqs
```

- Shipping only:
```powershell
python scripts/eval_freeform.py \
  --md docs\Grest_Use_Cases_Freeform.md \
  --out outputs\step2\test-reports\eval_freeform_report_docling_shipping_latest.md \
  --k 10 \
  --expected Shipping \
  --allowlist /policies/shipping/policy,/policies/refund/policy
```

## Notes
- Keep `.env` and secrets out of git. Use a local `.env` and never commit it.
- `outputs/` contains generated artifacts and should remain uncommitted.
- Docker/Slack are intentionally excluded from this snapshot; integrate later on a separate branch.
