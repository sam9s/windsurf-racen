# RACEN – AI Collaboration Guide

This file captures conventions and guardrails for working with the AI assistant on this repo. Read this (and `PLANNING.md` + `docs/TASK.md`) at the start of each new session.

## 1. Environment & Feature Toggles

- **[rule] Check env-gated features explicitly**  
  - Before claiming a feature "works end-to-end", inspect `.env` for its toggle.  
  - Example: `RACEN_QUERY_NORMALIZATION_ENABLE=1` must be set for the LLM query normalizer to actually call OpenAI.
- **[rule] Dont assume defaults**  
  - If a feature depends on an env var and it is missing, treat the feature as **off** and say so.

## 2. Running Code & Tests

- **[rule] No REPL-style inline commands**  
  - Avoid `python - <<` or interactive snippets that confuse PowerShell or the user.  
  - Use direct, reproducible commands only, e.g.:  
    - `.[0m.venv\Scripts\python.exe tests\queries\grest_holy_grail_comprehensive_51.py`  
    - `.[0m.venv\Scripts\python.exe -m pytest tests\queries -q`
- **[rule] Prefer existing harnesses**  
  - For 51 curated queries: `tests/queries/grest_holy_grail_comprehensive_51.py`.  
  - For messy-English normalization benchmark: `tests/queries/grest_messed_up_english_normalizer_report.py`.  
  - For normalization introspection (original vs normalized vs answer): `tests/queries/grest_messed_up_english_normalizer_introspection_report.py`.

## 3. Normalizer-Specific Checks

- **[rule] Verify normalizer is active**  
  - Confirm `RACEN_QUERY_NORMALIZATION_ENABLE=1` in `.env`.  
  - Optionally add tests or harness output showing `Original` vs `Normalized` queries differ for at least one known messy query.
- **[rule] Use introspection MD to debug behaviour**  
  - Regenerate and inspect:  
    - `tests/queries/grest_messed_up_english_normalizer_introspection.md`
  - This shows:  
    - Original messy query.  
    - Normalized query (LLM rewrite).  
    - Final answer + citations.

## 4. Session Bootstrap Procedure

At the start of a new day/session:

1. **Read project guides**  
   - `PLANNING.md` for architecture and design.  
   - `docs/TASK.md` for active/completed tasks.  
   - `docs/AI_Collaboration_Guide.md` (this file) for collaboration rules.
2. **Re-hydrate key context**  
   - Skim the latest benchmark/normalizer MDs in `tests/queries/` if working on NLP or normalization.  
   - Check `.env` for any feature flags relevant to the task.

## 5. Communication & Safety

- **[rule] Be explicit about whats proven**  
  - When reporting that something "works", specify whether:  
    - Only the pipeline answer looks good, or  
    - The intended feature (e.g., LLM normalizer) is definitively active and exercised.
- **[rule] Minimize user friction**  
  - Dont rely on the user to remind about env flags or historical runs.  
  - Proactively search prior harnesses, reports, and `.env` before making claims.
