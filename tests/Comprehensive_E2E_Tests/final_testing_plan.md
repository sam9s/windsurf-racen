# RACEN – Comprehensive E2E Testing Plan

Generated: 2025-11-26

This document captures the "mother of all tests" plan: a single, comprehensive E2E harness that exercises the full RACEN pipeline for many queries (51 now, ~2000 later) across English, Hinglish, and messy variants.

---

## 1. Core Principles

- **Single source of truth:**
  - One Python harness script + one Markdown result per run.
  - This harness becomes the primary regression test we run whenever we change core logic.
- **Separation of concerns:**
  - Core logic stays in `scripts/step4_answer.py` and other `src/racen/*` modules.
  - All testing and reporting lives under `tests/Comprehensive_E2E_Tests/`.
- **Dynamic, not hard-coded:**
  - The harness never hard-codes specific queries or intents.
  - Everything about which intents/variants to run comes from plain text query files.
- **Multi-variant intent testing:**
  - For each intent (e.g., "cheapest iphone"), we can test multiple variants:
    - Clean English.
    - Clean Hinglish.
    - Messy English.
    - Messy Hinglish.
  - The harness should show, side by side, how all variants behave.

---

## 2. Directory Structure

Parent folder:

- `tests/Comprehensive_E2E_Tests/`

Subfolders:

- `Test_Python_file/`
  - Contains the **single comprehensive harness script** (the mother test).
  - Example: `full_e2e_mother_test.py`.
- `Test_Queries/`
  - Contains one or more plain text query files.
  - Each file can define many intents and variants.
  - We can start from a file derived from the existing 51 Holy Grail queries, then later add a file with ~2000 queries.
- `Test_Results/`
  - Contains Markdown outputs from running the harness.
  - Each run produces exactly **one** MD report.

This keeps code, inputs, and outputs cleanly separated.

---

## 3. Query File Format (Dynamic, Intent + Variants)

Location: `tests/Comprehensive_E2E_Tests/Test_Queries/`

### 3.1. Line Types

- Comment lines:
  - Start with `#` and are ignored.
- Two supported formats:

1. **Simple (backwards-compatible)**
   - `query`
   - `query||k`
   - Use this when there is no explicit intent/variant grouping (for example, the current 51 Holy Grail queries).

2. **Grouped (for intent + variant)**
   - `intent_id||variant||query||k`
   - Where:
     - `intent_id`: logical name of the intent, e.g. `cheapest_iphone`.
     - `variant`: label such as `EN`, `HI_EN`, `MESSY_EN`, `MESSY_HI` (or any custom string).
     - `query`: the actual user text.
     - `k`: integer `top_k` value for retrieval.

### 3.2. Example for One Intent (Cheapest iPhone)

```text
# Cheapest iPhone intent – 4 variants
cheapest_iphone||EN||what is the cheapest iPhone you have?||10
cheapest_iphone||HI_EN||sabse sasta iphone kaun sa hai?||10
cheapest_iphone||MESSY_EN||waht is chepst iphene you have?||10
cheapest_iphone||MESSY_HI||sab sa sista iphne kon sa hai?||10
```

The harness does **not** assume there are 4 variants; it will accept any number of variants per `intent_id`.

---

## 4. Mother Harness Script

Location: `tests/Comprehensive_E2E_Tests/Test_Python_file/`

Example name: `full_e2e_mother_test.py`.

### 4.1. Inputs

- A query file path inside `Test_Queries/` (e.g., passed via CLI argument).
- An output MD path in `Test_Results/` (auto-generated from the query file name and timestamp if not specified).

### 4.2. Per-Query Processing

For each parsed query `(intent_id?, variant?, query, top_k)` the harness will:

1. **Parse metadata**
   - Extract:
     - `intent_id` (optional; empty when using simple format).
     - `variant` (optional; empty when using simple format).
     - `query` (required).
     - `top_k` (default `10` when omitted).

2. **Detect mode (EN vs Hinglish)**
   - Use existing helper from `scripts.step4_answer`:
     - `mode = sa._detect_mode(query)` → `EN` vs `HI_EN`.

3. **Normalize + family preservation** (no core changes)
   - Reuse existing core helpers:
     - `normalize_query` from `racen.query_normalizer`.
     - `_preserve_family_hints` from `scripts.step4_answer`.
   - Logic (in the harness):
     - Determine locale from mode:
       - `hi-IN` for `HI_EN`, otherwise `en-IN`.
     - Call `normalize_query(query, user_locale=locale)`.
     - `normalized_llm = (result.normalized_query or query).strip()`.
     - `effective_query = _preserve_family_hints(query, normalized_llm)`.
   - The harness logs **both**:
     - `normalized_llm` → "what the LLM understood in English".
     - `effective_query` → the text after family/brand preservation.

4. **Cold E2E run** (full pipeline)
   - Start a timer.
   - Call `sa.answer_query(query, top_k=k)` **once**.
   - Stop the timer → store `cold_ms`.
   - Capture:
     - Final `answer`.
     - `citations` list.
   - Classify status (using the same logic as the Hinglish harness):
     - `ok` / `fallback` / `error`.

5. **Hot E2E run** (latency only)
   - Separate second pass over all queries.
   - For each: call `sa.answer_query(query, top_k=k)` again, with timing only.
   - Record `hot_ms` per query + total `hot` time.

6. **Citations**
   - For the cold run, record citations as:
     - URL.
     - `start_line`.
     - `end_line`.

The harness **does not modify** core logic, it only calls into it.

---

## 5. Single Comprehensive Markdown Output

Location example:

- `tests/Comprehensive_E2E_Tests/Test_Results/full_e2e_<basename_of_queryfile>_<timestamp>.md`

### 5.1. Global Summary Section

- Total number of queries.
- Counts of:
  - `OK` answers.
  - `fallback` answers.
  - `error` cases.
- Total cold and hot time for all queries.
- Average cold and hot latency per query.

### 5.2. Per-Query Section

For each query, include **all** of the following in the MD report:

- `intent_id` (when present; otherwise blank).
- `variant` label (when present; otherwise blank).
- **Original text** (raw user input).
- **Detected mode** (`EN` / `HI_EN`).
- **Normalized LLM text** (English interpretation from `normalize_query`).
- **Effective query after family/brand preservation** (`_preserve_family_hints`).
- **Top-K**.
- **Status** (`ok` / `fallback` / `error`).
- **Latencies**:
  - `cold_ms` (full E2E).
  - `hot_ms` (second run).
- **Final answer (user-visible)**:
  - Exactly what an end user would see in Slack/Web UI.
  - Shown in a fenced `text` block.
- **Citations**:
  - Each citation as `- [j] url (lines start-end)`.

This gives one place where we can inspect, for each intent and variant, both:

- How the LLM is interpreting the query (normalized text).
- How the deterministic logic and rewriter respond (final answer, language, links).
- The performance characteristics (cold/hot latencies).

---

## 6. Rollout Strategy: 51 Queries Now, 2000 Later

### 6.1. Phase 1 – Wire Harness and Validate on 51 Queries

- Create a query file in `Test_Queries/` based on the existing 51 Holy Grail queries.
  - For backwards compatibility we can initially use the **simple format** (`query` or `query||k`).
  - Optionally, we can add `intent_id` + `variant` labels later, once we group them logically.
- Implement `full_e2e_mother_test.py` in `Test_Python_file/` using the plan above.
- Run the mother harness against the 51 queries and:
  - Compare high-level behavior to the existing holy-grail MD outputs (latency ranges, answer patterns).
  - Adjust only the reporting format if needed (fields/labels), not core logic.

### 6.2. Phase 2 – Introduce Intent + Variant Grouping

- For key intents (for example, "cheapest iphone"), create grouped lines in the query file using the 4-way pattern:
  - EN / HI_EN / MESSY_EN / MESSY_HI.
- Use the mother harness to see how these variants behave next to each other:
  - Do they normalize to similar English interpretations?
  - Do they all hit the same iPhone helper / product flow?
  - Are latencies in the same ballpark?

### 6.3. Phase 3 – Scale to ~2000 Queries

- Prepare one or more large query files under `Test_Queries/` with ~2000 lines.
  - Each line can carry `intent_id` and `variant` as needed.
- Run the **same** mother harness script over these large files to produce comprehensive MD outputs in `Test_Results/`.
- These runs become the main regression check before any major deploy or logic change.

---

## 7. Safety and Separation (No Core Changes for Testing)

- The comprehensive harness:
  - Lives entirely under `tests/Comprehensive_E2E_Tests/Test_Python_file/`.
  - Only imports existing helpers from `scripts.step4_answer` and `racen.query_normalizer`.
  - Does **not** add any new debug-only functions to `step4_answer.py`.
- All test inputs and outputs:
  - Stay within `Test_Queries/` and `Test_Results/`.
- This keeps the production codebase clean while still giving us a **very rich, end-to-end test view** of:
  - Intent understanding.
  - Normalization behavior.
  - Family/brand preservation.
  - Deterministic logic (like iPhone helper).
  - Final phrasing and language.
  - Latency characteristics.

---

This plan is intentionally written to be stable as we move from 51 queries today to ~2000 queries later, with the **same harness** and the **same MD format**.
