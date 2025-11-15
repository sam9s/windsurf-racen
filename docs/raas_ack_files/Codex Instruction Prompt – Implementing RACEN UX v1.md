# Codex Instruction Prompt — Implement RACEN UX v1

You are Codex.  
Your job is to implement RACEN Conversational UX Architecture (v1) exactly as defined.

Your responsibilities:
- Modify backend (FastAPI service on port 8011)
- Keep Slack bot thin (no logic)
- Maintain LLM-led logic
- Implement UX upgrades incrementally without breaking current endpoints

## Implement the following components:

### 1. Ack Classifier
- Build an endpoint-internal function:
  - First try LLM few-shot classifier (temperature 0)
  - If LLM unavailable, fallback to cosine similarity (>0.6 → ACK_CONTINUE)

### 2. Language Lock
- Detect language of last user message.
- Force response language to match, unless user explicitly switches.

### 3. Facet Bundles
- Create `config/facets/` folder.
- Define JSON bundles for each intent: contact, returns, warranty, shipping.
- Implement facet selector based on ack vs new topic.

### 4. Retrieval Biasing
- Implement allowlist boosting (`score *= 1.2`)
- Deduplicate near-duplicates (>0.92)
- Compute `retrieval_confidence` from top score.

### 5. Prompt Builder
- Build system prompt + developer rules + snippets + selected_facet context.
- Enforce:
  - Max 2 sentences first bubble
  - Chips from allowed list only
  - No citations in Slack

### 6. Fallback Logic
Use fallback template if:
- retrieval_confidence < threshold  
- facet missing  
- ambiguity detected

### 7. Post-Processing
- Verify language continuity  
- Enforce tone policy  
- Add chip (optional)  
- Final clean 2-sentence maximum  

### 8. Unit Tests (must add)
Add the 4 tests from “RACEN UX Architecture (v1)” file.

### 9. Metrics Logging
Log:
- ack_result
- facet_selected
- retrieval_confidence
- language_changed
- fallback_used

## Notes
- Do not expand Slack code; all logic resides in backend.
- Do not hardcode specific pages; use biasing and facet bundles.
- Keep system prompt short.

