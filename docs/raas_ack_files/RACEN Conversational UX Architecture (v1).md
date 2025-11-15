# RACEN Conversational UX Architecture (v1)
### Author: ChatGPT (Solutions Architect)
### Target Implementer: Codex (VS Code)

---

## 1. Purpose
Enhance RACEN’s conversational quality to match high-end GPT behavior while maintaining:
- Retrieval grounding (PostgreSQL + pgvector)
- LLM-led conversation control
- Zero Slack-side logic
- Natural, friendly, Indian customer-care voice

This document defines:
- Ack classification
- Language continuity
- Facet bundles
- Retrieval biasing
- Prompt architecture
- Fallbacks
- Evaluation metrics
- Code-level snippets
- Recommended implementation stages (2-week plan)

---

## 2. Architecture Overview

### Backend Responsibilities (FastAPI)
- **Detect language** (last user message)
- **Classify ack vs new topic**
- **Select facet** from intent bundles
- **Apply retrieval bias** using allowlist boosts
- **Infer retrieval confidence**
- **Build prompt** (system + developer + context + snippets)
- **Post-process** LLM output (tone, length, chips)

### Slack Responsibilities
- Nothing except forwarding user messages.

---

## 3. Conversation Logic Rules

### 3.1 Ack vs New Topic
If user says:  
“yes please”, “haan”, “sure”, “ok go ahead”, “need more info”,  
→ Treat as **ACK_CONTINUE** unless text contains explicit new topic nouns.

### 3.2 Language Continuity
Always reply in **last user language**, unless user switches language explicitly.

### 3.3 Tone
Default: **friendly, casual, warm**  
If negative sentiment: switch to **empathetic + formal**.

---

## 4. Facet Bundles

Stored as JSON, editable:

Example (contact bundle):
```json
{
  "intent": "contact",
  "facets": ["address", "phone", "email", "support_hours"],
  "priority": ["phone", "email", "address"]
}
```

Selection logic:
1. If ack → choose next facet by priority.
2. If no previous facet → choose priority[0].
3. If retrieval confidence low → fallback path.

---

## 5. Retrieval-Bias Algorithm

### Steps
1. Collect:
   - intent_hint  
   - last_user_message  
   - selected_facet  
2. Create query embedding.
3. Retrieve top-12.
4. Boost allowlisted docs: `score *= 1.2`
5. Deduplicate near duplicates (>0.92).
6. If top_score ≥ 0.78 → **HIGH** confidence  
   Else → MEDIUM/LOW

### Use in prompt:
- HIGH: expand facet confidently  
- LOW: provide short friendly summary + offer two options

---

## 6. Prompt Architecture

### SYSTEM (short)
```
You are RACEN — a warm, concise Indian customer-care assistant for Grest. Stay factual and helpful.
```

### DEVELOPER RULES
```
- Use same language as last user message.
- If ack and retrieval_confidence == HIGH, expand the selected facet confidently.
- First bubble ≤ 2 sentences + optional suggestion chip.
- If LOW confidence: give 1–2 line summary + two actionable options.
- Ask only 1 clarifier if ambiguity is high.
- Switch to empathetic formal tone if user upset.
```

### CONTEXT injected
- last 2–3 turns  
- selected_facet  
- retrieval snippets  
- retrieval confidence

### GENERATION INSTRUCTION
```
Follow selected_facet if available.
Be concise. No citations.
Offer only relevant chips.
```

---

## 7. Fallback Template (LOW confidence)
```
Short friendly summary (1–2 sentences).
Two options (e.g., "Phone & email" and "Full contact steps").
Open invite: “Or tell me exactly what you need.”
```

---

## 8. Test Scripts (Unit tests)

### Test 1 — English Contact Continuation
1) “Where is grest head office?”  
2) bot → address + “more details?”  
3) “yes please”  
Expected: phone+email in English, 2 sentences max.

### Test 2 — Hinglish Contact Continuation
1) “grest ka head office kaha hai?”  
2) bot → address in Hinglish  
3) “haan mujhe aur batao”  
Expected: phone+email in Hinglish.

### Test 3 — Ambiguous ack
1) “how long is warranty?”  
2) bot → “6 months, more details?”  
3) “yes”  
Expected fallback: summary + two options.

### Test 4 — Upset
1) “I’ve been chasing pickup for 5 days.”  
Expected: empathetic + offer human escalation.

---

## 9. Metrics

- **Language continuity** ≥ 98%  
- **Facet precision** ≥ 90%  
- **Askback rate** ≤ 5%  
- **Fallback helpfulness** ≥ 65%  
- **Persona warmth** ≥ 4/5  
- **Escalation correctness** ≥ 95%

---

## 10. 2-Week Implementation Plan

### Iteration A (3 days)
- Ack classifier  
- Language lock  
- First bubble rule  
- Metric hooks  

### Iteration B (4 days)
- Retrieval biasing  
- Confidence gating  
- Fallback refinement  

### Iteration C (7 days)
- Tone classifier  
- One-clarifier policy  
- A/B testing  
- Bundle refinement

---

## 11. Risks
- Over-bias retrieval → mitigated by gating  
- Over-restrict LLM → keep dev rules light  
- Wrong language detection → always prefer user last language  
- Too many clarifiers → enforce single-clarifier rule

---

## 12. Deliverables for Codex
- Implement ack classifier  
- Implement facet bundles  
- Implement retrieval-bias  
- Build prompt using architecture  
- Tests & metrics  
