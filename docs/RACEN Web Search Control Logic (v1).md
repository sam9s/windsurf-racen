# RACEN Web Search Control Logic (v1)
**Purpose:** Ensure RACEN triggers web search only when it is appropriate, safe, unbiased, and separated from internal Grest support logic.  
**Audience:** Codex (Backend Developer), Product, Astra (Solution Architect)  
**Scope:** Applies to all channels (Slack, WhatsApp, Web UI, Voice).

---

## 1. Web Search Modes (Two Distinct Pipelines)

### **Mode A — Serious Web Search**
Used for business-relevant, factual, external-info queries:
- Competitor comparisons  
  - “Grest vs Cashify”  
  - “How is Grest compared to X?”  
- Trust/safety checks  
  - “Is Cashify legit/safe?”  
  - “Are refurbished phones trustworthy?”  
- Market-level factual queries  
  - “What’s the current average resale value of iPhone 12?”  

**Goal:** Ground RACEN’s answer in publicly available information, not Grest-internal data.

---

### **Mode B — Fun Web Search**
Used only for opt-in, non-critical, entertainment-oriented queries:
- “Tell me a random tech fact”  
- “Give me a fun comparison about Android vs iPhone”  
- “Show me trending tech memes/news”  
- “Tech history trivia”  

**Goal:** Enhance engagement without influencing purchase decisions.

---

## 2. Web Search Bucketization (Classification Rules)

### **Bucket 1 — Internal-Only (Web Search = OFF)**
Never allow web search for:
- Returns, warranty, pickup, refund, replacements  
- Order tracking, delivery, COD, payments  
- Stock, pricing, inventory availability  
- Repair, IMEI, authenticity  
- Any policy or legal query  

These must be answered *only* from Grest’s deterministic/internal sources.

---

### **Bucket 2 — Competitor & Trust Queries (Web Search = SERIOUS MODE)**
Allow web search **only if**:
- The query explicitly references another company/brand  
  - “Cashify”, “Yaantra”, “Servify”, etc.  
- The intent is **comparison**, **trust**, or **credibility**  

Examples:
- “Why buy from Grest instead of Cashify?”  
- “Which is more reliable: Grest or Cashify?”  
- “Are Grest phones more trustworthy than X?”  

RACEN must preface results with:  
> “This is based on publicly available sources.”

---

### **Bucket 3 — Fun Requests (Web Search = FUN MODE)**
Allow web search **only if**:
- User explicitly requests fun/entertainment  
- User enters Fun Mode intentionally

Examples:
- “Fun mode on”  
- “Tell me something fun about phones”  
- “Give me a tech fact for timepass”  

Fun Mode must remain isolated from business logic.

---

## 3. Trigger Conditions (What actually activates web search)

### **A. Competitor-mode triggers (Exact patterns)**
Trigger **Serious Web Search** if message contains:
- `(grest + vs + competitor)`  
- `(compare + grest + competitor)`  
- `(why + grest + vs + competitor)`  
- competitor-alone when clearly trust-related:
  - “Is Cashify safe?”  
  - “Cashify trust score?”  

Competitor list is dynamic and expandable.

---

### **B. Fun-mode triggers (Opt-in)**
Trigger **Fun Web Search** only when user says:
- “fun mode on”  
- “tell me something fun”  
- “give me a tech fact / trivia”  
- “bored, show me something cool”  

Any fun request *must not* impact Grest’s policy, pricing, or purchase decisions.

---

### **C. Ambiguous queries (Require clarification)**
If RACEN cannot decide:
- “Do you want a factual comparison, or are you asking just for fun?”

This prevents accidental serious web search on fun queries and vice versa.

---

## 4. Guardrails & Safety Rules

### **1. Never mix Fun Mode and Serious Mode**
Fun Mode cannot answer competitor questions.  
Serious Mode cannot use meme/fun sources.

### **2. Persona + Disclaimers**
For competitor info:
> “This is from publicly available sources and may not reflect the latest company updates.”

### **3. No hallucinated comparisons**
- RACEN must ground comparisons in retrieved web snippets only.  
- If low-confidence:  
  > “I can’t confirm that confidently — would you like a human to assist?”

### **4. Internal supremacy rule**
If query relates even **indirectly** to:
- warranty  
- returns  
- refunds  
- Grest reliability  
- order flow  
→ internal DB takes precedence over web.

### **5. Rate-limits**
Prevent abuse or DDoS on web search API:
- max X web-search calls per user per hour  
- Fun mode rate limit separate from serious mode

### **6. PII safety**
Web search results should never expose:
- customer data  
- order data  
- internal metrics  

---

## 5. Output Structuring Rules

### **Serious Mode Output**
- 1–2 sentence summary  
- then offer:  
  - `[Detailed comparison]`  
  - `[Trust factors]`  
  - `[Talk to a human]`  

### **Fun Mode Output**
- playful but short  
- no policy or price claims  
- suggestion chips like:  
  - `[More fun]`  
  - `[Back to support]`  

---

## 6. Final Decision Tree (Simple Truth Table)

| Condition | Mode | Web Search | Action |
|----------|------|------------|--------|
| Grest policy / ops | Internal-only | ❌ | Use Grest DB only |
| Brand comparison | Serious | ✅ | Query web → concise safe summary |
| Trust / legitimacy | Serious | ✅ | Query web → cautious tone + disclaimer |
| Fun/trivia | Fun | ✅ | Query web → fun-only sources |
| Ambiguous | Ask user | ❓ | Clarify “fun or comparison?” |

---

## 7. Recommended Implementation Notes (No Code)

- Maintain a simple **intent → mode** mapping file (YAML/JSON).  
- Enable a per-request flag: `web_mode = {none, serious, fun}`.  
- Web retrieval service must know which mode to operate in.  
- Build separate safety filters for the two modes.  
- Log every web-search event into analytics for review.

---

## 8. Placement in RACEN Architecture
This module sits **before retrieval** and **before LLM generation**:
1. Normalizer →  
2. Intent classifier →  
3. **Web Search Decision Layer (this module)** →  
4. Retrieval (internal/web) →  
5. LLM composer →  
6. Reply shaper →  
7. Output

---

**End of Document — RACEN Web Search Control Logic (v1)**
