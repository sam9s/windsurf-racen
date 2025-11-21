# Plan for transactional RACEN use cases

This doc captures how we will evolve RACEN from an information-seeking assistant into an **action-taking agent** for high-impact, transactional flows (address updates, RMAs, warranty, etc.).

Authoritative list of target use cases (kept up to date there):

- `docs/High-impact, transactional use cases.md`

---

## 1. Architectural direction (high level)

We will move from a single, monolithic `step4_answer.py` to a **3-layer architecture**:

1. **Intent & routing layer (NLP router)**
   - Module: `src/racen/intent.py` (new).
   - Responsibilities:
     - Normalize/clean user text a bit.
     - Run **heuristic** checks for obvious patterns (cheap).
     - Run **small LLM classifiers** when needed to decide high-level intent.
     - Output a structured `Intent` object, e.g.:
       - `type`: `info_shipping`, `info_returns`, `product_specs`, `comparison`, `address_update`, `invoice_request`, `return_initiation`, `warranty_issue`, etc.
       - `confidence`: float.
       - `entities`: extracted fields (order id, product family, etc.).

2. **Skills / actions layer (business logic)**
   - Package: `src/racen/skills/` (new).
   - Each **transactional use case** becomes a separate skill module, e.g.:
     - `address_update.py`
     - `invoice_request.py`
     - `return_initiation.py`
     - `warranty_claim.py`
     - `order_status.py`
     - `product_finder.py`
   - Each skill:
     - Defines a **Pydantic model** for input (validated arguments).
     - Implements a pure Python `execute()` function (DB/api calls, no LLM).
     - Returns a structured result object (status, messages, references, ids).

3. **Conversation orchestrator**
   - `scripts/step4_answer.py` stays as the **entrypoint**, but becomes thinner:
     - Receives Slack/HTTP message.
     - Calls intent router to get `Intent`.
     - Branches:
       - **Information-seeking** → current RAG + answer LLM path.
       - **Transactional skill** → orchestrates slot filling + calls skill.
     - Uses LLM only for:
       - Clarifying questions / slot filling.
       - Final natural-language explanation of the skill result.

This supports **future growth**: when a new high-impact use case appears, we add a new skill and routing rule, not more if/else inside one giant file.

---

## 2. Phasing the work

### Phase T0 – Current state (baseline)

- RACEN handles **information-seeking** flows:
  - Returns, warranty, shipping, contact.
  - Product availability/spec queries via catalog + RAG.
  - Brand reputation via Trustpilot/Mouthshut.
- Logic is mostly centered in `scripts/step4_answer.py`.
- Comparison support (Phase 1.2) is being wired using:
  - SerpAPI DuckDuckGo client (`src/racen/web_comparison.py`).
  - A heuristic comparison detector (to be upgraded with a small LLM classifier).

### Phase T1 – Intent router + skills skeletons

Goals:

- Introduce a **proper intent router** and **skill interfaces** without changing behaviour for existing queries.
- Prepare the foundation for transactional flows.

Work items:

- Create `src/racen/intent.py` with:
  - A function that wraps current `_detect_intent` and `_classify_product_domain`.
  - A new small LLM-based classifier for:
    - `comparison` vs `info_product` vs `brand_reputation`.
  - A data structure for `Intent` and `Entity` extraction.
- Create `src/racen/skills/` package with **no-op or mocked skills**:
  - `AddressUpdateSkill`, `InvoiceSkill`, `OrderStatusSkill`, `ReturnInitSkill`, `WarrantyClaimSkill`.
  - Each exposes a typed `execute()` that currently just returns a placeholder result.
- Refactor `step4_answer.py` to:
  - Call the intent router first.
  - Keep existing behaviour for `info_*` intents.
  - For transactional intents, return a simple “not yet supported” message for now.

This keeps risk low while splitting responsibilities.

### Phase T2 – Implement top 5 transactional skills

Initial focus (from `High-impact, transactional use cases.md`):

1. **Auto Address/Contact Update Request**
2. **Invoice / GST Bill Request**
3. **Live stock / product availability** (stock-aware product finder)
4. **Warranty claim intake & triage** (with media capture hook)
5. **Return initiation (RMA) within 7 days**

For each of these:

- **Define inputs** (Pydantic models):
  - Minimal required fields (order id / email / phone / IMEI / address, etc.).
- **Implement business logic**:
  - Integrate with existing Grest backend APIs or DB tables where available.
  - If backend APIs don’t exist yet, define a clear interface the backend can implement.
- **Conversation flow pattern**:
  - Use LLM to:
    - Understand user phrase: “change my address…”, “send me invoice…”, “I want to return this…”.
    - Ask follow-up questions when required data is missing.
  - Call the skill `execute()` once enough data is collected.
  - Summarise the result and next steps for the user.

### Phase T3 – Expand to other high-impact cases

- COD eligibility & EMI/BNPL pre-check.
- Device setup concierge.
- Grade trade-offs advisor (Superb vs Good vs Fair).
- Fraud protection concierge.
- DOA / damage reporting and pickup coordination.
- Stock alerts & waitlists.
- Human escalation with context bundle.

Each of these maps to either:

- A new skill module in `src/racen/skills/`, or
- An extension of an existing skill (e.g., order_status + tracking, return_init + DOA path).

---

## 3. Design principles for transactional RACEN

To support the long-term goal of RACEN being a top-tier Indian AI agent:

- **LLM for understanding + phrasing; Python for actions**
  - LLMs decide *what* the user wants and phrase *how* we respond.
  - Deterministic Python skills decide *what to do* in the system.
- **Strong typing and validation**
  - Use Pydantic models for all skill inputs and outputs.
  - Reject or re-ask when required fields are missing/ambiguous.
- **Separation of concerns**
  - Intent/router logic in `racen.intent`.
  - Business flows in `racen.skills`.
  - Conversation + RAG in `scripts.step4_answer`.
- **Testability**
  - Each skill gets focused unit tests (no LLM, no Slack).
  - Intent router gets tests with mocked LLM responses.
  - End-to-end tests simulate Slack queries for key use cases.
- **Safety and auditability**
  - Every transactional action logs what was done, with which inputs, and why.
  - Easy to debug a wrong address update or misplaced return.

---

## 4. Near-term concrete steps

1. Finish the **LLM-based comparison classifier** and fold it into the new `intent` abstraction (even if the code physically still lives in `step4_answer.py` for now).
2. Sketch `src/racen/intent.py` and a simple `Intent` dataclass that wraps current behaviour.
3. Define skeletons for 2–3 core skills (`AddressUpdateSkill`, `InvoiceSkill`, `ReturnInitSkill`) in `src/racen/skills/` with Pydantic models and TODOs for backend integration.
4. Once stable, refactor `step4_answer.py` to call into `racen.intent` + `racen.skills` and keep it focused on RAG + conversation orchestration.
