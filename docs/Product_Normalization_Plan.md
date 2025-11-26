# RACEN Product Normalization & Scalable Retrieval Plan (v1)
**Purpose:** Ensure user queries like “cheapest iphne / phne / iphne 12” map correctly to intended products across all brands (Apple, Samsung, Realme, Redmi, etc.)  
**Audience:** Codex (Backend Developer)  
**Author:** Astra — Solution Architect  

---

## 1. Canonical Product Identity Model
Create a lightweight catalog map with the following fields:

- **canonical_name** – e.g., `"Apple iPhone 13"`
- **brand** – `"Apple"`
- **model_tokens** – `["iphone", "13"]`
- **category** – `"smartphone"`
- **aliases** – list of known surface forms:  
  - lowercased titles  
  - common misspellings (`iphne`, `iphon`)  
  - abbreviations (`ip13`, `i phone`)  

This map is the single source of truth for product normalization.

---

## 2. Automated Alias Layer (Dynamic & Incremental)
- Auto-generate aliases from product titles (tokenized, normalized).  
- Add simple typo-variants (edit distance 1–2).  
- Add keyboard-swap variants.  
- Weekly: mine logs for “unmatched tokens” → auto-suggest alias candidates for human approval.  
- Remove stale aliases not matched for 90 days.

This enables long-term scalability without manual babysitting.

---

## 3. Hierarchical Retrieval Logic (Brand-Aware)
For any price-related query:

1. Detect intent.  
2. Run alias/token scan → if match found, constrain retrieval to the canonical brand/model.  
3. If **no brand** detected → run two probes:
   - global cheapest across all phones  
   - brand-top candidates (Apple/Samsung/Realme/Redmi) via fuzzy/semantic match  
4. Compare confidence scores → if one candidate dominates, respond directly; else require clarifier.

This logic adapts automatically as new brands/models are added.

---

## 4. Disambiguation UX (Only When Needed)
When ambiguity is real, ask a single friendly clarifying question.

**English:**  
“Do you mean the cheapest **iPhone**, the cheapest **Samsung/Android**, or the cheapest **overall**?”

**Hinglish:**  
“Aapko cheapest **iPhone**, **Samsung/Android**, ya **overall** phone dekhna hai?”

**Suggestion Chips:**  
- `[Cheapest iPhone]`  
- `[Cheapest Android]`  
- `[All phones]`

---

## 5. Default Behavior Policy
- If the query contains a brand token (even fuzzy), assume that brand.  
- If recent conversation context shows a brand preference, inherit that.  
- If query is completely generic and no context → clarifier required.  
- Optional business rule: default to “overall cheapest” + brand chips as filters.

---

## 6. Confidence Thresholds

| Match Type              | Confidence | Behavior                     |
|-------------------------|-----------|------------------------------|
| Alias exact match       | HIGH      | Direct answer                |
| Fuzzy score > 0.85      | HIGH      | Direct answer                |
| Fuzzy 0.60–0.85         | MEDIUM    | Answer + “Did you mean X?”   |
| < 0.60                  | LOW       | Ask clarifier                |

Keep thresholds adjustable (config file).

---

## 7. Scalability Across Many Brands/Models
- Maintain a `popular_brands` list (top 20–30).  
- Use this list to order clarifier suggestions.  
- Run vector embeddings over all product titles for semantic matching.  
- New SKU onboarding automatically generates:
  - canonical_name  
  - normalized tokens  
  - alias seeds  

Zero rework needed when new brands are added.

---

## 8. Conversational Flow Examples

### Example A — Misspelled brand
**User:** “cheapest iphne”  
**System:** alias maps → “iphone”  
**RACEN:** “Cheapest iPhone today is **iPhone SE (2022)** at ₹X. Want specs or compare with Android?”

### Example B — No brand specified
**User:** “cheapest phne”  
**RACEN:** “Do you want the cheapest **iPhone**, cheapest **Samsung/Android**, or **overall**?”

### Example C — Use conversation context
**User:** “cheapest phone” (after Samsung discussion)  
**RACEN:** “For Samsung, the cheapest right now is … Want Android options too?”

---

## 9. Analytics & Auto-Learning Loop
Log:

- original user text  
- aliases matched  
- canonical normalization chosen  
- retrieval confidence  
- clarifier shown (yes/no)  
- user final choice  

Weekly job:

- auto-suggest new aliases from unmatched tokens  
- prune unused aliases  
- update popular_brands from sales frequency

---

## 10. UX Guardrails
- Only **one** clarifier question per ambiguity.  
- Replies: max **2 sentences**, simple, friendly, Hinglish when user uses Hinglish.  
- Never assume brand unless confidence HIGH or context strong.  
- Do not hallucinate specs — rely strictly on product catalog + retrieval.  
- Reset brand context after 10–12 turns or topic shift.

---

**End of Document (RACEN Normalization v1)**
