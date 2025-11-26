# GREST Hinglish iPhone Family Preservation – Results

Generated: 2025-11-26

This report summarizes the behavior of RACEN after adding the product family preservation wrapper around query normalization in `answer_query`.

## Harnesses Run

- **Hinglish output rewriter report**
  - Script: `tests/queries/grest_messed_up_Hinglish_output_rewriter_report.py`
  - Output: `tests/queries/grest_messed_up_Hinglish_output_rewriter_report.md`
- **Hinglish normalization introspection report**
  - Script: `tests/queries/grest_messed_up_Hinglish_normalizer_introspection_report.py`
  - Output: `tests/queries/grest_messed_up_Hinglish_normalizer_introspection.md`

All runs used the live LLM key and the normal RACEN backend stack (Postgres + embeddings).

## Query Set

Source: `tests/queries/grest_messed_up_Hinglish_queries.txt`

1. `sabse sasta iphne kaon sa hai?`
2. `eak sasts iphone batao?`
3. `25000 ke ander konsa iphne hai?`
4. `20000 aur 40000 ke beech konsa iphne hai?`
5. `aapka office kahan hai?`
6. `phone order kerne ka kaya process hai?`
7. `aapke phones ki battry kitni hai?`
8. `aapke phone ka screen size kitna hai?`

Queries **1–4** are clearly iPhone price/browse style family queries.

## Summary Outcomes

From `grest_messed_up_Hinglish_output_rewriter_report.md`:

- **Total queries:** 8
- **OK:** 8
- **Fallback:** 0
- **Error:** 0

No query returned the static fallback message (`Not found in sources provided...`).

## Behavior for iPhone Hinglish Price/Browse Queries

### 1. `sabse sasta iphne kaon sa hai?`

- **Introspection – normalized:** `Which is the cheapest phone?`
- **Final answer:**
  - Hinglish iPhone-specific answer.
  - Lists a single cheapest iPhone with real catalog price, e.g. `Refurbished Apple Iphone Xs` with price.
  - Includes the canonical collection link: `https://grest.in/collections/iphones`.
- **Routing interpretation:**
  - Despite the normalizer omitting the explicit token `iphone` in the raw normalized text, the wrapper preserved the iPhone family signal for routing.
  - The answer clearly comes from the **deterministic iPhone family helper**, not from the static fallback.

### 2. `eak sasts iphone batao?`

- **Introspection – normalized:** `Please tell me about the cheapest iPhone.`
- **Final answer:**
  - Hinglish answer naming the cheapest iPhone and price.
  - Includes the iPhone collection link.
- **Routing:**
  - Classified as iPhone family price query and answered via the deterministic helper.

### 3. `25000 ke ander konsa iphne hai?`

- **Introspection – normalized:** `Which phones are available under 25,000?`
- **Final answer:**
  - Hinglish price-band listing of **multiple iPhones** within the band.
  - Each entry is a real iPhone product with URL and price.
  - Ends with the iPhone collection link.
- **Routing:**
  - This query previously risked losing the iPhone family in normalization and falling back.
  - Now it is handled as an **iPhone family price-range** request and clearly uses the catalog-backed helper.

### 4. `20000 aur 40000 ke beech konsa iphne hai?`

- **Introspection – normalized:** `Which phones are available between 20,000 and 40,000?`
- **Final answer:**
  - Hinglish list of iPhones in the 20k–40k band with real prices and URLs.
  - Ends with the iPhone collection link.
- **Routing:**
  - Treated as an iPhone price-range browse query and answered via the deterministic iPhone family helper.

## Non‑iPhone Queries (5–8)

- **Office address (`aapka office kahan hai?`):**
  - Normalized to an English office-location question.
  - Answered with the full Gurugram address and rich Trustpilot/MouthShut citations as before.
- **Generic/underspecified phone questions (6–8):**
  - Normalized into generic process/spec questions about “phone(s)”.
  - Correctly trigger the **unclear-intent static rephrase prompt** in Hinglish, asking the user to clarify instead of guessing a product.
  - No catalog leakage or fake specs.

## Conclusions

- The **brand/family preservation layer around normalization is working as intended** for the messy iPhone Hinglish price/browse queries:
  - All four iPhone queries (1–4) are now handled by the **catalog-backed iPhone family helper**.
  - None of them fall back to the static "Not found in sources provided" message.
  - Output is in Hinglish with correct structured prices and product URLs.
- Non-iPhone, non-specific phone queries still:
  - Use normal domain routing (office address), or
  - Hit the graceful unclear-intent fallback, without inventing products.

This slice is compatible with the broader dynamic, alias-based multi-brand design in `docs/Product_Normalization_Plan.md`, and is scoped to iPhone family queries so it can be extended to additional families later without brittle hardcoding.
