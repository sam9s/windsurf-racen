# RACEN Product Database Design (Grest)

## 1. Goal

Provide a **single source of truth** for product prices and core specs so that RACEN:

- Always reads **prices, storage, conditions, warranty, canonical URL** from a Postgres table.
- Uses the **website (HTML/RAG)** only for descriptive text (process, benefits, marketing copy).
- Is immune to Shopify issues like multiple variants/URLs for the same model showing different prices.

This document covers:

- Minimal schema for a `products` table in Postgres.
- Rules that RACEN and the backend must follow.
- How data can flow from Shopify / CSV exports into this table.
- How the existing RACEN pipeline should use this table.

---

## 2. Minimal Schema: `products` Table

One row = **one sellable product variant** (e.g., "iPhone 11 / 64 GB / Good").

### 2.1 Recommended Columns

```sql
CREATE TABLE products (
  product_id           TEXT PRIMARY KEY,  -- our stable ID, not Shopify's
  family               TEXT NOT NULL,     -- e.g. 'iphone'
  base_model           TEXT NOT NULL,     -- e.g. '11', '14', '14-pro'
  title                TEXT NOT NULL,     -- display name, e.g. 'Apple iPhone 11'

  storage              TEXT NOT NULL,     -- e.g. '64 GB', '128 GB'
  condition            TEXT NOT NULL,     -- e.g. 'Fair', 'Good', 'Superb'
  color                TEXT,              -- optional, e.g. 'Black'

  price_rupees         INTEGER NOT NULL,  -- current selling price in rupees (e.g. 14499)
  mrp_rupees           INTEGER,           -- optional MRP / "was" price
  warranty_months      INTEGER NOT NULL,  -- e.g. 6

  canonical_url        TEXT NOT NULL,     -- single URL RACEN should show (may include ?variant=)
  shopify_product_id   TEXT,              -- optional linkage to Shopify
  shopify_variant_id   TEXT,              -- optional linkage to Shopify

  active               BOOLEAN NOT NULL DEFAULT TRUE,
  last_updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

**Key points:**

- `product_id` is **our** internal ID (e.g. `iphone11-64-good`), independent of Shopify.
- `canonical_url` is the **only** URL RACEN links to for that variant.
- `price_rupees` is the authoritative price. If HTML shows something else, RACEN trusts this column.

### 2.2 Example Rows

```text
product_id          | family | base_model | title             | storage | condition | price_rupees | mrp_rupees | warranty_months | canonical_url
--------------------+--------+------------+-------------------+---------+-----------+--------------+-----------+-----------------+-------------------------------------------------------------
iphone11-64-good    | iphone | 11         | Apple iPhone 11   | 64 GB   | Good      | 14499        | 43990     | 6               | https://grest.in/products/refurbished-iphone-11?variant=...
iphone14pro-128-fair| iphone | 14-pro     | Apple iPhone 14 Pro| 128 GB | Fair      | 54499        | 129900    | 6               | https://grest.in/products/refurbished-apple-iphone-14-pro?...
```

---

## 3. Rules for Backend + RACEN

### 3.1 Backend / Content Rules

- **Rule B1 – Single source of truth:**
  - `products.price_rupees`, `storage`, `condition`, `warranty_months`, and `canonical_url` are the **only authoritative values** for RACEN.
  - Shopify/HTML prices can be wrong; the DB is considered correct.

- **Rule B2 – One canonical URL per variant:**
  - For each sellable combination (model + storage + condition), backend picks **one** URL as `canonical_url`.
  - Old/extra Shopify variants may still exist, but RACEN will ignore their prices and only link to the `canonical_url`.

- **Rule B3 – Updates:**
  - When price or key specs change, backend **updates this table first**.
  - Shopify pages should eventually be aligned, but RACEN answers are driven by DB immediately.

- **Rule B4 – Activeness:**
  - `active = FALSE` for variants that should no longer be advertised (e.g., permanently out of stock).
  - RACEN should not surface inactive variants in suggestions.

### 3.2 RACEN / Answering Rules

- **Rule R1 – DB over HTML for facts:**
  - For any recognized product, RACEN must use the `products` table for:
    - Price (`price_rupees`)
    - Storage options
    - Condition labels
    - Warranty duration
    - Canonical URL
  - Even if HTML chunks contain other prices, RACEN **ignores** those for factual fields.

- **Rule R2 – HTML for narrative only:**
  - RACEN still retrieves product and blog/FAQ pages for:
    - Descriptions (camera, battery, design, etc.)
    - Refurbishing process, quality checks
    - Returns/shipping/warranty policy explanations
  - But it does not trust HTML for price/specs when `products` has data.

- **Rule R3 – Product selection flow:**
  - Use the existing product search / YAML catalog and intent logic to:
    1. Determine that the intent is `product`.
    2. Match the user’s text (e.g. "iphone 14 pro 128gb") to one or more `product_id`s.
    3. Fetch rows from `products` for those IDs.

- **Rule R4 – Answer composition:**
  - For each chosen product row, the answer template should:
    - Insert `title` and `canonical_url`.
    - Insert a **formatted price**, e.g. `₹54,499` from `price_rupees`.
    - Insert `storage`, `condition`, `warranty_months` as bullet points.
  - Surround these facts with narrative sentences built from retrieved HTML chunks.

- **Rule R5 – No contradiction with DB:**
  - If retrieved HTML mentions a different price/spec than the DB row:
    - RACEN should **not** surface that conflicting value in the final answer.

---

## 4. Data Ingestion into `products`

This is intentionally flexible. A few options:

### 4.1 Direct Backend Writes

- Backend services talk directly to Postgres and maintain `products` alongside Shopify updates.
- Pros:
  - Real-time updates.
  - Single source of truth managed by engineering.
- Cons:
  - Requires some backend work and discipline.

### 4.2 Shopify Export → CSV → Importer Script

1. Backend or operations exports a **CSV** of products/variants (once or periodically).
2. A Python script in this repo:
   - Reads the CSV.
   - Validates rows.
   - Upserts into `products` using `ON CONFLICT (product_id) DO UPDATE`.

Example CSV columns:

```csv
product_id,family,base_model,title,storage,condition,price_rupees,mrp_rupees,warranty_months,canonical_url,shopify_product_id,shopify_variant_id,active
iphone11-64-good,iphone,11,Apple iPhone 11,64 GB,Good,14499,43990,6,https://grest.in/products/refurbished-iphone-11?variant=47835864105191,123456,47835864105191,true
```

Importer responsibilities (to be implemented):

- Normalize/validate:
  - `price_rupees` and `mrp_rupees` are integers.
  - URLs are non-empty and well-formed.
- Insert/update records:

  ```sql
  INSERT INTO products (...)
  VALUES (...)
  ON CONFLICT (product_id) DO UPDATE SET
    price_rupees    = EXCLUDED.price_rupees,
    mrp_rupees      = EXCLUDED.mrp_rupees,
    storage         = EXCLUDED.storage,
    condition       = EXCLUDED.condition,
    canonical_url   = EXCLUDED.canonical_url,
    active          = EXCLUDED.active,
    last_updated_at = now();
  ```

### 4.3 Manually Maintained CSV (Phase 1, iPhones Only)

- For an initial phase, backend can maintain a **small curated CSV** for:
  - Only iPhones (the current Phase 1 focus).
- The same importer can still populate `products` from that CSV.
- Later, the CSV can be replaced with a fuller export or direct integration.

---

## 5. RACEN Integration Plan (High-Level)

1. **Create schema**:
   - Add `products` table to the RACEN Postgres DB (same DB as documents/chunks, or a separate schema).

2. **Implement importer**:
   - A script (e.g. `scripts/import_products_from_csv.py`) that:
     - Reads a CSV file from `Grest_Data/products.csv`.
     - Upserts into `products`.
   - Add simple pytest tests to ensure one good row, one bad row, and an update case behave correctly.

3. **Wire into answer pipeline**:
   - Extend `product_search` / orchestrator so that, after a product match, it:
     - Looks up `product_id` in `products`.
     - If found, uses DB fields for price/specs.
     - If not found, optionally falls back to **HTML-only** behavior with a warning.

4. **Update prompts and templates**:
   - Make sure the answer prompt and formatting layer expect and display:
     - Price from DB.
     - Storage/condition/warranty from DB.
     - Canonical URL from DB.

5. **Backoffice process**:
   - Document for backend/ops:
     - How to update a product row safely.
     - How often CSV exports/imports should run (if using CSV).
     - That RACEN will **not** pick up price changes until `products` is updated.

---

## 6. Summary for Stakeholders

- Current HTML-based approach is **inherently unstable** for prices/specs when:
  - Shopify contains multiple variants/URLs for the same model.
  - Different pages/cards show different prices.
- Introducing a `products` table in Postgres gives Grest a **single, explicit source of truth**:
  - One row per variant.
  - Canonical URL.
  - Canonical price and core specs.
- RACEN will:
  - Use this DB for all factual product answers.
  - Use the website only for descriptive text and policy explanations.
- Backend can populate this table via:
  - Direct writes, or
  - A controlled CSV→Postgres import process.

This design keeps RACEN’s answers stable even when Shopify content is messy, while still leveraging the existing RAG ingestion for rich text and explanations.
