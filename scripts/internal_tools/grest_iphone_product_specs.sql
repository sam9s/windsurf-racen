-- DDL for manual product specs table used by RACEN.
--
-- Schema: docling (see PGOPTIONS search_path in .env)
-- Table:  grest_iphone_product_specs
--
-- Columns:
--   slug          : canonical product slug (e.g. 'refurbished-apple-iphone-14'), primary key
--   model_details : human-readable model string from the Google Sheet
--   price_superb  : price in rupees for Superb condition (nullable)
--   price_good    : price in rupees for Good condition (nullable)
--   price_fair    : price in rupees for Fair condition (nullable)
--   product_url   : canonical base product URL (without ?variant=...)
--   details       : optional free-form notes
--
-- Usage:
--   Run this once against the 'racen' database (where search_path includes docling)
--   before wiring the sheet→DB sync script and price lookups.

CREATE TABLE IF NOT EXISTS docling.grest_iphone_product_specs (
    slug          TEXT PRIMARY KEY,
    model_details TEXT NOT NULL,
    price_superb  INTEGER,
    price_good    INTEGER,
    price_fair    INTEGER,
    product_url   TEXT NOT NULL,
    details       TEXT
);
