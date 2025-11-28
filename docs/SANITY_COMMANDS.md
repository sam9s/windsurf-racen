# DB Sanity Commands (Docker psql)

Use these to validate ingestion in the docling schema.

- All ingested URLs with non‑zero chunks
```
docker exec -i racen-pg psql -U postgres -d racen -c "select d.source, count(c.id) as chunks from docling.documents d join docling.chunks c on c.document_id=d.id group by d.source having count(c.id) > 0 order by chunks desc;"
```

- Totals snapshot (docling schema)
```
docker exec -i racen-pg psql -U postgres -d racen -c "select (select count(*) from docling.documents) as documents, (select count(*) from docling.chunks) as chunks, (select count(*) from docling.embeddings) as embeddings;"
```

- Only /pages/* URLs
```
docker exec -i racen-pg psql -U postgres -d racen -c "select d.source, count(c.id) as chunks from docling.documents d join docling.chunks c on c.document_id=d.id where d.source like '%/pages/%' group by d.source having count(c.id) > 0 order by chunks desc;"
```

- Top N by chunk count
```
docker exec -i racen-pg psql -U postgres -d racen -c "select d.source, count(c.id) as chunks from docling.documents d join docling.chunks c on c.document_id=d.id group by d.source order by chunks desc limit 20;"
```

Notes
- Column is `source` (not `url`).
- Our ingestion writes to `docling` schema (PGOPTIONS sets search_path).
- Use these exact commands for consistency across runs.

---

## Local Python sanity checks (no psql)

Use this from the Windsurf_Project root when psql is not available on the host.

- iPhone product specs table snapshot
```
python -c "from src.racen.step2_write import DBConfig, get_conn; cfg=DBConfig.from_env(); conn=get_conn(cfg); cur=conn.cursor(); cur.execute('SELECT slug FROM docling.grest_iphone_product_specs ORDER BY slug'); rows=cur.fetchall(); print('COUNT=', len(rows)); print('SLUGS=', [r['slug'] for r in rows]); conn.close()"
```
