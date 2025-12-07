"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import Header from "@/components/Header";

interface IphoneSpecsRow {
  s_no: string;
  model_details: string;
  superb: string;
  good: string;
  fair: string;
  slug: string;
  product_url: string;
  details: string;
}

interface IphoneSpecsSaveRow extends IphoneSpecsRow {
  is_new?: boolean;
}

export default function IphonePricingConsolePage() {
  const [rows, setRows] = useState<IphoneSpecsRow[]>([]);
  const [initialSlugs, setInitialSlugs] = useState<string[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const totalRows = useMemo(() => rows.length, [rows]);

  const loadRows = useCallback(async () => {
    setLoading(true);
    setError(null);
    setStatus(null);
    try {
      const res = await fetch("/api/iphone-specs/list", { cache: "no-store" });
      const data = await res.json();
      if (!res.ok) {
        const detail = (data && (data.detail || data.error)) || "Failed to load specs.";
        throw new Error(detail);
      }

      const rawRows: any[] = Array.isArray(data) ? data : [];
      const normalised: IphoneSpecsRow[] = rawRows.map((raw) => ({
        s_no: raw.s_no ?? raw["S.No."] ?? "",
        model_details: raw.model_details ?? raw["Model Details"] ?? "",
        superb: raw.superb ?? raw["Superb"] ?? "",
        good: raw.good ?? raw["Good"] ?? "",
        fair: raw.fair ?? raw["Fair"] ?? "",
        slug: raw.slug ?? raw["Slug"] ?? "",
        product_url: raw.product_url ?? raw["ProductURL"] ?? "",
        details: raw.details ?? raw["Details"] ?? "",
      }));

      setRows(normalised);
      setInitialSlugs(normalised.map((row) => row.slug));
      setStatus(`Loaded ${normalised.length} rows from sheet.`);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadRows();
  }, [loadRows]);

  const handleCellChange = useCallback(
    (index: number, field: keyof IphoneSpecsRow, value: string) => {
      setRows((prev) =>
        prev.map((row, i) => (i === index ? { ...row, [field]: value } : row)),
      );
    },
    [],
  );

  const handleAddRow = useCallback(() => {
    setRows((prev) => [
      ...prev,
      {
        s_no: "",
        model_details: "",
        superb: "",
        good: "",
        fair: "",
        slug: "",
        product_url: "",
        details: "",
      },
    ]);
  }, []);

  const handleSave = useCallback(async () => {
    if (!rows.length) {
      setStatus("Nothing to save – no rows loaded.");
      return;
    }

    setSaving(true);
    setError(null);
    setStatus(null);

    try {
      const originalSlugs = new Set(initialSlugs);
      const payloadRows: IphoneSpecsSaveRow[] = rows.map((row) => ({
        s_no: row.s_no ?? "",
        model_details: row.model_details ?? "",
        superb: row.superb ?? "",
        good: row.good ?? "",
        fair: row.fair ?? "",
        slug: row.slug ?? "",
        product_url: row.product_url ?? "",
        details: row.details ?? "",
        is_new: !originalSlugs.has(row.slug ?? ""),
      }));

      const res = await fetch("/api/iphone-specs/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: payloadRows }),
      });

      const data = await res.json();
      if (!res.ok || (data && data.status && data.status !== "ok")) {
        const detail = (data && (data.detail || data.message)) || "Save failed.";
        throw new Error(detail);
      }

      const message = data && data.message ? String(data.message) : "Saved changes.";
      setStatus(message);

      // Reload fresh data from sheet so local state always matches source of truth.
      await loadRows();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
    } finally {
      setSaving(false);
    }
  }, [rows, initialSlugs, loadRows]);

  const handleReload = useCallback(async () => {
    await loadRows();
  }, [loadRows]);

  return (
    <div className="page-shell admin-shell">
      <Header title="R . A . C . E . N" caption="IPHONE PRICING CONSOLE" />
      <main className="admin-main">
        <div className="admin-toolbar">
          <div>
            <div>Google Sheet: grest_iphone_product_specs</div>
            <div>{loading ? "Loading…" : `Rows: ${totalRows}`}</div>
          </div>
          <div className="admin-toolbar-buttons">
            <button
              type="button"
              className="admin-button"
              onClick={handleAddRow}
              disabled={loading || saving}
            >
              + Add row
            </button>
            <button
              type="button"
              className="admin-button"
              onClick={handleReload}
              disabled={loading || saving}
            >
              Reload
            </button>
            <button
              type="button"
              className="admin-button primary"
              onClick={handleSave}
              disabled={loading || saving}
            >
              {saving ? "Saving…" : "Submit changes"}
            </button>
          </div>
        </div>

        {status ? <div className="admin-status">{status}</div> : null}
        {error ? <div className="admin-error">{error}</div> : null}

        <div className="admin-table-wrapper">
          <div className="admin-table-scroll">
            <table className="admin-table">
              <thead>
                <tr>
                  <th style={{ width: "3.5rem" }}>S.No.</th>
                  <th style={{ width: "24%" }}>Model Details</th>
                  <th style={{ width: "9%" }}>Superb</th>
                  <th style={{ width: "9%" }}>Good</th>
                  <th style={{ width: "9%" }}>Fair</th>
                  <th style={{ width: "18%" }}>Slug</th>
                  <th style={{ width: "20%" }}>Product URL</th>
                  <th style={{ width: "15%" }}>Details</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr key={index}>
                    <td>
                      <input
                        type="text"
                        value={row.s_no}
                        onChange={(event) =>
                          handleCellChange(index, "s_no", event.currentTarget.value)
                        }
                      />
                    </td>
                    <td>
                      <input
                        type="text"
                        value={row.model_details}
                        onChange={(event) =>
                          handleCellChange(index, "model_details", event.currentTarget.value)
                        }
                      />
                    </td>
                    <td>
                      <input
                        type="text"
                        value={row.superb}
                        onChange={(event) =>
                          handleCellChange(index, "superb", event.currentTarget.value)
                        }
                      />
                    </td>
                    <td>
                      <input
                        type="text"
                        value={row.good}
                        onChange={(event) =>
                          handleCellChange(index, "good", event.currentTarget.value)
                        }
                      />
                    </td>
                    <td>
                      <input
                        type="text"
                        value={row.fair}
                        onChange={(event) =>
                          handleCellChange(index, "fair", event.currentTarget.value)
                        }
                      />
                    </td>
                    <td className="slug-cell">
                      <input
                        type="text"
                        value={row.slug}
                        onChange={(event) =>
                          handleCellChange(index, "slug", event.currentTarget.value)
                        }
                      />
                    </td>
                    <td>
                      <div className="url-cell">
                        <input
                          type="text"
                          value={row.product_url}
                          onChange={(event) =>
                            handleCellChange(index, "product_url", event.currentTarget.value)
                          }
                        />
                        {row.product_url ? (
                          <a
                            href={row.product_url}
                            target="_blank"
                            rel="noreferrer"
                            className="url-open"
                          >
                            Open
                          </a>
                        ) : null}
                      </div>
                    </td>
                    <td>
                      <input
                        type="text"
                        value={row.details}
                        onChange={(event) =>
                          handleCellChange(index, "details", event.currentTarget.value)
                        }
                      />
                    </td>
                  </tr>
                ))}
                {rows.length === 0 && !loading ? (
                  <tr>
                    <td colSpan={8}>
                      <div className="placeholder">No rows loaded from sheet.</div>
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </div>

        <p className="admin-footnote">
          Changes made here update the Google Sheet only. To push prices into RACEN&apos;s
          Postgres specs table, run the existing Slack admin sync command.
        </p>
      </main>
    </div>
  );
}
