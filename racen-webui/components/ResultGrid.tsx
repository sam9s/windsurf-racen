"use client";

import styles from "./ResultGrid.module.css";
import type { AssistantPayload, CatalogueItem, CatalogueMeta } from "@/lib/types";

interface ResultGridProps {
  items: CatalogueItem[];
  count: number;
  meta?: CatalogueMeta;
}

const rupeeFormatter = new Intl.NumberFormat("en-IN");

function formatPrice(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "-";
  }
  return `Rs ${rupeeFormatter.format(value)}`;
}

function formatChannels(value: number | null | undefined): string {
  if (value === null || value === undefined) return "-";
  return `${value}ch`;
}

function formatText(value: unknown): string {
  if (value === null || value === undefined) return "-";
  const text = String(value).trim();
  return text.length ? text : "-";
}

function extractButtonCounts(items: CatalogueItem[]): number[] {
  const counts = new Set<number>();
  const regex = /(\d{1,2})\s*-?\s*button/gi;
  for (const item of items) {
    const fields = [item.name, item.specs, ...(item.notes ?? []), ...(item.matched_snippets ?? [])];
    for (const field of fields) {
      if (!field) continue;
      const text = String(field);
      let match: RegExpExecArray | null;
      while ((match = regex.exec(text)) !== null) {
        const value = Number.parseInt(match[1], 10);
        if (!Number.isNaN(value)) {
          counts.add(value);
        }
      }
    }
  }
  return Array.from(counts).sort((a, b) => a - b);
}

function formatFilterValue(value: unknown): string {
  if (Array.isArray(value)) {
    return value.map((entry) => formatText(entry)).join(", ");
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return formatText(value);
}

export function buildAssistantSummary(payload: AssistantPayload | undefined): string {
  if (!payload) return "";
  const segments: string[] = [];
  const resolved = payload.meta?.resolved_command;
  if (resolved) {
    segments.push(`Interpreted as "${resolved}"`);
  }
  if (payload.count === 0) {
    segments.push(payload.meta?.message ?? "No products matched the current filters.");
    return segments.join("\n");
  }
  const head = `Found ${payload.count} product${payload.count === 1 ? "" : "s"} (showing top ${payload.items.length}).`;
  segments.push(head);
  if (payload.meta?.message) {
    segments.push(payload.meta.message);
  }
  const buttonCount = payload.meta?.button_count ?? null;
  if (buttonCount) {
    const observedCounts = extractButtonCounts(payload.items).filter((count) => count !== buttonCount);
    if (observedCounts.length) {
      const formatted =
        observedCounts.length === 1
          ? `${observedCounts[0]}-button`
          : `${observedCounts.slice(0, -1).map((n) => `${n}-button`).join(", ")} and ${observedCounts.at(-1)}-button`;
      segments.push(
        `Should I widen to ${formatted} options or try another brand? Just say "show ${observedCounts[0]} button" or "try another brand" to steer me.`
      );
    } else {
      segments.push(`Want me to widen beyond ${buttonCount}-button models or hop to another brand?`);
    }
  }
  return segments.join("\n");
}

export default function ResultGrid({ items, count, meta }: ResultGridProps) {
  const summaryText =
    count === 0
      ? meta?.message ?? "No products matched the current filters."
      : `Showing top ${Math.min(items.length, count)} result${count === 1 ? "" : "s"} of ${count}.`;

  return (
    <div className={styles.wrapper}>
      <header className={styles.summaryRow}>
        <span className={styles.summaryText}>{summaryText}</span>
        {meta?.resolved_command ? (
          <span className={styles.commandBadge}>
            Resolved as <code>{meta.resolved_command}</code>
          </span>
        ) : null}
        {meta?.resolved_filters ? (
          <span className={styles.filterSummary}>
            Filters:{" "}
            {Object.entries(meta.resolved_filters)
              .map(([key, value]) => `${key}=${formatFilterValue(value)}`)
              .join(", ")}
          </span>
        ) : null}
      </header>

      {items.length ? (
        <>
          <table className={styles.table}>
            <thead>
              <tr>
                <th scope="col">SKU</th>
                <th scope="col">Product</th>
                <th scope="col">Brand / Category</th>
                <th scope="col">Channels</th>
                <th scope="col">Voltage</th>
                <th scope="col" className={styles.money}>
                  Price
                </th>
              </tr>
            </thead>
            <tbody>
              {items.map((item, index) => {
                const key = item.sku ?? item.name ?? `row-${index}`;
                return (
                  <tr key={key}>
                    <td>{formatText(item.sku)}</td>
                    <td>{formatText(item.name)}</td>
                    <td>
                      <div className={styles.brand}>{formatText(item.brand)}</div>
                      <div className={styles.category}>{formatText(item.category)}</div>
                    </td>
                    <td>{formatChannels(item.channels)}</td>
                    <td>{formatText(item.voltage)}</td>
                    <td className={styles.money}>{formatPrice(item.price_inr)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <section className={styles.insights} aria-label="Why these matched">
            <h4>Why these matched</h4>
            <ul>
              {items.map((item, index) => {
                const key = item.sku ?? item.name ?? `why-${index}`;
                const bullets: string[] = [];
                if (item.matched_fields?.length) {
                  bullets.push(`Matched fields: ${item.matched_fields.join(", ")}`);
                }
                if (item.matched_snippets?.length) {
                  bullets.push(...item.matched_snippets.slice(0, 2));
                }
                if (item.notes?.length) {
                  bullets.push(...item.notes.slice(0, 2));
                }
                if (!bullets.length) {
                  bullets.push("No specific match metadata returned.");
                }
                const title = item.name ?? item.sku ?? `Result ${index + 1}`;
                return (
                  <li key={key}>
                    <strong>{title}</strong>
                    <ul>
                      {bullets.map((bullet, idx) => (
                        <li key={idx}>{bullet}</li>
                      ))}
                    </ul>
                  </li>
                );
              })}
            </ul>
          </section>
        </>
      ) : (
        <p className={styles.emptyState}>Try loosening the filters or removing the brand constraint.</p>
      )}

      {meta?.attempts?.length ? (
        <details className={styles.attempts}>
          <summary>Filter attempts</summary>
          <ol>
            {meta.attempts.map((attempt, index) => (
              <li key={`${attempt.label ?? "attempt"}-${index}`}>
                <span className={styles.attemptLabel}>{attempt.label ?? `Attempt ${index + 1}`}</span>
                {attempt.active_filters?.length ? (
                  <span className={styles.attemptFilters}>
                    Filters: {attempt.active_filters.join(", ")} | {attempt.count ?? 0} hit(s)
                  </span>
                ) : null}
              </li>
            ))}
          </ol>
        </details>
      ) : null}
    </div>
  );
}
