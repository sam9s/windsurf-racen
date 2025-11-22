import type { CatalogueItem, CatalogueMeta, CatalogueResponse, AnswerResponse } from "./types";

export interface QueryPayload extends Record<string, unknown> {
  text: string;
  limit: number;
}

// Use empty default so Next.js dev server rewrites (next.config.ts) proxy to 127.0.0.1:8011
const DEFAULT_ENDPOINT = "";
const DEFAULT_INTERPRET_MODE = "mock";

function getApiBase(): string {
  if (typeof process !== "undefined") {
    return process.env.NEXT_PUBLIC_RACEN_API ?? DEFAULT_ENDPOINT;
  }
  return DEFAULT_ENDPOINT;
}

function getInterpretMode(): string {
  if (typeof process !== "undefined") {
    return process.env.NEXT_PUBLIC_RACEN_INTERPRET_MODE ?? DEFAULT_INTERPRET_MODE;
  }
  return DEFAULT_INTERPRET_MODE;
}

export async function postJson<TResponse, TBody extends Record<string, unknown>>(
  path: string,
  body: TBody,
  init?: RequestInit
): Promise<TResponse> {
  const base = getApiBase().replace(/\/$/, "");
  const url = base ? `${base}${path}` : path; // when base is empty, use Next rewrite
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    ...init,
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return (await response.json()) as TResponse;
}

interface InterpretMatch {
  field: string;
  fragments?: string[];
}

interface InterpretHit {
  document: Record<string, unknown>;
  score?: number | null;
  matches?: InterpretMatch[];
}

interface InterpretResponse {
  query: string;
  mode: string;
  warnings?: string[];
  raw_response?: string;
  payload?: {
    q?: string;
    filters?: Record<string, unknown>;
    [key: string]: unknown;
  };
  hits?: InterpretHit[];
  applied_filters?: Array<{ field: string; match: string }>;
  total?: { value?: number; relation?: string };
}

function mapHitToItem(hit: InterpretHit): CatalogueItem {
  const doc = hit.document ?? {};
  const matches = hit.matches ?? [];
  const matchedFields = matches.map((match) => match.field);
  const matchedSnippets = matches.flatMap((match) => match.fragments ?? []);

  const tags = Array.isArray(doc.tags) ? (doc.tags as string[]) : [];

  return {
    sku: (doc.product_code as string) ?? undefined,
    brand: (doc.manufacturer as string) ?? undefined,
    category: (doc.device_type as string) ?? undefined,
    name: (doc.name as string) ?? undefined,
    channels: (doc.channels as number | null | undefined) ?? null,
    voltage: (doc.voltage as string | null | undefined) ?? null,
    price_inr: typeof doc.price === "number" ? (doc.price as number) : null,
    url: (doc.datasheet_url as string | null | undefined) ?? null,
    matched_fields: matchedFields.length ? matchedFields : undefined,
    matched_snippets: matchedSnippets.length ? matchedSnippets : undefined,
    notes: tags.length ? tags.slice(0, 5) : undefined,
  };
}

function buildMeta(response: InterpretResponse, count: number): CatalogueMeta {
  const resolvedFilters = (response.payload?.filters as Record<string, unknown> | undefined) ?? undefined;
  const warningMessage = response.warnings?.length ? response.warnings.join(" | ") : undefined;
  const attempts =
    response.applied_filters?.map((filter) => ({
      label: filter.field,
      active_filters: [filter.match],
      count,
    })) ?? undefined;

  return {
    message: warningMessage,
    attempts,
    resolved_filters: resolvedFilters,
    resolved_command: response.payload?.q ? String(response.payload.q) : undefined,
  };
}

function transformInterpretResponse(response: InterpretResponse): CatalogueResponse {
  const hits = response.hits ?? [];
  const items = hits.map(mapHitToItem);
  const total = response.total?.value ?? items.length;
  const meta = buildMeta(response, total);

  return {
    items,
    count: total,
    meta,
    summary: response.raw_response ?? null,
  };
}

export async function queryCatalogue(payload: QueryPayload, init?: RequestInit): Promise<CatalogueResponse> {
  const body = {
    text: payload.text,
    mode: getInterpretMode(),
    size: payload.limit,
  };

  const interpretResponse = await postJson<InterpretResponse, typeof body>("/interpret", body, init);
  return transformInterpretResponse(interpretResponse);
}

// New: Answer API client for RACEN Answer service (FastAPI)
export async function answerQuestion(args: {
  question: string;
  k?: number;
  previous_answer?: string;
  previous_user?: string;
}, init?: RequestInit): Promise<AnswerResponse> {
  const body = {
    question: args.question,
    k: args.k ?? 6,
    previous_answer: args.previous_answer ?? "",
    previous_user: args.previous_user ?? "",
  };
  // Goes through Next rewrite to http://127.0.0.1:8011/answer
  return await postJson<AnswerResponse, typeof body>("/api/answer", body, init);
}
