export type Role = "user" | "assistant";

export interface CatalogueItem {
  sku?: string;
  brand?: string;
  category?: string;
  name?: string;
  channels?: number | null;
  voltage?: string | null;
  price_inr?: number | null;
  url?: string | null;
  matched_fields?: string[];
  matched_snippets?: string[];
  notes?: string[];
  [key: string]: unknown;
}

export interface AttemptRecord {
  label?: string;
  active_filters?: string[];
  count?: number;
}

export interface CatalogueMeta {
  variant?: string;
  message?: string | null;
  attempts?: AttemptRecord[];
  original_query?: string;
  resolved_filters?: Record<string, unknown>;
  resolved_command?: string;
  button_count?: number | null;
}

export interface AssistantPayload {
  items: CatalogueItem[];
  count: number;
  meta?: CatalogueMeta;
}

export interface CatalogueResponse extends AssistantPayload {
  summary?: string | null;
}

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
  payload?: AssistantPayload;
}

// RACEN Answer API types
export interface CitationOut {
  url: string;
  start_line: number;
  end_line: number;
}

export interface AnswerResponse {
  answer: string;
  citations: CitationOut[];
  settings_summary: string;
}
