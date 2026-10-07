import type { JournalEntry, PreparedAnalysis } from "./types";

export type { JournalEntry } from "./types";

export type JournalEntryPage = {
  items: JournalEntry[];
  limit: number;
  offset: number;
};

export type JournalReflectionResult = {
  entry_id: string;
  reply: string;
  safety: PreparedAnalysis["safety"];
  model_run: PreparedAnalysis["model_run"] & {
    prompt_version?: string | null;
    prompt_tokens?: number | null;
    completion_tokens?: number | null;
  };
  generated_text_retained: false;
  /** Tentative suggestions from this reflection only; absent on older APIs. */
  feelings?: string[];
};
