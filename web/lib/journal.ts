import { apiRequest } from "./api";
import type { JournalEntry, JournalEntryPage, JournalReflectionResult } from "./journal-types";

const ENTRY_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function validJournalEntryId(value: string | null): string | null {
  return value && ENTRY_ID.test(value) ? value.toLowerCase() : null;
}

/** Retries retain one UUID and exact writing; the server returns the original entry. */
export function saveJournalEntry(text: string, requestId: string): Promise<JournalEntry> {
  return apiRequest("/v1/journal/entries", {
    method: "POST",
    body: JSON.stringify({ text, client_request_id: requestId }),
    retry: true,
  });
}

export function listJournalEntries(offset = 0, limit = 50): Promise<JournalEntryPage> {
  return apiRequest(`/v1/journal/entries?limit=${limit}&offset=${offset}`);
}

export function readJournalEntry(entryId: string, signal?: AbortSignal): Promise<JournalEntry> {
  return apiRequest(`/v1/journal/entries/${entryId}`, { signal });
}

export function deleteJournalEntry(entryId: string): Promise<void> {
  return apiRequest(`/v1/journal/entries/${entryId}`, { method: "DELETE" });
}

/** A paid reflection is never automatically replayed after a timeout. */
export function reflectJournalEntry(
  entryId: string,
  consent: boolean,
  locale: string,
  signal?: AbortSignal,
): Promise<JournalReflectionResult> {
  return apiRequest(`/v1/journal/entries/${entryId}/reflect`, {
    method: "POST",
    body: JSON.stringify({ llm_consent: consent, locale }),
    timeoutMs: 60_000,
    retry: false,
    signal,
  });
}
