import { apiRequest } from "./api";
import { LUNA_REQUEST_TIMEOUT_MS } from "./request-deadlines";
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
    timeoutMs: LUNA_REQUEST_TIMEOUT_MS,
    retry: false,
    signal,
  });
}

/**
 * "Sat, Sep 28" plus how long ago it was written, so an old entry is not mistaken for how
 * the person feels today. `now` is injectable for tests.
 */
export function entryDateLabel(createdAt: string, now: Date = new Date()): { date: string; age: string } {
  const written = new Date(createdAt);
  const date = written.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
  const startOf = (value: Date) => new Date(value.getFullYear(), value.getMonth(), value.getDate()).getTime();
  const days = Math.round((startOf(now) - startOf(written)) / 86_400_000);
  const age = days <= 0 ? "written today" : days === 1 ? "written yesterday" : `written ${days} days ago`;
  return { date, age };
}
