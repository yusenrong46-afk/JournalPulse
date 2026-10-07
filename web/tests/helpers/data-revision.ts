import { type APIRequestContext, expect } from "@playwright/test";

/** Direct integration setup writes follow the same protocol as browser writes. */
export async function currentDataRevisionHeaders(request: APIRequestContext, headers: Record<string, string>) {
  const response = await request.get("/v1/account/data-revision", { headers });
  expect(response.ok()).toBeTruthy();
  const { revision } = await response.json();
  expect(Number.isSafeInteger(revision) && revision >= 0).toBe(true);
  return { ...headers, "X-JournalPulse-Data-Revision": String(revision) };
}
