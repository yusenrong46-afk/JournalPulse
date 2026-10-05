import { afterEach, describe, expect, test, vi } from "vitest";

import {
  type DiscoveryResponse,
  DISCOVERY_TIMEOUT_MS,
  excludedSources,
  refinementRequest,
  searchDiscovery,
} from "@/lib/discovery";

const previous: DiscoveryResponse = {
  original_query: "grounding exercises for work breaks",
  updated_query: "grounding exercises for work breaks short guides",
  candidates: [],
  provenance: {
    search_provider: "brave", prompt_version: "test-only", retrieved_at: "2026-10-04T00:00:00Z",
    candidate_count: 0, search_calls: 1, page_fetches: 0, model_runs: [],
  },
  limitations: ["Full pages were not read."],
};

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("discovery refinement", () => {
  test("preserves the original goal and prior query while excluding seen and manually rejected sources", () => {
    const payload = refinementRequest({
      previous, feedback: "  Text only please  ", consent: true,
      seen: ["https://example.org/seen"], manuallyExcluded: "https://example.org/rejected\nhttps://example.org/seen",
    });
    expect(payload).toEqual({
      original_query: previous.original_query,
      previous_query: previous.updated_query,
      feedback: "Text only please",
      excluded_urls: ["https://example.org/seen", "https://example.org/rejected"],
      llm_consent: true,
    });
    expect(Object.keys(payload)).not.toContain("journal_text");
    expect(Object.keys(payload)).not.toContain("conversation_id");
  });

  test("keeps exclusions bounded instead of dropping older URLs and repeating them", () => {
    expect(() => excludedSources(Array.from({ length: 31 }, (_, i) => `https://example.org/${i}`), ""))
      .toThrow("up to 30");
  });

  test("rejects missing feedback and links carrying credentials before a request", () => {
    expect(() => refinementRequest({ previous, feedback: " ", consent: true, seen: [], manuallyExcluded: "" }))
      .toThrow("what you would like to change");
    expect(() => excludedSources([], "https://person:password@example.org/"))
      .toThrow("without sign-in details");
    expect(() => excludedSources([], "http://example.org/"))
      .toThrow("HTTPS");
  });
});

describe("discovery requests", () => {
  test("sends only the approved stateless fields, with no automatic paid-call retry", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValueOnce(new Response(JSON.stringify(previous), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const payload = {
      original_query: previous.original_query, excluded_urls: [], llm_consent: true,
    };
    await expect(searchDiscovery(payload)).resolves.toEqual(previous);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/v1/discovery/search");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual(payload);
  });

  test("allows a multi-call refinement past the default API timeout, then aborts at its own deadline", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn<typeof fetch>().mockImplementation((_url, init) => new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
    }));
    vi.stubGlobal("fetch", fetchMock);
    let settled = false;
    const outcome = searchDiscovery({ original_query: "grounding exercises", excluded_urls: [], llm_consent: true })
      .catch((error: Error) => {
        settled = true;
        return error.message;
      });
    await vi.advanceTimersByTimeAsync(15_001);
    expect(settled).toBe(false);
    await vi.advanceTimersByTimeAsync(DISCOVERY_TIMEOUT_MS - 15_001);
    await expect(outcome).resolves.toContain("took too long");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test("shows provider unavailability without manufacturing or retrying results", async () => {
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValueOnce(new Response(
      JSON.stringify({ detail: "Web discovery is unavailable." }), { status: 503 },
    ));
    vi.stubGlobal("fetch", fetchMock);
    await expect(searchDiscovery({ original_query: "grounding exercises", excluded_urls: [], llm_consent: true }))
      .rejects.toThrow("Web discovery is unavailable");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
