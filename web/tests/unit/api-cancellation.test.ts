import { withDataRevisionPreflight } from "../helpers/revision-fetch";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const auth = vi.hoisted(() => ({ getSupabase: vi.fn() }));
vi.mock("@/lib/supabase", () => ({ getSupabase: auth.getSupabase }));

import { apiRequest } from "@/lib/api";
import { activateBrowserAccount } from "@/lib/account-storage";

beforeEach(() => { activateBrowserAccount("alice"); auth.getSupabase.mockReset().mockResolvedValue(null); });
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("request cancellation", () => {
  test("switching accounts during authentication never sends the prior account's writing", async () => {
    let finishAuth!: (value: null) => void;
    auth.getSupabase.mockReturnValue(new Promise<null>((resolve) => { finishAuth = resolve; }));
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", withDataRevisionPreflight(fetchMock));
    const result = apiRequest("/v1/journal/entries", { method: "POST", body: JSON.stringify({ text: "Alice's private draft" }) })
      .catch((reason: Error) => reason.message);
    activateBrowserAccount("bob");
    finishAuth(null);
    await expect(result).resolves.toContain("account changed");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test("a delayed prior-account response is rejected after an account switch", async () => {
    let finish!: (value: Response) => void;
    const fetchMock = vi.fn().mockImplementation(() => new Promise<Response>((resolve) => { finish = resolve; }));
    vi.stubGlobal("fetch", withDataRevisionPreflight(fetchMock));
    const result = apiRequest("/v1/journal/entries", { method: "POST", body: "{}" })
      .catch((reason: Error) => reason.message);
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledOnce());
    activateBrowserAccount("bob");
    finish(new Response(JSON.stringify({ text: "Alice's saved writing" })));
    await expect(result).resolves.toContain("account changed");
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  test("a token from another account cannot authorize the mounted account's draft", async () => {
    auth.getSupabase.mockResolvedValue({ auth: { getSession: async () => ({ data: {
      session: { access_token: "bob-test-token", user: { id: "bob" } },
    } }) } });
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", withDataRevisionPreflight(fetchMock));
    await expect(apiRequest("/v1/journal/entries", { method: "POST", body: JSON.stringify({ text: "Alice's private draft" }) }))
      .rejects.toThrow("account changed");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test("an already cancelled write never starts authentication or fetch", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", withDataRevisionPreflight(fetchMock));
    const caller = new AbortController();
    caller.abort();
    await expect(apiRequest("/v1/discovery/search", { method: "POST", signal: caller.signal }))
      .rejects.toThrow("cancelled");
    expect(auth.getSupabase).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test("cancelling while authentication is pending prevents the paid request", async () => {
    let finishAuth!: (value: null) => void;
    auth.getSupabase.mockReturnValue(new Promise<null>((resolve) => { finishAuth = resolve; }));
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", withDataRevisionPreflight(fetchMock));
    const caller = new AbortController();
    const result = apiRequest("/v1/journal/entries/entry/reflect", { method: "POST", signal: caller.signal })
      .catch((reason: Error) => reason.message);
    caller.abort();
    finishAuth(null);
    await expect(result).resolves.toContain("cancelled");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  test("cancelling during retry backoff cannot start another write", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn().mockResolvedValueOnce(new Response("{}", { status: 503 }))
      .mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", withDataRevisionPreflight(fetchMock));
    const caller = new AbortController();
    const result = apiRequest("/v1/conversations", { method: "POST", retry: true, signal: caller.signal })
      .catch((reason: Error) => reason.message);
    await vi.advanceTimersByTimeAsync(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    caller.abort();
    await vi.runAllTimersAsync();
    await expect(result).resolves.toContain("cancelled");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  test("failed authentication cleans up the request deadline", async () => {
    vi.useFakeTimers();
    auth.getSupabase.mockRejectedValue(new Error("auth unavailable"));
    vi.stubGlobal("fetch", vi.fn());
    await expect(apiRequest("/v1/discovery/search", { method: "POST" })).rejects.toThrow("auth unavailable");
    expect(vi.getTimerCount()).toBe(0);
  });

  test.each(["caller", "deadline"] as const)("%s cancellation stays active during a slow response body", async (kind) => {
    vi.useFakeTimers();
    let body!: ReadableStreamDefaultController<Uint8Array>;
    let fetchSignal: AbortSignal | null | undefined;
    vi.stubGlobal("fetch", withDataRevisionPreflight(vi.fn<typeof fetch>().mockImplementation(async (_url, options) => {
      fetchSignal = options?.signal;
      const stream = new ReadableStream<Uint8Array>({ start(controller) { body = controller; } });
      fetchSignal?.addEventListener("abort", () => body.error(new DOMException("Aborted", "AbortError")), { once: true });
      return new Response(stream);
    })));
    const caller = new AbortController();
    let failure = "";
    const result = apiRequest("/v1/discovery/search", { method: "POST", signal: caller.signal, timeoutMs: 100 })
      .catch((reason: Error) => { failure = reason.message; });
    await vi.advanceTimersByTimeAsync(1);
    if (kind === "caller") caller.abort();
    await vi.advanceTimersByTimeAsync(100);
    try {
      expect(fetchSignal?.aborted).toBe(true);
      expect(failure).toContain(kind === "caller" ? "cancelled" : "took too long");
    } finally {
      if (!fetchSignal?.aborted) { body.enqueue(new TextEncoder().encode("{}")); body.close(); }
      await result;
    }
  });
});
