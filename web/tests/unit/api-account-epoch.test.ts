import { withDataRevisionPreflight } from "../helpers/revision-fetch";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const auth = vi.hoisted(() => ({ getSupabase: vi.fn() }));
vi.mock("@/lib/supabase", () => ({ getSupabase: auth.getSupabase }));

import { activateBrowserAccount, readAccountStorage, writeAccountStorage } from "@/lib/account-storage";
import { apiRequest } from "@/lib/api";

beforeEach(() => {
  activateBrowserAccount("alice");
  auth.getSupabase.mockResolvedValue(null);
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

describe("account switch history", () => {
  test("returning to the original account does not revive a response from its earlier workspace", async () => {
    let finishBody!: (value: unknown) => void;
    const response = {
      status: 200, ok: true, headers: new Headers(),
      json: vi.fn(() => new Promise((resolve) => { finishBody = resolve; })),
    };
    vi.stubGlobal("fetch", withDataRevisionPreflight(vi.fn().mockResolvedValue(response)));
    const result = apiRequest("/v1/journal/entries", { method: "POST", body: "{}" });
    await vi.waitFor(() => expect(response.json).toHaveBeenCalledOnce());
    activateBrowserAccount("bob");
    activateBrowserAccount("alice");
    finishBody({ text: "An obsolete response from Alice's prior workspace" });
    await expect(result).rejects.toThrow("account changed");
  });

  test("returning to the original account during backoff cannot restart an old write", async () => {
    vi.useFakeTimers();
    const fetchMock = vi.fn().mockResolvedValueOnce(new Response("{}", { status: 503 }))
      .mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", withDataRevisionPreflight(fetchMock));
    const result = apiRequest("/v1/conversations", { method: "POST", retry: true, body: "{}" })
      .catch((reason: Error) => reason.message);
    await vi.advanceTimersByTimeAsync(1);
    expect(fetchMock).toHaveBeenCalledOnce();
    activateBrowserAccount("bob");
    activateBrowserAccount("alice");
    await vi.runAllTimersAsync();
    await expect(result).resolves.toContain("account changed");
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  test("a failed removal masks a readable stale value for the same account", () => {
    const key = "epoch-review-stale-key";
    const storage = {
      getItem: () => "old private value",
      setItem: () => undefined,
      removeItem: () => { throw new DOMException("Storage is read-only", "SecurityError"); },
    };
    writeAccountStorage(key, null, storage);
    expect(readAccountStorage(key, storage)).toBeNull();
    activateBrowserAccount("bob");
    activateBrowserAccount("alice");
    expect(readAccountStorage(key, storage)).toBeNull();
  });
});
