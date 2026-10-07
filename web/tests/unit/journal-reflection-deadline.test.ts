import { withDataRevisionPreflight } from "../helpers/revision-fetch";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

vi.mock("@/lib/supabase", () => ({ getSupabase: async () => null }));

import { activateBrowserAccount } from "@/lib/account-storage";
import { reflectJournalEntry } from "@/lib/journal";

beforeEach(() => { activateBrowserAccount("reflection-fixture"); vi.useFakeTimers(); });
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

function delayedReflection(delay: number) {
  const request = vi.fn<typeof fetch>().mockImplementation((_url, init) => new Promise((resolve, reject) => {
    const timer = setTimeout(() => resolve(new Response(JSON.stringify({ reply: "A synthetic late success" }))), delay);
    init?.signal?.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    }, { once: true });
  }));
  vi.stubGlobal("fetch", withDataRevisionPreflight(request));
  return request;
}

test("a reflection completing inside the 100-second provider budget arrives without replay", async () => {
  const request = delayedReflection(90_000);
  const result = reflectJournalEntry("entry", true, "CA").catch((reason: Error) => reason);
  await vi.advanceTimersByTimeAsync(90_000);
  expect(await result).toEqual({ reply: "A synthetic late success" });
  expect(request).toHaveBeenCalledOnce();
});

test("exceeding the full request deadline reports failure and never replays generation", async () => {
  const request = delayedReflection(130_000);
  const result = reflectJournalEntry("entry", true, "CA").catch((reason: Error) => reason.message);
  await vi.advanceTimersByTimeAsync(125_000);
  expect(await result).toContain("took too long");
  expect(request).toHaveBeenCalledOnce();
  expect(vi.getTimerCount()).toBe(0);
});

test("explicit cancellation still stops a reflection before the longer deadline", async () => {
  const request = delayedReflection(90_000);
  const controller = new AbortController();
  const result = reflectJournalEntry("entry", true, "CA", controller.signal).catch((reason: Error) => reason.message);
  await vi.advanceTimersByTimeAsync(1000);
  controller.abort();
  expect(await result).toContain("cancelled");
  expect(request).toHaveBeenCalledOnce();
  expect(vi.getTimerCount()).toBe(0);
});
