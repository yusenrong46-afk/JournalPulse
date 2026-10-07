import { afterEach, beforeEach, expect, test, vi } from "vitest";

const auth = vi.hoisted(() => ({ getSupabase: vi.fn() }));
vi.mock("@/lib/supabase", () => ({ getSupabase: auth.getSupabase }));

import { apiRequest } from "@/lib/api";
import { activateBrowserAccount } from "@/lib/account-storage";

function deferred() {
  let resolve!: (value: unknown) => void;
  let reject!: (reason: Error) => void;
  const promise = new Promise((finish, fail) => { resolve = finish; reject = fail; });
  return { promise, resolve, reject };
}

beforeEach(() => {
  activateBrowserAccount("auth-deadline-fixture");
  auth.getSupabase.mockReset();
  vi.useFakeTimers();
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

const phases = ["sdk", "session", "401-sdk", "refresh"] as const;

test.each(phases.flatMap((phase) => ["deadline", "caller"].map((cause) => ({ phase, cause }))))(
  "$cause promptly settles preflight during $phase and late SDK resolution is harmless", async ({ phase, cause }) => {
    const gate = deferred();
    const session = { data: { session: { access_token: "synthetic-token", user: { id: "auth-deadline-fixture" } } } };
    const client = { auth: { getSession: vi.fn().mockResolvedValue(session), refreshSession: vi.fn().mockResolvedValue(session) } };
    auth.getSupabase.mockResolvedValue(client);
    if (phase === "sdk") auth.getSupabase.mockReturnValueOnce(gate.promise);
    if (phase === "session") client.auth.getSession.mockReturnValueOnce(gate.promise);
    if (phase === "401-sdk") auth.getSupabase.mockResolvedValueOnce(client).mockReturnValueOnce(gate.promise);
    if (phase === "refresh") client.auth.refreshSession.mockReturnValueOnce(gate.promise);
    const request = vi.fn<typeof fetch>().mockResolvedValue(new Response(JSON.stringify({ revision: 0 })));
    if (phase === "401-sdk" || phase === "refresh") request.mockResolvedValueOnce(new Response("{}", { status: 401 }));
    vi.stubGlobal("fetch", request);
    const caller = new AbortController();
    let settled = false;
    const result = apiRequest("/v1/journal/entries", { method: "POST", body: "{}", signal: caller.signal, timeoutMs: 125_000 })
      .then(() => { settled = true; return "unexpected success"; }, (reason: Error) => { settled = true; return reason.message; });
    await vi.advanceTimersByTimeAsync(1000);
    if (cause === "caller") caller.abort();
    await vi.advanceTimersByTimeAsync(cause === "deadline" ? 14_000 : 0);
    const callsBeforeLateResolution = request.mock.calls.length;
    try {
      expect(settled).toBe(true);
      expect(await result).toContain(cause === "deadline" ? "took too long" : "cancelled");
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      gate.resolve(phase === "sdk" || phase === "401-sdk" ? client : session);
      await vi.runAllTimersAsync();
      await result;
    }
    expect(request).toHaveBeenCalledTimes(callsBeforeLateResolution);
    expect(request.mock.calls.some(([, init]) => init?.method === "POST")).toBe(false);
    if (phase === "sdk") expect(client.auth.getSession).not.toHaveBeenCalled();
    if (phase === "401-sdk") expect(client.auth.refreshSession).not.toHaveBeenCalled();
  },
);

test.each(["sdk", "session"] as const)("late rejected %s promise is handled after cancellation", async (phase) => {
  const gate = deferred();
  const client = { auth: { getSession: vi.fn().mockReturnValue(gate.promise) } };
  auth.getSupabase.mockReturnValue(phase === "sdk" ? gate.promise : Promise.resolve(client));
  const request = vi.fn();
  vi.stubGlobal("fetch", request);
  const caller = new AbortController();
  let settled = false;
  const result = apiRequest("/v1/journal/entries", { method: "POST", body: "{}", signal: caller.signal })
    .catch((reason: Error) => { settled = true; return reason.message; });
  await vi.advanceTimersByTimeAsync(1);
  caller.abort();
  await vi.advanceTimersByTimeAsync(0);
  try { expect(settled).toBe(true); }
  finally {
    gate.reject(new Error("Synthetic late SDK rejection"));
    await vi.runAllTimersAsync();
    await result;
  }
  expect(await result).toContain("cancelled");
  expect(request).not.toHaveBeenCalled();
  expect(vi.getTimerCount()).toBe(0);
});
