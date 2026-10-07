import { withDataRevisionPreflight } from "../helpers/revision-fetch";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const auth = vi.hoisted(() => ({ getSupabase: vi.fn() }));
vi.mock("@/lib/supabase", () => ({ getSupabase: auth.getSupabase }));

import { activateBrowserAccount } from "@/lib/account-storage";
import { accountDataRequestRevision, invalidateAccountDataRequests } from "@/lib/account-data";
import { apiRequest } from "@/lib/api";

let key: string;
let accountSequence = 0;
const channels: MockChannel[] = [];
class MockChannel {
  onmessage: ((event: MessageEvent) => void) | null = null;
  postMessage = vi.fn();
  constructor() { channels.push(this); }
}

beforeEach(() => {
  vi.stubGlobal("BroadcastChannel", MockChannel);
  const account = `alice-${++accountSequence}`;
  activateBrowserAccount(account);
  key = `journalpulse_data_erasure_v1:${account}`;
  window.localStorage.clear();
  auth.getSupabase.mockReset().mockResolvedValue(null);
  accountDataRequestRevision();
  channels[0].postMessage.mockClear();
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

function remoteErasure(accountKey = key) {
  const token = crypto.randomUUID();
  window.localStorage.setItem(accountKey, token);
  window.dispatchEvent(new StorageEvent("storage", { key: accountKey, newValue: token }));
  return token;
}

test("a same-account erasure from another tab aborts an active request", async () => {
  let signal: AbortSignal | null | undefined;
  const request = vi.fn<typeof fetch>().mockImplementation((_url, init) => new Promise((_resolve, reject) => {
    signal = init?.signal;
    signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
  }));
  vi.stubGlobal("fetch", withDataRevisionPreflight(request));
  const result = apiRequest("/v1/journal/entries", { method: "POST", retry: true }).catch((reason: Error) => reason.message);
  await vi.waitFor(() => expect(request).toHaveBeenCalledOnce());
  remoteErasure();
  expect(await result).toContain("asked to delete");
  expect(signal?.aborted).toBe(true);
  expect(request).toHaveBeenCalledOnce();
});

test("the shared erasure token prevents an auth-delayed send before its storage event is delivered", async () => {
  let finishAuth!: (value: null) => void;
  auth.getSupabase.mockReturnValueOnce(new Promise<null>((resolve) => { finishAuth = resolve; }));
  const request = vi.fn().mockResolvedValue(new Response("{}"));
  vi.stubGlobal("fetch", withDataRevisionPreflight(request));
  const result = apiRequest("/v1/journal/entries", { method: "POST" }).catch((reason: Error) => reason.message);
  window.localStorage.setItem(key, crypto.randomUUID());
  finishAuth(null);
  expect(await result).toContain("asked to delete");
  expect(request).not.toHaveBeenCalled();
});

test("erasure during retry backoff never starts another write", async () => {
  vi.useFakeTimers();
  const request = vi.fn().mockRejectedValueOnce(new TypeError("Synthetic lost response"))
    .mockResolvedValue(new Response("{}"));
  vi.stubGlobal("fetch", withDataRevisionPreflight(request));
  const result = apiRequest("/v1/journal/entries", { method: "POST", retry: true }).catch((reason: Error) => reason.message);
  await vi.advanceTimersByTimeAsync(1);
  remoteErasure();
  await vi.runAllTimersAsync();
  expect(await result).toContain("asked to delete");
  expect(request).toHaveBeenCalledOnce();
});

test("another account's erasure and delayed old notifications leave a fresh save usable", async () => {
  const obsoleteToken = remoteErasure();
  remoteErasure();
  let finishAuth!: (value: null) => void;
  auth.getSupabase.mockReturnValueOnce(new Promise<null>((resolve) => { finishAuth = resolve; }));
  const request = vi.fn().mockResolvedValue(new Response(JSON.stringify({ saved: true })));
  vi.stubGlobal("fetch", withDataRevisionPreflight(request));
  const result = apiRequest("/v1/journal/entries", { method: "POST" });
  remoteErasure("journalpulse_data_erasure_v1:bob");
  window.dispatchEvent(new StorageEvent("storage", { key, newValue: obsoleteToken }));
  channels[0].onmessage?.(new MessageEvent("message", { data: { key, token: obsoleteToken, persisted: true } }));
  finishAuth(null);
  expect(await result).toEqual({ saved: true });
  expect(request).toHaveBeenCalledOnce();
});

test("BroadcastChannel cancellation works when persistent storage is blocked", async () => {
  vi.spyOn(window.localStorage, "getItem").mockImplementation(() => { throw new DOMException("Blocked", "SecurityError"); });
  vi.spyOn(window.localStorage, "setItem").mockImplementation(() => { throw new DOMException("Blocked", "SecurityError"); });
  let finishAuth!: (value: null) => void;
  auth.getSupabase.mockReturnValueOnce(new Promise<null>((resolve) => { finishAuth = resolve; }));
  const request = vi.fn().mockResolvedValue(new Response("{}"));
  vi.stubGlobal("fetch", withDataRevisionPreflight(request));
  const result = apiRequest("/v1/journal/entries", { method: "POST" }).catch((reason: Error) => reason.message);
  channels[0].onmessage?.(new MessageEvent("message", { data: { key, token: crypto.randomUUID(), persisted: false } }));
  finishAuth(null);
  expect(await result).toContain("asked to delete");
  expect(request).not.toHaveBeenCalled();
  const before = accountDataRequestRevision();
  invalidateAccountDataRequests();
  expect(accountDataRequestRevision()).not.toBe(before);
  expect(channels[0].postMessage).toHaveBeenCalledWith({ key, token: expect.any(String), persisted: false, erasedAtMs: expect.any(Number) });
});

test("local erasure publishes an account-scoped token without writing private content", () => {
  const before = accountDataRequestRevision();
  invalidateAccountDataRequests();
  expect(accountDataRequestRevision()).not.toBe(before);
  expect(channels[0].postMessage).toHaveBeenCalledWith({ key, token: window.localStorage.getItem(key), persisted: true, erasedAtMs: expect.any(Number) });
});

test("an older queued fallback notice cannot cancel fresh work after local erasure", async () => {
  vi.spyOn(window.localStorage, "getItem").mockImplementation(() => { throw new DOMException("Blocked", "SecurityError"); });
  vi.spyOn(window.localStorage, "setItem").mockImplementation(() => { throw new DOMException("Blocked", "SecurityError"); });
  const clock = vi.spyOn(performance, "now").mockReturnValue(1000);
  const earlierRemoteTime = Math.floor(performance.timeOrigin + performance.now());
  clock.mockReturnValue(2000);
  invalidateAccountDataRequests();
  clock.mockReturnValue(3000);
  invalidateAccountDataRequests();
  const before = accountDataRequestRevision();
  let finishAuth!: (value: null) => void;
  auth.getSupabase.mockReturnValueOnce(new Promise<null>((resolve) => { finishAuth = resolve; }));
  const request = vi.fn().mockResolvedValue(new Response(JSON.stringify({ saved: true })));
  vi.stubGlobal("fetch", withDataRevisionPreflight(request));
  const result = apiRequest("/v1/journal/entries", { method: "POST" }).catch((reason: Error) => reason.message);
  channels[0].onmessage?.(new MessageEvent("message", { data: {
    key, token: crypto.randomUUID(), persisted: false, erasedAtMs: earlierRemoteTime,
  } }));
  finishAuth(null);
  expect(await result).toEqual({ saved: true });
  expect(request).toHaveBeenCalledOnce();
  expect(accountDataRequestRevision()).toBe(before);
});

test.each(["newer", "equal", "legacy", "future", "invalid"] as const)(
  "%s remote fallback notices still cancel pending work after local erasure", async (kind) => {
    vi.spyOn(window.localStorage, "getItem").mockImplementation(() => { throw new DOMException("Blocked", "SecurityError"); });
    vi.spyOn(window.localStorage, "setItem").mockImplementation(() => { throw new DOMException("Blocked", "SecurityError"); });
    const clock = vi.spyOn(performance, "now").mockReturnValue(2000);
    invalidateAccountDataRequests();
    const localTime = Math.floor(performance.timeOrigin + performance.now());
    clock.mockReturnValue(3000);
    let finishAuth!: (value: null) => void;
    auth.getSupabase.mockReturnValueOnce(new Promise<null>((resolve) => { finishAuth = resolve; }));
    const request = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", withDataRevisionPreflight(request));
    const result = apiRequest("/v1/journal/entries", { method: "POST" }).catch((reason: Error) => reason.message);
    const erasedAtMs = kind === "newer" ? localTime + 500
      : kind === "equal" ? localTime : kind === "future" ? localTime + 10_000
        : kind === "invalid" ? Number.POSITIVE_INFINITY : undefined;
    channels[0].onmessage?.(new MessageEvent("message", { data: {
      key, token: crypto.randomUUID(), persisted: false, erasedAtMs,
    } }));
    finishAuth(null);
    expect(await result).toContain("asked to delete");
    expect(request).not.toHaveBeenCalled();
  },
);

test("an invalid future notice cannot suppress a later genuine erasure", async () => {
  vi.spyOn(window.localStorage, "getItem").mockImplementation(() => { throw new DOMException("Blocked", "SecurityError"); });
  vi.spyOn(window.localStorage, "setItem").mockImplementation(() => { throw new DOMException("Blocked", "SecurityError"); });
  const clock = vi.spyOn(performance, "now").mockReturnValue(2000);
  invalidateAccountDataRequests();
  const localTime = Math.floor(performance.timeOrigin + performance.now());
  channels[0].onmessage?.(new MessageEvent("message", { data: {
    key, token: crypto.randomUUID(), persisted: false, erasedAtMs: localTime + 10_000,
  } }));
  let finishAuth!: (value: null) => void;
  auth.getSupabase.mockReturnValueOnce(new Promise<null>((resolve) => { finishAuth = resolve; }));
  const request = vi.fn().mockResolvedValue(new Response("{}"));
  vi.stubGlobal("fetch", withDataRevisionPreflight(request));
  const result = apiRequest("/v1/journal/entries", { method: "POST" }).catch((reason: Error) => reason.message);
  clock.mockReturnValue(3000);
  channels[0].onmessage?.(new MessageEvent("message", { data: {
    key, token: crypto.randomUUID(), persisted: false, erasedAtMs: localTime + 500,
  } }));
  finishAuth(null);
  expect(await result).toContain("asked to delete");
  expect(request).not.toHaveBeenCalled();
});

test("a later remote erasure also supersedes an earlier queued fallback notice", () => {
  vi.spyOn(window.localStorage, "getItem").mockImplementation(() => { throw new DOMException("Blocked", "SecurityError"); });
  vi.spyOn(window.localStorage, "setItem").mockImplementation(() => { throw new DOMException("Blocked", "SecurityError"); });
  vi.spyOn(performance, "now").mockReturnValue(3000);
  const now = Math.floor(performance.timeOrigin + performance.now());
  channels[0].onmessage?.(new MessageEvent("message", { data: {
    key, token: crypto.randomUUID(), persisted: false, erasedAtMs: now - 1000,
  } }));
  const latest = accountDataRequestRevision();
  channels[0].onmessage?.(new MessageEvent("message", { data: {
    key, token: crypto.randomUUID(), persisted: false, erasedAtMs: now - 2000,
  } }));
  expect(accountDataRequestRevision()).toBe(latest);
});

test("an observed persisted erasure still orders old fallback notices if storage becomes blocked", () => {
  vi.spyOn(performance, "now").mockReturnValue(3000);
  const now = Math.floor(performance.timeOrigin + performance.now());
  const token = remoteErasure();
  channels[0].onmessage?.(new MessageEvent("message", { data: { key, token, persisted: true, erasedAtMs: now - 1000 } }));
  vi.spyOn(window.localStorage, "getItem").mockImplementation(() => { throw new DOMException("Blocked", "SecurityError"); });
  const before = accountDataRequestRevision();
  channels[0].onmessage?.(new MessageEvent("message", { data: {
    key, token: crypto.randomUUID(), persisted: false, erasedAtMs: now - 2000,
  } }));
  expect(accountDataRequestRevision()).toBe(before);
});
