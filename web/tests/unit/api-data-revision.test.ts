import { afterEach, beforeEach, expect, test, vi } from "vitest";

const auth = vi.hoisted(() => ({ getSupabase: vi.fn() }));
vi.mock("@/lib/supabase", () => ({ getSupabase: auth.getSupabase }));

import { apiRequest } from "@/lib/api";
import { activateBrowserAccount } from "@/lib/account-storage";
import { invalidateAccountDataRequests } from "@/lib/account-data";

const revisionPath = "/v1/account/data-revision";
const revisionHeader = "X-JournalPulse-Data-Revision";
const journalPath = "/v1/journal/entries";
const body = JSON.stringify({ text: "Synthetic writing submitted before erasure", client_request_id: "synthetic-receipt" });

function json(value: unknown, status = 200) { return new Response(JSON.stringify(value), { status }); }
function path(url: Parameters<typeof fetch>[0]) { return new URL(String(url), window.location.origin).pathname; }
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((finish) => { resolve = finish; });
  return { promise, resolve };
}

beforeEach(() => { activateBrowserAccount("revision-fixture"); auth.getSupabase.mockReset().mockResolvedValue(null); });
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

test("each mutation reads its revision before dispatch and retains it across normal retries", async () => {
  vi.useFakeTimers();
  let currentRevision = 4;
  const request = vi.fn<typeof fetch>().mockImplementation(async (url, init) => {
    if (path(url) === revisionPath) return json({ revision: currentRevision });
    const writes = request.mock.calls.filter(([, options]) => options?.method === "POST");
    expect(new Headers(init?.headers).get(revisionHeader)).toBe("4");
    if (writes.length === 1) { currentRevision = 5; throw new TypeError("Synthetic lost response"); }
    return json({ saved: true });
  });
  vi.stubGlobal("fetch", request);
  const result = apiRequest(journalPath, { method: "POST", body, retry: true, headers: { [revisionHeader]: "999" } })
    .catch((reason: Error) => reason);
  await vi.runAllTimersAsync();
  expect(await result).toEqual({ saved: true });
  expect(request.mock.calls.map(([url]) => path(url))).toEqual([revisionPath, journalPath, journalPath]);
  expect(request.mock.calls[0][1]?.body).toBeUndefined();
  expect(new Headers(request.mock.calls[0][1]?.headers).has(revisionHeader)).toBe(false);
});

test("erasure while the read-only revision preflight waits prevents the mutation", async () => {
  const pendingRevision = deferred<Response>();
  const request = vi.fn<typeof fetch>().mockImplementation(async (url) => path(url) === revisionPath
    ? pendingRevision.promise : json({ saved: true }));
  vi.stubGlobal("fetch", request);
  const result = apiRequest(journalPath, { method: "POST", body }).catch((reason: Error) => reason.message);
  await vi.waitFor(() => expect(request).toHaveBeenCalledOnce());
  invalidateAccountDataRequests();
  pendingRevision.resolve(json({ revision: 1 }));
  expect(await result).toContain("asked to delete");
  expect(request.mock.calls.map(([url]) => path(url))).toEqual([revisionPath]);
});

test("a revision captured before delayed SDK auth is never upgraded after another device erases data", async () => {
  const pendingAuth = deferred<null>();
  auth.getSupabase.mockResolvedValueOnce(null).mockReturnValueOnce(pendingAuth.promise);
  let currentRevision = 0;
  const headers: string[] = [];
  const request = vi.fn<typeof fetch>().mockImplementation(async (url, init) => {
    if (path(url) === revisionPath) return json({ revision: currentRevision });
    const observed = new Headers(init?.headers).get(revisionHeader);
    headers.push(String(observed));
    return observed === String(currentRevision) ? json({ saved: true }) : json({ detail: "Saved data was erased." }, 409);
  });
  vi.stubGlobal("fetch", request);
  const old = apiRequest(journalPath, { method: "POST", body, retry: true }).catch((reason: Error) => reason.message);
  await vi.waitFor(() => expect(auth.getSupabase).toHaveBeenCalledTimes(2));
  currentRevision = 1;
  pendingAuth.resolve(null);
  expect(await old).toContain("Saved data was erased");
  expect(headers).toEqual(["0"]);
  expect(request.mock.calls.map(([url]) => path(url))).toEqual([revisionPath, journalPath]);
  await expect(apiRequest(journalPath, { method: "POST", body: JSON.stringify({ text: "Fresh intentional writing" }) }))
    .resolves.toEqual({ saved: true });
  expect(headers).toEqual(["0", "1"]);
});

test("a dispatched request retains its old revision while server authentication is pending", async () => {
  const serverAuth = deferred<void>();
  let currentRevision = 0;
  let receivedHeader: string | null = null;
  const request = vi.fn<typeof fetch>().mockImplementation(async (url, init) => {
    if (path(url) === revisionPath) return json({ revision: currentRevision });
    if (path(url) === "/v1/account/data") { currentRevision += 1; return json({ deleted_records: 1 }); }
    receivedHeader = new Headers(init?.headers).get(revisionHeader);
    await serverAuth.promise;
    return receivedHeader === String(currentRevision) ? json({ saved: true }) : json({ detail: "Saved data was erased." }, 409);
  });
  vi.stubGlobal("fetch", request);
  const old = apiRequest(journalPath, { method: "POST", body, retry: true }).catch((reason: Error) => reason.message);
  await vi.waitFor(() => expect(request.mock.calls.some(([url]) => path(url) === journalPath)).toBe(true));
  await apiRequest("/v1/account/data", { method: "DELETE" });
  serverAuth.resolve();
  expect(await old).toContain("Saved data was erased");
  expect(receivedHeader).toBe("0");
  expect(request.mock.calls.map(([url]) => path(url))).toEqual([revisionPath, journalPath, "/v1/account/data"]);
});

test.each([-1, 0.5, Number.MAX_SAFE_INTEGER + 1, "0", null, undefined])(
  "invalid revision %s cannot authorize a write", async (revision) => {
    const request = vi.fn<typeof fetch>().mockImplementation(async (url) => path(url) === revisionPath
      ? json({ revision }) : json({ saved: true }));
    vi.stubGlobal("fetch", request);
    await expect(apiRequest(journalPath, { method: "POST", body })).rejects.toThrow("couldn’t confirm");
    expect(request.mock.calls.map(([url]) => path(url))).toEqual([revisionPath]);
  },
);

test("explicit account deletion and safe reads need no revision preflight", async () => {
  const request = vi.fn<typeof fetch>().mockResolvedValue(json({ deleted_records: 0 }));
  vi.stubGlobal("fetch", request);
  await apiRequest("/v1/account/data", { method: "DELETE" });
  expect(request.mock.calls.map(([url]) => path(url))).toEqual(["/v1/account/data"]);
  expect(new Headers(request.mock.calls[0][1]?.headers).has(revisionHeader)).toBe(false);
});

test("refreshing an expired SDK token never refreshes the captured data revision", async () => {
  const session = { access_token: "synthetic-token", user: { id: "revision-fixture" } };
  const refreshSession = vi.fn(async () => ({ data: { session } }));
  auth.getSupabase.mockResolvedValue({ auth: {
    getSession: async () => ({ data: { session } }), refreshSession,
  } });
  let writes = 0;
  const request = vi.fn<typeof fetch>().mockImplementation(async (url, init) => {
    if (path(url) === revisionPath) return json({ revision: 7 });
    expect(new Headers(init?.headers).get(revisionHeader)).toBe("7");
    return ++writes === 1 ? json({ detail: "Expired token" }, 401) : json({ detail: "Saved data was erased." }, 409);
  });
  vi.stubGlobal("fetch", request);
  await expect(apiRequest(journalPath, { method: "POST", body, retry: true })).rejects.toThrow("Saved data was erased");
  expect(refreshSession).toHaveBeenCalledOnce();
  expect(request.mock.calls.map(([url]) => path(url))).toEqual([revisionPath, journalPath, journalPath]);
});

test("the revision preflight has its own deadline and never sends a write after timeout", async () => {
  vi.useFakeTimers();
  const request = vi.fn<typeof fetch>().mockImplementation((_url, init) => new Promise((_resolve, reject) => {
    init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")), { once: true });
  }));
  vi.stubGlobal("fetch", request);
  const result = apiRequest(journalPath, { method: "POST", body, timeoutMs: 125_000 })
    .catch((reason: Error) => reason.message);
  await vi.advanceTimersByTimeAsync(15_000);
  expect(await result).toContain("took too long");
  expect(request.mock.calls.map(([url]) => path(url))).toEqual([revisionPath]);
  expect(vi.getTimerCount()).toBe(0);
});

test("a slow preflight does not consume the subsequent Luna request's deadline", async () => {
  vi.useFakeTimers();
  const request = vi.fn<typeof fetch>().mockImplementation((url, init) => new Promise((resolve, reject) => {
    const preflight = path(url) === revisionPath;
    const timer = setTimeout(() => resolve(json(preflight ? { revision: 2 } : { reply: "Synthetic late success" })), preflight ? 14_000 : 124_000);
    init?.signal?.addEventListener("abort", () => { clearTimeout(timer); reject(new DOMException("Aborted", "AbortError")); }, { once: true });
  }));
  vi.stubGlobal("fetch", request);
  const result = apiRequest("/v1/journal/entries/entry/reflect", { method: "POST", body, timeoutMs: 125_000, retry: false })
    .catch((reason: Error) => reason);
  await vi.advanceTimersByTimeAsync(138_000);
  expect(await result).toEqual({ reply: "Synthetic late success" });
  expect(request).toHaveBeenCalledTimes(2);
  expect(vi.getTimerCount()).toBe(0);
});

test.each(["caller", "account"] as const)("%s cancellation during revision preflight prevents the write", async (kind) => {
  const pendingRevision = deferred<Response>();
  const request = vi.fn<typeof fetch>().mockImplementation(async (url) => path(url) === revisionPath
    ? pendingRevision.promise : json({ saved: true }));
  vi.stubGlobal("fetch", request);
  const controller = new AbortController();
  const result = apiRequest(journalPath, { method: "POST", body, signal: controller.signal }).catch((reason: Error) => reason.message);
  await vi.waitFor(() => expect(request).toHaveBeenCalledOnce());
  if (kind === "caller") controller.abort();
  else activateBrowserAccount("another-account");
  pendingRevision.resolve(json({ revision: 1 }));
  expect(await result).toContain("cancelled");
  expect(request.mock.calls.map(([url]) => path(url))).toEqual([revisionPath]);
});
