import { getSupabase } from "./supabase";
import { ACCOUNT_CHANGED_EVENT, browserAccount, browserAccountRevision } from "./account-storage";

const CONFIGURED_API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "").replace(/\/$/, "");
const DEFAULT_TIMEOUT_MS = 15_000;
const CLIENT_ID_KEY = "journalpulse-preview-client-id";

export type ApiRequestInit = RequestInit & {
  timeoutMs?: number;
  retry?: boolean;
};

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public requestId?: string,
    public retryAfterSeconds?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function wait(milliseconds: number, signal?: AbortSignal | null): Promise<void> {
  return new Promise((resolve, reject) => {
    const cancel = () => {
      window.clearTimeout(timer);
      signal?.removeEventListener("abort", cancel);
      reject(new ApiError("The request was cancelled.", 0));
    };
    const timer = window.setTimeout(() => {
      signal?.removeEventListener("abort", cancel);
      resolve();
    }, milliseconds);
    signal?.addEventListener("abort", cancel, { once: true });
    if (signal?.aborted) cancel();
  });
}

function apiBase(): string {
  if (CONFIGURED_API_BASE) return CONFIGURED_API_BASE;
  // Only the Next.js dev server (port 3000) talks to a separate local API. Every other
  // origin, including FastAPI serving the exported site, is same-origin.
  const nextDevServer = ["localhost", "127.0.0.1"].includes(window.location.hostname)
    && window.location.port === "3000";
  return nextDevServer ? "http://127.0.0.1:8000" : "";
}

function previewClientId(): string {
  if (process.env.NEXT_PUBLIC_DEV_USER_ID) return process.env.NEXT_PUBLIC_DEV_USER_ID;
  try {
    const existing = window.localStorage.getItem(CLIENT_ID_KEY);
    if (existing) return existing;
    const created = crypto.randomUUID();
    window.localStorage.setItem(CLIENT_ID_KEY, created);
    return created;
  } catch {
    return "00000000-0000-4000-8000-000000000001";
  }
}

async function accessToken(refresh: boolean, expectedAccount: string | null | undefined): Promise<string | undefined> {
  const supabase = await getSupabase();
  if (!supabase) return undefined;
  const result = refresh ? await supabase.auth.refreshSession() : await supabase.auth.getSession();
  const session = result.data.session;
  if (expectedAccount && session?.user.id !== expectedAccount) {
    // Another tab can update SDK storage before its auth event reaches this tab.
    // Never send the mounted account's words with that other account's token.
    throw new ApiError("The request was cancelled because your account changed.", 0);
  }
  return session?.access_token;
}

function mayRetry(method: string, init: ApiRequestInit): boolean {
  return method === "GET" || method === "HEAD" || init.retry === true;
}

function errorDetail(detail: unknown): string {
  if (typeof detail === "string" && detail.trim()) return detail.trim().slice(0, 600);
  if (Array.isArray(detail)) {
    // FastAPI validation details also contain submitted input and context. Show
    // only bounded validation messages, never serialize the private request.
    const labels: Record<string, string> = {
      text: "Message", feedback: "Search feedback", original_query: "Search topic", note: "Note",
    };
    const messages = detail.flatMap((item: unknown) => {
      if (!item || typeof item !== "object" || !("msg" in item) || typeof item.msg !== "string") return [];
      const location = "loc" in item && Array.isArray(item.loc) ? item.loc.at(-1) : undefined;
      const label = typeof location === "string" ? labels[location] : undefined;
      return [`${label ? `${label}: ` : ""}${item.msg.trim().slice(0, 180)}`];
    }).filter(Boolean);
    if (messages.length) return [...new Set(messages)].slice(0, 3).join("; ").slice(0, 600);
  }
  return "JournalPulse could not complete the request. Please check your answers and try again.";
}

export async function apiRequest<T>(path: string, init: ApiRequestInit = {}): Promise<T> {
  const startedAccount = browserAccount();
  const startedRevision = browserAccountRevision();
  // Returning to the same account does not revive requests from an earlier
  // workspace. Capture the whole switch history, including retry backoff.
  const accountChanged = () => browserAccountRevision() !== startedRevision;
  const accountError = () => new ApiError("The request was cancelled because your account changed.", 0);
  const method = (init.method ?? "GET").toUpperCase();
  const attempts = mayRetry(method, init) ? 2 : 1;
  let refreshedSession = false;
  let lastError: unknown;

  for (let attempt = 0; attempt < attempts; attempt += 1) {
    if (accountChanged()) throw accountError();
    if (init.signal?.aborted) throw new ApiError("The request was cancelled.", 0);
    const controller = new AbortController();
    let timedOut = false;
    const timeout = window.setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, init.timeoutMs ?? DEFAULT_TIMEOUT_MS);
    const headers = new Headers(init.headers);
    headers.set("Accept", "application/json");
    headers.set("X-Request-ID", crypto.randomUUID());
    if (init.body) headers.set("Content-Type", "application/json");

    // Register before awaiting auth: leaving a page must not start a paid request
    // after a delayed session lookup completes.
    const abortFromCaller = () => controller.abort();
    const abortFromAccount = () => { if (accountChanged()) controller.abort(); };
    init.signal?.addEventListener("abort", abortFromCaller, { once: true });
    window.addEventListener(ACCOUNT_CHANGED_EVENT, abortFromAccount);
    try {
      const token = await accessToken(refreshedSession, startedAccount);
      if (accountChanged()) throw accountError();
      controller.signal.throwIfAborted();
      if (token) {
        headers.set("Authorization", `Bearer ${token}`);
      } else {
        headers.set("X-JournalPulse-User", previewClientId());
      }
      const response = await fetch(`${apiBase()}${path}`, {
        ...init,
        headers,
        signal: controller.signal,
        cache: "no-store",
      });
      if (accountChanged()) throw accountError();
      const requestId = response.headers.get("X-Request-ID") ?? undefined;
      if (response.status === 401 && (await getSupabase()) && !refreshedSession) {
        refreshedSession = true;
        attempt -= 1;
        continue;
      }
      if (!response.ok) {
        const detail = await response.json().catch(() => null);
        const retryAfter = Number(response.headers.get("Retry-After")) || undefined;
        const error = new ApiError(
          errorDetail(detail?.detail),
          response.status,
          requestId,
          retryAfter,
        );
        if (
          attempt + 1 < attempts
          && (response.status === 408 || response.status === 429 || response.status >= 500)
        ) {
          await wait(retryAfter ? Math.min(retryAfter * 1000, 3000) : 350 * (attempt + 1), init.signal);
          lastError = error;
          continue;
        }
        throw error;
      }
      if (response.status === 204) return undefined as T;
      // Keep the deadline and caller cancellation active until the body is read.
      const result = await response.json() as T;
      if (accountChanged()) throw accountError();
      return result;
    } catch (reason) {
      lastError = reason;
      if (accountChanged()) throw accountError();
      if (init.signal?.aborted) throw new ApiError("The request was cancelled.", 0);
      if (timedOut) {
        const timeoutError = new ApiError(
          "The request took too long. Your work is still on this page.",
          0,
        );
        if (attempt + 1 >= attempts) throw timeoutError;
        lastError = timeoutError;
        await wait(350 * (attempt + 1), init.signal);
        continue;
      }
      if (reason instanceof ApiError || attempt + 1 >= attempts) throw reason;
      await wait(350 * (attempt + 1), init.signal);
    } finally {
      window.clearTimeout(timeout);
      init.signal?.removeEventListener("abort", abortFromCaller);
      window.removeEventListener(ACCOUNT_CHANGED_EVENT, abortFromAccount);
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new ApiError("JournalPulse could not reach the reflection service.", 0);
}
