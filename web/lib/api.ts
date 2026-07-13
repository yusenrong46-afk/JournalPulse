import { getSupabase } from "./supabase";

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

function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function apiBase(): string {
  if (CONFIGURED_API_BASE) return CONFIGURED_API_BASE;
  const localFrontend = ["localhost", "127.0.0.1"].includes(window.location.hostname)
    && window.location.port !== "8000";
  return localFrontend ? "http://127.0.0.1:8000" : "";
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

async function accessToken(refresh: boolean): Promise<string | undefined> {
  const supabase = await getSupabase();
  if (!supabase) return undefined;
  if (refresh) {
    const refreshed = await supabase.auth.refreshSession();
    return refreshed.data.session?.access_token;
  }
  return (await supabase.auth.getSession()).data.session?.access_token;
}

function mayRetry(method: string, init: ApiRequestInit): boolean {
  return method === "GET" || method === "HEAD" || init.retry === true;
}

export async function apiRequest<T>(path: string, init: ApiRequestInit = {}): Promise<T> {
  const method = (init.method ?? "GET").toUpperCase();
  const attempts = mayRetry(method, init) ? 2 : 1;
  let refreshedSession = false;
  let lastError: unknown;

  for (let attempt = 0; attempt < attempts; attempt += 1) {
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

    const token = await accessToken(refreshedSession);
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    } else {
      headers.set("X-JournalPulse-User", previewClientId());
    }

    const abortFromCaller = () => controller.abort();
    init.signal?.addEventListener("abort", abortFromCaller, { once: true });
    try {
      const response = await fetch(`${apiBase()}${path}`, {
        ...init,
        headers,
        signal: controller.signal,
        cache: "no-store",
      });
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
          detail?.detail ?? "JournalPulse could not complete the request.",
          response.status,
          requestId,
          retryAfter,
        );
        if (
          attempt + 1 < attempts
          && (response.status === 408 || response.status === 429 || response.status >= 500)
        ) {
          await wait(retryAfter ? Math.min(retryAfter * 1000, 3000) : 350 * (attempt + 1));
          lastError = error;
          continue;
        }
        throw error;
      }
      if (response.status === 204) return undefined as T;
      return response.json() as Promise<T>;
    } catch (reason) {
      lastError = reason;
      if (init.signal?.aborted) throw new ApiError("The request was cancelled.", 0);
      if (timedOut) {
        const timeoutError = new ApiError(
          "The request took too long. Your work is still on this page.",
          0,
        );
        if (attempt + 1 >= attempts) throw timeoutError;
        lastError = timeoutError;
        await wait(350 * (attempt + 1));
        continue;
      }
      if (reason instanceof ApiError || attempt + 1 >= attempts) throw reason;
      await wait(350 * (attempt + 1));
    } finally {
      window.clearTimeout(timeout);
      init.signal?.removeEventListener("abort", abortFromCaller);
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new ApiError("JournalPulse could not reach the reflection service.", 0);
}
