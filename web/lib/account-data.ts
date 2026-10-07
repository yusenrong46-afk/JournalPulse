import { accountStorageKey } from "./account-storage";

export const ACCOUNT_DATA_CHANGED_EVENT = "journalpulse-account-data-changed";
const ERASURE_KEY = "journalpulse_data_erasure_v1";
const fallbackTokens = new Map<string, string>();
const erasureTimes = new Map<string, number>();
let listening = false;
let channel: BroadcastChannel | undefined;

type ErasureNotice = { key: string; token: string; persisted: boolean; erasedAtMs?: number };

function monotonicTime(): number {
  // These timestamps are comparable between same-origin windows and do not
  // move backwards when the system wall clock changes. Equal milliseconds are
  // deliberately ambiguous and must still invalidate older requests.
  return Math.floor(performance.timeOrigin + performance.now());
}

function noticeTime(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0 && value <= monotonicTime()
    ? value : undefined;
}

function notify() {
  window.dispatchEvent(new Event(ACCOUNT_DATA_CHANGED_EVENT));
}

function listen() {
  if (listening) return;
  listening = true;
  window.addEventListener("storage", (event) => {
    if (event.key === accountStorageKey(ERASURE_KEY)) notify();
  });
  // Storage events cover other tabs; BroadcastChannel also works when a browser
  // blocks persistent storage. Neither message contains the person's writing.
  if (typeof window.BroadcastChannel !== "function") return;
  try {
    channel = new window.BroadcastChannel(ERASURE_KEY);
    channel.onmessage = ({ data }: MessageEvent<ErasureNotice>) => {
      if (!data || data.key !== accountStorageKey(ERASURE_KEY)
        || typeof data.token !== "string" || typeof data.persisted !== "boolean") return;
      let readable = false;
      let stored: string | null = null;
      try { stored = window.localStorage.getItem(data.key); readable = true; } catch { /* Use the notice below. */ }
      const erasedAtMs = noticeTime(data.erasedAtMs);
      const latest = erasureTimes.get(data.key);
      // Read the current storage token when possible: an older queued notice
      // must not invalidate intentional work begun after the latest erasure.
      if (!data.persisted || !readable) {
        if (erasedAtMs !== undefined && latest !== undefined && erasedAtMs < latest) return;
        // Legacy/invalid/future timestamps cannot establish order. Cancel
        // conservatively, without letting that metadata suppress later notices.
        if (erasedAtMs !== undefined) erasureTimes.set(data.key, erasedAtMs);
        fallbackTokens.set(data.key, data.token);
      } else if (stored === data.token && erasedAtMs !== undefined && (latest === undefined || erasedAtMs > latest)) {
        // Retain the observed boundary if storage later becomes unavailable.
        erasureTimes.set(data.key, erasedAtMs);
      }
      notify();
    };
  } catch { /* The same-tab boundary and storage events remain available. */ }
}

/** Captured by every request, including requests still waiting for auth. */
export function accountDataRequestRevision(): string {
  listen();
  const key = accountStorageKey(ERASURE_KEY);
  if (!key) return "";
  let stored = "";
  try { stored = window.localStorage.getItem(key) ?? ""; } catch { /* In-memory fallback. */ }
  return `${fallbackTokens.get(key) ?? ""}:${stored}`;
}

/** Invalidate older work at both the start and successful end of account erasure. */
export function invalidateAccountDataRequests(): void {
  listen();
  const key = accountStorageKey(ERASURE_KEY);
  if (!key) return;
  const token = crypto.randomUUID();
  const erasedAtMs = monotonicTime();
  erasureTimes.set(key, erasedAtMs);
  let persisted = false;
  try { window.localStorage.setItem(key, token); persisted = true; }
  catch { fallbackTokens.set(key, token); }
  notify();
  try { channel?.postMessage({ key, token, persisted, erasedAtMs } satisfies ErasureNotice); }
  catch { /* Storage events still notify tabs with working storage. */ }
}
