import { readAccountStorage, writeAccountStorage } from "./account-storage";

const RECEIPT_KEY = "journalpulse_account_closure_v1";
const MAX_AGE_MS = 14 * 60 * 1000;

type ClosureReceipt = { request_id: string; receipt_secret: string; created_at: number };

/** A short-lived, action-only retry capability. Never put it in a URL or localStorage. */
export function closureReceipt(): Pick<ClosureReceipt, "request_id" | "receipt_secret"> {
  let storage: Storage | undefined;
  try { storage = window.sessionStorage; } catch { /* use the account-scoped memory fallback */ }
  const raw = storage ? readAccountStorage(RECEIPT_KEY, storage) : null;
  if (raw) {
    try {
      const value = JSON.parse(raw) as Partial<ClosureReceipt>;
      if (typeof value.request_id === "string" && /^[a-f0-9-]{36}$/.test(value.request_id)
          && typeof value.receipt_secret === "string" && /^[a-f0-9]{64}$/.test(value.receipt_secret)
          && typeof value.created_at === "number" && value.created_at <= Date.now()
          && Date.now() - value.created_at < MAX_AGE_MS) {
        return { request_id: value.request_id, receipt_secret: value.receipt_secret };
      }
    } catch { /* discard malformed local state */ }
  }
  const value: ClosureReceipt = {
    request_id: crypto.randomUUID(),
    receipt_secret: Array.from(crypto.getRandomValues(new Uint8Array(32)), (byte) => byte.toString(16).padStart(2, "0")).join(""),
    created_at: Date.now(),
  };
  // When session storage is blocked the caller retains this value in a component ref.
  if (storage) writeAccountStorage(RECEIPT_KEY, JSON.stringify(value), storage);
  return { request_id: value.request_id, receipt_secret: value.receipt_secret };
}
