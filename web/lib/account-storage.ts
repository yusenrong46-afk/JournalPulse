/** Browser data is scoped before an authenticated workspace is mounted. */
type AccountScope = string | null | undefined;
export const ACCOUNT_CHANGED_EVENT = "journalpulse-account-changed";

// Undefined is the local, unsigned preview. Null is a configured app with no session.
let activeAccount: AccountScope;
let accountRevision = 0;
const sessionValues = new Map<string, string | null>();

export function browserAccount(): AccountScope {
  return activeAccount;
}

export function browserAccountRevision(): number {
  return accountRevision;
}

export function activateBrowserAccount(userId: string | null): void {
  if (activeAccount === userId) return;
  activeAccount = userId;
  accountRevision += 1;
  window.dispatchEvent(new Event(ACCOUNT_CHANGED_EVENT));
}

export function accountStorageKey(baseKey: string): string | null {
  if (activeAccount === null) return null;
  return activeAccount ? `${baseKey}:${activeAccount}` : baseKey;
}

type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;

export function readAccountStorage(baseKey: string, storage?: Pick<StorageLike, "getItem">): string | null {
  const key = accountStorageKey(baseKey);
  if (!key) return null;
  if (sessionValues.has(key)) return sessionValues.get(key)!;
  try {
    return (storage ?? window.localStorage).getItem(key);
  } catch {
    return null;
  }
}

export function writeAccountStorage(baseKey: string, value: string | null, storage?: StorageLike): void {
  const key = accountStorageKey(baseKey);
  if (!key) return;
  try {
    const target = storage ?? window.localStorage;
    if (value === null) target.removeItem(key);
    else target.setItem(key, value);
    sessionValues.delete(key);
  } catch {
    // A blocked/quota-limited store must not crash the workspace. The explicit
    // choice stays usable in memory and is never transferred to another account.
    sessionValues.set(key, value);
  }
}
