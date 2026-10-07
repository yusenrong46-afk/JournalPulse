import { beforeEach, expect, test, vi } from "vitest";
import { activateBrowserAccount } from "@/lib/account-storage";
import { invalidateAccountDataRequests } from "@/lib/account-data";
import { chatDraftKey, clearSourceDrafts, clearTabSession, readTabValue, tabValueIsVolatile, writeTabValue } from "@/lib/tab-session";

beforeEach(() => { window.sessionStorage.clear(); clearTabSession(); activateBrowserAccount("audit-a"); clearTabSession(); });

test("drafts are scoped to account, chat incarnation and selected source", () => {
  const old = { id: "chat", incarnation_id: "first", source_entry_id: "entry" };
  writeTabValue(chatDraftKey(old), "Private writing");
  expect(readTabValue(chatDraftKey({ ...old, incarnation_id: "second" }))).toBe("");
  clearSourceDrafts("entry");
  expect(readTabValue(chatDraftKey(old))).toBe("");
  writeTabValue("journal", "Account A");
  activateBrowserAccount("audit-b"); expect(readTabValue("journal")).toBe("");
  activateBrowserAccount("audit-a"); expect(readTabValue("journal")).toBe("");
});

test("erasure clears writing and sign-out prevents storing it", () => {
  writeTabValue("journal", "Unsaved"); invalidateAccountDataRequests();
  expect(readTabValue("journal")).toBe("");
  activateBrowserAccount(null); writeTabValue("journal", "No account");
  activateBrowserAccount("audit-a"); expect(readTabValue("journal")).toBe("");
});

test("blocked storage preserves writing in memory and warns before refresh", () => {
  const denied = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new DOMException("Blocked"); });
  writeTabValue("journal", "Exact writing\n🙂");
  expect(readTabValue("journal")).toBe("Exact writing\n🙂");
  expect(tabValueIsVolatile("journal")).toBe(true);
  const unload = new Event("beforeunload", { cancelable: true }); window.dispatchEvent(unload);
  expect(unload.defaultPrevented).toBe(true);
  writeTabValue("journal", "");
  const empty = new Event("beforeunload", { cancelable: true }); window.dispatchEvent(empty);
  expect(empty.defaultPrevented).toBe(false);
  denied.mockRestore();
});
