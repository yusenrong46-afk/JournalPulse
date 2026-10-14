import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const mocks = vi.hoisted(() => ({ replace: vi.fn(), api: vi.fn(), signOut: vi.fn(),
  getUser: vi.fn(), signIn: vi.fn(), getClient: vi.fn() }));
vi.mock("next/navigation", () => {
  const router = { replace: mocks.replace };
  return { useRouter: () => router };
});
vi.mock("@/lib/api", () => ({ apiRequest: mocks.api,
  ApiError: class extends Error { constructor(message: string, public status: number) { super(message); } } }));
vi.mock("@/lib/supabase", () => ({ getSupabase: mocks.getClient }));

import { CloseAccount } from "@/components/close-account";
import { activateBrowserAccount, browserAccount, clearClosedAccountStorage } from "@/lib/account-storage";
import { ApiError } from "@/lib/api";

const OWNER = "11111111-1111-4111-8111-111111111111";
const OTHER = "22222222-2222-4222-8222-222222222222";
let root: Root;
let container: HTMLDivElement;
const client = { auth: { signOut: mocks.signOut, getUser: mocks.getUser, signInWithOtp: mocks.signIn } };

beforeEach(async () => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  activateBrowserAccount(OWNER);
  clearClosedAccountStorage(OWNER);
  window.sessionStorage.clear(); window.localStorage.clear();
  mocks.replace.mockReset(); mocks.signOut.mockReset().mockResolvedValue({ error: null });
  mocks.getUser.mockReset().mockResolvedValue({ data: { user: { id: OWNER, email: "fixture@example.test" } }, error: null });
  mocks.signIn.mockReset().mockResolvedValue({ error: null });
  mocks.getClient.mockReset().mockResolvedValue(client);
  mocks.api.mockReset().mockImplementation(async (path) => path === "/v1/system/status" ? { account_closure: "configured" } : undefined);
  container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container);
  await act(async () => root.render(createElement(CloseAccount, { onBusyChange: vi.fn() })));
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); activateBrowserAccount(null); });
async function confirm() {
  await act(async () => {
    const input = container.querySelector("input")!;
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, "delete my account");
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}
async function click(label: string) {
  await act(async () => [...container.querySelectorAll("button")].find(b => b.textContent === label)!.click());
}
function closureCalls() { return mocks.api.mock.calls.filter(call => call[0] === "/v1/account"); }

test("confirmed closure clears only the closed account and signs out locally", async () => {
  window.localStorage.setItem(`journalpulse_preferences_v1:${OWNER}`, "owner data");
  window.localStorage.setItem(`journalpulse_preferences_v1:${OTHER}`, "other data");
  await confirm(); await click("Close my account");
  expect(closureCalls()).toHaveLength(1);
  const options = closureCalls()[0][1];
  const body = JSON.parse(options.body);
  expect(body.confirmation).toBe("delete my account");
  expect(body.receipt_secret).toMatch(/^[a-f0-9]{64}$/);
  expect(body).not.toHaveProperty("user_id");
  expect(options.retry).toBe(false);
  expect(window.localStorage.getItem(`journalpulse_preferences_v1:${OWNER}`)).toBeNull();
  expect(window.localStorage.getItem(`journalpulse_preferences_v1:${OTHER}`)).toBe("other data");
  expect(mocks.signOut).toHaveBeenCalledWith({ scope: "local" });
  expect(browserAccount()).toBeNull();
  expect(mocks.replace).toHaveBeenCalledWith("/login");
});

test("failed closure keeps data and retries with the same action-only receipt", async () => {
  window.localStorage.setItem(`journalpulse_preferences_v1:${OWNER}`, "keep until confirmed");
  let attempts = 0;
  mocks.api.mockImplementation(async (path) => {
    if (path === "/v1/account" && attempts++ === 0) throw new ApiError("Closure could not be confirmed.", 503);
    return { account_closure: "configured" };
  });
  await confirm(); await click("Close my account");
  expect(mocks.signOut).not.toHaveBeenCalled();
  expect(window.localStorage.getItem(`journalpulse_preferences_v1:${OWNER}`)).toBe("keep until confirmed");
  const first = JSON.parse(closureCalls()[0][1].body);
  await click("Close my account");
  expect(JSON.parse(closureCalls()[1][1].body)).toEqual(first);
});

test("fresh sign-in email is sent only after the explicit button click", async () => {
  expect(mocks.signIn).not.toHaveBeenCalled();
  await click("Send a fresh sign-in link");
  expect(mocks.signIn).toHaveBeenCalledWith({ email: "fixture@example.test", options: {
    shouldCreateUser: false, emailRedirectTo: `${window.location.origin}/me/`,
  } });
  expect(closureCalls()).toHaveLength(0);
});

test("account switch during client initialization cannot start a late closure", async () => {
  let release!: (value: typeof client) => void;
  mocks.getClient.mockReturnValue(new Promise(resolve => { release = resolve; }));
  await confirm(); await click("Close my account");
  await act(async () => { activateBrowserAccount(OTHER); release(client); });
  expect(closureCalls()).toHaveLength(0);
  expect(mocks.signOut).not.toHaveBeenCalled();
});

test("a local sign-out error does not falsely say server deletion failed", async () => {
  mocks.signOut.mockResolvedValue({ error: new Error("Local sign-out failed") });
  await confirm(); await click("Close my account");
  expect(container.querySelector('[role="alert"]')?.textContent).toContain("Your account was closed");
  expect(container.querySelector('[role="alert"]')?.textContent).toContain("browser sign-out");
});
