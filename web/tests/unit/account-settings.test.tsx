import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const auth = vi.hoisted(() => ({ signOut: vi.fn(), replace: vi.fn() }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace: auth.replace }) }));
vi.mock("@/lib/supabase", () => ({ getSupabase: async () => ({ auth: { signOut: auth.signOut } }) }));
vi.mock("@/lib/api", () => ({ apiRequest: vi.fn().mockResolvedValue({ deleted_records: 3 }) }));

import MePage from "@/app/me/page";
import { readOpenConversationId, writeOpenConversationId } from "@/lib/conversation";
import { saveReminder } from "@/lib/reminders";

let root: Root;
let container: HTMLDivElement;
beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
  auth.replace.mockReset();
  auth.signOut.mockReset().mockResolvedValue({ error: null });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => { await act(async () => { root.unmount(); }); container.remove(); });
async function mount() { await act(async () => { root.render(createElement(MePage)); }); }
async function click(label: string) {
  await act(async () => { [...container.querySelectorAll("button")].find((button) => button.textContent?.trim() === label)!.click(); });
}

describe("account settings", () => {
  test("a failed sign-out stays on the page and reports that the user remains signed in", async () => {
    auth.signOut.mockResolvedValue({ error: new Error("auth service unavailable") });
    await mount();
    await click("Sign out");
    expect(auth.replace).not.toHaveBeenCalled();
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("still signed in");
  });

  test("deleting saved data also removes local resume and follow-up references", async () => {
    writeOpenConversationId("10000000-0000-4000-8000-000000000001");
    saveReminder({ decisionId: "deleted-decision", actionId: "walk", actionTitle: "A walk", dueAt: "2026-10-05T18:00:00Z" });
    await mount();
    await act(async () => {
      const input = container.querySelector('input[autocomplete="off"]')!;
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, "delete my journal");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await click("Delete my journal");
    expect(readOpenConversationId(window.localStorage)).toBeNull();
    expect(window.localStorage.getItem("journalpulse_reminders_v1")).toBe("[]");
  });
});
