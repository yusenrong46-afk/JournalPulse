import { act, createElement, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const auth = vi.hoisted(() => ({
  user: "alice",
  listener: undefined as undefined | ((_event: string, session: { user: { id: string } } | null) => void),
  getSession: vi.fn(),
  replace: vi.fn(),
}));
vi.mock("next/navigation", () => {
  const router = { replace: auth.replace };
  return { usePathname: () => "/talk", useRouter: () => router };
});
vi.mock("@/lib/supabase", () => ({
  isSupabaseConfigured: () => true,
  getSupabase: async () => ({ auth: {
    getSession: auth.getSession,
    onAuthStateChange: (callback: typeof auth.listener) => {
      auth.listener = callback;
      return { data: { subscription: { unsubscribe: vi.fn() } } };
    },
  } }),
}));

import { AuthBoundary } from "@/components/auth-boundary";
import { readOpenConversationId, writeOpenConversationId } from "@/lib/conversation";
import { DEFAULT_PREFERENCES, savePreferences, usePreferences } from "@/lib/preferences";
import { saveReminder, useReminders } from "@/lib/reminders";

const CHAT_ID = "10000000-0000-4000-8000-000000000001";
let root: Root;
let container: HTMLDivElement;

function PrivateWorkspace() {
  const [privateText] = useState(() => `Private writing by ${auth.user}`);
  const [preferences] = usePreferences();
  const reminders = useReminders();
  return createElement("p", null, JSON.stringify({
    privateText, consent: preferences.llmConsent, reminders: reminders.length,
    resumed: readOpenConversationId(window.localStorage),
  }));
}

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
  auth.user = "alice";
  auth.listener = undefined;
  auth.getSession.mockResolvedValue({ data: { session: { user: { id: "alice" } } } });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
});
async function mount() {
  await act(async () => { root.render(createElement(AuthBoundary, null, createElement(PrivateWorkspace))); });
}
async function switchTo(user: string | null) {
  auth.user = user ?? "signed-out";
  await act(async () => { auth.listener!("SIGNED_IN", user ? { user: { id: user } } : null); });
}

describe("account isolation", () => {
  test("switching accounts on the same route discards the prior account's mounted private state", async () => {
    await mount();
    expect(container.textContent).toContain("Private writing by alice");
    await switchTo("bob");
    expect(container.textContent).toContain("Private writing by bob");
    expect(container.textContent).not.toContain("Private writing by alice");
  });

  test("consent, resumable chat and reminders belong to the account that saved them", async () => {
    await mount();
    await act(async () => {
      savePreferences({ ...DEFAULT_PREFERENCES, llmConsent: true });
      writeOpenConversationId(CHAT_ID);
      saveReminder({ decisionId: "alice-decision", actionId: "walk", actionTitle: "A walk", dueAt: "2026-10-05T17:00:00Z" });
    });
    expect(container.textContent).toContain('"consent":true');
    await switchTo("bob");
    expect(container.textContent).toContain('"consent":false');
    expect(container.textContent).toContain('"reminders":0');
    expect(container.textContent).toContain('"resumed":null');
    await switchTo("alice");
    expect(container.textContent).toContain('"consent":true');
    expect(container.textContent).toContain('"reminders":1');
    expect(container.textContent).toContain(CHAT_ID);
  });

  test("a late initial session lookup cannot overwrite a newer sign-out event", async () => {
    let resolve!: (value: unknown) => void;
    auth.getSession.mockReturnValue(new Promise((done) => { resolve = done; }));
    await mount();
    await switchTo(null);
    await act(async () => { resolve({ data: { session: { user: { id: "alice" } } } }); });
    expect(container.textContent).not.toContain("Private writing");
    expect(auth.replace).toHaveBeenCalledWith("/login?next=%2Ftalk");
  });
});
