import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const setSession = vi.fn();
const getSession = vi.fn();
let configured = false;
vi.mock("@/lib/supabase", () => ({
  isSupabaseConfigured: () => configured,
  getSupabase: vi.fn(async () => (configured ? { auth: { setSession, getSession } } : null)),
}));
vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams() }));

import LoginPage from "@/app/login/page";
import { DemoBanner } from "@/components/demo-banner";
import { DEMO_LABEL, DemoUnavailableError, isLocalDemoUser, LOCAL_DEMO_PREFIX, startDemo } from "@/lib/demo";

const LOCAL_ID = `${LOCAL_DEMO_PREFIX}1234-4abc-8def-0123456789ab`;
let root: Root;
let container: HTMLDivElement;

function reply(status: number, body: unknown) {
  return vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
}

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  configured = false;
  window.localStorage.clear();
  setSession.mockReset().mockResolvedValue({ error: null });
  getSession.mockReset().mockResolvedValue({ data: { session: null } });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
  vi.unstubAllEnvs();
});

test("only the local demo prefix marks a local identity as a demo", () => {
  expect(isLocalDemoUser(LOCAL_ID)).toBe(true);
  expect(isLocalDemoUser("00000000-0000-4000-8000-000000000001")).toBe(false);
  expect(isLocalDemoUser(null)).toBe(false);
});

test("a local demo session switches the preview identity and keeps AI off", async () => {
  const fetcher = reply(201, { user_id: LOCAL_ID, label: DEMO_LABEL, access_token: null, refresh_token: null });
  await expect(startDemo(fetcher as unknown as typeof fetch)).resolves.toBe(LOCAL_ID);
  expect(fetcher).toHaveBeenCalledWith(expect.stringMatching(/\/v1\/demo\/sessions$/), expect.objectContaining({ method: "POST" }));
  expect(window.localStorage.getItem("journalpulse-preview-client-id")).toBe(LOCAL_ID);
  const preferences = JSON.parse(window.localStorage.getItem("journalpulse_preferences_v1")!);
  expect(preferences).toMatchObject({ onboarded: true, llmConsent: false });
  expect(setSession).not.toHaveBeenCalled();
});

test("a hosted demo session signs in with the anonymous session it was given", async () => {
  configured = true;
  const fetcher = reply(201, { user_id: "8f6c1c1e-0000-4000-8000-000000000001", label: DEMO_LABEL, access_token: "a", refresh_token: "r" });
  await startDemo(fetcher as unknown as typeof fetch);
  expect(setSession).toHaveBeenCalledWith({ access_token: "a", refresh_token: "r" });
  const stored = window.localStorage.getItem("journalpulse_preferences_v1:8f6c1c1e-0000-4000-8000-000000000001");
  expect(JSON.parse(stored!).llmConsent).toBe(false);
});

test("a rate-limited demo request explains itself", async () => {
  await expect(startDemo(reply(429, { detail: "slow down" }) as unknown as typeof fetch))
    .rejects.toThrow(new DemoUnavailableError("Too many demo sessions from this network. Please try again later."));
  expect(window.localStorage.getItem("journalpulse-preview-client-id")).toBeNull();
});

test("the demo banner labels a demo identity and keeps the crisis line visible", async () => {
  window.localStorage.setItem("journalpulse-preview-client-id", LOCAL_ID);
  await act(async () => { root.render(createElement(DemoBanner)); });
  expect(container.textContent).toContain(DEMO_LABEL);
  expect(container.querySelector('a[href="tel:988"]')).not.toBeNull();
});

test("the demo banner stays hidden for an ordinary identity", async () => {
  window.localStorage.setItem("journalpulse-preview-client-id", "00000000-0000-4000-8000-000000000001");
  await act(async () => { root.render(createElement(DemoBanner)); });
  expect(container.textContent).toBe("");
});

test("a hosted anonymous session shows the demo banner", async () => {
  configured = true;
  getSession.mockResolvedValue({ data: { session: { user: { id: "x", is_anonymous: true } } } });
  await act(async () => { root.render(createElement(DemoBanner)); });
  expect(container.textContent).toContain(DEMO_LABEL);
});

test("sign-in offers the demo only when it is enabled, beside the crisis line", async () => {
  await act(async () => { root.render(createElement(LoginPage)); });
  expect(container.textContent).not.toContain("Try the demo");
  await act(async () => { root.unmount(); });
  vi.stubEnv("NEXT_PUBLIC_JOURNALPULSE_DEMO_ENABLED", "true");
  root = createRoot(container);
  await act(async () => { root.render(createElement(LoginPage)); });
  expect(container.textContent).toContain("Try the demo");
  expect(container.textContent).toContain(DEMO_LABEL);
  expect(container.textContent).toContain("9-8-8 Suicide Crisis Helpline");
});
