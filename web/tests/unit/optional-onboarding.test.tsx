import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { beforeEach, afterEach, expect, test, vi } from "vitest";
const navigation = vi.hoisted(() => ({ replace: vi.fn(), query: "next=%2Fjournal%3Fentry%3D123" }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace: navigation.replace }), useSearchParams: () => new URLSearchParams(navigation.query) }));
vi.mock("next/link", () => ({ default: ({ href, children, ...props }: { href: string; children: ReactNode }) => createElement("a", { href, ...props }, children) }));
vi.mock("@/lib/api", async (original) => ({ ...await original<typeof import("@/lib/api")>(), apiRequest: vi.fn(async () => ({ items: [] })) }));
import HomePage from "@/app/page";
import WelcomePage from "@/app/welcome/page";
import { DEFAULT_PREFERENCES, savePreferences } from "@/lib/preferences";
let root: Root; let container: HTMLDivElement;
beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear(); savePreferences(DEFAULT_PREFERENCES); navigation.replace.mockClear();
  container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container);
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); });
async function click(label: string) { await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent?.includes(label))!.click()); }
test("incomplete setup does not redirect Home or grant consent", async () => {
  await act(async () => root.render(createElement(HomePage)));
  expect(navigation.replace).not.toHaveBeenCalled();
  expect(container.textContent).toContain("private defaults");
  expect(JSON.parse(window.localStorage.getItem("journalpulse_preferences_v1")!).llmConsent).toBe(false);
});
test.each([ ["next=%2Fjournal%3Fentry%3D123", "/journal?entry=123"], ["next=https%3A%2F%2Fevil.test", "/talk"], ["next=%2F%2Fevil.test", "/talk"] ])("setup returns safely with query %s", async (query, expected) => {
  navigation.query = query; await act(async () => root.render(createElement(WelcomePage)));
  await click("Nice to meet you"); expect(container.textContent).toContain("reported feelings and activity choices remain saved");
  await click("Simple Luna"); await click("Continue"); await click("Let’s begin");
  expect(navigation.replace).toHaveBeenCalledWith(expected);
});
