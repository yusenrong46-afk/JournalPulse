import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const navigation = vi.hoisted(() => ({ query: "goal=settle&entry=private-entry&text=private-writing" }));
vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams(navigation.query) }));
vi.mock("next/link", () => ({ default: ({ href, children, ...props }: { href: string; children: ReactNode }) =>
  createElement("a", { href, ...props }, children) }));
vi.mock("@/lib/discovery", async (original) => ({
  ...await original<typeof import("@/lib/discovery")>(), searchDiscovery: vi.fn(),
}));
vi.mock("@/lib/api", async (original) => ({
  ...await original<typeof import("@/lib/api")>(), apiRequest: vi.fn(),
}));

import DiscoverPage from "@/app/discover/page";
import { searchDiscovery } from "@/lib/discovery";
import { apiRequest } from "@/lib/api";
import { clearTabSession } from "@/lib/tab-session";

let root: Root;
let container: HTMLDivElement;
const search = vi.mocked(searchDiscovery);

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
  window.sessionStorage.clear(); clearTabSession();
  vi.mocked(apiRequest).mockReset().mockImplementation(async (path) => path === "/v1/capabilities" ? { discovery: "configured" } : { items: [] });
  window.localStorage.setItem("journalpulse_open_conversation_v1", "private-conversation");
  navigation.query = "goal=settle&entry=private-entry&text=private-writing";
  search.mockReset();
  container = document.createElement("div"); document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); });

async function mount() { await act(async () => root.render(createElement(DiscoverPage))); }

describe("discovery connected to chat", () => {
  test("unavailable search is disclosed before consent and keeps the app collection usable", async () => {
    vi.mocked(apiRequest).mockImplementation(async (path) => path === "/v1/capabilities" ? { discovery: "unavailable" } : { items: [] });
    await mount();
    expect(container.querySelector("input[type=checkbox]")).toBeNull();
    expect(container.textContent).toContain("Web search isn’t available right now");
    expect(container.textContent).toContain("Choose a reviewed app activity");
    expect(search).not.toHaveBeenCalled();
  });

  test("restores the topic and exclusions after returning but requires new consent", async () => {
    await mount();
    await act(async () => {
      const input = container.querySelector<HTMLInputElement>("#discovery-topic")!;
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, "quiet reflection guides");
      input.dispatchEvent(new Event("input", { bubbles: true }));
      container.querySelector<HTMLInputElement>("input[type=checkbox]")!.click();
    });
    await act(async () => root.render(null)); await mount();
    expect(container.querySelector<HTMLInputElement>("#discovery-topic")!.value).toBe("quiet reflection guides");
    expect(container.querySelector<HTMLInputElement>("input[type=checkbox]")!.checked).toBe(false);
    expect(search).not.toHaveBeenCalled();
  });
  test("prefills only a general goal and requires fresh search consent", async () => {
    await mount();
    expect(container.querySelector<HTMLInputElement>("#discovery-topic")!.value)
      .toBe("short grounding exercises for everyday stress");
    expect(container.querySelector<HTMLInputElement>("input[type=checkbox]")!.checked).toBe(false);
    expect(container.querySelector<HTMLButtonElement>("button[type=submit]")!.disabled).toBe(true);
    expect(container.querySelector("a[href='/talk']")).toBeTruthy();
    expect(container.textContent).toContain("Luna may add a short focus from your feedback");
    expect(search).not.toHaveBeenCalled();
    expect(container.textContent).not.toContain("private-writing");
  });

  test("unknown URL values cannot become a search topic", async () => {
    navigation.query = "goal=private-writing&topic=private-writing";
    await mount();
    expect(container.querySelector<HTMLInputElement>("#discovery-topic")!.value).toBe("");
    expect(search).not.toHaveBeenCalled();
  });

  test("an approved edited topic sends no chat identity or journal content", async () => {
    search.mockRejectedValue(new Error("Provider unavailable"));
    await mount();
    const input = container.querySelector<HTMLInputElement>("#discovery-topic")!;
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, "short text grounding guides");
      input.dispatchEvent(new Event("input", { bubbles: true }));
      container.querySelector<HTMLInputElement>("input[type=checkbox]")!.click();
    });
    await act(async () => {
      container.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    });
    expect(search).toHaveBeenCalledTimes(1);
    expect(search.mock.calls[0][0]).toEqual({
      original_query: "short text grounding guides", excluded_urls: [], llm_consent: true, locale: "CA",
    });
    expect(JSON.stringify(search.mock.calls)).not.toContain("private-");
    expect(input.value).toBe("short text grounding guides");
    expect(container.textContent).toContain("Provider unavailable");
  });
});
