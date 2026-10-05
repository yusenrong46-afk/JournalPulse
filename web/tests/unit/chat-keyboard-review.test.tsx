import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const navigation = vi.hoisted(() => ({
  router: { replace: vi.fn() },
  params: new URLSearchParams(),
}));
vi.mock("next/navigation", () => ({
  useRouter: () => navigation.router,
  useSearchParams: () => navigation.params,
}));
vi.mock("next/link", () => ({
  default: ({ href, children, ...props }: { href: string; children: ReactNode }) =>
    createElement("a", { href, ...props }, children),
}));
vi.mock("@/lib/api", async (original) => ({
  ...await original<typeof import("@/lib/api")>(),
  apiRequest: vi.fn(async () => ({ analysis_mode: "ai_configured" })),
}));

import TalkPage from "@/app/talk/page";
import { activateBrowserAccount } from "@/lib/account-storage";
import { apiRequest } from "@/lib/api";

let root: Root;
let container: HTMLDivElement;

beforeEach(async () => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  activateBrowserAccount("keyboard-review");
  window.localStorage.clear();
  navigation.params = new URLSearchParams();
  vi.mocked(apiRequest).mockImplementation(async () => ({ analysis_mode: "ai_configured" }));
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => { root.render(createElement(TalkPage)); });
});
afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
});

async function openPrivacy(): Promise<HTMLButtonElement> {
  const trigger = container.querySelector<HTMLButtonElement>(".privacy-pill")!;
  expect(trigger).not.toBeNull();
  trigger.focus();
  await act(async () => { trigger.click(); });
  return trigger;
}

describe("chat privacy keyboard behavior", () => {
  test("opening privacy moves keyboard focus into its modal", async () => {
    await openPrivacy();
    const dialog = container.querySelector('[role="dialog"]')!;
    expect(dialog.contains(document.activeElement)).toBe(true);
  });

  test("Escape dismisses privacy and restores focus to its opener", async () => {
    const trigger = await openPrivacy();
    await act(async () => {
      document.activeElement!.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(container.querySelector('[role="dialog"]')).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  test("Tab from the final modal control wraps to its first control", async () => {
    await openPrivacy();
    const dialog = container.querySelector('[role="dialog"]')!;
    const first = dialog.querySelector<HTMLInputElement>("input")!;
    const last = dialog.querySelector<HTMLButtonElement>("button")!;
    last.focus();
    await act(async () => {
      last.dispatchEvent(new KeyboardEvent("keydown", { key: "Tab", bubbles: true, cancelable: true }));
    });
    expect(document.activeElement).toBe(first);
  });

  test("Shift+Tab from the first modal control wraps to its final control", async () => {
    await openPrivacy();
    const dialog = container.querySelector('[role="dialog"]')!;
    const first = dialog.querySelector<HTMLInputElement>("input")!;
    const last = dialog.querySelector<HTMLButtonElement>("button")!;
    first.focus();
    await act(async () => {
      first.dispatchEvent(new KeyboardEvent("keydown", {
        key: "Tab", shiftKey: true, bubbles: true, cancelable: true,
      }));
    });
    expect(document.activeElement).toBe(last);
  });
});

describe("chat options keyboard behavior", () => {
  async function openMenu(): Promise<HTMLButtonElement> {
    await act(async () => { root.unmount(); });
    navigation.params = new URLSearchParams("c=10000000-0000-4000-8000-000000000099");
    vi.mocked(apiRequest).mockImplementation(async (path) => path === "/v1/system/status"
      ? { analysis_mode: "ai_configured" }
      : { conversation: {
        id: "10000000-0000-4000-8000-000000000099", user_id: "keyboard-review",
        created_at: "2026-10-05T12:00:00Z", updated_at: "2026-10-05T12:00:00Z",
        mode: "guided", status: "open", llm_consent: false, retain_text: false,
        safety_mode: "normal", locale: "CA", prompt_version: "mock", revision: 0,
      }, messages: [] });
    root = createRoot(container);
    await act(async () => { root.render(createElement(TalkPage)); });
    const trigger = container.querySelector<HTMLButtonElement>('[aria-label="Chat options"]')!;
    expect(trigger).not.toBeNull();
    trigger.focus();
    await act(async () => { trigger.click(); });
    return trigger;
  }

  test("opening options moves focus into its menu", async () => {
    await openMenu();
    expect(container.querySelector('[role="menu"]')!.contains(document.activeElement)).toBe(true);
  });

  test("Escape closes options and returns focus to its opener", async () => {
    const trigger = await openMenu();
    await act(async () => {
      document.activeElement!.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(container.querySelector('[role="menu"]')).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  test("Tab leaves options for the next workspace control and closes the menu", async () => {
    await openMenu();
    await act(async () => {
      document.activeElement!.dispatchEvent(new KeyboardEvent("keydown", {
        key: "Tab", bubbles: true, cancelable: true,
      }));
    });
    expect(container.querySelector('[role="menu"]')).toBeNull();
    expect(document.activeElement?.textContent).toBe("Just talk");
  });
});
