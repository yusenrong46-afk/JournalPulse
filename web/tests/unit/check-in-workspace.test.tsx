import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const navigation = vi.hoisted(() => ({ query: "decision=first", listeners: new Set<() => void>() }));
vi.mock("next/navigation", async () => {
  const { useMemo, useSyncExternalStore } = await import("react");
  return { useSearchParams: () => {
    const query = useSyncExternalStore((callback) => {
      navigation.listeners.add(callback);
      return () => { navigation.listeners.delete(callback); };
    }, () => navigation.query);
    return useMemo(() => new URLSearchParams(query), [query]);
  } };
});
vi.mock("next/link", () => ({ default: ({ href, children, ...props }: { href: string; children: ReactNode }) => createElement("a", { href, ...props }, children) }));
vi.mock("@/lib/api", async (original) => ({ ...await original<typeof import("@/lib/api")>(), apiRequest: vi.fn() }));

import CheckInPage from "@/app/check-in/page";
import { ApiError, apiRequest } from "@/lib/api";

const history = { items: ["first", "second"].map((id) => ({
  created_at: "2026-10-05T10:00:00Z", decision: { decision_id: id, action_id: id },
})) };
const catalog = { items: ["first", "second"].map((id) => ({ id, title: `${id} activity` })) };
const requested = vi.mocked(apiRequest);
let root: Root;
let container: HTMLDivElement;
beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  window.localStorage.clear();
  navigation.query = "decision=first";
  requested.mockReset();
  requested.mockImplementation(async (path) => path.startsWith("/v1/reflections") ? history : path === "/v1/resources" ? catalog : { items: [] });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
  navigation.listeners.clear();
});
async function mount() { await act(async () => { root.render(createElement(CheckInPage)); }); }
async function navigate(query: string) { await act(async () => { navigation.query = query; navigation.listeners.forEach((callback) => callback()); }); }
async function click(label: string) {
  await act(async () => { [...container.querySelectorAll("button")].find((button) => button.textContent?.trim() === label)!.click(); });
}

describe("check-in identity", () => {
  test("an old load cannot replace the check-in selected by a newer URL", async () => {
    let finish!: (value: typeof history) => void;
    requested.mockImplementationOnce(() => new Promise((resolve) => { finish = resolve; }));
    await mount();
    await navigate("decision=second");
    expect(container.textContent).toContain("second activity");
    await act(async () => { finish(history); });
    expect(container.textContent).toContain("second activity");
    expect(container.textContent).not.toContain("first activity");
  });

  test("changing the decision resets completed UI and creates a distinct receipt", async () => {
    await mount();
    await click("Not yet");
    await click("Skip this one");
    expect(container.textContent).toContain("Thank you!");
    await navigate("decision=second");
    expect(container.textContent).toContain("second activity");
    expect(container.textContent).not.toContain("Thank you!");
    await click("Not yet");
    await click("Skip this one");
    const receipts = requested.mock.calls.filter(([path, options]) => path === "/v1/outcomes" && options?.method === "POST")
      .map(([, options]) => JSON.parse(String(options!.body)));
    expect(receipts).toHaveLength(2);
    expect(receipts[0].decision_id).toBe("first");
    expect(receipts[1].decision_id).toBe("second");
    expect(receipts[0].client_request_id).not.toBe(receipts[1].client_request_id);
  });

  test("an ambiguous save keeps answers fixed and retries the complete original receipt", async () => {
    const writes: string[] = [];
    requested.mockImplementation(async (path, options) => {
      if (path === "/v1/outcomes" && options?.method === "POST") {
        writes.push(String(options.body));
        if (writes.length === 1) throw new Error("Lost response after commit");
        return JSON.parse(writes[0]);
      }
      return path.startsWith("/v1/reflections") ? history : path === "/v1/resources" ? catalog : { items: [] };
    });
    await mount();
    await click("Yes, I did");
    const face = (label: string) => [...container.querySelectorAll<HTMLButtonElement>(".face")]
      .find((button) => button.textContent?.includes(label))!;
    await act(async () => { face("Not at all").click(); });
    await click("Save my check-in");
    expect(container.textContent).toContain("submitted answers are kept unchanged");
    expect(face("A lot").disabled).toBe(true);
    expect(container.querySelector("textarea")!.disabled).toBe(true);
    expect([...container.querySelectorAll<HTMLButtonElement>(".chips button")].every((button) => button.disabled)).toBe(true);
    await act(async () => { face("A lot").click(); });
    expect(face("Not at all").getAttribute("aria-pressed")).toBe("true");
    await click("Retry saving check-in");
    expect(writes).toHaveLength(2);
    expect(writes[1]).toBe(writes[0]);
    expect(JSON.parse(writes[1]).helpfulness).toBe(1);
    expect(container.textContent).toContain("Thank you!");
  });

  test("a validation rejection keeps the answers editable and creates a new corrected receipt", async () => {
    const writes: string[] = [];
    requested.mockImplementation(async (path, options) => {
      if (path === "/v1/outcomes" && options?.method === "POST") {
        writes.push(String(options.body));
        if (writes.length === 1) throw new ApiError("Check your answers", 422);
        return JSON.parse(writes[1]);
      }
      return path.startsWith("/v1/reflections") ? history : path === "/v1/resources" ? catalog : { items: [] };
    });
    await mount(); await click("Not yet"); await click("Skip this one");
    expect(container.textContent).not.toContain("submitted answers are kept unchanged");
    await click("Skip this one");
    expect(JSON.parse(writes[0]).client_request_id).not.toBe(JSON.parse(writes[1]).client_request_id);
  });
});
