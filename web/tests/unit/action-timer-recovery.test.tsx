import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { ActionTimer } from "@/components/action-timer";
import { clearTabSession } from "@/lib/tab-session";

let root: Root; let container: HTMLDivElement;
beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  clearTabSession(); vi.useFakeTimers(); vi.setSystemTime(new Date("2026-10-07T12:00:00Z"));
  container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container);
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); vi.useRealTimers(); });
async function mount(id = "decision-a") { await act(async () => root.render(createElement(ActionTimer, { minutes: 7, sessionKey: id }))); }
async function click(label: string) { await act(async () => [...container.querySelectorAll("button")].find((button) => button.textContent === label)!.click()); }

test("a running timer continues through remount and a paused remainder survives", async () => {
  await mount(); await click("Start a 7-minute timer");
  await act(async () => vi.advanceTimersByTime(9000));
  await act(async () => root.render(null)); await act(async () => vi.advanceTimersByTime(5000)); await mount();
  expect(container.querySelector("[role=timer]")!.textContent).toBe("6:46");
  await click("Pause"); await act(async () => root.render(null)); await act(async () => vi.advanceTimersByTime(5000)); await mount();
  expect(container.querySelector("[role=timer]")!.textContent).toBe("6:46");
  await click("Reset"); await act(async () => root.render(null)); await mount();
  expect(container.querySelector("[role=timer]")!.textContent).toBe("7:00");
  expect(container.textContent).toContain("Start a 7-minute timer");
});

test("another decision never inherits the running timer", async () => {
  await mount(); await click("Start a 7-minute timer"); await act(async () => vi.advanceTimersByTime(9000));
  await act(async () => root.render(null)); await mount("decision-b");
  expect(container.querySelector("[role=timer]")!.textContent).toBe("7:00");
});
