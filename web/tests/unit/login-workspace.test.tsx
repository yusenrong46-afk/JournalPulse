import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

vi.mock("next/navigation", () => ({ useSearchParams: () => new URLSearchParams("next=%2Fjournal") }));
vi.mock("@/lib/supabase", () => ({ getSupabase: vi.fn() }));

import LoginPage from "@/app/login/page";
import { getSupabase } from "@/lib/supabase";

const signIn = vi.fn();
const client = { auth: { signInWithOtp: signIn } } as unknown as NonNullable<Awaited<ReturnType<typeof getSupabase>>>;
let root: Root;
let container: HTMLDivElement;

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => { resolve = done; });
  return { promise, resolve };
}

beforeEach(async () => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  vi.mocked(getSupabase).mockReset().mockResolvedValue(client);
  signIn.mockReset().mockResolvedValue({ error: null });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => { root.render(createElement(LoginPage)); });
  const input = container.querySelector("input")!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, "reader@example.com");
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
});

async function submit() {
  await act(async () => { container.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); });
}

test("failed sign-in initialization shows an error and restores the form", async () => {
  vi.mocked(getSupabase).mockRejectedValueOnce(new Error("SDK failed to initialize"));
  await submit();
  expect(container.querySelector("[role=alert]")?.textContent).toContain("Check your connection and try again");
  expect(container.querySelector("button")!.disabled).toBe(false);
  expect(container.querySelector("input")!.value).toBe("reader@example.com");
  expect(signIn).not.toHaveBeenCalled();
});

test("a rejected sign-in request restores controls and can be retried", async () => {
  signIn.mockRejectedValueOnce(new Error("Network failed"));
  await submit();
  expect(container.querySelector("button")!.disabled).toBe(false);
  expect(container.querySelector("input")!.disabled).toBe(false);
  await submit();
  expect(signIn).toHaveBeenCalledTimes(2);
  expect(container.textContent).toContain("Check your email");
  expect(container.querySelector("[role=alert]")).toBeNull();
});

test("repeated submits during SDK initialization send only one sign-in email", async () => {
  const waiting = deferred<typeof client>();
  vi.mocked(getSupabase).mockReturnValueOnce(waiting.promise);
  await submit();
  expect(container.querySelector("button")!.disabled).toBe(true);
  await submit();
  expect(getSupabase).toHaveBeenCalledTimes(1);
  await act(async () => { waiting.resolve(client); });
  expect(signIn).toHaveBeenCalledTimes(1);
  expect(signIn.mock.calls[0][0].email).toBe("reader@example.com");
});

test("leaving the page during initialization does not send a late email", async () => {
  const waiting = deferred<typeof client>();
  vi.mocked(getSupabase).mockReturnValueOnce(waiting.promise);
  await submit();
  await act(async () => { root.render(createElement("p", null, "Another page")); });
  await act(async () => { waiting.resolve(client); });
  expect(signIn).not.toHaveBeenCalled();
});
