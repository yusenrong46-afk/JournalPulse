import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const state = vi.hoisted(() => ({ path: "/privacy", replace: vi.fn(), getSupabase: vi.fn() }));
vi.mock("next/navigation", () => {
  const router = { replace: state.replace };
  return { usePathname: () => state.path, useRouter: () => router };
});
vi.mock("@/lib/supabase", () => ({ isSupabaseConfigured: () => true, getSupabase: state.getSupabase }));
import { AuthBoundary } from "@/components/auth-boundary";

let root: Root;
let container: HTMLDivElement;
beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  state.getSupabase.mockReset().mockRejectedValue(new Error("Auth service unavailable"));
  state.replace.mockClear();
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); });
async function render(path: string) {
  state.path = path;
  await act(async () => root.render(createElement(AuthBoundary, null, createElement("p", null, "Page content"))));
}

test.each(["/privacy", "/privacy/", "/terms", "/how-ai-works"])("%s does not initialize or wait for authentication", async (path) => {
  await render(path);
  expect(container.textContent).toBe("Page content");
  expect(state.getSupabase).not.toHaveBeenCalled();
  expect(state.replace).not.toHaveBeenCalled();
});
test("sign-in and its disclosures remain visible during an auth outage", async () => {
  await render("/login");
  expect(container.textContent).toBe("Page content");
});
test("private writing stays hidden when authentication fails", async () => {
  await render("/journal");
  expect(container.textContent).not.toContain("Page content");
  expect(container.querySelector("[role=alert]")).not.toBeNull();
});
test("moving from public information to private writing still requires sign-in", async () => {
  state.getSupabase.mockResolvedValue({ auth: {
    getSession: async () => ({ data: { session: null } }),
    onAuthStateChange: () => ({ data: { subscription: { unsubscribe: vi.fn() } } }),
  } });
  await render("/privacy");
  await render("/journal");
  expect(container.textContent).not.toContain("Page content");
  expect(state.replace).toHaveBeenCalledWith("/login?next=%2Fjournal");
});

test("returning to a formerly authorized page does not reuse its old session", async () => {
  state.getSupabase.mockResolvedValue({ auth: {
    getSession: async () => ({ data: { session: { user: { id: "alice" } } } }),
    onAuthStateChange: () => ({ data: { subscription: { unsubscribe: vi.fn() } } }),
  } });
  await render("/journal");
  expect(container.textContent).toBe("Page content");
  await render("/privacy");
  // A session expired or was signed out elsewhere while viewing public content.
  state.getSupabase.mockReturnValue(new Promise(() => {}));
  await render("/journal");
  expect(container.textContent).not.toContain("Page content");
});
