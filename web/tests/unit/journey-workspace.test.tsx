import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

vi.mock("next/link", () => ({ default: ({ href, children, ...props }: { href: string; children: ReactNode }) => createElement("a", { href, ...props }, children) }));
vi.mock("@/lib/api", () => ({ apiRequest: vi.fn() }));
import JourneyPage from "@/app/journey/page";
import { apiRequest } from "@/lib/api";

let root: Root;
let container: HTMLDivElement;
beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  vi.spyOn(window, "confirm").mockReturnValue(true);
  const records = ["first", "second"].map((id) => ({
    id, created_at: "2026-10-05T10:00:00Z", decision: { decision_id: `${id}-decision`, action_id: "walk" },
    state: { valence: 0.2, emotion_tags: [] }, reflection: { summary: `${id} fictional summary` },
  }));
  vi.mocked(apiRequest).mockReset().mockImplementation(async (path, options) => {
    if (options?.method === "DELETE") return;
    if (path.startsWith("/v1/reflections")) return { items: records };
    if (path === "/v1/outcomes") return { items: records.map((record) => ({ decision_id: record.decision.decision_id, completed: true, helpfulness: 4 })) };
    return { items: [{ id: "walk", title: "A walk" }] };
  });
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});
afterEach(async () => { await act(async () => { root.unmount(); }); container.remove(); vi.restoreAllMocks(); });

test("deleting a reflection immediately removes its cascaded outcome from journey totals", async () => {
  await act(async () => { root.render(createElement(JourneyPage)); });
  const totals = () => [...container.querySelectorAll(".stat strong")].map((value) => value.textContent);
  expect(totals()).toEqual(["2", "2", "2"]);
  await act(async () => { container.querySelector<HTMLButtonElement>(".entry button")!.click(); });
  expect(container.textContent).not.toContain("first fictional summary");
  expect(container.textContent).toContain("second fictional summary");
  expect(totals()).toEqual(["1", "1", "1"]);
});
