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
    if (path.startsWith("/v1/activity-history")) return { items: [] };
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
  expect(totals()).toEqual(["2", "2", "2", "0"]);
  await act(async () => { container.querySelector<HTMLButtonElement>(".entry button")!.click(); });
  expect(container.textContent).not.toContain("first fictional summary");
  expect(container.textContent).toContain("second fictional summary");
  expect(totals()).toEqual(["1", "1", "1", "0"]);
});

test("reported chat activities join the garden without implying benefit", async () => {
  vi.mocked(apiRequest).mockImplementation(async (path) => {
    if (path.startsWith("/v1/activity-history")) {
      return { items: [
        { id: "a1", conversation_id: "c1", title: "Two-minute quiet meditation", kind: "meditation", goal: "settle",
          participation: "not_tried", fit: null, state_change: null, helpfulness: null, reported_at: "2026-10-05T11:00:00Z" },
        { id: "a2", conversation_id: "c1", title: "Short walk", kind: "movement", goal: "settle",
          participation: "partial", fit: "good", state_change: "same", helpfulness: null, reported_at: "2026-10-05T12:00:00Z" },
      ] };
    }
    if (path.startsWith("/v1/reflections")) return { items: [] };
    if (path === "/v1/outcomes") return { items: [] };
    return { items: [] };
  });
  await act(async () => { root.render(createElement(JourneyPage)); });
  // Chat activities alone are enough to show the journey instead of the empty state.
  expect(container.textContent).not.toContain("Your garden is ready to grow.");
  // The moon calendar marks the day of a report without implying it helped.
  const marks = [...container.querySelectorAll(".moon-grid [role=img]")].map((mark) => mark.getAttribute("aria-label") ?? "");
  expect(marks).toHaveLength(14);
  const reportedDay = new Date("2026-10-05T12:00:00Z").toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
  if (Date.now() - Date.parse("2026-10-05T12:00:00Z") < 13 * 86_400_000) {
    expect(marks.find((label) => label.startsWith(reportedDay))).toContain("activity check-in");
  }
  expect(marks.join(" ")).not.toMatch(/helped|improv/i);
  expect(container.textContent).toContain("Tried part of it · about the same");
  expect([...container.querySelectorAll(".stat strong")].map((item) => item.textContent)).toEqual(["2", "1", "0", "2"]); // Actual participation joins the totals; no benefit is inferred.
  expect(container.textContent).toContain("Finishing a timer is never counted as trying it.");
});

test("a failed activity history keeps legacy check-ins visible", async () => {
  vi.mocked(apiRequest).mockImplementation(async (path) => {
    if (path.startsWith("/v1/activity-history")) throw new Error("older API");
    if (path.startsWith("/v1/reflections")) {
      return { items: [{ id: "first", created_at: "2026-10-05T10:00:00Z", decision: { decision_id: "d", action_id: "walk" },
        state: { valence: 0, emotion_tags: [] }, reflection: { summary: "first fictional summary" } }] };
    }
    return { items: [] };
  });
  await act(async () => { root.render(createElement(JourneyPage)); });
  expect(container.textContent).toContain("first fictional summary");
  expect(container.textContent).toContain("Activities from your chats couldn’t load.");
  expect([...container.querySelectorAll(".stat strong")].map((item) => item.textContent)).toEqual(["—", "—", "—", "—"]);
});

test("chat ratings join the helpfulness breakdown without treating participation or unchanged mood as benefit", async () => {
  const original = vi.mocked(apiRequest).getMockImplementation()!;
  vi.mocked(apiRequest).mockImplementation(async (path, options) => {
    if (path.startsWith("/v1/activity-history")) return { items: [
      { id: "a1", conversation_id: "c1", title: "A quiet pause", kind: "meditation", goal: "settle",
        participation: "completed", state_change: "toward_target", helpfulness: 5, reported_at: "2026-10-05T11:00:00Z" },
      { id: "a2", conversation_id: "c1", title: "A quiet pause", kind: "meditation", goal: "settle",
        participation: "partial", state_change: "same", helpfulness: 1, reported_at: "2026-10-05T12:00:00Z" },
      { id: "a3", conversation_id: "c1", title: "A quiet pause", kind: "meditation", goal: "settle",
        participation: "not_tried", state_change: null, helpfulness: 5, reported_at: "2026-10-05T13:00:00Z" },
      { id: "a4", conversation_id: "c1", title: "A short walk", kind: "movement", goal: "settle",
        participation: "completed", state_change: "same", helpfulness: null, reported_at: "2026-10-05T14:00:00Z" },
    ] };
    return original(path, options);
  });
  await act(async () => { root.render(createElement(JourneyPage)); });
  const ratings = container.querySelector('[aria-labelledby="helped-heading"]')!;
  expect(ratings.textContent).toContain("A quiet pause3.0 / 5From chats · 2 ratings · 1 rated helpful");
  expect(ratings.textContent).toContain("A walk4.0 / 5From reflections · 2 ratings · 2 rated helpful");
  expect(ratings.textContent).not.toContain("A short walk");
  expect([...container.querySelectorAll(".stat strong")].map((item) => item.textContent)).toEqual(["6", "5", "3", "4"]);
});

test("search filters chat and reflection histories while preserving summary totals", async () => {
  const original = vi.mocked(apiRequest).getMockImplementation()!;
  vi.mocked(apiRequest).mockImplementation(async (path, options) => {
    if (path.startsWith("/v1/activity-history")) return { items: [
      { id: "a1", conversation_id: "c1", title: "A quiet pause", kind: "meditation", goal: "settle",
        participation: "partial", state_change: "same", helpfulness: null, reported_at: "2026-10-05T11:00:00Z" },
    ] };
    return original(path, options);
  });
  await act(async () => { root.render(createElement(JourneyPage)); });
  const totals = [...container.querySelectorAll(".stat strong")].map((item) => item.textContent);
  const search = container.querySelector<HTMLInputElement>("#journey-search")!;
  const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
  await act(async () => {
    setValue.call(search, "about the same");
    search.dispatchEvent(new Event("input", { bubbles: true }));
  });
  expect(container.querySelectorAll(".activity-history li")).toHaveLength(1);
  expect(container.querySelectorAll(".entry")).toHaveLength(0);
  expect(container.textContent).toContain("1 matching check-in");
  expect([...container.querySelectorAll(".stat strong")].map((item) => item.textContent)).toEqual(totals);
  await act(async () => {
    setValue.call(search, "first fictional");
    search.dispatchEvent(new Event("input", { bubbles: true }));
  });
  expect(container.querySelectorAll(".activity-history li")).toHaveLength(0);
  expect(container.querySelectorAll(".entry")).toHaveLength(1);
  expect(container.textContent).toContain("No activity check-ins match this search.");
});

test("a failed reflection endpoint preserves reported chat activities and leaves incomplete totals unavailable", async () => {
  vi.mocked(apiRequest).mockImplementation(async (path) => {
    if (path.startsWith("/v1/reflections")) throw new Error("offline");
    if (path.startsWith("/v1/activity-history")) return { items: [
      { id: "a1", conversation_id: "c1", title: "A quiet pause", kind: "meditation", goal: "settle",
        participation: "completed", state_change: "same", helpfulness: null, reported_at: "2026-10-05T11:00:00Z" },
    ] };
    return { items: [] };
  });
  await act(async () => { root.render(createElement(JourneyPage)); });
  expect(container.querySelector(".activity-history")?.textContent).toContain("A quiet pause");
  expect(container.textContent).toContain("Some of your journey couldn’t load.");
  expect([...container.querySelectorAll(".stat strong")].map((item) => item.textContent)).toEqual(["—", "—", "—", "1"]);
});

test("pending suggestions are not presented as tried and unrelated older outcomes do not inflate recent totals", async () => {
  const original = vi.mocked(apiRequest).getMockImplementation()!;
  vi.mocked(apiRequest).mockImplementation(async (path, options) => {
    if (path === "/v1/outcomes") return { items: [
      { decision_id: "first-decision", completed: true, helpfulness: 4 },
      { decision_id: "older-not-loaded-decision", completed: true, helpfulness: 5 },
    ] };
    return original(path, options);
  });
  await act(async () => { root.render(createElement(JourneyPage)); });
  expect(container.querySelectorAll(".entry")[1].textContent).toContain("Suggested: A walk");
  expect(container.querySelectorAll(".entry")[1].textContent).not.toContain("Tried:");
  expect([...container.querySelectorAll(".stat strong")].map((item) => item.textContent)).toEqual(["2", "1", "1", "0"]);
});
