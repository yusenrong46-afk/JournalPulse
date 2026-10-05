import { act, createElement, StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

vi.mock("@/lib/api", async (original) => ({
  ...await original<typeof import("@/lib/api")>(), apiRequest: vi.fn(),
}));

import { ActionTimer } from "@/components/action-timer";
import { ActivitySessionPanel } from "@/components/activity-session-panel";
import { ActivitySessionWorkspace } from "@/components/activity-session-workspace";
import { ApiError, apiRequest } from "@/lib/api";
import { activityClock, activitySecondsLeft, canApplyActivity, type ActivitySession } from "@/lib/activity-session";
import type { Conversation } from "@/lib/types";

const CHAT = "10000000-0000-4000-8000-000000000001";
const SESSION = "20000000-0000-4000-8000-000000000001";
const SERVER = Date.parse("2026-10-05T12:00:00Z");
const chat: Conversation = {
  id: CHAT, user_id: "owner", created_at: new Date(SERVER).toISOString(), updated_at: new Date(SERVER).toISOString(),
  status: "open", mode: "ai", llm_consent: true, retain_text: false, safety_mode: "normal", locale: "CA",
  prompt_version: "test", revision: 2,
  card: {
    resource_intent: "ground", card_reason: "A quiet pause fits the two minutes you have.",
    goal: "settle", actions: [{ id: "guided_meditation_2m", title: "Two-minute quiet meditation", url: "",
      summary: "A quiet pause.", provider: "JournalPulse", resource_type: "activity", coping_style: "ground", duration_minutes: 2 }],
    decision_preview: { decision_id: "test", action_id: "guided_meditation_2m", propensity: 1, policy_name: "mock",
      policy_version: "test", safe_action_ids: ["guided_meditation_2m"], explanation: "fits", selection_source: "policy", eligible_for_ope: false },
  },
};

function session(status: ActivitySession["status"] = "offered", revision = 0): ActivitySession {
  return {
    id: SESSION, user_id: "owner", conversation_id: CHAT, source_entry_id: null, revision, status,
    resource: { id: "guided_meditation_2m", title: "Two-minute quiet meditation", url: null, provider: "JournalPulse",
      resource_type: "activity", format: "timer", kind: "meditation", duration_seconds: 120,
      instructions: ["Sit comfortably.", "Notice where you are, without changing your breathing."], provenance: "builtin" },
    recommendation_reason: "A quiet pause fits the two minutes you have.",
    selection: { selection_source: "llm", recommended_resource_id: "guided_meditation_2m", selected_resource_id: "guided_meditation_2m", eligible_for_ope: false, propensity: null },
    goal: "settle", duration_seconds: 120, remaining_seconds: 120,
    expires_at: status === "active" ? new Date(SERVER + 120_000).toISOString() : null,
    created_at: new Date(SERVER).toISOString(), updated_at: new Date(SERVER).toISOString(),
    started_at: status === "offered" ? null : new Date(SERVER).toISOString(), check_in_issued: false,
    report: null, reported_at: null, follow_up_status: "none", follow_up_reply: null,
    follow_up_attempts: 0, final_follow_up: false, server_now: new Date(SERVER).toISOString(), conversation_revision: 2,
  };
}

let root: Root;
let container: HTMLDivElement;
const requested = vi.mocked(apiRequest);

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  vi.useFakeTimers({ toFake: ["setInterval", "clearInterval", "setTimeout", "clearTimeout", "performance"] });
  requested.mockReset();
  container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container);
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); vi.useRealTimers(); vi.restoreAllMocks(); });

function button(label: string): HTMLButtonElement {
  const found = [...container.querySelectorAll("button")].find((item) => item.textContent?.trim() === label);
  expect(found, label).toBeTruthy(); return found!;
}
async function click(label: string) { await act(async () => button(label).click()); }

describe("authoritative activity clock", () => {
  test("server deadline ignores local wall-clock changes and interval count", () => {
    const active = session("active", 1); const anchor = activityClock(active, 1000);
    vi.setSystemTime(new Date("2040-01-01"));
    expect(activitySecondsLeft(active, anchor, 46_000)).toBe(75);
    expect(activitySecondsLeft(active, anchor, 121_000)).toBe(0);
  });
  test("paused duration survives a long interruption and old tab state cannot rewind a session", () => {
    const paused = { ...session("paused", 3), remaining_seconds: 71 };
    expect(activitySecondsLeft(paused, activityClock(paused, 0), 500_000)).toBe(71);
    expect(canApplyActivity(paused, session("active", 2), CHAT)).toBe(false);
    expect(canApplyActivity(paused, { ...paused, conversation_id: "other" }, CHAT)).toBe(false);
  });
});

describe("legacy timer", () => {
  test("StrictMode completion calls the hook once and does not infer participation", async () => {
    const done = vi.fn();
    await act(async () => root.render(createElement(StrictMode, null, createElement(ActionTimer, { minutes: 1, onDone: done }))));
    await click("Start a 1-minute timer");
    await act(async () => vi.advanceTimersByTime(60_000));
    expect(done).toHaveBeenCalledOnce();
    await act(async () => vi.advanceTimersByTime(10_000));
    expect(done).toHaveBeenCalledOnce(); expect(container.textContent).not.toContain("Nice work");
  });
  test("pause time does not consume the remaining activity duration", async () => {
    await act(async () => root.render(createElement(ActionTimer, { minutes: 1 })));
    await click("Start a 1-minute timer"); await act(async () => vi.advanceTimersByTime(10_000)); await click("Pause");
    await act(async () => vi.advanceTimersByTime(40_000)); expect(container.querySelector('[role="timer"]')?.textContent).toBe("0:50");
    await click("Keep going"); await act(async () => vi.advanceTimersByTime(15_000));
    expect(container.querySelector('[role="timer"]')?.textContent).toBe("0:35");
  });
});

describe("inline activity lifecycle", () => {
  let stored: ActivitySession | null;
  let followupFails: boolean;
  const refresh = vi.fn(async () => undefined);
  beforeEach(() => {
    stored = null; followupFails = false; refresh.mockClear();
    requested.mockImplementation(async (path, init) => {
      if (path === `/v1/conversations/${CHAT}/activity-sessions` && !init?.method) return stored
        ? { ...stored, server_now: new Date(SERVER + performance.now()).toISOString() } : null;
      if (path === `/v1/conversations/${CHAT}/activity-sessions` && init?.method === "POST") { stored = session(); return stored; }
      const body = init?.body ? JSON.parse(String(init.body)) : null;
      if (path === `/v1/activity-sessions/${SESSION}/commands`) {
        const value = stored!;
        const status = body.command === "start" || body.command === "resume" ? "active" : body.command === "pause" ? "paused" :
          body.command === "stop" ? "stopped" : body.command === "decline" ? "declined" : "awaiting_report";
        stored = { ...value, status, revision: value.revision + 1,
          started_at: body.command === "start" ? new Date(SERVER).toISOString() : value.started_at,
          expires_at: status === "active" ? new Date(SERVER + performance.now() + value.remaining_seconds * 1000).toISOString() : null,
          server_now: new Date(SERVER + performance.now()).toISOString(), check_in_issued: status === "awaiting_report" };
        return stored;
      }
      if (path === `/v1/activity-sessions/${SESSION}/report`) {
        stored = { ...stored!, status: "completed", revision: stored!.revision + 1,
          report: { participation: body.participation, state_change: body.state_change ?? null }, follow_up_status: "pending" };
        return stored;
      }
      if (path === `/v1/activity-sessions/${SESSION}/follow-up`) {
        if (followupFails) throw new ApiError("Temporary follow-up outage", 503);
        stored = { ...stored!, revision: stored!.revision + 1, follow_up_status: "ready", follow_up_reply: "Thanks for telling me you did not try it.",
          follow_up_message_id: "assistant-outcome" }; return stored;
      }
      throw new Error(`Unexpected ${path}`);
    });
  });
  async function mount() { await act(async () => root.render(createElement(ActivitySessionWorkspace, {
    conversation: chat, disabled: false, onRefresh: refresh, onBusyChange: vi.fn(),
  }))); }

  test("starting creates a separate activity without accepting or clearing the chat", async () => {
    await mount(); await click("Start activity");
    expect(stored?.status).toBe("active"); expect(button("Pause")).toBeTruthy();
    expect(requested.mock.calls.some(([path]) => path.endsWith("/accept"))).toBe(false);
    expect(refresh).not.toHaveBeenCalled();
  });

  test("an additive activity card starts in chat while the legacy card stays empty", async () => {
    const compatible = { ...chat, card: null, activity_card: chat.card } as Conversation;
    await act(async () => root.render(createElement(ActivitySessionWorkspace, {
      conversation: compatible, disabled: false, onRefresh: refresh, onBusyChange: vi.fn(),
    })));
    await click("Start activity");
    expect(stored?.resource.id).toBe("guided_meditation_2m");
    expect(stored?.status).toBe("active");
    expect(compatible.card).toBeNull();
    expect(requested.mock.calls.some(([path]) => path.endsWith("/accept"))).toBe(false);
  });

  test("expiry renders one participation check and no paid generation before a report", async () => {
    await mount(); await click("Start activity");
    await act(async () => vi.advanceTimersByTime(120_000));
    expect(container.textContent?.match(/Did you try it\?/g)).toHaveLength(1);
    expect(stored?.report).toBeNull();
    const expiry = requested.mock.calls.filter(([path, init]) => path.endsWith("/commands") && JSON.parse(String(init?.body)).command === "expire");
    expect(expiry.length).toBeLessThanOrEqual(1);
    expect(requested.mock.calls.some(([path]) => path.endsWith("/follow-up"))).toBe(false);
  });

  test("a saved not-tried report survives a follow-up outage and retry does not submit it twice", async () => {
    followupFails = true;
    await mount(); await click("Start activity"); await click("Finish early");
    const choice = container.querySelector<HTMLInputElement>('input[value="not_tried"]')!;
    await act(async () => choice.click()); await click("Save check-in");
    expect(stored?.report?.participation).toBe("not_tried"); expect(container.textContent).toContain("Check-in saved");
    const failed = requested.mock.calls.find(([path]) => path.endsWith("/follow-up"))!;
    followupFails = false; await click("Retry Luna’s follow-up");
    expect(requested.mock.calls.filter(([path]) => path.endsWith("/report"))).toHaveLength(1);
    const retries = requested.mock.calls.filter(([path]) => path.endsWith("/follow-up"));
    expect(retries[1][1]?.body).toBe(failed[1]?.body); expect(refresh).toHaveBeenCalledOnce();
  });

  test("a follow-up completed by another tab after 202 appears after one canonical refresh", async () => {
    stored = { ...session("completed", 3), report: { participation: "partial" }, follow_up_status: "pending" };
    const fallback = requested.getMockImplementation()!;
    requested.mockImplementation(async (path, init) => {
      if (path.endsWith("/follow-up")) { stored = { ...stored!, revision: 4, follow_up_status: "generating" }; return stored; }
      return fallback(path, init);
    });
    await mount(); await click("Retry Luna’s follow-up");
    expect(refresh).not.toHaveBeenCalled();
    stored = { ...stored!, revision: 5, follow_up_status: "ready", follow_up_message_id: "other-tab-message" };
    await act(async () => window.dispatchEvent(new Event("focus")));
    await act(async () => window.dispatchEvent(new Event("focus")));
    expect(refresh).toHaveBeenCalledOnce();
    expect(requested.mock.calls.filter(([path]) => path.endsWith("/follow-up"))).toHaveLength(1);
  });

  test("a fresh same-resource offer appears after a completed session, but the consumed offer does not", async () => {
    stored = { ...session("completed", 4), offered_message_id: "old-offer", report: { participation: "completed" }, follow_up_status: "ready" };
    await act(async () => root.render(createElement(ActivitySessionWorkspace, { conversation: {
      ...chat, card: { ...chat.card!, offered_message_id: "old-offer" },
    }, disabled: false, onRefresh: refresh, onBusyChange: vi.fn() })));
    expect([...container.querySelectorAll("button")].some((item) => item.textContent === "Start activity")).toBe(false);
    await act(async () => root.render(createElement(ActivitySessionWorkspace, { conversation: {
      ...chat, revision: 3, card: { ...chat.card!, offered_message_id: "new-offer" },
    }, disabled: false, onRefresh: refresh, onBusyChange: vi.fn() })));
    expect(button("Start activity")).toBeTruthy();
  });

  test("started-session controls and check-in remain usable at the ordinary message limit", async () => {
    stored = session("paused", 2);
    await act(async () => root.render(createElement(ActivitySessionWorkspace, {
      conversation: chat, ordinaryMessages: 20, disabled: false, onRefresh: refresh, onBusyChange: vi.fn(),
    })));
    expect(button("Finish early").disabled).toBe(false); await click("Finish early");
    expect(container.querySelector<HTMLInputElement>('input[value="not_tried"]')?.disabled).toBe(false);
    // A started activity has nothing to replace, so the search entry is not offered at all.
    expect([...container.querySelectorAll("button")].some((item) => item.textContent?.trim() === "Find another resource")).toBe(false);
  });

  test("a stop conversation suppresses a stale expired timer and later cannot restore its old question", async () => {
    stored = { ...session("active", 2), expires_at: new Date(SERVER).toISOString(), remaining_seconds: 0 };
    const paused = { ...chat, activity_move: "pause" } as Conversation;
    await act(async () => root.render(createElement(ActivitySessionWorkspace, {
      conversation: paused, disabled: false, onRefresh: refresh, onBusyChange: vi.fn(),
    })));
    await act(async () => vi.advanceTimersByTime(1000));
    expect(container.textContent).not.toContain("Did you try it?");
    expect(requested.mock.calls.some(([path]) => path.endsWith("/commands") || path.endsWith("/follow-up"))).toBe(false);
    stored = { ...stored, status: "stopped", revision: 3, check_in_issued: false, expires_at: null };
    await act(async () => root.render(createElement(ActivitySessionWorkspace, {
      conversation: { ...chat, revision: 3 }, disabled: false, onRefresh: refresh, onBusyChange: vi.fn(),
    })));
    expect(container.textContent).not.toContain("Did you try it?");
    expect(container.textContent).toContain("Activity stopped");
  });
});

test("an invalidated unstarted offer cannot invite a participant report", async () => {
  const stopped = { ...session("stopped"), started_at: null };
  await act(async () => root.render(createElement(ActivitySessionPanel, { session: stopped, secondsLeft: 120,
    busy: false, expiryPending: false, onCommand: vi.fn(), onReport: vi.fn(), onFollowUp: vi.fn() })));
  expect(container.textContent).not.toContain("Did you try it?"); expect(container.textContent).toContain("No activity started");
});

test("a conversation stop is silent while explicit activity Stop can offer an honest report", async () => {
  const stopped = { ...session("stopped", 3), check_in_issued: false };
  const props = { session: stopped, secondsLeft: 0, busy: false, expiryPending: false,
    onCommand: vi.fn(), onReport: vi.fn(), onFollowUp: vi.fn() };
  await act(async () => root.render(createElement(ActivitySessionPanel, props)));
  expect(container.textContent).not.toContain("Did you try it?");
  await act(async () => root.render(createElement(ActivitySessionPanel, {
    ...props, session: { ...stopped, check_in_issued: true },
  })));
  expect(container.textContent?.match(/Did you try it\?/g)).toHaveLength(1);
  expect(container.querySelector<HTMLInputElement>('input[value="stopped"]')?.checked).toBe(false);
});
