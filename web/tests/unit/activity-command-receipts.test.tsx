import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

vi.mock("@/lib/api", async (original) => ({ ...await original<typeof import("@/lib/api")>(), apiRequest: vi.fn() }));

import { ActivitySessionWorkspace } from "@/components/activity-session-workspace";
import { ApiError, apiRequest } from "@/lib/api";
import type { Conversation } from "@/lib/types";

// Fictional fixtures only.
const CHAT = "10000000-0000-4000-8000-000000000001";
const SESSION = "20000000-0000-4000-8000-000000000001";
const NOW = "2026-10-05T12:00:00Z";
const chat = {
  id: CHAT, user_id: "owner", created_at: NOW, updated_at: NOW, status: "open", mode: "ai", llm_consent: true,
  retain_text: false, safety_mode: "normal", locale: "CA", prompt_version: "test", revision: 2,
} as unknown as Conversation;

type Stored = Record<string, unknown> & { status: string; revision: number };
let stored: Stored;
let seen: Set<string>;
let applied: string[];
let loseNextResponse: boolean;
let root: Root;
let container: HTMLDivElement;

beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  vi.useFakeTimers({ toFake: ["setInterval", "clearInterval", "setTimeout", "clearTimeout", "performance"] });
  stored = {
    id: SESSION, user_id: "owner", conversation_id: CHAT, source_entry_id: null, revision: 1, status: "active",
    resource: {
      id: "quiet", title: "Quiet pause", url: null, provider: "JournalPulse", resource_type: "activity", format: "timer",
      kind: "meditation", duration_seconds: 600, instructions: [], provenance: "builtin",
    },
    recommendation_reason: null, selection: {}, duration_seconds: 600, remaining_seconds: 600,
    expires_at: "2026-10-05T12:10:00Z", created_at: NOW, updated_at: NOW, started_at: NOW, check_in_issued: false,
    report: null, reported_at: null, follow_up_status: "none", follow_up_reply: null, follow_up_attempts: 0,
    final_follow_up: false, server_now: NOW, conversation_revision: 2,
  };
  seen = new Set();
  applied = [];
  loseNextResponse = false;
  vi.mocked(apiRequest).mockReset().mockImplementation((async (path: string, init?: RequestInit) => {
    if (path.endsWith("/activity-sessions") && !init?.method) return { ...stored };
    const body = JSON.parse(String(init!.body));
    // Mirrors command_activity_session: a replayed receipt returns the current session.
    if (seen.has(body.client_request_id)) return { ...stored };
    seen.add(body.client_request_id);
    stored = { ...stored, status: body.command === "pause" ? "paused" : "active", revision: stored.revision + 1 };
    applied.push(body.command);
    if (loseNextResponse) {
      loseNextResponse = false;
      throw new ApiError("The request took too long. Your work is still on this page.", 0);
    }
    return { ...stored };
  }) as typeof apiRequest);
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => { root.unmount(); });
  container.remove();
  vi.useRealTimers();
});

async function click(label: string) {
  await act(async () => {
    [...container.querySelectorAll("button")].find((button) => button.textContent?.trim() === label)!.click();
  });
}
const visible = (label: string) => [...container.querySelectorAll("button")].some((button) => button.textContent?.trim() === label);

describe("activity command receipts", () => {
  test("a lost pause response does not swallow a later pause", async () => {
    // Regression: the receipt of a committed but unconfirmed Pause was reused for a
    // later Pause click, so the server replayed the old write and nothing happened.
    await act(async () => {
      root.render(createElement(ActivitySessionWorkspace, {
        conversation: chat, disabled: false, onRefresh: async () => null, onBusyChange: () => {},
      }));
    });
    loseNextResponse = true;
    await click("Pause");
    await act(async () => { vi.advanceTimersByTime(15_000); });
    expect(visible("Resume")).toBe(true);
    await click("Resume");
    expect(visible("Pause")).toBe(true);
    await click("Pause");
    expect(applied).toEqual(["pause", "resume", "pause"]);
    expect(stored.status).toBe("paused");
    expect(visible("Resume")).toBe(true);
  });

  test("retrying before the state changes still reuses the original receipt", async () => {
    await act(async () => {
      root.render(createElement(ActivitySessionWorkspace, {
        conversation: chat, disabled: false, onRefresh: async () => null, onBusyChange: () => {},
      }));
    });
    loseNextResponse = true;
    await click("Pause");
    const pauses = () => vi.mocked(apiRequest).mock.calls
      .filter(([, init]) => init?.method === "POST")
      .map(([, init]) => JSON.parse(String(init!.body)).client_request_id);
    await click("Pause");
    expect(new Set(pauses()).size).toBe(1);
    expect(applied).toEqual(["pause"]);
  });
});
