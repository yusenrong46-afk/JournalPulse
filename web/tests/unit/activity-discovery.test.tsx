import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

vi.mock("@/lib/api", async (original) => ({
  ...await original<typeof import("@/lib/api")>(), apiRequest: vi.fn(),
}));

import { ActivitySessionDiscovery } from "@/components/activity-session-discovery";
import { apiRequest } from "@/lib/api";
import type { components } from "@/lib/generated-api";
import type { Conversation } from "@/lib/types";

const constraints = { time_minutes: 2, no_audio: true, no_video: true, seated: true, avoid_breath_focus: true };
const conversation: Conversation & { activity_constraints: typeof constraints } = {
  id: "10000000-0000-4000-8000-000000000001", user_id: "owner", status: "open", mode: "ai",
  created_at: "2026-10-05T12:00:00Z", updated_at: "2026-10-05T12:00:00Z", revision: 4,
  llm_consent: true, retain_text: false, safety_mode: "normal", locale: "CA", prompt_version: "test",
  activity_constraints: constraints,
};

const requested = vi.mocked(apiRequest);
let root: Root;
let container: HTMLDivElement;
beforeEach(() => {
  (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
  requested.mockReset().mockResolvedValue({
    original_query: "quiet meditation", updated_query: "quiet meditation", candidates: [], offers: [],
    conversation_revision: 4,
  });
  container = document.createElement("div"); document.body.appendChild(container); root = createRoot(container);
});
afterEach(async () => { await act(async () => root.unmount()); container.remove(); });

test("inline search sends confirmed constraints in the API's nested request field", async () => {
  await act(async () => root.render(createElement(ActivitySessionDiscovery, {
    conversation, disabled: false, onSave: vi.fn(),
  })));
  await act(async () => { container.querySelector<HTMLInputElement>('input[type="checkbox"]')!.click(); });
  await act(async () => { container.querySelector("form")!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true })); });

  expect(requested).toHaveBeenCalledOnce();
  const [path, init] = requested.mock.calls[0];
  const payload = JSON.parse(String(init?.body));
  const expected: components["schemas"]["InlineDiscoveryRequest"] = {
    goal: "settle", style: "ground", constraints, llm_consent: true, expected_revision: 4, excluded_urls: [],
  };
  expect(path).toBe(`/v1/conversations/${conversation.id}/discover`);
  expect(payload).toEqual(expected);
  expect(init?.signal).toBeInstanceOf(AbortSignal);
  expect(init?.retry).toBe(false);
  expect(container.textContent).toContain("No fitting sources this time");
});
