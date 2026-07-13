import { beforeEach, describe, expect, test } from "vitest";

import { clearReminder, saveReminder } from "@/lib/reminders";

beforeEach(() => window.localStorage.clear());

describe("device follow-up reminders", () => {
  test("stores only safe action metadata and replaces the same decision", () => {
    saveReminder({
      decisionId: "decision-1",
      actionId: "approved-action",
      actionTitle: "A reviewed action",
      dueAt: "2026-07-12T12:10:00Z",
    });
    saveReminder({
      decisionId: "decision-1",
      actionId: "approved-action",
      actionTitle: "Updated reviewed action",
      dueAt: "2026-07-12T12:20:00Z",
    });

    const stored = JSON.parse(window.localStorage.getItem("journalpulse_reminders_v1") ?? "[]");
    expect(stored).toHaveLength(1);
    expect(stored[0].actionTitle).toBe("Updated reviewed action");
    expect(JSON.stringify(stored)).not.toContain("journal");

    clearReminder("decision-1");
    expect(window.localStorage.getItem("journalpulse_reminders_v1")).toBe("[]");
  });
});
