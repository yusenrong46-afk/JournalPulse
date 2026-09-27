import { describe, expect, test } from "vitest";

import {
  OPEN_CONVERSATION_KEY,
  chatStage,
  readOpenConversationId,
  readyForSomething,
  writeOpenConversationId,
} from "@/lib/conversation";

function memory() {
  const values = new Map<string, string>();
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
  };
}

describe("open conversation id", () => {
  test("stores only a conversation id and drops anything else", () => {
    const storage = memory();
    writeOpenConversationId(storage, "10000000-0000-4000-8000-000000000010");
    expect(storage.getItem(OPEN_CONVERSATION_KEY)).toBe("10000000-0000-4000-8000-000000000010");
    expect(readOpenConversationId(storage)).toBe("10000000-0000-4000-8000-000000000010");
    writeOpenConversationId(storage, "not-a-message");
    expect(readOpenConversationId(storage)).toBeNull();
    expect(storage.getItem(OPEN_CONVERSATION_KEY)).toBeNull();
  });
});

describe("chat stage", () => {
  const base = { saved: false, status: "open" as const, safetyMode: "normal" as const, hasCard: false, step: null };

  test("starts at the welcome until a conversation is open", () => {
    expect(chatStage({ ...base, status: null })).toBe("welcome");
    expect(chatStage({ ...base, status: "closed" })).toBe("welcome");
    expect(chatStage(base)).toBe("chat");
  });

  test("walks through feelings, goal, and the offered card", () => {
    expect(chatStage({ ...base, step: "feelings" })).toBe("feelings");
    expect(chatStage({ ...base, step: "goal" })).toBe("goal");
    expect(chatStage({ ...base, hasCard: true })).toBe("offer");
    expect(chatStage({ ...base, hasCard: true, saved: true })).toBe("saved");
  });

  test("support mode overrides every other step", () => {
    expect(chatStage({ ...base, safetyMode: "support", step: "goal", hasCard: true })).toBe("support");
  });
});

describe("ready prompt", () => {
  test("follows Luna or appears after two messages", () => {
    expect(readyForSomething({ userMessages: 1 })).toBe(false);
    expect(readyForSomething({ userMessages: 1, readyForAction: true })).toBe(true);
    expect(readyForSomething({ userMessages: 2 })).toBe(true);
  });
});
