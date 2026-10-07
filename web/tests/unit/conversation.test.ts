import { describe, expect, test } from "vitest";

import {
  OPEN_CONVERSATION_KEY,
  canApplyConversation,
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
    writeOpenConversationId("10000000-0000-4000-8000-000000000010", storage);
    expect(storage.getItem(OPEN_CONVERSATION_KEY)).toBe("10000000-0000-4000-8000-000000000010");
    expect(readOpenConversationId(storage)).toBe("10000000-0000-4000-8000-000000000010");
    writeOpenConversationId("not-a-message", storage);
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
  test("an explicit listening choice overrides Luna and the message-count shortcut", () => {
    expect(readyForSomething({ userMessages: 12, readyForAction: true, preference: "listen" })).toBe(false);
  });
  test("follows current readiness without inferring consent from a turn count", () => {
    expect(readyForSomething({ userMessages: 1 })).toBe(false);
    expect(readyForSomething({ userMessages: 1, readyForAction: true })).toBe(true);
    expect(readyForSomething({ userMessages: 2 })).toBe(false);
    expect(readyForSomething({ userMessages: 12, readyForAction: false })).toBe(false);
  });
});

describe("canonical conversation updates", () => {
  const current = { id: "chat-a", revision: 5, status: "open", safety_mode: "normal" };
  test("ignores a response older than the saved choice and accepts a newer one", () => {
    expect(canApplyConversation(current, { ...current, revision: 4 })).toBe(false);
    expect(canApplyConversation(current, { ...current, revision: 6 })).toBe(true);
  });
  test("cannot reopen a closed chat or downgrade support", () => {
    expect(canApplyConversation({ ...current, status: "closed" }, current)).toBe(false);
    expect(canApplyConversation({ ...current, safety_mode: "support" }, current)).toBe(false);
    expect(canApplyConversation(current, { ...current, id: "chat-b" })).toBe(false);
  });
});

test("a reused chat ID cannot admit a response from another incarnation", () => {
  const current={id:"same",status:"open",revision:0,incarnation_id:"new"};
  expect(canApplyConversation(current,{...current,revision:10,incarnation_id:"old"})).toBe(false);
  expect(canApplyConversation(current,{...current,revision:1})).toBe(true);
});
