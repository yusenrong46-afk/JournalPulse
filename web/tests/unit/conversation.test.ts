import { describe, expect, test } from "vitest";

import {
  OPEN_CONVERSATION_KEY,
  readOpenConversationId,
  talkView,
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

describe("talk view", () => {
  test("follows availability, support, and a saved choice", () => {
    expect(talkView({ aiAvailable: false })).toBe("unavailable");
    expect(talkView({ aiAvailable: null })).toBe("start");
    expect(talkView({ aiAvailable: true, status: "open" })).toBe("chat");
    expect(talkView({ aiAvailable: true, status: "open", safetyMode: "support" })).toBe("support");
    expect(talkView({ aiAvailable: true, status: "closed" })).toBe("start");
    expect(talkView({ aiAvailable: false, saved: true })).toBe("saved");
  });
});
