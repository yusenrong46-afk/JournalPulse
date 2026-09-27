export const OPEN_CONVERSATION_KEY = "journalpulse_open_conversation_v1";

const CONVERSATION_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

type StorageLike = {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
};

export function readOpenConversationId(storage: Pick<StorageLike, "getItem">): string | null {
  const value = storage.getItem(OPEN_CONVERSATION_KEY);
  return value && CONVERSATION_ID.test(value) ? value : null;
}

export function writeOpenConversationId(storage: StorageLike, id: string | null): void {
  if (id && CONVERSATION_ID.test(id)) storage.setItem(OPEN_CONVERSATION_KEY, id);
  else storage.removeItem(OPEN_CONVERSATION_KEY);
}

/**
 * Where the chat is. The feelings and goal steps are prompts Luna shows on the page; the
 * server only learns the result, as a normal message carrying the chosen goal.
 */
export type ChatStage = "welcome" | "chat" | "feelings" | "goal" | "offer" | "support" | "saved";

export function chatStage(input: {
  saved: boolean;
  status?: "open" | "closed" | null;
  safetyMode?: "normal" | "support" | null;
  hasCard: boolean;
  step: "feelings" | "goal" | null;
}): ChatStage {
  if (input.saved) return "saved";
  if (input.status !== "open") return "welcome";
  if (input.safetyMode === "support") return "support";
  if (input.step) return input.step;
  if (input.hasCard) return "offer";
  return "chat";
}

/** Offer the "find one small thing" prompt once Luna says so, or after a couple of turns. */
export function readyForSomething(input: { readyForAction?: boolean; userMessages: number }): boolean {
  return Boolean(input.readyForAction) || input.userMessages >= 2;
}
