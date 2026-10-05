import { readAccountStorage, writeAccountStorage } from "./account-storage";

export const OPEN_CONVERSATION_KEY = "journalpulse_open_conversation_v1";

const CONVERSATION_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

/** A source URL selects a single entry, never an arbitrary history query. */
export function journalSourceId(value: string | null): string | null {
  return value && CONVERSATION_ID.test(value) ? value.toLowerCase() : null;
}

/** A selected entry must never silently replace the source of an existing chat. */
export function needsNewJournalChat(
  conversation: { source_entry_id?: string | null } | null,
  entryId: string,
): boolean {
  return Boolean(conversation && conversation.source_entry_id?.toLowerCase() !== entryId.toLowerCase());
}

/** Workspace changes invalidate all pending responses, including a first chat creation. */
export function isCurrentChatRequest(startedGeneration: number, currentGeneration: number): boolean {
  return startedGeneration === currentGeneration;
}

type StorageLike = {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
};

export function readOpenConversationId(storage?: Pick<StorageLike, "getItem">): string | null {
  const value = readAccountStorage(OPEN_CONVERSATION_KEY, storage);
  return value && CONVERSATION_ID.test(value) ? value : null;
}

export function writeOpenConversationId(id: string | null, storage?: StorageLike): void {
  writeAccountStorage(OPEN_CONVERSATION_KEY, id && CONVERSATION_ID.test(id) ? id : null, storage);
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

/** The current turn can invite action; a message count does not express user intent. */
export function readyForSomething(input: {
  readyForAction?: boolean; userMessages: number; preference?: "auto" | "listen" | "act";
}): boolean {
  if (input.preference === "listen") return false;
  return Boolean(input.readyForAction);
}

/** An HTTP response can arrive after a newer choice, close, or support transition. */
export function canApplyConversation(
  current: { id: string; revision?: number; status: string; safety_mode?: string } | null,
  next: { id: string; revision?: number; status: string; safety_mode?: string },
): boolean {
  if (!current) return true;
  if (current.id !== next.id || (next.revision ?? 0) < (current.revision ?? 0)) return false;
  if (current.status === "closed" && next.status !== "closed") return false;
  return current.safety_mode !== "support" || next.safety_mode === "support";
}
