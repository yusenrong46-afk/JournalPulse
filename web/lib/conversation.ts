export const OPEN_CONVERSATION_KEY = "journalpulse_open_conversation_v1";

const CONVERSATION_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export type TalkView = "unavailable" | "start" | "chat" | "support" | "saved";

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

export function talkView(input: {
  aiAvailable: boolean | null;
  status?: "open" | "closed" | null;
  safetyMode?: "normal" | "support" | null;
  saved?: boolean;
}): TalkView {
  if (input.saved) return "saved";
  if (input.status === "open" && input.safetyMode === "support") return "support";
  if (input.status === "open") return "chat";
  if (input.aiAvailable === false) return "unavailable";
  return "start";
}
