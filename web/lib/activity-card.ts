import type { ActionCard, Conversation } from "./types";

/** New activity recommendations are separate so older production clients can still parse legacy cards. */
export function currentActivityCard(conversation: Conversation | null | undefined): ActionCard | null {
  const current = conversation as (Conversation & { activity_card?: ActionCard | null }) | null | undefined;
  return current?.activity_card ?? current?.card ?? null;
}
