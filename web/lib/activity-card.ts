import type { ActionCard, Conversation } from "./types";

/** New activity recommendations are separate so older production clients can still parse legacy cards. */
export function currentActivityCard(conversation: Conversation | null | undefined): ActionCard | null {
  const current = conversation as (Conversation & { activity_card?: ActionCard | null }) | null | undefined;
  // Support resources take precedence even if an older server returns a stale
  // ordinary activity card alongside the canonical support card.
  if (current?.safety_mode === "support") return current.card ?? null;
  return current?.activity_card ?? current?.card ?? null;
}
