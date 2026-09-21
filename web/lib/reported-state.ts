import type { AffectiveState } from "./types";

/** Return the reported state unchanged, including whatever confidence it already has. */
export function reportedState(state: AffectiveState): AffectiveState {
  return { ...state };
}
