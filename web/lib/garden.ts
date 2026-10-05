import type { PlantStage } from "@/components/plant";

import { apiRequest } from "./api";
import type { components } from "./generated-api";

export type ActivityHistoryItem = components["schemas"]["ActivityHistoryItem"];

export const PARTICIPATION_WORDS: Record<ActivityHistoryItem["participation"], string> = {
  completed: "Tried it",
  partial: "Tried part of it",
  not_tried: "Didn’t try it",
  stopped: "Stopped early",
};

/**
 * Plants follow what the person reported, never the timer. "Not tried" stays a seed rather
 * than a failure, and growth needs the person's own rating, not mere participation.
 */
export function activityPlantStage(item: ActivityHistoryItem): PlantStage {
  if (item.participation === "not_tried" || item.participation === "stopped") return "seed";
  if ((item.helpfulness ?? 0) >= 4) return "flower";
  if (item.helpfulness === 3 || item.state_change === "toward_target") return "leafy";
  return "sprout";
}

/** Chat activity reports, newest first. A failure leaves the legacy garden usable. */
export async function loadActivityHistory(limit = 50): Promise<ActivityHistoryItem[] | null> {
  try {
    return (await apiRequest<{ items: ActivityHistoryItem[] }>(`/v1/activity-history?limit=${limit}`)).items;
  } catch {
    return null;
  }
}
