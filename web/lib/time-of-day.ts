"use client";

import { useSyncExternalStore } from "react";

export type TimeOfDay = "morning" | "afternoon" | "evening" | "night";

export function timeOfDay(date: Date): TimeOfDay {
  const hour = date.getHours();
  if (hour >= 5 && hour < 12) return "morning";
  if (hour >= 12 && hour < 17) return "afternoon";
  if (hour >= 17 && hour < 22) return "evening";
  return "night";
}

export function greeting(time: TimeOfDay): string {
  if (time === "morning") return "Good morning";
  if (time === "afternoon") return "Good afternoon";
  if (time === "evening") return "Good evening";
  return "Hi, night owl";
}

function subscribe(callback: () => void) {
  const interval = window.setInterval(callback, 60_000);
  return () => window.clearInterval(interval);
}

/** The reader's local part of the day. Resolved on the client so static pages never bake it in. */
export function useTimeOfDay(): TimeOfDay | null {
  return useSyncExternalStore(
    subscribe,
    () => timeOfDay(new Date()),
    () => null,
  );
}
