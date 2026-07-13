"use client";

import { useMemo, useSyncExternalStore } from "react";

const STORAGE_KEY = "journalpulse_reminders_v1";
const EMPTY_SNAPSHOT = "[]";

export type FollowUpReminder = {
  decisionId: string;
  actionId: string;
  actionTitle: string;
  dueAt: string;
};

function subscribe(callback: () => void) {
  window.addEventListener("journalpulse-reminders", callback);
  window.addEventListener("storage", callback);
  return () => {
    window.removeEventListener("journalpulse-reminders", callback);
    window.removeEventListener("storage", callback);
  };
}

function snapshot() {
  return window.localStorage.getItem(STORAGE_KEY) ?? EMPTY_SNAPSHOT;
}

function parsed(value: string): FollowUpReminder[] {
  try {
    const reminders = JSON.parse(value) as FollowUpReminder[];
    return Array.isArray(reminders) ? reminders : [];
  } catch {
    return [];
  }
}

function write(reminders: FollowUpReminder[]) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(reminders));
  window.dispatchEvent(new Event("journalpulse-reminders"));
}

export function saveReminder(reminder: FollowUpReminder): void {
  write([...parsed(snapshot()).filter((item) => item.decisionId !== reminder.decisionId), reminder]);
}

export function clearReminder(decisionId: string): void {
  write(parsed(snapshot()).filter((item) => item.decisionId !== decisionId));
}

export function useReminders(): FollowUpReminder[] {
  const value = useSyncExternalStore(subscribe, snapshot, () => EMPTY_SNAPSHOT);
  return useMemo(() => parsed(value), [value]);
}
