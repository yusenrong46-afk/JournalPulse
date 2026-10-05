"use client";

import { useMemo, useSyncExternalStore } from "react";

import { ACCOUNT_CHANGED_EVENT, readAccountStorage, writeAccountStorage } from "./account-storage";

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
  window.addEventListener(ACCOUNT_CHANGED_EVENT, callback);
  return () => {
    window.removeEventListener("journalpulse-reminders", callback);
    window.removeEventListener("storage", callback);
    window.removeEventListener(ACCOUNT_CHANGED_EVENT, callback);
  };
}

function snapshot() {
  return readAccountStorage(STORAGE_KEY) ?? EMPTY_SNAPSHOT;
}

function parsed(value: string): FollowUpReminder[] {
  try {
    const reminders: unknown = JSON.parse(value);
    if (!Array.isArray(reminders)) return [];
    return reminders.filter((item): item is FollowUpReminder => Boolean(
      item && typeof item === "object" && typeof item.decisionId === "string"
      && typeof item.actionId === "string" && typeof item.actionTitle === "string"
      && typeof item.dueAt === "string" && Number.isFinite(Date.parse(item.dueAt)),
    ));
  } catch {
    return [];
  }
}

function write(reminders: FollowUpReminder[]) {
  writeAccountStorage(STORAGE_KEY, JSON.stringify(reminders));
  window.dispatchEvent(new Event("journalpulse-reminders"));
}

export function saveReminder(reminder: FollowUpReminder): void {
  write([...parsed(snapshot()).filter((item) => item.decisionId !== reminder.decisionId), reminder]);
}

export function clearReminder(decisionId: string): void {
  write(parsed(snapshot()).filter((item) => item.decisionId !== decisionId));
}

export function clearReminders(): void {
  write([]);
}

export function useReminders(): FollowUpReminder[] {
  const value = useSyncExternalStore(subscribe, snapshot, () => EMPTY_SNAPSHOT);
  return useMemo(() => parsed(value), [value]);
}
