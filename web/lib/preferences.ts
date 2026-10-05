"use client";

import { useMemo, useSyncExternalStore } from "react";

import { ACCOUNT_CHANGED_EVENT, readAccountStorage, writeAccountStorage } from "./account-storage";

export type Preferences = {
  onboarded: boolean;
  llmConsent: boolean;
  retainText: boolean;
  encryptedDrafts: boolean;
  followUpMinutes: number;
  locale: string;
};

export const DEFAULT_PREFERENCES: Preferences = {
  onboarded: false,
  llmConsent: false,
  retainText: false,
  encryptedDrafts: false,
  followUpMinutes: 10,
  locale: "CA",
};

const STORAGE_KEY = "journalpulse_preferences_v1";
const DEFAULT_SNAPSHOT = "__journalpulse_default__";
const SERVER_SNAPSHOT = "__journalpulse_server__";

function subscribe(callback: () => void) {
  window.addEventListener("journalpulse-preferences", callback);
  window.addEventListener("storage", callback);
  window.addEventListener(ACCOUNT_CHANGED_EVENT, callback);
  return () => {
    window.removeEventListener("journalpulse-preferences", callback);
    window.removeEventListener("storage", callback);
    window.removeEventListener(ACCOUNT_CHANGED_EVENT, callback);
  };
}

function getSnapshot() {
  return readAccountStorage(STORAGE_KEY) ?? DEFAULT_SNAPSHOT;
}

function getServerSnapshot() {
  return SERVER_SNAPSHOT;
}

export function savePreferences(preferences: Preferences): void {
  writeAccountStorage(STORAGE_KEY, JSON.stringify(validatedPreferences(preferences)));
  window.dispatchEvent(new Event("journalpulse-preferences"));
}

function validatedPreferences(value: unknown): Preferences {
  if (!value || typeof value !== "object" || Array.isArray(value)) return DEFAULT_PREFERENCES;
  const stored = value as Record<string, unknown>;
  // Browser storage is untrusted input: a string such as "false" must never
  // become consent through JavaScript truthiness.
  return {
    onboarded: stored.onboarded === true,
    llmConsent: stored.llmConsent === true,
    retainText: stored.retainText === true,
    encryptedDrafts: stored.encryptedDrafts === true,
    followUpMinutes: typeof stored.followUpMinutes === "number" && [5, 10, 20, 60].includes(stored.followUpMinutes)
      ? stored.followUpMinutes : DEFAULT_PREFERENCES.followUpMinutes,
    locale: typeof stored.locale === "string" && /^[a-z]{2}(?:-[a-z]{2})?$/i.test(stored.locale)
      ? stored.locale : DEFAULT_PREFERENCES.locale,
  };
}

export function usePreferences(): [Preferences, (value: Preferences) => void, boolean] {
  const snapshot = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const preferences = useMemo(() => {
    if (snapshot === SERVER_SNAPSHOT || snapshot === DEFAULT_SNAPSHOT) return DEFAULT_PREFERENCES;
    try {
      return validatedPreferences(JSON.parse(snapshot));
    } catch {
      return DEFAULT_PREFERENCES;
    }
  }, [snapshot]);
  return [preferences, savePreferences, snapshot !== SERVER_SNAPSHOT];
}
