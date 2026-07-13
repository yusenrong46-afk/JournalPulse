"use client";

import { useMemo, useSyncExternalStore } from "react";

export type Preferences = {
  onboarded: boolean;
  llmConsent: boolean;
  retainText: boolean;
  followUpMinutes: number;
  locale: string;
};

export const DEFAULT_PREFERENCES: Preferences = {
  onboarded: false,
  llmConsent: false,
  retainText: false,
  followUpMinutes: 10,
  locale: "CA",
};

const STORAGE_KEY = "journalpulse_preferences_v1";
const DEFAULT_SNAPSHOT = "__journalpulse_default__";
const SERVER_SNAPSHOT = "__journalpulse_server__";

function subscribe(callback: () => void) {
  window.addEventListener("journalpulse-preferences", callback);
  window.addEventListener("storage", callback);
  return () => {
    window.removeEventListener("journalpulse-preferences", callback);
    window.removeEventListener("storage", callback);
  };
}

function getSnapshot() {
  return window.localStorage.getItem(STORAGE_KEY) ?? DEFAULT_SNAPSHOT;
}

function getServerSnapshot() {
  return SERVER_SNAPSHOT;
}

export function savePreferences(preferences: Preferences): void {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
  window.dispatchEvent(new Event("journalpulse-preferences"));
}

export function usePreferences(): [Preferences, (value: Preferences) => void, boolean] {
  const snapshot = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const preferences = useMemo(() => {
    if (snapshot === SERVER_SNAPSHOT || snapshot === DEFAULT_SNAPSHOT) return DEFAULT_PREFERENCES;
    try {
      return { ...DEFAULT_PREFERENCES, ...JSON.parse(snapshot) } as Preferences;
    } catch {
      return DEFAULT_PREFERENCES;
    }
  }, [snapshot]);
  return [preferences, savePreferences, snapshot !== SERVER_SNAPSHOT];
}
