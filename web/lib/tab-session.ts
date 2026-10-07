"use client";

import { useMemo, useSyncExternalStore } from "react";
import { ACCOUNT_CHANGED_EVENT, browserAccount } from "./account-storage";
import { ACCOUNT_DATA_CHANGED_EVENT, accountDataRequestRevision } from "./account-data";

const PREFIX = "journalpulse_tab_v1:";
const CHANGE = "journalpulse-tab-session";
const memory = new Map<string, string>();
const volatile = new Set<string>();
const clearedScopes = new Set<string>();
let listening = false;
let previousAccount: ReturnType<typeof browserAccount>;

function prefix(account = browserAccount()): string {
  return `${PREFIX}${account === undefined ? "preview" : account ?? "signed-out"}:`;
}

function listen() {
  if (listening || typeof window === "undefined") return;
  listening = true;
  previousAccount = browserAccount();
  window.addEventListener(ACCOUNT_CHANGED_EVENT, () => {
    clearTabSession(previousAccount);
    previousAccount = browserAccount();
    window.dispatchEvent(new Event(CHANGE));
  });
  window.addEventListener(ACCOUNT_DATA_CHANGED_EVENT, () => clearTabSession());
  window.addEventListener("beforeunload", (event) => {
    const hasWriting = [...volatile].some((key) => key.startsWith(prefix())
      && /:(journal|chat:)/.test(key) && Boolean(memory.get(key)));
    if (hasWriting) { event.preventDefault(); event.returnValue = ""; }
  });
}

export function clearTabSession(account = browserAccount()): void {
  const scope = prefix(account);
  clearedScopes.add(scope);
  for (const key of memory.keys()) if (key.startsWith(scope)) { memory.delete(key); volatile.delete(key); }
  try {
    for (let index = window.sessionStorage.length - 1; index >= 0; index--) {
      const key = window.sessionStorage.key(index);
      if (key?.startsWith(scope)) window.sessionStorage.removeItem(key);
    }
  } catch { /* Tombstones below prevent inaccessible old records from returning. */ }
  window.dispatchEvent(new Event(CHANGE));
}

export function readTabValue(name: string): string {
  if (typeof window === "undefined" || browserAccount() === null) return "";
  listen();
  const key = prefix() + name;
  if (memory.has(key)) return memory.get(key)!;
  if (clearedScopes.has(prefix())) return "";
  try {
    const stored = window.sessionStorage.getItem(key);
    if (!stored) return "";
    const value = JSON.parse(stored) as { epoch?: string; value?: unknown };
    if (value.epoch !== accountDataRequestRevision() || typeof value.value !== "string") return "";
    memory.set(key, value.value);
    return value.value;
  } catch { return ""; }
}

export function writeTabValue(name: string, value: string): void {
  if (browserAccount() === null) return;
  listen();
  const key = prefix() + name;
  memory.set(key, value);
  try {
    if (value) window.sessionStorage.setItem(key, JSON.stringify({ epoch: accountDataRequestRevision(), value }));
    else window.sessionStorage.removeItem(key);
    volatile.delete(key);
  } catch { if (value) volatile.add(key); }
  window.dispatchEvent(new Event(CHANGE));
}

export function tabValueIsVolatile(name: string): boolean { return volatile.has(prefix() + name); }

function subscribe(callback: () => void) {
  window.addEventListener(CHANGE, callback);
  window.addEventListener(ACCOUNT_CHANGED_EVENT, callback);
  return () => { window.removeEventListener(CHANGE, callback); window.removeEventListener(ACCOUNT_CHANGED_EVENT, callback); };
}

export function useTabValue(name: string): [string, (value: string) => void, boolean] {
  const snapshot = useSyncExternalStore(subscribe, () => JSON.stringify([readTabValue(name), tabValueIsVolatile(name)]), () => '["",false]');
  const [value, temporary] = useMemo(() => JSON.parse(snapshot) as [string, boolean], [snapshot]);
  return [value, (next) => writeTabValue(name, next), temporary];
}

export function chatDraftKey(chat: { id: string; incarnation_id?: string | null; source_entry_id?: string | null }): string {
  return `chat:${chat.id}:${chat.incarnation_id ?? "legacy"}:source:${chat.source_entry_id ?? "none"}`;
}

export function clearSourceDrafts(sourceId: string): void {
  listen();
  const suffix = `:source:${sourceId}`;
  const keys = new Set(memory.keys());
  try { for (let i = 0; i < window.sessionStorage.length; i++) { const key = window.sessionStorage.key(i); if (key) keys.add(key); } } catch { /* Memory still clears. */ }
  for (const key of keys) if (key.startsWith(prefix()) && key.endsWith(suffix)) {
    writeTabValue(key.slice(prefix().length), "");
  }
  writeTabValue(`chat:new:${sourceId}`, "");
}

export const TAB_DRAFT_NOTE = "Unsaved writing stays in this browser tab through navigation and refresh. Save or discard it before leaving a shared device.";
export const VOLATILE_DRAFT_NOTE = "This browser cannot hold your draft through refresh. Your writing stays while you move between pages; save or copy it before closing or refreshing.";
