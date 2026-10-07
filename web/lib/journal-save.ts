"use client";

import { useSyncExternalStore } from "react";
import { ACCOUNT_CHANGED_EVENT, browserAccountRevision } from "./account-storage";
import { ACCOUNT_DATA_CHANGED_EVENT, accountDataRequestRevision } from "./account-data";
import { ApiError } from "./api";
import { saveJournalEntry } from "./journal";
import type { JournalEntry } from "./journal-types";
import { readTabValue, writeTabValue } from "./tab-session";

type SaveState = { status: "idle" | "saving" | "saved" | "failed"; entry?: JournalEntry; error?: string; revision: number };
const EMPTY: SaveState = { status: "idle", revision: 0 };
const listeners = new Set<() => void>();
let state: SaveState = EMPTY;
let generation = 0;
let attached = false;
let operation: Promise<JournalEntry | null> | null = null;

function publish(next: Omit<SaveState, "revision">) {
  state = { ...next, revision: state.revision + 1 };
  listeners.forEach((listener) => listener());
}

function attach() {
  if (attached) return;
  attached = true;
  const clear = () => { generation++; operation = null; publish({ status: "idle" }); };
  window.addEventListener(ACCOUNT_CHANGED_EVENT, clear);
  window.addEventListener(ACCOUNT_DATA_CHANGED_EVENT, clear);
}

function subscribe(callback: () => void) { attach(); listeners.add(callback); return () => { listeners.delete(callback); }; }
export function useJournalSave(): SaveState { return useSyncExternalStore(subscribe, () => state, () => EMPTY); }

/** The receipt outlives a page; account/erasure boundaries still invalidate its result. */
export function submitJournalSave(text: string): Promise<JournalEntry | null> {
  attach();
  if (operation) return operation;
  const epoch = accountDataRequestRevision();
  const account = browserAccountRevision();
  const started = generation;
  let receipt: { text: string; id: string; epoch: string } | null = null;
  try { receipt = JSON.parse(readTabValue("journal-receipt") || "null"); } catch { /* New receipt. */ }
  if (!receipt || receipt.text !== text || receipt.epoch !== epoch) receipt = { text, id: crypto.randomUUID(), epoch };
  writeTabValue("journal-receipt", JSON.stringify(receipt));
  publish({ status: "saving" });
  const valid = () => started === generation && account === browserAccountRevision() && epoch === accountDataRequestRevision();
  const pending = saveJournalEntry(text, receipt.id).then((entry) => {
    if (!valid()) return null;
    if (readTabValue("journal") === text) writeTabValue("journal", "");
    writeTabValue("journal-receipt", "");
    publish({ status: "saved", entry });
    return entry;
  }).catch((reason: unknown) => {
    if (!valid()) return null;
    if (reason instanceof ApiError && reason.status === 409) writeTabValue("journal-receipt", "");
    publish({ status: "failed", error: reason instanceof ApiError ? reason.message : "Your entry couldn’t save. Your writing is still in the editor." });
    return null;
  }).finally(() => { if (operation === pending) operation = null; });
  operation = pending;
  return pending;
}
