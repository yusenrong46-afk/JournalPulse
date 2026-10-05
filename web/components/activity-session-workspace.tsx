"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "@/lib/api";
import { currentActivityCard } from "@/lib/activity-card";
import Link from "next/link";
import { discoveryHref } from "@/lib/discovery";
import {
  activityClock, activityReceipt, activitySecondsLeft, canApplyActivity, commandActivity,
  createActivity, followUpActivity, readCurrentActivity, reportActivity,
  type ActivityClock, type ActivityCommand, type ActivityReport, type ActivityResource, type ActivitySession,
} from "@/lib/activity-session";
import type { Conversation } from "@/lib/types";
import { ActivitySessionDiscovery } from "./activity-session-discovery";
import { ActivitySessionPanel } from "./activity-session-panel";

type Props = {
  conversation: Conversation; disabled: boolean; onRefresh(): Promise<unknown>; onBusyChange(value: boolean): void;
  ordinaryMessages?: number;
};
type PendingOperation = { key: string; payload: string };

/** All authoritative state lives on the server. This component keeps only display clocks and retry receipts. */
export function ActivitySessionWorkspace({ conversation, disabled, onRefresh, onBusyChange, ordinaryMessages = 0 }: Props) {
  const pausedByConversation = (conversation as Conversation & { activity_move?: string }).activity_move === "pause";
  const [session, setSession] = useState<ActivitySession | null>(null);
  const [secondsLeft, setSecondsLeft] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchOpen, setSearchOpen] = useState(false);
  const [retryOperation, setRetryOperation] = useState<(() => Promise<void>) | null>(null);
  const current = useRef<ActivitySession | null>(null);
  const chat = useRef(conversation);
  const clock = useRef<ActivityClock | null>(null);
  const mounted = useRef(true);
  const generation = useRef(0);
  const pending = useRef<AbortController | null>(null);
  const readPending = useRef<AbortController | null>(null);
  const receipts = useRef(new Map<string, PendingOperation>());
  const expiryAttempt = useRef<string | null>(null);
  const refreshedFollowUp = useRef<string | null>(null);
  const inFlight = useRef(false);
  const busyCallback = useRef(onBusyChange);
  const refreshCallback = useRef(onRefresh);
  useEffect(() => {
    chat.current = conversation;
    busyCallback.current = onBusyChange;
    refreshCallback.current = onRefresh;
  }, [conversation, onBusyChange, onRefresh]);

  const apply = useCallback((next: ActivitySession | null) => {
    if (!mounted.current) return false;
    if (next && !canApplyActivity(current.current, next, chat.current.id)) return false;
    // A latest read may legitimately be empty after source/chat deletion.
    current.current = next;
    clock.current = next ? activityClock(next, performance.now()) : null;
    setSession(next);
    setSecondsLeft(next && clock.current ? activitySecondsLeft(next, clock.current, performance.now()) : 0);
    if (next?.follow_up_status === "ready" && next.follow_up_message_id
      && refreshedFollowUp.current !== next.follow_up_message_id) {
      const messageId = next.follow_up_message_id;
      refreshedFollowUp.current = messageId;
      // A different tab can finish generation after our HTTP 202. The saved message
      // must also appear when a later sync discovers it, without another model call.
      void refreshCallback.current().then((result) => {
        if (result === null && refreshedFollowUp.current === messageId) refreshedFollowUp.current = null;
      }).catch(() => { if (refreshedFollowUp.current === messageId) refreshedFollowUp.current = null; });
    }
    return true;
  }, []);

  const sync = useCallback(async () => {
    const started = generation.current;
    const controller = new AbortController();
    readPending.current?.abort();
    readPending.current = controller;
    try {
      const next = await readCurrentActivity(chat.current.id, controller.signal);
      if (!mounted.current || generation.current !== started || readPending.current !== controller) return;
      if (apply(next)) setLoading(false);
    } catch (reason) {
      if (!mounted.current || generation.current !== started || readPending.current !== controller) return;
      setLoading(false);
      if (reason instanceof ApiError && reason.status === 404) {
        apply(null);
        await refreshCallback.current();
      } else setError("We couldn’t sync your activity. Your chat is still available; try syncing again.");
    } finally {
      if (readPending.current === controller) readPending.current = null;
    }
  }, [apply]);

  useEffect(() => {
    mounted.current = true;
    queueMicrotask(() => { if (mounted.current) void sync(); });
    const resume = () => { if (document.visibilityState !== "hidden") void sync(); };
    window.addEventListener("focus", resume);
    document.addEventListener("visibilitychange", resume);
    const poll = window.setInterval(resume, 15_000);
    return () => {
      mounted.current = false;
      generation.current += 1;
      pending.current?.abort();
      readPending.current?.abort();
      busyCallback.current(false);
      window.clearInterval(poll);
      window.removeEventListener("focus", resume);
      document.removeEventListener("visibilitychange", resume);
    };
  }, [sync]);

  useEffect(() => {
    // A chat revision may revoke an offer/source or switch to support while a response is delayed.
    generation.current += 1;
    pending.current?.abort();
    pending.current = null;
    inFlight.current = false;
    queueMicrotask(() => {
      if (mounted.current) { busyCallback.current(false); setBusy(false); setRetryOperation(null); void sync(); }
    });
  }, [conversation.revision, sync]);

  useEffect(() => {
    const topic = (conversation as Conversation & { activity_search_topic?: string | null }).activity_search_topic;
    if (topic) queueMicrotask(() => { if (mounted.current) setSearchOpen(true); });
  }, [conversation]);

  useEffect(() => {
    const repaint = () => {
      const value = current.current;
      const anchor = clock.current;
      if (value && anchor) setSecondsLeft(activitySecondsLeft(value, anchor, performance.now()));
    };
    const interval = window.setInterval(repaint, 250);
    return () => window.clearInterval(interval);
  }, []);

  function receipt<T>(key: string, build: (requestId: string) => T): T {
    const existing = receipts.current.get(key);
    // A lost HTTP response may hide a committed write. Preserve its complete command,
    // including the old revision, so the server can recover the original receipt.
    if (existing) return JSON.parse(existing.payload) as T;
    const payload = build(crypto.randomUUID());
    receipts.current.set(key, { key, payload: JSON.stringify(payload) });
    return payload;
  }

  async function operation(action: (signal: AbortSignal) => Promise<void>, retry?: () => Promise<void>) {
    if (inFlight.current || disabled || pausedByConversation) return;
    const started = generation.current;
    const controller = new AbortController();
    readPending.current?.abort();
    readPending.current = null;
    pending.current?.abort();
    pending.current = controller;
    inFlight.current = true;
    setBusy(true);
    busyCallback.current(true);
    setError(null);
    setRetryOperation(null);
    try { await action(controller.signal); }
    catch (reason) {
      if (!mounted.current || generation.current !== started || pending.current !== controller) return;
      setError(reason instanceof Error ? reason.message : "This activity change could not be confirmed.");
      if (reason instanceof ApiError && (reason.status === 409 || reason.status === 404)) {
        receipts.current.clear();
        await sync();
        await refreshCallback.current();
      } else if (retry) setRetryOperation(() => retry);
    } finally {
      if (mounted.current && generation.current === started && pending.current === controller) {
        pending.current = null;
        inFlight.current = false;
        setBusy(false);
        busyCallback.current(false);
      }
    }
  }

  async function control(command: ActivityCommand) {
    const value = current.current;
    if (!value) return;
    const started = generation.current;
    const key = `command:${value.id}:${command}`;
    const payload = receipt(key, (requestId) => ({ ...activityReceipt(value, chat.current, requestId), command }));
    await operation(async (signal) => {
      const next = await commandActivity(value.id, payload, signal);
      if (generation.current === started && !signal.aborted) {
        receipts.current.delete(key);
        if (command === "start") receipts.current.delete(`create:${next.resource.id}`);
        apply(next);
      }
    }, () => control(command));
  }

  async function saveOffer(resourceId: string, resourceToken?: string, start = false) {
    const started = generation.current;
    const expected = chat.current.revision ?? 0;
    const key = `create:${resourceId}`;
    const payload = receipt(key, (requestId) => ({
      client_request_id: requestId, expected_conversation_revision: expected, resource_id: resourceId,
      ...(resourceToken ? { resource_token: resourceToken } : {}),
    }));
    let confirmed = false;
    await operation(async (signal) => {
      const offered = await createActivity(chat.current.id, payload, signal);
      if (generation.current !== started || signal.aborted) return;
      apply(offered);
      confirmed = true;
      if (start) {
        const startKey = `command:${offered.id}:start`;
        const startPayload = receipt(startKey, (requestId) => ({
          ...activityReceipt(offered, chat.current, requestId), command: "start" as const,
        }));
        const active = await commandActivity(offered.id, startPayload, signal);
        if (generation.current === started && !signal.aborted) { receipts.current.delete(startKey); apply(active); }
      }
      receipts.current.delete(key);
    }, async () => { await saveOffer(resourceId, resourceToken, start); });
    return confirmed;
  }

  async function followUp(value = current.current) {
    if (!value?.report || value.follow_up_status === "ready") return;
    const started = generation.current;
    const key = `followup:${value.id}`;
    const payload = receipt(key, (requestId) => activityReceipt(value, chat.current, requestId));
    const run = async (signal: AbortSignal) => {
      const next = await followUpActivity(value.id, payload, signal);
      if (generation.current !== started || signal.aborted) return;
      receipts.current.delete(key);
      apply(next);
    };
    await operation(run, () => followUp());
  }

  async function saveReport(report: ActivityReport) {
    const value = current.current;
    if (!value) return;
    const started = generation.current;
    const key = `report:${value.id}:${JSON.stringify(report)}`;
    const payload = receipt(key, (requestId) => ({ ...activityReceipt(value, chat.current, requestId), ...report }));
    let saved: ActivitySession | null = null;
    await operation(async (signal) => {
      const next = await reportActivity(value.id, payload, signal);
      if (generation.current === started && !signal.aborted) {
        receipts.current.delete(key); apply(next); saved = next;
        // A concerning report may atomically move the chat to support. Show that
        // authoritative route immediately, before attempting a normal follow-up.
        if ((next.conversation_revision ?? 0) > (chat.current.revision ?? 0)) await refreshCallback.current();
      }
    }, () => saveReport(report));
    // Only the independently confirmed report triggers a model response. Failed generation leaves it intact.
    if (saved && mounted.current && generation.current === started) await followUp(saved);
  }

  // A newer chat can invalidate the timer before its canonical session read arrives.
  // Never revive an old completion question during that synchronization window.
  const sessionNeedsSync = Boolean(session && (session.conversation_revision ?? 0) < (conversation.revision ?? 0));
  const expiryPending = !disabled && !pausedByConversation && !sessionNeedsSync
    && Boolean(session?.status === "active" && session.resource.format === "timer" && secondsLeft === 0);
  useEffect(() => {
    if (!expiryPending || !session || disabled || busy) return;
    const key = `${session.id}:${session.revision}`;
    if (expiryAttempt.current === key) return;
    expiryAttempt.current = key;
    // One deterministic question is rendered locally; the server receipt makes tabs/retries converge.
    void control("expire");
  });

  const terminal = session && ["completed", "stopped", "declined"].includes(session.status);
  const card = currentActivityCard(conversation);
  const primary = card?.actions.find((resource) => resource.id === card.decision_preview.action_id) ?? card?.actions[0];
  const consumedOffer = (session as (ActivitySession & { offered_message_id?: string | null }) | null)?.offered_message_id;
  const freshOffer = card?.offered_message_id && consumedOffer && card.offered_message_id !== consumedOffer;
  const canStart = ordinaryMessages < 20 && !pausedByConversation;
  const showOffer = Boolean(canStart && primary && (!session || (terminal && (freshOffer || primary.id !== session.resource.id))));

  async function saveSearch(resource: ActivityResource, token: string) {
    const saved = await saveOffer(resource.id, token);
    if (!saved) throw new Error("The resource save is not confirmed. Check the activity status and retry.");
  }

  return (
    <div className="stack">
      {loading && <p className="small muted" role="status">Syncing your activity…</p>}
      {showOffer && primary && <section className="card stack" aria-label="Activity with Luna">
        <h2>{primary.title}</h2><p>{card?.card_reason ?? primary.summary}</p>
        {!primary.id.startsWith("guided_") && <p className="small muted">Saved resource collection</p>}
        {primary.duration_minutes && <p className="small muted">About {primary.duration_minutes} minutes</p>}
        <div className="row"><button className="btn btn-primary" type="button" disabled={busy || disabled || loading}
          onClick={() => void saveOffer(primary.id, undefined, true)}>Start activity</button>
          <button className="btn btn-ghost" type="button" disabled={busy || disabled || loading}
            onClick={() => void saveOffer(primary.id).then((saved) => { if (saved) void control("decline"); })}>Not now</button></div>
        <p className="small muted">You can tell Luna what would fit better in the chat.</p>
        <Link className="small muted" href={discoveryHref(card?.goal)}>Search for other resources</Link>
      </section>}
      {session && <ActivitySessionPanel session={session} secondsLeft={secondsLeft} busy={busy || disabled || pausedByConversation || sessionNeedsSync}
        suppressQuestions={disabled || pausedByConversation || sessionNeedsSync}
        canStart={canStart}
        expiryPending={expiryPending} onCommand={(command) => void control(command)}
        onReport={(report) => void saveReport(report)} onFollowUp={() => void followUp()} />}
      <button className="chip" type="button" disabled={busy || disabled || !canStart} onClick={() => setSearchOpen((value) => !value)}>
        {searchOpen ? "Close activity search" : "Find another resource"}
      </button>
      {searchOpen && <ActivitySessionDiscovery conversation={conversation} disabled={busy || disabled || !canStart} onSave={saveSearch} />}
      {error && <div className="stack"><p role="alert" className="note error">{error}</p>
        {retryOperation && <button className="btn btn-soft" type="button" disabled={busy || disabled}
          onClick={() => void retryOperation()}>Retry activity change</button>}
        <button className="btn btn-ghost" type="button" disabled={busy || disabled} onClick={() => void sync()}>Sync activity</button>
      </div>}
    </div>
  );
}
