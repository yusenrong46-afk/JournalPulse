"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { ApiError } from "@/lib/api";
import { currentActivityCard } from "@/lib/activity-card";
import {
  activityClock, activityReceipt, activitySecondsLeft, canApplyActivity, commandActivity,
  createActivity, followUpActivity, readCurrentActivity, reportActivity,
  type ActivityClock, type ActivityCommand, type ActivityReport, type ActivityResource, type ActivitySession,
} from "@/lib/activity-session";
import type { Conversation } from "@/lib/types";
import type { ActivityReaction } from "@/lib/luna-motion";
import { ActivityBar } from "./activity-bar";
import { ActivitySessionDiscovery } from "./activity-session-discovery";
import { ActivitySessionPanel, OfferCard } from "./activity-session-panel";

/** What the activity flow currently asks of the person, so the page can hide duplicate choices. */
export type ActivityPresence = "idle" | "offer" | "running" | "check-in";

type Props = {
  conversation: Conversation; disabled: boolean; onRefresh(): Promise<unknown>; onBusyChange(value: boolean): void;
  ordinaryMessages?: number;
  /** Slot above the composer for the running-activity bar. Without one, the bar renders inline. */
  dock?: HTMLElement | null;
  onPresenceChange?(presence: ActivityPresence): void;
  onReactionChange?(reaction: ActivityReaction): void;
  /** "Just talk" on Luna's offer: the same server preference as the chat-level choice. */
  onJustTalk?(): void;
};
type PendingOperation = { key: string; payload: string };

/** Saved activity limits, shown back to the person in their own terms on the offer. */
function constraintTags(conversation: Conversation): string[] {
  const limits = (conversation as Conversation & {
    activity_constraints?: { no_audio?: boolean; no_video?: boolean; seated?: boolean; avoid_breath_focus?: boolean } | null;
  }).activity_constraints;
  if (!limits) return [];
  return [
    limits.no_audio && "No audio", limits.no_video && "No video",
    limits.seated && "Seated", limits.avoid_breath_focus && "No breath focus",
  ].filter((tag): tag is string => Boolean(tag));
}

/** All authoritative state lives on the server. This component keeps only display clocks and retry receipts. */
export function ActivitySessionWorkspace({
  conversation, disabled, onRefresh, onBusyChange, ordinaryMessages = 0, dock = null, onPresenceChange, onJustTalk, onReactionChange,
}: Props) {
  const pausedByConversation = (conversation as Conversation & { activity_move?: string }).activity_move === "pause";
  const [session, setSession] = useState<ActivitySession | null>(null);
  useEffect(() => {
    onReactionChange?.({
      conversationId: conversation.id,
      status: session?.status,
      followUp: session?.follow_up_status,
      participation: session?.report?.participation,
      stateChange: session?.report?.state_change,
      messageId: session?.follow_up_message_id,
    });
  }, [conversation.id, session, onReactionChange]);
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
  const presenceCallback = useRef(onPresenceChange);
  const checkInRef = useRef<HTMLFormElement>(null);
  const searchRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    chat.current = conversation;
    busyCallback.current = onBusyChange;
    refreshCallback.current = onRefresh;
    presenceCallback.current = onPresenceChange;
  }, [conversation, onBusyChange, onRefresh, onPresenceChange]);

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
      presenceCallback.current?.("idle");
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
  const expiryKey = expiryPending && session && !disabled && !busy ? `${session.id}:${session.revision}` : null;
  const controlRef = useRef(control);
  controlRef.current = control;
  useEffect(() => {
    if (!expiryKey || expiryAttempt.current === expiryKey) return;
    expiryAttempt.current = expiryKey;
    // One deterministic question is rendered locally; the server receipt makes tabs/retries converge.
    void controlRef.current("expire");
  }, [expiryKey]);

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
    setSearchOpen(false);
  }

  function openSearch() {
    setSearchOpen(true);
    // The panel mounts on the next render; bring it into view for keyboard and screen-reader users.
    queueMicrotask(() => searchRef.current?.scrollIntoView?.({ block: "nearest" }));
  }

  function goToCheckIn() {
    const form = checkInRef.current;
    form?.scrollIntoView?.({ block: "center" });
    form?.querySelector<HTMLInputElement>("input[type=radio]")?.focus();
  }

  const running = Boolean(session && (session.status === "active" || session.status === "paused"));
  const waiting = Boolean(session && !disabled && !pausedByConversation && !sessionNeedsSync && (
    session.status === "awaiting_report" || expiryPending
    || (session.status === "stopped" && session.check_in_issued && Boolean(session.started_at) && !session.report)));
  const presence: ActivityPresence = waiting ? "check-in" : running ? "running"
    : showOffer || session?.status === "offered" ? "offer" : "idle";
  useEffect(() => { presenceCallback.current?.(presence); }, [presence]);

  // While a check-in is due, the form in the chat is the one place to answer; the bar only
  // remains for the brief moment the expiry is being confirmed.
  const bar = session && ((running && !waiting) || expiryPending) ? (
    <ActivityBar session={session} secondsLeft={secondsLeft}
      busy={busy || disabled || pausedByConversation || sessionNeedsSync}
      waiting={waiting} expiryPending={expiryPending}
      onCommand={(command) => void control(command)} onGoToCheckIn={goToCheckIn} />
  ) : null;
  // Search is one choice, offered when nothing else is on screen; an offer card has its own
  // "Something else", and a running activity has nothing to replace.
  const searchEntry = presence === "idle" && !searchOpen;

  return (
    <div className="stack">
      {loading && <p className="small muted" role="status">Syncing your activity…</p>}
      {showOffer && primary && <OfferCard title={primary.title} reason={card?.card_reason ?? primary.summary}
        minutes={primary.duration_minutes} tags={constraintTags(conversation)} url={primary.url} busy={busy || disabled || loading} canStart
        onStart={() => void saveOffer(primary.id, undefined, true)}
        onSomethingElse={canStart ? openSearch : undefined}
        dismiss={onJustTalk ? { label: "Just talk", onClick: onJustTalk } : {
          label: "Not now", onClick: () => void saveOffer(primary.id).then((saved) => { if (saved) void control("decline"); }),
        }} />}
      {session && <ActivitySessionPanel ref={checkInRef} session={session} secondsLeft={secondsLeft}
        busy={busy || disabled || pausedByConversation || sessionNeedsSync}
        suppressQuestions={pausedByConversation}
        canStart={canStart} onSomethingElse={canStart ? openSearch : undefined}
        expiryPending={expiryPending} onCommand={(command) => void control(command)}
        onReport={(report) => void saveReport(report)} onFollowUp={() => void followUp()} />}
      {searchEntry && <button className="chip" type="button" disabled={busy || disabled || !canStart} onClick={openSearch}>
        Find another resource
      </button>}
      {searchOpen && <div ref={searchRef} className="stack">
        <ActivitySessionDiscovery conversation={conversation} disabled={busy || disabled || !canStart} onSave={saveSearch} />
        <button className="btn btn-ghost" type="button" onClick={() => setSearchOpen(false)}>Close activity search</button>
      </div>}
      {error && <div className="stack"><p role="alert" className="note error">{error}</p>
        {retryOperation && <button className="btn btn-soft" type="button" disabled={busy || disabled}
          onClick={() => void retryOperation()}>Retry activity change</button>}
        <button className="btn btn-ghost" type="button" disabled={busy || disabled} onClick={() => void sync()}>Sync activity</button>
      </div>}
      {bar && (dock ? createPortal(bar, dock) : bar)}
    </div>
  );
}
