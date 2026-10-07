"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Fragment, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ActivityPreferences, activityPreferencesMessage } from "@/components/activity-preferences";
import { ActionTimer } from "@/components/action-timer";
import { type ActivityPresence, ActivitySessionWorkspace } from "@/components/activity-session-workspace";
import { Luna, type LunaMood } from "@/components/luna";
import { resolveLunaMood, type ActivityReaction } from "@/lib/luna-motion";
import { Icon } from "@/components/nav-icon";
import { ApiError, apiRequest } from "@/lib/api";
import { LUNA_REQUEST_TIMEOUT_MS } from "@/lib/request-deadlines";
import { currentActivityCard } from "@/lib/activity-card";
import { discoveryHref } from "@/lib/discovery";
import {
  canApplyConversation,
  chatStage,
  isCurrentChatRequest,
  journalSourceId,
  needsNewJournalChat,
  readOpenConversationId,
  readyForSomething,
  writeOpenConversationId,
} from "@/lib/conversation";
import {
  FEELINGS,
  GOALS,
  type GoalOption,
  MOODS,
  type Mood,
  actionEmoji,
  actionTone,
  goalSentence,
  moodByScore,
  selfReport,
} from "@/lib/feelings";
import { usePreferences } from "@/lib/preferences";
import { saveReminder } from "@/lib/reminders";
import { entryDateLabel } from "@/lib/journal";
import { greeting, useTimeOfDay } from "@/lib/time-of-day";
import { useStickToBottom, useVisualViewportHeight } from "@/lib/use-chat-scroll";
import type {
  ActivityConstraints,
  Conversation,
  ConversationDetail,
  ConversationMessage,
  ConversationTurn,
  JournalEntry,
  ReflectionRecord,
  Resource,
  SystemStatus,
} from "@/lib/types";

const CHAT_MESSAGE_LIMIT = 20;
const THINKING_LINES = ["Luna is thinking…", "Mulling it over…", "Finding the right words…"];
// A new time label appears only after a real pause in the conversation.
const TIME_GAP_MS = 20 * 60_000;

function timeLabel(value: string) {
  const at = new Date(value);
  const today = new Date();
  const sameDay = at.toDateString() === today.toDateString();
  const clock = at.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  return sameDay ? `Today · ${clock}` : `${at.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })} · ${clock}`;
}

type Busy = "send" | "accept" | "close" | null;
type Outgoing = { text: string; goal?: GoalOption["id"]; moodScore?: number; feelings?: string[]; constraints?: ActivityConstraints };
type PendingSend = { id: string; text: string; status: "sending" | "failed" };
type PendingCommand = { id: string; payload: string; outgoing: Outgoing };

function ChatWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const time = useTimeOfDay();
  const [preferences, , preferencesLoaded] = usePreferences();

  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState<Busy>(null);
  const [error, setError] = useState<string | null>(null);
  const [retryText, setRetryText] = useState<Outgoing | null>(null);
  const [ended, setEnded] = useState(false);
  const [step, setStep] = useState<"feelings" | "goal" | null>(null);
  const [feelings, setFeelings] = useState<string[]>([]);
  const [moodValence, setMoodValence] = useState<number | null>(null);
  const [selectedAction, setSelectedAction] = useState("");
  const [saved, setSaved] = useState<{ record: ReflectionRecord; resource: Resource | null } | null>(null);
  const [consent, setConsent] = useState<boolean | null>(null);
  const [retain, setRetain] = useState<boolean | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [privacyOpen, setPrivacyOpen] = useState(false);
  const [activityPreferencesOpen, setActivityPreferencesOpen] = useState(false);
  const [preferenceBusy, setPreferenceBusy] = useState(false);
  const [preferenceRetry, setPreferenceRetry] = useState<"listen" | "act" | null>(null);
  const [answering, setAnswering] = useState(false);
  const [thinkingLine, setThinkingLine] = useState(0);
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);
  const [loadedSource, setLoadedSource] = useState<Pick<JournalEntry, "id" | "created_at"> | null>(null);
  const [sourceFailure, setSourceFailure] = useState<{ id: string; message: string } | null>(null);
  const [guidedNotice, setGuidedNotice] = useState(false);
  const [aiAvailable, setAiAvailable] = useState<boolean | null>(null);
  const [resuming, setResuming] = useState(true);
  const [resumeFailure, setResumeFailure] = useState<string | null>(null);
  const [resumeAttempt, setResumeAttempt] = useState(0);
  const [pendingSend, setPendingSend] = useState<PendingSend | null>(null);
  const [activityBusy, setActivityBusy] = useState(false);
  const [activityReaction, setActivityReaction] = useState<ActivityReaction | null>(null);
  const [activityPresence, setActivityPresence] = useState<ActivityPresence>("idle");

  const pendingMessage = useRef<PendingCommand | null>(null);
  const pendingByChat = useRef(new Map<string, PendingCommand>());
  // UUID reuse must not transfer cached words to a replacement chat after
  // navigation has cleared latestConversation. No private data is persisted.
  const knownIncarnations = useRef(new Map<string, Conversation["incarnation_id"]>());
  const pendingCreation = useRef<{ id: string; payload: string } | null>(null);
  const resumeTarget = useRef<string | null | undefined>(undefined);
  const sendInFlight = useRef(false);
  const draftsByChat = useRef(new Map<string, string>());
  const acceptRequestId = useRef("");
  const loadedId = useRef<string | null>(null);
  // The activity bar renders into this slot above the composer, outside the scrolling log,
  // while its state stays in the single ActivitySessionWorkspace inside the log.
  const [activityDock, setActivityDock] = useState<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const menuTriggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const privacyRef = useRef<HTMLDivElement>(null);
  const privacyOpener = useRef<HTMLElement | null>(null);
  const latestConversation = useRef<Conversation | null>(null);
  const preferenceInFlight = useRef(false);
  const workspaceGeneration = useRef(0);
  const preferenceCommand = useRef<{
    client_request_id: string; expected_revision: number; preference: "listen" | "act";
  } | null>(null);

  const useAi = consent ?? preferences.llmConsent;
  const keepText = retain ?? preferences.retainText;
  const requestedChatId = searchParams.get("c");
  const requestedEntry = searchParams.get("entry");
  const requestedSourceId = journalSourceId(requestedEntry);
  const activeSourceId = conversation?.source_entry_id ?? selectedSourceId;
  const visibleSourceId = requestedSourceId ?? activeSourceId;
  const sourceEntry = loadedSource?.id === visibleSourceId ? loadedSource : null;
  const sourceError = requestedEntry && !requestedSourceId
    ? "This journal entry link is invalid. Open an entry from your journal."
    : sourceFailure?.id === visibleSourceId ? sourceFailure?.message : null;
  const sourceNeedsNewChat = visibleSourceId
    ? needsNewJournalChat(conversation, visibleSourceId) : false;
  const sourcePending = Boolean(requestedSourceId && !conversation && requestedSourceId !== selectedSourceId);
  const linkedSourceUnavailable = Boolean(activeSourceId && ((!conversation && !useAi) || aiAvailable === false));

  const clearUnavailableChat = useCallback((message: string, canReload: boolean) => {
    const previous = latestConversation.current;
    ++workspaceGeneration.current;
    sendInFlight.current = false;
    preferenceInFlight.current = false;
    preferenceCommand.current = null;
    pendingCreation.current = null;
    pendingMessage.current = null;
    const unavailableId = previous?.id ?? resumeTarget.current;
    if (unavailableId) {
      pendingByChat.current.delete(unavailableId);
      draftsByChat.current.delete(unavailableId);
      knownIncarnations.current.delete(unavailableId);
    }
    latestConversation.current = null;
    loadedId.current = null;
    writeOpenConversationId(null);
    setConversation(null); setMessages([]); setDraft(""); setPendingSend(null); setRetryText(null);
    setBusy(null); setPreferenceBusy(false); setPreferenceRetry(null); setActivityBusy(false);
    setActivityReaction(null); setActivityPresence("idle"); setStep(null); setSaved(null); setAnswering(false);
    setFeelings([]); setMoodValence(null); setSelectedAction(""); setSelectedSourceId(null);
    setMenuOpen(false); setPrivacyOpen(false); setActivityPreferencesOpen(false);
    setResuming(false); setEnded(!canReload);
    setError(canReload ? null : message); setResumeFailure(canReload ? message : null);
  }, []);

  const rejectReplacedChat = useCallback((next: Conversation) => {
    const current = latestConversation.current;
    const known = current?.id === next.id ? current.incarnation_id : knownIncarnations.current.get(next.id);
    const hasKnown = current?.id === next.id || knownIncarnations.current.has(next.id);
    if (!hasKnown || !(known || next.incarnation_id) || known === next.incarnation_id) return false;
    clearUnavailableChat("This chat was replaced elsewhere. Load it again to open its current version.", true);
    return true;
  }, [clearUnavailableChat]);

  /** Take the person's own report from the server, never Luna's suggestion, after a reload. */
  const applyServerState = useCallback((next: Conversation) => {
    if (rejectReplacedChat(next)) return false;
    if (!canApplyConversation(latestConversation.current, next)) return false;
    latestConversation.current = next;
    knownIncarnations.current.set(next.id, next.incarnation_id);
    setConversation(next);
    if (next.confirmed_feelings) setFeelings(next.confirmed_feelings);
    const mood = moodByScore(next.reported_mood);
    if (mood) setMoodValence(mood.valence);
    setSelectedAction(currentActivityCard(next)?.decision_preview.action_id ?? "");
    if (next.interaction_preference === "listen" || next.safety_mode === "support") setStep(null);
    return true;
  }, [rejectReplacedChat]);

  useEffect(() => () => {
    // Leaving this workspace must also invalidate sends, not only resume reads.
    workspaceGeneration.current += 1;
  }, []);

  useEffect(() => {
    if (!menuOpen) return;
    const items = Array.from(menuRef.current?.querySelectorAll<HTMLButtonElement>("[role=menuitem]") ?? []);
    let returnFocus: HTMLElement | null = menuTriggerRef.current;
    items[0]?.focus();
    function keydown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        setMenuOpen(false);
      } else if (event.key === "Tab") {
        event.preventDefault();
        const outside = Array.from(document.querySelectorAll<HTMLElement>(
          'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex="0"]',
        )).filter((element) => !menuRef.current?.contains(element));
        const position = outside.indexOf(menuTriggerRef.current!);
        returnFocus = outside[position + (event.shiftKey ? -1 : 1)] ?? menuTriggerRef.current;
        setMenuOpen(false);
      } else if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
        if (!menuRef.current?.contains(document.activeElement)) return;
        event.preventDefault();
        const current = items.indexOf(document.activeElement as HTMLButtonElement);
        const next = event.key === "Home" ? 0 : event.key === "End" ? items.length - 1
          : (current + (event.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
        items[next]?.focus();
      }
    }
    document.addEventListener("keydown", keydown);
    return () => {
      document.removeEventListener("keydown", keydown);
      if (returnFocus?.isConnected) returnFocus.focus();
    };
  }, [menuOpen]);

  useEffect(() => {
    if (!privacyOpen) return;
    const dialog = privacyRef.current;
    if (!dialog) return;
    const controls = () => Array.from(dialog.querySelectorAll<HTMLElement>(
      'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex="0"]',
    ));
    // aria-modal must match keyboard behavior: focus enters the sheet, stays
    // inside it, then returns to the control that opened it.
    controls()[0]?.focus();
    function keydown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        event.preventDefault();
        setPrivacyOpen(false);
      } else if (event.key === "Tab") {
        const items = controls();
        const first = items[0], last = items.at(-1);
        if (!first || !last) return;
        if (!dialog!.contains(document.activeElement)
          || (event.shiftKey && document.activeElement === first)
          || (!event.shiftKey && document.activeElement === last)) {
          event.preventDefault();
          (event.shiftKey ? last : first).focus();
        }
      }
    }
    document.addEventListener("keydown", keydown);
    return () => {
      document.removeEventListener("keydown", keydown);
      if (privacyOpener.current?.isConnected) privacyOpener.current.focus();
    };
  }, [privacyOpen]);

  useEffect(() => {
    let cancelled = false;
    apiRequest<SystemStatus>("/v1/system/status")
      .then((status) => { if (!cancelled) setAiAvailable(status.analysis_mode === "ai_configured"); })
      .catch(() => { if (!cancelled) setAiAvailable(null); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    let cancelled = false;
    if (requestedEntry && !requestedSourceId) return;
    if (!visibleSourceId) return;
    apiRequest<JournalEntry>(`/v1/journal/entries/${visibleSourceId}`)
      .then((entry) => {
        // Keep identity and date only. The original words remain on the journal page.
        if (!cancelled) {
          setLoadedSource({ id: entry.id, created_at: entry.created_at });
          setSourceFailure(null);
        }
      })
      .catch((reason) => {
        if (!cancelled) setSourceFailure({ id: visibleSourceId, message: reason instanceof ApiError && reason.status === 404
          ? "This journal entry is no longer available."
          : "We couldn’t load this journal entry. Open your journal and try again." });
      });
    return () => { cancelled = true; };
  }, [requestedEntry, requestedSourceId, visibleSourceId]);

  useEffect(() => {
    const requested = requestedChatId ?? readOpenConversationId();
    let cancelled = false;
    const switching = resumeTarget.current !== undefined && resumeTarget.current !== requested
      && requested !== latestConversation.current?.id;
    resumeTarget.current = requested;
    if (switching) {
      workspaceGeneration.current += 1;
      latestConversation.current = null;
      loadedId.current = null;
      pendingMessage.current = null;
      pendingCreation.current = null;
      acceptRequestId.current = "";
      sendInFlight.current = false;
      preferenceCommand.current = null;
      preferenceInFlight.current = false;
    }
    const generation = workspaceGeneration.current;
    queueMicrotask(() => {
      if (!cancelled && isCurrentChatRequest(generation, workspaceGeneration.current)) {
        if (switching) {
          setConversation(null);
          setMessages([]);
          // Restore only after the server confirms the cached chat incarnation.
          setDraft("");
          setBusy(null);
          setPendingSend(null);
          setRetryText(null);
          setStep(null);
          setSaved(null);
          setFeelings([]);
          setMoodValence(null);
          setSelectedAction("");
          setSelectedSourceId(null);
          setPreferenceBusy(false);
          setPreferenceRetry(null);
          setEnded(false);
          setError(null);
        }
        setResumeFailure(null);
        setResuming(Boolean(requested && loadedId.current !== requested));
      }
    });
    if (!requested || loadedId.current === requested) return () => { cancelled = true; };
    apiRequest<ConversationDetail>(`/v1/conversations/${requested}`)
      .then((detail) => {
        if (cancelled || !isCurrentChatRequest(generation, workspaceGeneration.current)) return;
        if (detail.conversation.status !== "open") {
          writeOpenConversationId(null);
          setEnded(true);
          setError("This chat has ended. Start a new chat to keep talking.");
          return;
        }
        loadedId.current = detail.conversation.id;
        writeOpenConversationId(detail.conversation.id);
        setEnded(false);
        if (applyServerState(detail.conversation)) {
          setMessages(detail.messages);
          setDraft(draftsByChat.current.get(detail.conversation.id) ?? "");
          const pending = pendingByChat.current.get(detail.conversation.id);
          if (pending) {
            if (detail.messages.some((message) => message.client_message_id === pending.id)) {
              pendingByChat.current.delete(detail.conversation.id);
              pendingMessage.current = null;
              setPendingSend(null);
              setRetryText(null);
            } else {
              // A submitted message is separate from an unsent draft. Recover its
              // original receipt only after checking whether the server saved it.
              pendingMessage.current = pending;
              setDraft(draftsByChat.current.get(detail.conversation.id) ?? pending.outgoing.text);
              setPendingSend({ id: pending.id, text: pending.outgoing.text, status: "failed" });
              setRetryText(pending.outgoing);
              setError("The earlier message’s delivery is not confirmed. You can retry it.");
            }
          }
        }
      })
      .catch((reason) => {
        if (!cancelled && isCurrentChatRequest(generation, workspaceGeneration.current)) {
          if (reason instanceof ApiError && reason.status === 404) {
            clearUnavailableChat("This chat is no longer available. Start a new chat to keep talking.", false);
          } else {
            // Losing a read is not evidence that the saved chat disappeared.
            setResumeFailure("We couldn’t load this chat. Try again to continue where you left off.");
          }
        }
      })
      .finally(() => {
        if (!cancelled && isCurrentChatRequest(generation, workspaceGeneration.current)) setResuming(false);
      });
    return () => {
      cancelled = true;
    };
  }, [requestedChatId, resumeAttempt, applyServerState, clearUnavailableChat]);

  useEffect(() => {
    if (busy !== "send") return;
    const interval = window.setInterval(() => setThinkingLine((value) => (value + 1) % THINKING_LINES.length), 2600);
    return () => window.clearInterval(interval);
  }, [busy]);

  const { logRef, contentRef: logContentRef, unseen, scrollToLatest } = useStickToBottom();
  const chatRef = useVisualViewportHeight();

  useEffect(() => {
    // Grow with the draft up to the CSS max-height, then scroll inside the field.
    const input = inputRef.current;
    if (!input) return;
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  }, [draft]);

  const userMessages = messages.filter((message) => message.role === "user").length;
  const stage = chatStage({
    saved: Boolean(saved),
    status: conversation?.status,
    safetyMode: conversation?.safety_mode,
    // AI activities remain in the open chat; the guided path keeps its legacy accept flow.
    hasCard: Boolean(currentActivityCard(conversation)) && conversation?.mode !== "ai",
    step,
  });
  const showReady =
    stage === "chat" && conversation?.mode !== "ai" &&
    busy === null &&
    readyForSomething({
      readyForAction: conversation?.ready_for_action, userMessages,
      preference: conversation?.interaction_preference,
    });

  const mood: LunaMood = useMemo(() => {
    const latest = messages.at(-1);
    const currentUser = messages.findLast((message) => message.role === "user");
    const activity = activityReaction?.conversationId === conversation?.id ? activityReaction : null;
    // A past report may remain in the activity workspace. Only the corresponding
    // delivered follow-up can use it to drive the current expression.
    const currentActivity = activity ? { ...activity,
      participation: latest?.id === activity.messageId ? activity.participation : undefined,
      stateChange: latest?.id === activity.messageId ? activity.stateChange : undefined,
    } : null;
    return resolveLunaMood({
      support: stage === "support", error: Boolean(error),
      waiting: busy === "send" || busy === "accept" || activityBusy,
      typing: Boolean(draft.trim()), closed: ended,
      move: (conversation as (Conversation & { activity_move?: string }) | null)?.activity_move,
      hasOffer: activityPresence === "offer", hasReply: answering || Boolean(latest?.role === "assistant"),
      replyFeelings: conversation?.feelings,
      confirmedFeelings: currentUser?.request_inputs?.goal
        ? currentUser.request_inputs.confirmed_feelings : undefined,
      openingMood: userMessages <= 1 ? conversation?.reported_mood : null,
      activity: currentActivity,
    });
  }, [activityBusy, activityPresence, activityReaction, answering, busy, conversation, draft, ended, error, messages, stage, userMessages]);

  const remember = useCallback(
    (next: Conversation) => {
      if (rejectReplacedChat(next)) return false;
      if (!canApplyConversation(latestConversation.current, next)) return false;
      latestConversation.current = next;
      knownIncarnations.current.set(next.id, next.incarnation_id);
      loadedId.current = next.id;
      writeOpenConversationId(next.id);
      if (searchParams.get("c") !== next.id) {
        const entry = requestedSourceId ?? next.source_entry_id;
        router.replace(`/talk?c=${next.id}${entry ? `&entry=${entry}` : ""}`);
      }
      setConversation(next);
      return true;
    },
    [router, searchParams, requestedSourceId, rejectReplacedChat],
  );

  async function refreshConversation(id: string, generation = workspaceGeneration.current): Promise<ConversationDetail | null> {
    try {
      const detail = await apiRequest<ConversationDetail>(`/v1/conversations/${id}`);
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return null;
      if (applyServerState(detail.conversation)) {
        setMessages(detail.messages);
        if (detail.conversation.status !== "open") {
          writeOpenConversationId(null);
          setEnded(true);
        }
        return detail;
      }
    } catch (reason) {
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return null;
      if (latestConversation.current?.id !== id) return null;
      if (reason instanceof ApiError && reason.status === 404) {
        clearUnavailableChat("This chat was deleted. Start a new chat when you’re ready.", false);
      } else {
        // A failed read is not evidence that the chat was closed or deleted.
        setError("We couldn’t refresh this chat. Your draft is still on this page.");
      }
    }
    return null;
  }

  async function ensureConversation(generation: number): Promise<Conversation> {
    if (conversation && conversation.status === "open") return conversation;
    const fields = {
      llm_consent: useAi, retain_text: keepText, locale: preferences.locale,
      ...(selectedSourceId ? { source_entry_id: selectedSourceId } : {}),
    };
    const payload = JSON.stringify(fields);
    if (pendingCreation.current?.payload !== payload) {
      pendingCreation.current = { id: crypto.randomUUID(), payload };
    }
    const created = await apiRequest<Conversation>("/v1/conversations", {
      method: "POST",
      retry: true,
      timeoutMs: LUNA_REQUEST_TIMEOUT_MS,
      body: JSON.stringify({
        // A lost first response must recover the same chat, not create another.
        client_request_id: pendingCreation.current.id,
        ...fields,
      }),
    });
    if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return created;
    pendingCreation.current = null;
    latestConversation.current = null;
    remember(created);
    setMessages([]);
    setEnded(false);
    return created;
  }

  async function send(outgoing: Outgoing) {
    const trimmed = outgoing.text.trim();
    if (!trimmed || userMessages >= CHAT_MESSAGE_LIMIT || busy || activityBusy || sendInFlight.current || preferenceInFlight.current || resuming || resumeFailure || sourcePending || sourceNeedsNewChat || linkedSourceUnavailable
      || (activeSourceId && (!sourceEntry || sourceError))) return;
    const generation = workspaceGeneration.current;
    const sendingComposerDraft = draft.trim() === trimmed;
    sendInFlight.current = true;
    setBusy("send");
    setError(null);
    setRetryText(null);
    scrollToLatest();
    const messageFields = {
      text: trimmed,
      ...(outgoing.goal ? { goal: outgoing.goal, confirmed_feelings: outgoing.feelings ?? [] } : {}),
      ...(outgoing.moodScore ? { mood_score: outgoing.moodScore } : {}),
      ...(outgoing.constraints ? { activity_constraints: outgoing.constraints } : {}),
    };
    const payload = JSON.stringify(messageFields);
    // Only an identical retry shares a receipt. Edited wording is a new message.
    const messageId = pendingMessage.current?.payload === payload ? pendingMessage.current.id : crypto.randomUUID();
    const editedRetry = Boolean(pendingMessage.current && pendingMessage.current.payload !== payload);
    const command = { id: messageId, payload, outgoing: { ...outgoing, text: trimmed } };
    if (!editedRetry) pendingMessage.current = command;
    setPendingSend({ id: messageId, text: trimmed, status: "sending" });
    let active = conversation?.status === "open" ? conversation : null;
    if (active && !editedRetry) {
      pendingByChat.current.set(active.id, command);
      if (sendingComposerDraft) draftsByChat.current.delete(active.id);
    }
    try {
      active = await ensureConversation(generation);
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
      if (editedRetry) {
        // Recover any earlier committed pair before sending changed wording as
        // a new message, so the transcript never quietly omits that prior turn.
        const recovered = await refreshConversation(active.id, generation);
        if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
        if (!recovered) throw new Error("We couldn’t confirm the earlier message. Your updated wording is still in the editor. Try again.");
      }
      pendingMessage.current = command;
      pendingByChat.current.set(active.id, command);
      if (sendingComposerDraft) draftsByChat.current.delete(active.id);
      const turn = await apiRequest<ConversationTurn>(`/v1/conversations/${active.id}/messages`, {
        method: "POST",
        retry: true,
        timeoutMs: LUNA_REQUEST_TIMEOUT_MS,
        body: JSON.stringify({
          client_message_id: messageId,
          ...(active.incarnation_id !== undefined ? { expected_incarnation_id: active.incarnation_id } : {}),
          ...messageFields,
        }),
      });
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
      if (!remember(turn.conversation)) {
        if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
        // The turn may have committed before the preference command but arrived
        // afterward. Reload its messages without replacing newer canonical state.
        pendingMessage.current = null;
        pendingByChat.current.delete(active.id);
        setPendingSend(null);
        // A successful HTTP turn contains the stored user message, so this draft
        // is already saved even when its conversation snapshot is superseded.
        if (sendingComposerDraft) setDraft("");
        await refreshConversation(active.id, generation);
        return;
      }
      pendingMessage.current = null;
      pendingByChat.current.delete(active.id);
      setPendingSend(null);
      // Retrying an older message must not discard newer unsent wording.
      if (sendingComposerDraft) {
        setDraft("");
        draftsByChat.current.delete(active.id);
      }
      // A refresh can discover a committed turn before its retry receipt arrives.
      setMessages((current) => {
        const ids = new Set(current.map((message) => message.id));
        return [...current, ...[turn.user_message, turn.assistant_message].filter((message) => !ids.has(message.id))];
      });
      applyServerState(turn.conversation);
      setAnswering(true);
      window.setTimeout(() => {
        if (isCurrentChatRequest(generation, workspaceGeneration.current)) setAnswering(false);
      }, 2600);
    } catch (reason) {
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
      const message = reason instanceof Error ? reason.message : "Luna couldn’t reply just now.";
      if (reason instanceof ApiError && (selectedSourceId || active?.source_entry_id)) {
        // Provider readiness can change after the page's status request. Keep the
        // explicit guided alternative available when the server refuses source AI.
        if (reason.status === 409 && message.includes("guided chat without")) setAiAvailable(false);
        if (reason.status === 404 && !active && selectedSourceId) {
          setSourceFailure({ id: selectedSourceId, message: "This journal entry is no longer available." });
        }
      }
      if (reason instanceof ApiError && (reason.status === 409 || reason.status === 404) && active) {
        // The chat moved on elsewhere (closed, deleted, or another reply landed first).
        // Nothing from this turn was saved; show the server's current state.
        pendingMessage.current = null;
        pendingByChat.current.delete(active.id);
        await refreshConversation(active.id, generation);
      } else if (active) {
        const recovered = await refreshConversation(active.id, generation);
        if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
        if (recovered?.messages.some((message) => message.client_message_id === messageId)) {
          // The reply was lost in transit, but the canonical read confirms save.
          pendingMessage.current = null;
          pendingByChat.current.delete(active.id);
          setPendingSend(null);
          if (sendingComposerDraft) {
            setDraft("");
            draftsByChat.current.delete(active.id);
          }
          setRetryText(null);
          setError(null);
          return;
        }
      }
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
      setRetryText(outgoing);
      setPendingSend({ id: messageId, text: trimmed, status: "failed" });
      setError(message);
    } finally {
      if (isCurrentChatRequest(generation, workspaceGeneration.current)) {
        sendInFlight.current = false;
        setBusy(null);
      }
    }
  }

  function tapMood(choice: Mood) {
    setMoodValence(choice.valence);
    void send({ text: choice.sentence, moodScore: choice.score });
  }

  function startCheck() {
    // Keep what the person already confirmed; only suggest Luna's guess the first time.
    if (conversation?.confirmed_feelings) setFeelings(conversation.confirmed_feelings);
    else if (conversation?.feelings?.length) setFeelings(conversation.feelings);
    setStep("feelings");
  }

  async function changePreference(value: "listen" | "act") {
    const active = latestConversation.current;
    if (!active || active.status !== "open" || preferenceInFlight.current) return;
    if (busy === "accept" || busy === "close") return;
    const generation = workspaceGeneration.current;
    preferenceInFlight.current = true;
    setPreferenceBusy(true);
    setPreferenceRetry(null);
    setError(null);
    if (preferenceCommand.current?.preference !== value) {
      preferenceCommand.current = {
        client_request_id: crypto.randomUUID(), expected_revision: active.revision ?? 0, preference: value,
      };
    }
    try {
      const next = await apiRequest<Conversation>(`/v1/conversations/${active.id}/preference`, {
        method: "POST", retry: true, body: JSON.stringify(preferenceCommand.current),
      });
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
      preferenceCommand.current = null;
      if (remember(next)) {
        applyServerState(next);
        acceptRequestId.current = "";
        if (next.mode !== "ai" && next.interaction_preference === "act" && next.status === "open" && next.safety_mode !== "support") {
          setFeelings(next.confirmed_feelings ?? next.feelings ?? []);
          setStep("feelings");
        } else {
          setStep(null);
          inputRef.current?.focus();
        }
      } else {
        await refreshConversation(active.id, generation);
      }
    } catch (reason) {
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
      if (reason instanceof ApiError && (reason.status === 409 || reason.status === 404)) {
        preferenceCommand.current = null;
        await refreshConversation(active.id, generation);
      } else {
        // Keep the same command ID/revision: a lost response may hide a successful
        // write. A retry must discover that receipt rather than apply a new choice.
        setPreferenceRetry(value);
      }
      setError(reason instanceof Error ? reason.message : "Your choice could not be confirmed.");
    } finally {
      if (isCurrentChatRequest(generation, workspaceGeneration.current)) {
        preferenceInFlight.current = false;
        setPreferenceBusy(false);
      }
    }
  }

  function toggleFeeling(id: string) {
    setFeelings((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id].slice(-6),
    );
  }

  function chooseGoal(goal: GoalOption) {
    setStep(null);
    void send({ text: goalSentence(feelings, goal), goal: goal.id, feelings });
  }

  async function accept() {
    if (!conversation?.card || !selectedAction || busy || preferenceInFlight.current) return;
    if (!acceptRequestId.current) acceptRequestId.current = crypto.randomUUID();
    const generation = workspaceGeneration.current;
    setBusy("accept");
    setError(null);
    try {
      const record = await apiRequest<ReflectionRecord>(`/v1/conversations/${conversation.id}/accept`, {
        method: "POST",
        retry: true,
        timeoutMs: LUNA_REQUEST_TIMEOUT_MS,
        body: JSON.stringify({
          client_request_id: acceptRequestId.current,
          action_id: selectedAction,
          expected_revision: conversation.revision,
          self_report: selfReport(feelings, moodValence),
        }),
      });
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
      const resource = conversation.card.actions.find((item) => item.id === record.decision.action_id) ?? null;
      saveReminder({
        decisionId: record.decision.decision_id,
        actionId: record.decision.action_id,
        actionTitle: resource?.title ?? "Your small step",
        // The original acceptance timestamp also keeps a retry from restarting the wait.
        dueAt: new Date(new Date(record.created_at).getTime() + preferences.followUpMinutes * 60_000).toISOString(),
      });
      writeOpenConversationId(null);
      loadedId.current = null;
      setSaved({ record, resource });
    } catch (reason) {
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
      setError(reason instanceof Error ? reason.message : "Your choice wasn’t saved. Please try again.");
      if (reason instanceof ApiError && (reason.status === 409 || reason.status === 404)) {
        acceptRequestId.current = "";
        await refreshConversation(conversation.id, generation);
      }
    } finally {
      if (isCurrentChatRequest(generation, workspaceGeneration.current)) setBusy(null);
    }
  }

  async function startSelectedChat(withSource: boolean, aiWithoutSource = false) {
    if (resuming || preferenceInFlight.current || busy === "accept" || busy === "close") return;
    if (withSource && (!sourceEntry || !useAi || aiAvailable === false)) return;
    const current = latestConversation.current;
    if ((current || draft.trim() || step || saved) && !window.confirm(
      "End the current chat and start a new one? Any unsent draft will be discarded. "
      + "The current chat follows its existing message retention choice.",
    )) return;
    const generation = ++workspaceGeneration.current;
    sendInFlight.current = false;
    setPendingSend(null);
    setBusy("close");
    try {
      if (current?.status === "open") {
        try {
          await apiRequest<Conversation>(`/v1/conversations/${current.id}/close`, { method: "POST", retry: true });
        } catch (reason) {
          // A deleted source also deletes its chat. There is no old flow left to close.
          if (!(reason instanceof ApiError && reason.status === 404)) throw reason;
        }
      }
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
      latestConversation.current = null;
      loadedId.current = null;
      pendingMessage.current = null;
      pendingCreation.current = null;
      sendInFlight.current = false;
      acceptRequestId.current = "";
      preferenceCommand.current = null;
      preferenceInFlight.current = false;
      writeOpenConversationId(null);
      setConversation(null);
      setMessages([]);
      setPendingSend(null);
      setDraft("");
      setStep(null);
      setFeelings([]);
      setMoodValence(null);
      setSelectedAction("");
      setSaved(null);
      setError(null);
      setRetryText(null);
      setPreferenceRetry(null);
      setPreferenceBusy(false);
      setEnded(false);
      setAnswering(false);
      setSelectedSourceId(withSource ? sourceEntry!.id : null);
      setGuidedNotice(!withSource && !aiWithoutSource);
      if (!withSource) setConsent(aiWithoutSource);
      router.replace(withSource ? `/talk?entry=${sourceEntry!.id}` : "/talk");
      inputRef.current?.focus();
    } catch (reason) {
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
      setError(reason instanceof Error ? reason.message : "The current chat could not be ended. Please try again.");
      if (current) await refreshConversation(current.id, generation);
    } finally {
      if (isCurrentChatRequest(generation, workspaceGeneration.current)) setBusy(null);
    }
  }

  function startOver() {
    writeOpenConversationId(null);
    // A full reload discards private drafts and in-memory retry receipts, even
    // when the fresh chat uses the same route and has no query string to change.
    // eslint-disable-next-line @next/next/no-location-assign-relative-destination
    window.location.assign("/talk");
  }

  async function endConversation(remove: boolean) {
    setMenuOpen(false);
    if (!conversation) return;
    if (remove && !window.confirm("Delete this chat? This can’t be undone.")) return;
    const generation = ++workspaceGeneration.current;
    sendInFlight.current = false;
    setPendingSend(null);
    preferenceInFlight.current = false;
    preferenceCommand.current = null;
    setPreferenceBusy(false);
    setPreferenceRetry(null);
    setBusy("close");
    setError(null);
    try {
      if (remove) {
        await apiRequest<void>(`/v1/conversations/${conversation.id}`, { method: "DELETE" });
      } else {
        await apiRequest<Conversation>(`/v1/conversations/${conversation.id}/close`, { method: "POST", retry: true });
      }
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
      writeOpenConversationId(null);
      loadedId.current = null;
      router.replace("/");
    } catch (reason) {
      if (!isCurrentChatRequest(generation, workspaceGeneration.current)) return;
      setError(reason instanceof Error ? reason.message : "That didn’t work. Please try again.");
      await refreshConversation(conversation.id, generation);
    } finally {
      if (isCurrentChatRequest(generation, workspaceGeneration.current)) setBusy(null);
    }
  }

  const status =
    stage === "support"
      ? "Here with you"
      : busy === "send"
        ? "Thinking…"
        : conversation?.mode === "guided"
          ? "Simple mode · no AI"
          : conversation
            ? "Private chat"
            : useAi
              ? "Private chat"
              : "Simple mode · no AI";

  const card = currentActivityCard(conversation);
  // Conditions that make typing pointless (limit, missing chat or source) disable the field.
  // A reply or activity change in progress only pauses sending.
  // An open entry chooser must be answered first: a message typed under it would go to the
  // current chat without the entry the screen is offering.
  const composerBlocked = userMessages >= CHAT_MESSAGE_LIMIT || resuming || Boolean(resumeFailure) || sourcePending
    || sourceNeedsNewChat || linkedSourceUnavailable || Boolean(activeSourceId && sourceError);
  const composerWaiting = busy !== null || activityBusy || preferenceBusy;
  const lastLunaIndex = messages.map((message) => message.role).lastIndexOf("assistant");

  return (
    <div className="chat" ref={chatRef}>
      <header className="chat-top">
        <Link className="icon-btn" href="/" aria-label="Back to home"><Icon name="back" /></Link>
        <div className="chat-who">
          {stage !== "welcome" && <Luna mood={mood} size={44} />}
          <span className="chat-who-text">
            <strong>Luna</strong>
            <small aria-live="polite">{status}</small>
          </span>
        </div>
        {conversation && stage !== "saved" ? (
          <button ref={menuTriggerRef} className="icon-btn" type="button" aria-label="Chat options" aria-haspopup="menu" aria-expanded={menuOpen} onClick={() => setMenuOpen((open) => !open)}>
            <Icon name="more" />
          </button>
        ) : <span />}
      </header>

      {menuOpen && (
        <div ref={menuRef} className="menu" role="menu" aria-label="Chat options">
          <button role="menuitem" type="button" onClick={() => { privacyOpener.current = menuTriggerRef.current; setMenuOpen(false); setPrivacyOpen(true); }}>How this chat is kept</button>
          {conversation?.mode === "ai" && conversation.safety_mode !== "support" && <button role="menuitem" type="button"
            disabled={busy !== null || activityBusy || userMessages >= CHAT_MESSAGE_LIMIT}
            onClick={() => { setMenuOpen(false); setActivityPreferencesOpen(true); }}>Activity preferences</button>}
          <button role="menuitem" type="button" onClick={() => void endConversation(false)}>End this chat</button>
          <button role="menuitem" type="button" className="danger" onClick={() => void endConversation(true)}>Delete this chat</button>
        </div>
      )}

      {activeSourceId && (
        <section className="source-strip" aria-label="Journal entry in this chat">
          <span className="source-strip-icon" aria-hidden="true"><Icon name="journal" /></span>
          <div className="source-strip-text" role="status">
            <strong>Using your selected journal entry{sourceEntry?.id === activeSourceId
              ? ` from ${entryDateLabel(sourceEntry.created_at).date}` : ""}</strong>
            <small>
              {sourceEntry?.id === activeSourceId ? `${entryDateLabel(sourceEntry.created_at).age} · ` : ""}
              Luna reads only this entry. How you feel now may be different.
            </small>
          </div>
          <div className="source-strip-actions">
            <Link className="link-btn" href={`/journal?entry=${activeSourceId}`}>Open entry</Link>
            <button type="button" className="link-btn" disabled={busy !== null || preferenceBusy}
              onClick={() => void startSelectedChat(false, true)}>New chat without it</button>
          </div>
        </section>
      )}

      <div className="chat-log" ref={logRef} role="log" aria-live="polite" aria-label="Chat with Luna">
        <div className="chat-log-content" ref={logContentRef}>
        {resumeFailure && <div className="chat-panel">
          <p className="note error" role="alert">{resumeFailure}</p>
          <button className="btn btn-soft" type="button" onClick={() => setResumeAttempt((value) => value + 1)}>Try loading chat again</button>
        </div>}
        {(sourcePending || sourceNeedsNewChat || (requestedEntry && sourceError)) && (
          <div className="chat-panel">
            <div className="card" style={{ display: "grid", gap: 12 }}>
              <h2>Talk about one journal entry</h2>
              {sourceEntry && <p>Selected entry from {entryDateLabel(sourceEntry.created_at).date} ({entryDateLabel(sourceEntry.created_at).age}).{" "}
                <Link href={`/journal?entry=${sourceEntry.id}`}>Open original entry</Link></p>}
              {sourceError ? <p className="note error" role="alert">{sourceError}</p>
                : !sourceEntry ? <p role="status">Loading the selected entry…</p>
                : <>
                  {sourceNeedsNewChat && <p>Your current chat has different context. Start a new chat to use this entry.</p>}
                  <p className="small muted">With your AI consent, only this entry and this chat are sent for replies.
                    Other journal entries stay outside the conversation.</p>
                  {!useAi && <p>AI help is off. You can keep the entry private and use guided prompts below.</p>}
                  {aiAvailable === false && <p>AI help is unavailable. Guided chat is still available; your entry will not be sent.</p>}
                  {!useAi && aiAvailable !== false && <button className="btn btn-soft" type="button"
                    onClick={() => setConsent(true)}>Allow AI for this new chat</button>}
                  <button className="btn btn-primary" type="button"
                    disabled={resuming || !useAi || aiAvailable === false || busy !== null || preferenceBusy}
                    onClick={() => void startSelectedChat(true)}>Use this entry in a new AI chat</button>
                </>}
              <button className="btn btn-soft" type="button" disabled={resuming || busy !== null || preferenceBusy}
                onClick={() => void startSelectedChat(false)}>Start guided chat without sending the entry</button>
              {conversation && <button className="btn btn-ghost" type="button" onClick={() => {
                router.replace(`/talk?c=${conversation.id}${conversation.source_entry_id ? `&entry=${conversation.source_entry_id}` : ""}`);
              }}>Keep current chat</button>}
            </div>
          </div>
        )}
        {activeSourceId && sourceError && !requestedEntry && (
          <div className="chat-panel"><p className="note error" role="alert">{sourceError}</p>
            <button className="btn btn-soft" type="button" disabled={busy !== null || preferenceBusy}
              onClick={() => void startSelectedChat(false)}>Start guided chat without the entry</button></div>
        )}
        {linkedSourceUnavailable && !sourcePending && !sourceNeedsNewChat && (
          <div className="chat-panel"><p className="note" role="status">
            AI help is off or unavailable. Your entry will not be sent for a reply.
            You can reflect on one detail using guided prompts.
          </p><button className="btn btn-soft" type="button" disabled={busy !== null || preferenceBusy}
            onClick={() => void startSelectedChat(false)}>Start guided chat without sending the entry</button></div>
        )}
        {guidedNotice && stage === "welcome" && (
          <div className="chat-panel"><p className="note" role="status">
            Guided chat uses no AI. Your journal entry is not sent. Think of one detail from your entry:
            what feels most important about it now? You can answer with a mood below or type in your own words.
          </p></div>
        )}
        {stage === "welcome" && (
          <div className="chat-welcome">
            <Luna mood={mood} size={104} />
            <h1 className="display">
              {time ? greeting(time) : "Hi there"}. How are you arriving?
            </h1>
            <p>Choose a mood or just type. There’s no wrong answer.</p>
            <div className="faces" role="group" aria-label="How are you feeling?" style={{ width: "100%", maxWidth: 420, marginTop: 8 }}>
              {MOODS.map((choice) => (
                <button key={choice.score} className="face" type="button" disabled={busy !== null || resuming || Boolean(resumeFailure) || sourcePending} aria-pressed={moodValence === choice.valence} onClick={() => tapMood(choice)}>
                  <span>{choice.label}</span>
                </button>
              ))}
            </div>
            {preferencesLoaded && (
              <button className="privacy-pill" type="button" onClick={(event) => { privacyOpener.current = event.currentTarget; setPrivacyOpen(true); }}>
                <Icon name="lock" />
                {useAi ? "AI help on" : "AI help off"} · {keepText ? "messages kept" : "messages cleared after"}
                <span className="sr-only">Change chat privacy</span>
              </button>
            )}
          </div>
        )}

        {messages.map((message, index) => {
          if (!message.content) return null;
          const previous = messages.slice(0, index).findLast((item) => item.content);
          // Time is context, not decoration: show it once, then only after a real pause.
          const showTime = !previous
            || Date.parse(message.created_at) - Date.parse(previous.created_at) > TIME_GAP_MS;
          // Luna's small mark opens each run of her replies instead of repeating on every line.
          const opensLunaRun = message.role === "assistant" && (showTime || previous?.role !== "assistant");
          return (
            <Fragment key={message.id}>
              {showTime && <time className="msg-time-divider" dateTime={message.created_at}>
                {timeLabel(message.created_at)}
              </time>}
              <div className={`msg ${message.role === "user" ? "from-me" : "from-luna"}${message.safety_mode === "support" && message.role === "assistant" ? " from-support" : ""}`}>
                {message.role === "assistant" && (
                  <span className="msg-avatar">{opensLunaRun
                    && <Luna mood={index === lastLunaIndex ? mood : "idle"} size={30} decorative />}</span>
                )}
                <div className="message-content"><div className="bubble">
                  <span className="sr-only">{message.role === "user" ? "You said: " : "Luna said: "}</span>
                  {message.content}
                </div></div>
              </div>
            </Fragment>
          );
        })}

        {pendingSend && !messages.some((message) => message.client_message_id === pendingSend.id) && (
          <div className="msg from-me" data-message-status={pendingSend.status}>
            <div className="bubble">
              <span className="sr-only">You said: </span>{pendingSend.text}
              <small className="small muted" style={{ display: "block" }}>
                {pendingSend.status === "sending" ? "Sending…" : ended
                  ? "Delivery not confirmed. Start a new chat to continue."
                  : "Delivery not confirmed. You can retry."}
              </small>
            </div>
          </div>
        )}

        {busy === "send" && (
          <div className="msg from-luna is-thinking" role="status">
            <span className="msg-avatar"><Luna mood="thinking" size={30} decorative /></span>
            <span className="sr-only">Luna is replying</span>
            <span className="thinking-line" aria-hidden="true">
              <span className="typing"><i /><i /><i /></span>
              <span className="typing-note">{THINKING_LINES[thinkingLine]}</span>
            </span>
          </div>
        )}

        {conversation?.mode === "ai" && conversation.status === "open" && conversation.safety_mode !== "support"
          && (conversation as Conversation & { activity_move?: string }).activity_move !== "pause"
          && conversation.interaction_preference !== "listen" && !sourcePending && !sourceNeedsNewChat
          && !linkedSourceUnavailable && !sourceError && (
          <div className="chat-panel">
            <ActivitySessionWorkspace key={conversation.id} conversation={conversation}
              ordinaryMessages={userMessages}
              disabled={busy !== null || preferenceBusy || resuming || Boolean(resumeFailure)}
              onBusyChange={setActivityBusy}
              onRefresh={() => refreshConversation(conversation.id)}
              dock={activityDock} onPresenceChange={setActivityPresence}
              onReactionChange={setActivityReaction}
              onJustTalk={() => void changePreference("listen")} />
          </div>
        )}

        {/* An offer card carries its own Just talk / Something else, and a started activity
            keeps the chat open without competing choices, so chat-level chips step aside. */}
        {conversation?.status === "open" && stage !== "support" && stage !== "saved"
          && (conversation.mode !== "ai" || activityPresence === "idle" || conversation.interaction_preference === "listen") && (
          <div className="chat-panel chat-choices">
            {conversation.interaction_preference === "listen" ? (
              <>
                <p className="note" role="status">Just talking. We’ll stay with your thoughts.</p>
                <button className="chip" type="button" disabled={busy !== null || preferenceBusy}
                  onClick={() => void changePreference("act")}>Find a small step</button>
              </>
            ) : (
              <button className="chip" type="button"
                disabled={preferenceBusy || busy === "accept" || busy === "close"}
                onClick={() => void changePreference("listen")}>Just talk</button>
            )}
            {preferenceBusy && <p className="note" role="status">Saving your choice…</p>}
            {/* AI chats search inline, inside the activity flow; the separate library stays on Home. */}
            {stage === "chat" && conversation.mode !== "ai" && <Link className="chip" href={discoveryHref()}>Find resources</Link>}
          </div>
        )}

        {conversation?.status === "open" && userMessages >= CHAT_MESSAGE_LIMIT && (
          <div className="chat-panel"><p className="note" role="status">This chat has reached its 20-message limit.
            You can still finish a started activity and save its check-in.</p>
            <button className="btn btn-soft" type="button" onClick={startOver}>Start a new chat</button></div>
        )}

        {showReady && !preferenceBusy && (
          <div className="chat-panel">
            <div className="chips">
              <button className="chip" type="button" onClick={startCheck}><Icon name="leaf" />Yes, let’s find one small thing</button>
            </div>
          </div>
        )}

        {stage === "feelings" && (
          <>
            <div className="msg from-luna">
              <span className="msg-avatar"><Luna mood="listening" size={34} decorative /></span>
              <div className="bubble">Before I suggest anything, which of these feel true right now? Pick as many as you like.</div>
            </div>
            <div className="chat-panel">
              <div className="chips" role="group" aria-label="Feelings">
                {FEELINGS.map((item) => (
                  <button key={item.id} className="chip" type="button" aria-pressed={feelings.includes(item.id)} onClick={() => toggleFeeling(item.id)}>
                    {item.label}
                  </button>
                ))}
              </div>
              <div className="row">
                <button className="btn btn-primary" type="button" onClick={() => setStep("goal")}>{feelings.length ? "That’s it" : "I’m not sure"}</button>
                <button className="btn btn-ghost" type="button" disabled={preferenceBusy}
                  onClick={() => void changePreference("listen")}>Back to chatting</button>
              </div>
            </div>
          </>
        )}

        {stage === "goal" && (
          <>
            <div className="msg from-luna">
              <span className="msg-avatar"><Luna mood="answering" size={34} decorative /></span>
              <div className="bubble">Thanks for telling me. What would help most right now?</div>
            </div>
            <div className="chat-panel">
              <div className="stack" role="group" aria-label="What would help">
                {GOALS.map((goal) => (
                  <button key={goal.id} className="chip goal" type="button" onClick={() => chooseGoal(goal)}>
                    {goal.label}
                  </button>
                ))}
              </div>
              <button className="btn btn-ghost" type="button" onClick={() => setStep("feelings")}>Back</button>
            </div>
          </>
        )}

        {stage === "offer" && card && (
          <div className="chat-panel">
            <p className="small muted">Saved resource collection</p>
            <div className="actions" role="group" aria-label="Small things to try">
              {card.actions.map((item) => (
                <button key={item.id} className="action" type="button" aria-pressed={selectedAction === item.id} onClick={() => setSelectedAction(item.id)}>
                  <span className={`action-icon ${actionTone(item.coping_style, item.resource_type)}`} aria-hidden="true">{actionEmoji(item.coping_style, item.resource_type)}</span>
                  <span>
                    {item.id === card.decision_preview.action_id && <span className="pick-badge">Luna’s pick</span>}
                    <strong>{item.title}</strong>
                    <small>{item.duration_minutes ? `${item.duration_minutes} min · ` : ""}{item.provider}</small>
                  </span>
                  <span className="radio-dot" aria-hidden="true" />
                </button>
              ))}
            </div>
            <Link className="btn btn-soft" href={discoveryHref(card.goal)}>Search for other resources</Link>
            <div className="row">
              <button className="btn btn-primary btn-big" type="button"
                disabled={!selectedAction || busy !== null || preferenceBusy} onClick={accept}>
                {busy === "accept" ? "Saving…" : "Let’s try it"}
              </button>
              <button className="btn btn-ghost" type="button" onClick={() => setStep("goal")}>Show me other ideas</button>
            </div>
          </div>
        )}

        {stage === "support" && (
          <div className="chat-panel">
            <div className="crisis">
              <strong>You don’t have to handle this alone.</strong>
              <p>If you might act on thoughts of hurting yourself, or you’re in danger, please reach a person now.</p>
              <div className="row">
                <a className="btn btn-primary" href="https://988.ca/" target="_blank" rel="noreferrer">Call or text 9-8-8</a>
                <a className="btn btn-soft" href="tel:911">Emergency: 911</a>
              </div>
            </div>
            {card?.actions.filter((item) => item.url).map((item) => (
              <a key={item.id} className="action" href={item.url} target="_blank" rel="noreferrer">
                <span className="action-icon support" aria-hidden="true">🏮</span>
                <span><strong>{item.title}</strong><small>{item.provider}</small></span>
                <span aria-hidden="true">↗</span>
              </a>
            ))}
          </div>
        )}

        {stage === "saved" && saved && (
          <div className="chat-panel">
            <div className="card center">
              <Luna mood="proud" size={110} />
              <h2>Nice choice.</h2>
              <p className="muted">{saved.resource?.title ?? "Your small step is saved."}</p>
              <p className="small muted">Saved resource collection</p>
              {saved.resource?.url && (
                <a className="btn btn-primary" href={saved.resource.url} target="_blank" rel="noreferrer">
                  Open the {saved.resource.resource_type}
                </a>
              )}
              <ActionTimer minutes={saved.resource?.duration_minutes ?? 5} />
              <p className="small muted">I’ll check in with you in about {preferences.followUpMinutes} minutes. You’ll find it on Home.</p>
              <div className="row" style={{ justifyContent: "center" }}>
                <Link className="btn btn-soft" href={discoveryHref(saved.record.target.goal)}>Search for other resources</Link>
                <Link className="btn btn-soft" href={`/check-in?decision=${saved.record.decision.decision_id}`}>I’m done, check in now</Link>
                <Link className="btn btn-ghost" href="/">Back home</Link>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="chat-panel">
            <p className="note error" role="alert">{error}</p>
            {preferenceRetry && (
              <button className="btn btn-soft" type="button" disabled={preferenceBusy}
                onClick={() => void changePreference(preferenceRetry)}>Try saving choice again</button>
            )}
            {ended ? (
              <button className="btn btn-primary" type="button" onClick={startOver}>Start a new chat</button>
            ) : retryText && (
              <button className="btn btn-soft" type="button" onClick={() => void send(retryText)}>
                {draft.trim() && draft.trim() !== retryText.text.trim() ? "Retry original message" : "Try again"}
              </button>
            )}
          </div>
        )}
        </div>
      </div>

      <div className="chat-dock">
        {unseen && (
          <button className="jump-latest" type="button" onClick={() => scrollToLatest(true)}>
            New messages <span aria-hidden="true">↓</span>
          </button>
        )}
        <div ref={setActivityDock} />
      </div>

      {(stage === "welcome" || stage === "chat") && (
        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void send({ text: draft });
          }}
        >
          <label className="sr-only" htmlFor="chat-input">Message Luna</label>
          <textarea
            id="chat-input"
            ref={inputRef}
            rows={1}
            maxLength={2000}
            value={draft}
            disabled={composerBlocked}
            readOnly={composerWaiting}
            aria-busy={composerWaiting}
            placeholder="What's on your mind?"
            onChange={(event) => {
              setDraft(event.target.value);
              if (conversation) draftsByChat.current.set(conversation.id, event.target.value);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing && event.nativeEvent.keyCode !== 229) {
                event.preventDefault();
                void send({ text: draft });
              }
            }}
          />
          <button className="send-btn" type="submit" aria-label="Send"
            disabled={composerBlocked || composerWaiting || !draft.trim()}>
            <Icon name="send" />
          </button>
        </form>
      )}
      {stage === "chat" && conversation?.mode !== "ai" && conversation?.interaction_preference !== "listen"
        && userMessages >= 1 && !showReady && busy === null && !preferenceBusy && (
        <div className="chat-hint" style={{ paddingBottom: "calc(10px + env(safe-area-inset-bottom))" }}>
          <button className="link-btn" type="button" onClick={startCheck}>Skip ahead: find one small thing to try</button>
        </div>
      )}

      {activityPreferencesOpen && conversation?.status === "open" && conversation.safety_mode !== "support" && (
        <ActivityPreferences key={conversation.id} current={conversation.activity_constraints}
          busy={busy !== null || activityBusy} opener={menuTriggerRef}
          onClose={() => setActivityPreferencesOpen(false)}
          onApply={(constraints) => {
            setActivityPreferencesOpen(false);
            void send({ text: activityPreferencesMessage(constraints), constraints });
          }} />
      )}
      {privacyOpen && (
        <div className="sheet-backdrop" onClick={() => setPrivacyOpen(false)}>
          <div ref={privacyRef} className="sheet" role="dialog" aria-modal="true" aria-labelledby="privacy-title" onClick={(event) => event.stopPropagation()}>
            <h2 id="privacy-title">How this chat is kept</h2>
            {conversation ? (
              <p className="muted">
                This chat uses {conversation.mode === "guided" ? "simple mode, with no AI" : "private AI help"}. Its messages
                are {conversation.retain_text ? "kept after it ends" : "cleared when it ends"}. A short summary and your choice are
                saved either way. You can change the defaults for new chats in Settings.
              </p>
            ) : (
              <>
                <div className="settings">
                  <label className="setting">
                    <span><strong>Let Luna use AI</strong><small>Uses AI for replies, with zero data retention required from the provider.</small></span>
                    <span className="switch"><input type="checkbox" checked={useAi} onChange={(event) => setConsent(event.target.checked)} /><span /></span>
                  </label>
                  <label className="setting">
                    <span><strong>Keep my messages</strong><small>Off clears the words when the chat ends.</small></span>
                    <span className="switch"><input type="checkbox" checked={keepText} onChange={(event) => setRetain(event.target.checked)} /><span /></span>
                  </label>
                </div>
                <p className="small muted">Without AI, Luna asks a few simple questions instead. A short summary and your choice are saved either way.</p>
              </>
            )}
            <button className="btn btn-primary" type="button" onClick={() => setPrivacyOpen(false)}>Done</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default function TalkPage() {
  return (
    <Suspense fallback={<div className="loading-luna"><Luna mood="idle" size={90} decorative /></div>}>
      <ChatWorkspace />
    </Suspense>
  );
}
