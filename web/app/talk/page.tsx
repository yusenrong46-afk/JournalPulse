"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { ActionTimer } from "@/components/action-timer";
import { Luna, type LunaMood } from "@/components/luna";
import { Icon } from "@/components/nav-icon";
import { apiRequest } from "@/lib/api";
import {
  chatStage,
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
  selfReport,
} from "@/lib/feelings";
import { usePreferences } from "@/lib/preferences";
import { saveReminder } from "@/lib/reminders";
import { greeting, useTimeOfDay } from "@/lib/time-of-day";
import type {
  Conversation,
  ConversationDetail,
  ConversationMessage,
  ConversationTurn,
  ReflectionRecord,
  Resource,
} from "@/lib/types";

const CHAT_TIMEOUT_MS = 60_000;
const THINKING_LINES = ["Luna is thinking…", "Mulling it over…", "Finding the right words…"];

type Busy = "send" | "accept" | "close" | null;

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
  const [retryText, setRetryText] = useState<{ text: string; goal?: GoalOption["id"] } | null>(null);
  const [step, setStep] = useState<"feelings" | "goal" | null>(null);
  const [feelings, setFeelings] = useState<string[]>([]);
  const [moodValence, setMoodValence] = useState<number | null>(null);
  const [selectedAction, setSelectedAction] = useState("");
  const [saved, setSaved] = useState<{ record: ReflectionRecord; resource: Resource | null } | null>(null);
  const [consent, setConsent] = useState<boolean | null>(null);
  const [retain, setRetain] = useState<boolean | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [privacyOpen, setPrivacyOpen] = useState(false);
  const [dismissedReady, setDismissedReady] = useState(0);
  const [answering, setAnswering] = useState(false);
  const [thinkingLine, setThinkingLine] = useState(0);

  const pendingMessageId = useRef<string | null>(null);
  const acceptRequestId = useRef("");
  const loadedId = useRef<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const useAi = consent ?? preferences.llmConsent;
  const keepText = retain ?? preferences.retainText;

  useEffect(() => {
    const requested = searchParams.get("c") ?? readOpenConversationId(window.localStorage);
    if (!requested || loadedId.current === requested) return;
    let cancelled = false;
    apiRequest<ConversationDetail>(`/v1/conversations/${requested}`)
      .then((detail) => {
        if (cancelled) return;
        if (detail.conversation.status !== "open") {
          writeOpenConversationId(window.localStorage, null);
          return;
        }
        loadedId.current = detail.conversation.id;
        writeOpenConversationId(window.localStorage, detail.conversation.id);
        setConversation(detail.conversation);
        setMessages(detail.messages);
        setFeelings(detail.conversation.feelings ?? []);
        if (detail.conversation.card) setSelectedAction(detail.conversation.card.decision_preview.action_id);
      })
      .catch(() => {
        if (!cancelled) writeOpenConversationId(window.localStorage, null);
      });
    return () => {
      cancelled = true;
    };
  }, [searchParams]);

  useEffect(() => {
    if (busy !== "send") return;
    const interval = window.setInterval(() => setThinkingLine((value) => (value + 1) % THINKING_LINES.length), 2600);
    return () => window.clearInterval(interval);
  }, [busy]);

  useEffect(() => {
    const log = logRef.current;
    if (log) log.scrollTop = log.scrollHeight;
  }, [messages, step, busy, error, saved]);

  const userMessages = messages.filter((message) => message.role === "user").length;
  const stage = chatStage({
    saved: Boolean(saved),
    status: conversation?.status,
    safetyMode: conversation?.safety_mode,
    hasCard: Boolean(conversation?.card),
    step,
  });
  const showReady =
    stage === "chat" &&
    busy === null &&
    dismissedReady < userMessages &&
    readyForSomething({ readyForAction: conversation?.ready_for_action, userMessages });

  const mood: LunaMood = useMemo(() => {
    if (stage === "support") return "support";
    if (error) return "oops";
    if (busy === "send" || busy === "accept") return "thinking";
    if (stage === "saved") return "proud";
    if (answering) return "answering";
    if (draft.trim()) return "listening";
    return "idle";
  }, [answering, busy, draft, error, stage]);

  const remember = useCallback(
    (next: Conversation) => {
      loadedId.current = next.id;
      writeOpenConversationId(window.localStorage, next.id);
      if (searchParams.get("c") !== next.id) router.replace(`/talk?c=${next.id}`);
      setConversation(next);
    },
    [router, searchParams],
  );

  async function ensureConversation(): Promise<Conversation> {
    if (conversation && conversation.status === "open") return conversation;
    const created = await apiRequest<Conversation>("/v1/conversations", {
      method: "POST",
      retry: true,
      timeoutMs: CHAT_TIMEOUT_MS,
      body: JSON.stringify({
        client_request_id: crypto.randomUUID(),
        llm_consent: useAi,
        retain_text: keepText,
        locale: preferences.locale,
      }),
    });
    remember(created);
    setMessages([]);
    return created;
  }

  async function send(text: string, goal?: GoalOption["id"]) {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    setBusy("send");
    setError(null);
    setRetryText(null);
    const messageId = pendingMessageId.current ?? crypto.randomUUID();
    pendingMessageId.current = messageId;
    try {
      const active = await ensureConversation();
      const turn = await apiRequest<ConversationTurn>(`/v1/conversations/${active.id}/messages`, {
        method: "POST",
        retry: true,
        timeoutMs: CHAT_TIMEOUT_MS,
        body: JSON.stringify({ client_message_id: messageId, text: trimmed, ...(goal ? { goal } : {}) }),
      });
      pendingMessageId.current = null;
      setDraft("");
      setMessages((current) => [...current, turn.user_message, turn.assistant_message]);
      remember(turn.conversation);
      if (turn.conversation.card) setSelectedAction(turn.conversation.card.decision_preview.action_id);
      setAnswering(true);
      window.setTimeout(() => setAnswering(false), 2600);
    } catch (reason) {
      setRetryText({ text: trimmed, goal });
      setError(reason instanceof Error ? reason.message : "Luna couldn’t reply just now.");
    } finally {
      setBusy(null);
    }
  }

  function tapMood(choice: Mood) {
    setMoodValence(choice.valence);
    void send(choice.sentence);
  }

  function startCheck() {
    setFeelings(conversation?.feelings?.length ? conversation.feelings : feelings);
    setStep("feelings");
  }

  function toggleFeeling(id: string) {
    setFeelings((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id].slice(-6),
    );
  }

  function chooseGoal(goal: GoalOption) {
    setStep(null);
    void send(goalSentence(feelings, goal), goal.id);
  }

  async function accept() {
    if (!conversation?.card || !selectedAction || busy) return;
    if (!acceptRequestId.current) acceptRequestId.current = crypto.randomUUID();
    setBusy("accept");
    setError(null);
    try {
      const record = await apiRequest<ReflectionRecord>(`/v1/conversations/${conversation.id}/accept`, {
        method: "POST",
        retry: true,
        timeoutMs: CHAT_TIMEOUT_MS,
        body: JSON.stringify({
          client_request_id: acceptRequestId.current,
          action_id: selectedAction,
          self_report: selfReport(feelings, moodValence),
        }),
      });
      const resource = conversation.card.actions.find((item) => item.id === record.decision.action_id) ?? null;
      saveReminder({
        decisionId: record.decision.decision_id,
        actionId: record.decision.action_id,
        actionTitle: resource?.title ?? "Your small step",
        dueAt: new Date(Date.now() + preferences.followUpMinutes * 60_000).toISOString(),
      });
      writeOpenConversationId(window.localStorage, null);
      loadedId.current = null;
      setSaved({ record, resource });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Your choice wasn’t saved. Please try again.");
    } finally {
      setBusy(null);
    }
  }

  async function endConversation(remove: boolean) {
    setMenuOpen(false);
    if (!conversation) return;
    if (remove && !window.confirm("Delete this chat? This can’t be undone.")) return;
    setBusy("close");
    setError(null);
    try {
      if (remove) {
        await apiRequest<void>(`/v1/conversations/${conversation.id}`, { method: "DELETE" });
      } else {
        await apiRequest<Conversation>(`/v1/conversations/${conversation.id}/close`, { method: "POST", retry: true });
      }
      writeOpenConversationId(window.localStorage, null);
      loadedId.current = null;
      router.replace("/");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "That didn’t work. Please try again.");
    } finally {
      setBusy(null);
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

  const card = conversation?.card ?? null;
  const lastLunaIndex = messages.map((message) => message.role).lastIndexOf("assistant");

  return (
    <div className="chat">
      <header className="chat-top">
        <Link className="icon-btn" href="/" aria-label="Back to home"><Icon name="back" /></Link>
        <div className="chat-who">
          {stage !== "welcome" && <Luna mood={mood} size={80} />}
          <strong>Luna</strong>
          <small aria-live="polite">{status}</small>
        </div>
        {conversation && stage !== "saved" ? (
          <button className="icon-btn" type="button" aria-label="Chat options" aria-expanded={menuOpen} onClick={() => setMenuOpen((open) => !open)}>
            <Icon name="more" />
          </button>
        ) : <span />}
      </header>

      {menuOpen && (
        <div className="menu" role="menu">
          <button role="menuitem" type="button" onClick={() => { setMenuOpen(false); setPrivacyOpen(true); }}>How this chat is kept</button>
          <button role="menuitem" type="button" onClick={() => void endConversation(false)}>End this chat</button>
          <button role="menuitem" type="button" className="danger" onClick={() => void endConversation(true)}>Delete this chat</button>
        </div>
      )}

      <div className="chat-log" ref={logRef} role="log" aria-live="polite" aria-label="Chat with Luna">
        {stage === "welcome" && (
          <div className="chat-welcome">
            <Luna mood={mood} size={150} />
            <h1 className="display" style={{ fontSize: "1.7rem" }}>
              {time ? greeting(time) : "Hi there"}. How are you arriving?
            </h1>
            <p>Tap a face or just type. There’s no wrong answer.</p>
            <div className="faces" role="group" aria-label="How are you feeling?" style={{ width: "100%", maxWidth: 420, marginTop: 8 }}>
              {MOODS.map((choice) => (
                <button key={choice.score} className="face" type="button" disabled={busy !== null} aria-pressed={moodValence === choice.valence} onClick={() => tapMood(choice)}>
                  <span aria-hidden="true">{choice.emoji}</span>
                  <span>{choice.label}</span>
                </button>
              ))}
            </div>
            {preferencesLoaded && (
              <button className="privacy-pill" type="button" onClick={() => setPrivacyOpen(true)}>
                <span aria-hidden="true">🔒</span>
                {useAi ? "AI help on" : "AI help off"} · {keepText ? "messages kept" : "messages cleared after"}
                <span className="sr-only">Change chat privacy</span>
              </button>
            )}
          </div>
        )}

        {messages.map((message, index) =>
          message.content ? (
            <div key={message.id} className={`msg ${message.role === "user" ? "from-me" : "from-luna"}${message.safety_mode === "support" && message.role === "assistant" ? " from-support" : ""}`}>
              {message.role === "assistant" && (
                <span className="msg-avatar"><Luna mood={index === lastLunaIndex && stage === "support" ? "support" : "idle"} size={34} decorative /></span>
              )}
              <div className="bubble">
                <span className="sr-only">{message.role === "user" ? "You said: " : "Luna said: "}</span>
                {message.content}
              </div>
            </div>
          ) : null,
        )}

        {busy === "send" && (
          <div className="msg from-luna" aria-label={THINKING_LINES[thinkingLine]}>
            <span className="msg-avatar"><Luna mood="thinking" size={34} decorative /></span>
            <div className="typing" aria-hidden="true"><i /><i /><i /></div>
          </div>
        )}
        {busy === "send" && <div className="typing-note" aria-hidden="true">{THINKING_LINES[thinkingLine]}</div>}

        {showReady && (
          <div className="chat-panel">
            <div className="chips">
              <button className="chip" type="button" onClick={startCheck}><span className="chip-emoji" aria-hidden="true">✨</span>Yes, let’s find one small thing</button>
              <button className="chip" type="button" onClick={() => { setDismissedReady(userMessages); inputRef.current?.focus(); }}>Keep talking</button>
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
                    <span className="chip-emoji" aria-hidden="true">{item.emoji}</span>{item.label}
                  </button>
                ))}
              </div>
              <div className="row">
                <button className="btn btn-primary" type="button" onClick={() => setStep("goal")}>{feelings.length ? "That’s it" : "I’m not sure"}</button>
                <button className="btn btn-ghost" type="button" onClick={() => { setStep(null); setDismissedReady(userMessages); }}>Back to chatting</button>
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
                    <span className="chip-emoji" aria-hidden="true">{goal.emoji}</span>{goal.label}
                  </button>
                ))}
              </div>
              <button className="btn btn-ghost" type="button" onClick={() => setStep("feelings")}>Back</button>
            </div>
          </>
        )}

        {stage === "offer" && card && (
          <div className="chat-panel">
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
            <div className="row">
              <button className="btn btn-primary btn-big" type="button" disabled={!selectedAction || busy !== null} onClick={accept}>
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
              {saved.resource?.url && (
                <a className="btn btn-primary" href={saved.resource.url} target="_blank" rel="noreferrer">
                  Open the {saved.resource.resource_type}
                </a>
              )}
              <ActionTimer minutes={saved.resource?.duration_minutes ?? 5} />
              <p className="small muted">I’ll check in with you in about {preferences.followUpMinutes} minutes. You’ll find it on Home.</p>
              <div className="row" style={{ justifyContent: "center" }}>
                <Link className="btn btn-soft" href={`/check-in?decision=${saved.record.decision.decision_id}`}>I’m done, check in now</Link>
                <Link className="btn btn-ghost" href="/">Back home</Link>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className="chat-panel">
            <p className="note error" role="alert">{error}</p>
            {retryText && (
              <button className="btn btn-soft" type="button" onClick={() => void send(retryText.text, retryText.goal)}>Try again</button>
            )}
          </div>
        )}
      </div>

      {(stage === "welcome" || stage === "chat") && (
        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void send(draft);
          }}
        >
          <label className="sr-only" htmlFor="chat-input">Message Luna</label>
          <textarea
            id="chat-input"
            ref={inputRef}
            rows={1}
            value={draft}
            disabled={busy !== null}
            placeholder={stage === "welcome" ? "Or tell Luna what’s going on…" : "Type something…"}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send(draft);
              }
            }}
          />
          <button className="send-btn" type="submit" aria-label="Send" disabled={busy !== null || !draft.trim()}>
            <Icon name="send" />
          </button>
        </form>
      )}
      {stage === "chat" && userMessages >= 1 && !showReady && busy === null && (
        <div className="chat-hint" style={{ paddingBottom: "calc(10px + env(safe-area-inset-bottom))" }}>
          <button className="link-btn" type="button" onClick={startCheck}>Skip ahead: find one small thing to try</button>
        </div>
      )}

      {privacyOpen && (
        <div className="sheet-backdrop" onClick={() => setPrivacyOpen(false)}>
          <div className="sheet" role="dialog" aria-modal="true" aria-labelledby="privacy-title" onClick={(event) => event.stopPropagation()}>
            <h2 id="privacy-title">How this chat is kept</h2>
            {conversation ? (
              <p className="muted">
                This chat uses {conversation.mode === "guided" ? "simple mode, with no AI" : "private AI help"}. Its messages
                are {conversation.retain_text ? "kept after it ends" : "cleared when it ends"}. A short summary and your choice are
                saved either way. You can change the defaults for new chats on the Me page.
              </p>
            ) : (
              <>
                <div className="settings">
                  <label className="setting">
                    <span><strong>Let Luna use AI</strong><small>Smarter replies. Sent privately, with no data kept by the AI provider.</small></span>
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
