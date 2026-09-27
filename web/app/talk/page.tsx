"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";

import { StateControls } from "@/components/state-controls";
import { apiRequest } from "@/lib/api";
import { readOpenConversationId, talkView, writeOpenConversationId } from "@/lib/conversation";
import { usePreferences } from "@/lib/preferences";
import { saveReminder } from "@/lib/reminders";
import type {
  AffectiveState,
  Conversation,
  ConversationDetail,
  ConversationMessage,
  ConversationTurn,
  ReflectionRecord,
  Resource,
  SystemStatus,
} from "@/lib/types";

const CHAT_TIMEOUT_MS = 60_000;
const TALK_REQUIRES_AI = "Talk needs private AI analysis. The guided reflection is still available.";
const RETENTION_COPY =
  "While this conversation is open, its messages are stored on the server so a reload can restore them. When it ends, original text is cleared unless you choose to keep it. The summary, card reason, and decision can still be saved. An open conversation with no activity for 24 hours is closed the next time you use Talk.";

const initialState: AffectiveState = {
  valence: 0,
  arousal: 0.5,
  agency: 0.5,
  emotion_tags: [],
  confidence: 0,
};

function TalkWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [preferences, , preferencesLoaded] = usePreferences();
  const [aiAvailable, setAiAvailable] = useState<boolean | null>(null);
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [messages, setMessages] = useState<ConversationMessage[]>([]);
  const [consent, setConsent] = useState(false);
  const [retain, setRetain] = useState(false);
  const [draft, setDraft] = useState("");
  const [selectedAction, setSelectedAction] = useState("");
  const [selfReport, setSelfReport] = useState(initialState);
  const [saved, setSaved] = useState<ReflectionRecord | null>(null);
  const [savedResource, setSavedResource] = useState<Resource | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const acceptRequestId = useRef("");
  const pendingMessageId = useRef<string | null>(null);
  const preferencesApplied = useRef(false);
  const loadedId = useRef<string | null>(null);

  useEffect(() => {
    apiRequest<SystemStatus>("/v1/system/status")
      .then((status) => setAiAvailable(status.analysis_mode === "ai_configured"))
      .catch(() => setAiAvailable(false));
  }, []);

  useEffect(() => {
    if (!preferencesLoaded || preferencesApplied.current) return;
    preferencesApplied.current = true;
    setConsent(preferences.llmConsent);
    setRetain(preferences.retainText);
  }, [preferences.llmConsent, preferences.retainText, preferencesLoaded]);

  useEffect(() => {
    const requested = searchParams.get("c") ?? readOpenConversationId(window.localStorage);
    if (!requested || loadedId.current === requested) return;
    let cancelled = false;
    apiRequest<ConversationDetail>(`/v1/conversations/${requested}`)
      .then((detail) => {
        if (cancelled) return;
        if (detail.conversation.status !== "open") {
          loadedId.current = null;
          writeOpenConversationId(window.localStorage, null);
          setConversation(null);
          setMessages([]);
          return;
        }
        loadedId.current = detail.conversation.id;
        writeOpenConversationId(window.localStorage, detail.conversation.id);
        setConversation(detail.conversation);
        setMessages(detail.messages);
      })
      .catch(() => {
        if (cancelled) return;
        writeOpenConversationId(window.localStorage, null);
      });
    return () => {
      cancelled = true;
    };
  }, [searchParams]);

  function remember(next: Conversation) {
    loadedId.current = next.id;
    writeOpenConversationId(window.localStorage, next.id);
    if (searchParams.get("c") !== next.id) router.replace(`/talk?c=${next.id}`);
    setConversation(next);
  }

  async function start() {
    setLoading(true);
    setError(null);
    try {
      const created = await apiRequest<Conversation>("/v1/conversations", {
        method: "POST",
        retry: true,
        timeoutMs: CHAT_TIMEOUT_MS,
        body: JSON.stringify({
          client_request_id: crypto.randomUUID(),
          llm_consent: consent,
          retain_text: retain,
          locale: preferences.locale,
        }),
      });
      remember(created);
      setMessages([]);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : TALK_REQUIRES_AI);
    } finally {
      setLoading(false);
    }
  }

  async function send() {
    if (!conversation || !draft.trim() || loading) return;
    const text = draft;
    const messageId = pendingMessageId.current ?? crypto.randomUUID();
    pendingMessageId.current = messageId;
    setLoading(true);
    setError(null);
    try {
      const turn = await apiRequest<ConversationTurn>(`/v1/conversations/${conversation.id}/messages`, {
        method: "POST",
        retry: true,
        timeoutMs: CHAT_TIMEOUT_MS,
        body: JSON.stringify({ client_message_id: messageId, text }),
      });
      pendingMessageId.current = null;
      setDraft("");
      setMessages((current) => [...current, turn.user_message, turn.assistant_message]);
      remember(turn.conversation);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The reply was not saved.");
    } finally {
      setLoading(false);
    }
  }

  async function accept() {
    if (!conversation || !selectedAction) return;
    if (!acceptRequestId.current) acceptRequestId.current = crypto.randomUUID();
    setLoading(true);
    setError(null);
    try {
      const record = await apiRequest<ReflectionRecord>(`/v1/conversations/${conversation.id}/accept`, {
        method: "POST",
        retry: true,
        timeoutMs: CHAT_TIMEOUT_MS,
        body: JSON.stringify({
          client_request_id: acceptRequestId.current,
          action_id: selectedAction,
          self_report: selfReport,
        }),
      });
      const resource = conversation.card?.actions.find((item) => item.id === record.decision.action_id) ?? null;
      saveReminder({
        decisionId: record.decision.decision_id,
        actionId: record.decision.action_id,
        actionTitle: resource?.title ?? "Your selected action",
        dueAt: new Date(Date.now() + preferences.followUpMinutes * 60_000).toISOString(),
      });
      loadedId.current = null;
      writeOpenConversationId(window.localStorage, null);
      setSaved(record);
      setSavedResource(resource);
      setConversation(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The action could not be saved.");
    } finally {
      setLoading(false);
    }
  }

  async function closeConversation() {
    if (!conversation) return;
    setLoading(true);
    setError(null);
    try {
      await apiRequest<Conversation>(`/v1/conversations/${conversation.id}/close`, {
        method: "POST",
        retry: true,
      });
      loadedId.current = null;
      writeOpenConversationId(window.localStorage, null);
      setConversation(null);
      setMessages([]);
      router.replace("/talk");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The conversation could not be closed.");
    } finally {
      setLoading(false);
    }
  }

  async function deleteConversation() {
    if (!conversation) return;
    setLoading(true);
    setError(null);
    try {
      await apiRequest<void>(`/v1/conversations/${conversation.id}`, { method: "DELETE" });
      loadedId.current = null;
      writeOpenConversationId(window.localStorage, null);
      setConversation(null);
      setMessages([]);
      router.replace("/talk");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The conversation could not be deleted.");
    } finally {
      setLoading(false);
    }
  }

  const view = talkView({
    aiAvailable,
    status: conversation?.status,
    safetyMode: conversation?.safety_mode,
    saved: Boolean(saved),
  });
  const supportMessage = conversation?.safety?.support_message
    ?? messages.findLast((message) => message.role === "assistant")?.content;

  return (
    <div className="page-wrap narrow reveal">
      <header className="flow-header">
        <div className="page-heading-copy">
          <span className="kicker">Short conversation</span>
          <h1>Talk it through.</h1>
          <p>A few turns with a private model, then one reviewed action if you want it. This is not a diagnosis.</p>
        </div>
      </header>

      {view === "unavailable" && (
        <section className="flow-sheet">
          <span className="folio">Unavailable</span>
          <h2>Talk needs private AI analysis.</h2>
          <p>{TALK_REQUIRES_AI}</p>
          <Link className="button primary" href="/reflect">Open a guided reflection <span aria-hidden="true">→</span></Link>
        </section>
      )}

      {view === "start" && (
        <section className="flow-sheet">
          <span className="folio">Before the first message</span>
          <h2>Confirm how this conversation is handled.</h2>
          <p className="section-lede">{RETENTION_COPY}</p>
          <div className="consent-box">
            <label>
              <input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} />
              <span><strong>Use private AI analysis for this conversation</strong><small>Sent through a zero-data-retention route. Required for Talk.</small></span>
            </label>
            <label>
              <input type="checkbox" checked={retain} onChange={(event) => setRetain(event.target.checked)} />
              <span><strong>Keep the original messages after this conversation ends</strong><small>Off clears every message when you accept an action or end the conversation.</small></span>
            </label>
          </div>
          {!consent && <p>{TALK_REQUIRES_AI} <Link href="/reflect">Open a guided reflection</Link></p>}
          <button className="button primary" type="button" onClick={start} disabled={loading || !consent}>
            {loading ? "Starting…" : "Start this conversation"}
          </button>
        </section>
      )}

      {(view === "chat" || view === "support") && conversation && (
        <section className="flow-sheet">
          <span className="folio">{view === "support" ? "Support mode" : "This conversation"}</span>
          {view === "support" ? (
            <>
              <h2>Human support comes first.</h2>
              <p>{supportMessage}</p>
              <a className="button urgent" href="https://988.ca/" target="_blank" rel="noreferrer">Open 9-8-8 Canada</a>
            </>
          ) : (
            <h2>Write the next thing that still has energy.</h2>
          )}
          <ol className="talk-log" role="log" aria-live="polite" aria-label="Conversation">
            {messages.map((message) => (
              message.content ? (
                <li key={message.id} className={message.role}>{message.content}</li>
              ) : null
            ))}
          </ol>
          {loading && view === "chat" && <p className="talk-status" role="status">Luna is replying…</p>}
          {view === "chat" && (
            <div className="talk-composer">
              <label className="field-label">
                Your message
                <textarea
                  value={draft}
                  disabled={loading}
                  onChange={(event) => setDraft(event.target.value)}
                  placeholder="Write the next thing you want to say…"
                />
              </label>
              <button className="button primary" type="button" onClick={send} disabled={loading || !draft.trim()}>
                {loading ? "Waiting for a reply…" : "Send"}
              </button>
            </div>
          )}
          {conversation.card && (
            <div className="talk-card">
              <h2>One small next move</h2>
              <p>{conversation.card.card_reason}</p>
              <div className="resource-choice-grid">
                {conversation.card.actions.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={selectedAction === item.id ? "resource-choice active" : "resource-choice"}
                    onClick={() => setSelectedAction(item.id)}
                  >
                    <span className="resource-meta">{item.duration_minutes ?? "—"} min · {item.provider}</span>
                    <strong>{item.title}</strong>
                    <p>{item.summary}</p>
                  </button>
                ))}
              </div>
              {selectedAction && (
                <>
                  <StateControls state={selfReport} onChange={setSelfReport} />
                  <button className="button primary" type="button" onClick={accept} disabled={loading}>
                    {loading ? "Saving your choice…" : "Use this action"}
                  </button>
                </>
              )}
            </div>
          )}
          <div className="button-row">
            <button className="button secondary" type="button" onClick={closeConversation} disabled={loading}>End conversation</button>
            <button className="button urgent" type="button" onClick={deleteConversation} disabled={loading}>Delete this conversation</button>
          </div>
        </section>
      )}

      {view === "saved" && saved && (
        <section className="flow-sheet action-sheet">
          <span className="folio">Next</span>
          <h2>{savedResource?.title ?? "Your choice is saved"}</h2>
          <p>{savedResource?.summary ?? saved.reflection.summary}</p>
          <div className="button-row">
            {savedResource && <a className="button primary" href={savedResource.url} target="_blank" rel="noreferrer">Open {savedResource.resource_type}</a>}
            <Link className="button secondary" href={`/check-in?decision=${saved.decision.decision_id}`}>Check in afterward</Link>
          </div>
        </section>
      )}
      {error && <p className="error-note" role="alert">{error}</p>}
    </div>
  );
}

export default function TalkPage() {
  return (
    <Suspense fallback={<div className="page-wrap narrow"><section className="paper-card skeleton-card" /></div>}>
      <TalkWorkspace />
    </Suspense>
  );
}
