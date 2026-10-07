"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";

import { ApiError, apiRequest } from "@/lib/api";
import { currentActivityCard } from "@/lib/activity-card";
import type { ActivityResource } from "@/lib/activity-session";
import { DISCOVERY_TIMEOUT_MS, MAX_EXCLUDED_SOURCES, type DiscoveryResponse } from "@/lib/discovery";
import type { Conversation } from "@/lib/types";
import { useDiscoveryCapability } from "@/lib/capabilities";
import { DiscoveryAvailability } from "./discovery-availability";
import { ReviewedActivities } from "./reviewed-activities";

type InlineDiscoveryResult = DiscoveryResponse & {
  offers: { resource: ActivityResource; resource_token: string }[];
  conversation_revision: number;
};
type Props = { conversation: Conversation; disabled: boolean; onSave(resource: ActivityResource, token: string): Promise<void>; onChooseApp?(id: string): void };
type ActivityConversation = Conversation & {
  activity_goal?: "settle" | "move" | "understand" | "connect" | "act" | null;
  activity_search_topic?: string | null;
  activity_constraints?: { time_minutes?: number | null; no_audio?: boolean; no_video?: boolean;
    seated?: boolean; avoid_breath_focus?: boolean };
};

/** Only the dedicated general-topic fields leave this panel; the composer and journal never do. */
export function ActivitySessionDiscovery({ conversation, disabled, onSave, onChooseApp }: Props) {
  const [approved, setApproved] = useState(false);
  const capability = useDiscoveryCapability();
  const [topic, setTopic] = useState(() => (conversation as ActivityConversation).activity_search_topic ?? "");
  const [feedback, setFeedback] = useState("");
  const [result, setResult] = useState<InlineDiscoveryResult | null>(null);
  const [seen, setSeen] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pending = useRef<AbortController | null>(null);
  const chat = useRef(conversation);
  useEffect(() => { chat.current = conversation; }, [conversation]);

  useEffect(() => () => pending.current?.abort(), []);
  useEffect(() => {
    // An intervening preference/support/source decision invalidates in-flight retrieval.
    pending.current?.abort();
    pending.current = null;
    queueMicrotask(() => { setBusy(false); setResult(null); setFeedback(""); });
  }, [conversation.revision, disabled]);

  useEffect(() => {
    const suggested = (conversation as ActivityConversation).activity_search_topic;
    if (suggested) queueMicrotask(() => setTopic(suggested));
  }, [conversation]);

  async function search(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!approved || busy || disabled || capability.status !== "configured") return;
    const current = chat.current;
    const controller = new AbortController();
    pending.current?.abort();
    pending.current = controller;
    setBusy(true);
    setError(null);
    try {
      if (seen.length > MAX_EXCLUDED_SOURCES) throw new Error("Start a new search after skipping 30 sources.");
      const activityChat = current as ActivityConversation;
      const card = currentActivityCard(current);
      const payload = {
        goal: activityChat.activity_goal ?? card?.goal ?? "settle",
        style: card?.resource_intent ?? "ground",
        constraints: activityChat.activity_constraints,
        llm_consent: true,
        expected_revision: current.revision ?? 0,
        ...(result ? { original_query: result.original_query } : topic.trim() ? { original_query: topic.trim() } : {}),
        ...(result ? { previous_query: result.updated_query, feedback: feedback.trim() } : {}),
        excluded_urls: seen,
      };
      const response = await apiRequest<InlineDiscoveryResult>(`/v1/conversations/${current.id}/discover`, {
        method: "POST", body: JSON.stringify(payload), timeoutMs: DISCOVERY_TIMEOUT_MS,
        retry: false, signal: controller.signal,
      });
      if (response.provenance) console.info("JournalPulse discovery accounting", JSON.stringify({ search_calls: response.provenance.search_calls, model_runs: response.provenance.model_runs.map((run) => ({ generation_id: run.generation_id, cost_usd: run.cost_usd, model: run.model })) }));
      if (pending.current !== controller || chat.current.id !== current.id || chat.current.revision !== current.revision) return;
      setResult(response);
      setSeen((values) => [...new Set([...values, ...response.candidates.map((candidate) => candidate.url)])]);
      setFeedback("");
    } catch (reason) {
      if (pending.current !== controller) return;
      setError(reason instanceof Error ? reason.message : "Search could not finish. You can keep talking or try a catalog activity.");
    } finally {
      if (pending.current === controller) { pending.current = null; setBusy(false); }
    }
  }

  async function save(resource: ActivityResource, token: string) {
    if (busy || disabled) return;
    setBusy(true);
    setError(null);
    try { await onSave(resource, token); }
    catch (reason) { setError(reason instanceof ApiError || reason instanceof Error ? reason.message : "The resource could not be saved."); }
    finally { setBusy(false); }
  }

  return (
    <section className="card stack" aria-label="Find another activity">
      <h2>Find something that fits better</h2>
      <DiscoveryAvailability status={capability.status} onRetry={capability.retry} />
      {capability.status === "configured" && <>
      <p className="small muted">Brave receives a general activity topic. Your chat and journal are not included. Luna selects search snippets; full pages are not reviewed.</p>
      <label className="row"><input type="checkbox" checked={approved} disabled={disabled}
        onChange={(event) => {
          setApproved(event.target.checked);
          if (!event.target.checked) { pending.current?.abort(); pending.current = null; setBusy(false); }
        }} />Allow this general activity search with Brave and Luna</label>
      <form className="stack" onSubmit={search}>
        <label className="text-field">General activity topic (optional)
          <input value={topic} onChange={(event) => setTopic(event.target.value)} maxLength={160}
            placeholder="Leave blank for a general topic based on your goal" disabled={busy || disabled || Boolean(result)} />
        </label>
        {result && <label className="text-field">What would fit better? Keep it general.
          <textarea value={feedback} onChange={(event) => setFeedback(event.target.value)} maxLength={160}
            placeholder="For example: shorter seated no video" disabled={busy || disabled} required />
        </label>}
        <button className="btn btn-soft" type="submit" disabled={!approved || busy || disabled || Boolean(result && !feedback.trim())}>
          {busy ? "Searching…" : result ? "Find different sources" : "Search activities"}
        </button>
      </form>
      </>}
      <details open={capability.status !== "configured"}><summary>Browse app activities</summary>
        <ReviewedActivities goal={(conversation as ActivityConversation).activity_goal} constraints={conversation.activity_constraints} onChoose={onChooseApp} disabled={busy || disabled} />
      </details>
      {busy && <p role="status">Finding a fitting activity…</p>}
      {error && <p className="note error" role="alert">{error}</p>}
      {result && <>
        {!result.offers.length && <p>No fitting sources this time. You can change the format or keep talking.</p>}
        {result.offers.map(({ resource, resource_token }) => <article className="stack" key={resource.id}>
          <h3>{resource.title}</h3><p className="small muted">{resource.provider} · Search snippet</p>
          {resource.url && <a href={resource.url} target="_blank" rel="noopener noreferrer">Open link</a>}
          <button className="btn btn-soft" type="button" disabled={busy || disabled}
            onClick={() => void save(resource, resource_token)}>Save this activity</button>
        </article>)}
        <p className="small muted">Opening a link does not save it or mark it completed. Refinement skips all sources already shown.</p>
        <button className="btn btn-ghost" type="button" disabled={busy} onClick={() => {
          pending.current?.abort(); pending.current = null; setResult(null); setSeen([]); setFeedback(""); setError(null);
        }}>Start a new search</button>
      </>}
    </section>
  );
}
