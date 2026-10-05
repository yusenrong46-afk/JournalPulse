"use client";

import { type FormEvent, Suspense, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

import { ApiError } from "@/lib/api";
import {
  type DiscoveryRequest,
  type DiscoveryResponse,
  discoveryTopicForGoal,
  excludedSources,
  refinementRequest,
  searchDiscovery,
} from "@/lib/discovery";
import { usePreferences } from "@/lib/preferences";

function DiscoveryWorkspace() {
  const searchParams = useSearchParams();
  const [preferences] = usePreferences();
  const [topic, setTopic] = useState(() => discoveryTopicForGoal(searchParams.get("goal")));
  const [approved, setApproved] = useState(false);
  const [response, setResponse] = useState<DiscoveryResponse | null>(null);
  const [feedback, setFeedback] = useState("");
  const [seen, setSeen] = useState<string[]>([]);
  const [rejected, setRejected] = useState<string[]>([]);
  const [manualExclusions, setManualExclusions] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [supportRequired, setSupportRequired] = useState(false);
  const pending = useRef<AbortController | null>(null);

  useEffect(() => () => pending.current?.abort(), []);

  async function run(payload: DiscoveryRequest) {
    // A reset or consent change invalidates an older response before it can restore the screen.
    const controller = new AbortController();
    pending.current?.abort();
    pending.current = controller;
    setBusy(true);
    setError(null);
    setSupportRequired(false);
    try {
      const result = await searchDiscovery(payload, controller.signal);
      if (pending.current !== controller) return;
      setResponse(result);
      setSeen((current) => [...new Set([...current, ...result.candidates.map((item) => item.url)])]);
      setFeedback("");
    } catch (reason) {
      if (pending.current !== controller) return;
      setSupportRequired(reason instanceof ApiError && reason.status === 422
        && reason.message.includes("Web discovery is paused"));
      setError(reason instanceof ApiError || reason instanceof Error
        ? reason.message : "Resources could not be found. Please try again.");
    } finally {
      if (pending.current === controller) {
        pending.current = null;
        setBusy(false);
      }
    }
  }

  function start(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!approved) return;
    try {
      const original = topic.trim();
      if (original.length < 3) throw new Error("Write a general topic of at least 3 characters.");
      void run({
        original_query: original,
        excluded_urls: excludedSources([], manualExclusions),
        llm_consent: approved,
        locale: preferences.locale,
      });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Check your search topic.");
    }
  }

  function refine(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!response || !approved) return;
    try {
      void run(refinementRequest({
        previous: response, feedback, seen: [...seen, ...rejected],
        manuallyExcluded: manualExclusions, consent: approved,
        locale: preferences.locale,
      }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Check your search feedback.");
    }
  }

  function reset() {
    pending.current?.abort();
    pending.current = null;
    setBusy(false);
    setResponse(null);
    setFeedback("");
    setSeen([]);
    setRejected([]);
    setManualExclusions("");
    setError(null);
    setSupportRequired(false);
  }

  function changeApproval(value: boolean) {
    setApproved(value);
    if (!value) {
      pending.current?.abort();
      pending.current = null;
      setBusy(false);
    }
  }

  return (
    <div className="page page-wide stack">
      <header className="page-head">
        <h1>Find a resource</h1>
        <p className="muted">Explore a topic, then tell Luna what would fit better.</p>
        <Link href="/talk">Return to chat</Link>
      </header>

      <section className="card stack" aria-label="Approve your search topic">
        <p>Write a general topic you are comfortable sharing, such as “short grounding exercises for work breaks”.
          Leave out names and personal journal details.</p>
        <p className="muted">Luna selects from Brave Search snippets. Full pages are not read, and their claims are not fact-checked.</p>
        <label className="row">
          <input type="checkbox" checked={approved} onChange={(event) => changeApproval(event.target.checked)} />
          I approve sending this topic and a short focus Luna derives from your feedback to Brave Search,
          and this topic and feedback to Luna.
        </label>
        <p className="small muted">Review or edit the topic before searching. Your journal and chat messages
          are not included automatically. Keep feedback general too.</p>
        {!response ? (
          <form className="stack" onSubmit={start}>
            <label className="text-field" htmlFor="discovery-topic">
              General topic to search
              <textarea id="discovery-topic" value={topic} onChange={(event) => setTopic(event.target.value)}
                minLength={3} maxLength={160} required disabled={busy} />
            </label>
            <label className="text-field" htmlFor="discovery-exclude">
              Sources to skip (optional, one HTTPS link per line)
              <textarea id="discovery-exclude" value={manualExclusions}
                onChange={(event) => setManualExclusions(event.target.value)} maxLength={6000} disabled={busy} />
            </label>
            <button type="submit" className="btn btn-primary" disabled={!approved || busy || topic.trim().length < 3}>
              {busy ? "Searching…" : "Search this topic"}
            </button>
          </form>
        ) : (
          <>
            <p><strong>Your original goal:</strong> {response.original_query}</p>
            <p><strong>Latest search:</strong> {response.updated_query}</p>
            <button type="button" className="btn btn-soft" onClick={reset}>Start a new search</button>
          </>
        )}
      </section>

      {busy && <p role="status">Luna is looking through search snippets. This may take a minute.</p>}
      {error && <p className="error" role="alert">{error}</p>}
      {supportRequired && <Link href="/talk" className="btn btn-primary">Talk with Luna</Link>}

      {response && (
        <section className="stack" aria-label="Resource suggestions" aria-busy={busy}>
          <h2>{response.candidates.length ? "A few sources to explore" : "No matching sources this time"}</h2>
          {!response.candidates.length && <p>Try describing a different format or focus below.</p>}
          {response.candidates.map((candidate) => (
            <article key={candidate.url} className="card stack">
              <h3><a href={candidate.url} target="_blank" rel="noopener noreferrer">{candidate.title}</a></h3>
              <p className="muted">{new URL(candidate.url).hostname} · Search snippet</p>
              <p>{candidate.description}</p>
              <p><strong>Why Luna chose it:</strong> {candidate.why_selected}</p>
              <button type="button" className="btn btn-soft" disabled={busy || rejected.includes(candidate.url)}
                onClick={() => setRejected((current) => [...current, candidate.url])}>
                {rejected.includes(candidate.url) ? "Skipped on the next search" : "Skip this source"}
              </button>
            </article>
          ))}
          <details className="explain">
            <summary>How these were chosen</summary>
            <div>
              {response.limitations.map((limitation) => <p key={limitation}>{limitation}</p>)}
              <p>Brave returned {response.provenance.candidate_count} usable snippets.
                Luna selected {response.candidates.length} sources from that list.</p>
            </div>
          </details>
          <form className="card stack" onSubmit={refine}>
            <h2>What would fit better?</h2>
            <label className="text-field" htmlFor="discovery-feedback">
              Feedback for Luna
              <textarea id="discovery-feedback" value={feedback} onChange={(event) => setFeedback(event.target.value)}
                placeholder="For example: shorter, practical exercises with no video" maxLength={600} required disabled={busy} />
            </label>
            <p className="muted">The next search keeps your original goal and skips every source already shown.</p>
            <details className="explain">
              <summary>Sources skipped on the next search ({seen.length})</summary>
              <div>
                {seen.map((url) => <p key={url}>{url}</p>)}
                <label className="text-field" htmlFor="discovery-more-exclude">
                  Additional sources to skip (one HTTPS link per line)
                  <textarea id="discovery-more-exclude" value={manualExclusions}
                    onChange={(event) => setManualExclusions(event.target.value)} maxLength={6000} disabled={busy} />
                </label>
              </div>
            </details>
            <button type="submit" className="btn btn-primary" disabled={!approved || busy || !feedback.trim()}>
              {busy ? "Finding alternatives…" : "Find different sources"}
            </button>
          </form>
        </section>
      )}
    </div>
  );
}

export default function DiscoverPage() {
  return <Suspense fallback={<p role="status">Opening resource search…</p>}><DiscoveryWorkspace /></Suspense>;
}
