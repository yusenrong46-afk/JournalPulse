"use client";

import { type FormEvent, Suspense, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Icon } from "@/components/nav-icon";
import { Luna } from "@/components/luna";
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
    <div className="page page-wide stack discovery-page">
      <header className="page-head companion-page-head">
        <Luna mood={supportRequired ? "support" : busy ? "thinking" : response ? "reflecting" : "idle"} size={72} decorative />
        <span className="eyebrow">A little guidance, at your pace</span>
        <h1>Find something that fits.</h1>
        <p className="muted">A quiet exercise, a useful read, a different perspective.</p>
        <Link className="subtle-link" href="/talk">Return to chat <Icon name="arrow" /></Link>
      </header>

      <section className="card stack discovery-consent" aria-label="Approve your search topic">
        {!response ? (
          <form className="discovery-query" onSubmit={start}>
            <label className="text-field" htmlFor="discovery-topic">
              What would you like to explore?
              <span className="search-input-wrap"><Icon name="search" />
                <input id="discovery-topic" value={topic} onChange={(event) => setTopic(event.target.value)}
                  placeholder="Try quiet grounding activities"
                  aria-describedby="discovery-topic-note" minLength={3} maxLength={160} required disabled={busy} />
              </span>
            </label>
            <button type="submit" className="btn btn-primary" disabled={!approved || busy || topic.trim().length < 3}>
              {busy ? "Searching…" : "Search this topic"}
            </button>
            <p id="discovery-topic-note" className="small muted discovery-topic-note">Use a general topic. Leave out names and private details.</p>
            <details className="explain discovery-exclusions"><summary>Sources to skip (optional)</summary>
              <div><label className="text-field" htmlFor="discovery-exclude">
                Sources to skip (optional, one HTTPS link per line)
                <textarea id="discovery-exclude" value={manualExclusions}
                  onChange={(event) => setManualExclusions(event.target.value)} maxLength={6000} disabled={busy} />
              </label></div>
            </details>
          </form>
        ) : (
          <div className="discovery-current">
            <div><span className="eyebrow">Your topic</span><h2>{response.original_query}</h2>
              {response.updated_query !== response.original_query && <p className="small muted">Latest search: {response.updated_query}</p>}
            </div>
            <button type="button" className="btn btn-ghost" onClick={reset}>New search <Icon name="arrow" /></button>
          </div>
        )}
        <div className="search-permission">
          <label className="check-row">
            <input type="checkbox" checked={approved} onChange={(event) => changeApproval(event.target.checked)} />
            <span>Share this topic and general refinement feedback with Brave Search and Luna.</span>
          </label>
          <p className="small muted">Your journal and chat are not included. Luna may add a short focus from your feedback.
            Results use snippets; full pages and their claims are not reviewed.</p>
        </div>
      </section>

      {busy && <div className="search-progress" role="status"><Luna mood="thinking" size={44} decorative /><p>Luna is looking through search snippets. This may take a minute.</p></div>}
      {error && <p className="note error" role="alert">{error}</p>}
      {supportRequired && <Link href="/talk" className="btn btn-primary">Talk with Luna</Link>}

      {response && (
        <section className="stack discovery-results" aria-label="Resource suggestions" aria-busy={busy}>
          <div className="section-title"><h2>{response.candidates.length ? "A few sources to explore" : "No matching sources this time"}</h2>
            {response.candidates.length > 0 && <span className="small muted">{response.candidates.length} suggestions</span>}
          </div>
          {!response.candidates.length && <p>Try describing a different format or focus below.</p>}
          {response.candidates.map((candidate, index) => (
            <article key={candidate.url} className="card stack discovery-result" data-skipped={rejected.includes(candidate.url)}>
              <div className="result-source"><span className="result-index">{String(index + 1).padStart(2, "0")}</span>
                <span>{new URL(candidate.url).hostname}</span><span className="result-kind">Search snippet</span></div>
              <h3><a href={candidate.url} target="_blank" rel="noopener noreferrer">{candidate.title}<Icon name="external" /></a></h3>
              <p>{candidate.description}</p>
              <p className="result-reason"><strong>Why it may fit</strong> {candidate.why_selected}</p>
              <button type="button" className="btn btn-ghost result-skip" disabled={busy || rejected.includes(candidate.url)}
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
          <form className="card stack discovery-refine" onSubmit={refine}>
            <div><span className="eyebrow">Keep exploring</span><h2>What would fit better?</h2></div>
            <label className="text-field" htmlFor="discovery-feedback">
              Feedback for Luna
              <textarea id="discovery-feedback" value={feedback} onChange={(event) => setFeedback(event.target.value)}
                placeholder="For example: shorter, practical exercises with no video" maxLength={600} required disabled={busy} />
            </label>
            <p className="small muted">The next search keeps your original goal and skips every source already shown.</p>
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
