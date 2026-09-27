"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { apiRequest } from "@/lib/api";
import { usePreferences } from "@/lib/preferences";
import { useReminders } from "@/lib/reminders";
import type { Insights, ReflectionRecord } from "@/lib/types";

export function TodayDashboard() {
  const [preferences, , preferencesLoaded] = usePreferences();
  const reminders = useReminders();
  const [history, setHistory] = useState<ReflectionRecord[]>([]);
  const [insights, setInsights] = useState<Insights | null>(null);
  const [offline, setOffline] = useState(false);
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const interval = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    Promise.all([
      apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=10"),
      apiRequest<Insights>("/v1/insights"),
    ])
      .then(([records, patterns]) => {
        setHistory(records.items);
        setInsights(patterns);
      })
      .catch(() => setOffline(true))
      .finally(() => setLoading(false));
  }, [retry]);

  function retryLoad() {
    setLoading(true);
    setOffline(false);
    setRetry((value) => value + 1);
  }

  if (preferencesLoaded && !preferences.onboarded) {
    return (
      <div className="today-layout">
        <section className="paper-card onboarding-card hero-card">
          <span className="folio">Your first five minutes</span>
          <h2>Decide what leaves the page before you write.</h2>
          <p>Choose how analysis and text retention work, then begin a guided reflection with a boundary you control.</p>
          <Link className="button primary" href="/welcome">Set my privacy boundary <span aria-hidden="true">→</span></Link>
        </section>
        <MethodStrip />
      </div>
    );
  }

  if (offline) {
    return (
      <div className="today-layout">
        <section className="paper-card empty-card hero-card offline-card">
          <div className="card-symbol" aria-hidden="true">↗</div>
          <span className="folio">Connection note</span>
          <h2>The reflection service is taking a pause.</h2>
          <p>Nothing was sent or stored. Reconnect the service before saving a reflection.</p>
          <div className="button-row">
            <button className="button primary" type="button" onClick={retryLoad}>Try the connection again</button>
            <Link className="button secondary" href="/privacy">Review data controls</Link>
          </div>
        </section>
        <MethodStrip />
      </div>
    );
  }

  const latest = history[0] ?? null;
  const pendingId = insights?.pending_decision_ids[0];
  const pending = history.find((item) => item.decision.decision_id === pendingId);
  const reminder = reminders.find((item) => item.decisionId === pendingId);
  const minutesUntilDue = reminder
    ? Math.max(0, Math.ceil((new Date(reminder.dueAt).getTime() - now) / 60_000))
    : 0;

  const completionRate = Math.round((insights?.completion_rate ?? 0) * 100);

  return (
    <div className="today-layout" aria-busy={loading}>
      <div className="today-grid">
      {pending ? (
        <section className="paper-card primary-card hero-card pending-card">
          <div className="card-symbol" aria-hidden="true">↻</div>
          <span className="folio">
            {minutesUntilDue > 0 ? `Open loop / check in in ${minutesUntilDue} min` : "Open loop / check-in due"}
          </span>
          <h2>What changed after your last action?</h2>
          <p>{pending.reflection.summary}</p>
          <div className="action-recall"><span>Action</span><strong>{reminder?.actionTitle ?? pending.decision.action_id.replaceAll("_", " ")}</strong></div>
          <div className="button-row">
            <Link className="button primary" href={`/check-in?decision=${pending.decision.decision_id}`}>Record the outcome</Link>
            <Link className="button secondary" href={`/action?decision=${pending.decision.decision_id}`}>Reopen the action</Link>
          </div>
        </section>
      ) : (
        <section className="paper-card primary-card hero-card">
          <div className="card-symbol" aria-hidden="true">01</div>
          <span className="folio">Today / field note</span>
          <h2>{latest ? "Begin another useful observation" : "Begin with one honest observation"}</h2>
          <p>{latest ? "Your previous loop is closed. Start again only when there is something worth noticing." : "Write what happened, correct the system’s read, and choose what you want to change—not how you are supposed to feel."}</p>
          <div className="button-row">
            <Link className="button primary" href="/reflect">Open a reflection <span aria-hidden="true">→</span></Link>
            <Link className="button secondary" href="/talk">Talk it through</Link>
          </div>
        </section>
      )}
        <aside className="evidence-panel" aria-label="Your evidence so far">
          <div className="evidence-heading"><span className="folio">Evidence so far</span><span className="live-mark"><i /> updated</span></div>
          <div className="evidence-stat"><strong>{insights?.reflection_count ?? "—"}</strong><span>reflections recorded</span></div>
          <div className="completion-meter" role="progressbar" aria-label="Reflection loops closed" aria-valuemin={0} aria-valuemax={100} aria-valuenow={completionRate}><span style={{ width: `${completionRate}%` }} /></div>
          <div className="evidence-row"><span>Loops closed</span><strong>{completionRate}%</strong></div>
          <div className="evidence-row"><span>Outcomes captured</span><strong>{insights?.completed_outcomes ?? "—"}</strong></div>
          <Link className="panel-link" href="/patterns">See the full pattern <span aria-hidden="true">↗</span></Link>
        </aside>
      </div>

      {latest && (
        <section className="latest-observation" aria-labelledby="latest-observation-heading">
          <div><span className="kicker">Latest observation</span><time>{new Date(latest.created_at).toLocaleDateString("en-CA", { month: "short", day: "numeric" })}</time></div>
          <div><h2 id="latest-observation-heading">{latest.reflection.summary}</h2><p>{latest.reflection.interpretation}</p></div>
          <div className="state-stamp"><span>Agency</span><strong>{Math.round(latest.state.agency * 100)}</strong><small>self-corrected</small></div>
        </section>
      )}

      <MethodStrip />
    </div>
  );
}

function MethodStrip() {
  return (
    <section className="method-strip" aria-label="How JournalPulse works">
      <header><span className="kicker">The method</span><p>One loop, three deliberate decisions.</p></header>
      <ol>
        <li><span>01</span><div><strong>Observe</strong><small>Write the event, signal, and unresolved edge.</small></div></li>
        <li><span>02</span><div><strong>Correct</strong><small>Edit the proposed state until it reflects you.</small></div></li>
        <li><span>03</span><div><strong>Test</strong><small>Try one reviewed action and record what changed.</small></div></li>
      </ol>
      <p className="method-boundary">Patterns are descriptive evidence, never a diagnosis or promise.</p>
    </section>
  );
}
