"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { apiRequest } from "@/lib/api";
import { clearReminder, useReminders } from "@/lib/reminders";
import type { ReflectionRecord, Resource } from "@/lib/types";

function ActionWorkspace() {
  const searchParams = useSearchParams();
  const decisionId = searchParams.get("decision");
  const reminders = useReminders();
  const [reflection, setReflection] = useState<ReflectionRecord | null>(null);
  const [resource, setResource] = useState<Resource | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const interval = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    Promise.all([
      apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=100"),
      apiRequest<{ items: Resource[] }>("/v1/resources"),
    ])
      .then(([history, catalog]) => {
        const selected = history.items.find((item) => item.decision.decision_id === decisionId) ?? null;
        setReflection(selected);
        setResource(
          selected
            ? catalog.items.find((item) => item.id === selected.decision.action_id) ?? null
            : null,
        );
      })
      .catch(() => setError("The selected action could not be loaded."))
      .finally(() => setLoading(false));
  }, [decisionId]);

  if (loading) {
    return <div className="page-wrap narrow"><section className="paper-card skeleton-card" aria-label="Loading action" /></div>;
  }
  if (error) {
    return <div className="page-wrap narrow"><section className="paper-card empty-card"><span className="folio">Connection note</span><h2>Action unavailable.</h2><p>{error}</p><Link className="button secondary" href="/">Return to Today</Link></section></div>;
  }
  if (!reflection) {
    return <div className="page-wrap narrow"><section className="paper-card empty-card"><span className="folio">No open action</span><h2>Choose an action from a reflection first.</h2><Link className="button primary" href="/reflect">Start a reflection</Link></section></div>;
  }

  const reminder = reminders.find((item) => item.decisionId === reflection.decision.decision_id);
  const remaining = reminder ? Math.max(0, Math.ceil((new Date(reminder.dueAt).getTime() - now) / 60_000)) : 0;

  return (
    <div className="page-wrap narrow reveal">
      <header className="flow-header"><div><span className="kicker">Selected action</span><h1>Give one small experiment a fair try.</h1></div></header>
      <section className="flow-sheet action-sheet">
        <span className="folio">{remaining > 0 ? `Check in in about ${remaining} minutes` : "Ready for a check-in"}</span>
        <div className="action-duration">{resource?.duration_minutes ?? "—"}<small>minutes</small></div>
        <h2>{resource?.title ?? reminder?.actionTitle ?? reflection.decision.action_id}</h2>
        <p>{resource?.summary ?? "Return to the action you selected, then notice what changes."}</p>
        <div className="evidence-slip"><span>Why this</span><p>{reflection.decision.explanation}</p></div>
        <div className="button-row">
          {resource && <a className="button primary" href={resource.url} target="_blank" rel="noreferrer">Open {resource.resource_type}</a>}
          <Link className="button secondary" href={`/check-in?decision=${reflection.decision.decision_id}`}>Record what changed</Link>
        </div>
        {reminder && <button className="text-button" type="button" onClick={() => clearReminder(reflection.decision.decision_id)}>Dismiss device reminder</button>}
      </section>
    </div>
  );
}

export default function ActionPage() {
  return <Suspense fallback={<div className="page-wrap narrow"><section className="paper-card skeleton-card" /></div>}><ActionWorkspace /></Suspense>;
}
