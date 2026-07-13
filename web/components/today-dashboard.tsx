"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { apiRequest } from "@/lib/api";
import { usePreferences } from "@/lib/preferences";
import type { Insights, ReflectionRecord } from "@/lib/types";

export function TodayDashboard() {
  const [preferences, , preferencesLoaded] = usePreferences();
  const [history, setHistory] = useState<ReflectionRecord[]>([]);
  const [insights, setInsights] = useState<Insights | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    Promise.all([
      apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=10"),
      apiRequest<Insights>("/v1/insights"),
    ])
      .then(([records, patterns]) => {
        setHistory(records.items);
        setInsights(patterns);
      })
      .catch(() => setOffline(true));
  }, []);

  if (preferencesLoaded && !preferences.onboarded) {
    return (
      <section className="paper-card onboarding-card">
        <span className="folio">Start with the boundary</span>
        <h2>Decide what leaves the page before you write.</h2>
        <p>Set AI-processing and text-retention preferences, then begin the first guided reflection.</p>
        <Link className="button primary" href="/welcome">Review privacy and begin</Link>
      </section>
    );
  }

  if (offline) {
    return (
      <section className="paper-card empty-card">
        <span className="folio">Connection note</span>
        <h2>The reflection service is offline.</h2>
        <p>Your journal text has not been sent or stored. Start the API service, then return here.</p>
      </section>
    );
  }

  const latest = history[0] ?? null;
  const pendingId = insights?.pending_decision_ids[0];
  const pending = history.find((item) => item.decision.decision_id === pendingId);

  return (
    <div className="today-grid">
      {pending ? (
        <section className="paper-card primary-card pending-card">
          <span className="folio">Open loop / check-in due</span>
          <h2>What changed after your last action?</h2>
          <p>{pending.reflection.summary}</p>
          <div className="action-recall"><span>Action</span><strong>{pending.decision.action_id.replaceAll("_", " ")}</strong></div>
          <Link className="button primary" href={`/check-in?decision=${pending.decision.decision_id}`}>Record the outcome</Link>
        </section>
      ) : (
        <section className="paper-card primary-card">
          <span className="folio">Today / field note</span>
          <h2>{latest ? "Begin another useful observation" : "Begin with one honest observation"}</h2>
          <p>{latest ? "Your previous loop is closed. Start again only when there is something worth noticing." : "Write what happened, correct the system’s read, and choose what you want to change—not how you are supposed to feel."}</p>
          <Link className="button primary" href="/reflect">Open a reflection</Link>
        </section>
      )}
      <section className="paper-card metric-card">
        <span className="folio">Evidence so far</span>
        <div className="large-number">{insights?.reflection_count ?? "—"}</div>
        <p>recorded reflections</p>
        <div className="hairline" />
        <strong>{Math.round((insights?.completion_rate ?? 0) * 100)}% of loops closed</strong>
      </section>
      <section className="margin-note">
        <span>Method note 01</span>
        <p>A recommendation remains a baseline decision until repeated outcomes justify personalization.</p>
      </section>
    </div>
  );
}
