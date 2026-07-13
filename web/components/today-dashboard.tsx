"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { apiRequest } from "@/lib/api";
import type { Insights, ReflectionRecord } from "@/lib/types";

export function TodayDashboard() {
  const [latest, setLatest] = useState<ReflectionRecord | null>(null);
  const [insights, setInsights] = useState<Insights | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    Promise.all([
      apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=1"),
      apiRequest<Insights>("/v1/insights"),
    ])
      .then(([history, patterns]) => {
        setLatest(history.items[0] ?? null);
        setInsights(patterns);
      })
      .catch(() => setOffline(true));
  }, []);

  if (offline) {
    return (
      <section className="paper-card empty-card">
        <span className="folio">Connection note</span>
        <h2>The reflection service is offline.</h2>
        <p>Your journal text has not been sent or stored. Start the FastAPI service, then return here.</p>
      </section>
    );
  }

  return (
    <div className="today-grid">
      <section className="paper-card primary-card">
        <span className="folio">Today / field note</span>
        <h2>{latest ? "Continue from the last signal" : "Begin with one honest observation"}</h2>
        <p>
          {latest
            ? latest.reflection.summary
            : "Write what happened, check the system’s read, and choose what you want to change—not how you are supposed to feel."}
        </p>
        <Link className="button primary" href="/reflect">
          {latest ? "Start a new reflection" : "Open a reflection"}
        </Link>
      </section>
      <section className="paper-card metric-card">
        <span className="folio">Evidence so far</span>
        <div className="large-number">{insights?.reflection_count ?? "—"}</div>
        <p>recorded reflections</p>
        <div className="hairline" />
        <strong>{insights?.completed_outcomes ?? "—"} outcomes completed</strong>
      </section>
      <section className="margin-note">
        <span>Method note 01</span>
        <p>A recommendation is a baseline decision until repeated outcomes justify personalization.</p>
      </section>
    </div>
  );
}
