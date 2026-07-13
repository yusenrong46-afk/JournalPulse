"use client";

import { useEffect, useState } from "react";

import { TrajectoryChart } from "@/components/trajectory-chart";
import { apiRequest } from "@/lib/api";
import type { Insights } from "@/lib/types";

export default function PatternsPage() {
  const [data, setData] = useState<Insights | null>(null);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(true);
  useEffect(() => { apiRequest<Insights>("/v1/insights").then(setData).catch(() => setError(true)).finally(() => setLoading(false)); }, []);
  const changes = data?.average_state_change ?? {};

  return (
    <div className="page-wrap reveal">
      <header className="page-header"><div className="page-heading-copy"><span className="kicker">Descriptive evidence</span><h1>Patterns, with uncertainty attached.</h1><p>See what repeatedly changes around your actions without turning correlation into a claim.</p></div><div className="page-counter"><strong>{String(data?.reflection_count ?? 0).padStart(2, "0")}</strong><span>observations</span></div></header>
      {error && <p className="error-note">Patterns are unavailable while the API is offline.</p>}
      {!error && <section className="evidence-boundary"><span>How to read this</span><p>{data?.note ?? "JournalPulse waits for your observations before drawing a pattern."}</p><strong>{loading ? "Loading evidence…" : `${data?.completed_outcomes ?? 0} completed loops`}</strong></section>}
      <div className="pattern-grid" aria-busy={loading}>
        <section className="paper-card metric-card pattern-metric"><span className="folio">Completed loops</span><div className="large-number">{data?.completed_outcomes ?? "—"}</div><p>outcomes available for comparison</p><div className="completion-meter"><span style={{ width: `${Math.round((data?.completion_rate ?? 0) * 100)}%` }} /></div><strong>{Math.round((data?.completion_rate ?? 0) * 100)}% completion rate</strong></section>
        <section className="paper-card trajectory-card"><span className="folio">State trajectory</span><h2>Movement across reflections</h2><TrajectoryChart points={data?.state_trajectory ?? []} /></section>
        <section className="paper-card state-change-card"><span className="folio">Average post-action movement</span><h2>What moved afterward</h2>{["valence", "arousal", "agency"].map((key) => { const value = changes[key] ?? 0; return <div className="pattern-row" key={key}><span>{key}</span><div><i style={{ width: `${Math.min(Math.abs(value) * 100, 100)}%` }} /></div><strong>{value > 0 ? "+" : ""}{value.toFixed(2)}</strong></div>; })}<small className="chart-footnote">Direction only. Small samples can move this average sharply.</small></section>
        <section className="paper-card action-observation-card"><span className="folio">Action observations</span><h2>What you actually tried</h2>{Object.entries(data?.action_counts ?? {}).map(([action, count]) => <div className="action-row" key={action}><span>{action.replaceAll("_", " ")}</span><strong>{count}× <small>{data?.average_helpfulness_by_action[action]?.toFixed(1) ?? "—"}/5 helpful</small></strong></div>)}{Object.keys(data?.action_counts ?? {}).length === 0 && <div className="inline-empty"><strong>No action outcomes yet.</strong><span>Close a reflection loop to begin this view.</span></div>}</section>
      </div>
    </div>
  );
}
