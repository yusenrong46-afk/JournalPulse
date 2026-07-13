"use client";

import { useEffect, useState } from "react";

import { TrajectoryChart } from "@/components/trajectory-chart";
import { apiRequest } from "@/lib/api";
import type { Insights } from "@/lib/types";

export default function PatternsPage() {
  const [data, setData] = useState<Insights | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => { apiRequest<Insights>("/v1/insights").then(setData).catch(() => setError(true)); }, []);
  const changes = data?.average_state_change ?? {};

  return (
    <div className="page-wrap reveal">
      <header className="page-header"><div><span className="kicker">Descriptive evidence</span><h1>Patterns with uncertainty attached.</h1></div></header>
      {error && <p className="error-note">Patterns are unavailable while the API is offline.</p>}
      <div className="pattern-grid">
        <section className="paper-card metric-card"><span className="folio">Completed loop</span><div className="large-number">{data?.completed_outcomes ?? "—"}</div><p>outcomes available for comparison</p><div className="hairline" /><strong>{Math.round((data?.completion_rate ?? 0) * 100)}% completion rate</strong></section>
        <section className="paper-card trajectory-card"><span className="folio">State trajectory</span><h2>Movement across reflections</h2><TrajectoryChart points={data?.state_trajectory ?? []} /></section>
        <section className="paper-card"><span className="folio">Average post-action movement</span>{["valence", "arousal", "agency"].map((key) => { const value = changes[key] ?? 0; return <div className="pattern-row" key={key}><span>{key}</span><div><i style={{ width: `${Math.min(Math.abs(value) * 100, 100)}%` }} /></div><strong>{value > 0 ? "+" : ""}{value.toFixed(2)}</strong></div>; })}</section>
        <section className="paper-card"><span className="folio">Action observations</span>{Object.entries(data?.action_counts ?? {}).map(([action, count]) => <div className="action-row" key={action}><span>{action.replaceAll("_", " ")}</span><strong>{count} selected · {data?.average_helpfulness_by_action[action]?.toFixed(1) ?? "—"}/5 helpful</strong></div>)}{Object.keys(data?.action_counts ?? {}).length === 0 && <p>No action outcomes yet.</p>}</section>
        <aside className="margin-note"><span>Interpretation boundary</span><p>{data?.note ?? "These views remain empty until observations exist."}</p></aside>
      </div>
    </div>
  );
}
