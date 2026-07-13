"use client";

import { useEffect, useState } from "react";

import { apiRequest } from "@/lib/api";
import type { Insights } from "@/lib/types";

export default function PatternsPage() {
  const [data, setData] = useState<Insights | null>(null);
  useEffect(() => { apiRequest<Insights>("/v1/insights").then(setData).catch(() => setData(null)); }, []);
  const changes = data?.average_state_change ?? {};
  return <div className="page-wrap reveal"><header className="page-header"><div><span className="kicker">Descriptive evidence</span><h1>Patterns with uncertainty attached.</h1></div></header><div className="pattern-grid"><section className="paper-card metric-card"><span className="folio">Completed loop</span><div className="large-number">{data?.completed_outcomes ?? "—"}</div><p>outcomes available for comparison</p></section><section className="paper-card"><span className="folio">Average state movement</span>{["valence", "arousal", "agency"].map((key) => { const value = changes[key] ?? 0; return <div className="pattern-row" key={key}><span>{key}</span><div><i style={{ width: `${Math.min(Math.abs(value) * 100, 100)}%` }} /></div><strong>{value > 0 ? "+" : ""}{value.toFixed(2)}</strong></div>; })}</section><section className="paper-card"><span className="folio">Action observations</span>{Object.entries(data?.action_counts ?? {}).map(([action, count]) => <div className="action-row" key={action}><span>{action}</span><strong>{count}</strong></div>)}{Object.keys(data?.action_counts ?? {}).length === 0 && <p>No action outcomes yet.</p>}</section><aside className="margin-note"><span>Interpretation boundary</span><p>{data?.note ?? "These views remain empty until the API is available."}</p></aside></div></div>;
}
