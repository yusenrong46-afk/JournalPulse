"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { apiRequest } from "@/lib/api";
import type { OutcomeRecord, ReflectionRecord } from "@/lib/types";

export default function HistoryPage() {
  const [items, setItems] = useState<ReflectionRecord[]>([]);
  const [outcomes, setOutcomes] = useState<OutcomeRecord[]>([]);
  const [query, setQuery] = useState("");
  const [tag, setTag] = useState("all");
  const [status, setStatus] = useState("all");
  const [error, setError] = useState(false);

  useEffect(() => {
    Promise.all([
      apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=100"),
      apiRequest<{ items: OutcomeRecord[] }>("/v1/outcomes"),
    ])
      .then(([history, recordedOutcomes]) => {
        setItems(history.items);
        setOutcomes(recordedOutcomes.items);
      })
      .catch(() => setError(true));
  }, []);

  const completedIds = useMemo(() => new Set(outcomes.map((item) => item.decision_id)), [outcomes]);
  const tags = useMemo(() => Array.from(new Set(items.flatMap((item) => item.state.emotion_tags))).sort(), [items]);
  const filtered = items.filter((item) => {
    const searchable = [item.reflection.summary, item.reflection.interpretation, ...item.state.emotion_tags, ...Object.values(item.context)].join(" ").toLowerCase();
    const matchesQuery = !query || searchable.includes(query.toLowerCase());
    const matchesTag = tag === "all" || item.state.emotion_tags.includes(tag);
    const complete = completedIds.has(item.decision.decision_id);
    const matchesStatus = status === "all" || (status === "closed" ? complete : !complete);
    return matchesQuery && matchesTag && matchesStatus;
  });

  async function remove(id: string) {
    if (!window.confirm("Delete this reflection and its linked outcome? This cannot be undone.")) return;
    await apiRequest(`/v1/reflections/${id}`, { method: "DELETE" });
    setItems((current) => current.filter((item) => item.id !== id));
  }

  return (
    <div className="page-wrap reveal">
      <header className="page-header"><div><span className="kicker">Private record</span><h1>History, without a fixed story.</h1></div><span>{filtered.length} shown</span></header>
      <section className="history-controls" aria-label="History filters">
        <label>Search<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Situation, signal, or phrase" /></label>
        <label>Signal<select value={tag} onChange={(event) => setTag(event.target.value)}><option value="all">All signals</option>{tags.map((item) => <option key={item} value={item}>{item.replaceAll("_", " ")}</option>)}</select></label>
        <label>Loop<select value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">All loops</option><option value="open">Needs check-in</option><option value="closed">Closed</option></select></label>
      </section>
      {error && <p className="error-note">History is unavailable. Nothing new was stored.</p>}
      <div className="timeline">
        {filtered.map((item, index) => {
          const complete = completedIds.has(item.decision.decision_id);
          return <article className="timeline-entry" key={item.id}><div className="timeline-index">{String(filtered.length - index).padStart(2, "0")}</div><div><time>{new Date(item.created_at).toLocaleDateString("en-CA", { month: "short", day: "numeric", year: "numeric" })}</time><h2>{item.reflection.summary}</h2><p>{item.reflection.interpretation}</p><div className="tag-row">{item.state.emotion_tags.map((signal) => <span key={signal}>{signal.replaceAll("_", " ")}</span>)}</div><small>Agency {item.state.agency.toFixed(2)} · {item.decision.selection_source.replaceAll("_", " ")}</small>{!complete && <Link className="text-button" href={`/check-in?decision=${item.decision.decision_id}`}>Complete check-in</Link>}{complete && <span className="closed-loop">Loop closed</span>}</div><button className="text-button danger" onClick={() => remove(item.id)}>Delete</button></article>;
        })}
      </div>
      {!error && filtered.length === 0 && <section className="paper-card empty-card"><h2>No matching reflections.</h2><p>Adjust the filters or begin a new guided reflection.</p></section>}
    </div>
  );
}
