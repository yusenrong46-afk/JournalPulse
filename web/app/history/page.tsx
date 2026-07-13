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
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=100"),
      apiRequest<{ items: OutcomeRecord[] }>("/v1/outcomes"),
    ])
      .then(([history, recordedOutcomes]) => {
        setItems(history.items);
        setOutcomes(recordedOutcomes.items);
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
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
      <header className="page-header">
        <div className="page-heading-copy"><span className="kicker">Private record</span><h1>History, without a fixed story.</h1><p>Review what you noticed, which actions you chose, and which loops still need an outcome.</p></div>
        <div className="page-counter"><strong>{String(filtered.length).padStart(2, "0")}</strong><span>entries shown</span></div>
      </header>
      <section className="history-summary" aria-label="History summary">
        <div><span>All observations</span><strong>{items.length}</strong></div>
        <div><span>Closed loops</span><strong>{completedIds.size}</strong></div>
        <div><span>Open loops</span><strong>{Math.max(items.length - completedIds.size, 0)}</strong></div>
        <Link href="/reflect">New reflection <span aria-hidden="true">→</span></Link>
      </section>
      <section className="history-controls" aria-label="History filters">
        <label>Search<input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Situation, signal, or phrase" /></label>
        <label>Signal<select value={tag} onChange={(event) => setTag(event.target.value)}><option value="all">All signals</option>{tags.map((item) => <option key={item} value={item}>{item.replaceAll("_", " ")}</option>)}</select></label>
        <label>Loop<select value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">All loops</option><option value="open">Needs check-in</option><option value="closed">Closed</option></select></label>
      </section>
      {error && <p className="error-note">History is unavailable. Nothing new was stored.</p>}
      {loading && <section className="history-loading" aria-label="Loading history"><span /><span /><span /></section>}
      {!loading && <div className="timeline">
        {filtered.map((item, index) => {
          const complete = completedIds.has(item.decision.decision_id);
          return (
            <article className="timeline-entry" key={item.id}>
              <div className="timeline-index">{String(filtered.length - index).padStart(2, "0")}</div>
              <div className="timeline-content">
                <div className="timeline-meta"><time>{new Date(item.created_at).toLocaleDateString("en-CA", { month: "short", day: "numeric", year: "numeric" })}</time><span className={complete ? "loop-state complete" : "loop-state"}>{complete ? "Loop closed" : "Check-in due"}</span></div>
                <h2>{item.reflection.summary}</h2>
                <p>{item.reflection.interpretation}</p>
                <div className="tag-row">{item.state.emotion_tags.map((signal) => <span key={signal}>{signal.replaceAll("_", " ")}</span>)}</div>
                <div className="timeline-footer"><small>Agency {Math.round(item.state.agency * 100)} · {item.decision.selection_source.replaceAll("_", " ")}</small>{!complete && <Link className="text-button" href={`/check-in?decision=${item.decision.decision_id}`}>Complete check-in →</Link>}</div>
              </div>
              <button className="icon-button danger" aria-label={`Delete reflection from ${new Date(item.created_at).toLocaleDateString("en-CA")}`} onClick={() => remove(item.id)}>×</button>
            </article>
          );
        })}
      </div>}
      {!loading && !error && filtered.length === 0 && <section className="paper-card empty-card refined-empty"><span className="folio">Quiet record</span><h2>{items.length ? "No reflections match these filters." : "Your first observation will appear here."}</h2><p>{items.length ? "Change the search, signal, or loop filter to widen the record." : "Begin with one moment worth noticing. You decide what gets saved."}</p><Link className="button primary" href="/reflect">Start a reflection <span aria-hidden="true">→</span></Link></section>}
    </div>
  );
}
