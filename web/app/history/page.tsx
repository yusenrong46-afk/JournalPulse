"use client";

import { useEffect, useState } from "react";

import { apiRequest } from "@/lib/api";
import type { ReflectionRecord } from "@/lib/types";

export default function HistoryPage() {
  const [items, setItems] = useState<ReflectionRecord[]>([]);
  const [error, setError] = useState(false);
  useEffect(() => { apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=100").then((data) => setItems(data.items)).catch(() => setError(true)); }, []);
  async function remove(id: string) {
    if (!window.confirm("Delete this reflection and its linked outcome? This cannot be undone.")) return;
    await apiRequest(`/v1/reflections/${id}`, { method: "DELETE" });
    setItems((current) => current.filter((item) => item.id !== id));
  }
  return <div className="page-wrap reveal"><header className="page-header"><div><span className="kicker">Private record</span><h1>History, without a fixed story.</h1></div><span>{items.length} observations</span></header>{error && <p className="error-note">History is unavailable. Nothing new was stored.</p>}<div className="timeline">{items.map((item, index) => <article className="timeline-entry" key={item.id}><div className="timeline-index">{String(items.length - index).padStart(2, "0")}</div><div><time>{new Date(item.created_at).toLocaleDateString("en-CA", { month: "short", day: "numeric", year: "numeric" })}</time><h2>{item.reflection.summary}</h2><p>{item.reflection.interpretation}</p><div className="tag-row">{item.state.emotion_tags.map((tag) => <span key={tag}>{tag.replaceAll("_", " ")}</span>)}</div><small>Action: {item.decision.action_id} · Agency {item.state.agency.toFixed(2)}</small></div><button className="text-button danger" onClick={() => remove(item.id)}>Delete</button></article>)}</div>{!error && items.length === 0 && <section className="paper-card empty-card"><h2>No stored reflections.</h2><p>Your history begins only after you explicitly save a guided reflection.</p></section>}</div>;
}
