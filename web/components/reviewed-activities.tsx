"use client";

import { useEffect, useState } from "react";
import { apiRequest } from "@/lib/api";
import type { ActivityConstraints } from "@/lib/types";

type ReviewedActivity = { id: string; title: string; summary: string; url: string | null; duration_minutes: number | null };

export function ReviewedActivities({ goal, constraints, onChoose, disabled = false }: {
  goal?: string | null; constraints?: ActivityConstraints; onChoose?(id: string): void; disabled?: boolean;
}) {
  const [items, setItems] = useState<ReviewedActivity[] | null>(null);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const query = new URLSearchParams();
  if (goal && ["settle", "move", "understand", "connect", "act"].includes(goal)) query.set("goal", goal);
  if (constraints) for (const [name, value] of Object.entries(constraints)) if (value != null) query.set(name, String(value));
  const queryString = query.toString();
  useEffect(() => {
    const controller = new AbortController();
    void apiRequest<{ items: ReviewedActivity[] }>(`/v1/activity-resources?${queryString}`, { signal: controller.signal })
      .then((result) => { if (!controller.signal.aborted) { setItems(result.items); setError(false); } })
      .catch(() => { if (!controller.signal.aborted) setError(true); });
    return () => controller.abort();
  }, [queryString, attempt]);
  return <section className="card stack" aria-label="Reviewed app activities">
    <h2>Choose a reviewed app activity</h2>
    <p className="small muted">A limited collection you can choose from yourself. No web search or AI is used to browse it.</p>
    {error ? <><p role="status">We couldn’t load the collection. Your writing and chat are still available.</p>
      <button className="btn btn-soft" type="button" onClick={() => setAttempt((value) => value + 1)}>Reload app activities</button></>
      : !items ? <p role="status">Loading app activities…</p> : !items.length ? <p>No reviewed activities fit these preferences. You can change your activity preferences or keep talking.</p>
      : <ul className="stack">{items.map((item) => <li key={item.id} className="stack">
        <strong>{item.title}</strong><p className="small muted">{item.summary}</p>
        {item.url && <a href={item.url} target="_blank" rel="noopener noreferrer">Open resource</a>}
        {onChoose && <button className="btn btn-soft" type="button" disabled={disabled} onClick={() => onChoose(item.id)}>Choose {item.title}</button>}
      </li>)}</ul>}
    {!onChoose && <p className="small muted">Opening a resource does not save a choice or mark it completed. You can write your own small step in your journal.</p>}
  </section>;
}
