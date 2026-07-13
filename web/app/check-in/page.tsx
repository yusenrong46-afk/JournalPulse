"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useState } from "react";

import { StateControls } from "@/components/state-controls";
import { apiRequest } from "@/lib/api";
import { usePreferences } from "@/lib/preferences";
import type { AffectiveState, OutcomeRecord, ReflectionRecord, Resource } from "@/lib/types";

function CheckInWorkspace() {
  const [preferences] = usePreferences();
  const searchParams = useSearchParams();
  const requestedDecision = searchParams.get("decision");
  const [reflection, setReflection] = useState<ReflectionRecord | null>(null);
  const [resource, setResource] = useState<Resource | null>(null);
  const [postState, setPostState] = useState<AffectiveState | null>(null);
  const [existing, setExisting] = useState<OutcomeRecord | null>(null);
  const [completed, setCompleted] = useState(true);
  const [helpfulness, setHelpfulness] = useState(3);
  const [effort, setEffort] = useState(2);
  const [elapsedOverride, setElapsedOverride] = useState<number | null>(null);
  const [note, setNote] = useState("");
  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const elapsed = elapsedOverride ?? preferences.followUpMinutes;

  useEffect(() => {
    Promise.all([
      apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=100"),
      apiRequest<{ items: OutcomeRecord[] }>("/v1/outcomes"),
      apiRequest<{ items: Resource[] }>("/v1/resources"),
    ])
      .then(([history, outcomes, catalog]) => {
        const completedIds = new Set(outcomes.items.map((item) => item.decision_id));
        const selected = requestedDecision
          ? history.items.find((item) => item.decision.decision_id === requestedDecision)
          : history.items.find((item) => !completedIds.has(item.decision.decision_id));
        if (!selected) return;
        setReflection(selected);
        setPostState({ ...selected.state, confidence: 1 });
        setExisting(outcomes.items.find((item) => item.decision_id === selected.decision.decision_id) ?? null);
        setResource(catalog.items.find((item) => item.id === selected.decision.action_id) ?? null);
      })
      .catch(() => setError("The check-in could not be loaded."))
      .finally(() => setLoading(false));
  }, [requestedDecision]);

  async function submit() {
    if (!reflection || !postState) return;
    setLoading(true);
    setError("");
    try {
      await apiRequest<OutcomeRecord>("/v1/outcomes", {
        method: "POST",
        body: JSON.stringify({
          decision_id: reflection.decision.decision_id,
          completed,
          post_state: postState,
          helpfulness,
          effort,
          elapsed_minutes: elapsed,
          note: note || null,
        }),
      });
      setSaved(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The check-in could not be saved.");
    } finally {
      setLoading(false);
    }
  }

  if (loading && !reflection) return <div className="page-wrap narrow"><section className="paper-card skeleton-card" aria-label="Loading check-in" /></div>;
  if (error && !reflection) return <div className="page-wrap narrow"><p className="error-note">{error}</p></div>;
  if (!reflection || !postState) return <div className="page-wrap narrow"><section className="paper-card empty-card"><span className="folio">No pending loop</span><h2>Nothing needs a check-in.</h2><p>Complete a reflection and choose an action first.</p><Link className="button primary" href="/reflect">Start a reflection</Link></section></div>;
  if (existing || saved) return <div className="page-wrap narrow"><section className="flow-sheet complete-sheet"><span className="folio">Loop closed</span><h2>One observation recorded.</h2><p>This result informs your descriptive patterns. It does not prove that the action caused the change.</p><div className="button-row"><Link className="button primary" href="/patterns">Review patterns</Link><Link className="button secondary" href="/">Return to today</Link></div></section></div>;

  return (
    <div className="page-wrap narrow reveal">
      <header className="flow-header"><div><span className="kicker">Delayed outcome</span><h1>What changed after the action?</h1></div></header>
      <section className="flow-sheet">
        <span className="folio">Action recalled</span>
        <h2>{resource?.title ?? reflection.decision.action_id}</h2>
        <p>{reflection.reflection.summary}</p>
        <div className="check-in-baseline"><span>Before</span><strong>Valence {reflection.state.valence.toFixed(2)}</strong><strong>Activation {reflection.state.arousal.toFixed(2)}</strong><strong>Agency {reflection.state.agency.toFixed(2)}</strong></div>
        <StateControls state={postState} onChange={setPostState} />
        <div className="outcome-form">
          <label>Did you try it?<select value={completed ? "yes" : "no"} onChange={(event) => setCompleted(event.target.value === "yes")}><option value="yes">Yes</option><option value="no">Not yet</option></select></label>
          <label>Helpfulness, 1–5<input value={helpfulness} onChange={(event) => setHelpfulness(Number(event.target.value))} type="number" min="1" max="5" /></label>
          <label>Effort, 1–5<input value={effort} onChange={(event) => setEffort(Number(event.target.value))} type="number" min="1" max="5" /></label>
          <label>Minutes elapsed<input value={elapsed} onChange={(event) => setElapsedOverride(Number(event.target.value))} type="number" min="0" max="10080" /></label>
        </div>
        <label className="field-label">Optional observation<input value={note} onChange={(event) => setNote(event.target.value)} placeholder="What helped, resisted, or surprised you?" /></label>
        <button className="button primary" onClick={submit} disabled={loading}>{loading ? "Recording…" : "Record this outcome"}</button>
        {error && <p className="error-note" role="alert">{error}</p>}
      </section>
    </div>
  );
}

export default function CheckInPage() {
  return <Suspense fallback={<div className="page-wrap narrow"><section className="paper-card skeleton-card" /></div>}><CheckInWorkspace /></Suspense>;
}
