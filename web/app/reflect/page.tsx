"use client";

import { useMemo, useState } from "react";

import { StateControls } from "@/components/state-controls";
import { apiRequest } from "@/lib/api";
import type { AffectiveState, PreparedAnalysis, ReflectionRecord, Resource, TargetState } from "@/lib/types";

const initialState: AffectiveState = {
  valence: 0,
  arousal: 0.5,
  agency: 0.5,
  emotion_tags: [],
  confidence: 1,
};

const goals = [
  ["settle", "Lower the noise"],
  ["move", "Regain momentum"],
  ["understand", "Understand the signal"],
  ["connect", "Feel less alone"],
  ["act", "Prepare one next step"],
];

export default function ReflectPage() {
  const [step, setStep] = useState(1);
  const [text, setText] = useState("");
  const [context, setContext] = useState("");
  const [consent, setConsent] = useState(false);
  const [retain, setRetain] = useState(false);
  const [analysis, setAnalysis] = useState<PreparedAnalysis | null>(null);
  const [state, setState] = useState(initialState);
  const [target, setTarget] = useState<TargetState>({ goal: "settle", valence: 0, arousal: 0.35, agency: 0.65 });
  const [record, setRecord] = useState<ReflectionRecord | null>(null);
  const [resource, setResource] = useState<Resource | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const progress = useMemo(() => `${Math.min(step, 5)}/5`, [step]);

  async function analyze() {
    if (text.trim().length < 8) return setError("Write at least one complete thought before continuing.");
    setLoading(true);
    setError(null);
    try {
      const result = await apiRequest<PreparedAnalysis>("/v1/reflections/analyze", {
        method: "POST",
        body: JSON.stringify({ text, context: context ? { situation: context } : {}, llm_consent: consent, locale: "CA" }),
      });
      setAnalysis(result);
      setState({ ...result.state, confidence: 1 });
      setStep(result.safety.mode === "support" ? 4 : 2);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  async function decide() {
    if (!analysis) return;
    setLoading(true);
    setError(null);
    try {
      const saved = await apiRequest<ReflectionRecord>("/v1/reflections", {
        method: "POST",
        body: JSON.stringify({
          text,
          context: context ? { situation: context } : {},
          self_report: state,
          target,
          llm_consent: consent,
          retain_text: retain,
          locale: "CA",
          prepared_analysis: analysis,
        }),
      });
      setRecord(saved);
      const catalog = await apiRequest<{ items: Resource[] }>("/v1/resources");
      setResource(catalog.items.find((item) => item.id === saved.decision.action_id) ?? null);
      setStep(4);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Decision failed.");
    } finally {
      setLoading(false);
    }
  }

  async function submitOutcome(form: FormData) {
    if (!record) return;
    setLoading(true);
    await apiRequest("/v1/outcomes", {
      method: "POST",
      body: JSON.stringify({
        decision_id: record.decision.decision_id,
        completed: form.get("completed") === "yes",
        post_state: state,
        helpfulness: Number(form.get("helpfulness")),
        effort: Number(form.get("effort")),
        elapsed_minutes: Number(form.get("elapsed")),
      }),
    });
    setLoading(false);
    setStep(5);
  }

  return (
    <div className="page-wrap narrow reveal">
      <header className="flow-header">
        <div><span className="kicker">Guided reflection</span><h1>One signal at a time.</h1></div>
        <div className="step-counter"><span>{progress}</span><small>field sequence</small></div>
      </header>
      <div className="progress-track"><span style={{ width: `${Math.min(step, 5) * 20}%` }} /></div>

      {step === 1 && (
        <section className="flow-sheet">
          <span className="folio">01 / Observe</span>
          <h2>What happened, what feeling is strongest, and what still feels unresolved?</h2>
          <textarea value={text} onChange={(event) => setText(event.target.value)} placeholder="Write without trying to sound composed…" autoFocus />
          <label className="field-label">Optional situation<input value={context} onChange={(event) => setContext(event.target.value)} placeholder="After work, before sleep, with someone…" /></label>
          <div className="consent-box">
            <label><input type="checkbox" checked={consent} onChange={(event) => setConsent(event.target.checked)} /><span><strong>Use private AI analysis</strong><small>Send this entry through an OpenRouter zero-data-retention route.</small></span></label>
            <label><input type="checkbox" checked={retain} onChange={(event) => setRetain(event.target.checked)} /><span><strong>Keep my original text</strong><small>Otherwise only your approved structured state is saved.</small></span></label>
          </div>
          <button className="button primary" onClick={analyze} disabled={loading}>{loading ? "Reading carefully…" : "Check the signal"}</button>
        </section>
      )}

      {step === 2 && analysis && (
        <section className="flow-sheet">
          <span className="folio">02 / Correct</span>
          <h2>The system’s read is a proposal, not a verdict.</h2>
          <blockquote>{analysis.reflection.summary}</blockquote>
          <p className="interpretation">{analysis.reflection.interpretation}</p>
          <StateControls state={state} onChange={setState} />
          <div className="tag-row">{state.emotion_tags.map((tag) => <span key={tag}>{tag.replaceAll("_", " ")}</span>)}</div>
          {state.uncertainty && <p className="method-note">Uncertainty: {state.uncertainty}</p>}
          <button className="button primary" onClick={() => setStep(3)}>This reflects me</button>
        </section>
      )}

      {step === 3 && (
        <section className="flow-sheet">
          <span className="folio">03 / Orient</span>
          <h2>What would feel meaningfully different?</h2>
          <div className="goal-grid">{goals.map(([value, label]) => <button key={value} className={target.goal === value ? "goal active" : "goal"} onClick={() => setTarget({ ...target, goal: value })}><span>{label}</span><small>{value}</small></button>)}</div>
          <div className="target-row"><label>Desired activation<input type="range" min="0" max="1" step="0.05" value={target.arousal ?? 0.35} onChange={(event) => setTarget({ ...target, arousal: Number(event.target.value) })} /></label><label>Desired agency<input type="range" min="0" max="1" step="0.05" value={target.agency ?? 0.65} onChange={(event) => setTarget({ ...target, agency: Number(event.target.value) })} /></label></div>
          <button className="button primary" onClick={decide} disabled={loading}>{loading ? "Choosing from safe options…" : "Find one next move"}</button>
        </section>
      )}

      {step === 4 && analysis?.safety.mode === "support" && !record && (
        <section className="flow-sheet support-sheet"><span className="folio">Support mode</span><h2>Human support comes first.</h2><p>{analysis.safety.support_message}</p><a className="button urgent" href="https://988.ca/" target="_blank" rel="noreferrer">Open 9-8-8 Canada</a></section>
      )}

      {step === 4 && record && (
        <section className="flow-sheet">
          <span className="folio">04 / Act</span>
          <h2>{resource?.title ?? "Take a deliberate pause"}</h2>
          <p>{resource?.summary ?? "Step away for two minutes and notice what changes without forcing it."}</p>
          <div className="evidence-slip"><span>Why this</span><p>{record.decision.explanation}</p><small>{record.decision.policy_name} · propensity {record.decision.propensity.toFixed(2)}</small></div>
          {resource && <a className="button primary" href={resource.url} target="_blank" rel="noreferrer">Open {resource.resource_type}</a>}
          <details><summary>Record a check-in now</summary><form action={submitOutcome} className="outcome-form"><label>Did you try it?<select name="completed" defaultValue="yes"><option value="yes">Yes</option><option value="no">Not yet</option></select></label><label>Helpfulness, 1–5<input name="helpfulness" type="number" min="1" max="5" defaultValue="3" /></label><label>Effort, 1–5<input name="effort" type="number" min="1" max="5" defaultValue="2" /></label><label>Minutes elapsed<input name="elapsed" type="number" min="0" defaultValue="10" /></label><button className="button secondary" disabled={loading}>Save outcome</button></form></details>
        </section>
      )}

      {step === 5 && <section className="flow-sheet complete-sheet"><span className="folio">05 / Learn</span><h2>Outcome recorded.</h2><p>This is one observation, not a conclusion. Repeated outcomes are what allow the policy to learn responsibly.</p><button className="button primary" onClick={() => window.location.assign("/")}>Return to today</button></section>}
      {error && <p className="error-note" role="alert">{error}</p>}
    </div>
  );
}
