"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { StateControls } from "@/components/state-controls";
import { apiRequest } from "@/lib/api";
import { usePreferences } from "@/lib/preferences";
import type {
  ActionPreview,
  AffectiveState,
  PreparedAnalysis,
  ReflectionRecord,
  Resource,
  TargetState,
} from "@/lib/types";

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
  const [preferences] = usePreferences();
  const [step, setStep] = useState(1);
  const [text, setText] = useState("");
  const [situation, setSituation] = useState("");
  const [energy, setEnergy] = useState("medium");
  const [socialContext, setSocialContext] = useState("alone");
  const [consentOverride, setConsentOverride] = useState<boolean | null>(null);
  const [retainOverride, setRetainOverride] = useState<boolean | null>(null);
  const [analysis, setAnalysis] = useState<PreparedAnalysis | null>(null);
  const [state, setState] = useState(initialState);
  const [newTag, setNewTag] = useState("");
  const [target, setTarget] = useState<TargetState>({
    goal: "settle",
    valence: 0,
    arousal: 0.35,
    agency: 0.65,
  });
  const [preview, setPreview] = useState<ActionPreview | null>(null);
  const [selectedAction, setSelectedAction] = useState("");
  const [record, setRecord] = useState<ReflectionRecord | null>(null);
  const [resource, setResource] = useState<Resource | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const progress = useMemo(() => `${Math.min(step, 5)}/5`, [step]);
  const consent = consentOverride ?? preferences.llmConsent;
  const retain = retainOverride ?? preferences.retainText;
  const context = useMemo(
    () => ({ situation, energy, social_context: socialContext }),
    [energy, situation, socialContext],
  );

  function addTag() {
    const tag = newTag.trim().toLowerCase().replaceAll(" ", "_");
    if (!tag || state.emotion_tags.includes(tag) || state.emotion_tags.length >= 6) return;
    setState({ ...state, emotion_tags: [...state.emotion_tags, tag] });
    setNewTag("");
  }

  async function analyze() {
    if (text.trim().length < 8) {
      setError("Write at least one complete thought before continuing.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const result = await apiRequest<PreparedAnalysis>("/v1/reflections/analyze", {
        method: "POST",
        body: JSON.stringify({ text, context, llm_consent: consent, locale: preferences.locale }),
      });
      setAnalysis(result);
      setState({ ...result.state, confidence: 1 });
      setStep(result.safety.mode === "support" ? 5 : 2);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Analysis failed.");
    } finally {
      setLoading(false);
    }
  }

  async function previewActions() {
    if (!analysis) return;
    setLoading(true);
    setError(null);
    try {
      const result = await apiRequest<ActionPreview>("/v1/actions/preview", {
        method: "POST",
        body: JSON.stringify({
          state,
          target,
          context,
          resource_intent: analysis.resource_intent,
        }),
      });
      setPreview(result);
      setSelectedAction(result.decision.action_id);
      setStep(4);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Safe actions are unavailable.");
    } finally {
      setLoading(false);
    }
  }

  async function saveChoice() {
    if (!analysis || !preview || !selectedAction) return;
    setLoading(true);
    setError(null);
    try {
      const saved = await apiRequest<ReflectionRecord>("/v1/reflections", {
        method: "POST",
        body: JSON.stringify({
          text,
          context,
          self_report: state,
          target,
          llm_consent: consent,
          retain_text: retain,
          locale: preferences.locale,
          prepared_analysis: analysis,
          chosen_action_id: selectedAction,
        }),
      });
      setRecord(saved);
      setResource(preview.actions.find((item) => item.id === saved.decision.action_id) ?? null);
      setStep(5);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The action could not be saved.");
    } finally {
      setLoading(false);
    }
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
          <label className="field-label">Optional situation<input value={situation} onChange={(event) => setSituation(event.target.value)} placeholder="After work, before sleep, with someone…" /></label>
          <div className="context-grid">
            <label>Energy<select value={energy} onChange={(event) => setEnergy(event.target.value)}><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></label>
            <label>Social context<select value={socialContext} onChange={(event) => setSocialContext(event.target.value)}><option value="alone">Alone</option><option value="with_others">With others</option><option value="after_contact">After contact</option></select></label>
          </div>
          <details className="processing-details">
            <summary>Processing choices for this entry</summary>
            <div className="consent-box">
              <label><input type="checkbox" checked={consent} onChange={(event) => setConsentOverride(event.target.checked)} /><span><strong>Use private AI analysis</strong><small>Send through a zero-data-retention route when configured.</small></span></label>
              <label><input type="checkbox" checked={retain} onChange={(event) => setRetainOverride(event.target.checked)} /><span><strong>Keep my original text</strong><small>Otherwise only your approved structured state is saved.</small></span></label>
            </div>
          </details>
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
          <div className="tag-editor">
            <div className="tag-row">{state.emotion_tags.map((tag) => <button key={tag} onClick={() => setState({ ...state, emotion_tags: state.emotion_tags.filter((item) => item !== tag) })}>{tag.replaceAll("_", " ")} ×</button>)}</div>
            <label>Add your own signal<input value={newTag} onChange={(event) => setNewTag(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); addTag(); } }} placeholder="for example: disappointed" /></label>
            <button className="text-button" type="button" onClick={addTag}>Add tag</button>
          </div>
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
          <button className="button primary" onClick={previewActions} disabled={loading}>{loading ? "Checking the reviewed catalog…" : "Show safe options"}</button>
        </section>
      )}

      {step === 4 && preview && (
        <section className="flow-sheet">
          <span className="folio">04 / Choose</span>
          <h2>Choose the action you are actually willing to try.</h2>
          <p>The first option is the transparent baseline recommendation. Choosing another is recorded as your decision, not model performance.</p>
          <div className="resource-choice-grid">
            {preview.actions.map((item) => {
              const recommended = item.id === preview.decision.action_id;
              return <button key={item.id} className={selectedAction === item.id ? "resource-choice active" : "resource-choice"} onClick={() => setSelectedAction(item.id)}><span className="resource-meta">{recommended ? "Baseline pick" : "Safe alternative"} · {item.duration_minutes ?? "—"} min</span><strong>{item.title}</strong><p>{item.summary}</p><small>{item.provider} · {item.resource_type}</small></button>;
            })}
          </div>
          <button className="button primary" onClick={saveChoice} disabled={loading || !selectedAction}>{loading ? "Saving your choice…" : "Use this action"}</button>
        </section>
      )}

      {step === 5 && analysis?.safety.mode === "support" && !record && (
        <section className="flow-sheet support-sheet"><span className="folio">Support mode</span><h2>Human support comes first.</h2><p>{analysis.safety.support_message}</p><a className="button urgent" href="https://988.ca/" target="_blank" rel="noreferrer">Open 9-8-8 Canada</a></section>
      )}

      {step === 5 && record && (
        <section className="flow-sheet action-sheet">
          <span className="folio">05 / Act</span>
          <div className="action-duration">{resource?.duration_minutes ?? "—"}<small>minutes</small></div>
          <h2>{resource?.title ?? "Take a deliberate pause"}</h2>
          <p>{resource?.summary ?? "Step away briefly and notice what changes without forcing it."}</p>
          <div className="evidence-slip"><span>Why this</span><p>{record.decision.explanation}</p><small>{record.decision.selection_source.replaceAll("_", " ")} · {record.decision.eligible_for_ope ? "eligible for policy evaluation" : "excluded from policy evaluation"}</small></div>
          <div className="button-row">
            {resource && <a className="button primary" href={resource.url} target="_blank" rel="noreferrer">Open {resource.resource_type}</a>}
            <Link className="button secondary" href={`/check-in?decision=${record.decision.decision_id}`}>Check in afterward</Link>
          </div>
          <Link className="text-button" href="/">I’ll check in later</Link>
        </section>
      )}
      {error && <p className="error-note" role="alert">{error}</p>}
    </div>
  );
}
