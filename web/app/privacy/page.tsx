"use client";

import { useState } from "react";

import { apiRequest } from "@/lib/api";
import { usePreferences } from "@/lib/preferences";
import { clearReflectionDraft, loadReflectionDraft } from "@/lib/reflection-draft";

const DELETE_PHRASE = "delete my journal data";

export default function PrivacyPage() {
  const [preferences, updatePreferences, preferencesLoaded] = usePreferences();
  const [confirmation, setConfirmation] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function exportData() {
    setError("");
    try {
      const payload = await apiRequest<Record<string, unknown>>("/v1/export");
      const activeDraft = preferences.encryptedDrafts ? await loadReflectionDraft() : null;
      const completeExport = { ...payload, active_local_draft: activeDraft };
      const href = URL.createObjectURL(
        new Blob([JSON.stringify(completeExport, null, 2)], { type: "application/json" }),
      );
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = "journalpulse-export.json";
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(href), 0);
      setMessage("Your export was created locally in this browser.");
    } catch {
      setError("The export could not be created. Your stored data was not changed.");
    }
  }

  async function deleteData() {
    if (confirmation !== DELETE_PHRASE) return;
    setBusy(true);
    setError("");
    try {
      const result = await apiRequest<{ deleted_records: number }>("/v1/account/data", {
        method: "DELETE",
      });
      await clearReflectionDraft();
      setConfirmation("");
      setMessage(`${result.deleted_records} stored records were deleted. Your sign-in remains active.`);
    } catch {
      setError("Deletion did not complete. No success is assumed; please retry.");
    } finally {
      setBusy(false);
    }
  }

  async function setDraftRecovery(enabled: boolean) {
    updatePreferences({ ...preferences, encryptedDrafts: enabled });
    if (!enabled) await clearReflectionDraft();
  }

  return (
    <div className="page-wrap narrow reveal">
      <header className="page-header">
        <div className="page-heading-copy">
          <span className="kicker">Data boundary</span>
          <h1>Your reflections are not the product.</h1>
          <p>Control analysis, retention, local recovery, export, and deletion from one place.</p>
        </div>
      </header>
      <section className="privacy-dashboard" aria-label="Privacy status">
        <article><span>AI analysis</span><strong>{preferences.llmConsent ? "Allowed" : "Off"}</strong><small>Per-entry control</small></article>
        <article><span>Original text</span><strong>{preferences.retainText ? "Retained" : "Not retained"}</strong><small>Default setting</small></article>
        <article><span>Draft recovery</span><strong>{preferences.encryptedDrafts ? "Encrypted" : "Off"}</strong><small>This device only</small></article>
      </section>
      <section className="flow-sheet">
        <span className="folio">Processing choices</span>
        {preferencesLoaded && (
          <div className="preference-panel">
            <label><span><strong>Private AI analysis</strong><small>Default for new reflections</small></span><input type="checkbox" checked={preferences.llmConsent} onChange={(event) => updatePreferences({ ...preferences, llmConsent: event.target.checked })} /></label>
            <label><span><strong>Retain original text</strong><small>Summaries, interpretations, and situation notes can still be stored on the server</small></span><input type="checkbox" checked={preferences.retainText} onChange={(event) => updatePreferences({ ...preferences, retainText: event.target.checked })} /></label>
            <label><span><strong>Recover unfinished drafts</strong><small>Encrypted on this device and removed after 24 hours</small></span><input type="checkbox" checked={preferences.encryptedDrafts} onChange={(event) => void setDraftRecovery(event.target.checked)} /></label>
            <label><span><strong>Default follow-up window</strong><small>Used when recording elapsed time</small></span><select value={preferences.followUpMinutes} onChange={(event) => updatePreferences({ ...preferences, followUpMinutes: Number(event.target.value) })}><option value="5">5 minutes</option><option value="10">10 minutes</option><option value="20">20 minutes</option><option value="60">1 hour</option></select></label>
          </div>
        )}
        <div className="privacy-list">
          <article>
            <strong>AI analysis</strong>
            <p>Your saved default can still be changed for each reflection. Requests require zero-data-retention routing.</p>
          </article>
          <article>
            <strong>Original text</strong>
            <p>Off by default. Turning this off omits the original entry. Summaries, interpretations, and the situation you provide can still be saved on the server. Preview storage is the server database, not a private copy that exists only in this browser.</p>
          </article>
          <article>
            <strong>Personal memory</strong>
            <p>Disabled until retrieval, provenance, and deletion behavior are validated.</p>
          </article>
          <article>
            <strong>Safety</strong>
            <p>Support routing happens before AI analysis, retrieval, or adaptive selection.</p>
          </article>
        </div>
        <div className="button-row">
          <button className="button secondary" onClick={exportData} type="button">
            Export my data
          </button>
        </div>
      </section>
      <section className="flow-sheet danger-zone" aria-labelledby="delete-data-heading">
        <span className="folio">Irreversible</span>
        <h2 id="delete-data-heading">Delete all journal data</h2>
        <p>
          This removes reflections, outcomes, model records, safety events, and memories. It does
          not remove your sign-in identity.
        </p>
        <label className="field-label">
          Type “{DELETE_PHRASE}” to confirm
          <input
            autoComplete="off"
            onChange={(event) => setConfirmation(event.target.value)}
            value={confirmation}
          />
        </label>
        <button
          className="button urgent"
          disabled={busy || confirmation !== DELETE_PHRASE}
          onClick={deleteData}
          type="button"
        >
          {busy ? "Deleting…" : "Delete journal data"}
        </button>
      </section>
      {message && <p className="success-note" role="status">{message}</p>}
      {error && <p className="error-note" role="alert">{error}</p>}
    </div>
  );
}
