"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { Luna } from "@/components/luna";
import { apiRequest } from "@/lib/api";
import { usePreferences } from "@/lib/preferences";
import { clearReflectionDraft } from "@/lib/reflection-draft";
import { getSupabase } from "@/lib/supabase";

const DELETE_PHRASE = "delete my journal";

export default function MePage() {
  const router = useRouter();
  const [preferences, updatePreferences, preferencesLoaded] = usePreferences();
  const [confirmation, setConfirmation] = useState("");
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function exportData() {
    setError("");
    try {
      const payload = await apiRequest<Record<string, unknown>>("/v1/export");
      const href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
      const anchor = document.createElement("a");
      anchor.href = href;
      anchor.download = "journalpulse-export.json";
      anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(href), 0);
      setMessage("Your data was downloaded to this device.");
    } catch {
      setError("The download didn’t work. Nothing was changed.");
    }
  }

  async function deleteData() {
    if (confirmation.trim().toLowerCase() !== DELETE_PHRASE) return;
    setBusy(true);
    setError("");
    try {
      const result = await apiRequest<{ deleted_records: number }>("/v1/account/data", { method: "DELETE" });
      await clearReflectionDraft();
      setConfirmation("");
      setMessage(`Done. ${result.deleted_records} saved items were deleted. You’re still signed in.`);
    } catch {
      setError("Deletion didn’t finish. Nothing is assumed deleted; please try again.");
    } finally {
      setBusy(false);
    }
  }

  async function signOut() {
    const client = await getSupabase();
    await client?.auth.signOut();
    router.replace("/login");
  }

  return (
    <div className="page">
      <header className="page-head row" style={{ gap: 14 }}>
        <Luna mood="idle" size={72} />
        <div>
          <h1>Your space</h1>
          <p>Choose how Luna works with you. You can change these anytime.</p>
        </div>
      </header>

      {preferencesLoaded && (
        <section className="settings" aria-label="Chat settings">
          <label className="setting">
            <span><strong>Let Luna use AI</strong><small>Smarter, more personal replies. Sent privately; the AI provider keeps nothing. Off means Luna asks simple questions instead.</small></span>
            <span className="switch"><input type="checkbox" checked={preferences.llmConsent} onChange={(event) => updatePreferences({ ...preferences, llmConsent: event.target.checked })} /><span /></span>
          </label>
          <label className="setting">
            <span><strong>Keep my messages</strong><small>Off clears the words of a chat when it ends. A short summary and your choice are still saved.</small></span>
            <span className="switch"><input type="checkbox" checked={preferences.retainText} onChange={(event) => updatePreferences({ ...preferences, retainText: event.target.checked })} /><span /></span>
          </label>
          <label className="setting">
            <span><strong>Check in after</strong><small>When Luna asks how your small step went.</small></span>
            <select value={preferences.followUpMinutes} onChange={(event) => updatePreferences({ ...preferences, followUpMinutes: Number(event.target.value) })}>
              <option value="5">5 minutes</option>
              <option value="10">10 minutes</option>
              <option value="20">20 minutes</option>
              <option value="60">1 hour</option>
            </select>
          </label>
        </section>
      )}

      <section className="stack" aria-labelledby="how-heading">
        <h2 id="how-heading">How Luna works</h2>
        <details className="explain">
          <summary>Luna is a companion, not a therapist</summary>
          <div>
            <p>Luna helps you notice how you feel and try one small thing. It doesn’t diagnose or treat anything.</p>
            <p>If you’re in crisis in Canada, call or text <a href="https://988.ca/" target="_blank" rel="noreferrer">9-8-8</a>. In an emergency, call 911.</p>
          </div>
        </details>
        <details className="explain">
          <summary>Safety comes first</summary>
          <div>
            <p>Every message is checked for signs of crisis before anything else happens. If there are signs, Luna skips the AI and points you to people who can help right away.</p>
          </div>
        </details>
        <details className="explain">
          <summary>Where Luna’s ideas come from</summary>
          <div>
            <p>Luna only suggests activities from a reviewed list, like breathing exercises, short walks, or reading from trusted health sites. The AI never invents links.</p>
            <p>Luna’s pick comes from a simple, fixed rule. You can always choose a different option, and that choice is yours.</p>
          </div>
        </details>
        <details className="explain">
          <summary>What gets saved</summary>
          <div>
            <p>While a chat is open, its messages are stored so you can come back to it. When it ends, the words are cleared unless you chose to keep them. A short summary, your feelings, and the small step you picked are saved.</p>
            <p>Chats left open for 24 hours are closed the next time you use Luna.</p>
          </div>
        </details>
      </section>

      <section className="stack" aria-labelledby="data-heading">
        <h2 id="data-heading">Your data</h2>
        <div className="row">
          <button className="btn btn-soft" type="button" onClick={exportData}>Download my data</button>
          <button className="btn btn-ghost" type="button" onClick={() => void signOut()}>Sign out</button>
        </div>
        <div className="card danger-card">
          <h3>Delete everything</h3>
          <p className="muted small">This removes your chats, check-ins, and everything Luna saved. Your sign-in stays.</p>
          <label className="text-field">
            Type “{DELETE_PHRASE}” to confirm
            <input autoComplete="off" value={confirmation} onChange={(event) => setConfirmation(event.target.value)} />
          </label>
          <button className="btn btn-danger" type="button" disabled={busy || confirmation.trim().toLowerCase() !== DELETE_PHRASE} onClick={deleteData}>
            {busy ? "Deleting…" : "Delete my journal"}
          </button>
        </div>
      </section>

      {message && <p className="note ok" role="status">{message}</p>}
      {error && <p className="note error" role="alert">{error}</p>}
    </div>
  );
}
