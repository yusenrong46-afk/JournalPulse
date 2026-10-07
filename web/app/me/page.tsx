"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { Luna } from "@/components/luna";
import { apiRequest } from "@/lib/api";
import { invalidateAccountDataRequests } from "@/lib/account-data";
import { writeOpenConversationId } from "@/lib/conversation";
import { usePreferences } from "@/lib/preferences";
import { clearReflectionDraft } from "@/lib/reflection-draft";
import { clearReminders } from "@/lib/reminders";
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
    // Deletion supersedes submitted work, even if the DELETE later fails. An
    // explicit new save remains possible; an old automatic retry never revives.
    invalidateAccountDataRequests();
    try {
      const result = await apiRequest<{ deleted_records: number }>("/v1/account/data", { method: "DELETE" });
      // Cancel work begun in another tab while deletion was in progress too.
      invalidateAccountDataRequests();
      writeOpenConversationId(null);
      clearReminders();
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
    setBusy(true);
    setError("");
    try {
      const client = await getSupabase();
      const result = await client?.auth.signOut();
      if (result?.error) throw result.error;
      router.replace("/login");
    } catch {
      setError("Sign-out didn’t finish. You’re still signed in; please try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page page-wide settings-page">
      <header className="page-head companion-page-head">
        <Luna mood="idle" size={72} decorative />
        <div>
          <span className="eyebrow">On your terms</span>
          <h1>Your space</h1>
          <p>Choose how Luna works with you. You can change these anytime.</p>
        </div>
      </header>

      <div className="settings-layout">
      <div className="settings-primary stack">
      <div><h2>Your preferences</h2><p className="small muted">AI and message settings apply to new chats. Changes save automatically.</p></div>
      {preferencesLoaded && (
        <section className="settings" aria-label="Chat settings">
          <label className="setting">
            <span><strong>Let Luna use AI</strong><small>Uses AI for replies in new chats, with zero data retention required from the provider. Off starts a simple guided chat.</small></span>
            <span className="switch"><input type="checkbox" checked={preferences.llmConsent} onChange={(event) => updatePreferences({ ...preferences, llmConsent: event.target.checked })} /><span /></span>
          </label>
          <label className="setting">
            <span><strong>Keep my messages</strong><small>For new chats. Off clears message text when the chat ends; your summary and activity choices remain saved.</small></span>
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
          <label className="setting">
            <span><strong>Animation</strong><small>Luna’s movements and gentle transitions. Switch off for a still experience.</small></span>
            <span className="switch"><input type="checkbox" checked={preferences.animateLuna !== false}
              onChange={(event) => updatePreferences({ ...preferences, animateLuna: event.target.checked })} /><span /></span>
          </label>
        </section>
      )}

      </div>
      <section className="stack settings-explainer" aria-labelledby="how-heading">
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
            <p>JournalPulse checks messages for certain clear crisis phrases. These checks can miss signs of distress. When a phrase is detected, Luna switches to support information and points you toward human help.</p>
          </div>
        </details>
        <details className="explain">
          <summary>Where Luna’s ideas come from</summary>
          <div>
            <p>Luna can suggest built-in activities and options from a reviewed resource collection, such as short walks and reading from health sites.</p>
            <p>With your permission, Luna can also search Brave and select links from search snippets. These web results have not been reviewed; full pages are not read or fact-checked.</p>
            <p>With AI help on, Luna uses the current conversation and any journal entry you explicitly share to choose from available activities. Simple mode uses a fixed selection rule. You can decline a suggestion or choose another option.</p>
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

      </div>
      <section className="stack settings-data" aria-labelledby="data-heading">
        <h2 id="data-heading">Your data</h2>
        <div className="row">
          <button className="btn btn-soft" type="button" onClick={exportData}>Download my data</button>
          <button className="btn btn-ghost" type="button" disabled={busy} onClick={() => void signOut()}>Sign out</button>
        </div>
        <details className="explain danger-card">
          <summary>Delete saved data</summary>
          <div className="stack">
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
        </details>
      </section>

      {message && <p className="note ok" role="status">{message}</p>}
      {error && <p className="note error" role="alert">{error}</p>}
    </div>
  );
}
