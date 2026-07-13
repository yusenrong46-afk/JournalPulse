"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { DEFAULT_PREFERENCES, savePreferences } from "@/lib/preferences";

export default function WelcomePage() {
  const router = useRouter();
  const [llmConsent, setLlmConsent] = useState(false);
  const [retainText, setRetainText] = useState(false);
  const [encryptedDrafts, setEncryptedDrafts] = useState(false);

  function continueToReflection() {
    savePreferences({
      ...DEFAULT_PREFERENCES,
      onboarded: true,
      llmConsent,
      retainText,
      encryptedDrafts,
    });
    router.push("/reflect");
  }

  return (
    <div className="page-wrap narrow welcome-page">
      <header className="flow-header">
        <div className="page-heading-copy">
          <span className="kicker">Before the first entry</span>
          <h1>A field journal, not a diagnosis.</h1>
          <p>Set the boundary once. You can change any choice for each individual reflection.</p>
        </div>
      </header>
      <section className="flow-sheet">
        <span className="folio">Your boundary</span>
        <h2>You remain the authority on your own state.</h2>
        <p>
          JournalPulse proposes a structured reading, lets you correct it, and offers activities from a
          reviewed catalog. It does not provide therapy, treatment, or emergency care.
        </p>
        <div className="boundary-grid">
          <article><strong>1</strong><span>Write privately</span><small>Original text is not retained unless you choose it.</small></article>
          <article><strong>2</strong><span>Correct the read</span><small>Automated interpretation is always editable.</small></article>
          <article><strong>3</strong><span>Test one action</span><small>Outcomes become observations, not clinical claims.</small></article>
        </div>
        <div className="consent-box">
          <label>
            <input type="checkbox" checked={llmConsent} onChange={(event) => setLlmConsent(event.target.checked)} />
            <span><strong>Allow private AI analysis</strong><small>Opt in to zero-data-retention processing. You can change this later.</small></span>
          </label>
          <label>
            <input type="checkbox" checked={retainText} onChange={(event) => setRetainText(event.target.checked)} />
            <span><strong>Keep my original journal text</strong><small>Off by default. Structured states can be saved without the entry.</small></span>
          </label>
          <label>
            <input type="checkbox" checked={encryptedDrafts} onChange={(event) => setEncryptedDrafts(event.target.checked)} />
            <span><strong>Recover unfinished drafts</strong><small>Encrypt one active draft on this device for up to 24 hours.</small></span>
          </label>
        </div>
        <button className="button primary" onClick={continueToReflection}>Set my preferences <span aria-hidden="true">→</span></button>
        <p className="method-note">If language suggests immediate danger, support mode bypasses AI and experimentation.</p>
      </section>
    </div>
  );
}
