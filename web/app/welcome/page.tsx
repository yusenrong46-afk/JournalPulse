"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { DEFAULT_PREFERENCES, savePreferences } from "@/lib/preferences";

export default function WelcomePage() {
  const router = useRouter();
  const [llmConsent, setLlmConsent] = useState(false);
  const [retainText, setRetainText] = useState(false);

  function continueToReflection() {
    savePreferences({
      ...DEFAULT_PREFERENCES,
      onboarded: true,
      llmConsent,
      retainText,
    });
    router.push("/reflect");
  }

  return (
    <div className="page-wrap narrow welcome-page">
      <header className="flow-header">
        <div>
          <span className="kicker">Before the first entry</span>
          <h1>A field journal, not a diagnosis.</h1>
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
        </div>
        <button className="button primary" onClick={continueToReflection}>Set my preferences</button>
        <p className="method-note">If language suggests immediate danger, support mode bypasses AI and experimentation.</p>
      </section>
    </div>
  );
}
