"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

import { Icon } from "@/components/nav-icon";
import { Luna } from "@/components/luna";
import { DEFAULT_PREFERENCES, savePreferences, usePreferences } from "@/lib/preferences";

export default function WelcomePage() {
  const router = useRouter();
  const [preferences] = usePreferences();
  const [step, setStep] = useState(0);
  const [aiChoice, setAiChoice] = useState<boolean | null>(null);
  const [retainText, setRetainText] = useState(false);

  function finish() {
    savePreferences({
      ...DEFAULT_PREFERENCES,
      ...preferences,
      onboarded: true,
      llmConsent: aiChoice === true,
      retainText,
    });
    router.replace("/talk");
  }

  return (
    <div className="focus-page">
      <div className="dots" aria-hidden="true">
        {[0, 1, 2].map((index) => <i key={index} className={index === step ? "on" : undefined} />)}
      </div>

      {step === 0 && (
        <>
          <span className="welcome-luna"><Luna mood="checkin" size={150} /></span>
          <h1>Hi, I’m Luna.</h1>
          <p className="welcome-lede">A quiet place to put the day down.</p>
          <p>We talk for a minute. If you want, I’ll suggest one small thing, and later you can tell me honestly how it went.</p>
          <ul className="welcome-promises">
            <li><Icon name="chat" />Talk it through, in your own words</li>
            <li><Icon name="leaf" />One small step, only if you want it</li>
            <li><Icon name="lock" />Private by default, your choices kept</li>
          </ul>
          <button className="btn btn-primary btn-big btn-block" type="button" onClick={() => setStep(1)}>Nice to meet you</button>
        </>
      )}

      {step === 1 && (
        <>
          <Luna mood="listening" size={130} />
          <h1>Your words stay yours.</h1>
          <p>How should Luna reply to you?</p>
          <div className="choice-cards" role="group" aria-label="How Luna replies">
            <button className="choice-card" type="button" aria-pressed={aiChoice === true} onClick={() => setAiChoice(true)}>
              <strong>Smart Luna</strong>
              <small>Uses AI for personal replies, with zero data retention required from the provider.</small>
              <span className="radio-dot" aria-hidden="true" />
            </button>
            <button className="choice-card" type="button" aria-pressed={aiChoice === false} onClick={() => setAiChoice(false)}>
              <strong>Simple Luna</strong>
              <small>No AI. Luna asks a few gentle questions instead.</small>
              <span className="radio-dot" aria-hidden="true" />
            </button>
          </div>
          <div className="settings" style={{ textAlign: "left" }}>
            <label className="setting">
              <span><strong>Keep my messages</strong><small>Off clears the words when a chat ends.</small></span>
              <span className="switch"><input type="checkbox" checked={retainText} onChange={(event) => setRetainText(event.target.checked)} /><span /></span>
            </label>
          </div>
          <button className="btn btn-primary btn-big btn-block" type="button" disabled={aiChoice === null} onClick={() => setStep(2)}>Continue</button>
          <p className="small">You can change this anytime in Settings.</p>
        </>
      )}

      {step === 2 && (
        <>
          <Luna mood="answering" size={130} />
          <h1>One small thing at a time.</h1>
          <ol className="how-steps">
            <li><span style={{ background: "var(--lav-soft)" }} aria-hidden="true"><Icon name="chat" /></span><div><strong>Talk</strong><p className="small">Tell Luna how you’re doing, in your own words or with a tap.</p></div></li>
            <li><span style={{ background: "var(--sun-soft)" }} aria-hidden="true"><Icon name="leaf" /></span><div><strong>Try</strong><p className="small">Pick one small idea from a reviewed list.</p></div></li>
            <li><span style={{ background: "var(--sage-soft)" }} aria-hidden="true"><Icon name="journey" /></span><div><strong>Check in</strong><p className="small">Tell Luna how it went, and watch your garden grow.</p></div></li>
          </ol>
          <button className="btn btn-primary btn-big btn-block" type="button" onClick={finish}>Let’s begin</button>
          <p className="small">Luna is a companion, not a therapist. In a crisis in Canada, call or text 9-8-8.</p>
        </>
      )}

      {step > 0 && (
        <button className="btn btn-ghost" type="button" onClick={() => setStep(step - 1)}>Back</button>
      )}
    </div>
  );
}
