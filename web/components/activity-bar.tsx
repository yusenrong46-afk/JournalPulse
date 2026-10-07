"use client";

import { useId, useState } from "react";

import type { ActivityCommand, ActivitySession } from "@/lib/activity-session";

import { Luna } from "./luna";

type Props = {
  session: ActivitySession; secondsLeft: number; busy: boolean;
  /** The person's check-in is due (timer expiry, Finish early, or an explicit Stop). */
  waiting: boolean; expiryPending: boolean;
  onCommand(command: ActivityCommand): void; onGoToCheckIn(): void;
};

function clockLabel(seconds: number) {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
}

/**
 * The compact controls for a started activity. It sits above the composer, outside the chat
 * log, so the conversation stays readable while the activity runs. It never reports anything
 * itself: finishing, stopping or expiry only lead to the person's own check-in.
 */
export function ActivityBar({ session, secondsLeft, busy, waiting, expiryPending, onCommand, onGoToCheckIn }: Props) {
  const [stepsOpen, setStepsOpen] = useState(false);
  const stepsId = useId();
  const timer = session.resource.format === "timer";
  const paused = session.status === "paused";

  if (waiting || expiryPending) {
    const timeUp = expiryPending || (timer && session.remaining_seconds === 0);
    return (
      <section className="activity-bar is-due" aria-label="Current activity">
        <div className="activity-bar-row">
          <div className="activity-bar-text">
            <strong>{timeUp ? "Time’s up" : "Ready for your check-in"}</strong>
            <span>{expiryPending ? "Confirming…" : "Tell Luna how it went, whenever you’re ready."}</span>
          </div>
          <button className="btn btn-soft activity-bar-btn" type="button" disabled={expiryPending} onClick={onGoToCheckIn}>
            Check in
          </button>
        </div>
      </section>
    );
  }

  const total = Math.max(session.duration_seconds, 1);
  const progress = timer ? Math.min(1, Math.max(0, secondsLeft / total)) : 1;
  const circumference = 2 * Math.PI * 25;
  const steps = session.resource.instructions;
  // A gentle pointer to where someone might be, paced by elapsed time; it never measures them.
  const stepIndex = steps.length ? Math.min(steps.length - 1, Math.floor((1 - progress) * steps.length)) : -1;
  return (
    <section className={`activity-bar${paused ? " is-paused" : ""}`} aria-label="Current activity" aria-busy={busy}>
      {stepsOpen && session.resource.instructions.length > 0 && (
        <ol id={stepsId} className="activity-steps">
          {session.resource.instructions.map((instruction) => <li key={instruction}>{instruction}</li>)}
        </ol>
      )}
      {!stepsOpen && stepIndex >= 0 && timer && (
        <p className="activity-step-hint">Step {stepIndex + 1} of {steps.length} · {steps[stepIndex]}</p>
      )}
      <div className="activity-bar-row">
        <div className="activity-bar-info">
          {timer && (
            <span className="activity-ring" aria-hidden="true">
              <svg width="58" height="58" viewBox="0 0 58 58">
                <circle cx="29" cy="29" r="25" fill="none" strokeWidth="4.5" className="activity-ring-track" />
                <circle cx="29" cy="29" r="25" fill="none" strokeWidth="4.5" strokeLinecap="round"
                  className="activity-ring-fill" strokeDasharray={circumference}
                  strokeDashoffset={circumference * (1 - progress)} transform="rotate(-90 29 29)" />
              </svg>
              <span className="activity-ring-luna"><Luna mood={paused ? "resting" : "grounded"} size={34} decorative /></span>
            </span>
          )}
          <div className="activity-bar-text">
            <strong>{session.resource.title}</strong>
            {session.resource.url && <a className="small activity-resource-link" href={session.resource.url}
              target="_blank" rel="noopener noreferrer">Open resource</a>}
            {timer ? (
              <span className="activity-time" role="timer" aria-live="off" aria-label="Activity time remaining">
                {paused ? "Paused · " : ""}{clockLabel(secondsLeft)} <em>left</em>
              </span>
            ) : <span>{paused ? "Paused" : "In progress"}</span>}
          </div>
          {session.resource.instructions.length > 0 && (
            <button className="link-btn activity-bar-steps" type="button" aria-expanded={stepsOpen}
              aria-controls={stepsOpen ? stepsId : undefined} onClick={() => setStepsOpen((open) => !open)}>
              {stepsOpen ? "Hide steps" : "Steps"}
            </button>
          )}
        </div>
        <div className="activity-bar-actions">
          {timer && (
            <button className={`btn activity-bar-btn ${paused ? "btn-primary" : "btn-soft"}`} type="button" disabled={busy}
              onClick={() => onCommand(paused ? "resume" : "pause")}>
              {paused ? "Resume" : "Pause"}
            </button>
          )}
          <button className="btn btn-soft activity-bar-btn" type="button" disabled={busy} onClick={() => onCommand("finish_early")}>
            {timer ? "Finish early" : "Done / check in"}
          </button>
          <button className="btn btn-ghost activity-bar-btn" type="button" disabled={busy} onClick={() => onCommand("stop")}>
            Stop activity
          </button>
        </div>
      </div>
    </section>
  );
}
