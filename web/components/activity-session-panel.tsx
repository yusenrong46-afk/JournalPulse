"use client";

import { forwardRef, useState, type FormEvent } from "react";

import type { ActivityCommand, ActivityReport, ActivitySession } from "@/lib/activity-session";

type Props = {
  session: ActivitySession; secondsLeft: number; busy: boolean; expiryPending: boolean;
  canStart?: boolean;
  suppressQuestions?: boolean;
  /** Opens the inline search; offered sessions show it as "Something else". */
  onSomethingElse?(): void;
  onCommand(command: ActivityCommand): void; onReport(report: ActivityReport): void; onFollowUp(): void;
};

type OfferProps = {
  title: string; reason?: string | null; minutes?: number | null; instructions?: string[];
  searchSnippet?: boolean; url?: string | null; busy: boolean; canStart: boolean;
  onStart(): void; onSomethingElse?(): void; dismiss: { label: string; onClick(): void };
};

/** One optional proposal, with the same layout whether Luna suggested it or the person saved it. */
export function OfferCard({ title, reason, minutes, instructions = [], searchSnippet, url, busy, canStart, onStart, onSomethingElse, dismiss }: OfferProps) {
  return (
    <section className="card offer-card" aria-label="Activity with Luna">
      <span className="offer-kicker">Optional idea</span>
      <h2>{title}</h2>
      {(minutes || searchSnippet) && <div className="offer-meta">
        {minutes ? <span className="tag sage">About {minutes} minute{minutes === 1 ? "" : "s"}</span> : null}
        {searchSnippet ? <span className="tag">From a search snippet · page not reviewed</span> : null}
      </div>}
      {reason && <p className="offer-reason">{reason}</p>}
      {instructions.length > 0 && <details className="offer-steps">
        <summary>What you’d do</summary>
        <ol>{instructions.map((instruction) => <li key={instruction}>{instruction}</li>)}</ol>
      </details>}
      {url && <a className="small" href={url} target="_blank" rel="noopener noreferrer">Open resource</a>}
      <button className="btn btn-primary offer-start" type="button" disabled={busy || !canStart} onClick={onStart}>Start activity</button>
      <div className="offer-alternatives">
        {onSomethingElse && <button className="btn btn-soft" type="button" disabled={busy} onClick={onSomethingElse}>Something else</button>}
        <button className="btn btn-ghost" type="button" disabled={busy} onClick={dismiss.onClick}>{dismiss.label}</button>
      </div>
      <p className="small muted">You can also tell Luna what would fit better.</p>
    </section>
  );
}

const PARTICIPATION = [
  ["completed", "Completed"], ["partial", "Partly tried"], ["not_tried", "Not tried"], ["stopped", "Stopped"],
] as const;

export const CheckIn = forwardRef<HTMLFormElement, Pick<Props, "session" | "busy" | "onReport">>(
  function CheckIn({ session, busy, onReport }, ref) {
    const [participation, setParticipation] = useState<ActivityReport["participation"] | "">("");
    const [fit, setFit] = useState("");
    const [change, setChange] = useState("");
    const [progress, setProgress] = useState("");
    const [note, setNote] = useState("");

    function submit(event: FormEvent<HTMLFormElement>) {
      event.preventDefault();
      if (!participation || busy) return;
      onReport({ participation,
        ...(fit ? { fit: fit as ActivityReport["fit"] } : {}),
        ...(change ? { state_change: change as ActivityReport["state_change"] } : {}),
        ...(progress ? { goal_progress: progress as ActivityReport["goal_progress"] } : {}),
        ...(note.trim() ? { note: note.trim() } : {}),
      });
    }

    return (
      <form ref={ref} className="card check-in-card" aria-label="Activity check-in" onSubmit={submit} tabIndex={-1}>
        <fieldset disabled={busy} className="stack" style={{ border: 0, padding: 0, margin: 0 }}>
          <legend><h2 className="check-in-title">Did you try it?</h2></legend>
          <p className="small muted">The timer can’t tell whether you took part or how you feel. Your answer is what counts.</p>
          <div className="participation">
            {PARTICIPATION.map(([value, label]) => (
              <label className={`participation-choice${participation === value ? " is-selected" : ""}`} key={value}>
                <input type="radio" name={`participation-${session.id}`} value={value}
                  checked={participation === value} onChange={() => setParticipation(value)} /> {label}
              </label>
            ))}
          </div>
          <label className="text-field">Anything you want Luna to know? (optional)
            <textarea value={note} onChange={(event) => setNote(event.target.value)} maxLength={1000} rows={2} />
          </label>
          {/* Optional detail stays available for research-quality reports without crowding the one required answer. */}
          <details className="check-in-more">
            <summary>Add more detail (optional)</summary>
            <div className="stack">
              <label className="text-field">How well did this fit?
                <select value={fit} onChange={(event) => setFit(event.target.value)}>
                  <option value="">Choose if you want</option><option value="good">Good fit</option>
                  <option value="mixed">Mixed fit</option><option value="poor">Poor fit</option><option value="unsure">Unsure</option>
                </select>
              </label>
              <label className="text-field">What changed?
                <select value={change} onChange={(event) => setChange(event.target.value)}>
                  <option value="">Choose if you want</option><option value="toward_target">More like I hoped</option>
                  <option value="same">About the same</option><option value="away_from_target">Less like I hoped</option>
                  <option value="unsure">Unsure</option>
                </select>
              </label>
              <label className="text-field">Progress toward your goal
                <select value={progress} onChange={(event) => setProgress(event.target.value)}>
                  <option value="">Choose if you want</option><option value="closer">A little closer</option>
                  <option value="same">About the same</option><option value="further">Further away</option><option value="unsure">Unsure</option>
                </select>
              </label>
            </div>
          </details>
          <button className="btn btn-primary" type="submit" disabled={!participation || busy}>
            {busy ? "Saving…" : "Save check-in"}
          </button>
        </fieldset>
      </form>
    );
  },
);

const REPORTED: Record<ActivityReport["participation"], string> = {
  not_tried: "You reported that you did not try it.",
  partial: "You reported partly trying it.",
  stopped: "You reported stopping.",
  completed: "You reported completing it.",
};

/** Parts of an activity that belong in the conversation: an offered session, the check-in, and its receipt. */
export const ActivitySessionPanel = forwardRef<HTMLFormElement, Props>(function ActivitySessionPanel(
  { session, busy, expiryPending, canStart = true, suppressQuestions = false, onSomethingElse, onCommand, onReport, onFollowUp },
  checkInRef,
) {
  const waiting = !suppressQuestions && (session.status === "awaiting_report" || expiryPending
    || (session.status === "stopped" && session.check_in_issued && Boolean(session.started_at) && !session.report));
  const minutes = session.duration_seconds ? Math.round(session.duration_seconds / 60) || null : null;

  return (
    <>
      {session.status === "offered" && <OfferCard title={session.resource.title} reason={session.recommendation_reason}
        minutes={session.resource.format === "timer" ? minutes : null} instructions={session.resource.instructions}
        searchSnippet={session.resource.provenance === "search_snippet"} url={session.resource.url}
        busy={busy} canStart={canStart} onStart={() => onCommand("start")} onSomethingElse={onSomethingElse}
        dismiss={{ label: "Not now", onClick: () => onCommand("decline") }} />}
      {waiting && <CheckIn ref={checkInRef} key={session.id} session={session} busy={busy || expiryPending} onReport={onReport} />}
      {/* Once Luna's follow-up is in the chat it acknowledges the report there, so the receipt
          only stays while that reply is pending, failed, or could not be placed as a message. */}
      {session.report && !(session.follow_up_status === "ready" && session.follow_up_message_id && !session.final_follow_up)
        && <div className="activity-receipt" aria-label="Activity check-in saved">
        <p role="status"><strong>Check-in saved.</strong> {REPORTED[session.report.participation]}</p>
        {session.follow_up_status === "pending" || session.follow_up_status === "generating"
          ? <p className="small muted" role="status">Luna’s follow-up is on its way.</p> : null}
        {(session.follow_up_status === "failed" || session.follow_up_status === "pending") && session.follow_up_attempts < 3 && <button className="btn btn-soft" type="button"
          disabled={busy} onClick={onFollowUp}>Retry Luna’s follow-up</button>}
        {session.follow_up_status === "ready" && session.follow_up_reply && !session.follow_up_message_id && <p>{session.follow_up_reply}</p>}
        {session.final_follow_up && <p className="small muted">This chat has reached its message limit. Your check-in is saved; start a new chat when you want to continue.</p>}
      </div>}
      {(session.status === "declined" || (session.status === "stopped" && !session.started_at)) && <p className="note activity-note">No activity started. You can keep talking.</p>}
      {session.status === "stopped" && session.started_at && !session.check_in_issued && !session.report
        && <p className="note activity-note">Activity stopped. You can keep talking.</p>}
    </>
  );
});
