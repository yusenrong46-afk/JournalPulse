"use client";

import { forwardRef, useState, type FormEvent } from "react";

import type { ActivityCommand, ActivityReport, ActivitySession } from "@/lib/activity-session";
import { Luna } from "./luna";
import { OfferScene } from "./offer-scene";

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
  /** Plain-language limits already saved for this chat, such as "No audio" or "Seated". */
  tags?: string[];
  onStart(): void; onSomethingElse?(): void; dismiss: { label: string; onClick(): void };
};

/** One optional proposal, with the same layout whether Luna suggested it or the person saved it. */
export function OfferCard({ title, reason, minutes, instructions = [], searchSnippet, url, busy, canStart, onStart, onSomethingElse, dismiss, tags = [] }: OfferProps) {
  return (
    <section className="card offer-card" aria-label="Activity with Luna">
      <OfferScene />
      <div className="offer-content">
        <span className="offer-kicker">
          <span className="offer-luna"><Luna mood="offering" size={22} decorative /></span>
          An optional idea
        </span>
        <h2>{title}</h2>
        {(minutes || tags.length > 0) && <ul className="offer-tags" aria-label="About this idea">
          {minutes ? <li><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8.5" /><path d="M12 7.5V12l3 2" /></svg>
            About {minutes} minute{minutes === 1 ? "" : "s"}</li> : null}
          {tags.map((tag) => <li key={tag}>{tag}</li>)}
        </ul>}
        {searchSnippet && <p className="offer-provenance">From a search snippet · the full page was not reviewed</p>}
        {reason && <p className="offer-reason">{reason}</p>}
        {instructions.length > 0 && <details className="offer-steps">
          <summary>What you’d do</summary>
          <ol>{instructions.map((instruction) => <li key={instruction}>{instruction}</li>)}</ol>
        </details>}
        {url && <a className="small activity-resource-link" href={url} target="_blank" rel="noopener noreferrer">Open resource</a>}
      </div>
      <div className="offer-controls">
        <button className="btn btn-primary offer-start" type="button" disabled={busy || !canStart} onClick={onStart}>Start activity</button>
        {onSomethingElse && <button className="btn btn-quiet" type="button" disabled={busy} onClick={onSomethingElse}>Something else</button>}
        <button className="btn btn-quiet" type="button" disabled={busy} onClick={dismiss.onClick}>{dismiss.label}</button>
      </div>
      <p className="offer-footnote">Nothing starts until you choose. You can also tell Luna what would fit better.</p>
    </section>
  );
}

const PARTICIPATION = [
  ["completed", "Completed", "the whole way"], ["partial", "Partly tried", "some of it"],
  ["not_tried", "Not tried", "not this time"], ["stopped", "Stopped", "it wasn’t right"],
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
          <p className="check-in-lede">The timer can’t tell whether you took part or how you feel. Only you can, and any answer is fine.</p>
          <div className="participation">
            {PARTICIPATION.map(([value, label, hint]) => (
              <label className={`participation-choice${participation === value ? " is-selected" : ""}`} key={value}>
                {/* The native radio stays for keyboard and screen readers; CSS draws it as a moon. */}
                <input className={`moon-radio moon-${value}`} type="radio" name={`participation-${session.id}`} value={value}
                  aria-label={label} checked={participation === value} onChange={() => setParticipation(value)} />
                <span className="participation-text"><span>{label}</span><small aria-hidden="true">{hint}</small></span>
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
      {(waiting || session.report) && session.resource.url && !suppressQuestions && (
        <a className="small activity-resource-link" href={session.resource.url}
          target="_blank" rel="noopener noreferrer">Open resource: {session.resource.title}</a>
      )}
      {waiting && <CheckIn ref={checkInRef} key={session.id} session={session} busy={busy || expiryPending} onReport={onReport} />}
      {/* Keep the person's own report visible after generated follow-up wording.
          A reply must not become the only remaining record of what was reported. */}
      {session.report && <div className="activity-receipt" aria-label="Activity check-in saved">
        <p role="status"><strong>Check-in saved.</strong> {REPORTED[session.report.participation]}</p>
        {session.report.note && <p className="activity-receipt-note">“{session.report.note}”</p>}
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
