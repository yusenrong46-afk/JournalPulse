"use client";

import { useState, type FormEvent } from "react";

import type { ActivityCommand, ActivityReport, ActivitySession } from "@/lib/activity-session";

type Props = {
  session: ActivitySession; secondsLeft: number; busy: boolean; expiryPending: boolean;
  canStart?: boolean;
  suppressQuestions?: boolean;
  onCommand(command: ActivityCommand): void; onReport(report: ActivityReport): void; onFollowUp(): void;
};

function CheckIn({ session, busy, onReport }: Pick<Props, "session" | "busy" | "onReport">) {
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
    <form className="stack" aria-label="Activity check-in" onSubmit={submit}>
      <fieldset disabled={busy} className="stack" style={{ border: 0, padding: 0, margin: 0 }}>
        <legend><strong>Did you try it?</strong></legend>
        <p className="small muted">The timer cannot tell whether you participated or how you feel.</p>
        <div className="chips">
          {([
            ["completed", "Completed"], ["partial", "Partly tried"], ["not_tried", "Not tried"], ["stopped", "Stopped"],
          ] as const).map(([value, label]) => (
            <label className="chip" key={value}>
              <input type="radio" name={`participation-${session.id}`} value={value}
                checked={participation === value} onChange={() => setParticipation(value)} /> {label}
            </label>
          ))}
        </div>
        <label className="text-field">How well did this fit? (optional)
          <select value={fit} onChange={(event) => setFit(event.target.value)}>
            <option value="">Choose if you want</option><option value="good">Good fit</option>
            <option value="mixed">Mixed fit</option><option value="poor">Poor fit</option><option value="unsure">Unsure</option>
          </select>
        </label>
        <label className="text-field">What changed? (optional)
          <select value={change} onChange={(event) => setChange(event.target.value)}>
            <option value="">Choose if you want</option><option value="toward_target">More like I hoped</option>
            <option value="same">About the same</option><option value="away_from_target">Less like I hoped</option>
            <option value="unsure">Unsure</option>
          </select>
        </label>
        <label className="text-field">Progress toward your goal (optional)
          <select value={progress} onChange={(event) => setProgress(event.target.value)}>
            <option value="">Choose if you want</option><option value="closer">A little closer</option>
            <option value="same">About the same</option><option value="further">Further away</option><option value="unsure">Unsure</option>
          </select>
        </label>
        <label className="text-field">Anything you want Luna to know? (optional)
          <textarea value={note} onChange={(event) => setNote(event.target.value)} maxLength={1000} rows={2} />
        </label>
        <button className="btn btn-primary" type="submit" disabled={!participation || busy}>
          {busy ? "Saving…" : "Save check-in"}
        </button>
      </fieldset>
    </form>
  );
}

export function ActivitySessionPanel({ session, secondsLeft, busy, expiryPending, canStart = true, suppressQuestions = false, onCommand, onReport, onFollowUp }: Props) {
  const waiting = !suppressQuestions && (session.status === "awaiting_report" || expiryPending
    || (session.status === "stopped" && session.check_in_issued && Boolean(session.started_at) && !session.report));
  const running = session.status === "active" || session.status === "paused";
  const label = `${Math.floor(secondsLeft / 60)}:${String(secondsLeft % 60).padStart(2, "0")}`;

  return (
    <section className="card stack" aria-label="Activity with Luna" aria-busy={busy}>
      <h2>{session.resource.title}</h2>
      {session.recommendation_reason && <p>{session.recommendation_reason}</p>}
      {session.resource.instructions.length > 0 && (
        <ol>{session.resource.instructions.map((instruction) => <li key={instruction}>{instruction}</li>)}</ol>
      )}
      {session.resource.provenance === "search_snippet" && <p className="small muted">Chosen from a search snippet. The full page was not reviewed.</p>}
      {session.resource.url && <a className="btn btn-soft" href={session.resource.url} target="_blank" rel="noopener noreferrer">
        Open resource
      </a>}
      {(session.status === "offered" || running) && session.resource.format === "timer" && (
        <div role="timer" aria-live="off" aria-label="Activity time remaining" className="display center">{label}</div>
      )}
      {session.status === "offered" && <div className="row">
        <button className="btn btn-primary" type="button" disabled={busy || !canStart} onClick={() => onCommand("start")}>Start activity</button>
        <button className="btn btn-ghost" type="button" disabled={busy} onClick={() => onCommand("decline")}>Not now</button>
      </div>}
      {running && !expiryPending && <div className="row">
        {session.resource.format === "timer" && <button className="btn btn-soft" type="button" disabled={busy}
          onClick={() => onCommand(session.status === "paused" ? "resume" : "pause")}>
          {session.status === "paused" ? "Resume" : "Pause"}
        </button>}
        <button className="btn btn-primary" type="button" disabled={busy} onClick={() => onCommand("finish_early")}>
          {session.resource.format === "timer" ? "Finish early" : "Done / check in"}
        </button>
        <button className="btn btn-ghost" type="button" disabled={busy} onClick={() => onCommand("stop")}>Stop activity</button>
      </div>}
      {waiting && <>
        <p role="status">{expiryPending ? "Time is up. Confirming your check-in…" : "Your activity is ready for a check-in."}</p>
        <CheckIn key={session.id} session={session} busy={busy || expiryPending} onReport={onReport} />
      </>}
      {session.report && <>
        <p role="status">Check-in saved. {session.report.participation === "not_tried" ? "You reported that you did not try it." :
          session.report.participation === "partial" ? "You reported partly trying it." :
          session.report.participation === "stopped" ? "You reported stopping." : "You reported completing it."}</p>
        {session.follow_up_status === "pending" || session.follow_up_status === "generating"
          ? <p role="status">Your report is saved; Luna’s follow-up is pending.</p> : null}
        {(session.follow_up_status === "failed" || session.follow_up_status === "pending") && session.follow_up_attempts < 3 && <button className="btn btn-soft" type="button"
          disabled={busy} onClick={onFollowUp}>Retry Luna’s follow-up</button>}
        {session.follow_up_status === "ready" && session.follow_up_reply && !session.follow_up_message_id && <p>{session.follow_up_reply}</p>}
        {session.final_follow_up && <p className="small muted">This chat has reached its message limit. Your check-in is saved; start a new chat when you want to continue.</p>}
      </>}
      {(session.status === "declined" || (session.status === "stopped" && !session.started_at)) && <p className="note">No activity started. You can keep talking.</p>}
      {session.status === "stopped" && session.started_at && !session.check_in_issued && !session.report
        && <p className="note">Activity stopped. You can keep talking.</p>}
      {running && <p className="small muted">You can pause or stop. If the browser closes, we’ll sync the activity when you return.</p>}
    </section>
  );
}
