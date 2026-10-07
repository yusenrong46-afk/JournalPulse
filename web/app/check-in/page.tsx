"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";

import { Luna } from "@/components/luna";
import { Icon } from "@/components/nav-icon";
import { ApiError, apiRequest } from "@/lib/api";
import { FEELINGS, selfReport } from "@/lib/feelings";
import { clearReminder } from "@/lib/reminders";
import type { OutcomeRecord, ReflectionRecord, Resource } from "@/lib/types";

const HELP_FACES = [
  { score: 1, label: "Not at all" },
  { score: 2, label: "A little" },
  { score: 3, label: "Somewhat" },
  { score: 4, label: "Helped" },
  { score: 5, label: "A lot" },
];

function CheckInWorkspace() {
  const searchParams = useSearchParams();
  const requestedDecision = searchParams.get("decision");
  const prefilled = Number(searchParams.get("h"));
  const [reflection, setReflection] = useState<ReflectionRecord | null>(null);
  const [resource, setResource] = useState<Resource | null>(null);
  const [alreadyDone, setAlreadyDone] = useState(false);
  const [tried, setTried] = useState<boolean | null>(
    searchParams.get("tried") === "no" ? false : prefilled >= 1 && prefilled <= 5 ? true : null,
  );
  const [helpfulness, setHelpfulness] = useState<number | null>(prefilled >= 1 && prefilled <= 5 ? prefilled : null);
  const [feelings, setFeelings] = useState<string[]>([]);
  const [note, setNote] = useState("");
  const [saved, setSaved] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [retryPending, setRetryPending] = useState(false);
  const [error, setError] = useState("");
  const pendingOutcome = useRef<string | null>(null);
  const submission = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=100", { signal: controller.signal }),
      apiRequest<{ items: OutcomeRecord[] }>("/v1/outcomes", { signal: controller.signal }),
      apiRequest<{ items: Resource[] }>("/v1/resources", { signal: controller.signal }),
    ])
      .then(([history, outcomes, catalog]) => {
        if (controller.signal.aborted) return;
        const done = new Set(outcomes.items.map((item) => item.decision_id));
        const selected = requestedDecision
          ? history.items.find((item) => item.decision.decision_id === requestedDecision)
          : history.items.find((item) => !done.has(item.decision.decision_id));
        if (!selected) return;
        setReflection(selected);
        setAlreadyDone(done.has(selected.decision.decision_id));
        setResource(catalog.items.find((item) => item.id === selected.decision.action_id) ?? null);
      })
      .catch(() => { if (!controller.signal.aborted) setError("Luna couldn’t load this check-in. Please try again in a moment."); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => {
      controller.abort();
      submission.current?.abort();
    };
  }, [requestedDecision]);

  async function submit(completed: boolean) {
    if (!reflection || submission.current) return;
    const controller = new AbortController();
    submission.current = controller;
    setBusy(true);
    setError("");
    if (!pendingOutcome.current) {
      const elapsed = Math.round((Date.now() - new Date(reflection.created_at).getTime()) / 60_000);
      pendingOutcome.current = JSON.stringify({
        client_request_id: crypto.randomUUID(),
        decision_id: reflection.decision.decision_id,
        completed,
        post_state: completed && feelings.length ? selfReport(feelings, null) : null,
        helpfulness: completed ? helpfulness : null,
        effort: null,
        elapsed_minutes: Math.min(Math.max(elapsed, 0), 10080),
        note: note.trim() || null,
      });
    }
    try {
      await apiRequest<OutcomeRecord>("/v1/outcomes", {
        method: "POST",
        retry: true,
        signal: controller.signal,
        // The server stores one outcome per decision and replays the first
        // receipt. A lost response must retry these exact answers and timestamp.
        body: pendingOutcome.current,
      });
      if (controller.signal.aborted) return;
      clearReminder(reflection.decision.decision_id);
      setRetryPending(false);
      setSaved(true);
    } catch (reason) {
      if (!controller.signal.aborted) {
        // Validation rejects before saving, so those answers remain editable.
        const rejected = reason instanceof ApiError && [400, 422].includes(reason.status);
        if (rejected) pendingOutcome.current = null;
        setRetryPending(!rejected);
        setError(reason instanceof Error ? reason.message : "The save could not be confirmed. Please retry.");
      }
    } finally {
      if (!controller.signal.aborted) setBusy(false);
      if (submission.current === controller) submission.current = null;
    }
  }

  const title = resource?.title ?? "your small step";

  let body: React.ReactNode;
  if (loading) {
    body = <div className="loading-luna" role="status"><Luna mood="checkin" size={110} decorative /><span>Finding your check-in…</span></div>;
  } else if (!reflection) {
    body = (
      <>
        <Luna mood="idle" size={120} />
        <h1>Nothing to check in on.</h1>
        <p>{error || "When you try something Luna suggested, you can tell Luna how it went here."}</p>
        <Link className="btn btn-primary" href="/talk">Talk with Luna</Link>
      </>
    );
  } else if (saved || alreadyDone) {
    body = (
      <>
        <Luna mood={saved && tried && helpfulness != null && helpfulness >= 4 ? "grounded" : "reflecting"} size={130} />
        <h1>{saved ? "Thank you!" : "Already checked in"}</h1>
        <p>{saved ? "Your garden grew a little. Over time you’ll see which small things help you most." : "You already told Luna how this one went."}</p>
        <div className="row" style={{ justifyContent: "center" }}>
          <Link className="btn btn-primary" href="/journey">See your garden</Link>
          <Link className="btn btn-ghost" href="/">Home</Link>
        </div>
      </>
    );
  } else if (tried === null) {
    body = (
      <>
        <Luna mood="checkin" size={130} />
        <h1>Did you get to try “{title}”?</h1>
        <div className="stack">
          <button className="btn btn-primary btn-big btn-block" type="button" onClick={() => setTried(true)}>Yes, I did</button>
          <button className="btn btn-soft btn-block" type="button" onClick={() => setTried(false)}>Not yet</button>
        </div>
      </>
    );
  } else if (!tried) {
    body = (
      <>
        <Luna mood="idle" size={120} />
        <h1>That’s okay.</h1>
        <p>Small steps work best when they fit your day. Want Luna to ask again later, or skip this one?</p>
        <div className="stack">
          <Link className="btn btn-primary btn-block" href="/">Ask me later</Link>
          <button className="btn btn-soft btn-block" type="button" disabled={busy} onClick={() => void submit(false)}>{busy ? "Saving…" : retryPending ? "Retry saving check-in" : "Skip this one"}</button>
        </div>
        {error && <p className="note error" role="alert">{error}</p>}
      </>
    );
  } else {
    body = (
      <>
        <Luna mood={helpfulness && helpfulness >= 4 ? "answering" : "checkin"} size={110} />
        <h1>How much did it help?</h1>
        <div className="faces" role="group" aria-label="How much did it help?">
          {HELP_FACES.map((face) => (
            <button key={face.score} className="face" type="button" disabled={busy || retryPending} aria-pressed={helpfulness === face.score} onClick={() => setHelpfulness(face.score)}>
              <span className="scale-number" aria-hidden="true">{face.score}</span>
              <span>{face.label}</span>
            </button>
          ))}
        </div>
        <div className="stack" style={{ textAlign: "left" }}>
          <h2 style={{ fontSize: "1.15rem" }}>How do you feel now? <span className="muted small">(optional)</span></h2>
          <div className="chips" role="group" aria-label="How you feel now">
            {FEELINGS.map((item) => (
              <button
                key={item.id}
                className="chip"
                type="button"
                disabled={busy || retryPending}
                aria-pressed={feelings.includes(item.id)}
                onClick={() => setFeelings((current) => (current.includes(item.id) ? current.filter((id) => id !== item.id) : [...current, item.id]))}
              >
                {item.label}
              </button>
            ))}
          </div>
          <label className="text-field">
            Anything you noticed? <span className="muted small">(optional)</span>
            <textarea value={note} maxLength={1000} disabled={busy || retryPending} onChange={(event) => setNote(event.target.value)} placeholder="What helped, what didn’t, what surprised you…" />
          </label>
        </div>
        <button className="btn btn-primary btn-big btn-block" type="button" disabled={busy || helpfulness === null} onClick={() => void submit(true)}>
          {busy ? "Saving…" : retryPending ? "Retry saving check-in" : "Save my check-in"}
        </button>
        {error && <p className="note error" role="alert">{error}</p>}
      </>
    );
  }

  return (
    <div className="focus-page">
      <div style={{ display: "flex", justifyContent: "flex-start" }}>
        <Link className="icon-btn" href="/" aria-label="Back to home"><Icon name="back" /></Link>
      </div>
      {body}
      {retryPending && !saved && !alreadyDone && <p className="note" role="status">
        Your save is not confirmed. Your submitted answers are kept unchanged; retry to confirm the same check-in.
      </p>}
    </div>
  );
}

function CheckInRoute() {
  const searchParams = useSearchParams();
  // A different decision is a different form and idempotency receipt. Keying the
  // workspace also cancels the previous load and discards its unsaved answers.
  const identity = JSON.stringify([searchParams.get("decision"), searchParams.get("h"), searchParams.get("tried")]);
  return <CheckInWorkspace key={identity} />;
}

export default function CheckInPage() {
  return (
    <Suspense fallback={<div className="loading-luna"><Luna mood="checkin" size={100} decorative /></div>}>
      <CheckInRoute />
    </Suspense>
  );
}
