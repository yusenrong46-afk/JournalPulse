"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";

import { Luna } from "@/components/luna";
import { Icon } from "@/components/nav-icon";
import { apiRequest } from "@/lib/api";
import { FEELINGS, selfReport } from "@/lib/feelings";
import { clearReminder } from "@/lib/reminders";
import type { OutcomeRecord, ReflectionRecord, Resource } from "@/lib/types";

const HELP_FACES = [
  { score: 1, emoji: "😣", label: "Not at all" },
  { score: 2, emoji: "😕", label: "A little" },
  { score: 3, emoji: "😐", label: "Somewhat" },
  { score: 4, emoji: "🙂", label: "Helped" },
  { score: 5, emoji: "😄", label: "A lot" },
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
  const [error, setError] = useState("");
  const requestId = useRef("");

  useEffect(() => {
    Promise.all([
      apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=100"),
      apiRequest<{ items: OutcomeRecord[] }>("/v1/outcomes"),
      apiRequest<{ items: Resource[] }>("/v1/resources"),
    ])
      .then(([history, outcomes, catalog]) => {
        const done = new Set(outcomes.items.map((item) => item.decision_id));
        const selected = requestedDecision
          ? history.items.find((item) => item.decision.decision_id === requestedDecision)
          : history.items.find((item) => !done.has(item.decision.decision_id));
        if (!selected) return;
        setReflection(selected);
        setAlreadyDone(done.has(selected.decision.decision_id));
        setResource(catalog.items.find((item) => item.id === selected.decision.action_id) ?? null);
      })
      .catch(() => setError("Luna couldn’t load this check-in. Please try again in a moment."))
      .finally(() => setLoading(false));
  }, [requestedDecision]);

  async function submit(completed: boolean) {
    if (!reflection) return;
    setBusy(true);
    setError("");
    const elapsed = Math.round((Date.now() - new Date(reflection.created_at).getTime()) / 60_000);
    try {
      await apiRequest<OutcomeRecord>("/v1/outcomes", {
        method: "POST",
        retry: true,
        body: JSON.stringify({
          client_request_id: requestId.current || (requestId.current = crypto.randomUUID()),
          decision_id: reflection.decision.decision_id,
          completed,
          post_state: completed && feelings.length ? selfReport(feelings, null) : null,
          helpfulness: completed ? helpfulness : null,
          effort: null,
          elapsed_minutes: Math.min(Math.max(elapsed, 0), 10080),
          note: note.trim() || null,
        }),
      });
      clearReminder(reflection.decision.decision_id);
      setSaved(true);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "That didn’t save. Please try again.");
    } finally {
      setBusy(false);
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
        <Luna mood="proud" size={130} />
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
          <button className="btn btn-soft btn-block" type="button" disabled={busy} onClick={() => void submit(false)}>{busy ? "Saving…" : "Skip this one"}</button>
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
            <button key={face.score} className="face" type="button" aria-pressed={helpfulness === face.score} onClick={() => setHelpfulness(face.score)}>
              <span aria-hidden="true">{face.emoji}</span>
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
                aria-pressed={feelings.includes(item.id)}
                onClick={() => setFeelings((current) => (current.includes(item.id) ? current.filter((id) => id !== item.id) : [...current, item.id]))}
              >
                <span className="chip-emoji" aria-hidden="true">{item.emoji}</span>{item.label}
              </button>
            ))}
          </div>
          <label className="text-field">
            Anything you noticed? <span className="muted small">(optional)</span>
            <textarea value={note} maxLength={1000} onChange={(event) => setNote(event.target.value)} placeholder="What helped, what didn’t, what surprised you…" />
          </label>
        </div>
        <button className="btn btn-primary btn-big btn-block" type="button" disabled={busy || helpfulness === null} onClick={() => void submit(true)}>
          {busy ? "Saving…" : "Save my check-in"}
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
    </div>
  );
}

export default function CheckInPage() {
  return (
    <Suspense fallback={<div className="loading-luna"><Luna mood="checkin" size={100} decorative /></div>}>
      <CheckInWorkspace />
    </Suspense>
  );
}
