"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState, useSyncExternalStore } from "react";

import { Luna } from "@/components/luna";
import { Plant, plantStage } from "@/components/plant";
import { apiRequest } from "@/lib/api";
import { readOpenConversationId } from "@/lib/conversation";
import { usePreferences } from "@/lib/preferences";
import { useReminders } from "@/lib/reminders";
import { greeting, useTimeOfDay } from "@/lib/time-of-day";
import type { OutcomeRecord, ReflectionRecord, Resource } from "@/lib/types";

const HELP_FACES = [
  { score: 1, emoji: "😣", label: "Not at all" },
  { score: 2, emoji: "😕", label: "A little" },
  { score: 3, emoji: "😐", label: "Somewhat" },
  { score: 4, emoji: "🙂", label: "Helped" },
  { score: 5, emoji: "😄", label: "A lot" },
];

function subscribeToStorage(callback: () => void) {
  window.addEventListener("storage", callback);
  return () => window.removeEventListener("storage", callback);
}

export default function HomePage() {
  const router = useRouter();
  const time = useTimeOfDay();
  const [preferences, , preferencesLoaded] = usePreferences();
  const reminders = useReminders();
  const [reflections, setReflections] = useState<ReflectionRecord[]>([]);
  const [outcomes, setOutcomes] = useState<OutcomeRecord[]>([]);
  const [catalog, setCatalog] = useState<Resource[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [offline, setOffline] = useState(false);
  const openChat = useSyncExternalStore(
    subscribeToStorage,
    () => Boolean(readOpenConversationId()),
    () => false,
  );
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (preferencesLoaded && !preferences.onboarded) router.replace("/welcome");
  }, [preferences.onboarded, preferencesLoaded, router]);

  useEffect(() => {
    const interval = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    Promise.all([
      apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=30"),
      apiRequest<{ items: OutcomeRecord[] }>("/v1/outcomes"),
      apiRequest<{ items: Resource[] }>("/v1/resources"),
    ])
      .then(([history, recorded, resources]) => {
        setReflections(history.items);
        setOutcomes(recorded.items);
        setCatalog(resources.items);
      })
      .catch(() => setOffline(true))
      .finally(() => setLoaded(true));
  }, []);

  const outcomeByDecision = useMemo(
    () => new Map(outcomes.map((item) => [item.decision_id, item])),
    [outcomes],
  );
  const pending = reflections.find((item) => !outcomeByDecision.has(item.decision.decision_id)) ?? null;
  const pendingTitle = pending
    ? catalog.find((item) => item.id === pending.decision.action_id)?.title ??
      reminders.find((item) => item.decisionId === pending.decision.decision_id)?.actionTitle ??
      "your small step"
    : "";
  const reminder = pending ? reminders.find((item) => item.decisionId === pending.decision.decision_id) : undefined;
  const minutesLeft = reminder ? Math.ceil((new Date(reminder.dueAt).getTime() - now) / 60_000) : 0;

  const lunaMood = pending && minutesLeft <= 0 ? "checkin" : time === "night" ? "sleepy" : "idle";
  const recent = reflections.slice(0, 8).reverse();

  return (
    <>
      <section className="hero" aria-labelledby="home-greeting">
        <svg className="hills" viewBox="0 0 400 160" preserveAspectRatio="none" aria-hidden="true">
          <path d="M0 90 C70 50 130 60 200 88 C270 116 330 70 400 80 V160 H0 Z" fill="var(--hill-back)" opacity="0.7" />
          <path d="M0 120 C80 96 150 110 220 124 C290 138 340 110 400 118 V160 H0 Z" fill="var(--hill-front)" />
        </svg>
        <Luna mood={lunaMood} size={132} />
        <h1 id="home-greeting">{time ? greeting(time) : "Hello"}</h1>
        <p>{pending && minutesLeft <= 0 ? "I’ve been wondering how your small step went." : "How are you arriving today?"}</p>
        <Link className="btn btn-primary btn-big" href="/talk">
          {openChat ? "Continue with Luna" : "Talk with Luna"}
        </Link>
        <Link className="btn btn-ghost" href="/journal">Write a journal entry</Link>
        <Link className="btn btn-ghost" href="/discover">Explore useful resources</Link>
      </section>

      <div className="page page-wide">
        {offline && (
          <p className="note error" role="status">Luna can’t reach your journal right now. Your chat will still try when you’re back online.</p>
        )}

        <div className="home-grid">
          {pending ? (
            <section className="card sun" aria-labelledby="checkin-heading">
              <span className="eyebrow">Check-in</span>
              <h2 id="checkin-heading">Earlier you picked “{pendingTitle}”. Did it help?</h2>
              {minutesLeft > 0 ? (
                <p className="muted">Give it a try first. I’ll ask again in about {minutesLeft} minute{minutesLeft === 1 ? "" : "s"}.</p>
              ) : null}
              <div className="faces" role="group" aria-label="Did it help?">
                {HELP_FACES.map((face) => (
                  <Link key={face.score} className="face" href={`/check-in?decision=${pending.decision.decision_id}&h=${face.score}`}>
                    <span aria-hidden="true">{face.emoji}</span>
                    <span>{face.label}</span>
                  </Link>
                ))}
              </div>
              <Link className="link-btn" href={`/check-in?decision=${pending.decision.decision_id}&tried=no`}>I haven’t tried it yet</Link>
            </section>
          ) : (
            <section className="card lav" aria-labelledby="how-heading">
              <span className="eyebrow">How it works</span>
              <h2 id="how-heading">Talk, try one small thing, check in.</h2>
              <p className="muted">A quick chat with Luna, one gentle idea from a reviewed list, and a check-in later to see what helped.</p>
            </section>
          )}

          <section className="card" aria-labelledby="garden-heading">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2 id="garden-heading">Your garden</h2>
              <Link className="link-btn" href="/journey">See your journey</Link>
            </div>
            {!loaded ? (
              <div className="skeleton" style={{ minHeight: 70 }} />
            ) : recent.length ? (
              <div className="mini-garden">
                {recent.map((item, index) => {
                  const outcome = outcomeByDecision.get(item.decision.decision_id);
                  return <Plant key={item.id} index={index} size={36} stage={plantStage(outcome?.helpfulness, outcome?.completed)} />;
                })}
              </div>
            ) : (
              <p className="muted">Each check-in plants something here. Your first one is a chat away.</p>
            )}
          </section>
        </div>

        <p className="small muted" style={{ textAlign: "center" }}>
          Luna is a companion, not a therapist. In a crisis in Canada, call or text <a href="https://988.ca/" target="_blank" rel="noreferrer">9-8-8</a>.
        </p>
      </div>
    </>
  );
}
