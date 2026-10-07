"use client";

import Link from "next/link";
import { useEffect, useMemo, useState, useSyncExternalStore } from "react";

import { Icon } from "@/components/nav-icon";
import { Luna } from "@/components/luna";
import { MoonMark } from "@/components/moon-mark";
import { plantStage } from "@/components/plant";
import { moonDays } from "@/lib/moon-days";
import { apiRequest } from "@/lib/api";
import { readOpenConversationId } from "@/lib/conversation";
import { activityPlantStage, loadActivityHistory, PARTICIPATION_WORDS, type ActivityHistoryItem } from "@/lib/garden";
import { usePreferences } from "@/lib/preferences";
import { useReminders } from "@/lib/reminders";
import { greeting, useTimeOfDay } from "@/lib/time-of-day";
import type { OutcomeRecord, ReflectionRecord, Resource } from "@/lib/types";

const HELP_FACES = [
  { score: 1, label: "Not at all" },
  { score: 2, label: "A little" },
  { score: 3, label: "Somewhat" },
  { score: 4, label: "Helped" },
  { score: 5, label: "A lot" },
];

function subscribeToStorage(callback: () => void) {
  window.addEventListener("storage", callback);
  return () => window.removeEventListener("storage", callback);
}

export default function HomePage() {
  const time = useTimeOfDay();
  const [preferences, , preferencesLoaded] = usePreferences();
  const reminders = useReminders();
  const [reflections, setReflections] = useState<ReflectionRecord[]>([]);
  const [outcomes, setOutcomes] = useState<OutcomeRecord[]>([]);
  const [catalog, setCatalog] = useState<Resource[]>([]);
  const [activities, setActivities] = useState<ActivityHistoryItem[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [offline, setOffline] = useState(false);
  const openChat = useSyncExternalStore(
    subscribeToStorage,
    () => Boolean(readOpenConversationId()),
    () => false,
  );
  const [now, setNow] = useState(() => Date.now());

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
    void loadActivityHistory(8).then((items) => setActivities(items ?? []));
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
  // Legacy check-ins and chat activity reports share one garden, oldest to newest.
  const recent = [
    ...reflections.slice(0, 8).map((item) => {
      const outcome = outcomeByDecision.get(item.decision.decision_id);
      return { key: item.id, at: item.created_at, stage: plantStage(outcome?.helpfulness, outcome?.completed), label: undefined as string | undefined };
    }),
    ...activities.map((item) => ({
      key: item.id, at: item.reported_at, stage: activityPlantStage(item),
      label: `${item.title}: ${PARTICIPATION_WORDS[item.participation]}`,
    })),
  ].sort((a, b) => Date.parse(b.at) - Date.parse(a.at)).slice(0, 8).reverse();

  // Day marks only say that a moment was recorded; they never grade how it went.
  const week = moonDays(reflections.map((item) => item.created_at), activities.map((item) => item.reported_at), 7, new Date(now));

  return (
    <>
      <section className="hero" aria-labelledby="home-greeting">
        <Luna mood={lunaMood} size={132} />
        <span className="kicker home-date"><MoonMark kind="activity" size={15} />
          {time ? new Date().toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" }) : "\u00a0"}</span>
        <h1 id="home-greeting">{time ? greeting(time) : "Hello"}</h1>
        <p>{pending && minutesLeft <= 0 ? "I’ve been wondering how your small step went." : "How are you arriving today?"}</p>
        <Link className="btn btn-primary btn-big" href="/talk">
          {openChat ? "Continue with Luna" : "Talk with Luna"}
        </Link>
        <div className="home-paths">
          <Link href="/journal"><Icon name="journal" /><span><strong>Write a journal entry</strong><small>Make room for what’s on your mind.</small></span><Icon name="arrow" /></Link>
          <Link href="/discover"><Icon name="explore" /><span><strong>Explore useful resources</strong><small>Find a small place to begin.</small></span><Icon name="arrow" /></Link>
        </div>
      </section>

      <div className="page page-wide home-details">
        {preferencesLoaded && !preferences.onboarded && <p className="note">
          You can use JournalPulse with private defaults. <Link href="/welcome?next=%2F">Choose your optional setup preferences</Link> whenever you’re ready.
        </p>}
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
                    <span className="scale-number" aria-hidden="true">{face.score}</span>
                    <span>{face.label}</span>
                  </Link>
                ))}
              </div>
              <Link className="link-btn" href={`/check-in?decision=${pending.decision.decision_id}&tried=no`}>I haven’t tried it yet</Link>
            </section>
          ) : loaded && recent.length > 0 ? null : (
            // Returning people already know the rhythm; the explanation is for a first visit.
            <section className="card lav" aria-labelledby="how-heading">
              <span className="eyebrow">How it works</span>
              <h2 id="how-heading">Talk, try one small thing, check in.</h2>
              <p className="muted">A quick chat with Luna, one gentle idea from a reviewed list, and a check-in later to see what helped.</p>
            </section>
          )}

          <section className="card week-card" aria-labelledby="garden-heading">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2 id="garden-heading">This week</h2>
              <Link className="link-btn" href="/journey">See your journey</Link>
            </div>
            {!loaded ? (
              <div className="skeleton" style={{ minHeight: 70 }} />
            ) : (
              <>
                <ol className="moon-week" aria-label="Days you showed up this week">
                  {week.map((day) => (
                    <li key={day.key} aria-current={day.today ? "date" : undefined}>
                      <MoonMark kind={day.kind} today={day.today} size={28} label={day.label} />
                      <span aria-hidden="true">{day.date.toLocaleDateString(undefined, { weekday: "narrow" })}</span>
                    </li>
                  ))}
                </ol>
                <p className="small muted">{recent.length
                  ? `${recent.length} recent ${recent.length === 1 ? "moment" : "moments"}, in your own words.`
                  : "Each check-in leaves a small mark here. Your first one is a chat away."}</p>
              </>
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
