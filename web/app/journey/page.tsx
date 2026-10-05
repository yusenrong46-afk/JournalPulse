"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { Luna } from "@/components/luna";
import { Plant, plantStage } from "@/components/plant";
import { apiRequest } from "@/lib/api";
import { feelingById } from "@/lib/feelings";
import { activityPlantStage, loadActivityHistory, PARTICIPATION_WORDS, type ActivityHistoryItem } from "@/lib/garden";
import type { OutcomeRecord, ReflectionRecord, Resource } from "@/lib/types";

const HELP_WORDS = ["", "Didn’t help", "Helped a little", "Helped somewhat", "Helped", "Helped a lot"];
const CHANGE_WORDS: Record<string, string> = {
  toward_target: "felt closer to what you wanted",
  same: "about the same",
  away_from_target: "felt further away",
  unsure: "not sure what changed",
};

function moodColor(valence: number) {
  if (valence >= 0.25) return "var(--sage)";
  if (valence <= -0.25) return "var(--lav)";
  return "#f1d9a6";
}

function dateLabel(value: string) {
  return new Date(value).toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
}

export default function JourneyPage() {
  const [reflections, setReflections] = useState<ReflectionRecord[]>([]);
  const [outcomes, setOutcomes] = useState<OutcomeRecord[]>([]);
  const [catalog, setCatalog] = useState<Resource[]>([]);
  const [activities, setActivities] = useState<ActivityHistoryItem[]>([]);
  const [activitiesFailed, setActivitiesFailed] = useState(false);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    Promise.all([
      apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=100"),
      apiRequest<{ items: OutcomeRecord[] }>("/v1/outcomes"),
      apiRequest<{ items: Resource[] }>("/v1/resources"),
      // Loaded separately so an older API or a failure never hides legacy check-ins.
      loadActivityHistory(100),
    ])
      .then(([history, recorded, resources, chatActivities]) => {
        setReflections(history.items);
        setOutcomes(recorded.items);
        setCatalog(resources.items);
        setActivities(chatActivities ?? []);
        setActivitiesFailed(chatActivities === null);
      })
      .catch(() => setError(true))
      .finally(() => setLoading(false));
  }, []);

  const outcomeByDecision = useMemo(() => new Map(outcomes.map((item) => [item.decision_id, item])), [outcomes]);
  const titleOf = (id: string) => catalog.find((item) => item.id === id)?.title ?? id.replaceAll("_", " ");

  const oldestFirst = [...reflections].reverse();
  const tried = outcomes.filter((item) => item.completed);
  const helpedCount = tried.filter((item) => (item.helpfulness ?? 0) >= 4).length;

  const helped = useMemo(() => {
    const byAction = new Map<string, { total: number; helped: number; sum: number }>();
    for (const reflection of reflections) {
      const outcome = outcomeByDecision.get(reflection.decision.decision_id);
      if (!outcome?.completed || !outcome.helpfulness) continue;
      const entry = byAction.get(reflection.decision.action_id) ?? { total: 0, helped: 0, sum: 0 };
      entry.total += 1;
      entry.sum += outcome.helpfulness;
      if (outcome.helpfulness >= 4) entry.helped += 1;
      byAction.set(reflection.decision.action_id, entry);
    }
    return [...byAction.entries()]
      .map(([id, entry]) => ({ id, ...entry, average: entry.sum / entry.total }))
      .sort((a, b) => b.average - a.average || b.total - a.total)
      .slice(0, 5);
  }, [outcomeByDecision, reflections]);

  const filtered = reflections.filter((item) => {
    if (!query.trim()) return true;
    const text = [
      item.reflection.summary,
      titleOf(item.decision.action_id),
      ...item.state.emotion_tags.map((tag) => feelingById(tag)?.label ?? tag),
    ]
      .join(" ")
      .toLowerCase();
    return text.includes(query.trim().toLowerCase());
  });

  async function remove(id: string) {
    if (!window.confirm("Delete this entry and its check-in? This can’t be undone.")) return;
    const decisionId = reflections.find((item) => item.id === id)?.decision.decision_id;
    try {
      await apiRequest(`/v1/reflections/${id}`, { method: "DELETE" });
      setReflections((current) => current.filter((item) => item.id !== id));
      // The server cascades the check-in too. Keep the local totals and garden
      // consistent immediately, rather than counting the deleted outcome.
      setOutcomes((current) => current.filter((item) => item.decision_id !== decisionId));
    } catch {
      setError(true);
    }
  }

  const garden = [
    ...oldestFirst.map((item) => {
      const outcome = outcomeByDecision.get(item.decision.decision_id);
      const status = !outcome ? "check-in waiting" : outcome.completed ? HELP_WORDS[outcome.helpfulness ?? 0] || "tried it" : "skipped";
      return {
        key: item.id, at: item.created_at,
        stage: plantStage(outcome?.helpfulness, outcome?.completed),
        label: `${dateLabel(item.created_at)}: ${status}`,
      };
    }),
    ...activities.map((item) => ({
      key: item.id, at: item.reported_at, stage: activityPlantStage(item),
      label: `${dateLabel(item.reported_at)}: ${item.title}, ${PARTICIPATION_WORDS[item.participation].toLowerCase()}`,
    })),
  ].sort((a, b) => Date.parse(a.at) - Date.parse(b.at));

  if (!loading && !error && reflections.length === 0 && activities.length === 0) {
    return (
      <div className="page">
        <header className="page-head"><h1>Your journey</h1></header>
        <section className="card center">
          <Luna mood="checkin" size={120} />
          <h2>Your garden is ready to grow.</h2>
          <p className="muted">Each time you try a small step and check in, a new plant appears here.</p>
          <Link className="btn btn-primary" href="/talk">Talk with Luna</Link>
        </section>
      </div>
    );
  }

  return (
    <div className="page page-wide">
      <header className="page-head">
        <h1>Your journey</h1>
        <p>Every check-in grows your garden. These are your own patterns, not a diagnosis.</p>
      </header>

      {error && <p className="note error" role="status">Some of your journey couldn’t load. Nothing was changed.</p>}

      <section aria-label="Your garden">
        {loading ? (
          <div className="skeleton" />
        ) : (
          <div className="garden">
            {garden.map((item, index) => (
              <Plant key={item.key} index={index} stage={item.stage} size={48} label={item.label} />
            ))}
          </div>
        )}
      </section>

      <section className="stat-grid" aria-label="Your numbers">
        <div className="stat"><strong>{reflections.length}</strong><span>check-ins with Luna</span></div>
        <div className="stat"><strong>{tried.length}</strong><span>small steps tried</span></div>
        <div className="stat"><strong>{helpedCount}</strong><span>really helped</span></div>
        <div className="stat"><strong>{activities.length}</strong><span>activity check-ins from chats</span></div>
      </section>

      <section className="card" aria-labelledby="chat-activities-heading">
        <h2 id="chat-activities-heading">Activities from your chats</h2>
        {activitiesFailed ? (
          <p className="muted" role="status">Activities from your chats couldn’t load. Nothing was changed.</p>
        ) : activities.length ? (
          <ul className="activity-history">
            {activities.map((item) => (
              <li key={item.id}>
                <strong>{item.title}</strong>
                <span className="small muted">
                  <time dateTime={item.reported_at}>{dateLabel(item.reported_at)}</time>
                  {" · "}{PARTICIPATION_WORDS[item.participation]}
                  {item.state_change ? ` · ${CHANGE_WORDS[item.state_change]}` : ""}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">When you try an activity with Luna and check in, it appears here in your own words.</p>
        )}
        <p className="small muted">These are your own check-ins. Finishing a timer is never counted as trying it.</p>
      </section>

      <div className="home-grid">
        <section className="card" aria-labelledby="helped-heading">
          <h2 id="helped-heading">What helps you</h2>
          {helped.length ? (
            <div className="helped">
              {helped.map((item) => (
                <div className="helped-row" key={item.id}>
                  <strong>{titleOf(item.id)}</strong>
                  <span className="small muted">helped {item.helped} of {item.total}</span>
                  <div className="helped-bar"><i style={{ width: `${(item.average / 5) * 100}%` }} /></div>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">After a few check-ins, Luna will show which small steps help you most.</p>
          )}
        </section>

        <section className="card" aria-labelledby="mood-heading">
          <h2 id="mood-heading">How you’ve been feeling</h2>
          <div className="mood-dots" role="img" aria-label={`Mood over your last ${Math.min(oldestFirst.length, 30)} check-ins`}>
            {oldestFirst.slice(-30).map((item) => (
              <span key={item.id} style={{ background: moodColor(item.state.valence) }} title={dateLabel(item.created_at)} />
            ))}
          </div>
          <div className="row small muted">
            <span><span className="tag" style={{ background: "var(--lav)" }}>&nbsp;</span> heavier</span>
            <span><span className="tag" style={{ background: "#f1d9a6" }}>&nbsp;</span> in between</span>
            <span><span className="tag" style={{ background: "var(--sage)" }}>&nbsp;</span> lighter</span>
          </div>
        </section>
      </div>

      <section className="stack" aria-labelledby="entries-heading">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h2 id="entries-heading">Past check-ins</h2>
        </div>
        <label className="sr-only" htmlFor="journey-search">Search your check-ins</label>
        <input id="journey-search" className="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search feelings or small steps" />
        {filtered.map((item) => {
          const outcome = outcomeByDecision.get(item.decision.decision_id);
          return (
            <article className="entry" key={item.id}>
              <div className="entry-head">
                <time dateTime={item.created_at}>{dateLabel(item.created_at)}</time>
                <button className="link-btn" type="button" aria-label={`Delete the check-in from ${dateLabel(item.created_at)}`} onClick={() => void remove(item.id)}>Delete</button>
              </div>
              {item.state.emotion_tags.length > 0 && (
                <div className="chips">
                  {item.state.emotion_tags.map((tag) => {
                    const feeling = feelingById(tag);
                    return <span className="tag" key={tag}>{feeling ? `${feeling.emoji} ${feeling.label}` : tag.replaceAll("_", " ")}</span>;
                  })}
                </div>
              )}
              <p>{item.reflection.summary}</p>
              <div className="row small">
                <span className="tag sun">Tried: {titleOf(item.decision.action_id)}</span>
                {outcome ? (
                  <span className="tag sage">{outcome.completed ? HELP_WORDS[outcome.helpfulness ?? 0] || "Tried it" : "Skipped"}</span>
                ) : (
                  <Link className="link-btn" href={`/check-in?decision=${item.decision.decision_id}`}>Check in now</Link>
                )}
              </div>
            </article>
          );
        })}
        {!loading && filtered.length === 0 && reflections.length > 0 && <p className="muted">Nothing matches that search.</p>}
      </section>
    </div>
  );
}
