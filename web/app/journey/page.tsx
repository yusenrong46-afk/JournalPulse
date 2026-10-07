"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { Luna } from "@/components/luna";
import { MoonMark } from "@/components/moon-mark";
import { moonDays } from "@/lib/moon-days";
import { apiRequest } from "@/lib/api";
import { feelingById } from "@/lib/feelings";
import { loadActivityHistory, PARTICIPATION_WORDS, type ActivityHistoryItem } from "@/lib/garden";
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

function activityWasTried(item: ActivityHistoryItem) {
  return item.participation === "completed" || item.participation === "partial";
}

function matchesQuery(query: string, ...values: (string | null | undefined)[]) {
  return values.filter(Boolean).join(" ").toLowerCase().includes(query);
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
  const [countsUnavailable, setCountsUnavailable] = useState(false);
  const [deleteError, setDeleteError] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([
      apiRequest<{ items: ReflectionRecord[] }>("/v1/reflections?limit=100"),
      apiRequest<{ items: OutcomeRecord[] }>("/v1/outcomes"),
      apiRequest<{ items: Resource[] }>("/v1/resources"),
      // Loaded separately so an older API or a failure never hides legacy check-ins.
      loadActivityHistory(100),
    ])
      .then(([history, recorded, resources, chatActivities]) => {
        if (cancelled) return;
        if (history.status === "fulfilled") setReflections(history.value.items);
        if (recorded.status === "fulfilled") setOutcomes(recorded.value.items);
        if (resources.status === "fulfilled") setCatalog(resources.value.items);
        const chatFailed = chatActivities.status === "rejected" || chatActivities.value === null;
        if (chatActivities.status === "fulfilled") setActivities(chatActivities.value ?? []);
        setActivitiesFailed(chatFailed);
        setCountsUnavailable(history.status === "rejected" || recorded.status === "rejected" || chatFailed);
        setError([history, recorded, resources].some((result) => result.status === "rejected"));
        setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  const outcomeByDecision = useMemo(() => new Map(outcomes.map((item) => [item.decision_id, item])), [outcomes]);
  const titleById = useMemo(() => new Map(catalog.map((item) => [item.id, item.title])), [catalog]);
  const titleOf = (id: string) => titleById.get(id) ?? id.replaceAll("_", " ");

  const oldestFirst = [...reflections].sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at));
  const tried = reflections.flatMap((item) => {
    const outcome = outcomeByDecision.get(item.decision.decision_id);
    return outcome?.completed ? [outcome] : [];
  });
  const triedActivities = activities.filter(activityWasTried);
  const helpedCount = tried.filter((item) => (item.helpfulness ?? 0) >= 4).length
    + triedActivities.filter((item) => (item.helpfulness ?? 0) >= 4).length;

  const helped = useMemo(() => {
    const byAction = new Map<string, { title: string; source: string; total: number; helped: number; sum: number }>();
    for (const reflection of reflections) {
      const outcome = outcomeByDecision.get(reflection.decision.decision_id);
      if (!outcome?.completed || !outcome.helpfulness) continue;
      const id = `reflection:${reflection.decision.action_id}`;
      const entry = byAction.get(id) ?? {
        title: titleById.get(reflection.decision.action_id) ?? reflection.decision.action_id.replaceAll("_", " "),
        source: "From reflections", total: 0, helped: 0, sum: 0,
      };
      entry.total += 1;
      entry.sum += outcome.helpfulness;
      if (outcome.helpfulness >= 4) entry.helped += 1;
      byAction.set(id, entry);
    }
    for (const activity of activities) {
      if (!activityWasTried(activity) || !activity.helpfulness) continue;
      // History exposes titles and kinds, not resource identities. Keep these
      // ratings separate from catalog actions instead of assuming they match.
      const id = `chat:${JSON.stringify([activity.kind, activity.title])}`;
      const entry = byAction.get(id) ?? { title: activity.title, source: "From chats", total: 0, helped: 0, sum: 0 };
      entry.total += 1;
      entry.sum += activity.helpfulness;
      if (activity.helpfulness >= 4) entry.helped += 1;
      byAction.set(id, entry);
    }
    return [...byAction.entries()]
      .map(([id, entry]) => ({ id, ...entry, average: entry.sum / entry.total }))
      .sort((a, b) => b.average - a.average || b.total - a.total)
      .slice(0, 5);
  }, [activities, outcomeByDecision, reflections, titleById]);

  const searchTerm = query.trim().toLowerCase();
  const filtered = reflections.filter((item) => {
    const outcome = outcomeByDecision.get(item.decision.decision_id);
    return matchesQuery(searchTerm,
      item.reflection.summary,
      titleOf(item.decision.action_id),
      !outcome ? "check-in waiting" : outcome.completed ? HELP_WORDS[outcome.helpfulness ?? 0] || "tried it" : "skipped",
      ...item.state.emotion_tags.map((tag) => feelingById(tag)?.label ?? tag),
    );
  });
  const filteredActivities = activities.filter((item) => matchesQuery(searchTerm,
    item.title, item.kind, item.goal, PARTICIPATION_WORDS[item.participation],
    item.state_change ? CHANGE_WORDS[item.state_change] : null,
    activityWasTried(item) ? HELP_WORDS[item.helpfulness ?? 0] : null,
  ));

  async function remove(id: string) {
    if (deletingId) return;
    if (!window.confirm("Delete this entry and its check-in? This can’t be undone.")) return;
    setDeletingId(id);
    setDeleteError(false);
    const decisionId = reflections.find((item) => item.id === id)?.decision.decision_id;
    try {
      await apiRequest(`/v1/reflections/${id}`, { method: "DELETE" });
      setReflections((current) => current.filter((item) => item.id !== id));
      // The server cascades the check-in too. Keep the local totals and garden
      // consistent immediately, rather than counting the deleted outcome.
      setOutcomes((current) => current.filter((item) => item.decision_id !== decisionId));
    } catch {
      setDeleteError(true);
    } finally {
      setDeletingId(null);
    }
  }


  const fortnight = moonDays(reflections.map((item) => item.created_at), activities.map((item) => item.reported_at), 14);

  if (!loading && !error && !activitiesFailed && reflections.length === 0 && activities.length === 0) {
    return (
      <div className="page">
        <header className="page-head"><h1>Your journey</h1></header>
        <section className="card center">
          <Luna mood="checkin" size={120} />
          <h2>A little space to look back.</h2>
          <p className="muted">Your reflections and activity check-ins will gather here. There’s no pace to keep.</p>
          <Link className="btn btn-primary" href="/talk">Talk with Luna</Link>
        </section>
      </div>
    );
  }

  return (
    <div className="page page-wide journey-page">
      <header className="page-head">
        <h1>Your journey</h1>
        <p>A quiet look at what you’ve felt, tried, and found helpful.</p>
      </header>

      {error && <p className="note error" role="status">Some of your journey couldn’t load. The history available is shown below; nothing was changed.</p>}
      {deleteError && <p className="note error" role="alert">That check-in couldn’t be deleted. Please try again.</p>}

      <section className="moon-calendar" aria-labelledby="calendar-heading">
        <div className="moon-calendar-head">
          <h2 id="calendar-heading">Last two weeks</h2>
          <span className="small muted">a mark for each day you showed up</span>
        </div>
        {loading ? (
          <div className="skeleton" role="status" aria-label="Loading your journey" />
        ) : (
          <>
            <ol className="moon-grid">
              {fortnight.map((day) => (
                <li key={day.key} aria-current={day.today ? "date" : undefined}>
                  <span aria-hidden="true">{day.date.toLocaleDateString(undefined, { weekday: "narrow" })}</span>
                  <MoonMark kind={day.kind} today={day.today} size={26} label={day.label} />
                </li>
              ))}
            </ol>
            <p className="moon-legend small muted">
              <span><MoonMark kind="activity" size={14} /> activity check-in</span>
              <span><MoonMark kind="moment" size={14} /> reflection or chat moment</span>
            </p>
          </>
        )}
      </section>

      <section className="stat-grid" aria-label="Your numbers">
        <div className="stat"><strong>{loading || countsUnavailable ? "—" : reflections.length + activities.length}</strong><span>moments recorded</span></div>
        <div className="stat"><strong>{loading || countsUnavailable ? "—" : tried.length + triedActivities.length}</strong><span>small steps tried</span></div>
        <div className="stat"><strong>{loading || countsUnavailable ? "—" : helpedCount}</strong><span>rated helpful</span></div>
        <div className="stat"><strong>{loading || activitiesFailed ? "—" : activities.length}</strong><span>activity check-ins from chats</span></div>
      </section>
      <p className="small muted">Based on your latest 100 reflections and 100 activity check-ins. “Rated helpful” counts your ratings of 4 or 5 after trying a step.</p>

      <section className="stack" aria-labelledby="history-heading">
        <h2 id="history-heading">Your check-ins</h2>
        <label className="sr-only" htmlFor="journey-search">Search your check-ins</label>
        <input id="journey-search" className="search" type="search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search feelings, activities, or reflections" aria-describedby="journey-search-hint" />
        <p id="journey-search-hint" className="small muted">Search both histories below. Your summary above stays the same.</p>
        {searchTerm && !loading && <p className="small muted" role="status">{filtered.length + filteredActivities.length} {filtered.length + filteredActivities.length === 1 ? "matching check-in" : "matching check-ins"}</p>}
      </section>

      <section className="journey-list" aria-labelledby="chat-activities-heading">
        <h2 id="chat-activities-heading">Activities from your chats</h2>
        {loading ? <p className="muted">Loading your activity check-ins…</p> : activitiesFailed ? (
          <p className="muted" role="status">Activities from your chats couldn’t load. Nothing was changed.</p>
        ) : filteredActivities.length ? (
          <ul className="activity-history">
            {filteredActivities.map((item) => (
              <li key={item.id}>
                <strong>{item.title}</strong>
                <span className="small muted">
                  <time dateTime={item.reported_at}>{dateLabel(item.reported_at)}</time>
                  {" · "}{PARTICIPATION_WORDS[item.participation]}
                  {item.state_change ? ` · ${CHANGE_WORDS[item.state_change]}` : ""}
                  {activityWasTried(item) && item.helpfulness ? ` · ${HELP_WORDS[item.helpfulness]}` : ""}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="muted">{searchTerm ? "No activity check-ins match this search." : "When you report back on an activity with Luna, your check-in appears here—even if you didn’t try it."}</p>
        )}
        <p className="small muted">These are your own check-ins. Finishing a timer is never counted as trying it.</p>
      </section>

      <section className="stack" aria-labelledby="entries-heading">
        <h2 id="entries-heading">Past reflections</h2>
        {loading && <p className="muted">Loading your reflections…</p>}
        {filtered.map((item) => {
          const outcome = outcomeByDecision.get(item.decision.decision_id);
          return (
            <article className="entry" key={item.id}>
              <div className="entry-head">
                <time dateTime={item.created_at}>{dateLabel(item.created_at)}</time>
                <button className="link-btn" type="button" disabled={deletingId !== null} aria-label={`Delete the check-in from ${dateLabel(item.created_at)}`} onClick={() => void remove(item.id)}>{deletingId === item.id ? "Deleting…" : "Delete"}</button>
              </div>
              {item.state.emotion_tags.length > 0 && (
                <div className="chips">
                  {item.state.emotion_tags.map((tag) => {
                    const feeling = feelingById(tag);
                    return <span className="tag" key={tag}>{feeling?.label ?? tag.replaceAll("_", " ")}</span>;
                  })}
                </div>
              )}
              <p>{item.reflection.summary}</p>
              <div className="row small">
                <span className="tag sun">{outcome?.completed ? "Tried" : "Suggested"}: {titleOf(item.decision.action_id)}</span>
                {outcome ? (
                  <span className="tag sage">{outcome.completed ? HELP_WORDS[outcome.helpfulness ?? 0] || "Tried it" : "Skipped"}</span>
                ) : (
                  <Link className="link-btn" href={`/check-in?decision=${item.decision.decision_id}`}>Check in now</Link>
                )}
              </div>
            </article>
          );
        })}
        {!loading && filtered.length === 0 && <p className="muted">{searchTerm ? "No reflections match this search." : "Your saved reflections will appear here."}</p>}
      </section>
      <div className="home-grid journey-insights">
        <section className="card" aria-labelledby="helped-heading">
          <h2 id="helped-heading">What you found helpful</h2>
          {loading ? <p className="muted">Loading your ratings…</p> : helped.length ? (
            <div className="helped">
              {helped.map((item) => (
                <div className="helped-row" key={item.id}>
                  <strong>{item.title}</strong>
                  <span className="small muted">{item.average.toFixed(1)} / 5</span>
                  <span className="small muted">{item.source} · {item.total} {item.total === 1 ? "rating" : "ratings"} · {item.helped} rated helpful</span>
                  <div className="helped-bar" aria-hidden="true"><i style={{ width: `${(item.average / 5) * 100}%` }} /></div>
                </div>
              ))}
            </div>
          ) : (
            <p className="muted">When you rate a step you’ve tried, your ratings will appear here. Feeling the same or skipping a step doesn’t count as a helpfulness rating.</p>
          )}
        </section>

        <section className="card" aria-labelledby="mood-heading">
          <h2 id="mood-heading">Feelings from your reflections</h2>
          {loading ? <p className="muted">Loading your reflections…</p> : oldestFirst.length ? <>
          <div className="mood-dots" role="img" aria-label={oldestFirst.slice(-30).map((item) => `${dateLabel(item.created_at)}: ${item.state.valence >= 0.25 ? "lighter" : item.state.valence <= -0.25 ? "heavier" : "in between"}`).join("; ")}>
            {oldestFirst.slice(-30).map((item) => (
              <span key={item.id} style={{ background: moodColor(item.state.valence) }} title={dateLabel(item.created_at)} />
            ))}
          </div>
          <div className="row small muted">
            <span><span className="tag" style={{ background: "var(--lav)" }}>&nbsp;</span> heavier</span>
            <span><span className="tag" style={{ background: "#f1d9a6" }}>&nbsp;</span> in between</span>
            <span><span className="tag" style={{ background: "var(--sage)" }}>&nbsp;</span> lighter</span>
          </div>
          <p className="small muted">Up to 30 reflections, oldest to newest. Activity reports don’t add a mood estimate.</p>
          </> : <p className="muted">Your saved reflections will add a view of how you’ve been feeling. Activity check-ins stay in your history below.</p>}
        </section>
      </div>

    </div>
  );
}
