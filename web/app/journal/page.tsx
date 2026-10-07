"use client";

import Link from "next/link";
import { Luna } from "@/components/luna";
import { resolveLunaMood } from "@/lib/luna-motion";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";

import { ApiError } from "@/lib/api";
import {
  deleteJournalEntry,
  listJournalEntries,
  readJournalEntry,
  reflectJournalEntry,
  validJournalEntryId,
} from "@/lib/journal";
import type { JournalEntry, JournalReflectionResult } from "@/lib/journal-types";
import { usePreferences } from "@/lib/preferences";
import { useTimeOfDay } from "@/lib/time-of-day";
import { submitJournalSave, useJournalSave } from "@/lib/journal-save";
import { TAB_DRAFT_NOTE, VOLATILE_DRAFT_NOTE, clearSourceDrafts, readTabValue, useTabValue, writeTabValue } from "@/lib/tab-session";

function dateLabel(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit",
  });
}

function dayLabel(value?: string): string {
  return (value ? new Date(value) : new Date()).toLocaleDateString(undefined, {
    weekday: "long", month: "long", day: "numeric",
  });
}

const PROMPTS = [
  "What would you like to remember?",
  "What stayed with you today?",
  "One small good thing",
  "What would you like to set down?",
];

function wordCount(text: string): number {
  return text.trim() ? text.trim().split(/\s+/).length : 0;
}

function momentLabel(): string {
  const now = new Date();
  const hour = now.getHours();
  const part = hour < 12 ? "Morning" : hour < 17 ? "Afternoon" : hour < 22 ? "Evening" : "Night";
  return `${part} · ${now.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })}`;
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback;
}

/** Keyed by the URL selection: changing entries resets consent and transient replies. */
function SavedEntry({ entryId, invalid, onDeleted }: {
  entryId: string | null;
  invalid: boolean;
  onDeleted: (id: string) => void;
}) {
  const router = useRouter();
  const [preferences] = usePreferences();
  const [entry, setEntry] = useState<JournalEntry | null>(null);
  const [opening, setOpening] = useState(Boolean(entryId));
  const [deleting, setDeleting] = useState(false);
  const [reflecting, setReflecting] = useState(false);
  const [consent, setConsent] = useState(false);
  const [reflection, setReflection] = useState<JournalReflectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);
  const alive = useRef(true);
  const reflectionController = useRef<AbortController | null>(null);

  useEffect(() => {
    alive.current = true;
    const controller = new AbortController();
    if (entryId) {
      readJournalEntry(entryId, controller.signal)
        .then((saved) => { if (!controller.signal.aborted) setEntry(saved); })
        .catch((reason) => {
          if (!controller.signal.aborted) setError(errorMessage(reason, "This entry couldn’t load."));
        })
        .finally(() => { if (!controller.signal.aborted) setOpening(false); });
    }
    return () => {
      alive.current = false;
      controller.abort();
      reflectionController.current?.abort();
    };
  }, [entryId]);

  async function reflect() {
    if (!entry || !consent || reflecting || deleting) return;
    reflectionController.current?.abort();
    const controller = new AbortController();
    reflectionController.current = controller;
    const requestGeneration = ++generation.current;
    setReflecting(true);
    setReflection(null);
    setError(null);
    try {
      const reply = await reflectJournalEntry(entry.id, consent, preferences.locale, controller.signal);
      // Selection changes unmount this pane and abort the response; deletion also
      // invalidates its generation. The server independently checks source identity.
      if (!controller.signal.aborted && requestGeneration === generation.current) setReflection(reply);
    } catch (reason) {
      if (!controller.signal.aborted && requestGeneration === generation.current) {
        setError(errorMessage(reason, "Luna couldn’t reflect just now. Your entry is still saved."));
        if (reason instanceof ApiError && reason.status === 404) setEntry(null);
      }
    } finally {
      if (alive.current && requestGeneration === generation.current) setReflecting(false);
    }
  }

  async function remove() {
    if (!entry || deleting) return;
    const removedId = entry.id;
    if (!window.confirm("Delete this entry and any chats and saved choices based on it? This can’t be undone.")) return;
    generation.current += 1;
    reflectionController.current?.abort();
    setReflecting(false);
    setReflection(null);
    setDeleting(true);
    setError(null);
    try {
      await deleteJournalEntry(removedId);
      onDeleted(removedId);
      if (alive.current) {
        // Clear private content as soon as deletion succeeds; a cached route
        // transition can lag and must not keep displaying the deleted source.
        setEntry(null);
        setConsent(false);
        router.replace("/journal", { scroll: false });
      }
    } catch (reason) {
      if (alive.current) setError(errorMessage(reason, "This entry couldn’t be deleted. Please try again."));
    } finally {
      if (alive.current) setDeleting(false);
    }
  }

  return (
    <section className="journal-sheet journal-reading" aria-labelledby="saved-entry-heading" aria-busy={opening}>
      <h2 id="saved-entry-heading" className={entry ? "kicker saved-entry-label" : undefined}>{entry ? "Your saved entry" : "Return to an entry"}</h2>
      {invalid && <p className="note error" role="alert">This entry link is not valid. Choose a saved entry.</p>}
      {error && <p className="note error" role="alert">{error}</p>}
      {opening ? <p role="status">Opening your entry…</p> : entry ? (
        <>
          <header className="journal-sheet-head">
            <time className="journal-day" dateTime={entry.created_at}>{dayLabel(entry.created_at)}</time>
            <span className="saved-stamp" aria-hidden="true">Saved</span>
            <p className="small muted"><strong>Saved writing</strong> · {new Date(entry.created_at).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })} · Kept until you delete this entry.</p>
          </header>
          <p className="journal-entry-text">{entry.text}</p>
          <div className="journal-entry-actions">
            <button className="btn btn-ghost" disabled={deleting} onClick={() => void remove()}>
              {deleting ? "Deleting…" : "Delete entry"}
            </button>
          </div>
          <div className="journal-reflect journal-aside">
            <div className="journal-companion"><Luna mood={resolveLunaMood({
              waiting: reflecting, support: reflection?.safety.mode === "support",
              hasReply: Boolean(reflection), replyFeelings: reflection?.feelings,
            })} size={56} decorative /><h3>Reflect with Luna</h3></div>
            <p className="small muted">Optional: get one brief reply about this entry.</p>
            {/* Explain retention before consent, while the person can still choose. */}
            <p className="small muted">This reply disappears when you leave this entry or reload.</p>
            <label className="check-row">
              <input type="checkbox" checked={consent} disabled={reflecting || deleting}
                onChange={(event) => setConsent(event.target.checked)} />
              <span>Allow this entry to be sent to the private AI provider for a reflection.</span>
            </label>
            <button className="btn btn-primary" disabled={!consent || reflecting || deleting}
              onClick={() => void reflect()}>{reflecting ? "Luna is reflecting…" : "Reflect on this entry"}</button>
            {reflection && (
              <div className="note journal-reflection" role="status">
                <strong>{reflection.safety.mode === "support" ? "Support information" : "Luna’s reflection"}</strong>
                <p className="journal-entry-text">{reflection.reply}</p>
                <p className="small muted">Temporary reply. Your saved writing stays in your journal.</p>
              </div>
            )}
          </div>
          <div className="journal-reflect journal-aside">
            <h3>Discuss your entry</h3>
            <p className="small muted">Optional: keep exploring in a conversation with Luna.</p>
            <p className="small muted">
              Chat uses your saved entry only if you choose to share it; the temporary reflection is not carried into the conversation.
              You’ll choose whether to share this entry on the next screen.
            </p>
            <Link className="btn btn-soft" href={`/talk?entry=${entry.id}`}>Discuss with Luna</Link>
          </div>
        </>
      ) : <p className="muted">Choose a saved entry to reread it or request a reflection.</p>}
    </section>
  );
}

function JournalWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedParameter = searchParams.get("entry");
  const selectedId = validJournalEntryId(selectedParameter);
  const [entries, setEntries] = useState<JournalEntry[]>([]);
  const [draft, setDraft, draftVolatile] = useTabValue("journal");
  const [listLoading, setListLoading] = useState(true);
  const [hasMore, setHasMore] = useState(false);
  const saveState = useJournalSave();
  const saving = saveState.status === "saving";
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [prompt, setPrompt] = useState(PROMPTS[0]);
  // Local date and time exist only in the browser; rendering them on the server would
  // disagree with the reader's time zone and break hydration.
  const clientTime = useTimeOfDay();
  const listGeneration = useRef(0);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      listGeneration.current += 1;
    };
  }, []);

  useEffect(() => {
    let active = true;
    const requestGeneration = ++listGeneration.current;
    listJournalEntries()
      .then((page) => {
        if (!active || requestGeneration !== listGeneration.current) return;
        setEntries(page.items);
        setHasMore(page.items.length === page.limit);
      })
      .catch((reason) => {
        if (active && requestGeneration === listGeneration.current) {
          setError(errorMessage(reason, "Your saved entries couldn’t load. Please try again."));
        }
      })
      .finally(() => {
        if (active && requestGeneration === listGeneration.current) setListLoading(false);
      });
    return () => { active = false; };
  }, []);

  async function loadEntries(offset = 0) {
    const requestGeneration = ++listGeneration.current;
    setListLoading(true);
    setError(null);
    try {
      const page = await listJournalEntries(offset);
      if (requestGeneration !== listGeneration.current) return;
      setEntries((current) => offset === 0 ? page.items : [
        ...current, ...page.items.filter((item) => !current.some((saved) => saved.id === item.id)),
      ]);
      setHasMore(page.items.length === page.limit);
    } catch (reason) {
      if (requestGeneration === listGeneration.current) {
        setError(errorMessage(reason, "Your saved entries couldn’t load. Please try again."));
      }
    } finally {
      if (requestGeneration === listGeneration.current) setListLoading(false);
    }
  }

  async function save() {
    if (saving || !draft.trim()) return;
    setError(null);
    setNotice(null);
    const submitted = draft;
    const saved = await submitJournalSave(submitted);
    if (saved && mounted.current && !readTabValue("journal")) {
      router.replace(`/journal?entry=${saved.id}`, { scroll: false });
    }
  }

  useEffect(() => {
    if (saveState.status === "saved" && saveState.entry) {
      const saved = saveState.entry;
      queueMicrotask(() => {
        if (!mounted.current) return;
        setEntries((current) => [saved, ...current.filter((item) => item.id !== saved.id)]);
        setNotice("Entry saved. You can leave it here or ask Luna to reflect.");
        void loadEntries();
      });
    }
    if (saveState.status === "failed") queueMicrotask(() => { if (mounted.current) setError(saveState.error ?? "Your save could not be confirmed."); });
    // A completed save is reconciled on every mount, including returning before it finishes.
  }, [saveState]);

  function removed(id: string) {
    clearSourceDrafts(id);
    setEntries((current) => current.filter((item) => item.id !== id));
    setNotice("Entry and its linked chats deleted.");
    // The refreshed request supersedes any read that still contains deleted text.
    void loadEntries();
  }

  return (
    <div className="page page-wide journal-page">
      <header className="page-head">
        <div className="journal-title-row">
          <h1>Your journal</h1>
          {entries.length > 0 && <a className="entry-count" href="#entries-heading">
            {entries.length}{hasMore ? "+" : ""} {entries.length === 1 ? "entry" : "entries"}</a>}
        </div>
        <p>Your own words, kept as you wrote them. Reflection is always your choice.</p>
        <a className="subtle-link journal-jump" href="#entries-heading">Browse saved entries</a>
      </header>
      {error && <p className="note error" role="alert">{error}</p>}
      {notice && <p className="note" role="status">{notice}</p>}
      <div className="journal-layout">
      <section className="journal-archive" aria-labelledby="entries-heading" aria-busy={listLoading}>
        <h2 id="entries-heading">Saved entries</h2>
        <Link className="btn btn-soft" href="/journal" scroll={false}>New entry</Link>
        {entries.length === 0 && !listLoading && <p className="muted">Your saved writing will appear here.</p>}
        <ul className="journal-entry-list">
          {entries.map((saved) => (
            <li key={saved.id}>
              <Link className="journal-entry-link" href={`/journal?entry=${saved.id}`} scroll={false}
                aria-current={saved.id === selectedId ? "true" : undefined}>
                <time className="small muted" dateTime={saved.created_at}>{dateLabel(saved.created_at)}</time>
                <span className="journal-entry-preview">{saved.text.length > 150 ? `${saved.text.slice(0, 150)}…` : saved.text}</span>
                <span className="small journal-entry-open">Read entry</span>
              </Link>
            </li>
          ))}
        </ul>
        {listLoading && <p role="status">Loading entries…</p>}
        {hasMore && <button className="btn btn-soft" disabled={listLoading}
          onClick={() => void loadEntries(entries.length)}>Load older entries</button>}
        {error && !listLoading && <button className="btn btn-ghost" onClick={() => void loadEntries()}>Reload entries</button>}
      </section>
        <div className="journal-main">
          {selectedParameter ? (
          <SavedEntry key={selectedParameter ?? "empty"} entryId={selectedId}
            invalid={Boolean(selectedParameter && !selectedId)} onDeleted={removed} />
          ) : (
        <section className="journal-sheet journal-paper" aria-labelledby="write-heading">
          <div className="prompt-chips" role="group" aria-label="Optional prompts">
            {PROMPTS.slice(1).map((item) => (
              <button key={item} type="button" className="prompt-chip" aria-pressed={prompt === item}
                onClick={() => setPrompt(prompt === item ? PROMPTS[0] : item)}>{item}</button>
            ))}
          </div>
          <header className="journal-sheet-head">
            <span className="journal-day-wrap"><span className="kicker">{clientTime ? momentLabel() : "\u00a0"}</span>
              <span className="journal-day">{clientTime ? dayLabel() : "Today"}</span></span>
            <span className="journal-companion"><Luna mood={draft.trim() ? "listening" : "idle"} size={34} decorative />
              <h2 id="write-heading">Write at your pace.</h2></span>
          </header>
          {/* The field keeps one stable accessible name; the chosen prompt is a visible nudge only. */}
          <label className="journal-prompt" htmlFor="journal-writing">{prompt}</label>
          <textarea id="journal-writing" aria-label="What would you like to remember?" rows={10} maxLength={5000} value={draft}
            onChange={(event) => { setDraft(event.target.value); setNotice(null); }}
            placeholder="Something that happened, a thought that stayed, or how today felt…"
            aria-describedby="journal-save-note journal-length" />
          <div className="journal-sheet-foot">
            <p id="journal-length" className="small muted">
              {draft.length > 0 && <span className="unsaved" role="status">Unsaved writing · </span>}
              {wordCount(draft)} {wordCount(draft) === 1 ? "word" : "words"} · {draft.length.toLocaleString()} / 5,000 characters
            </p>
            <button className="btn btn-primary" disabled={saving || !draft.trim()} onClick={() => void save()}>
              {saving ? "Saving…" : "Save entry"}
            </button>
          </div>
          <p id="journal-save-note" className="small muted journal-save-note">
            Saving keeps your exact words until you delete the entry, even if temporary chat text is turned off.
            AI is optional. Saved entries keep their original wording.
          </p>
          <p className="small muted" role={draftVolatile ? "status" : undefined}>{draftVolatile ? VOLATILE_DRAFT_NOTE : TAB_DRAFT_NOTE}</p>
          {draft && <button className="btn btn-ghost" type="button" onClick={() => {
            if (window.confirm("Discard your unsaved journal writing?")) { setDraft(""); writeTabValue("journal-receipt", ""); }
          }}>Discard draft</button>}
        </section>
          )}
        </div>
      </div>
    </div>
  );
}

export default function JournalPage() {
  return <Suspense fallback={<div className="page"><p role="status">Opening your journal…</p></div>}>
    <JournalWorkspace />
  </Suspense>;
}
