"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";

import { ApiError } from "@/lib/api";
import {
  deleteJournalEntry,
  listJournalEntries,
  readJournalEntry,
  reflectJournalEntry,
  saveJournalEntry,
  validJournalEntryId,
} from "@/lib/journal";
import type { JournalEntry, JournalReflectionResult } from "@/lib/journal-types";
import { usePreferences } from "@/lib/preferences";

function dateLabel(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit",
  });
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
    <section className="card" aria-labelledby="saved-entry-heading" aria-busy={opening}>
      <h2 id="saved-entry-heading">{entry ? "Your saved entry" : "Return to an entry"}</h2>
      {invalid && <p className="note error" role="alert">This entry link is not valid. Choose a saved entry below.</p>}
      {error && <p className="note error" role="alert">{error}</p>}
      {opening ? <p role="status">Opening your entry…</p> : entry ? (
        <>
          <p className="small"><strong>Saved writing</strong> · Kept until you delete this entry.</p>
          <p className="small muted"><time dateTime={entry.created_at}>{dateLabel(entry.created_at)}</time></p>
          <p className="journal-entry-text">{entry.text}</p>
          <div className="journal-entry-actions">
            <button className="btn btn-ghost" disabled={deleting} onClick={() => void remove()}>
              {deleting ? "Deleting…" : "Delete entry"}
            </button>
          </div>
          <div className="journal-reflect">
            <h3>Reflect with Luna</h3>
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
          <div className="journal-reflect">
            <h3>Discuss your entry</h3>
            <p className="small muted">Optional: keep exploring in a conversation with Luna.</p>
            <p className="small muted">
              Chat uses your saved entry only if you choose to share it; the temporary reflection is not carried into the conversation.
              You’ll choose whether to share this entry on the next screen.
            </p>
            <Link className="btn btn-soft" href={`/talk?entry=${entry.id}`}>Discuss with Luna</Link>
          </div>
        </>
      ) : <p className="muted">Choose a saved entry below to reread it or request a reflection.</p>}
    </section>
  );
}

function JournalWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const selectedParameter = searchParams.get("entry");
  const selectedId = validJournalEntryId(selectedParameter);
  const [entries, setEntries] = useState<JournalEntry[]>([]);
  const [draft, setDraft] = useState("");
  const [listLoading, setListLoading] = useState(true);
  const [hasMore, setHasMore] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const pendingSave = useRef<{ text: string; id: string } | null>(null);
  const listGeneration = useRef(0);

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
    // Keep the UUID and exact writing across network retries. Editing a failed
    // draft creates a new intentional entry rather than rewriting its saved copy.
    if (!pendingSave.current || pendingSave.current.text !== draft) {
      pendingSave.current = { text: draft, id: crypto.randomUUID() };
    }
    setSaving(true);
    setError(null);
    setNotice(null);
    try {
      const saved = await saveJournalEntry(pendingSave.current.text, pendingSave.current.id);
      setEntries((current) => [saved, ...current.filter((item) => item.id !== saved.id)]);
      setDraft("");
      pendingSave.current = null;
      setNotice("Entry saved. You can leave it here or ask Luna to reflect.");
      // A list read captured before this mutation must not erase the new entry.
      // Refresh the canonical first page and invalidate every older pending read.
      void loadEntries();
      router.replace(`/journal?entry=${saved.id}`, { scroll: false });
    } catch (reason) {
      setError(errorMessage(reason, "Your entry couldn’t save. Your writing is still in the editor."));
    } finally {
      setSaving(false);
    }
  }

  function removed(id: string) {
    setEntries((current) => current.filter((item) => item.id !== id));
    setNotice("Entry and its linked chats deleted.");
    // The refreshed request supersedes any read that still contains deleted text.
    void loadEntries();
  }

  return (
    <div className="page page-wide journal-page">
      <header className="page-head">
        <h1>Your journal</h1>
        <p>Write and save first. Then choose a reflection or a conversation with Luna, if you want.</p>
      </header>
      {error && <p className="note error" role="alert">{error}</p>}
      {notice && <p className="note" role="status">{notice}</p>}
      <div className="home-grid journal-grid">
        <section className="card" aria-labelledby="write-heading">
          <h2 id="write-heading">Write and save</h2>
          {draft.length > 0 && <p className="small muted" role="status">Unsaved writing</p>}
          <label htmlFor="journal-writing">What would you like to remember?</label>
          <textarea id="journal-writing" rows={8} maxLength={5000} value={draft}
            onChange={(event) => { setDraft(event.target.value); setNotice(null); }} disabled={saving}
            placeholder="Something that happened, a thought that stayed, or how today felt…"
            aria-describedby="journal-save-note journal-length" />
          <p id="journal-length" className="small muted">{draft.length.toLocaleString()} / 5,000 characters</p>
          <p id="journal-save-note" className="small muted">
            Saving keeps your exact words until you delete the entry, even if temporary chat text is turned off.
            AI is optional. Saved entries keep their original wording.
          </p>
          <button className="btn btn-primary" disabled={saving || !draft.trim()} onClick={() => void save()}>
            {saving ? "Saving…" : "Save entry"}
          </button>
        </section>
        <SavedEntry key={selectedParameter ?? "empty"} entryId={selectedId}
          invalid={Boolean(selectedParameter && !selectedId)} onDeleted={removed} />
      </div>
      <section className="card" aria-labelledby="entries-heading" aria-busy={listLoading}>
        <h2 id="entries-heading">Saved entries</h2>
        {entries.length === 0 && !listLoading && <p className="muted">Your saved writing will appear here.</p>}
        <ul className="journal-entry-list">
          {entries.map((saved) => (
            <li key={saved.id}>
              <Link className="journal-entry-link" href={`/journal?entry=${saved.id}`} scroll={false}
                aria-current={saved.id === selectedId ? "true" : undefined}>
                <time className="small muted" dateTime={saved.created_at}>{dateLabel(saved.created_at)}</time>
                <span>{saved.text.length > 150 ? `${saved.text.slice(0, 150)}…` : saved.text}</span>
                <span className="small">Open entry →</span>
              </Link>
            </li>
          ))}
        </ul>
        {listLoading && <p role="status">Loading entries…</p>}
        {hasMore && <button className="btn btn-soft" disabled={listLoading}
          onClick={() => void loadEntries(entries.length)}>Load older entries</button>}
        {error && !listLoading && <button className="btn btn-ghost" onClick={() => void loadEntries()}>Reload entries</button>}
      </section>
    </div>
  );
}

export default function JournalPage() {
  return <Suspense fallback={<div className="page"><p role="status">Opening your journal…</p></div>}>
    <JournalWorkspace />
  </Suspense>;
}
