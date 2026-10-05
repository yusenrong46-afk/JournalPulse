"""Exercise the journal extension on the disposable jp_verify database.

Run verify_postgres_schema.py first. This script never resets a database and
refuses a configured remote host; it reuses the actual signed adapter payloads.
"""

from __future__ import annotations

import os
from datetime import timedelta

import verify_postgres_schema as base
from journalpulse.journal_models import JournalEntry


def main() -> None:
    base.require_local_postgres_dsn(os.getenv("JOURNALPULSE_PG_DSN"))

    source = JournalEntry(user_id=base.USER_A, created_at=base.STARTED, text="Fictional meeting reflection.")
    other = JournalEntry(user_id=base.USER_B, created_at=base.STARTED, text="Another person's fictional entry.")
    changed = source.model_copy(update={"text": "A conflicting retry."})
    replay = source.model_copy(update={"created_at": base.STARTED + timedelta(minutes=1)})

    def save(entry: JournalEntry) -> str:
        return base.captured(lambda repo: repo.save_journal_entry(entry))

    def delete(entry: JournalEntry) -> str:
        return base.captured(lambda repo: repo.delete_journal_entry(entry.user_id, entry.id))

    linked = base.conversation(base.USER_A).model_copy(update={
        "source_entry_id": source.id, "source_entry_created_at": source.created_at,
    })
    unlinked = base.conversation(base.USER_A)
    foreign = base.conversation(base.USER_B).model_copy(update={
        "source_entry_id": source.id, "source_entry_created_at": source.created_at,
    })
    stale_source = base.conversation(base.USER_A).model_copy(update={
        "source_entry_id": source.id, "source_entry_created_at": source.created_at + timedelta(seconds=1),
    })
    user, assistant = base.pair(linked, "Fictional follow-up.", 0)
    record = base.reflection(base.USER_A)
    deletion_sql = delete(source)
    sql = f"""
    {base.as_user(base.USER_A)}
    select {save(source)};
    {base.check(f"({save(replay)}->>'created_at')::timestamptz = '{source.created_at.isoformat()}'::timestamptz", "journal replay returns the original timestamp")}
    {base.expect_error(save(changed), "Journal request ID is already in use", "changed journal replay cannot overwrite writing")}
    {base.expect_denied("insert into public.journal_entries (id,user_id,created_at,record) values (gen_random_uuid(),auth.uid(),now(),'{}'::jsonb)", "unsigned journal insertion is denied")}
    {base.expect_denied(f"delete from public.journal_entries where id = '{source.id}'", "direct deletion cannot bypass derived-content cleanup")}
    {base.as_user(base.USER_B)}
    {base.check(f"(select count(*) from public.journal_entries where id = '{source.id}') = 0", "journal source is hidden by owner RLS")}
    {base.expect_error(base.create(foreign), "Journal entry not found", "a chat cannot attach another person's source")}
    select {save(other)};
    {base.as_user(base.USER_A)}
    {base.expect_error(base.create(stale_source), "Journal entry not found", "source creation time is checked atomically")}
    select {base.create(linked)};
    select {base.create(unlinked)};
    select {base.commit(linked, user, assistant, 0)};
    select {base.accept(linked, record, 1)};
    {base.check(f"exists(select 1 from public.reflections where id = '{record.id}')", "linked action reflection was saved before source deletion")}
    {base.check(deletion_sql, "source deletion succeeds")}
    {base.check(f"not exists(select 1 from public.conversations where id = '{linked.id}')", "source deletion removes linked conversation")}
    {base.check(f"not exists(select 1 from public.conversation_messages where conversation_id = '{linked.id}')", "source deletion removes derived messages")}
    {base.check(f"not exists(select 1 from public.reflections where id = '{record.id}')", "source deletion removes the accepted derived reflection")}
    {base.check(f"exists(select 1 from public.conversations where id = '{unlinked.id}')", "source deletion preserves unrelated conversations")}
    {base.expect_error(base.commit(linked, user, assistant, 1), "Conversation not found", "a delayed turn cannot recreate deleted source context")}
    {base.expect_error(base.create(linked), "Journal entry not found", "a delayed start cannot create an orphan source chat")}
    select {save(source)};
    select public.delete_my_journalpulse_data();
    {base.check("(select count(*) from public.journal_entries) = 0", "global journal deletion includes standalone writing")}
    {base.as_user(base.USER_B)}
    {base.check(f"exists(select 1 from public.journal_entries where id = '{other.id}')", "global deletion preserves the other person's writing")}
    select {delete(other)};
    select 'Connected journal schema verification passed' as result;
    """
    output = base.psql(base.DATABASE, sql)
    for line in output.splitlines():
        if "ok:" in line:
            print(line.split("NOTICE:", 1)[-1].strip())
    if "Connected journal schema verification passed" not in output:
        raise SystemExit("Journal schema verification did not complete.")
    print("Connected journal schema verification passed")


if __name__ == "__main__":
    main()
