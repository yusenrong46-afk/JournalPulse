"""Verify demo-account isolation and 24-hour erasure on local jp_verify only.

Run verify_postgres_schema.py first (it applies scripts/pg_harness.sql and every
migration). Two anonymous demo identities write fictional entries under row-level
security; neither can read or delete the other's. The scheduled erasure job then
removes only expired anonymous accounts, through public.delete_my_journalpulse_data.
No hosted database or provider is touched.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from uuid import UUID, uuid4

import verify_postgres_schema as base
from journalpulse.erasure import bind_repository
from journalpulse.journal_models import JournalEntry


def entry(owner: UUID, text: str) -> JournalEntry:
    return JournalEntry(user_id=owner, created_at=datetime.now(UTC), text=text)


def save(record: JournalEntry) -> str:
    return base.captured(lambda repo: bind_repository(repo, record.user_id, None).save_journal_entry(record))


def remove(actor: UUID, entry_id: UUID) -> str:
    """A signed delete request from ``actor`` that names ``entry_id``."""
    return base.captured(lambda repo: repo.delete_journal_entry(actor, entry_id))


def demo_checks() -> str:
    first, second, expired, email_user = uuid4(), uuid4(), uuid4(), uuid4()
    a, b = entry(first, "Demo entry (fictional): first identity."), entry(second, "Demo entry (fictional): second identity.")
    old = entry(expired, "Demo entry (fictional): an expired sandbox.")
    kept = entry(email_user, "Fictional entry from an older email account.")
    return base.psql(base.DATABASE, f"""
    begin;
    insert into auth.users(id,email,is_anonymous,created_at) values
      ('{first}',null,true,now()),('{second}',null,true,now()),
      ('{expired}',null,true,now()-interval '25 hours'),
      ('{email_user}','old-email-fixture@example.test',false,now()-interval '40 days');
    {base.as_user(first)} select {save(a)};
    {base.as_user(second)} select {save(b)};
    {base.as_user(expired)} select {save(old)};
    {base.as_user(email_user)} select {save(kept)};
    {base.as_user(first)}
    {base.check("(select count(*) from public.journal_entries)=1", 'demo identity A sees only its own entry under RLS')}
    {base.check(f"not exists(select 1 from public.journal_entries where id='{b.id}')", 'demo identity A cannot read demo identity B')}
    {base.expect_denied(f"delete from public.journal_entries where id='{b.id}'", 'demo identity A has no direct delete on journal entries')}
    {base.check(f"not {remove(first, b.id)}", 'demo identity A cannot delete demo identity B through the delete RPC')}
    {base.as_user(second)}
    {base.check(f"exists(select 1 from public.journal_entries where id='{b.id}')", 'demo identity A cannot delete demo identity B')}
    {base.check("(select count(*) from public.journal_entries)=1", 'demo identity B sees only its own entry under RLS')}
    {base.as_user(first)}
    {base.expect_denied("select private.jp_erase_expired_demo_accounts()", 'signed-in users cannot run the demo erasure job')}
    reset role; select set_config('request.jwt.claims','',false);
    {base.check("private.jp_erase_expired_demo_accounts()=1", 'erasure job removes exactly the one expired anonymous account')}
    {base.check(f"not exists(select 1 from public.journal_entries where user_id='{expired}')", 'expired demo writing is erased')}
    {base.check(f"not exists(select 1 from auth.users where id='{expired}')", 'expired demo identity is removed')}
    {base.check(f"not exists(select 1 from private.account_erasure_revisions where user_id='{expired}')", 'expired demo identity leaves no erasure marker behind')}
    {base.check(f"(select count(*) from public.journal_entries where user_id in ('{first}','{second}'))=2", 'fresh demo identities keep their data')}
    {base.check(f"exists(select 1 from public.journal_entries where user_id='{email_user}')", 'email accounts are never erased by the demo job, whatever their age')}
    {base.check("coalesce(current_setting('request.jwt.claims',true),'')=''", 'erasure job leaves no subject claim behind')}
    {base.expect_error("private.jp_erase_expired_demo_accounts(interval '5 minutes')", 'Demo accounts are kept for at least one hour', 'erasure job refuses a retention shorter than one hour')}
    rollback;
    """)


def main() -> None:
    base.require_local_postgres_dsn(os.getenv("JOURNALPULSE_PG_DSN"))
    output = demo_checks()
    for line in output.splitlines():
        if "ok:" in line:
            print(line.split("NOTICE:", 1)[-1].strip())
    print(json.dumps({"demo_postgres_checks": "passed", "provider_calls": 0}))


if __name__ == "__main__":
    main()
