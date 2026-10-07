"""Verify retired identities and account erasure fencing on local jp_verify only.

Run verify_postgres_schema.py first. Uses real adapter signatures, synthetic content,
transaction rollback and two disposable sessions; no hosted DB or provider calls.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from uuid import uuid4

import verify_activity_schema as activity
import verify_postgres_schema as base
from journalpulse.activity_models import ActivityFollowUpRequest, ActivityReportRequest
from journalpulse.domain import InteractionPreference
from journalpulse.erasure import bind_repository
from journalpulse.journal_models import JournalEntry
from journalpulse.signing import readiness_probe

RETIRED = "Creation request ID was deleted; use a new request ID"


def entry(owner=base.USER_A):
    return JournalEntry(user_id=owner, created_at=datetime.now(UTC), text="Fictional erased writing.")


def save(record, *, revision=None):
    return base.captured(lambda repo: bind_repository(repo, record.user_id, revision).save_journal_entry(record))


def delete(record):
    return base.captured(lambda repo: repo.delete_journal_entry(record.user_id, record.id))


def current_revision(owner=base.USER_A):
    output = base.psql(base.DATABASE, f"{base.as_user(owner)} select public.jp_account_data_revision();")
    return int(next(line.strip() for line in output.splitlines() if line.strip().isdigit()))


def retired_identity_checks(scope):
    source = entry()
    chat = base.conversation(base.USER_A).model_copy(update={
        "created_at": source.created_at, "updated_at": source.created_at, "incarnation_id": uuid4(),
        "source_entry_id": source.id if scope == "source" else None,
        "source_entry_created_at": source.created_at if scope == "source" else None,
    })
    session = activity.session(chat)
    record = base.reflection(base.USER_A)
    fresh_chat = chat.model_copy(update={"id": uuid4(), "source_entry_id": None, "source_entry_created_at": None})
    reused_session = session.model_copy(update={"conversation_id": fresh_chat.id, "source_entry_id": None})
    erase = delete(source) if scope == "source" else "public.delete_my_journalpulse_data()" if scope == "account" else f"public.jp_delete_conversation('{chat.id}')"
    report = ActivityReportRequest(client_request_id=uuid4(), expected_revision=0, expected_conversation_revision=0, participation="not_tried")
    followup = ActivityFollowUpRequest(client_request_id=uuid4(), expected_revision=0, expected_conversation_revision=0)
    return base.psql(base.DATABASE, f"""
    begin;
    {base.as_user(base.USER_A)}
    select {save(source)};
    select {base.create(chat)};
    select {activity.offer(session)};
    select {erase};
    {base.expect_error(base.create(chat.model_copy(update={'source_entry_id': None, 'source_entry_created_at': None})), RETIRED, f'{scope}: retired chat ID cannot be recreated')}
    {base.expect_error(base.accept(chat, record, 0), 'Conversation not found', f'{scope}: delayed acceptance cannot restore a deleted summary')}
    {base.expect_error(activity.offer(session), 'Conversation not found', f'{scope}: delayed offer cannot attach to a deleted chat')}
    {base.expect_error(base.preference(chat, InteractionPreference.LISTEN, 0, uuid4()), 'Conversation not found', f'{scope}: pre-read preference targets no replacement')}
    {base.expect_error(activity.command(session, 'start', 0), 'Activity not found', f'{scope}: old activity start is rejected')}
    {base.expect_error(activity.report(session, report), 'Activity not found', f'{scope}: old activity report is rejected')}
    {base.expect_error(activity.claim(session, followup), 'Activity not found', f'{scope}: old follow-up claim is rejected')}
    select {base.create(fresh_chat)};
    {base.expect_error(activity.offer(reused_session), RETIRED, f'{scope}: session ID cannot be recreated on a new chat')}
    {base.check(f"not exists(select 1 from public.reflections where id='{record.id}')", f'{scope}: no old summary was stored')}
    {base.check(f"not exists(select 1 from public.conversation_messages where conversation_id='{chat.id}')", f'{scope}: no deleted message was restored')}
    rollback;
    """)


def journal_reflection_replay_checks():
    source = entry()
    record = base.reflection(base.USER_A)
    save_reflection = base.captured(lambda repo: repo.save_reflection(record))
    # The adapter's reflection deletion is an authenticated table DELETE, not RPC.
    delete_reflection = f"delete from public.reflections where user_id='{record.user_id}' and id='{record.id}'"
    return base.psql(base.DATABASE, f"""
    begin;
    {base.as_user(base.USER_A)}
    select {save(source)};
    {base.check(f"{save(source)}={save(source)}", 'existing journal retry remains idempotent')}
    select {delete(source)};
    {base.expect_error(save(source), RETIRED, 'entry deletion rejects a lost-response save retry')}
    select {save_reflection};
    {delete_reflection};
    {base.expect_error(save_reflection, RETIRED, 'reflection deletion rejects a summary replay')}
    select public.delete_my_journalpulse_data();
    {base.expect_error(save(source), RETIRED, 'account erasure retains the deleted entry marker')}
    {base.expect_error(save_reflection, RETIRED, 'account erasure retains the deleted reflection marker')}
    {base.expect_denied('select * from private.deleted_object_ids', 'deletion markers are not client-readable')}
    {base.expect_denied('delete from private.deleted_object_ids', 'deletion markers cannot be reset by clients')}
    {base.as_user(base.USER_B)}
    {base.expect_error(save(source.model_copy(update={'user_id':base.USER_B})), RETIRED, 'another account cannot claim an erased global UUID')}
    rollback;
    """)


def account_revision_checks():
    revision = current_revision()
    source, fresh = entry(), entry()
    probe = readiness_probe(base.SIGNING_KEY)
    return base.psql(base.DATABASE, f"""
    begin;
    {base.as_user(base.USER_A)}
    select public.delete_my_journalpulse_data();
    {base.expect_error(save(source, revision=revision), 'Account data was erased', 'never-committed authenticated save is fenced by account erase')}
    {base.check(f"not exists(select 1 from public.journal_entries where id='{source.id}')", 'rejected new ID writes no text')}
    select {save(fresh, revision=revision+1)};
    {base.check(f"exists(select 1 from public.journal_entries where id='{fresh.id}')", 'fresh authenticated request after erasure can save')}
    select {save(entry())};
    {base.check(f"({base.rpc_call('jp_readiness_v4', probe)})->>'schema'='sentinel-boundaries-1'", 'legacy readiness contract remains compatible')}
    {base.check(f"({base.rpc_call('jp_readiness_v5', probe)})->>'schema'='erasure-boundaries-1'", 'new readiness identifies erasure guard schema')}
    {base.check(f"({base.rpc_call('jp_readiness_v5', probe)})->>'erasure'='ready'", 'new readiness verifies all identity triggers')}
    rollback;
    """)


def auth_identity_cascade_check():
    owner = uuid4()
    source, second = entry(owner), entry(owner)
    return base.psql(base.DATABASE, f"""
    begin;
    insert into auth.users(id,email) values('{owner}','erasure-fixture@example.test');
    {base.as_user(owner)}
    select {save(source)};
    select public.delete_my_journalpulse_data();
    select {save(second)};
    reset role;
    delete from auth.users where id='{owner}';
    {base.check(f"not exists(select 1 from private.deleted_object_ids where user_id='{owner}')", 'auth identity deletion removes prior and cascading markers')}
    {base.check(f"not exists(select 1 from private.account_erasure_revisions where user_id='{owner}')", 'auth identity deletion removes account revision')}
    rollback;
    """)


def concurrency_checks():
    """Observe holder's sleep inside its transaction before sending the contender."""
    def contend(holder_sql, contender_sql):
        name = f"jp-erasure-{uuid4()}"
        holding = subprocess.Popen(base.psql_command(base.DATABASE), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert holding.stdin is not None
        holding.stdin.write(f"set application_name='{name}'; {base.as_user(base.USER_A)} begin; select {holder_sql}; select pg_sleep(2); commit;")
        holding.stdin.close()
        try:
            for _ in range(40):
                observed = base.psql(base.DATABASE, f"select count(*) from pg_stat_activity where application_name='{name}' and wait_event='PgSleep';")
                if any(line.strip() == '1' for line in observed.splitlines()):
                    break
                time.sleep(0.05)
            else:
                raise SystemExit('Erasure concurrency holder never reached its transaction barrier')
            contender = subprocess.run(base.psql_command(base.DATABASE), input=f"{base.as_user(base.USER_A)} select {contender_sql};", text=True, capture_output=True, timeout=15)
            holding.wait(timeout=15)
            assert holding.stderr is not None
            if holding.returncode:
                raise SystemExit(holding.stderr.read())
            return contender
        finally:
            if holding.poll() is None:
                holding.kill()
                holding.wait()

    revision = current_revision()
    source = entry()
    erased = contend(save(source, revision=revision), 'public.delete_my_journalpulse_data()')
    if erased.returncode:
        raise SystemExit(erased.stderr)
    base.psql(base.DATABASE, base.check(f"not exists(select 1 from public.journal_entries where id='{source.id}')", 'account erase waits for an earlier writer then removes its content'))
    revision = current_revision()
    fresh = entry()
    refused = contend('public.delete_my_journalpulse_data()', save(fresh, revision=revision))
    if refused.returncode == 0 or 'Account data was erased' not in refused.stderr:
        raise SystemExit(f'Account barrier failed to reject waiting stale writer: {refused.stderr}')
    base.psql(base.DATABASE, base.check(f"not exists(select 1 from public.journal_entries where id='{fresh.id}')", 'writer waiting behind erase checks latest revision and saves nothing'))
    print('ok: account erase and content writes serialize correctly in both transaction orders')


def main():
    base.require_local_postgres_dsn(os.getenv('JOURNALPULSE_PG_DSN'))
    outputs = [retired_identity_checks(scope) for scope in ('chat', 'source', 'account')]
    outputs += [journal_reflection_replay_checks(), account_revision_checks(), auth_identity_cascade_check()]
    for output in outputs:
        for line in output.splitlines():
            if 'ok:' in line:
                print(line.split('NOTICE:', 1)[-1].strip())
    concurrency_checks()
    print(json.dumps({'erasure_postgres_checks': 'passed', 'provider_calls': 0}))


if __name__ == '__main__':
    main()
