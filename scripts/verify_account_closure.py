"""Real local PostgreSQL closure guards; fictional identities only, no Auth service calls."""
# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from uuid import uuid4

import verify_postgres_schema as base
from journalpulse.journal_models import JournalEntry
from journalpulse.signing import readiness_probe, signed_payload

PENDING = "Account closure is pending or complete"


def main() -> None:
    base.require_local_postgres_dsn(os.getenv("JOURNALPULSE_PG_DSN"))
    owner, other, operation = uuid4(), uuid4(), uuid4()
    proof = hashlib.sha256(b"fictional-closure-proof").hexdigest()
    entry = JournalEntry(user_id=owner, text="Fictional closure entry.")
    foreign = JournalEntry(user_id=other, text="Other fictional account keeps its writing.")
    new_entry = entry.model_copy(update={"id": uuid4()})
    begin = base.rpc_call("jp_begin_account_closure", signed_payload("begin_account_closure", owner,
        {"request_id": str(operation), "receipt_hash": proof}, base.SIGNING_KEY))
    resolve = f"public.jp_resolve_account_closure('{operation}','{proof}')"
    finish = f"public.jp_finish_account_closure('{operation}','{proof}')"
    def save(value):
        return base.captured(lambda repo: repo.save_journal_entry(value))
    linked = base.conversation(owner).model_copy(update={"source_entry_id": entry.id,
                                                       "source_entry_created_at": entry.created_at})
    user, assistant = base.pair(linked, "Fictional linked conversation.", 0)
    reflection = base.reflection(owner)
    probe = base.rpc_call("jp_readiness_v6", readiness_probe(base.SIGNING_KEY))
    output = base.psql(base.DATABASE, f"""
      reset role;
      insert into auth.users(id,email) values('{owner}','closure-a@example.test'),('{other}','closure-b@example.test');
      {base.as_user(owner)} select {save(entry)};
      select {base.create(linked)};
      select {base.commit(linked,user,assistant,0)};
      select {base.accept(linked,reflection,1)};
      {base.as_user(other)} select {save(foreign)};
      {base.expect_error(begin, 'Record owner does not match authenticated user', 'closure cannot target a different authenticated owner')}
      {base.as_user(owner)}
      {base.expect_denied('select * from private.account_closures', 'closure recovery ledger is private')}
      {base.expect_denied('select '+resolve, 'browser cannot resolve privileged closure receipts directly')}
      {base.expect_denied('select '+finish, 'browser cannot mark closure complete')}
      select {begin};
      {base.check(f"({begin}->>'state')='pending'", 'same closure request is idempotent')}
      {base.expect_error(save(new_entry), PENDING, 'new signed writes are blocked while closure is pending')}
      {base.expect_error('public.jp_account_data_revision()', PENDING, 'pending closure blocks model-entry mutations early')}
      {base.as_user(other)}
      {base.check(f"exists(select 1 from public.journal_entries where id='{foreign.id}')", 'other account remains readable')}
      reset role; set role service_role;
      {base.check(f"({resolve}->>'user_id')='{owner}'", 'scoped receipt resolves only its original owner')}
      {base.check(f"public.jp_resolve_account_closure('{operation}','{'0'*64}') is null", 'incorrect receipt is rejected')}
      {base.expect_error(finish, 'Authentication identity still exists', 'completion cannot be reported before identity removal')}
      reset role;
      update private.account_closures set receipt_expires_at=now()-interval '1 second' where user_id='{owner}';
      set role service_role;
      {base.check(resolve+' is null', 'expired receipt is rejected')}
      reset role; {base.as_user(owner)} select {begin};
      reset role;
      {base.check(f"(select receipt_expires_at>now() from private.account_closures where user_id='{owner}')", 'fresh signed initiation renews an expired same-request receipt')}
      -- Simulate the Auth service transaction, not a live user's deletion.
      delete from auth.users where id='{owner}';
      {base.check(f"not exists(select 1 from public.journal_entries where user_id='{owner}')", 'identity cascade deletes writing')}
      {base.check(f"not exists(select 1 from public.conversations where user_id='{owner}')", 'identity cascade deletes linked conversations')}
      {base.check(f"not exists(select 1 from public.reflections where user_id='{owner}')", 'identity cascade deletes linked reflections')}
      {base.check(f"exists(select 1 from private.account_closures where user_id='{owner}')", 'content-free recovery record survives identity deletion')}
      set role service_role;
      {base.check(f"({finish}->>'state')='complete'", 'completion verifies cascaded application deletion')}
      {base.check(f"({resolve}->>'state')='complete'", 'lost completion response can be retried')}
      reset role;
      {base.check(f"({probe}->>'account_closure')='ready'", 'readiness verifies closure write guards')}
      {base.check(f"({probe}->>'schema')='account-closure-1'", 'readiness advertises closure schema')}
    """)
    for line in output.splitlines():
        if "ok:" in line:
            print(line.split("NOTICE:", 1)[-1].strip())
    races()
    print(json.dumps({"account_closure_postgres": "passed", "provider_calls": 0,
                      "live_auth_deletion_tested": False}))


def races() -> None:
    """Both orderings of an ordinary signed save and the closure barrier."""
    for close_first in (True, False):
        owner, operation = uuid4(), uuid4()
        entry = JournalEntry(user_id=owner, text="Fictional concurrent closure test.")
        proof = hashlib.sha256(str(uuid4()).encode()).hexdigest()
        begin = base.rpc_call("jp_begin_account_closure", signed_payload("begin_account_closure", owner,
            {"request_id": str(operation), "receipt_hash": proof}, base.SIGNING_KEY))
        save = base.captured(lambda repo, entry=entry: repo.save_journal_entry(entry))
        base.psql(base.DATABASE, f"insert into auth.users(id,email) values('{owner}','{owner}@example.test')")
        first, second = (begin, save) if close_first else (save, begin)
        holding = subprocess.Popen(base.psql_command(base.DATABASE), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert holding.stdin is not None
        holding.stdin.write(f"begin; {base.as_user(owner)} select {first}; select pg_sleep(1); commit;")
        holding.stdin.close()
        time.sleep(0.25)
        waiting = subprocess.run(base.psql_command(base.DATABASE), input=f"{base.as_user(owner)} select {second};",
                                 text=True, capture_output=True, check=False)
        holding.wait(timeout=20)
        if holding.returncode or (close_first and PENDING not in waiting.stderr) or (not close_first and waiting.returncode):
            raise SystemExit("Account-closure concurrency check failed")
        print('ok: closure barrier serializes with writes, closure_first='+str(close_first))


if __name__ == "__main__":
    main()
