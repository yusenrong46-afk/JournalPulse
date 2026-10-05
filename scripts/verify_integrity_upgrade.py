"""Exercise additive audit integrity fixes on existing local scratch PostgreSQL.

Run verify_postgres_schema.py first. This script never resets any database and
refuses remote hosts. It produces signed RPC payloads using the actual adapter.
"""

from __future__ import annotations

import os

import verify_postgres_schema as base
from journalpulse.domain import ConversationRequestInputs
from journalpulse.signing import signed_payload


def main() -> None:
    base.require_local_postgres_dsn(os.getenv("JOURNALPULSE_PG_DSN"))

    chat = base.conversation(base.USER_A, retain=False)
    user, assistant = base.pair(chat, "Fictional original writing.", 0)
    user = user.model_copy(update={"request_inputs": ConversationRequestInputs(mood_score=2)})
    changed_user = user.model_copy(update={"content": "Different fictional writing."})
    changed_inputs = user.model_copy(update={"request_inputs": ConversationRequestInputs(mood_score=4)})
    rate_body = signed_payload("consume_rate_limit", base.USER_A, {
        "bucket": "generation", "max_events": 20, "window_seconds": 60,
    }, base.SIGNING_KEY)
    forged = {**rate_body, "signature": "0" * 64}
    accepted = base.conversation(base.USER_A)
    other = base.conversation(base.USER_A)
    receipt = base.reflection(base.USER_A).model_copy(update={
        "context": {"source": "conversation", "conversation_id": str(accepted.id)},
    })
    changed_receipt = receipt.model_copy(update={
        "decision": receipt.decision.model_copy(update={"action_id": "a-different-action"}),
    })
    sql = f"""
    reset role;
    delete from public.rate_limit_events where user_id = '{base.USER_A}' and bucket = 'generation';
    insert into public.rate_limit_events (user_id, bucket, created_at)
      select '{base.USER_A}', 'generation', now() - interval '30 seconds' from generate_series(1,20);
    {base.as_user(base.USER_A)}
    {base.check(
        "not (public.jp_consume_rate_limit('generation',20,60)->>'allowed')::boolean",
        "legacy production generation policy still enforces the window",
    )}
    {base.expect_error(
        "public.jp_consume_rate_limit('generation',20,1)", "Invalid legacy rate limit policy",
        "direct RPC cannot shorten and erase usage history",
    )}
    {base.expect_error(
        "public.jp_consume_rate_limit('generation',1000,60)", "Invalid legacy rate limit policy",
        "direct RPC cannot widen generation limits",
    )}
    {base.expect_error(
        base.rpc_call('jp_consume_rate_limit_v2', forged), "Untrusted write",
        "signed usage RPC rejects a forged signature",
    )}
    {base.check(
        f"not ({base.rpc_call('jp_consume_rate_limit_v2', rate_body)}->>'allowed')::boolean",
        "signed and legacy generation calls share the same usage history",
    )}
    reset role;
    {base.check(
        f"(select count(*) from public.rate_limit_events where user_id = '{base.USER_A}' "
        "and bucket = 'generation') = 20",
        "rejected policy changes never delete usage events",
    )}
    {base.as_user(base.USER_B)}
    {base.expect_error(
        base.rpc_call('jp_consume_rate_limit_v2', rate_body),
        "Record owner does not match authenticated user",
        "signed usage policy remains owner bound",
    )}
    {base.as_user(base.USER_A)}
    select {base.create(chat)};
    {base.check(
        f"({base.create(chat)}->>'id')::uuid = '{chat.id}'",
        "identical creation retry returns the same conversation",
    )}
    {base.expect_error(
        base.create(chat.model_copy(update={'retain_text': True})),
        "Conversation request ID is already in use",
        "creation retry cannot change retention choice",
    )}
    {base.expect_error(
        base.create(chat.model_copy(update={'llm_consent': False})),
        "Conversation request ID is already in use",
        "creation retry cannot change AI consent",
    )}
    {base.expect_error(
        base.create(chat.model_copy(update={'locale': 'US'})), "Conversation request ID is already in use",
        "creation retry cannot change locale",
    )}
    select {base.commit(chat, user, assistant, 0)};
    {base.check(
        f"({base.commit(chat, user, assistant, 0)}->'user_message'->>'content') "
        "= 'Fictional original writing.'",
        "identical turn replay does not regenerate",
    )}
    {base.expect_error(
        base.commit(chat, changed_user, assistant, 0), "Message request ID is already in use",
        "changed text cannot reuse a turn ID",
    )}
    {base.expect_error(
        base.commit(chat, changed_inputs, assistant, 0), "Message request ID is already in use",
        "changed reported mood cannot reuse a turn ID",
    )}
    select public.jp_close_conversation('{chat.id}');
    {base.expect_error(
        base.commit(chat, user, assistant, 0),
        "This message's text is no longer retained; its retry cannot be verified.",
        "purged turn replay is refused without retaining text hashes",
    )}
    select public.jp_delete_conversation('{chat.id}');
    select {base.create(accepted)};
    select {base.create(other)};
    select {base.accept(accepted, receipt, 0)};
    {base.check(
        f"({base.accept(accepted, receipt, 0)}->>'id')::uuid = '{receipt.id}'",
        "identical acceptance retry returns the same receipt",
    )}
    {base.expect_error(
        base.accept(other, receipt, 0), "Reflection request ID is already in use",
        "one acceptance receipt cannot be attached to another conversation",
    )}
    {base.check(
        f"(select status from public.conversations where id = '{other.id}') = 'open'",
        "conflicting acceptance leaves the second conversation open",
    )}
    {base.expect_error(
        base.accept(accepted, changed_receipt, 0), "Reflection request ID is already in use",
        "acceptance replay cannot change the selected action",
    )}
    select public.jp_delete_conversation('{accepted.id}');
    select public.jp_delete_conversation('{other.id}');
    select 'Audit integrity verification passed' as result;
    """
    output = base.psql(base.DATABASE, sql)
    for line in output.splitlines():
        if "ok:" in line:
            print(line.split("NOTICE:", 1)[-1].strip())
    if "Audit integrity verification passed" not in output:
        raise SystemExit("Audit integrity verification did not complete.")
    print("Audit integrity verification passed")


if __name__ == "__main__":
    main()
