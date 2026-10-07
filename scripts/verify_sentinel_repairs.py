"""Sentinel regressions on the named local scratch database after base verification.

Uses real signed adapter payloads and PostgreSQL transactions, never a provider.
Run after verify_postgres_schema.py. Does not reset or contact hosted databases.
"""

# ruff: noqa: E501
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import verify_activity_schema as activity
import verify_postgres_schema as base
from journalpulse.activity_models import ActivityFollowUpRequest, ActivityReportRequest
from journalpulse.domain import ActionCard, ConversationMode, Goal, PolicyDecision, SafetyMode
from journalpulse.signing import readiness_probe
from scratch_postgres import require_local_postgres_dsn


def recent_chat(**changes):
    now = datetime.now(UTC)
    return base.conversation(base.USER_A).model_copy(
        update={
            "created_at": now,
            "updated_at": now,
            "incarnation_id": uuid4(),
            **changes,
        }
    )


def turn(chat, *, legacy_rpc=False):
    user, assistant = base.pair(chat, "Fictional boundary turn.", 0)
    user = user.model_copy(update={"created_at": chat.created_at})
    assistant = assistant.model_copy(update={"created_at": chat.created_at + timedelta(microseconds=1)})
    call = base.commit(chat, user, assistant, 0)
    return call.replace("jp_commit_turn_v2(", "jp_commit_turn(") if legacy_rpc else call


def incarnation_check(*, legacy_rpc=False):
    original = recent_chat(retain_text=True)
    replacement = original.model_copy(
        update={
            "incarnation_id": uuid4(),
            "mode": ConversationMode.GUIDED,
            "llm_consent": False,
            "retain_text": False,
        }
    )
    return base.psql(
        base.DATABASE,
        f"""
    begin;
    {base.as_user(base.USER_A)}
    select {base.create(original)};
    select public.jp_delete_conversation('{original.id}');
    {base.expect_error(base.create(replacement), "Creation request ID was deleted; use a new request ID", "deleted logical chat UUID cannot be recreated")}
    {base.expect_error(turn(original, legacy_rpc=legacy_rpc), "Conversation not found", "late turn refused after retired UUID")}
    {base.check(f"(select count(*) from public.conversations where id='{original.id}')=0", "deleted chat remains absent")}
    {base.check(f"(select count(*) from public.conversation_messages where conversation_id='{original.id}')=0", "no deleted messages reappear")}
    rollback;
    """,
    )


def support_response_check(*, legacy_rpc=False):
    ordinary = ActionCard(
        resource_intent="read",
        card_reason="A fictional optional guide.",
        actions=[{"id": "ordinary", "title": "Fictional ordinary guide", "url": "https://example.org/guide"}],
        goal=Goal.UNDERSTAND,
        decision_preview=PolicyDecision(
            action_id="ordinary",
            propensity=None,
            eligible_for_ope=False,
            policy_name="fixture",
            policy_version="1",
            safe_action_ids=["ordinary"],
            context_snapshot={},
            explanation="fixture",
        ),
    )
    support = ordinary.model_copy(
        update={
            "actions": [{"id": "support", "title": "Support information", "url": "https://988.ca/"}],
            "decision_preview": ordinary.decision_preview.model_copy(update={"propensity": 1.0}),
        }
    )
    chat = recent_chat(activity_card=ordinary)
    updated = chat.model_copy(update={"safety_mode": SafetyMode.SUPPORT, "card": support})
    return base.psql(
        base.DATABASE,
        f"""
    begin;
    {base.as_user(base.USER_A)}
    select {base.create(chat)};
    create temporary table sentinel_return as select {turn(updated, legacy_rpc=legacy_rpc)} as receipt;
    {base.check("(select receipt->'conversation'->'activity_card'='null'::jsonb from sentinel_return)", "support response clears ordinary activity card")}
    {base.check(f"(select receipt->'conversation'=(select record from public.conversations where id='{chat.id}') from sentinel_return)", "immediate response matches trigger-normalized stored row")}
    rollback;
    """,
    )


def lease_check():
    chat = recent_chat()
    session = activity.session(chat)
    report = ActivityReportRequest(
        client_request_id=uuid4(),
        expected_revision=2,
        expected_conversation_revision=0,
        participation="not_tried",
    )
    claim = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=3, expected_conversation_revision=0
    )
    return base.psql(
        base.DATABASE,
        f"""
    begin;
    {base.as_user(base.USER_A)}
    select {base.create(chat)};
    select {activity.offer(session)};
    select {activity.command(session, "start", 0)};
    select {activity.command(session, "finish_early", 1)};
    select {activity.report(session, report)};
    create temporary table sentinel_claim as select {activity.claim(session, claim)} as receipt;
    {base.check("(select (receipt->'claimed')::boolean from sentinel_claim)", "follow-up claim acquired")}
    {base.check("(select (receipt->'session'->>'follow_up_lease_until')::timestamptz > now()+interval '120 seconds' from sentinel_claim)", "lease outlasts provider budget and host deadline")}
    rollback;
    """,
    )


def followup_identity_check():
    chat = recent_chat()
    session = activity.session(chat)
    report = ActivityReportRequest(
        client_request_id=uuid4(),
        expected_revision=2,
        expected_conversation_revision=0,
        participation="not_tried",
    )
    claim = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=3, expected_conversation_revision=0
    )
    base.psql(
        base.DATABASE,
        f"""
    {base.as_user(base.USER_A)}
    select {base.create(chat)}; select {activity.offer(session)};
    select {activity.command(session, "start", 0)};
    select {activity.command(session, "finish_early", 1)};
    select {activity.report(session, report)}; select {activity.claim(session, claim)};
    """,
    )
    stored = activity.stored(session)

    def finish(expected):
        return activity.finish(stored, chat, claim, expected_incarnation=expected)

    return base.psql(
        base.DATABASE,
        f"""
    {base.as_user(base.USER_A)}
    {base.expect_error(finish(uuid4()), "Activity changed", "follow-up cannot cross parent incarnation")}
    {base.check(f"(select count(*) from public.conversation_messages where conversation_id='{chat.id}')=0", "stale outcome stores no reply")}
    select {finish(chat.incarnation_id)};
    select {finish(chat.incarnation_id)};
    {base.check(f"(select count(*) from public.conversation_messages where conversation_id='{chat.id}')=1", "correct incarnation finishes exactly once")}
    """,
    )


def readiness_check():
    probe = readiness_probe(base.SIGNING_KEY)
    return base.psql(
        base.DATABASE,
        f"""
    {base.check(f"({base.rpc_call('jp_readiness_v4', probe)})->>'repairs'='ready'", "readiness identifies repaired contract")}
    """,
    )


def main():
    require_local_postgres_dsn(base.os.getenv("JOURNALPULSE_PG_DSN"))
    outputs = [
        incarnation_check(),
        support_response_check(),
        lease_check(),
        followup_identity_check(),
        readiness_check(),
    ]
    for output in outputs:
        for line in output.splitlines():
            if "ok:" in line:
                print(line.split("NOTICE:", 1)[-1].strip())
    print(json.dumps({"sentinel_postgres_checks": "passed", "provider_calls": 0}))


if __name__ == "__main__":
    main()
