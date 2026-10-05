"""Verify signed guided-action RPCs on local jp_verify after the main schema check.

No reset and no hosted routing: fixtures and adapter payloads are disposable. The
checks exercise actual PostgreSQL ownership, receipts, timer/report semantics,
retention, and compatibility; no live AI or Brave request is made.
"""

# SQL assertion expressions stay readable on one line, matching the schema harness.
# ruff: noqa: E501
from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

import httpx

import verify_postgres_schema as base
from journalpulse.activity_models import (
    ActivityCommandRequest,
    ActivityFollowUpRequest,
    ActivityReportRequest,
    ActivityResource,
    ActivitySelectionProvenance,
    ActivitySession,
)
from journalpulse.domain import (
    ActionCard,
    ActivityConstraintInputs,
    ActivityFollowUpDirective,
    Conversation,
    InteractionPreference,
    ModelRun,
    PolicyDecision,
)
from journalpulse.journal_models import JournalEntry
from journalpulse.signing import readiness_probe


def session(chat: Conversation) -> ActivitySession:
    return ActivitySession(
        user_id=chat.user_id,
        conversation_id=chat.id,
        source_entry_id=chat.source_entry_id,
        resource=ActivityResource(
            id="guided_meditation_2m",
            title="Two-minute quiet meditation",
            format="timer",
            kind="meditation",
            duration_seconds=120,
            provenance="builtin",
        ),
        selection=ActivitySelectionProvenance(
            selection_source="llm",
            recommended_resource_id="guided_meditation_2m",
            selected_resource_id="guided_meditation_2m",
        ),
        duration_seconds=120,
        remaining_seconds=120,
    )


def stored(activity: ActivitySession) -> ActivitySession:
    output = base.psql(
        base.DATABASE, f"select record from public.activity_sessions where id='{activity.id}';"
    )
    return ActivitySession.model_validate(
        json.loads(next(line.strip() for line in output.splitlines() if line.strip().startswith("{")))
    )


def offer(activity: ActivitySession, revision: int = 0) -> str:
    return base.captured(
        lambda repo: repo.offer_activity_session(
            activity, request_id=activity.id, expected_conversation_revision=revision, now=datetime.now(UTC)
        )
    )


def command(
    activity: ActivitySession,
    name: Literal["start", "pause", "resume", "finish_early", "expire", "stop", "decline"],
    revision: int,
    request_id=None,
    chat_revision=0,
) -> str:
    request = ActivityCommandRequest(
        client_request_id=request_id or uuid4(),
        expected_revision=revision,
        expected_conversation_revision=chat_revision,
        command=name,
    )
    return base.captured(
        lambda repo: repo.command_activity_session(
            activity.user_id, activity.id, request, now=datetime.now(UTC)
        )
    )


def report(activity: ActivitySession, request: ActivityReportRequest) -> str:
    return base.captured(
        lambda repo: repo.report_activity_session(
            activity.user_id, activity.id, request, now=datetime.now(UTC)
        )
    )


def claim(activity: ActivitySession, request: ActivityFollowUpRequest) -> str:
    return base.captured(
        lambda repo: repo.claim_activity_follow_up(
            activity.user_id, activity.id, request, now=datetime.now(UTC)
        )
    )


def finish(
    activity: ActivitySession,
    chat: Conversation,
    request: ActivityFollowUpRequest,
    directive: ActivityFollowUpDirective | None = None,
) -> str:
    capture = base.Capture()

    def handler(request_http: httpx.Request) -> httpx.Response:
        name = request_http.url.path.rsplit("/", 1)[-1]
        if name == "activity_sessions":
            return httpx.Response(200, json=[{"record": activity.storage_payload()}])
        if name == "conversations":
            return httpx.Response(
                200, json=[{"record": chat.model_dump(mode="json"), "revision": chat.revision}]
            )
        capture.bodies.append((name, json.loads(request_http.content)))
        return httpx.Response(200, json={})

    capture.handler = handler  # type: ignore[method-assign,assignment]
    try:
        base.adapter(capture).finish_activity_follow_up(
            activity.user_id,
            activity.id,
            request_id=request.client_request_id,
            expected_revision=activity.revision,
            expected_conversation_revision=chat.revision,
            expected_session_created_at=activity.created_at,
            reply="The report says it stayed the same. We can leave it here.",
            model_run=ModelRun(model="fixture", latency_ms=1, schema_valid=True),
            now=datetime.now(UTC),
            directive=directive,
        )
    except Exception:  # A captured reply is intentionally not a real session.
        pass
    name, body = capture.bodies[-1]
    return base.rpc_call(name, body)


def notices(output: str) -> int:
    checks = 0
    for line in output.splitlines():
        if "ok:" in line:
            print(line.split("NOTICE:", 1)[-1].strip())
            checks += 1
    return checks


def preference_and_message_bound_checks(card: ActionCard) -> int:
    """Use real owner privileges for card withdrawal and supported privacy deletion."""
    count = 0
    now = datetime.now(UTC)
    for wanted in (InteractionPreference.LISTEN, InteractionPreference.ACT):
        chat = base.conversation(base.USER_A).model_copy(
            update={"created_at": now, "updated_at": now, "activity_card": card, "ready_for_action": True}
        )
        activity = session(chat)
        preference = base.preference(chat, wanted, 0, uuid4())
        expected_status = "stopped" if wanted == InteractionPreference.LISTEN else "active"
        count += notices(base.psql(base.DATABASE, f"""
        {base.as_user(base.USER_A)}
        select {base.create(chat)};select {offer(activity)};select {command(activity, "start", 0)};
        do $test$ declare response jsonb; current_record jsonb; begin
          response := {preference};
          select record into current_record from public.conversations where id='{chat.id}';
          if response is distinct from current_record or response->>'activity_card' is not null
             or response->>'card' is not null then
            raise exception 'Preference response differs from committed withdrawal';
          end if;
          raise notice 'ok: {wanted} preference returns the committed card withdrawal';
        end $test$;
        {base.check(f"({preference})=(select record from public.conversations where id='{chat.id}')", f"{wanted} retry returns current state")}
        {base.check(f"(select status from public.activity_sessions where id='{activity.id}')='{expected_status}'", f"{wanted} preserves the intended active-session lifecycle")}
        """))

    bounded = base.conversation(base.USER_A).model_copy(update={"created_at": now, "updated_at": now})
    commands = [base.as_user(base.USER_A), f"select {base.create(bounded)};"]
    for index in range(20):
        user, assistant = base.pair(bounded, f"Fictional bounded turn {index}", index)
        commands.append(f"select {base.commit(bounded, user, assistant, index)};")
    commands.extend([
        base.expect_denied(
            f"delete from public.conversation_messages where conversation_id='{bounded.id}'",
            "owner cannot reset the turn bound through partial message deletion",
        ),
        base.check(
            f"(select count(*) from public.conversation_messages where conversation_id='{bounded.id}' and role='user')=20",
            "denied partial deletion preserves all twenty turn records",
        ),
        base.expect_error(offer(session(bounded), revision=20), "This chat has reached its activity limit",
                          "activity offer remains blocked at the preserved turn bound"),
        f"select {base.captured(lambda repo: repo.delete_conversation(base.USER_A, bounded.id))};",
        base.check(f"not exists(select 1 from public.conversation_messages where conversation_id='{bounded.id}')",
                   "existing full-conversation RPC still deletes all owned messages"),
    ])
    count += notices(base.psql(base.DATABASE, "\n".join(commands)))
    return count


def superseded_offer_checks(card: ActionCard) -> int:
    """A newer recommendation withdraws only an activity that has not started."""
    count = 0
    now = datetime.now(UTC)
    for label, change in (
        ("new recommendation", {"activity_card": card.model_copy(update={"offered_message_id": uuid4()})}),
        ("shorter requested duration", {"activity_constraints": ActivityConstraintInputs(time_minutes=1)}),
        ("changed goal", {"activity_goal": "move"}),
        ("new public search topic", {"activity_search_topic": "gentle walking"}),
        ("withdrawn recommendation", {"activity_card": None}),
    ):
        chat = base.conversation(base.USER_A).model_copy(
            update={"created_at": now, "updated_at": now, "activity_card": card}
        )
        activity = session(chat)
        user, assistant = base.pair(chat, "A fictional clarification.", 0)
        updated = chat.model_copy(update={"updated_at": now, **change})
        count += notices(base.psql(base.DATABASE, f"""
        {base.as_user(base.USER_A)}
        select {base.create(chat)};select {offer(activity)};
        select {base.commit(updated, user, assistant, 0)};
        {base.check(f"(select status='stopped' and revision=1 and record->>'expires_at' is null and not (record->>'check_in_issued')::boolean from public.activity_sessions where id='{activity.id}')", f"{label} atomically withdraws the unstarted offer")}
        {base.expect_error(command(activity, "start", 1, chat_revision=1), "This activity changed; refresh before trying again", f"{label} cannot start the withdrawn offer")}
        select {offer(session(updated), revision=1)};
        """))
    for state in ("offered", "active", "paused", "awaiting_report"):
        chat = base.conversation(base.USER_A).model_copy(
            update={"created_at": now, "updated_at": now, "activity_card": card}
        )
        activity = session(chat)
        setup = [base.as_user(base.USER_A), f"select {base.create(chat)};select {offer(activity)};"]
        if state != "offered":
            setup.append(f"select {command(activity, 'start', 0)};")
        if state == "paused":
            setup.append(f"select {command(activity, 'pause', 1)};")
        elif state == "awaiting_report":
            setup.append(f"select {command(activity, 'finish_early', 1)};")
        user, assistant = base.pair(chat, "A fictional next turn.", 0)
        updated = chat.model_copy(update={"updated_at": now, "summary": "A new summary."})
        if state != "offered":
            updated = updated.model_copy(update={"activity_card": None})
        setup.append(f"select {base.commit(updated, user, assistant, 0)};")
        setup.append(base.check(
            f"(select status from public.activity_sessions where id='{activity.id}')='{state}'",
            f"ordinary updates preserve {state} activity when withdrawal is not required",
        ))
        count += notices(base.psql(base.DATABASE, "\n".join(setup)))
    return count


def followup_pause_checks() -> int:
    """A completed check-in can pause a second activity without losing its own reply."""
    now = datetime.now(UTC)
    chat = base.conversation(base.USER_A).model_copy(update={"created_at": now, "updated_at": now})
    completed = session(chat)
    active = session(chat)
    report_request = ActivityReportRequest(
        client_request_id=uuid4(), expected_revision=2, expected_conversation_revision=0,
        participation="partial", note="Fictional report before another activity.",
    )
    follow = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=3, expected_conversation_revision=0
    )
    base.psql(base.DATABASE, f"""
    {base.as_user(base.USER_A)}
    select {base.create(chat)};select {offer(completed)};
    select {command(completed, "start", 0)};select {command(completed, "finish_early", 1)};
    select {report(completed, report_request)};select {claim(completed, follow)};
    select {offer(active)};select {command(active, "start", 0)};
    """)
    claimed = stored(completed)
    pause = finish(claimed, chat, follow, ActivityFollowUpDirective(move="pause"))
    return notices(base.psql(base.DATABASE, f"""
    {base.as_user(base.USER_A)}
    do $test$ declare response jsonb; current_record jsonb; begin
      response := {pause};
      select record into current_record from public.activity_sessions where id='{completed.id}';
      if response is distinct from current_record or response->>'follow_up_status' <> 'ready'
         or response->'report'->>'participation' <> 'partial' then
        raise exception 'Pause follow-up differs from its saved report and reply';
      end if;
      raise notice 'ok: pause follow-up returns its actual saved report and successful reply';
    end $test$;
    {base.check(f"(select record->>'activity_move' from public.conversations where id='{chat.id}')='pause'", "outcome reply persists the pause directive")}
    {base.check(f"(select status='stopped' and record->>'expires_at' is null and not (record->>'check_in_issued')::boolean from public.activity_sessions where id='{active.id}')", "outcome pause atomically stops a second active timer without a check-in")}
    select {pause};
    {base.check(f"(select count(*) from public.conversation_messages where conversation_id='{chat.id}' and role='assistant')=1", "pause follow-up remains idempotent after activity invalidation")}
    """))


def main() -> None:
    base.require_local_postgres_dsn(os.getenv("JOURNALPULSE_PG_DSN"))
    now = datetime.now(UTC)
    chat = base.conversation(base.USER_A).model_copy(update={"created_at": now, "updated_at": now})
    activity = session(chat)
    start_id = uuid4()
    start = command(activity, "start", 0, start_id)
    other = session(chat)
    check = base.check
    report_request = ActivityReportRequest(
        client_request_id=uuid4(),
        expected_revision=4,
        expected_conversation_revision=0,
        participation="partial",
        fit="mixed",
        state_change="same",
        note="Fictional check-in note.",
    )
    report_sql = report(activity, report_request)
    count = notices(
        base.psql(
            base.DATABASE,
            f"""
    {base.as_user(base.USER_A)}
    select {base.create(chat)};
    select {offer(activity)};
    {check(f"({offer(activity)}->>'revision')::integer=0", "offer retry is idempotent")}
    {base.expect_denied("insert into public.activity_sessions(id,user_id,conversation_id,status,revision,created_at,updated_at,record) values(gen_random_uuid(),auth.uid(),gen_random_uuid(),'offered',0,now(),now(),'{}')", "unsigned activity insertion is denied")}
    select {start};
    {check(f"({start}->>'revision')::integer=1", "duplicate start does not restart the clock")}
    {check(f"(select status from public.conversations where id='{chat.id}')='open'", "starting an in-chat activity does not close its conversation")}
    {check(f"(select revision from public.conversations where id='{chat.id}')=0", "controls do not change the ordinary chat revision")}
    {base.expect_error(offer(other), "Finish or stop the current activity before starting another", "one nonterminal activity per conversation")}
    {base.expect_error(command(activity, "expire", 1), "The timer has not finished yet", "client cannot claim premature timer expiry")}
    select {command(activity, "pause", 1)};
    {check(f"(select status from public.activity_sessions where id='{activity.id}')='paused'", "pause is persisted")}
    select {command(activity, "resume", 2)};
    select {command(activity, "finish_early", 3)};
    {check(f"(select record->>'report' from public.activity_sessions where id='{activity.id}') is null", "finish early does not invent participant feedback")}
    {check(f"(select (record->>'check_in_issued')::boolean from public.activity_sessions where id='{activity.id}')", "one deterministic check-in is issued")}
    select {report_sql};
    {check(f"({report_sql}->>'revision')::integer=5", "duplicate report stores one outcome")}
    {check(f"(select record->'report'->>'participation' from public.activity_sessions where id='{activity.id}')='partial'", "partial report remains distinct from completed")}
    {check("not exists(select 1 from public.outcomes where user_id=auth.uid())", "partial report never fabricates a legacy completed outcome")}
    {base.expect_error(report(activity, report_request.model_copy(update={"note": "A conflicting retry"})), "Activity request ID is already in use", "same request ID cannot change its report")}
    {base.as_user(base.USER_B)}
    {check(f"not exists(select 1 from public.activity_sessions where id='{activity.id}')", "activity RLS hides another owner's session")}
    {check(f"not exists(select 1 from public.activity_receipts where session_id='{activity.id}')", "receipt RLS hides another owner's requests")}
    {base.expect_error(command(activity, "stop", 5), "Record owner does not match authenticated user", "owner cannot replay a different owner's signed command")}
    """,
        )
    )
    saved = stored(activity)
    follow = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=saved.revision, expected_conversation_revision=0
    )
    claim_sql = claim(saved, follow)
    count += notices(
        base.psql(
            base.DATABASE,
            f"""
    {base.as_user(base.USER_A)}
    {check(f"({claim_sql}->>'claimed')::boolean", "one server obtains the generation lease")}
    {check(f"not ({claim_sql}->>'claimed')::boolean", "claim replay cannot obtain a second paid generation")}
    """,
        )
    )
    claimed = stored(activity)
    next_card = ActionCard(
        resource_intent="ground",
        card_reason="A user-welcomed quiet alternative.",
        actions=[{"id": "guided_meditation_2m", "title": "Quiet meditation"}],
        decision_preview=PolicyDecision(
            action_id="guided_meditation_2m",
            recommended_action_id="guided_meditation_2m",
            propensity=None,
            eligible_for_ope=False,
            policy_name="luna-guided-action",
            policy_version="fixture",
            safe_action_ids=["guided_meditation_2m"],
            context_snapshot={},
            explanation="A user-welcomed alternative.",
        ),
    )
    count += preference_and_message_bound_checks(next_card)
    count += superseded_offer_checks(next_card)
    count += followup_pause_checks()
    finish_sql = finish(claimed, chat, follow, ActivityFollowUpDirective(card=next_card))
    stop_user, stop_assistant = base.pair(chat, "Stop. Please do not ask any more questions.", 0)
    stop_user = stop_user.model_copy(update={"created_at": now})
    stop_assistant = stop_assistant.model_copy(update={"created_at": now})
    pause_reported_chat = chat.model_copy(update={"updated_at": now, "activity_move": "pause"})
    pause_reported = base.commit(pause_reported_chat, stop_user, stop_assistant, 1)
    count += notices(
        base.psql(
            base.DATABASE,
            f"""
    {base.as_user(base.USER_A)}
    select {finish_sql};
    select {finish_sql};
    {check(f"(select count(*) from public.conversation_messages where conversation_id='{chat.id}')=1", "follow-up retry appends one assistant message")}
    {check(f"(select revision from public.conversations where id='{chat.id}')=1", "follow-up atomically advances the conversation revision")}
    {check(f"(select record->>'follow_up_reply' from public.activity_sessions where id='{activity.id}') is null", "activity row does not duplicate journal-derived model text")}
    {check(f"(select record->>'card' from public.conversations where id='{chat.id}') is null", "legacy conversation card remains compatible with the old production parser")}
    {check(f"(select record->'activity_card'->'decision_preview'->>'propensity' from public.conversations where id='{chat.id}') is null", "new LLM card has no fabricated propensity")}
    {check(f"(select record->'activity_card'->>'offered_message_id' from public.conversations where id='{chat.id}')=(select record->>'follow_up_message_id' from public.activity_sessions where id='{activity.id}')", "revised card references the atomically stored assistant message")}
    {check("(select attnotnull from pg_attribute where attrelid='public.policy_decisions'::regclass and attname='propensity')", "legacy policy-decision propensity schema stays unchanged")}
    select {pause_reported};
    {check(f"(select status='completed' and record->'report'->>'participation'='partial' from public.activity_sessions where id='{activity.id}')", "a chat stop preserves the already saved participation report")}
    {check(f"(select not (record->>'check_in_issued')::boolean from public.activity_sessions where id='{activity.id}')", "a chat stop withdraws further report questions")}
    select public.jp_close_conversation('{chat.id}');
    {check(f"(select record->>'activity_card' from public.conversations where id='{chat.id}') is null", "closing invalidates the additive recommendation too")}

    {check(f"(select record->'report'->>'note' from public.activity_sessions where id='{activity.id}') is null", "closing an unretained chat clears outcome notes")}
    {check(f"not exists(select 1 from public.activity_receipts where session_id='{activity.id}')", "unretained close clears request fingerprints")}
    {check(f"not exists(select 1 from public.conversation_messages where conversation_id='{chat.id}' and record->>'content' is not null)", "canonical follow-up and stop-turn text follow existing retention")}
    """,
        )
    )
    pause_chat = base.conversation(base.USER_A).model_copy(update={"created_at": now, "updated_at": now})
    pause_activity = session(pause_chat)
    pause_user, pause_assistant = base.pair(pause_chat, "Stop. No more questions.", 0)
    pause_user = pause_user.model_copy(update={"created_at": now})
    pause_assistant = pause_assistant.model_copy(update={"created_at": now})
    paused_chat = pause_chat.model_copy(update={"updated_at": now, "activity_move": "pause"})
    pause_turn = base.commit(paused_chat, pause_user, pause_assistant, 0)
    count += notices(
        base.psql(
            base.DATABASE,
            f"""
    {base.as_user(base.USER_A)}
    select {base.create(pause_chat)};select {offer(pause_activity)};
    select {command(pause_activity, "start", 0)};
    select {pause_turn};
    {check(f"(select status='stopped' and record->>'expires_at' is null and not (record->>'check_in_issued')::boolean from public.activity_sessions where id='{pause_activity.id}')", "signed ordinary-chat stop atomically cancels the active timer without a check-in")}
    {base.expect_error(command(pause_activity, "start", 2, chat_revision=1), "This chat has paused activities", "a stopped chat cannot restart activity controls")}
    {base.expect_error(offer(session(pause_chat), revision=1), "This chat has paused activities", "a stopped chat cannot offer a new activity before a new user turn")}
    """,
        )
    )
    source = JournalEntry(user_id=base.USER_A, text="Fictional old journal", created_at=now)
    linked = base.conversation(base.USER_A).model_copy(
        update={
            "created_at": now,
            "updated_at": now,
            "source_entry_id": source.id,
            "source_entry_created_at": source.created_at,
        }
    )
    linked_activity = session(linked)
    linked_user, linked_assistant = base.pair(linked, "Fictional source discussion.", 0)
    save_source = base.captured(lambda repo: repo.save_journal_entry(source))
    delete_source = base.captured(lambda repo: repo.delete_journal_entry(source.user_id, source.id))
    probe = readiness_probe(base.SIGNING_KEY)
    readiness = base.rpc_call("jp_readiness_v3", probe)
    count += notices(
        base.psql(
            base.DATABASE,
            f"""
    {base.as_user(base.USER_A)}
    select {save_source};select {base.create(linked)};
    select {base.commit(linked, linked_user, linked_assistant, 0)};select {offer(linked_activity, revision=1)};
    select {delete_source};
    {check(f"not exists(select 1 from public.activity_sessions where id='{linked_activity.id}')", "source deletion cascades activity sessions")}
    {check(f"not exists(select 1 from public.activity_receipts where session_id='{linked_activity.id}')", "source deletion cascades activity receipts")}
    {check(f"not exists(select 1 from public.conversation_messages where conversation_id='{linked.id}')", "signed source deletion still cascades its message metadata")}
    {base.expect_error(command(linked_activity, "start", 0), "Activity not found", "deleted source cannot be revived by an old command")}
    {check(f"({readiness}->>'schema')='guided-action-1'", "new preview requires its additive readiness version")}
    {check(f"({readiness}->>'activities')='ready'", "readiness verifies activity RPCs and retention integration")}
    select public.delete_my_journalpulse_data();
    {check("not exists(select 1 from public.activity_sessions)", "account data deletion includes activity sessions")}
    {check("not exists(select 1 from public.activity_receipts)", "account data deletion includes activity receipts")}
    {check("not exists(select 1 from public.conversation_messages)", "existing account deletion still removes owned message metadata")}
    """,
        )
    )
    print(f"Guided activity PostgreSQL verification passed ({count} assertions)")


if __name__ == "__main__":
    main()
