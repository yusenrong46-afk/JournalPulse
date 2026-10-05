"""Apply the Supabase migrations to PostgreSQL and exercise RLS, provenance, and lifecycle rules.

Drops and recreates the database named jp_verify. Set JOURNALPULSE_PG_DSN to a
maintenance-database URI (for example the CI service). Without it, the script uses local
peer authentication as the postgres user.

Payloads are produced by the real SupabaseRepository through a capturing transport, so
the SQL is tested against exactly what the API sends.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

import httpx

from journalpulse.config import Settings
from journalpulse.domain import (
    AffectiveState,
    Conversation,
    ConversationMessage,
    InteractionPreference,
    MessageRole,
    ModelRun,
    OutcomeRecord,
    PolicyDecision,
    ReflectionCopy,
    ReflectionRecord,
    SafetyMode,
    SafetyResult,
    TargetState,
)
from journalpulse.intelligence import CONVERSATION_PROMPT_VERSION
from journalpulse.persistence import SupabaseRepository
from journalpulse.signing import readiness_probe, signed_payload
from scratch_postgres import require_local_postgres_dsn

ROOT = Path(__file__).resolve().parents[1]
DATABASE = "jp_verify"
SIGNING_KEY = "verification-only-signing-key-0123456789abcdef"
USER_A = UUID("11111111-1111-4111-8111-111111111111")
USER_B = UUID("22222222-2222-4222-8222-222222222222")
STARTED = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def dsn_for(database: str) -> str | None:
    raw = os.environ.get("JOURNALPULSE_PG_DSN")
    require_local_postgres_dsn(raw)
    if not raw:
        return None
    parts = urlsplit(raw)
    return urlunsplit(parts._replace(path=f"/{database}"))


def psql_command(database: str) -> list[str]:
    dsn = dsn_for(database)
    if dsn:
        return ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-X", "-q"]
    return ["sudo", "-u", "postgres", "psql", "-d", database, "-v", "ON_ERROR_STOP=1", "-X", "-q"]


def psql(database: str, sql: str) -> str:
    completed = subprocess.run(psql_command(database), input=sql, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
    return completed.stdout + completed.stderr


def literal(value: str) -> str:
    if "$jp$" in value:
        raise SystemExit("payload contains the SQL dollar-quote delimiter")
    return f"$jp${value}$jp$"


class Capture:
    """Records the RPC body the adapter would send, and answers with a canned response."""

    def __init__(self) -> None:
        self.bodies: list[tuple[str, dict[str, Any]]] = []
        self.reply: Any = {}

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append((request.url.path.rsplit("/", 1)[-1], json.loads(request.content or b"{}")))
        return httpx.Response(200, json=self.reply)


def adapter(capture: Capture) -> SupabaseRepository:
    settings = Settings(
        environment="test",
        database_path=Path("unused.db"),
        resource_catalog_path=Path("unused.json"),
        openrouter_api_key=None,
        openrouter_model="unused",
        openrouter_base_url="https://openrouter.ai/api/v1",
        openrouter_zdr=True,
        openrouter_timeout_seconds=1,
        supabase_url="https://verify.invalid",
        supabase_anon_key="anon",
        raw_text_retention_default=False,
        write_signing_key=SIGNING_KEY,
    )
    return SupabaseRepository(
        settings, "unused", client=httpx.Client(transport=httpx.MockTransport(capture.handler))
    )


def rpc_call(name: str, body: dict[str, Any]) -> str:
    """SQL that calls one RPC with the adapter's named arguments."""
    args = ", ".join(
        f"{key} => {literal(value) if isinstance(value, str) else literal(json.dumps(value)) + '::jsonb'}"
        for key, value in body.items()
    )
    return f"public.{name}({args})"


def captured(call: Any) -> str:
    capture = Capture()
    capture.reply = {"conversation": {}, "user_message": {}, "assistant_message": {}}
    try:
        call(adapter(capture))
    except Exception:  # noqa: BLE001 - the canned reply need not parse; only the body matters
        pass
    name, body = capture.bodies[-1]
    return rpc_call(name, body)


def rate(bucket: str = "generation", limit: int = 3, window: int = 60) -> str:
    return captured(lambda repo: repo.consume_rate_limit(
        USER_A, bucket, limit=limit, window_seconds=window, now=datetime.now(UTC),
    ))


def conversation(owner: UUID, *, retain: bool = False) -> Conversation:
    return Conversation(
        id=uuid4(),
        user_id=owner,
        created_at=STARTED,
        updated_at=STARTED,
        llm_consent=True,
        retain_text=retain,
        locale="CA",
        prompt_version=CONVERSATION_PROMPT_VERSION,
    )


def pair(chat: Conversation, text: str, offset: int) -> tuple[ConversationMessage, ConversationMessage]:
    at = STARTED + timedelta(minutes=offset)
    user = ConversationMessage(
        conversation_id=chat.id,
        client_message_id=uuid4(),
        role=MessageRole.USER,
        content=text,
        created_at=at,
        safety_mode=SafetyMode.NORMAL,
    )
    assistant = ConversationMessage(
        conversation_id=chat.id,
        role=MessageRole.ASSISTANT,
        content=f"reply to {text}",
        created_at=at + timedelta(seconds=1),
        safety_mode=SafetyMode.NORMAL,
        model_run=ModelRun(model="openai/gpt-6-luna", latency_ms=20, schema_valid=True),
    )
    return user, assistant


def reflection(owner: UUID, *, reflection_id: UUID | None = None) -> ReflectionRecord:
    return ReflectionRecord(
        id=reflection_id or uuid4(),
        user_id=owner,
        created_at=STARTED,
        state=AffectiveState(valence=-0.2, arousal=0.6, agency=0.4, emotion_tags=["tired"]),
        target=TargetState(goal="settle"),
        reflection=ReflectionCopy(
            summary="A summary.", interpretation="Calm.", reflection_question="What changed?"
        ),
        safety=SafetyResult(mode=SafetyMode.NORMAL, locale="CA", exploration_allowed=True),
        decision=PolicyDecision(
            action_id="site_nhs_breathing",
            propensity=1,
            policy_name="fixed-baseline",
            policy_version="1.0.0",
            safe_action_ids=["site_nhs_breathing"],
            context_snapshot={},
            explanation="Deterministic baseline.",
        ),
        model_run=ModelRun(model="openai/gpt-6-luna", latency_ms=12, schema_valid=True),
    )


def create(chat: Conversation) -> str:
    return captured(lambda repo: repo.create_conversation(chat))


def commit(
    chat: Conversation, user: ConversationMessage, assistant: ConversationMessage, revision: int
) -> str:
    return captured(lambda repo: repo.commit_turn(chat, user, assistant, expected_revision=revision))


def accept(chat: Conversation, record: ReflectionRecord, revision: int) -> str:
    return captured(
        lambda repo: repo.accept_conversation(chat.user_id, chat.id, record, expected_revision=revision)
    )


def preference(chat: Conversation, value: InteractionPreference, revision: int, request: UUID) -> str:
    return captured(
        lambda repo: repo.change_preference(
            chat.user_id, chat.id, request_id=request, preference=value,
            expected_revision=revision, now=STARTED,
        )
    )


def preference_checks() -> None:
    """Use signed adapter payloads to verify the choice through real RLS and row locks."""
    chat = conversation(USER_A)
    legacy = conversation(USER_A)
    legacy_accept = signed_payload("accept_conversation", USER_A, {
        "conversation_id": str(legacy.id), "expected_revision": 0,
        "reflection": reflection(USER_A).model_dump(mode="json"),
    }, SIGNING_KEY)
    first_id = uuid4()
    listen = preference(chat, InteractionPreference.LISTEN, 0, first_id)
    act = preference(chat, InteractionPreference.ACT, 1, uuid4())
    user, assistant = pair(chat, "a delayed turn", 0)
    offered = chat.model_copy(update={"ready_for_action": True})
    next_user, next_assistant = pair(chat, "later turn", 1)
    record = reflection(USER_A)
    missing_revision = signed_payload("accept_conversation", USER_A, {
        "conversation_id": str(chat.id), "expected_revision": 6,
        "reflection": record.model_dump(mode="json"),
    }, SIGNING_KEY)
    output = psql(DATABASE, f"""
    {as_user(USER_A)}
    select {create(legacy)};
    reset role;
    update public.conversations set record = record - 'interaction_preference' where id = '{legacy.id}';
    {as_user(USER_A)}
    select {rpc_call("jp_accept_conversation", legacy_accept)};
    {check(f"(select status from public.conversations where id = '{legacy.id}') = 'closed'", "legacy JSON without preference keeps its original acceptance contract")}
    {as_user(USER_A)}
    select {create(chat)};
    select {listen};
    {check(f"(select record->>'interaction_preference' from public.conversations where id = '{chat.id}') = 'listen'", "preference persists in PostgreSQL")}
    select {act};
    {check(f"({listen}->>'interaction_preference') = 'act'", "replayed Listen returns current Act without changing it")}
    {expect_error(preference(chat, InteractionPreference.ACT, 0, first_id), "Preference request ID is already in use", "conflicting preference ID is refused")}
    {expect_error(commit(chat, user, assistant, 0), "Conversation changed", "a delayed turn cannot restore the withdrawn card")}
    {check(f"(select count(*) from public.conversation_messages where conversation_id = '{chat.id}') = 0", "rejected turn inserts no messages")}
    select {commit(chat, user, assistant, 2)};
    {check(f"(select record->>'interaction_preference' from public.conversations where id = '{chat.id}') = 'act'", "older server turn cannot overwrite preference")}
    select {preference(chat, InteractionPreference.LISTEN, 3, uuid4())};
    select {commit(offered, next_user, next_assistant, 4)};
    {check(f"(select record->>'ready_for_action' from public.conversations where id = '{chat.id}') = 'false'", "database suppresses a model offer while listening")}
    {expect_error(accept(chat, record, 5), "Conversation changed", "listening cannot accept an ordinary card")}
    select {preference(chat, InteractionPreference.ACT, 5, uuid4())};
    {expect_error(rpc_call("jp_accept_conversation", missing_revision), "Conversation changed", "explicit choice requires client card revision")}
    {as_user(USER_B)}
    {expect_error(preference(chat.model_copy(update={"user_id": USER_B}), InteractionPreference.LISTEN, 6, uuid4()), "Conversation not found", "other owner cannot change preference")}
    {check(f"(select count(*) from public.conversation_preference_requests where conversation_id = '{chat.id}') = 0", "RLS hides another person's command receipts")}
    {expect_denied("insert into public.conversation_preference_requests default values", "direct preference receipt writes are forbidden")}
    {as_user(USER_A)}
    select {accept(chat, record, 6)};
    {check(f"({listen}->>'status') = 'closed'", "duplicate command after close returns closed state")}
    {expect_error(preference(chat, InteractionPreference.LISTEN, 7, uuid4()), "Conversation is closed", "new preference cannot reopen a closed chat")}
    select public.jp_delete_conversation('{chat.id}');
    {check(f"(select count(*) from public.conversation_preference_requests where conversation_id = '{chat.id}') = 0", "conversation deletion cascades to receipts")}
    """)
    for line in output.splitlines():
        if "ok:" in line:
            print(line.split("NOTICE:", 1)[-1].strip())


def as_user(user: UUID | None) -> str:
    claims = json.dumps({"sub": str(user), "role": "authenticated"}) if user else ""
    return f"set role authenticated; select set_config('request.jwt.claims', {literal(claims)}, false);"


def expect_error(sql: str, message: str, label: str) -> str:
    label = label.replace("'", "''")
    return f"""
    do $check$
    begin
      perform {sql};
      raise exception 'FAILED (no error): {label}';
    exception
      when others then
        if sqlerrm like 'FAILED%' then raise; end if;
        if sqlerrm <> {literal(message)} then
          raise exception 'FAILED: {label}: expected "%", got "%"', {literal(message)}, sqlerrm;
        end if;
        raise notice 'ok: {label}';
    end
    $check$;
    """


def expect_denied(sql: str, label: str) -> str:
    label = label.replace("'", "''")
    return f"""
    do $check$
    begin
      execute {literal(sql)};
      raise exception 'FAILED (allowed): {label}';
    exception
      when insufficient_privilege or undefined_function then
        raise notice 'ok: {label}';
    end
    $check$;
    """


def check(condition: str, label: str) -> str:
    return f"select public._expect(({condition}), {literal(label)});"


def text_count(chat: Conversation) -> str:
    return (
        "(select count(*) from public.conversation_messages where conversation_id = "
        f"'{chat.id}' and jsonb_typeof(record->'content') = 'string')"
    )


def behavior_sql() -> str:
    lifecycle = conversation(USER_A)
    first_user, first_assistant = pair(lifecycle, "first", 0)
    second_user, second_assistant = pair(lifecycle, "second", 1)
    stale_user, stale_assistant = pair(lifecycle, "stale", 2)
    late_user, late_assistant = pair(lifecycle, "late", 3)

    accepted = conversation(USER_A)
    accepted_user, accepted_assistant = pair(accepted, "accept me", 0)
    accepted_record = reflection(USER_A)
    second_accept = reflection(USER_A)

    stale_accept_chat = conversation(USER_A)
    stale_accept_user, stale_accept_assistant = pair(stale_accept_chat, "moved on", 0)

    kept = conversation(USER_A, retain=True)
    kept_user, kept_assistant = pair(kept, "keep this", 0)

    deleted = conversation(USER_A)
    deleted_user, deleted_assistant = pair(deleted, "gone", 0)

    idle = conversation(USER_A)
    idle_user, idle_assistant = pair(idle, "idle text", 0)
    idle_kept = conversation(USER_A, retain=True)
    idle_kept_user, idle_kept_assistant = pair(idle_kept, "idle kept", 0)

    b_chat = conversation(USER_B)
    b_user, b_assistant = pair(b_chat, "b private", 0)
    forged_owner = lifecycle.model_copy(update={"user_id": USER_B})

    create_body = json.loads(json.dumps({"conversation": lifecycle.model_dump(mode="json")}))
    forged = {
        "payload": signed_payload("create_conversation", USER_A, create_body, SIGNING_KEY)["payload"],
        "signature": "0" * 64,
    }
    wrong_purpose = signed_payload("commit_turn", USER_A, create_body, SIGNING_KEY)
    wrong_key = signed_payload("create_conversation", USER_A, create_body, "x" * 40)
    outcome = OutcomeRecord(
        user_id=USER_A, decision_id=accepted_record.decision.decision_id, completed=True, helpfulness=4
    )
    duplicate_outcome = outcome.model_copy(update={"id": uuid4()})
    probe = readiness_probe(SIGNING_KEY)
    bad_probe = readiness_probe("y" * 40)

    return f"""
    insert into auth.users (id, email) values ('{USER_A}', 'a@example.test'), ('{USER_B}', 'b@example.test');
    insert into private.server_secrets (name, value) values ('write_signing_key', {literal(SIGNING_KEY)});

    create or replace function public._expect(condition boolean, label text)
    returns void language plpgsql as $expect$
    begin
      if condition then raise notice 'ok: %', label; else raise exception 'FAILED: %', label; end if;
    end
    $expect$;
    grant execute on function public._expect(boolean, text) to anon, authenticated;

    {as_user(USER_A)}

    -- Trusted provenance: no direct writes, only signed functions.
    {
        expect_denied(
            f"insert into public.conversations (id, user_id, created_at, updated_at, status, record) "
            f"values ('{uuid4()}', '{USER_A}', now(), now(), 'open', '{{}}'::jsonb)",
            "signed-in user cannot insert a conversation directly",
        )
    }
    {
        expect_denied(
            f"insert into public.policy_decisions (id, user_id, policy_name, policy_version, action_id, "
            f"propensity, available_actions, context_snapshot) values ('{uuid4()}', '{USER_A}', 'forged', "
            f"'1', 'x', 1, '[]', '{{}}')",
            "signed-in user cannot forge a policy decision",
        )
    }
    {
        expect_denied(
            f"insert into public.model_runs (user_id, model, provider, latency_ms, schema_valid) "
            f"values ('{USER_A}', 'forged', 'x', 0, true)",
            "signed-in user cannot forge a model run",
        )
    }
    {
        expect_denied(
            "select public.save_reflection_bundle('{}'::jsonb)", "unsigned reflection function is gone"
        )
    }
    {
        expect_denied(
            "select public.jp_purge_expired_conversations()", "signed-in user cannot run the global purge"
        )
    }
    {
        expect_error(
            rpc_call("jp_create_conversation", forged), "Untrusted write", "forged signature is rejected"
        )
    }
    {
        expect_error(
            rpc_call("jp_create_conversation", wrong_key),
            "Untrusted write",
            "signature from another key is rejected",
        )
    }
    {
        expect_error(
            rpc_call("jp_create_conversation", wrong_purpose),
            "Untrusted write",
            "a payload signed for another purpose is rejected",
        )
    }
    {
        expect_error(
            create(forged_owner),
            "Record owner does not match authenticated user",
            "cannot create a conversation for someone else",
        )
    }

    -- Lifecycle: revision-guarded commits.
    select {create(lifecycle)};
    select {create(lifecycle)};
    {
        check(
            f"(select count(*) from public.conversations where id = '{lifecycle.id}') = 1",
            "create is idempotent",
        )
    }
    {
        check(
            f"(select revision from public.conversations where id = '{lifecycle.id}') = 0",
            "new chat is revision 0",
        )
    }
    {
        expect_denied(
            f"update public.conversations set status = 'open' where id = '{lifecycle.id}'",
            "signed-in user cannot update a conversation directly",
        )
    }
    select {commit(lifecycle, first_user, first_assistant, 0)};
    select {commit(lifecycle, first_user, first_assistant, 0)};
    {
        check(
            f"(select count(*) from public.conversation_messages where conversation_id = '{lifecycle.id}') = 2",
            "retried turn is stored once",
        )
    }
    {
        expect_error(
            commit(lifecycle, stale_user, stale_assistant, 0),
            "Conversation changed",
            "turn computed on an old revision is rejected",
        )
    }
    select {commit(lifecycle, second_user, second_assistant, 1)};
    {
        check(
            f"(select revision from public.conversations where id = '{lifecycle.id}') = 2",
            "each commit bumps revision",
        )
    }
    select public.jp_close_conversation('{lifecycle.id}');
    {
        check(
            f"(select status from public.conversations where id = '{lifecycle.id}') = 'closed'",
            "close writes status",
        )
    }
    {check(f"{text_count(lifecycle)} = 0", "close clears message text")}
    {
        expect_error(
            commit(lifecycle, late_user, late_assistant, 2),
            "Conversation is closed",
            "delayed reply after close is rejected",
        )
    }
    {check(f"{text_count(lifecycle)} = 0", "delayed reply cannot restore cleared text")}
    {
        check(
            f"(select status from public.conversations where id = '{lifecycle.id}') = 'closed'",
            "delayed reply cannot reopen the chat",
        )
    }

    -- Accept: one atomic transition, exactly once.
    select {create(accepted)};
    select {commit(accepted, accepted_user, accepted_assistant, 0)};
    select {accept(accepted, accepted_record, 1)};
    select {accept(accepted, accepted_record, 1)};
    {
        check(
            f"(select count(*) from public.reflections where conversation_id = '{accepted.id}') = 1",
            "accept retry stores one reflection",
        )
    }
    {
        check(
            f"(select reflection_id from public.conversations where id = '{accepted.id}') = '{accepted_record.id}'",
            "conversation links the reflection",
        )
    }
    {
        check(
            f"(select status from public.conversations where id = '{accepted.id}') = 'closed'",
            "accept closes the chat",
        )
    }
    {check(f"{text_count(accepted)} = 0", "accept purges text when not retained")}
    {
        check(
            f"(select count(*) from public.policy_decisions where reflection_id = '{accepted_record.id}') = 1",
            "accept writes the decision row",
        )
    }
    {
        check(
            f"(select count(*) from public.model_runs where reflection_id = '{accepted_record.id}') = 1",
            "accept writes the model run",
        )
    }
    {
        expect_error(
            accept(accepted, second_accept, 1),
            "Conversation already accepted",
            "a different acceptance is refused",
        )
    }
    {
        check(
            f"(select count(*) from public.reflections where id = '{second_accept.id}') = 0",
            "refused acceptance writes nothing",
        )
    }
    select {create(stale_accept_chat)};
    select {commit(stale_accept_chat, stale_accept_user, stale_accept_assistant, 0)};
    {
        expect_error(
            accept(stale_accept_chat, reflection(USER_A), 0),
            "Conversation changed",
            "accept against an old card is rejected",
        )
    }
    {
        check(
            f"(select status from public.conversations where id = '{stale_accept_chat.id}') = 'open'",
            "rejected accept leaves the chat open",
        )
    }
    select {create(kept)};
    select {commit(kept, kept_user, kept_assistant, 0)};
    select {accept(kept, reflection(USER_A), 1)};
    {check(f"{text_count(kept)} = 2", "accept keeps text when the person chose to keep it")}

    -- Delete while a reply is in flight.
    select {create(deleted)};
    select public.jp_delete_conversation('{deleted.id}');
    {
        expect_error(
            commit(deleted, deleted_user, deleted_assistant, 0),
            "Conversation not found",
            "reply after delete cannot recreate the chat",
        )
    }
    {
        check(
            f"(select count(*) from public.conversation_messages where conversation_id = '{deleted.id}') = 0",
            "reply after delete stores no messages",
        )
    }

    -- Outcomes are the person's own report.
    select public.save_outcome_record({literal(outcome.model_dump_json())}::jsonb);
    {
        expect_error(
            f"public.save_outcome_record({literal(duplicate_outcome.model_dump_json())}::jsonb)",
            "An outcome already exists for this decision",
            "one outcome per decision",
        )
    }

    -- Shared rate limit.
    select {rate()};
    select {rate()};
    {
        check(
            f"({rate()}->>'allowed')::boolean",
            "third event is allowed",
        )
    }
    {
        check(
            f"not ({rate()}->>'allowed')::boolean",
            "fourth event is refused",
        )
    }
    {
        check(
            f"({rate()}->>'retry_after')::int between 1 and 60",
            "refusal carries retry_after within the window",
        )
    }
    {check(f"({rate('other')}->>'allowed')::boolean", "buckets are independent")}
    {expect_denied("select * from public.rate_limit_events", "usage counters are not readable")}

    -- Isolation.
    {as_user(USER_B)}
    {check("(select count(*) from public.conversations) = 0", "B cannot see A conversations")}
    {check("(select count(*) from public.reflections) = 0", "B cannot see A reflections")}
    {
        expect_error(
            commit(lifecycle.model_copy(update={"user_id": USER_B}), late_user, late_assistant, 2),
            "Conversation not found",
            "B cannot attach a turn to A's chat",
        )
    }
    {
        expect_error(
            commit(lifecycle, late_user, late_assistant, 2),
            "Record owner does not match authenticated user",
            "B cannot replay a payload signed for A",
        )
    }
    select {create(b_chat)};
    select {commit(b_chat, b_user, b_assistant, 0)};
    {
        expect_error(
            f"public.jp_close_conversation('{accepted.id}')",
            "Conversation not found",
            "B cannot close A's chat",
        )
    }
    {as_user(USER_A)}
    {
        check(
            f"(select revision from public.conversations where id = '{accepted.id}') = 2",
            "B cannot close A chat",
        )
    }

    -- Retention: the scheduled purge closes idle chats and clears text.
    select {create(idle)};
    select {commit(idle, idle_user, idle_assistant, 0)};
    select {create(idle_kept)};
    select {commit(idle_kept, idle_kept_user, idle_kept_assistant, 0)};
    reset role;
    update public.conversations set updated_at = now() - interval '25 hours'
    where id in ('{idle.id}', '{idle_kept.id}');
    select public.jp_purge_expired_conversations();
    {
        check(
            f"(select status from public.conversations where id = '{idle.id}') = 'closed'",
            "purge closes idle chats",
        )
    }
    {check(f"{text_count(idle)} = 0", "purge clears idle chat text")}
    {check(f"{text_count(idle_kept)} = 2", "purge keeps text the person chose to keep")}
    update public.conversation_messages set record = jsonb_set(record, '{{content}}', '"leaked"')
    where conversation_id = '{accepted.id}';
    select public.jp_purge_expired_conversations();
    {check(f"{text_count(accepted)} = 0", "purge repairs text left on a closed chat")}

    -- Readiness.
    set role anon;
    {
        check(
            f"(public.jp_readiness({literal(probe['probe'])}, {literal(probe['signature'])})->>'schema')"
            " = 'phase-a-1'",
            "the existing API retains its readiness schema contract",
        )
    }
    {
        check(
            f"(public.jp_readiness({literal(probe['probe'])}, {literal(probe['signature'])})->>'signing') = 'valid'",
            "the existing API can still verify its shared signing key",
        )
    }
    {
        check(
            f"(public.jp_readiness_v2({literal(bad_probe['probe'])}, {literal(bad_probe['signature'])})->>'signing')"
            " = 'invalid'",
            "readiness detects a mismatched key",
        )
    }
    {
        check(
            f"(public.jp_readiness_v2({literal(probe['probe'])}, {literal(probe['signature'])})->>'schema')"
            " = 'phase-a-2'",
            "readiness reports the schema version",
        )
    }
    {
        check(
            f"(public.jp_readiness_v2({literal(probe['probe'])}, {literal(probe['signature'])})->>'signing') = 'valid'",
            "the preview API verifies the same shared signing key",
        )
    }
    {expect_denied("select count(*) from public.conversations", "anonymous callers read nothing")}

    -- Deletion covers every journal table and leaves others alone.
    {as_user(USER_A)}
    select public.delete_my_journalpulse_data();
    reset role;
    {
        check(
            f'''(select sum(n) from (
        select count(*) n from public.reflections where user_id = '{USER_A}'
        union all select count(*) from public.outcomes where user_id = '{USER_A}'
        union all select count(*) from public.policy_decisions where user_id = '{USER_A}'
        union all select count(*) from public.model_runs where user_id = '{USER_A}'
        union all select count(*) from public.safety_events where user_id = '{USER_A}'
        union all select count(*) from public.affective_observations where user_id = '{USER_A}'
        union all select count(*) from public.conversations where user_id = '{USER_A}'
        union all select count(*) from public.conversation_messages where user_id = '{USER_A}'
      ) counts) = 0''',
            "A deletion removes every A journal row",
        )
    }
    {check(f"(select count(*) from public.rate_limit_events where user_id = '{USER_A}') > 0", "deleting a journal does not reset the generation limit")}
    update public.rate_limit_events set created_at = now() - interval '2 days' where user_id = '{USER_A}';
    select public.jp_purge_expired_conversations();
    {check(f"(select count(*) from public.rate_limit_events where user_id = '{USER_A}') = 0", "the scheduled job expires old usage counters")}
    {
        check(
            f"(select count(*) from public.conversations where user_id = '{USER_B}') = 1",
            "A deletion leaves B alone",
        )
    }
    select 'PostgreSQL schema verification passed' as result;
    """


def race_sql(user: UUID, first: str, hold_seconds: float) -> str:
    return f"{as_user(user)} begin; select {first}; select pg_sleep({hold_seconds}); commit;"


def retention_checks() -> None:
    """Real PostgreSQL functions, with disposable cron metadata and a controlled clock.

    These cases do not claim the pg_cron worker actually fired. Metadata stand-ins
    are rolled back, including when the extension exists on the test server.
    """
    at = "2026-10-04 12:00:00+00"

    def observed(key: str, moment: str = at) -> str:
        return f"(private.retention_diagnostics('{moment}'::timestamptz)->>'{key}')"

    output = psql(DATABASE, f"""
    begin;
    reset role;
    create schema if not exists cron;
    create table if not exists cron.job (
      jobid bigint primary key, jobname text, schedule text, command text, active boolean
    );
    create table if not exists cron.job_run_details (
      runid bigint primary key, jobid bigint, status text, start_time timestamptz, end_time timestamptz
    );
    delete from cron.job where jobname = 'journalpulse-retention';
    update private.retention_state set monitoring_started_at = '{at}', last_success_at = null;
    {check(f"{observed('retention_state')} = 'missing'", "retention distinguishes a missing job")}
    insert into cron.job(jobid, jobname, schedule, command, active)
      values(99001, 'journalpulse-retention', '*/15 * * * *',
             'select public.jp_purge_expired_conversations()', true);
    {check(f"{observed('retention_state')} = 'never_succeeded'", "a schedule alone is never success")}
    {check(f"{observed('retention_overdue', '2026-10-04 12:30:00+00')} = 'false'", "initial grace includes exactly 30 minutes")}
    {check(f"{observed('retention_overdue', '2026-10-04 12:30:01+00')} = 'true'", "initial grace expires after 30 minutes")}
    select public.jp_purge_expired_conversations();
    {check("(select last_success_at from private.retention_state) is not null", "successful scheduled entry point records completion")}
    {check(f"{observed('retention_state')} = 'never_succeeded'", "manual entry point cannot masquerade as a successful scheduler")}
    update private.retention_state set last_success_at = '{at}';
    {as_user(USER_A)}
    select public.jp_close_my_stale_conversations();
    reset role;
    {check(f"(select last_success_at from private.retention_state) = '{at}'::timestamptz", "request-time cleanup cannot repair scheduler evidence")}
    insert into cron.job_run_details(runid, jobid, status, start_time, end_time)
      values (99000, 99001, 'succeeded', '{at}', '{at}');
    {check(f"{observed('retention_state', '2026-10-04 12:30:00+00')} = 'healthy'", "success is fresh at the 30-minute boundary")}
    {check(f"{observed('retention_state', '2026-10-04 12:30:01+00')} = 'overdue'", "stale success is overdue after 30 minutes")}
    insert into cron.job_run_details(runid, jobid, status, start_time, end_time)
      values (99001, 99001, 'failed', '2026-10-04 12:10:00+00', '2026-10-04 12:11:00+00');
    {check(f"{observed('retention_state', '2026-10-04 12:12:00+00')} = 'failed'", "failure after success is observable")}
    update private.retention_state set last_success_at = '2026-10-04 12:15:00+00';
    {check(f"{observed('retention_state', '2026-10-04 12:16:00+00')} = 'failed'", "manual completion cannot hide a cron failure")}
    insert into cron.job_run_details(runid, jobid, status, start_time, end_time)
      values (99002, 99001, 'succeeded', '2026-10-04 12:14:00+00', '2026-10-04 12:15:00+00');
    {check(f"{observed('retention_state', '2026-10-04 12:16:00+00')} = 'healthy'", "later completion recovers from failure")}
    alter table cron.job_run_details rename to job_run_details_unavailable;
    {check(f"{observed('retention_state')} = 'unknown'", "unavailable cron history is unknown")}
    alter table cron.job_run_details_unavailable rename to job_run_details;
    alter table cron.job_run_details rename column status to unavailable_status;
    {check(f"{observed('retention_state')} = 'unknown'", "unreadable cron metadata is unknown")}
    alter table cron.job_run_details rename column unavailable_status to status;
    update cron.job set schedule = '* * * * *' where jobid = 99001;
    {check(f"{observed('retention_state')} = 'unknown'", "unexpected schedule is not called healthy")}
    update cron.job set active = false where jobid = 99001;
    {check(f"{observed('retention_state')} = 'missing'", "inactive cron job is missing")}
    create or replace function private.purge_conversations(max_idle interval, only_owner uuid)
    returns jsonb language plpgsql security definer set search_path = private, public as $fail$
    begin raise exception 'forced cleanup failure'; end;
    $fail$;
    {expect_error("public.jp_purge_expired_conversations()", "forced cleanup failure", "failed cleanup propagates rather than claiming success")}
    {check("(select last_success_at from private.retention_state) = '2026-10-04 12:15:00+00'::timestamptz", "failed cleanup leaves previous success unchanged")}
    {as_user(USER_A)}
    {expect_denied("select public.jp_purge_expired_conversations()", "authenticated user cannot mark scheduler success")}
    {expect_denied("select * from private.retention_state", "authenticated user cannot read or write private monitoring data")}
    reset role; set role anon;
    {check("public.jp_readiness_v2('readiness:test', 'invalid')->>'retention_state' = 'missing'", "anonymous readiness exposes bounded operational status")}
    {expect_denied("select private.retention_diagnostics()", "public callers cannot choose the diagnostic clock")}
    rollback;
    """)
    for line in output.splitlines():
        if "ok:" in line:
            print(line.split("NOTICE:", 1)[-1].strip())
    print("note: retention clock/history checks use cron metadata stand-ins; no worker execution claimed")


def concurrency_checks() -> None:
    """Two real sessions contend for the same row; the loser must fail, not overwrite."""
    close_chat = conversation(USER_A)
    user, assistant = pair(close_chat, "racing", 0)
    accept_chat = conversation(USER_A)
    accept_user, accept_assistant = pair(accept_chat, "racing accept", 0)
    first_record = reflection(USER_A)
    second_record = reflection(USER_A)
    psql(
        DATABASE,
        f"""
        {as_user(USER_A)}
        select {create(close_chat)};
        select {create(accept_chat)};
        select {commit(accept_chat, accept_user, accept_assistant, 0)};
        """,
    )

    def contend(holder: str, contender: str) -> tuple[str, str]:
        holding = subprocess.Popen(
            psql_command(DATABASE),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert holding.stdin is not None
        holding.stdin.write(race_sql(USER_A, holder, 1.5))
        holding.stdin.close()
        time.sleep(0.5)
        second = subprocess.run(
            psql_command(DATABASE),
            input=f"{as_user(USER_A)} select {contender};",
            text=True,
            capture_output=True,
            check=False,
        )
        holding.wait(timeout=30)
        assert holding.stdout is not None and holding.stderr is not None
        if holding.returncode != 0:
            raise SystemExit(f"holder failed: {holding.stderr.read()}")
        return second.stdout, second.stderr

    _, error = contend(
        f"public.jp_close_conversation('{close_chat.id}')", commit(close_chat, user, assistant, 0)
    )
    if "Conversation is closed" not in error:
        raise SystemExit(f"FAILED: concurrent close did not reject the waiting turn: {error}")
    print("ok: a turn waiting on a concurrent close is rejected")

    _, error = contend(accept(accept_chat, first_record, 1), accept(accept_chat, second_record, 1))
    if "Conversation already accepted" not in error:
        raise SystemExit(f"FAILED: concurrent accept did not refuse the second: {error}")
    counts = psql(
        DATABASE,
        f"select 'count=' || count(*) from public.reflections where conversation_id = '{accept_chat.id}';",
    )
    if "count=1" not in counts:
        raise SystemExit(f"FAILED: concurrent accepts did not store exactly one reflection: {counts}")
    print("ok: concurrent accepts store exactly one reflection")

    choice_chat = conversation(USER_A)
    choice_user, choice_assistant = pair(choice_chat, "pending offer", 0)
    psql(DATABASE, f"{as_user(USER_A)} select {create(choice_chat)};")
    _, error = contend(
        preference(choice_chat, InteractionPreference.LISTEN, 0, uuid4()),
        commit(choice_chat, choice_user, choice_assistant, 0),
    )
    if "Conversation changed" not in error:
        raise SystemExit(f"FAILED: concurrent preference did not reject the waiting turn: {error}")
    print("ok: a turn waiting on a concurrent preference choice is rejected")


def reset_database() -> None:
    require_local_postgres_dsn(os.getenv("JOURNALPULSE_PG_DSN"))
    psql(
        "postgres",
        """
        select pg_terminate_backend(pid)
        from pg_stat_activity
        where datname = 'jp_verify' and pid <> pg_backend_pid();
        drop database if exists jp_verify;
        create database jp_verify;
        """,
    )


def apply_schema() -> None:
    psql(DATABASE, (ROOT / "scripts" / "pg_harness.sql").read_text())
    for path in sorted((ROOT / "supabase" / "migrations").glob("*.sql")):
        psql(DATABASE, path.read_text())


def main() -> None:
    reset_database()
    apply_schema()
    output = psql(DATABASE, behavior_sql())
    for line in output.splitlines():
        if "ok:" in line:
            print(line.split("NOTICE:", 1)[-1].strip())
    if "PostgreSQL schema verification passed" not in output:
        sys.stderr.write(output)
        raise SystemExit("schema verification did not report success")
    concurrency_checks()
    preference_checks()
    retention_checks()
    print("PostgreSQL schema verification passed")


if __name__ == "__main__":
    main()
