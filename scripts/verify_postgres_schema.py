"""Apply the Supabase migrations to PostgreSQL and exercise RLS and the write functions.

Drops and recreates the database named jp_verify. Set JOURNALPULSE_PG_DSN to a
maintenance-database URI (for example the CI service). Without it, the script
uses local peer authentication as the postgres user.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

from journalpulse.domain import (
    AffectiveState,
    Conversation,
    ConversationMessage,
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

ROOT = Path(__file__).resolve().parents[1]
DATABASE = "jp_verify"
USER_A = UUID("11111111-1111-4111-8111-111111111111")
USER_B = UUID("22222222-2222-4222-8222-222222222222")
CONVERSATION_ID = UUID("aaaaaaaa-0000-4000-8000-000000000010")
STARTED = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def dsn_for(database: str) -> str | None:
    raw = os.environ.get("JOURNALPULSE_PG_DSN")
    if not raw:
        return None
    parts = urlsplit(raw)
    return urlunsplit(parts._replace(path=f"/{database}"))


def psql(database: str, sql: str) -> str:
    dsn = dsn_for(database)
    if dsn:
        command = ["psql", dsn, "-v", "ON_ERROR_STOP=1", "-X"]
    else:
        command = ["sudo", "-u", "postgres", "psql", "-d", database, "-v", "ON_ERROR_STOP=1", "-X"]
    completed = subprocess.run(command, input=sql, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(completed.returncode)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return completed.stdout


def quote(value: object) -> str:
    text = json.dumps(value)
    if "$jp$" in text:
        raise SystemExit("payload contains the SQL dollar-quote delimiter")
    return f"$jp${text}$jp$"


def state() -> AffectiveState:
    return AffectiveState(
        valence=-0.2,
        arousal=0.6,
        agency=0.4,
        emotion_tags=["frustration"],
        confidence=0.8,
    )


def reflection() -> ReflectionRecord:
    return ReflectionRecord(
        user_id=USER_A,
        created_at=STARTED,
        text=None,
        text_retained=False,
        state=state(),
        target=TargetState(goal="settle", arousal=0.3, agency=0.7),
        reflection=ReflectionCopy(
            summary="A summary.",
            interpretation="An interpretation.",
            reflection_question="What changed?",
        ),
        safety=SafetyResult(mode=SafetyMode.NORMAL, locale="CA", exploration_allowed=True),
        decision=PolicyDecision(
            action_id="approved-resource",
            propensity=1,
            policy_name="fixed-baseline",
            policy_version="1.0.0",
            safe_action_ids=["approved-resource"],
            context_snapshot={},
            explanation="Deterministic baseline.",
        ),
        model_run=ModelRun(model="openai/gpt-6-luna", latency_ms=12, schema_valid=True),
    )


def conversation() -> Conversation:
    return Conversation(
        id=CONVERSATION_ID,
        user_id=USER_A,
        created_at=STARTED,
        updated_at=STARTED,
        llm_consent=True,
        retain_text=False,
        locale="CA",
        prompt_version=CONVERSATION_PROMPT_VERSION,
    )


def message(
    *,
    role: MessageRole,
    content: str,
    client_message_id: UUID | None,
    at: datetime,
) -> ConversationMessage:
    run = None
    if role is MessageRole.ASSISTANT:
        run = ModelRun(model="openai/gpt-6-luna", latency_ms=20, schema_valid=True)
    return ConversationMessage(
        id=uuid4(),
        conversation_id=CONVERSATION_ID,
        client_message_id=client_message_id,
        role=role,
        content=content,
        created_at=at,
        safety_mode=SafetyMode.NORMAL,
        model_run=run,
    )


def turn(user_message: ConversationMessage, assistant_message: ConversationMessage, owner: UUID) -> str:
    stored = conversation().model_copy(update={"user_id": owner, "updated_at": assistant_message.created_at})
    payload = {
        "conversation": stored.model_dump(mode="json"),
        "user_message": user_message.model_dump(mode="json"),
        "assistant_message": assistant_message.model_dump(mode="json"),
    }
    return quote(payload)


def reset_database() -> None:
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


def apply_sql(path: Path) -> None:
    psql(DATABASE, path.read_text())


def behavior_sql() -> str:
    first_client = uuid4()
    second_client = uuid4()
    first_user = message(
        role=MessageRole.USER,
        content="private note",
        client_message_id=first_client,
        at=STARTED,
    )
    first_assistant = message(
        role=MessageRole.ASSISTANT,
        content="a reply",
        client_message_id=None,
        at=datetime(2026, 9, 27, 12, 0, 2, tzinfo=UTC),
    )
    second_user = message(
        role=MessageRole.USER,
        content="second note",
        client_message_id=second_client,
        at=datetime(2026, 9, 27, 12, 1, tzinfo=UTC),
    )
    second_assistant = message(
        role=MessageRole.ASSISTANT,
        content="second reply",
        client_message_id=None,
        at=datetime(2026, 9, 27, 12, 1, 2, tzinfo=UTC),
    )
    attack_user = message(
        role=MessageRole.USER,
        content="planted",
        client_message_id=uuid4(),
        at=datetime(2026, 9, 27, 12, 2, tzinfo=UTC),
    )
    attack_assistant = message(
        role=MessageRole.ASSISTANT,
        content="planted reply",
        client_message_id=None,
        at=datetime(2026, 9, 27, 12, 2, 2, tzinfo=UTC),
    )
    spoof_user = message(
        role=MessageRole.USER,
        content="spoofed",
        client_message_id=uuid4(),
        at=datetime(2026, 9, 27, 12, 3, tzinfo=UTC),
    )
    spoof_assistant = message(
        role=MessageRole.ASSISTANT,
        content="spoofed reply",
        client_message_id=None,
        at=datetime(2026, 9, 27, 12, 3, 2, tzinfo=UTC),
    )
    stored_reflection = reflection()
    outcome = OutcomeRecord(
        user_id=USER_A,
        decision_id=stored_reflection.decision.decision_id,
        created_at=datetime(2026, 9, 27, 12, 4, tzinfo=UTC),
        completed=True,
        helpfulness=4,
    )
    duplicate = outcome.model_copy(update={"id": uuid4()})
    stored = conversation()
    return f"""
    grant all on all tables in schema public to anon, authenticated, service_role;
    grant all on all sequences in schema public to anon, authenticated, service_role;

    insert into auth.users (id, email) values
      ('{USER_A}', 'a@example.test'),
      ('{USER_B}', 'b@example.test');

    create or replace function public._expect(condition boolean, label text)
    returns void language plpgsql as $expect$
    begin
      if condition then
        raise notice 'ok: %', label;
      else
        raise exception 'FAILED: %', label;
      end if;
    end
    $expect$;

    set role authenticated;
    select set_config('request.jwt.claim.sub', '{USER_A}', false);

    insert into public.conversations (
      id, user_id, created_at, updated_at, status, record
    ) values (
      '{stored.id}', '{stored.user_id}', '{stored.created_at.isoformat()}',
      '{stored.updated_at.isoformat()}', 'open', {quote(stored.model_dump(mode="json"))}::jsonb
    );

    select public.save_conversation_turn({turn(first_user, first_assistant, USER_A)}::jsonb);
    select public._expect(
      (select count(*) from public.conversation_messages) = 2,
      'turn stores both messages'
    );
    select public.save_conversation_turn({turn(first_user, first_assistant, USER_A)}::jsonb);
    select public._expect(
      (select count(*) from public.conversation_messages) = 2,
      'retry does not duplicate'
    );
    select public.save_conversation_turn({turn(second_user, second_assistant, USER_A)}::jsonb);
    select public._expect((select count(*) from public.conversation_messages) = 4, 'second turn appends');

    select set_config('request.jwt.claim.sub', '{USER_B}', false);
    select public._expect((select count(*) from public.conversations) = 0, 'B cannot see A conversation');
    select public._expect((select count(*) from public.conversation_messages) = 0, 'B cannot see A messages');

    do $attack$
    begin
      perform public.save_conversation_turn({turn(attack_user, attack_assistant, USER_B)}::jsonb);
      raise exception 'B attached a turn';
    exception
      when others then
        if sqlerrm = 'B attached a turn' then
          raise;
        end if;
        if sqlerrm <> 'Conversation not found' then
          raise;
        end if;
        raise notice 'ok: B cannot attach a turn to A''s conversation';
    end
    $attack$;

    do $spoof$
    begin
      perform public.save_conversation_turn({turn(spoof_user, spoof_assistant, USER_A)}::jsonb);
      raise exception 'B wrote as A';
    exception
      when others then
        if sqlerrm = 'B wrote as A' then
          raise;
        end if;
        if sqlerrm <> 'Record owner does not match authenticated user' then
          raise;
        end if;
        raise notice 'ok: B cannot spoof A as the record owner';
    end
    $spoof$;

    select set_config('request.jwt.claim.sub', '{USER_A}', false);
    select public.close_conversation('{CONVERSATION_ID}', true);
    select public._expect(
      (select status from public.conversations where id = '{CONVERSATION_ID}') = 'closed',
      'close writes the status column'
    );
    select public._expect(
      (select count(*) from public.conversation_messages) = 4,
      'close keeps the message rows'
    );
    select public._expect(
      (
        select count(*) from public.conversation_messages
        where jsonb_typeof(record->'content') is distinct from 'null'
      ) = 0,
      'close clears message text'
    );

    select public.save_reflection_bundle({quote(stored_reflection.model_dump(mode="json"))}::jsonb);
    select public.save_reflection_bundle({quote(stored_reflection.model_dump(mode="json"))}::jsonb);
    select public._expect((select count(*) from public.reflections) = 1, 'reflection save is idempotent');
    select public._expect((select count(*) from public.policy_decisions) = 1, 'decision row written');
    select public._expect((select count(*) from public.model_runs) = 1, 'model run written');
    select public._expect((select count(*) from public.safety_events) = 1, 'safety event written');
    select public._expect(
      (select count(*) from public.affective_observations) = 1,
      'observation written'
    );
    select public._expect(
      (select raw_text is null and text_retained = false from public.reflections),
      'reflection text stays withheld'
    );

    select set_config('request.jwt.claim.sub', '{USER_B}', false);
    select public._expect((select count(*) from public.reflections) = 0, 'B cannot read A reflection');
    select public.delete_my_journalpulse_data();

    select set_config('request.jwt.claim.sub', '{USER_A}', false);
    select public._expect((select count(*) from public.reflections) = 1, 'B erasure leaves A reflection');
    select public.save_outcome_record({quote(outcome.model_dump(mode="json"))}::jsonb);

    do $dup$
    begin
      perform public.save_outcome_record({quote(duplicate.model_dump(mode="json"))}::jsonb);
      raise exception 'duplicate outcome inserted';
    exception
      when unique_violation then
        raise notice 'ok: one outcome per decision';
    end
    $dup$;

    select public.delete_my_journalpulse_data();
    select public._expect((select count(*) from public.reflections) = 0, 'A erasure removes reflections');
    select public._expect((select count(*) from public.conversations) = 0, 'A erasure removes conversations');
    select public._expect((select count(*) from public.outcomes) = 0, 'A erasure removes outcomes');

    select set_config('request.jwt.claim.sub', '', false);
    do $anon$
    begin
      perform public.save_conversation_turn(
        '{{"conversation":{{}},"user_message":{{}},"assistant_message":{{}}}}'::jsonb
      );
      raise exception 'anonymous turn stored';
    exception
      when others then
        if sqlerrm = 'anonymous turn stored' then
          raise;
        end if;
        if sqlerrm <> 'Authentication required' then
          raise;
        end if;
        raise notice 'ok: anonymous turn rejected';
    end
    $anon$;

    select set_config('request.jwt.claim.sub', '{USER_A}', false);
    do $catalog$
    begin
      insert into public.intervention_catalog (id, payload) values ('unreviewed', '{{}}'::jsonb);
      raise exception 'catalog write allowed';
    exception
      when insufficient_privilege then
        raise notice 'ok: signed-in user cannot rewrite the catalog';
    end
    $catalog$;

    reset role;
    select 'PostgreSQL schema verification passed' as result;
    """


def main() -> None:
    reset_database()
    apply_sql(ROOT / "scripts" / "pg_harness.sql")
    for path in sorted((ROOT / "supabase" / "migrations").glob("*.sql")):
        apply_sql(path)
    stdout = psql(DATABASE, behavior_sql())
    if "PostgreSQL schema verification passed" not in stdout:
        sys.stderr.write(stdout)
        raise SystemExit("schema verification did not report success")
    print("PostgreSQL schema verification passed")


if __name__ == "__main__":
    main()
