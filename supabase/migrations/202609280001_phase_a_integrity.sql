-- Phase A: database-enforced conversation lifecycle, trusted provenance, shared rate
-- limiting, and bounded retention.
--
-- 1. Every conversation carries a revision. A turn, accept, close, sweep, or delete
--    changes or removes the row, so a model reply computed against an older revision
--    can no longer be committed. No transaction is held open while the model runs:
--    the API reads, calls the model, then commits conditionally.
-- 2. Signed-in users keep SELECT and DELETE on their own rows (RLS), but lose INSERT
--    and UPDATE. All writes go through the functions below. Writes that carry
--    server provenance (policy, model, safety, catalog cards) must be signed with a
--    key shared only by the API and this database, so a user holding their own JWT
--    cannot forge evidence the product later treats as system-generated.
-- 3. Temporary chat text is purged on a schedule (pg_cron, when available), not only
--    when the person returns.

create schema if not exists private;
revoke all on schema private from public;
do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    execute 'revoke all on schema private from anon';
  end if;
  if exists (select 1 from pg_roles where rolname = 'authenticated') then
    execute 'revoke all on schema private from authenticated';
  end if;
end
$$;

-- The operator sets the value once per environment; it is never committed:
--   insert into private.server_secrets (name, value) values ('write_signing_key', '<key>')
--   on conflict (name) do update set value = excluded.value;
create table if not exists private.server_secrets (
  name text primary key,
  value text not null check (length(value) >= 32)
);
revoke all on table private.server_secrets from public;

create or replace function private.signature_valid(payload text, signature text)
returns boolean
language plpgsql
stable
security definer
set search_path = private, public, extensions
as $$
declare
  signing_key text;
begin
  select value into signing_key from private.server_secrets where name = 'write_signing_key';
  if signing_key is null or signature is null then
    return false;
  end if;
  return encode(hmac(convert_to(payload, 'UTF8'), convert_to(signing_key, 'UTF8'), 'sha256'), 'hex')
    = lower(signature);
end;
$$;

-- Verifies and unwraps a signed server payload bound to one purpose and one owner.
create or replace function private.verified_payload(payload text, signature text, purpose text)
returns jsonb
language plpgsql
stable
security definer
set search_path = private, public, extensions
as $$
declare
  owner_id uuid := auth.uid();
  body jsonb;
begin
  if owner_id is null then
    raise exception 'Authentication required' using errcode = 'PT401';
  end if;
  if not private.signature_valid(payload, signature) then
    raise exception 'Untrusted write' using errcode = 'PT403';
  end if;
  body := payload::jsonb;
  if body->>'purpose' is distinct from purpose then
    raise exception 'Untrusted write' using errcode = 'PT403';
  end if;
  if (body->>'user_id')::uuid is distinct from owner_id then
    raise exception 'Record owner does not match authenticated user' using errcode = 'PT403';
  end if;
  if (body->>'issued_at')::timestamptz < now() - interval '15 minutes'
     or (body->>'issued_at')::timestamptz > now() + interval '5 minutes' then
    raise exception 'Expired write' using errcode = 'PT403';
  end if;
  return body;
end;
$$;

-- Schema changes -------------------------------------------------------------

alter table public.conversations add column if not exists revision integer not null default 0;

alter table public.reflections
  add column if not exists conversation_id uuid references public.conversations(id) on delete set null;
create unique index if not exists reflections_conversation_unique_idx
  on public.reflections (conversation_id) where conversation_id is not null;

create table if not exists public.rate_limit_events (
  id bigserial primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  bucket text not null,
  created_at timestamptz not null default now()
);
create index if not exists rate_limit_events_user_bucket_idx
  on public.rate_limit_events (user_id, bucket, created_at);
alter table public.rate_limit_events enable row level security;

create index if not exists conversations_open_updated_idx
  on public.conversations (updated_at) where status = 'open';

-- Signed-in users read and delete their own rows; every write goes through a function.
do $$
declare
  table_name text;
begin
  foreach table_name in array array[
    'profiles', 'consents', 'reflections', 'affective_observations', 'policy_decisions',
    'outcomes', 'episodic_memories', 'model_runs', 'safety_events', 'conversations',
    'conversation_messages'
  ] loop
    execute format('revoke insert, update, truncate, references, trigger on public.%I from anon, authenticated', table_name);
    execute format('revoke all on public.%I from anon', table_name);
    execute format('grant select, delete on public.%I to authenticated', table_name);
  end loop;
end
$$;
revoke all on table public.rate_limit_events from anon, authenticated;
revoke all on sequence public.rate_limit_events_id_seq from anon, authenticated;

-- The previous write functions accepted unsigned provenance from any signed-in user.
drop function if exists public.save_reflection_bundle(jsonb);
drop function if exists public.save_conversation_turn(jsonb);
drop function if exists public.close_conversation(uuid, boolean);

-- Internal helpers -------------------------------------------------------------

create or replace function private.insert_reflection_bundle(
  owner_id uuid,
  payload jsonb,
  linked_conversation uuid
)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  target_reflection_id uuid := (payload->>'id')::uuid;
  decision_payload jsonb := payload->'decision';
  model_payload jsonb := payload->'model_run';
  safety_payload jsonb := payload->'safety';
  existing_record jsonb;
begin
  if (payload->>'user_id')::uuid is distinct from owner_id then
    raise exception 'Record owner does not match authenticated user' using errcode = 'PT403';
  end if;

  select reflections.record into existing_record
  from public.reflections
  where reflections.id = target_reflection_id and reflections.user_id = owner_id;
  if found then
    return existing_record;
  end if;

  insert into public.reflections (
    id, user_id, created_at, raw_text, text_retained, context, state, target,
    reflection, safety, decision, model_run, record, conversation_id
  ) values (
    target_reflection_id,
    owner_id,
    (payload->>'created_at')::timestamptz,
    payload->>'text',
    coalesce((payload->>'text_retained')::boolean, false),
    coalesce(payload->'context', '{}'::jsonb),
    payload->'state',
    payload->'target',
    payload->'reflection',
    safety_payload,
    decision_payload,
    model_payload,
    payload,
    linked_conversation
  );

  insert into public.affective_observations (user_id, reflection_id, observation_kind, state)
  values (owner_id, target_reflection_id, 'self_report', payload->'state');

  insert into public.policy_decisions (
    id, user_id, reflection_id, policy_name, policy_version, action_id,
    recommended_action_id, selection_source, eligible_for_ope, propensity,
    available_actions, context_snapshot
  ) values (
    (decision_payload->>'decision_id')::uuid,
    owner_id,
    target_reflection_id,
    decision_payload->>'policy_name',
    decision_payload->>'policy_version',
    decision_payload->>'action_id',
    decision_payload->>'recommended_action_id',
    decision_payload->>'selection_source',
    coalesce((decision_payload->>'eligible_for_ope')::boolean, true),
    (decision_payload->>'propensity')::double precision,
    decision_payload->'safe_action_ids',
    decision_payload->'context_snapshot'
  );

  if model_payload is not null and jsonb_typeof(model_payload) <> 'null' then
    insert into public.model_runs (
      user_id, reflection_id, model, provider, latency_ms, prompt_tokens,
      completion_tokens, schema_valid, fallback_reason
    ) values (
      owner_id,
      target_reflection_id,
      model_payload->>'model',
      model_payload->>'provider',
      (model_payload->>'latency_ms')::integer,
      nullif(model_payload->>'prompt_tokens', '')::integer,
      nullif(model_payload->>'completion_tokens', '')::integer,
      (model_payload->>'schema_valid')::boolean,
      model_payload->>'fallback_reason'
    );
  end if;

  insert into public.safety_events (user_id, reflection_id, mode, reason_codes, locale)
  values (
    owner_id,
    target_reflection_id,
    safety_payload->>'mode',
    coalesce(safety_payload->'reasons', '[]'::jsonb),
    safety_payload->>'locale'
  );

  return payload;
end;
$$;

-- Closes one locked conversation and clears message text unless the person chose to
-- keep it. The caller holds the row lock.
create or replace function private.close_locked_conversation(
  target_conversation_id uuid,
  linked_reflection uuid
)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  stored jsonb;
begin
  update public.conversations
  set status = 'closed',
      updated_at = now(),
      revision = public.conversations.revision + 1,
      reflection_id = coalesce(linked_reflection, public.conversations.reflection_id),
      record = public.conversations.record
        || jsonb_build_object(
          'status', 'closed',
          'updated_at', to_jsonb(now()),
          'revision', public.conversations.revision + 1,
          'reflection_id', to_jsonb(coalesce(linked_reflection, public.conversations.reflection_id))
        )
  where public.conversations.id = target_conversation_id
  returning public.conversations.record into stored;

  if coalesce((stored->>'retain_text')::boolean, false) = false then
    update public.conversation_messages
    set record = jsonb_set(public.conversation_messages.record, '{content}', 'null'::jsonb)
    where public.conversation_messages.conversation_id = target_conversation_id
      and jsonb_typeof(public.conversation_messages.record->'content') is distinct from 'null';
  end if;
  return stored;
end;
$$;

create or replace function private.stored_turn(
  target_conversation_id uuid,
  owner_id uuid,
  target_client_message_id uuid
)
returns jsonb
language plpgsql
stable
security definer
set search_path = private, public
as $$
declare
  existing_user jsonb;
  existing_assistant jsonb;
begin
  select messages.record into existing_user
  from public.conversation_messages as messages
  where messages.conversation_id = target_conversation_id
    and messages.client_message_id = target_client_message_id
    and messages.user_id = owner_id;
  if not found then
    return null;
  end if;
  select messages.record into existing_assistant
  from public.conversation_messages as messages
  where messages.conversation_id = target_conversation_id
    and messages.user_id = owner_id
    and messages.role = 'assistant'
    and messages.created_at >= (existing_user->>'created_at')::timestamptz
  order by messages.created_at asc
  limit 1;
  return jsonb_build_object(
    'conversation', (
      select conversations.record || jsonb_build_object('revision', conversations.revision)
      from public.conversations
      where conversations.id = target_conversation_id
    ),
    'user_message', existing_user,
    'assistant_message', existing_assistant
  );
end;
$$;

-- Public write functions ---------------------------------------------------------

create or replace function public.jp_create_conversation(payload text, signature text)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  body jsonb := private.verified_payload(payload, signature, 'create_conversation');
  owner_id uuid := auth.uid();
  conversation jsonb := body->'conversation';
  target_conversation_id uuid := (conversation->>'id')::uuid;
  existing_owner uuid;
  existing_record jsonb;
begin
  if (conversation->>'user_id')::uuid is distinct from owner_id then
    raise exception 'Record owner does not match authenticated user' using errcode = 'PT403';
  end if;
  select conversations.user_id, conversations.record || jsonb_build_object('revision', conversations.revision)
  into existing_owner, existing_record
  from public.conversations
  where conversations.id = target_conversation_id;
  if found then
    if existing_owner <> owner_id then
      raise exception 'Conversation request ID is already in use' using errcode = 'PT409';
    end if;
    return existing_record;
  end if;
  insert into public.conversations (id, user_id, created_at, updated_at, status, revision, record)
  values (
    target_conversation_id,
    owner_id,
    (conversation->>'created_at')::timestamptz,
    (conversation->>'updated_at')::timestamptz,
    'open',
    0,
    conversation || jsonb_build_object('status', 'open', 'revision', 0, 'reflection_id', null)
  );
  return conversation || jsonb_build_object('status', 'open', 'revision', 0, 'reflection_id', null);
end;
$$;

-- Commits one turn only if the conversation is still open at the revision the turn
-- was computed from. A retry with the same client message ID returns the stored turn.
create or replace function public.jp_commit_turn(payload text, signature text)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  body jsonb := private.verified_payload(payload, signature, 'commit_turn');
  owner_id uuid := auth.uid();
  conversation jsonb := body->'conversation';
  user_message jsonb := body->'user_message';
  assistant_message jsonb := body->'assistant_message';
  target_conversation_id uuid := (conversation->>'id')::uuid;
  target_client_message_id uuid := (user_message->>'client_message_id')::uuid;
  expected_revision integer := (body->>'expected_revision')::integer;
  current_status text;
  current_revision integer;
  stored jsonb;
  next_revision integer;
  committed jsonb;
begin
  if (user_message->>'conversation_id')::uuid is distinct from target_conversation_id
     or (assistant_message->>'conversation_id')::uuid is distinct from target_conversation_id then
    raise exception 'Message does not belong to this conversation' using errcode = 'PT400';
  end if;
  if target_client_message_id is null or expected_revision is null then
    raise exception 'Turn is missing its request or revision' using errcode = 'PT400';
  end if;

  select conversations.status, conversations.revision
  into current_status, current_revision
  from public.conversations
  where conversations.id = target_conversation_id and conversations.user_id = owner_id
  for update;
  if not found then
    raise exception 'Conversation not found' using errcode = 'PT404';
  end if;

  stored := private.stored_turn(target_conversation_id, owner_id, target_client_message_id);
  if stored is not null then
    return stored;
  end if;
  if current_status <> 'open' then
    raise exception 'Conversation is closed' using errcode = 'PT409';
  end if;
  if current_revision <> expected_revision then
    raise exception 'Conversation changed' using errcode = 'PT409';
  end if;

  insert into public.conversation_messages (
    id, conversation_id, user_id, client_message_id, role, created_at, record
  ) values
    (
      (user_message->>'id')::uuid, target_conversation_id, owner_id, target_client_message_id,
      'user', (user_message->>'created_at')::timestamptz, user_message
    ),
    (
      (assistant_message->>'id')::uuid, target_conversation_id, owner_id, null,
      'assistant', (assistant_message->>'created_at')::timestamptz, assistant_message
    );

  next_revision := current_revision + 1;
  committed := conversation || jsonb_build_object(
    'status', 'open', 'revision', next_revision, 'reflection_id', null
  );
  update public.conversations
  set updated_at = (conversation->>'updated_at')::timestamptz,
      revision = next_revision,
      record = committed
  where public.conversations.id = target_conversation_id;

  return jsonb_build_object(
    'conversation', committed,
    'user_message', user_message,
    'assistant_message', assistant_message
  );
end;
$$;

-- Accept, save the reflection bundle, link it, close, and purge in one transaction.
create or replace function public.jp_accept_conversation(payload text, signature text)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  body jsonb := private.verified_payload(payload, signature, 'accept_conversation');
  owner_id uuid := auth.uid();
  target_conversation_id uuid := (body->>'conversation_id')::uuid;
  expected_revision integer := (body->>'expected_revision')::integer;
  reflection jsonb := body->'reflection';
  target_reflection_id uuid := (reflection->>'id')::uuid;
  current_status text;
  current_revision integer;
  linked_reflection uuid;
  saved jsonb;
begin
  select conversations.status, conversations.revision, conversations.reflection_id
  into current_status, current_revision, linked_reflection
  from public.conversations
  where conversations.id = target_conversation_id and conversations.user_id = owner_id
  for update;
  if not found then
    raise exception 'Conversation not found' using errcode = 'PT404';
  end if;

  if linked_reflection is not null then
    if linked_reflection = target_reflection_id then
      select reflections.record into saved from public.reflections
      where reflections.id = linked_reflection and reflections.user_id = owner_id;
      return saved;
    end if;
    raise exception 'Conversation already accepted' using errcode = 'PT409';
  end if;
  if current_status <> 'open' then
    raise exception 'Conversation is closed' using errcode = 'PT409';
  end if;
  if current_revision <> expected_revision then
    raise exception 'Conversation changed' using errcode = 'PT409';
  end if;

  saved := private.insert_reflection_bundle(owner_id, reflection, target_conversation_id);
  perform private.close_locked_conversation(target_conversation_id, target_reflection_id);
  return saved;
end;
$$;

-- Legacy guided-reflection save. Same bundle, no conversation.
create or replace function public.jp_save_reflection(payload text, signature text)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  body jsonb := private.verified_payload(payload, signature, 'save_reflection');
begin
  return private.insert_reflection_bundle(auth.uid(), body->'reflection', null);
end;
$$;

-- Closing is the person's own decision, so it needs no server signature.
create or replace function public.jp_close_conversation(p_conversation_id uuid)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  owner_id uuid := auth.uid();
  current_status text;
  stored jsonb;
begin
  if owner_id is null then
    raise exception 'Authentication required' using errcode = 'PT401';
  end if;
  select conversations.status, conversations.record || jsonb_build_object('revision', conversations.revision)
  into current_status, stored
  from public.conversations
  where conversations.id = p_conversation_id and conversations.user_id = owner_id
  for update;
  if not found then
    raise exception 'Conversation not found' using errcode = 'PT404';
  end if;
  if current_status = 'closed' then
    return stored;
  end if;
  return private.close_locked_conversation(p_conversation_id, null);
end;
$$;

create or replace function public.jp_delete_conversation(p_conversation_id uuid)
returns boolean
language plpgsql
security definer
set search_path = private, public
as $$
declare
  owner_id uuid := auth.uid();
  linked_reflection uuid;
begin
  if owner_id is null then
    raise exception 'Authentication required' using errcode = 'PT401';
  end if;
  select conversations.reflection_id into linked_reflection
  from public.conversations
  where conversations.id = p_conversation_id and conversations.user_id = owner_id
  for update;
  if not found then
    return false;
  end if;
  if linked_reflection is not null then
    delete from public.reflections
    where reflections.id = linked_reflection and reflections.user_id = owner_id;
  end if;
  delete from public.conversations
  where conversations.id = p_conversation_id and conversations.user_id = owner_id;
  return true;
end;
$$;

-- Outcomes are the person's own report, so they need no server signature.
create or replace function public.save_outcome_record(payload jsonb)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  owner_id uuid := auth.uid();
  outcome_id uuid := (payload->>'id')::uuid;
  target_decision_id uuid := (payload->>'decision_id')::uuid;
  existing_record jsonb;
begin
  if owner_id is null then
    raise exception 'Authentication required' using errcode = 'PT401';
  end if;
  if (payload->>'user_id')::uuid is distinct from owner_id then
    raise exception 'Record owner does not match authenticated user' using errcode = 'PT403';
  end if;

  select outcomes.record into existing_record
  from public.outcomes
  where outcomes.id = outcome_id and outcomes.user_id = owner_id;
  if found then
    if (existing_record->>'decision_id')::uuid <> target_decision_id then
      raise exception 'Outcome request ID is already in use' using errcode = 'PT409';
    end if;
    return existing_record;
  end if;

  if not exists (
    select 1 from public.policy_decisions
    where policy_decisions.id = target_decision_id and policy_decisions.user_id = owner_id
  ) then
    raise exception 'Policy decision does not belong to authenticated user' using errcode = 'PT404';
  end if;

  begin
    insert into public.outcomes (id, user_id, decision_id, created_at, record)
    values (outcome_id, owner_id, target_decision_id, (payload->>'created_at')::timestamptz, payload);
  exception
    when unique_violation then
      raise exception 'An outcome already exists for this decision' using errcode = 'PT409';
  end;
  return payload;
end;
$$;

create or replace function public.delete_my_journalpulse_data()
returns integer
language plpgsql
security definer
set search_path = private, public
as $$
declare
  owner_id uuid := auth.uid();
  deleted_count integer := 0;
  affected integer := 0;
  table_name text;
begin
  if owner_id is null then
    raise exception 'Authentication required' using errcode = 'PT401';
  end if;
  -- Children first so each row is counted once instead of vanishing in a cascade.
  foreach table_name in array array[
    'outcomes', 'affective_observations', 'policy_decisions', 'model_runs', 'safety_events',
    'conversation_messages', 'episodic_memories', 'reflections', 'conversations',
    'consents', 'profiles'
  ] loop
    execute format('delete from public.%I where user_id = $1', table_name) using owner_id;
    get diagnostics affected = row_count;
    deleted_count := deleted_count + affected;
  end loop;
  -- Usage counters are removed too, but they are not journal records, so not counted.
  delete from public.rate_limit_events where rate_limit_events.user_id = owner_id;
  return deleted_count;
end;
$$;

-- Shared rate limit ---------------------------------------------------------------

create or replace function public.jp_consume_rate_limit(
  p_bucket text,
  p_max_events integer,
  p_window_seconds integer
)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  owner_id uuid := auth.uid();
  window_length interval := make_interval(secs => p_window_seconds);
  used integer;
  oldest timestamptz;
begin
  if owner_id is null then
    raise exception 'Authentication required' using errcode = 'PT401';
  end if;
  if p_max_events < 1 or p_window_seconds < 1 or p_window_seconds > 86400 or length(p_bucket) > 40 then
    raise exception 'Invalid rate limit' using errcode = 'PT400';
  end if;
  -- One writer per person and bucket, so concurrent instances see the same count.
  perform pg_advisory_xact_lock(hashtextextended(owner_id::text || ':' || p_bucket, 0));
  delete from public.rate_limit_events
  where rate_limit_events.user_id = owner_id
    and rate_limit_events.bucket = p_bucket
    and rate_limit_events.created_at <= now() - window_length;
  select count(*), min(rate_limit_events.created_at) into used, oldest
  from public.rate_limit_events
  where rate_limit_events.user_id = owner_id and rate_limit_events.bucket = p_bucket;
  if used >= p_max_events then
    return jsonb_build_object(
      'allowed', false,
      'retry_after', greatest(1, ceil(extract(epoch from (oldest + window_length - now())))::integer),
      'remaining', 0
    );
  end if;
  insert into public.rate_limit_events (user_id, bucket) values (owner_id, p_bucket);
  return jsonb_build_object('allowed', true, 'retry_after', 0, 'remaining', p_max_events - used - 1);
end;
$$;

-- Retention ------------------------------------------------------------------------

-- Closes chats idle longer than the limit and clears text that should not survive.
-- The second step also repairs any closed, non-retained chat that still holds text.
create or replace function private.purge_conversations(max_idle interval, only_owner uuid)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  closed_count integer := 0;
  purged_count integer := 0;
  repaired integer := 0;
  target record;
begin
  for target in
    select conversations.id, coalesce((conversations.record->>'retain_text')::boolean, false) as retained
    from public.conversations
    where conversations.status = 'open'
      and conversations.updated_at < now() - max_idle
      and (only_owner is null or conversations.user_id = only_owner)
    for update skip locked
  loop
    if not target.retained then
      purged_count := purged_count + (
        select count(*) from public.conversation_messages
        where conversation_messages.conversation_id = target.id
          and jsonb_typeof(conversation_messages.record->'content') is distinct from 'null'
      );
    end if;
    perform private.close_locked_conversation(target.id, null);
    closed_count := closed_count + 1;
  end loop;

  update public.conversation_messages as messages
  set record = jsonb_set(messages.record, '{content}', 'null'::jsonb)
  from public.conversations
  where messages.conversation_id = conversations.id
    and conversations.status = 'closed'
    and coalesce((conversations.record->>'retain_text')::boolean, false) = false
    and jsonb_typeof(messages.record->'content') is distinct from 'null'
    and (only_owner is null or conversations.user_id = only_owner);
  get diagnostics repaired = row_count;
  purged_count := purged_count + repaired;

  return jsonb_build_object('closed', closed_count, 'purged_messages', purged_count);
end;
$$;

-- Scheduled job entry point. Not callable by signed-in users.
create or replace function public.jp_purge_expired_conversations()
returns jsonb
language sql
security definer
set search_path = private, public
as $$
  select private.purge_conversations(interval '24 hours', null);
$$;

-- Request-time defense in depth for the signed-in person only.
create or replace function public.jp_close_my_stale_conversations()
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
begin
  if auth.uid() is null then
    raise exception 'Authentication required' using errcode = 'PT401';
  end if;
  return private.purge_conversations(interval '24 hours', auth.uid());
end;
$$;

do $$
begin
  if exists (select 1 from pg_available_extensions where name = 'pg_cron') then
    create extension if not exists pg_cron;
    perform cron.schedule(
      'journalpulse-retention',
      '*/15 * * * *',
      'select public.jp_purge_expired_conversations()'
    );
  end if;
exception
  when others then
    raise notice 'pg_cron is unavailable (%); request-time cleanup remains the only purge path', sqlerrm;
end
$$;

-- Readiness ------------------------------------------------------------------------

-- Lets /ready prove the schema is current and that the API and database share the
-- signing key, without revealing the key. Only probes of the form "readiness:..." are
-- accepted, so a verified probe can never double as a write payload.
create or replace function public.jp_readiness(probe text, signature text)
returns jsonb
language plpgsql
stable
security definer
set search_path = private, public
as $$
declare
  signing text;
  retention text := 'not_scheduled';
begin
  if not exists (select 1 from private.server_secrets where name = 'write_signing_key') then
    signing := 'missing';
  elsif probe like 'readiness:%' and private.signature_valid(probe, signature) then
    signing := 'valid';
  else
    signing := 'invalid';
  end if;
  if to_regclass('cron.job') is not null then
    execute 'select case when exists (select 1 from cron.job where jobname = $1 and active)
             then ''scheduled'' else ''not_scheduled'' end'
      into retention using 'journalpulse-retention';
  end if;
  return jsonb_build_object('schema', 'phase-a-1', 'signing', signing, 'retention_job', retention);
end;
$$;

-- Grants -------------------------------------------------------------------------

revoke all on all functions in schema private from public;
-- Supabase grants EXECUTE on new public functions to anon and authenticated by
-- default, so each one is reset before the intended grants below.
do $$
declare
  signature text;
begin
  foreach signature in array array[
    'public.jp_create_conversation(text, text)',
    'public.jp_commit_turn(text, text)',
    'public.jp_accept_conversation(text, text)',
    'public.jp_save_reflection(text, text)',
    'public.jp_close_conversation(uuid)',
    'public.jp_delete_conversation(uuid)',
    'public.save_outcome_record(jsonb)',
    'public.delete_my_journalpulse_data()',
    'public.jp_consume_rate_limit(text, integer, integer)',
    'public.jp_close_my_stale_conversations()',
    'public.jp_purge_expired_conversations()',
    'public.jp_readiness(text, text)'
  ] loop
    execute format('revoke all on function %s from public, anon, authenticated', signature);
  end loop;
end
$$;

grant execute on function public.jp_create_conversation(text, text) to authenticated;
grant execute on function public.jp_commit_turn(text, text) to authenticated;
grant execute on function public.jp_accept_conversation(text, text) to authenticated;
grant execute on function public.jp_save_reflection(text, text) to authenticated;
grant execute on function public.jp_close_conversation(uuid) to authenticated;
grant execute on function public.jp_delete_conversation(uuid) to authenticated;
grant execute on function public.save_outcome_record(jsonb) to authenticated;
grant execute on function public.delete_my_journalpulse_data() to authenticated;
grant execute on function public.jp_consume_rate_limit(text, integer, integer) to authenticated;
grant execute on function public.jp_close_my_stale_conversations() to authenticated;
grant execute on function public.jp_readiness(text, text) to anon, authenticated;
do $$
begin
  if exists (select 1 from pg_roles where rolname = 'service_role') then
    execute 'grant execute on function public.jp_purge_expired_conversations() to service_role';
  end if;
end
$$;
