-- Additive audit fixes. Previously deployed migration files remain immutable.
-- Keeps the old production generation policy (20/minute), while new writers use
-- signed usage policy. Request UUIDs cannot silently change privacy settings or
-- retained turn text. Existing owner/source/preference lifecycle guards remain.

create or replace function private.consume_rate_limit(
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
  if p_bucket is null or p_max_events is null or p_window_seconds is null
     or p_max_events < 1 or p_max_events > 1000
     or p_window_seconds < 1 or p_window_seconds > 86400
     or length(p_bucket) not between 1 and 40 then
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

revoke all on function private.consume_rate_limit(text, integer, integer) from public, anon, authenticated;

-- New servers sign the complete owner-bound usage policy, just like provenance
-- writes. A person holding a session token cannot shorten the cleanup window.
create function public.jp_consume_rate_limit_v2(payload text, signature text)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  body jsonb := private.verified_payload(payload, signature, 'consume_rate_limit');
begin
  return private.consume_rate_limit(
    body->>'bucket', (body->>'max_events')::integer, (body->>'window_seconds')::integer);
end;
$$;
revoke all on function public.jp_consume_rate_limit_v2(text, text) from public, anon, authenticated;
grant execute on function public.jp_consume_rate_limit_v2(text, text) to authenticated;

-- Existing production uses this older RPC with the default generation policy.
-- Preserve that contract, but reject caller-selected windows, limits, and buckets.
-- Alternate limits require the new signed RPC; legacy failures remain fail-closed.
create or replace function public.jp_consume_rate_limit(
  p_bucket text, p_max_events integer, p_window_seconds integer
)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
begin
  if p_bucket is distinct from 'generation' or p_max_events is distinct from 20
     or p_window_seconds is distinct from 60 then
    raise exception 'Invalid legacy rate limit policy' using errcode = 'PT400';
  end if;
  return private.consume_rate_limit('generation', 20, 60);
end;
$$;
revoke all on function public.jp_consume_rate_limit(text, integer, integer) from public, anon, authenticated;
grant execute on function public.jp_consume_rate_limit(text, integer, integer) to authenticated;

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
  target_id uuid := (conversation->>'id')::uuid;
  source_id uuid := (conversation->>'source_entry_id')::uuid;
  source_created timestamptz := (conversation->>'source_entry_created_at')::timestamptz;
  existing public.conversations%rowtype;
  fresh jsonb;
begin
  if (conversation->>'user_id')::uuid is distinct from owner_id then
    raise exception 'Record owner does not match authenticated user' using errcode = 'PT403';
  end if;
  perform pg_advisory_xact_lock(hashtextextended('conversation:' || target_id::text, 0));
  -- Lock the owned source across creation; a delete between API read and insert
  -- cannot create an orphan. The composite FK independently enforces ownership.
  if source_id is not null then
    perform 1 from public.journal_entries
    where id = source_id and user_id = owner_id and created_at = source_created for key share;
    if not found then
      raise exception 'Journal entry not found' using errcode = 'PT404';
    end if;
  end if;
  select * into existing from public.conversations where id = target_id;
  if found then
    if existing.user_id <> owner_id or existing.source_entry_id is distinct from source_id
       or existing.source_entry_created_at is distinct from source_created
       or (existing.record->>'llm_consent')::boolean is distinct from (conversation->>'llm_consent')::boolean
       or (existing.record->>'retain_text')::boolean is distinct from (conversation->>'retain_text')::boolean
       or existing.record->>'locale' is distinct from conversation->>'locale' then
      raise exception 'Conversation request ID is already in use' using errcode = 'PT409';
    end if;
    return existing.record || jsonb_build_object('revision', existing.revision);
  end if;
  fresh := conversation || jsonb_build_object('status', 'open', 'revision', 0, 'reflection_id', null);
  insert into public.conversations (
    id, user_id, created_at, updated_at, status, revision, record, source_entry_id, source_entry_created_at)
  values (target_id, owner_id, (fresh->>'created_at')::timestamptz, (fresh->>'updated_at')::timestamptz,
          'open', 0, fresh, source_id, source_created);
  return fresh;
end;
$$;

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
  current_record jsonb;
begin
  if (user_message->>'conversation_id')::uuid is distinct from target_conversation_id
     or (assistant_message->>'conversation_id')::uuid is distinct from target_conversation_id then
    raise exception 'Message does not belong to this conversation' using errcode = 'PT400';
  end if;
  if target_client_message_id is null or expected_revision is null then
    raise exception 'Turn is missing its request or revision' using errcode = 'PT400';
  end if;

  select conversations.status, conversations.revision, conversations.record
  into current_status, current_revision, current_record
  from public.conversations
  where conversations.id = target_conversation_id and conversations.user_id = owner_id
  for update;
  if not found then
    raise exception 'Conversation not found' using errcode = 'PT404';
  end if;

  stored := private.stored_turn(target_conversation_id, owner_id, target_client_message_id);
  if stored is not null then
    -- No text digest is retained. Once text has been purged, safe request identity
    -- cannot be established, so retries fail explicitly without writing a new turn.
    if stored->'user_message'->>'content' is null then
      raise exception 'This message''s text is no longer retained; its retry cannot be verified.' using errcode = 'PT409';
    end if;
    if stored->'user_message'->>'content' is distinct from user_message->>'content'
       or (jsonb_typeof(stored->'user_message'->'request_inputs') = 'object'
           and stored->'user_message'->'request_inputs' is distinct from user_message->'request_inputs') then
      raise exception 'Message request ID is already in use' using errcode = 'PT409';
    end if;
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
  -- Preserve the database-owned preference even if an older server omits it.
  committed := committed || jsonb_build_object('interaction_preference',
    coalesce(current_record->>'interaction_preference', 'auto'));
  if current_record->>'interaction_preference' = 'listen'
     and committed->>'safety_mode' <> 'support' then
    committed := committed || jsonb_build_object('card', null, 'ready_for_action', false);
  end if;
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

revoke all on function public.jp_create_conversation(text, text) from public, anon, authenticated;
grant execute on function public.jp_create_conversation(text, text) to authenticated;
revoke all on function public.jp_commit_turn(text, text) from public, anon, authenticated;
grant execute on function public.jp_commit_turn(text, text) to authenticated;

-- Acceptance receipts bind one conversation and one reported choice.
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
  existing_conversation uuid;
begin
  if (payload->>'user_id')::uuid is distinct from owner_id then
    raise exception 'Record owner does not match authenticated user' using errcode = 'PT403';
  end if;

  perform pg_advisory_xact_lock(hashtextextended('reflection:' || target_reflection_id::text, 0));
  select reflections.record, reflections.conversation_id into existing_record, existing_conversation
  from public.reflections
  where reflections.id = target_reflection_id and reflections.user_id = owner_id;
  if found then
    if existing_conversation is distinct from linked_conversation
       or (linked_conversation is not null and (
         existing_record->'decision'->>'action_id' is distinct from payload->'decision'->>'action_id'
         or existing_record->'state' is distinct from payload->'state'
         or existing_record->'target' is distinct from payload->'target'
         or coalesce(existing_record->'self_report_input', 'null'::jsonb)
            is distinct from coalesce(payload->'self_report_input', 'null'::jsonb))) then
      raise exception 'Reflection request ID is already in use' using errcode = 'PT409';
    end if;
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
  saved_conversation uuid;
  current_record jsonb;
begin
  select conversations.status, conversations.revision, conversations.reflection_id, conversations.record
  into current_status, current_revision, linked_reflection, current_record
  from public.conversations
  where conversations.id = target_conversation_id and conversations.user_id = owner_id
  for update;
  if not found then
    raise exception 'Conversation not found' using errcode = 'PT404';
  end if;

  if linked_reflection is not null then
    if linked_reflection = target_reflection_id then
      select reflections.record, reflections.conversation_id into saved, saved_conversation
      from public.reflections
      where reflections.id = linked_reflection and reflections.user_id = owner_id;
      if saved_conversation is distinct from target_conversation_id
         or saved->'decision'->>'action_id' is distinct from reflection->'decision'->>'action_id'
         or saved->'state' is distinct from reflection->'state'
         or saved->'target' is distinct from reflection->'target' then
        raise exception 'Reflection request ID is already in use' using errcode = 'PT409';
      end if;
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

  -- New preference-aware chats require evidence of the card revision checked by
  -- the API. Old automatic chats keep their original acceptance contract.
  if coalesce(current_record->>'interaction_preference', 'auto') <> 'auto'
     and (body->>'client_card_revision')::integer is distinct from current_revision then
    raise exception 'Conversation changed' using errcode = 'PT409';
  end if;
  if current_record->>'interaction_preference' = 'listen'
     and current_record->>'safety_mode' <> 'support' then
    raise exception 'Conversation changed' using errcode = 'PT409';
  end if;
  saved := private.insert_reflection_bundle(owner_id, reflection, target_conversation_id);
  perform private.close_locked_conversation(target_conversation_id, target_reflection_id);
  return saved;
end;
$$;
revoke all on function private.insert_reflection_bundle(uuid,jsonb,uuid) from public, anon, authenticated;
revoke all on function public.jp_accept_conversation(text,text) from public, anon, authenticated;
grant execute on function public.jp_accept_conversation(text,text) to authenticated;
