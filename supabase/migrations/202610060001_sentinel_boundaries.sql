-- Sentinel repairs; previous migrations remain immutable.
-- Chat incarnation fencing is independent of UUID and timestamp reuse.

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

  -- New writers bind an explicit incarnation, including null for legacy rows.
  -- Older writers have no field; still fence their original server timestamp.
  if (body ? 'expected_incarnation_id' and
      current_record->>'incarnation_id' is distinct from body->>'expected_incarnation_id')
     or (current_record->>'created_at')::timestamptz is distinct from
        (conversation->>'created_at')::timestamptz then
    raise exception 'Conversation changed' using errcode = 'PT409';
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
           and ('{"activity_constraints":null}'::jsonb || stored->'user_message'->'request_inputs')
              is distinct from ('{"activity_constraints":null}'::jsonb || user_message->'request_inputs')) then
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
    'status', 'open', 'revision', next_revision, 'reflection_id', null,
    'incarnation_id', current_record->'incarnation_id', 'created_at', current_record->'created_at',
    'mode', current_record->'mode', 'llm_consent', current_record->'llm_consent',
    'retain_text', current_record->'retain_text'
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
  where public.conversations.id = target_conversation_id
  returning record into committed;

  return jsonb_build_object(
    'conversation', committed,
    'user_message', user_message,
    'assistant_message', assistant_message
  );
end;
$$;


-- Renewed requests cannot overtake a provider still within its supported deadline.
create or replace function public.jp_claim_activity_followup_v1(payload text,signature text)
returns jsonb language plpgsql security definer set search_path=private,public as $$
declare
  body jsonb:=private.verified_payload(payload,signature,'claim_activity_follow_up'); owner_id uuid:=auth.uid();
  target_id uuid:=(body->>'session_id')::uuid; request jsonb:=body->'request';
  request_id uuid:=(request->>'client_request_id')::uuid; chat public.conversations%rowtype;
  saved public.activity_sessions%rowtype; replay boolean; changed jsonb; is_final boolean;
begin
  select * into saved from public.activity_sessions where id=target_id and user_id=owner_id;
  if not found then raise exception 'Activity not found' using errcode='PT404'; end if;
  chat:=private.activity_locked_chat(saved.conversation_id,owner_id);
  select * into saved from public.activity_sessions where id=target_id and user_id=owner_id for update;
  if not found then raise exception 'Activity not found' using errcode='PT404'; end if;
  replay:=private.activity_receipt(owner_id,target_id,request_id,'follow_up',body->>'request_hash');
  if saved.record->>'follow_up_status'='ready' or (saved.record->>'follow_up_status'='generating'
     and (saved.record->>'follow_up_lease_until')::timestamptz>now()) then
    return jsonb_build_object('session',saved.record,'claimed',false);
  end if;
  perform private.activity_guard(chat,(request->>'expected_conversation_revision')::integer,true,true);
  if not replay and saved.revision<>(request->>'expected_revision')::integer then
    raise exception 'Activity changed' using errcode='PT409'; end if;
  if saved.record->>'report' is null or (saved.record->>'follow_up_attempts')::integer>=3 then
    raise exception 'The saved check-in has no available follow-up attempt' using errcode='PT409'; end if;
  select count(*)>=20 into is_final from public.conversation_messages where conversation_id=chat.id and user_id=owner_id and role='user';
  if is_final and exists(select 1 from public.activity_sessions where conversation_id=chat.id and user_id=owner_id
    and id<>target_id and (record->>'final_follow_up')::boolean) then
    raise exception 'This chat has reached its final check-in response' using errcode='PT409'; end if;
  changed:=saved.record||jsonb_build_object('follow_up_status','generating','follow_up_request_id',request_id,
    'follow_up_lease_until',now()+interval '180 seconds','follow_up_attempts',(saved.record->>'follow_up_attempts')::integer+1,
    'final_follow_up',is_final or (saved.record->>'final_follow_up')::boolean,'revision',saved.revision+1,'updated_at',now());
  update public.activity_sessions set revision=saved.revision+1,updated_at=now(),record=changed where id=target_id;
  if not replay then perform private.activity_store_receipt(owner_id,target_id,request_id,'follow_up',body->>'request_hash'); end if;
  return jsonb_build_object('session',changed,'claimed',true);
end; $$;


-- New API writers require this versioned RPC, so missing migration fails closed.
create function public.jp_commit_turn_v2(payload text, signature text)
returns jsonb language plpgsql security definer set search_path=private,public as $$
declare body jsonb := private.verified_payload(payload,signature,'commit_turn');
begin
  if not (body ? 'expected_incarnation_id') then
    raise exception 'Turn is missing its incarnation' using errcode='PT400';
  end if;
  return public.jp_commit_turn(payload,signature);
end; $$;
revoke all on function public.jp_commit_turn_v2(text,text) from public,anon,authenticated;
grant execute on function public.jp_commit_turn_v2(text,text) to authenticated;

-- Preview/production rollout must prove this additive boundary migration exists.
create function public.jp_readiness_v4(probe text,signature text)
returns jsonb language plpgsql stable security definer set search_path=private,public as $$
begin
  return public.jp_readiness_v3(probe,signature) || jsonb_build_object(
    'schema','sentinel-boundaries-1','repairs',
    case when to_regprocedure('public.jp_commit_turn_v2(text,text)') is not null
      and to_regprocedure('public.jp_finish_activity_followup_v2(text,text)') is not null
      then 'ready' else 'missing' end);
end; $$;
revoke all on function public.jp_readiness_v4(text,text) from public,anon,authenticated;
grant execute on function public.jp_readiness_v4(text,text) to anon,authenticated;

-- Delayed outcome generation is bound to the parent chat incarnation too.
create or replace function public.jp_finish_activity_followup_v1(payload text,signature text)
returns jsonb language plpgsql security definer set search_path=private,public as $$
declare
  body jsonb:=private.verified_payload(payload,signature,'finish_activity_follow_up'); owner_id uuid:=auth.uid();
  target_id uuid:=(body->>'session_id')::uuid; request_id uuid:=(body->>'request_id')::uuid;
  chat public.conversations%rowtype; saved public.activity_sessions%rowtype; changed jsonb; assistant jsonb:=body->'assistant_message';
  directive jsonb:=body->'directive';
begin
  select * into saved from public.activity_sessions where id=target_id and user_id=owner_id;
  if not found then raise exception 'Activity not found' using errcode='PT404'; end if;
  chat:=private.activity_locked_chat(saved.conversation_id,owner_id);
  select * into saved from public.activity_sessions where id=target_id and user_id=owner_id for update;
  if not found then raise exception 'Activity not found' using errcode='PT404'; end if;
  if (body ? 'expected_conversation_incarnation_id' and
      chat.record->>'incarnation_id' is distinct from body->>'expected_conversation_incarnation_id')
     or saved.created_at is distinct from (body->>'expected_session_created_at')::timestamptz then
    raise exception 'Activity changed' using errcode='PT409'; end if;
  if saved.record->>'follow_up_status'='ready' and (saved.record->>'follow_up_request_id')::uuid=request_id then return saved.record; end if;
  perform private.activity_guard(chat,(body->>'expected_conversation_revision')::integer,true,true);
  if saved.revision<>(body->>'expected_revision')::integer or saved.record->>'follow_up_status'<>'generating'
     or (saved.record->>'follow_up_request_id')::uuid is distinct from request_id then
    raise exception 'This follow-up changed' using errcode='PT409'; end if;
  if assistant->>'content' is null then
    changed:=saved.record||jsonb_build_object('follow_up_status','failed','follow_up_lease_until',null,
      'revision',saved.revision+1,'updated_at',now());
  else
    if assistant->>'role'<>'assistant' or (assistant->>'conversation_id')::uuid is distinct from chat.id
       or length(assistant->>'content') not between 1 and 1200 then
      raise exception 'Invalid activity follow-up' using errcode='PT400'; end if;
    if directive is not null and directive<>'null'::jsonb
       and (directive->>'card' is not null or directive->>'search_topic' is not null)
       and ((saved.record->>'final_follow_up')::boolean or chat.record->>'safety_mode'='support'
         or chat.record->>'interaction_preference'='listen'
         or (select count(*) from public.conversation_messages where user_id=owner_id and conversation_id=chat.id and role='user')>=20) then
      raise exception 'This check-in cannot offer another activity' using errcode='PT409'; end if;
    -- No user message is synthesized. The one outcome reply has a distinct UUID
    -- and cannot become another route around the ordinary 20-user-message cap.
    insert into public.conversation_messages(id,conversation_id,user_id,client_message_id,role,created_at,record)
    values((assistant->>'id')::uuid,chat.id,owner_id,null,'assistant',now(),assistant||jsonb_build_object('created_at',now()));
    update public.conversations set revision=revision+1,updated_at=now(),record=record||jsonb_build_object(
      'revision',revision+1,'updated_at',now())
      || case when directive is null or directive='null'::jsonb then '{}'::jsonb else jsonb_build_object(
        'card',null,'activity_card',case when directive->>'card' is null then 'null'::jsonb else directive->'card'||jsonb_build_object('offered_message_id',assistant->>'id') end,'activity_constraints',coalesce(nullif(directive->'constraints','null'::jsonb),record->'activity_constraints'),
        'activity_goal',coalesce(nullif(directive->'goal','null'::jsonb),record->'activity_goal'),
        'activity_search_topic',directive->'search_topic','activity_move',directive->'move',
        'ready_for_action',directive->>'card' is not null) end where id=chat.id;
    changed:=saved.record||jsonb_build_object('follow_up_status','ready','follow_up_reply',null,
      'follow_up_model_run',assistant->'model_run','follow_up_message_id',assistant->>'id','follow_up_lease_until',null,
      'revision',saved.revision+1,'updated_at',now());
  end if;
  update public.activity_sessions set revision=saved.revision+1,updated_at=now(),record=changed where id=target_id;
  return changed;
end; $$;


create function public.jp_finish_activity_followup_v2(payload text,signature text)
returns jsonb language plpgsql security definer set search_path=private,public as $$
declare body jsonb := private.verified_payload(payload,signature,'finish_activity_follow_up');
begin
  if not (body ? 'expected_conversation_incarnation_id') then
    raise exception 'Follow-up is missing its incarnation' using errcode='PT400';
  end if;
  return public.jp_finish_activity_followup_v1(payload,signature);
end; $$;
revoke all on function public.jp_finish_activity_followup_v2(text,text) from public,anon,authenticated;
grant execute on function public.jp_finish_activity_followup_v2(text,text) to authenticated;
