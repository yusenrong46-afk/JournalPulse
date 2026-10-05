-- An explicit conversation choice must outlive a browser tab and invalidate work
-- computed from the previous revision. Receipts contain command metadata, not text.
create table public.conversation_preference_requests (
  id uuid not null,
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  preference text not null check (preference in ('listen', 'act')),
  expected_revision integer not null check (expected_revision >= 0),
  created_at timestamptz not null default now(),
  primary key (conversation_id, id)
);
create index conversation_preference_requests_owner
  on public.conversation_preference_requests (user_id, created_at, id);
alter table public.conversation_preference_requests enable row level security;
create policy preference_requests_owner on public.conversation_preference_requests
  for select using (auth.uid() = user_id);
revoke all on public.conversation_preference_requests from public, anon, authenticated;
grant select on public.conversation_preference_requests to authenticated;

create or replace function public.jp_change_preference(payload text, signature text)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  body jsonb := private.verified_payload(payload, signature, 'change_preference');
  owner_id uuid := auth.uid();
  target_id uuid := (body->>'conversation_id')::uuid;
  request_id uuid := (body->>'client_request_id')::uuid;
  wanted text := body->>'preference';
  expected integer := (body->>'expected_revision')::integer;
  current_row public.conversations%rowtype;
  receipt public.conversation_preference_requests%rowtype;
  changed jsonb;
  moment timestamptz := now();
begin
  if request_id is null or expected is null or expected < 0
     or wanted is null or wanted not in ('listen', 'act') then
    raise exception 'Invalid preference command' using errcode = 'PT400';
  end if;
  select * into current_row from public.conversations
  where id = target_id and user_id = owner_id for update;
  if not found then
    raise exception 'Conversation not found' using errcode = 'PT404';
  end if;
  -- Check receipts before the revision: a retry is not a new command. Returning
  -- current state prevents replaying Listen from undoing a later Act choice.
  select * into receipt from public.conversation_preference_requests
  where conversation_id = target_id and id = request_id;
  if found then
    if receipt.preference <> wanted or receipt.expected_revision <> expected then
      raise exception 'Preference request ID is already in use' using errcode = 'PT409';
    end if;
    return current_row.record || jsonb_build_object('revision', current_row.revision);
  end if;
  if current_row.status <> 'open' then
    raise exception 'Conversation is closed' using errcode = 'PT409';
  end if;
  if current_row.revision <> expected then
    raise exception 'Conversation changed' using errcode = 'PT409';
  end if;
  if current_row.record->>'safety_mode' = 'support' then
    raise exception 'Support mode cannot be changed' using errcode = 'PT409';
  end if;
  changed := current_row.record || jsonb_build_object(
    'interaction_preference', wanted, 'card', null, 'ready_for_action', wanted = 'act',
    'revision', current_row.revision + 1, 'updated_at', moment
  );
  update public.conversations set record = changed,
    revision = current_row.revision + 1, updated_at = moment where id = target_id;
  insert into public.conversation_preference_requests
    (id, conversation_id, user_id, preference, expected_revision, created_at)
  values (request_id, target_id, owner_id, wanted, expected, moment);
  return changed;
end;
$$;
revoke all on function public.jp_change_preference(text, text) from public, anon, authenticated;
grant execute on function public.jp_change_preference(text, text) to authenticated;

-- Replace lifecycle functions additively; the earlier migration stays immutable.
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

