create table if not exists public.conversations (
  id uuid primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null,
  updated_at timestamptz not null,
  status text not null check (status in ('open', 'closed')),
  reflection_id uuid references public.reflections(id) on delete set null,
  record jsonb not null
);

create table if not exists public.conversation_messages (
  id uuid primary key,
  conversation_id uuid not null references public.conversations(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  client_message_id uuid,
  role text not null check (role in ('user', 'assistant')),
  created_at timestamptz not null,
  record jsonb not null
);

create unique index if not exists conversation_messages_client_id_idx
  on public.conversation_messages (conversation_id, client_message_id)
  where client_message_id is not null;

create index if not exists conversations_user_updated_idx
  on public.conversations (user_id, status, updated_at);

alter table public.conversations enable row level security;
alter table public.conversation_messages enable row level security;

create policy conversations_owner on public.conversations
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create policy conversation_messages_owner on public.conversation_messages
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create or replace function public.save_conversation_turn(payload jsonb)
returns jsonb
language plpgsql
security invoker
set search_path = public
as $$
declare
  owner_id uuid := auth.uid();
  conversation_id uuid := (payload->'conversation'->>'id')::uuid;
  client_message_id uuid := (payload->'user_message'->>'client_message_id')::uuid;
  existing_user jsonb;
  existing_assistant jsonb;
begin
  if owner_id is null then
    raise exception 'Authentication required';
  end if;
  if (payload->'conversation'->>'user_id')::uuid <> owner_id then
    raise exception 'Record owner does not match authenticated user';
  end if;
  if (payload->'user_message'->>'conversation_id')::uuid <> conversation_id
     or (payload->'assistant_message'->>'conversation_id')::uuid <> conversation_id then
    raise exception 'Message does not belong to this conversation';
  end if;

  select record into existing_user
  from public.conversation_messages
  where conversation_id = save_conversation_turn.conversation_id
    and client_message_id = save_conversation_turn.client_message_id
    and user_id = owner_id;
  if found then
    select record into existing_assistant
    from public.conversation_messages
    where conversation_id = save_conversation_turn.conversation_id
      and user_id = owner_id
      and role = 'assistant'
      and created_at >= (existing_user->>'created_at')::timestamptz
    order by created_at asc
    limit 1;
    return jsonb_build_object(
      'conversation', (
        select record from public.conversations
        where id = save_conversation_turn.conversation_id and user_id = owner_id
      ),
      'user_message', existing_user,
      'assistant_message', existing_assistant
    );
  end if;

  insert into public.conversation_messages (
    id, conversation_id, user_id, client_message_id, role, created_at, record
  ) values (
    (payload->'user_message'->>'id')::uuid,
    conversation_id,
    owner_id,
    client_message_id,
    payload->'user_message'->>'role',
    (payload->'user_message'->>'created_at')::timestamptz,
    payload->'user_message'
  );

  insert into public.conversation_messages (
    id, conversation_id, user_id, client_message_id, role, created_at, record
  ) values (
    (payload->'assistant_message'->>'id')::uuid,
    conversation_id,
    owner_id,
    nullif(payload->'assistant_message'->>'client_message_id', '')::uuid,
    payload->'assistant_message'->>'role',
    (payload->'assistant_message'->>'created_at')::timestamptz,
    payload->'assistant_message'
  );

  update public.conversations
  set updated_at = (payload->'conversation'->>'updated_at')::timestamptz,
      status = payload->'conversation'->>'status',
      reflection_id = nullif(payload->'conversation'->>'reflection_id', '')::uuid,
      record = payload->'conversation'
  where id = conversation_id and user_id = owner_id;

  return jsonb_build_object(
    'conversation', payload->'conversation',
    'user_message', payload->'user_message',
    'assistant_message', payload->'assistant_message'
  );
end;
$$;

create or replace function public.close_conversation(conversation_id uuid, purge boolean)
returns jsonb
language plpgsql
security invoker
set search_path = public
as $$
declare
  owner_id uuid := auth.uid();
  stored jsonb;
begin
  if owner_id is null then
    raise exception 'Authentication required';
  end if;

  update public.conversations
  set status = 'closed',
      updated_at = now(),
      record = jsonb_set(
        jsonb_set(record, '{status}', '"closed"'::jsonb),
        '{updated_at}',
        to_jsonb(now())
      )
  where id = conversation_id and user_id = owner_id
  returning record into stored;

  if stored is null then
    raise exception 'Conversation not found';
  end if;

  if purge then
    update public.conversation_messages
    set record = jsonb_set(record, '{content}', 'null'::jsonb)
    where conversation_id = close_conversation.conversation_id
      and user_id = owner_id;
  end if;

  return stored;
end;
$$;

create or replace function public.delete_my_journalpulse_data()
returns integer
language plpgsql
security invoker
set search_path = public
as $$
declare
  owner_id uuid := auth.uid();
  deleted_count integer := 0;
  affected integer := 0;
begin
  if owner_id is null then
    raise exception 'Authentication required';
  end if;

  delete from public.outcomes where user_id = owner_id;
  get diagnostics affected = row_count;
  deleted_count := deleted_count + affected;

  delete from public.conversation_messages where user_id = owner_id;
  get diagnostics affected = row_count;
  deleted_count := deleted_count + affected;

  delete from public.conversations where user_id = owner_id;
  get diagnostics affected = row_count;
  deleted_count := deleted_count + affected;

  delete from public.episodic_memories where user_id = owner_id;
  get diagnostics affected = row_count;
  deleted_count := deleted_count + affected;

  delete from public.reflections where user_id = owner_id;
  get diagnostics affected = row_count;
  deleted_count := deleted_count + affected;

  delete from public.consents where user_id = owner_id;
  get diagnostics affected = row_count;
  deleted_count := deleted_count + affected;

  delete from public.profiles where user_id = owner_id;
  get diagnostics affected = row_count;
  deleted_count := deleted_count + affected;

  return deleted_count;
end;
$$;

revoke all on function public.save_conversation_turn(jsonb) from public;
revoke all on function public.close_conversation(uuid, boolean) from public;
revoke all on function public.delete_my_journalpulse_data() from public;
grant execute on function public.save_conversation_turn(jsonb) to authenticated;
grant execute on function public.close_conversation(uuid, boolean) to authenticated;
grant execute on function public.delete_my_journalpulse_data() to authenticated;
