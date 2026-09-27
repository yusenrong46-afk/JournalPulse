-- save_conversation_turn and close_conversation never completed on PostgreSQL.
-- Their variables are named conversation_id and client_message_id, which are also
-- columns, so the unqualified references abort with "column reference is ambiguous"
-- before a turn can be stored. Closing with text removal (the default) hits the
-- same error, so the privacy-preserving close could not run.
-- Argument names stay the same: PostgREST matches the JSON body to those names.

create or replace function public.save_conversation_turn(payload jsonb)
returns jsonb
language plpgsql
security invoker
set search_path = public
as $$
declare
  owner_id uuid := auth.uid();
  target_conversation_id uuid := (payload->'conversation'->>'id')::uuid;
  target_client_message_id uuid := (payload->'user_message'->>'client_message_id')::uuid;
  existing_user jsonb;
  existing_assistant jsonb;
begin
  if owner_id is null then
    raise exception 'Authentication required';
  end if;
  if (payload->'conversation'->>'user_id')::uuid <> owner_id then
    raise exception 'Record owner does not match authenticated user';
  end if;
  if (payload->'user_message'->>'conversation_id')::uuid <> target_conversation_id
     or (payload->'assistant_message'->>'conversation_id')::uuid <> target_conversation_id then
    raise exception 'Message does not belong to this conversation';
  end if;

  -- RLS hides other people's rows, so a missing row and someone else's row
  -- look the same here. Either way the turn must not be stored.
  if not exists (
    select 1
    from public.conversations
    where public.conversations.id = target_conversation_id
      and public.conversations.user_id = owner_id
  ) then
    raise exception 'Conversation not found';
  end if;

  select messages.record into existing_user
  from public.conversation_messages as messages
  where messages.conversation_id = target_conversation_id
    and messages.client_message_id = target_client_message_id
    and messages.user_id = owner_id;
  if found then
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
        select conversations.record
        from public.conversations
        where conversations.id = target_conversation_id
          and conversations.user_id = owner_id
      ),
      'user_message', existing_user,
      'assistant_message', existing_assistant
    );
  end if;

  insert into public.conversation_messages (
    id, conversation_id, user_id, client_message_id, role, created_at, record
  ) values (
    (payload->'user_message'->>'id')::uuid,
    target_conversation_id,
    owner_id,
    target_client_message_id,
    payload->'user_message'->>'role',
    (payload->'user_message'->>'created_at')::timestamptz,
    payload->'user_message'
  );

  insert into public.conversation_messages (
    id, conversation_id, user_id, client_message_id, role, created_at, record
  ) values (
    (payload->'assistant_message'->>'id')::uuid,
    target_conversation_id,
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
  where public.conversations.id = target_conversation_id
    and public.conversations.user_id = owner_id;
  if not found then
    raise exception 'Conversation not found';
  end if;

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
        jsonb_set(public.conversations.record, '{status}', '"closed"'::jsonb),
        '{updated_at}',
        to_jsonb(now())
      )
  where public.conversations.id = close_conversation.conversation_id
    and public.conversations.user_id = owner_id
  returning public.conversations.record into stored;

  if stored is null then
    raise exception 'Conversation not found';
  end if;

  if purge then
    update public.conversation_messages
    set record = jsonb_set(public.conversation_messages.record, '{content}', 'null'::jsonb)
    where public.conversation_messages.conversation_id = close_conversation.conversation_id
      and public.conversation_messages.user_id = owner_id;
  end if;

  return stored;
end;
$$;

revoke all on function public.save_conversation_turn(jsonb) from public;
revoke all on function public.close_conversation(uuid, boolean) from public;
grant execute on function public.save_conversation_turn(jsonb) to authenticated;
grant execute on function public.close_conversation(uuid, boolean) to authenticated;

-- The shared catalog is unused by the API. Without RLS, Supabase's default
-- grants would let every signed-in user rewrite it.
alter table public.intervention_catalog enable row level security;
revoke all on table public.intervention_catalog from anon, authenticated;
