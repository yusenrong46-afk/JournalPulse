-- Standalone saved writing. Earlier deployed migrations and readiness versions stay
-- unchanged: journal support expands the schema without changing Phase A's contract.
create table public.journal_entries (
  id uuid primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null,
  record jsonb not null,
  unique (user_id, id),
  unique (user_id, id, created_at),
  check (jsonb_typeof(record->'text') = 'string'),
  check (length(record->>'text') between 1 and 5000),
  check ((record->>'text') !~ '^[[:space:]]*$'),
  check ((record->>'id')::uuid = id),
  check ((record->>'user_id')::uuid = user_id),
  check ((record->>'created_at')::timestamptz = created_at)
);
create index journal_entries_owner_created_idx
  on public.journal_entries (user_id, created_at desc, id desc);
alter table public.journal_entries enable row level security;
create policy journal_entries_owner on public.journal_entries
  for select using (auth.uid() = user_id);
-- Even text-only writes use the API's signed owner-bound envelope. There is no
-- UPDATE path, and direct DELETE cannot bypass the derived-content lifecycle.
revoke all on public.journal_entries from public, anon, authenticated;
grant select on public.journal_entries to authenticated;

alter table public.conversations add column source_entry_id uuid;
alter table public.conversations add column source_entry_created_at timestamptz;
alter table public.conversations add constraint conversations_owned_journal_source_fk
  foreign key (user_id, source_entry_id, source_entry_created_at)
  references public.journal_entries (user_id, id, created_at) on delete cascade;
create index conversations_journal_source_idx
  on public.conversations (user_id, source_entry_id) where source_entry_id is not null;

-- Link identity is canonical database state. An old writer may omit this new field,
-- but cannot detach or rewrite it and thereby escape source deletion.
create or replace function private.guard_conversation_journal_source()
returns trigger
language plpgsql
security definer
set search_path = private, public
as $$
declare
  requested uuid := (new.record->>'source_entry_id')::uuid;
  source_created timestamptz := (new.record->>'source_entry_created_at')::timestamptz;
begin
  if tg_op = 'INSERT' then
    new.source_entry_id := requested;
    new.source_entry_created_at := source_created;
    if requested is not null then
      perform 1 from public.journal_entries
      where id = requested and user_id = new.user_id and created_at = source_created for key share;
      if not found then
        raise exception 'Journal entry not found' using errcode = 'PT404';
      end if;
    end if;
  else
    if new.user_id is distinct from old.user_id then
      raise exception 'Record owner cannot be changed' using errcode = 'PT403';
    end if;
    if (requested is not null and requested is distinct from old.source_entry_id)
       or (source_created is not null and source_created is distinct from old.source_entry_created_at)
       or new.source_entry_id is distinct from old.source_entry_id
       or new.source_entry_created_at is distinct from old.source_entry_created_at then
      raise exception 'Journal source cannot be changed' using errcode = 'PT409';
    end if;
    new.source_entry_id := old.source_entry_id;
    new.source_entry_created_at := old.source_entry_created_at;
  end if;
  new.record := new.record || jsonb_build_object(
    'source_entry_id', new.source_entry_id, 'source_entry_created_at', new.source_entry_created_at);
  return new;
end;
$$;
create trigger conversation_journal_source_guard
  before insert or update on public.conversations
  for each row execute function private.guard_conversation_journal_source();

create or replace function public.jp_save_journal_entry(payload text, signature text)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  body jsonb := private.verified_payload(payload, signature, 'save_journal_entry');
  owner_id uuid := auth.uid();
  entry jsonb := body->'entry';
  target_id uuid := (entry->>'id')::uuid;
  existing public.journal_entries%rowtype;
begin
  if (entry->>'user_id')::uuid is distinct from owner_id then
    raise exception 'Record owner does not match authenticated user' using errcode = 'PT403';
  end if;
  if target_id is null or jsonb_typeof(entry->'text') is distinct from 'string'
     or length(entry->>'text') not between 1 and 5000
     or (entry->>'text') ~ '^[[:space:]]*$'
     or entry->>'created_at' is null then
    raise exception 'Invalid journal entry' using errcode = 'PT400';
  end if;
  -- Serialize by UUID across owners as well, so simultaneous reuse has exactly the
  -- same conflict rule as a later retry and never overwrites another person's text.
  perform pg_advisory_xact_lock(hashtextextended('journal-entry:' || target_id::text, 0));
  select * into existing from public.journal_entries where id = target_id;
  if found then
    if existing.user_id <> owner_id or existing.record->>'text' is distinct from entry->>'text' then
      raise exception 'Journal request ID is already in use' using errcode = 'PT409';
    end if;
    -- The first server timestamp wins; a retry has a fresh envelope, not a new entry.
    return existing.record;
  end if;
  insert into public.journal_entries (id, user_id, created_at, record)
  values (target_id, owner_id, (entry->>'created_at')::timestamptz, entry);
  return entry;
end;
$$;
revoke all on function public.jp_save_journal_entry(text, text) from public, anon, authenticated;
grant execute on function public.jp_save_journal_entry(text, text) to authenticated;

-- Save summaries and accepted action records can contain source-derived content.
-- Delete those bundles before their conversations; reflection foreign keys then
-- cascade to outcomes, model runs, safety events, and affective observations.
-- A trigger covers RPC, account deletion, and auth-user cascades alike.
create or replace function private.delete_journal_derived_content()
returns trigger
language plpgsql
security definer
set search_path = private, public
as $$
declare
  linked public.conversations%rowtype;
begin
  for linked in select * from public.conversations
    where user_id = old.user_id and source_entry_id = old.id order by id for update
  loop
    delete from public.reflections
    where user_id = old.user_id
      and (conversation_id = linked.id or id = linked.reflection_id);
    -- Removing this row under the same transaction also removes messages and
    -- preference receipts. A pending turn subsequently finds no row to commit.
    delete from public.conversations where id = linked.id and user_id = old.user_id;
  end loop;
  return old;
end;
$$;
create trigger journal_entries_delete_derived_content
  before delete on public.journal_entries
  for each row execute function private.delete_journal_derived_content();

-- Existing RLS grants also allow a person to delete their conversation directly.
-- Source-derived accepted reflections must follow that route too. AFTER DELETE
-- avoids updating the conversation being deleted via its reflection SET NULL FK.
create or replace function private.delete_linked_conversation_reflection()
returns trigger
language plpgsql
security definer
set search_path = private, public
as $$
begin
  if old.source_entry_id is not null then
    delete from public.reflections where user_id = old.user_id
      and (id = old.reflection_id or conversation_id = old.id);
  end if;
  return old;
end;
$$;
create trigger conversation_journal_delete_reflection
  after delete on public.conversations
  for each row execute function private.delete_linked_conversation_reflection();

create or replace function public.jp_delete_journal_entry(payload text, signature text)
returns boolean
language plpgsql
security definer
set search_path = private, public
as $$
declare
  body jsonb := private.verified_payload(payload, signature, 'delete_journal_entry');
  owner_id uuid := auth.uid();
  target_id uuid := (body->>'entry_id')::uuid;
begin
  perform 1 from public.journal_entries where id = target_id and user_id = owner_id for update;
  if not found then
    return false;
  end if;
  delete from public.journal_entries where id = target_id and user_id = owner_id;
  return true;
end;
$$;
revoke all on function public.jp_delete_journal_entry(text, text) from public, anon, authenticated;
grant execute on function public.jp_delete_journal_entry(text, text) to authenticated;

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
       or existing.source_entry_created_at is distinct from source_created then
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
revoke all on function public.jp_create_conversation(text, text) from public, anon, authenticated;
grant execute on function public.jp_create_conversation(text, text) to authenticated;

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
  -- Children first so the count covers each journal row exactly once, including
  -- source-linked chats, their receipts, and the standalone writing itself.
  foreach table_name in array array[
    'outcomes', 'affective_observations', 'policy_decisions', 'model_runs', 'safety_events',
    'conversation_messages', 'conversation_preference_requests', 'episodic_memories',
    'reflections', 'conversations', 'journal_entries', 'consents', 'profiles'
  ] loop
    execute format('delete from public.%I where user_id = $1', table_name) using owner_id;
    get diagnostics affected = row_count;
    deleted_count := deleted_count + affected;
  end loop;
  -- Keep content-free rate-limit counters: account deletion must not buy more paid calls.
  return deleted_count;
end;
$$;
revoke all on function public.delete_my_journalpulse_data() from public, anon, authenticated;
grant execute on function public.delete_my_journalpulse_data() to authenticated;
