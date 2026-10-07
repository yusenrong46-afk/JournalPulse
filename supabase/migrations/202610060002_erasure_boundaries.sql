-- Deleted UUIDs name one retired logical creation. Keep only owner/object identity,
-- never writing, summaries, report text, text hashes, model output, or timestamps.
-- These private markers and account revision deliberately survive data erasure;
-- deleting the auth identity removes them. Existing RPC/readiness contracts remain.
create table private.deleted_object_ids (
  object_kind text not null check (object_kind in ('journal_entries','conversations','activity_sessions','reflections')),
  object_id uuid not null,
  user_id uuid not null references auth.users(id) on delete cascade,
  primary key(object_kind,object_id)
);
create index deleted_object_ids_owner on private.deleted_object_ids(user_id);
create table private.account_erasure_revisions (
  user_id uuid primary key references auth.users(id) on delete cascade,
  revision bigint not null check(revision>=0)
);
revoke all on private.deleted_object_ids, private.account_erasure_revisions from public,anon,authenticated;

create function private.guard_logical_creation_identity()
returns trigger language plpgsql volatile security definer set search_path=private,public as $$
begin
  -- Serialize deletion with a new insert of this UUID, including direct RLS deletes.
  perform pg_advisory_xact_lock(hashtextextended(
    'retired-object:' || tg_table_name || ':' || coalesce(new.id,old.id)::text, 0));
  if tg_op='DELETE' then
    -- Auth-user deletion is a separate privileged cascade: there is no remaining
    -- account to replay under, and inserting its marker would violate the FK.
    if exists(select 1 from auth.users where id=old.user_id) then
      insert into private.deleted_object_ids(object_kind,object_id,user_id)
      values(tg_table_name,old.id,old.user_id) on conflict do nothing;
    end if;
    return old;
  end if;
  if exists(select 1 from private.deleted_object_ids
            where object_kind=tg_table_name and object_id=new.id) then
    raise exception 'Creation request ID was deleted; use a new request ID' using errcode='PT409';
  end if;
  return new;
end; $$;

-- Cascades use the same guards, covering source deletion, chat deletion, account
-- deletion, and the legacy direct RLS delete paths without duplicating endpoints.
do $$ declare table_name text; begin
  foreach table_name in array array['journal_entries','conversations','activity_sessions','reflections'] loop
    execute format('create trigger logical_creation_identity before insert or delete on public.%I '
      || 'for each row execute function private.guard_logical_creation_identity()', table_name);
  end loop;
end; $$;

create function public.jp_account_data_revision()
returns bigint language plpgsql volatile security definer set search_path=private,public as $$
declare owner_id uuid:=auth.uid(); result bigint;
begin
  if owner_id is null then raise exception 'Authentication required' using errcode='PT401'; end if;
  perform pg_advisory_xact_lock_shared(hashtextextended('account-erasure:' || owner_id::text,0));
  select revision into result from private.account_erasure_revisions where user_id=owner_id;
  return coalesce(result,0);
end; $$;
revoke all on function public.jp_account_data_revision() from public,anon,authenticated;
grant execute on function public.jp_account_data_revision() to authenticated;

create or replace function private.verified_payload(payload text, signature text, purpose text)
returns jsonb
language plpgsql
volatile
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
  -- Every signed operation obtains the account lock before any object row lock.
  -- Shared writers retain concurrency; whole-account erasure is exclusive.
  perform pg_advisory_xact_lock_shared(hashtextextended('account-erasure:' || owner_id::text, 0));
  -- Older deployed servers omit this field during the additive migration rollout.
  -- New servers preserve the client's pre-submission revision across authentication
  -- and generation; they never replace a stale revision with a fresh database read.
  if body ? 'expected_erasure_revision' and
     (jsonb_typeof(body->'expected_erasure_revision') is distinct from 'number'
      or (body->>'expected_erasure_revision')::bigint is distinct from
         coalesce((select revision from private.account_erasure_revisions where user_id=owner_id),0)) then
    raise exception 'Account data was erased' using errcode='PT409';
  end if;
  return body;
end;
$$;

create or replace function public.delete_my_journalpulse_data()
returns integer language plpgsql security definer set search_path=private,public as $$
declare owner_id uuid:=auth.uid(); deleted_count integer:=0; affected integer; table_name text;
begin
  if owner_id is null then raise exception 'Authentication required' using errcode='PT401'; end if;
  perform pg_advisory_xact_lock(hashtextextended('account-erasure:' || owner_id::text,0));
  insert into private.account_erasure_revisions(user_id,revision) values(owner_id,1)
  on conflict(user_id) do update set revision=private.account_erasure_revisions.revision+1;
  foreach table_name in array array[
    'activity_receipts','activity_sessions','outcomes','affective_observations','policy_decisions','model_runs','safety_events',
    'conversation_messages','conversation_preference_requests','episodic_memories','reflections','conversations',
    'journal_entries','consents','profiles'
  ] loop
    execute format('delete from public.%I where user_id=$1',table_name) using owner_id;
    get diagnostics affected=row_count; deleted_count:=deleted_count+affected;
  end loop;
  return deleted_count;
end; $$;

-- Versioned readiness lets the new API require the new guard while v1-v4 keep
-- their old response contracts for the live server during database-first rollout.
create function public.jp_readiness_v5(probe text,signature text)
returns jsonb language plpgsql stable security definer set search_path=private,public as $$
declare guards_ready boolean;
begin
  guards_ready:=to_regclass('private.deleted_object_ids') is not null
    and to_regprocedure('public.jp_account_data_revision()') is not null
    and (select count(*) from pg_trigger where tgname='logical_creation_identity' and not tgisinternal)=4;
  return public.jp_readiness_v4(probe,signature) || jsonb_build_object(
    'schema','erasure-boundaries-1','erasure',case when guards_ready then 'ready' else 'missing' end);
end; $$;
revoke all on function public.jp_readiness_v5(text,text) from public,anon,authenticated;
grant execute on function public.jp_readiness_v5(text,text) to anon,authenticated;
notify pgrst, 'reload schema';
