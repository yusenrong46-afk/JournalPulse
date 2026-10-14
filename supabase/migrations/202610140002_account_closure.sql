-- Content-free recovery record deliberately survives removal of auth.users.
-- Receipt hashes permit a short, action-only retry after Auth invalidates the session.
create table private.account_closures (
  user_id uuid primary key,
  request_id uuid not null unique,
  receipt_hash text not null check (receipt_hash ~ '^[a-f0-9]{64}$'),
  requested_at timestamptz not null default now(),
  receipt_expires_at timestamptz not null,
  completed_at timestamptz
);
revoke all on private.account_closures from public,anon,authenticated;

create function private.reject_closing_account()
returns trigger language plpgsql security definer set search_path=private,public as $$
begin
  perform pg_advisory_xact_lock_shared(hashtextextended('account-erasure:' || new.user_id::text,0));
  -- Auth deletion can cascade SET NULL updates before deleting dependent rows.
  if tg_op='UPDATE' and pg_trigger_depth()>1
      and not exists(select 1 from auth.users where id=new.user_id) then return new; end if;
  if exists(select 1 from private.account_closures where user_id=new.user_id) then
    raise exception 'Account closure is pending or complete' using errcode='PT409';
  end if;
  return new;
end; $$;
revoke all on function private.reject_closing_account() from public,anon,authenticated;

do $$ declare t record; begin
  for t in select table_name from information_schema.columns
    where table_schema='public' and column_name='user_id' and udt_name='uuid'
      and table_name in (select tablename from pg_tables where schemaname='public')
  loop
    execute format('create trigger account_closure_write_guard before insert or update on public.%I '
      'for each row execute function private.reject_closing_account()',t.table_name);
  end loop;
end; $$;

create function public.jp_begin_account_closure(payload text,signature text)
returns jsonb language plpgsql security definer set search_path=private,public as $$
declare owner_id uuid:=auth.uid(); body jsonb; operation uuid; proof text; saved private.account_closures;
begin
  if owner_id is null then raise exception 'Authentication required' using errcode='PT401'; end if;
  -- Take the exclusive lock before verified_payload takes its shared lock: no lock-upgrade race.
  perform pg_advisory_xact_lock(hashtextextended('account-erasure:' || owner_id::text,0));
  body:=private.verified_payload(payload,signature,'begin_account_closure');
  operation:=(body->>'request_id')::uuid; proof:=body->>'receipt_hash';
  if operation is null or proof is null or proof !~ '^[a-f0-9]{64}$' then
    raise exception 'Invalid closure request' using errcode='PT400';
  end if;
  if not exists(select 1 from auth.users where id=owner_id) then
    raise exception 'Authentication required' using errcode='PT401';
  end if;
  select * into saved from private.account_closures where user_id=owner_id;
  if found and saved.request_id=operation and saved.receipt_hash=proof then
    if saved.receipt_expires_at<=now() then
      update private.account_closures set receipt_expires_at=now()+interval '15 minutes' where user_id=owner_id;
    end if;
    return jsonb_build_object('user_id',owner_id,'request_id',operation,
      'state',case when saved.completed_at is null then 'pending' else 'complete' end);
  end if;
  insert into private.account_closures(user_id,request_id,receipt_hash,receipt_expires_at)
    values(owner_id,operation,proof,now()+interval '15 minutes')
  on conflict(user_id) do update set request_id=excluded.request_id,receipt_hash=excluded.receipt_hash,
    receipt_expires_at=excluded.receipt_expires_at;
  return jsonb_build_object('user_id',owner_id,'request_id',operation,'state','pending');
end; $$;
revoke all on function public.jp_begin_account_closure(text,text) from public,anon,authenticated;
grant execute on function public.jp_begin_account_closure(text,text) to authenticated;

create function public.jp_resolve_account_closure(operation_id uuid,proof_hash text)
returns jsonb language sql stable security definer set search_path=private,public as $$
  select jsonb_build_object('user_id',user_id,'request_id',request_id,
    'state',case when completed_at is null then 'pending' else 'complete' end)
  from private.account_closures where request_id=operation_id and receipt_hash=proof_hash
    and receipt_expires_at>statement_timestamp();
$$;
revoke all on function public.jp_resolve_account_closure(uuid,text) from public,anon,authenticated;
grant execute on function public.jp_resolve_account_closure(uuid,text) to service_role;

create function public.jp_finish_account_closure(operation_id uuid,proof_hash text)
returns jsonb language plpgsql security definer set search_path=private,public as $$
declare saved private.account_closures; t record; remains boolean;
begin
  select * into saved from private.account_closures where request_id=operation_id and receipt_hash=proof_hash;
  if not found then raise exception 'Invalid closure receipt' using errcode='PT403'; end if;
  perform pg_advisory_xact_lock(hashtextextended('account-erasure:' || saved.user_id::text,0));
  if exists(select 1 from auth.users where id=saved.user_id) then
    raise exception 'Authentication identity still exists' using errcode='PT409';
  end if;
  for t in select table_schema,table_name from information_schema.columns
    where table_schema in ('public','private') and column_name='user_id' and udt_name='uuid'
      and table_name<>'account_closures'
  loop
    execute format('select exists(select 1 from %I.%I where user_id=$1)',t.table_schema,t.table_name)
      into remains using saved.user_id;
    if remains then raise exception 'Dependent account records remain' using errcode='PT409'; end if;
  end loop;
  update private.account_closures set completed_at=coalesce(completed_at,now()) where user_id=saved.user_id;
  return jsonb_build_object('state','complete');
end; $$;
revoke all on function public.jp_finish_account_closure(uuid,text) from public,anon,authenticated;
grant execute on function public.jp_finish_account_closure(uuid,text) to service_role;

-- Stop new model-backed mutations early, while triggers fence already-running writers.
create or replace function public.jp_account_data_revision()
returns bigint language plpgsql stable security definer set search_path=private,public as $$
declare owner_id uuid:=auth.uid(); result bigint;
begin
  if owner_id is null then raise exception 'Authentication required' using errcode='PT401'; end if;
  if exists(select 1 from private.account_closures where user_id=owner_id) then
    raise exception 'Account closure is pending or complete' using errcode='PT409';
  end if;
  select revision into result from private.account_erasure_revisions where user_id=owner_id;
  return coalesce(result,0);
end; $$;

create function public.jp_readiness_v6(probe text,signature text)
returns jsonb language plpgsql stable security definer set search_path=private,public as $$
declare guarded boolean;
begin
  guarded:=to_regclass('private.account_closures') is not null and not exists(
    select 1 from information_schema.columns c where c.table_schema='public'
      and c.column_name='user_id' and c.udt_name='uuid'
      and c.table_name in (select tablename from pg_tables where schemaname='public') and not exists(
        select 1 from pg_trigger t where t.tgrelid=format('public.%I',c.table_name)::regclass
          and t.tgname='account_closure_write_guard' and not t.tgisinternal));
  return public.jp_readiness_v5(probe,signature) || jsonb_build_object('schema','account-closure-1',
    'account_closure',case when guarded then 'ready' else 'missing' end);
end; $$;
revoke all on function public.jp_readiness_v6(text,text) from public,anon,authenticated;
grant execute on function public.jp_readiness_v6(text,text) to anon,authenticated;
notify pgrst,'reload schema';
