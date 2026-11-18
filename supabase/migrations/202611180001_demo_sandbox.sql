-- Portfolio demo sandbox: erase anonymous (demo) accounts after 24 hours.
--
-- Demo identities are Supabase anonymous users (auth.users.is_anonymous = true). They
-- carry only fictional seed data, and the API never sends their text to a model or
-- search provider (src/journalpulse/demo.py, AuthContext.demo).
--
-- Erasure goes through the existing account erasure path: for each expired anonymous
-- user this job runs public.delete_my_journalpulse_data() under that user's subject
-- claim, exactly as the user's own "delete my data" request would. It then removes the
-- anonymous auth identity, which cascades the private retirement markers. If the auth
-- row cannot be removed (for example a platform permission change), the user's data
-- is still erased and the job carries on; later runs erase again, which is harmless.
--
-- This file adds objects only. It changes no earlier migration, table or policy.

create or replace function private.jp_erase_expired_demo_accounts(
  max_age interval default interval '24 hours',
  batch_size integer default 500
)
returns integer
language plpgsql
volatile
security definer
set search_path = private, public
as $$
declare
  demo_user uuid;
  erased integer := 0;
begin
  if max_age < interval '1 hour' then
    raise exception 'Demo accounts are kept for at least one hour' using errcode = '22023';
  end if;
  for demo_user in
    select id from auth.users
    where is_anonymous is true and created_at < now() - max_age
    order by created_at
    limit greatest(batch_size, 1)
  loop
    -- Act as that user for the existing erasure function, then drop the claim again.
    perform set_config('request.jwt.claims', jsonb_build_object('sub', demo_user, 'role', 'authenticated')::text, true);
    perform public.delete_my_journalpulse_data();
    perform set_config('request.jwt.claims', '', true);
    begin
      delete from auth.users where id = demo_user and is_anonymous is true;
    exception
      when insufficient_privilege then
        raise notice 'demo identity % kept: auth row removal not permitted; its data was erased', demo_user;
    end;
    erased := erased + 1;
  end loop;
  return erased;
end;
$$;

revoke all on function private.jp_erase_expired_demo_accounts(interval, integer) from public, anon, authenticated;

do $$
begin
  if exists (select 1 from pg_available_extensions where name = 'pg_cron') then
    create extension if not exists pg_cron;
    perform cron.schedule(
      'journalpulse-demo-erasure',
      '17 * * * *',
      'select private.jp_erase_expired_demo_accounts()'
    );
  end if;
exception
  when others then
    raise notice 'pg_cron is unavailable (%); demo accounts are not erased automatically', sqlerrm;
end
$$;
