-- A schedule is not evidence that cleanup completed. This private singleton holds
-- only timestamps; neither journal content nor scheduler error messages belong here.
create table private.retention_state (
  singleton boolean primary key default true check (singleton),
  monitoring_started_at timestamptz not null default clock_timestamp(),
  last_success_at timestamptz
);
insert into private.retention_state (singleton) values (true);
revoke all on private.retention_state from public, anon, authenticated;

create or replace function public.jp_purge_expired_conversations()
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  result jsonb;
begin
  result := private.purge_conversations(interval '24 hours', null);
  -- Written after cleanup and in the same transaction. An error or rollback must
  -- leave the previous success unchanged. Request-time cleanup never writes this.
  update private.retention_state set last_success_at = clock_timestamp() where singleton;
  return result;
end;
$$;
revoke all on function public.jp_purge_expired_conversations() from public, anon, authenticated;
do $$
begin
  if exists (select 1 from pg_roles where rolname = 'service_role') then
    grant execute on function public.jp_purge_expired_conversations() to service_role;
  end if;
end
$$;

-- The clock argument is private and makes boundary checks reproducible. Public
-- callers see only the current observation through jp_readiness_v2.
create function private.retention_diagnostics(moment timestamptz default now())
returns jsonb
language plpgsql
stable
security definer
set search_path = private, public
as $$
declare
  started timestamptz;
  completed timestamptz;
  succeeded timestamptz;
  successful_start timestamptz;
  job_id bigint;
  job_schedule text;
  job_command text;
  last_status text;
  last_run_at timestamptz;
  job_state text := 'not_scheduled';
  execution_state text := 'missing';
  overdue text := 'unknown';
begin
  select monitoring_started_at, last_success_at into started, completed
    from private.retention_state where singleton;
  if to_regclass('cron.job') is not null then
    execute 'select jobid, schedule, command from cron.job where jobname = $1 and active
             order by jobid desc limit 1'
      into job_id, job_schedule, job_command using 'journalpulse-retention';
    if job_id is not null then
      job_state := 'scheduled';
      execution_state := 'unknown';
      if job_schedule = '*/15 * * * *'
         and trim(trailing ';' from btrim(job_command)) = 'select public.jp_purge_expired_conversations()'
         and to_regclass('cron.job_run_details') is not null and started is not null then
        execute 'select status, coalesce(end_time, start_time) from cron.job_run_details
                 where jobid = $1 order by start_time desc, runid desc limit 1'
          into last_status, last_run_at using job_id;
        -- A manual invocation also writes the completion timestamp. Only cron's
        -- completed run can prove that the scheduler itself succeeded. Old runs
        -- predating this migration cannot establish the new monitoring contract.
        execute 'select end_time, start_time from cron.job_run_details
                 where jobid = $1 and status = ''succeeded'' and end_time >= $2
                 order by end_time desc, runid desc limit 1'
          into succeeded, successful_start using job_id, started;
        -- Two missed 15-minute intervals are the proposed alert threshold. First
        -- installation gets the same 30-minute grace; it is still never_succeeded.
        overdue := (moment > coalesce(succeeded, started) + interval '30 minutes')::text;
        if last_status is not null and (last_run_at is null
             or last_status not in ('succeeded', 'failed', 'running', 'starting')) then
          execution_state := 'unknown';
        elsif succeeded is not null and (completed is null or completed < successful_start) then
          execution_state := 'unknown';
        elsif last_status = 'failed' and (succeeded is null or last_run_at > succeeded) then
          execution_state := 'failed';
        elsif succeeded is null then
          execution_state := 'never_succeeded';
        elsif overdue = 'true' then
          execution_state := 'overdue';
        else
          execution_state := 'healthy';
        end if;
      end if;
    end if;
  end if;
  return jsonb_build_object(
    'retention_job', job_state,
    'retention_state', execution_state,
    'retention_last_success', coalesce(succeeded::text, 'never'),
    'retention_last_run_status', case
      when last_status in ('succeeded', 'failed', 'running', 'starting') then last_status
      else 'unknown' end,
    'retention_overdue', overdue
  );
exception
  when others then
    -- Missing history privileges/columns must not masquerade as a healthy job.
    return jsonb_build_object('retention_job', 'unknown', 'retention_state', 'unknown',
      'retention_last_success', coalesce(succeeded::text, 'never'),
      'retention_last_run_status', 'unknown', 'retention_overdue', 'unknown');
end;
$$;
revoke all on function private.retention_diagnostics(timestamptz) from public, anon, authenticated;

-- Preserve jp_readiness and its phase-a-1 response for the existing live API.
-- The preview must prove its newer schema through a separate, additive RPC.
create function public.jp_readiness_v2(probe text, signature text)
returns jsonb
language plpgsql
stable
security definer
set search_path = private, public
as $$
declare
  signing text;
begin
  if not exists (select 1 from private.server_secrets where name = 'write_signing_key') then
    signing := 'missing';
  elsif probe like 'readiness:%' and private.signature_valid(probe, signature) then
    signing := 'valid';
  else
    signing := 'invalid';
  end if;
  return jsonb_build_object('schema', 'phase-a-2', 'signing', signing)
    || private.retention_diagnostics();
end;
$$;
revoke all on function public.jp_readiness_v2(text, text) from public, anon, authenticated;
grant execute on function public.jp_readiness_v2(text, text) to anon, authenticated;
