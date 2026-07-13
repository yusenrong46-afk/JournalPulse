create extension if not exists pgcrypto;
create extension if not exists vector;

create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  locale text not null default 'CA',
  created_at timestamptz not null default now()
);

create table if not exists public.consents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  llm_processing boolean not null default false,
  raw_text_retention boolean not null default false,
  memory_enabled boolean not null default false,
  accepted_at timestamptz not null default now(),
  revoked_at timestamptz
);

create table if not exists public.reflections (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null default now(),
  raw_text text,
  text_retained boolean not null default false,
  context jsonb not null default '{}'::jsonb,
  state jsonb not null,
  target jsonb not null,
  reflection jsonb not null,
  safety jsonb not null,
  decision jsonb not null,
  model_run jsonb,
  record jsonb not null
);

create table if not exists public.affective_observations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  reflection_id uuid references public.reflections(id) on delete cascade,
  observation_kind text not null check (observation_kind in ('inferred', 'self_report', 'target', 'outcome')),
  state jsonb not null,
  observed_at timestamptz not null default now()
);

create table if not exists public.intervention_catalog (
  id text primary key,
  payload jsonb not null,
  active boolean not null default true,
  reviewed_at timestamptz
);

create table if not exists public.policy_decisions (
  id uuid primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  reflection_id uuid references public.reflections(id) on delete cascade,
  policy_name text not null,
  policy_version text not null,
  action_id text not null,
  propensity double precision not null check (propensity > 0 and propensity <= 1),
  available_actions jsonb not null,
  context_snapshot jsonb not null,
  created_at timestamptz not null default now()
);

create table if not exists public.outcomes (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  decision_id uuid not null references public.policy_decisions(id) on delete cascade,
  created_at timestamptz not null default now(),
  record jsonb not null
);

create table if not exists public.episodic_memories (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  reflection_id uuid references public.reflections(id) on delete cascade,
  summary text not null,
  links jsonb not null default '[]'::jsonb,
  embedding vector(768),
  valid boolean not null default true,
  created_at timestamptz not null default now(),
  invalidated_at timestamptz
);

create table if not exists public.model_runs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  reflection_id uuid references public.reflections(id) on delete cascade,
  model text not null,
  provider text not null,
  latency_ms integer not null,
  prompt_tokens integer,
  completion_tokens integer,
  schema_valid boolean not null,
  fallback_reason text,
  created_at timestamptz not null default now()
);

create table if not exists public.safety_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  reflection_id uuid references public.reflections(id) on delete cascade,
  mode text not null,
  reason_codes jsonb not null,
  locale text not null,
  created_at timestamptz not null default now()
);

alter table public.profiles enable row level security;
alter table public.consents enable row level security;
alter table public.reflections enable row level security;
alter table public.affective_observations enable row level security;
alter table public.policy_decisions enable row level security;
alter table public.outcomes enable row level security;
alter table public.episodic_memories enable row level security;
alter table public.model_runs enable row level security;
alter table public.safety_events enable row level security;

do $$
declare table_name text;
begin
  foreach table_name in array array[
    'profiles', 'consents', 'reflections', 'affective_observations',
    'policy_decisions', 'outcomes', 'episodic_memories', 'model_runs', 'safety_events'
  ] loop
    execute format(
      'create policy %I on public.%I for all using (auth.uid() = user_id) with check (auth.uid() = user_id)',
      table_name || '_owner', table_name
    );
  end loop;
end $$;

create index if not exists reflections_user_created_idx on public.reflections(user_id, created_at desc);
create index if not exists decisions_user_created_idx on public.policy_decisions(user_id, created_at desc);
create index if not exists outcomes_user_created_idx on public.outcomes(user_id, created_at desc);
create index if not exists memories_user_created_idx on public.episodic_memories(user_id, created_at desc);

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

revoke all on function public.delete_my_journalpulse_data() from public;
grant execute on function public.delete_my_journalpulse_data() to authenticated;
