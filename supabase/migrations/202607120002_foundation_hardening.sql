create or replace function public.save_reflection_bundle(payload jsonb)
returns jsonb
language plpgsql
security invoker
set search_path = public
as $$
declare
  owner_id uuid := auth.uid();
  reflection_id uuid := (payload->>'id')::uuid;
  decision_payload jsonb := payload->'decision';
  model_payload jsonb := payload->'model_run';
  safety_payload jsonb := payload->'safety';
  existing_record jsonb;
begin
  if owner_id is null then
    raise exception 'Authentication required';
  end if;
  if (payload->>'user_id')::uuid <> owner_id then
    raise exception 'Record owner does not match authenticated user';
  end if;

  select record into existing_record
  from public.reflections
  where id = reflection_id and user_id = owner_id;
  if found then
    return existing_record;
  end if;

  insert into public.reflections (
    id, user_id, created_at, raw_text, text_retained, context, state, target,
    reflection, safety, decision, model_run, record
  ) values (
    reflection_id,
    owner_id,
    (payload->>'created_at')::timestamptz,
    payload->>'text',
    coalesce((payload->>'text_retained')::boolean, false),
    coalesce(payload->'context', '{}'::jsonb),
    payload->'state',
    payload->'target',
    payload->'reflection',
    safety_payload,
    decision_payload,
    model_payload,
    payload
  );

  insert into public.affective_observations (
    user_id, reflection_id, observation_kind, state
  ) values (
    owner_id, reflection_id, 'self_report', payload->'state'
  );

  insert into public.policy_decisions (
    id, user_id, reflection_id, policy_name, policy_version, action_id,
    recommended_action_id, selection_source, eligible_for_ope, propensity,
    available_actions, context_snapshot
  ) values (
    (decision_payload->>'decision_id')::uuid,
    owner_id,
    reflection_id,
    decision_payload->>'policy_name',
    decision_payload->>'policy_version',
    decision_payload->>'action_id',
    decision_payload->>'recommended_action_id',
    decision_payload->>'selection_source',
    coalesce((decision_payload->>'eligible_for_ope')::boolean, true),
    (decision_payload->>'propensity')::double precision,
    decision_payload->'safe_action_ids',
    decision_payload->'context_snapshot'
  );

  if model_payload is not null and jsonb_typeof(model_payload) <> 'null' then
    insert into public.model_runs (
      user_id, reflection_id, model, provider, latency_ms, prompt_tokens,
      completion_tokens, schema_valid, fallback_reason
    ) values (
      owner_id,
      reflection_id,
      model_payload->>'model',
      model_payload->>'provider',
      (model_payload->>'latency_ms')::integer,
      nullif(model_payload->>'prompt_tokens', '')::integer,
      nullif(model_payload->>'completion_tokens', '')::integer,
      (model_payload->>'schema_valid')::boolean,
      model_payload->>'fallback_reason'
    );
  end if;

  insert into public.safety_events (
    user_id, reflection_id, mode, reason_codes, locale
  ) values (
    owner_id,
    reflection_id,
    safety_payload->>'mode',
    coalesce(safety_payload->'reasons', '[]'::jsonb),
    safety_payload->>'locale'
  );

  return payload;
end;
$$;

create or replace function public.save_outcome_record(payload jsonb)
returns jsonb
language plpgsql
security invoker
set search_path = public
as $$
declare
  owner_id uuid := auth.uid();
  outcome_id uuid := (payload->>'id')::uuid;
  decision_id uuid := (payload->>'decision_id')::uuid;
  existing_record jsonb;
begin
  if owner_id is null then
    raise exception 'Authentication required';
  end if;
  if (payload->>'user_id')::uuid <> owner_id then
    raise exception 'Record owner does not match authenticated user';
  end if;

  select record into existing_record
  from public.outcomes
  where id = outcome_id and user_id = owner_id;
  if found then
    if (existing_record->>'decision_id')::uuid <> decision_id then
      raise exception 'Outcome request ID is already in use';
    end if;
    return existing_record;
  end if;

  if not exists (
    select 1 from public.policy_decisions
    where id = decision_id and user_id = owner_id
  ) then
    raise exception 'Policy decision does not belong to authenticated user';
  end if;

  insert into public.outcomes (
    id, user_id, decision_id, created_at, record
  ) values (
    outcome_id,
    owner_id,
    decision_id,
    (payload->>'created_at')::timestamptz,
    payload
  );

  return payload;
end;
$$;

revoke all on function public.save_reflection_bundle(jsonb) from public;
revoke all on function public.save_outcome_record(jsonb) from public;
grant execute on function public.save_reflection_bundle(jsonb) to authenticated;
grant execute on function public.save_outcome_record(jsonb) to authenticated;
