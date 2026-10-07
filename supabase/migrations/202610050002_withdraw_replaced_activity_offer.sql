-- Correct new offer replacements without rewriting ambiguous historical declines.
-- The signature and grants are preserved; apply after guided_activity_sessions.

create or replace function public.jp_offer_activity_v1(payload text,signature text)
returns jsonb language plpgsql security definer set search_path=private,public as $$
declare
  body jsonb:=private.verified_payload(payload,signature,'offer_activity');
  owner_id uuid:=auth.uid(); offered jsonb:=body->'session'; target_id uuid:=(offered->>'id')::uuid;
  request_id uuid:=(body->>'request_id')::uuid; chat public.conversations%rowtype;
  saved public.activity_sessions%rowtype; pending public.activity_sessions%rowtype; fresh jsonb;
begin
  chat:=private.activity_locked_chat((offered->>'conversation_id')::uuid,owner_id);
  perform pg_advisory_xact_lock(hashtextextended('activity:'||target_id::text,0));
  select * into saved from public.activity_sessions where id=target_id;
  if found then
    if saved.user_id<>owner_id or saved.conversation_id<>chat.id then
      raise exception 'Activity request ID is already in use' using errcode='PT409';
    end if;
    if private.activity_receipt(owner_id,target_id,request_id,'offer',body->>'request_hash') then return saved.record; end if;
  end if;
  perform private.activity_guard(chat,(body->>'expected_conversation_revision')::integer,false,false);
  if (offered->>'user_id')::uuid is distinct from owner_id
     or (offered->>'source_entry_id')::uuid is distinct from chat.source_entry_id then
    raise exception 'Activity source changed' using errcode='PT409';
  end if;
  if (select count(*) from public.conversation_messages where conversation_id=chat.id and user_id=owner_id and role='user')>=20
     or (select count(*) from public.activity_sessions where conversation_id=chat.id and user_id=owner_id)>=20 then
    raise exception 'This chat has reached its activity limit' using errcode='PT409';
  end if;
  if offered->'selection'->'eligible_for_ope' is distinct from 'false'::jsonb
     or offered->'selection'->'propensity' is distinct from 'null'::jsonb
     or (offered->>'duration_seconds')::integer not between 0 and 3600 then
    raise exception 'Invalid activity provenance' using errcode='PT400';
  end if;
  select * into pending from public.activity_sessions where user_id=owner_id and conversation_id=chat.id
    and status in ('offered','active','paused','awaiting_report') for update;
  if found then
    if pending.status<>'offered' then
      raise exception 'Finish or stop the current activity before starting another' using errcode='PT409';
    end if;
    -- Choosing a different saved option withdraws this offer; only an explicit
    -- decline is a rejection that should exclude a resource from future choices.
    update public.activity_sessions set status='stopped',revision=revision+1,updated_at=now(),
      record=record||jsonb_build_object('status','stopped','revision',revision+1,'updated_at',now(),
        'expires_at',null,'check_in_issued',false) where id=pending.id;
  end if;
  fresh:=(offered-'server_now'-'conversation_revision')||jsonb_build_object(
    'status','offered','revision',0,'created_at',now(),'updated_at',now());
  insert into public.activity_sessions(id,user_id,conversation_id,source_entry_id,status,revision,created_at,updated_at,record)
  values(target_id,owner_id,chat.id,chat.source_entry_id,'offered',0,now(),now(),fresh);
  perform private.activity_store_receipt(owner_id,target_id,request_id,'offer',body->>'request_hash');
  return fresh;
end; $$;

