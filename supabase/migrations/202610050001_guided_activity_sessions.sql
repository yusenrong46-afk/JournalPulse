-- Additive guided-action storage. Legacy acceptance/readiness/RPC contracts stay
-- available for production while the preview adopts owner-bound in-chat sessions.
create unique index conversations_owner_id_activity_idx on public.conversations(user_id,id);
create table public.activity_sessions (
  id uuid primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  conversation_id uuid not null,
  source_entry_id uuid,
  status text not null check(status in ('offered','active','paused','awaiting_report','completed','stopped','declined')),
  revision integer not null default 0 check(revision>=0),
  created_at timestamptz not null,
  updated_at timestamptz not null,
  record jsonb not null,
  foreign key(user_id,conversation_id) references public.conversations(user_id,id) on delete cascade,
  foreign key(user_id,source_entry_id) references public.journal_entries(user_id,id) on delete cascade,
  check((record->>'id')::uuid=id),
  check((record->>'user_id')::uuid=user_id),
  check((record->>'conversation_id')::uuid=conversation_id),
  check(record->>'status'=status),
  check((record->>'revision')::integer=revision),
  check((record->>'created_at')::timestamptz=created_at),
  check((record->>'duration_seconds')::integer between 0 and 3600),
  check((record->>'remaining_seconds')::integer between 0 and 3600),
  check(record->'selection'->'eligible_for_ope'='false'::jsonb),
  check(record->'selection'->'propensity'='null'::jsonb)
);
create unique index activity_one_nonterminal_idx on public.activity_sessions(user_id,conversation_id)
  where status in ('offered','active','paused','awaiting_report');
create index activity_owner_created_idx on public.activity_sessions(user_id,created_at desc,id desc);
create table public.activity_receipts (
  id uuid not null,
  user_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid not null references public.activity_sessions(id) on delete cascade,
  operation text not null,
  request_hash text not null check(request_hash ~ '^[a-f0-9]{64}$'),
  created_at timestamptz not null default now(),
  primary key(user_id,id)
);
create index activity_receipts_session_idx on public.activity_receipts(session_id);
alter table public.activity_sessions enable row level security;
alter table public.activity_receipts enable row level security;
create policy activity_sessions_owner on public.activity_sessions for select using(auth.uid()=user_id);
create policy activity_receipts_owner on public.activity_receipts for select using(auth.uid()=user_id);
revoke all on public.activity_sessions,public.activity_receipts from public,anon,authenticated;
grant select on public.activity_sessions,public.activity_receipts to authenticated;

-- Turn and final-check-in bounds depend on retained message metadata. Individual
-- message deletion would reset those bounds without ending the conversation.
-- Owned conversation/account RPCs and source cascades still delete the full chat.
revoke delete on public.conversation_messages from authenticated;

create function private.activity_receipt(owner_id uuid,target_id uuid,request_id uuid,operation_name text,request_digest text)
returns boolean language plpgsql security definer set search_path=private,public as $$
declare saved public.activity_receipts%rowtype; used integer;
begin
  select * into saved from public.activity_receipts where user_id=owner_id and id=request_id;
  if found then
    if saved.session_id<>target_id or saved.operation<>operation_name or saved.request_hash<>request_digest then
      raise exception 'Activity request ID is already in use' using errcode='PT409';
    end if;
    return true;
  end if;
  select count(*) into used from public.activity_receipts where session_id=target_id;
  if used >= (case when operation_name in ('start','pause','resume') then 48 else 64 end) then
    raise exception 'Activity control limit reached; finish the activity to check in' using errcode='PT409';
  end if;
  return false;
end; $$;
create function private.activity_store_receipt(owner_id uuid,target_id uuid,request_id uuid,operation_name text,request_digest text)
returns void language sql security definer set search_path=private,public as $$
  insert into public.activity_receipts(id,user_id,session_id,operation,request_hash)
  values(request_id,owner_id,target_id,operation_name,request_digest);
$$;

-- Lock ordering is owned source, parent chat, then session, including deletion and
-- delayed follow-ups. The API never holds these locks while the model runs.
create function private.activity_locked_chat(target_id uuid,owner_id uuid)
returns public.conversations language plpgsql security definer set search_path=private,public as $$
declare chat public.conversations%rowtype;
begin
  select * into chat from public.conversations where id=target_id and user_id=owner_id;
  if not found then raise exception 'Conversation not found' using errcode='PT404'; end if;
  if chat.source_entry_id is not null then
    perform 1 from public.journal_entries where id=chat.source_entry_id and user_id=owner_id
      and created_at=chat.source_entry_created_at for key share;
    if not found then raise exception 'Journal entry not found' using errcode='PT404'; end if;
  end if;
  select * into chat from public.conversations where id=target_id and user_id=owner_id for update;
  if not found then raise exception 'Conversation not found' using errcode='PT404'; end if;
  return chat;
end; $$;
create function private.activity_guard(chat public.conversations,chat_revision integer,allow_support boolean,allow_listen boolean)
returns void language plpgsql security definer set search_path=private,public as $$
begin
  if chat.status<>'open' or chat.updated_at<now()-interval '24 hours' then
    raise exception 'Conversation is closed' using errcode='PT409'; end if;
  if chat.revision<>chat_revision then raise exception 'Conversation changed' using errcode='PT409'; end if;
  if not allow_support and chat.record->>'safety_mode'='support' then
    raise exception 'Support mode pauses ordinary activities' using errcode='PT409';
  end if;
  if not allow_listen and chat.record->>'activity_move'='pause' then
    raise exception 'This chat has paused activities' using errcode='PT409';
  end if;
  if not allow_listen and chat.record->>'interaction_preference'='listen' then
    raise exception 'This chat is set to Just talk' using errcode='PT409';
  end if;
end; $$;

create function public.jp_offer_activity_v1(payload text,signature text)
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
    update public.activity_sessions set status='declined',revision=revision+1,updated_at=now(),
      record=record||jsonb_build_object('status','declined','revision',revision+1,'updated_at',now()) where id=pending.id;
  end if;
  fresh:=(offered-'server_now'-'conversation_revision')||jsonb_build_object(
    'status','offered','revision',0,'created_at',now(),'updated_at',now());
  insert into public.activity_sessions(id,user_id,conversation_id,source_entry_id,status,revision,created_at,updated_at,record)
  values(target_id,owner_id,chat.id,chat.source_entry_id,'offered',0,now(),now(),fresh);
  perform private.activity_store_receipt(owner_id,target_id,request_id,'offer',body->>'request_hash');
  return fresh;
end; $$;

create function public.jp_activity_command_v1(payload text,signature text)
returns jsonb language plpgsql security definer set search_path=private,public as $$
declare
  body jsonb:=private.verified_payload(payload,signature,'activity_command'); owner_id uuid:=auth.uid();
  target_id uuid:=(body->>'session_id')::uuid; request jsonb:=body->'request';
  request_id uuid:=(request->>'client_request_id')::uuid; operation_name text:=request->>'command';
  chat public.conversations%rowtype; saved public.activity_sessions%rowtype; changed jsonb; next_status text;
  remaining integer; deadline timestamptz;
begin
  select * into saved from public.activity_sessions where id=target_id and user_id=owner_id;
  if not found then raise exception 'Activity not found' using errcode='PT404'; end if;
  chat:=private.activity_locked_chat(saved.conversation_id,owner_id);
  select * into saved from public.activity_sessions where id=target_id and user_id=owner_id for update;
  if not found then raise exception 'Activity not found' using errcode='PT404'; end if;
  if private.activity_receipt(owner_id,target_id,request_id,operation_name,body->>'request_hash') then return saved.record; end if;
  perform private.activity_guard(chat,(request->>'expected_conversation_revision')::integer,false,operation_name in ('stop','decline'));
  if saved.revision<>(request->>'expected_revision')::integer then
    raise exception 'Activity changed' using errcode='PT409';
  end if;
  changed:=saved.record; next_status:=saved.status; remaining:=(changed->>'remaining_seconds')::integer;
  deadline:=(changed->>'expires_at')::timestamptz;
  if saved.status='active' and deadline is not null then
    remaining:=least((changed->>'duration_seconds')::integer,greatest(0,ceil(extract(epoch from deadline-now()))::integer));
  end if;
  if operation_name='start' and saved.status='offered' then
    if (select count(*) from public.conversation_messages where conversation_id=chat.id and user_id=owner_id and role='user')>=20 then
      raise exception 'This chat has reached its activity limit' using errcode='PT409';
    end if;
    next_status:='active'; changed:=changed||jsonb_build_object('started_at',now());
    if changed->'resource'->>'format'='timer' then deadline:=now()+make_interval(secs=>remaining); end if;
  elsif operation_name='pause' and saved.status='active' and changed->'resource'->>'format'='timer' then
    next_status:=case when remaining=0 then 'awaiting_report' else 'paused' end; deadline:=null;
    if remaining=0 then changed:=changed||jsonb_build_object('check_in_issued',true); end if;
  elsif operation_name='resume' and saved.status='paused' then
    next_status:='active'; deadline:=now()+make_interval(secs=>remaining);
  elsif operation_name='expire' and saved.status='active' and deadline is not null then
    if now()<deadline then raise exception 'The timer has not finished yet' using errcode='PT409'; end if;
    next_status:='awaiting_report'; remaining:=0; deadline:=null;
    changed:=changed||jsonb_build_object('check_in_issued',true);
  elsif operation_name='finish_early' and saved.status in ('active','paused') then
    next_status:='awaiting_report'; deadline:=null; changed:=changed||jsonb_build_object('check_in_issued',true);
  elsif operation_name='decline' and saved.status='offered' then
    next_status:='declined'; deadline:=null;
  elsif operation_name='stop' and saved.status in ('offered','active','paused','awaiting_report') then
    next_status:='stopped'; deadline:=null;
    changed:=changed||jsonb_build_object('check_in_issued',changed->>'started_at' is not null);
  else raise exception 'This activity changed; refresh before trying again' using errcode='PT409'; end if;
  changed:=changed||jsonb_build_object('status',next_status,'remaining_seconds',remaining,'expires_at',deadline,
    'revision',saved.revision+1,'updated_at',now());
  update public.activity_sessions set status=next_status,revision=saved.revision+1,updated_at=now(),record=changed where id=target_id;
  perform private.activity_store_receipt(owner_id,target_id,request_id,operation_name,body->>'request_hash');
  return changed;
end; $$;

create function public.jp_report_activity_v1(payload text,signature text)
returns jsonb language plpgsql security definer set search_path=private,public as $$
declare
  body jsonb:=private.verified_payload(payload,signature,'report_activity'); owner_id uuid:=auth.uid();
  target_id uuid:=(body->>'session_id')::uuid; request jsonb:=body->'request'; report jsonb;
  request_id uuid:=(request->>'client_request_id')::uuid; chat public.conversations%rowtype;
  saved public.activity_sessions%rowtype; changed jsonb; next_status text; support jsonb:=body->'safety';
begin
  select * into saved from public.activity_sessions where id=target_id and user_id=owner_id;
  if not found then raise exception 'Activity not found' using errcode='PT404'; end if;
  chat:=private.activity_locked_chat(saved.conversation_id,owner_id);
  select * into saved from public.activity_sessions where id=target_id and user_id=owner_id for update;
  if not found then raise exception 'Activity not found' using errcode='PT404'; end if;
  if private.activity_receipt(owner_id,target_id,request_id,'report',body->>'request_hash') then return saved.record; end if;
  perform private.activity_guard(chat,(request->>'expected_conversation_revision')::integer,true,true);
  if saved.revision<>(request->>'expected_revision')::integer then raise exception 'Activity changed' using errcode='PT409'; end if;
  if saved.status not in ('awaiting_report','stopped') or saved.record->>'report' is not null or saved.record->>'started_at' is null then
    raise exception 'This activity is not awaiting a report' using errcode='PT409';
  end if;
  report:=request-'client_request_id'-'expected_revision'-'expected_conversation_revision';
  if report->>'participation' not in ('completed','partial','not_tried','stopped')
     or length(coalesce(report->>'note',''))>1000 then raise exception 'Invalid activity report' using errcode='PT400'; end if;
  next_status:=case when report->>'participation'='stopped' then 'stopped' else 'completed' end;
  changed:=saved.record||jsonb_build_object('status',next_status,'report',report,'reported_at',now(),
    'follow_up_status','pending','expires_at',null,'revision',saved.revision+1,'updated_at',now());
  update public.activity_sessions set status=next_status,revision=saved.revision+1,updated_at=now(),record=changed where id=target_id;
  perform private.activity_store_receipt(owner_id,target_id,request_id,'report',body->>'request_hash');
  if support->>'mode'='support' and chat.record->>'safety_mode'<>'support' then
    update public.conversations set revision=revision+1,updated_at=now(),record=record||jsonb_build_object(
      'safety_mode','support','safety',support,'card',null,'activity_card',null,'ready_for_action',false,'revision',revision+1,'updated_at',now()) where id=chat.id;
    select record into changed from public.activity_sessions where id=target_id;
  end if;
  return changed;
end; $$;

create function public.jp_claim_activity_followup_v1(payload text,signature text)
returns jsonb language plpgsql security definer set search_path=private,public as $$
declare
  body jsonb:=private.verified_payload(payload,signature,'claim_activity_follow_up'); owner_id uuid:=auth.uid();
  target_id uuid:=(body->>'session_id')::uuid; request jsonb:=body->'request';
  request_id uuid:=(request->>'client_request_id')::uuid; chat public.conversations%rowtype;
  saved public.activity_sessions%rowtype; replay boolean; changed jsonb; is_final boolean;
begin
  select * into saved from public.activity_sessions where id=target_id and user_id=owner_id;
  if not found then raise exception 'Activity not found' using errcode='PT404'; end if;
  chat:=private.activity_locked_chat(saved.conversation_id,owner_id);
  select * into saved from public.activity_sessions where id=target_id and user_id=owner_id for update;
  if not found then raise exception 'Activity not found' using errcode='PT404'; end if;
  replay:=private.activity_receipt(owner_id,target_id,request_id,'follow_up',body->>'request_hash');
  if saved.record->>'follow_up_status'='ready' or (saved.record->>'follow_up_status'='generating'
     and (saved.record->>'follow_up_lease_until')::timestamptz>now()) then
    return jsonb_build_object('session',saved.record,'claimed',false);
  end if;
  perform private.activity_guard(chat,(request->>'expected_conversation_revision')::integer,true,true);
  if not replay and saved.revision<>(request->>'expected_revision')::integer then
    raise exception 'Activity changed' using errcode='PT409'; end if;
  if saved.record->>'report' is null or (saved.record->>'follow_up_attempts')::integer>=3 then
    raise exception 'The saved check-in has no available follow-up attempt' using errcode='PT409'; end if;
  select count(*)>=20 into is_final from public.conversation_messages where conversation_id=chat.id and user_id=owner_id and role='user';
  if is_final and exists(select 1 from public.activity_sessions where conversation_id=chat.id and user_id=owner_id
    and id<>target_id and (record->>'final_follow_up')::boolean) then
    raise exception 'This chat has reached its final check-in response' using errcode='PT409'; end if;
  changed:=saved.record||jsonb_build_object('follow_up_status','generating','follow_up_request_id',request_id,
    'follow_up_lease_until',now()+interval '90 seconds','follow_up_attempts',(saved.record->>'follow_up_attempts')::integer+1,
    'final_follow_up',is_final or (saved.record->>'final_follow_up')::boolean,'revision',saved.revision+1,'updated_at',now());
  update public.activity_sessions set revision=saved.revision+1,updated_at=now(),record=changed where id=target_id;
  if not replay then perform private.activity_store_receipt(owner_id,target_id,request_id,'follow_up',body->>'request_hash'); end if;
  return jsonb_build_object('session',changed,'claimed',true);
end; $$;

create function public.jp_finish_activity_followup_v1(payload text,signature text)
returns jsonb language plpgsql security definer set search_path=private,public as $$
declare
  body jsonb:=private.verified_payload(payload,signature,'finish_activity_follow_up'); owner_id uuid:=auth.uid();
  target_id uuid:=(body->>'session_id')::uuid; request_id uuid:=(body->>'request_id')::uuid;
  chat public.conversations%rowtype; saved public.activity_sessions%rowtype; changed jsonb; assistant jsonb:=body->'assistant_message';
  directive jsonb:=body->'directive';
begin
  select * into saved from public.activity_sessions where id=target_id and user_id=owner_id;
  if not found then raise exception 'Activity not found' using errcode='PT404'; end if;
  chat:=private.activity_locked_chat(saved.conversation_id,owner_id);
  select * into saved from public.activity_sessions where id=target_id and user_id=owner_id for update;
  if not found then raise exception 'Activity not found' using errcode='PT404'; end if;
  if saved.created_at is distinct from (body->>'expected_session_created_at')::timestamptz then
    raise exception 'Activity changed' using errcode='PT409'; end if;
  if saved.record->>'follow_up_status'='ready' and (saved.record->>'follow_up_request_id')::uuid=request_id then return saved.record; end if;
  perform private.activity_guard(chat,(body->>'expected_conversation_revision')::integer,true,true);
  if saved.revision<>(body->>'expected_revision')::integer or saved.record->>'follow_up_status'<>'generating'
     or (saved.record->>'follow_up_request_id')::uuid is distinct from request_id then
    raise exception 'This follow-up changed' using errcode='PT409'; end if;
  if assistant->>'content' is null then
    changed:=saved.record||jsonb_build_object('follow_up_status','failed','follow_up_lease_until',null,
      'revision',saved.revision+1,'updated_at',now());
  else
    if assistant->>'role'<>'assistant' or (assistant->>'conversation_id')::uuid is distinct from chat.id
       or length(assistant->>'content') not between 1 and 1200 then
      raise exception 'Invalid activity follow-up' using errcode='PT400'; end if;
    if directive is not null and directive<>'null'::jsonb
       and (directive->>'card' is not null or directive->>'search_topic' is not null)
       and ((saved.record->>'final_follow_up')::boolean or chat.record->>'safety_mode'='support'
         or chat.record->>'interaction_preference'='listen'
         or (select count(*) from public.conversation_messages where user_id=owner_id and conversation_id=chat.id and role='user')>=20) then
      raise exception 'This check-in cannot offer another activity' using errcode='PT409'; end if;
    -- No user message is synthesized. The one outcome reply has a distinct UUID
    -- and cannot become another route around the ordinary 20-user-message cap.
    insert into public.conversation_messages(id,conversation_id,user_id,client_message_id,role,created_at,record)
    values((assistant->>'id')::uuid,chat.id,owner_id,null,'assistant',now(),assistant||jsonb_build_object('created_at',now()));
    update public.conversations set revision=revision+1,updated_at=now(),record=record||jsonb_build_object(
      'revision',revision+1,'updated_at',now())
      || case when directive is null or directive='null'::jsonb then '{}'::jsonb else jsonb_build_object(
        'card',null,'activity_card',case when directive->>'card' is null then 'null'::jsonb else directive->'card'||jsonb_build_object('offered_message_id',assistant->>'id') end,'activity_constraints',coalesce(nullif(directive->'constraints','null'::jsonb),record->'activity_constraints'),
        'activity_goal',coalesce(nullif(directive->'goal','null'::jsonb),record->'activity_goal'),
        'activity_search_topic',directive->'search_topic','activity_move',directive->'move',
        'ready_for_action',directive->>'card' is not null) end where id=chat.id;
    changed:=saved.record||jsonb_build_object('follow_up_status','ready','follow_up_reply',null,
      'follow_up_model_run',assistant->'model_run','follow_up_message_id',assistant->>'id','follow_up_lease_until',null,
      'revision',saved.revision+1,'updated_at',now());
  end if;
  update public.activity_sessions set revision=saved.revision+1,updated_at=now(),record=changed where id=target_id;
  return changed;
end; $$;

-- New LLM cards use a versioned extra field. Old deployed models ignore it;
-- their original card/PolicyDecision still has a real nonnull propensity.
create function private.guard_guided_activity_card()
returns trigger language plpgsql security definer set search_path=private,public as $$
begin
  if new.record->'card'->'decision_preview'->>'propensity' is null
     and new.record->'card'->'decision_preview'->'eligible_for_ope'='false'::jsonb then
    raise exception 'Guided Luna cards must use the additive activity card field' using errcode='PT400';
  end if;
  if new.status='closed' or new.record->>'safety_mode'='support'
     or new.record->>'interaction_preference'='listen' or new.record->>'activity_move'='pause' then
    new.record:=new.record||jsonb_build_object('activity_card',null);
  end if;
  return new;
end; $$;
create trigger conversation_guided_card_compatibility before insert or update on public.conversations
  for each row execute function private.guard_guided_activity_card();

-- Existing close/sweep/accept RPCs all update the conversation, so this additive
-- trigger gives them the new lifecycle without changing their deployed signature.
create function private.invalidate_conversation_activities()
returns trigger language plpgsql security definer set search_path=private,public as $$
declare target public.activity_sessions%rowtype; changed jsonb; clear_text boolean;
begin
  if new.status='closed' or (new.record->>'safety_mode'='support' and old.record->>'safety_mode' is distinct from 'support')
     or (new.record->>'interaction_preference'='listen' and old.record->>'interaction_preference' is distinct from 'listen')
     or (new.record->>'activity_move'='pause' and old.record->>'activity_move' is distinct from 'pause') then
    clear_text:=new.status='closed' and not coalesce((new.record->>'retain_text')::boolean,false);
    for target in select * from public.activity_sessions where conversation_id=new.id and user_id=new.user_id for update loop
      changed:=target.record||jsonb_build_object('expires_at',null,'check_in_issued',false,'follow_up_request_id',null,'follow_up_lease_until',null,
        'revision',target.revision+1,'updated_at',now());
      if target.status in ('offered','active','paused','awaiting_report') then changed:=changed||jsonb_build_object('status','stopped'); end if;
      if target.record->>'follow_up_status' in ('pending','generating') then changed:=changed||jsonb_build_object('follow_up_status','failed'); end if;
      if clear_text then
        changed:=changed||jsonb_build_object('recommendation_reason',null,'follow_up_reply',null);
        if target.record->>'report' is not null then changed:=changed||jsonb_build_object('report',(target.record->'report')||jsonb_build_object('note',null)); end if;
      end if;
      update public.activity_sessions set status=changed->>'status',revision=target.revision+1,updated_at=now(),record=changed where id=target.id;
    end loop;
    if clear_text then delete from public.activity_receipts where session_id in(select id from public.activity_sessions where conversation_id=new.id); end if;
  elsif coalesce(old.record->'card','null'::jsonb) is distinct from coalesce(new.record->'card','null'::jsonb)
     or coalesce(old.record->'activity_card','null'::jsonb) is distinct from coalesce(new.record->'activity_card','null'::jsonb)
     or coalesce(old.record->'activity_constraints','null'::jsonb) is distinct from coalesce(new.record->'activity_constraints','null'::jsonb)
     or coalesce(old.record->'activity_goal','null'::jsonb) is distinct from coalesce(new.record->'activity_goal','null'::jsonb)
     or coalesce(old.record->'activity_search_topic','null'::jsonb) is distinct from coalesce(new.record->'activity_search_topic','null'::jsonb) then
    -- A newer recommendation or constraints withdraw an unstarted offer. This
    -- is supersession, not a participant decline; ongoing activities stay intact.
    update public.activity_sessions set status='stopped',revision=revision+1,updated_at=now(),
      record=record||jsonb_build_object('status','stopped','revision',revision+1,'updated_at',now(),
        'expires_at',null,'check_in_issued',false)
    where conversation_id=new.id and user_id=new.user_id and status='offered'
      and record->>'started_at' is null;
  end if;
  return new;
end; $$;
create trigger conversation_activity_lifecycle after update on public.conversations
  for each row execute function private.invalidate_conversation_activities();

-- Preference changes withdraw either generation of recommendation. Return the
-- database row after BEFORE triggers, so mutation responses match later reads.
create or replace function public.jp_change_preference(payload text, signature text)
returns jsonb
language plpgsql
security definer
set search_path = private, public
as $$
declare
  body jsonb := private.verified_payload(payload, signature, 'change_preference');
  owner_id uuid := auth.uid();
  target_id uuid := (body->>'conversation_id')::uuid;
  request_id uuid := (body->>'client_request_id')::uuid;
  wanted text := body->>'preference';
  expected integer := (body->>'expected_revision')::integer;
  current_row public.conversations%rowtype;
  receipt public.conversation_preference_requests%rowtype;
  changed jsonb;
  moment timestamptz := now();
begin
  if request_id is null or expected is null or expected < 0
     or wanted is null or wanted not in ('listen', 'act') then
    raise exception 'Invalid preference command' using errcode = 'PT400';
  end if;
  select * into current_row from public.conversations
  where id = target_id and user_id = owner_id for update;
  if not found then
    raise exception 'Conversation not found' using errcode = 'PT404';
  end if;
  select * into receipt from public.conversation_preference_requests
  where conversation_id = target_id and id = request_id;
  if found then
    if receipt.preference <> wanted or receipt.expected_revision <> expected then
      raise exception 'Preference request ID is already in use' using errcode = 'PT409';
    end if;
    return current_row.record || jsonb_build_object('revision', current_row.revision);
  end if;
  if current_row.status <> 'open' then
    raise exception 'Conversation is closed' using errcode = 'PT409';
  end if;
  if current_row.revision <> expected then
    raise exception 'Conversation changed' using errcode = 'PT409';
  end if;
  if current_row.record->>'safety_mode' = 'support' then
    raise exception 'Support mode cannot be changed' using errcode = 'PT409';
  end if;
  changed := current_row.record || jsonb_build_object(
    'interaction_preference', wanted, 'card', null, 'activity_card', null,
    'ready_for_action', wanted = 'act', 'revision', current_row.revision + 1, 'updated_at', moment
  );
  update public.conversations set record = changed,
    revision = current_row.revision + 1, updated_at = moment where id = target_id
    returning record into changed;
  insert into public.conversation_preference_requests
    (id, conversation_id, user_id, preference, expected_revision, created_at)
  values (request_id, target_id, owner_id, wanted, expected, moment);
  return changed;
end;
$$;
revoke all on function public.jp_change_preference(text, text) from public, anon, authenticated;
grant execute on function public.jp_change_preference(text, text) to authenticated;

create or replace function public.delete_my_journalpulse_data()
returns integer language plpgsql security definer set search_path=private,public as $$
declare owner_id uuid:=auth.uid(); deleted_count integer:=0; affected integer; table_name text;
begin
  if owner_id is null then raise exception 'Authentication required' using errcode='PT401'; end if;
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

create function public.jp_readiness_v3(probe text,signature text)
returns jsonb language plpgsql stable security definer set search_path=private,public as $$
declare result jsonb; activity_ready boolean;
begin
  result:=public.jp_readiness_v2(probe,signature);
  activity_ready:=to_regprocedure('public.jp_offer_activity_v1(text,text)') is not null
    and to_regprocedure('public.jp_finish_activity_followup_v1(text,text)') is not null
    and exists(select 1 from pg_trigger where tgname='conversation_activity_lifecycle' and not tgisinternal)
    and exists(select 1 from pg_trigger where tgname='conversation_guided_card_compatibility' and not tgisinternal);
  return result||jsonb_build_object('schema','guided-action-1','activities',case when activity_ready then 'ready' else 'missing' end);
end; $$;

revoke all on function private.activity_receipt(uuid,uuid,uuid,text,text),
  private.activity_store_receipt(uuid,uuid,uuid,text,text),private.activity_locked_chat(uuid,uuid),
  private.activity_guard(public.conversations,integer,boolean,boolean),private.invalidate_conversation_activities(),
  private.guard_guided_activity_card()
  from public,anon,authenticated;
revoke all on function public.jp_offer_activity_v1(text,text),public.jp_activity_command_v1(text,text),
  public.jp_report_activity_v1(text,text),public.jp_claim_activity_followup_v1(text,text),
  public.jp_finish_activity_followup_v1(text,text),public.jp_readiness_v3(text,text)
  from public,anon,authenticated;
grant execute on function public.jp_offer_activity_v1(text,text),public.jp_activity_command_v1(text,text),
  public.jp_report_activity_v1(text,text),public.jp_claim_activity_followup_v1(text,text),
  public.jp_finish_activity_followup_v1(text,text) to authenticated;
grant execute on function public.jp_readiness_v3(text,text) to anon,authenticated;
