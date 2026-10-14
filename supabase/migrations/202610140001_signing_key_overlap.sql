-- Add bounded verification overlap without changing the signed-write envelope.
-- New writes still use write_signing_key. The previous key is verification-only.
-- No keys are generated, installed or rotated by this migration.
alter table private.server_secrets add column if not exists expires_at timestamptz;

create or replace function private.signature_valid(payload text, signature text)
returns boolean
language plpgsql
stable
security definer
set search_path = private, public, extensions
as $$
declare
  signing_key text;
  previous_key text;
  previous_deadline timestamptz;
begin
  if signature is null or payload is null then
    return false;
  end if;
  select value into signing_key from private.server_secrets where name = 'write_signing_key';
  if signing_key is null then
    return false;
  end if;
  if encode(hmac(convert_to(payload, 'UTF8'), convert_to(signing_key, 'UTF8'), 'sha256'), 'hex')
      = lower(signature) then
    return true;
  end if;
  select value, expires_at into previous_key, previous_deadline from private.server_secrets
    where name = 'write_signing_key_previous';
  if previous_key is null or length(previous_key) < 32 or previous_deadline is null
      or not isfinite(previous_deadline)
      or previous_deadline <= statement_timestamp() then
    return false;
  end if;
  return encode(hmac(convert_to(payload, 'UTF8'), convert_to(previous_key, 'UTF8'), 'sha256'), 'hex')
    = lower(signature);
end;
$$;
revoke all on function private.signature_valid(text, text) from public, anon, authenticated;
