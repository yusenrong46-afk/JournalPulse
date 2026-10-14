# Account closure contract and release gate

`DELETE /v1/account/data` keeps its existing meaning: delete saved app data while
keeping sign-in. `DELETE /v1/account` separately closes the authenticated account
and its dependent saved records. Privacy, Terms and Settings use the approved
contact `yusenrong46@gmail.com`.

## Server boundary

Enable only after migration `202610140002_account_closure.sql` and disposable
hosted-account verification. Server-only configuration:

- `SUPABASE_SERVICE_ROLE_KEY`: privileged Admin credential; never `NEXT_PUBLIC_*`,
  browser state, source control, logs or command arguments.
- `JOURNALPULSE_ACCOUNT_CLOSURE_ENABLED=true`: explicit release switch. Default false.
- Existing Supabase URL/public key and write-signing key remain required.

The initial request verifies its bearer token through Supabase Auth. Its `sub` must
match the returned identity. Auth-issued AMR must show a real sign-in within five
minutes; refreshed token `iat` and user-editable metadata do not count. The wire
request accepts only `confirmation: "delete my account"`, a UUID4 request ID and a
256-bit random receipt. Supplying a user ID is rejected.

A signed owner-scoped RPC creates the private pending record under the same
transaction lock used by writes. Every public user-owned table rejects new writes
while closure is pending or complete. The data-revision preflight rejects new
model-backed mutations early; already-running work still faces database guards.
Other owners remain unaffected.

The API performs a hard delete through Supabase Admin, verifies the identity is
absent, then calls a service-only RPC that checks dependent application rows are
gone. It returns 204 only after these checks. A lost or uncertain response returns
503 with honest retry guidance. The short-lived receipt can resume only this
operation after Auth invalidates the session; a valid session for another owner
is rejected. Receipt plaintext stays in account-scoped tab session storage/memory,
never URLs or logs; the database retains its hash. After 15 minutes, a still-existing
account needs fresh sign-in; an already-deleted identity with unconfirmed closure
needs operator help. The operator must verify absence before saying it is complete.

The UI clears the closed account's drafts, cached preferences and local session
only after confirmed completion. It reports a browser sign-out error separately
from a confirmed server deletion. Fresh sign-in email is sent only when the user
presses its explicit button; no automated test sends email to a real user.

## Verification and deployment

Local API tests use a fake Auth transport. SQL tests use fictional identities in
real PostgreSQL and exercise owner isolation, stale/invalid receipts, pending-write
protection, both concurrency orderings, cascade deletion and retained recovery
markers. These tests do not prove hosted Admin API behavior.

Before production enablement, use newly created disposable accounts and fictional
writing on the tested staged build. Verify fresh sign-in, stale-auth rejection,
extra owner-field rejection, cross-account receipt rejection, blocked concurrent
writes, actual Admin identity removal, dependent-row absence, retry after a lost
response, and truthful errors. Never use an existing real user's account.

Apply the additive migration before the new API. `/ready` on the new API requires
`jp_readiness_v6` / `account-closure-1`; prior v5 remains available for rollback.
Capture migration status and the known-good deployment ID. Promote only the tested
build after all release gates pass.

Rollback by disabling new closures and restoring the prior application deployment;
retain this migration and pending/complete markers. Do not drop the guards or
ledger to get rid of an error, and do not restore deleted identities. Resolve a
pending operation privately, verify its state, and use a forward repair if needed.
Existing pending accounts remain write-blocked during rollback.

## Prepared hosted smoke command

After the real encrypted recovery is verified, apply the compatible migration and
stage the tested application with closure configured. Capture the verified Vercel
project ID, exact source commit and deployment origin in a release record:

```json
{
  "project_id": "prj_EOSYVlo1KjkdnOAT8Smm72LdgFrV",
  "commit": "EXACT_40_CHARACTER_COMMIT_FROM_VERIFIED_DEPLOYMENT",
  "url": "https://VERIFIED_JOURNALPULSE_DEPLOYMENT.vercel.app"
}
```

Then run in your interactive terminal, using the actual release-record path and a
new report path:

```sh
.venv/bin/python scripts/verify_hosted_closure.py --deployment-record RELEASE_RECORD.json --output HOSTED_CLOSURE_RESULT.json
```

The Admin and public anon keys are hidden prompts. The privileged credential goes
only to this project's Supabase API. The runner requires a successful encrypted
recovery record, creates two new fictional identities tagged with its unique run
ID, and stores only fictional writing. It never accepts an existing user ID as a
test target, lists real users, sends sign-in email, or makes a model call.
[Supabase's Admin creation API](https://supabase.com/docs/reference/javascript/auth-admin-createuser)
supports confirming these disposable test emails without an invitation.

The test waits just over five minutes, refreshes the token and proves that refresh
does not satisfy fresh sign-in. It then signs in freshly, closes the first account,
checks identity absence and the verified completion record, retries the same
closure receipt, rejects that receipt from the second account, and checks the
second account's writing is unchanged. Cleanup verifies the exact returned ID,
fictional email and server-owned test marker before deleting any remaining fixture.

A timeout during creation records an uncertain fictional email/run marker rather
than retrying or guessing which account to delete. Reconcile that fixture privately
before another attempt. A STOPPED report does not establish hosted completion.
The script's failure-injection and concurrent-write coverage remain the separately
recorded local API/PostgreSQL tests; it does not claim to inject production faults.
Do not publish raw responses or tokens. The committed guard tests use only fake
HTTP responses and are not evidence that this hosted smoke has run.
