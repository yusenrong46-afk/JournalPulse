# Encrypted local recovery — JournalPulse

This is a manual logical-export procedure for the existing Free Supabase project
`cggixttnyprrbaotmjxj`. It follows the free-plan export approach in the
[Supabase backup guidance](https://supabase.com/docs/guides/platform/backups).
It does **not** establish managed daily backups, point-in-time recovery, or a
working hosted Auth service after a disaster.

## Protection and frequency

Run a backup manually every week and before database changes. The tool retains
three **successfully restored and verified** encrypted generations. An export that
fails verification is kept for investigation and never replaces the sole good copy.
A weekly routine can lose up to a week's new data; missed runs make that interval
longer. No automatic schedule has been configured. Recovery duration is unmeasured
until the real export/restore drill completes.

Backups are outside Git at `/Users/thomas/.local/share/journalpulse-backups` (0700),
with ciphertext and verification files restricted to the owner (0600). Keep the
passphrase in a password manager separately from the encrypted files. Losing it
means the backup cannot be recovered. Do not share it in chat, SQL editor snippets,
command arguments, screenshots or issue trackers.

## Create and verify

Prerequisites: Docker Desktop, `gpg`, `uv`, and the existing database password.
Do not reset the password for this exercise. Get this project's **session pooler**
URI on port 5432 from Supabase Connect. Percent-encode special password characters
in the URI; do not add query parameters. The script accepts only the approved
project and verifies TLS. Unset any `PG*` environment overrides first.

Run from your own interactive terminal:

```sh
uv run --no-project --script /Users/thomas/Documents/ChatGPT/JP/scripts/live_backup.py create
```

Paste the connection URI at the hidden prompt; enter and repeat a new passphrase
of 16–1024 characters. It is never echoed. The script:

1. Opens a consistent, repeatable-read source snapshot and records PostgreSQL,
   schema, extension and role metadata. Row counts and checksums stay in the
   encrypted manifest; no journal content is printed.
2. Exports `public`, `private`, `auth`, `storage` and `extensions` with a pinned
   Supabase PostgreSQL 17 image, keeping the same source snapshot throughout.
3. Packs and encrypts the archive in memory with GPG AES256 **before writing it**.
   It then decrypts in memory to check integrity.
4. Restores into a unique disposable container with no network, no published
   ports, no host data binds, tmpfs database storage, no preload/job libraries,
   and no application or AI process. It verifies every exported table fingerprint,
   private-secret ACLs, signed writes and cross-account reads using fictional users.
5. Simulates deletion after a fixture backup, replays the deletion in another
   isolated restore, and checks that the deleted object ID cannot be recreated.
6. Removes the restore container, writes a content-free `.verified.json` result,
   and only then rotates old verified generations.

The reviewed limit is 128 MiB and PostgreSQL major 17. Storage objects and Vault
secrets cause a stop because this procedure does not back up their external bytes
or encryption dependencies. Role names and RLS bypass attributes are preserved
without importing role passwords; this is not a reconstruction of cloud services.
A failed private operation suppresses raw diagnostics because they may contain
sensitive SQL values. Keep any failed export encrypted; do not decrypt to debug logs.

## Recheck an existing backup

Use the actual encrypted file path produced by `create`:

```sh
uv run --no-project --script /Users/thomas/Documents/ChatGPT/JP/scripts/live_backup.py verify /Users/thomas/.local/share/journalpulse-backups/backup-TIMESTAMP-ID.gpg
```

Only the passphrase is requested. This performs the same isolated restore drill,
removes its container, and refreshes the verification result. It never overwrites
production or leaves a restored database available to users.

## Deletions and a real incident

A backup can contain writing or an identity deleted after that backup. Before any
future recovery is made accessible, an operator must obtain a complete, trusted
post-backup deletion history. Relevant content-free records include
`private.deleted_object_ids`, account erasure revisions, and (after migration
`202610140002`) `private.account_closures`. Retain and protect those records when
handling a deletion request; they are not journal text.

Replay object/data deletions and completed or pending account closures against the
isolated recovery. For a closed account, remove its recovered Auth identity and
all dependent rows, preserve its closure marker, and verify absence. Reconcile
signing-key rotation as well. A fresh source export can preserve current deletion
history while the source remains available. This plan has **not** built an
independently replicated deletion ledger; if the source is lost and later deletion
history cannot be proved complete, **do not open the restored data to anyone**.
The supplied drill always destroys the local recovery, so it cannot accidentally
publish an unreconciled snapshot.

Record only: encrypted file checksum, timestamp, tool/source identity, PASS/STOPPED,
isolation/check results and measured duration. Do not copy the encrypted manifest,
private identifiers, row digests, credentials or journal text into Git. The real
recovery status is tracked in `RELIABILITY_PROGRESS_2026-10-14.md`; fictional smoke
success must never be substituted for live recovery evidence.
