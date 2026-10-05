# Connected Luna: narrow feature boundaries

This work extends the existing FastAPI, Next.js, SQLite/Supabase, and OpenRouter
application. It does not replace the Phase A conversation lifecycle.

## Writing and reflection

`journal_models.py` defines immutable saved writing, independent of action records.
`journals.py` owns owner-scoped CRUD and explicit reflection. `persistence.py`
implements the same contracts for SQLite and Supabase. Generated one-entry
reflections are transient; source writing is saved only when requested.

`reflection_prompts.py` holds shared, versioned reflection instructions. The
general chat prompt incorporates them. Source writing is passed in user messages;
only server-defined instructions enter the trusted instruction layer. Output
schema validation does not prove that an interpretation is useful or supported.

## Explicit journal context

A conversation may reference one source entry. Its immutable creation timestamp
distinguishes that source from a later entry reusing the same UUID. The server
checks ownership, consent, availability, and source identity. Database validation
also protects the race between reading the source and creating the conversation.
Source deletion invalidates linked work. The browser displays which entry is in
use and guards against old asynchronous responses replacing a newer selection.

## Optional discovery

Chat and saved action cards now link to this workflow through fixed general goal
keys. Discover maps those keys to editable general topics and requires fresh
approval. The handoff URL carries no entry/chat identifier or private writing.
The return control resumes the open chat. Saved catalog cards remain separately
labelled; live discovery does not silently replace the collection.

`discovery_models.py`, `discovery.py`, and `discovery_prompts.py` own a stateless
workflow. It receives an approved general topic, optional feedback and previous
query, and rejected links; it has no journal-repository retrieval capability.

The initial implementation uses Brave search snippets. It does not read candidate
pages, resolve their DNS, verify claims, or guarantee link availability. Selection
is constrained to retrieved candidate IDs, so the server constructs final URLs
from those candidates rather than trusting model-generated links. Structural URL
validation rejects unsafe syntax. Results expose their evidence type and limits.

At most one search and two model calls serve a request. Refinement keeps the
original goal in the query and can add a focus from feedback. Normalized excluded
URLs are filtered before model selection. Both providers must be configured,
discovery enabled, and AI consent supplied; otherwise no fake results are returned.

## Evidence boundaries

The latest release is documented in `AUDIT_UPGRADE_RELEASE.md`. It adds strict
local conversation-output validation, current-turn invitation readiness and a
versioned fictional evaluation set. Just talk remains a persisted server
preference. Historical readiness and turn count do not establish fresh consent
to an activity. Model structure and semantic usefulness are evaluated separately.

Unit/API tests use controlled model and search responses. PostgreSQL checks
exercise actual migrations, signed writes, RLS, and lifecycle behavior. Browser
integration checks exercise real UI, API, PostgREST, and PostgreSQL while replacing
provider and auth issuance. None establishes real model reflection quality,
search relevance, clinical benefit, or production readiness by itself.

The additive journal SQL is now applied to the existing Supabase project, as
recorded in `CONNECTED_LUNA_PREVIEW.md`. Applied migration files are immutable.
Future deployments still require confirming provider settings, migration history,
source revision, and the user acceptance sequence.
