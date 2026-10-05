"""An interrupted worker's expired lease permits a bounded, explicit retry."""

from datetime import timedelta
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from journalpulse.activity_models import ActivityFollowUpRequest
from journalpulse.api import create_app
from journalpulse.persistence import SQLiteRepository
from test_guided_action_boundaries import (
    OWNER,
    OWNER_HEADERS,
    Clock,
    FollowUpDouble,
    command,
    follow_up_body,
    offered,
    reported,
    seed_conversation,
    settings,
)


@pytest.mark.parametrize("elapsed,visible_status", [(89, "generating"), (90, "failed")])
def test_reads_expose_expired_claim_for_retry_without_mutating_storage(
    tmp_path: Path, elapsed: int, visible_status: str
) -> None:
    configured, clock, model = settings(tmp_path), Clock(), FollowUpDouble()
    repository = SQLiteRepository(configured.database_path)
    chat = seed_conversation(configured, clock)
    with TestClient(
        create_app(settings=configured, clock=clock, activity_follow_up_generator=model)
    ) as client:
        session = reported(client, command(client, command(client, offered(client, chat), "start"), "stop"))
        body = follow_up_body(session)
        claimed, owns = repository.claim_activity_follow_up(
            OWNER, UUID(session["id"]), ActivityFollowUpRequest.model_validate(body), now=clock()
        )
        assert owns
        # Simulate a worker exiting after acquiring the claim, before saving a reply.
        clock.now += timedelta(seconds=elapsed)
        for endpoint in (
            f"/v1/activity-sessions/{session['id']}",
            f"/v1/conversations/{chat.id}/activity-sessions",
        ):
            read = client.get(endpoint, headers=OWNER_HEADERS)
            assert read.status_code == 200, read.text
            assert read.json()["follow_up_status"] == visible_status
            assert read.json()["revision"] == claimed.revision
            assert read.json()["report"] == session["report"]
        stored = repository.get_activity_session(OWNER, claimed.id)
        assert stored == claimed, "Reading an expired lease must not rewrite the canonical claim"

        retry = client.post(
            f"/v1/activity-sessions/{session['id']}/follow-up", headers=OWNER_HEADERS, json=body
        )
        if visible_status == "generating":
            assert retry.status_code == 202
            assert model.calls == 0
        else:
            assert retry.status_code == 200, retry.text
            assert retry.json()["follow_up_status"] == "ready"
            assert retry.json()["follow_up_attempts"] == 2
            assert model.calls == 1
            replay = client.post(
                f"/v1/activity-sessions/{session['id']}/follow-up", headers=OWNER_HEADERS, json=body
            )
            assert replay.status_code == 200
            assert model.calls == 1
            assert len(repository.list_messages(OWNER, chat.id)) == 1


def test_expired_claim_recovery_preserves_the_three_attempt_limit(tmp_path: Path) -> None:
    configured, clock, model = settings(tmp_path), Clock(), FollowUpDouble()
    repository = SQLiteRepository(configured.database_path)
    chat = seed_conversation(configured, clock)
    with TestClient(
        create_app(settings=configured, clock=clock, activity_follow_up_generator=model)
    ) as client:
        session = reported(client, command(client, command(client, offered(client, chat), "start"), "stop"))
        body = follow_up_body(session)
        request = ActivityFollowUpRequest.model_validate(body)
        for expected_attempts in range(1, 4):
            claimed, owns = repository.claim_activity_follow_up(
                OWNER, UUID(session["id"]), request, now=clock()
            )
            assert owns and claimed.follow_up_attempts == expected_attempts
            clock.now += timedelta(seconds=90)
        read = client.get(f"/v1/activity-sessions/{session['id']}", headers=OWNER_HEADERS)
        assert read.json()["follow_up_status"] == "failed"
        assert read.json()["follow_up_attempts"] == 3
        retry = client.post(
            f"/v1/activity-sessions/{session['id']}/follow-up", headers=OWNER_HEADERS, json=body
        )
        assert retry.status_code == 409
        assert model.calls == 0
        assert repository.get_activity_session(OWNER, claimed.id) == claimed
        assert repository.list_messages(OWNER, chat.id) == []
