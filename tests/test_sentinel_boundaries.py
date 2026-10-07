"""Regression boundaries from the October Sentinel audit; no live services."""

from datetime import timedelta
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from journalpulse.activity_chat import chat_activity_context, validated_activity_update
from journalpulse.activity_models import ActivityFollowUpRequest
from journalpulse.activity_resources import ActivityConstraints
from journalpulse.api import create_app
from journalpulse.config import CHAT_PROVIDER_BUDGET_SECONDS
from journalpulse.domain import Goal
from journalpulse.persistence import SQLiteRepository
from test_activity_chat import OWNER, RecordingClient, completion, configured, conversation
from test_activity_sessions_repository import NOW, awaiting, report, setup
from test_inline_discovery import DiscoveryDouble
from test_inline_discovery import configured as search_settings
from test_research_beta_api import reflection_payload
from test_research_beta_api import settings as reflection_settings


def test_model_cannot_relax_saved_limits_to_offer_incompatible_resource(tmp_path):
    settings = configured(tmp_path)
    repo = SQLiteRepository(settings.database_path)
    current = repo.create_conversation(
        conversation(
            activity_constraints=ActivityConstraints(
                time_minutes=1,
                no_audio=True,
                no_video=True,
                seated=True,
                avoid_breath_focus=True,
            )
        )
    )
    context = chat_activity_context(settings, repo, current)
    candidate = next(item for item in context.candidates if item["id"] == "video_meditation_start_day")
    with pytest.raises(HTTPException) as error:
        validated_activity_update(
            completion(selected=candidate["id"], constraints=ActivityConstraints()), context, uuid4()
        )
    assert error.value.status_code == 502


def test_reflection_turn_cannot_erase_saved_activity_limits(tmp_path):
    settings = configured(tmp_path)
    repo = SQLiteRepository(settings.database_path)
    limits = ActivityConstraints(time_minutes=1, no_audio=True, no_video=True)
    current = repo.create_conversation(conversation(activity_constraints=limits))
    update = validated_activity_update(
        completion(constraints=ActivityConstraints()), chat_activity_context(settings, repo, current), uuid4()
    )
    assert update["activity_constraints"] == limits



def fixture_clock():
    # Fixtures are stamped at NOW. Using the real clock let the idle-chat sweep close them a
    # day later, so these checks failed (or passed on "chat closed" instead of the limit).
    return NOW

@pytest.mark.parametrize("mode", ["ai", "guided"])
def test_compatibility_goal_card_respects_saved_limits(tmp_path, mode):
    from journalpulse.activity_resources import (
        activity_resource_matches_constraints,
        resolve_activity_resource,
    )
    from journalpulse.domain import ConversationMode

    settings = configured(tmp_path)
    repo = SQLiteRepository(settings.database_path)
    limits = ActivityConstraints(no_video=True)
    current = repo.create_conversation(
        conversation(activity_constraints=limits).model_copy(update={"mode": ConversationMode(mode)})
    )
    model = RecordingClient(completion())
    with TestClient(create_app(clock=fixture_clock, settings=settings, conversation_client=model)) as client:
        response = client.post(
            f"/v1/conversations/{current.id}/messages", headers={"X-JournalPulse-User": str(OWNER)},
            json={"client_message_id": str(uuid4()), "text": "Help me settle.", "goal": "settle"},
        )
        assert response.status_code == 200, response.text
        state = response.json()["conversation"]
        for card in (state.get("card"), state.get("activity_card")):
            for action in card["actions"] if card else []:
                resolved = resolve_activity_resource(settings.resource_catalog_path, action["id"])
                assert resolved is not None and activity_resource_matches_constraints(resolved, limits)
        assert not state["ready_for_action"] or state.get("card") or state.get("activity_card")
        assert model.calls == []


@pytest.mark.parametrize("path", ["accept", "activity-sessions"])
def test_old_catalog_card_cannot_be_saved_against_current_limits(tmp_path, path):
    from journalpulse.conversations import PREVIEW_STATE, _catalog_card
    from journalpulse.policy import FixedBaselinePolicy

    settings = configured(tmp_path)
    card = _catalog_card(settings, FixedBaselinePolicy(), intent="ground", reason="Fictional earlier offer.",
                         message_id=uuid4(), state=PREVIEW_STATE, goal=Goal.SETTLE)
    assert card is not None
    video = next(item for item in card.actions if item["resource_type"] == "video")
    repo = SQLiteRepository(settings.database_path)
    current = repo.create_conversation(
        conversation(card=card, activity_constraints=ActivityConstraints(no_video=True))
    )
    body = {"action_id": video["id"], "expected_revision": 0} if path == "accept" else {
        "client_request_id": str(uuid4()), "resource_id": video["id"], "expected_conversation_revision": 0,
    }
    with TestClient(create_app(clock=fixture_clock, settings=settings)) as client:
        response = client.post(f"/v1/conversations/{current.id}/{path}",
                               headers={"X-JournalPulse-User": str(OWNER)}, json=body)
        assert response.status_code in {409, 422}, response.text
    assert repo.list_activity_sessions(OWNER, current.id) == []
    assert repo.get_conversation(OWNER, current.id).reflection_id is None


def test_explicit_person_constraint_update_is_used_and_bound_to_retry(tmp_path):
    settings = configured(tmp_path)
    repo = SQLiteRepository(settings.database_path)
    current = repo.create_conversation(
        conversation(activity_constraints=ActivityConstraints(time_minutes=1, no_video=True))
    )
    new_limits = ActivityConstraints(time_minutes=10)
    model = RecordingClient(completion(selected="video_meditation_start_day", constraints=new_limits))
    with TestClient(create_app(clock=fixture_clock, settings=settings, conversation_client=model)) as client:
        body = {
            "client_message_id": str(uuid4()),
            "text": "Please use the activity limits I selected.",
            "activity_constraints": new_limits.model_dump(),
        }
        url = f"/v1/conversations/{current.id}/messages"
        headers = {"X-JournalPulse-User": str(OWNER)}
        response = client.post(url, headers=headers, json=body)
        assert response.status_code == 200, response.text
        assert model.calls[-1][1].constraints == new_limits
        assert (
            response.json()["user_message"]["request_inputs"]["activity_constraints"]
            == new_limits.model_dump()
        )
        assert client.post(url, headers=headers, json=body).status_code == 200
        changed = {**body, "activity_constraints": ActivityConstraints(time_minutes=1).model_dump()}
        assert client.post(url, headers=headers, json=changed).status_code == 409
        assert len(model.calls) == 1


@pytest.mark.parametrize("requested_goal", [None, "connect"])
def test_saved_inline_resource_keeps_exact_search_goal(tmp_path, requested_goal):
    settings = search_settings(tmp_path)
    repo = SQLiteRepository(settings.database_path)
    current = repo.create_conversation(conversation(activity_goal=Goal.MOVE))
    headers = {"X-JournalPulse-User": str(OWNER)}
    app = create_app(clock=fixture_clock, settings=settings, discovery_client=DiscoveryDouble())
    with TestClient(app) as client:
        body = {"expected_revision": 0, "llm_consent": True}
        if requested_goal is not None:
            body["goal"] = requested_goal
        response = client.post(f"/v1/conversations/{current.id}/discover", headers=headers, json=body)
        assert response.status_code == 200, response.text
        offer = response.json()["offers"][0]
        saved = client.post(
            f"/v1/conversations/{current.id}/activity-sessions",
            headers=headers,
            json={
                "client_request_id": str(uuid4()),
                "expected_conversation_revision": 0,
                "resource_id": offer["resource"]["id"],
                "resource_token": offer["resource_token"],
            },
        )
        assert saved.status_code == 201, saved.text
        assert saved.json()["goal"] == (requested_goal or "move")
        assert repo.get_activity_session(OWNER, UUID(saved.json()["id"])).goal.value == (
            requested_goal or "move"
        )


def test_followup_cannot_be_reclaimed_during_valid_provider_budget(tmp_path):
    from test_activity_sessions_repository import OWNER as SESSION_OWNER

    repo, chat, offered, _ = setup(tmp_path)
    saved, _ = report(repo, awaiting(repo, offered))
    moment = NOW + timedelta(seconds=122)
    request = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=saved.revision, expected_conversation_revision=0
    )
    claimed, owns = repo.claim_activity_follow_up(SESSION_OWNER, saved.id, request, now=moment)
    assert owns
    elapsed = moment + timedelta(seconds=CHAT_PROVIDER_BUDGET_SECONDS - 1)
    duplicate = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=claimed.revision, expected_conversation_revision=0
    )
    current, owns_again = repo.claim_activity_follow_up(SESSION_OWNER, saved.id, duplicate, now=elapsed)
    assert not owns_again
    assert current.follow_up_attempts == 1
    assert current.follow_up_lease_until > elapsed


def test_saved_reflection_receipt_survives_exhausted_generation_limit(tmp_path):
    from dataclasses import replace

    settings = replace(reflection_settings(tmp_path), analysis_rate_limit_per_minute=1)
    payload = reflection_payload(client_request_id=str(uuid4()))
    with TestClient(create_app(clock=fixture_clock, settings=settings)) as client:
        first = client.post("/v1/reflections", json=payload)
        retry = client.post("/v1/reflections", json=payload)
        assert first.status_code == retry.status_code == 201
        assert retry.json() == first.json()


def test_search_receipt_cannot_cross_recreated_chat_identity(tmp_path):
    settings = search_settings(tmp_path)
    headers = {"X-JournalPulse-User": str(OWNER)}
    app = create_app(clock=fixture_clock, settings=settings, discovery_client=DiscoveryDouble())
    with TestClient(app) as client:
        cid = str(uuid4())
        body = {"client_request_id": cid, "llm_consent": True}
        assert client.post("/v1/conversations", headers=headers, json=body).status_code == 201
        found = client.post(
            f"/v1/conversations/{cid}/discover",
            headers=headers,
            json={"expected_revision": 0, "llm_consent": True},
        ).json()["offers"][0]
        assert client.delete(f"/v1/conversations/{cid}", headers=headers).status_code == 204
        assert client.post("/v1/conversations", headers=headers, json=body).status_code == 409
        saved = client.post(
            f"/v1/conversations/{cid}/activity-sessions",
            headers=headers,
            json={
                "client_request_id": str(uuid4()),
                "expected_conversation_revision": 0,
                "resource_id": found["resource"]["id"],
                "resource_token": found["resource_token"],
            },
        )
        assert saved.status_code == 404


def test_late_search_cannot_offer_results_to_replacement_chat(tmp_path):
    settings = search_settings(tmp_path)
    headers = {"X-JournalPulse-User": str(OWNER)}

    class ReplacingSearch(DiscoveryDouble):
        replace = None

        def search(self, payload):
            self.replace()
            return super().search(payload)

    search = ReplacingSearch()
    with TestClient(create_app(clock=fixture_clock, settings=settings, discovery_client=search)) as client:
        cid = str(uuid4())
        body = {"client_request_id": cid, "llm_consent": True}
        assert client.post("/v1/conversations", headers=headers, json=body).status_code == 201

        def replace():
            assert client.delete(f"/v1/conversations/{cid}", headers=headers).status_code == 204
            assert client.post("/v1/conversations", headers=headers, json=body).status_code == 409

        search.replace = replace
        response = client.post(
            f"/v1/conversations/{cid}/discover",
            headers=headers,
            json={"expected_revision": 0, "llm_consent": True},
        )
        assert response.status_code == 404
        assert len(search.requests) == 1


def test_late_followup_cannot_cross_recreated_parent_chat(tmp_path):
    from test_activity_sessions_repository import OWNER as SESSION_OWNER

    repo, chat, offered, _ = setup(tmp_path)
    saved, _ = report(repo, awaiting(repo, offered))
    request = ActivityFollowUpRequest(
        client_request_id=uuid4(), expected_revision=saved.revision, expected_conversation_revision=0
    )
    claimed, owns = repo.claim_activity_follow_up(SESSION_OWNER, saved.id, request, now=NOW)
    assert owns
    assert repo.delete_conversation(SESSION_OWNER, chat.id)
    with pytest.raises(ValueError, match="deleted"):
        repo.create_conversation(chat.model_copy(update={"incarnation_id": uuid4()}))
    replacement = repo.create_conversation(chat.model_copy(update={"id": uuid4(), "incarnation_id": uuid4()}))
    from journalpulse.activity_lifecycle import ActivityNotFound
    with pytest.raises(ActivityNotFound):
        repo.finish_activity_follow_up(
            SESSION_OWNER,
            claimed.id,
            request_id=request.client_request_id,
            expected_revision=claimed.revision,
            expected_conversation_revision=0,
            expected_session_created_at=claimed.created_at,
            reply="Fictional stale report reply.",
            model_run=None,
            now=NOW,
        )
    assert repo.list_messages(SESSION_OWNER, replacement.id) == []
