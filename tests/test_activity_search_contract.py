"""The provider and runtime agree on search categories without disclosing prose."""

import json
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from journalpulse.activity_resources import (
    ACTIVITY_SEARCH_TOPICS,
    ActivityConstraints,
    general_search_topic,
    validate_general_search_topic,
)
from journalpulse.api import create_app
from journalpulse.config import Settings
from journalpulse.guided_action import GUIDED_ACTION_JSON_SCHEMA, ActivityDirective
from journalpulse.intelligence import OpenRouterConversationClient


def test_every_provider_search_category_compiles_with_all_public_constraint_fields():
    exposed = GUIDED_ACTION_JSON_SCHEMA["schema"]["properties"]["activity"]["properties"]
    assert set(exposed["search_topic"]["enum"]) == {*ACTIVITY_SEARCH_TOPICS, None}
    for topic in ACTIVITY_SEARCH_TOPICS:
        directive = ActivityDirective(
            move="propose", goal="move", selected_resource_id=None, search_topic=topic,
            constraints=ActivityConstraints(
                time_minutes=6, no_audio=True, no_video=True, seated=True, avoid_breath_focus=True,
            ),
        )
        query = general_search_topic(
            goal=directive.goal, style="move", constraints=directive.constraints,
            topic=directive.search_topic,
        )
        assert validate_general_search_topic(query) == query
        assert "6 minute silent text seated without breathing exercises" in query


@pytest.mark.parametrize("untrusted", [
    "brief corridor walking movement break", "meditation for PRIVATE_NAME_ALICE",
    "walking https://example.org/private", "focus 5551239876", "Walking", "",
])
def test_search_category_cannot_carry_model_prose_or_identifiers(untrusted: str):
    with pytest.raises(ValueError):
        ActivityDirective(
            move="propose", goal="move", selected_resource_id=None, search_topic=untrusted,
            constraints=ActivityConstraints(),
        )


def test_historical_walking_failure_is_delivered_and_saved_with_the_new_category(tmp_path: Path):
    configured = Settings(
        environment="test", database_path=tmp_path / "regression.db",
        resource_catalog_path=Path(__file__).resolve().parents[1] / "assets/resources/catalog.json",
        openrouter_api_key="test-only-key", openrouter_model="openai/gpt-6-luna",
        openrouter_base_url="https://openrouter.ai/api/v1", openrouter_zdr=True,
        openrouter_timeout_seconds=2, openrouter_max_attempts=1, supabase_url=None,
        supabase_anon_key=None, raw_text_retention_default=False,
    )
    calls = []
    output = {
        "reply": (
            "A little movement sounds like a good fit after a long call. "
            "Would you like me to look for a brief walking break that fits your six minutes?"
        ),
        "offer_action": True, "resource_intent": "move",
        "card_reason": "A short walking break fits your time and movement preference.",
        "summary": "The person welcomes a little movement after a long call.", "feelings": ["anxious"],
        "activity": {
            "move": "propose", "goal": "move", "selected_resource_id": None,
            "search_topic": "walking", "constraints": ActivityConstraints(time_minutes=6).model_dump(),
        },
    }

    def provider(request: httpx.Request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={
            "model": "openai/gpt-6-luna", "provider": "test-double",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(output)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 100},
        })

    model = OpenRouterConversationClient(
        configured, client=httpx.Client(transport=httpx.MockTransport(provider)),
    )
    headers = {"X-JournalPulse-User": str(uuid4())}
    with TestClient(create_app(settings=configured, conversation_client=model)) as client:
        created = client.post("/v1/conversations", headers=headers, json={"llm_consent": True})
        assert created.status_code == 201
        chat_id = created.json()["id"]
        result = client.post(f"/v1/conversations/{chat_id}/messages", headers=headers, json={
            "text": (
                "I am restless after a long call. I have six minutes and a safe corridor nearby; "
                "a little movement sounds welcome."
            ),
            "client_message_id": str(uuid4()),
        })
        assert result.status_code == 200, result.text
        current = result.json()["conversation"]
        assert current["activity_search_topic"] == "gentle movement gentle walking 6 minute"
        assert current["ready_for_action"] is True
        assert result.json()["assistant_message"]["content"] == output["reply"]
        loaded = client.get(f"/v1/conversations/{chat_id}", headers=headers)
        assert loaded.status_code == 200
        assert loaded.json()["conversation"]["activity_search_topic"] == current["activity_search_topic"]
    # The proposal makes one paid-model-shaped request; no search provider is
    # configured or contacted before the person gives explicit search consent.
    assert len(calls) == 1
    assert "corridor" not in current["activity_search_topic"]
    with pytest.raises(ValueError):
        validate_general_search_topic("brief corridor walking movement break")
