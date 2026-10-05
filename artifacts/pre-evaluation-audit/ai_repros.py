"""Local synthetic audit cases; no network, provider, deployment, or persistent user data."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from fastapi.testclient import TestClient

ROOT = Path('/workspace/JournalPulse')
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]

from journalpulse.activity_chat import generate_activity_follow_up
from journalpulse.activity_models import ActivityReport, ActivitySession, ActivityStatus
from journalpulse.api import create_app
from journalpulse.domain import Goal
from journalpulse.intelligence import OpenRouterConversationClient, build_guided_request
from journalpulse.persistence import SQLiteRepository
from test_activity_chat import RecordingClient, activity, configured, conversation
from test_conversations_api import chat_settings, say, start
from test_guided_action_prompt import payload


def follow_up_resource_context() -> dict:
    with TemporaryDirectory(prefix='jp-ai-audit-', dir='/tmp') as directory:
        settings = configured(Path(directory))
        repository = SQLiteRepository(settings.database_path)
        current = repository.create_conversation(conversation())
        report = ActivityReport(participation='partial', fit='good', state_change='same')
        first_session = activity(current, status=ActivityStatus.COMPLETED, report=report, goal=Goal.SETTLE)
        second_data = first_session.model_dump(mode='json')
        second_data.update({
            'goal': 'connect', 'duration_seconds': 0, 'remaining_seconds': 0,
            'resource': {
                'id': 'search_876', 'title': 'Fictional assertive communication workbook',
                'url': 'https://example.com/communication', 'provider': 'Example',
                'format': 'external', 'kind': 'reading', 'provenance': 'search_snippet',
            },
            'selection': {
                'selection_source': 'search', 'recommended_resource_id': 'search_876',
                'selected_resource_id': 'search_876',
            },
        })
        second_session = ActivitySession.model_validate(second_data)
        requests = []
        for session in (first_session, second_session):
            # A read-only repository response fixture supplies a real validated
            # current session; actual request assembly remains production code.
            repository.list_activity_sessions = lambda *args, session=session: [session]
            client = RecordingClient()
            generate_activity_follow_up(settings, repository, current, [], session, client)
            requests.append(build_guided_request(settings, *client.calls[0]))
        return {
            'case': 'follow_up_resource_context',
            'resource_a': first_session.resource.title,
            'goal_a': first_session.goal.value,
            'resource_b': second_session.resource.title,
            'goal_b': second_session.goal.value,
            'requests_identical': requests[0] == requests[1],
            'resource_b_title_in_request': second_session.resource.title in json.dumps(requests[1]),
            'current_activity_state': json.loads(requests[1]['messages'][1]['content'])['activity_context']['activity_state'],
        }


def empty_feeling_correction() -> dict:
    with TemporaryDirectory(prefix='jp-ai-audit-', dir='/tmp') as directory:
        settings = chat_settings(Path(directory))
        calls = []

        def fake_provider(request: httpx.Request) -> httpx.Response:
            calls.append(json.loads(request.content))
            output = payload(move='reflect', goal=None, selected_resource_id=None)
            output.update({
                'reply': 'We can use your current account.', 'offer_action': False,
                'card_reason': '', 'feelings': ['anxious'] if len(calls) == 1 else [],
            })
            return httpx.Response(200, json={
                'model': 'synthetic-audit', 'provider': 'OpenAI',
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(output)}}],
            })

        with httpx.Client(transport=httpx.MockTransport(fake_provider)) as transport:
            model = OpenRouterConversationClient(settings, client=transport)
            with TestClient(create_app(settings=settings, conversation_client=model)) as client:
                chat = start(client)
                first = say(client, chat['id'], 'I am anxious about the fictional presentation.')
                second = say(client, chat['id'], 'That feeling has passed. Please clear that label; I do not want a feeling suggested.')
                return {
                    'case': 'empty_feeling_correction',
                    'statuses': [first.status_code, second.status_code],
                    'first_feelings': first.json()['conversation']['feelings'],
                    'second_model_feelings': [],
                    'second_stored_feelings': second.json()['conversation']['feelings'],
                    'actual_guided_parser_exercised': 'activity' in calls[-1]['response_format']['json_schema']['schema']['properties'],
                    'external_calls': 0,
                }


if __name__ == '__main__':
    print(json.dumps([follow_up_resource_context(), empty_feeling_correction()], indent=2))
