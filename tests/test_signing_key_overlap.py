from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from journalpulse.activity_resources import issue_resource_token, verify_resource_token
from test_activity_resources import discovered
from test_research_beta_api import settings

OLD = "old-verification-only-key-0123456789abcdef"
NEW = "new-verification-only-key-0123456789abcdef"


def test_old_resource_receipt_survives_only_the_explicit_overlap(tmp_path):
    now = datetime(2026, 10, 14, 16, tzinfo=UTC)
    owner, chat = uuid4(), uuid4()
    old = replace(settings(tmp_path), write_signing_key=OLD)
    rotated = replace(old, write_signing_key=NEW, write_signing_key_previous=OLD,
                      write_signing_key_previous_expires_at=(now + timedelta(minutes=1)).isoformat())
    token = issue_resource_token(old, user_id=owner, conversation_id=chat,
                                conversation_revision=0, resource=discovered(), now=now)
    def verify(config, time, user=owner):
        return verify_resource_token(config, token, user_id=user, conversation_id=chat,
                                     conversation_revision=0, now=time)
    assert verify(rotated, now)
    with pytest.raises(ValueError):
        verify(rotated, now + timedelta(minutes=1))
    with pytest.raises(ValueError):
        verify(replace(rotated, write_signing_key_previous=None), now)
    with pytest.raises(ValueError):
        verify(rotated, now, uuid4())
    for expiry in (None, "bad", "2026-10-15T16:00:00"):
        invalid = replace(rotated, write_signing_key_previous_expires_at=expiry)
        assert "previous_signing_key_invalid" in invalid.configuration_issues
        with pytest.raises(ValueError):
            verify(invalid, now)


def test_new_receipts_are_signed_only_with_current_key(tmp_path):
    now = datetime.now(UTC)
    owner, chat = uuid4(), uuid4()
    rotated = replace(settings(tmp_path), write_signing_key=NEW, write_signing_key_previous=OLD,
                      write_signing_key_previous_expires_at=(now + timedelta(minutes=10)).isoformat())
    token = issue_resource_token(rotated, user_id=owner, conversation_id=chat,
                                conversation_revision=0, resource=discovered(), now=now)
    current_only = replace(rotated, write_signing_key_previous=None)
    assert verify_resource_token(current_only, token, user_id=owner, conversation_id=chat,
                                 conversation_revision=0, now=now)
    with pytest.raises(ValueError):
        verify_resource_token(replace(current_only, write_signing_key=OLD), token, user_id=owner,
                              conversation_id=chat, conversation_revision=0, now=now)
