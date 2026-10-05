"""Pure session transitions shared by the local repository and lifecycle tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from math import ceil

from .activity_models import ActivitySession, ActivityStatus


class ActivityNotFound(ValueError):
    pass


class ActivityConflict(ValueError):
    pass


def remaining_seconds(session: ActivitySession, now: datetime) -> int:
    if session.status == ActivityStatus.ACTIVE and session.expires_at is not None:
        return min(session.duration_seconds, max(0, ceil((session.expires_at - now).total_seconds())))
    return session.remaining_seconds


def apply_activity_command(session: ActivitySession, command: str, now: datetime) -> ActivitySession:
    """The server clock owns time; a browser interval only refreshes its display."""
    changes: dict = {"updated_at": now, "revision": session.revision + 1}
    if command == "start" and session.status == ActivityStatus.OFFERED:
        changes.update(status=ActivityStatus.ACTIVE, started_at=now)
        if session.resource.format == "timer":
            changes["expires_at"] = now + timedelta(seconds=session.remaining_seconds)
    elif (
        command == "pause" and session.status == ActivityStatus.ACTIVE and session.resource.format == "timer"
    ):
        remaining = remaining_seconds(session, now)
        if remaining == 0:
            changes.update(
                status=ActivityStatus.AWAITING_REPORT,
                remaining_seconds=0,
                expires_at=None,
                check_in_issued=True,
            )
        else:
            changes.update(status=ActivityStatus.PAUSED, remaining_seconds=remaining, expires_at=None)
    elif command == "resume" and session.status == ActivityStatus.PAUSED:
        changes.update(
            status=ActivityStatus.ACTIVE, expires_at=now + timedelta(seconds=session.remaining_seconds)
        )
    elif command == "expire" and session.status == ActivityStatus.ACTIVE and session.expires_at is not None:
        if now < session.expires_at:
            raise ActivityConflict("The timer has not finished yet")
        changes.update(
            status=ActivityStatus.AWAITING_REPORT, remaining_seconds=0, expires_at=None, check_in_issued=True
        )
    elif command == "finish_early" and session.status in {ActivityStatus.ACTIVE, ActivityStatus.PAUSED}:
        changes.update(
            status=ActivityStatus.AWAITING_REPORT,
            remaining_seconds=remaining_seconds(session, now),
            expires_at=None,
            check_in_issued=True,
        )
    elif command == "decline" and session.status == ActivityStatus.OFFERED:
        changes.update(status=ActivityStatus.DECLINED, expires_at=None)
    elif command == "stop" and session.status in {
        ActivityStatus.OFFERED,
        ActivityStatus.ACTIVE,
        ActivityStatus.PAUSED,
        ActivityStatus.AWAITING_REPORT,
    }:
        changes.update(
            status=ActivityStatus.STOPPED,
            remaining_seconds=remaining_seconds(session, now),
            expires_at=None,
            check_in_issued=session.started_at is not None,
        )
    else:
        raise ActivityConflict("This activity changed; refresh before trying again")
    return ActivitySession.model_validate({**session.model_dump(), **changes})


def invalidated_activity(session: ActivitySession, now: datetime, *, clear_text: bool) -> ActivitySession:
    """Closing unretained chats removes derived free text, including outcome notes."""
    changes: dict = {
        "revision": session.revision + 1,
        "updated_at": now,
        "expires_at": None,
        # Conversation stop/listen/support is withdrawal, not a request for a check-in.
        "check_in_issued": False,
        "follow_up_request_id": None,
        "follow_up_lease_until": None,
    }
    if session.status in {
        ActivityStatus.OFFERED,
        ActivityStatus.ACTIVE,
        ActivityStatus.PAUSED,
        ActivityStatus.AWAITING_REPORT,
    }:
        changes["status"] = ActivityStatus.STOPPED
    if session.follow_up_status in {"pending", "generating"}:
        changes["follow_up_status"] = "failed"
    if clear_text:
        changes.update(recommendation_reason=None, follow_up_reply=None)
        if session.report is not None:
            changes["report"] = session.report.model_copy(update={"note": None})
    return ActivitySession.model_validate({**session.model_dump(), **changes})
