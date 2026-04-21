"""CalendarAPI acceptance tests — Phase 3."""
from __future__ import annotations

import pytest

from models import DriftEvent
from tools.calendar import CalendarAPI


def _fresh_calendar(seed_events: list[dict] | None = None) -> CalendarAPI:
    seed = {"events": list(seed_events or [])}
    return CalendarAPI(seed_data=seed)


def _seeded_calendar() -> CalendarAPI:
    return _fresh_calendar([
        {
            "event_id": "evt_1",
            "title": "Kickoff",
            "start": "2026-04-25T10:00:00Z",
            "end": "2026-04-25T11:00:00Z",
            "attendees": ["a@x.com"],
            "status": "confirmed",
        },
        {
            "event_id": "evt_2",
            "title": "Review",
            "start": "2026-04-26T10:00:00Z",
            "end": "2026-04-26T11:00:00Z",
            "attendees": ["b@x.com"],
            "status": "confirmed",
        },
        {
            "event_id": "evt_3",
            "title": "Retro",
            "start": "2026-04-27T10:00:00Z",
            "end": "2026-04-27T11:00:00Z",
            "attendees": ["c@x.com"],
            "status": "confirmed",
        },
    ])


def test_baseline_create_event() -> None:
    cal = _fresh_calendar()
    resp = cal.call(
        "create_event",
        {
            "title": "Launch",
            "start": "2026-04-25T10:00:00Z",
            "end": "2026-04-25T11:00:00Z",
            "attendees": ["a@x.com", "b@x.com"],
        },
    )
    assert resp.ok is True
    assert resp.status == 200
    assert resp.body is not None
    assert "attendees" in resp.body
    assert "participants" not in resp.body
    assert resp.body["status"] == "confirmed"
    assert resp.body["attendees"] == ["a@x.com", "b@x.com"]
    assert resp.body["event_id"].startswith("evt_")


def test_drift_field_rename_create_event() -> None:
    cal = _fresh_calendar()
    event = DriftEvent(
        tool="calendar",
        endpoint="create_event",
        kind="field_rename",
        fires_at_step=1,
        details={"from": "attendees", "to": "participants"},
    )
    cal.apply_drift(event)

    schema = cal.get_schema("create_event")
    assert "participants" in schema["params"]
    assert "attendees" not in schema["params"]
    assert "participants" in schema["required"]
    assert "attendees" not in schema["required"]
    assert "participants" in schema["response_shape"]
    assert "attendees" not in schema["response_shape"]

    bad = cal.call(
        "create_event",
        {
            "title": "Launch",
            "start": "2026-04-25T10:00:00Z",
            "end": "2026-04-25T11:00:00Z",
        },
    )
    assert bad.ok is False
    assert bad.status == 400
    assert bad.error is not None
    assert "participants" in bad.error

    ok = cal.call(
        "create_event",
        {
            "title": "Launch",
            "start": "2026-04-25T10:00:00Z",
            "end": "2026-04-25T11:00:00Z",
            "participants": [
                {"email": "a@x.com", "role": "required"},
                {"email": "b@x.com", "role": "optional"},
            ],
        },
    )
    assert ok.ok is True
    assert ok.status == 200
    assert ok.body is not None
    assert "participants" in ok.body
    assert "attendees" not in ok.body
    assert isinstance(ok.body["participants"], list)
    assert ok.body["participants"][0]["email"] == "a@x.com"


def test_drift_tool_removal_delete_event() -> None:
    cal = _seeded_calendar()
    event = DriftEvent(
        tool="calendar",
        endpoint="delete_event",
        kind="tool_removal",
        fires_at_step=1,
        details={"replacement": "update_event(status=cancelled)"},
    )
    cal.apply_drift(event)

    gone = cal.call("delete_event", {"event_id": "evt_1"})
    assert gone.ok is False
    assert gone.status == 410

    updated = cal.call("update_event", {"event_id": "evt_1", "status": "cancelled"})
    assert updated.ok is True
    assert updated.status == 200
    assert updated.body is not None
    assert updated.body["status"] == "cancelled"

    still_there = [e for e in cal.events if e["event_id"] == "evt_1"]
    assert len(still_there) == 1
    assert still_there[0]["status"] == "cancelled"


def test_update_event_not_found() -> None:
    cal = _seeded_calendar()
    resp = cal.call("update_event", {"event_id": "nonexistent", "status": "cancelled"})
    assert resp.ok is False
    assert resp.status == 404


def test_list_events() -> None:
    cal = _seeded_calendar()

    wide = cal.call(
        "list_events",
        {"date_from": "2026-04-25T00:00:00Z", "date_to": "2026-04-28T00:00:00Z"},
    )
    assert wide.ok is True
    assert wide.body is not None
    assert len(wide.body["events"]) == 3

    narrow = cal.call(
        "list_events",
        {"date_from": "2026-04-25T00:00:00Z", "date_to": "2026-04-25T23:59:59Z"},
    )
    assert narrow.ok is True
    assert narrow.body is not None
    assert len(narrow.body["events"]) == 1
    assert narrow.body["events"][0]["event_id"] == "evt_1"


def test_unknown_drift_kind_raises() -> None:
    cal = _fresh_calendar()
    event = DriftEvent(
        tool="calendar",
        endpoint="create_event",
        kind="rate_limit_tightening",
        fires_at_step=1,
        details={},
    )
    with pytest.raises(ValueError):
        cal.apply_drift(event)
