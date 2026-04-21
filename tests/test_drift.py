"""DriftInjector acceptance tests — Phase 3."""
from __future__ import annotations

from drift import DriftInjector
from models import DriftEvent, EpisodeState
from tools.calendar import CalendarAPI
from tools.mail import MailAPI


def _make_state(drifts: list[DriftEvent]) -> EpisodeState:
    return EpisodeState(
        episode_id="ep-test",
        task_id="test-task",
        difficulty="easy",
        max_steps=10,
        token_budget=4000,
        token_budget_remaining=4000,
        drift_plan=drifts,
        ground_truth_final_state={},
    )


def _make_tools() -> dict:
    mail = MailAPI(seed_data={"messages": [
        {"id": "m1", "from": "a@x.com", "to": "u@org.com",
         "subject": "s", "body": "b", "folder": "inbox"}
    ]})
    cal = CalendarAPI(seed_data={"events": [
        {"event_id": "evt_1", "title": "x",
         "start": "2026-04-25T10:00:00Z", "end": "2026-04-25T11:00:00Z",
         "attendees": ["a@x.com"], "status": "confirmed"}
    ]})
    return {"mail": mail, "calendar": cal}


def test_single_drift_fires_at_step() -> None:
    drift = DriftEvent(
        tool="mail", endpoint="list_messages", kind="field_rename",
        fires_at_step=3, details={},
    )
    state = _make_state([drift])
    tools = _make_tools()

    state.step = 1
    assert DriftInjector.tick(state, tools) == []
    assert "messages" in tools["mail"].active_schemas["list_messages"].response_shape

    state.step = 3
    fired = DriftInjector.tick(state, tools)
    assert len(fired) == 1
    assert fired[0] is drift
    mail_shape = tools["mail"].active_schemas["list_messages"].response_shape
    assert "items" in mail_shape
    assert "messages" not in mail_shape

    state.step = 4
    assert DriftInjector.tick(state, tools) == []


def test_multiple_drifts_different_steps() -> None:
    mail_drift = DriftEvent(
        tool="mail", endpoint="send_message", kind="endpoint_deprecation",
        fires_at_step=2, details={"replacement": "messages.send"},
    )
    cal_drift = DriftEvent(
        tool="calendar", endpoint="create_event", kind="field_rename",
        fires_at_step=5, details={},
    )
    state = _make_state([mail_drift, cal_drift])
    tools = _make_tools()

    state.step = 2
    fired = DriftInjector.tick(state, tools)
    assert [e.tool for e in fired] == ["mail"]
    assert "send_message" not in tools["mail"].active_schemas
    assert "messages.send" in tools["mail"].active_schemas
    assert "attendees" in tools["calendar"].active_schemas["create_event"].params

    state.step = 3
    assert DriftInjector.tick(state, tools) == []

    state.step = 5
    fired = DriftInjector.tick(state, tools)
    assert [e.tool for e in fired] == ["calendar"]
    cal_params = tools["calendar"].active_schemas["create_event"].params
    assert "participants" in cal_params
    assert "attendees" not in cal_params

    state.step = 6
    assert DriftInjector.tick(state, tools) == []


def test_drift_not_fired_before_step() -> None:
    drift = DriftEvent(
        tool="calendar", endpoint="create_event", kind="field_rename",
        fires_at_step=5, details={},
    )
    state = _make_state([drift])
    tools = _make_tools()

    state.step = 2
    fired = DriftInjector.tick(state, tools)
    assert fired == []
    cal_params = tools["calendar"].active_schemas["create_event"].params
    assert "attendees" in cal_params
    assert "participants" not in cal_params


def test_detected_by_agent_unchanged_by_tick() -> None:
    drift = DriftEvent(
        tool="mail", endpoint="list_messages", kind="field_rename",
        fires_at_step=1, details={},
    )
    state = _make_state([drift])
    tools = _make_tools()
    assert drift.detected_by_agent is False

    state.step = 1
    fired = DriftInjector.tick(state, tools)
    assert len(fired) == 1
    assert fired[0].detected_by_agent is False
    assert state.drift_plan[0].detected_by_agent is False
