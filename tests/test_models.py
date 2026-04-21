"""Validation tests for models.py — Phase 1 acceptance."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from models import (
    Action,
    CompleteParams,
    DriftEvent,
    DriftReportParams,
    EpisodeState,
    HistoryStep,
    InspectParams,
    Observation,
    RetryParams,
    RewardBreakdown,
    ToolCallParams,
    ToolResponse,
)


def test_imports() -> None:
    classes = [
        ToolCallParams, InspectParams, RetryParams, DriftReportParams, CompleteParams,
        Action, ToolResponse, HistoryStep, Observation, DriftEvent, RewardBreakdown,
        EpisodeState,
    ]
    assert len(classes) == 12
    for cls in classes:
        assert hasattr(cls, "model_validate")
        assert hasattr(cls, "model_dump_json")


def test_action_serialization() -> None:
    actions = [
        Action(
            type="call_tool",
            tool_call=ToolCallParams(
                tool="mail", endpoint="send_message",
                params={"to": "a@b.com", "subject": "hi", "body": "hello"},
            ),
        ),
        Action(type="inspect_schema", inspect=InspectParams(tool="calendar")),
        Action(
            type="retry_with_variant",
            retry=RetryParams(
                tool="crm", endpoint="search_contacts",
                params={"email_address": "x@y.com"},
            ),
        ),
        Action(
            type="report_drift",
            report=DriftReportParams(
                tool="calendar", drift_kind="field_rename",
                description="attendees renamed to participants",
            ),
        ),
        Action(type="complete_task", complete=CompleteParams(summary="all done")),
    ]
    assert len(actions) == 5
    for original in actions:
        payload = original.model_dump_json()
        assert isinstance(payload, str) and len(payload) > 0
        roundtrip = Action.model_validate_json(payload)
        assert roundtrip == original


def test_observation_validation() -> None:
    step_action = Action(type="inspect_schema", inspect=InspectParams(tool="mail"))
    resp = ToolResponse(ok=True, status=200, body={"messages": []}, error=None)
    hist = [HistoryStep(step=0, action=step_action, response=resp,
                        reward_breakdown={"shaped_total": 0.10})]
    obs = Observation(
        episode_id="ep-1",
        task_id="E1_onboard_new_hire",
        difficulty="easy",
        step=1,
        max_steps=8,
        token_budget_remaining=3900,
        task_description="Send welcome email and create orientation event.",
        success_criteria=["welcome email sent", "calendar event created"],
        tool_schemas={"mail": {"send_message": {"required": ["to", "subject", "body"]}}},
        known_state={"mail.sent_count": 0},
        history=hist,
        last_response=resp,
        drift_events_visible=[{"tool": "calendar", "kind": "field_rename"}],
        done=False,
        feedback="ok",
    )
    assert obs.step == 1
    assert obs.history[0].response is not None
    assert obs.last_response is not None and obs.last_response.ok is True


def test_episode_state_with_drifts() -> None:
    drifts = [
        DriftEvent(
            tool="mail", endpoint="send_message", kind="endpoint_deprecation",
            fires_at_step=1, details={"replacement": "messages.send"},
        ),
        DriftEvent(
            tool="calendar", endpoint="create_event", kind="field_rename",
            fires_at_step=3, details={"from": "attendees", "to": "participants"},
        ),
    ]
    state = EpisodeState(
        episode_id="ep-xyz",
        task_id="E1_onboard_new_hire",
        difficulty="easy",
        max_steps=8,
        token_budget=4000,
        token_budget_remaining=4000,
        drift_plan=drifts,
        ground_truth_final_state={"mail.sent_count": 1, "calendar.events_count": 1},
    )
    assert state.step == 0
    assert len(state.drift_plan) == 2
    assert state.drift_plan[0].tool == "mail"
    assert state.drift_plan[1].fires_at_step == 3
    assert state.drift_plan[0].detected_by_agent is False
    assert state.agent_state == {}
    assert state.history == []
    assert state.cumulative_reward == 0.0


def test_reward_breakdown_defaults() -> None:
    r = RewardBreakdown()
    assert r.task_completion == 0.0
    assert r.drift_detection == 0.0
    assert r.adaptation_quality == 0.0
    assert r.efficiency == 0.0
    assert r.catastrophic_gate == 1.0
    assert r.correct_final_gate == 1.0
    assert r.step_shaping == 0.0
    assert r.shaped_total == 0.0
    assert r.binary == 0.0


def test_literal_enforcement() -> None:
    with pytest.raises(ValidationError):
        Action(type="teleport")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        InspectParams(tool="salesforce")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        DriftReportParams(
            tool="mail",
            drift_kind="meteor_strike",  # type: ignore[arg-type]
            description="not a real drift kind",
        )
