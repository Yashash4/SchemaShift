"""SchemaShiftEnvironment acceptance tests — Phase 6 (end-to-end RL loop)."""
from __future__ import annotations

import pytest

from models import (
    Action,
    CompleteParams,
    DriftReportParams,
    InspectParams,
    RetryParams,
    ToolCallParams,
)
from server.environment import SchemaShiftEnvironment


# ──────────────────────────────────────────────────────────────────
# Test 1 — Round 1 bug prevention
# ──────────────────────────────────────────────────────────────────

def test_step_before_reset_raises() -> None:
    env = SchemaShiftEnvironment()
    action = Action(type="complete_task", complete=CompleteParams(summary="noop"))
    with pytest.raises(RuntimeError) as excinfo:
        env.step(action)
    assert "reset" in str(excinfo.value).lower()


# ──────────────────────────────────────────────────────────────────
# Test 2 — reset returns a valid observation
# ──────────────────────────────────────────────────────────────────

def test_reset_returns_valid_observation() -> None:
    env = SchemaShiftEnvironment()
    obs = env.reset("E1_onboard_new_hire")
    assert obs.task_id == "E1_onboard_new_hire"
    assert obs.step == 0
    assert obs.max_steps == 8
    assert "mail" in obs.tool_schemas
    assert "calendar" in obs.tool_schemas
    assert "crm" not in obs.tool_schemas
    assert obs.done is False
    assert obs.difficulty == "easy"
    assert len(obs.success_criteria) >= 1


# ──────────────────────────────────────────────────────────────────
# Test 3 — reset on unknown task raises
# ──────────────────────────────────────────────────────────────────

def test_reset_unknown_task_raises() -> None:
    env = SchemaShiftEnvironment()
    with pytest.raises(ValueError):
        env.reset("nonexistent_task")


# ──────────────────────────────────────────────────────────────────
# Test 4 — call_tool success updates agent_state
# ──────────────────────────────────────────────────────────────────

def test_call_tool_success_updates_state() -> None:
    env = SchemaShiftEnvironment()
    env.reset("E1_onboard_new_hire")

    obs, reward = env.step(Action(
        type="call_tool",
        tool_call=ToolCallParams(
            tool="mail",
            endpoint="send_message",
            params={
                "to": "priya@company.com",
                "subject": "Welcome!",
                "body": "Welcome to the team.",
            },
        ),
    ))
    assert obs.last_response is not None
    assert obs.last_response.ok is True
    assert obs.known_state["mail.sent_count"] == 1
    assert obs.known_state["mail.last_sent_to"] == "priya@company.com"
    assert obs.known_state["mail.last_subject_contains_welcome"] is True


# ──────────────────────────────────────────────────────────────────
# Test 5 — FULL E1 episode with drift → inspect → retry → report → complete
# ──────────────────────────────────────────────────────────────────

def test_e1_full_episode_with_adaptation() -> None:
    env = SchemaShiftEnvironment()
    env.reset("E1_onboard_new_hire")

    # Step 1: send welcome email (pre-drift)
    obs, r1 = env.step(Action(
        type="call_tool",
        tool_call=ToolCallParams(
            tool="mail", endpoint="send_message",
            params={"to": "priya@company.com", "subject": "Welcome aboard!",
                    "body": "Welcome to the team."},
        ),
    ))
    assert obs.last_response is not None and obs.last_response.ok is True

    # Step 2: inspect calendar (pre-drift)
    obs, r2 = env.step(Action(
        type="inspect_schema", inspect=InspectParams(tool="calendar"),
    ))
    assert obs.last_response is not None and obs.last_response.ok is True

    # Step 3: drift fires at state.step=3; call with stale attendees fails
    obs, r3 = env.step(Action(
        type="call_tool",
        tool_call=ToolCallParams(
            tool="calendar", endpoint="create_event",
            params={"title": "New Hire Orientation",
                    "start": "2026-04-27T10:00:00Z",
                    "end": "2026-04-27T11:00:00Z",
                    "attendees": ["priya@company.com", "alex@company.com"]},
        ),
    ))
    assert obs.last_response is not None and obs.last_response.ok is False

    # Step 4: inspect calendar (now shows participants schema)
    obs, r4 = env.step(Action(
        type="inspect_schema", inspect=InspectParams(tool="calendar"),
    ))
    assert obs.last_response is not None and obs.last_response.ok is True
    cal_schema = obs.tool_schemas["calendar"]["create_event"]
    assert "participants" in cal_schema["params"]

    # Step 5: retry with participants format
    obs, r5 = env.step(Action(
        type="retry_with_variant",
        retry=RetryParams(
            tool="calendar", endpoint="create_event",
            params={"title": "New Hire Orientation",
                    "start": "2026-04-27T10:00:00Z",
                    "end": "2026-04-27T11:00:00Z",
                    "participants": [
                        {"email": "priya@company.com", "role": "required"},
                        {"email": "alex@company.com", "role": "required"},
                    ]},
        ),
    ))
    assert obs.last_response is not None and obs.last_response.ok is True

    # Step 6: report drift
    obs, r6 = env.step(Action(
        type="report_drift",
        report=DriftReportParams(
            tool="calendar", drift_kind="field_rename",
            description="create_event attendees renamed to participants",
        ),
    ))

    # Step 7: complete
    obs, r7 = env.step(Action(
        type="complete_task",
        complete=CompleteParams(
            summary="Onboarded Priya with welcome email and orientation event.",
        ),
    ))

    state = env._state
    assert state is not None
    assert obs.done is True
    assert state.agent_state["mail.sent_count"] == 1
    assert state.agent_state["calendar.events_count"] == 1
    assert state.agent_state["calendar.last_event_has_both_attendees"] is True
    assert state.drift_plan[0].detected_by_agent is True
    assert r7.task_completion == 1.0
    assert r7.drift_detection == 1.0
    assert r7.adaptation_quality == 1.0
    assert r7.shaped_total > 0.5
    assert r7.binary == 1.0


# ──────────────────────────────────────────────────────────────────
# Test 6 — max_steps terminates episode
# ──────────────────────────────────────────────────────────────────

def test_max_steps_terminates_episode() -> None:
    env = SchemaShiftEnvironment()
    env.reset("E2_meeting_invite_blast")
    inspect = Action(type="inspect_schema", inspect=InspectParams(tool="mail"))
    obs = None
    for _ in range(6):
        obs, _ = env.step(inspect)
    assert obs is not None
    assert obs.done is True
    assert obs.step == 6


# ──────────────────────────────────────────────────────────────────
# Test 7 — step_shaping +0.10 for inspect after failure
# ──────────────────────────────────────────────────────────────────

def test_step_shaping_applied_correctly() -> None:
    env = SchemaShiftEnvironment()
    env.reset("E1_onboard_new_hire")

    # Step 1: send_message missing required 'body' → 400
    env.step(Action(
        type="call_tool",
        tool_call=ToolCallParams(
            tool="mail", endpoint="send_message",
            params={"to": "x@y.com", "subject": "hi"},
        ),
    ))

    # Step 2: inspect_schema after failure → +0.10 shaping
    obs, reward = env.step(Action(
        type="inspect_schema", inspect=InspectParams(tool="mail"),
    ))
    assert reward.step_shaping == pytest.approx(0.10)


# ──────────────────────────────────────────────────────────────────
# Test 8 — dumb retry penalty
# ──────────────────────────────────────────────────────────────────

def test_dumb_retry_penalty() -> None:
    env = SchemaShiftEnvironment()
    env.reset("E1_onboard_new_hire")

    # Step 1: call_tool mail.send_message with only {"to": "x"} → 400
    env.step(Action(
        type="call_tool",
        tool_call=ToolCallParams(
            tool="mail", endpoint="send_message",
            params={"to": "x@y.com"},
        ),
    ))

    # Step 2: same call again → dumb retry → -0.05 penalty
    obs, reward = env.step(Action(
        type="call_tool",
        tool_call=ToolCallParams(
            tool="mail", endpoint="send_message",
            params={"to": "x@y.com"},
        ),
    ))
    assert reward.step_shaping == pytest.approx(-0.05)
