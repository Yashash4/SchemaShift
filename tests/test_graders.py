"""Grader acceptance tests — Phase 5 (the GRPO reward signal)."""
from __future__ import annotations

from graders import (
    AdaptationRubric,
    CompletionRubric,
    DriftDetectionRubric,
    EfficiencyRubric,
    build_grader,
    compute_step_shaping,
)
from models import (
    Action,
    DriftEvent,
    DriftReportParams,
    EpisodeState,
    HistoryStep,
    InspectParams,
    RetryParams,
    RewardBreakdown,
    ToolCallParams,
    ToolResponse,
)


def _state_with(
    *,
    step: int = 0,
    max_steps: int = 8,
    token_budget: int = 4000,
    token_budget_remaining: int | None = None,
    drift_plan: list[DriftEvent] | None = None,
    history: list[HistoryStep] | None = None,
    agent_state: dict | None = None,
    ground_truth: dict | None = None,
    done: bool = False,
) -> EpisodeState:
    s = EpisodeState(
        episode_id="ep",
        task_id="t",
        difficulty="easy",
        max_steps=max_steps,
        token_budget=token_budget,
        token_budget_remaining=(
            token_budget_remaining if token_budget_remaining is not None else token_budget
        ),
        drift_plan=drift_plan or [],
        ground_truth_final_state=ground_truth or {},
    )
    s.step = step
    if agent_state:
        s.agent_state = dict(agent_state)
    if history:
        s.history = list(history)
    s.done = done
    return s


def test_completion_rubric_full() -> None:
    s = _state_with(
        ground_truth={"mail.sent_count": 1, "mail.last_sent_to": "a@x.com"},
        agent_state={"mail.sent_count": 1, "mail.last_sent_to": "a@x.com"},
    )
    name, val, details = CompletionRubric().score(s)
    assert name == "task_completion"
    assert val == 1.0
    assert details["satisfied"] == 2
    assert details["total"] == 2


def test_completion_rubric_partial() -> None:
    s = _state_with(
        ground_truth={"mail.sent_count": 1, "mail.last_sent_to": "a@x.com"},
        agent_state={"mail.sent_count": 1},
    )
    _, val, details = CompletionRubric().score(s)
    assert val == 0.5
    assert details["satisfied"] == 1
    assert details["total"] == 2


def test_completion_rubric_empty_gt() -> None:
    s = _state_with(ground_truth={})
    _, val, details = CompletionRubric().score(s)
    assert val == 0.0
    assert "reason" in details


def test_drift_detection_rubric() -> None:
    d1 = DriftEvent(
        tool="mail", endpoint="send_message", kind="endpoint_deprecation",
        fires_at_step=3, details={}, detected_by_agent=True,
    )
    d2 = DriftEvent(
        tool="calendar", endpoint="create_event", kind="field_rename",
        fires_at_step=5, details={}, detected_by_agent=False,
    )
    s = _state_with(step=5, drift_plan=[d1, d2])
    _, val, details = DriftDetectionRubric().score(s)
    assert val == 0.5
    assert details["detected"] == 1
    assert details["fired_total"] == 2


def test_adaptation_rubric_success() -> None:
    drift = DriftEvent(
        tool="calendar", endpoint="create_event", kind="field_rename",
        fires_at_step=3, details={},
    )
    retry_action = Action(
        type="retry_with_variant",
        retry=RetryParams(
            tool="calendar", endpoint="create_event",
            params={"title": "x", "start": "t1", "end": "t2",
                    "participants": [{"email": "a@x.com", "role": "required"}]},
        ),
    )
    hist = [HistoryStep(
        step=4, action=retry_action,
        response=ToolResponse(ok=True, status=200, body={"event_id": "evt_1"}),
    )]
    s = _state_with(step=5, drift_plan=[drift], history=hist)
    _, val, details = AdaptationRubric().score(s)
    assert val == 1.0
    assert details["adapted"] == 1
    assert details["opportunities"] == 1


def test_adaptation_rubric_multi_drift_same_tool() -> None:
    """M3-style stress test: two drifts on the same tool (calendar).

    History:
      step 2 — call_tool calendar.delete_event → 410 (post-Drift-A tool_removal)
      step 5 — call_tool calendar.create_event with attendees → 400 (post-Drift-B field_rename)
      step 7 — retry_with_variant calendar.create_event with participants → 200 success

    Expected rubric behavior (per Phase 5 judgment call #2):
      - Drift A (fires_at_step=2): first post-drift calendar call = step 5 (failed). opp=1, adapted=0.
      - Drift B (fires_at_step=5): first post-drift calendar call = step 7 (succeeded). opp=1, adapted=1.
      - Score = 1/2 = 0.5.

    Documents intentional denominator behavior: partial credit for partial adaptation.
    Dense step_shaping (+0.20 for successful retry after failure) catches the step 7
    recovery independently, so the rubric staying conservative is acceptable.
    """
    drifts = [
        DriftEvent(
            tool="calendar", endpoint="delete_event", kind="tool_removal",
            fires_at_step=2, details={}, detected_by_agent=True,
        ),
        DriftEvent(
            tool="calendar", endpoint="create_event", kind="field_rename",
            fires_at_step=5, details={}, detected_by_agent=True,
        ),
    ]
    history = [
        HistoryStep(
            step=2,
            action=Action(
                type="call_tool",
                tool_call=ToolCallParams(
                    tool="calendar", endpoint="delete_event",
                    params={"event_id": "evt_2"},
                ),
            ),
            response=ToolResponse(ok=False, status=410, error="removed"),
        ),
        HistoryStep(
            step=5,
            action=Action(
                type="call_tool",
                tool_call=ToolCallParams(
                    tool="calendar", endpoint="create_event",
                    params={"title": "x", "start": "t1", "end": "t2",
                            "attendees": ["a@x.com"]},
                ),
            ),
            response=ToolResponse(ok=False, status=400, error="missing required"),
        ),
        HistoryStep(
            step=7,
            action=Action(
                type="retry_with_variant",
                retry=RetryParams(
                    tool="calendar", endpoint="create_event",
                    params={"title": "x", "start": "t1", "end": "t2",
                            "participants": [{"email": "a@x.com", "role": "required"}]},
                ),
            ),
            response=ToolResponse(ok=True, status=200, body={"event_id": "evt_3"}),
        ),
    ]
    s = _state_with(step=7, drift_plan=drifts, history=history)
    _, val, details = AdaptationRubric().score(s)
    assert val == 0.5, f"Expected 0.5, got {val}"
    assert details["adapted"] == 1
    assert details["opportunities"] == 2


def test_adaptation_rubric_no_post_drift_calls() -> None:
    drift = DriftEvent(
        tool="calendar", endpoint="create_event", kind="field_rename",
        fires_at_step=3, details={},
    )
    s = _state_with(step=5, drift_plan=[drift], history=[])
    _, val, details = AdaptationRubric().score(s)
    assert val == 0.0
    assert "reason" in details


def test_efficiency_rubric_fresh() -> None:
    s = _state_with(step=0, max_steps=8, token_budget=4000, token_budget_remaining=4000)
    _, val, details = EfficiencyRubric().score(s)
    assert val == 1.0
    assert details["step_eff"] == 1.0
    assert details["tok_eff"] == 1.0


def test_step_shaping_inspect_after_failure() -> None:
    prev_action = Action(
        type="call_tool",
        tool_call=ToolCallParams(tool="mail", endpoint="send_message", params={}),
    )
    failed = ToolResponse(ok=False, status=400, error="bad")
    hist = [HistoryStep(step=1, action=prev_action, response=failed)]
    s = _state_with(step=2, history=hist)
    inspect = Action(type="inspect_schema", inspect=InspectParams(tool="mail"))
    assert compute_step_shaping(s, inspect, None) == 0.10


def test_step_shaping_correct_drift_report() -> None:
    drift = DriftEvent(
        tool="mail", endpoint="send_message", kind="endpoint_deprecation",
        fires_at_step=1, details={},
    )
    s = _state_with(step=2, drift_plan=[drift])
    report = Action(
        type="report_drift",
        report=DriftReportParams(
            tool="mail",
            drift_kind="endpoint_deprecation",
            description="send_message deprecated",
        ),
    )
    assert compute_step_shaping(s, report, None) == 0.15


def test_step_shaping_dumb_retry_penalty() -> None:
    prev_action = Action(
        type="call_tool",
        tool_call=ToolCallParams(tool="mail", endpoint="send_message", params={}),
    )
    failed = ToolResponse(ok=False, status=400, error="bad")
    hist = [HistoryStep(step=1, action=prev_action, response=failed)]
    s = _state_with(step=2, history=hist)
    same = Action(
        type="call_tool",
        tool_call=ToolCallParams(tool="mail", endpoint="send_message", params={}),
    )
    assert compute_step_shaping(s, same, None) == -0.05


def test_build_grader_full_pipeline() -> None:
    drift = DriftEvent(
        tool="mail", endpoint="send_message", kind="endpoint_deprecation",
        fires_at_step=2, details={}, detected_by_agent=True,
    )
    retry_action = Action(
        type="retry_with_variant",
        retry=RetryParams(
            tool="mail", endpoint="messages.send",
            params={"to": "a@x.com", "subject": "hi", "body": "hello"},
        ),
    )
    hist = [HistoryStep(
        step=3, action=retry_action,
        response=ToolResponse(ok=True, status=200, body={"message_id": "m1"}),
    )]
    s = _state_with(
        step=3, max_steps=8, token_budget=4000, token_budget_remaining=3000,
        drift_plan=[drift], history=hist,
        agent_state={"mail.sent_count": 1},
        ground_truth={"mail.sent_count": 1},
        done=True,
    )
    reward = build_grader()(s)
    assert isinstance(reward, RewardBreakdown)
    assert reward.task_completion > 0.9
    assert reward.drift_detection > 0.9
    assert reward.adaptation_quality > 0.9
    assert reward.catastrophic_gate == 1.0
    assert reward.correct_final_gate == 1.0
    assert reward.binary == 1.0
    assert reward.shaped_total > 0.7
    assert reward.step_shaping == 0.0
