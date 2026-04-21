"""Tests for eval.py baseline agents — Phase 9."""
from __future__ import annotations

import pytest

from eval import (
    EpisodeResult,
    NaiveHeuristicAgent,
    PolicyAwareHeuristicAgent,
    build_agent,
    print_baseline_table,
)
from models import (
    Observation,
    ToolResponse,
)


def _make_obs(
    task_description: str = "Send a welcome email",
    success_criteria: list | None = None,
    tool_schemas: dict | None = None,
    history: list | None = None,
    last_response: ToolResponse | None = None,
    step: int = 0,
    max_steps: int = 8,
    done: bool = False,
) -> Observation:
    return Observation(
        episode_id="test-ep",
        task_id="E1_onboard_new_hire",
        difficulty="easy",
        step=step,
        max_steps=max_steps,
        token_budget_remaining=4000,
        task_description=task_description,
        success_criteria=success_criteria or [],
        tool_schemas=tool_schemas or {},
        known_state={},
        history=history or [],
        last_response=last_response,
        drift_events_visible=[],
        done=done,
        feedback="test",
    )


def test_naive_heuristic_returns_call_tool_first() -> None:
    agent = NaiveHeuristicAgent()
    obs = _make_obs(
        tool_schemas={"mail": {"send_message": {"params": {"to": "str"}, "required": ["to"]}}},
    )
    action = agent.act(obs)
    assert action.type == "call_tool"
    assert action.tool_call is not None
    assert action.tool_call.tool == "mail"


def test_naive_heuristic_completes_after_3_steps() -> None:
    agent = NaiveHeuristicAgent()
    obs = _make_obs(
        tool_schemas={"mail": {"send_message": {"params": {"to": "str"}, "required": ["to"]}}},
    )
    agent.act(obs)
    agent.act(obs)
    action = agent.act(obs)
    assert action.type == "complete_task"


def test_policy_aware_inspects_after_failure() -> None:
    agent = PolicyAwareHeuristicAgent()
    obs1 = _make_obs(
        task_description="Send welcome email to priya@company.com",
        tool_schemas={"mail": {"send_message": {"params": {"to": "str"}, "required": ["to"]}}},
    )
    action1 = agent.act(obs1)
    assert action1.type == "call_tool"
    assert action1.tool_call is not None
    assert action1.tool_call.tool == "mail"

    failed_response = ToolResponse(ok=False, status=400, error="validation failed")
    obs2 = _make_obs(last_response=failed_response, step=1)
    action2 = agent.act(obs2)
    assert action2.type == "inspect_schema"
    assert action2.inspect is not None
    assert action2.inspect.tool == "mail"


def test_policy_aware_reports_drift_after_inspecting() -> None:
    agent = PolicyAwareHeuristicAgent()
    obs1 = _make_obs(
        task_description="Send welcome email to priya@company.com",
        tool_schemas={"mail": {"send_message": {"params": {"to": "str"}, "required": ["to"]}}},
    )
    agent.act(obs1)  # task_specific → mail call
    failed = ToolResponse(ok=False, status=400, error="bad")
    obs2 = _make_obs(last_response=failed, step=1)
    agent.act(obs2)  # inspect
    obs3 = _make_obs(
        last_response=ToolResponse(ok=True, status=200, body={"schema": {}}),
        step=2,
        tool_schemas={"mail": {"messages.send": {"params": {"to": "str"}, "required": ["to"]}}},
    )
    action = agent.act(obs3)
    assert action.type == "report_drift"
    assert action.report is not None
    assert action.report.tool == "mail"


def test_build_agent_factory() -> None:
    assert isinstance(build_agent("naive_heuristic"), NaiveHeuristicAgent)
    assert isinstance(build_agent("policy_aware_heuristic"), PolicyAwareHeuristicAgent)
    with pytest.raises(ValueError):
        build_agent("nonexistent_baseline")


def test_print_baseline_table_format() -> None:
    results = [
        EpisodeResult(
            task_id="E1_onboard_new_hire", seed=0,
            completion=1.0, shaped_total=0.85, binary=1.0, steps_used=7,
        ),
        EpisodeResult(
            task_id="E1_onboard_new_hire", seed=1,
            completion=0.5, shaped_total=0.40, binary=0.0, steps_used=8,
        ),
    ]
    table = print_baseline_table("test_baseline", results)
    assert "## Eval results" in table
    assert "E1_onboard_new_hire" in table
    assert "0.850" in table or "0.85" in table
    assert "OVERALL" in table
