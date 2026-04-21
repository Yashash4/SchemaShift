"""Scenario structure tests — Phase 4."""
from __future__ import annotations

from models import DriftEvent
from scenarios import SCENARIOS


REQUIRED_KEYS = {
    "difficulty",
    "max_steps",
    "token_budget",
    "task_description",
    "success_criteria",
    "seed_data",
    "drift_plan",
    "ground_truth_final_state",
    "required_tools",
}


def test_all_scenarios_present() -> None:
    assert set(SCENARIOS.keys()) == {
        "E1_onboard_new_hire",
        "E2_meeting_invite_blast",
        "E3_customer_lookup",
        "M1_customer_escalation",
        "M2_weekly_report",
        "M3_event_cleanup",
    }


def test_medium_scenarios_present() -> None:
    for task_id in ("M1_customer_escalation", "M2_weekly_report", "M3_event_cleanup"):
        assert task_id in SCENARIOS, f"{task_id} missing from SCENARIOS"
        assert SCENARIOS[task_id]["difficulty"] == "medium"


def test_medium_scenarios_multi_drift() -> None:
    for task_id in ("M1_customer_escalation", "M2_weekly_report", "M3_event_cleanup"):
        plan = SCENARIOS[task_id]["drift_plan"]
        assert len(plan) == 2, f"{task_id}: expected 2 drifts, got {len(plan)}"


def test_m3_same_tool_multi_drift() -> None:
    """M3 is the judgment-call-#2 stress test: both drifts target calendar."""
    plan = SCENARIOS["M3_event_cleanup"]["drift_plan"]
    tools = [d.tool for d in plan]
    assert tools == ["calendar", "calendar"], (
        f"M3 drifts must both target calendar, got {tools}"
    )


def test_medium_required_tools() -> None:
    assert SCENARIOS["M1_customer_escalation"]["required_tools"] == ["mail", "calendar", "crm"]
    assert SCENARIOS["M2_weekly_report"]["required_tools"] == ["mail", "calendar", "crm"]
    assert SCENARIOS["M3_event_cleanup"]["required_tools"] == ["mail", "calendar"]


def test_each_scenario_has_required_fields() -> None:
    for name, sc in SCENARIOS.items():
        missing = REQUIRED_KEYS - set(sc.keys())
        assert not missing, f"{name} missing keys: {missing}"


def test_drift_plans_contain_valid_events() -> None:
    valid_tools = {"mail", "calendar", "crm", "chat", "docs"}
    for name, sc in SCENARIOS.items():
        plan = sc["drift_plan"]
        assert isinstance(plan, list)
        assert len(plan) > 0, f"{name}: drift_plan must have at least one event"
        for d in plan:
            assert isinstance(d, DriftEvent), f"{name}: non-DriftEvent in drift_plan"
            assert d.tool in valid_tools
            assert isinstance(d.fires_at_step, int)
            assert d.fires_at_step >= 0


def test_required_tools_match_scenario_intent() -> None:
    assert SCENARIOS["E1_onboard_new_hire"]["required_tools"] == ["mail", "calendar"]
    assert SCENARIOS["E2_meeting_invite_blast"]["required_tools"] == ["mail"]
    assert SCENARIOS["E3_customer_lookup"]["required_tools"] == ["crm"]

    for name, sc in SCENARIOS.items():
        seed_keys = set(sc["seed_data"].keys())
        req = set(sc["required_tools"])
        assert seed_keys.issubset(req), (
            f"{name}: seed_data keys {seed_keys} not a subset of required_tools {req}"
        )


def test_ground_truth_keys_non_empty() -> None:
    for name, sc in SCENARIOS.items():
        gt = sc["ground_truth_final_state"]
        assert isinstance(gt, dict)
        assert len(gt) > 0, f"{name}: ground_truth_final_state is empty"
