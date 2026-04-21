"""Deploy smoke test — runs against LIVE deployed HF Space.

Skipped unless SCHEMASHIFT_DEPLOY_URL env var is set.

Run:
    SCHEMASHIFT_DEPLOY_URL=https://YOUR-USERNAME-schemashift.hf.space pytest tests/test_deploy_smoke.py -v
"""
from __future__ import annotations

import os

import httpx
import pytest

from client import SchemaShiftEnvClient
from models import Action, InspectParams, ToolCallParams


DEPLOY_URL = os.getenv("SCHEMASHIFT_DEPLOY_URL")
REQUIRES_DEPLOY = pytest.mark.skipif(
    not DEPLOY_URL,
    reason="SCHEMASHIFT_DEPLOY_URL not set — skipping deploy smoke tests",
)


@REQUIRES_DEPLOY
def test_deployed_health() -> None:
    r = httpx.get(f"{DEPLOY_URL}/health", timeout=30.0)
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"


@REQUIRES_DEPLOY
def test_deployed_tasks_list() -> None:
    r = httpx.get(f"{DEPLOY_URL}/tasks", timeout=30.0)
    assert r.status_code == 200
    body = r.json()
    assert body.get("count") == 6
    task_ids = [t["task_id"] for t in body["tasks"]]
    for required in (
        "E1_onboard_new_hire",
        "E2_meeting_invite_blast",
        "E3_customer_lookup",
        "M1_customer_escalation",
        "M2_weekly_report",
        "M3_event_cleanup",
    ):
        assert required in task_ids, f"{required} missing from /tasks response"


@REQUIRES_DEPLOY
def test_deployed_reset_and_step() -> None:
    with SchemaShiftEnvClient(base_url=DEPLOY_URL, timeout=60.0) as client:
        assert client.health()
        obs = client.reset("E1_onboard_new_hire")
        assert obs.step == 0
        assert obs.task_id == "E1_onboard_new_hire"
        assert "mail" in obs.tool_schemas
        assert "calendar" in obs.tool_schemas

        action = Action(type="inspect_schema", inspect=InspectParams(tool="mail"))
        obs, reward = client.step(action, tokens_used=30)
        assert obs.step == 1
        assert reward.shaped_total is not None
        # First step, no prior failure → step_shaping must be 0.0 exactly
        assert reward.step_shaping == 0.0


@REQUIRES_DEPLOY
def test_deployed_step_shaping_fires() -> None:
    """Critical: verify dense shaping survives production HTTP."""
    with SchemaShiftEnvClient(base_url=DEPLOY_URL, timeout=60.0) as client:
        client.reset("E1_onboard_new_hire")

        # Trigger a failure first (missing required 'subject' and 'body')
        bad_action = Action(
            type="call_tool",
            tool_call=ToolCallParams(
                tool="mail",
                endpoint="send_message",
                params={"to": "x@y.com"},
            ),
        )
        obs, reward = client.step(bad_action, tokens_used=50)
        assert obs.last_response is not None
        assert not obs.last_response.ok

        # Inspect → must earn +0.10 step_shaping
        inspect_action = Action(
            type="inspect_schema", inspect=InspectParams(tool="mail")
        )
        obs, reward = client.step(inspect_action, tokens_used=30)
        assert abs(reward.step_shaping - 0.10) < 1e-6, (
            f"Deployed step_shaping = {reward.step_shaping}, expected 0.10. "
            "Dense shaping NOT preserved through production HTTP."
        )
