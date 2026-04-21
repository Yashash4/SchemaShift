"""FastAPI server acceptance tests — Phase 7.

Each test gets a fresh SchemaShiftEnvironment via monkeypatch so state doesn't
leak across tests (the default module-level env would pollute test ordering).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import app as app_module
from server.environment import SchemaShiftEnvironment


@pytest.fixture
def client(monkeypatch):
    """Fresh env per test for isolation."""
    fresh_env = SchemaShiftEnvironment()
    monkeypatch.setattr(app_module, "env", fresh_env)
    return TestClient(app_module.app)


def test_root_endpoint(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "SchemaShift"
    assert "endpoints" in body


def test_health_endpoint(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": "0.1.0"}


def test_tasks_endpoint(client) -> None:
    r = client.get("/tasks")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    task_ids = {t["task_id"] for t in body["tasks"]}
    assert task_ids == {
        "E1_onboard_new_hire",
        "E2_meeting_invite_blast",
        "E3_customer_lookup",
    }
    for t in body["tasks"]:
        assert t["difficulty"] == "easy"
        assert isinstance(t["required_tools"], list)


def test_reset_valid_task(client) -> None:
    r = client.post("/reset", json={"task_id": "E1_onboard_new_hire"})
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "E1_onboard_new_hire"
    assert body["step"] == 0
    assert body["done"] is False
    assert "mail" in body["tool_schemas"]
    assert "calendar" in body["tool_schemas"]


def test_reset_invalid_task(client) -> None:
    r = client.post("/reset", json={"task_id": "nonexistent_task"})
    assert r.status_code == 400
    body = r.json()
    assert "detail" in body
    assert "nonexistent_task" in body["detail"]


def test_step_before_reset_returns_400(client) -> None:
    action_payload = {
        "action": {
            "type": "inspect_schema",
            "inspect": {"tool": "mail"},
        },
        "tokens_used": 0,
    }
    r = client.post("/step", json=action_payload)
    assert r.status_code == 400
    assert "reset" in r.json()["detail"].lower()


def test_step_valid_action_after_reset(client) -> None:
    client.post("/reset", json={"task_id": "E1_onboard_new_hire"})
    action_payload = {
        "action": {
            "type": "inspect_schema",
            "inspect": {"tool": "mail"},
        },
        "tokens_used": 0,
    }
    r = client.post("/step", json=action_payload)
    assert r.status_code == 200
    body = r.json()
    assert "observation" in body
    assert "reward" in body
    assert body["observation"]["step"] == 1
    assert body["observation"]["last_response"]["ok"] is True


def test_state_endpoint_before_reset_returns_400(client) -> None:
    r = client.get("/state")
    assert r.status_code == 400


def test_state_endpoint_after_reset(client) -> None:
    client.post("/reset", json={"task_id": "E1_onboard_new_hire"})
    r = client.get("/state")
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == "E1_onboard_new_hire"
    assert body["step"] == 0
    assert body["max_steps"] == 8
    assert body["done"] is False
    assert isinstance(body["drift_plan"], list)


def test_grader_endpoint_after_reset(client) -> None:
    client.post("/reset", json={"task_id": "E1_onboard_new_hire"})
    r = client.get("/grader")
    assert r.status_code == 200
    body = r.json()
    assert body["cumulative_reward"] == 0.0
    assert body["step"] == 0
    assert body["max_steps"] == 8
    assert body["done"] is False
    assert "current_breakdown" in body
