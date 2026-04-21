"""SchemaShiftEnvClient tests (mocked HTTP responses)."""
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from client import SchemaShiftEnvClient


def test_client_health_true() -> None:
    client = SchemaShiftEnvClient(base_url="http://example.com")
    with patch.object(client._client, "get") as mock_get:
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: {"status": "ok"},
        )
        assert client.health() is True
    client.close()


def test_client_health_false_on_exception() -> None:
    client = SchemaShiftEnvClient(base_url="http://example.com")
    with patch.object(client._client, "get", side_effect=Exception("connection refused")):
        assert client.health() is False
    client.close()


def test_client_reset_returns_observation() -> None:
    client = SchemaShiftEnvClient(base_url="http://example.com")
    fake_obs = {
        "episode_id": "test-id",
        "task_id": "E1_onboard_new_hire",
        "difficulty": "easy",
        "step": 0,
        "max_steps": 8,
        "token_budget_remaining": 4000,
        "task_description": "test",
        "success_criteria": [],
        "tool_schemas": {},
        "known_state": {},
        "history": [],
        "last_response": None,
        "drift_events_visible": [],
        "done": False,
        "feedback": "Episode started.",
    }
    with patch.object(client._client, "post") as mock_post:
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: fake_obs,
            raise_for_status=lambda: None,
        )
        obs = client.reset("E1_onboard_new_hire")
    assert obs.task_id == "E1_onboard_new_hire"
    assert obs.step == 0
    assert obs.difficulty == "easy"
    client.close()


def test_client_reset_400_raises_value_error() -> None:
    client = SchemaShiftEnvClient(base_url="http://example.com")
    with patch.object(client._client, "post") as mock_post:
        mock_post.return_value = Mock(
            status_code=400,
            json=lambda: {"detail": "Unknown task_id: foo"},
        )
        with pytest.raises(ValueError, match="Unknown task_id"):
            client.reset("foo")
    client.close()
