"""SchemaShiftEnvClient — HTTP client for SchemaShift env server.

Used by training loops (Kaggle, Colab, local) to drive episodes.
"""
from __future__ import annotations

import httpx

from models import Action, Observation, RewardBreakdown


class SchemaShiftEnvClient:
    """Synchronous HTTP client for SchemaShift env server."""

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "SchemaShiftEnvClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────

    def health(self) -> bool:
        """Ping server. True if healthy."""
        try:
            r = self._client.get(f"{self.base_url}/health")
            return r.status_code == 200 and r.json().get("status") == "ok"
        except Exception:
            return False

    def list_tasks(self) -> list[dict]:
        """Return list of available scenarios with metadata."""
        r = self._client.get(f"{self.base_url}/tasks")
        r.raise_for_status()
        return r.json().get("tasks", [])

    def reset(self, task_id: str, seed: int = 0) -> Observation:
        """Start a new episode. Returns initial Observation."""
        r = self._client.post(
            f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed},
        )
        if r.status_code == 400:
            raise ValueError(f"Reset rejected: {r.json().get('detail')}")
        r.raise_for_status()
        return Observation.model_validate(r.json())

    def step(
        self, action: Action, tokens_used: int = 0
    ) -> tuple[Observation, RewardBreakdown]:
        """Submit action, receive observation + reward."""
        r = self._client.post(
            f"{self.base_url}/step",
            json={
                "action": action.model_dump(),
                "tokens_used": tokens_used,
            },
        )
        if r.status_code == 400:
            raise RuntimeError(f"Step rejected: {r.json().get('detail')}")
        r.raise_for_status()
        data = r.json()
        obs = Observation.model_validate(data["observation"])
        reward = RewardBreakdown.model_validate(data["reward"])
        return obs, reward

    def get_state(self) -> dict:
        """Full episode state (debugging only)."""
        r = self._client.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def get_grader(self) -> dict:
        """Current grader breakdown."""
        r = self._client.get(f"{self.base_url}/grader")
        r.raise_for_status()
        return r.json()
