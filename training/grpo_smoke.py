"""5-step smoke test of GRPO-compatible env interaction.

Does NOT run GRPO training — that's in the Kaggle notebook.
Verifies:
  1. Server is reachable
  2. Client roundtrips action/observation cleanly
  3. Reward signal has correct shape
  4. Dense shaping fires when expected (inspect-after-failure on step 4)

Run:
    # Terminal 1: uvicorn server.app:app --port 7860
    # Terminal 2: python training/grpo_smoke.py
"""
from __future__ import annotations

import os
import sys

from client import SchemaShiftEnvClient
from models import (
    Action,
    CompleteParams,
    InspectParams,
    ToolCallParams,
)


def main() -> int:
    url = os.getenv("SCHEMASHIFT_URL", "http://localhost:7860")
    print(f"Connecting to env at {url}...")

    with SchemaShiftEnvClient(base_url=url) as client:
        if not client.health():
            print("ERROR: Env server not reachable. Start uvicorn first.")
            return 1
        print("[OK] Health check passed")

        tasks = client.list_tasks()
        print(f"[OK] Tasks available: {[t['task_id'] for t in tasks]}")

        print("\n--- Running E1_onboard_new_hire smoke ---")
        obs = client.reset("E1_onboard_new_hire")
        print(
            f"[OK] Reset: step={obs.step}, max_steps={obs.max_steps}, "
            f"tools={list(obs.tool_schemas)}"
        )

        # Step 1: send welcome email (pre-drift)
        obs, reward = client.step(
            Action(
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
            ),
            tokens_used=50,
        )
        print(
            f"Step 1 (send welcome email): "
            f"ok={obs.last_response.ok if obs.last_response else None}, "
            f"shaped={reward.shaped_total:.4f}, step_shaping={reward.step_shaping:.4f}"
        )

        # Step 2: inspect calendar (pre-drift)
        obs, reward = client.step(
            Action(type="inspect_schema", inspect=InspectParams(tool="calendar")),
            tokens_used=30,
        )
        print(
            f"Step 2 (inspect calendar): "
            f"shaped={reward.shaped_total:.4f}, step_shaping={reward.step_shaping:.4f}"
        )

        # Step 3: create_event with OLD schema (drift fires at step 3 → fails)
        obs, reward = client.step(
            Action(
                type="call_tool",
                tool_call=ToolCallParams(
                    tool="calendar",
                    endpoint="create_event",
                    params={
                        "title": "Orientation",
                        "start": "2026-04-27T10:00:00Z",
                        "end": "2026-04-27T11:00:00Z",
                        "attendees": ["priya@company.com", "alex@company.com"],
                    },
                ),
            ),
            tokens_used=100,
        )
        print(
            f"Step 3 (create_event OLD schema): "
            f"ok={obs.last_response.ok if obs.last_response else None}, "
            f"status={obs.last_response.status if obs.last_response else None}, "
            f"shaped={reward.shaped_total:.4f}"
        )

        # Step 4: inspect_schema after failure → +0.10 dense shaping
        obs, reward = client.step(
            Action(type="inspect_schema", inspect=InspectParams(tool="calendar")),
            tokens_used=30,
        )
        print(
            f"Step 4 (inspect after failure): "
            f"shaped={reward.shaped_total:.4f}, "
            f"step_shaping={reward.step_shaping:.4f} (should be ~0.10)"
        )
        step4_shaping = reward.step_shaping

        # Step 5: complete task
        obs, reward = client.step(
            Action(
                type="complete_task",
                complete=CompleteParams(summary="Smoke test complete."),
            ),
            tokens_used=40,
        )
        print(
            f"Step 5 (complete): done={obs.done}, "
            f"shaped={reward.shaped_total:.4f}"
        )

        # Verify the critical diagnostic
        if abs(step4_shaping - 0.10) > 1e-6:
            print(
                f"\nERROR: Step 4 step_shaping was {step4_shaping:.4f}, expected 0.10."
            )
            print("Dense shaping is NOT flowing through the HTTP pipeline.")
            return 2

        print("\n--- Smoke test PASSED ---")
        print("Step-shaping fired correctly at Step 4 as expected.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
