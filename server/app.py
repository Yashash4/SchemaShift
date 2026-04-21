"""FastAPI server exposing SchemaShiftEnvironment as OpenEnv HTTP service."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models import Action
from scenarios import SCENARIOS
from server.environment import SchemaShiftEnvironment


app = FastAPI(
    title="SchemaShift OpenEnv",
    description="RL environment for training adaptive tool use under schema drift.",
    version="0.1.0",
)
env = SchemaShiftEnvironment()


# ─────────────────────────────────────────────────────────────────
# Request / response models
# ─────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: str
    seed: int = 0


class StepRequest(BaseModel):
    action: Action
    tokens_used: int = Field(
        default=0, ge=0,
        description="Tokens consumed by the agent on this step",
    )


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict:
    return {
        "name": "SchemaShift",
        "version": "0.1.0",
        "description": "Adaptive tool use under schema drift",
        "endpoints": ["/health", "/reset", "/step", "/state", "/tasks", "/grader"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/reset")
def reset(req: ResetRequest) -> dict:
    """Start new episode. Returns initial Observation as dict."""
    try:
        obs = env.reset(req.task_id, req.seed)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {e}")


@app.post("/step")
def step(req: StepRequest) -> dict:
    """Submit action, get observation + reward."""
    try:
        obs, reward = env.step(req.action, req.tokens_used)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {e}")


@app.get("/state")
def get_state() -> dict:
    """Return full current episode state for debugging."""
    if env._state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    return env._state.model_dump()


@app.get("/tasks")
def get_tasks() -> dict:
    """List all available scenarios with metadata."""
    tasks = []
    for task_id, scenario in SCENARIOS.items():
        desc = scenario["task_description"]
        trimmed = desc[:120] + ("..." if len(desc) > 120 else "")
        tasks.append({
            "task_id": task_id,
            "difficulty": scenario["difficulty"],
            "max_steps": scenario["max_steps"],
            "required_tools": scenario["required_tools"],
            "description": trimmed,
        })
    return {"tasks": tasks, "count": len(tasks)}


@app.get("/grader")
def get_grader_breakdown() -> dict:
    """Return grader scoring for current episode state."""
    if env._state is None:
        raise HTTPException(status_code=400, detail="No active episode.")
    reward = env._grader(env._state)
    return {
        "cumulative_reward": env._state.cumulative_reward,
        "current_breakdown": reward.model_dump(),
        "step": env._state.step,
        "max_steps": env._state.max_steps,
        "done": env._state.done,
    }
