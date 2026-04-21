"""SchemaShift — typed contracts for actions, observations, rewards."""
from __future__ import annotations
from typing import Literal, Optional, Any
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────
# ACTION SPACE
# ─────────────────────────────────────────────────────────────────

ToolName = Literal["mail", "calendar", "crm", "chat", "docs"]

ActionType = Literal[
    "call_tool", "inspect_schema", "retry_with_variant",
    "report_drift", "complete_task",
]

class ToolCallParams(BaseModel):
    tool: ToolName
    endpoint: str
    params: dict[str, Any] = Field(default_factory=dict)

class InspectParams(BaseModel):
    tool: ToolName

class RetryParams(BaseModel):
    tool: ToolName
    endpoint: str
    params: dict[str, Any]

class DriftReportParams(BaseModel):
    tool: ToolName
    drift_kind: Literal[
        "field_rename", "endpoint_deprecation", "response_restructure",
        "new_required_param", "error_code_remap", "tool_removal",
        "rate_limit_tightening"
    ]
    description: str

class CompleteParams(BaseModel):
    summary: str

class Action(BaseModel):
    type: ActionType
    tool_call: Optional[ToolCallParams] = None
    inspect: Optional[InspectParams] = None
    retry: Optional[RetryParams] = None
    report: Optional[DriftReportParams] = None
    complete: Optional[CompleteParams] = None


# ─────────────────────────────────────────────────────────────────
# OBSERVATION
# ─────────────────────────────────────────────────────────────────

class ToolResponse(BaseModel):
    ok: bool
    status: int
    body: dict[str, Any] | None = None
    error: str | None = None

class HistoryStep(BaseModel):
    step: int
    action: Action
    response: ToolResponse | None = None
    reward_breakdown: dict[str, float] | None = None

class Observation(BaseModel):
    episode_id: str
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    step: int
    max_steps: int
    token_budget_remaining: int
    task_description: str
    success_criteria: list[str]
    tool_schemas: dict[str, dict]                  # current (possibly drifted)
    known_state: dict[str, Any]
    history: list[HistoryStep]
    last_response: ToolResponse | None
    drift_events_visible: list[dict] = []
    done: bool = False
    feedback: str = ""


# ─────────────────────────────────────────────────────────────────
# DRIFT
# ─────────────────────────────────────────────────────────────────

class DriftEvent(BaseModel):
    tool: ToolName
    endpoint: str | None = None
    kind: Literal[
        "field_rename", "endpoint_deprecation", "response_restructure",
        "new_required_param", "error_code_remap", "tool_removal",
        "rate_limit_tightening"
    ]
    fires_at_step: int
    details: dict[str, Any]
    detected_by_agent: bool = False


# ─────────────────────────────────────────────────────────────────
# REWARD — with dense shaping
# ─────────────────────────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    # Terminal-ish rubric dimensions
    task_completion: float = 0.0
    drift_detection: float = 0.0
    adaptation_quality: float = 0.0
    efficiency: float = 0.0
    # Gates
    catastrophic_gate: float = 1.0
    correct_final_gate: float = 1.0
    # Step-level dense shaping (NEW in v2)
    step_shaping: float = 0.0
    # Totals
    shaped_total: float = 0.0                      # rubric × gates + step_shaping
    binary: float = 0.0                            # {0,1} for GRPO

class EpisodeState(BaseModel):
    episode_id: str
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    step: int = 0
    max_steps: int
    token_budget: int
    token_budget_remaining: int
    drift_plan: list[DriftEvent]
    ground_truth_final_state: dict[str, Any]
    agent_state: dict[str, Any] = Field(default_factory=dict)
    history: list[HistoryStep] = Field(default_factory=list)
    done: bool = False
    cumulative_reward: float = 0.0
