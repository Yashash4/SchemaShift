"""Composable rubric grader + dense step-level reward shaping.

Reward flow per step:
    compute_step_shaping(state, action, response)  -> float  (dense signal)
    build_grader()(state)                          -> RewardBreakdown (terminal-ish)
Environment combines them into reward.shaped_total and reward.binary.
"""
from __future__ import annotations

from typing import Any, Callable

from models import Action, EpisodeState, RewardBreakdown, ToolResponse


# ─────────────────────────────────────────────────────────────────
# Rubric base
# ─────────────────────────────────────────────────────────────────

class Rubric:
    name: str = ""

    def score(self, state: EpisodeState) -> tuple[str, float, dict]:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────
# Concrete rubrics
# ─────────────────────────────────────────────────────────────────

class CompletionRubric(Rubric):
    name = "task_completion"

    def score(self, state: EpisodeState) -> tuple[str, float, dict]:
        gt = state.ground_truth_final_state
        if not gt:
            return (self.name, 0.0, {"reason": "no ground truth"})
        satisfied = 0
        for key, expected in gt.items():
            if _matches(state.agent_state.get(key), expected):
                satisfied += 1
        total = len(gt)
        return (
            self.name,
            satisfied / total,
            {"satisfied": satisfied, "total": total},
        )


class DriftDetectionRubric(Rubric):
    name = "drift_detection"

    def score(self, state: EpisodeState) -> tuple[str, float, dict]:
        fired = [d for d in state.drift_plan if d.fires_at_step <= state.step]
        if not fired:
            return (self.name, 0.0, {"reason": "no drifts fired yet"})
        detected = sum(1 for d in fired if d.detected_by_agent)
        return (
            self.name,
            detected / len(fired),
            {"detected": detected, "fired_total": len(fired)},
        )


class AdaptationRubric(Rubric):
    name = "adaptation_quality"

    def score(self, state: EpisodeState) -> tuple[str, float, dict]:
        fired = [d for d in state.drift_plan if d.fires_at_step <= state.step]
        if not fired:
            return (self.name, 0.0, {"reason": "no drifts fired yet"})

        adapted = 0
        opportunities = 0
        for drift in fired:
            first_post = None
            for h in state.history:
                if h.step <= drift.fires_at_step:
                    continue
                act = h.action
                if act.type == "call_tool" and act.tool_call is not None and act.tool_call.tool == drift.tool:
                    first_post = h
                    break
                if act.type == "retry_with_variant" and act.retry is not None and act.retry.tool == drift.tool:
                    first_post = h
                    break
            if first_post is not None:
                opportunities += 1
                if first_post.response is not None and first_post.response.ok:
                    adapted += 1

        if opportunities == 0:
            return (self.name, 0.0, {"reason": "no post-drift calls yet"})
        return (
            self.name,
            adapted / opportunities,
            {"adapted": adapted, "opportunities": opportunities},
        )


class EfficiencyRubric(Rubric):
    name = "efficiency"

    def score(self, state: EpisodeState) -> tuple[str, float, dict]:
        step_eff = max(0.0, 1.0 - state.step / state.max_steps)
        tok_eff = state.token_budget_remaining / max(1, state.token_budget)
        return (
            self.name,
            0.5 * step_eff + 0.5 * tok_eff,
            {"step_eff": step_eff, "tok_eff": tok_eff},
        )


# ─────────────────────────────────────────────────────────────────
# Composition primitives
# ─────────────────────────────────────────────────────────────────

class WeightedSum:
    def __init__(self, rubrics_with_weights: list[tuple[Rubric, float]]) -> None:
        total_w = sum(w for _, w in rubrics_with_weights)
        assert abs(total_w - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total_w}"
        self.items = rubrics_with_weights

    def score(self, state: EpisodeState) -> tuple[float, dict]:
        total = 0.0
        breakdown: dict[str, dict] = {}
        for rubric, w in self.items:
            name, val, details = rubric.score(state)
            total += w * val
            breakdown[name] = {
                "value": val,
                "weight": w,
                "weighted": w * val,
                "details": details,
            }
        return total, breakdown


class Gate:
    def __init__(self, name: str, predicate: Callable[[EpisodeState], bool]) -> None:
        self.name = name
        self.predicate = predicate

    def score(self, state: EpisodeState) -> tuple[str, float]:
        return (self.name, 1.0 if self.predicate(state) else 0.0)


# ─────────────────────────────────────────────────────────────────
# Dense step-level shaping (GRPO convergence fix)
# ─────────────────────────────────────────────────────────────────

def compute_step_shaping(
    state: EpisodeState,
    action: Action,
    response: ToolResponse | None,
) -> float:
    """Dense step-level reward — fires during episode to densify GRPO signal."""
    shaped = 0.0
    history = state.history
    prev_response = history[-1].response if history else None

    # +0.10 — inspecting schema right after a failure
    if action.type == "inspect_schema":
        if prev_response is not None and not prev_response.ok:
            shaped += 0.10

    # +0.15 — correct drift report (matching tool + kind, already fired, not yet detected)
    if action.type == "report_drift" and action.report is not None:
        for d in state.drift_plan:
            matches = (
                d.tool == action.report.tool
                and d.kind == action.report.drift_kind
                and d.fires_at_step <= state.step
                and not d.detected_by_agent
            )
            if matches:
                shaped += 0.15
                break

    # +0.20 — successful retry after prior failure (recovery from drift)
    if action.type == "retry_with_variant":
        if (response is not None and response.ok
                and prev_response is not None and not prev_response.ok):
            shaped += 0.20

    # -0.05 — dumb retry (same failing endpoint twice without inspecting)
    if action.type == "call_tool" and prev_response is not None and not prev_response.ok:
        if history:
            prev_action = history[-1].action
            if (prev_action.type == "call_tool"
                    and prev_action.tool_call is not None
                    and action.tool_call is not None
                    and prev_action.tool_call.endpoint == action.tool_call.endpoint
                    and prev_action.tool_call.tool == action.tool_call.tool):
                shaped -= 0.05

    return shaped


# ─────────────────────────────────────────────────────────────────
# Match helper + final-state gate
# ─────────────────────────────────────────────────────────────────

def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return isinstance(actual, list) and all(e in actual for e in expected)
    if isinstance(expected, bool):
        return actual == expected
    return actual == expected


def _final_state_acceptable(state: EpisodeState) -> bool:
    """Gate: lenient mid-episode (1.0), strict at terminal (0.0 if GT unmet)."""
    if state.agent_state.get("catastrophe", False):
        return False
    if not state.done:
        return True
    gt = state.ground_truth_final_state
    for k, v in gt.items():
        if not _matches(state.agent_state.get(k), v):
            return False
    return True


# ─────────────────────────────────────────────────────────────────
# Composed grader
# ─────────────────────────────────────────────────────────────────

def build_grader() -> Callable[[EpisodeState], RewardBreakdown]:
    weighted = WeightedSum([
        (CompletionRubric(),     0.40),
        (DriftDetectionRubric(), 0.25),
        (AdaptationRubric(),     0.20),
        (EfficiencyRubric(),     0.15),
    ])

    gates = [
        Gate("catastrophic_ok", lambda s: not s.agent_state.get("catastrophe", False)),
        Gate("correct_final_gate", _final_state_acceptable),
    ]

    def grade(state: EpisodeState) -> RewardBreakdown:
        weighted_score, breakdown = weighted.score(state)

        gate_vals: dict[str, float] = {}
        gate_product = 1.0
        for g in gates:
            name, v = g.score(state)
            gate_vals[name] = v
            gate_product *= v

        shaped = weighted_score * gate_product
        completion_val = breakdown["task_completion"]["value"]
        binary = 1.0 if (completion_val >= 0.95 and gate_product >= 1.0) else 0.0

        return RewardBreakdown(
            task_completion=completion_val,
            drift_detection=breakdown["drift_detection"]["value"],
            adaptation_quality=breakdown["adaptation_quality"]["value"],
            efficiency=breakdown["efficiency"]["value"],
            catastrophic_gate=gate_vals["catastrophic_ok"],
            correct_final_gate=gate_vals["correct_final_gate"],
            step_shaping=0.0,
            shaped_total=shaped,
            binary=binary,
        )

    return grade
