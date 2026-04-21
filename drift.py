"""Drift injector — fires scheduled DriftEvents at the configured step."""
from __future__ import annotations

from models import DriftEvent, EpisodeState


class DriftInjector:
    """Stateless helper — reads drift plan from state and mutates tools."""

    @staticmethod
    def tick(state: EpisodeState, tools: dict) -> list[DriftEvent]:
        """Fire any drifts whose fires_at_step == current state.step.

        Returns list of events that fired this tick. Does NOT mark
        detected_by_agent — that's done in environment.step() when
        agent calls report_drift.
        """
        fired = []
        for event in state.drift_plan:
            if event.fires_at_step == state.step and not _already_fired(event):
                tools[event.tool].apply_drift(event)
                event.details["_fired"] = True
                fired.append(event)
        return fired


def _already_fired(event: DriftEvent) -> bool:
    return event.details.get("_fired", False)
