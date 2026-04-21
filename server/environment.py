"""SchemaShiftEnvironment — episode scheduler. reset/step loop with drift ticks + grader.

Round 1 bug prevention: step() raises RuntimeError if called before reset(). Never lazy-init.
"""
from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any

from drift import DriftInjector
from graders import build_grader, compute_step_shaping
from models import (
    Action,
    DriftReportParams,
    EpisodeState,
    HistoryStep,
    Observation,
    RewardBreakdown,
    ToolResponse,
)
from scenarios import SCENARIOS


def _instantiate_tool(name: str, seed_data: dict) -> Any:
    """Lazy import so missing stretch tools don't break core scenarios."""
    if name == "mail":
        from tools.mail import MailAPI
        return MailAPI(seed_data)
    if name == "calendar":
        from tools.calendar import CalendarAPI
        return CalendarAPI(seed_data)
    if name == "crm":
        from tools.crm import CRMAPI
        return CRMAPI(seed_data)
    if name == "chat":
        from tools.chat import ChatAPI  # type: ignore[attr-defined]
        return ChatAPI(seed_data)
    if name == "docs":
        from tools.docs import DocsAPI  # type: ignore[attr-defined]
        return DocsAPI(seed_data)
    raise ValueError(f"Unknown tool: {name}")


class SchemaShiftEnvironment:
    """The SchemaShift RL environment. One instance = one episode at a time."""

    def __init__(self) -> None:
        self._state: EpisodeState | None = None
        self._tools: dict[str, Any] = {}
        self._grader = build_grader()

    # ──────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────

    def reset(self, task_id: str, seed: int = 0) -> Observation:
        if task_id not in SCENARIOS:
            raise ValueError(
                f"Unknown task_id: {task_id}. Available: {list(SCENARIOS.keys())}"
            )

        scenario = SCENARIOS[task_id]
        required_tools = scenario["required_tools"]

        self._tools = {}
        for tool_name in required_tools:
            tool_seed = scenario["seed_data"].get(tool_name, {})
            self._tools[tool_name] = _instantiate_tool(tool_name, tool_seed)

        self._state = EpisodeState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            difficulty=scenario["difficulty"],
            step=0,
            max_steps=scenario["max_steps"],
            token_budget=scenario["token_budget"],
            token_budget_remaining=scenario["token_budget"],
            drift_plan=deepcopy(scenario["drift_plan"]),
            ground_truth_final_state=dict(scenario["ground_truth_final_state"]),
            agent_state={},
            history=[],
            done=False,
            cumulative_reward=0.0,
        )

        return self._observation("Episode started.")

    def step(
        self, action: Action, tokens_used: int = 0
    ) -> tuple[Observation, RewardBreakdown]:
        if self._state is None:
            raise RuntimeError(
                "Call reset() before step(). "
                "SchemaShiftEnvironment requires an active episode."
            )
        s = self._state
        if s.done:
            raise RuntimeError(
                "Episode already done. Call reset() to start a new episode."
            )

        s.step += 1
        s.token_budget_remaining = max(0, s.token_budget_remaining - tokens_used)

        # 1. Apply any scheduled drifts for this step
        fired_drifts = DriftInjector.tick(s, self._tools)

        # 2. Dispatch action (does NOT mark drift detected — that happens after shaping)
        response, feedback = self._dispatch_action(action)

        # 3. Compute step shaping BEFORE marking drift detected
        #    (shaping checks `not d.detected_by_agent` — must run pre-mark)
        step_shape = compute_step_shaping(s, action, response)

        # 4. Now apply drift-detection mark (so grader sees the detection this step)
        if action.type == "report_drift" and action.report is not None:
            self._mark_drift_detected(action.report)

        # 5. Update agent_state from action+response
        self._update_agent_state(action, response)

        # 6. Log the history step (reward will be filled below)
        history_step = HistoryStep(
            step=s.step, action=action, response=response, reward_breakdown=None,
        )
        s.history.append(history_step)

        # 7. Check terminal conditions
        if s.step >= s.max_steps or s.token_budget_remaining <= 0:
            s.done = True
        if action.type == "complete_task":
            s.done = True

        # 8. Run grader (sees marked drifts + updated agent_state + done flag)
        reward = self._grader(s)
        reward.step_shaping = step_shape
        reward.shaped_total += step_shape

        # 9. Log reward into history
        s.history[-1].reward_breakdown = reward.model_dump()
        s.cumulative_reward += reward.shaped_total

        # 10. Decorate feedback with drift info and return
        if fired_drifts:
            feedback += (
                f" [DRIFT FIRED: {len(fired_drifts)} drift event(s) on step {s.step}.]"
            )

        return self._observation(feedback), reward

    # ──────────────────────────────────────────────────────────────
    # Action dispatch (pure — no state mutation for drift detection)
    # ──────────────────────────────────────────────────────────────

    def _dispatch_action(self, action: Action) -> tuple[ToolResponse | None, str]:
        s = self._state
        assert s is not None

        if action.type == "call_tool":
            if action.tool_call is None:
                return None, "Invalid call_tool: missing tool_call params."
            if action.tool_call.tool not in self._tools:
                return (
                    ToolResponse(
                        ok=False, status=404,
                        error=f"Tool '{action.tool_call.tool}' not available in this scenario.",
                    ),
                    "Tool not available.",
                )
            response = self._tools[action.tool_call.tool].call(
                action.tool_call.endpoint, action.tool_call.params
            )
            return (
                response,
                f"Called {action.tool_call.tool}.{action.tool_call.endpoint}: status={response.status}",
            )

        if action.type == "inspect_schema":
            if action.inspect is None or action.inspect.tool not in self._tools:
                return (
                    ToolResponse(
                        ok=False, status=404,
                        error="Tool unavailable for inspection.",
                    ),
                    "Inspect target missing.",
                )
            schema = self._tools[action.inspect.tool].get_schema()
            return (
                ToolResponse(ok=True, status=200, body={"schema": schema}),
                f"Inspected {action.inspect.tool} schema.",
            )

        if action.type == "retry_with_variant":
            if action.retry is None or action.retry.tool not in self._tools:
                return (
                    ToolResponse(
                        ok=False, status=404,
                        error="Retry target unavailable.",
                    ),
                    "Retry target missing.",
                )
            response = self._tools[action.retry.tool].call(
                action.retry.endpoint, action.retry.params
            )
            return (
                response,
                f"Retried {action.retry.tool}.{action.retry.endpoint}: status={response.status}",
            )

        if action.type == "report_drift":
            if action.report is None:
                return None, "Invalid report_drift: missing report params."
            for d in s.drift_plan:
                if (d.tool == action.report.tool
                        and d.kind == action.report.drift_kind
                        and d.fires_at_step <= s.step
                        and not d.detected_by_agent):
                    return (
                        None,
                        f"Drift correctly reported: {d.kind} on {d.tool} at step {d.fires_at_step}.",
                    )
            return None, "Drift report did not match any undetected fired drift."

        if action.type == "complete_task":
            summary = action.complete.summary if action.complete else ""
            s.agent_state["_completion_summary"] = summary
            self._check_completion_summary(summary)
            return None, f"Episode marked complete. Summary: {summary[:80]}"

        return None, f"Unknown action type: {action.type}"

    # ──────────────────────────────────────────────────────────────
    # Post-shaping state mutations
    # ──────────────────────────────────────────────────────────────

    def _mark_drift_detected(self, report: DriftReportParams) -> None:
        s = self._state
        assert s is not None
        for d in s.drift_plan:
            if (d.tool == report.tool
                    and d.kind == report.drift_kind
                    and d.fires_at_step <= s.step
                    and not d.detected_by_agent):
                d.detected_by_agent = True
                return

    # ──────────────────────────────────────────────────────────────
    # State tracking — populates agent_state so grader can read it
    # ──────────────────────────────────────────────────────────────

    def _update_agent_state(
        self, action: Action, response: ToolResponse | None
    ) -> None:
        if response is None or not response.ok:
            return
        s = self._state
        assert s is not None
        st = s.agent_state

        tool: str | None = None
        endpoint: str | None = None
        params: dict = {}
        if action.type == "call_tool" and action.tool_call is not None:
            tool = action.tool_call.tool
            endpoint = action.tool_call.endpoint
            params = action.tool_call.params
        elif action.type == "retry_with_variant" and action.retry is not None:
            tool = action.retry.tool
            endpoint = action.retry.endpoint
            params = action.retry.params
        else:
            return

        # MAIL ────────────────────────────────────────────────────
        if tool == "mail":
            if endpoint in ("send_message", "messages.send"):
                st["mail.sent_count"] = st.get("mail.sent_count", 0) + 1
                sent_to = params.get("to", "")
                st["mail.last_sent_to"] = sent_to
                subject = str(params.get("subject", "")).lower()
                if "welcome" in subject:
                    st["mail.last_subject_contains_welcome"] = True
                if "all-hands" in subject or "all hands" in subject:
                    st["mail.last_subject_contains_allhands"] = True
                if "priority support" in subject:
                    st["mail.last_subject_contains_priority_support"] = True
                if "weekly" in subject:
                    st["mail.last_subject_contains_weekly"] = True
                if "calendar updated" in subject:
                    st["mail.last_subject_contains_calendar_updated"] = True
                recipients: list[str] = st.get("mail.all_recipients", [])
                if sent_to and sent_to not in recipients:
                    recipients.append(sent_to)
                    st["mail.all_recipients"] = recipients
                e2_required = {"alex@company.com", "jordan@company.com", "sam@company.com"}
                if e2_required.issubset(set(recipients)):
                    st["mail.sent_to_all_three_recipients"] = True

        # CALENDAR ────────────────────────────────────────────────
        if tool == "calendar":
            if endpoint == "create_event":
                st["calendar.events_count"] = st.get("calendar.events_count", 0) + 1
                body = response.body or {}
                raw = body.get("attendees") or body.get("participants") or []
                emails: list[str] = []
                for a in raw:
                    if isinstance(a, str):
                        emails.append(a)
                    elif isinstance(a, dict):
                        emails.append(a.get("email", ""))
                st["calendar.last_event_attendees"] = emails
                # Recognised attendee pairs (E1 + M1 share this key by design).
                priya_alex = (
                    "priya@company.com" in emails and "alex@company.com" in emails
                )
                bob_alex = (
                    "bob@customer.com" in emails and "alex@company.com" in emails
                )
                if priya_alex or bob_alex:
                    st["calendar.last_event_has_both_attendees"] = True
                if "sarah@company.com" in emails and "mike@company.com" in emails:
                    st["calendar.last_event_has_both_sales_leads"] = True
                # M3: Friday Wrap-up event counter
                title = str(body.get("title") or params.get("title") or "").lower()
                if "friday wrap-up" in title:
                    st["calendar.events_count_new_friday_wrapup"] = (
                        st.get("calendar.events_count_new_friday_wrapup", 0) + 1
                    )
            elif endpoint == "update_event":
                # M3: track per-event status transitions (cancellations)
                event_id = params.get("event_id", "")
                status = params.get("status")
                if event_id and status:
                    st[f"calendar.{event_id}_status"] = status

        # CRM ─────────────────────────────────────────────────────
        if tool == "crm":
            if endpoint in ("update_contact", "contacts.patch"):
                cid = params.get("contact_id", "")
                status = params.get("status")
                if cid and status:
                    st[f"crm.contact_{cid}_status"] = status

    def _check_completion_summary(self, summary: str) -> None:
        s = self._state
        assert s is not None
        st = s.agent_state
        gt = s.ground_truth_final_state
        if "complete_summary_mentions_company" in gt:
            for c in ("Globex", "Acme", "Initech"):
                if c.lower() in summary.lower():
                    st["complete_summary_mentions_company"] = True
                    break

    # ──────────────────────────────────────────────────────────────
    # Observation construction
    # ──────────────────────────────────────────────────────────────

    def _observation(self, feedback: str) -> Observation:
        s = self._state
        assert s is not None
        scenario = SCENARIOS[s.task_id]

        return Observation(
            episode_id=s.episode_id,
            task_id=s.task_id,
            difficulty=s.difficulty,
            step=s.step,
            max_steps=s.max_steps,
            token_budget_remaining=s.token_budget_remaining,
            task_description=scenario["task_description"],
            success_criteria=list(scenario["success_criteria"]),
            tool_schemas={name: t.get_schema() for name, t in self._tools.items()},
            known_state=dict(s.agent_state),
            history=list(s.history[-5:]),
            last_response=s.history[-1].response if s.history else None,
            drift_events_visible=[
                {
                    "tool": d.tool,
                    "kind": d.kind,
                    "endpoint": d.endpoint,
                    "fires_at_step": d.fires_at_step,
                }
                for d in s.drift_plan
                if d.detected_by_agent
            ],
            done=s.done,
            feedback=feedback,
        )
