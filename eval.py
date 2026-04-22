"""Evaluation harness for SchemaShift baselines.

Supports:
  - naive_heuristic: always calls first endpoint of first tool (floor baseline)
  - policy_aware_heuristic: on 4xx/5xx, inspects schema, reports drift, retries
  - openai:<model>: OpenAI API (GPT-4o-mini is the pitch target)
  - hf:<model_id> (or llm:<model_id>): HF Inference Router
  - checkpoint:<hub_id>: trained model checkpoint (Phase 13)

Usage:
    python eval.py --baseline naive_heuristic --seeds 0,1,2,3,4
    python eval.py --baseline policy_aware_heuristic --seeds 0,1,2,3,4
    python eval.py --baseline openai:gpt-4o-mini --seeds 0,1,2,3,4
    python eval.py --baseline hf:Qwen/Qwen2.5-7B-Instruct --seeds 0,1,2,3,4

Output: markdown table to stdout + JSON to eval_results/<baseline>_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from client import SchemaShiftEnvClient
from models import (
    Action,
    CompleteParams,
    DriftReportParams,
    InspectParams,
    Observation,
    RetryParams,
    RewardBreakdown,
    ToolCallParams,
)


# ═══════════════════════════════════════════════════════════════════════
# Agent protocol
# ═══════════════════════════════════════════════════════════════════════

class BaseAgent:
    name: str = "base"

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════════
# Baseline 1: Naive heuristic — floor
# ═══════════════════════════════════════════════════════════════════════

class NaiveHeuristicAgent(BaseAgent):
    """Always calls first endpoint with empty params, then completes. Expected score: ~0."""
    name = "naive_heuristic"

    def __init__(self) -> None:
        self._step_count = 0

    def reset(self) -> None:
        self._step_count = 0

    def act(self, obs: Observation) -> Action:
        self._step_count += 1
        if self._step_count >= 3:
            return Action(
                type="complete_task",
                complete=CompleteParams(summary="Naive agent done."),
            )
        if obs.tool_schemas:
            tool_name = list(obs.tool_schemas.keys())[0]
            schemas = obs.tool_schemas[tool_name]
            if isinstance(schemas, dict) and schemas:
                endpoint = list(schemas.keys())[0]
                return Action(
                    type="call_tool",
                    tool_call=ToolCallParams(
                        tool=tool_name, endpoint=endpoint, params={},
                    ),
                )
        return Action(
            type="complete_task",
            complete=CompleteParams(summary="No tools."),
        )


# ═══════════════════════════════════════════════════════════════════════
# Baseline 2: Policy-aware heuristic — ceiling for rule-based
# ═══════════════════════════════════════════════════════════════════════

class PolicyAwareHeuristicAgent(BaseAgent):
    """Inspects schema after 4xx/5xx, reports drift, retries with adapted params."""
    name = "policy_aware_heuristic"

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_action_was_call: bool = False
        self._last_tool_called: Optional[str] = None
        self._last_endpoint_called: Optional[str] = None
        self._last_params: dict = {}
        self._inspected_tools: set = set()
        self._reported_drifts: set = set()
        self._retried_tools: set = set()
        self._contact_id_seen: Optional[str] = None
        self._last_company_seen: Optional[str] = None
        self._attempted_update: bool = False

    def act(self, obs: Observation) -> Action:
        self._capture_crm_metadata(obs)

        # PRIORITY 1: inspect after failure
        if (
            self._last_action_was_call
            and obs.last_response is not None
            and not obs.last_response.ok
            and self._last_tool_called is not None
            and self._last_tool_called not in self._inspected_tools
        ):
            tool = self._last_tool_called
            self._inspected_tools.add(tool)
            self._last_action_was_call = False
            return Action(type="inspect_schema", inspect=InspectParams(tool=tool))

        # PRIORITY 2: report drift after inspecting
        if (
            self._last_tool_called is not None
            and self._last_tool_called in self._inspected_tools
            and self._last_tool_called not in self._reported_drifts
        ):
            tool = self._last_tool_called
            self._reported_drifts.add(tool)
            kind = self._guess_drift_kind(obs, tool)
            return Action(
                type="report_drift",
                report=DriftReportParams(
                    tool=tool,
                    drift_kind=kind,
                    description=f"Detected {kind} on {tool}",
                ),
            )

        # PRIORITY 3: retry with adapted params (once per tool)
        if (
            self._last_tool_called is not None
            and self._last_tool_called in self._reported_drifts
            and self._last_tool_called not in self._retried_tools
            and self._last_endpoint_called is not None
        ):
            tool = self._last_tool_called
            self._retried_tools.add(tool)
            new_endpoint = self._adapt_endpoint(obs, tool, self._last_endpoint_called)
            new_params = self._adapt_params(obs, tool, new_endpoint, self._last_params)
            self._last_action_was_call = True
            self._last_endpoint_called = new_endpoint
            self._last_params = new_params
            return Action(
                type="retry_with_variant",
                retry=RetryParams(tool=tool, endpoint=new_endpoint, params=new_params),
            )

        # Step-budget safety net
        if obs.step >= obs.max_steps - 1:
            return Action(
                type="complete_task",
                complete=CompleteParams(summary=self._compose_summary()),
            )

        # PRIORITY 4: task-specific action
        return self._task_specific_action(obs)

    # ─────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────

    def _capture_crm_metadata(self, obs: Observation) -> None:
        """Pick up contact_id + company from the most recent successful CRM response."""
        if (
            self._last_tool_called == "crm"
            and obs.last_response is not None
            and obs.last_response.ok
            and obs.last_response.body
        ):
            body = obs.last_response.body
            contacts = body.get("contacts")
            if isinstance(contacts, list) and contacts:
                c = contacts[0]
                self._contact_id_seen = c.get("contact_id", self._contact_id_seen)
                self._last_company_seen = c.get("company", self._last_company_seen)

    def _task_specific_action(self, obs: Observation) -> Action:
        desc = obs.task_description.lower()
        ks = obs.known_state
        mail_done = ks.get("mail.sent_count", 0) > 0
        calendar_done = ks.get("calendar.events_count", 0) > 0
        crm_update_done = any(
            k.startswith("crm.contact_") and k.endswith("_status") for k in ks
        )
        mail_avail = "mail" in obs.tool_schemas
        calendar_avail = "calendar" in obs.tool_schemas
        crm_avail = "crm" in obs.tool_schemas

        # Mail
        if mail_avail and ("email" in desc or "send" in desc) and not mail_done:
            to_match = re.search(r"([\w.+-]+@[\w-]+(?:\.[\w-]+)+)", obs.task_description)
            to = to_match.group(1) if to_match else "recipient@company.com"
            if "welcome" in desc:
                subject = "Welcome!"
            elif "all-hands" in desc or "all hands" in desc:
                subject = "All-Hands: Friday 3pm"
            else:
                subject = "Update"
            self._last_action_was_call = True
            self._last_tool_called = "mail"
            self._last_endpoint_called = "send_message"
            self._last_params = {"to": to, "subject": subject, "body": "Automated message."}
            return Action(
                type="call_tool",
                tool_call=ToolCallParams(
                    tool="mail", endpoint="send_message", params=self._last_params,
                ),
            )

        # Calendar
        if (
            calendar_avail
            and ("calendar" in desc or "event" in desc or "orientation" in desc)
            and not calendar_done
        ):
            self._last_action_was_call = True
            self._last_tool_called = "calendar"
            self._last_endpoint_called = "create_event"
            self._last_params = {
                "title": "New Hire Orientation" if "orientation" in desc else "Event",
                "start": "2026-04-27T10:00:00Z",
                "end": "2026-04-27T11:00:00Z",
                "attendees": ["priya@company.com", "alex@company.com"],
            }
            return Action(
                type="call_tool",
                tool_call=ToolCallParams(
                    tool="calendar", endpoint="create_event", params=self._last_params,
                ),
            )

        # CRM: search first, then update (if task mentions "update")
        if crm_avail and (
            "customer" in desc or "crm" in desc or "contact" in desc or "lookup" in desc
        ):
            if self._contact_id_seen is None:
                email_match = re.search(r"([\w.+-]+@[\w-]+(?:\.[\w-]+)+)", obs.task_description)
                email = email_match.group(1) if email_match else "bob@customer.com"
                self._last_action_was_call = True
                self._last_tool_called = "crm"
                self._last_endpoint_called = "search_contacts"
                self._last_params = {"customer_email": email}
                return Action(
                    type="call_tool",
                    tool_call=ToolCallParams(
                        tool="crm", endpoint="search_contacts", params=self._last_params,
                    ),
                )
            if "update" in desc and not crm_update_done and not self._attempted_update:
                status_match = re.search(r"'([\w_]+)'", obs.task_description)
                status = status_match.group(1) if status_match else "updated"
                self._attempted_update = True
                self._last_action_was_call = True
                self._last_tool_called = "crm"
                self._last_endpoint_called = "update_contact"
                self._last_params = {
                    "contact_id": self._contact_id_seen,
                    "status": status,
                }
                return Action(
                    type="call_tool",
                    tool_call=ToolCallParams(
                        tool="crm", endpoint="update_contact", params=self._last_params,
                    ),
                )

        return Action(
            type="complete_task",
            complete=CompleteParams(summary=self._compose_summary()),
        )

    def _compose_summary(self) -> str:
        if self._last_company_seen:
            return f"Task complete. Contact's company: {self._last_company_seen}."
        return "Policy-aware agent completed."

    def _guess_drift_kind(self, obs: Observation, tool: str) -> str:
        schema = obs.tool_schemas.get(tool, {})
        if tool == "calendar":
            cs = schema.get("create_event", {})
            if "participants" in cs.get("params", {}):
                return "field_rename"
        if tool == "mail":
            if "messages.send" in schema and "send_message" not in schema:
                return "endpoint_deprecation"
        if tool == "crm":
            cs = schema.get("search_contacts", {})
            if "email_address" in cs.get("params", {}):
                return "field_rename"
        return "field_rename"

    def _adapt_endpoint(
        self, obs: Observation, tool: str, old_endpoint: str
    ) -> str:
        schemas = obs.tool_schemas.get(tool, {})
        if old_endpoint not in schemas:
            if tool == "mail" and old_endpoint == "send_message" and "messages.send" in schemas:
                return "messages.send"
            if tool == "crm" and old_endpoint == "update_contact" and "contacts.patch" in schemas:
                return "contacts.patch"
        return old_endpoint

    def _adapt_params(
        self, obs: Observation, tool: str, endpoint: str, old_params: dict
    ) -> dict:
        schema = obs.tool_schemas.get(tool, {}).get(endpoint, {})
        if not schema:
            return old_params
        new_params = dict(old_params)
        schema_params = schema.get("params", {})
        # Calendar: attendees → participants (list of dicts)
        if tool == "calendar" and endpoint == "create_event" and "participants" in schema_params:
            if "attendees" in new_params:
                attendees = new_params.pop("attendees")
                new_params["participants"] = [
                    {"email": e, "role": "required"} for e in attendees
                ]
        # CRM: customer_email → email_address
        if tool == "crm" and "email_address" in schema_params:
            if "customer_email" in new_params:
                new_params["email_address"] = new_params.pop("customer_email")
        # Strip unknown params (strict base.py validation will reject them otherwise)
        valid = set(schema_params.keys())
        new_params = {k: v for k, v in new_params.items() if k in valid}
        return new_params


# ═══════════════════════════════════════════════════════════════════════
# Baselines 3-5: LLM agents (HF Inference Router + OpenAI)
# ═══════════════════════════════════════════════════════════════════════

class LLMAgent(BaseAgent):
    """Generic LLM-backed agent. Configure via provider + model_id."""

    def __init__(self, provider: str, model_id: str) -> None:
        self.provider = provider
        self.model_id = model_id
        self.name = f"{provider}:{model_id}"
        self._client = None
        self._setup()

    def _setup(self) -> None:
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                raise RuntimeError("openai package not installed.")
        elif self.provider == "hf":
            try:
                from huggingface_hub import InferenceClient
                self._client = InferenceClient(
                    model=self.model_id, token=os.getenv("HF_TOKEN")
                )
            except ImportError:
                raise RuntimeError("huggingface_hub not installed.")
        elif self.provider == "ollama":
            key = os.getenv("OLLAMA_API_KEY")
            if not key:
                raise RuntimeError(
                    "OLLAMA_API_KEY not set (populate .env or export the variable)."
                )
            self._ollama_key = key
            self._client = None  # httpx call is stateless; no client object needed
        elif self.provider == "checkpoint":
            raise NotImplementedError("Checkpoint loading implemented in Phase 13.")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def act(self, obs: Observation) -> Action:
        prompt = self._build_prompt(obs)
        try:
            text = self._call(prompt)
        except Exception as e:
            return Action(
                type="complete_task",
                complete=CompleteParams(summary=f"LLM error: {e}"),
            )
        return self._parse(text)

    def _build_prompt(self, obs: Observation) -> str:
        schemas_text = json.dumps(obs.tool_schemas, indent=2) if obs.tool_schemas else "{}"
        history_text = self._format_history(obs.history)
        return f"""You are an autonomous workflow agent. Complete the task using tools.

TASK: {obs.task_description}

SUCCESS CRITERIA:
{chr(10).join(f'- {c}' for c in obs.success_criteria)}

CURRENT TOOL SCHEMAS:
{schemas_text}

RECENT HISTORY (last {len(obs.history)} steps):
{history_text}

AVAILABLE ACTIONS:
- call_tool(tool, endpoint, params)
- inspect_schema(tool)
- retry_with_variant(tool, endpoint, params)
- report_drift(tool, drift_kind, description)
- complete_task(summary)

IMPORTANT: Output ONLY a JSON object matching the Action schema. No explanation, no markdown, no code fences.

Example:
{{"type": "call_tool", "tool_call": {{"tool": "mail", "endpoint": "send_message", "params": {{"to": "x@y.com", "subject": "Hi", "body": "Hello"}}}}}}

Your next action:"""

    def _format_history(self, history: list) -> str:
        if not history:
            return "(no history yet)"
        lines = []
        for step in history[-3:]:
            act_type = step.action.type
            resp_status = step.response.status if step.response else "N/A"
            resp_ok = step.response.ok if step.response else False
            lines.append(f"Step {step.step}: {act_type} -> status={resp_status}, ok={resp_ok}")
        return "\n".join(lines)

    def _call(self, prompt: str) -> str:
        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0,
            )
            return response.choices[0].message.content or ""
        if self.provider == "hf":
            response = self._client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.01,
            )
            return response.choices[0].message.content or ""
        if self.provider == "ollama":
            import httpx
            r = httpx.post(
                "https://ollama.com/api/chat",
                headers={"Authorization": f"Bearer {self._ollama_key}"},
                json={
                    "model": self.model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 500},
                },
                timeout=120.0,
            )
            r.raise_for_status()
            body = r.json()
            return body.get("message", {}).get("content", "") or ""
        raise ValueError(f"Provider not callable: {self.provider}")

    def _parse(self, text: str) -> Action:
        """Brace-balanced JSON extractor with graceful complete_task fallback."""
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", text)
        depth = 0
        start = -1
        for i, ch in enumerate(cleaned):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start >= 0:
                    chunk = cleaned[start:i + 1]
                    try:
                        obj = json.loads(chunk)
                        return Action.model_validate(obj)
                    except Exception:
                        start = -1
                        continue
        return Action(
            type="complete_task",
            complete=CompleteParams(summary=f"Parse error — LLM output: {text[:120]}"),
        )


# ═══════════════════════════════════════════════════════════════════════
# Agent factory
# ═══════════════════════════════════════════════════════════════════════

def build_agent(baseline: str) -> BaseAgent:
    if baseline == "naive_heuristic":
        return NaiveHeuristicAgent()
    if baseline == "policy_aware_heuristic":
        return PolicyAwareHeuristicAgent()
    if baseline.startswith("openai:"):
        return LLMAgent(provider="openai", model_id=baseline.split(":", 1)[1])
    if baseline.startswith("llm:") or baseline.startswith("hf:"):
        return LLMAgent(provider="hf", model_id=baseline.split(":", 1)[1])
    if baseline.startswith("ollama:"):
        return LLMAgent(provider="ollama", model_id=baseline.split(":", 1)[1])
    if baseline.startswith("checkpoint:"):
        return LLMAgent(provider="checkpoint", model_id=baseline.split(":", 1)[1])
    raise ValueError(f"Unknown baseline: {baseline}")


# ═══════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EpisodeResult:
    task_id: str
    seed: int
    completion: float = 0.0
    drift_detection: float = 0.0
    adaptation: float = 0.0
    efficiency: float = 0.0
    shaped_total: float = 0.0
    cumulative_reward: float = 0.0
    binary: float = 0.0
    steps_used: int = 0
    final_action_type: str = ""
    error: Optional[str] = None


def run_episode(
    agent: BaseAgent,
    client: SchemaShiftEnvClient,
    task_id: str,
    seed: int,
) -> EpisodeResult:
    result = EpisodeResult(task_id=task_id, seed=seed)
    try:
        agent.reset()
        obs = client.reset(task_id, seed=seed)
        last_reward: Optional[RewardBreakdown] = None
        while not obs.done:
            action = agent.act(obs)
            obs, reward = client.step(action, tokens_used=0)
            last_reward = reward
            result.final_action_type = action.type
            result.steps_used = obs.step
        if last_reward:
            result.completion = last_reward.task_completion
            result.drift_detection = last_reward.drift_detection
            result.adaptation = last_reward.adaptation_quality
            result.efficiency = last_reward.efficiency
            result.shaped_total = last_reward.shaped_total
            result.binary = last_reward.binary
        try:
            grader = client.get_grader()
            result.cumulative_reward = float(grader.get("cumulative_reward", 0.0))
        except Exception:
            pass
    except Exception as e:
        result.error = str(e)
    return result


# ═══════════════════════════════════════════════════════════════════════
# Output formatting
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_TASKS = ["E1_onboard_new_hire", "E2_meeting_invite_blast", "E3_customer_lookup"]


def print_baseline_table(baseline: str, results: list[EpisodeResult]) -> str:
    lines = [f"## Eval results — {baseline}", ""]
    lines.append("| Task | Seed | Compl | Drift | Adapt | Effic | Shaped | Cumul | Binary |")
    lines.append("|------|------|-------|-------|-------|-------|--------|-------|--------|")
    for r in results:
        lines.append(
            f"| {r.task_id[:16]} | {r.seed} | {r.completion:.3f} | {r.drift_detection:.3f} | "
            f"{r.adaptation:.3f} | {r.efficiency:.3f} | {r.shaped_total:.3f} | "
            f"{r.cumulative_reward:.3f} | {r.binary:.0f} |"
        )
    lines.append("")
    lines.append("### Aggregates")
    by_task: dict[str, list[EpisodeResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)
    for task_id, rs in by_task.items():
        mean_shaped = sum(r.shaped_total for r in rs) / len(rs)
        mean_cumul = sum(r.cumulative_reward for r in rs) / len(rs)
        binary_rate = sum(r.binary for r in rs) / len(rs)
        lines.append(
            f"- **{task_id}**: mean_shaped={mean_shaped:.3f}, "
            f"mean_cumul={mean_cumul:.3f}, binary_rate={binary_rate:.2%}"
        )
    if results:
        overall_shaped = sum(r.shaped_total for r in results) / len(results)
        overall_cumul = sum(r.cumulative_reward for r in results) / len(results)
        overall_binary = sum(r.binary for r in results) / len(results)
        lines.append(
            f"- **OVERALL**: mean_shaped={overall_shaped:.3f}, "
            f"mean_cumul={overall_cumul:.3f}, binary_rate={overall_binary:.2%}"
        )
    return "\n".join(lines)


def save_results_json(
    baseline: str, results: list[EpisodeResult], out_dir: Path
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = baseline.replace("/", "_").replace(":", "_")
    path = out_dir / f"{safe_name}_{timestamp}.json"
    payload = {
        "baseline": baseline,
        "timestamp": timestamp,
        "results": [r.__dict__ for r in results],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description="SchemaShift baseline eval")
    parser.add_argument(
        "--baseline", required=True,
        help="Agent: naive_heuristic | policy_aware_heuristic | openai:<model> | hf:<model_id>",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument(
        "--url", default=os.getenv("SCHEMASHIFT_URL", "http://localhost:7860"),
    )
    parser.add_argument("--out-dir", default="eval_results")
    args = parser.parse_args()

    # Load .env so secrets (OLLAMA_API_KEY, OPENAI_API_KEY, HF_TOKEN) are available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"# Eval: {args.baseline}")
    print(f"Tasks: {tasks}")
    print(f"Seeds: {seeds}")
    print(f"Env URL: {args.url}")
    print()

    agent = build_agent(args.baseline)
    results: list[EpisodeResult] = []

    with SchemaShiftEnvClient(base_url=args.url) as client:
        if not client.health():
            print(f"ERROR: Env not reachable at {args.url}. Start server first.")
            return 1
        for task_id in tasks:
            for seed in seeds:
                print(f"  {task_id} seed={seed}...", end=" ", flush=True)
                start = time.time()
                r = run_episode(agent, client, task_id, seed)
                elapsed = time.time() - start
                if r.error:
                    print(f"ERROR: {r.error} ({elapsed:.1f}s)")
                else:
                    print(
                        f"shaped={r.shaped_total:.3f} cumul={r.cumulative_reward:.3f} "
                        f"binary={r.binary:.0f} steps={r.steps_used} ({elapsed:.1f}s)"
                    )
                results.append(r)

    agent.close()

    print()
    table = print_baseline_table(args.baseline, results)
    print(table)
    path = save_results_json(args.baseline, results, Path(args.out_dir))
    print(f"\nResults JSON saved to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
