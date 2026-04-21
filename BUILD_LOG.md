# SchemaShift Build Log

Running log of everything built, tested, trained, and deployed. Append-only — never delete.

**Team Tripod:** Yashash Sheshagiri (lead), Gajanand V Dhayagode, Likith B S
**Event:** Meta × HF × PyTorch OpenEnv Hackathon 2026 · Round 2 · Bangalore · April 25-26

---

## PHASES COMPLETED

### Phase 0 — Project Scaffolding
- **Date:** Tuesday April 21, 2026 (evening)
- **Commit:** `d4ab0f1`
- **Tests:** 0 (scaffolding only — no logic yet)
- **Time:** ~30 min
- **Notes:** 29 files created. Python venv built. Deps installed cleanly (fastapi 0.136, pydantic 2.13, openai 2.32, openenv-core 0.2.3, gradio 6.13). Repo pushed to https://github.com/Yashash4/SchemaShift.

### Phase 1 — Pydantic Models
- **Date:** Tuesday April 21, 2026
- **Commit:** `8cdb02b`
- **Tests:** 6 passing (0.22s runtime)
- **Line counts:** models.py = 135 lines, test_models.py = ~120 lines
- **Time:** ~20 min
- **Judgment calls:**
  1. 12 classes + 2 Literal type aliases (not 13 classes as prompt suggested)
- **Notes:** Spec code copied verbatim. No deviations.

### Phase 2 — Mail Tool
- **Date:** Tuesday April 21, 2026
- **Commit:** `8410ff1`
- **Tests:** 7 new (13 total), 0.16s runtime
- **Line counts:** tools/base.py = 60, tools/mail.py = 121, tests/test_mail.py = 149
- **Time:** ~90 min
- **Judgment calls:**
  1. Absolute imports (`from models import`) over relative imports — matches Phase 1 convention
  2. `get_schema()` returns `{}` for unknown endpoints post-deprecation
  3. 501 fallback in `BaseTool.call` for schema-without-handler (dev-time safety)
  4. `messages.send` is deep-copy rename of `send_message` after deprecation drift
  5. Pagination tokens: `"tok_xyz"` / `"cur_abc123"` when >10 results, `None` otherwise
- **Notes:** 3 endpoints (list_messages, send_message, get_message), 3 drifts (field_rename, endpoint_deprecation, new_required_param).

### Phase 3 — Calendar Tool + DriftInjector
- **Date:** Tuesday April 21, 2026
- **Commit:** `27dd958`
- **Tests:** 10 new (23 total), 0.24s runtime
- **Line counts:** tools/calendar.py = 164, drift.py = 28, tests/test_calendar.py = 182, tests/test_drift.py = 122
- **Time:** ~100 min
- **Judgment calls:**
  1. Overlap predicate for `list_events` (inclusive both sides, ISO string sort)
  2. `_create_event` handles both attendees and participants gracefully
  3. `_update_event` accepts both attendees and participants (no second drift handler needed)
  4. Cancelled events remain in storage (matches Google/Outlook behavior)
  5. `_fired` sentinel in `event.details` (underscore prefix avoids collision)
  6. `rate_limit_tightening` chosen for unknown-drift test (semantically meaningful)
- **Notes:** 4 endpoints (list/create/update/delete_event), 2 drifts (field_rename attendees→participants, tool_removal delete_event). DriftInjector is stateless.

### Phase 4 — CRM Tool + Scenarios E1/E2/E3
- **Date:** Tuesday April 21, 2026
- **Commit:** `bc8f184`
- **Tests:** 11 new (34 total), 0.24s runtime
- **Line counts:** tools/crm.py = 213, scenarios.py = 124, tests/test_crm.py = 142, tests/test_scenarios.py = 65
- **Time:** ~100 min
- **Judgment calls:**
  1. Strict param validation in `BaseTool.call` — rejects unknown params with 400 "Unknown params" (matches Stripe/Google API behavior)
  2. Internal storage stays `customer_email`; projection on read via `_project_contact`
  3. `search_contacts()` with no filters returns ALL contacts (REST convention)
  4. Rate-limit counter resets on drift fire (pre-drift calls don't count against post-drift budget)
  5. Drift-order commutativity for `contacts.patch` (either order produces same final state)
  6. E2 task description doesn't disclose deprecation (agent must discover via 410 response)
  7. E3 drift fires at step 2 (gives agent one clean baseline call)
  8. `DriftEvent.details` populated with hints (`from`, `to`) for grader/scenario auditing
- **Notes:** 4 endpoints (search/get/create/update_contact), 3 drifts (tool-wide field_rename, rate_limit_tightening, endpoint_deprecation update→contacts.patch).

### Phase 5 — Composable Grader + Dense Step Shaping
- **Date:** Tuesday April 21, 2026
- **Commit:** `46b8c56`
- **Tests:** 11 new (45 total), 0.25s runtime
- **Line counts:** graders.py = 267, tests/test_graders.py = 224
- **Time:** ~150 min
- **Judgment calls:**
  1. Binary threshold 0.95 (not 1.0) — floating point safety
  2. `AdaptationRubric` iterates all fired drifts (TODO: revisit for Phase 11 M-scenarios with multi-drift-per-tool)
  3. `_final_state_acceptable` lenient mid-episode, strict at terminal — preserves training signal
  4. `compute_step_shaping` checks `not d.detected_by_agent` — anti-farming guard
  5. Dumb-retry penalty requires same tool AND same endpoint
  6. `assert` for weight-sum invariant (dev-time, not runtime)
  7. Rubric names coupled to `RewardBreakdown` field names (intentional reward contract)
  8. Bonus test asserts `step_shaping = 0` in grader output (env fills it, not grader)
- **Notes:** 4 rubrics (Completion, DriftDetection, Adaptation, Efficiency) + 2 gates (catastrophic, correct_final) + WeightedSum + Gate + `compute_step_shaping` + `build_grader`.

### Phase 6 — Environment Scheduler
- **Date:** Tuesday April 21, 2026
- **Commit:** `c618a61`
- **Tests:** 8 new (53 total), 0.29s runtime
- **Line counts:** server/environment.py = 363, tests/test_environment.py = 242
- **Time:** ~120 min
- **E1 FULL-EPISODE BENCHMARK:**
  - `shaped_total = 0.934375` (expected ~0.93, exact match on weighted sum)
  - `task_completion = 1.0`
  - `drift_detection = 1.0`
  - `adaptation_quality = 1.0`
  - `efficiency = 0.5625` (used 7 of 8 max steps)
  - `catastrophic_gate = 1.0, correct_final_gate = 1.0`
  - `binary = 1.0`
  - `cumulative_reward = 4.2975` across 7 steps
  - **Dense shaping fired correctly:** +0.10 at step 4 (inspect-after-failure), +0.15 at step 6 (drift report)
- **Judgment calls:**
  1. Shaping-before-marking ordering: dispatch→compute_step_shaping→mark detected→grader
  2. `tokens_used` clamped to non-negative with `max(0, remaining - tokens_used)`
  3. Lazy tool imports in `_instantiate_tool` — stretch tools don't break core
  4. `deepcopy(drift_plan)` on reset — prevents state leak across episodes
  5. Observation history windowed to last 5 steps for LLM prompt budget
  6. Calendar body parser handles both attendees (strings) and participants (dicts)
  7. `_completion_summary` stored with underscore prefix (no GT key collision)
  8. E2 three-recipient check tracked in `_update_agent_state`
  9. `done` flag set AFTER dispatch but BEFORE grader (enables strict gate at terminal)
  10. Dispatch returns `None` for `report_drift` and `complete_task` (no tool response)
- **Notes:** Round 1 `_engine is None` bug pattern explicitly prevented via `RuntimeError` in step() when state is None.

### Phase 7 — FastAPI Server + Docker
- **Date:** Tuesday April 21, 2026 (late evening)
- **Commit:** `cb33205`
- **Tests:** 10 new (63 total), 0.94s runtime
- **Line counts:** server/app.py = 120, tests/test_server.py = 129, Dockerfile = 14, openenv.yaml = 14
- **Time:** ~60 min
- **Live server smoke tests:** uvicorn on 127.0.0.1:7860 verified /health, /tasks (count=3), /reset (E1 → step=0, tools=['mail', 'calendar']) all 200 OK
- **`pip install -e .` confirmation:** Editable wheel built successfully. Docker container will resolve absolute imports correctly.
- **Judgment calls:**
  1. Removed root `__init__.py` — conflict with `py-modules` in pyproject
  2. `py-modules` + `packages` split in pyproject (preserves `from models import` convention)
  3. Dropped unused imports from spec's server code
  4. TestClient monkeypatch fixture — fresh env per test prevents cross-contamination
  5. `/step` catches `RuntimeError` → 400 not 500 (client bug not server bug)
  6. Read-only endpoints (`/state`, `/grader`) don't auto-reset
  7. `/tasks` trims description to 120 chars (lightweight listing)
  8. No CORS middleware yet (defer to Phase 10 if needed)
  9. Hardcoded version "0.1.0" in two places (pragmatic duplication)
  10. `urllib.request` for smoke test (Windows-safe, no curl dependency)
- **Notes:** Endpoints: /, /health, /reset, /step, /state, /tasks, /grader. Port 7860 for HF Spaces compatibility.

### Phase 8 — Env Client + Training Skeleton
- **Status:** NOT YET STARTED
- **Assigned to:** Yashash (lead) with Gajanand reviewing Kaggle notebook
- **Expected time:** ~2-2.5 hours
- (To be filled in on completion)

---

## PHASES REMAINING

- [ ] Phase 8 — Env client + training skeleton
- [ ] Phase 9 — Baseline eval (heuristics + 3 LLMs + GPT-4o-mini)
- [ ] Phase 10 — HF Space deploy
- [ ] Phase 11 — Medium scenarios (M1/M2/M3)
- [ ] Phase 12 — Insurance video recording (2 videos: 60s + 2min)
- [ ] Phase 13 — Kaggle training runs (Stage 1 single account → Stage 2 parallelize)
- [ ] Phase 14 — Pitch + blog + video
- [ ] Phase 15 — Onsite Saturday-Sunday

---

## TRAINING RUNS

(Will be populated during Phase 13. Template for each run below.)

### Run 1 — Account [N] — [Person] — [Date Time]
- **Model:** Qwen 2.5 1.5B Instruct OR Qwen 2.5 Coder 1.5B Instruct
- **Reward variant:** binary / shaped_total / shaped_total+procedural
- **Scenario set:** deterministic / procedural
- **LoRA config:** r=16, alpha=32, targets=[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
- **GRPO config:** lr=5e-6, num_generations=4, max_completion_length=1536, gradient_accumulation=4, max_steps=100
- **Start time:**
- **End time:**
- **Steps completed:**
- **First batch mean reward:**
- **Final mean reward:**
- **Delta:**
- **Checkpoint HF Hub ID:**
- **Eval E1 score:**
- **Eval E2 score:**
- **Eval E3 score:**
- **Reward curve image:** `training_logs/run_N.png`
- **Errors / stops:**
- **Notes:**

---

## EVAL RUNS

(Will be populated during Phase 9. Template below.)

### Eval [baseline_name] — [Date]
- **Model:** e.g., naive_heuristic, policy_aware_heuristic, Qwen 2.5 7B, Llama 3.1 8B, GPT-4o-mini, Qwen 2.5 1.5B (trained)
- **Seeds:** 5 per task
- **E1_onboard_new_hire avg:**
- **E2_meeting_invite_blast avg:**
- **E3_customer_lookup avg:**
- **Full raw scores table:**

  | Task | Seed | Completion | DriftDetection | Adaptation | Efficiency | Shaped | Binary |
  |------|------|------------|----------------|------------|------------|--------|--------|
  | E1   | 0    |            |                |            |            |        |        |
  | E1   | 1    |            |                |            |            |        |        |
  | ...  |      |            |                |            |            |        |        |

- **Failure examples:**
  1. Action JSON + Response JSON
  2. ...
  3. ...

---

## DEPLOYMENTS

(Will be populated during Phase 10+.)

### Deploy v[N] — [Date]
- **Target:** HF Space / local / Kaggle
- **Commit:** 
- **URL:** 
- **Health check:** 
- **Config secrets set:** 
- **Notes:**

---

## DECISIONS LOG (strategic, append-only)

### Tuesday ~9 PM — Domain lock-in
- **Decision:** Schema Drift Adaptation Arena (SaaS Admin Workflows)
- **Rationale:** Direct Patronus sub-theme hit; Scaler secondary; no competitor finalist has it; authentic to Yashash's APEX OS/VisionX experience; cleanest before/after RL story

### Tuesday ~10 PM — Staged Kaggle approach (v2.3)
- **Decision:** Run ONE Kaggle account (Gajanand/Account 2, main config) first. Parallelize only if Stage 1 converges.
- **Rationale:** If reward function or parser has a bug, 3 simultaneous accounts waste 3× GPU quota. Debug once on one account. Main config = primary pitch claim, so prove that first.

### Tuesday ~10 PM — Qwen 2.5 Coder 1.5B ablation on Account 3 (v2.2)
- **Decision:** Account 3 uses Coder-1.5B instead of same model + curriculum only.
- **Rationale:** More informative ablation — tests whether code-pretraining helps schema-adaptation learning. Gives blog/pitch a secondary result regardless of outcome.

### Tuesday ~10 PM — Two videos in Phase 12 (v2.1)
- **Decision:** Produce 60-second core cut + 2-minute full version, not one 90-second video.
- **Rationale:** 60s fits pitch timing with hook+close room. 2min fits HF blog context depth. Recording once (Thursday), using twice.

### Tuesday ~10 PM — One-sentence pitch opener (v2.1)
- **Decision:** Open README/pitch/video with: *"SchemaShift teaches agents to recover when the tool schema changes under them."*
- **Rationale:** Gallery winners (Kube SRE Gym, GAIA) lead with one-sentence claims. Judges remember one clean idea, not four themes.

### Tuesday ~10 PM — Procedural drift scheduler on Account 3 (v2.1)
- **Decision:** Account 3 variant adds ~30 LOC scheduler that rotates drift steps and adds secondary drift after step 150.
- **Rationale:** Kube SRE Gym uses Claude-powered adversarial designer; we achieve similar curriculum story without API credit burn.

---

## REJECTED IDEAS (do not revive — posterity record)

### Validator / oversight agent — rejected Tuesday
- **Why rejected:** Would break 63 passing tests, adds scope at wrong phase, theme-benefit marginal (multi-agent claim already defensible via scripted tool servers)

### Uncertainty injection across failure types — rejected Tuesday  
- **Why rejected:** Dilutes pitch (becomes "uncertainty" not "schema drift"), risks GRPO convergence by introducing probabilistic reward, already solved by dense shaping + policy-aware heuristic baseline proving non-gameability

### Claude-powered adversarial drift designer — rejected Tuesday
- **Why rejected:** Time sink, API credit burn, provides marginal benefit over procedural scheduler

### Pivot to "decision-making under uncertainty" framing — rejected Tuesday
- **Why rejected:** Breaks Patronus sub-theme fit, muddies demo story, opens scope unnecessarily

### Qwen3 family migration (primary training) — rejected Tuesday
- **Why rejected:** Newer but less stable, prompt-behavior churn, migration risk for 48-hour build. Stick with Qwen 2.5 1.5B.

### 3B+ primary model — rejected Tuesday
- **Why rejected:** Slow iteration (2x fewer experiments per Kaggle hour), tight T4 memory, doesn't improve pitch. "1.5B beats GPT-4o-mini" is a stronger claim than "3B beats GPT-4o-mini".

---

## OPEN QUESTIONS / TODOS

- [ ] Phase 8: verify step_shaping survives HTTP serialization roundtrip
- [ ] Phase 9: do baseline LLMs hit at least 0.30 on drifted E1? (if too high, env is too easy)
- [ ] Phase 10: deploy to HF Space, verify remote /health works
- [ ] Phase 11: when adding M1/M2/M3, verify AdaptationRubric denominator still makes sense (flagged in Phase 5 judgment call #2)
- [ ] Phase 13 Stage 1: go/no-go decision at 20 steps — document the reward curve image
- [ ] Phase 14: draft strong-claim and softer-claim versions of pitch, pick based on training results
- [ ] Phase 15 onsite: bring a backup video file on USB in case HF Space is slow to load

---

## ONSITE LOGISTICS

**Event:** Saturday-Sunday April 25-26, 2026 · SST Campus Bangalore
**Travel:** (add details)
**Accommodation:** (add details)
**Equipment to bring:**
- 2 laptops (primary + backup)
- USB with video files + repo backup
- HDMI adapter for presentation
- Chargers + extension cord
- Both phones with hotspot capability (in case venue wifi fails)

---
