# SchemaShift Training Log

**Purpose:** Immutable record of every training run, every checkpoint, every evaluation. Append-only. Never edit existing entries. If a number turns out wrong later, add a correction note below it — never overwrite.

**Why this log exists:** By Sunday pitch you will quote numbers. "Our 1.5B model improved from 0.15 to 0.78 on drifted E1." "We trained for 100 steps over 2.5 hours." "Binary-only reward failed to converge; shaped reward converged in 60 steps." Every single one of these claims requires logged data. Memory will not survive five days of continuous building. Write it down now or rebuild it from bad memory on Sunday morning.

**Team:** Tripod (Yashash / Gajanand / Likith)
**Event:** Meta × HF × PyTorch OpenEnv Hackathon · Round 2 · April 25-26, 2026

---

## HOW TO USE THIS LOG

Four sections below, each append-only:

1. **PRE-TRAINING BASELINES** — untrained model scores (the "before" number for the pitch)
2. **TRAINING RUNS** — every run you attempt, successful or failed, with per-checkpoint data
3. **CHECKPOINT EVALS** — every eval you run against any checkpoint (linked by checkpoint ID)
4. **HEAD-TO-HEAD COMPARISONS** — the tables that will go in the blog / pitch

**Logging frequency during Phase 13:**
- Before any training run starts: full baseline eval against the untrained model
- Every 25 GRPO steps during training: checkpoint auto-saves, run eval against it
- After training completes: full eval + reward curve screenshot + judgment notes
- After each eval run on any baseline LLM: add to CHECKPOINT EVALS section
- End of Thursday: fill in first HEAD-TO-HEAD table with everything you have so far

---

## 1. PRE-TRAINING BASELINES

These are the "before" numbers. Every trained checkpoint gets compared against these.

### Baseline — UNTRAINED Qwen 2.5 1.5B Instruct (base model, no training)

- **Evaluated:** [DATE/TIME — fill in before Stage 1 training starts]
- **Seeds:** 0, 1, 2, 3, 4 (five per task)
- **Scenarios:** E1_onboard_new_hire, E2_meeting_invite_blast, E3_customer_lookup
- **Eval runner:** `python eval.py --baseline untrained_qwen_1.5b_instruct --seeds 0,1,2,3,4`

#### Per-seed raw scores

| Task | Seed | Completion | DriftDetection | Adaptation | Efficiency | Shaped | Binary |
|------|------|------------|----------------|------------|------------|--------|--------|
| E1   | 0    |            |                |            |            |        |        |
| E1   | 1    |            |                |            |            |        |        |
| E1   | 2    |            |                |            |            |        |        |
| E1   | 3    |            |                |            |            |        |        |
| E1   | 4    |            |                |            |            |        |        |
| E2   | 0    |            |                |            |            |        |        |
| E2   | 1    |            |                |            |            |        |        |
| E2   | 2    |            |                |            |            |        |        |
| E2   | 3    |            |                |            |            |        |        |
| E2   | 4    |            |                |            |            |        |        |
| E3   | 0    |            |                |            |            |        |        |
| E3   | 1    |            |                |            |            |        |        |
| E3   | 2    |            |                |            |            |        |        |
| E3   | 3    |            |                |            |            |        |        |
| E3   | 4    |            |                |            |            |        |        |

#### Aggregates

- **E1 mean shaped_total:**
- **E2 mean shaped_total:**
- **E3 mean shaped_total:**
- **Overall mean shaped_total:**
- **Overall binary rate (fraction of episodes that hit binary=1.0):**

#### Behavioral observations

What did the untrained model consistently do wrong? Quote 3 specific failure patterns:

1.
2.
3.

#### Failure examples (paste full Action + Response JSON for 3 bad actions)

Example 1 — [scenario, seed, step number]:
```json
{"action": ..., "response": ...}
```
Commentary: what was wrong

Example 2 — [scenario, seed, step number]:
```json
{"action": ..., "response": ...}
```
Commentary: what was wrong

Example 3 — [scenario, seed, step number]:
```json
{"action": ..., "response": ...}
```
Commentary: what was wrong

#### Runtime

- Total eval time: X minutes
- Kaggle GPU used: yes/no, which account
- Cost: X GPU-hours

---

### Baseline — UNTRAINED Qwen 2.5 Coder 1.5B Instruct (for v2.2 ablation)

(Same structure as above — fill in before Account 3 Stage 2 training starts)

- **Evaluated:** [DATE/TIME]
- **Seeds:** 0, 1, 2, 3, 4

(Fill the same per-seed table, aggregates, observations, failures, runtime.)

---

### Baseline — naive_heuristic

A dumb policy that always calls the first endpoint of the first tool. Proves the environment cannot be trivially solved.

**Partial data already collected:** See Section 3 / Eval 1 (3 seeds, local) and Eval 3 (3 seeds, production). Both runs returned 0.000 shaped / 0% binary across E1/E2/E3. For the final pitch table, need 5-seed production run in Phase 10+ sweep.

- **Evaluated:** [DATE/TIME]
- **Seeds:** 0, 1, 2, 3, 4

(Fill table + aggregates.)

---

### Baseline — policy_aware_heuristic

A smarter rule-based policy: on 4xx/5xx response, call inspect_schema next. Upper bound for non-RL solutions.

**Partial data already collected:** See Section 3 / Eval 2 (3 seeds, local) and Eval 3 (3 seeds, production). Both runs returned 0.348 overall mean_shaped / 66.67% binary. For the final pitch table, need 5-seed production run in Phase 10+ sweep.

- **Evaluated:** [DATE/TIME]
- **Seeds:** 0, 1, 2, 3, 4

(Fill table + aggregates.)

---

### Baseline — Qwen 2.5 7B Instruct (via HF router)

Larger-but-still-small LLM baseline. Should do better than 1.5B untrained but worse than trained 1.5B.

- **Evaluated:** [DATE/TIME]
- **Seeds:** 0, 1, 2, 3, 4
- **API provider:** HF Inference Router

(Fill table + aggregates.)

---

### Baseline — Llama 3.1 8B Instruct (via HF router)

Meta's comparable-size baseline.

- **Evaluated:** [DATE/TIME]
- **Seeds:** 0, 1, 2, 3, 4
- **API provider:** HF Inference Router

(Fill table + aggregates.)

---

### Baseline — GPT-4o-mini (the pitch target)

The frontier proxy we need to beat. This is THE number to beat.

- **Evaluated:** [DATE/TIME]
- **Seeds:** 0, 1, 2, 3, 4
- **API provider:** OpenAI
- **Cost:** $X.XX in API credits

(Fill table + aggregates.)

**PITCH-CRITICAL NOTE:** GPT-4o-mini's score on E1 is the headline comparison. If Qwen 1.5B + SchemaShift beats this number, we have the pitch. If not, we reframe to "trained small model beats untrained baseline" which is weaker.

---

## 2. TRAINING RUNS

Every training attempt, successful or failed. Number them sequentially across the team.

### Run 1 — Account 2 (Gajanand) — STAGE 1 MAIN — [Date Time start]

**This is Stage 1 — the pipeline-validation run. Critical go/no-go for Stage 2.**

#### Config snapshot (immutable — record exactly what was used)

- **Model:** Qwen 2.5 1.5B Instruct (`Qwen/Qwen2.5-1.5B-Instruct`)
- **Reward variant:** `return reward.shaped_total` (full: rubric + dense step shaping + gates)
- **Scenario set:** deterministic (original SCENARIOS from scenarios.py, no procedural scheduler)
- **LoRA config:**
  - r: 16
  - lora_alpha: 32
  - lora_dropout: 0.0
  - target_modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **GRPO config:**
  - learning_rate: 5e-6
  - num_generations: 4
  - max_completion_length: 1536
  - per_device_train_batch_size: 1
  - gradient_accumulation_steps: 4
  - logging_steps: 5
  - save_steps: 25
  - max_steps: 100
  - use_vllm: true, vllm_mode: colocate
- **Quantization:** 4-bit via Unsloth
- **Hardware:** Kaggle T4 x2 (or P100 fallback — note which)
- **SchemaShift env URL:** [HF Space URL from Phase 10]
- **Commit hash of repo at training time:**

#### Pre-training check (Cells 1-8 verification before Cell 9)

- [ ] Model loaded without OOM: yes/no, peak GPU memory: X GB
- [ ] env_client.health() returned True: yes/no
- [ ] Sample reward_fn call with dummy completion returned float: yes/no, value: X
- [ ] parse_completion_to_actions test on realistic Qwen output succeeded: yes/no

#### Per-batch reward log (first 20 steps, critical go/no-go)

| Step | Timestamp | Mean reward | Min | Max | Notes |
|------|-----------|-------------|-----|-----|-------|
| 1    |           |             |     |     |       |
| 2    |           |             |     |     |       |
| 3    |           |             |     |     |       |
| 4    |           |             |     |     |       |
| 5    |           |             |     |     |       |
| 10   |           |             |     |     |       |
| 15   |           |             |     |     |       |
| 20   |           |             |     |     |       |

**STAGE 1 GO/NO-GO DECISION at step 20:**
- Reward curve trending up: yes/no
- At least one reward > 0: yes/no
- Not stuck at single value: yes/no
- DECISION: PROCEED / STOP AND DEBUG
- Rationale:

#### Per-checkpoint evaluation (every 25 steps — MANDATORY)

For each checkpoint, run eval against E1/E2/E3 with 5 seeds, compare to untrained baseline.

##### Checkpoint @ step 25
- HF Hub ID:
- Eval timestamp:
- E1 mean shaped: X (vs untrained: Y, delta: +Z)
- E2 mean shaped:
- E3 mean shaped:
- Overall mean:
- Binary rate: X%
- **Key behavioral change observed vs untrained:**
- Screenshot: `training_logs/run_1_checkpoint_25_rewardcurve.png`

##### Checkpoint @ step 50
(Same structure)

##### Checkpoint @ step 75
(Same structure)

##### Checkpoint @ step 100 (FINAL)
- HF Hub ID:
- Eval timestamp:
- E1 mean shaped: X (vs untrained baseline: Y, delta: +Z, % improvement: W%)
- E2 mean shaped:
- E3 mean shaped:
- Overall mean:
- Binary rate: X%
- Compared to GPT-4o-mini: BEATS / TIES / LOSES
- Screenshot: `training_logs/run_1_final_rewardcurve.png`

#### Training run metadata

- **Start time:**
- **End time:**
- **Wall-clock duration:**
- **Total steps completed:** (if less than 100, note why)
- **First batch mean reward:**
- **Final batch mean reward:**
- **Delta first→final:**
- **% improvement first→final:**
- **Peak GPU memory during training:**
- **vLLM collocate stable throughout: yes/no**
- **HF checkpoint push count:** (should be 4: at 25, 50, 75, 100)

#### Errors encountered (full traceback for each)

If any:
1. Step X — [error type] — paste full traceback
   - Resolution: (what you did)
   - Re-ran from: (checkpoint or from scratch)

If none: "No errors."

#### Iteration note

Is this Run 1 for this config, or did you re-run after fixing something?
- If re-run: what config change was made vs prior attempt? What did the prior attempt fail on?

#### Judgment calls made during this run

(E.g., "reduced max_completion_length from 1536 to 1024 because of OOM at step 40" — document any deviations from the planned config)

#### Verdict

- [ ] Stage 1 passes — pipeline validated, proceed to Stage 2 with Accounts 1 and 3
- [ ] Stage 1 fails — specific failure mode:
- Next action:

---

### Run 2 — Account 1 (Yashash) — STAGE 2 BINARY CONTROL — [Date Time start]

(Only start after Run 1 Stage 1 passes. Same structure as Run 1, but with:)

- **Reward variant:** `return reward.binary` (sparse 0/1 signal only)
- **Model:** Qwen 2.5 1.5B Instruct (same as Run 1)
- **Hypothesis:** Will sparse binary reward converge without dense shaping?

(Fill same template: config, pre-training check, per-batch log, per-checkpoint eval, metadata, errors, verdict.)

---

### Run 3 — Account 3 (Likith) — STAGE 2 CODER ABLATION — [Date Time start]

(Same structure, but:)

- **Reward variant:** `return reward.shaped_total` (same as Run 1)
- **Model:** Qwen 2.5 **Coder** 1.5B Instruct (`Qwen/Qwen2.5-Coder-1.5B-Instruct`)
- **Scenario set:** procedural (build_procedural_scenarios enabled)
- **Hypothesis:** Does code-pretraining help schema-adaptation learning?

(Fill same template.)

---

### Run N — [placeholder for additional iterations]

When you re-run any config with changes, create a new Run entry. Never edit Run 1-3. New runs get new numbers. This preserves the experimental history.

---

## 3. CHECKPOINT EVALS (all eval runs, any model)

Every eval.py invocation gets one entry here, even if it was just to sanity-check something.

### Eval [N] — [model/checkpoint name] — [Date Time]

- **Target:** untrained base / checkpoint HF ID / baseline LLM name
- **Seeds:** 0, 1, 2, 3, 4
- **Scenarios:** E1, E2, E3
- **Full raw table:** (same format as baseline template above)
- **Aggregates:**
  - E1 mean shaped:
  - E2 mean shaped:
  - E3 mean shaped:
  - Overall:
  - Binary rate:
- **3 failure examples (Action+Response JSON):**
- **Cost (if API-based):** $X.XX
- **Commentary:**

---

### Eval 1 — naive_heuristic — Tuesday April 21, 2026 (late night, local server)

- **Target:** naive_heuristic (floor baseline)
- **Seeds:** 0, 1, 2
- **Scenarios:** E1, E2, E3
- **Aggregates:**
  - E1 mean shaped: 0.000 (binary_rate=0%)
  - E2 mean shaped: 0.000 (binary_rate=0%)
  - E3 mean shaped: 0.000 (binary_rate=0%)
  - **Overall mean_shaped: 0.000**
  - **Overall cumulative reward: 0.235**
  - **Overall binary rate: 0.00%**
- **3 failure examples:** Always calls first endpoint of first tool with empty params → 400 missing_required_params → completes after 3 steps without adaptation. Drift goes undetected in all 9 episodes.
- **Cost:** $0 (local env, no API)
- **Commentary:** Floor baseline. Proves env cannot be trivially solved. Zero shaped reward across all 9 episodes. Small non-zero cumulative (0.235) comes from efficiency rubric crediting unused step budget — not from any real task progress.

---

### Eval 2 — policy_aware_heuristic — Tuesday April 21, 2026 (late night, local server)

- **Target:** policy_aware_heuristic (ceiling for non-RL solutions)
- **Seeds:** 0, 1, 2
- **Scenarios:** E1, E2, E3
- **Aggregates:**
  - E1 mean shaped: 0.522 (binary_rate=100%)
  - E2 mean shaped: 0.000 (binary_rate=0% — unreachable by rule-based agent within step budget)
  - E3 mean shaped: 0.522 (binary_rate=100%)
  - **Overall mean_shaped: 0.348**
  - **Overall cumulative reward: 1.284**
  - **Overall binary rate: 66.67%**
- **Behavior observed:**
  - E1: heuristic executes mail send + calendar create in 3 steps, narrowly beating the drift fired at step 3. Completion GT met on both tools.
  - E2: drift fires step 1 (unavoidable), heuristic recovers via inspect→report→retry, sends 1 of 3 required emails. Remaining step budget insufficient for 2 more sends.
  - E3: search captures contact_id + company from response. Step 2 update_contact succeeds (drift transparent since field_rename doesn't affect email-less params). Summary mentions "Globex Industries" → complete_summary_mentions_company GT satisfied.
- **Cost:** $0 (local env, no API)
- **Commentary:** Discriminability vs naive: **+0.348 shaped, +66.67 percentage points binary.** Env has proven learnable-but-nontrivial gap. E2's unreachable-for-rules property is the RL-advantage test — a trained policy that plans ahead across multiple sends should exceed this ceiling. E2 is the scenario that will drive the pitch's "small model beats frontier" claim.

---

### Eval 3 — policy_aware_heuristic — production HF Space — Tuesday April 21, 2026 23:31 IST

- **Target:** policy_aware_heuristic against https://yashash045-schemashift.hf.space
- **Seeds:** 0, 1, 2
- **Aggregates:**
  - E1 mean shaped: 0.522 (binary_rate=100%)
  - E2 mean shaped: 0.000 (binary_rate=0% — ceiling: rule-based can't send 3 emails within 6-step budget after drift recovery)
  - E3 mean shaped: 0.522 (binary_rate=100%)
  - **Overall mean_shaped: 0.348**
  - **Overall cumulative reward: 1.284**
  - **Overall binary rate: 66.67%**
- **Cost:** $0 (HF Space CPU-basic free tier + local client)
- **Commentary:** Production matches local numbers to 3 decimal places. Env is deterministic and HF deploy preserves all scoring. **Critical pitch milestone: dense shaping (step_shaping=0.1000) survives production HTTP path verified independently via training/grpo_smoke.py.** Kaggle training in Phase 13 can now target this URL with confidence that shaped rewards propagate correctly end-to-end.

---

### Eval 4 — policy_aware_heuristic — M-tier local — Wednesday April 22, 2026 (early morning)

- **Target:** policy_aware_heuristic on M1/M2/M3, local server
- **Seeds:** 0, 1, 2
- **Scenarios:** M1_customer_escalation, M2_weekly_report, M3_event_cleanup
- **Aggregates:**
  - M1 mean shaped: 0.000 (binary=0%), cumul=2.510 — dense shaping captured partial progress
  - M2 mean shaped: 0.000 (binary=0%), cumul=3.013 — highest M-tier cumul
  - M3 mean shaped: 0.000 (binary=0%), cumul=0.441 — lowest M-tier cumul (multi-drift same-tool stress)
  - **Overall mean_shaped: 0.000**
  - **Overall cumulative reward: 1.988** (HIGHER than E-tier 1.284 — dense signal fires on multi-drift)
  - **Overall binary rate: 0.00%**
- **Behavior observed:** Heuristic's keyword-trigger task dispatcher misses M1's "schedule"/"check-in call" → fails to create calendar event. M2's multi-step aggregate task exceeds heuristic's one-step planning. M3's multi-event cleanup exceeds single-tool-single-endpoint dispatcher.
- **Commentary:** **M-tier establishes three-tier discriminability: naive=0, rule-based=0 (M-tier), rule-based=0.348 (E-tier). Any positive shaped score on M-tier requires real planning across 10-15 steps with 2 drifts per episode. This is the "must be RL" tier — no rule-based agent can solve it regardless of how many keywords are added. Pitch-critical: this is where "trained 1.5B beats GPT-4o-mini" gets proven.**
- **Heuristic brittleness note (pitch material):** Rule-based agent literally cannot parse natural language task variation. "Send welcome email" triggers mail. "Schedule a check-in call" does not. Frontier LLMs can parse both; our trained RL agent (Phase 13) should also parse both. Env rewards language generalization, not regex.

---

### Eval 5 — policy_aware_heuristic — M-tier production — Wednesday April 22, 2026 (early morning)

- **Target:** policy_aware_heuristic on M1/M2/M3, https://yashash045-schemashift.hf.space
- **Seeds:** 0, 1, 2
- **Aggregates:** Identical to Eval 4 to 3 decimals (2.510 / 3.013 / 0.441 cumul matches local)
- **Commentary:** Deterministic parity confirmed on M-tier. Production HF Space correctly serves 6 scenarios now. Kaggle training in Phase 13 can target this URL with full E+M scenario diversity.

---

### Eval 6 — naive_heuristic — 5 seeds × 6 scenarios — production — Wednesday April 22, 2026 (early morning)

- **Target:** naive_heuristic on E1/E2/E3/M1/M2/M3 against https://yashash045-schemashift.hf.space
- **Seeds:** 0, 1, 2, 3, 4 (30 episodes total)
- **Results JSON:** `eval_results/naive_5seed_full.json`
- **Per-seed table (all seeds identical — env is deterministic):**

| Task | Seed | Compl | Drift | Adapt | Effic | Shaped | Cumul | Binary |
|------|------|-------|-------|-------|-------|--------|-------|--------|
| E1_onboard_new_hire | 0 | 0.000 | 0.000 | 0.000 | 0.812 | 0.000 | 0.222 | 0 |
| E1_onboard_new_hire | 1 | 0.000 | 0.000 | 0.000 | 0.812 | 0.000 | 0.222 | 0 |
| E1_onboard_new_hire | 2 | 0.000 | 0.000 | 0.000 | 0.812 | 0.000 | 0.222 | 0 |
| E1_onboard_new_hire | 3 | 0.000 | 0.000 | 0.000 | 0.812 | 0.000 | 0.222 | 0 |
| E1_onboard_new_hire | 4 | 0.000 | 0.000 | 0.000 | 0.812 | 0.000 | 0.222 | 0 |
| E2_meeting_invite_blast | 0 | 0.000 | 0.000 | 0.000 | 0.750 | 0.000 | 0.213 | 0 |
| E2_meeting_invite_blast | 1 | 0.000 | 0.000 | 0.000 | 0.750 | 0.000 | 0.213 | 0 |
| E2_meeting_invite_blast | 2 | 0.000 | 0.000 | 0.000 | 0.750 | 0.000 | 0.213 | 0 |
| E2_meeting_invite_blast | 3 | 0.000 | 0.000 | 0.000 | 0.750 | 0.000 | 0.213 | 0 |
| E2_meeting_invite_blast | 4 | 0.000 | 0.000 | 0.000 | 0.750 | 0.000 | 0.213 | 0 |
| E3_customer_lookup | 0 | 0.000 | 0.000 | 0.000 | 0.812 | 0.000 | 0.272 | 0 |
| E3_customer_lookup | 1 | 0.000 | 0.000 | 0.000 | 0.812 | 0.000 | 0.272 | 0 |
| E3_customer_lookup | 2 | 0.000 | 0.000 | 0.000 | 0.812 | 0.000 | 0.272 | 0 |
| E3_customer_lookup | 3 | 0.000 | 0.000 | 0.000 | 0.812 | 0.000 | 0.272 | 0 |
| E3_customer_lookup | 4 | 0.000 | 0.000 | 0.000 | 0.812 | 0.000 | 0.272 | 0 |
| M1_customer_escalation | 0 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.231 | 0 |
| M1_customer_escalation | 1 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.231 | 0 |
| M1_customer_escalation | 2 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.231 | 0 |
| M1_customer_escalation | 3 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.231 | 0 |
| M1_customer_escalation | 4 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.231 | 0 |
| M2_weekly_report | 0 | 0.000 | 0.000 | 0.000 | 0.850 | 0.000 | 0.227 | 0 |
| M2_weekly_report | 1 | 0.000 | 0.000 | 0.000 | 0.850 | 0.000 | 0.227 | 0 |
| M2_weekly_report | 2 | 0.000 | 0.000 | 0.000 | 0.850 | 0.000 | 0.227 | 0 |
| M2_weekly_report | 3 | 0.000 | 0.000 | 0.000 | 0.850 | 0.000 | 0.227 | 0 |
| M2_weekly_report | 4 | 0.000 | 0.000 | 0.000 | 0.850 | 0.000 | 0.227 | 0 |
| M3_event_cleanup | 0 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.231 | 0 |
| M3_event_cleanup | 1 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.231 | 0 |
| M3_event_cleanup | 2 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.231 | 0 |
| M3_event_cleanup | 3 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.231 | 0 |
| M3_event_cleanup | 4 | 0.000 | 0.000 | 0.000 | 0.875 | 0.000 | 0.231 | 0 |

- **Aggregates:**
  - E1 mean shaped: 0.000 (binary=0%), cumul=0.222
  - E2 mean shaped: 0.000 (binary=0%), cumul=0.213
  - E3 mean shaped: 0.000 (binary=0%), cumul=0.272
  - M1 mean shaped: 0.000 (binary=0%), cumul=0.231
  - M2 mean shaped: 0.000 (binary=0%), cumul=0.227
  - M3 mean shaped: 0.000 (binary=0%), cumul=0.231
  - **Overall mean_shaped: 0.000**
  - **Overall cumulative reward: 0.233**
  - **Overall binary rate: 0.00%**
- **Cost:** $0 (prod HF Space CPU-basic + local client, ~1.3s per episode)
- **Commentary:** Definitive floor baseline with 5 seeds × 6 scenarios on production. Zero shaped reward and zero binary on every single episode across both tiers. Low non-zero cumulative (0.21–0.27 per scenario) is pure efficiency-rubric credit from unused step budget. This is the "before" number the pitch quotes.

---

### Eval 7 — policy_aware_heuristic — 5 seeds × 6 scenarios — production — Wednesday April 22, 2026 (early morning)

- **Target:** policy_aware_heuristic on E1/E2/E3/M1/M2/M3 against https://yashash045-schemashift.hf.space
- **Seeds:** 0, 1, 2, 3, 4 (30 episodes total)
- **Results JSON:** `eval_results/policy_aware_5seed_full.json`
- **Per-seed table (all seeds identical — env is deterministic):**

| Task | Seed | Compl | Drift | Adapt | Effic | Shaped | Cumul | Binary |
|------|------|-------|-------|-------|-------|--------|-------|--------|
| E1_onboard_new_hire | 0 | 1.000 | 0.000 | 0.000 | 0.812 | 0.522 | 1.434 | 1 |
| E1_onboard_new_hire | 1 | 1.000 | 0.000 | 0.000 | 0.812 | 0.522 | 1.434 | 1 |
| E1_onboard_new_hire | 2 | 1.000 | 0.000 | 0.000 | 0.812 | 0.522 | 1.434 | 1 |
| E1_onboard_new_hire | 3 | 1.000 | 0.000 | 0.000 | 0.812 | 0.522 | 1.434 | 1 |
| E1_onboard_new_hire | 4 | 1.000 | 0.000 | 0.000 | 0.812 | 0.522 | 1.434 | 1 |
| E2_meeting_invite_blast | 0 | 0.000 | 1.000 | 1.000 | 0.583 | 0.000 | 1.425 | 0 |
| E2_meeting_invite_blast | 1 | 0.000 | 1.000 | 1.000 | 0.583 | 0.000 | 1.425 | 0 |
| E2_meeting_invite_blast | 2 | 0.000 | 1.000 | 1.000 | 0.583 | 0.000 | 1.425 | 0 |
| E2_meeting_invite_blast | 3 | 0.000 | 1.000 | 1.000 | 0.583 | 0.000 | 1.425 | 0 |
| E2_meeting_invite_blast | 4 | 0.000 | 1.000 | 1.000 | 0.583 | 0.000 | 1.425 | 0 |
| E3_customer_lookup | 0 | 1.000 | 0.000 | 0.000 | 0.812 | 0.522 | 0.994 | 1 |
| E3_customer_lookup | 1 | 1.000 | 0.000 | 0.000 | 0.812 | 0.522 | 0.994 | 1 |
| E3_customer_lookup | 2 | 1.000 | 0.000 | 0.000 | 0.812 | 0.522 | 0.994 | 1 |
| E3_customer_lookup | 3 | 1.000 | 0.000 | 0.000 | 0.812 | 0.522 | 0.994 | 1 |
| E3_customer_lookup | 4 | 1.000 | 0.000 | 0.000 | 0.812 | 0.522 | 0.994 | 1 |
| M1_customer_escalation | 0 | 0.500 | 0.500 | 0.000 | 0.708 | 0.000 | 2.510 | 0 |
| M1_customer_escalation | 1 | 0.500 | 0.500 | 0.000 | 0.708 | 0.000 | 2.510 | 0 |
| M1_customer_escalation | 2 | 0.500 | 0.500 | 0.000 | 0.708 | 0.000 | 2.510 | 0 |
| M1_customer_escalation | 3 | 0.500 | 0.500 | 0.000 | 0.708 | 0.000 | 2.510 | 0 |
| M1_customer_escalation | 4 | 0.500 | 0.500 | 0.000 | 0.708 | 0.000 | 2.510 | 0 |
| M2_weekly_report | 0 | 0.250 | 0.000 | 1.000 | 0.500 | 0.000 | 3.013 | 0 |
| M2_weekly_report | 1 | 0.250 | 0.000 | 1.000 | 0.500 | 0.000 | 3.013 | 0 |
| M2_weekly_report | 2 | 0.250 | 0.000 | 1.000 | 0.500 | 0.000 | 3.013 | 0 |
| M2_weekly_report | 3 | 0.250 | 0.000 | 1.000 | 0.500 | 0.000 | 3.013 | 0 |
| M2_weekly_report | 4 | 0.250 | 0.000 | 1.000 | 0.500 | 0.000 | 3.013 | 0 |
| M3_event_cleanup | 0 | 0.200 | 0.000 | 0.000 | 0.875 | 0.000 | 0.441 | 0 |
| M3_event_cleanup | 1 | 0.200 | 0.000 | 0.000 | 0.875 | 0.000 | 0.441 | 0 |
| M3_event_cleanup | 2 | 0.200 | 0.000 | 0.000 | 0.875 | 0.000 | 0.441 | 0 |
| M3_event_cleanup | 3 | 0.200 | 0.000 | 0.000 | 0.875 | 0.000 | 0.441 | 0 |
| M3_event_cleanup | 4 | 0.200 | 0.000 | 0.000 | 0.875 | 0.000 | 0.441 | 0 |

- **Aggregates:**
  - E1 mean shaped: 0.522 (binary=100%), cumul=1.434
  - E2 mean shaped: 0.000 (binary=0%), cumul=1.425 — unreachable for rule-based (intentional ceiling)
  - E3 mean shaped: 0.522 (binary=100%), cumul=0.994
  - M1 mean shaped: 0.000 (binary=0%), cumul=2.510
  - M2 mean shaped: 0.000 (binary=0%), cumul=3.013
  - M3 mean shaped: 0.000 (binary=0%), cumul=0.441
  - **Overall mean_shaped: 0.174** (only E1 and E3 binary=1 passed; 4 of 6 scenarios gate-zeroed)
  - **Overall cumulative reward: 1.636** (7× naive's 0.233 — dense shaping fires consistently)
  - **Overall binary rate: 33.33%** (10 of 30 episodes binary=1, all E1 and E3)
- **Cost:** $0 (prod HF Space CPU-basic + local client, ~1.3s per episode)
- **Commentary:** Rule-based ceiling with 5-seed stability. Discriminability vs naive: **+0.174 shaped, +1.403 cumul, +33.33pp binary**. The 33.33% binary rate (E1 + E3 only) is the rule-based ceiling that trained RL must exceed. E2's binary=0 is the "requires multi-step planning" signal. M1/M2/M3 binary=0 is the "requires language understanding beyond keyword matching" signal. Together these 4 gate-zeroed scenarios are where the pitch lives: a trained 1.5B model that handles any of them proves RL learned something beyond regex.
- **Pitch claim positioning:** "Our 1.5B model improved from X.XX to Y.YY shaped reward on E2/M-tier scenarios where rule-based agents score 0.000 regardless of effort" — X will be untrained Qwen 1.5B shaped (TBD after Run 1), Y will be trained checkpoint shaped.

---

## 4. HEAD-TO-HEAD COMPARISONS (for blog, pitch, judges)

These tables get built progressively as runs complete. They ARE your pitch data.

### Table A: Qwen 2.5 1.5B — untrained vs SchemaShift-trained (primary pitch number)

| Model state | E1 shaped | E2 shaped | E3 shaped | Overall | Binary rate |
|-------------|-----------|-----------|-----------|---------|-------------|
| Untrained base |       |           |           |         |             |
| SchemaShift-trained |  |           |           |         |             |
| **Improvement %** |    |           |           |         |             |

### Table B: Frontier models fail, trained small model succeeds (the headline claim)

| Model | Size | E1 | E2 | E3 | Avg |
|-------|------|-----|-----|-----|-----|
| naive_heuristic | — |  |  |  |  |
| policy_aware_heuristic | — |  |  |  |  |
| Qwen 2.5 7B Instruct | 7B |  |  |  |  |
| Llama 3.1 8B Instruct | 8B |  |  |  |  |
| GPT-4o-mini | ~8B* |  |  |  |  |
| **Qwen 2.5 1.5B + SchemaShift (ours)** | **1.5B** |  |  |  |  |

*GPT-4o-mini parameter count estimated — OpenAI has not disclosed publicly.

### Table C: Reward function ablation (binary vs shaped)

| Run | Reward | Model | Final E1 | Convergence step | Verdict |
|-----|--------|-------|----------|------------------|---------|
| Run 1 (Account 2) | shaped_total | Qwen 1.5B |  |  |  |
| Run 2 (Account 1) | binary only | Qwen 1.5B |  |  |  |

Hypothesis being tested: dense shaping is necessary for GRPO convergence in sparse-reward tool-use domains.

### Table D: Model ablation (Instruct vs Coder pretraining)

| Run | Model | Final E1 | Final E2 | Final E3 | Convergence step |
|-----|-------|----------|----------|----------|------------------|
| Run 1 (Account 2) | Qwen 2.5 1.5B Instruct |  |  |  |  |
| Run 3 (Account 3) | Qwen 2.5 Coder 1.5B Instruct |  |  |  |  |

Hypothesis: code-pretraining transfers to schema-adaptation.

### Table E: Training efficiency

| Run | Steps | Wall-clock | GPU-hours | Cost (USD) | Peak reward |
|-----|-------|------------|-----------|------------|-------------|
| Run 1 |  |  |  | (Kaggle free) |  |
| Run 2 |  |  |  | (Kaggle free) |  |
| Run 3 |  |  |  | (Kaggle free) |  |

If Saturday onsite uses H100 credits, add row for that run and log cost.

---

## 5. ITERATION TIMELINE (what we tried, in order)

A chronological summary of every experiment. One line per iteration. This is the "we iterated" proof.

| # | Date/time | Run | Config change from previous | Outcome |
|---|-----------|-----|------------------------------|---------|
| 1 | 2026-04-22 early AM | M-tier validation | Added M1/M2/M3 scenarios + AdaptationRubric multi-drift test | M-tier discriminability confirmed (rule-based=0 on M-tier, >0 on E-tier). Dense shaping fires correctly. No rubric change needed. |
| 2 |  | Run 1 (Stage 1) | First attempt — Qwen 1.5B + shaped | |
| 3 |  |  |  |  |

(Add rows as you go. Include failed runs — failures are data too.)

---

## 6. LESSONS LEARNED (captured while fresh)

Every time training surprises you — good or bad — note it here immediately. Do not defer. You will forget.

### Lesson [N] — [short title] — [Date]

- Context:
- What happened:
- Why it happened:
- How to avoid / exploit it next time:

---

## 7. OPEN QUESTIONS FOR ONSITE

Things you couldn't resolve during dev that the onsite H100 run should answer.

1.
2.
3.

---

**This log is part of your submission evidence.** Judges may ask: "How many configurations did you try?" "What was your eval methodology?" "Show the reward improvement." Every good answer points back to a filled-in entry in this file.
