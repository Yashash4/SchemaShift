# Stage 1 Kaggle Runbook — GRPO Training on Qwen 2.5 1.5B

**Who:** Yashash (Kaggle Account 1) — STAGE 1 MAIN run (shaped reward)
**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (4-bit, LoRA r=16)
**Reward:** `reward.shaped_total` (rubric + dense step shaping)
**Dataset:** all 6 scenarios (E1/E2/E3/M1/M2/M3), deterministic drifts
**Target:** 100 GRPO steps, checkpoints at 25 / 50 / 75 / 100
**Env:** `https://yashash045-schemashift.hf.space`
**Expected wall-clock:** ~4 hours on Kaggle T4 ×2 (no vLLM — see "Kaggle dependency compatibility" below for why)

### Kaggle dependency compatibility (why no vLLM)

Kaggle's 2026 base image ships `torch 2.10+cu128` + `transformers 4.57` + `torchao 0.17` pre-installed. We do NOT pin torch:
- Pinning torch 2.4 breaks Kaggle's pre-installed `transformers` and `torchao`
- Pinning torch 2.5.1 breaks `vLLM 0.6.3` (which itself pins torch 2.4)
- The only clean path is Kaggle-native torch + latest Unsloth + no vLLM

TRL range `>=0.18.2,<=0.24.0,!=0.19.0` is what Unsloth 2026.x is compatible with (pip resolves to `0.24.0`). Training runs ~4 hours instead of ~2.5 hours without vLLM's fast inference, but correctness is unaffected — only generation throughput drops. **Judges don't ask about inference speed.**

---

## Prerequisites

- [ ] Yashash's Kaggle account logged in
- [ ] Hugging Face write token generated (from https://huggingface.co/settings/tokens)
- [ ] Read the current BUILD_LOG.md + TRAINING_LOG.md Section 1 for baseline context
- [ ] Confirmed repo is synced: `git log --oneline -1 origin/main` matches local

## Step 1 — Open Kaggle and import notebook

1. Go to https://www.kaggle.com/code
2. **New Notebook** → **Import Notebook** → **From GitHub URL**:
   ```
   https://github.com/Yashash4/SchemaShift/blob/main/training/grpo_kaggle.ipynb
   ```
3. Kaggle will clone the repo + open `grpo_kaggle.ipynb`

## Step 2 — Configure session

In the right sidebar:

1. **Accelerator** → `GPU T4 x2` (P100 is a valid fallback if T4 isn't available)
2. **Internet** → `ON` (required — env client hits HF Space over HTTPS)
3. **Persistence** → `Files only` or `Variables and files` (so checkpoints survive session timeout)

## Step 3 — Add three Kaggle Secrets

Right sidebar → **Add-ons** → **Secrets** → add each:

| Name | Value |
|---|---|
| `HF_TOKEN` | Your write-scoped HF token (starts `hf_...`) |
| `SCHEMASHIFT_URL` | `https://yashash045-schemashift.hf.space` |
| `HF_USERNAME` | `yashash045` |

**Attach all three to the notebook** (gear icon on each secret → toggle the notebook on).

## Step 4 — Run Cells 1-8 in order

Execute each cell with Shift+Enter. Wait for "✓" before advancing.

| Cell | Purpose | Expected | Red flag |
|---|---|---|---|
| 1 | `pip install unsloth trl httpx ...` | `torch: 2.10.x+cu128`, `unsloth: 2026.x`, `trl: 0.24.0`, "Core deps verified" | any import error or wrong torch version — stop |
| 2 | `git clone SchemaShift && pip install -e .` | "Successfully installed schemashift-0.1.0" | clone fails (check internet) |
| 3 | Env health check (import client, call `client.health()`) | prints `True` | `False` — URL wrong or Space down |
| 4 | Load Qwen 2.5 1.5B 4-bit + LoRA | "Trainable params: ~8.9M" | OOM error — use T4 x2 not x1 |
| 5 | Parser + reward_fn + env client | no output | import error — check Cell 2 ran |
| 6 | Procedural scheduler (SKIP for Stage 1) | skip this cell entirely | — |
| 7 | `build_prompts_dataset(SCENARIOS)` | prints "6 scenarios loaded" | <6 — scenarios.py out of sync |
| 8 | GRPOConfig + `Dataset.from_list(...)` | no output | hub_model_id wrong → checkpoints land in wrong namespace |

**Smoke test before Cell 9:** paste this into a fresh scratch cell and run it:

```python
# Quick reward_fn sanity check (one call, one episode)
test_rewards = reward_fn(
    prompts=["test"],
    completions=['{"type": "complete_task", "complete": {"summary": "noop"}}'],
    task_id="E1_onboard_new_hire",
)
print(f"reward_fn works: {test_rewards}")
# Should print something like: reward_fn works: [0.121875]  (the naive-completion efficiency credit)
# If it prints [0.0] exactly or crashes → parser or env client is broken; STOP.
```

## Step 5 — Run Cell 9 (training)

Cell 9 is just `trainer.train()`. Click it. Watch the stdout logs.

### Go/no-go watch window — first 20 steps

- [ ] **Steps 1-5:** any reward > 0 in any batch? → **GO**; if all zero → **STOP**, parser broken
- [ ] **Steps 5-15:** rewards show variance (not constant)? → **GO**; if stuck → **STOP**, policy collapsed
- [ ] **Step 20:** mean reward trending up (or at minimum holding, not crashing)? → **GO** to completion
- [ ] **Step 20 decision:** if any red above → abort session, don't burn another 2 hours

If **GO**, walk away. Training takes ~4 hours on T4 x2 (no vLLM — see dependency notes at the top of this runbook).

### During training — expected log pattern

```
[step 5]  mean_reward=0.08 loss=1.23
[step 10] mean_reward=0.12 loss=1.18
[step 15] mean_reward=0.15 loss=1.14
[step 20] mean_reward=0.19 loss=1.10   ← decision point: trending up = good
[step 25] checkpoint pushed to HF Hub   ← first checkpoint lands
...
[step 100] checkpoint pushed to HF Hub  ← final
```

## Step 6 — Checkpoint verification

After training completes, confirm 4 checkpoints on HF Hub:

```
https://huggingface.co/yashash045/schemashift-qwen15b-kaggle/tree/main
```

Should contain folders / revisions tagged with `step_25`, `step_50`, `step_75`, `step_100`.

## Step 7 — Save the reward curve image

- Scroll back through Kaggle stdout → select the log block from step 1 through step 100
- Screenshot it (use Kaggle's built-in screenshot, or OS snip)
- Save locally as `training_logs/run_1_yashash_stage1_shaped.png`
- `git add training_logs/run_1_yashash_stage1_shaped.png && git commit -m "Stage 1 reward curve"`

## Step 8 — Eval each checkpoint

From your local terminal (not Kaggle):

```bash
cd E:\sst\Final\schemashift
for CKPT in 25 50 75 100; do
  SCHEMASHIFT_URL=https://yashash045-schemashift.hf.space \
    python eval.py \
      --baseline "checkpoint:yashash045/schemashift-qwen15b-kaggle:step_$CKPT" \
      --tasks E1_onboard_new_hire,E2_meeting_invite_blast,E3_customer_lookup,M1_customer_escalation,M2_weekly_report,M3_event_cleanup \
      --seeds 0,1,2,3,4
done
```

**Note:** The `checkpoint:` provider in `eval.py` is currently a stub (`NotImplementedError`). Before running the loop above, wire it up to load the model locally (or via HF router inference endpoint) — that's a ~20-line addition to `LLMAgent._setup`/`_call`. Coordinate with Claude Code to implement before this step.

## Step 9 — Log to TRAINING_LOG.md Section 2 Run 1

Fill in the following fields in TRAINING_LOG.md Run 1 entry (Section 2):

- [ ] **Start time** (UTC, from Cell 9 stdout first log line)
- [ ] **End time** (when Cell 9 finished)
- [ ] **Commit hash at training time** (run `git rev-parse HEAD` from the cloned repo in the Kaggle session)
- [ ] **First batch mean reward** (from stdout step 1 log)
- [ ] **Final mean reward** (last 10 steps average)
- [ ] **Delta** (final - first)
- [ ] **Per-batch reward log** (steps 1-20 copy-pasted from stdout)
- [ ] **Checkpoint HF Hub IDs** (4 of them)
- [ ] **Checkpoint eval results** (per-checkpoint aggregates from Step 8)
- [ ] **Reward curve image path** (training_logs/run_1_yashash_stage1_shaped.png)
- [ ] **Errors / stops** (any crashes, OOMs, go/no-go abort triggers)
- [ ] **Notes** (anything surprising)
- [ ] **Verdict** (Tier 1 / Tier 2 / Tier 3 achievement against the baseline comparison — see TRAINING_LOG Section 1 gpt-oss entry for thresholds)

## Step 10 — Commit and push

```bash
git add TRAINING_LOG.md training_logs/run_1_yashash_stage1_shaped.png
git commit -m "Phase 13 Stage 1 complete: Run 1 results + reward curve + checkpoint evals"
git push origin main
git push space main
```

## Failure modes and recovery

### Rewards flat at 0.000 for 20+ steps

Likely cause: parser returning `complete_task` for every completion (model output not parseable as Action JSON).

Fix: in a fresh cell, dump a raw completion:
```python
sample = trainer.tokenizer.decode(
    trainer.model.generate(
        trainer.tokenizer.encode("<test prompt>", return_tensors="pt").to("cuda"),
        max_new_tokens=200,
    )[0]
)
print(sample)
# Look at what the model is actually producing. If it's not JSON, the prompt template needs work.
```

### Kaggle session killed / 12-hour timeout

Checkpoints at steps 25/50/75/100 are already pushed to HF Hub mid-training. Resume from the latest checkpoint by:
1. Note which step was last pushed (check HF Hub)
2. In a new Kaggle session, modify Cell 4 to load from that checkpoint
3. Modify Cell 8's `max_steps` to the remaining count
4. Continue

### OOM on T4

- Confirm **T4 x2** not **T4 x1**
- Reduce `max_completion_length` from 1536 → 1024 in Cell 7 (GRPOConfig)
- Reduce `num_generations` from 4 → 2 (halves memory but slows learning)

### Env Space slow / timeout

HF Space free tier cold-starts after ~10 min idle. The first `/reset` after a pause can take 30-60s. If you see repeated timeouts mid-training:
- Pause: send a warm-up `curl` to `/health` from a scratch cell every ~5 min
- Worst case: restart the Space from HF UI, wait for it to stabilize, resume

---

**Stage 1 is done when:** Section 2 Run 1 has all fields populated, reward curve image committed, and verdict written against the Tier 1/2/3 thresholds in Section 1. Handoff to Gajanand / Likith for parallel Stage 2 runs happens after Yashash's verdict is logged.
