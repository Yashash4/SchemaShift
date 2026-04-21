# SchemaShift Deploy Guide

How to deploy the env to Hugging Face Spaces.

## Prerequisites

- HuggingFace account with access to create Spaces
- `huggingface_hub` CLI installed locally: `pip install -U huggingface_hub`
- Logged in: `huggingface-cli login` (use a write-scoped token)

## Step 1: Create the Space

Option A — via CLI:
```bash
huggingface-cli repo create schemashift --type space --space_sdk docker
```

Option B — via web:
1. Go to https://huggingface.co/new-space
2. Name: `schemashift`
3. SDK: **Docker**
4. Hardware: **CPU basic** (free tier is sufficient for env serving — no training happens here)
5. Visibility: **Public**

## Step 2: Push code to the Space

```bash
cd E:\sst\Final\schemashift
git remote add space https://huggingface.co/spaces/<YOUR_USERNAME>/schemashift
git push space main
```

First push takes 3-5 minutes to build the Docker image. Watch the build logs on the Space page.

## Step 3: Verify deployment

Once the build shows "Running", test the endpoints:

```bash
# Replace with your actual Space URL
export SS_URL=https://<YOUR_USERNAME>-schemashift.hf.space

curl $SS_URL/health
# Expected: {"status":"ok","version":"0.1.0"}

curl $SS_URL/tasks
# Expected: {"tasks":[...], "count":3}

curl -X POST $SS_URL/reset -H "Content-Type: application/json" -d '{"task_id":"E1_onboard_new_hire"}'
# Expected: JSON with task_id, step:0, tool_schemas, etc.
```

## Step 4: Run the production smoke test

```bash
SCHEMASHIFT_URL=$SS_URL python training/grpo_smoke.py
```

Must see: `Step 4 (inspect after failure): step_shaping=0.1000` — confirms dense shaping survives production deploy.

## Step 5: Run baseline eval against deployed env

Heuristics (free, no API keys):

```bash
SCHEMASHIFT_URL=$SS_URL python eval.py --baseline naive_heuristic --seeds 0,1,2,3,4
SCHEMASHIFT_URL=$SS_URL python eval.py --baseline policy_aware_heuristic --seeds 0,1,2,3,4
```

LLMs (requires API keys):

```bash
# Qwen 7B via HF router
export HF_TOKEN=hf_xxx
SCHEMASHIFT_URL=$SS_URL python eval.py --baseline hf:Qwen/Qwen2.5-7B-Instruct --seeds 0,1,2,3,4

# Llama 3.1 8B via HF router
SCHEMASHIFT_URL=$SS_URL python eval.py --baseline hf:meta-llama/Meta-Llama-3.1-8B-Instruct --seeds 0,1,2,3,4

# GPT-4o-mini via OpenAI
export OPENAI_API_KEY=sk-xxx
SCHEMASHIFT_URL=$SS_URL python eval.py --baseline openai:gpt-4o-mini --seeds 0,1,2,3,4
```

## Step 6: Run the deploy smoke test suite

```bash
SCHEMASHIFT_DEPLOY_URL=$SS_URL pytest tests/test_deploy_smoke.py -v
```

All 4 tests should pass. The most critical one: `test_deployed_step_shaping_fires` asserts the +0.10 dense-shaping reward survives through the production HTTP roundtrip.

## Step 7: Log results to TRAINING_LOG.md

Each eval run populates Section 1 (pre-training baselines) of TRAINING_LOG.md. See that file's template.

## Updating the Space later

Any changes to main push to both remotes:
```bash
git push origin main
git push space main
```

Or configure a single push that goes to both:
```bash
git remote set-url --add --push origin https://huggingface.co/spaces/<YOUR_USERNAME>/schemashift
```

## Troubleshooting

**Build fails with "pyproject.toml parse error":** check py-modules + packages config matches Phase 7 (`py-modules = ["models", "drift", ...]`, `packages = ["tools", "server", "training"]`).

**Server starts but /reset returns 500:** SCENARIOS dict import failed — check `scenarios.py` is at repo root.

**step_shaping returns 0.0 in smoke test:** RewardBreakdown serialization is broken — check `/step` JSON response has `"step_shaping"` field. Run `curl -X POST $SS_URL/step -H "Content-Type: application/json" -d '{"action":{"type":"inspect_schema","inspect":{"tool":"mail"}},"tokens_used":0}'` against a freshly reset episode.

**CORS errors from browser:** add `fastapi.middleware.cors.CORSMiddleware` to `server/app.py`. Not needed for Python clients or curl.

**Build succeeds but Space shows "Runtime error":** check Space logs — usually a missing dep. Verify `requirements.txt` includes everything you use.

**HF Space cold start is slow:** first request after idle can take 30-60 seconds. Subsequent requests are fast. If you hit a timeout on first call, retry.
