---
title: SchemaShift
emoji: 🔄
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - reinforcement-learning
  - openenv
  - tool-use
  - schema-drift
  - patronus
  - scaler
---

# SchemaShift — Adaptive Tool Use Under Schema Drift

> **SchemaShift teaches agents to recover when the tool schema changes under them.**

An OpenEnv-compliant RL environment where workflow agents must complete multi-step SaaS admin tasks across Mail, Calendar, and CRM tools — while those tool APIs drift mid-episode. Trains adaptive tool use, a meta-skill that frontier LLMs lack because they're trained on static documentation.

**Team Tripod:** Yashash Sheshagiri (lead), Gajanand V Dhayagode, Likith B S
**Event:** Meta × Hugging Face × PyTorch OpenEnv Hackathon 2026 · Round 2 · Bangalore
**Themes hit:** Multi-Agent · Long-Horizon · World Modeling
**Sub-themes:** Patronus (Schema Drift — direct hit) · Scaler (Multi-App Enterprise)

## What it does

A trained agent handles what frontier LLMs silently fail at: when Gmail renames a field, when Stripe deprecates an endpoint, when Calendar restructures a response. Our env injects these drifts mid-episode and rewards agents that detect-inspect-adapt instead of retrying blindly.

## The claim we're testing

A Qwen 2.5 1.5B model trained with GRPO on SchemaShift beats GPT-4o-mini on drifted tasks. A small cheap model that learned skepticism beats a large expensive model that memorized docs.

## Status

Live environment with 3 scenarios (E1 onboard new hire, E2 meeting invite blast, E3 customer lookup), 3 tools (Mail 3 endpoints, Calendar 4 endpoints, CRM 4 endpoints), 7 drift types, composable rubric grader with dense step shaping, 73 tests passing. Discriminability gap verified (policy-aware heuristic 0.348 shaped vs naive 0.000). See BUILD_LOG.md for phase history and TRAINING_LOG.md for eval/training data.

## Endpoints

- `GET /` — metadata
- `GET /health` — health check
- `POST /reset` — start new episode, body: `{"task_id": "E1_onboard_new_hire", "seed": 0}`
- `POST /step` — submit action, body: `{"action": {...}, "tokens_used": 0}`
- `GET /state` — debug: current episode state
- `GET /tasks` — list available scenarios
- `GET /grader` — current grader breakdown

## Repo

https://github.com/Yashash4/SchemaShift

## License

MIT
