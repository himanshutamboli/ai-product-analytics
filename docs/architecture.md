# Architecture — ai-product-analytics

The **PM / growth-analytics** view of a (fictional) live GenAI product, `Lumen` — the
product-side counterpart to [`llm-observatory`](https://github.com/himanshutamboli/llm-observatory)'s
engineer-side observability. This document covers the data model, the metric definitions, and
the design decisions; the [README](../README.md) is the quickstart.

## Data flow

```
 data.py ──► generate(seed) ──► Dataset(users, sessions)   [polars, deterministic]
                                      │
                       metrics/ ──────┤  pure functions: (filtered frames) → chartable frames
                                      ▼
                       app.py ──► Streamlit: KPI header + 5 tabbed sections + filters (Plotly)
```

Two tables are the single source of truth; every metric derives from them.

| Table | Grain | Key columns |
|---|---|---|
| `users` | one row per signup | `user_id`, `signup_date`, `plan`, `channel`, `region` |
| `sessions` | one row per usage session | `user_id`, `date`, `feature`, `model`, `tokens`, `cost_usd`, `quality`, `feedback`, `refused`, `resolved` |

## The seeded narrative

The generator is deterministic (seeded `random`, no clock, no network) and encodes a story the
dashboard is built to surface — the same technique that makes `llm-observatory`'s demo land:

- **Growth** — signups accelerate over ~90 days, so DAU/WAU/MAU climb.
- **v2 launch (2026-05-30)** — a new model that uses ~40% more tokens: **cost per session steps
  up ~70% and stays elevated**, and **answer quality dips** for ~two weeks.
- **Quality fix (2026-06-13)** — quality recovers to slightly better than v1; the higher cost
  remains. The lasting cost step is the point: capability upgrades have a permanent unit-economics
  price a PM has to manage.

## Metric definitions

| Metric | Definition |
|---|---|
| **DAU / WAU / MAU** | Distinct users active in the trailing 1 / 7 / 28 days as of each day. Rolling *distinct* counts, computed over per-day user sets (not a groupby). |
| **Stickiness** | DAU / MAU — the share of monthly users who show up on a given day. |
| **Cohort retention** | Users grouped by signup week (Monday); `retention[cohort][k]` = fraction of that cohort with ≥1 session in week `k`. Week 0 is the signup week. |
| **Feature reach** | Distinct users of a feature ÷ all users. |
| **Activation funnel** | Signed up → Activated (≥1 session) → Engaged (≥2 active days) → Retained (a session ≥7 days after signup). |
| **CSAT** | 👍 share *among rated sessions* — `up / (up + down)`; unrated sessions don't count. |
| **Refusal / containment** | Share of sessions the assistant refused / resolved without escalation. |
| **Cost per active user** | Daily spend ÷ daily distinct actives (DAU) — the metric that shows whether a capability upgrade stays affordable as usage scales. |
| **A/B lift & significance** | Treatment − control on a binary success metric, tested with a **two-proportion z-test** (normal CDF via `math.erf`, no scipy). Reports absolute/relative lift, a 95% CI on the lift, a p-value, and a ship/stop/keep-testing decision. See `experiments.py`. |

## Design decisions

1. **Two flat tables, metrics as pure functions.** Every chart is a small polars transform over
   `users`/`sessions`; no hidden state, each function independently testable against hand-computed
   values.
2. **Deterministic synthetic data with a built-in story.** A dashboard with random noise proves
   nothing; a seeded narrative lets the tests assert the story holds and gives viewers something
   real to read.
3. **CSAT among rated only.** Counting unrated sessions as neutral would dilute a genuine signal;
   the rated-share is reported separately so the denominator is honest.
4. **Cohorts/activation use full history.** Date/feature slices distort cohort curves, so those
   two views intentionally ignore those filters (plan still applies) — a deliberate analytics
   choice, documented rather than silent.
5. **The product counterpart to observability.** This repo answers "is the product working for
   users and does the economics hold," where `llm-observatory` answers "is the system healthy" —
   two audiences, one telemetry substrate.

## Limitations & future work

- Synthetic data models a plausible product, not a real one; the schema is shaped so a real
  event stream (or a query over `llm-observatory` traces) could drop in behind the metrics.
- Retention is session-based; revenue/expansion retention would need a billing table.
- The A/B layer (`experiments.py`) uses a two-proportion z-test for binary metrics; continuous
  metrics (e.g. cost/session) would want a Welch's t-test, and sequential testing would need
  alpha-spending. Both are natural extensions.
