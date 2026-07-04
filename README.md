# ai-product-analytics 📈

[![CI](https://github.com/himanshutamboli/ai-product-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/himanshutamboli/ai-product-analytics/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/lint-ruff-orange.svg)](https://github.com/astral-sh/ruff)
[![status](https://img.shields.io/badge/status-building-blue.svg)](#roadmap)

> The **PM / growth analytics** view of a live GenAI product — North Star + adoption, retention
> cohorts, feature funnels, AI quality & CSAT, and unit economics — in one Streamlit dashboard.
> The product-side counterpart to [`llm-observatory`](https://github.com/himanshutamboli/llm-observatory)'s
> engineer-side view.

## Why this exists

Engineers watch traces, latency, and error rates. **Product** watches a different scoreboard:
are people adopting the AI features, coming back, getting value — and does the unit economics
work as usage grows? This dashboard answers those questions for a (fictional) AI writing
assistant, on a deterministic synthetic dataset that tells a real story: steady growth, then a
**v2 model launch that lifts capability but spikes token cost and briefly dips quality** — the
kind of tradeoff a PM has to see and manage.

## What it shows

- **North Star & KPIs** — the headline metric + adoption (DAU/WAU/MAU, stickiness).
- **Retention** — weekly signup-cohort retention curves.
- **Feature funnel** — adoption and drop-off across the product's AI features.
- **AI quality** — answer-quality trend, CSAT (👍/👎), refusal & containment rates.
- **Unit economics** — token spend and cost per active user / session, by plan and model.

## Quickstart

```bash
uv sync --dev
uv run ai-product-analytics          # generate the dataset → data/*.parquet + summary
uv run streamlit run app.py          # launch the dashboard
uv run pytest
```

## Roadmap

| Step | Deliverable |
|---|---|
| 1 ✅ | Synthetic GenAI-product dataset (users + sessions) with a built-in growth/quality/cost narrative |
| 2 | Metrics package: adoption, retention cohorts, feature funnel, AI quality, unit economics |
| 3 | Streamlit dashboard: KPI header + sectioned analytics with filters |
| 4 | Docs + demo screenshot; ship v1.0 |

## License

MIT
