"""Unit economics: token spend and cost per active user / session over time.

Cost-per-active-user joins daily spend against daily distinct actives (DAU) — the metric that
tells a PM whether a capability upgrade (e.g. a pricier, token-hungrier model) is affordable as
usage scales.
"""

import polars as pl

from ai_product_analytics.metrics.adoption import active_users_series


def cost_series(sessions: pl.DataFrame) -> pl.DataFrame:
    return (
        sessions.group_by("date")
        .agg(cost=pl.col("cost_usd").sum(), tokens=pl.col("tokens").sum(), sessions=pl.len())
        .sort("date")
    )


def unit_economics_series(sessions: pl.DataFrame) -> pl.DataFrame:
    daily = cost_series(sessions)
    dau = active_users_series(sessions, 1)
    return (
        daily.join(dau, on="date", how="left")
        .with_columns(
            cost_per_active_user=pl.col("cost") / pl.col("active_users"),
            cost_per_session=pl.col("cost") / pl.col("sessions"),
        )
        .sort("date")
    )


def spend_by(sessions: pl.DataFrame, dim: str) -> pl.DataFrame:
    """Total spend / tokens / sessions grouped by a dimension (plan, model, feature)."""
    g = (
        sessions.group_by(dim)
        .agg(cost=pl.col("cost_usd").sum(), tokens=pl.col("tokens").sum(), sessions=pl.len())
        .sort("cost", descending=True)
    )
    total = g["cost"].sum()
    return g.with_columns(share=pl.col("cost") / total)


def economics_summary(sessions: pl.DataFrame) -> dict:
    return {
        "total_cost": sessions["cost_usd"].sum(),
        "total_tokens": int(sessions["tokens"].sum()),
        "cost_per_session": sessions["cost_usd"].mean(),
        "sessions": sessions.height,
    }
