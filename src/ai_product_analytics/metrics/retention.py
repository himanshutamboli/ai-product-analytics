"""Weekly signup-cohort retention.

Each user belongs to the cohort of their signup week (Monday). `cohort_retention` returns a
tidy frame — one row per (cohort, week_index) — where `retention` is the fraction of that
cohort with at least one session in that week. Week 0 is the signup week.
"""

import polars as pl


def cohort_retention(
    sessions: pl.DataFrame, users: pl.DataFrame, max_weeks: int = 8
) -> pl.DataFrame:
    u = users.select("user_id", "signup_date").with_columns(
        cohort=pl.col("signup_date").dt.truncate("1w")
    )
    s = (
        sessions.select("user_id", "date")
        .join(u, on="user_id", how="inner")
        .with_columns(
            week_index=((pl.col("date") - pl.col("cohort")).dt.total_days() // 7).cast(pl.Int64)
        )
    )
    sizes = u.group_by("cohort").agg(cohort_size=pl.len())
    active = (
        s.filter((pl.col("week_index") >= 0) & (pl.col("week_index") <= max_weeks))
        .select("cohort", "week_index", "user_id")
        .unique()
        .group_by("cohort", "week_index")
        .agg(active=pl.len())
    )
    return (
        active.join(sizes, on="cohort", how="left")
        .with_columns(retention=pl.col("active") / pl.col("cohort_size"))
        .sort(["cohort", "week_index"])
    )
