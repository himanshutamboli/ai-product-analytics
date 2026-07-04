"""AI quality signals: answer quality, CSAT (👍/👎), refusal rate, containment.

CSAT is the thumbs-up share *among rated sessions* (up / (up + down)); sessions with no
feedback don't count either way. Containment is the share of sessions resolved without escalation.
"""

import polars as pl


def _csat_expr() -> pl.Expr:
    up = pl.col("feedback").eq("up").sum()
    down = pl.col("feedback").eq("down").sum()
    return pl.when((up + down) > 0).then(up / (up + down)).otherwise(None).alias("csat")


def quality_series(sessions: pl.DataFrame) -> pl.DataFrame:
    """Daily quality, CSAT, refusal rate, and containment rate."""
    return (
        sessions.group_by("date")
        .agg(
            quality=pl.col("quality").mean(),
            csat=_csat_expr(),
            refusal_rate=pl.col("refused").cast(pl.Float64).mean(),
            containment_rate=pl.col("resolved").cast(pl.Float64).mean(),
            sessions=pl.len(),
        )
        .sort("date")
    )


def quality_summary(sessions: pl.DataFrame) -> dict:
    up = int(sessions.select(pl.col("feedback").eq("up").sum()).item())
    down = int(sessions.select(pl.col("feedback").eq("down").sum()).item())
    rated = up + down
    return {
        "avg_quality": sessions["quality"].mean(),
        "csat": up / rated if rated else None,
        "refusal_rate": sessions["refused"].cast(pl.Float64).mean(),
        "containment_rate": sessions["resolved"].cast(pl.Float64).mean(),
        "rated_share": rated / sessions.height if sessions.height else None,
    }
