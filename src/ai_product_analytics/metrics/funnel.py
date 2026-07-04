"""Feature adoption and the activation funnel.

* `feature_adoption` — reach (distinct users) and usage per AI feature.
* `activation_funnel` — the signup → activated → engaged → retained drop-off, the funnel a PM
  watches to see whether new users actually stick.
"""

import polars as pl


def feature_adoption(sessions: pl.DataFrame) -> pl.DataFrame:
    total_users = sessions["user_id"].n_unique()
    return (
        sessions.group_by("feature")
        .agg(
            users=pl.col("user_id").n_unique(),
            sessions=pl.len(),
            cost=pl.col("cost_usd").sum(),
        )
        .with_columns(reach=pl.col("users") / total_users)
        .sort("users", descending=True)
    )


def activation_funnel(sessions: pl.DataFrame, users: pl.DataFrame) -> pl.DataFrame:
    signed_up = users.height
    activated = sessions["user_id"].n_unique()  # ≥1 session

    days_per_user = (
        sessions.select("user_id", "date").unique().group_by("user_id").agg(days=pl.len())
    )
    engaged = days_per_user.filter(pl.col("days") >= 2).height  # active on ≥2 distinct days

    with_signup = sessions.select("user_id", "date").join(
        users.select("user_id", "signup_date"), on="user_id", how="inner"
    )
    retained = with_signup.filter((pl.col("date") - pl.col("signup_date")).dt.total_days() >= 7)[
        "user_id"
    ].n_unique()

    rows = [
        {"stage": "Signed up", "users": signed_up},
        {"stage": "Activated", "users": activated},
        {"stage": "Engaged (2+ days)", "users": engaged},
        {"stage": "Retained (wk 2+)", "users": retained},
    ]
    return pl.DataFrame(rows).with_columns(pct=pl.col("users") / signed_up)
