"""Adoption & engagement: active-user counts and stickiness over time.

`active_users_series` is parameterized by window — DAU (1), WAU (7), MAU (28) — and counts
*distinct* users active within the trailing window as of each calendar day. Rolling distinct
counts don't reduce to a groupby, so we compute them directly over per-day user sets.
"""

from datetime import date, timedelta

import polars as pl

_EMPTY = pl.DataFrame(schema={"date": pl.Date, "active_users": pl.Int64})


def _users_by_day(sessions: pl.DataFrame) -> dict[date, set[str]]:
    du = sessions.select("date", "user_id").unique()
    by_day: dict[date, set[str]] = {}
    for d, u in zip(du["date"].to_list(), du["user_id"].to_list(), strict=True):
        by_day.setdefault(d, set()).add(u)
    return by_day


def active_users_series(sessions: pl.DataFrame, window_days: int = 1) -> pl.DataFrame:
    """Distinct users active in the trailing `window_days` as of each day (DAU=1, WAU=7, MAU=28)."""
    if sessions.height == 0:
        return _EMPTY
    by_day = _users_by_day(sessions)
    start, end = min(by_day), max(by_day)
    rows = []
    for i in range((end - start).days + 1):
        d = start + timedelta(days=i)
        active: set[str] = set()
        for j in range(window_days):
            active |= by_day.get(d - timedelta(days=j), set())
        rows.append({"date": d, "active_users": len(active)})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def stickiness_series(sessions: pl.DataFrame) -> pl.DataFrame:
    """DAU/MAU per day — the classic 'how many monthly users show up daily' ratio."""
    dau = active_users_series(sessions, 1).rename({"active_users": "dau"})
    mau = active_users_series(sessions, 28).rename({"active_users": "mau"})
    return dau.join(mau, on="date", how="left").with_columns(
        (pl.col("dau") / pl.col("mau")).alias("stickiness")
    )
