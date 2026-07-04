"""Metrics verified against a tiny hand-computed fixture.

users:   u1(free, signup 04-06), u2(pro, 04-06), u3(free, 04-13)   [04-06 and 04-13 are Mondays]
sessions:
  04-06 u1 chat  up   resolved      04-06 u2 code  down resolved
  04-07 u1 draft none REFUSED (not resolved)
  04-13 u1 chat  up   resolved      04-13 u3 chat  none resolved
"""

from datetime import date

import polars as pl
import pytest

from ai_product_analytics.metrics import (
    activation_funnel,
    active_users_series,
    cohort_retention,
    feature_adoption,
    quality_summary,
    unit_economics_series,
)

USERS = pl.DataFrame(
    {
        "user_id": ["u1", "u2", "u3"],
        "signup_date": [date(2026, 4, 6), date(2026, 4, 6), date(2026, 4, 13)],
        "plan": ["free", "pro", "free"],
    }
).with_columns(pl.col("signup_date").cast(pl.Date))

SESSIONS = pl.DataFrame(
    {
        "user_id": ["u1", "u2", "u1", "u1", "u3"],
        "date": [
            date(2026, 4, 6),
            date(2026, 4, 6),
            date(2026, 4, 7),
            date(2026, 4, 13),
            date(2026, 4, 13),
        ],
        "feature": ["chat", "code", "draft", "chat", "chat"],
        "feedback": ["up", "down", "none", "up", "none"],
        "refused": [False, False, True, False, False],
        "resolved": [True, True, False, True, True],
        "quality": [0.90, 0.50, 0.80, 0.88, 0.85],
        "cost_usd": [0.010, 0.020, 0.010, 0.010, 0.010],
        "tokens": [100, 200, 120, 110, 90],
    }
).with_columns(pl.col("date").cast(pl.Date))


def _at(df, d):
    return df.filter(pl.col("date") == date.fromisoformat(d))


def test_dau_counts_distinct_users_per_day():
    dau = active_users_series(SESSIONS, 1)
    assert _at(dau, "2026-04-06")["active_users"].item() == 2
    assert _at(dau, "2026-04-07")["active_users"].item() == 1
    assert _at(dau, "2026-04-13")["active_users"].item() == 2
    # the full calendar range is filled, including the quiet days between
    assert dau.height == (date(2026, 4, 13) - date(2026, 4, 6)).days + 1


def test_wau_rolls_distinct_users_over_window():
    wau = active_users_series(SESSIONS, 7)
    # trailing 7 days ending 04-13 covers 04-07..04-13 → {u1, u3}
    assert _at(wau, "2026-04-13")["active_users"].item() == 2
    # ending 04-12 covers 04-06..04-12 → {u1, u2}
    assert _at(wau, "2026-04-12")["active_users"].item() == 2


def test_quality_summary_csat_refusal_containment():
    q = quality_summary(SESSIONS)
    assert q["csat"] == pytest.approx(2 / 3)  # 2 up, 1 down, 2 unrated
    assert q["refusal_rate"] == pytest.approx(1 / 5)
    assert q["containment_rate"] == pytest.approx(4 / 5)
    assert q["rated_share"] == pytest.approx(3 / 5)


def test_cohort_retention_week0_and_week1():
    r = cohort_retention(SESSIONS, USERS)
    c0 = r.filter(pl.col("cohort") == date(2026, 4, 6))
    assert c0.filter(pl.col("week_index") == 0)["retention"].item() == pytest.approx(1.0)
    assert c0.filter(pl.col("week_index") == 1)["retention"].item() == pytest.approx(0.5)
    c1 = r.filter(pl.col("cohort") == date(2026, 4, 13))
    assert c1.filter(pl.col("week_index") == 0)["retention"].item() == pytest.approx(1.0)


def test_feature_adoption_reach():
    fa = feature_adoption(SESSIONS)
    chat = fa.filter(pl.col("feature") == "chat")
    assert chat["users"].item() == 2 and chat["sessions"].item() == 3
    assert chat["reach"].item() == pytest.approx(2 / 3)


def test_activation_funnel_stages():
    f = activation_funnel(SESSIONS, USERS)
    got = dict(zip(f["stage"].to_list(), f["users"].to_list(), strict=True))
    assert got == {"Signed up": 3, "Activated": 3, "Engaged (2+ days)": 1, "Retained (wk 2+)": 1}


def test_cost_per_active_user_by_day():
    ue = unit_economics_series(SESSIONS)
    assert _at(ue, "2026-04-06")["cost_per_active_user"].item() == pytest.approx(0.03 / 2)
    assert _at(ue, "2026-04-07")["cost_per_active_user"].item() == pytest.approx(0.01)
    assert _at(ue, "2026-04-13")["cost_per_active_user"].item() == pytest.approx(0.02 / 2)
