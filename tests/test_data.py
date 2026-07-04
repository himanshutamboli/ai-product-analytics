import polars as pl

from ai_product_analytics.data import END, MODEL_CHANGE, QUALITY_RECOVERY, START, generate


def test_schema_and_shape():
    ds = generate()
    assert ds.users.height > 500
    assert ds.sessions.height > 5000
    assert set(ds.users.columns) == {"user_id", "signup_date", "plan", "channel", "region"}
    assert {
        "session_id",
        "user_id",
        "date",
        "feature",
        "model",
        "tokens",
        "cost_usd",
        "quality",
        "feedback",
        "refused",
        "resolved",
    } <= set(ds.sessions.columns)
    # every session belongs to a known user, and never predates that user's signup
    joined = ds.sessions.join(ds.users, on="user_id", how="left")
    assert joined["signup_date"].null_count() == 0
    assert (joined["date"] >= joined["signup_date"]).all()


def test_dataset_is_deterministic():
    a, b = generate(seed=7), generate(seed=7)
    assert a.sessions.equals(b.sessions) and a.users.equals(b.users)
    assert not generate(seed=1).sessions.equals(a.sessions)


def test_dates_are_within_range():
    ds = generate()
    assert ds.sessions["date"].min() >= START
    assert ds.sessions["date"].max() <= END


def test_narrative_v2_raises_cost_and_dips_quality():
    s = generate().sessions
    pre = s.filter(pl.col("date") < MODEL_CHANGE)
    bad = s.filter((pl.col("date") >= MODEL_CHANGE) & (pl.col("date") < QUALITY_RECOVERY))
    fixed = s.filter(pl.col("date") >= QUALITY_RECOVERY)

    # v2 uses more tokens → higher cost per session, and it persists after the quality fix
    assert bad["cost_usd"].mean() > pre["cost_usd"].mean() * 1.2
    assert fixed["cost_usd"].mean() > pre["cost_usd"].mean() * 1.2
    # quality dips during the regression window, then recovers
    assert bad["quality"].mean() < pre["quality"].mean()
    assert fixed["quality"].mean() > bad["quality"].mean()
    # only v2 runs after the model change
    assert s.filter(pl.col("date") >= MODEL_CHANGE)["model"].unique().to_list() == ["assistant-v2"]
