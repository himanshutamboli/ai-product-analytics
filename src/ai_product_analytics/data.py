"""A deterministic synthetic dataset for a fictional GenAI product ("Lumen", an AI writing
assistant), so the dashboard has a realistic story to tell — offline, in CI, reproducibly.

The data encodes a deliberate narrative the dashboard should surface (the same trick that makes
`llm-observatory`'s demo compelling): the product grows steadily, then **ships model v2 around
2026-05-30** — which lifts capability but *raises token usage and cost* and *briefly dips
answer quality* until a fix lands ~two weeks later. So Adoption trends up, Unit Economics steps
up (and stays up), and AI Quality shows a dip-and-recovery you can point at.

Two tables:
* `users`    — one row per signup (id, date, plan, channel, region).
* `sessions` — one row per usage session (feature, model, tokens, cost, quality, feedback, …).

Everything derives from these via the `metrics` package. Deterministic given a seed — no clock,
no network. Run `ai-product-analytics` (or `python -m ai_product_analytics.data`) to write
Parquet snapshots and print a summary.
"""

import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from ai_product_analytics.logging_config import get_logger

logger = get_logger(__name__)

PRODUCT = "Lumen"

START = date(2026, 4, 2)
END = date(2026, 6, 30)  # ~90 days of history
MODEL_CHANGE = date(2026, 5, 30)  # v2 ships: capability ↑, tokens/cost ↑, quality briefly ↓
QUALITY_RECOVERY = date(2026, 6, 13)  # a fix restores quality; the higher cost stays

PLANS = ["free", "pro", "team"]
PLAN_WEIGHTS = [0.70, 0.24, 0.06]
PLAN_ACTIVITY = {"free": 0.16, "pro": 0.42, "team": 0.58}  # daily-active propensity
PLAN_INTENSITY = {"free": 1.0, "pro": 1.6, "team": 2.2}  # sessions-per-active-day multiplier

FEATURES = ["chat", "summarize", "draft", "code", "translate"]
FEATURE_WEIGHTS = [0.38, 0.22, 0.20, 0.12, 0.08]
FEATURE_QUALITY = {"chat": 0.82, "summarize": 0.85, "draft": 0.80, "code": 0.72, "translate": 0.83}
# base (tokens_in, tokens_out) per session by feature
FEATURE_TOKENS = {
    "chat": (420, 260),
    "summarize": (1500, 300),
    "draft": (300, 620),
    "code": (520, 700),
    "translate": (360, 360),
}

CHANNELS = ["organic", "paid_search", "referral", "content", "partner"]
CHANNEL_WEIGHTS = [0.34, 0.26, 0.18, 0.14, 0.08]
REGIONS = ["NA", "EU", "APAC", "LATAM"]
REGION_WEIGHTS = [0.44, 0.30, 0.18, 0.08]

# $ per 1M tokens (input, output)
PRICES = {"assistant-v1": (2.0, 10.0), "assistant-v2": (2.5, 12.0)}


@dataclass
class Dataset:
    users: pl.DataFrame
    sessions: pl.DataFrame


def _daterange(start: date, end: date):
    for i in range((end - start).days + 1):
        yield start + timedelta(days=i)


def _model_for(day: date) -> str:
    return "assistant-v2" if day >= MODEL_CHANGE else "assistant-v1"


def _quality_delta(day: date) -> float:
    if day < MODEL_CHANGE:
        return 0.0
    if day < QUALITY_RECOVERY:
        return -0.12  # the regression window
    return 0.02  # v2, fixed — slightly better than v1


def _generate_users(rng: random.Random) -> list[dict]:
    users: list[dict] = []
    uid = 0
    for i, day in enumerate(_daterange(START, END)):
        signups = max(1, round(6 + 0.28 * i + rng.gauss(0, 2)))
        for _ in range(signups):
            uid += 1
            users.append(
                {
                    "user_id": f"u{uid:05d}",
                    "signup_date": day,
                    "plan": rng.choices(PLANS, PLAN_WEIGHTS)[0],
                    "channel": rng.choices(CHANNELS, CHANNEL_WEIGHTS)[0],
                    "region": rng.choices(REGIONS, REGION_WEIGHTS)[0],
                }
            )
    return users


def _active_probability(plan: str, days_since_signup: int) -> float:
    """Retention curve: a launch burst that decays toward a habitual-use floor."""
    p0 = PLAN_ACTIVITY[plan]
    floor = p0 * 0.30
    return floor + (p0 - floor) * 2.718281828 ** (-days_since_signup / 28.0)


def _make_session(rng: random.Random, sid: int, user: dict, day: date) -> dict:
    feature = rng.choices(FEATURES, FEATURE_WEIGHTS)[0]
    model = _model_for(day)
    tok_mult = 1.4 if model == "assistant-v2" else 1.0  # v2 uses more context
    base_in, base_out = FEATURE_TOKENS[feature]
    tokens_in = max(1, int(base_in * tok_mult * rng.uniform(0.8, 1.25)))
    tokens_out = max(1, int(base_out * tok_mult * rng.uniform(0.8, 1.25)))
    price_in, price_out = PRICES[model]
    cost = (tokens_in * price_in + tokens_out * price_out) / 1e6

    quality = FEATURE_QUALITY[feature] + _quality_delta(day) + rng.gauss(0, 0.06)
    quality = min(0.99, max(0.30, quality))

    r = rng.random()
    if r < quality * 0.28:
        feedback = "up"
    elif r < quality * 0.28 + (1 - quality) * 0.14:
        feedback = "down"
    else:
        feedback = "none"

    refuse_p = 0.02 + (0.06 if MODEL_CHANGE <= day < QUALITY_RECOVERY else 0.0)
    refuse_p += 0.05 if quality < 0.5 else 0.0
    refused = rng.random() < refuse_p
    resolved = (not refused) and (rng.random() < min(0.97, quality * 0.95))

    latency = int(
        (650 + (tokens_in + tokens_out) * 0.35) * (1.15 if model == "assistant-v2" else 1.0)
    )
    latency = int(latency * rng.uniform(0.85, 1.25))

    return {
        "session_id": f"s{sid:06d}",
        "user_id": user["user_id"],
        "date": day,
        "plan": user["plan"],
        "feature": feature,
        "model": model,
        "messages": 1 + int(rng.random() * 6),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens": tokens_in + tokens_out,
        "latency_ms": latency,
        "cost_usd": round(cost, 6),
        "quality": round(quality, 4),
        "feedback": feedback,
        "refused": refused,
        "resolved": resolved,
    }


def _generate_sessions(rng: random.Random, users: list[dict]) -> list[dict]:
    sessions: list[dict] = []
    sid = 0
    for day in _daterange(START, END):
        for user in users:
            if user["signup_date"] > day:
                continue
            days_since = (day - user["signup_date"]).days
            if rng.random() >= _active_probability(user["plan"], days_since):
                continue
            n = 1 + int(rng.random() * 3 * PLAN_INTENSITY[user["plan"]])
            for _ in range(n):
                sid += 1
                sessions.append(_make_session(rng, sid, user, day))
    return sessions


def generate(seed: int = 7) -> Dataset:
    """Build the full deterministic dataset for the given seed."""
    rng = random.Random(seed)
    users = _generate_users(rng)
    sessions = _generate_sessions(rng, users)
    users_df = pl.DataFrame(users).with_columns(pl.col("signup_date").cast(pl.Date))
    sessions_df = pl.DataFrame(sessions).with_columns(pl.col("date").cast(pl.Date))
    return Dataset(users=users_df, sessions=sessions_df)


def main() -> None:
    ds = generate()
    out = Path("data")
    out.mkdir(exist_ok=True)
    ds.users.write_parquet(out / "users.parquet")
    ds.sessions.write_parquet(out / "sessions.parquet")

    s = ds.sessions
    pre = s.filter(pl.col("date") < MODEL_CHANGE)["cost_usd"].mean()
    post = s.filter(pl.col("date") >= MODEL_CHANGE)["cost_usd"].mean()
    logger.info(
        "%s: %d users, %d sessions (%s → %s)", PRODUCT, ds.users.height, s.height, START, END
    )
    logger.info(
        "total cost $%.2f · avg quality %.3f · cost/session pre-v2 $%.5f → post-v2 $%.5f (+%.0f%%)",
        s["cost_usd"].sum(),
        s["quality"].mean(),
        pre,
        post,
        (post / pre - 1) * 100,
    )


if __name__ == "__main__":
    main()
