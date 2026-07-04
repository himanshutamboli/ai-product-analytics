"""A/B experimentation readouts with proper significance testing.

Product teams don't ship on eyeballed deltas — they ship on experiments. This module assigns
enrolled users to control/treatment for a set of experiments and reports each readout with a
**two-proportion z-test**: absolute & relative lift, a 95% confidence interval, a p-value, and a
ship / stop / keep-testing decision.

The z-test is computed from scratch (normal CDF via `math.erf`) — no scipy dependency, and it
shows the statistics rather than hiding them behind a library call. Assignment and outcomes are
deterministic given a seed, so the readouts (and their tests) are reproducible.

The seeded experiments are an honest mix: a clear winner, a clear loser, and an inconclusive one.
"""

import math
import random
from dataclasses import dataclass

import polars as pl


@dataclass
class Experiment:
    key: str
    name: str
    metric: str  # the binary success metric being moved
    control_rate: float
    treatment_rate: float
    enroll_fraction: float
    hypothesis: str


EXPERIMENTS = [
    Experiment(
        "onboarding_v2",
        "Guided onboarding v2",
        "activation rate",
        0.62,
        0.68,
        1.0,
        "A guided first-run flow lifts activation.",
    ),
    Experiment(
        "concise_prompt",
        "Concise system prompt",
        "CSAT",
        0.860,
        0.872,
        0.8,
        "A tighter system prompt nudges satisfaction up.",
    ),
    Experiment(
        "proactive_suggest",
        "Proactive suggestions",
        "containment rate",
        0.72,
        0.66,
        1.0,
        "Proactively suggesting next actions boosts self-serve resolution.",
    ),
]


def _phi(z: float) -> float:
    """Standard-normal CDF."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def two_proportion_z_test(x_c: int, n_c: int, x_t: int, n_t: int) -> dict:
    """Two-proportion z-test (treatment vs control) with a 95% CI on the absolute lift."""
    p_c, p_t = x_c / n_c, x_t / n_t
    pooled = (x_c + x_t) / (n_c + n_t)
    se_pooled = math.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    z = (p_t - p_c) / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - _phi(abs(z)))
    diff = p_t - p_c
    se_diff = math.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    return {
        "rate_control": p_c,
        "rate_treatment": p_t,
        "abs_lift": diff,
        "rel_lift": diff / p_c if p_c else 0.0,
        "z": z,
        "p_value": p_value,
        "ci_low": diff - 1.96 * se_diff,
        "ci_high": diff + 1.96 * se_diff,
        "significant": p_value < 0.05,
    }


def generate_assignments(users: pl.DataFrame, seed: int = 11) -> pl.DataFrame:
    """Enroll users into each experiment, assign 50/50, and draw a binary outcome per user."""
    rng = random.Random(seed)
    uids = users["user_id"].to_list()
    rows = []
    for exp in EXPERIMENTS:
        for uid in uids:
            if rng.random() > exp.enroll_fraction:
                continue
            variant = "treatment" if rng.random() < 0.5 else "control"
            rate = exp.treatment_rate if variant == "treatment" else exp.control_rate
            rows.append(
                {
                    "experiment": exp.key,
                    "name": exp.name,
                    "metric": exp.metric,
                    "hypothesis": exp.hypothesis,
                    "user_id": uid,
                    "variant": variant,
                    "converted": rng.random() < rate,
                }
            )
    return pl.DataFrame(rows)


def _decision(stat: dict) -> str:
    if stat["significant"] and stat["abs_lift"] > 0:
        return "Ship 🚀"
    if stat["significant"] and stat["abs_lift"] < 0:
        return "Stop 🛑"
    return "Keep testing ⏳"


def ab_results(assignments: pl.DataFrame) -> pl.DataFrame:
    """Per-experiment readout: rates, lift, CI, p-value, and a decision."""
    rows = []
    for key in assignments["experiment"].unique(maintain_order=True).to_list():
        g = assignments.filter(pl.col("experiment") == key)
        c = g.filter(pl.col("variant") == "control")
        t = g.filter(pl.col("variant") == "treatment")
        stat = two_proportion_z_test(
            int(c["converted"].sum()), c.height, int(t["converted"].sum()), t.height
        )
        rows.append(
            {
                "experiment": key,
                "name": g["name"][0],
                "metric": g["metric"][0],
                "hypothesis": g["hypothesis"][0],
                "n_control": c.height,
                "n_treatment": t.height,
                **stat,
                "decision": _decision(stat),
            }
        )
    return pl.DataFrame(rows)
