import pytest

from ai_product_analytics.data import generate
from ai_product_analytics.experiments import (
    ab_results,
    generate_assignments,
    two_proportion_z_test,
)


def test_z_test_matches_hand_computed_case():
    # 50/100 vs 70/100: pooled p=0.6, se=sqrt(0.6*0.4*0.02)=0.06928, z=0.2/0.06928≈2.887
    r = two_proportion_z_test(50, 100, 70, 100)
    assert r["z"] == pytest.approx(2.887, abs=0.01)
    assert r["p_value"] == pytest.approx(0.0039, abs=0.001)
    assert r["abs_lift"] == pytest.approx(0.20)
    assert r["significant"]


def test_z_test_flat_difference_is_not_significant():
    r = two_proportion_z_test(500, 1000, 505, 1000)  # 50.0% vs 50.5%
    assert not r["significant"] and r["p_value"] > 0.05


def test_assignments_are_deterministic_and_balanced():
    users = generate().users
    a, b = generate_assignments(users, seed=11), generate_assignments(users, seed=11)
    assert a.equals(b)
    # roughly 50/50 within each experiment
    for key in a["experiment"].unique().to_list():
        g = a.filter(a["experiment"] == key)
        share = g.filter(g["variant"] == "treatment").height / g.height
        assert 0.45 < share < 0.55


def test_ab_results_tell_an_honest_mix():
    res = ab_results(generate_assignments(generate().users))
    by = {r["experiment"]: r for r in res.to_dicts()}

    assert by["onboarding_v2"]["significant"] and by["onboarding_v2"]["decision"] == "Ship 🚀"
    assert (
        by["proactive_suggest"]["significant"] and by["proactive_suggest"]["decision"] == "Stop 🛑"
    )
    assert not by["concise_prompt"]["significant"]
    assert by["concise_prompt"]["decision"] == "Keep testing ⏳"
    # the CI on a significant winner excludes zero
    assert by["onboarding_v2"]["ci_low"] > 0
