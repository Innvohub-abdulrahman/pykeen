"""Economics.

The numbers here are *computed* from the parameter block in
:mod:`agora.economics`, not asserted. These tests pin the ratios, which are far
more stable than the absolutes, and check the model against the figures
published in the system report with an explicit tolerance rather than
pretending to a precision the model does not have.
"""

from __future__ import annotations

import pytest

from agora.economics import (
    ALL_LEVERS,
    FRONTIER,
    LOCAL,
    MID,
    Levers,
    Mix,
    TaskShape,
    VerificationCost,
    cost_of_task,
    email_triage_scenario,
    feature_task_scenario,
    honest_factor,
    judge_model_comparison,
    mobile_app_scenario,
    naive_total,
    waterfall,
)

#: How far a computed figure may sit from the published one. The report itself
#: says the prices are placeholders; holding the model to the cent would be
#: false precision.
TOLERANCE = 0.12


def _rel(actual: float, expected: float) -> float:
    return abs(actual - expected) / expected


# --------------------------------------------------------------------------
# Each lever does what it claims
# --------------------------------------------------------------------------


def test_every_lever_reduces_cost():
    rows = waterfall()
    costs = [row.cost_usd for row in rows]
    assert costs == sorted(costs, reverse=True)
    assert all(row.delta_pct < 0 for row in rows[1:])


@pytest.mark.parametrize(
    "index,label,expected",
    [
        (0, "baseline", 16.25),
        (1, "reference passing", 8.21),
        (2, "compiled context", 3.98),
        (3, "prompt caching", 2.36),
        (4, "plan cache and turn caps", 1.57),
        (5, "model ladder", 0.35),
    ],
)
def test_the_waterfall_matches_the_published_figures(index, label, expected):
    actual = waterfall()[index].cost_usd
    assert _rel(actual, expected) < TOLERANCE, (
        f"{label}: computed ${actual:.2f} against a published ${expected:.2f}"
    )


def test_turn_caps_are_exactly_linear_in_turns():
    """24 turns to 16 is a third of the work, and the model should say so."""
    rows = waterfall()
    assert rows[4].delta_pct == pytest.approx(-33.3, abs=0.2)


def test_the_cumulative_factor_is_in_the_right_neighbourhood():
    factor = waterfall()[-1].cumulative_factor
    assert 35 <= factor <= 60, f"{factor:.0f}x"


def test_the_honest_number_is_smaller_than_the_headline():
    """Against engineering that already passes references and caches prompts."""
    factor = honest_factor()
    assert 3 <= factor <= 8, (
        f"{factor:.1f}x against competent engineering; the report says 3-8x and "
        f"says not to put 46x in a deck"
    )


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario_fn,naive,optimised",
    [
        (feature_task_scenario, 23.32, 0.40),
        (email_triage_scenario, 4500.0, 96.0),
        (mobile_app_scenario, 1129.0, 19.0),
    ],
)
def test_scenarios_land_near_the_published_figures(scenario_fn, naive, optimised):
    scenario = scenario_fn()
    assert _rel(scenario.naive_usd, naive) < TOLERANCE, (
        f"{scenario.name}: naive ${scenario.naive_usd:,.2f} vs ${naive:,.2f}"
    )
    assert _rel(scenario.optimised_usd, optimised) < 0.25, (
        f"{scenario.name}: optimised ${scenario.optimised_usd:,.2f} vs ${optimised:,.2f}"
    )


def test_email_triage_saves_by_not_calling_a_model_at_all():
    """The instructive one: 185 of 200 messages a day never reach a paid model."""
    scenario = email_triage_scenario()
    assert scenario.factor > 20
    assert "never reach a paid model" in scenario.note


def test_a_local_model_has_no_marginal_cost():
    assert LOCAL.cost(input_tokens=1_000_000, output_tokens=1_000_000) == 0.0
    assert MID.input_per_mtok < FRONTIER.input_per_mtok


def test_prompt_caching_only_discounts_the_cached_prefix():
    fresh = FRONTIER.cost(input_tokens=10_000, output_tokens=0)
    cached = FRONTIER.cost(input_tokens=10_000, output_tokens=0, cached_tokens=10_000)
    assert cached == pytest.approx(fresh * FRONTIER.cached_input_multiplier)


def test_the_model_ladder_mix_normalises():
    mix = Mix(local=2, mid=1, frontier=1).normalised()
    assert mix.local + mix.mid + mix.frontier == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Verification economics
# --------------------------------------------------------------------------


def test_the_universal_floor_is_free_and_grade_a_verification_nearly_so():
    costs = VerificationCost()
    assert costs.universal_floor == 0.0
    assert costs.cost_for_grade("A") == 0.0
    assert costs.cost_for_grade("B") < costs.cost_for_grade("C")
    assert costs.cost_for_grade("D") > costs.cost_for_grade("B")


def test_judging_everything_with_a_model_costs_real_money():
    judge, floor = judge_model_comparison(artifacts_per_day=500, days=365)
    assert judge > 2_000
    assert floor == 0.0


def test_a_naive_run_pays_for_planning_judging_and_rework():
    shape = TaskShape()
    assert naive_total(shape) > cost_of_task(shape, Levers()), (
        "the costs a naive framework pays without noticing are not zero"
    )
