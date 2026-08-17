"""Measured trust: what the ledger will and will not say, and what the router
does with each answer."""

from __future__ import annotations

import random

import pytest

from agora.contracts import AuthorityEnvelope, DataClass, TaskContract
from agora.errors import NoRouteAvailable
from agora.ledger import (
    BucketSpec,
    CapabilityLedger,
    CrossTenantPriors,
    EvidenceClass,
    Outcome,
    OutcomeSource,
    shrink_factor,
)
from agora.registry import AgentCard, AgentRegistry, Probation
from agora.router import GapLog, PolicyFilter, Router, Stage
from agora.stats import BetaPosterior, beta_ppf, betainc

from conftest import seed_outcomes

AGENT = "agent://eng/impl"
CAP = "code.implement"
FAMILIAR = ("python", "small", "familiar")


# --------------------------------------------------------------------------
# The maths
# --------------------------------------------------------------------------


def test_betainc_matches_a_closed_form():
    assert betainc(1, 1, 0.37) == pytest.approx(0.37, abs=1e-9)
    assert betainc(2, 3, 0.5) == pytest.approx(0.6875, abs=1e-9)


def test_ppf_inverts_the_cdf():
    for a, b, q in [(2, 5, 0.05), (7, 3, 0.5), (30, 4, 0.95)]:
        assert betainc(a, b, beta_ppf(a, b, q)) == pytest.approx(q, abs=1e-6)


def test_the_interval_narrows_as_evidence_accumulates():
    widths = [
        BetaPosterior(1 + round(n * 0.92), 1 + n - round(n * 0.92)).interval().width
        for n in (2, 8, 30, 67)
    ]
    assert widths == sorted(widths, reverse=True)
    assert widths[0] > 0.5, "two samples should say almost nothing"
    assert widths[-1] < 0.15, "sixty-seven should say quite a lot"


# --------------------------------------------------------------------------
# Evidence classes
# --------------------------------------------------------------------------


def test_thirty_production_outcomes_reach_production_measured(ledger):
    seed_outcomes(ledger, AGENT, CAP, FAMILIAR, 67, 62)
    reading = ledger.read(AGENT, CAP, FAMILIAR)
    assert reading.evidence_class is EvidenceClass.PRODUCTION_MEASURED
    assert reading.routable
    assert not reading.verification_mandatory


def test_harness_evidence_forces_verification(ledger):
    seed_outcomes(
        ledger, AGENT, CAP, ("swift", "small", "familiar"), 12, 11,
        source=OutcomeSource.HARNESS,
    )
    reading = ledger.read(AGENT, CAP, ("swift", "small", "familiar"))
    assert reading.evidence_class is EvidenceClass.HARNESS_VERIFIED
    assert reading.verification_mandatory


def test_two_samples_are_a_guess_not_a_measurement(ledger):
    seed_outcomes(ledger, AGENT, CAP, ("terraform", "novel"), 2, 2)
    reading = ledger.read(AGENT, CAP, ("terraform", "novel"))
    assert reading.evidence_class is EvidenceClass.DECLARED
    assert not reading.routable


def test_a_capability_below_the_floor_is_deprecated(ledger):
    seed_outcomes(ledger, AGENT, CAP, ("rust", "large", "novel"), 40, 17)
    reading = ledger.read(AGENT, CAP, ("rust", "large", "novel"))
    assert reading.evidence_class is EvidenceClass.DEPRECATED
    assert not reading.routable
    assert ledger.get(AGENT, CAP, ("rust", "large", "novel")).re_eval_queued


# --------------------------------------------------------------------------
# The load-bearing behaviours
# --------------------------------------------------------------------------


def test_self_reported_success_never_enters_the_ledger(ledger):
    result = ledger.record(
        Outcome(AGENT, CAP, FAMILIAR, True, OutcomeSource.SELF_REPORTED)
    )
    assert result is None
    assert ledger.rejected_self_reports == 1
    assert ledger.read(AGENT, CAP, FAMILIAR).n == 0


def test_a_verified_outcome_must_name_its_verifier():
    with pytest.raises(Exception):
        Outcome(AGENT, CAP, FAMILIAR, True, OutcomeSource.PRODUCTION, verified_by="")


def test_borrowed_evidence_collapses_toward_the_prior(ledger):
    """0.92 on small familiar Python must not become 0.92 on novel Terraform."""
    seed_outcomes(ledger, AGENT, CAP, FAMILIAR, 67, 62)
    direct = ledger.read(AGENT, CAP, FAMILIAR)
    borrowed = ledger.read(AGENT, CAP, ("terraform", "novel"))

    assert borrowed.borrowed
    assert borrowed.mean < direct.mean - 0.25
    assert borrowed.interval.width > direct.interval.width * 2
    assert borrowed.effective_n < direct.n / 4


def test_borrowed_evidence_always_forces_verification(ledger):
    seed_outcomes(ledger, AGENT, CAP, FAMILIAR, 200, 199)
    borrowed = ledger.read(AGENT, CAP, ("terraform", "novel"))
    assert borrowed.verification_mandatory, (
        "however good the number looks, a borrowed reading can never authorise "
        "unsupervised execution"
    )
    assert borrowed.evidence_class is not EvidenceClass.PRODUCTION_MEASURED


def test_shrinkage_grows_with_distance():
    assert shrink_factor(0) == 1.0
    assert shrink_factor(1) > shrink_factor(2) > shrink_factor(3)


def test_decay_widens_the_interval_over_time(ledger):
    seed_outcomes(ledger, AGENT, CAP, FAMILIAR, 60, 55)
    before = ledger.read(AGENT, CAP, FAMILIAR).interval.width
    ledger.decay(now_days=180)
    after = ledger.read(AGENT, CAP, FAMILIAR).interval.width
    assert after > before, "stale confidence cannot persist"


def test_decay_eventually_returns_to_the_prior(ledger):
    seed_outcomes(ledger, AGENT, CAP, FAMILIAR, 60, 55)
    ledger.decay(now_days=5_000)
    assert ledger.read(AGENT, CAP, FAMILIAR).mean == pytest.approx(0.5, abs=0.05)


def test_non_stationarity_is_detected_even_though_it_is_not_solved(ledger):
    for i in range(20):
        ledger.record(
            Outcome(
                AGENT, CAP, FAMILIAR, True, OutcomeSource.PRODUCTION,
                verified_by="v", model_fingerprint="model-a" if i < 10 else "model-b",
            )
        )
    warnings = ledger.stationarity_warnings()
    assert warnings and "stationarity" in warnings[0]


# --------------------------------------------------------------------------
# Cross-tenant priors
# --------------------------------------------------------------------------


def test_pooling_requires_consent():
    pool = CrossTenantPriors()
    assert not pool.contribute(
        tenant_id="t1", capability=CAP, bucket=FAMILIAR, success=True, consented=False
    )
    assert pool.prior_for(CAP, FAMILIAR) is None


def test_pooling_requires_enough_tenants_to_be_anonymous():
    pool = CrossTenantPriors(min_tenants=3)
    for tenant in ("t1", "t2"):
        pool.contribute(
            tenant_id=tenant, capability=CAP, bucket=FAMILIAR, success=True, consented=True
        )
    assert pool.prior_for(CAP, FAMILIAR) is None
    pool.contribute(
        tenant_id="t3", capability=CAP, bucket=FAMILIAR, success=True, consented=True
    )
    assert pool.prior_for(CAP, FAMILIAR) is not None


def test_a_pooled_prior_solves_cold_start_without_claiming_local_evidence(ledger):
    pool = CrossTenantPriors(min_tenants=3)
    for i in range(300):
        pool.contribute(
            tenant_id=f"t{i % 40}", capability=CAP, bucket=FAMILIAR,
            success=i % 10 != 0, consented=True,
        )
    reading = pool.seed(ledger, AGENT, CAP, FAMILIAR)
    assert reading is not None
    assert reading.mean > 0.7
    assert reading.evidence_class is not EvidenceClass.PRODUCTION_MEASURED
    assert ledger.get(AGENT, CAP, FAMILIAR).n_production == 0


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def _contract(capability=CAP, bucket=None, **kwargs) -> TaskContract:
    return TaskContract.root(
        goal="do the thing",
        capability=capability,
        authority=AuthorityEnvelope.root(["*"], max_cost_usd=5.0),
        budget_usd=5.0,
        context_bucket=dict(zip(("language", "size", "familiarity"), bucket or FAMILIAR)),
        **kwargs,
    )


def test_a_single_candidate_routes_deterministically(registry, ledger):
    router = Router(registry, ledger, rng=random.Random(0))
    decision = router.route(_contract("web.research"))
    assert decision.stage is Stage.DETERMINISTIC
    assert decision.agent == "agent://res/spec"


def test_several_candidates_go_to_the_bandit(registry, ledger):
    seed_outcomes(ledger, "agent://eng/impl", CAP, FAMILIAR, 40, 38)
    seed_outcomes(ledger, "agent://eng/impl2", CAP, FAMILIAR, 40, 22)
    router = Router(registry, ledger, rng=random.Random(7))
    stages = {router.route(_contract()).stage for _ in range(5)}
    assert stages == {Stage.BANDIT}


def test_the_bandit_prefers_the_better_measured_agent(registry, ledger):
    seed_outcomes(ledger, "agent://eng/impl", CAP, FAMILIAR, 60, 57)
    seed_outcomes(ledger, "agent://eng/impl2", CAP, FAMILIAR, 60, 30)
    router = Router(registry, ledger, rng=random.Random(11))
    picks = [router.route(_contract()).agent for _ in range(40)]
    assert picks.count("agent://eng/impl") > picks.count("agent://eng/impl2") * 3


def test_no_evidence_falls_through_to_the_constrained_router(registry, ledger):
    router = Router(registry, ledger, rng=random.Random(3))
    decision = router.route(_contract())
    assert decision.stage is Stage.LLM
    assert decision.agent in {"agent://eng/impl", "agent://eng/impl2"}
    assert decision.verification_mandatory


def test_the_stage_three_router_cannot_invent_an_agent(registry, ledger):
    router = Router(registry, ledger, llm_router=lambda c, cands: "agent://made/up")
    with pytest.raises(NoRouteAvailable):
        router.route(_contract())


def test_policy_filters_before_statistics_are_consulted(ledger):
    registry = AgentRegistry(
        [
            AgentCard(
                id="agent://eng/impl",
                name="impl",
                capabilities=(CAP,),
                probation=Probation.GRADUATED,
                data_classes=frozenset({DataClass.PUBLIC}),
            )
        ]
    )
    seed_outcomes(ledger, "agent://eng/impl", CAP, FAMILIAR, 100, 99)
    router = Router(registry, ledger)
    with pytest.raises(NoRouteAvailable):
        router.route(_contract(data_class=DataClass.RESTRICTED))
    assert len(router.gap_log) == 1


def test_a_deprecated_agent_is_excluded_not_merely_ranked_last(registry, ledger):
    seed_outcomes(ledger, "agent://eng/impl", CAP, FAMILIAR, 40, 12)
    seed_outcomes(ledger, "agent://eng/impl2", CAP, FAMILIAR, 40, 38)
    router = Router(registry, ledger, rng=random.Random(5))
    for _ in range(10):
        assert router.route(_contract()).agent == "agent://eng/impl2"


def test_every_unroutable_task_is_logged_as_a_gap(registry, ledger):
    router = Router(registry, ledger, gap_log=GapLog())
    for _ in range(12):
        with pytest.raises(NoRouteAvailable):
            router.route(_contract("legal.redline"))
    clusters = router.gap_log.clusters(min_occurrences=10)
    assert clusters and clusters[0].capability == "legal.redline"
    assert clusters[0].count == 12


def test_exploration_routes_to_the_least_measured_candidate(registry, ledger):
    seed_outcomes(ledger, "agent://eng/impl", CAP, FAMILIAR, 80, 76)
    seed_outcomes(ledger, "agent://eng/impl2", CAP, FAMILIAR, 6, 5)
    router = Router(registry, ledger, exploration_rate=1.0, rng=random.Random(2))
    decision = router.route(_contract(), low_stakes=True)
    assert decision.exploration
    assert decision.agent == "agent://eng/impl2"
    assert decision.verification_mandatory


def test_exploration_does_not_touch_high_stakes_traffic(registry, ledger):
    seed_outcomes(ledger, "agent://eng/impl", CAP, FAMILIAR, 80, 76)
    seed_outcomes(ledger, "agent://eng/impl2", CAP, FAMILIAR, 6, 5)
    router = Router(registry, ledger, exploration_rate=1.0, rng=random.Random(2))
    decision = router.route(_contract(), low_stakes=False)
    assert not decision.exploration
