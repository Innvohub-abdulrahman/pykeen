"""The agent factory, the commitment ladder, and the eval-first pipeline."""

from __future__ import annotations

import pytest

from agora.contracts import AuthorityEnvelope
from agora.errors import CharterError
from agora.evolution import (
    Candidate,
    CoreViolation,
    EvalCase,
    EvalSuite,
    SkillSynthesis,
    Stage,
)
from agora.factory import (
    OVERLAP_THRESHOLD,
    AgentFactory,
    Charter,
    Demand,
    Rung,
    Skeptic,
    entropy_sweep,
    which,
)
from agora.fleet import starter_charters, validate_all
from agora.registry import AgentCard, AgentRegistry, Probation, Skill
from agora.router import GapEntry


# --------------------------------------------------------------------------
# The commitment ladder
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "demand,expected",
    [
        (Demand("x", occurrences_30d=2), Rung.P0_NOTHING),
        (Demand("x", occurrences_30d=20, fits_existing_agent="agent://eng/impl"), Rung.P1_SKILL),
        (Demand("x", occurrences_30d=20), Rung.P1_SKILL),
        (Demand("x", occurrences_30d=6, multistep=True, needs_tools=True), Rung.P2_EPHEMERAL),
        (
            Demand("x", occurrences_30d=22, multistep=True, needs_tools=True, needs_continuity=True),
            Rung.P3_PERSISTENT,
        ),
        (
            Demand(
                "x", occurrences_30d=30, multistep=True, needs_tools=True,
                needs_continuity=True, sibling_agents=4,
            ),
            Rung.P4_DEPARTMENT,
        ),
    ],
)
def test_the_ladder_picks_the_smallest_thing_that_works(demand, expected):
    rung, reasons = which(demand)
    assert rung is expected
    assert reasons


def test_the_default_answer_is_build_nothing():
    rung, _ = which(Demand("x", occurrences_30d=1, multistep=True, needs_tools=True))
    assert rung is Rung.P0_NOTHING
    assert "just ask" in rung.description


# --------------------------------------------------------------------------
# Charter review
# --------------------------------------------------------------------------


@pytest.fixture
def existing_registry() -> AgentRegistry:
    return AgentRegistry(
        [
            AgentCard(
                id="agent://eng/implementation",
                name="Implementation",
                capabilities=("code.implement", "repo.refactor", "test.author"),
                probation=Probation.GRADUATED,
                retirement_criteria="below 0.70 over 30 outcomes",
            )
        ]
    )


def _charter(**overrides) -> Charter:
    base = dict(
        id="agent://revenue/deal-desk",
        name="Deal Desk",
        department="revenue",
        job="prices non-standard deals",
        capabilities=("deal.structure",),
        retirement_criteria="fewer than 5 deals a month for two months",
        eval_suite="evals/revenue/deal_v1.yaml",
        authority=AuthorityEnvelope.of(["crm.read", "crm.write"], max_cost_usd=2.0),
    )
    base.update(overrides)
    return Charter(**base)


def _demand() -> Demand:
    return Demand(
        "deal.structure",
        occurrences_30d=22,
        multistep=True,
        needs_tools=True,
        needs_continuity=True,
    )


def test_a_sound_charter_is_approved(existing_registry):
    verdict = AgentFactory(existing_registry).review(_charter(), _demand())
    assert verdict.approved, verdict.blockers


def test_overlap_above_the_threshold_is_refused(existing_registry):
    charter = _charter(
        id="agent://eng/test-writer",
        capabilities=("code.implement", "repo.refactor", "test.author"),
    )
    verdict = AgentFactory(existing_registry).review(charter, _demand())
    assert not verdict.approved
    assert any("overlap" in b for b in verdict.blockers)
    assert verdict.worst_overlap[1] > OVERLAP_THRESHOLD


def test_no_retirement_criteria_blocks(existing_registry):
    verdict = AgentFactory(existing_registry).review(
        _charter(retirement_criteria=""), _demand()
    )
    assert not verdict.approved
    assert "no retirement criteria" in verdict.blockers


def test_no_eval_suite_blocks(existing_registry):
    verdict = AgentFactory(existing_registry).review(_charter(eval_suite=None), _demand())
    assert not verdict.approved
    assert "no eval suite" in verdict.blockers


def test_a_charter_claiming_authority_no_parent_holds_is_refused(existing_registry):
    parent = AuthorityEnvelope.root(["crm.read", "crm.write"], max_cost_usd=25.0)
    charter = _charter(
        authority=AuthorityEnvelope.of(["prod.deploy", "crm.read"], max_cost_usd=2.0)
    )
    verdict = AgentFactory(existing_registry).review(
        charter, _demand(), parent_authority=parent
    )
    assert not verdict.approved
    assert any("prod.deploy" in b for b in verdict.blockers)


def test_weak_demand_blocks_agent_creation(existing_registry):
    verdict = AgentFactory(existing_registry).review(
        _charter(), Demand("deal.structure", occurrences_30d=1)
    )
    assert not verdict.approved
    assert any("no agent should be created" in b for b in verdict.blockers)


def test_committing_needs_a_human(existing_registry):
    factory = AgentFactory(existing_registry)
    verdict = factory.review(_charter(), _demand())
    with pytest.raises(CharterError, match="human decision"):
        factory.commit(_charter(), verdict, human_approved=False)


def test_a_new_agent_enters_at_probation_not_trusted(existing_registry):
    factory = AgentFactory(existing_registry)
    charter = _charter()
    card = factory.commit(
        charter, factory.review(charter, _demand()), human_approved=True
    )
    assert card.probation is Probation.PROBATION
    assert not card.probation is Probation.GRADUATED


def test_rollback_is_one_command(existing_registry):
    factory = AgentFactory(existing_registry)
    charter = _charter()
    factory.commit(charter, factory.review(charter, _demand()), human_approved=True)
    assert len(existing_registry) == 2
    factory.rollback(charter.id)
    assert len(existing_registry) == 1


def test_the_skeptic_argues_for_the_lowest_rung():
    skeptic = Skeptic()
    rung, argument = skeptic.argue(
        Demand("x", occurrences_30d=6, multistep=True, needs_tools=True),
        Rung.P3_PERSISTENT,
    )
    assert rung is Rung.P2_EPHEMERAL
    assert "evidence supports" in argument
    assert skeptic.rejection_rate == 1.0


def test_a_skeptic_that_approves_everything_should_retire():
    skeptic = Skeptic()
    for _ in range(10):
        skeptic.argue(
            Demand("x", occurrences_30d=22, multistep=True, needs_tools=True,
                   needs_continuity=True),
            Rung.P3_PERSISTENT,
        )
    assert skeptic.rejection_rate == 0.0
    assert skeptic.should_retire


# --------------------------------------------------------------------------
# Entropy control
# --------------------------------------------------------------------------


def test_the_sweep_finds_idle_agents_and_dead_skills():
    registry = AgentRegistry(
        [
            AgentCard(
                id="agent://eng/impl",
                name="impl",
                capabilities=("code.implement",),
                retirement_criteria="stated",
                skills=[Skill(name="Old trick", tier="T1")],
            )
        ]
    )
    findings = entropy_sweep(registry, usage_30d={}, skill_usage_30d={})
    kinds = {f.kind for f in findings}
    assert "idle-agent" in kinds
    assert "dead-skill" in kinds


# --------------------------------------------------------------------------
# The starter fleet
# --------------------------------------------------------------------------


def test_all_sixteen_charters_validate():
    results = validate_all()
    assert len(results) == 16
    bad = [r for r in results if not r.ok]
    assert not bad, [(r.charter_id, r.problems) for r in bad]


def test_phase_one_is_local_inference_only():
    """Phase 1 touches personal data, so it never leaves the machine."""
    phase1 = [c for c in starter_charters() if c.phase == 1]
    assert len(phase1) == 3
    assert all(c.local_inference_only for c in phase1)
    assert all(c.cost_hint_usd == 0.0 for c in phase1)


def test_every_charter_names_how_it_dies():
    assert all(c.retirement_criteria.strip() for c in starter_charters())


def test_charters_round_trip_through_json():
    for charter in starter_charters():
        assert Charter.from_dict(charter.to_dict()).capabilities == charter.capabilities


# --------------------------------------------------------------------------
# Skill synthesis
# --------------------------------------------------------------------------


def _suite(cases: int = 24, adversarial: int = 6) -> EvalSuite:
    return EvalSuite(
        id="evals/eng/migration_v1.yaml",
        capability="data.migrate",
        cases=[
            EvalCase(id=f"c{i}", given="x", expect="y", adversarial=i < adversarial)
            for i in range(cases)
        ],
    )


@pytest.fixture
def synthesis() -> SkillSynthesis:
    registry = AgentRegistry(
        [
            AgentCard(
                id="agent://eng/impl",
                name="impl",
                capabilities=("code.implement",),
                probation=Probation.GRADUATED,
            )
        ]
    )
    return SkillSynthesis(registry)


def _gap(count: int = 14) -> GapEntry:
    return GapEntry(capability="data.migrate", bucket=("sql",), reason="unroutable", count=count)


def test_a_one_off_gap_builds_nothing(synthesis):
    candidate = synthesis.consider(_gap(count=3), agent_id="agent://eng/impl")
    assert candidate.stage is Stage.DROPPED
    assert "no one-off skills" in candidate.reason


def test_the_eval_comes_before_the_skill(synthesis):
    """The critical inversion: no skill is written before its eval is approved."""
    candidate = synthesis.consider(_gap(), agent_id="agent://eng/impl")
    with pytest.raises(CoreViolation, match="before its eval is approved"):
        synthesis.synthesise(candidate)


def test_a_thin_eval_suite_is_rejected(synthesis):
    candidate = synthesis.consider(_gap(), agent_id="agent://eng/impl")
    synthesis.draft_eval(candidate, _suite(cases=5, adversarial=1))
    assert candidate.stage is Stage.DROPPED


def test_an_eval_without_adversarial_cases_is_rejected(synthesis):
    candidate = synthesis.consider(_gap(), agent_id="agent://eng/impl")
    synthesis.draft_eval(candidate, _suite(cases=24, adversarial=0))
    assert candidate.stage is Stage.DROPPED


def _to_sandbox(synthesis: SkillSynthesis, tier: str = "T1") -> Candidate:
    candidate = synthesis.consider(_gap(), agent_id="agent://eng/impl", tier=tier)
    synthesis.draft_eval(candidate, _suite())
    synthesis.approve_eval(candidate, by="human", approve=True)
    synthesis.synthesise(candidate)
    return candidate


def test_a_t2_skill_always_sees_a_human(synthesis):
    candidate = _to_sandbox(synthesis, tier="T2")
    with pytest.raises(CoreViolation, match="goes to a human"):
        synthesis.sandbox(candidate, pass_rate=0.99)


def test_a_weak_sandbox_run_drops_the_candidate(synthesis):
    candidate = _to_sandbox(synthesis)
    synthesis.sandbox(candidate, pass_rate=0.60)
    assert candidate.stage is Stage.DROPPED


def test_shadow_requires_fifty_clean_executions(synthesis):
    candidate = _to_sandbox(synthesis)
    synthesis.sandbox(candidate, pass_rate=0.95)
    synthesis.shadow(candidate, executions=20)
    assert candidate.stage is Stage.DROPPED


def test_any_shadow_violation_drops_the_candidate(synthesis):
    candidate = _to_sandbox(synthesis)
    synthesis.sandbox(candidate, pass_rate=0.95)
    synthesis.shadow(candidate, executions=80, violations=1)
    assert candidate.stage is Stage.DROPPED


def test_a_canary_without_lift_rolls_back(synthesis):
    candidate = _to_sandbox(synthesis)
    synthesis.sandbox(candidate, pass_rate=0.95)
    synthesis.shadow(candidate, executions=60)
    synthesis.canary(candidate, measured_lift=-0.02)
    assert candidate.stage is Stage.ROLLED_BACK


def test_the_full_happy_path_promotes(synthesis):
    candidate = _to_sandbox(synthesis)
    synthesis.sandbox(candidate, pass_rate=0.92)
    synthesis.shadow(candidate, executions=60)
    synthesis.canary(candidate, measured_lift=0.08)
    synthesis.promote(candidate, name="Migration writer")
    assert candidate.stage is Stage.PROMOTED
    card = synthesis.registry.get("agent://eng/impl")
    assert card.skill("Migration writer").promoted


def test_promotion_is_reversible_in_one_command(synthesis):
    candidate = _to_sandbox(synthesis)
    synthesis.sandbox(candidate, pass_rate=0.92)
    synthesis.shadow(candidate, executions=60)
    synthesis.canary(candidate, measured_lift=0.08)
    synthesis.promote(candidate, name="Migration writer")
    synthesis.rollback(candidate, reason="regression found in production")
    assert candidate.stage is Stage.ROLLED_BACK
    assert synthesis.registry.get("agent://eng/impl").skill("Migration writer") is None


def test_stages_cannot_be_skipped(synthesis):
    candidate = _to_sandbox(synthesis)
    with pytest.raises(CoreViolation, match="shadow follows sandbox"):
        synthesis.shadow(candidate, executions=60)
