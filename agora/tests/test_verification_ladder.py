"""The ladder, assurance grading, and the published capability index."""

from __future__ import annotations

import pytest

from agora.artifacts import Artifact, Source, Trust
from agora.contracts import AcceptanceCriterion, CheckKind, TaskContract
from agora.verification.ladder import (
    RELIABILITY_GAP_THRESHOLD,
    DomainPack,
    Grade,
    Level,
    Position,
    RubricGrader,
    VerificationLadder,
    adjudicate,
    grade_capability,
)
from agora.verification.packs import CAPABILITY_INDEX, PackRegistry, default_packs
from agora.verification.universal import CheckResult, Finding, Severity


def _ok(name: str):
    return lambda artifact, contract: CheckResult(check=name)


def _fail(name: str):
    def check(artifact, contract):
        result = CheckResult(check=name)
        result.findings.append(
            Finding(check=name, severity=Severity.BLOCK, message="nope")
        )
        return result

    return check


def _artifact(content: str = "# Summary\n\nAll good.\n", **kwargs) -> Artifact:
    return Artifact(content=content, **kwargs)


# --------------------------------------------------------------------------
# Grading is derived, never asserted
# --------------------------------------------------------------------------


def test_grade_is_derived_from_the_checks_a_pack_supplies():
    assert grade_capability(DomainPack("p").add(Level.L0, _ok("x"))) is Grade.A
    assert grade_capability(DomainPack("p").add(Level.L2, _ok("x"))) is Grade.B
    assert grade_capability(DomainPack("p").add(Level.L1, _ok("x"))) is Grade.B
    assert grade_capability(DomainPack("p").add(Level.L3, _ok("x"))) is Grade.C
    assert grade_capability(None) is Grade.D


def test_grade_d_still_gets_the_floor():
    """Grade D is not "unsupported". It is governance without autonomy."""
    assert not Grade.D.may_run_unsupervised
    assert "full human review" in Grade.D.verification_policy


def test_only_a_and_b_run_unsupervised():
    assert Grade.A.may_run_unsupervised and Grade.B.may_run_unsupervised
    assert not Grade.C.may_run_unsupervised and not Grade.D.may_run_unsupervised


@pytest.mark.parametrize("capability,expected", sorted(CAPABILITY_INDEX.items()))
def test_every_capability_grades_exactly_as_published(capability, expected):
    """The published index is a claim. A pack that quietly loses a check must
    fail a test rather than quietly downgrade a customer's autonomy."""
    assert PackRegistry().grade_for(capability).value == expected


def test_strategy_has_no_pack_and_does_not_pretend_to():
    registry = PackRegistry()
    assert registry.for_capability("strategy.position") is None
    assert registry.grade_for("strategy.position") is Grade.D


def test_one_pack_can_carry_different_grades_per_capability():
    """Reconciliation ties out to the cent; a forecast has nothing to tie to."""
    registry = PackRegistry()
    assert registry.grade_for("finance.reconcile") is Grade.A
    assert registry.grade_for("finance.forecast") is Grade.B
    assert registry.for_capability("finance.reconcile").name == "finance"
    assert registry.for_capability("finance.forecast").name == "finance"


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


def test_the_floor_gates_everything_above_it():
    """A rubric is never called for an artifact that failed a free check."""
    pack = DomainPack("p").add(Level.L3, _ok("l3"))
    rubric = RubricGrader(score_fn=lambda a, c: 1.0)
    ladder = VerificationLadder(pack, rubric=rubric)
    bad = _artifact("Adoption grew 47% year on year.")
    report = ladder.verify(bad, TaskContract(goal="g", capability="marketing.campaign"))
    assert not report.passed
    assert report.failed_at is Level.UNIVERSAL
    assert report.model_calls == 0, "the expensive rung must not have run"
    assert report.cost_usd == 0.0


def test_a_clean_artifact_climbs_the_whole_ladder():
    pack = DomainPack("p").add(Level.L0, _ok("l0")).add(Level.L1, _ok("l1"))
    ladder = VerificationLadder(pack)
    report = ladder.verify(_artifact(), TaskContract(goal="g", capability="c"))
    assert report.passed
    assert report.reached is Level.L3
    assert report.grade is Grade.A


def test_a_failing_domain_check_stops_the_climb():
    pack = DomainPack("p").add(Level.L0, _ok("l0")).add(Level.L1, _fail("l1"))
    report = VerificationLadder(pack).verify(
        _artifact(), TaskContract(goal="g", capability="c")
    )
    assert not report.passed
    assert report.failed_at is Level.L1


def test_deterministic_acceptance_criteria_run_at_l0():
    criterion = AcceptanceCriterion(
        id="has-summary",
        kind=CheckKind.PREDICATE,
        description="mentions a summary",
        predicate=lambda artifact: "Summary" in artifact.content,
    )
    contract = TaskContract(goal="g", capability="c", acceptance=(criterion,))
    assert VerificationLadder().verify(_artifact(), contract).passed
    assert not VerificationLadder().verify(_artifact("nothing here"), contract).passed


def test_the_rubric_is_blind_to_the_producer():
    seen = {}

    def score(artifact, criteria):
        seen["reasoning"] = artifact.reasoning
        seen["producer"] = artifact.producer
        return 1.0

    contract = TaskContract(
        goal="g",
        capability="c",
        acceptance=(
            AcceptanceCriterion(
                id="r", kind=CheckKind.RUBRIC, description="graded", threshold=0.7
            ),
        ),
    )
    artifact = _artifact(reasoning="I was not sure so I guessed", producer="agent://x")
    VerificationLadder(rubric=RubricGrader(score_fn=score)).verify(artifact, contract)
    assert seen["reasoning"] is None
    assert seen["producer"] == "<blinded>"


def test_the_rubric_is_the_only_rung_that_costs_money():
    contract = TaskContract(
        goal="g",
        capability="c",
        acceptance=(
            AcceptanceCriterion(
                id="r", kind=CheckKind.RUBRIC, description="graded", threshold=0.6
            ),
        ),
    )
    report = VerificationLadder(rubric=RubricGrader()).verify(_artifact(), contract)
    assert report.model_calls == 1
    assert report.cost_usd > 0
    assert report.floor.cost_usd == 0.0


# --------------------------------------------------------------------------
# Contest and adjudication
# --------------------------------------------------------------------------


def test_a_decidable_test_settles_it():
    verdict = adjudicate(
        Position("a", "x is true", 0.5),
        Position("b", "x is false", 0.9),
        decidable_test=lambda: "a",
    )
    assert verdict.winner == "a"
    assert verdict.method == "decidable-test"
    assert not verdict.escalated


def test_a_wide_reliability_gap_decides_when_no_test_exists():
    verdict = adjudicate(Position("a", "x", 0.90), Position("b", "y", 0.60))
    assert verdict.winner == "a"
    assert verdict.method == "reliability-gap"


def test_a_narrow_gap_escalates_to_a_human():
    verdict = adjudicate(Position("a", "x", 0.82), Position("b", "y", 0.78))
    assert verdict.winner is None
    assert verdict.escalated
    assert 0.82 - 0.78 < RELIABILITY_GAP_THRESHOLD


# --------------------------------------------------------------------------
# Domain packs
# --------------------------------------------------------------------------


def test_engineering_pack_requires_a_clean_test_report():
    pack = default_packs()["engineering"]
    contract = TaskContract(goal="g", capability="code.implement")
    failing = _artifact(metadata={"test_report": {"exit_code": 1, "failed": 2}})
    assert not VerificationLadder(pack).verify(failing, contract).passed
    passing = _artifact(
        metadata={"test_report": {"exit_code": 0, "failed": 0, "collected": 9}}
    )
    assert VerificationLadder(pack).verify(passing, contract).passed


def test_calendar_pack_catches_an_infeasible_schedule():
    pack = default_packs()["calendar"]
    artifact = _artifact(
        metadata={
            "schedule": [
                {"title": "A", "start": 9.0, "end": 10.0, "travel_min": 30},
                {"title": "B", "start": 10.25, "end": 11.0},
            ]
        }
    )
    report = VerificationLadder(pack).verify(
        artifact, TaskContract(goal="g", capability="schedule.resolve")
    )
    assert not report.passed
    assert "travel time" in report.blocking[0].message


def test_finance_pack_catches_a_ledger_that_does_not_tie_out():
    pack = default_packs()["finance"]
    artifact = _artifact(
        metadata={
            "ledger": {"lines": [{"amount": 100.0}, {"amount": 50.0}], "total": 160.0}
        }
    )
    report = VerificationLadder(pack).verify(
        artifact, TaskContract(goal="g", capability="finance.reconcile")
    )
    assert not report.passed


def test_l2_catches_what_the_floor_structurally_cannot():
    """A figure cited to a source that does not contain it.

    The universal floor asks whether a figure traces to *any* supplied source.
    Only recomputation asks whether it traces to *the source it claims*.
    """
    artifact = _artifact(
        "Renewal rate is 91% [S1].",
        sources=(Source(id="S1", text="renewal rate 88%", trust=Trust.PRIMARY),),
    )
    contract = TaskContract(goal="g", capability="web.research")

    from agora.verification.universal import run_universal_floor

    assert run_universal_floor(artifact, contract).passed, (
        "the floor sees 91 present in the sources and lets it through"
    )
    report = VerificationLadder(default_packs()["research"]).verify(artifact, contract)
    assert not report.passed
    assert report.failed_at is Level.L2


def test_planning_pack_catches_a_dependency_cycle():
    pack = default_packs()["planning"]
    artifact = _artifact(
        metadata={
            "plan": {
                "budget_usd": 10.0,
                "nodes": [
                    {"id": "a", "depends_on": ["b"], "acceptance": ["x"], "budget_usd": 5},
                    {"id": "b", "depends_on": ["a"], "acceptance": ["x"], "budget_usd": 5},
                ],
            }
        }
    )
    report = VerificationLadder(pack).verify(
        artifact, TaskContract(goal="g", capability="plan.decompose")
    )
    assert not report.passed
    assert "cycle" in report.blocking[0].message
