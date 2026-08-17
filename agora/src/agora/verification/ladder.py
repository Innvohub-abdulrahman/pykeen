"""The verification ladder and assurance grading.

Domains do not differ in how you orchestrate them. They differ in how cheaply
you can tell whether the answer is right. That single fact is what makes the
framework general -- and what limits it.

::

    artifact -> UNIVERSAL FLOOR   free, always, every domain
             -> L0 executable      tests, compiler, solver, identities
             -> L1 constraints     schema, policy, invariants
             -> L2 recomputation   independent path, source tie-out
             -> L3 rubric          model-graded, the only tier that costs money
             -> L5 escalation      a human

A failure does not end the story. If the disagreement is decidable, run the
test and let evidence settle it. If it is not, a reliability gap over 15 points
lets the more reliable position stand. Failing both, it escalates to a person.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..artifacts import Artifact
from ..contracts import AcceptanceCriterion, CheckKind, TaskContract
from .universal import (
    CheckResult,
    Finding,
    FloorReport,
    Severity,
    run_universal_floor,
)


class Level(str, Enum):
    """Rungs of the ladder, cheapest first."""

    UNIVERSAL = "universal"
    L0 = "L0"   # executable: tests, compiler, solver, identities
    L1 = "L1"   # domain constraints: schema, policy, invariants
    L2 = "L2"   # recomputation: independent path, source tie-out
    L3 = "L3"   # rubric: model-graded
    L5 = "L5"   # human

    @property
    def rank(self) -> int:
        return _LEVEL_ORDER.index(self)

    @property
    def costs_money(self) -> bool:
        return self in (Level.L2, Level.L3, Level.L5)


_LEVEL_ORDER = [Level.UNIVERSAL, Level.L0, Level.L1, Level.L2, Level.L3, Level.L5]


class Grade(str, Enum):
    """What a capability has earned the right to do, given its checks.

    Rather than claim uniform coverage, every capability carries a grade
    derived from the checks its domain pack can actually supply. Nothing is
    refused; everything is labelled.
    """

    A = "A"  # verified   -- deterministic domain checks
    B = "B"  # checked    -- recomputation or strong constraints + floor
    C = "C"  # guarded    -- floor + rubric
    D = "D"  # observed   -- floor only

    @property
    def label(self) -> str:
        return {
            Grade.A: "Verified",
            Grade.B: "Checked",
            Grade.C: "Guarded",
            Grade.D: "Observed",
        }[self]

    @property
    def may_run_unsupervised(self) -> bool:
        return self in (Grade.A, Grade.B)

    @property
    def verification_policy(self) -> str:
        return {
            Grade.A: "autonomous within budget, sampled verification",
            Grade.B: "autonomous, every output verified by a second path",
            Grade.C: "drafts only, human approves before anything leaves",
            Grade.D: "drafts only, full human review",
        }[self]

    @property
    def rank(self) -> int:
        return [Grade.A, Grade.B, Grade.C, Grade.D].index(self)


# --------------------------------------------------------------------------
# Domain packs
# --------------------------------------------------------------------------

#: A domain check takes (artifact, contract) and returns a CheckResult.
DomainCheck = Callable[[Artifact, "TaskContract | None"], CheckResult]


@dataclass
class DomainPack:
    """Domain-specific checks layered above the universal floor.

    A pack declares which rungs it can supply for which capabilities. That
    declaration is what produces the assurance grade -- the grade is derived
    from the checks that exist, never asserted by hand.
    """

    name: str
    capabilities: tuple[str, ...] = ()
    checks: dict[Level, list[DomainCheck]] = field(default_factory=dict)
    description: str = ""
    #: Per-capability overrides. A pack often supplies deep checks for some of
    #: its capabilities and only shallow ones for others -- reconciliation ties
    #: out against source rows, but a cash forecast has no ground truth to tie
    #: to. Grading the whole pack at its best capability would be exactly the
    #: overclaiming this system exists to avoid.
    capability_levels: dict[str, tuple[Level, ...]] = field(default_factory=dict)

    def add(self, level: Level | str, check: DomainCheck) -> "DomainPack":
        self.checks.setdefault(Level(level), []).append(check)
        return self

    def levels_supplied(self, capability: str | None = None) -> set[Level]:
        supplied = {lvl for lvl, fns in self.checks.items() if fns}
        if capability and capability in self.capability_levels:
            return supplied & set(self.capability_levels[capability])
        return supplied

    def run(
        self, level: Level, artifact: Artifact, contract: TaskContract | None
    ) -> list[CheckResult]:
        return [fn(artifact, contract) for fn in self.checks.get(level, [])]

    def covers(self, capability: str) -> bool:
        if not self.capabilities:
            return True
        return capability in self.capabilities or any(
            capability.startswith(c.rstrip("*")) for c in self.capabilities if "*" in c
        )


def grade_capability(
    pack: DomainPack | None,
    *,
    has_rubric: bool = False,
    capability: str | None = None,
) -> Grade:
    """Derive an assurance grade from the checks a pack can actually supply."""
    levels = pack.levels_supplied(capability) if pack else set()
    if Level.L0 in levels:
        return Grade.A
    if Level.L2 in levels or Level.L1 in levels:
        return Grade.B
    if has_rubric or Level.L3 in levels:
        return Grade.C
    return Grade.D


# --------------------------------------------------------------------------
# Ladder results
# --------------------------------------------------------------------------


@dataclass
class LadderReport:
    artifact_id: str
    floor: FloorReport
    results: list[CheckResult] = field(default_factory=list)
    reached: Level = Level.UNIVERSAL
    failed_at: Level | None = None
    escalated: bool = False
    escalation_reason: str = ""
    grade: Grade = Grade.D

    @property
    def findings(self) -> list[Finding]:
        return [*self.floor.findings, *[f for r in self.results for f in r.findings]]

    @property
    def blocking(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.BLOCK]

    @property
    def passed(self) -> bool:
        return self.failed_at is None and not self.escalated

    @property
    def model_calls(self) -> int:
        return self.floor.model_calls + sum(r.model_calls for r in self.results)

    @property
    def cost_usd(self) -> float:
        return self.floor.cost_usd + sum(r.cost_usd for r in self.results)

    def summary(self) -> str:
        state = (
            "VERIFIED"
            if self.passed
            else ("ESCALATED" if self.escalated else f"FAILED at {self.failed_at.value}")
        )
        return (
            f"{state} · grade {self.grade.value} · reached {self.reached.value} · "
            f"{len(self.blocking)} blocking · ${self.cost_usd:.4f} · "
            f"{self.model_calls} model call(s)"
        )


# --------------------------------------------------------------------------
# Rubric grading
# --------------------------------------------------------------------------


class RubricGrader:
    """A model-graded rubric. The only rung of the ladder that costs money.

    Deliberately blind: it receives the artifact and the criteria, never the
    producer's reasoning. Supply ``score_fn`` to plug in a real model; the
    default is a deterministic stand-in so the ladder is testable end to end
    without a network call.
    """

    def __init__(
        self,
        score_fn: Callable[[Artifact, Sequence[AcceptanceCriterion]], float] | None = None,
        *,
        cost_usd: float = 0.012,
        family: str = "verifier-family",
    ) -> None:
        self.score_fn = score_fn
        self.cost_usd = cost_usd
        self.family = family

    def __call__(
        self, artifact: Artifact, criteria: Sequence[AcceptanceCriterion]
    ) -> CheckResult:
        blind = artifact.blind_view()
        result = CheckResult(check="rubric", model_calls=1, cost_usd=self.cost_usd)
        threshold = max(
            (c.threshold for c in criteria if c.threshold is not None), default=0.75
        )
        score = (
            self.score_fn(blind, criteria)
            if self.score_fn is not None
            else _heuristic_rubric(blind, criteria)
        )
        result.note = f"rubric {score:.2f} vs threshold {threshold:.2f}"
        if score < threshold:
            result.findings.append(
                Finding(
                    check="rubric",
                    severity=Severity.BLOCK,
                    message=f"rubric score {score:.2f} below threshold {threshold:.2f}",
                    evidence=result.note,
                )
            )
        return result


def _heuristic_rubric(
    artifact: Artifact, criteria: Sequence[AcceptanceCriterion]
) -> float:
    """A deterministic stand-in for a graded rubric.

    Scores structure and sourcing, which are the two things a rubric agrees
    with a deterministic checker about. It is not a quality judgement and does
    not pretend to be one.
    """
    score = 0.5
    if artifact.sources:
        score += 0.2
    if artifact.sections:
        score += 0.15
    if len(artifact.content.split()) > 40:
        score += 0.1
    if artifact.confidence >= 0.8:
        score += 0.05
    return min(1.0, score)


# --------------------------------------------------------------------------
# Executable criteria (L0)
# --------------------------------------------------------------------------


def run_acceptance_criteria(
    artifact: Artifact,
    criteria: Sequence[AcceptanceCriterion],
    *,
    runner: Callable[[AcceptanceCriterion, Artifact], bool] | None = None,
) -> CheckResult:
    """Execute the decidable acceptance criteria attached to the contract."""
    result = CheckResult(check="acceptance")
    for criterion in criteria:
        if not criterion.deterministic:
            continue
        ok: bool
        if criterion.kind is CheckKind.PREDICATE and criterion.predicate is not None:
            ok = bool(criterion.predicate(artifact))
        elif runner is not None:
            ok = bool(runner(criterion, artifact))
        else:
            result.note = "no runner supplied; command/test criteria not executed"
            continue
        if not ok:
            result.findings.append(
                Finding(
                    check="acceptance",
                    severity=Severity.BLOCK,
                    message=f"acceptance criterion {criterion.id!r} failed",
                    evidence=criterion.description,
                    location=criterion.kind.value,
                )
            )
    return result


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------


class VerificationLadder:
    """Runs the floor, then each rung the domain pack can supply.

    Cheap rungs gate expensive ones: the rubric is never called for an
    artifact that already failed a free deterministic check, which is most of
    why verification is affordable at volume.
    """

    def __init__(
        self,
        pack: DomainPack | None = None,
        *,
        rubric: RubricGrader | None = None,
        criterion_runner: Callable[[AcceptanceCriterion, Artifact], bool] | None = None,
    ) -> None:
        self.pack = pack
        self.rubric = rubric
        self.criterion_runner = criterion_runner

    def verify(
        self,
        artifact: Artifact,
        contract: TaskContract | None = None,
        *,
        run_rubric: bool | None = None,
    ) -> LadderReport:
        criteria = tuple(contract.acceptance) if contract else ()
        wants_rubric = (
            run_rubric
            if run_rubric is not None
            else any(c.kind is CheckKind.RUBRIC for c in criteria)
        )
        report = LadderReport(
            artifact_id=artifact.id,
            floor=run_universal_floor(artifact, contract),
            grade=grade_capability(self.pack, has_rubric=wants_rubric),
        )

        if not report.floor.passed:
            report.failed_at = Level.UNIVERSAL
            return report

        subject = artifact.sanitized()

        # L0 -- executable
        l0 = [run_acceptance_criteria(subject, criteria, runner=self.criterion_runner)]
        if self.pack:
            l0 += self.pack.run(Level.L0, subject, contract)
        if not self._advance(report, Level.L0, l0):
            return report

        # L1 -- domain constraints
        if not self._advance(
            report, Level.L1, self.pack.run(Level.L1, subject, contract) if self.pack else []
        ):
            return report

        # L2 -- recomputation
        if not self._advance(
            report, Level.L2, self.pack.run(Level.L2, subject, contract) if self.pack else []
        ):
            return report

        # L3 -- rubric, the only rung that costs money
        l3: list[CheckResult] = []
        if self.pack:
            l3 += self.pack.run(Level.L3, subject, contract)
        if wants_rubric and self.rubric is not None:
            l3.append(self.rubric(subject, criteria))
        if not self._advance(report, Level.L3, l3):
            return report

        return report

    def _advance(
        self, report: LadderReport, level: Level, results: Sequence[CheckResult]
    ) -> bool:
        report.results.extend(results)
        report.reached = level
        if any(not r.passed for r in results):
            report.failed_at = level
            return False
        return True


# --------------------------------------------------------------------------
# Contest and adjudication
# --------------------------------------------------------------------------


@dataclass
class Position:
    """One side of a disagreement about an artifact."""

    holder: str
    claim: str
    reliability: float
    artifact: Artifact | None = None


@dataclass
class Adjudication:
    winner: str | None
    method: str
    detail: str
    escalated: bool = False


#: Reliability difference, in points, that lets one position stand over another
#: without running anything. Below this the gap is noise, and noise should not
#: settle an argument.
RELIABILITY_GAP_THRESHOLD = 0.15


def adjudicate(
    a: Position,
    b: Position,
    *,
    decidable_test: Callable[[], str | None] | None = None,
    gap_threshold: float = RELIABILITY_GAP_THRESHOLD,
) -> Adjudication:
    """Resolve a disagreement: run a test, else reliability gap, else a human.

    Evidence beats argument. Only when there is no test to run does measured
    reliability get a vote, and only when the gap is real does it decide.
    """
    if decidable_test is not None:
        winner = decidable_test()
        if winner is not None:
            return Adjudication(
                winner=winner,
                method="decidable-test",
                detail="a test was available and it settled the question",
            )

    gap = abs(a.reliability - b.reliability)
    if gap > gap_threshold:
        winner = a.holder if a.reliability > b.reliability else b.holder
        return Adjudication(
            winner=winner,
            method="reliability-gap",
            detail=f"reliability gap {gap:.2f} exceeds {gap_threshold:.2f}",
        )

    return Adjudication(
        winner=None,
        method="escalation",
        detail=(
            f"no decidable test and reliability gap {gap:.2f} is within noise; "
            f"a human decides"
        ),
        escalated=True,
    )


__all__ = [
    "Level",
    "Grade",
    "DomainPack",
    "DomainCheck",
    "grade_capability",
    "LadderReport",
    "VerificationLadder",
    "RubricGrader",
    "run_acceptance_criteria",
    "Position",
    "Adjudication",
    "adjudicate",
    "RELIABILITY_GAP_THRESHOLD",
]
