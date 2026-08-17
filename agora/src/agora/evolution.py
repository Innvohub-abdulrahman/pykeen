"""Bounded self-improvement: four loops, and the eval-first promotion pipeline.

The system does not improve because a model got better. It improves because
four feedback loops run at different time constants, each writing into the one
outside it and never inward:

======  =================  =====================================================
Loop    Time constant      What closes it
======  =================  =====================================================
ROUTING seconds            verified outcome -> Beta posterior -> next route
PLANS   hours to days      plan verified -> fragment -> cached plan
SKILLS  days to weeks      gap clustered -> eval written -> skill promoted
AGENTS  weeks to months    recurring demand -> charter -> agent, or archived
======  =================  =====================================================

Routing evidence can promote a plan; a plan cannot rewrite the router. Nothing
here can modify the orchestrator, the verifier, the policy engine or the budget
enforcer. If the system can rewrite its own verifier, every other safety
property becomes unenforceable.

The critical inversion in skill synthesis: **the eval is generated and
approved before the skill exists**. This stops the system writing a skill and
then writing an eval the skill happens to pass, and it gives the reviewer
twenty readable test cases instead of an opaque generated procedure.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence

from .registry import AgentRegistry, Skill
from .router import GapEntry, GapLog


class Loop(str, Enum):
    ROUTING = "routing"
    PLANS = "plans"
    SKILLS = "skills"
    AGENTS = "agents"

    @property
    def time_constant(self) -> str:
        return {
            Loop.ROUTING: "seconds",
            Loop.PLANS: "hours to days",
            Loop.SKILLS: "days to weeks",
            Loop.AGENTS: "weeks to months",
        }[self]

    @property
    def writes_into(self) -> "Loop | None":
        """Each loop writes outward only."""
        return {
            Loop.ROUTING: Loop.PLANS,
            Loop.PLANS: Loop.SKILLS,
            Loop.SKILLS: Loop.AGENTS,
            Loop.AGENTS: None,
        }[self]

    def may_write_to(self, other: "Loop") -> bool:
        """A loop may write outward, never inward."""
        order = [Loop.ROUTING, Loop.PLANS, Loop.SKILLS, Loop.AGENTS]
        return order.index(other) > order.index(self)


#: Components no generated artifact may ever modify. This list is the boundary
#: between L3 (the immutable core) and the mutable periphery.
IMMUTABLE_CORE: frozenset[str] = frozenset(
    {"orchestrator", "verifier", "policy-engine", "budget-enforcer", "audit-log"}
)


class CoreViolation(Exception):
    """Something in the evolution plane tried to write into the core."""


def assert_writes_outward(source: Loop, target: Loop) -> None:
    if not source.may_write_to(target):
        raise CoreViolation(
            f"the {source.value} loop may not write into {target.value}: loops "
            f"write outward only"
        )


def assert_not_core(component: str) -> None:
    if component in IMMUTABLE_CORE:
        raise CoreViolation(
            f"{component!r} is part of the immutable core and is changed only by "
            f"human code review"
        )


# --------------------------------------------------------------------------
# Skill synthesis
# --------------------------------------------------------------------------


class Stage(str, Enum):
    """The promotion pipeline. Every stage can only be entered from the last."""

    GAP = "gap"
    EVAL_DRAFTED = "eval-drafted"
    EVAL_APPROVED = "eval-approved"
    SYNTHESISED = "synthesised"
    HUMAN_REVIEWED = "human-reviewed"
    SANDBOX = "sandbox"
    SHADOW = "shadow"
    CANARY = "canary"
    PROMOTED = "promoted"
    DROPPED = "dropped"
    ROLLED_BACK = "rolled-back"

    @property
    def terminal(self) -> bool:
        return self in (Stage.PROMOTED, Stage.DROPPED, Stage.ROLLED_BACK)


#: A gap must recur before anything is built. No one-off skills.
GAP_OCCURRENCE_THRESHOLD = 10
GAP_WINDOW_DAYS = 14

#: The eval suite must be substantial enough to be worth approving.
MIN_EVAL_CASES = 20
MIN_ADVERSARIAL_CASES = 4

#: Bars at each stage.
SANDBOX_PASS_RATE = 0.85
SHADOW_EXECUTIONS = 50
CANARY_TRAFFIC_SHARE = 0.05


@dataclass
class EvalCase:
    id: str
    given: str
    expect: str
    adversarial: bool = False


@dataclass
class EvalSuite:
    """Written *first*, approved *first*, and never by the thing it tests."""

    id: str
    capability: str
    cases: list[EvalCase] = field(default_factory=list)
    approved_by: str | None = None

    @property
    def adversarial_count(self) -> int:
        return sum(1 for c in self.cases if c.adversarial)

    def problems(self) -> list[str]:
        issues: list[str] = []
        if len(self.cases) < MIN_EVAL_CASES:
            issues.append(
                f"{len(self.cases)} cases; at least {MIN_EVAL_CASES} are required"
            )
        if self.adversarial_count < MIN_ADVERSARIAL_CASES:
            issues.append(
                f"{self.adversarial_count} adversarial cases; at least "
                f"{MIN_ADVERSARIAL_CASES} are required"
            )
        return issues

    @property
    def valid(self) -> bool:
        return not self.problems()


@dataclass
class Candidate:
    """A skill working its way from a measured gap to production."""

    id: str
    capability: str
    agent_id: str
    tier: str = "T1"  # T0 declarative · T1 sandboxed · T2 side-effecting
    stage: Stage = Stage.GAP
    eval_suite: EvalSuite | None = None
    occurrences: int = 0
    sandbox_pass_rate: float = 0.0
    shadow_executions: int = 0
    shadow_violations: int = 0
    canary_lift: float = 0.0
    canary_violations: int = 0
    history: list[str] = field(default_factory=list)
    reason: str = ""

    def log(self, message: str) -> None:
        self.history.append(message)

    @property
    def needs_human_review(self) -> bool:
        return self.tier == "T2"


class SkillSynthesis:
    """The eval-first promotion pipeline.

    Nothing in here is optional, and the order is the point. A skill that
    reaches production has been described by an approved eval, then written,
    then run without secrets, then run against real traffic with its output
    thrown away, then run on 5% of low-stakes traffic with verification forced
    -- and can be rolled back in one command at any point after.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        gap_threshold: int = GAP_OCCURRENCE_THRESHOLD,
    ) -> None:
        self.registry = registry
        self.gap_threshold = gap_threshold
        self.candidates: dict[str, Candidate] = {}
        self.promoted: list[Candidate] = []

    # -- 1 · gap ----------------------------------------------------------

    def consider(self, gap: GapEntry, *, agent_id: str, tier: str = "T1") -> Candidate:
        candidate = Candidate(
            id=f"cand-{uuid.uuid4().hex[:8]}",
            capability=gap.capability,
            agent_id=agent_id,
            tier=tier,
            occurrences=gap.count,
        )
        self.candidates[candidate.id] = candidate
        if gap.count < self.gap_threshold:
            candidate.stage = Stage.DROPPED
            candidate.reason = (
                f"{gap.count} occurrence(s) in {GAP_WINDOW_DAYS} days is below the "
                f"threshold of {self.gap_threshold} — wait, no one-off skills"
            )
            candidate.log(candidate.reason)
            return candidate
        candidate.log(f"gap clustered at {gap.count} occurrences")
        return candidate

    # -- 2 · the eval, first ----------------------------------------------

    def draft_eval(self, candidate: Candidate, suite: EvalSuite) -> Candidate:
        if candidate.stage is not Stage.GAP:
            raise CoreViolation(
                f"cannot draft an eval from stage {candidate.stage.value}"
            )
        problems = suite.problems()
        if problems:
            candidate.stage = Stage.DROPPED
            candidate.reason = "; ".join(problems)
            candidate.log(f"eval rejected: {candidate.reason}")
            return candidate
        candidate.eval_suite = suite
        candidate.stage = Stage.EVAL_DRAFTED
        candidate.log(
            f"eval drafted: {len(suite.cases)} cases "
            f"({suite.adversarial_count} adversarial)"
        )
        return candidate

    def approve_eval(self, candidate: Candidate, *, by: str, approve: bool) -> Candidate:
        """A human approves *the eval*, before the skill exists."""
        if candidate.stage is not Stage.EVAL_DRAFTED:
            raise CoreViolation("the eval must be drafted before it can be approved")
        if not approve:
            candidate.stage = Stage.DROPPED
            candidate.reason = f"eval rejected by {by}"
            candidate.log(candidate.reason)
            return candidate
        assert candidate.eval_suite is not None
        candidate.eval_suite.approved_by = by
        candidate.stage = Stage.EVAL_APPROVED
        candidate.log(f"eval approved by {by} — the skill may now be written")
        return candidate

    # -- 3 · synthesise ---------------------------------------------------

    def synthesise(self, candidate: Candidate) -> Candidate:
        if candidate.stage is not Stage.EVAL_APPROVED:
            raise CoreViolation(
                "no skill is written before its eval is approved — that inversion "
                "is the whole point of this pipeline"
            )
        candidate.stage = Stage.SYNTHESISED
        candidate.log(f"skill synthesised at tier {candidate.tier}")
        return candidate

    def human_review(self, candidate: Candidate, *, by: str, approve: bool) -> Candidate:
        """Mandatory for T2 (side-effecting) skills. Always available for the rest."""
        if not approve:
            candidate.stage = Stage.DROPPED
            candidate.reason = f"skill rejected by {by}"
            candidate.log(candidate.reason)
            return candidate
        candidate.stage = Stage.HUMAN_REVIEWED
        candidate.log(f"skill reviewed by {by}")
        return candidate

    # -- 4 · sandbox, shadow, canary --------------------------------------

    def sandbox(self, candidate: Candidate, *, pass_rate: float) -> Candidate:
        if candidate.needs_human_review and candidate.stage is not Stage.HUMAN_REVIEWED:
            raise CoreViolation(
                "a T2 side-effecting skill goes to a human before it goes anywhere else"
            )
        if candidate.stage not in (Stage.SYNTHESISED, Stage.HUMAN_REVIEWED):
            raise CoreViolation("sandbox follows synthesis")
        candidate.sandbox_pass_rate = pass_rate
        # No secrets, no production credentials, ever.
        if pass_rate < SANDBOX_PASS_RATE:
            candidate.stage = Stage.DROPPED
            candidate.reason = (
                f"sandbox pass rate {pass_rate:.0%} is below {SANDBOX_PASS_RATE:.0%}"
            )
            candidate.log(candidate.reason)
            return candidate
        candidate.stage = Stage.SANDBOX
        candidate.log(f"sandbox passed at {pass_rate:.0%}, no secrets present")
        return candidate

    def shadow(
        self, candidate: Candidate, *, executions: int, violations: int = 0
    ) -> Candidate:
        if candidate.stage is not Stage.SANDBOX:
            raise CoreViolation("shadow follows sandbox")
        candidate.shadow_executions = executions
        candidate.shadow_violations = violations
        if executions < SHADOW_EXECUTIONS or violations > 0:
            candidate.stage = Stage.DROPPED
            candidate.reason = (
                f"shadow: {executions} execution(s), {violations} violation(s); "
                f"{SHADOW_EXECUTIONS} clean are required"
            )
            candidate.log(candidate.reason)
            return candidate
        candidate.stage = Stage.SHADOW
        candidate.log(
            f"shadow clean over {executions} real executions (output discarded)"
        )
        return candidate

    def canary(
        self, candidate: Candidate, *, measured_lift: float, violations: int = 0
    ) -> Candidate:
        if candidate.stage is not Stage.SHADOW:
            raise CoreViolation("canary follows shadow")
        candidate.canary_lift = measured_lift
        candidate.canary_violations = violations
        candidate.stage = Stage.CANARY
        candidate.log(
            f"canary on {CANARY_TRAFFIC_SHARE:.0%} of low-stakes traffic, "
            f"verification mandatory"
        )
        if violations > 0 or measured_lift <= 0:
            return self.rollback(
                candidate,
                reason=(
                    f"canary: lift {measured_lift:+.1%}, {violations} violation(s)"
                ),
            )
        return candidate

    # -- 5 · promote and roll back ----------------------------------------

    def promote(self, candidate: Candidate, *, name: str) -> Candidate:
        if candidate.stage is not Stage.CANARY:
            raise CoreViolation("promotion follows a clean canary")
        card = self.registry.get(candidate.agent_id)
        if card is None:
            raise CoreViolation(f"unknown agent {candidate.agent_id!r}")
        assert candidate.eval_suite is not None
        card.add_skill(
            Skill(
                name=name,
                tier=candidate.tier,
                evals=candidate.eval_suite.id,
                promoted=True,
                description=f"synthesised for {candidate.capability}",
            )
        )
        candidate.stage = Stage.PROMOTED
        candidate.log(f"promoted as {name!r} under {candidate.agent_id}")
        self.promoted.append(candidate)
        return candidate

    def rollback(self, candidate: Candidate, *, reason: str) -> Candidate:
        """Always available, at every stage, in one command.

        A promotion you can undo is a decision; one you cannot is a gamble.
        """
        card = self.registry.get(candidate.agent_id)
        if card is not None:
            card.skills = [
                s for s in card.skills if s.evals != (candidate.eval_suite.id if candidate.eval_suite else None)
            ]
        candidate.stage = Stage.ROLLED_BACK
        candidate.reason = reason
        candidate.log(f"rolled back: {reason}")
        if candidate in self.promoted:
            self.promoted.remove(candidate)
        return candidate


# --------------------------------------------------------------------------
# The four loops, as a description you can assert against
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoopSpec:
    loop: Loop
    time_constant: str
    trigger: str
    effect: str
    reversible_by: str


LOOPS: tuple[LoopSpec, ...] = (
    LoopSpec(
        Loop.ROUTING,
        "seconds",
        "a verified outcome",
        "updates a Beta posterior, which changes the next route",
        "decay, and re-measurement",
    ),
    LoopSpec(
        Loop.PLANS,
        "hours to days",
        "a plan whose run verified",
        "becomes a fragment and a cached plan",
        "evicting the cached plan",
    ),
    LoopSpec(
        Loop.SKILLS,
        "days to weeks",
        "a gap that clustered",
        "an approved eval, then a skill, then sandbox/shadow/canary",
        "one rollback command",
    ),
    LoopSpec(
        Loop.AGENTS,
        "weeks to months",
        "recurring demand no existing agent covers",
        "a charter, a review, and an agent — or an archive proposal",
        "deregistering the agent",
    ),
)


def loop_writes_are_outward_only() -> bool:
    """Property check: no loop may write into a faster one."""
    return all(
        not spec.loop.may_write_to(other.loop)
        for spec in LOOPS
        for other in LOOPS
        if LOOPS.index(other) < LOOPS.index(spec)
    )


__all__ = [
    "CANARY_TRAFFIC_SHARE",
    "Candidate",
    "CoreViolation",
    "EvalCase",
    "EvalSuite",
    "GAP_OCCURRENCE_THRESHOLD",
    "IMMUTABLE_CORE",
    "LOOPS",
    "Loop",
    "LoopSpec",
    "SHADOW_EXECUTIONS",
    "SkillSynthesis",
    "Stage",
    "assert_not_core",
    "assert_writes_outward",
    "loop_writes_are_outward_only",
]
