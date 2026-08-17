"""Planning: recall, compose, synthesise -- in that order of preference.

Most planning is not planning. It is recognising a task shape you have solved
before, which costs nothing. The order here is a cost decision:

1. **Recall.** The task shape matches a validated cached plan. Zero cost.
2. **Compose.** No whole-plan match, but previously *verified* fragments fit.
3. **Synthesise.** Genuinely novel. A model plans, and the output becomes a
   candidate fragment only if the run that used it verified.

The last clause is the loop that matters: plans earn their way into the cache
by working, not by looking reasonable when they were written.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contracts import AcceptanceCriterion, CheckKind, Tier


class PlanSource(str, Enum):
    RECALL = "recall"
    COMPOSE = "compose"
    SYNTHESISE = "synthesise"

    @property
    def cost_usd(self) -> float:
        return {
            PlanSource.RECALL: 0.0,
            PlanSource.COMPOSE: 0.0,
            PlanSource.SYNTHESISE: 0.08,
        }[self]


@dataclass
class Step:
    """One node of a plan: a capability, a goal, and how it will be checked."""

    id: str
    capability: str
    goal: str
    budget_usd: float = 0.0
    depends_on: tuple[str, ...] = ()
    acceptance: tuple[AcceptanceCriterion, ...] = ()
    context_bucket: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.depends_on = tuple(self.depends_on)
        self.acceptance = tuple(self.acceptance)


@dataclass
class Plan:
    """A DAG of steps with a total budget."""

    id: str
    goal: str
    steps: tuple[Step, ...]
    budget_usd: float
    tier: Tier = Tier.PROJECT
    source: PlanSource = PlanSource.SYNTHESISE
    shape: str = ""
    verified_runs: int = 0

    def __post_init__(self) -> None:
        self.steps = tuple(self.steps)
        self.tier = Tier(self.tier)

    @property
    def allocated_usd(self) -> float:
        return sum(s.budget_usd for s in self.steps)

    def to_metadata(self) -> dict[str, Any]:
        """The shape the planning domain pack checks invariants against."""
        return {
            "budget_usd": self.budget_usd,
            "nodes": [
                {
                    "id": s.id,
                    "goal": s.goal,
                    "capability": s.capability,
                    "budget_usd": s.budget_usd,
                    "depends_on": list(s.depends_on),
                    "acceptance": [c.id for c in s.acceptance],
                }
                for s in self.steps
            ],
        }

    def order(self) -> list[Step]:
        """Topological order. Raises if the plan is not a DAG."""
        by_id = {s.id: s for s in self.steps}
        pending = {s.id: set(s.depends_on) & set(by_id) for s in self.steps}
        out: list[Step] = []
        while pending:
            ready = sorted(k for k, deps in pending.items() if not deps)
            if not ready:
                raise ValueError(f"plan {self.id} has a dependency cycle")
            for k in ready:
                out.append(by_id[k])
                pending.pop(k)
            for deps in pending.values():
                deps.difference_update(ready)
        return out


# --------------------------------------------------------------------------
# Task shape
# --------------------------------------------------------------------------

_SHAPE_NOISE = re.compile(r"[^a-z0-9 ]+")
_SHAPE_STOP = frozenset(
    "the a an of to for in on and or with our my your this that please can".split()
)


def task_shape(goal: str, capability: str, bucket: Mapping[str, str] | None = None) -> str:
    """A stable fingerprint of "what kind of task is this".

    Deliberately lossy. Two requests to add ZATCA validation to different
    services should hash the same, or the plan cache never hits.
    """
    words = _SHAPE_NOISE.sub(" ", goal.lower()).split()
    salient = sorted({w for w in words if w not in _SHAPE_STOP and len(w) > 3})
    parts = [capability, *salient]
    if bucket:
        parts += [f"{k}={v}" for k, v in sorted(bucket.items())]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Caches
# --------------------------------------------------------------------------


class PlanCache:
    """Validated plans, keyed by task shape. A hit costs nothing."""

    def __init__(self) -> None:
        self._by_shape: dict[str, Plan] = {}
        self.hits = 0
        self.misses = 0

    def recall(self, shape: str) -> Plan | None:
        plan = self._by_shape.get(shape)
        if plan is None:
            self.misses += 1
            return None
        self.hits += 1
        return plan

    def remember(self, plan: Plan) -> Plan:
        """Cache a plan that has actually worked."""
        if plan.verified_runs < 1:
            raise ValueError(
                "only a plan whose run verified may enter the cache; a plan that "
                "looked reasonable is not evidence"
            )
        self._by_shape[plan.shape] = plan
        return plan

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def __len__(self) -> int:
        return len(self._by_shape)


class FragmentLibrary:
    """Previously verified plan fragments, indexed by capability."""

    def __init__(self) -> None:
        self._by_capability: dict[str, list[Step]] = {}

    def add(self, step: Step) -> None:
        self._by_capability.setdefault(step.capability, []).append(step)

    def fragments_for(self, capabilities: Sequence[str]) -> list[Step]:
        return [
            step
            for capability in capabilities
            for step in self._by_capability.get(capability, [])[:1]
        ]

    def covers(self, capabilities: Sequence[str]) -> bool:
        return all(c in self._by_capability for c in capabilities)

    def __len__(self) -> int:
        return sum(len(v) for v in self._by_capability.values())


# --------------------------------------------------------------------------
# The planner
# --------------------------------------------------------------------------

Synthesiser = Callable[[str, str, float], Sequence[Step]]


def _default_synthesiser(goal: str, capability: str, budget_usd: float) -> list[Step]:
    """A deterministic stand-in for LLM planning.

    Produces the shape a plan for an unfamiliar task tends to have -- research,
    do, check -- so the system is runnable end to end without a model. Swap it
    for a real planner via ``Planner(synthesiser=...)``.
    """
    share = budget_usd / 3.0 if budget_usd else 0.0
    return [
        Step(
            id="research",
            capability="web.research",
            goal=f"gather primary sources for: {goal}",
            budget_usd=round(share * 0.32, 4),
            acceptance=(
                AcceptanceCriterion(
                    id="sources-present",
                    kind=CheckKind.PREDICATE,
                    description="at least one primary source is attached",
                    predicate=lambda artifact: bool(getattr(artifact, "sources", ())),
                ),
            ),
        ),
        Step(
            id="do",
            capability=capability,
            goal=goal,
            budget_usd=round(share * 1.5, 4),
            depends_on=("research",),
            acceptance=(
                AcceptanceCriterion(
                    id="non-empty",
                    kind=CheckKind.PREDICATE,
                    description="the artifact has content",
                    predicate=lambda artifact: bool(
                        getattr(artifact, "content", "").strip()
                    ),
                ),
            ),
        ),
        Step(
            id="review",
            capability="code.review",
            goal=f"review the result of: {goal}",
            budget_usd=round(share * 0.5, 4),
            depends_on=("do",),
            acceptance=(
                AcceptanceCriterion(
                    id="reviewed",
                    kind=CheckKind.PREDICATE,
                    description="a review verdict is attached",
                    predicate=lambda artifact: "verdict"
                    in getattr(artifact, "metadata", {}),
                ),
            ),
        ),
    ]


@dataclass
class PlanChoice:
    plan: Plan
    source: PlanSource
    cost_usd: float
    rationale: str


class Planner:
    """Recall, then compose, then synthesise."""

    def __init__(
        self,
        *,
        cache: PlanCache | None = None,
        fragments: FragmentLibrary | None = None,
        synthesiser: Synthesiser | None = None,
    ) -> None:
        self.cache = cache or PlanCache()
        self.fragments = fragments or FragmentLibrary()
        self.synthesiser = synthesiser or _default_synthesiser

    def plan(
        self,
        goal: str,
        capability: str,
        *,
        budget_usd: float,
        tier: Tier = Tier.PROJECT,
        bucket: Mapping[str, str] | None = None,
        required_capabilities: Sequence[str] = (),
    ) -> PlanChoice:
        shape = task_shape(goal, capability, bucket)

        cached = self.cache.recall(shape)
        if cached is not None:
            return PlanChoice(
                plan=cached,
                source=PlanSource.RECALL,
                cost_usd=0.0,
                rationale="plan cache hit — recalled, not synthesised",
            )

        if required_capabilities and self.fragments.covers(required_capabilities):
            steps = self.fragments.fragments_for(required_capabilities)
            plan = Plan(
                id=f"plan-{uuid.uuid4().hex[:10]}",
                goal=goal,
                steps=tuple(steps),
                budget_usd=budget_usd,
                tier=tier,
                source=PlanSource.COMPOSE,
                shape=shape,
            )
            return PlanChoice(
                plan=plan,
                source=PlanSource.COMPOSE,
                cost_usd=0.0,
                rationale="assembled from previously verified fragments",
            )

        steps = list(self.synthesiser(goal, capability, budget_usd))
        plan = Plan(
            id=f"plan-{uuid.uuid4().hex[:10]}",
            goal=goal,
            steps=tuple(steps),
            budget_usd=budget_usd,
            tier=tier,
            source=PlanSource.SYNTHESISE,
            shape=shape,
        )
        return PlanChoice(
            plan=plan,
            source=PlanSource.SYNTHESISE,
            cost_usd=PlanSource.SYNTHESISE.cost_usd,
            rationale="genuinely novel task shape — synthesised",
        )

    def promote(self, plan: Plan) -> Plan:
        """A plan whose run verified becomes a cached plan and fragments.

        This is the hours-to-days feedback loop. It writes outward only: a plan
        can be promoted by routing evidence, and can never rewrite the router.
        """
        plan.verified_runs += 1
        self.cache.remember(plan)
        for step in plan.steps:
            self.fragments.add(step)
        return plan


__all__ = [
    "FragmentLibrary",
    "Plan",
    "PlanCache",
    "PlanChoice",
    "PlanSource",
    "Planner",
    "Step",
    "task_shape",
]
