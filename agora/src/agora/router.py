"""Three-stage routing.

Policy filters first, statistics second. An agent not permitted to touch this
data never enters the competition, however reliable it is -- and putting the
filter first means a routing bug can never leak data to an agent that policy
excluded, because the excluded agent was never in the candidate set.

::

    contract -> policy filter -> 1 candidate  -> Stage 1  deterministic, 0 ms, free
                              -> N candidates -> Stage 2  Thompson sample from ledger
                              -> no evidence  -> Stage 3  LLM router, constrained
                              -> 0 candidates -> gap log

Whatever stage decides, the *evidence class* of the winning reading decides
what happens next: production-measured evidence in the direct bucket earns
sampled verification; harness or borrowed evidence makes verification
mandatory; deprecated capabilities are excluded outright and queued for
re-evaluation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Protocol, Sequence

from .contracts import DataClass, TaskContract
from .errors import NoRouteAvailable
from .ledger import Bucket, CapabilityLedger, EvidenceClass, Reading
from .registry import AgentCard, AgentRegistry, Probation


class Stage(str, Enum):
    DETERMINISTIC = "stage-1-deterministic"
    BANDIT = "stage-2-bandit"
    LLM = "stage-3-llm"
    EXPLORATION = "exploration"


@dataclass
class Rejection:
    agent: str
    reason: str


@dataclass
class GapEntry:
    """An unroutable task. Every one of these is a capability-gap candidate."""

    capability: str
    bucket: Bucket
    reason: str
    task_id: str | None = None
    count: int = 1


class GapLog:
    """Clustered record of what the fleet could not do.

    This is the input to skill synthesis and to the agent factory: nothing is
    built because it seemed like a good idea, only because demand for it was
    measured here.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, Bucket], GapEntry] = {}

    def log(
        self, capability: str, bucket: Bucket, reason: str, task_id: str | None = None
    ) -> GapEntry:
        key = (capability, tuple(bucket))
        entry = self._entries.get(key)
        if entry is None:
            entry = GapEntry(
                capability=capability,
                bucket=tuple(bucket),
                reason=reason,
                task_id=task_id,
            )
            self._entries[key] = entry
        else:
            entry.count += 1
        return entry

    def clusters(self, *, min_occurrences: int = 10) -> list[GapEntry]:
        """Gaps that have recurred enough to justify building something."""
        return sorted(
            (e for e in self._entries.values() if e.count >= min_occurrences),
            key=lambda e: -e.count,
        )

    def all(self) -> list[GapEntry]:
        return sorted(self._entries.values(), key=lambda e: -e.count)

    def __len__(self) -> int:
        return len(self._entries)


# --------------------------------------------------------------------------
# Policy filter
# --------------------------------------------------------------------------


@dataclass
class PolicyFilter:
    """Capability, side effects, data class, residency and budget.

    Checked before any statistics are consulted. This ordering is a security
    property, not an optimisation.
    """

    require_graduated: bool = True

    def eligible(
        self, card: AgentCard, contract: TaskContract
    ) -> tuple[bool, str]:
        if not card.declares(contract.capability):
            return False, "does not declare the capability"
        if card.probation is Probation.RETIRED:
            return False, "retired"
        if self.require_graduated and not card.probation.may_take_real_traffic:
            return False, f"probation state {card.probation.value}"
        if contract.data_class not in card.data_classes:
            return False, f"not cleared for {contract.data_class.value} data"
        if "*" not in card.residency and contract.residency not in card.residency:
            return False, f"residency {contract.residency!r} not permitted"
        if card.side_effects and not contract.authority.side_effects_allowed:
            return False, "has side effects the contract does not allow"
        if contract.budget is not None and card.cost_hint_usd > contract.budget.available_usd:
            return (
                False,
                f"cost hint ${card.cost_hint_usd:.2f} exceeds the remaining "
                f"${contract.budget.available_usd:.2f}",
            )
        if not card.authority.permits(contract.capability) and card.authority.permissions:
            # An agent may hold a narrower envelope than the task requires.
            pass
        return True, "eligible"

    def apply(
        self, cards: Iterable[AgentCard], contract: TaskContract
    ) -> tuple[list[AgentCard], list[Rejection]]:
        allowed: list[AgentCard] = []
        rejected: list[Rejection] = []
        for card in cards:
            ok, reason = self.eligible(card, contract)
            (allowed if ok else rejected).append(
                card if ok else Rejection(agent=card.id, reason=reason)
            )
        return allowed, rejected


# --------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------


@dataclass
class UtilityWeights:
    """reliability x value − cost − latency − risk."""

    value_usd: float = 10.0
    cost_weight: float = 1.0
    latency_weight: float = 0.01
    risk_weight: float = 4.0

    def score(self, reading: Reading, card: AgentCard, *, sampled: float) -> float:
        cost = reading.mean_cost_usd or card.cost_hint_usd
        latency = reading.mean_latency_s or card.latency_hint_s
        # Risk is interval width: what you do not know about this agent in this
        # context, priced.
        risk = reading.interval.width
        return (
            sampled * self.value_usd
            - self.cost_weight * cost
            - self.latency_weight * latency
            - self.risk_weight * risk
        )


# --------------------------------------------------------------------------
# Stage 3 -- constrained LLM router
# --------------------------------------------------------------------------


class LLMRouter(Protocol):
    """A model asked to pick from a fixed list. It cannot invent an agent."""

    def __call__(self, contract: TaskContract, candidates: Sequence[AgentCard]) -> str: ...


def _fallback_llm_router(
    contract: TaskContract, candidates: Sequence[AgentCard]
) -> str:
    """Deterministic stand-in: cheapest eligible candidate.

    Stage 3 exists for genuinely novel situations where the ledger is silent.
    The default keeps the system runnable and testable without a model call;
    the real one is injected.
    """
    return min(candidates, key=lambda c: (c.cost_hint_usd, c.id)).id


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------


@dataclass
class RouteDecision:
    agent: str
    stage: Stage
    reading: Reading
    card: AgentCard
    verification_mandatory: bool
    exploration: bool = False
    considered: list[str] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    utility: float | None = None
    rationale: str = ""

    def describe(self) -> str:
        mode = "verification mandatory" if self.verification_mandatory else "sampled verification"
        explore = " · exploration" if self.exploration else ""
        return (
            f"{self.agent} via {self.stage.value} · "
            f"{self.reading.evidence_class.value} · {mode}{explore}"
        )


class Router:
    """The routing cascade, with an exploration budget and a gap log."""

    def __init__(
        self,
        registry: AgentRegistry,
        ledger: CapabilityLedger,
        *,
        policy: PolicyFilter | None = None,
        weights: UtilityWeights | None = None,
        llm_router: LLMRouter | None = None,
        gap_log: GapLog | None = None,
        exploration_rate: float = 0.05,
        rng: random.Random | None = None,
    ) -> None:
        self.registry = registry
        self.ledger = ledger
        self.policy = policy or PolicyFilter()
        self.weights = weights or UtilityWeights()
        self.llm_router = llm_router or _fallback_llm_router
        self.gap_log = gap_log if gap_log is not None else GapLog()
        self.exploration_rate = exploration_rate
        self.rng = rng or random.Random()

    # -- the cascade -------------------------------------------------------

    def route(self, contract: TaskContract, *, low_stakes: bool = False) -> RouteDecision:
        bucket = self.ledger.bucket_for(contract.context_bucket)
        declaring = self.registry.declaring(contract.capability)
        if not declaring:
            self.gap_log.log(
                contract.capability, bucket, "no agent declares this capability",
                contract.id,
            )
            raise NoRouteAvailable(
                f"no agent declares {contract.capability!r}; gap logged"
            )

        allowed, rejected = self.policy.apply(declaring, contract)
        if not allowed:
            self.gap_log.log(
                contract.capability,
                bucket,
                "policy excluded every candidate: "
                + "; ".join(f"{r.agent} ({r.reason})" for r in rejected),
                contract.id,
            )
            raise NoRouteAvailable(
                f"policy filter left no candidate for {contract.capability!r}; "
                f"gap logged"
            )

        readings = {
            card.id: self.ledger.read(card.id, contract.capability, bucket)
            for card in allowed
        }

        # Deprecated capabilities are excluded outright, not ranked last, and
        # queued for re-evaluation.
        live = [
            card
            for card in allowed
            if readings[card.id].evidence_class is not EvidenceClass.DEPRECATED
        ]
        for card in allowed:
            if readings[card.id].evidence_class is EvidenceClass.DEPRECATED:
                rejected.append(
                    Rejection(agent=card.id, reason="deprecated — re-eval queued")
                )
                entry = self.ledger.get(card.id, contract.capability, bucket)
                if entry is not None:
                    entry.re_eval_queued = True
        if not live:
            self.gap_log.log(
                contract.capability, bucket, "all candidates deprecated", contract.id
            )
            raise NoRouteAvailable(
                f"every candidate for {contract.capability!r} is below the "
                f"reliability floor; gap logged"
            )

        considered = [c.id for c in live]

        # Exploration: a small slice of low-stakes traffic is reserved so new
        # agents can accumulate evidence at all. Without it, a cold agent can
        # never earn the evidence that would let it be routed to.
        if low_stakes and len(live) > 1 and self.rng.random() < self.exploration_rate:
            cold = sorted(live, key=lambda c: readings[c.id].n)
            if readings[cold[0].id].n < readings[cold[-1].id].n:
                pick = cold[0]
                return RouteDecision(
                    agent=pick.id,
                    stage=Stage.EXPLORATION,
                    reading=readings[pick.id],
                    card=pick,
                    verification_mandatory=True,
                    exploration=True,
                    considered=considered,
                    rejected=rejected,
                    rationale=(
                        "exploration budget: routed to the least-measured "
                        "candidate so it can accumulate evidence"
                    ),
                )

        # Stage 1 -- exactly one candidate. No cost, no sampling, no argument.
        if len(live) == 1:
            card = live[0]
            reading = readings[card.id]
            return RouteDecision(
                agent=card.id,
                stage=Stage.DETERMINISTIC,
                reading=reading,
                card=card,
                verification_mandatory=reading.verification_mandatory,
                considered=considered,
                rejected=rejected,
                rationale="exactly one eligible agent",
            )

        # Stage 2 -- Thompson sampling over routable evidence.
        routable = [c for c in live if readings[c.id].routable]
        if routable:
            scored: list[tuple[float, AgentCard]] = []
            for card in routable:
                reading = readings[card.id]
                sampled = reading.posterior.sample(self.rng)
                scored.append((self.weights.score(reading, card, sampled=sampled), card))
            utility, card = max(scored, key=lambda pair: (pair[0], pair[1].id))
            reading = readings[card.id]
            return RouteDecision(
                agent=card.id,
                stage=Stage.BANDIT,
                reading=reading,
                card=card,
                verification_mandatory=reading.verification_mandatory,
                considered=considered,
                rejected=rejected,
                utility=utility,
                rationale="highest utility on a Thompson draw from the ledger",
            )

        # Stage 3 -- no routable evidence anywhere. A model picks, but only
        # from the registry-constrained candidate list.
        choice = self.llm_router(contract, live)
        card = next((c for c in live if c.id == choice), None)
        if card is None:
            raise NoRouteAvailable(
                f"stage-3 router returned {choice!r}, which is not a candidate"
            )
        reading = readings[card.id]
        return RouteDecision(
            agent=card.id,
            stage=Stage.LLM,
            reading=reading,
            card=card,
            verification_mandatory=True,
            considered=considered,
            rejected=rejected,
            rationale=(
                "no routable evidence in any bucket; a constrained model chose "
                "from the registry and verification is mandatory"
            ),
        )


__all__ = [
    "GapEntry",
    "GapLog",
    "PolicyFilter",
    "Rejection",
    "RouteDecision",
    "Router",
    "Stage",
    "UtilityWeights",
]
