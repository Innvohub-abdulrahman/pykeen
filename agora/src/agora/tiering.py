"""The tier gate.

An inbox triage and an app build differ by four orders of magnitude in cost.
Running one machinery over both is the fastest way to burn money, so every
task is classified before anything else happens:

===========  ==============================  =======  ==========  ===================
Tier         Example                         Agents   Budget      Overhead rule
===========  ==============================  =======  ==========  ===================
REFLEX       inbox triage, calendar conflict  1 local  ~$0         bypasses control plane
TASK         draft a reply, write a function  1        cents       bypasses control plane
PROJECT      add a feature, monthly close     3-6      dollars     <10% of task cost
CAMPAIGN     ship a mobile app                5-15     tens-100s   <10% of task cost
===========  ==============================  =======  ==========  ===================

The classifier is deterministic by default and runs locally at zero marginal
cost. A local model can be plugged in via ``TierGate(model=...)``; its verdict
is *bounded* by the deterministic signals rather than trusted outright, because
a misclassification downward silently removes the control plane from a task
that needed it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .contracts import Tier

# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

_SIDE_EFFECT_HINTS = re.compile(
    r"\b(send|deploy|publish|commit|push|book|pay|charge|refund|delete|"
    r"provision|merge|release|migrate)\b",
    re.IGNORECASE,
)

_DECOMPOSITION_HINTS = re.compile(
    r"\b(and then|after that|first .* then|multi[- ]?step|end[- ]to[- ]end|"
    r"pipeline|across (?:the )?(?:repo|system|services)|refactor|integrate|"
    r"implement .* and .*|feature|epic)\b",
    re.IGNORECASE,
)

_CAMPAIGN_HINTS = re.compile(
    r"\b(ship|launch|build (?:a|an|the) (?:app|product|platform)|"
    r"over (?:the )?(?:next )?(?:weeks?|months?|quarter)|roadmap|"
    r"program|multi[- ]?session|campaign|rollout)\b",
    re.IGNORECASE,
)

#: Action verbs. A goal naming two or more distinct ones ("add the validator
#: *and* wire it into the pipeline") describes work that has to be decomposed,
#: however short the sentence is. Phrase matching alone missed this: the
#: coordination is in the verbs, not in the connective.
_ACTION_VERB = re.compile(
    r"(?i)\b(add|implement|build|wire|integrate|migrate|refactor|deploy|"
    r"write|create|update|remove|delete|fix|test|document|validate|connect|"
    r"expose|generate|publish|configure|instrument|backfill|rename)\b"
)

_REFLEX_HINTS = re.compile(
    r"\b(classify|triage|label|tag|look ?up|check|is there|conflict|"
    r"dedupe|sort|summari[sz]e (?:this|the) (?:email|message))\b",
    re.IGNORECASE,
)


class TierClassifier(Protocol):
    """A pluggable local model. Must be cheap; it runs on every single task."""

    def __call__(self, text: str) -> str: ...  # returns a Tier value


@dataclass
class TierSignals:
    """Everything the gate looked at, kept for the audit trail."""

    capabilities: int = 1
    has_side_effects: bool = False
    needs_decomposition: bool = False
    spans_sessions: bool = False
    explicit_tier: Tier | None = None
    model_vote: Tier | None = None
    reasons: list[str] = field(default_factory=list)


@dataclass
class TierDecision:
    tier: Tier
    signals: TierSignals
    bypasses_control_plane: bool

    @property
    def reasons(self) -> list[str]:
        return self.signals.reasons


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


class TierGate:
    """Classifies every task Reflex / Task / Project / Campaign.

    The deterministic rules establish a *floor*: a task with side effects can
    never be classified REFLEX, and a task needing decomposition can never
    bypass the control plane. A model vote may raise the tier above the floor
    but never lower it below.
    """

    def __init__(self, model: TierClassifier | None = None) -> None:
        self.model = model

    def classify(
        self,
        text: str,
        *,
        capabilities: int = 1,
        side_effects: bool | None = None,
        subtasks: int = 0,
        spans_sessions: bool | None = None,
        explicit_tier: Tier | str | None = None,
        hints: Mapping[str, Any] | None = None,
    ) -> TierDecision:
        hints = dict(hints or {})
        signals = TierSignals(capabilities=capabilities)

        if explicit_tier is not None:
            signals.explicit_tier = Tier(explicit_tier)

        signals.has_side_effects = (
            bool(_SIDE_EFFECT_HINTS.search(text))
            if side_effects is None
            else bool(side_effects)
        )
        distinct_verbs = {m.group(0).lower() for m in _ACTION_VERB.finditer(text)}
        signals.needs_decomposition = bool(
            subtasks > 1
            or _DECOMPOSITION_HINTS.search(text)
            or len(distinct_verbs) > 1
        )
        signals.spans_sessions = (
            bool(_CAMPAIGN_HINTS.search(text))
            if spans_sessions is None
            else bool(spans_sessions)
        )

        floor = self._deterministic_floor(text, signals, hints)

        if self.model is not None:
            try:
                vote = Tier(self.model(text))
            except Exception:  # a broken classifier must not break the gate
                vote = None
            signals.model_vote = vote
            if vote is not None and vote.rank > floor.rank:
                floor = vote
                signals.reasons.append(f"local model raised tier to {vote.value}")

        if signals.explicit_tier is not None:
            if signals.explicit_tier.rank >= floor.rank:
                signals.reasons.append(
                    f"caller declared {signals.explicit_tier.value}"
                )
                floor = signals.explicit_tier
            else:
                signals.reasons.append(
                    f"caller declared {signals.explicit_tier.value} but signals "
                    f"require {floor.value}; the gate does not go down"
                )

        return TierDecision(
            tier=floor,
            signals=signals,
            bypasses_control_plane=floor.bypasses_control_plane,
        )

    # -- rules -------------------------------------------------------------

    def _deterministic_floor(
        self, text: str, signals: TierSignals, hints: Mapping[str, Any]
    ) -> Tier:
        reasons = signals.reasons

        if signals.spans_sessions:
            reasons.append("spans sessions -> CAMPAIGN")
            return Tier.CAMPAIGN

        if signals.needs_decomposition or signals.capabilities > 1:
            reasons.append(
                "needs decomposition or multiple capabilities -> PROJECT"
            )
            return Tier.PROJECT

        if signals.has_side_effects:
            reasons.append("has side effects -> TASK floor (never REFLEX)")
            return Tier.TASK

        if hints.get("uses_tools") or hints.get("tools"):
            reasons.append("uses tools -> TASK")
            return Tier.TASK

        if _REFLEX_HINTS.search(text) and signals.capabilities == 1:
            reasons.append("single capability, no side effects -> REFLEX")
            return Tier.REFLEX

        reasons.append("single agent, default -> TASK")
        return Tier.TASK


# --------------------------------------------------------------------------
# Tier policy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TierPolicy:
    """What each tier is allowed to cost and how it is verified."""

    tier: Tier
    max_agents: int
    typical_budget_usd: float
    overhead_rule: str
    verification: str
    durable_workflow: bool
    ledger_writes_batched: bool


TIER_POLICIES: dict[Tier, TierPolicy] = {
    Tier.REFLEX: TierPolicy(
        tier=Tier.REFLEX,
        max_agents=1,
        typical_budget_usd=0.0,
        overhead_rule="bypasses control plane",
        verification="universal checks only",
        durable_workflow=False,
        ledger_writes_batched=True,
    ),
    Tier.TASK: TierPolicy(
        tier=Tier.TASK,
        max_agents=1,
        typical_budget_usd=0.05,
        overhead_rule="bypasses control plane",
        verification="cheap verification",
        durable_workflow=False,
        ledger_writes_batched=True,
    ),
    Tier.PROJECT: TierPolicy(
        tier=Tier.PROJECT,
        max_agents=6,
        typical_budget_usd=2.50,
        overhead_rule="<10% of task cost",
        verification="full ladder",
        durable_workflow=True,
        ledger_writes_batched=False,
    ),
    Tier.CAMPAIGN: TierPolicy(
        tier=Tier.CAMPAIGN,
        max_agents=15,
        typical_budget_usd=50.0,
        overhead_rule="<10% of task cost",
        verification="phase gates + growing regression suite",
        durable_workflow=True,
        ledger_writes_batched=False,
    ),
}


def policy_for(tier: Tier | str) -> TierPolicy:
    return TIER_POLICIES[Tier(tier)]


def overhead_within_budget(
    tier: Tier | str, task_cost_usd: float, overhead_usd: float
) -> bool:
    """True if control-plane overhead respects the tier's overhead rule."""
    tier = Tier(tier)
    if tier.bypasses_control_plane:
        return overhead_usd <= 1e-6
    if task_cost_usd <= 0:
        return overhead_usd <= 1e-6
    return overhead_usd <= 0.10 * task_cost_usd + 1e-9


__all__ = [
    "TierGate",
    "TierDecision",
    "TierSignals",
    "TierPolicy",
    "TIER_POLICIES",
    "policy_for",
    "overhead_within_budget",
]
