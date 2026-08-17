"""Task contracts, authority envelopes and budgets.

This module is part of L3, the immutable core. Nothing the system generates
about itself may modify it.

Three primitives:

``AuthorityEnvelope``
    What an actor may do. Envelopes narrow monotonically down the delegation
    tree: a child's *effective* envelope is the intersection of its declared
    envelope with its parent's actual one, computed and stored at commit time.
    Escalation is therefore structurally impossible rather than merely
    forbidden -- there is no code path that widens an envelope.

``Budget``
    What an actor may spend. A root ceiling subdivides down to every leaf;
    reserving a child budget debits the parent immediately, so the sum of all
    outstanding leaves can never exceed the root. ``onExceed: continue`` is
    not a legal value.

``TaskContract``
    Scope, budget, authority, termination and acceptance criteria for one unit
    of delegated work. The specification gate refuses any contract whose
    acceptance criteria are not machine-checkable.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from .errors import (
    BudgetExceeded,
    DeadlineExceeded,
    DepthExceeded,
    EscalationError,
    SpecificationError,
)

# --------------------------------------------------------------------------
# Data classes and residency
# --------------------------------------------------------------------------


class DataClass(str, Enum):
    """Sensitivity classes, ordered from least to most restricted."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"

    @property
    def rank(self) -> int:
        return _DATA_CLASS_ORDER.index(self)

    def permits(self, other: "DataClass") -> bool:
        """True if an actor cleared for ``self`` may touch ``other`` data."""
        return other.rank <= self.rank


_DATA_CLASS_ORDER = [
    DataClass.PUBLIC,
    DataClass.INTERNAL,
    DataClass.CONFIDENTIAL,
    DataClass.RESTRICTED,
    DataClass.PERSONAL,
]


class OnExceed(str, Enum):
    """What a budget does when a charge would breach the ceiling.

    ``CONTINUE`` is deliberately absent. A budget that can be exceeded is not
    a budget, and every downstream safety property assumes charges are hard.
    """

    HALT = "halt"
    ESCALATE = "escalate"


# --------------------------------------------------------------------------
# Authority
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AuthorityEnvelope:
    """The set of things an actor may do, with the ceilings that bound them.

    Permissions are dotted action names (``fs.write``, ``git.commit``,
    ``email.draft``). A permission ending in ``.*`` is a wildcard over its
    namespace; ``*`` alone grants everything and is only ever legitimate at
    the root.
    """

    permissions: frozenset[str] = frozenset()
    max_depth: int = 0
    max_cost_usd: float = 0.0
    data_classes: frozenset[DataClass] = frozenset({DataClass.PUBLIC})
    residency: frozenset[str] = frozenset({"*"})
    side_effects_allowed: bool = False

    # -- construction ------------------------------------------------------

    @classmethod
    def of(
        cls,
        permissions: Iterable[str] = (),
        *,
        max_depth: int = 0,
        max_cost_usd: float = 0.0,
        data_classes: Iterable[DataClass | str] = (DataClass.PUBLIC,),
        residency: Iterable[str] = ("*",),
        side_effects_allowed: bool | None = None,
    ) -> "AuthorityEnvelope":
        perms = frozenset(permissions)
        if side_effects_allowed is None:
            side_effects_allowed = any(_is_side_effecting(p) for p in perms)
        return cls(
            permissions=perms,
            max_depth=max_depth,
            max_cost_usd=float(max_cost_usd),
            data_classes=frozenset(DataClass(d) for d in data_classes),
            residency=frozenset(residency),
            side_effects_allowed=side_effects_allowed,
        )

    @classmethod
    def root(
        cls,
        permissions: Iterable[str],
        *,
        max_depth: int = 3,
        max_cost_usd: float = 25.0,
        data_classes: Iterable[DataClass | str] = (
            DataClass.PUBLIC,
            DataClass.INTERNAL,
            DataClass.CONFIDENTIAL,
        ),
        residency: Iterable[str] = ("*",),
    ) -> "AuthorityEnvelope":
        """A root envelope. Only a human ever mints one of these."""
        return cls.of(
            permissions,
            max_depth=max_depth,
            max_cost_usd=max_cost_usd,
            data_classes=data_classes,
            residency=residency,
        )

    # -- queries -----------------------------------------------------------

    def permits(self, action: str) -> bool:
        """True if ``action`` falls inside this envelope."""
        if "*" in self.permissions:
            return True
        if action in self.permissions:
            return True
        parts = action.split(".")
        for i in range(len(parts) - 1, 0, -1):
            if ".".join(parts[:i]) + ".*" in self.permissions:
                return True
        return False

    def permits_data(self, data_class: DataClass | str) -> bool:
        return DataClass(data_class) in self.data_classes

    def permits_residency(self, region: str) -> bool:
        return "*" in self.residency or region in self.residency

    # -- narrowing ---------------------------------------------------------

    def intersect(self, declared: "AuthorityEnvelope") -> "AuthorityEnvelope":
        """Intersect a declared child envelope with this (parent) envelope.

        The result is what the child actually gets. It is always a subset of
        both inputs on every axis, so authority can only narrow going down the
        tree. The child's declared envelope is a *request*, never a grant.
        """
        granted = frozenset(p for p in declared.permissions if self.permits(p))
        return AuthorityEnvelope(
            permissions=granted,
            max_depth=max(0, min(declared.max_depth, self.max_depth - 1)),
            max_cost_usd=min(declared.max_cost_usd, self.max_cost_usd),
            data_classes=declared.data_classes & self.data_classes,
            residency=(
                self.residency
                if "*" in declared.residency
                else declared.residency & self.residency
                if "*" not in self.residency
                else declared.residency
            ),
            side_effects_allowed=(
                declared.side_effects_allowed and self.side_effects_allowed
            ),
        )

    def narrow(self, **overrides: Any) -> "AuthorityEnvelope":
        """Explicitly drop capability. Refuses to widen anything."""
        candidate = replace(self, **overrides)
        if not candidate.issubset(self):
            raise EscalationError(
                "narrow() may only remove authority; "
                f"gained {sorted(candidate.permissions - self.permissions)}"
            )
        return candidate

    def issubset(self, other: "AuthorityEnvelope") -> bool:
        """True if this envelope is contained by ``other`` on every axis."""
        return (
            all(other.permits(p) for p in self.permissions)
            and self.max_depth <= other.max_depth
            and self.max_cost_usd <= other.max_cost_usd + 1e-9
            and self.data_classes <= other.data_classes
            and ("*" in other.residency or self.residency <= other.residency)
            and (other.side_effects_allowed or not self.side_effects_allowed)
        )

    # -- reporting ---------------------------------------------------------

    def dropped_from(self, parent: "AuthorityEnvelope") -> list[str]:
        """Permissions the parent held that this envelope does not."""
        return sorted(p for p in parent.permissions if not self.permits(p))

    def describe(self) -> str:
        perms = " · ".join(sorted(self.permissions)) or "read-only"
        return (
            f"{perms} · ${self.max_cost_usd:.2f} · depth ≤ {self.max_depth}"
        )


_SIDE_EFFECTING_PREFIXES = (
    "fs.write",
    "fs.delete",
    "git.commit",
    "git.push",
    "email.send",
    "calendar.write",
    "crm.write",
    "prod.deploy",
    "payment",
    "shell.exec",
)


def _is_side_effecting(permission: str) -> bool:
    return any(permission.startswith(p) for p in _SIDE_EFFECTING_PREFIXES)


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


@dataclass
class Budget:
    """A hard spending ceiling that subdivides down the delegation tree.

    Reserving a child debits the parent at reservation time, not at spend
    time. That is what makes the invariant hold: at any instant,
    ``spent + reserved <= ceiling`` for every node.
    """

    ceiling_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    on_exceed: OnExceed = OnExceed.HALT
    label: str = "root"
    parent: "Budget | None" = field(default=None, repr=False, compare=False)
    children: list["Budget"] = field(default_factory=list, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.ceiling_usd < 0:
            raise ValueError("budget ceiling must be non-negative")
        self.on_exceed = OnExceed(self.on_exceed)

    # -- queries -----------------------------------------------------------

    @property
    def available_usd(self) -> float:
        return self.ceiling_usd - self.spent_usd - self.reserved_usd

    @property
    def total_committed_usd(self) -> float:
        return self.spent_usd + self.reserved_usd

    def would_exceed(self, amount_usd: float) -> bool:
        return amount_usd > self.available_usd + 1e-9

    # -- operations --------------------------------------------------------

    def charge(self, amount_usd: float, *, reason: str = "") -> float:
        """Spend against this budget. Raises rather than overspending."""
        if amount_usd < 0:
            raise ValueError("charge must be non-negative")
        if self.would_exceed(amount_usd):
            raise BudgetExceeded(
                f"{self.label}: charge ${amount_usd:.4f} exceeds "
                f"${self.available_usd:.4f} available"
                + (f" ({reason})" if reason else "")
            )
        self.spent_usd += amount_usd
        return self.available_usd

    def reserve(self, amount_usd: float, *, label: str) -> "Budget":
        """Carve a child budget out of this one."""
        if self.would_exceed(amount_usd):
            raise BudgetExceeded(
                f"{self.label}: cannot reserve ${amount_usd:.4f}; "
                f"${self.available_usd:.4f} available"
            )
        self.reserved_usd += amount_usd
        child = Budget(
            ceiling_usd=amount_usd,
            on_exceed=self.on_exceed,
            label=label,
            parent=self,
        )
        self.children.append(child)
        return child

    def release(self) -> float:
        """Return this child's unspent remainder to its parent."""
        if self.parent is None:
            return 0.0
        unspent = self.ceiling_usd - self.spent_usd
        self.parent.reserved_usd -= self.ceiling_usd
        self.parent.spent_usd += self.spent_usd
        self.ceiling_usd = self.spent_usd
        return unspent

    def rollup_usd(self) -> float:
        """Total spent by this node and everything beneath it."""
        return self.spent_usd + sum(c.rollup_usd() for c in self.children)

    def invariant_holds(self) -> bool:
        """``spent + reserved <= ceiling`` at this node and every descendant."""
        if self.total_committed_usd > self.ceiling_usd + 1e-9:
            return False
        return all(c.invariant_holds() for c in self.children)


# --------------------------------------------------------------------------
# Acceptance criteria and the specification gate
# --------------------------------------------------------------------------


class CheckKind(str, Enum):
    """How an acceptance criterion is decided.

    Everything except ``RUBRIC`` and ``HUMAN`` is decidable by running
    something. ``RUBRIC`` is machine-checkable but model-graded and therefore
    expensive and fallible; ``HUMAN`` is not machine-checkable at all.
    """

    COMMAND = "command"          # exit code of a command
    TEST = "test"                # named test suite passes
    SCHEMA = "schema"            # artifact conforms to a schema
    PREDICATE = "predicate"      # a Python callable over the artifact
    IDENTITY = "identity"        # a numeric identity must hold
    RUBRIC = "rubric"            # model-graded score against a threshold
    HUMAN = "human"              # a person decides

    @property
    def machine_checkable(self) -> bool:
        return self is not CheckKind.HUMAN

    @property
    def deterministic(self) -> bool:
        return self not in (CheckKind.RUBRIC, CheckKind.HUMAN)


@dataclass(frozen=True)
class AcceptanceCriterion:
    """One machine-checkable statement of what "done" means."""

    id: str
    kind: CheckKind
    description: str
    spec: Mapping[str, Any] = field(default_factory=dict)
    threshold: float | None = None
    predicate: Callable[[Any], bool] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", CheckKind(self.kind))
        if self.kind is CheckKind.RUBRIC and self.threshold is None:
            raise SpecificationError(
                f"criterion {self.id!r}: a rubric criterion needs a threshold"
            )
        if self.kind is CheckKind.PREDICATE and self.predicate is None:
            raise SpecificationError(
                f"criterion {self.id!r}: a predicate criterion needs a callable"
            )

    @property
    def machine_checkable(self) -> bool:
        return self.kind.machine_checkable

    @property
    def deterministic(self) -> bool:
        return self.kind.deterministic


# Vague words that make an acceptance criterion undecidable. The gate rejects
# criteria whose description leans on them without a decidable kind.
_VAGUE_TERMS = re.compile(
    r"\b(good|better|nice|clean|elegant|reasonable|appropriate|sensible|"
    r"high[- ]quality|professional|polished|compelling|as needed|etc)\b",
    re.IGNORECASE,
)


def specification_gate(criteria: Sequence[AcceptanceCriterion]) -> None:
    """Refuse a task whose acceptance criteria are not machine-checkable.

    This is the single highest-value rule in the system. It addresses the
    largest category of multi-agent failure by declining work at the point
    where the ambiguity is cheapest to resolve.
    """
    if not criteria:
        raise SpecificationError(
            "no acceptance criteria: refusing the task. If you cannot say how "
            "you would check the result, delegating it only moves the "
            "ambiguity somewhere more expensive."
        )
    problems: list[str] = []
    for c in criteria:
        if not c.machine_checkable:
            problems.append(f"{c.id}: kind {c.kind.value!r} is not machine-checkable")
            continue
        if not c.deterministic:
            continue  # rubric with a threshold is checkable, just expensive
        vague = _VAGUE_TERMS.search(c.description)
        if vague and not c.spec and c.predicate is None:
            problems.append(
                f"{c.id}: {vague.group(0)!r} is not decidable and no spec was given"
            )
    if problems:
        raise SpecificationError(
            "specification gate refused the task:\n  - " + "\n  - ".join(problems)
        )


# --------------------------------------------------------------------------
# Termination
# --------------------------------------------------------------------------


@dataclass
class TerminationPolicy:
    """Runtime-enforced stopping conditions.

    Termination is guaranteed by the orchestrator, not by agent cooperation.
    An agent that never decides it is finished is stopped anyway.
    """

    max_depth: int = 3
    max_turns: int = 24
    wall_clock_s: float = 900.0
    no_progress_turns: int = 3
    started_at: float = field(default_factory=time.monotonic)

    def check_depth(self, depth: int) -> None:
        if depth > self.max_depth:
            raise DepthExceeded(f"depth {depth} exceeds cap {self.max_depth}")

    def check_deadline(self, *, now: float | None = None) -> None:
        elapsed = (now if now is not None else time.monotonic()) - self.started_at
        if elapsed > self.wall_clock_s:
            raise DeadlineExceeded(
                f"wall clock {elapsed:.1f}s exceeds {self.wall_clock_s:.1f}s"
            )

    def remaining_s(self, *, now: float | None = None) -> float:
        elapsed = (now if now is not None else time.monotonic()) - self.started_at
        return max(0.0, self.wall_clock_s - elapsed)


# --------------------------------------------------------------------------
# Task contract
# --------------------------------------------------------------------------


class Tier(str, Enum):
    """Execution tiers, four orders of magnitude apart in cost.

    ``REFLEX`` and ``TASK`` bypass the control plane entirely. Running the
    full contract machinery over an inbox triage is the fastest way to burn
    money on overhead.
    """

    REFLEX = "reflex"
    TASK = "task"
    PROJECT = "project"
    CAMPAIGN = "campaign"

    @property
    def bypasses_control_plane(self) -> bool:
        return self in (Tier.REFLEX, Tier.TASK)

    @property
    def rank(self) -> int:
        return [Tier.REFLEX, Tier.TASK, Tier.PROJECT, Tier.CAMPAIGN].index(self)


@dataclass
class Scope:
    """What a task may and may not touch, declared per task not per domain."""

    allowed: frozenset[str] = frozenset()
    barred: frozenset[str] = frozenset()

    @classmethod
    def of(
        cls, allowed: Iterable[str] = (), barred: Iterable[str] = ()
    ) -> "Scope":
        return cls(frozenset(allowed), frozenset(barred))

    def bars(self, target: str) -> bool:
        return any(_glob_match(pattern, target) for pattern in self.barred)

    def allows(self, target: str) -> bool:
        if self.bars(target):
            return False
        if not self.allowed:
            return True
        return any(_glob_match(pattern, target) for pattern in self.allowed)


def _glob_match(pattern: str, target: str) -> bool:
    """Cheap glob: ``*`` matches any run of characters, case-insensitively."""
    regex = "^" + ".*".join(re.escape(p) for p in pattern.split("*")) + "$"
    return re.match(regex, target, re.IGNORECASE) is not None


@dataclass
class TaskContract:
    """One unit of delegated work, fully specified before anyone starts it."""

    goal: str
    capability: str
    tier: Tier = Tier.PROJECT
    id: str = field(default_factory=lambda: f"task-{uuid.uuid4().hex[:12]}")
    parent_id: str | None = None
    depth: int = 0
    scope: Scope = field(default_factory=Scope)
    authority: AuthorityEnvelope = field(default_factory=AuthorityEnvelope)
    budget: Budget | None = None
    termination: TerminationPolicy = field(default_factory=TerminationPolicy)
    acceptance: tuple[AcceptanceCriterion, ...] = ()
    data_class: DataClass = DataClass.INTERNAL
    residency: str = "*"
    context_bucket: Mapping[str, str] = field(default_factory=dict)
    inputs: Mapping[str, Any] = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    granted_offers: tuple[str, ...] = ()
    required_sections: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tier = Tier(self.tier)
        self.data_class = DataClass(self.data_class)
        self.acceptance = tuple(self.acceptance)

    # -- gate --------------------------------------------------------------

    def enforce_specification_gate(self) -> None:
        specification_gate(self.acceptance)

    @property
    def deterministic_criteria(self) -> tuple[AcceptanceCriterion, ...]:
        return tuple(c for c in self.acceptance if c.deterministic)

    # -- delegation --------------------------------------------------------

    def delegate(
        self,
        *,
        goal: str,
        capability: str,
        declared_authority: AuthorityEnvelope,
        budget_usd: float,
        acceptance: Sequence[AcceptanceCriterion] = (),
        scope: Scope | None = None,
        context_bucket: Mapping[str, str] | None = None,
        inputs: Mapping[str, Any] | None = None,
        tier: Tier | None = None,
    ) -> "TaskContract":
        """Create a child contract with authority already narrowed.

        The child's effective envelope is computed here, at commit time, and
        stored pre-narrowed. Nothing downstream can widen it, because nothing
        downstream ever sees the declared envelope again.
        """
        self.termination.check_depth(self.depth + 1)
        effective = self.authority.intersect(declared_authority)
        if not effective.issubset(self.authority):
            raise EscalationError(
                "child envelope is not contained by parent -- refusing to commit"
            )
        if self.budget is None:
            raise BudgetExceeded("parent contract has no budget to subdivide")
        child_budget = self.budget.reserve(budget_usd, label=f"{capability}@{self.id}")
        effective = replace(
            effective, max_cost_usd=min(effective.max_cost_usd, budget_usd)
        )
        child = TaskContract(
            goal=goal,
            capability=capability,
            tier=tier or self.tier,
            parent_id=self.id,
            depth=self.depth + 1,
            scope=scope or self.scope,
            authority=effective,
            budget=child_budget,
            termination=self.termination,
            acceptance=tuple(acceptance),
            data_class=self.data_class,
            residency=self.residency,
            context_bucket=dict(context_bucket or self.context_bucket),
            inputs=dict(inputs or {}),
            sources=self.sources,
            granted_offers=self.granted_offers,
        )
        return child

    # -- convenience -------------------------------------------------------

    @classmethod
    def root(
        cls,
        goal: str,
        capability: str,
        *,
        authority: AuthorityEnvelope,
        budget_usd: float,
        acceptance: Sequence[AcceptanceCriterion] = (),
        **kwargs: Any,
    ) -> "TaskContract":
        contract = cls(
            goal=goal,
            capability=capability,
            authority=authority,
            budget=Budget(ceiling_usd=budget_usd, label="root"),
            acceptance=tuple(acceptance),
            **kwargs,
        )
        contract.termination.max_depth = authority.max_depth
        return contract

    def bucket_key(self) -> tuple[str, ...]:
        """The ledger context bucket, in declaration order.

        Order is semantic, not cosmetic: the ledger generalises by dropping
        dimensions from the right, so the leftmost dimension is the one it
        considers least safe to generalise across. Sorting these alphabetically
        would silently reorder that hierarchy, which is why insertion order is
        preserved here and why :meth:`CapabilityLedger.bucket_for` -- which
        knows the declared :class:`BucketSpec` -- is preferred where available.
        """
        return tuple(str(v) for v in self.context_bucket.values())

    def summary(self) -> str:
        spent = self.budget.rollup_usd() if self.budget else 0.0
        ceiling = self.budget.ceiling_usd if self.budget else 0.0
        return (
            f"{self.id} [{self.tier.value}] {self.capability} :: {self.goal} "
            f"(${spent:.2f} of ${ceiling:.2f}, depth {self.depth})"
        )
