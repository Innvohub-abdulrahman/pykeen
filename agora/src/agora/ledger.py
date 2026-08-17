"""The Capability Ledger.

Capability is a claim; proficiency is a measurement. Every platform in this
market can tell you what an agent *claims* to do. This tells you how well it
actually does a specific thing in a specific context, with a confidence
interval, and refuses to route work to capabilities that have not earned it.

Four mechanisms, each load-bearing:

*Verified outcomes only.* An outcome enters the ledger if and only if an
independent verifier decided it. Self-reported success is discarded on the
way in, not weighted down later.

*Context buckets.* Reliability is per ``(agent, capability, bucket)``. An
agent that is excellent at small familiar Python is not thereby good at novel
Terraform, and a single scalar cannot say so.

*Decay.* Evidence half-lives toward the uniform prior. Stale confidence
cannot persist just because nobody re-measured it.

*Borrowing with shrinkage.* An unseen bucket inherits a coarser posterior,
shrunk in proportion to generalisation distance. Borrowed evidence widens the
interval, and a wide interval can never authorise unsupervised execution.

Known limitation, stated where the code is rather than in a footnote: these
posteriors assume stationarity that model updates violate. Calendar decay does
not capture a model swap. The ledger records a model fingerprint per outcome
and can *detect* the discontinuity (see :meth:`CapabilityLedger.stationarity_warnings`),
but correcting for it is unsolved here.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator, Mapping, Sequence

from .errors import LedgerError
from .stats import BetaPosterior, Interval

Bucket = tuple[str, ...]


# --------------------------------------------------------------------------
# Evidence classes
# --------------------------------------------------------------------------


class EvidenceClass(str, Enum):
    """How much a posterior is worth, independent of what it says.

    The same headline number is three completely different facts depending on
    the evidence behind it. This enum is where that distinction becomes
    operational instead of rhetorical.
    """

    PRODUCTION_MEASURED = "production-measured"
    HARNESS_VERIFIED = "harness-verified"
    DECLARED = "declared"
    DEPRECATED = "deprecated"

    @property
    def routable(self) -> bool:
        return self in (
            EvidenceClass.PRODUCTION_MEASURED,
            EvidenceClass.HARNESS_VERIFIED,
        )

    @property
    def verification_mandatory(self) -> bool:
        """Anything short of direct production measurement gets checked."""
        return self is not EvidenceClass.PRODUCTION_MEASURED

    @property
    def note(self) -> str:
        return {
            EvidenceClass.PRODUCTION_MEASURED: "routable unsupervised",
            EvidenceClass.HARNESS_VERIFIED: "verification mandatory",
            EvidenceClass.DECLARED: "not routable — this is a guess",
            EvidenceClass.DEPRECATED: "below floor, excluded from routing",
        }[self]


class OutcomeSource(str, Enum):
    """Where an outcome came from. Only two of these may enter the ledger."""

    PRODUCTION = "production"      # verified in a real run
    HARNESS = "harness"            # verified in an eval harness
    SELF_REPORTED = "self-reported"  # discarded
    DECLARED = "declared"          # an AgentCard claim; advisory only


#: Verified outcomes needed in a bucket before it counts as production-measured.
#: Reaching this across ~40 capabilities at 3-5 buckets each is exactly why
#: cross-tenant priors are a necessity rather than an optimisation for smaller
#: organisations.
PRODUCTION_THRESHOLD = 30

#: Minimum evidence before a posterior is worth routing on at all.
HARNESS_THRESHOLD = 5

#: Reliability floor. Below this, with enough evidence to be sure, a
#: capability is deprecated and excluded rather than merely ranked last.
RELIABILITY_FLOOR = 0.50


# --------------------------------------------------------------------------
# Outcomes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """One independently verified result."""

    agent: str
    capability: str
    bucket: Bucket
    success: bool
    source: OutcomeSource = OutcomeSource.PRODUCTION
    verified_by: str = ""
    at: float = 0.0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    model_fingerprint: str = ""
    task_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", OutcomeSource(self.source))
        object.__setattr__(self, "bucket", tuple(self.bucket))
        if self.source in (OutcomeSource.PRODUCTION, OutcomeSource.HARNESS):
            if not self.verified_by:
                raise LedgerError(
                    "a verified outcome must name its verifier; self-reported "
                    "success is not evidence"
                )


# --------------------------------------------------------------------------
# Bucket generalisation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BucketSpec:
    """Ordered dimensions of a context bucket, coarsest first.

    Ordering is a design decision with teeth: generalisation drops dimensions
    from the *right*, so the leftmost dimension is the one the system considers
    least safe to generalise across.
    """

    dimensions: tuple[str, ...] = ("language", "size", "familiarity")

    def of(self, values: Mapping[str, str]) -> Bucket:
        return tuple(str(values[d]) for d in self.dimensions if d in values)

    def generalisations(self, bucket: Bucket) -> list[Bucket]:
        """Coarser buckets, nearest first, ending at the pooled root."""
        return [tuple(bucket[:i]) for i in range(len(bucket) - 1, -1, -1)]

    def distance(self, want: Bucket, have: Bucket) -> float:
        """How far the ledger had to reach to answer about ``want``.

        Dropping a dimension costs 1. Substituting a *different* value at a
        dimension costs 2, because "some other language" generalises far worse
        than "language unspecified".
        """
        if want == have:
            return 0.0
        shared = 0
        for a, b in zip(want, have):
            if a == b:
                shared += 1
            else:
                break
        dropped = len(want) - shared
        substituted = len(have) - shared
        return dropped + 2.0 * substituted


#: How hard borrowed evidence is shrunk per unit of distance. At distance 2 --
#: the reach from "python/small/familiar" to a pooled root that answers for
#: Terraform -- roughly 85% of the evidence is discarded.
SHRINK_PER_DISTANCE = 0.95


def shrink_factor(distance: float, per_distance: float = SHRINK_PER_DISTANCE) -> float:
    if distance <= 0:
        return 1.0
    return math.exp(-per_distance * distance)


def is_deprecated(posterior: BetaPosterior, n: int) -> bool:
    """Below the floor, with enough evidence that it is not noise.

    Two ways in. Either there is enough evidence to say the mean sits below
    the floor, or -- with less evidence -- the whole credible interval does.
    A capability that trips this is *excluded* from routing rather than ranked
    last, and queued for re-evaluation.
    """
    if n >= 20 and posterior.mean < RELIABILITY_FLOOR:
        return True
    return n >= HARNESS_THRESHOLD and posterior.interval().hi < RELIABILITY_FLOOR


# --------------------------------------------------------------------------
# Ledger entries
# --------------------------------------------------------------------------


@dataclass
class Entry:
    """The posterior for one ``(agent, capability, bucket)`` triple."""

    agent: str
    capability: str
    bucket: Bucket
    posterior: BetaPosterior = field(default_factory=lambda: BetaPosterior(1.0, 1.0))
    n_production: int = 0
    n_harness: int = 0
    last_update: float = 0.0
    total_cost_usd: float = 0.0
    total_latency_s: float = 0.0
    model_fingerprints: dict[str, int] = field(default_factory=dict)
    deprecated: bool = False
    re_eval_queued: bool = False

    @property
    def n(self) -> int:
        return self.n_production + self.n_harness

    @property
    def mean(self) -> float:
        return self.posterior.mean

    def interval(self, mass: float = 0.90) -> Interval:
        return self.posterior.interval(mass)

    @property
    def mean_cost_usd(self) -> float:
        return self.total_cost_usd / self.n if self.n else 0.0

    @property
    def mean_latency_s(self) -> float:
        return self.total_latency_s / self.n if self.n else 0.0

    def evidence_class(self) -> EvidenceClass:
        if self.deprecated or is_deprecated(self.posterior, self.n):
            return EvidenceClass.DEPRECATED
        if self.n_production >= PRODUCTION_THRESHOLD:
            return EvidenceClass.PRODUCTION_MEASURED
        if self.n >= HARNESS_THRESHOLD:
            return EvidenceClass.HARNESS_VERIFIED
        return EvidenceClass.DECLARED

    def describe(self) -> str:
        cls = self.evidence_class()
        return (
            f"{self.mean:.2f} {self.interval()} · n={self.n} · {cls.value} · "
            f"{cls.note}"
        )


@dataclass
class Reading:
    """What the router gets back when it asks about a capability."""

    agent: str
    capability: str
    bucket: Bucket
    posterior: BetaPosterior
    evidence_class: EvidenceClass
    borrowed_from: Bucket | None = None
    distance: float = 0.0
    n: int = 0
    mean_cost_usd: float = 0.0
    mean_latency_s: float = 0.0

    @property
    def mean(self) -> float:
        return self.posterior.mean

    @property
    def interval(self) -> Interval:
        return self.posterior.interval()

    @property
    def borrowed(self) -> bool:
        return self.borrowed_from is not None and self.distance > 0

    @property
    def effective_n(self) -> float:
        """Evidence *after* shrinkage. For a borrowed reading this is the
        honest number; ``n`` is how many raw outcomes were reached for."""
        return self.posterior.evidence

    @property
    def routable(self) -> bool:
        return self.evidence_class.routable

    @property
    def verification_mandatory(self) -> bool:
        """Borrowed evidence always forces verification, whatever it says.

        With no data for this exact context the router fell back to a coarser
        bucket and shrank the posterior in proportion to how far it reached.
        That result can never authorise unsupervised execution.
        """
        return self.borrowed or self.evidence_class.verification_mandatory

    def describe(self) -> str:
        origin = (
            f" (borrowed from {self.borrowed_from or '()'} at distance "
            f"{self.distance:g}, {self.n} raw outcomes shrunk to "
            f"{self.effective_n:.1f})"
            if self.borrowed
            else ""
        )
        return (
            f"{self.agent} · {self.capability} · {self.bucket} → "
            f"{self.mean:.2f} {self.interval} n={self.n} "
            f"[{self.evidence_class.value}]{origin}"
        )


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


class CapabilityLedger:
    """Beta posteriors per ``(agent, capability, bucket)``, verified only."""

    def __init__(
        self,
        *,
        spec: BucketSpec | None = None,
        half_life_days: float = 45.0,
        shrink_per_distance: float = SHRINK_PER_DISTANCE,
        prior: BetaPosterior | None = None,
    ) -> None:
        self.spec = spec or BucketSpec()
        self.half_life_days = half_life_days
        self.shrink_per_distance = shrink_per_distance
        self.prior = prior or BetaPosterior(1.0, 1.0)
        self._entries: dict[tuple[str, str, Bucket], Entry] = {}
        self.rejected_self_reports = 0
        self.clock: float = 0.0  # days; injected so tests are deterministic

    # -- writing -----------------------------------------------------------

    def record(self, outcome: Outcome) -> Entry | None:
        """Record a verified outcome. Returns ``None`` if it was discarded.

        Self-reported and declared outcomes are counted and dropped. They are
        not weighted down, not stored with a caveat, and never routed on --
        an agent's opinion of its own work is not evidence about it.
        """
        if outcome.source in (OutcomeSource.SELF_REPORTED, OutcomeSource.DECLARED):
            self.rejected_self_reports += 1
            return None

        key = (outcome.agent, outcome.capability, tuple(outcome.bucket))
        entry = self._entries.get(key)
        if entry is None:
            entry = Entry(
                agent=outcome.agent,
                capability=outcome.capability,
                bucket=tuple(outcome.bucket),
                posterior=self.prior,
                last_update=outcome.at or self.clock,
            )
            self._entries[key] = entry

        entry.posterior = entry.posterior.update(
            successes=1.0 if outcome.success else 0.0,
            failures=0.0 if outcome.success else 1.0,
        )
        if outcome.source is OutcomeSource.PRODUCTION:
            entry.n_production += 1
        else:
            entry.n_harness += 1
        entry.last_update = outcome.at or self.clock
        entry.total_cost_usd += outcome.cost_usd
        entry.total_latency_s += outcome.latency_s
        if outcome.model_fingerprint:
            entry.model_fingerprints[outcome.model_fingerprint] = (
                entry.model_fingerprints.get(outcome.model_fingerprint, 0) + 1
            )
        if is_deprecated(entry.posterior, entry.n):
            entry.deprecated = True
            entry.re_eval_queued = True
        return entry

    def record_many(self, outcomes: Iterable[Outcome]) -> int:
        return sum(1 for o in outcomes if self.record(o) is not None)

    # -- maintenance -------------------------------------------------------

    def decay(self, *, now_days: float | None = None) -> int:
        """Nightly half-life decay toward the uniform prior."""
        now = self.clock if now_days is None else now_days
        self.clock = now
        touched = 0
        for entry in self._entries.values():
            age = max(0.0, now - entry.last_update)
            if age <= 0:
                continue
            factor = 0.5 ** (age / self.half_life_days)
            entry.posterior = entry.posterior.decay(factor)
            entry.last_update = now
            touched += 1
        return touched

    def prune(self, *, min_evidence: float = 0.05) -> list[tuple[str, str, Bucket]]:
        """Drop entries decayed back to noise. Part of the monthly sweep."""
        dead = [
            key
            for key, entry in self._entries.items()
            if entry.posterior.evidence < min_evidence
        ]
        for key in dead:
            del self._entries[key]
        return dead

    # -- reading -----------------------------------------------------------

    def get(self, agent: str, capability: str, bucket: Bucket) -> Entry | None:
        return self._entries.get((agent, capability, tuple(bucket)))

    def _pooled(
        self, agent: str, capability: str, prefix: Bucket
    ) -> tuple[BetaPosterior, int, float, float] | None:
        """Pool every entry whose bucket starts with ``prefix``."""
        alpha = self.prior.alpha
        beta = self.prior.beta
        n = 0
        cost = 0.0
        latency = 0.0
        found = False
        for (a, c, bucket), entry in self._entries.items():
            if a != agent or c != capability:
                continue
            if bucket[: len(prefix)] != prefix:
                continue
            found = True
            alpha += entry.posterior.alpha - self.prior.alpha
            beta += entry.posterior.beta - self.prior.beta
            n += entry.n
            cost += entry.total_cost_usd
            latency += entry.total_latency_s
        if not found:
            return None
        return BetaPosterior(alpha, beta), n, cost, latency

    def read(
        self, agent: str, capability: str, bucket: Bucket | Mapping[str, str]
    ) -> Reading:
        """Read a posterior, borrowing from coarser buckets when it must."""
        if isinstance(bucket, Mapping):
            bucket = self.spec.of(bucket)
        bucket = tuple(bucket)

        entry = self.get(agent, capability, bucket)
        if entry is not None and entry.n > 0:
            return Reading(
                agent=agent,
                capability=capability,
                bucket=bucket,
                posterior=entry.posterior,
                evidence_class=entry.evidence_class(),
                n=entry.n,
                mean_cost_usd=entry.mean_cost_usd,
                mean_latency_s=entry.mean_latency_s,
            )

        for coarser in self.spec.generalisations(bucket):
            pooled = self._pooled(agent, capability, coarser)
            if pooled is None:
                continue
            posterior, n, cost, latency = pooled
            distance = self.spec.distance(bucket, coarser)
            shrunk = posterior.borrow(
                shrink_factor(distance, self.shrink_per_distance), self.prior
            )
            # Borrowed evidence never counts as production-measured, however
            # much of it there is.
            cls = (
                EvidenceClass.HARNESS_VERIFIED
                if shrunk.evidence >= HARNESS_THRESHOLD
                else EvidenceClass.DECLARED
            )
            if is_deprecated(shrunk, int(shrunk.evidence)):
                cls = EvidenceClass.DEPRECATED
            return Reading(
                agent=agent,
                capability=capability,
                bucket=bucket,
                posterior=shrunk,
                evidence_class=cls,
                borrowed_from=coarser,
                distance=distance,
                n=n,
                mean_cost_usd=cost / n if n else 0.0,
                mean_latency_s=latency / n if n else 0.0,
            )

        return Reading(
            agent=agent,
            capability=capability,
            bucket=bucket,
            posterior=self.prior,
            evidence_class=EvidenceClass.DECLARED,
            n=0,
        )

    def bucket_for(self, context: Mapping[str, str] | Sequence[str]) -> Bucket:
        """Resolve a context mapping to a bucket using the declared spec.

        The single place bucket ordering is decided. Every caller that turns a
        contract's context into a ledger key must come through here, or a
        writer and a reader can disagree about what "the same bucket" means --
        which shows up as a silently empty posterior rather than an error.
        """
        if isinstance(context, Mapping):
            resolved = self.spec.of(context)
            if resolved:
                return resolved
            return tuple(str(v) for v in context.values())
        return tuple(str(v) for v in context)

    def candidates(self, capability: str) -> list[str]:
        return sorted(
            {a for (a, c, _) in self._entries if c == capability}
        )

    def entries(self) -> Iterator[Entry]:
        return iter(self._entries.values())

    # -- non-stationarity --------------------------------------------------

    def stationarity_warnings(self, *, dominance: float = 0.6) -> list[str]:
        """Flag entries whose evidence spans a model change.

        This does not correct for non-stationarity -- nothing here does. It
        makes the assumption visible, which is the most this layer can honestly
        offer.
        """
        warnings: list[str] = []
        for entry in self._entries.values():
            counts = entry.model_fingerprints
            if len(counts) < 2:
                continue
            total = sum(counts.values())
            top = max(counts.values())
            if top / total < dominance:
                warnings.append(
                    f"{entry.agent}/{entry.capability}/{entry.bucket}: evidence "
                    f"spans {len(counts)} model fingerprints; the posterior "
                    f"assumes a stationarity the model set does not have"
                )
        return warnings

    # -- reporting ---------------------------------------------------------

    def table(self, capability: str | None = None) -> list[Entry]:
        rows = [
            e
            for e in self._entries.values()
            if capability is None or e.capability == capability
        ]
        return sorted(rows, key=lambda e: (-e.mean, -e.n))


# --------------------------------------------------------------------------
# Cross-tenant priors
# --------------------------------------------------------------------------


@dataclass
class PooledPrior:
    capability: str
    bucket: Bucket
    posterior: BetaPosterior
    n: int
    tenants: int


class CrossTenantPriors:
    """Anonymised ``(capability, bucket, outcome)`` pooled across tenants.

    This solves cold start, creates a genuine network effect, and builds a data
    asset a competitor cannot copy by writing the same algorithm. It also has
    to be designed in from the first line of code: retrofitting a consent basis
    for data pooling after launch is close to impossible, which is why consent
    is a required argument here rather than a configuration flag.

    Nothing agent-identifying or tenant-identifying is stored. A contribution
    is a capability, a bucket, and a boolean.
    """

    def __init__(self, *, min_tenants: int = 3, dilution: float = 0.25) -> None:
        self.min_tenants = min_tenants
        self.dilution = dilution
        self._counts: dict[tuple[str, Bucket], list[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        self._tenants: dict[tuple[str, Bucket], set[str]] = defaultdict(set)

    def contribute(
        self,
        *,
        tenant_id: str,
        capability: str,
        bucket: Bucket,
        success: bool,
        consented: bool,
    ) -> bool:
        """Add one anonymised outcome. Refuses without an explicit consent basis."""
        if not consented:
            return False
        key = (capability, tuple(bucket))
        self._counts[key][0 if success else 1] += 1.0
        self._tenants[key].add(tenant_id)
        return True

    def prior_for(self, capability: str, bucket: Bucket) -> PooledPrior | None:
        key = (capability, tuple(bucket))
        if key not in self._counts:
            return None
        tenants = len(self._tenants[key])
        if tenants < self.min_tenants:
            return None  # k-anonymity: too few contributors to release
        successes, failures = self._counts[key]
        # Diluted so a pooled prior informs a new tenant without swamping the
        # handful of local outcomes that actually describe *their* environment.
        return PooledPrior(
            capability=capability,
            bucket=tuple(bucket),
            posterior=BetaPosterior(
                1.0 + successes * self.dilution, 1.0 + failures * self.dilution
            ),
            n=int(successes + failures),
            tenants=tenants,
        )

    def seed(
        self, ledger: CapabilityLedger, agent: str, capability: str, bucket: Bucket
    ) -> Reading | None:
        """Seed a new tenant's ledger entry from the pool."""
        pooled = self.prior_for(capability, bucket)
        if pooled is None:
            return None
        key = (agent, capability, tuple(bucket))
        entry = ledger._entries.setdefault(
            key,
            Entry(
                agent=agent,
                capability=capability,
                bucket=tuple(bucket),
                posterior=ledger.prior,
                last_update=ledger.clock,
            ),
        )
        entry.posterior = pooled.posterior
        # A seeded entry is a prior, never a measurement of *this* tenant. It
        # is deliberately left out of the production-measured counters.
        return Reading(
            agent=agent,
            capability=capability,
            bucket=tuple(bucket),
            posterior=pooled.posterior,
            evidence_class=EvidenceClass.HARNESS_VERIFIED,
            borrowed_from=tuple(bucket),
            distance=1.0,
            n=pooled.n,
        )


__all__ = [
    "Bucket",
    "BucketSpec",
    "CapabilityLedger",
    "CrossTenantPriors",
    "Entry",
    "EvidenceClass",
    "HARNESS_THRESHOLD",
    "Outcome",
    "OutcomeSource",
    "PRODUCTION_THRESHOLD",
    "PooledPrior",
    "RELIABILITY_FLOOR",
    "Reading",
    "shrink_factor",
]
