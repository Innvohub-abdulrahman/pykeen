"""Beta-posterior arithmetic, implemented from scratch.

The Capability Ledger needs three things from a Beta distribution: its mean,
its credible interval, and a sample. None of them justify a numerical stack
dependency, and a control plane that must run in a customer's own environment
is better off with no compiled dependencies at all.

The interval is the part that matters. A percentage cannot express the
difference between 0.92 measured over 67 verified outcomes and 0.92 guessed
from 2, and it is the interval -- not the mean -- that decides whether a
capability may run unsupervised.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betacf(a: float, b: float, x: float, *, iterations: int = 300) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, iterations + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)`` -- the Beta CDF."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(a * math.log(x) + b * math.log1p(-x) - log_beta(a, b))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        b * math.log1p(-x) + a * math.log(x) - log_beta(b, a)
    ) * _betacf(b, a, 1.0 - x) / b


def beta_ppf(a: float, b: float, q: float, *, tol: float = 1e-9) -> float:
    """Inverse Beta CDF by bisection.

    Bisection rather than Newton because it cannot diverge, and a routing
    decision that silently returns a wrong quantile is worse than one that
    takes fifty extra iterations.
    """
    if q <= 0.0:
        return 0.0
    if q >= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Interval:
    """A credible interval. ``width`` is what you do not know."""

    lo: float
    hi: float
    mass: float = 0.90

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def __str__(self) -> str:
        return f"[{self.lo:.2f}–{self.hi:.2f}]"


@dataclass(frozen=True)
class BetaPosterior:
    """A Beta(alpha, beta) belief about a success rate."""

    alpha: float = 1.0
    beta: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError("alpha and beta must be positive")

    # -- summaries ---------------------------------------------------------

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def evidence(self) -> float:
        """Pseudo-observations behind this posterior, excluding the prior."""
        return max(0.0, self.alpha + self.beta - 2.0)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1.0))

    def interval(self, mass: float = 0.90) -> Interval:
        tail = (1.0 - mass) / 2.0
        return Interval(
            lo=beta_ppf(self.alpha, self.beta, tail),
            hi=beta_ppf(self.alpha, self.beta, 1.0 - tail),
            mass=mass,
        )

    def lower_bound(self, mass: float = 0.90) -> float:
        """The conservative estimate: routing on this is routing on evidence."""
        return self.interval(mass).lo

    # -- updates -----------------------------------------------------------

    def update(self, *, successes: float = 0.0, failures: float = 0.0) -> "BetaPosterior":
        return BetaPosterior(self.alpha + successes, self.beta + failures)

    def decay(self, factor: float) -> "BetaPosterior":
        """Pull evidence toward the uniform prior by ``factor`` in [0, 1].

        A factor of 1.0 keeps everything; 0.0 forgets everything and returns
        Beta(1, 1). Stale confidence cannot persist.
        """
        factor = max(0.0, min(1.0, factor))
        return BetaPosterior(
            1.0 + (self.alpha - 1.0) * factor,
            1.0 + (self.beta - 1.0) * factor,
        )

    def borrow(
        self, factor: float, prior: "BetaPosterior | None" = None
    ) -> "BetaPosterior":
        """Shrink borrowed evidence toward the prior, mean included.

        Distinct from :meth:`decay` in more than name. Decay is about age: you
        keep your best estimate and lose confidence in it. Borrowing is about
        generalisation distance, and there the *estimate itself* is suspect --
        being good at small familiar Python is weak evidence about novel
        Terraform, so both the mean and the evidence count must move toward
        the prior.

        In practice 0.93 measured on small Python, borrowed across two
        dimensions, lands near 0.55 with a very wide interval -- which can
        never authorise unsupervised execution, and that is the point.
        """
        factor = max(0.0, min(1.0, factor))
        prior = prior or UNIFORM_PRIOR
        effective_n = self.evidence * factor
        blended_mean = factor * self.mean + (1.0 - factor) * prior.mean
        return BetaPosterior(
            prior.alpha + effective_n * blended_mean,
            prior.beta + effective_n * (1.0 - blended_mean),
        )

    def sample(self, rng: random.Random | None = None) -> float:
        """One Thompson draw."""
        rng = rng or random
        x = rng.gammavariate(self.alpha, 1.0)
        y = rng.gammavariate(self.beta, 1.0)
        return x / (x + y) if (x + y) > 0 else self.mean

    def __str__(self) -> str:
        ci = self.interval()
        return f"{self.mean:.2f} {ci} n≈{self.evidence:.0f}"


UNIFORM_PRIOR = BetaPosterior(1.0, 1.0)


__all__ = [
    "BetaPosterior",
    "Interval",
    "UNIFORM_PRIOR",
    "beta_ppf",
    "betainc",
    "log_beta",
]
