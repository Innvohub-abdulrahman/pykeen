#!/usr/bin/env python3
"""capability_ledger.py — measured trust, demonstrated.

The same headline number is three completely different facts depending on the
evidence behind it. This prints all four cases from the report's Figure 12,
then shows an interval narrowing as verified outcomes accumulate.

    python bin/capability_ledger.py
"""
from __future__ import annotations

import sys

from _path import bootstrap

bootstrap()

from agora.ledger import (  # noqa: E402
    CapabilityLedger,
    Outcome,
    OutcomeSource,
)

AGENT = "agent://eng/implementation"
CAP = "code.implement"


def seed(ledger: CapabilityLedger, bucket, n: int, successes: int, source) -> None:
    for i in range(n):
        ledger.record(
            Outcome(
                agent=AGENT,
                capability=CAP,
                bucket=bucket,
                success=i < successes,
                source=source,
                verified_by="blind-verifier",
                cost_usd=0.35,
                latency_s=90.0,
            )
        )


def main() -> int:
    ledger = CapabilityLedger()
    seed(ledger, ("python", "small", "familiar"), 67, 62, OutcomeSource.PRODUCTION)
    seed(ledger, ("swift", "small", "familiar"), 12, 11, OutcomeSource.HARNESS)
    seed(ledger, ("rust", "large", "novel"), 40, 17, OutcomeSource.PRODUCTION)

    print("SAME HEADLINE NUMBER, DIFFERENT FACTS")
    print()
    for bucket in [
        ("python", "small", "familiar"),
        ("swift", "small", "familiar"),
        ("terraform", "novel"),
        ("rust", "large", "novel"),
    ]:
        reading = ledger.read(AGENT, CAP, bucket)
        print(f"  {' · '.join(bucket):<28} {reading.mean:.2f} {reading.interval}")
        print(
            f"  {'':<28} n={reading.n} · {reading.evidence_class.value} · "
            f"{reading.evidence_class.note}"
        )
        if reading.borrowed:
            print(
                f"  {'':<28} borrowed from {reading.borrowed_from or '(pooled root)'}"
                f" at distance {reading.distance:g}; "
                f"{reading.n} outcomes shrank to {reading.effective_n:.1f}"
            )
        print()

    print("AN INTERVAL NARROWING WITH EVIDENCE")
    print()
    fresh = CapabilityLedger()
    for n in (2, 8, 12, 30, 67):
        fresh = CapabilityLedger()
        seed(fresh, ("python", "small", "familiar"), n, round(n * 0.92), OutcomeSource.PRODUCTION)
        reading = fresh.read(AGENT, CAP, ("python", "small", "familiar"))
        bar = "█" * max(1, round(reading.interval.width * 40))
        print(
            f"  n={n:<3} mean {reading.mean:.2f}  {reading.interval}  "
            f"width {reading.interval.width:.2f} {bar}"
        )
    print()
    print("The mean is knowable early. The confidence is not — and confidence is")
    print("what buys autonomy.")

    print()
    print("SELF-REPORTED SUCCESS IS DISCARDED")
    before = ledger.rejected_self_reports
    ledger.record(
        Outcome(AGENT, CAP, ("python", "small", "familiar"), True, OutcomeSource.SELF_REPORTED)
    )
    print(f"  rejected self-reports: {before} → {ledger.rejected_self_reports}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
