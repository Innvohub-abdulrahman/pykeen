#!/usr/bin/env python3
"""universal_verifiers.py — the wedge.

Runs the six domain-agnostic deterministic checks across four unrelated
domains plus a clean control. Zero model calls, zero marginal cost.

    python bin/universal_verifiers.py
"""
from __future__ import annotations

import sys

from _path import bootstrap

bootstrap()

from agora.verification.corpus import run_corpus  # noqa: E402
from agora.verification.universal import UNIVERSAL_CHECKS  # noqa: E402


def main() -> int:
    print("UNIVERSAL FLOOR · " + " · ".join(UNIVERSAL_CHECKS))
    print()
    report = run_corpus()
    for result in report.results:
        print(f"── {result.case.domain}")
        if not result.report.findings:
            print("   clean")
        for finding in result.report.findings:
            print(f"   {finding}")
        print()
    print(report.headline())
    if report.clean:
        print("Every seeded fault caught; the clean control produced nothing.")
        return 0
    print(f"missed={report.missed} false_positives={report.false_positives}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
