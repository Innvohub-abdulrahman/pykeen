#!/usr/bin/env python3
"""agora_cost_model.py — parameterised economics.

    python bin/agora_cost_model.py
"""
from __future__ import annotations

import sys

from _path import bootstrap

bootstrap()

from agora.economics import (  # noqa: E402
    judge_model_comparison,
    render_scenarios,
    render_waterfall,
)


def main() -> int:
    print(render_waterfall())
    print()
    print(render_scenarios())
    print()
    judge, floor = judge_model_comparison()
    print("VERIFYING EVERYTHING, FOR A YEAR")
    print(f"  LLM judge on every artifact : ${judge:,.0f}")
    print(f"  deterministic floor         : ${floor:,.0f}")
    print("  A ledger fed by sampled verification is measuring a biased sample.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
