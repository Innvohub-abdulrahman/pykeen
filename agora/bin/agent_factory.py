#!/usr/bin/env python3
"""agent_factory.py — the commitment ladder, demonstrated.

    python bin/agent_factory.py
"""
from __future__ import annotations

import sys

from _path import bootstrap

bootstrap()

from agora.contracts import AuthorityEnvelope  # noqa: E402
from agora.factory import AgentFactory, Charter, Demand, which  # noqa: E402
from agora.registry import AgentCard, AgentRegistry, Probation  # noqa: E402

CASES = [
    ("asked twice this month", Demand("report.weekly", occurrences_30d=2)),
    ("fits an existing agent", Demand("code.lint", occurrences_30d=20, fits_existing_agent="agent://eng/implementation")),
    ("single-step, no tools", Demand("doc.summarise", occurrences_30d=15)),
    ("multi-step, no continuity", Demand("data.migrate", occurrences_30d=6, multistep=True, needs_tools=True)),
    ("multi-step, needs continuity", Demand("deal.desk", occurrences_30d=22, multistep=True, needs_tools=True, needs_continuity=True)),
    ("four siblings share memory", Demand("revenue.ops", occurrences_30d=30, multistep=True, needs_tools=True, needs_continuity=True, sibling_agents=4)),
]


def main() -> int:
    print("THE COMMITMENT LADDER")
    print()
    for label, demand in CASES:
        rung, reasons = which(demand)
        print(f"  {label:<30} → {rung.value}")
        print(f"  {'':<30}   {rung.description}")
    print()

    registry = AgentRegistry(
        [
            AgentCard(
                id="agent://eng/implementation",
                name="Implementation",
                capabilities=("code.implement", "repo.refactor", "test.author"),
                probation=Probation.GRADUATED,
                retirement_criteria="verified success below 0.70 over 30 outcomes",
            )
        ]
    )
    factory = AgentFactory(registry)
    parent = AuthorityEnvelope.root(["fs.write", "git.commit", "net.fetch"], max_cost_usd=25.0)

    print("CHARTER REVIEW")
    print()
    overlapping = Charter(
        id="agent://eng/test-writer",
        name="Test Writer",
        department="engineering",
        job="writes tests",
        capabilities=("code.implement", "repo.refactor", "test.author"),
        retirement_criteria="never used for a month",
        eval_suite="evals/eng/tests_v1.yaml",
        authority=AuthorityEnvelope.of(["fs.write"], max_cost_usd=1.0),
    )
    verdict = factory.review(
        overlapping,
        Demand("test.author", occurrences_30d=30, multistep=True, needs_tools=True, needs_continuity=True),
        parent_authority=parent,
    )
    print(f"  {verdict.describe()}")
    print(f"    skeptic: {verdict.skeptic_position}")

    escalating = Charter(
        id="agent://ops/deployer",
        name="Deployer",
        department="ops",
        job="deploys to production",
        capabilities=("prod.deploy",),
        retirement_criteria="no deploys for 60 days",
        eval_suite="evals/ops/deploy_v1.yaml",
        authority=AuthorityEnvelope.of(["prod.deploy", "fs.write"], max_cost_usd=5.0),
    )
    verdict = factory.review(
        escalating,
        Demand("prod.deploy", occurrences_30d=30, multistep=True, needs_tools=True, needs_own_authority=True),
        parent_authority=parent,
    )
    print(f"  {verdict.describe()}")

    good = Charter(
        id="agent://revenue/deal-desk",
        name="Deal Desk",
        department="revenue",
        job="prices and structures non-standard deals",
        capabilities=("deal.structure", "deal.price"),
        retirement_criteria="fewer than 5 deals a month for two months",
        eval_suite="evals/revenue/deal_v1.yaml",
        authority=AuthorityEnvelope.of(["crm.read", "crm.write"], max_cost_usd=2.0),
    )
    verdict = factory.review(
        good,
        Demand("deal.structure", occurrences_30d=22, multistep=True, needs_tools=True, needs_continuity=True),
        parent_authority=AuthorityEnvelope.root(["crm.read", "crm.write"], max_cost_usd=25.0),
    )
    print(f"  {verdict.describe()}")
    card = factory.commit(good, verdict, human_approved=True)
    print(
        f"    committed at probation={card.probation.value} — no routable evidence "
        f"yet, so every output is verified"
    )
    factory.rollback(card.id)
    print(f"    rolled back in one command; registry now holds {len(registry)} agent(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
