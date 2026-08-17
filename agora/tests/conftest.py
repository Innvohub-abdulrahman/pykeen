"""Shared fixtures. Keeps AGORA importable from a clean checkout."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agora.adapter import CallableAdapter, Invocation  # noqa: E402
from agora.artifacts import Artifact, Source, Trust  # noqa: E402
from agora.contracts import (  # noqa: E402
    AcceptanceCriterion,
    AuthorityEnvelope,
    CheckKind,
    TaskContract,
)
from agora.ledger import CapabilityLedger, Outcome, OutcomeSource  # noqa: E402
from agora.registry import AgentCard, AgentRegistry, Probation  # noqa: E402
from agora.router import Router  # noqa: E402


@pytest.fixture
def root_authority() -> AuthorityEnvelope:
    return AuthorityEnvelope.root(
        [
            "fs.write",
            "git.commit",
            "email.draft",
            "calendar.write",
            "net.fetch",
            "code.implement",
            "web.research",
            "code.review",
        ],
        max_depth=3,
        max_cost_usd=25.0,
    )


@pytest.fixture
def registry() -> AgentRegistry:
    reg = AgentRegistry()
    for agent_id, caps in [
        ("agent://eng/impl", ("code.implement",)),
        ("agent://eng/impl2", ("code.implement",)),
        ("agent://res/spec", ("web.research",)),
        ("agent://eng/review", ("code.review",)),
    ]:
        reg.register(
            AgentCard(
                id=agent_id,
                name=agent_id,
                capabilities=caps,
                probation=Probation.GRADUATED,
                authority=AuthorityEnvelope.of(["*"], max_cost_usd=25.0),
                cost_hint_usd=0.05,
            )
        )
    return reg


@pytest.fixture
def ledger() -> CapabilityLedger:
    return CapabilityLedger()


@pytest.fixture
def router(registry, ledger) -> Router:
    import random

    return Router(registry, ledger, rng=random.Random(1234))


def _echo(invocation: Invocation) -> Artifact:
    return Artifact(
        content=f"# Result\n\nCompleted: {invocation.goal}. Setup takes 11 minutes [S1].\n",
        capability=invocation.capability,
        sources=(
            Source(id="S1", text="setup completed in 11 minutes", trust=Trust.PRIMARY),
        ),
        metadata={
            "test_report": {"exit_code": 0, "failed": 0, "collected": 4},
            "verdict": "ok",
        },
    )


@pytest.fixture
def adapters(registry) -> dict:
    return {
        card.id: CallableAdapter(card.id, _echo, capabilities=card.capabilities)
        for card in registry
    }


@pytest.fixture
def machine_checkable() -> list[AcceptanceCriterion]:
    return [
        AcceptanceCriterion(
            id="non-empty",
            kind=CheckKind.PREDICATE,
            description="the artifact has content",
            predicate=lambda artifact: bool(artifact.content.strip()),
        )
    ]


def seed_outcomes(
    ledger: CapabilityLedger,
    agent: str,
    capability: str,
    bucket,
    n: int,
    successes: int,
    source: OutcomeSource = OutcomeSource.PRODUCTION,
) -> None:
    for i in range(n):
        ledger.record(
            Outcome(
                agent=agent,
                capability=capability,
                bucket=bucket,
                success=i < successes,
                source=source,
                verified_by="test-verifier",
                cost_usd=0.05,
                latency_s=10.0,
            )
        )
