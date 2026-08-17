"""The master flow, end to end, including the worked trace from the report."""

from __future__ import annotations

import random

import pytest

from agora.adapter import (
    CallableAdapter,
    HumanAdapter,
    Invocation,
    ProcessAdapter,
    QueueItem,
    TaskState,
)
from agora.artifacts import Artifact, Source, Trust
from agora.audit import AuditLog, EventKind
from agora.blackboard import Blackboard
from agora.contracts import (
    AcceptanceCriterion,
    AuthorityEnvelope,
    CheckKind,
    DataClass,
    Scope,
    TaskContract,
    Tier,
)
from agora.ledger import CapabilityLedger, EvidenceClass
from agora.orchestrator import Orchestrator, RunState
from agora.planner import PlanSource, Planner
from agora.registry import AgentRegistry
from agora.router import Router
from agora.tiering import TierGate, overhead_within_budget

from conftest import seed_outcomes

BUCKET = {"language": "python", "size": "small", "familiarity": "familiar"}


@pytest.fixture
def orchestrator(registry, ledger, router, adapters) -> Orchestrator:
    return Orchestrator(
        registry=registry,
        ledger=ledger,
        router=router,
        adapters=adapters,
        audit=AuditLog(model_set="pinned-2026-08"),
        blackboard=Blackboard(),
    )


def _run(orchestrator, goal="Add ZATCA Phase-2 validation and wire it in", **kwargs):
    defaults = dict(
        authority=AuthorityEnvelope.root(
            ["fs.write", "git.commit", "net.fetch", "code.implement", "web.research",
             "code.review"],
            max_depth=3,
            max_cost_usd=2.50,
        ),
        budget_usd=2.50,
        acceptance=[
            AcceptanceCriterion(
                id="non-empty",
                kind=CheckKind.PREDICATE,
                description="the artifact has content",
                predicate=lambda artifact: bool(artifact.content.strip()),
            )
        ],
        bucket=BUCKET,
    )
    defaults.update(kwargs)
    return orchestrator.run(goal, "code.implement", **defaults)


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------


def test_a_project_runs_end_to_end_and_verifies(orchestrator):
    report = _run(orchestrator)
    assert report.state is RunState.DELIVERED, report.detail
    assert report.tier.tier is Tier.PROJECT
    assert all(step.ok for step in report.steps)
    assert report.artifact_uri


def test_the_run_stays_inside_its_budget(orchestrator):
    report = _run(orchestrator, budget_usd=2.50)
    assert report.cost_usd < 2.50


def test_control_plane_overhead_respects_the_tier_rule(orchestrator):
    """Overhead is bounded ex ante, against the budget the task declared.

    The budget is what you know before you start; actual execution cost is not.
    A rule you can only evaluate after the money is spent is not a rule.
    """
    report = _run(orchestrator, budget_usd=2.50)
    assert overhead_within_budget(Tier.PROJECT, 2.50, report.overhead_usd), (
        f"overhead ${report.overhead_usd:.4f} against a $2.50 budget"
    )


def test_a_project_too_cheap_to_plan_skips_planning(orchestrator):
    """If synthesising the plan costs more than a tenth of the whole budget,
    the control plane is the expensive part and skips itself."""
    report = _run(orchestrator, budget_usd=0.40)
    assert report.plan is None
    assert report.planning_cost_usd == 0.0
    assert report.state is RunState.DELIVERED, report.detail


def test_every_step_is_audited_and_the_chain_holds(orchestrator):
    _run(orchestrator)
    orchestrator.audit.verify()
    kinds = {r.kind for r in orchestrator.audit}
    assert EventKind.TIER_DECIDED in kinds
    assert EventKind.ROUTE_DECIDED in kinds
    assert EventKind.VERIFICATION in kinds
    assert EventKind.LEDGER_UPDATED in kinds
    assert orchestrator.audit.replay().faithful


def test_cost_is_attributed_per_agent(orchestrator):
    _run(orchestrator)
    costs = orchestrator.audit.costs()
    assert costs.total > 0
    assert any(agent.startswith("agent://") for agent in costs.by_agent)


# --------------------------------------------------------------------------
# Refusal
# --------------------------------------------------------------------------


def test_a_task_with_no_checkable_criteria_is_refused_not_attempted(orchestrator):
    report = _run(orchestrator, acceptance=[])
    assert report.state is RunState.REFUSED
    assert report.steps == [], "nothing should have been executed"
    assert report.cost_usd == 0.0
    assert orchestrator.audit.of_kind(EventKind.REFUSED)


def test_refusal_is_cheaper_than_any_attempt(orchestrator):
    refused = _run(orchestrator, acceptance=[])
    attempted = _run(orchestrator)
    assert refused.cost_usd < attempted.cost_usd


def test_an_unroutable_capability_is_logged_as_a_gap(registry, ledger, adapters):
    router = Router(registry, ledger)
    orchestrator = Orchestrator(
        registry=registry, ledger=ledger, router=router, adapters=adapters
    )
    report = orchestrator.run(
        "Redline this NDA",
        "legal.redline",
        authority=AuthorityEnvelope.root(["legal.redline"], max_cost_usd=1.0),
        budget_usd=1.0,
        acceptance=[
            AcceptanceCriterion(
                id="x", kind=CheckKind.PREDICATE, description="has content",
                predicate=lambda a: True,
            )
        ],
        explicit_tier=Tier.TASK,
    )
    assert report.state is RunState.NO_ROUTE
    assert len(router.gap_log) == 1


# --------------------------------------------------------------------------
# Tiering
# --------------------------------------------------------------------------


def test_reflex_and_task_bypass_the_control_plane(orchestrator):
    report = _run(orchestrator, goal="Classify this email", explicit_tier=Tier.TASK)
    assert report.tier.tier is Tier.TASK
    assert report.plan is None, "no planning machinery for a Task-tier job"
    assert len(report.steps) == 1


def test_the_gate_will_not_be_talked_down():
    """A caller may raise the tier, never lower it below the signals."""
    gate = TierGate()
    decision = gate.classify(
        "ship the mobile app over the next quarter", explicit_tier=Tier.REFLEX
    )
    assert decision.tier is Tier.CAMPAIGN
    assert any("does not go down" in r for r in decision.reasons)


def test_side_effects_can_never_be_reflex():
    decision = TierGate().classify("send the invoice to the customer")
    assert decision.tier is not Tier.REFLEX


# --------------------------------------------------------------------------
# The dotted arrows -- how the system learns
# --------------------------------------------------------------------------


def test_a_verified_outcome_updates_the_ledger(orchestrator, ledger):
    before = ledger.read("agent://eng/impl", "code.implement", tuple(BUCKET.values()))
    _run(orchestrator)
    after = ledger.read("agent://eng/impl", "code.implement", tuple(BUCKET.values()))
    assert after.n > before.n


def test_a_synthesised_plan_that_verified_becomes_a_cached_plan(orchestrator):
    first = _run(orchestrator)
    assert first.plan.source is PlanSource.SYNTHESISE
    assert first.planning_cost_usd > 0

    second = _run(orchestrator)
    assert second.plan.source is PlanSource.RECALL
    assert second.planning_cost_usd == 0.0, "a plan cache hit costs nothing"


def test_a_plan_only_enters_the_cache_after_it_verified():
    planner = Planner()
    choice = planner.plan("novel goal", "code.implement", budget_usd=1.0)
    with pytest.raises(ValueError, match="verified"):
        planner.cache.remember(choice.plan)


def test_routing_improves_as_evidence_accumulates(registry, ledger, adapters):
    """The seconds-scale loop: measured reliability changes the next route."""
    seed_outcomes(ledger, "agent://eng/impl", "code.implement", tuple(BUCKET.values()), 60, 57)
    seed_outcomes(ledger, "agent://eng/impl2", "code.implement", tuple(BUCKET.values()), 60, 25)
    router = Router(registry, ledger, rng=random.Random(9))
    orchestrator = Orchestrator(
        registry=registry, ledger=ledger, router=router, adapters=adapters
    )
    report = _run(orchestrator, goal="Implement the validator only")
    executed = [s.decision.agent for s in report.steps if s.decision]
    assert "agent://eng/impl" in executed


# --------------------------------------------------------------------------
# Adapters
# --------------------------------------------------------------------------


def test_an_adapter_refuses_work_outside_its_envelope():
    adapter = CallableAdapter(
        "agent://x/y", lambda inv: Artifact(content="done"), capabilities=("doc.write",)
    )
    contract = TaskContract.root(
        "g", "doc.write",
        authority=AuthorityEnvelope.root(["something.else"], max_cost_usd=1.0),
        budget_usd=1.0,
    )
    assert adapter.execute(contract).state is TaskState.REJECTED


def test_an_adapter_refuses_a_capability_it_does_not_declare():
    adapter = CallableAdapter(
        "agent://x/y", lambda inv: Artifact(content="done"), capabilities=("doc.write",)
    )
    contract = TaskContract.root(
        "g", "prod.deploy",
        authority=AuthorityEnvelope.root(["prod.deploy"], max_cost_usd=1.0),
        budget_usd=1.0,
    )
    assert adapter.execute(contract).state is TaskState.REJECTED


def test_an_adapter_halts_on_an_exhausted_budget():
    adapter = CallableAdapter(
        "agent://x/y",
        lambda inv: {"content": "done", "cost_usd": 5.0},
        capabilities=("doc.write",),
    )
    contract = TaskContract.root(
        "g", "doc.write",
        authority=AuthorityEnvelope.root(["doc.write"], max_cost_usd=0.10),
        budget_usd=0.10,
    )
    result = adapter.execute(contract)
    assert result.state is TaskState.FAILED
    assert "exceeds" in result.error


def test_a_runtime_exception_becomes_a_task_failure_not_a_crash():
    def explode(invocation: Invocation):
        raise RuntimeError("the runtime fell over")

    adapter = CallableAdapter("agent://x/y", explode, capabilities=("doc.write",))
    contract = TaskContract.root(
        "g", "doc.write",
        authority=AuthorityEnvelope.root(["doc.write"], max_cost_usd=1.0),
        budget_usd=1.0,
    )
    result = adapter.execute(contract)
    assert result.state is TaskState.FAILED
    assert "fell over" in result.error


def test_a_process_adapter_wraps_a_cli_agent():
    calls = []

    def runner(argv):
        calls.append(argv)
        return "# Result\n\nDone.\n"

    adapter = ProcessAdapter(
        "agent://cli/claude-code",
        ["claude", "-p", "{goal}", "--budget", "{budget}"],
        runner,
        capabilities=("code.implement",),
    )
    contract = TaskContract.root(
        "fix the bug", "code.implement",
        authority=AuthorityEnvelope.root(["code.implement"], max_cost_usd=1.0),
        budget_usd=1.0,
    )
    result = adapter.execute(contract)
    assert result.ok
    assert calls[0][2] == "fix the bug"


def test_a_human_is_modelled_as_an_agent():
    """Grade C and D work routes through the same machinery as everything else."""
    queue: list[QueueItem] = []
    adapter = HumanAdapter(
        "human://legal/counsel",
        queue,
        capabilities=("legal.redline",),
        answer_fn=lambda item: "Clause 7 is acceptable as drafted.",
    )
    contract = TaskContract.root(
        "review clause 7", "legal.redline",
        authority=AuthorityEnvelope.root(["legal.redline"], max_cost_usd=1.0),
        budget_usd=1.0,
    )
    result = adapter.execute(contract)
    assert result.ok
    assert len(queue) == 1 and queue[0].answer
    assert result.artifact.sources[0].trust is Trust.PRIMARY

    card = adapter.agent_card()
    assert card.latency_hint_s > 60, "a person is slow, and the ledger should know"
    assert card.cost_hint_usd == 0.0


def test_an_unanswered_human_queue_item_is_not_an_error():
    queue: list[QueueItem] = []
    adapter = HumanAdapter(
        "human://legal/counsel", queue, capabilities=("legal.redline",), answer_fn=None
    )
    contract = TaskContract.root(
        "review clause 7", "legal.redline",
        authority=AuthorityEnvelope.root(["legal.redline"], max_cost_usd=1.0),
        budget_usd=1.0,
    )
    result = adapter.execute(contract)
    assert result.state is TaskState.FAILED
    assert "awaiting a human" in result.error
    assert len(queue) == 1


# --------------------------------------------------------------------------
# Blackboard
# --------------------------------------------------------------------------


def test_the_blackboard_hands_over_references_not_payloads():
    board = Blackboard()
    uri = board.publish(Artifact(content="x" * 40_000, capability="code.implement"))
    reference = board.reference(uri)
    assert "x" * 100 not in str(reference)
    assert board.savings_ratio() > 10
