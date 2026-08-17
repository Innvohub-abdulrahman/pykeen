"""The orchestrator: L3, the immutable core.

Solid path: something needs doing -> tier gate -> plan -> contract ->
acceptance-criteria gate -> route -> execute under budget and authority ->
blackboard -> universal floor -> domain checks -> deliverable.

Dotted path, and the only reason the system improves: verified outcome ->
Capability Ledger -> next routing decision; verified plan -> fragment ->
cached plan; unroutable task -> gap log.

The load-bearing detail is the acceptance-criteria gate. It *refuses* tasks
rather than attempting them. If you cannot say how you would check the result,
delegating it only moves the ambiguity somewhere more expensive. This single
rule addresses the largest category of multi-agent failure.

Nothing in the evolution plane may modify this module, the verifier, the
policy engine or the budget enforcer. If the system can rewrite its own
verifier, every other safety property becomes unenforceable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .adapter import A2AAdapter, TaskResult, TaskState
from .artifacts import Artifact
from .audit import AuditLog, EventKind
from .blackboard import Blackboard
from .contracts import (
    AuthorityEnvelope,
    Budget,
    DataClass,
    Scope,
    TaskContract,
    Tier,
)
from .errors import (
    BudgetExceeded,
    NoRouteAvailable,
    SpecificationError,
    TerminationError,
)
from .ledger import CapabilityLedger, Outcome, OutcomeSource
from .planner import Plan, PlanChoice, PlanSource, Planner
from .registry import AgentRegistry
from .router import RouteDecision, Router
from .tiering import TierDecision, TierGate, overhead_within_budget, policy_for
from .verification.ladder import LadderReport, VerificationLadder
from .verification.packs import PackRegistry


class RunState(str, Enum):
    DELIVERED = "delivered"
    REFUSED = "refused"
    ESCALATED = "escalated"
    FAILED = "failed"
    NO_ROUTE = "no-route"


@dataclass
class StepRun:
    """What happened to one step of a plan."""

    step_id: str
    capability: str
    decision: RouteDecision | None = None
    result: TaskResult | None = None
    report: LadderReport | None = None
    artifact_uri: str | None = None
    state: RunState = RunState.FAILED
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.state is RunState.DELIVERED


@dataclass
class RunReport:
    """The whole run: what was decided, what it cost, and whether it verified."""

    task_id: str
    goal: str
    tier: TierDecision | None = None
    plan: PlanChoice | None = None
    steps: list[StepRun] = field(default_factory=list)
    state: RunState = RunState.FAILED
    detail: str = ""
    cost_usd: float = 0.0
    planning_cost_usd: float = 0.0
    verification_cost_usd: float = 0.0
    model_calls: int = 0
    artifact_uri: str | None = None

    @property
    def delivered(self) -> bool:
        return self.state is RunState.DELIVERED

    @property
    def overhead_usd(self) -> float:
        return round(self.planning_cost_usd + self.verification_cost_usd, 6)

    def summary(self) -> str:
        return (
            f"{self.task_id} · {self.state.value} · ${self.cost_usd:.4f} "
            f"(overhead ${self.overhead_usd:.4f}) · {len(self.steps)} step(s)"
        )


class Orchestrator:
    """Durable-workflow-shaped execution with termination guaranteed by the
    runtime rather than by agent cooperation."""

    def __init__(
        self,
        *,
        registry: AgentRegistry,
        ledger: CapabilityLedger,
        router: Router,
        adapters: Mapping[str, A2AAdapter],
        planner: Planner | None = None,
        blackboard: Blackboard | None = None,
        audit: AuditLog | None = None,
        packs: PackRegistry | None = None,
        tier_gate: TierGate | None = None,
    ) -> None:
        self.registry = registry
        self.ledger = ledger
        self.router = router
        self.adapters = dict(adapters)
        self.planner = planner if planner is not None else Planner()
        # `x or Default()` would be wrong for these two: Blackboard and
        # AuditLog define __len__, so an empty one passed in by the caller is
        # falsy and would be silently replaced with a fresh instance.
        self.blackboard = blackboard if blackboard is not None else Blackboard()
        self.audit = audit if audit is not None else AuditLog()
        self.packs = packs if packs is not None else PackRegistry()
        self.tier_gate = tier_gate if tier_gate is not None else TierGate()

    # -- the master flow ---------------------------------------------------

    def run(
        self,
        goal: str,
        capability: str,
        *,
        authority: AuthorityEnvelope,
        budget_usd: float,
        acceptance: Sequence[Any] = (),
        bucket: Mapping[str, str] | None = None,
        data_class: DataClass = DataClass.INTERNAL,
        scope: Scope | None = None,
        inputs: Mapping[str, Any] | None = None,
        sources: Sequence[str] = (),
        required_sections: Sequence[str] = (),
        granted_offers: Sequence[str] = (),
        low_stakes: bool = False,
        explicit_tier: Tier | None = None,
    ) -> RunReport:
        contract = TaskContract.root(
            goal=goal,
            capability=capability,
            authority=authority,
            budget_usd=budget_usd,
            acceptance=tuple(acceptance),
            context_bucket=dict(bucket or {}),
            data_class=data_class,
            scope=scope or Scope(),
            inputs=dict(inputs or {}),
            sources=tuple(sources),
            required_sections=tuple(required_sections),
            granted_offers=tuple(granted_offers),
        )
        report = RunReport(task_id=contract.id, goal=goal)
        self.audit.append(
            EventKind.TASK_SUBMITTED,
            {"goal": goal, "capability": capability},
            task_id=contract.id,
        )

        # 1 · Tier gate. Reflex and Task bypass the control plane entirely.
        decision = self.tier_gate.classify(
            goal, explicit_tier=explicit_tier, hints={"uses_tools": True}
        )
        contract.tier = decision.tier
        report.tier = decision
        self.audit.append(
            EventKind.TIER_DECIDED,
            {"tier": decision.tier.value, "reasons": decision.reasons},
            task_id=contract.id,
        )

        # 2 · Specification gate. Refusal is a valid, cheap outcome.
        try:
            contract.enforce_specification_gate()
        except SpecificationError as exc:
            report.state = RunState.REFUSED
            report.detail = str(exc)
            self.audit.append(
                EventKind.REFUSED, {"reason": str(exc)}, task_id=contract.id
            )
            return report

        if decision.bypasses_control_plane:
            return self._run_direct(contract, report)

        # 3 · Plan: recall, compose, or synthesise.
        #
        # The tier's overhead rule is enforced here rather than hoped for. If
        # synthesising a plan would cost more than a tenth of the task's whole
        # budget, the control plane is the expensive part of the task and the
        # right move is to skip it. Cheap paths -- recall and compose -- are
        # always allowed, because they are free.
        choice = self.planner.plan(
            goal,
            capability,
            budget_usd=budget_usd,
            tier=decision.tier,
            bucket=bucket,
        )
        if choice.source is PlanSource.SYNTHESISE and not overhead_within_budget(
            decision.tier, budget_usd, choice.cost_usd
        ):
            self.audit.append(
                EventKind.PLAN_SELECTED,
                {
                    "source": "skipped",
                    "rationale": (
                        f"synthesis at ${choice.cost_usd:.4f} would exceed 10% of a "
                        f"${budget_usd:.2f} budget; running direct instead"
                    ),
                },
                task_id=contract.id,
            )
            return self._run_direct(contract, report)
        report.plan = choice
        report.planning_cost_usd = choice.cost_usd
        report.cost_usd += choice.cost_usd
        self.audit.append(
            EventKind.PLAN_SELECTED,
            {"source": choice.source.value, "rationale": choice.rationale},
            task_id=contract.id,
            cost_usd=choice.cost_usd,
        )

        # 4 · Execute the plan, one step at a time, under one budget tree.
        produced: dict[str, str] = {}
        try:
            for step in choice.plan.order():
                child = contract.delegate(
                    goal=step.goal,
                    capability=step.capability,
                    declared_authority=authority.narrow(
                        max_depth=max(0, authority.max_depth - 1),
                        max_cost_usd=min(
                            authority.max_cost_usd, step.budget_usd or budget_usd
                        ),
                    ),
                    budget_usd=min(
                        step.budget_usd or budget_usd,
                        contract.budget.available_usd,
                    ),
                    acceptance=step.acceptance,
                    context_bucket=step.context_bucket or contract.context_bucket,
                    inputs={
                        "references": self.blackboard.references_for(
                            [produced[d] for d in step.depends_on if d in produced]
                        )
                    },
                )
                child.required_sections = contract.required_sections
                child.granted_offers = contract.granted_offers
                child.sources = contract.sources

                run = self._run_step(step.id, child, low_stakes=low_stakes)
                report.steps.append(run)
                report.cost_usd += run.result.cost_usd if run.result else 0.0
                if run.report is not None:
                    report.verification_cost_usd += run.report.cost_usd
                    report.cost_usd += run.report.cost_usd
                    report.model_calls += run.report.model_calls
                if run.artifact_uri:
                    produced[step.id] = run.artifact_uri
                if not run.ok:
                    report.state = run.state
                    report.detail = run.detail
                    return report
        except (BudgetExceeded, TerminationError) as exc:
            report.state = RunState.FAILED
            report.detail = f"{type(exc).__name__}: {exc}"
            self.audit.append(
                EventKind.REFUSED, {"reason": report.detail}, task_id=contract.id
            )
            return report

        # 5 · The dotted arrows: a plan that verified becomes a cached plan.
        if choice.source is PlanSource.SYNTHESISE:
            self.planner.promote(choice.plan)
            self.audit.append(
                EventKind.PROMOTED,
                {"plan": choice.plan.id, "shape": choice.plan.shape},
                task_id=contract.id,
            )

        report.state = RunState.DELIVERED
        report.artifact_uri = produced.get(choice.plan.steps[-1].id)
        report.detail = "verified"
        return report

    # -- single-step paths -------------------------------------------------

    def _run_direct(self, contract: TaskContract, report: RunReport) -> RunReport:
        """Reflex and Task tiers: one agent, no orchestration, no workflow engine."""
        run = self._run_step("direct", contract, low_stakes=True)
        report.steps.append(run)
        report.cost_usd += run.result.cost_usd if run.result else 0.0
        if run.report is not None:
            report.verification_cost_usd += run.report.cost_usd
            report.model_calls += run.report.model_calls
        report.state = run.state
        report.detail = run.detail
        report.artifact_uri = run.artifact_uri
        return report

    def _run_step(
        self, step_id: str, contract: TaskContract, *, low_stakes: bool
    ) -> StepRun:
        run = StepRun(step_id=step_id, capability=contract.capability)

        try:
            decision = self.router.route(contract, low_stakes=low_stakes)
        except NoRouteAvailable as exc:
            run.state = RunState.NO_ROUTE
            run.detail = str(exc)
            self.audit.append(
                EventKind.ROUTE_FAILED,
                {"capability": contract.capability, "reason": str(exc)},
                task_id=contract.id,
            )
            return run
        run.decision = decision
        self.audit.append(
            EventKind.ROUTE_DECIDED,
            {
                "agent": decision.agent,
                "stage": decision.stage.value,
                "evidence": decision.reading.evidence_class.value,
                "verification_mandatory": decision.verification_mandatory,
            },
            task_id=contract.id,
            agent=decision.agent,
        )

        adapter = self.adapters.get(decision.agent)
        if adapter is None:
            run.state = RunState.FAILED
            run.detail = f"no adapter registered for {decision.agent}"
            return run

        self.audit.append(
            EventKind.AUTHORITY_GRANTED,
            {"envelope": contract.authority.describe()},
            task_id=contract.id,
            agent=decision.agent,
        )
        result = adapter.execute(contract, budget=contract.budget)
        run.result = result
        self.audit.append(
            EventKind.EXECUTION_FINISHED,
            {"state": result.state.value, "error": result.error},
            task_id=contract.id,
            agent=decision.agent,
            cost_usd=result.cost_usd,
        )
        if not result.ok or result.artifact is None:
            run.state = RunState.FAILED
            run.detail = result.error or "execution failed"
            self._record_outcome(contract, decision, success=False, result=result)
            return run

        uri = self.blackboard.publish(result.artifact)
        run.artifact_uri = uri
        self.audit.append(
            EventKind.ARTIFACT_PUBLISHED,
            {"uri": uri, "digest": result.artifact.digest},
            task_id=contract.id,
            agent=decision.agent,
        )

        # Verification. Sampled when production-measured evidence backs the
        # direct bucket; mandatory otherwise -- including for every borrowed
        # reading, however good the number looks.
        pack = self.packs.for_capability(contract.capability)
        ladder = VerificationLadder(pack)
        report = ladder.verify(result.artifact, contract)
        run.report = report
        self.audit.append(
            EventKind.VERIFICATION,
            {
                "passed": report.passed,
                "reached": report.reached.value,
                "grade": report.grade.value,
                "findings": [str(f) for f in report.blocking],
            },
            task_id=contract.id,
            agent=decision.agent,
            cost_usd=report.cost_usd,
        )

        self._record_outcome(
            contract, decision, success=report.passed, result=result, report=report
        )

        if report.passed:
            run.state = RunState.DELIVERED
            run.detail = report.summary()
        elif report.escalated:
            run.state = RunState.ESCALATED
            run.detail = report.escalation_reason or "escalated to a human"
            self.audit.append(
                EventKind.ESCALATED, {"detail": run.detail}, task_id=contract.id
            )
        else:
            run.state = RunState.FAILED
            run.detail = "; ".join(str(f) for f in report.blocking)
        return run

    # -- the dotted arrow --------------------------------------------------

    def _record_outcome(
        self,
        contract: TaskContract,
        decision: RouteDecision,
        *,
        success: bool,
        result: TaskResult | None = None,
        report: LadderReport | None = None,
    ) -> None:
        """Only a verified outcome updates the ledger.

        An execution that failed before verification is recorded as a failure;
        one that was never verified at all is not recorded, because an
        unverified success is not evidence of anything.
        """
        if report is None and success:
            return
        self.ledger.record(
            Outcome(
                agent=decision.agent,
                capability=contract.capability,
                bucket=self.ledger.bucket_for(contract.context_bucket),
                success=success,
                source=OutcomeSource.PRODUCTION,
                verified_by="universal-floor+pack" if report else "execution",
                cost_usd=result.cost_usd if result else 0.0,
                latency_s=result.latency_s if result else 0.0,
                task_id=contract.id,
                at=self.ledger.clock,
            )
        )
        self.audit.append(
            EventKind.LEDGER_UPDATED,
            {
                "success": success,
                "bucket": list(self.ledger.bucket_for(contract.context_bucket)),
            },
            task_id=contract.id,
            agent=decision.agent,
        )


__all__ = ["Orchestrator", "RunReport", "RunState", "StepRun"]
