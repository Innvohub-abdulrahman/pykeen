"""The starter fleet: sixteen charters across five phases.

Three classes, and conflating them causes most fleet design errors:
control-plane roles, department agents, and ephemeral workers.

**Install phase 1 only, then run it for a fortnight before touching anything
else.** The fleet chart is a destination, not an installation order. Every
agent you add before you have evidence you need it is an agent you will later
have to argue about retiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .contracts import AuthorityEnvelope, DataClass
from .factory import Charter
from .registry import AgentClass


def _envelope(
    permissions: Sequence[str], *, budget: float, depth: int = 1
) -> AuthorityEnvelope:
    return AuthorityEnvelope.of(permissions, max_depth=depth, max_cost_usd=budget)


def starter_charters() -> list[Charter]:
    """All sixteen, in install order."""
    return [
        # ---- Phase 1 · Founder's Office — local inference only -----------
        Charter(
            id="agent://personal/calendar-warden",
            name="Calendar Warden",
            department="personal",
            phase=1,
            grade="A",
            agent_class=AgentClass.PERSISTENT,
            job=(
                "Every meeting request, reschedule and travel change passes here "
                "first. Holds the calendar to conflicts, travel time, working "
                "hours, Hijri dates, Ramadan hours and prayer gaps. Proposes; "
                "never books."
            ),
            capabilities=("schedule.resolve", "schedule.reflow", "schedule.negotiate"),
            authority=_envelope(["calendar.read", "calendar.propose"], budget=0.05),
            data_classes=frozenset({DataClass.PERSONAL, DataClass.INTERNAL}),
            local_inference_only=True,
            memory_scopes=("agent-own", "personal"),
            eval_suite="evals/personal/calendar_v1.yaml",
            retirement_criteria=(
                "retire if fewer than 5 scheduling decisions a week for two "
                "consecutive months, or if proposal acceptance falls below 60%"
            ),
            cost_hint_usd=0.0,
            latency_hint_s=2.0,
        ),
        Charter(
            id="agent://personal/inbox-triage",
            name="Inbox Triage",
            department="personal",
            phase=1,
            grade="B",
            agent_class=AgentClass.PERSISTENT,
            job=(
                "Classifies ~200 messages a day locally at zero marginal cost. "
                "Roughly 15 warrant a reply; those get a draft. Never sends."
            ),
            capabilities=("email.classify", "email.draft"),
            authority=_envelope(["email.read", "email.draft"], budget=0.05),
            data_classes=frozenset({DataClass.PERSONAL, DataClass.INTERNAL}),
            local_inference_only=True,
            memory_scopes=("agent-own", "personal"),
            eval_suite="evals/personal/inbox_v1.yaml",
            retirement_criteria=(
                "retire if draft acceptance falls below 40% for a month, or if "
                "misclassification of urgent mail exceeds 2%"
            ),
            cost_hint_usd=0.0,
            latency_hint_s=1.5,
        ),
        Charter(
            id="agent://personal/daily-brief",
            name="Daily Brief",
            department="personal",
            phase=1,
            grade="B",
            agent_class=AgentClass.EPHEMERAL,
            job=(
                "Runs at 06:30, dies at 06:31. One page from calendar, inbox, "
                "finance and open projects. Surfaces decisions not revisited."
            ),
            capabilities=("brief.compile",),
            authority=_envelope(["calendar.read", "email.read", "doc.write"], budget=0.02),
            data_classes=frozenset({DataClass.PERSONAL, DataClass.INTERNAL}),
            local_inference_only=True,
            memory_scopes=("task",),
            eval_suite="evals/personal/brief_v1.yaml",
            retirement_criteria="retire if the brief goes unread for 10 consecutive days",
            cost_hint_usd=0.0,
            latency_hint_s=8.0,
        ),
        # ---- Phase 2 · Engineering ---------------------------------------
        Charter(
            id="agent://eng/implementation",
            name="Implementation",
            department="engineering",
            phase=2,
            grade="A",
            agent_class=AgentClass.PERSISTENT,
            job=(
                "Takes a contract with tests attached and lands the change on a "
                "branch. Refuses tasks without machine-checkable criteria."
            ),
            capabilities=("code.implement", "repo.refactor", "test.author"),
            authority=_envelope(
                ["fs.write", "git.commit", "net.fetch", "code.implement",
                 "repo.refactor", "test.author"],
                budget=4.00,
                depth=2,
            ),
            memory_scopes=("agent-own", "project"),
            eval_suite="evals/eng/implementation_v1.yaml",
            retirement_criteria=(
                "retire if verified success falls below 0.70 in its primary "
                "bucket over 30 outcomes"
            ),
            cost_hint_usd=0.35,
            latency_hint_s=90.0,
        ),
        Charter(
            id="agent://eng/code-reviewer",
            name="Code Reviewer",
            department="engineering",
            phase=2,
            grade="A",
            agent_class=AgentClass.PERSISTENT,
            job=(
                "Scores changes against criteria while blind to the author's "
                "reasoning. Deliberately starved of context."
            ),
            capabilities=("code.review", "verify.rubric"),
            authority=_envelope(["fs.read", "code.review", "verify.rubric"], budget=0.60),
            memory_scopes=("agent-own",),
            eval_suite="evals/eng/review_v1.yaml",
            retirement_criteria=(
                "retire if its verdicts and the deterministic checks disagree on "
                "more than 15% of artifacts"
            ),
            cost_hint_usd=0.08,
            latency_hint_s=25.0,
        ),
        Charter(
            id="agent://eng/build-matrix",
            name="Build Matrix",
            department="engineering",
            phase=2,
            grade="A",
            agent_class=AgentClass.PERSISTENT,
            job=(
                "Builds, runs the device matrix, diffs screenshots, lints store "
                "policy, guards permission deltas."
            ),
            capabilities=("build.compile", "build.device_matrix", "build.store_lint",
                          "test.execute"),
            authority=_envelope(
                ["shell.exec", "fs.read", "build.compile", "build.device_matrix",
                 "build.store_lint", "test.execute"],
                budget=1.20,
            ),
            memory_scopes=("agent-own", "project"),
            eval_suite="evals/eng/build_v1.yaml",
            retirement_criteria="retire if the device matrix has not changed in 90 days",
            cost_hint_usd=0.15,
            latency_hint_s=300.0,
        ),
        Charter(
            id="agent://eng/spec-research",
            name="Spec Research",
            department="engineering",
            phase=2,
            grade="B",
            agent_class=AgentClass.EPHEMERAL,
            job=(
                "Pulls primary sources, tags everything untrusted. Strips "
                "instruction-shaped content before handoff."
            ),
            capabilities=("web.research", "doc.extract"),
            authority=_envelope(["net.fetch", "web.research", "doc.extract"], budget=0.80),
            memory_scopes=("task",),
            eval_suite="evals/eng/research_v1.yaml",
            retirement_criteria="retire if source tie-out failures exceed 10% over 20 briefs",
            cost_hint_usd=0.12,
            latency_hint_s=45.0,
        ),
        # ---- Phase 3 · Finance -------------------------------------------
        Charter(
            id="agent://finance/reconciler",
            name="Reconciler",
            department="finance",
            phase=3,
            grade="A",
            agent_class=AgentClass.PERSISTENT,
            job=(
                "Ties settlements to the ledger monthly and flags every gap with "
                "its source row. Cannot hand you an untraceable figure."
            ),
            capabilities=("finance.reconcile", "finance.trace"),
            authority=_envelope(
                ["finance.read", "finance.reconcile", "finance.trace"], budget=0.90
            ),
            data_classes=frozenset({DataClass.CONFIDENTIAL, DataClass.INTERNAL}),
            memory_scopes=("agent-own", "department"),
            eval_suite="evals/finance/reconcile_v1.yaml",
            retirement_criteria="retire if any month closes with an unexplained variance",
            cost_hint_usd=0.20,
            latency_hint_s=60.0,
        ),
        Charter(
            id="agent://finance/runway-watch",
            name="Runway Watch",
            department="finance",
            phase=3,
            grade="B",
            agent_class=AgentClass.PERSISTENT,
            job=(
                "Projects cash at 30/60/90 days and names what breaks first. "
                "Self-retires if forecast error exceeds 15% for two months."
            ),
            capabilities=("finance.forecast", "finance.scenario"),
            authority=_envelope(["finance.read", "finance.forecast"], budget=0.40),
            data_classes=frozenset({DataClass.CONFIDENTIAL, DataClass.INTERNAL}),
            memory_scopes=("agent-own", "department"),
            eval_suite="evals/finance/forecast_v1.yaml",
            retirement_criteria=(
                "self-retires if 30-day forecast error exceeds 15% for two "
                "consecutive months"
            ),
            cost_hint_usd=0.10,
            latency_hint_s=30.0,
        ),
        # ---- Phase 4 · Revenue and Customer ------------------------------
        Charter(
            id="agent://revenue/pipeline-keeper",
            name="Pipeline Keeper",
            department="revenue",
            phase=4,
            grade="B",
            agent_class=AgentClass.PERSISTENT,
            job=(
                "Weekly sweep for stale deals, missing fields and single-threaded "
                "accounts. Writes to the CRM, never to a customer."
            ),
            capabilities=("crm.hygiene", "crm.enrich"),
            authority=_envelope(["crm.read", "crm.write"], budget=0.30),
            memory_scopes=("agent-own", "department"),
            eval_suite="evals/revenue/hygiene_v1.yaml",
            retirement_criteria="retire if fewer than 10 corrections a month for two months",
            cost_hint_usd=0.06,
            latency_hint_s=40.0,
        ),
        Charter(
            id="agent://revenue/account-research",
            name="Account Research",
            department="revenue",
            phase=4,
            grade="B",
            agent_class=AgentClass.EPHEMERAL,
            job=(
                "One-page brief before a call. Drops unsourced claims rather than "
                "hedging them."
            ),
            capabilities=("account.research",),
            authority=_envelope(["net.fetch", "crm.read", "account.research"], budget=0.50),
            memory_scopes=("task",),
            eval_suite="evals/revenue/account_v1.yaml",
            retirement_criteria="retire if briefs go unopened before 50% of calls",
            cost_hint_usd=0.10,
            latency_hint_s=50.0,
        ),
        Charter(
            id="agent://customer/ticket-triage",
            name="Ticket Triage",
            department="customer",
            phase=4,
            grade="B",
            agent_class=AgentClass.PERSISTENT,
            job=(
                "Prioritises, dedupes and routes. Sorts but never answers — "
                "answering is a different authority envelope."
            ),
            capabilities=("support.triage", "support.dedupe"),
            authority=_envelope(["support.read", "support.route"], budget=0.15),
            memory_scopes=("agent-own", "department"),
            eval_suite="evals/customer/triage_v1.yaml",
            retirement_criteria="retire if mis-routing exceeds 8% over 200 tickets",
            cost_hint_usd=0.02,
            latency_hint_s=5.0,
        ),
        Charter(
            id="agent://customer/reply-drafter",
            name="Reply Drafter",
            department="customer",
            phase=4,
            grade="C",
            agent_class=AgentClass.PERSISTENT,
            job=(
                "Drafts from the knowledge base with a no-commitment guard. Earns "
                "auto-send per bucket; never for billing or contracts."
            ),
            capabilities=("support.draft",),
            authority=_envelope(["support.read", "support.draft"], budget=0.25),
            memory_scopes=("agent-own", "department"),
            eval_suite="evals/customer/draft_v1.yaml",
            retirement_criteria=(
                "retire if human edit distance on drafts exceeds 50% for a month"
            ),
            cost_hint_usd=0.05,
            latency_hint_s=20.0,
        ),
        # ---- Phase 5 · Control plane -------------------------------------
        Charter(
            id="agent://control/planner",
            name="Planner",
            department="control",
            phase=5,
            grade="B",
            agent_class=AgentClass.CONTROL_PLANE,
            job=(
                "Turns intent into a contract tree. Recalls cached plans, composes "
                "fragments, synthesises fresh only when it must."
            ),
            capabilities=("plan.decompose", "plan.synthesise"),
            authority=_envelope(["plan.decompose", "plan.synthesise"], budget=0.30, depth=3),
            memory_scopes=("agent-own", "project"),
            eval_suite="evals/control/plan_v1.yaml",
            retirement_criteria="retire if plan-cache hit rate stays below 20% after 200 tasks",
            cost_hint_usd=0.08,
            latency_hint_s=15.0,
        ),
        Charter(
            id="agent://control/archivist",
            name="Archivist",
            department="control",
            phase=5,
            grade="A",
            agent_class=AgentClass.CONTROL_PLANE,
            job=(
                "Monthly entropy sweep — idle agents, dead skills, stale ledger "
                "entries. Proposes only."
            ),
            capabilities=("fleet.audit", "fleet.archive_propose"),
            authority=_envelope(["fleet.read", "fleet.propose"], budget=0.10),
            memory_scopes=("agent-own",),
            eval_suite="evals/control/archivist_v1.yaml",
            retirement_criteria="retire if it proposes nothing for three consecutive sweeps",
            cost_hint_usd=0.02,
            latency_hint_s=30.0,
        ),
        Charter(
            id="agent://control/council-skeptic",
            name="Council Skeptic",
            department="control",
            phase=5,
            grade="B",
            agent_class=AgentClass.EPHEMERAL,
            job=(
                "Spawned when a gap clusters. Argues for the smallest thing that "
                "solves it. Retires if its rejection rate drops below 40%."
            ),
            capabilities=("charter.critique",),
            authority=_envelope(["fleet.read", "charter.critique"], budget=0.10),
            memory_scopes=("task",),
            eval_suite="evals/control/skeptic_v1.yaml",
            retirement_criteria="retires if rejection rate falls below 40% over 10 reviews",
            cost_hint_usd=0.03,
            latency_hint_s=12.0,
        ),
    ]


@dataclass
class CharterValidation:
    charter_id: str
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def validate_charter(charter: Charter) -> CharterValidation:
    """Every charter must be installable without further argument."""
    problems: list[str] = []
    if not charter.id.startswith("agent://"):
        problems.append("id must be an agent:// URI")
    if not charter.capabilities:
        problems.append("no capabilities declared")
    if not charter.job.strip():
        problems.append("no job description")
    if not charter.retirement_criteria.strip():
        problems.append("no retirement criteria")
    if not charter.eval_suite:
        problems.append("no eval suite")
    if charter.authority.max_cost_usd <= 0:
        problems.append("no budget ceiling")
    if charter.grade not in ("A", "B", "C", "D"):
        problems.append(f"invalid assurance grade {charter.grade!r}")
    if not 1 <= charter.phase <= 5:
        problems.append(f"invalid install phase {charter.phase}")
    if charter.local_inference_only and charter.cost_hint_usd > 0:
        problems.append("local-inference-only agents must have zero marginal cost")
    if charter.agent_class is AgentClass.EPHEMERAL and "project" in charter.memory_scopes:
        problems.append("an ephemeral agent should not hold project memory")
    return CharterValidation(charter_id=charter.id, problems=problems)


def validate_all(charters: Sequence[Charter] | None = None) -> list[CharterValidation]:
    return [validate_charter(c) for c in (charters or starter_charters())]


def by_phase(phase: int) -> list[Charter]:
    return [c for c in starter_charters() if c.phase == phase]


def render_fleet() -> str:
    lines = ["STARTER FLEET · 16 charters across 5 phases", ""]
    for phase in range(1, 6):
        members = by_phase(phase)
        if not members:
            continue
        lines.append(f"PHASE {phase} · {members[0].department}")
        for charter in members:
            lines.append(
                f"  {charter.name:<20} {charter.agent_class.value:<13} "
                f"Grade {charter.grade}  {', '.join(charter.capabilities)}"
            )
        lines.append("")
    lines.append("Install phase 1 only, then run it for a fortnight.")
    return "\n".join(lines)


__all__ = [
    "CharterValidation",
    "by_phase",
    "render_fleet",
    "starter_charters",
    "validate_all",
    "validate_charter",
]
