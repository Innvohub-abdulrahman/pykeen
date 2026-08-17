"""The agent factory.

Agents that build agents, with the discipline that stops a self-expanding
fleet consuming the company running it.

The commitment ladder is the core of it. Three questions decide how much
machinery a new capability deserves, and the default answer is *none*:

======  ==========================================  ===================================
Rung    When                                        What you get
======  ==========================================  ===================================
P0      under 3 occurrences in 30 days              build nothing -- just ask
P1      fits an existing agent's job                a skill, no identity, no authority
P2      multi-step, but no continuity needed        an ephemeral agent
P3      10+ occurrences and continuity or authority a persistent agent
P4      4+ siblings sharing memory                  a department
======  ==========================================  ===================================

Two gates sit across the ladder regardless of rung. A charter whose capability
set overlaps an existing agent by more than 0.85 is rejected -- extend the
existing one. And no agent is created without retirement criteria and an eval
suite, because a fleet that can only grow is a fleet that will.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .contracts import AuthorityEnvelope, DataClass
from .errors import CharterError, EscalationError
from .registry import AgentCard, AgentClass, AgentRegistry, Probation

#: Capability-set overlap above which a new charter is refused outright.
OVERLAP_THRESHOLD = 0.85

#: Occurrences in 30 days below which the answer is "just ask".
MIN_OCCURRENCES_FOR_ANY_BUILD = 3

#: Occurrences required before an agent gets its own identity and continuity.
MIN_OCCURRENCES_FOR_PERSISTENT = 10

#: Sibling agents sharing memory before a department is warranted.
MIN_SIBLINGS_FOR_DEPARTMENT = 4


class Rung(str, Enum):
    """What to build, in ascending order of commitment."""

    P0_NOTHING = "P0-nothing"
    P1_SKILL = "P1-skill"
    P2_EPHEMERAL = "P2-ephemeral"
    P3_PERSISTENT = "P3-persistent"
    P4_DEPARTMENT = "P4-department"

    @property
    def needs_charter(self) -> bool:
        return self in (Rung.P2_EPHEMERAL, Rung.P3_PERSISTENT, Rung.P4_DEPARTMENT)

    @property
    def description(self) -> str:
        return {
            Rung.P0_NOTHING: "build nothing — just ask",
            Rung.P1_SKILL: "a skill under an existing agent — no identity, no authority",
            Rung.P2_EPHEMERAL: "an ephemeral agent — spawned, does the job, dies",
            Rung.P3_PERSISTENT: "a persistent agent with its own memory and authority",
            Rung.P4_DEPARTMENT: "a department — siblings sharing a memory scope",
        }[self]


@dataclass
class Demand:
    """The measured evidence that something needs building at all."""

    capability: str
    occurrences_30d: int = 0
    multistep: bool = False
    needs_tools: bool = False
    needs_continuity: bool = False
    needs_own_authority: bool = False
    sibling_agents: int = 0
    fits_existing_agent: str | None = None


def which(demand: Demand) -> tuple[Rung, list[str]]:
    """Which operation do I actually need?

    Three distinct operations get called "adding an agent". They carry very
    different costs, and picking the wrong one is where fleets go bad.
    """
    reasons: list[str] = []

    if demand.occurrences_30d < MIN_OCCURRENCES_FOR_ANY_BUILD:
        reasons.append(
            f"{demand.occurrences_30d} occurrence(s) in 30 days is below the "
            f"threshold of {MIN_OCCURRENCES_FOR_ANY_BUILD}"
        )
        return Rung.P0_NOTHING, reasons

    if demand.fits_existing_agent:
        reasons.append(f"fits the existing job of {demand.fits_existing_agent}")
        return Rung.P1_SKILL, reasons

    if not (demand.multistep and demand.needs_tools):
        reasons.append("single-step or tool-free work does not need an identity")
        return Rung.P1_SKILL, reasons

    if not (
        demand.occurrences_30d >= MIN_OCCURRENCES_FOR_PERSISTENT
        and (demand.needs_continuity or demand.needs_own_authority)
    ):
        reasons.append(
            "multi-step and tool-using, but without the volume plus continuity "
            "or authority that justifies a standing identity"
        )
        return Rung.P2_EPHEMERAL, reasons

    if demand.sibling_agents >= MIN_SIBLINGS_FOR_DEPARTMENT:
        reasons.append(
            f"{demand.sibling_agents} siblings already share this memory scope"
        )
        return Rung.P4_DEPARTMENT, reasons

    reasons.append(
        f"{demand.occurrences_30d} occurrences with "
        + ("continuity" if demand.needs_continuity else "its own authority")
    )
    return Rung.P3_PERSISTENT, reasons


# --------------------------------------------------------------------------
# Charters
# --------------------------------------------------------------------------


@dataclass
class Charter:
    """The document that has to survive review before an agent exists."""

    id: str
    name: str
    department: str
    job: str
    capabilities: tuple[str, ...]
    agent_class: AgentClass = AgentClass.PERSISTENT
    authority: AuthorityEnvelope = field(default_factory=AuthorityEnvelope)
    data_classes: frozenset[DataClass] = frozenset({DataClass.INTERNAL})
    retirement_criteria: str = ""
    eval_suite: str | None = None
    memory_scopes: tuple[str, ...] = ("agent-own",)
    local_inference_only: bool = False
    grade: str = "D"
    phase: int = 5
    runtime: str = "claude-code"
    cost_hint_usd: float = 0.05
    latency_hint_s: float = 20.0
    notes: str = ""

    def __post_init__(self) -> None:
        self.capabilities = tuple(self.capabilities)
        self.agent_class = AgentClass(self.agent_class)
        self.data_classes = frozenset(DataClass(d) for d in self.data_classes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "department": self.department,
            "job": self.job,
            "capabilities": list(self.capabilities),
            "agent_class": self.agent_class.value,
            "authority": {
                "permissions": sorted(self.authority.permissions),
                "max_depth": self.authority.max_depth,
                "max_cost_usd": self.authority.max_cost_usd,
            },
            "data_classes": sorted(d.value for d in self.data_classes),
            "retirement_criteria": self.retirement_criteria,
            "eval_suite": self.eval_suite,
            "memory_scopes": list(self.memory_scopes),
            "local_inference_only": self.local_inference_only,
            "grade": self.grade,
            "phase": self.phase,
            "runtime": self.runtime,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Charter":
        authority = data.get("authority", {})
        return cls(
            id=data["id"],
            name=data["name"],
            department=data.get("department", ""),
            job=data.get("job", ""),
            capabilities=tuple(data.get("capabilities", ())),
            agent_class=AgentClass(data.get("agent_class", "persistent")),
            authority=AuthorityEnvelope.of(
                authority.get("permissions", ()),
                max_depth=authority.get("max_depth", 0),
                max_cost_usd=authority.get("max_cost_usd", 0.0),
            ),
            data_classes=frozenset(
                DataClass(d) for d in data.get("data_classes", ["internal"])
            ),
            retirement_criteria=data.get("retirement_criteria", ""),
            eval_suite=data.get("eval_suite"),
            memory_scopes=tuple(data.get("memory_scopes", ("agent-own",))),
            local_inference_only=data.get("local_inference_only", False),
            grade=data.get("grade", "D"),
            phase=data.get("phase", 5),
            runtime=data.get("runtime", "claude-code"),
        )


@dataclass
class Verdict:
    """The outcome of charter review, with every objection recorded."""

    charter_id: str
    rung: Rung
    approved: bool
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    overlap: dict[str, float] = field(default_factory=dict)
    skeptic_position: str = ""

    @property
    def worst_overlap(self) -> tuple[str, float] | None:
        if not self.overlap:
            return None
        agent, score = max(self.overlap.items(), key=lambda kv: kv[1])
        return (agent, score) if score > 0 else None

    def describe(self) -> str:
        state = "APPROVED" if self.approved else "BLOCKED"
        return (
            f"{state} · {self.charter_id} · {self.rung.value}"
            + (f" · blockers: {'; '.join(self.blockers)}" if self.blockers else "")
        )


class Skeptic:
    """Argues for the smallest thing that solves the problem.

    A council role, not a formality. It retires itself if its rejection rate
    falls below 40% -- a skeptic that approves everything has stopped being a
    skeptic and has become overhead.
    """

    def __init__(self, *, retirement_floor: float = 0.40) -> None:
        self.retirement_floor = retirement_floor
        self.reviewed = 0
        self.rejected = 0

    @property
    def rejection_rate(self) -> float:
        return self.rejected / self.reviewed if self.reviewed else 0.0

    @property
    def should_retire(self) -> bool:
        return self.reviewed >= 10 and self.rejection_rate < self.retirement_floor

    def argue(self, demand: Demand, proposed: Rung) -> tuple[Rung, str]:
        """Return the lowest rung that plausibly solves it, and the argument."""
        self.reviewed += 1
        lowest, reasons = which(demand)
        if lowest is not proposed:
            self.rejected += 1
            return lowest, (
                f"the proposal asks for {proposed.value} but the evidence supports "
                f"{lowest.value}: {'; '.join(reasons)}"
            )
        return proposed, f"{proposed.value} is the smallest thing that solves it"


class AgentFactory:
    """Charter review, overlap detection and the commitment ladder."""

    def __init__(
        self,
        registry: AgentRegistry,
        *,
        overlap_threshold: float = OVERLAP_THRESHOLD,
        skeptic: Skeptic | None = None,
    ) -> None:
        self.registry = registry
        self.overlap_threshold = overlap_threshold
        self.skeptic = skeptic or Skeptic()

    # -- review ------------------------------------------------------------

    def review(
        self,
        charter: Charter,
        demand: Demand,
        *,
        parent_authority: AuthorityEnvelope | None = None,
        proposed_rung: Rung | None = None,
    ) -> Verdict:
        rung, _ = which(demand)
        proposed = proposed_rung or rung
        rung, argument = self.skeptic.argue(demand, proposed)

        verdict = Verdict(
            charter_id=charter.id,
            rung=rung,
            approved=False,
            skeptic_position=argument,
        )

        # Mechanical overlap detection, before anyone reads the prose.
        verdict.overlap = self.registry.overlap(charter.capabilities)
        worst = verdict.worst_overlap
        if worst and worst[1] > self.overlap_threshold:
            verdict.blockers.append(
                f"capability overlap {worst[1]:.2f} with {worst[0]} exceeds "
                f"{self.overlap_threshold:.2f} — extend that agent instead"
            )

        if not rung.needs_charter:
            verdict.blockers.append(
                f"the evidence supports {rung.value} ({rung.description}); "
                f"no agent should be created"
            )

        # Mandatory retirement criteria and eval suite. A fleet that can only
        # grow is a fleet that will.
        if not charter.retirement_criteria.strip():
            verdict.blockers.append("no retirement criteria")
        if not charter.eval_suite:
            verdict.blockers.append("no eval suite")

        if not charter.capabilities:
            verdict.blockers.append("charter declares no capabilities")
        if not charter.job.strip():
            verdict.blockers.append("charter has no job description")

        # Authority must be containable by the parent that will delegate to it.
        if parent_authority is not None:
            effective = parent_authority.intersect(charter.authority)
            missing = sorted(charter.authority.permissions - effective.permissions)
            if missing:
                verdict.blockers.append(
                    f"charter claims authority no parent holds: {missing}"
                )
        if charter.authority.max_cost_usd <= 0:
            verdict.warnings.append("no budget ceiling declared; it will get zero")

        if self.skeptic.should_retire:
            verdict.warnings.append(
                f"the skeptic's rejection rate has fallen to "
                f"{self.skeptic.rejection_rate:.0%}; it should retire"
            )

        verdict.approved = not verdict.blockers
        return verdict

    # -- commit ------------------------------------------------------------

    def commit(
        self,
        charter: Charter,
        verdict: Verdict,
        *,
        parent_authority: AuthorityEnvelope | None = None,
        human_approved: bool = False,
    ) -> AgentCard:
        """Create the agent. A human is always in this path.

        The new agent enters at zero trust regardless of how good its charter
        was. Trust is measured, and nothing has been measured yet.
        """
        if not verdict.approved:
            raise CharterError(
                f"cannot commit a blocked charter: {'; '.join(verdict.blockers)}"
            )
        if not human_approved:
            raise CharterError(
                "charter approval is a human decision; no agent is created without it"
            )
        if self.registry.get(charter.id) is not None:
            raise CharterError(f"agent {charter.id!r} already exists")

        authority = charter.authority
        if parent_authority is not None:
            authority = parent_authority.intersect(charter.authority)
            if not authority.issubset(parent_authority):
                raise EscalationError("committed authority escapes the parent envelope")

        card = AgentCard(
            id=charter.id,
            name=charter.name,
            runtime=charter.runtime,
            capabilities=charter.capabilities,
            agent_class=charter.agent_class,
            authority=authority,
            data_classes=charter.data_classes,
            probation=Probation.PROBATION,
            department=charter.department,
            charter_id=charter.id,
            retirement_criteria=charter.retirement_criteria,
            local_inference_only=charter.local_inference_only,
            cost_hint_usd=charter.cost_hint_usd,
            latency_hint_s=charter.latency_hint_s,
        )
        return self.registry.register(card)

    def rollback(self, agent_id: str) -> AgentCard | None:
        """Undo a promotion. Every promotion is reversible in one command.

        A promotion you can undo is a decision; one you cannot is a gamble.
        """
        return self.registry.deregister(agent_id)


# --------------------------------------------------------------------------
# Entropy control
# --------------------------------------------------------------------------


@dataclass
class SweepFinding:
    kind: str
    subject: str
    detail: str
    proposal: str


def entropy_sweep(
    registry: AgentRegistry,
    *,
    usage_30d: Mapping[str, int] | None = None,
    skill_usage_30d: Mapping[str, int] | None = None,
    idle_threshold: int = 0,
) -> list[SweepFinding]:
    """Monthly sweep for idle agents, dead skills and unretirable charters.

    The Archivist proposes; it never archives. Every finding here is a
    recommendation to a human.
    """
    usage = dict(usage_30d or {})
    skill_usage = dict(skill_usage_30d or {})
    findings: list[SweepFinding] = []

    for card in registry:
        used = usage.get(card.id, 0)
        if used <= idle_threshold and card.agent_class is not AgentClass.EPHEMERAL:
            findings.append(
                SweepFinding(
                    kind="idle-agent",
                    subject=card.id,
                    detail=f"{used} task(s) in 30 days",
                    proposal="archive, or state why it must stay",
                )
            )
        if not card.retirement_criteria:
            findings.append(
                SweepFinding(
                    kind="no-retirement-criteria",
                    subject=card.id,
                    detail="the charter has no retirement criteria",
                    proposal="write them, or archive the agent",
                )
            )
        for skill in card.skills:
            key = f"{card.id}#{skill.name}"
            if skill_usage.get(key, 0) <= idle_threshold:
                findings.append(
                    SweepFinding(
                        kind="dead-skill",
                        subject=key,
                        detail=f"{skill_usage.get(key, 0)} invocation(s) in 30 days",
                        proposal="remove the skill",
                    )
                )
    return findings


__all__ = [
    "AgentFactory",
    "Charter",
    "Demand",
    "OVERLAP_THRESHOLD",
    "Rung",
    "Skeptic",
    "SweepFinding",
    "Verdict",
    "entropy_sweep",
    "which",
]
