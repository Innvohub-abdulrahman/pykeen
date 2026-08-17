"""The agent registry: extended A2A AgentCards.

A card says what an agent *claims*. Those claims are marked advisory and are
never routed on -- they are how an agent becomes eligible to be considered,
and nothing more. What decides routing is the Capability Ledger.

That separation is the whole point. Every registry in this category lets a
component describe itself and then takes the description at face value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator, Mapping, Sequence

from .contracts import AuthorityEnvelope, DataClass


class AgentClass(str, Enum):
    """Three classes, and conflating them causes most fleet design errors."""

    CONTROL_PLANE = "control-plane"   # planner, archivist, skeptic
    PERSISTENT = "persistent"          # a department agent with continuity
    EPHEMERAL = "ephemeral"            # spawned, does one job, dies


class Probation(str, Enum):
    """A new runtime enters at zero trust and graduates on measured evidence."""

    ZERO_TRUST = "zero-trust"     # sandbox only, verification always
    PROBATION = "probation"        # real traffic, low stakes, verified always
    GRADUATED = "graduated"        # routed normally
    RETIRED = "retired"

    @property
    def may_take_real_traffic(self) -> bool:
        return self in (Probation.PROBATION, Probation.GRADUATED)


@dataclass
class Skill:
    """A procedure an agent can run, nested under it in the console."""

    name: str
    tier: str = "T0"  # T0 declarative · T1 sandboxed · T2 side-effecting
    evals: str | None = None
    promoted: bool = False
    description: str = ""

    @property
    def needs_human_review(self) -> bool:
        return self.tier == "T2"


@dataclass
class AgentCard:
    """An extended A2A AgentCard.

    ``capabilities`` is advisory: it is a claim by the agent about itself, and
    the field name is deliberately paired with ``advisory_only=True`` so that
    nothing downstream can mistake it for evidence.
    """

    id: str
    name: str
    runtime: str = "unknown"
    url: str | None = None
    capabilities: tuple[str, ...] = ()
    advisory_only: bool = True
    agent_class: AgentClass = AgentClass.PERSISTENT
    authority: AuthorityEnvelope = field(default_factory=AuthorityEnvelope)
    data_classes: frozenset[DataClass] = frozenset({DataClass.INTERNAL})
    residency: frozenset[str] = frozenset({"*"})
    side_effects: bool = False
    local_inference_only: bool = False
    skills: list[Skill] = field(default_factory=list)
    probation: Probation = Probation.ZERO_TRUST
    cost_hint_usd: float = 0.05
    latency_hint_s: float = 20.0
    department: str = ""
    charter_id: str | None = None
    retirement_criteria: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.capabilities = tuple(self.capabilities)
        self.agent_class = AgentClass(self.agent_class)
        self.probation = Probation(self.probation)
        self.data_classes = frozenset(DataClass(d) for d in self.data_classes)
        self.residency = frozenset(self.residency)

    def declares(self, capability: str) -> bool:
        return capability in self.capabilities

    def skill(self, name: str) -> Skill | None:
        return next((s for s in self.skills if s.name == name), None)

    def add_skill(self, skill: Skill) -> None:
        if self.skill(skill.name):
            raise ValueError(f"{self.id} already has a skill named {skill.name!r}")
        self.skills.append(skill)


class AgentRegistry:
    """Who exists, what they claim, and what they are permitted to touch."""

    def __init__(self, cards: Iterable[AgentCard] = ()) -> None:
        self._cards: dict[str, AgentCard] = {}
        for card in cards:
            self.register(card)

    def register(self, card: AgentCard) -> AgentCard:
        if card.id in self._cards:
            raise ValueError(f"agent {card.id!r} is already registered")
        self._cards[card.id] = card
        return card

    def deregister(self, agent_id: str) -> AgentCard | None:
        return self._cards.pop(agent_id, None)

    def get(self, agent_id: str) -> AgentCard | None:
        return self._cards.get(agent_id)

    def all(self) -> list[AgentCard]:
        return list(self._cards.values())

    def __iter__(self) -> Iterator[AgentCard]:
        return iter(self._cards.values())

    def __len__(self) -> int:
        return len(self._cards)

    def declaring(self, capability: str) -> list[AgentCard]:
        """Eligible to be *considered*. Not a statement about competence."""
        return [c for c in self._cards.values() if c.declares(capability)]

    def by_runtime(self, runtime: str) -> list[AgentCard]:
        return [c for c in self._cards.values() if c.runtime == runtime]

    def capabilities(self) -> set[str]:
        return {cap for card in self._cards.values() for cap in card.capabilities}

    def graduate(self, agent_id: str, to: Probation = Probation.GRADUATED) -> AgentCard:
        card = self._cards[agent_id]
        card.probation = Probation(to)
        return card

    def overlap(self, capabilities: Sequence[str]) -> dict[str, float]:
        """Jaccard overlap of a proposed capability set against every agent.

        Used by the agent factory to refuse a charter that duplicates an agent
        the fleet already has.
        """
        want = set(capabilities)
        scores: dict[str, float] = {}
        for card in self._cards.values():
            have = set(card.capabilities)
            union = want | have
            scores[card.id] = len(want & have) / len(union) if union else 0.0
        return scores


__all__ = [
    "AgentCard",
    "AgentClass",
    "AgentRegistry",
    "Probation",
    "Skill",
]
