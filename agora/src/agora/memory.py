"""Memory scopes and the canonicity ladder.

The architecture of memory is scope and write authority, not storage tiers.
Reads are broad; writes are narrow; the two scopes that matter most accept
proposals rather than writes.

===========  ==================  ===========================================
Scope        Agents may          Notes
===========  ==================  ===========================================
task         write               ephemeral working memory, dies with the run
agent-own    write               an agent's own notes
project      write               shared within one project
department   propose             a human promotes
company      propose             a human promotes
personal     never write         readable only by personal-domain agents,
                                 and only under local inference
===========  ==================  ===========================================

Agents write at ``observed``; only a human promotes to ``canonical``. This
prevents the dominant failure of company memory -- a plausible inference
hardening into fact and resurfacing in a pricing decision six months later
with no traceable origin.

Contradiction is detected on *write*, never on read. A contradiction found at
read time has already been acted on.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .errors import MemoryScopeError


class Scope(str, Enum):
    TASK = "task"
    AGENT_OWN = "agent-own"
    PROJECT = "project"
    DEPARTMENT = "department"
    COMPANY = "company"
    PERSONAL = "personal"

    @property
    def agent_writable(self) -> bool:
        return self in (Scope.TASK, Scope.AGENT_OWN, Scope.PROJECT)

    @property
    def proposal_only(self) -> bool:
        return self in (Scope.DEPARTMENT, Scope.COMPANY)

    @property
    def human_only(self) -> bool:
        return self is Scope.PERSONAL


class Canonicity(str, Enum):
    """How much weight a memory may carry when cited."""

    OBSERVED = "observed"      # an agent saw something
    ASSERTED = "asserted"      # an agent concluded something
    VERIFIED = "verified"      # checked against a source or a second path
    CANONICAL = "canonical"    # a human signed off

    @property
    def rank(self) -> int:
        return [
            Canonicity.OBSERVED,
            Canonicity.ASSERTED,
            Canonicity.VERIFIED,
            Canonicity.CANONICAL,
        ].index(self)

    @property
    def citable_as_fact(self) -> bool:
        return self is Canonicity.CANONICAL

    @property
    def usage(self) -> str:
        return {
            Canonicity.OBSERVED: "cite with a hedge",
            Canonicity.ASSERTED: "cite with a hedge",
            Canonicity.VERIFIED: "citable",
            Canonicity.CANONICAL: "citable as fact",
        }[self]


#: The highest level an agent may write at, unaided. Everything above needs
#: either a verifier (VERIFIED) or a person (CANONICAL).
AGENT_WRITE_CEILING = Canonicity.ASSERTED


@dataclass
class Memory:
    """One remembered thing, with its origin attached permanently."""

    key: str
    value: Any
    scope: Scope
    canonicity: Canonicity = Canonicity.OBSERVED
    author: str = ""
    task_id: str | None = None
    verified_by: str | None = None
    promoted_by: str | None = None
    at: float = 0.0
    id: str = field(default_factory=lambda: f"mem-{uuid.uuid4().hex[:10]}")
    supersedes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.scope = Scope(self.scope)
        self.canonicity = Canonicity(self.canonicity)

    def cite(self) -> str:
        hedge = "" if self.canonicity.citable_as_fact else " (unconfirmed)"
        return f"{self.value}{hedge} [{self.scope.value}/{self.canonicity.value}]"


@dataclass
class Proposal:
    """A write an agent wants into a scope it may not write to."""

    memory: Memory
    proposer: str
    rationale: str = ""
    id: str = field(default_factory=lambda: f"prop-{uuid.uuid4().hex[:10]}")
    decided: bool = False
    accepted: bool = False
    decided_by: str | None = None


@dataclass
class Contradiction:
    """An incoming write that conflicts with something already held."""

    incoming: Memory
    existing: Memory
    detail: str

    def describe(self) -> str:
        return (
            f"{self.incoming.key!r}: incoming {self.incoming.value!r} contradicts "
            f"{self.existing.canonicity.value} {self.existing.value!r} — {self.detail}"
        )


class ContradictionError(MemoryScopeError):
    def __init__(self, contradiction: Contradiction) -> None:
        super().__init__(contradiction.describe())
        self.contradiction = contradiction


class MemoryStore:
    """Scoped memory with write authority, canonicity and write-time conflict
    detection."""

    def __init__(self) -> None:
        self._items: dict[str, Memory] = {}
        self._by_key: dict[tuple[Scope, str], list[str]] = {}
        self.proposals: list[Proposal] = []

    # -- reading -----------------------------------------------------------

    def read(
        self,
        key: str,
        *,
        scopes: Sequence[Scope] = (
            Scope.TASK,
            Scope.AGENT_OWN,
            Scope.PROJECT,
            Scope.DEPARTMENT,
            Scope.COMPANY,
        ),
        personal_domain_agent: bool = False,
        local_inference: bool = False,
    ) -> list[Memory]:
        """Reads are broad -- but personal memory has two locks on it."""
        out: list[Memory] = []
        for scope in scopes:
            if scope is Scope.PERSONAL and not (
                personal_domain_agent and local_inference
            ):
                continue
            for mem_id in self._by_key.get((scope, key), []):
                out.append(self._items[mem_id])
        return sorted(out, key=lambda m: (-m.canonicity.rank, -m.at))

    def best(self, key: str, **kwargs: Any) -> Memory | None:
        found = self.read(key, **kwargs)
        return found[0] if found else None

    def all(self, scope: Scope | None = None) -> list[Memory]:
        return [
            m for m in self._items.values() if scope is None or m.scope is Scope(scope)
        ]

    # -- writing -----------------------------------------------------------

    def write(
        self,
        memory: Memory,
        *,
        by_human: bool = False,
        force: bool = False,
    ) -> Memory:
        """Write to a scope. Refuses anything an agent may not do directly."""
        scope = memory.scope
        if not by_human:
            if scope.human_only:
                raise MemoryScopeError(
                    f"no agent writes {scope.value} memory; it is not a permissions "
                    f"setting, it is the design"
                )
            if scope.proposal_only:
                raise MemoryScopeError(
                    f"{scope.value} accepts proposals, not writes — use propose()"
                )
            if memory.canonicity.rank > AGENT_WRITE_CEILING.rank:
                if memory.canonicity is Canonicity.VERIFIED and memory.verified_by:
                    pass  # a verifier signed it, which is what VERIFIED means
                else:
                    raise MemoryScopeError(
                        f"an agent may write at most {AGENT_WRITE_CEILING.value}; "
                        f"{memory.canonicity.value} needs a verifier or a human"
                    )

        if not force:
            conflict = self.detect_contradiction(memory)
            if conflict is not None:
                raise ContradictionError(conflict)

        self._items[memory.id] = memory
        self._by_key.setdefault((memory.scope, memory.key), []).append(memory.id)
        return memory

    def propose(
        self, memory: Memory, proposer: str, rationale: str = ""
    ) -> Proposal:
        """Offer a write into a scope that only humans commit to."""
        proposal = Proposal(memory=memory, proposer=proposer, rationale=rationale)
        self.proposals.append(proposal)
        return proposal

    def decide(
        self, proposal: Proposal, *, accept: bool, by: str, promote: bool = False
    ) -> Memory | None:
        """A human decides. Optionally promotes to canonical in the same act."""
        proposal.decided = True
        proposal.accepted = accept
        proposal.decided_by = by
        if not accept:
            return None
        memory = proposal.memory
        if promote:
            memory.canonicity = Canonicity.CANONICAL
            memory.promoted_by = by
        return self.write(memory, by_human=True, force=True)

    def promote(self, memory_id: str, *, by: str) -> Memory:
        """Promote to canonical. Only a human is ever in this path."""
        memory = self._items[memory_id]
        memory.canonicity = Canonicity.CANONICAL
        memory.promoted_by = by
        return memory

    def mark_verified(self, memory_id: str, *, verifier: str) -> Memory:
        memory = self._items[memory_id]
        if memory.canonicity.rank < Canonicity.VERIFIED.rank:
            memory.canonicity = Canonicity.VERIFIED
            memory.verified_by = verifier
        return memory

    # -- contradiction -----------------------------------------------------

    def detect_contradiction(self, incoming: Memory) -> Contradiction | None:
        """Detect conflict at write time, never at read time.

        A contradiction discovered at read time has already been acted on.
        """
        for scope in (Scope.COMPANY, Scope.DEPARTMENT, incoming.scope):
            for mem_id in self._by_key.get((scope, incoming.key), []):
                existing = self._items[mem_id]
                if existing.id == incoming.id:
                    continue
                if _values_agree(existing.value, incoming.value):
                    continue
                if existing.canonicity.rank >= incoming.canonicity.rank:
                    return Contradiction(
                        incoming=incoming,
                        existing=existing,
                        detail=(
                            f"the held value is {existing.canonicity.value} and the "
                            f"incoming one is only {incoming.canonicity.value}"
                        ),
                    )
        return None

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Memory]:
        return iter(self._items.values())


def _values_agree(a: Any, b: Any) -> bool:
    if a == b:
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 1e-9
    return _normalise(a) == _normalise(b)


def _normalise(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


__all__ = [
    "AGENT_WRITE_CEILING",
    "Canonicity",
    "Contradiction",
    "ContradictionError",
    "Memory",
    "MemoryStore",
    "Proposal",
    "Scope",
]
