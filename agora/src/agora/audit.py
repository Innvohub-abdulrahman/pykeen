"""The audit ledger: a hash-chained, append-only record.

Every routing decision, every authority grant, every verified outcome and
every dollar lands here. Two properties make it worth having:

*Tamper evidence.* Each record carries the hash of the one before it. Editing
any record invalidates every hash after it, and :func:`AuditLog.verify` finds
the first break.

*Replay.* A run can be replayed against a pinned model set. Without pinning,
"we replayed it and got something else" tells you nothing about whether the
system behaved correctly, because the model changed underneath you.

Cost attribution rides along: every token is attributed to a task, an agent, a
department and a tenant, which is what makes the economics in
:mod:`agora.economics` measurable rather than modelled.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .errors import AuditChainBroken

GENESIS = "0" * 64


class EventKind(str, Enum):
    TASK_SUBMITTED = "task.submitted"
    TIER_DECIDED = "tier.decided"
    SPEC_GATE = "spec.gate"
    PLAN_SELECTED = "plan.selected"
    CONTRACT_ISSUED = "contract.issued"
    AUTHORITY_GRANTED = "authority.granted"
    ROUTE_DECIDED = "route.decided"
    ROUTE_FAILED = "route.failed"
    EXECUTION_STARTED = "execution.started"
    EXECUTION_FINISHED = "execution.finished"
    ARTIFACT_PUBLISHED = "artifact.published"
    VERIFICATION = "verification"
    LEDGER_UPDATED = "ledger.updated"
    BUDGET_CHARGED = "budget.charged"
    ESCALATED = "escalated"
    REFUSED = "refused"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled.back"


@dataclass(frozen=True)
class Record:
    """One immutable entry. ``digest`` covers the payload and the previous hash."""

    seq: int
    kind: EventKind
    at: float
    payload: Mapping[str, Any]
    prev: str
    digest: str
    task_id: str | None = None
    agent: str | None = None
    department: str | None = None
    tenant: str | None = None
    cost_usd: float = 0.0
    model_set: str = ""

    def body(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind.value,
            "at": self.at,
            "payload": _canonical(self.payload),
            "prev": self.prev,
            "task_id": self.task_id,
            "agent": self.agent,
            "department": self.department,
            "tenant": self.tenant,
            "cost_usd": round(self.cost_usd, 6),
            "model_set": self.model_set,
        }

    def compute_digest(self) -> str:
        blob = json.dumps(self.body(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> Any:
    """Make a payload JSON-serialisable without losing the shape of it."""
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [_canonical(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dataclass_fields__"):
        return _canonical(asdict(value))
    return str(value)


@dataclass
class CostRollup:
    by_task: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_agent: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_department: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_tenant: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    @property
    def total(self) -> float:
        return round(sum(self.by_task.values()), 6)


class AuditLog:
    """Append-only, hash-chained, replayable."""

    def __init__(self, *, tenant: str = "default", model_set: str = "") -> None:
        self.tenant = tenant
        self.model_set = model_set
        self._records: list[Record] = []
        self.clock: float = 0.0

    # -- writing -----------------------------------------------------------

    def append(
        self,
        kind: EventKind | str,
        payload: Mapping[str, Any] | None = None,
        *,
        task_id: str | None = None,
        agent: str | None = None,
        department: str | None = None,
        cost_usd: float = 0.0,
        at: float | None = None,
    ) -> Record:
        prev = self._records[-1].digest if self._records else GENESIS
        draft = Record(
            seq=len(self._records),
            kind=EventKind(kind),
            at=self.clock if at is None else at,
            payload=dict(payload or {}),
            prev=prev,
            digest="",
            task_id=task_id,
            agent=agent,
            department=department,
            tenant=self.tenant,
            cost_usd=float(cost_usd),
            model_set=self.model_set,
        )
        record = Record(**{**draft.__dict__, "digest": draft.compute_digest()})
        self._records.append(record)
        return record

    # -- reading -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self._records)

    def records(self) -> list[Record]:
        return list(self._records)

    def for_task(self, task_id: str) -> list[Record]:
        return [r for r in self._records if r.task_id == task_id]

    def of_kind(self, kind: EventKind | str) -> list[Record]:
        kind = EventKind(kind)
        return [r for r in self._records if r.kind is kind]

    # -- integrity ---------------------------------------------------------

    def verify(self) -> None:
        """Raise on the first broken link. Silence means the chain holds."""
        prev = GENESIS
        for record in self._records:
            if record.prev != prev:
                raise AuditChainBroken(
                    f"record {record.seq} claims prev={record.prev[:8]} but the "
                    f"chain is at {prev[:8]}"
                )
            expected = record.compute_digest()
            if expected != record.digest:
                raise AuditChainBroken(
                    f"record {record.seq} ({record.kind.value}) has been altered"
                )
            prev = record.digest

    def is_intact(self) -> bool:
        try:
            self.verify()
        except AuditChainBroken:
            return False
        return True

    @property
    def head(self) -> str:
        return self._records[-1].digest if self._records else GENESIS

    # -- attribution -------------------------------------------------------

    def costs(self) -> CostRollup:
        rollup = CostRollup()
        for record in self._records:
            if not record.cost_usd:
                continue
            rollup.by_task[record.task_id or "-"] += record.cost_usd
            rollup.by_agent[record.agent or "-"] += record.cost_usd
            rollup.by_department[record.department or "-"] += record.cost_usd
            rollup.by_tenant[record.tenant or "-"] += record.cost_usd
        return rollup

    # -- replay ------------------------------------------------------------

    def replay(self, *, model_set: str | None = None) -> "ReplayResult":
        """Reconstruct a run's decision sequence against a pinned model set.

        A replay under a different model set is not a replay. It is a new run
        that happens to start from the same inputs, and reporting it as a
        replay is how "deterministic replay" becomes a marketing claim rather
        than an engineering one.
        """
        pinned = model_set or self.model_set
        mismatched = [
            r.seq for r in self._records if r.model_set and r.model_set != pinned
        ]
        steps = [
            (r.seq, r.kind.value, r.task_id, r.agent) for r in self._records
        ]
        return ReplayResult(
            model_set=pinned,
            steps=steps,
            faithful=not mismatched and self.is_intact(),
            mismatched_records=mismatched,
        )

    # -- serialisation -----------------------------------------------------

    def to_jsonl(self) -> str:
        return "\n".join(
            json.dumps({**r.body(), "digest": r.digest}, sort_keys=True)
            for r in self._records
        )


@dataclass
class ReplayResult:
    model_set: str
    steps: list[tuple[int, str, str | None, str | None]]
    faithful: bool
    mismatched_records: list[int] = field(default_factory=list)

    def summary(self) -> str:
        if self.faithful:
            return f"replay faithful over {len(self.steps)} step(s) at {self.model_set!r}"
        return (
            f"replay NOT faithful: "
            + (
                f"{len(self.mismatched_records)} record(s) ran under a different "
                f"model set"
                if self.mismatched_records
                else "the audit chain is broken"
            )
        )


__all__ = [
    "AuditLog",
    "AuditChainBroken",
    "CostRollup",
    "EventKind",
    "Record",
    "ReplayResult",
]
