"""The universal socket: an A2A adapter template.

This is how anything joins the fleet. Four blocks to fill in; everything else
-- AgentCard emission, task lifecycle, local enforcement, provenance tagging,
the untrusted boundary -- is done for you.

=====================================  ==========================  =======
What you are adding                    Path                        Effort
=====================================  ==========================  =======
Your own runtime                       four ADAPT ME blocks        ~1 day
A CLI agent (Claude Code)              wrap the process stream     ~1 day
Another framework (LangGraph, CrewAI)  wrap their invocation API   ~1 day
A no-code workflow (n8n, Zapier)       wrap the webhook            hours
A bare MCP server                      register tools directly     minutes
A REST API or internal service         thin adapter                hours
A human specialist                     open a queue item and wait  hours
=====================================  ==========================  =======

The last row is the one that matters. Modelling a human as an agent -- with an
authority envelope, a latency profile and a ledger entry -- means Grade C and
Grade D work routes through the same machinery as everything else. Your lawyer
becomes a capability with measured turnaround. The system does not have
separate "automated" and "manual" modes; it has one mode with different
assurance grades.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence

from .artifacts import Artifact, Source, Trust, strip_instruction_shaped
from .contracts import AuthorityEnvelope, Budget, TaskContract
from .errors import AdapterError, AuthorityError, BudgetExceeded
from .registry import AgentCard, AgentClass, Probation


class TaskState(str, Enum):
    """A2A task lifecycle."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
            TaskState.REJECTED,
        )


class Performative(str, Enum):
    """Typed messages. Agents never call each other directly.

    A request for help is a performative addressed to the control plane, which
    decides who answers. Point-to-point agent chatter is how a fleet becomes
    unauditable.
    """

    DELEGATE = "delegate"
    CONSULT = "consult"
    INFORM = "inform"
    CONTEST = "contest"
    ESCALATE = "escalate"
    COMPLETE = "complete"
    FAIL = "fail"


@dataclass
class Message:
    performative: Performative
    sender: str
    recipient: str
    body: Mapping[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    turn: int = 0
    id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex[:10]}")

    def __post_init__(self) -> None:
        self.performative = Performative(self.performative)


@dataclass
class TaskResult:
    """What an adapter hands back to the control plane."""

    state: TaskState
    artifact: Artifact | None = None
    cost_usd: float = 0.0
    latency_s: float = 0.0
    turns: int = 0
    error: str | None = None
    messages: list[Message] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.state is TaskState.COMPLETED and self.artifact is not None


@dataclass
class Invocation:
    """Everything the underlying runtime is told about the job.

    Deliberately *not* the transcript. Compiled context, references rather
    than payloads: the two levers that account for most of the cost reduction
    in :mod:`agora.economics`.
    """

    task_id: str
    capability: str
    goal: str
    inputs: Mapping[str, Any]
    references: Sequence[Mapping[str, Any]]
    authority: AuthorityEnvelope
    budget_usd: float
    deadline_s: float
    acceptance: Sequence[str]
    turn_cap: int


# --------------------------------------------------------------------------
# The adapter
# --------------------------------------------------------------------------


class A2AAdapter(ABC):
    """Subclass this and fill in four blocks.

    Local enforcement is not optional and not overridable. Every adapter
    checks authority before invoking, charges the budget after, and refuses to
    exceed the turn cap -- whether or not the wrapped runtime has any concept
    of those things. A runtime that ignores its budget is contained by the
    adapter rather than trusted to behave.
    """

    #: Adapters declare their runtime name so the registry can group them.
    runtime: str = "custom"

    def __init__(
        self,
        agent_id: str,
        *,
        capabilities: Sequence[str] = (),
        cost_hint_usd: float = 0.05,
        latency_hint_s: float = 20.0,
        turn_cap: int = 8,
        agent_class: AgentClass = AgentClass.PERSISTENT,
        trust: Trust = Trust.SECONDARY,
    ) -> None:
        self.agent_id = agent_id
        self.capabilities = tuple(capabilities)
        self.cost_hint_usd = cost_hint_usd
        self.latency_hint_s = latency_hint_s
        self.turn_cap = turn_cap
        self.agent_class = agent_class
        self.trust = trust
        self.state = TaskState.SUBMITTED
        self._turns = 0

    # ==== ADAPT ME 1 of 4 =================================================
    @abstractmethod
    def describe(self) -> Mapping[str, Any]:
        """Return what your runtime can do, as advisory metadata.

        This becomes the AgentCard. Nothing here is routed on -- it makes the
        agent eligible to be considered, and the ledger decides the rest.
        """

    # ==== ADAPT ME 2 of 4 =================================================
    @abstractmethod
    def invoke(self, invocation: Invocation) -> Any:
        """Call your runtime and return whatever it natively returns."""

    # ==== ADAPT ME 3 of 4 =================================================
    @abstractmethod
    def to_artifact(self, raw: Any, invocation: Invocation) -> Artifact:
        """Turn your runtime's native output into an Artifact."""

    # ==== ADAPT ME 4 of 4 =================================================
    def cancel(self, task_id: str) -> bool:
        """Stop an in-flight task. Default: report that you cannot.

        Returning ``False`` is honest and safe -- the control plane will stop
        routing to a runtime it cannot cancel for long-running work.
        """
        return False

    # ======================================================================
    # Everything below is done for you.
    # ======================================================================

    def agent_card(self, **overrides: Any) -> AgentCard:
        described = dict(self.describe())
        return AgentCard(
            id=self.agent_id,
            name=described.get("name", self.agent_id),
            runtime=self.runtime,
            url=described.get("url"),
            capabilities=tuple(described.get("capabilities", self.capabilities)),
            advisory_only=True,
            agent_class=self.agent_class,
            side_effects=bool(described.get("side_effects", False)),
            probation=Probation.ZERO_TRUST,
            cost_hint_usd=described.get("cost_hint_usd", self.cost_hint_usd),
            latency_hint_s=described.get("latency_hint_s", self.latency_hint_s),
            **overrides,
        )

    def cost_of(self, raw: Any, invocation: Invocation) -> float:
        """Actual cost of a run. Override when your runtime reports it."""
        if isinstance(raw, Mapping) and "cost_usd" in raw:
            return float(raw["cost_usd"])
        return self.cost_hint_usd

    # -- the enforced path -------------------------------------------------

    def execute(
        self,
        contract: TaskContract,
        *,
        references: Sequence[Mapping[str, Any]] = (),
        budget: Budget | None = None,
    ) -> TaskResult:
        """Run one task under enforced authority, budget and turn cap."""
        budget = budget or contract.budget
        if budget is None:
            raise AdapterError("a task contract must carry a budget to be executed")

        if not self._declares(contract.capability):
            self.state = TaskState.REJECTED
            return TaskResult(
                state=TaskState.REJECTED,
                error=f"{self.agent_id} does not declare {contract.capability!r}",
            )

        # Local enforcement, before the runtime is touched at all.
        if not contract.authority.permits(contract.capability):
            self.state = TaskState.REJECTED
            return TaskResult(
                state=TaskState.REJECTED,
                error=(
                    f"capability {contract.capability!r} is outside the granted "
                    f"authority envelope"
                ),
            )
        if self._turns >= self.turn_cap:
            self.state = TaskState.FAILED
            return TaskResult(state=TaskState.FAILED, error="turn cap exhausted")

        invocation = Invocation(
            task_id=contract.id,
            capability=contract.capability,
            goal=contract.goal,
            inputs=dict(contract.inputs),
            references=list(references),
            authority=contract.authority,
            budget_usd=budget.available_usd,
            deadline_s=contract.termination.remaining_s(),
            acceptance=[c.description for c in contract.acceptance],
            turn_cap=self.turn_cap - self._turns,
        )

        self.state = TaskState.WORKING
        started = time.monotonic()
        try:
            raw = self.invoke(invocation)
        except Exception as exc:  # a runtime failure is a task failure, not a crash
            self.state = TaskState.FAILED
            return TaskResult(
                state=TaskState.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                latency_s=time.monotonic() - started,
            )
        finally:
            self._turns += 1

        latency = time.monotonic() - started
        cost = self.cost_of(raw, invocation)
        try:
            budget.charge(cost, reason=f"{self.agent_id}/{contract.capability}")
        except BudgetExceeded as exc:
            self.state = TaskState.FAILED
            return TaskResult(
                state=TaskState.FAILED, error=str(exc), latency_s=latency, cost_usd=0.0
            )

        artifact = self.to_artifact(raw, invocation)
        artifact = self._tag_provenance(artifact, contract)

        # The untrusted boundary is crossed here, on the way back in.
        artifact = artifact.sanitized()

        self.state = TaskState.COMPLETED
        return TaskResult(
            state=TaskState.COMPLETED,
            artifact=artifact,
            cost_usd=cost,
            latency_s=latency,
            turns=self._turns,
        )

    # -- helpers -----------------------------------------------------------

    def _declares(self, capability: str) -> bool:
        declared = tuple(self.describe().get("capabilities", self.capabilities))
        return capability in declared

    def _tag_provenance(self, artifact: Artifact, contract: TaskContract) -> Artifact:
        artifact.producer = artifact.producer or self.agent_id
        artifact.capability = artifact.capability or contract.capability
        artifact.task_id = contract.id
        artifact.metadata.setdefault("runtime", self.runtime)
        artifact.metadata.setdefault("authority", contract.authority.describe())
        return artifact

    def consult(self, question: str, *, task_id: str | None = None) -> Message:
        """Ask the control plane, not another agent."""
        return Message(
            performative=Performative.CONSULT,
            sender=self.agent_id,
            recipient="control-plane",
            body={"question": question},
            task_id=task_id,
        )


# --------------------------------------------------------------------------
# Reference adapters
# --------------------------------------------------------------------------


class CallableAdapter(A2AAdapter):
    """Wraps any Python callable. The thin end of the wedge.

    A REST client, an MCP tool, an internal service, or a test double all fit
    here without ceremony.
    """

    runtime = "callable"

    def __init__(
        self,
        agent_id: str,
        fn: Callable[[Invocation], Any],
        *,
        capabilities: Sequence[str] = (),
        to_artifact_fn: Callable[[Any, Invocation], Artifact] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(agent_id, capabilities=capabilities, **kwargs)
        self.fn = fn
        self._to_artifact = to_artifact_fn

    def describe(self) -> Mapping[str, Any]:
        return {"name": self.agent_id, "capabilities": self.capabilities}

    def invoke(self, invocation: Invocation) -> Any:
        return self.fn(invocation)

    def to_artifact(self, raw: Any, invocation: Invocation) -> Artifact:
        if self._to_artifact is not None:
            return self._to_artifact(raw, invocation)
        if isinstance(raw, Artifact):
            return raw
        if isinstance(raw, Mapping) and "content" in raw:
            return Artifact(
                content=str(raw["content"]),
                capability=invocation.capability,
                metadata={k: v for k, v in raw.items() if k != "content"},
            )
        return Artifact(content=str(raw), capability=invocation.capability)


class ProcessAdapter(A2AAdapter):
    """Wraps a CLI agent by driving its process stream.

    Supply ``runner`` -- anything that takes an argv list and returns stdout.
    Keeping the process launch injectable is what makes this testable without
    a subprocess, and what lets a customer swap in their own sandbox.
    """

    runtime = "process"

    def __init__(
        self,
        agent_id: str,
        argv_template: Sequence[str],
        runner: Callable[[Sequence[str]], str],
        *,
        capabilities: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(agent_id, capabilities=capabilities, **kwargs)
        self.argv_template = list(argv_template)
        self.runner = runner

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.agent_id,
            "capabilities": self.capabilities,
            "side_effects": True,
        }

    def invoke(self, invocation: Invocation) -> Any:
        argv = [
            part.format(
                goal=invocation.goal,
                capability=invocation.capability,
                budget=f"{invocation.budget_usd:.2f}",
                task_id=invocation.task_id,
            )
            for part in self.argv_template
        ]
        return self.runner(argv)

    def to_artifact(self, raw: Any, invocation: Invocation) -> Artifact:
        return Artifact(content=str(raw), capability=invocation.capability)


class WebhookAdapter(A2AAdapter):
    """Wraps a no-code workflow behind a webhook."""

    runtime = "webhook"

    def __init__(
        self,
        agent_id: str,
        post: Callable[[str, Mapping[str, Any]], Mapping[str, Any]],
        url: str,
        *,
        capabilities: Sequence[str] = (),
        **kwargs: Any,
    ) -> None:
        super().__init__(agent_id, capabilities=capabilities, **kwargs)
        self.post = post
        self.url = url

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.agent_id,
            "capabilities": self.capabilities,
            "url": self.url,
        }

    def invoke(self, invocation: Invocation) -> Any:
        return self.post(
            self.url,
            {
                "task_id": invocation.task_id,
                "goal": invocation.goal,
                "inputs": dict(invocation.inputs),
                "references": list(invocation.references),
                "budget_usd": invocation.budget_usd,
            },
        )

    def to_artifact(self, raw: Any, invocation: Invocation) -> Artifact:
        body = raw.get("output", raw) if isinstance(raw, Mapping) else raw
        return Artifact(content=str(body), capability=invocation.capability)


@dataclass
class QueueItem:
    id: str
    task_id: str
    goal: str
    acceptance: Sequence[str]
    assignee: str
    answer: str | None = None
    answered_at: float | None = None


class HumanAdapter(A2AAdapter):
    """A human specialist, modelled as an agent.

    Opens a queue item and waits. The person gets an authority envelope, a
    latency profile and a ledger entry like anything else, which is what lets
    Grade C and Grade D work run through exactly the same machinery -- routed,
    budgeted, verified and audited -- instead of falling off the edge of the
    system into a "manual process".
    """

    runtime = "human"

    def __init__(
        self,
        agent_id: str,
        queue: list[QueueItem],
        *,
        capabilities: Sequence[str] = (),
        answer_fn: Callable[[QueueItem], str | None] | None = None,
        latency_hint_s: float = 3600.0,
        cost_hint_usd: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            agent_id,
            capabilities=capabilities,
            latency_hint_s=latency_hint_s,
            cost_hint_usd=cost_hint_usd,
            **kwargs,
        )
        self.queue = queue
        self.answer_fn = answer_fn

    def describe(self) -> Mapping[str, Any]:
        return {
            "name": self.agent_id,
            "capabilities": self.capabilities,
            "human": True,
            "latency_hint_s": self.latency_hint_s,
        }

    def invoke(self, invocation: Invocation) -> Any:
        item = QueueItem(
            id=f"q-{uuid.uuid4().hex[:8]}",
            task_id=invocation.task_id,
            goal=invocation.goal,
            acceptance=list(invocation.acceptance),
            assignee=self.agent_id,
        )
        self.queue.append(item)
        answer = self.answer_fn(item) if self.answer_fn else None
        if answer is None:
            # Nobody has answered yet. This is a normal state, not an error.
            self.state = TaskState.INPUT_REQUIRED
            raise AdapterError(f"queued for {self.agent_id}; awaiting a human")
        item.answer = answer
        item.answered_at = time.time()
        return answer

    def to_artifact(self, raw: Any, invocation: Invocation) -> Artifact:
        return Artifact(
            content=str(raw),
            capability=invocation.capability,
            sources=(
                Source(
                    id="human",
                    text=str(raw),
                    trust=Trust.PRIMARY,
                ),
            ),
        )


__all__ = [
    "A2AAdapter",
    "CallableAdapter",
    "HumanAdapter",
    "Invocation",
    "Message",
    "Performative",
    "ProcessAdapter",
    "QueueItem",
    "TaskResult",
    "TaskState",
    "WebhookAdapter",
]
