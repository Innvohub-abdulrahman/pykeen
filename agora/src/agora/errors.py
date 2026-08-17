"""Exception hierarchy for AGORA.

Every failure mode the control plane can enforce has its own exception, so
callers can distinguish "the agent did something it was not allowed to do"
from "the agent ran out of money" from "the agent produced a bad artifact".
"""

from __future__ import annotations


class AgoraError(Exception):
    """Base class for every AGORA error."""


class SpecificationError(AgoraError):
    """A task contract is not well formed enough to be executed.

    Raised by the specification gate when acceptance criteria are not
    machine-checkable. The system refuses such tasks rather than attempting
    them: if you cannot say how you would check the result, delegating it
    only moves the ambiguity somewhere more expensive.
    """


class AuthorityError(AgoraError):
    """An action was attempted outside the acting envelope."""


class EscalationError(AuthorityError):
    """A child envelope tried to claim authority no ancestor holds.

    This is raised at *commit* time, not at use time: authority narrowing is
    structural, so an escalating envelope can never be constructed at all.
    """


class BudgetExceeded(AgoraError):
    """A charge would take a budget past its ceiling.

    Budgets are physical. There is no ``onExceed: continue``.
    """


class TerminationError(AgoraError):
    """A run hit a termination guarantee enforced by the runtime."""


class DepthExceeded(TerminationError):
    """Delegation went deeper than the contract's depth cap."""


class DeadlineExceeded(TerminationError):
    """A run passed its wall-clock deadline."""


class NoProgressError(TerminationError):
    """A run produced no new blackboard state for too many turns."""


class CycleDetected(TerminationError):
    """Delegation revisited an ancestor task."""


class VerificationFailed(AgoraError):
    """An artifact failed the verification ladder."""


class RoutingError(AgoraError):
    """No agent could be routed to a task."""


class NoRouteAvailable(RoutingError):
    """The policy filter or the ledger left no eligible agent."""


class PolicyViolation(AgoraError):
    """The policy engine denied an action."""


class MemoryScopeError(AgoraError):
    """An agent tried to write to a scope it may only propose into."""


class LedgerError(AgoraError):
    """An invalid ledger operation, e.g. recording an unverified outcome."""


class CharterError(AgoraError):
    """An agent charter failed validation."""


class AdapterError(AgoraError):
    """A runtime adapter failed to satisfy the universal socket contract."""


class AuditChainBroken(AgoraError):
    """The hash chain over the audit log does not verify."""
