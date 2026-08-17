"""AGORA -- a control plane for agent fleets.

Add any agent. Run any task. Know which ones you can trust unsupervised.

Three properties define the system:

*Measured trust.* Routing reads a Beta posterior built only from
independently verified outcomes, bucketed by context, decayed by age, shrunk
when borrowed across contexts. Self-reported success is discarded.

*Structural containment.* Authority narrows monotonically down the delegation
tree. Budgets are physical. Termination is enforced by the runtime, not by
agent cooperation.

*Bounded self-improvement.* Four feedback loops at four time constants, each
writing outward only. Nothing generated can modify the orchestrator, verifier,
policy engine or budget enforcer.
"""

from __future__ import annotations

__version__ = "2.0.0"

from .artifacts import Artifact, Source, Trust
from .contracts import (
    AcceptanceCriterion,
    AuthorityEnvelope,
    Budget,
    CheckKind,
    DataClass,
    Scope,
    TaskContract,
    TerminationPolicy,
    Tier,
)
from .errors import (
    AgoraError,
    AuthorityError,
    BudgetExceeded,
    EscalationError,
    SpecificationError,
    VerificationFailed,
)
from .tiering import TierGate
from .verification import (
    Grade,
    Level,
    PackRegistry,
    VerificationLadder,
    run_universal_floor,
)

__all__ = [
    "__version__",
    "AcceptanceCriterion",
    "AgoraError",
    "Artifact",
    "AuthorityEnvelope",
    "AuthorityError",
    "Budget",
    "BudgetExceeded",
    "CheckKind",
    "DataClass",
    "EscalationError",
    "Grade",
    "Level",
    "PackRegistry",
    "Scope",
    "Source",
    "SpecificationError",
    "TaskContract",
    "TerminationPolicy",
    "Tier",
    "TierGate",
    "Trust",
    "VerificationFailed",
    "VerificationLadder",
    "run_universal_floor",
]
