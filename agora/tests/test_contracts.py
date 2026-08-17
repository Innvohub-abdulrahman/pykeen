"""Structural containment: authority narrows, budgets are physical, tasks
without machine-checkable criteria are refused."""

from __future__ import annotations

import pytest

from agora.contracts import (
    AcceptanceCriterion,
    AuthorityEnvelope,
    Budget,
    CheckKind,
    DataClass,
    OnExceed,
    Scope,
    TaskContract,
    TerminationPolicy,
    Tier,
    specification_gate,
)
from agora.errors import (
    BudgetExceeded,
    DeadlineExceeded,
    DepthExceeded,
    EscalationError,
    SpecificationError,
)


# --------------------------------------------------------------------------
# Authority
# --------------------------------------------------------------------------


def test_intersection_narrows_on_every_axis(root_authority):
    child = root_authority.intersect(
        AuthorityEnvelope.of(
            ["fs.write", "git.commit", "net.fetch", "prod.deploy"],
            max_depth=9,
            max_cost_usd=999.0,
            data_classes=[DataClass.PUBLIC, DataClass.RESTRICTED],
        )
    )
    assert "prod.deploy" not in child.permissions, "no ancestor holds prod.deploy"
    assert child.max_depth < root_authority.max_depth
    assert child.max_cost_usd <= root_authority.max_cost_usd
    assert child.data_classes <= root_authority.data_classes
    assert child.issubset(root_authority)


def test_escalation_is_structurally_impossible(root_authority):
    """A child cannot gain authority by asking for it."""
    greedy = AuthorityEnvelope.of(["prod.deploy", "payment.send"], max_cost_usd=1e9)
    effective = root_authority.intersect(greedy)
    assert effective.permissions == frozenset()
    assert not effective.permits("prod.deploy")


def test_narrow_refuses_to_widen(root_authority):
    with pytest.raises(EscalationError):
        root_authority.narrow(permissions=root_authority.permissions | {"prod.deploy"})


def test_wildcards_cover_their_namespace():
    envelope = AuthorityEnvelope.of(["fs.*"])
    assert envelope.permits("fs.write")
    assert envelope.permits("fs.read.metadata")
    assert not envelope.permits("git.commit")


def test_depth_shrinks_by_one_per_level(root_authority):
    envelope = root_authority
    depths = [envelope.max_depth]
    for _ in range(3):
        envelope = envelope.intersect(
            AuthorityEnvelope.of(["fs.write"], max_depth=99, max_cost_usd=1.0)
        )
        depths.append(envelope.max_depth)
    assert depths == sorted(depths, reverse=True)
    assert depths[-1] == 0, "a leaf may not delegate further"


def test_dropped_permissions_are_reportable(root_authority):
    child = root_authority.intersect(
        AuthorityEnvelope.of(["fs.write", "net.fetch"], max_cost_usd=4.0)
    )
    dropped = child.dropped_from(root_authority)
    assert "email.draft" in dropped and "calendar.write" in dropped


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def test_a_budget_cannot_be_overspent():
    budget = Budget(ceiling_usd=1.00)
    budget.charge(0.75)
    with pytest.raises(BudgetExceeded):
        budget.charge(0.30)
    assert budget.spent_usd == pytest.approx(0.75)


def test_on_exceed_continue_is_not_expressible():
    assert {e.value for e in OnExceed} == {"halt", "escalate"}
    with pytest.raises(ValueError):
        OnExceed("continue")


def test_reserving_a_child_debits_the_parent_immediately():
    root = Budget(ceiling_usd=10.0)
    root.reserve(4.0, label="a")
    root.reserve(4.0, label="b")
    assert root.available_usd == pytest.approx(2.0)
    with pytest.raises(BudgetExceeded):
        root.reserve(3.0, label="c")


def test_the_leaf_sum_never_exceeds_the_root():
    root = Budget(ceiling_usd=5.0)
    a = root.reserve(3.0, label="a")
    b = a.reserve(2.0, label="b")
    b.charge(1.5)
    a.charge(0.5)
    assert root.invariant_holds()
    assert root.rollup_usd() <= root.ceiling_usd


def test_releasing_returns_the_remainder():
    root = Budget(ceiling_usd=5.0)
    child = root.reserve(3.0, label="child")
    child.charge(1.0)
    returned = child.release()
    assert returned == pytest.approx(2.0)
    assert root.available_usd == pytest.approx(4.0)


# --------------------------------------------------------------------------
# The specification gate
# --------------------------------------------------------------------------


def test_a_task_with_no_criteria_is_refused():
    with pytest.raises(SpecificationError):
        specification_gate([])


def test_a_human_only_criterion_is_refused():
    with pytest.raises(SpecificationError) as exc:
        specification_gate(
            [
                AcceptanceCriterion(
                    id="vibes", kind=CheckKind.HUMAN, description="looks right to me"
                )
            ]
        )
    assert "machine-checkable" in str(exc.value)


def test_a_vague_criterion_with_no_spec_is_refused():
    with pytest.raises(SpecificationError):
        specification_gate(
            [
                AcceptanceCriterion(
                    id="quality",
                    kind=CheckKind.COMMAND,
                    description="the code should be clean and reasonable",
                )
            ]
        )


def test_a_rubric_needs_a_threshold():
    with pytest.raises(SpecificationError):
        AcceptanceCriterion(id="r", kind=CheckKind.RUBRIC, description="graded")


def test_a_decidable_criterion_passes(machine_checkable):
    specification_gate(machine_checkable)  # does not raise


# --------------------------------------------------------------------------
# Delegation and termination
# --------------------------------------------------------------------------


def test_delegation_stores_the_narrowed_envelope(root_authority, machine_checkable):
    parent = TaskContract.root(
        "build a thing",
        "code.implement",
        authority=root_authority,
        budget_usd=10.0,
        acceptance=machine_checkable,
    )
    child = parent.delegate(
        goal="research the spec",
        capability="web.research",
        declared_authority=AuthorityEnvelope.of(
            ["net.fetch", "prod.deploy"], max_depth=1, max_cost_usd=5.0
        ),
        budget_usd=0.80,
        acceptance=machine_checkable,
    )
    assert child.authority.permits("net.fetch")
    assert not child.authority.permits("prod.deploy")
    assert child.budget.ceiling_usd == pytest.approx(0.80)
    assert child.authority.max_cost_usd <= 0.80
    assert child.depth == parent.depth + 1


def test_delegation_cannot_outspend_the_parent(root_authority, machine_checkable):
    parent = TaskContract.root(
        "g", "code.implement", authority=root_authority, budget_usd=1.0,
        acceptance=machine_checkable,
    )
    with pytest.raises(BudgetExceeded):
        parent.delegate(
            goal="child",
            capability="web.research",
            declared_authority=AuthorityEnvelope.of(["net.fetch"], max_cost_usd=5.0),
            budget_usd=5.0,
        )


def test_depth_cap_is_enforced_by_the_runtime(root_authority, machine_checkable):
    contract = TaskContract.root(
        "g", "code.implement", authority=root_authority, budget_usd=10.0,
        acceptance=machine_checkable,
    )
    contract.termination.max_depth = 1
    child = contract.delegate(
        goal="c1",
        capability="web.research",
        declared_authority=AuthorityEnvelope.of(["net.fetch"], max_cost_usd=1.0),
        budget_usd=1.0,
    )
    with pytest.raises(DepthExceeded):
        child.delegate(
            goal="c2",
            capability="web.research",
            declared_authority=AuthorityEnvelope.of(["net.fetch"], max_cost_usd=0.5),
            budget_usd=0.5,
        )


def test_wall_clock_deadline_is_enforced():
    policy = TerminationPolicy(wall_clock_s=10.0, started_at=0.0)
    policy.check_deadline(now=9.0)
    with pytest.raises(DeadlineExceeded):
        policy.check_deadline(now=11.0)


# --------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------


def test_barred_beats_allowed():
    scope = Scope.of(allowed=["src/*"], barred=["src/payments/*"])
    assert scope.allows("src/app.py")
    assert not scope.allows("src/payments/charge.py")


def test_tiers_that_bypass_the_control_plane():
    assert Tier.REFLEX.bypasses_control_plane
    assert Tier.TASK.bypasses_control_plane
    assert not Tier.PROJECT.bypasses_control_plane
    assert not Tier.CAMPAIGN.bypasses_control_plane
