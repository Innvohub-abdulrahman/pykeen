"""The wedge. If any of these fail, nothing downstream is worth routing on.

The headline claim from the report is reproduced exactly here: eleven genuine
faults across four unrelated domains, zero false positives on a clean control,
zero model calls.
"""

from __future__ import annotations

import pytest

from agora.artifacts import Artifact, Source, Trust
from agora.contracts import AuthorityEnvelope, Scope, TaskContract
from agora.verification.corpus import all_cases, run_corpus
from agora.verification.universal import (
    UNIVERSAL_CHECKS,
    Severity,
    check_abstention,
    check_authority,
    check_consistency,
    check_provenance,
    check_scope,
    check_structure,
    run_universal_floor,
)


# --------------------------------------------------------------------------
# The corpus
# --------------------------------------------------------------------------


def test_corpus_catches_every_seeded_fault():
    report = run_corpus()
    assert report.seeded == 11, "the corpus should carry eleven seeded faults"
    assert report.caught == 11, f"missed: {report.missed}"


def test_corpus_produces_no_false_positives():
    report = run_corpus()
    assert report.false_positives == [], (
        "a noisy checker is worse than no checker, because people learn to "
        f"click past it: {report.false_positives}"
    )


def test_control_document_is_completely_clean():
    control = next(c for c in all_cases() if c.name == "control")
    floor = run_universal_floor(control.artifact, control.contract)
    assert floor.findings == [], [str(f) for f in floor.findings]
    assert floor.passed


def test_verification_costs_nothing_and_calls_no_model():
    report = run_corpus()
    assert report.model_calls == 0
    assert report.cost_usd == 0.0


def test_headline_matches_the_published_claim():
    assert run_corpus().headline() == (
        "11 GENUINE FAULTS · 0 FALSE POSITIVES · 0 MODEL CALLS"
    )


@pytest.mark.parametrize("case", [c for c in all_cases() if c.faults], ids=lambda c: c.name)
def test_each_domain_fires_only_the_checks_it_should(case):
    floor = run_universal_floor(case.artifact, case.contract)
    fired = {f.check for f in floor.findings}
    assert case.expected_checks <= fired
    assert fired <= case.expected_checks


def test_the_floor_runs_all_six_checks():
    assert len(UNIVERSAL_CHECKS) == 6
    control = next(c for c in all_cases() if c.name == "control")
    floor = run_universal_floor(control.artifact, control.contract)
    assert {r.check for r in floor.results} == set(UNIVERSAL_CHECKS)


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------


def _artifact(content: str, **kwargs) -> Artifact:
    return Artifact(content=content, **kwargs)


def test_provenance_accepts_a_figure_that_traces_to_a_source():
    artifact = _artifact(
        "Revenue was $18,500 last month.",
        sources=(Source(id="S1", text="invoiced revenue $18,500", trust=Trust.PRIMARY),),
    )
    assert check_provenance(artifact).findings == []


def test_provenance_rejects_an_invented_statistic():
    artifact = _artifact(
        "Adoption grew 47% year on year.",
        sources=(Source(id="S1", text="adoption grew", trust=Trust.PRIMARY),),
    )
    findings = check_provenance(artifact).findings
    assert len(findings) == 1
    assert "47%" in findings[0].message


def test_provenance_rejects_an_appeal_to_authority_with_no_source():
    artifact = _artifact("Studies show that teams prefer this workflow.")
    findings = check_provenance(artifact).findings
    assert any("appeal to authority" in f.message for f in findings)


def test_an_authorised_offer_is_not_a_claim_about_the_world():
    """The false positive that shipped, and the rule that fixed it.

    An offer you are permitted to make is a performative, not an assertion.
    """
    artifact = _artifact("We can apply the approved 15% discount on annual prepay.")
    contract = TaskContract(
        goal="reply",
        capability="support.draft",
        granted_offers=("15% discount on annual prepay",),
    )
    assert check_provenance(artifact, contract).findings == []


def test_the_same_offer_is_a_violation_without_the_grant():
    artifact = _artifact("I can offer you a 15% discount on annual prepay.")
    bare = TaskContract(goal="reply", capability="support.draft")
    assert check_authority(artifact, bare).findings != []


def test_ordinals_and_section_numbers_are_not_claims():
    artifact = _artifact("Phase 2 covers section 4 of version 3 of the plan.")
    assert check_provenance(artifact).findings == []


def test_scope_blocks_a_barred_target():
    artifact = _artifact("done", touched=("payments/ledger.py",))
    contract = TaskContract(
        goal="g",
        capability="code.implement",
        scope=Scope.of(allowed=["src/*"], barred=["payments/*"]),
    )
    findings = check_scope(artifact, contract).findings
    assert any(f.severity is Severity.BLOCK for f in findings)


def test_scope_is_skipped_when_nothing_is_declared():
    result = check_scope(_artifact("done"), TaskContract(goal="g", capability="c"))
    assert result.skipped
    assert result.passed


@pytest.mark.parametrize(
    "text,permission",
    [
        ("I can offer you a 30% discount if you sign this quarter.", "commerce.offer_discount"),
        ("We guarantee delivery before your board meeting.", "commit.delivery"),
        ("I'll have the signed order form to you by Friday.", "commit.deadline"),
        ("You should sign this as drafted.", "legal.advise"),
    ],
)
def test_authority_catches_commitments_nobody_granted(text, permission):
    contract = TaskContract(
        goal="g", capability="c", authority=AuthorityEnvelope.of(["doc.write"])
    )
    findings = check_authority(_artifact(text), contract).findings
    assert findings, f"{text!r} should have tripped {permission}"
    assert any(permission in f.message for f in findings)


def test_authority_permits_what_the_envelope_grants():
    contract = TaskContract(
        goal="g",
        capability="c",
        authority=AuthorityEnvelope.of(["commerce.offer_discount"]),
    )
    assert check_authority(_artifact("I can offer a 30% discount."), contract).findings == []


def test_consistency_catches_a_headline_the_body_contradicts():
    artifact = _artifact(
        "Onboarding time cut by 60% in the first week.\n"
        "Onboarding time drops by 45% for most teams.\n"
    )
    findings = check_consistency(artifact).findings
    assert len(findings) == 1
    assert "60%" in findings[0].message and "45%" in findings[0].message


def test_consistency_does_not_flag_values_scoped_to_different_periods():
    """Two figures with the same label and different months are data, not a
    contradiction. This distinction is what keeps the check usable."""
    artifact = _artifact(
        "March invoiced revenue was $18,500.\nApril invoiced revenue was $21,300.\n"
    )
    assert check_consistency(artifact).findings == []


def test_structure_requires_declared_sections():
    artifact = _artifact("# Summary\nbody text\n")
    contract = TaskContract(
        goal="g", capability="c", required_sections=("Summary", "Risks")
    )
    findings = check_structure(artifact, contract).findings
    assert len(findings) == 1
    assert "Risks" in findings[0].message


def test_abstention_blocks_an_invented_value():
    artifact = _artifact(
        "The governing law is English law.", unavailable_inputs=("governing law",)
    )
    findings = check_abstention(artifact).findings
    assert any(f.severity is Severity.BLOCK for f in findings)


def test_abstention_accepts_a_flagged_gap():
    artifact = _artifact(
        "The governing law was not provided, so the clause is left open.",
        unavailable_inputs=("governing law",),
    )
    assert check_abstention(artifact).findings == []


def test_abstention_warns_when_a_missing_input_is_silently_dropped():
    artifact = _artifact("Everything else looks fine.", unavailable_inputs=("governing law",))
    findings = check_abstention(artifact).findings
    assert findings and findings[0].severity is Severity.WARN


# --------------------------------------------------------------------------
# The stated limit
# --------------------------------------------------------------------------


def test_the_floor_catches_unsourced_not_false():
    """Being straight about the limit, in a test rather than a footnote.

    A cited claim that misrepresents its source passes the floor. That is what
    the L2 recomputation rung exists for, and pretending otherwise would be
    the same overclaiming this system is built to avoid.
    """
    artifact = _artifact(
        "Renewal rate is 88% [S1].",
        sources=(Source(id="S1", text="renewal rate 88% among trial accounts only"),),
    )
    assert check_provenance(artifact).findings == []
