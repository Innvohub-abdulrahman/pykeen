"""Domain packs: the checks that layer above the universal floor.

A pack is the honest unit of coverage in this system. Its assurance grade is
*derived* from the rungs it can supply, never asserted:

* supplies L0 (something executable)                -> Grade A
* supplies L1 or L2 (constraints or recomputation)  -> Grade B
* supplies only a rubric                            -> Grade C
* supplies nothing                                  -> Grade D, floor only

Grade D is not "unsupported". A Grade D capability still gets provenance
checking, scope enforcement, authority limits, an audit trail and cost
control. It just does not get autonomy.

Domain packs are the largest hidden cost in the system and the reason
verification debt is the top risk in the register. The five packs here are
reference implementations, not a complete library.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..artifacts import Artifact
from ..contracts import TaskContract
from .ladder import DomainPack, Grade, Level, grade_capability
from .universal import CheckResult, Finding, Severity, content_words, sentences


def _result(name: str) -> CheckResult:
    return CheckResult(check=name)


def _block(result: CheckResult, message: str, evidence: str = "") -> None:
    result.findings.append(
        Finding(
            check=result.check,
            severity=Severity.BLOCK,
            message=message,
            evidence=evidence,
        )
    )


# --------------------------------------------------------------------------
# Engineering -- Grade A (executable)
# --------------------------------------------------------------------------


def check_tests_pass(artifact: Artifact, contract: TaskContract | None) -> CheckResult:
    """L0: the attached test report must show a clean run.

    The artifact carries the evidence; the check reads it. There is no model
    involved and no judgement to make -- exit code zero or it did not pass.
    """
    result = _result("eng.tests")
    report = artifact.metadata.get("test_report")
    if report is None:
        _block(result, "no test report attached to a code artifact")
        return result
    if int(report.get("exit_code", 1)) != 0:
        _block(
            result,
            f"test suite exited {report.get('exit_code')}",
            evidence=str(report.get("summary", "")),
        )
    if int(report.get("failed", 0)) > 0:
        _block(result, f"{report['failed']} test(s) failed")
    if int(report.get("collected", 0)) == 0:
        _block(result, "test suite collected zero tests")
    return result


def check_patch_applies(artifact: Artifact, contract: TaskContract | None) -> CheckResult:
    """L0: the diff must apply and the build must compile."""
    result = _result("eng.build")
    meta = artifact.metadata
    if meta.get("patch_applies") is False:
        _block(result, "the produced diff does not apply to the base revision")
    if meta.get("compiles") is False:
        _block(result, "the change does not compile", evidence=str(meta.get("compiler_output", "")))
    return result


def check_no_new_lint(artifact: Artifact, contract: TaskContract | None) -> CheckResult:
    """L1: no new lint violations relative to the base revision."""
    result = _result("eng.lint")
    delta = int(artifact.metadata.get("lint_delta", 0))
    if delta > 0:
        _block(result, f"{delta} new lint violation(s) introduced")
    return result


def engineering_pack() -> DomainPack:
    pack = DomainPack(
        name="engineering",
        capabilities=(
            "code.implement",
            "repo.refactor",
            "test.author",
            "test.execute",
            "code.review",
            "verify.rubric",
            "build.compile",
            "build.device_matrix",
            "build.store_lint",
        ),
        description="Tests, compiler and lint -- the cheapest ground truth there is.",
    )
    pack.add(Level.L0, check_tests_pass)
    pack.add(Level.L0, check_patch_applies)
    pack.add(Level.L1, check_no_new_lint)
    return pack


# --------------------------------------------------------------------------
# Calendar -- Grade A (solver)
# --------------------------------------------------------------------------


def check_schedule_feasible(
    artifact: Artifact, contract: TaskContract | None
) -> CheckResult:
    """L0: a proposed schedule must actually be physically possible.

    Overlaps, travel time between locations, declared working hours and
    blocked windows (prayer gaps, Ramadan hours) are all decidable. A schedule
    is either feasible or it is not; no rubric is required to tell you.
    """
    result = _result("cal.feasible")
    events: Sequence[Mapping[str, Any]] = artifact.metadata.get("schedule", [])
    if not events:
        result.skipped = True
        result.note = "no schedule attached"
        return result

    ordered = sorted(events, key=lambda e: float(e["start"]))
    for earlier, later in zip(ordered, ordered[1:]):
        end = float(earlier["end"]) + float(earlier.get("travel_min", 0)) / 60.0
        if end > float(later["start"]) + 1e-9:
            _block(
                result,
                f"{earlier.get('title', '?')!r} runs into {later.get('title', '?')!r} "
                f"once travel time is counted",
                evidence=f"{earlier} -> {later}",
            )

    working = artifact.metadata.get("working_hours")
    if working:
        lo, hi = float(working[0]), float(working[1])
        for event in ordered:
            if float(event["start"]) < lo - 1e-9 or float(event["end"]) > hi + 1e-9:
                _block(
                    result,
                    f"{event.get('title', '?')!r} falls outside working hours "
                    f"{lo:g}-{hi:g}",
                    evidence=str(event),
                )

    for window in artifact.metadata.get("blocked_windows", []):
        b_lo, b_hi = float(window["start"]), float(window["end"])
        for event in ordered:
            if float(event["start"]) < b_hi and float(event["end"]) > b_lo:
                _block(
                    result,
                    f"{event.get('title', '?')!r} collides with the blocked window "
                    f"{window.get('label', '')!r}",
                    evidence=str(event),
                )
    return result


def calendar_pack() -> DomainPack:
    pack = DomainPack(
        name="calendar",
        capabilities=("schedule.resolve", "schedule.reflow", "schedule.negotiate"),
        description="Feasibility is a constraint problem, and constraints are decidable.",
    )
    pack.add(Level.L0, check_schedule_feasible)
    return pack


# --------------------------------------------------------------------------
# Finance -- Grade A (identities + recomputation)
# --------------------------------------------------------------------------


def check_ledger_identity(
    artifact: Artifact, contract: TaskContract | None
) -> CheckResult:
    """L0: the parts must sum to the whole, to the cent."""
    result = _result("fin.identity")
    ledger = artifact.metadata.get("ledger")
    if not ledger:
        result.skipped = True
        result.note = "no ledger attached"
        return result
    lines = ledger.get("lines", [])
    total = float(ledger.get("total", 0.0))
    computed = round(sum(float(line["amount"]) for line in lines), 2)
    if abs(computed - round(total, 2)) > 0.005:
        _block(
            result,
            f"line items sum to {computed:.2f} but the stated total is {total:.2f}",
            evidence=f"{len(lines)} line(s)",
        )
    return result


def check_source_tie_out(
    artifact: Artifact, contract: TaskContract | None
) -> CheckResult:
    """L2: every reported figure recomputes from the source rows.

    This is the rung that catches what the universal floor structurally
    cannot: a figure that *is* cited but does not follow from what it cites.
    """
    result = _result("fin.tie_out")
    rows = artifact.metadata.get("source_rows")
    reported = artifact.metadata.get("reported")
    if rows is None or reported is None:
        result.skipped = True
        result.note = "no source rows to recompute from"
        return result
    for key, stated in reported.items():
        recomputed = round(
            sum(float(r["amount"]) for r in rows if r.get("bucket") == key), 2
        )
        if abs(recomputed - round(float(stated), 2)) > 0.005:
            _block(
                result,
                f"{key!r} reported as {float(stated):.2f} but the source rows "
                f"recompute to {recomputed:.2f}",
                evidence=f"bucket={key}",
            )
    unmatched = [r for r in rows if not r.get("bucket")]
    if unmatched:
        _block(
            result,
            f"{len(unmatched)} source row(s) are untraceable to any reported bucket",
        )
    return result


def finance_pack() -> DomainPack:
    pack = DomainPack(
        name="finance",
        capabilities=(
            "finance.reconcile",
            "finance.trace",
            "finance.forecast",
            "finance.scenario",
        ),
        description="Identities hold or they do not. Reconciliation is arithmetic.",
    )
    pack.add(Level.L0, check_ledger_identity)
    pack.add(Level.L2, check_source_tie_out)
    # Reconciliation ties out to the cent; a forecast has nothing to tie to.
    # Same pack, different grades, and saying so is the point.
    pack.capability_levels = {
        "finance.reconcile": (Level.L0, Level.L2),
        "finance.trace": (Level.L0, Level.L2),
        "finance.forecast": (Level.L2,),
        "finance.scenario": (Level.L2,),
    }
    return pack


# --------------------------------------------------------------------------
# Research -- Grade B (recomputation against the cited source)
# --------------------------------------------------------------------------

_CITE = re.compile(r"\[(?P<id>[A-Za-z0-9_.\-]+)\]")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def check_citation_tie_out(
    artifact: Artifact, contract: TaskContract | None
) -> CheckResult:
    """L2: the cited source must actually contain the figure cited to it.

    The universal floor asks whether a figure traces to *any* supplied source.
    This asks the stronger question -- whether it traces to *the source it
    claims*.
    """
    result = _result("research.tie_out")
    by_id = {s.id: s.text for s in artifact.sources}
    if not by_id:
        result.skipped = True
        result.note = "no sources supplied"
        return result
    for sentence in sentences(artifact.content):
        cited = [m.group("id") for m in _CITE.finditer(sentence)]
        if not cited:
            continue
        numbers = {n.group(0).replace(",", "") for n in _NUM.finditer(sentence)}
        if not numbers:
            continue
        pool = " ".join(by_id.get(c, "") for c in cited).replace(",", "")
        for number in numbers:
            if number not in pool:
                _block(
                    result,
                    f"{number!r} is cited to {cited} but does not appear there",
                    evidence=sentence.strip(),
                )
    return result


def research_pack() -> DomainPack:
    pack = DomainPack(
        name="research",
        capabilities=("web.research", "doc.extract", "account.research"),
        description="Tie every figure back to the source it was cited to.",
    )
    pack.add(Level.L2, check_citation_tie_out)
    return pack


# --------------------------------------------------------------------------
# CRM / support -- Grade B (schema constraints)
# --------------------------------------------------------------------------


def check_record_schema(artifact: Artifact, contract: TaskContract | None) -> CheckResult:
    """L1: every written record carries its required fields."""
    result = _result("crm.schema")
    records = artifact.metadata.get("records")
    required = artifact.metadata.get("required_fields", [])
    if records is None or not required:
        result.skipped = True
        result.note = "no records or no schema declared"
        return result
    for i, record in enumerate(records):
        missing = [f for f in required if not record.get(f)]
        if missing:
            _block(
                result,
                f"record {i} is missing required field(s): {', '.join(missing)}",
                evidence=str(record),
            )
    return result


def crm_pack() -> DomainPack:
    pack = DomainPack(
        name="crm",
        capabilities=("crm.hygiene", "crm.enrich", "support.triage", "support.dedupe"),
        description="Schema conformance on every written record.",
    )
    pack.add(Level.L1, check_record_schema)
    return pack


# --------------------------------------------------------------------------
# Personal, support and fleet
# --------------------------------------------------------------------------


def check_label_set(artifact: Artifact, contract: TaskContract | None) -> CheckResult:
    """L1: a classification must come from the declared label set.

    Trivially decidable, and it catches the most common failure of a
    classifier wired to a language model: a plausible label nobody defined.
    """
    result = _result("personal.labels")
    labels = artifact.metadata.get("labels")
    allowed = artifact.metadata.get("label_set")
    if labels is None or not allowed:
        result.skipped = True
        result.note = "no labels or no declared label set"
        return result
    for label in labels:
        if label not in allowed:
            _block(result, f"label {label!r} is not in the declared label set")
    return result


def check_no_commitment(artifact: Artifact, contract: TaskContract | None) -> CheckResult:
    """L1: a draft must contain no commitment the sender cannot make.

    A thin wrapper over the universal authority check, run again at the domain
    level because for drafting capabilities it is the *whole* job.
    """
    from .universal import check_authority

    result = check_authority(artifact, contract)
    result.check = "personal.no_commitment"
    for i, finding in enumerate(result.findings):
        result.findings[i] = Finding(
            check="personal.no_commitment",
            severity=finding.severity,
            message=finding.message,
            evidence=finding.evidence,
            location=finding.location,
        )
    return result


def personal_pack() -> DomainPack:
    pack = DomainPack(
        name="personal",
        capabilities=("email.classify", "email.draft", "brief.compile"),
        description="Label sets and no-commitment guards. Constraints, not ground truth.",
    )
    pack.add(Level.L1, check_label_set)
    pack.add(Level.L1, check_no_commitment)
    return pack


def support_pack() -> DomainPack:
    """Customer replies. Floor plus rubric only -- Grade C, and it stays there.

    A support draft can be checked for commitments it may not make and for
    sourcing against the knowledge base, but not for whether it actually
    answers the question. That is a judgement, and judgements are Grade C.
    """
    pack = DomainPack(
        name="support",
        capabilities=("support.draft",),
        description="No executable ground truth for whether a reply is right.",
    )
    pack.add(Level.L3, check_no_commitment)
    return pack


def check_sweep_determinism(
    artifact: Artifact, contract: TaskContract | None
) -> CheckResult:
    """L0: a fleet audit must be reproducible from the state it read.

    The Archivist's proposals are a pure function of the registry and the usage
    counts. Re-running the sweep must produce the same list, or the sweep is
    not evidence of anything.
    """
    result = _result("fleet.determinism")
    first = artifact.metadata.get("sweep")
    second = artifact.metadata.get("sweep_rerun")
    if first is None or second is None:
        result.skipped = True
        result.note = "no sweep pair attached"
        return result
    if first != second:
        _block(result, "re-running the sweep produced a different proposal set")
    return result


def fleet_pack() -> DomainPack:
    pack = DomainPack(
        name="fleet",
        capabilities=("fleet.audit", "fleet.archive_propose"),
        description="A sweep is a pure function of state, so it is checkable.",
    )
    pack.add(Level.L0, check_sweep_determinism)
    return pack


# --------------------------------------------------------------------------
# Planning -- Grade B (structural invariants)
# --------------------------------------------------------------------------


def check_plan_invariants(
    artifact: Artifact, contract: TaskContract | None
) -> CheckResult:
    """L1: a plan must be a DAG, fully funded, and fully specified."""
    result = _result("plan.invariants")
    plan = artifact.metadata.get("plan")
    if not plan:
        result.skipped = True
        result.note = "no plan attached"
        return result

    nodes = {n["id"]: n for n in plan.get("nodes", [])}
    for node in nodes.values():
        for dep in node.get("depends_on", []):
            if dep not in nodes:
                _block(result, f"node {node['id']!r} depends on unknown node {dep!r}")

    # Cycle detection by iterative peeling -- a plan that cannot be ordered
    # cannot be executed, and detecting that here is free.
    remaining = {k: set(v.get("depends_on", [])) & set(nodes) for k, v in nodes.items()}
    while True:
        ready = [k for k, deps in remaining.items() if not deps]
        if not ready:
            break
        for k in ready:
            remaining.pop(k)
        for deps in remaining.values():
            deps.difference_update(ready)
    if remaining:
        _block(
            result,
            f"plan contains a dependency cycle among {sorted(remaining)}",
        )

    budget = float(plan.get("budget_usd", 0.0))
    allocated = sum(float(n.get("budget_usd", 0.0)) for n in nodes.values())
    if allocated > budget + 1e-9:
        _block(
            result,
            f"child budgets total {allocated:.2f} against a plan ceiling of "
            f"{budget:.2f}",
        )
    for node in nodes.values():
        if not node.get("acceptance"):
            _block(
                result,
                f"node {node['id']!r} has no acceptance criteria",
                evidence=str(node.get("goal", "")),
            )
    return result


def planning_pack() -> DomainPack:
    pack = DomainPack(
        name="planning",
        capabilities=("plan.decompose", "plan.synthesise", "charter.critique"),
        description="A plan is a graph; graphs have decidable properties.",
    )
    pack.add(Level.L1, check_plan_invariants)
    return pack


# --------------------------------------------------------------------------
# Marketing and legal -- Grade C (floor + rubric)
# --------------------------------------------------------------------------


def check_marketing_rubric(
    artifact: Artifact, contract: TaskContract | None
) -> CheckResult:
    """L3: banned-claim screen. Cheap, but not deterministic ground truth."""
    result = _result("marketing.claims")
    banned = artifact.metadata.get(
        "banned_claims", ["guaranteed results", "best in the world", "100% secure"]
    )
    lowered = artifact.content.lower()
    for phrase in banned:
        if phrase.lower() in lowered:
            _block(result, f"uses a banned claim: {phrase!r}", evidence=phrase)
    return result


def marketing_pack() -> DomainPack:
    pack = DomainPack(
        name="marketing",
        capabilities=("marketing.landing_copy", "marketing.campaign"),
        description="No executable ground truth exists here. Floor plus rubric, Grade C.",
    )
    pack.add(Level.L3, check_marketing_rubric)
    return pack


def check_clause_coverage(
    artifact: Artifact, contract: TaskContract | None
) -> CheckResult:
    """L3: every clause the reviewer was asked to cover is discussed."""
    result = _result("legal.coverage")
    required = artifact.metadata.get("required_clauses", [])
    if not required:
        result.skipped = True
        result.note = "no clause checklist supplied"
        return result
    body = content_words(artifact.content)
    for clause in required:
        if not content_words(clause) & body:
            _block(result, f"clause {clause!r} is not addressed anywhere in the summary")
    return result


def legal_pack() -> DomainPack:
    pack = DomainPack(
        name="legal",
        capabilities=("legal.nda_summary", "legal.clause_check"),
        description="Coverage is checkable; correctness is not. Grade C, and we say so.",
    )
    pack.add(Level.L3, check_clause_coverage)
    return pack


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def default_packs() -> dict[str, DomainPack]:
    return {
        p.name: p
        for p in (
            engineering_pack(),
            calendar_pack(),
            finance_pack(),
            research_pack(),
            crm_pack(),
            planning_pack(),
            marketing_pack(),
            legal_pack(),
            personal_pack(),
            support_pack(),
            fleet_pack(),
        )
    }


class PackRegistry:
    """Resolves a capability to the pack that covers it, and to its grade."""

    def __init__(self, packs: Mapping[str, DomainPack] | None = None) -> None:
        self.packs: dict[str, DomainPack] = dict(
            packs if packs is not None else default_packs()
        )

    def register(self, pack: DomainPack) -> None:
        self.packs[pack.name] = pack

    def for_capability(self, capability: str) -> DomainPack | None:
        for pack in self.packs.values():
            if pack.capabilities and capability in pack.capabilities:
                return pack
        for pack in self.packs.values():
            if pack.covers(capability) and pack.capabilities:
                return pack
        return None

    def grade_for(self, capability: str) -> Grade:
        """The assurance grade for a capability, derived from its pack.

        Strategy work resolves to no pack and is therefore Grade D. That is
        the honest answer, and stating it is what makes the Grade A claims
        credible.
        """
        pack = self.for_capability(capability)
        has_rubric = bool(pack and Level.L3 in pack.levels_supplied(capability))
        return grade_capability(pack, has_rubric=has_rubric, capability=capability)

    def coverage_table(self, capabilities: Sequence[str]) -> list[tuple[str, str, Grade]]:
        rows = []
        for capability in capabilities:
            pack = self.for_capability(capability)
            rows.append((capability, pack.name if pack else "-", self.grade_for(capability)))
        return rows


#: The capability index, with the grade each capability is *expected* to reach.
#: This is the published claim in Appendix A of the system report, kept next to
#: the code so a pack that quietly loses a check fails a test rather than
#: quietly downgrading a customer's autonomy.
CAPABILITY_INDEX: dict[str, str] = {
    # Personal
    "schedule.resolve": "A",
    "schedule.reflow": "A",
    "schedule.negotiate": "A",
    "email.classify": "B",
    "email.draft": "B",
    "brief.compile": "B",
    # Engineering
    "code.implement": "A",
    "repo.refactor": "A",
    "test.author": "A",
    "test.execute": "A",
    "code.review": "A",
    "verify.rubric": "A",
    "build.compile": "A",
    "build.device_matrix": "A",
    "build.store_lint": "A",
    "web.research": "B",
    "doc.extract": "B",
    # Finance
    "finance.reconcile": "A",
    "finance.trace": "A",
    "finance.forecast": "B",
    "finance.scenario": "B",
    # Revenue and customer
    "crm.hygiene": "B",
    "crm.enrich": "B",
    "account.research": "B",
    "support.triage": "B",
    "support.dedupe": "B",
    "support.draft": "C",
    # Marketing and legal
    "marketing.landing_copy": "C",
    "marketing.campaign": "C",
    "legal.nda_summary": "C",
    "legal.clause_check": "C",
    # Control
    "plan.decompose": "B",
    "plan.synthesise": "B",
    "charter.critique": "B",
    "fleet.audit": "A",
    "fleet.archive_propose": "A",
    # Strategy -- no pack exists, and none is claimed
    "strategy.position": "D",
    "strategy.assess": "D",
}


__all__ = [
    "CAPABILITY_INDEX",
    "PackRegistry",
    "default_packs",
    "engineering_pack",
    "calendar_pack",
    "finance_pack",
    "research_pack",
    "crm_pack",
    "planning_pack",
    "marketing_pack",
    "legal_pack",
    "personal_pack",
    "support_pack",
    "fleet_pack",
]
