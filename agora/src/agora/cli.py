"""``agora`` — the onboarding CLI.

Three distinct operations get called "adding an agent". They carry very
different costs, and picking the wrong one is where fleets go bad::

    new execution engine?      -> add-runtime  ~1 day    adapter work
    fits an existing agent?    -> add-skill    ~minutes  config, evals first
    evidence for a new role?   -> add-agent    ~hours    charter + review
    none of the above          -> build nothing, just ask

Everything the CLI creates is reversible with ``agora rollback``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .economics import render_scenarios, render_waterfall
from .factory import (
    AgentFactory,
    Charter,
    Demand,
    Rung,
    entropy_sweep,
    which,
)
from .fleet import render_fleet, starter_charters, validate_all
from .registry import AgentCard, AgentRegistry, Probation, Skill
from .verification.corpus import run_corpus
from .verification.packs import CAPABILITY_INDEX, PackRegistry

DEFAULT_STATE = Path("agora-state.json")


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------


class State:
    """Installed agents and runtimes, persisted as plain JSON.

    Deliberately boring and inspectable. An operator has to be able to read
    what the system did to itself without running the system.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"agents": {}, "runtimes": {}, "history": []}
        if path.exists():
            self.data = json.loads(path.read_text())

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True) + "\n")

    def registry(self) -> AgentRegistry:
        registry = AgentRegistry()
        for record in self.data["agents"].values():
            registry.register(
                AgentCard(
                    id=record["id"],
                    name=record["name"],
                    runtime=record.get("runtime", "unknown"),
                    capabilities=tuple(record.get("capabilities", ())),
                    department=record.get("department", ""),
                    retirement_criteria=record.get("retirement_criteria", ""),
                    probation=Probation(record.get("probation", "zero-trust")),
                    skills=[Skill(**s) for s in record.get("skills", [])],
                )
            )
        return registry

    def note(self, action: str, subject: str, detail: str = "") -> None:
        self.data["history"].append(
            {"action": action, "subject": subject, "detail": detail}
        )


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_which(args: argparse.Namespace) -> int:
    demand = Demand(
        capability=args.capability or "unnamed",
        occurrences_30d=args.occurrences,
        multistep=args.multistep,
        needs_tools=args.tools or args.multistep,
        needs_continuity=args.continuity,
        needs_own_authority=args.own_authority,
        sibling_agents=args.siblings,
        fits_existing_agent=args.fits,
    )
    rung, reasons = which(demand)
    print(f"{rung.value} — {rung.description}")
    for reason in reasons:
        print(f"  · {reason}")
    if rung is Rung.P0_NOTHING:
        print("\nThe cheapest capability is the one you did not build.")
    return 0


def cmd_add_agent(args: argparse.Namespace) -> int:
    state = State(args.state)
    charter = Charter.from_dict(json.loads(Path(args.charter).read_text()))
    registry = state.registry()
    factory = AgentFactory(registry)
    demand = Demand(
        capability=charter.capabilities[0] if charter.capabilities else "",
        occurrences_30d=args.occurrences,
        multistep=True,
        needs_tools=True,
        needs_continuity=True,
        sibling_agents=args.siblings,
    )
    verdict = factory.review(charter, demand)
    print(verdict.describe())
    print(f"  skeptic: {verdict.skeptic_position}")
    for warning in verdict.warnings:
        print(f"  warning: {warning}")
    if not verdict.approved:
        return 1
    if not args.yes:
        print("\nCharter approval is a human decision. Re-run with --yes to commit.")
        return 2
    card = factory.commit(charter, verdict, human_approved=True)
    state.data["agents"][card.id] = {
        "id": card.id,
        "name": card.name,
        "runtime": card.runtime,
        "capabilities": list(card.capabilities),
        "department": card.department,
        "retirement_criteria": card.retirement_criteria,
        "probation": card.probation.value,
        "skills": [],
    }
    state.note("add-agent", card.id, charter.job[:80])
    state.save()
    print(f"\ninstalled {card.id} at probation={card.probation.value}")
    print("It enters at zero trust. Trust is measured, and nothing is measured yet.")
    return 0


def cmd_add_skill(args: argparse.Namespace) -> int:
    state = State(args.state)
    record = state.data["agents"].get(args.agent)
    if record is None:
        print(f"unknown agent {args.agent!r}", file=sys.stderr)
        return 1
    if not args.evals or not Path(args.evals).exists():
        print(
            "the eval suite must exist before the skill does — that inversion is "
            "the whole point of the pipeline",
            file=sys.stderr,
        )
        return 1
    if any(s["name"] == args.name for s in record.get("skills", [])):
        print(f"{args.agent} already has a skill named {args.name!r}", file=sys.stderr)
        return 1
    record.setdefault("skills", []).append(
        {"name": args.name, "tier": args.tier, "evals": args.evals, "promoted": False}
    )
    state.note("add-skill", f"{args.agent}#{args.name}", args.tier)
    state.save()
    print(f"registered {args.name!r} on {args.agent} at tier {args.tier}")
    print("Next: sandbox → shadow → canary → promote. Rollback is one command.")
    return 0


def cmd_add_runtime(args: argparse.Namespace) -> int:
    state = State(args.state)
    state.data["runtimes"][args.name] = {
        "name": args.name,
        "url": args.url,
        "capabilities": args.capabilities,
        "probation": Probation.ZERO_TRUST.value,
    }
    state.note("add-runtime", args.name, args.url or "")
    state.save()
    print(f"registered runtime {args.name!r} at zero trust")
    print("Sandbox only, verification mandatory, graduating on measured evidence.")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    state = State(args.state)
    registry = state.registry()
    usage = state.data.get("usage_30d", {})
    findings = entropy_sweep(registry, usage_30d=usage)
    if not findings:
        print("monthly sweep: nothing to propose")
        return 0
    print(f"monthly sweep: {len(findings)} proposal(s) — the Archivist proposes only\n")
    for finding in findings:
        print(f"  [{finding.kind}] {finding.subject}")
        print(f"      {finding.detail} → {finding.proposal}")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    state = State(args.state)
    if args.agent not in state.data["agents"]:
        print(f"unknown agent {args.agent!r}", file=sys.stderr)
        return 1
    del state.data["agents"][args.agent]
    state.note("rollback", args.agent)
    state.save()
    print(f"rolled back {args.agent}")
    return 0


def cmd_install(args: argparse.Namespace) -> int:
    state = State(args.state)
    charters = [c for c in starter_charters() if c.phase == args.phase]
    for charter in charters:
        state.data["agents"][charter.id] = {
            "id": charter.id,
            "name": charter.name,
            "runtime": charter.runtime,
            "capabilities": list(charter.capabilities),
            "department": charter.department,
            "retirement_criteria": charter.retirement_criteria,
            "probation": Probation.PROBATION.value,
            "skills": [],
        }
        state.note("install", charter.id, f"phase {charter.phase}")
    state.save()
    print(f"installed phase {args.phase}: {len(charters)} agent(s)")
    for charter in charters:
        print(f"  {charter.name} — Grade {charter.grade}")
    if args.phase == 1:
        print("\nNow stop. Run it for a fortnight before touching anything else.")
    return 0


def cmd_fleet(args: argparse.Namespace) -> int:
    results = validate_all()
    bad = [r for r in results if not r.ok]
    print(render_fleet())
    print(f"\n{len(results) - len(bad)}/{len(results)} charters validate")
    for result in bad:
        print(f"  {result.charter_id}: {'; '.join(result.problems)}")
    return 1 if bad else 0


def cmd_verify(args: argparse.Namespace) -> int:
    report = run_corpus()
    print(report.headline())
    print()
    for result in report.results:
        state = "clean" if not result.report.findings else f"{len(result.report.findings)} finding(s)"
        print(f"  {result.case.domain:<18} {state}")
        for finding in result.report.findings:
            print(f"      {finding}")
    print()
    if report.clean:
        print("Every seeded fault caught. No false positives. No model calls.")
        return 0
    print(f"MISSED: {report.missed}  FALSE POSITIVES: {report.false_positives}")
    return 1


def cmd_grades(args: argparse.Namespace) -> int:
    packs = PackRegistry()
    capabilities = sorted(CAPABILITY_INDEX)
    print("CAPABILITY                    PACK          GRADE  MAY")
    mismatches = 0
    for capability, pack, grade in packs.coverage_table(capabilities):
        expected = CAPABILITY_INDEX[capability]
        flag = "" if grade.value == expected else f"  ← published index says {expected}"
        mismatches += bool(flag)
        print(
            f"{capability:<29} {pack:<13} {grade.value}      "
            f"{grade.verification_policy}{flag}"
        )
    if mismatches:
        print(f"\n{mismatches} capability grade(s) drifted from the published index.")
    print(
        "\nNothing is refused; everything is labelled. Stating what a capability "
        "cannot do is what makes the rest credible."
    )
    return 0


def cmd_costs(args: argparse.Namespace) -> int:
    print(render_waterfall())
    print()
    print(render_scenarios())
    return 0


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agora",
        description="AGORA — a control plane for agent fleets.",
    )
    parser.add_argument(
        "--state", type=Path, default=DEFAULT_STATE, help="path to the state file"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("which", help="which operation do I actually need?")
    p.add_argument("--capability", default=None)
    p.add_argument("--occurrences", type=int, default=0)
    p.add_argument("--multistep", action="store_true")
    p.add_argument("--tools", action="store_true")
    p.add_argument("--continuity", action="store_true")
    p.add_argument("--own-authority", action="store_true")
    p.add_argument("--siblings", type=int, default=0)
    p.add_argument("--fits", default=None, help="an existing agent this fits under")
    p.set_defaults(func=cmd_which)

    p = sub.add_parser("add-agent", help="charter review, then install")
    p.add_argument("--charter", required=True)
    p.add_argument("--occurrences", type=int, default=10)
    p.add_argument("--siblings", type=int, default=0)
    p.add_argument("--yes", action="store_true", help="the human approval step")
    p.set_defaults(func=cmd_add_agent)

    p = sub.add_parser("add-skill", help="add a procedure under an existing agent")
    p.add_argument("--agent", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--tier", default="T1", choices=["T0", "T1", "T2"])
    p.add_argument("--evals", required=True, help="must already exist")
    p.set_defaults(func=cmd_add_skill)

    p = sub.add_parser("add-runtime", help="onboard a whole new engine")
    p.add_argument("--name", required=True)
    p.add_argument("--url", default=None)
    p.add_argument("--capabilities", nargs="*", default=[])
    p.set_defaults(func=cmd_add_runtime)

    p = sub.add_parser("audit", help="monthly entropy sweep")
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("rollback", help="undo anything")
    p.add_argument("--agent", required=True)
    p.set_defaults(func=cmd_rollback)

    p = sub.add_parser("install", help="install a fleet phase")
    p.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4, 5])
    p.set_defaults(func=cmd_install)

    p = sub.add_parser("fleet", help="show and validate the starter fleet")
    p.set_defaults(func=cmd_fleet)

    p = sub.add_parser("verify", help="run the cross-domain fault corpus")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("grades", help="assurance grade per capability")
    p.set_defaults(func=cmd_grades)

    p = sub.add_parser("costs", help="cost waterfall and scenarios")
    p.set_defaults(func=cmd_costs)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BrokenPipeError:  # `agora verify | head` is a normal thing to do
        try:
            sys.stdout.close()
        finally:
            return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
