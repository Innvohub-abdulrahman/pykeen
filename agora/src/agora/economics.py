"""Parameterised economics: where the money actually goes.

Prices here are placeholders and will be wrong by the time you read them. The
*ratios* are far more stable than the absolutes, which is why every number is
computed from a parameter block rather than written down.

Read the waterfall honestly. The baseline is the default behaviour of most
multi-agent frameworks -- full transcript passing, artifacts inline, every
turn on a frontier model. Against competent engineering that already passes
references and caches prompts, expect **3-8x**, most of it from the model
ladder and the verification ladder. Do not put 46x in a deck.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence


# --------------------------------------------------------------------------
# Prices
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrice:
    """Dollars per million tokens."""

    name: str
    input_per_mtok: float
    output_per_mtok: float
    cached_input_multiplier: float = 0.10

    def cost(
        self, *, input_tokens: float, output_tokens: float, cached_tokens: float = 0.0
    ) -> float:
        fresh = max(0.0, input_tokens - cached_tokens)
        return (
            fresh * self.input_per_mtok
            + cached_tokens * self.input_per_mtok * self.cached_input_multiplier
            + output_tokens * self.output_per_mtok
        ) / 1_000_000.0


FRONTIER = ModelPrice("frontier", 3.00, 15.00)
MID = ModelPrice("mid", 0.80, 4.00)
LOCAL = ModelPrice("local", 0.0, 0.0)  # amortised GPU, zero marginal cost


# --------------------------------------------------------------------------
# Task shape
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskShape:
    """One Project-tier task: four subtasks, three agents, twenty-four turns.

    The defaults describe what a *naive* multi-agent framework actually does,
    because that is what the baseline is meant to represent:

    * a 27k-token persona and domain pack re-sent on every single turn;
    * an 19k-token contract and inputs, likewise;
    * artifacts of ~24k tokens (a diff plus its tests, a research brief)
      carried inline in every subsequent message;
    * the accumulated transcript re-read by each of the three agents, which is
      what ``context_replication`` counts.

    None of these are strawmen. They are the default behaviour you get when
    nobody has decided otherwise.
    """

    turns: int = 24
    subtasks: int = 4
    agents: int = 3
    persona_tokens: int = 26_800     # persona + domain pack, identical every turn
    task_tokens: int = 18_900        # the contract and its inputs
    output_tokens_per_turn: int = 1_700
    artifact_tokens: int = 24_000    # a diff plus tests, a research brief, a spec
    reference_tokens: int = 60       # uri + digest + kind + confidence
    compiled_context_tokens: int = 700   # what a compiled context costs instead
    context_replication: int = 3     # agents each re-reading the shared transcript
    planning_cost_per_subtask: float = 0.08   # synthesised from scratch, every time
    judge_verification_share: float = 1.0     # fraction of the artifact re-read by a judge
    rework_factor: float = 0.34      # unverified work that has to be done twice


@dataclass(frozen=True)
class Mix:
    """How turns are distributed across the model ladder."""

    local: float = 0.0
    mid: float = 0.0
    frontier: float = 1.0

    def normalised(self) -> "Mix":
        total = self.local + self.mid + self.frontier
        if total <= 0:
            return Mix(0.0, 0.0, 1.0)
        return Mix(self.local / total, self.mid / total, self.frontier / total)


# --------------------------------------------------------------------------
# Levers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Levers:
    """Each lever is a switch, applied cumulatively in the waterfall."""

    reference_passing: bool = False
    compiled_context: bool = False
    prompt_caching: bool = False
    plan_cache_and_turn_caps: bool = False
    model_ladder: bool = False

    capped_turns: int = 16
    ladder_mix: Mix = field(default_factory=lambda: Mix(local=0.55, mid=0.30, frontier=0.15))


def cost_of_task(shape: TaskShape, levers: Levers) -> float:
    """Cost one task under a given set of levers.

    The baseline is deliberately not a strawman: it is what you get from a
    framework that passes the transcript forward, inlines artifacts, and sends
    every turn to the best model available.
    """
    turns = levers.capped_turns if levers.plan_cache_and_turn_caps else shape.turns
    mix = levers.ladder_mix.normalised() if levers.model_ladder else Mix()

    turns_per_subtask = max(1, turns // max(1, shape.subtasks))
    replication = 1 if levers.compiled_context else max(1, shape.context_replication)
    total = 0.0

    for turn in range(turns):
        # --- input side -------------------------------------------------
        fixed = shape.persona_tokens + shape.task_tokens

        if levers.compiled_context:
            # No transcript accumulation: each turn is handed a compiled
            # context of roughly constant size.
            history = shape.compiled_context_tokens
        else:
            history = turn * shape.output_tokens_per_turn * replication

        finished_artifacts = min(shape.subtasks, turn // turns_per_subtask)
        if levers.reference_passing:
            artifacts = finished_artifacts * shape.reference_tokens
        else:
            artifacts = finished_artifacts * shape.artifact_tokens * replication

        input_tokens = fixed + history + artifacts
        cached = shape.persona_tokens if levers.prompt_caching else 0.0

        # --- price it ---------------------------------------------------
        for price, share in (
            (LOCAL, mix.local),
            (MID, mix.mid),
            (FRONTIER, mix.frontier),
        ):
            if share <= 0:
                continue
            total += share * price.cost(
                input_tokens=input_tokens,
                output_tokens=shape.output_tokens_per_turn,
                cached_tokens=cached,
            )
    return total


# --------------------------------------------------------------------------
# The waterfall
# --------------------------------------------------------------------------


@dataclass
class WaterfallRow:
    label: str
    cost_usd: float
    delta_pct: float | None = None
    cumulative_factor: float = 1.0


#: The levers in the order they are applied, each named as it appears in the
#: report's Figure 20.
WATERFALL: tuple[tuple[str, str], ...] = (
    ("", "Baseline — transcript passing, artifacts inline, all frontier"),
    ("reference_passing", "+ Reference passing — artifacts by URI, not payload"),
    ("compiled_context", "+ Compiled context — no transcript accumulation"),
    ("prompt_caching", "+ Prompt caching — persona and domain pack"),
    ("plan_cache_and_turn_caps", "+ Plan cache and turn caps — 24 turns to 16"),
    ("model_ladder", "+ Model ladder — local, mid, frontier by need"),
)


def waterfall(shape: TaskShape | None = None) -> list[WaterfallRow]:
    """Levers applied cumulatively to one Project-tier task."""
    shape = shape or TaskShape()
    levers = Levers()
    rows: list[WaterfallRow] = []
    baseline = cost_of_task(shape, levers)
    previous = baseline

    for attr, label in WATERFALL:
        if attr:
            levers = replace(levers, **{attr: True})
        cost = cost_of_task(shape, levers)
        rows.append(
            WaterfallRow(
                label=label,
                cost_usd=cost,
                delta_pct=None if not attr else (cost - previous) / previous * 100.0,
                cumulative_factor=baseline / cost if cost else float("inf"),
            )
        )
        previous = cost
    return rows


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    naive_usd: float
    optimised_usd: float
    note: str = ""

    @property
    def factor(self) -> float:
        return self.naive_usd / self.optimised_usd if self.optimised_usd else float("inf")


ALL_LEVERS = Levers(
    reference_passing=True,
    compiled_context=True,
    prompt_caching=True,
    plan_cache_and_turn_caps=True,
    model_ladder=True,
)

#: What a competent team already does without being told: hand artifacts over
#: by reference, cache the persona and domain pack, and compact context instead
#: of replaying the whole transcript. Measuring against *this* rather than
#: against the naive baseline is the number worth quoting -- it is what a buyer
#: who has already done the obvious work would actually see.
COMPETENT_BASELINE = Levers(
    reference_passing=True,
    compiled_context=True,
    prompt_caching=True,
)


def honest_factor(shape: TaskShape | None = None) -> float:
    """Improvement over competent engineering, not over a strawman.

    Lands in the 3-8x range, most of it from the model ladder and the
    verification ladder. The 46x cumulative figure is against a baseline
    nobody should be running, and it does not belong in a deck.
    """
    shape = shape or TaskShape()
    return cost_of_task(shape, COMPETENT_BASELINE) / cost_of_task(shape, ALL_LEVERS)


def naive_total(shape: TaskShape) -> float:
    """Execution plus the three costs a naive framework pays without noticing.

    It re-plans from scratch every time, grades every artifact with a judge
    model, and redoes a share of the work because nothing caught the failure
    the first time. All three are avoidable, and all three are real money.
    """
    execution = cost_of_task(shape, Levers())
    planning = shape.planning_cost_per_subtask * shape.subtasks
    judging = (
        shape.turns
        * shape.artifact_tokens
        * shape.judge_verification_share
        * FRONTIER.input_per_mtok
        / 1_000_000.0
    )
    rework = execution * shape.rework_factor
    return execution + planning + judging + rework


def feature_task_scenario(shape: TaskShape | None = None) -> Scenario:
    """One feature task, naive versus fully levered."""
    shape = shape or TaskShape()
    return Scenario(
        name="One feature task",
        naive_usd=naive_total(shape),
        optimised_usd=cost_of_task(shape, ALL_LEVERS),
        note="plan recalled rather than synthesised; deterministic verification is free",
    )


def email_triage_scenario(
    *,
    messages_per_day: int = 200,
    replies_per_day: int = 15,
    days: int = 365,
    naive_cost_per_message: float = 0.0616,
    drafting_cost_per_reply: float = 0.0175,
) -> Scenario:
    """A year of inbox triage.

    The instructive one. Most of the saving is not a cheaper model -- it is
    not calling a model at all for the 185 messages a day that need no reply.
    Classification runs on a local model at zero marginal cost; only the
    handful that warrant a response reach a paid model.
    """
    naive = messages_per_day * days * naive_cost_per_message
    optimised = replies_per_day * days * drafting_cost_per_reply
    return Scenario(
        name="Email triage, one year",
        naive_usd=naive,
        optimised_usd=optimised,
        note=(
            f"{messages_per_day - replies_per_day} of {messages_per_day} messages a "
            f"day are classified locally and never reach a paid model"
        ),
    )


def mobile_app_scenario(
    *,
    sessions: int = 40,
    shape: TaskShape | None = None,
) -> Scenario:
    """A Campaign-tier build across many sessions.

    The plan cache is what changes the shape here: after the first few
    sessions most of the planning is recall, and the marginal session is
    mostly execution.
    """
    shape = shape or TaskShape(turns=26, subtasks=5, agents=5, context_replication=3)
    naive = sessions * naive_total(shape)
    levered = replace(ALL_LEVERS, capped_turns=22)
    optimised = sessions * cost_of_task(shape, levered)
    return Scenario(
        name="Mobile app, 40 sessions",
        naive_usd=naive,
        optimised_usd=optimised,
        note="plan cache turns most later-session planning into recall",
    )


def scenarios() -> list[Scenario]:
    return [feature_task_scenario(), email_triage_scenario(), mobile_app_scenario()]


# --------------------------------------------------------------------------
# Verification economics
# --------------------------------------------------------------------------


@dataclass
class VerificationCost:
    """What it costs to check an artifact, per rung of the ladder."""

    universal_floor: float = 0.0   # deterministic, zero marginal cost
    l0_executable: float = 0.0     # a test run you were going to do anyway
    l1_constraints: float = 0.0
    l2_recomputation: float = 0.004
    l3_rubric: float = 0.012
    l5_human: float = 4.50         # a person's time, priced honestly

    def cost_for_grade(self, grade: str) -> float:
        return {
            "A": self.universal_floor + self.l0_executable,
            "B": self.universal_floor + self.l1_constraints + self.l2_recomputation,
            "C": self.universal_floor + self.l3_rubric + self.l5_human,
            "D": self.universal_floor + self.l5_human,
        }[grade.upper()]


def judge_model_comparison(
    *, artifacts_per_day: int = 500, days: int = 365, judge_cost: float = 0.012
) -> tuple[float, float]:
    """Deterministic floor versus an LLM judge, at volume.

    Returns ``(judge_annual_usd, floor_annual_usd)``. The floor is zero, which
    is the entire argument: a Capability Ledger needs verification cheap
    enough to run on *everything*, or it is measuring a biased sample.
    """
    return (artifacts_per_day * days * judge_cost, 0.0)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def render_waterfall(rows: Sequence[WaterfallRow] | None = None) -> str:
    rows = rows or waterfall()
    width = max(len(r.label) for r in rows)
    lines = ["EXECUTION COST PER TASK · levers applied cumulatively", ""]
    for row in rows:
        delta = f"  {row.delta_pct:+.1f}%" if row.delta_pct is not None else ""
        lines.append(f"{row.label:<{width}}  ${row.cost_usd:>7.2f}{delta}")
    lines.append("")
    lines.append(f"{rows[-1].cumulative_factor:.0f}× cumulative")
    lines.append(
        "Against competent engineering that already passes references and caches "
        "prompts, expect 3–8×."
    )
    return "\n".join(lines)


def render_scenarios(items: Sequence[Scenario] | None = None) -> str:
    items = items or scenarios()
    lines = ["ANNUALISED COST · naive versus optimised", ""]
    for s in items:
        lines.append(
            f"{s.name:<28} ${s.naive_usd:>10,.2f}  →  ${s.optimised_usd:>9,.2f}  "
            f"({s.factor:.0f}× cheaper)"
        )
        if s.note:
            lines.append(f"{'':<28} {s.note}")
    return "\n".join(lines)


__all__ = [
    "FRONTIER",
    "LOCAL",
    "MID",
    "Levers",
    "Mix",
    "ModelPrice",
    "ALL_LEVERS",
    "COMPETENT_BASELINE",
    "Scenario",
    "TaskShape",
    "honest_factor",
    "naive_total",
    "VerificationCost",
    "WaterfallRow",
    "cost_of_task",
    "email_triage_scenario",
    "feature_task_scenario",
    "judge_model_comparison",
    "mobile_app_scenario",
    "render_scenarios",
    "render_waterfall",
    "scenarios",
    "waterfall",
]
