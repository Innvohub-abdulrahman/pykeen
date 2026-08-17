# AGORA

**A control plane for agent fleets.** Add any agent. Run any task. Know which
ones you can trust unsupervised.

This is a working implementation of the system described in *AGORA — System &
Product Report v2.0*. Every mechanism in the report that had a stated
reference implementation has one here, with tests; the mechanisms the report
marked `spec` are implemented too, and marked as such below.

```
python bin/universal_verifiers.py     # the wedge: 11 faults, 0 false positives, 0 model calls
python bin/capability_ledger.py       # measured trust, with intervals
python bin/agent_factory.py           # the commitment ladder
python bin/a2a_adapter_template.py --selftest
python bin/starter_fleet.py           # 16 charters, generated and validated
python bin/agora_cost_model.py        # the cost waterfall
python -m pytest agora/tests -q       # 244 tests
```

---

## The claim

Every platform in this market can tell you what an agent *claims* to do. AGORA
tells you how well it actually does a specific thing in a specific context,
with a confidence interval, and refuses to route work to capabilities that have
not earned it.

**Capability is a claim; proficiency is a measurement.**

Two things make it different, both structural rather than clever:

1. **It enforces.** It sits in the execution path and stops things, where the
   entire observability category watches and scores.
2. **It grades with tests, not judge models.** A universal battery of
   deterministic checks runs on every artifact in every domain at zero marginal
   cost, with domain-specific checks layered on top where they exist.

## Three properties

| Property | What it means | Enforced by |
|---|---|---|
| **Measured trust** | Routing reads a Beta posterior built only from independently verified outcomes, bucketed by context, decayed by age, shrunk when borrowed. Self-reported success is discarded. | `ledger.py` |
| **Structural containment** | Authority narrows monotonically down the delegation tree. Budgets are physical. Termination is enforced by the runtime, not by agent cooperation. | `contracts.py` |
| **Bounded self-improvement** | Four feedback loops at four time constants, each writing outward only. Nothing generated can modify the orchestrator, verifier, policy engine or budget enforcer. | `evolution.py` |

---

## The wedge: the universal floor

Six domain-agnostic deterministic checks run first, on every artifact, in every
domain. They are the floor, not a fallback.

| Check | Question | Why it generalises |
|---|---|---|
| provenance | Does every checkable claim trace to a supplied source? | An invented statistic is the same failure in a landing page, a board memo and a contract summary |
| scope | Did it touch anything the contract barred? | Scope is declared per task, not per domain |
| authority | Did it commit to anything it may not commit to? | Promising a refund is the same overreach in support, sales and legal |
| consistency | Does the output contradict itself? | Headline 60%, body 45% — no domain knowledge needed |
| structure | Are declared sections present? | Schema conformance is universal |
| abstention | Where an input was missing, did it flag or invent? | The anti-hallucination check for domains with no ground truth |

Run against four unrelated domains with zero domain-specific code:

```
$ python bin/universal_verifiers.py
── Legal
   BLOCK provenance: the figure '$250,000' does not trace to any supplied source
   BLOCK authority [legal_advice]: gave legal advice without 'legal.advise' in the envelope
   BLOCK abstention [governing law]: input was unavailable, but the output asserts a value
── Marketing      4 provenance findings · 1 consistency finding
── Sales          2 provenance findings · 3 authority findings
── Planning       4 provenance findings · 1 consistency finding
── Control (clean)
   clean

11 GENUINE FAULTS · 0 FALSE POSITIVES · 0 MODEL CALLS
```

**Being straight about the limit:** these catch *unsourced*, not *false*. A
cited claim that misrepresents its source passes the floor — that is what the
L2 recomputation rung is for, and there is a test that documents exactly this
(`test_the_floor_catches_unsourced_not_false`).

**The false positive that shipped.** An early version flagged an *authorised*
discount offer as an unsourced claim. An offer you are permitted to make is a
performative, not an assertion about the world. That distinction now lives in
the code (`_is_authorised_offer`) and in the clean control document. A noisy
checker is worse than no checker, because people learn to click past it.

---

## Assurance grades

Rather than claim uniform coverage, every capability carries a grade **derived
from the checks its domain pack can actually supply** — never asserted by hand.

| Grade | What backs it | What the agent may do |
|---|---|---|
| **A — Verified** | Deterministic domain checks: tests, solvers, identities | Autonomous within budget, sampled verification |
| **B — Checked** | Recomputation or strong constraints + universal floor | Autonomous, every output verified by a second path |
| **C — Guarded** | Universal floor + rubric | Drafts only, human approves before anything leaves |
| **D — Observed** | Universal floor only | Drafts only, full human review — still with provenance, scope, authority limits, audit trail and cost control |

`python -m agora.cli grades` prints all 38 capabilities. Grade D is not
"unsupported"; it is governance without autonomy. Strategy work resolves to no
pack and grades D, and the code says so rather than pretending otherwise.

One pack can carry different grades per capability: `finance.reconcile` ties
out to the cent (Grade A) while `finance.forecast` has nothing to tie to
(Grade B). Grading the whole pack at its best capability would be exactly the
overclaiming this system exists to avoid.

---

## Measured trust

```
$ python bin/capability_ledger.py

  python · small · familiar    0.91 [0.85–0.96]
                               n=67 · production-measured · routable unsupervised

  swift · small · familiar     0.86 [0.68–0.97]
                               n=12 · harness-verified · verification mandatory

  terraform · novel            0.55 [0.32–0.77]
                               n=67 · harness-verified · verification mandatory
                               borrowed from (pooled root) at distance 2;
                               67 outcomes shrank to 10.0

  rust · large · novel         0.43 [0.31–0.55]
                               n=40 · deprecated · below floor, excluded from routing
```

The third row is the mechanism that matters. With no data for novel Terraform,
the router falls back to a coarser bucket and shrinks the posterior in
proportion to how far it reached — 0.91 collapses to 0.55 with a very wide
interval, and **borrowed evidence can never authorise unsupervised execution**,
however good the number looks.

Beta arithmetic (`stats.py`) is implemented from scratch: regularised
incomplete beta by continued fraction, its inverse by bisection. A control
plane that has to run in a customer's own environment is better off with no
compiled dependencies at all.

**Stated limitation, in the code rather than a footnote:** these posteriors
assume a stationarity that model updates violate. Calendar decay does not
capture a model swap. `CapabilityLedger.stationarity_warnings()` *detects* the
discontinuity; correcting for it is unsolved here.

---

## Layout

```
src/agora/
  contracts.py      TaskContract · AuthorityEnvelope · Budget · TerminationPolicy · the spec gate
  tiering.py        the tier gate — Reflex / Task / Project / Campaign
  artifacts.py      artifacts, sources, instruction stripping, confidence clamping
  blackboard.py     reference passing, never payload passing
  registry.py       extended A2A AgentCards — claims marked advisory, never routed on
  stats.py          Beta posteriors, credible intervals, Thompson sampling
  ledger.py         the Capability Ledger · buckets · decay · borrowing · cross-tenant priors
  router.py         policy filter → deterministic → bandit → constrained LLM · gap log
  adapter.py        the universal socket + process / webhook / human adapters
  memory.py         six scopes, write authority, the canonicity ladder
  audit.py          hash-chained append-only log · cost attribution · replay
  planner.py        recall → compose → synthesise
  orchestrator.py   the master flow (L3, immutable core)
  factory.py        the commitment ladder · charter review · overlap detection · entropy sweep
  evolution.py      four loops · eval-first skill synthesis · the core boundary
  economics.py      parameterised cost model
  fleet.py          the 16 starter charters
  cli.py            which / add-agent / add-skill / add-runtime / audit / rollback / install
  verification/
    universal.py    the six checks
    ladder.py       universal → L0 → L1 → L2 → L3 → escalation · grading · adjudication
    packs.py        eleven domain packs + the published capability index
    corpus.py       the four-domain fault corpus and the clean control
console/AgoraConsole.jsx   fleet console: evidence intervals, nested skills, factory verdicts
bin/                       the reference scripts named in the report
tests/                     244 tests
```

## Adding capability

Three distinct operations get called "adding an agent". Picking the wrong one
is where fleets go bad, so the CLI asks first:

```
$ python -m agora.cli which --occurrences 2
P0-nothing — build nothing — just ask
  · 2 occurrence(s) in 30 days is below the threshold of 3

The cheapest capability is the one you did not build.
```

| What you are adding | Path | Effort |
|---|---|---|
| Your own runtime | subclass `A2AAdapter`, four ADAPT ME blocks | ~1 day |
| A CLI agent (Claude Code) | `ProcessAdapter` | ~1 day |
| Another framework (LangGraph, CrewAI) | `CallableAdapter` around their API | ~1 day |
| A no-code workflow (n8n, Zapier) | `WebhookAdapter` | hours |
| A REST API or internal service | `CallableAdapter` | hours |
| **A human specialist** | `HumanAdapter` — opens a queue item and waits | hours |

The last row is the one that matters. Modelling a person as an agent — with an
authority envelope, a latency profile and a ledger entry — means Grade C and D
work routes through the same machinery as everything else. The system does not
have separate "automated" and "manual" modes; it has one mode with different
assurance grades.

## Economics

```
$ python bin/agora_cost_model.py

Baseline — transcript passing, artifacts inline, all frontier   $ 15.90
+ Reference passing — artifacts by URI, not payload             $  8.13  -48.9%
+ Compiled context — no transcript accumulation                 $  3.96  -51.3%
+ Prompt caching — persona and domain pack                      $  2.22  -43.9%
+ Plan cache and turn caps — 24 turns to 16                     $  1.48  -33.3%
+ Model ladder — local, mid, frontier by need                   $  0.34  -77.0%

47× cumulative
```

**Read this honestly.** The baseline is the default behaviour of most
multi-agent frameworks. Against competent engineering that already passes
references, compacts context and caches prompts, `honest_factor()` computes
**6.5×** — in the 3–8× range the report gives. Do not put 47× in a deck.

The tier overhead rule is enforced rather than hoped for: if synthesising a
plan would cost more than a tenth of a task's whole budget, the control plane
is the expensive part of the task and skips itself.

## Tests

```
$ python -m pytest agora/tests -q
244 passed
```

The suite pins the claims, not just the code paths — the eleven-fault result,
the zero-false-positive control, every published capability grade, the
waterfall figures within a stated tolerance, and the properties that are meant
to be structural (authority cannot widen, budgets cannot be exceeded, borrowed
evidence always forces verification, loops write outward only, a skill cannot
be written before its eval is approved).

## What is *not* here

Honesty about scope, in the same spirit as the assurance grades:

- **No durable workflow engine.** `orchestrator.py` is shaped like one —
  termination guarantees, an audit trail, replay against a pinned model set —
  but it runs in-process. Production wants Temporal or equivalent underneath.
- **No real model calls.** Every model-shaped seam (`TierClassifier`,
  `LLMRouter`, `RubricGrader.score_fn`, `Planner.synthesiser`) takes an
  injected callable and ships with a deterministic stand-in, so the whole
  system is testable without a network.
- **Eleven domain packs, not a library.** Domain packs are the largest hidden
  cost in this system and the top risk in the register. The ones here are
  reference implementations.
- **No multi-tenancy or persistence layer.** `CrossTenantPriors` implements the
  pooling, consent gate and k-anonymity threshold, but the store is in memory.
