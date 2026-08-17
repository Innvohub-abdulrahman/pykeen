# Report figures → implementation

Where each of the report's twenty-four figures lives in the code, and what
test holds it honest. Useful for reviewing whether the implementation actually
says what the document says.

## 1 · The system

| Figure | Mechanism | Code | Test |
|---|---|---|---|
| 1 · Layer stack | L3 immutable core, mutable periphery | `evolution.IMMUTABLE_CORE`, `assert_not_core` | `test_nothing_generated_may_touch_the_core` |
| 2 · Component topology | Control plane sits between ingress and runtimes; agents never call each other | `orchestrator.Orchestrator`, `adapter.Performative` | `test_a_project_runs_end_to_end_and_verifies` |

## 2 · Functionality

| Figure | Mechanism | Code | Test |
|---|---|---|---|
| 3 · Functional groups | what/who/may/did/learned/prove | the module layout itself | — |
| — · Function catalogue | every row | see the module map in `README.md` | the suite |
| — · The universal socket | four ADAPT ME blocks | `adapter.A2AAdapter` | `bin/a2a_adapter_template.py --selftest` (11 checks) |

## 3 · Task flow

| Figure | Mechanism | Code | Test |
|---|---|---|---|
| 4 · Master flow | tier → plan → contract → gate → route → execute → verify → ledger | `orchestrator.Orchestrator.run` | `test_end_to_end.py` |
| 4 · The acceptance-criteria gate | refuses rather than attempts | `contracts.specification_gate` | `test_a_task_with_no_checkable_criteria_is_refused_not_attempted` |
| 5 · Task tiering | Reflex / Task / Project / Campaign | `tiering.TierGate`, `TIER_POLICIES` | `test_the_gate_will_not_be_talked_down` |
| 6 · Three-stage routing | policy filter → deterministic → bandit → constrained LLM | `router.Router.route` | `test_ledger_and_router.py` |
| 6 · Borrowed evidence forces verification | shrinkage by generalisation distance | `ledger.CapabilityLedger.read` | `test_borrowed_evidence_always_forces_verification` |
| 7 · Worked trace | plan recall, budget subdivision, blind verification, ledger update | `orchestrator`, `planner`, `audit` | `test_every_step_is_audited_and_the_chain_holds` |

## 4 · Verification

| Figure | Mechanism | Code | Test |
|---|---|---|---|
| 8 · Verification tiers | universal → L0 → L1 → L2 → L3 → L5 | `verification.ladder.VerificationLadder` | `test_the_floor_gates_everything_above_it` |
| 8 · Contest and adjudication | run a test, else reliability gap, else escalate | `verification.ladder.adjudicate` | `test_a_narrow_gap_escalates_to_a_human` |
| — · The universal floor | six checks | `verification.universal` | `test_universal_verifiers.py` (30 tests) |
| 9 · Faults caught by domain | 11 faults, 0 false positives, 0 model calls | `verification.corpus` | `test_corpus_catches_every_seeded_fault` |
| 10 · Assurance coverage | grade derived from supplied checks | `verification.ladder.grade_capability` | `test_every_capability_grades_exactly_as_published` |

## 5 · Improvement

| Figure | Mechanism | Code | Test |
|---|---|---|---|
| 11 · Four nested loops | routing / plans / skills / agents, writing outward only | `evolution.LOOPS`, `Loop.may_write_to` | `test_a_plan_cannot_rewrite_the_router` |
| 12 · Same number, four facts | evidence classes | `ledger.EvidenceClass` | `test_two_samples_are_a_guess_not_a_measurement` |
| 13 · Interval narrowing | Beta posteriors and credible intervals | `stats.BetaPosterior` | `test_the_interval_narrows_as_evidence_accumulates` |
| 14 · Eval-first promotion | gap → eval → approval → skill → sandbox → shadow → canary | `evolution.SkillSynthesis` | `test_the_eval_comes_before_the_skill` |
| 15 · Charter review | commitment ladder, overlap, retirement criteria | `factory.AgentFactory` | `test_factory_and_evolution.py` |

## 6 · Containment

| Figure | Mechanism | Code | Test |
|---|---|---|---|
| 16 · Memory scopes | write / propose / never; canonicity ladder | `memory.MemoryStore` | `test_only_a_human_promotes_to_canonical` |
| 16 · Contradiction on write | never on read | `memory.MemoryStore.detect_contradiction` | `test_contradiction_is_detected_on_write_not_on_read` |
| 17 · Envelope intersection | authority narrows monotonically | `contracts.AuthorityEnvelope.intersect` | `test_escalation_is_structurally_impossible` |
| — · Untrusted boundary | instruction stripping, confidence clamping | `artifacts.strip_instruction_shaped` | `test_instruction_shaped_content_is_stripped` |

## 7 · The fleet

| Figure | Mechanism | Code | Test |
|---|---|---|---|
| 18 · Fleet at maturity | 16 charters across 5 phases | `fleet.starter_charters` | `test_all_sixteen_charters_validate` |
| 19 · Three paths | add-runtime / add-skill / add-agent / build nothing | `factory.which`, `cli` | `test_the_ladder_picks_the_smallest_thing_that_works` |

## 8 · Economics

| Figure | Mechanism | Code | Test |
|---|---|---|---|
| 20 · Cost waterfall | five levers, applied cumulatively | `economics.waterfall` | `test_the_waterfall_matches_the_published_figures` |
| 21 · Scenarios | feature task, email triage, mobile app | `economics.scenarios` | `test_scenarios_land_near_the_published_figures` |
| — · The honest number | 3–8× against competent engineering | `economics.honest_factor` | `test_the_honest_number_is_smaller_than_the_headline` |

## 9–10 · Market and delivery

Figures 22 (positioning), 23 (adoption by org size) and 24 (roadmap) are
strategy rather than mechanism, and have no implementation. Two claims made
near them *are* implemented, because they carry engineering consequences:

| Claim | Code | Test |
|---|---|---|
| Cross-tenant priors must be designed in from day one, not retrofitted | `ledger.CrossTenantPriors` — consent is a required argument, k-anonymity is enforced | `test_pooling_requires_consent` |
| Model non-stationarity is unsolved | `ledger.stationarity_warnings` detects it and claims nothing more | `test_non_stationarity_is_detected_even_though_it_is_not_solved` |

## Appendices

| Appendix | Code |
|---|---|
| A · Capability index | `verification.packs.CAPABILITY_INDEX` — 38 capabilities, each asserted against its derived grade |
| B · Commands | `bin/*.py` and `agora.cli` |
| C · Artifacts | every module in `src/agora`, plus `console/AgoraConsole.jsx` |
