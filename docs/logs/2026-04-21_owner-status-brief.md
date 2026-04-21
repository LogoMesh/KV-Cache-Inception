# Owner Status Brief — April 21, 2026

**Project:** KV-Cache Inception (NeurIPS 2026 D&B)
**Deadline:** May 6, 2026 (15 days)
**Prepared from:** All session logs, test data, scripts, and paper draft as of Apr 21

---

## TL;DR

- **Phase 2 complete.** All core infrastructure is implemented and tested. Phase 3 (H100 experiments) is the active blocker.
- **5 experiments planned. 0 run at scale.** Only 2 smoke tests exist (10–15 nodes on a 1B model). No full experimental data.
- **149 tests, all passing.** Suite is smoke-test coverage, not integration or scale coverage.
- **ρ_R bug was fixed Apr 17.** The old gate run data (Apr 16) is invalidated — the gate must be re-run with the corrected implementation before any experiment results are valid.
- **Highest-urgency blocker: OpenReview profiles.** Non-institutional email verification takes up to 2 weeks. May 2 is the safety deadline; abstract submission closes May 4.
- **PR #1 (batch dataset collector)** is code-complete with 10 passing tests. Needs Max to rebase and fix 2 issues before merge.

---

## Session Timeline

**Apr 11** — Phase 2 kickoff. Deep gap analysis written. All Phase 2 modules scaffolded: `kv_mcts.py`, `telemetry_matrix.py`, `orthogonal_escape.py`. 117 tests passing. Linear project set up with 16 issues (LOG-31–LOG-46).

**Apr 14** — DynamicCache runtime compatibility fixed (transformers 5.3 schema changed). FP32 accumulator added for Theorem 1 reversibility. Snapshot-isolated eval cache added. 130 tests passing.

**Apr 15** — Windows Unicode failures in runners fixed. Reproducibility controls added (seed handling, metadata persistence). `croissant_export.py` written with schema validation. Croissant evidence-format language added to paper. NeurIPS hardline rules contract written. 135 tests passing.

**Apr 16** — **Phase 3 gate passed** on `meta-llama/Llama-3.2-1B-Instruct`:
- KV-cache mutability probe: PASSED
- 10-node MCTS smoke test: PASSED
- 200-cycle Lipschitz drift validation: PASSED (FP32 accumulator exact; naive bf16 diverges to 5.97 by cycle 200)

Also: first substantive run on Qwen2.5-Coder-1.5B (15 nodes, seed 20260416, 1.8s). Paper formatted to NeurIPS double-blind spec.

**Apr 17** — Honest assessment session. Key finding: **paper reads as experiments-complete but only smoke tests exist.** ρ_R implementation bug found and fixed (was returning normalized [0,1] dot products; paper Eq. 4 requires raw unbounded dot products). Literature references corrected (COCONUT, CRSM, Okazaki-RAG, Bailey et al.). Unverified paper claims hedged. Experiment-readiness + paper-revision doc created with pre-flight checklist for all 5 experiments.

**Apr 18 (session 1)** — Janitorial pass. Context brief rewritten (52 lines, was 125). 6 stale docs archived to `docs/archive/`. `docs/reviews/` directory removed. Paper references updated.

**Apr 18 (session 2)** — PR #1 reviewed: Max's batch dataset collector (1336 lines added, 10 tests passing, 145/145 tests green on branch). **Two required fixes before merge:** (1) `full_rebuild` leaks stale source artifacts — needs `shutil.rmtree(sources_dir)` before rebuild; (2) branch needs rebase on main (conflict in `croissant_export.py` from commit `b7944bc`). Minor issues: duplicate SHA-256 helper, cross-script import fragility.

---

## Experiment Data That Exists

| Run | Model | Config | Key Results | Artifact |
|---|---|---|---|---|
| Qwen gate run | Qwen2.5-Coder-1.5B-Instruct | 15 nodes, depth 3, branches 3, seed 20260416 | OEI=0.996 (saturated), reward=0.433, σ_H=1.0, ρ_R=0.714, TDS=0.025 | `tmp/runs/2026-04-16/` |
| Llama mutability probe | Llama-3.2-1B-Instruct | Phase 2 gate | max_logit_delta=0.172, revert_delta=0.0, gate_passed=true | gate JSON |
| Llama MCTS smoke | Llama-3.2-1B-Instruct | 10 nodes | OEI=0.883, TDS=0.0084, reward=0.214, elapsed=0.6s | gate JSON |
| Lipschitz drift | Llama-3.2-1B-Instruct | 200 cycles | FP32 accumulator: 0.00e+00 (exact); naive bf16: 5.97 at cycle 200 | `docs/logs/drift_results_llama_gate_2026-04-16.csv` |

**Note:** All runs above used the old ρ_R implementation (normalized [0,1]). The Apr 17 fix changed the output range. Gate must be re-run before Phase 3 results carry any validity.

---

## Test Suite Status

149 tests passing. 0 failures.

| File | Tests | Coverage |
|---|---|---|
| `tests/test_sage.py` | ~30 | Phase A logomesh module unit tests, no LLM calls |
| `tests/test_whitebox.py` | 23 | PerLayerHonestyProjector, RepE probes |
| `tests/test_local_model_interface.py` | ~15 | Phase 2 LocalLlamaOracle KV cache API |
| `tests/test_phase2_modules.py` | 30+ | TelemetryMatrix, OEI, TDS, FP32Accumulator, MCTS smoke |
| `tests/test_croissant_export.py` | 5 | Croissant export schema validation |
| `tests/test_collect_dataset.py` | 10 | Batch dataset collector (PR #1, not yet merged) |

All tests are unit/smoke-level. No integration tests against a live model; no scale tests on H100.

---

## Implementation Status

### logomesh/ modules

| Module | Status | Notes |
|---|---|---|
| `oracle.py` | Complete | BaseModelClient ABC + OpenAIModelClient |
| `local_model.py` | Complete | LocalLlamaOracle; handles dense + MoE; KV-cache API |
| `hneuron_monitor.py` | Complete | Dense path (Llama) + MoE path (router logit entropy); same top-K neurons used globally across layers (per-layer retraining deferred to Phase 3) |
| `whitebox.py` | Partial | PerLayerHonestyProjector fully implemented; 5 probes (GCG, attention, loss landscape, saliency, RepE) are reference-algorithm stubs only |
| `telemetry_matrix.py` | Complete | T_t matrix, 4 diagnostic states, Eq. 8 reward; classify() thresholds are provisional placeholders — must be re-calibrated on H100 |
| `orthogonal_escape.py` | Complete | NullSpaceProjector, OEICalculator (Eq. 10), TDSCalculator |
| `kv_mcts.py` | Complete | ReversibleMCTS, FP32Accumulator (Theorem 1), KVCacheNode; handles both legacy tuple and DynamicCache KV formats |
| `search_policy.py` | Complete | UCB1 bandit, novelty/failure tracking |
| `payload_library.py` | Complete | PayloadEntry, PayloadLibrary with top-k querying |
| `croissant_export.py` | Complete | Croissant 1.1 + RAI 1.0 export; 24-column interventions.csv |
| `evidence_store.py` | Complete | Structured per-run logging, dict-compatible |
| `graders.py` | Complete | PluginGrader, RuleBasedGrader, CompositeGrader |
| `ablation.py` | Complete | 14 toggles for controlled experiments |
| `threat_model.py` | Complete | ThreatModel, GoalTaxonomy, AttackSurface |
| `library_loader.py` | Complete | Battle-time payload scheduling with priority tiers |

### scripts/

| Script | Status | Purpose |
|---|---|---|
| `probe_kv_cache_mutability.py` | Complete | Phase 2 gate: in-place mutation + reversibility validation |
| `measure_lipschitz_drift.py` | Complete | Theorem 1: FP32 accumulator vs naive bf16 drift |
| `run_kv_mcts.py` | Complete | Main MCTS runner — produces JSON + optional Croissant export |
| `run_offline_mcts.py` | Complete | Text-space MCTS baseline (Exp 2 comparison) |
| `train_lat_probes.py` | Complete | LAT probe training for RepE ρ_R calibration |
| `export_kv_mcts_to_croissant.py` | Complete | Post-hoc artifact → Croissant package converter |
| `collect_dataset.py` | Complete (PR #1) | Batch dataset collector; not yet merged |
| Exp 3, 4, 5 scripts | **Not written** | Memory efficiency, reproducibility, cross-model transfer |

---

## Open Blockers

**Hardware:** H100 access confirmed — Lambda rental instances, $400 credits available. Tianyu has also given the OK.
**OpenReview:** All 4 co-authors (Josh, Max, Bakul, Asuka) already have verified profiles.

| Priority | Blocker | Owner | Deadline / Notes |
|---|---|---|---|
| **HIGH** | Re-run gate smoke test with corrected ρ_R implementation | Team | Before Experiment 1 — old data invalidated by Apr 17 fix |
| **HIGH** | Re-calibrate classify() thresholds based on real H100 ρ_R distribution | Team | Before any experiment uses DiagnosticState |
| **HIGH** | Experiment priority decision — which of the 5 experiments can actually run before May 5? | Team | Now — drives paper scope decision |
| **MEDIUM** | PR #1 merge (Max: rebase on main + 2 bug fixes) | Max / Josh | Before batch-collection pipeline is usable |
| **MEDIUM** | gpt-oss-20b licence and availability confirmed on HuggingFace | Josh | Before Experiment 1 on 20B model |

---

## Known Technical Gaps

**OEI signal range:** The gate run saturated OEI at 0.996 — not useful for discrimination. Must find an α range where OEI ∈ [0.3, 0.9] before Experiment 1 is informative. Currently unknown what α achieves this for the target model.

**ρ_R calibration:** All existing telemetry data used the old normalized implementation. The new raw dot-product output is unbounded. The classify() threshold (currently hard-coded to 0.7) is meaningless until re-calibrated against H100 output distribution.

**Per-layer H-Neuron monitoring:** `hneuron_monitor.py` applies the same top-K neuron indices across all layers. This is a Phase 2 approximation. Paper Eq. 3 implies per-layer calibrated σ_H^(l). Phase 3 requires retraining with layer-specific calibration.

**Orthogonal Escape unvalidated:** OE is listed as a headline contribution but remains a hypothesis. The paper's Limitations section correctly flags this — but it is still 1 of 3 claimed contributions. Experimental validation is the highest empirical burden item.

**Evidence burden (reviewer risk):** Paper requires a validation chain: continuous KV-cache interventions → detectable deceptive intent in behavioral outputs. This chain is not yet established empirically.

---

## Paper Status

| Item | Status |
|---|---|
| Canonical draft | `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` |
| NeurIPS E&D double-blind formatting | Done |
| Present-tense unverified claims hedged | Done (Apr 17) |
| Literature corrections applied | Done (COCONUT, CRSM, Okazaki-RAG, Bailey et al.) |
| Results sections | Empty — awaiting H100 experiments |
| Final anonymization pass | Not done |
| PDF compilation check | Not done |

**Three scope options still on the table** (decision needed at next team meeting):
- **Option A (focused):** Submit with 2 contributions (T_t matrix + Reversible MCTS), drop OE as headline. Most achievable.
- **Option B (dataset-first):** Reframe as benchmark/dataset paper around the Croissant artifact. Requires scale.
- **Option C (delay):** Submit to a later venue. Depends on team timeline and advisor input.

---

## What Needs to Happen This Week

1. **Team:** Decide which of the 5 experiments to run (and what to cut) before May 5.
2. **Max:** Rebase PR #1 on main, apply the 2 required fixes, re-run tests.
3. **Team:** Re-run gate with corrected ρ_R on H100; find valid α range for OEI.
4. **Josh:** Confirm gpt-oss-20b licence/availability on HuggingFace before targeting it for Experiment 1.
