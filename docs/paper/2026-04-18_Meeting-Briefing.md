# LogoMesh Team Meeting — April 18, 2026

**Time:** 8–9 PM  
**Attendees:** Josh, Max, Bakul, Asuka (+ new members)  
**Deadline:** NeurIPS 2026 E&D track — May 6, 2026 (**18 days**)

---

## What This Meeting Needs to Decide

Three decisions drive everything else. The paper cannot be responsibly scoped, the experiments cannot be prioritized, and the claims cannot be hedged correctly until these are settled tonight.

1. **OE scope** — Is Orthogonal Escape a validated contribution, a preliminary observation, or a discussion point?
2. **Experiment priority** — Which of the five experiments will definitely produce results before May 5?
3. **Operational items** — OpenReview profiles, H100 access, hosting platform.

---

## For New Members: What We've Built

**The paper:** *KV-Cache Inception* — a framework for detecting and stress-testing alignment faking in LLMs by intervening directly in the model's internal representations (KV-cache) rather than testing behavior at the output level.

**Three contributions:**
1. **T_t matrix** — a per-layer telemetry matrix fusing two independent signals (H-Neuron stress σ_H and RepE honesty ρ_R) into a 2×L diagnostic matrix per MCTS node. Classifies nodes into four states: Creative Synthesis, Contextual Fabrication, Alignment Faking, Orthogonal Escape.
2. **Reversible MCTS** — MCTS that mutates KV-cache tensors in-place and rolls back exactly via FP32 accumulators, reducing memory from O(b^d · M) to O(M + d·K). Enables deep search on a single H100 for 20B models.
3. **Orthogonal Escape formalization** — formal definition + experimental protocol for the hypothesis that steering pressure along monitored directions forces deceptive encodings into unmonitored orthogonal subspaces.

**Track:** NeurIPS 2026 Evaluations & Datasets (E&D). The dataset of MCTS intervention traces with full Croissant 1.1/RAI metadata is a primary deliverable alongside the paper.

**Current implementation state:**
- All three contributions are implemented and unit-tested (139/139 green)
- Phase 3 gate passed 2026-04-16 on Llama-3.2-1B: reversibility confirmed (FP32 drift = 0.00), non-degenerate telemetry confirmed (OEI = 0.883, σ_H = 0.003, ρ_R = 0.362)
- Croissant export pipeline is production-quality and mlcroissant-validated
- **No H100 experiments have been run.** All performance claims in the paper are projected.

---

## Decision 1 — Orthogonal Escape Scope

The paper's third contribution is the "first formal definition and experimental protocol for Orthogonal Escape." The Limitations section honestly states it "has not yet been empirically validated." If Experiment 2 does not run before May 6, that remains true.

| Option | What it means | What changes in the paper |
|---|---|---|
| **A — Run Experiment 2 before May 6** | OE becomes a validated contribution | Present-tense claims become empirical results; Limitations caveat removed |
| **B — Reframe as emerging observation** | OE stays in the paper as preliminary evidence + future work | Contribution 3 reframed: "we describe preliminary observations consistent with orthogonal redistribution and propose a formal protocol for systematic study" |
| **C — Drop OE from contributions** | Two-contribution paper; OE moves to Discussion | Contribution 3 removed; paper tightened around T_t and Reversible MCTS |

**What makes A viable:** H100 access confirmed quickly + signal quality is good at scale.  
**What makes B or C the safer call:** If H100 time is uncertain, a tighter honest paper beats an overscoped one.

**Team decision:** ____________________

---

## Decision 2 — Experiment Priority Order

Five experiments are planned. The answer to *"which will definitely produce results before May 5?"* determines the paper's claim posture.

| Experiment | What it validates | Dependency | Estimated H100 time |
|---|---|---|---|
| **Exp 1** — OE Detection (α sweep, OEI/TDS curves) | Core diagnostic metric; alignment faking signal | α calibration run first (OEI was saturated at ~0.996 on Qwen 1B — must find discriminative α range) | TBD |
| **Exp 2** — MCTS vs. text-space comparison | MCTS advantage over GCG / text-MCTS | Exp 1 complete | TBD |
| **Exp 3** — Memory efficiency | Theorem 1 empirical confirmation; 80GB GPU claim | Independent | Low — benchmarking only |
| **Exp 4** — Evaluation reproducibility | LLM-as-Judge variance claim | 500 artifacts needed | TBD |
| **Exp 5** — Cross-model transfer (Procrustes) | Generalization across scales | Exp 1 + 7B model selected | TBD |

**Dependency note:** Exp 1 results feed into Exps 2 and 5. Exp 3 is independent. Exp 4 requires a large artifact set that only exists after Exp 1 or 2 at scale.

**Before any experiment runs on H100:**
- H100 access confirmed ❌
- Gate smoke test re-run with corrected ρ_R ❌
- classify() thresholds recalibrated (ClassifyConfig ready, needs data) ❌
- LAT probes trained on target model ❌
- Target model selected and verified (gpt-oss-20b availability unconfirmed) ❌

**Priority table — fill tonight:**

| Experiment | Run before May 5? | Owner | Notes |
|---|---|---|---|
| Exp 1 | | | |
| Exp 2 | | | |
| Exp 3 | | | |
| Exp 4 | | | |
| Exp 5 | | | |

---

## Decision 3 — Operational Items

| Item | Status | Owner | Action needed |
|---|---|---|---|
| OpenReview profiles — Josh, Max, Bakul, Asuka | ❌ Unknown | All | **Critical: non-institutional emails take up to 2 weeks for moderation. May 2 is the safe cutoff for May 4 abstract deadline. Verify or create profiles tonight.** |
| H100 access from Tianyu's lab | ❌ Pending | Josh | Confirm timeline; gates all Phase B work |
| Dataset hosting platform | ❌ Undecided | Team | HuggingFace (gated) is tentative recommendation — see `neurips-2026-data-requirements.md` |
| Human ground-truth labeling protocol | ❌ Undecided | Team | Required for `ground_truth_label` field in dataset; internal / MTurk / other? |
| Anonymization strategy for dataset URL | ❌ Undecided | Team | Anonymous org on hosting platform? Private-during-review? |

---

## Open Claims Requiring Team Input

From the claims audit — items that need a decision before paper is revised:

| # | Claim | Question |
|---|---|---|
| 4 | "first formal definition of Orthogonal Escape" | Defensible after literature search? Has anyone formalized OE-like phenomena? |
| 5 | O(b^d · M) → O(M + d·K) memory reduction | Is the algebraic derivation in Theorem 1 complete and correct? |
| 6 | "enables deep MCTS on single 80GB GPU for 20B models" | Experiment 3 confirmed in scope? Needed to keep current future-tense hedge. |
| 7 | LLM-as-Judge divergences 0.10–0.20 | Does the cited paper [llmasjudge2025limitations, arXiv:2503.05061] actually state this figure? |
| 11 | Accumulator adds ~50MB overhead for 20B model | Verify calculation: L × S'=10 × d_model × 4 bytes |

---

## Key Reference Docs

| Doc | What it's for |
|---|---|
| [experiment-readiness-and-paper-revision.md](experiment-readiness-and-paper-revision.md) | Pre-flight gates + exact paper text that changes after each experiment |
| [paper-revision-claims-audit.md](paper-revision-claims-audit.md) | Full claims inventory + decision record table |
| [neurips-2026-submission-guidelines.md](neurips-2026-submission-guidelines.md) | Deadlines, formatting, anonymization rules |
| [neurips-2026-data-requirements.md](neurips-2026-data-requirements.md) | Croissant compliance state, hosting options |
