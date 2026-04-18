# Paper Revision Claims Audit

**Based on:** `docs/NeurIPS/04.17.2026-NeurIPS-Research-Proposal.tex`
**Review session:** Saturday April 18, 2026 — LogoMesh Weekly Meeting
**Last updated:** 2026-04-18 (pruned resolved sections; updated claims table)

**Purpose:** Meeting input document. Every strong claim in the paper audited against
current implementation state. Team fills in the Decision column at the meeting; those
decisions drive paper edits and experiment scope.

---

## 0. Structural Concerns (Read Before Claim-by-Claim Review)

### 0.1 The OEI Signal Quality Issue

The Qwen gate run showed OEI saturating at ~0.996 across all non-root nodes. A diagnostic
metric that returns ~1.0 for every node is not discriminating between alignment-faking and
non-faking states — it may be responding to perturbation magnitude rather than alignment
structure. Three interpretations:

1. **α is too large** — perturbation overwhelms the representation regardless of content.
2. **1.5B model is too small** — at this scale, KV-cache interventions produce chaotic rather
   than structured orthogonal redistribution.
3. **OEI is correctly measuring that all perturbed states escape to null spaces** — a finding,
   but requires a baseline comparison to distinguish from interpretations 1 or 2.

The Llama gate run showed OEI = 0.883, which is more discriminative.

**Required action before Experiment 1:** Sweep α ∈ {0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0}
on a single prompt and confirm OEI varies meaningfully across the range. This is gated in
`experiment-readiness-and-paper-revision.md` as a pre-flight requirement for Experiment 1.

---

### 0.2 Orthogonal Escape: Formalization Without Validation

The paper's third contribution is the "first formal definition and experimental protocol for
Orthogonal Escape." The Limitations section honestly states it "has not yet been empirically
validated in the specific setting of KV-cache interventions." If Experiment 2 does not run
before May 6, this contribution is math without confirmation. E&D reviewers will ask where
the evidence is.

**Three options for the meeting to decide:**

| Option | What it means | Risk | Strength |
|---|---|---|---|
| **A. Run Experiment 2 before May 6** | OE becomes a validated contribution | Timeline pressure; H100-dependent; signal quality uncertain | All three contributions grounded |
| **B. Reframe OE as "emerging observation"** | Demote to preliminary evidence + future work; describe gate run observations consistent with OE | Smaller claim | Honest; still novel; reviewer-resilient |
| **C. Drop OE from contributions** | Two-contribution paper (T_t + Reversible MCTS); OE in discussion only | Loses most ambitious claim | Cleanest scope; strongest if H100 time is tight |

---

## 1. Claims Inventory

All strong claims in the current paper. Decide posture before H100 runs begin so experiments
confirm decided claims rather than generating post-hoc justifications.

| # | Claim | Section | Type | Current posture | Status |
|---|---|---|---|---|---|
| 1 | "first MCTS via continuous arithmetic KV-cache mutations with exact FP32 rollback" | 1.1 / 2.4 | Priority claim | Reframed from "first MCTS in latent space" after literature review (COCONUT, CRSM invalidated broader claim) | ✅ Applied (commit e1df154) |
| 2 | "all existing MCTS-based red-teaming operates in text space" [wu2025mcts, damon2025, li2025ace] | 2.4 | Comparative | Unhedged; three citations | ✅ Verified by literature review — all three cited works confirmed text-space |
| 3 | "no prior work deployed heuristic tree-search in KV-cache at inference time without gradient access" | 2.2 | Priority claim | Qualified from broader "no prior adversarial latent search" after Bailey et al. and RL-Obfuscation found | ✅ Applied (commit e1df154) |
| 4 | "first formal definition and experimental protocol for Orthogonal Escape" | 1.1 | Priority claim | Unhedged | ❓ Team decision — see Section 0.2 |
| 5 | O(b^d · M) → O(M + d · K) memory reduction | 1.1 | Mathematical claim | Unhedged | ❓ Verify algebraic derivation in Theorem 1 section is complete |
| 6 | "enables deep MCTS on a single 80GB GPU for models up to 20B parameters" | 1.1 | Capability claim | Hedged to "designed to enable... verify in Experiment 3" (commit ec7a897) | ⚠ Future-tense applied; confirm Experiment 3 is in scope |
| 7 | LLM-as-Judge divergences 0.10–0.20 on normalized scales | 2.5 | Quantitative cited claim | Citation [llmasjudge2025limitations] given | ❓ Verify cited paper states this figure |
| 8 | "FP32 accumulator achieves zero semantic degradation" | 4.2 | Empirical | Unhedged | ✅ Confirmed — Lipschitz drift = 0.00e+00 over 200 cycles (gate run seed 20260416) |
| 9 | "Orthogonal Escape has not yet been empirically validated" | Limitations | Honest limitation | Present in draft | ✅ Keep — required for reviewer credibility |
| 12 | σ_H confound risk: stress signal may conflate hallucination, mode-collapse, and strategic masking unless discriminant controls added | Limitations | Reviewer attack surface | **Not in current draft** | ❓ Add to Limitations before submission |
| 13 | OEI false negative risk: may miss nonlinear/manifold-structured redistribution not concentrated along linear null-space directions | Limitations | Reviewer attack surface | **Not in current draft** | ❓ Add to Limitations before submission |
| 14 | Evidence burden: paper needs explicit validation chain linking continuous latent interventions to discrete deceptive intent outcomes | Limitations | Reviewer attack surface | **Not in current draft** | ❓ Add to Limitations before submission |
| 10 | "20B model KV cache ≈ 40GB in bf16" | 4.2 | Specification | Unhedged | ✅ Calculable from architecture; keep |
| 11 | "accumulators add ~50MB overhead for 20B model with S'=10 positions" | 4.2 | Memory claim | Unhedged | ❓ Verify calculation |

---

## 2. Decision Record (Fill at Meeting)

| Item | Decision | Owner | Notes |
|---|---|---|---|
| OE scope — Option A, B, or C (Section 0.2) | | | Drives whether Experiment 2 is in scope |
| Claim 4: "first formal definition of OE" — defensible? | | | |
| Claim 5: memory reduction theorem — proof complete? | | | |
| Claim 6: Experiment 3 confirmed in scope? | | | Required to keep current future-tense hedge |
| Claim 7: LLM-as-Judge figure — citation verified? | | | |
| Claim 11: accumulator memory calculation — verified? | | | |
| Experiment priority order (which run before May 5?) | | | Drives all claim framing; see experiment-readiness doc |
| Claim 12: σ_H confound risk — add to Limitations? | | | |
| Claim 13: OEI false negative risk — add to Limitations? | | | |
| Claim 14: Evidence burden — add validation chain to Limitations? | | | |

---

## Appendix: Verbatim Abstract (for reference at meeting)

> Current approaches to evaluating the safety and alignment of large language models (LLMs)
> rely predominantly on black-box behavioral testing or external LLM-as-a-Judge frameworks.
> These methods are fundamentally blind to a critical failure mode: *alignment faking*, wherein
> a model strategically complies with safety constraints during evaluation while preserving
> misaligned capabilities in its latent representations. Recent work has demonstrated that LLMs
> can be trained via reinforcement learning to evade latent-space monitors while maintaining
> identical black-box behavior (Gupta and Jenner, 2025), and that deceptive behaviors persist
> through standard safety training (Hubinger et al., 2024). We argue that detecting and
> stress-testing alignment faking requires moving evaluation *inside* the model---from behavioral
> auditing to mechanistic intervention. To this end, we propose **KV-Cache Inception**, a
> framework that combines three novel components: (1) a **multi-dimensional micro-telemetry
> matrix** that fuses bottom-up hallucination-associated neuron (H-Neuron) activation tracking
> with top-down Representation Engineering (RepE) projections to provide a continuous,
> differentiable signal for distinguishing genuine reasoning from compliance-driven fabrication;
> (2) a **Reversible MCTS algorithm** that conducts Monte Carlo Tree Search directly in the
> model's KV-cache latent space---rather than in text space---to systematically explore the
> intervention landscape for alignment-breaking attack vectors, using a mathematically reversible
> state-rollback mechanism with FP32 accumulators to eliminate memory scaling barriers on
> commodity hardware; and (3) the formalization of **Orthogonal Escape**---the phenomenon
> whereby evolutionary pressure applied to a model's monitored latent subspace forces deceptive
> encodings into unmonitored orthogonal null spaces---together with the first systematic
> experimental protocol for its detection and characterization. All released evidence artifacts
> are specified in dual form: raw runtime JSON plus a standards-compliant Croissant 1.1 JSON-LD
> manifest (with RAI extension metadata) for reproducible dataset packaging and external
> validation.

*Note: abstract reflects the `04.02.2026` draft. Verify against `04.17.2026` before the meeting.*

---

## Update Log

| Date | What changed |
|---|---|
| 2026-04-16 | Initial version created from paper audit (commit 041087b) |
| 2026-04-17 | Section 0 structural concerns added |
| 2026-04-18 | Pruned resolved sections (ρ_R fix, all three `\reviewedit{}` edits done, superseded structural sections); updated claims table to reflect e1df154 rewrites; tightened to meeting-input focus |
