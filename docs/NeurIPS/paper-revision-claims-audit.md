# Paper Revision Claims Audit

**Based on:** `docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex` (commit 041087b, 2026-04-16)
**Review session:** Saturday April 18, 2026 — LogoMesh Weekly Meeting

**Purpose of this document:** Review every strong claim, proposed edit, and comparison statement
before scaled H100 experiments begin. The principle governing this audit: experiments should
confirm claims the team has already decided on — not generate post-hoc justifications for
assertions added under uncertainty. Every claim in the submitted paper will be checked by
reviewers. Decide the posture here, then the paper is edited to match, then experiments run.

**How to use at the meeting:** Start with Section 0 (structural concerns — read before anything
else). Then work through Sections 1–3. Fill in the "Team decision" column in each table. Assign
owners. Section 4 is the blank decision record.

---

## 0. Structural Concerns (Read Before Claim-by-Claim Review)

*Added 2026-04-17. These are upstream of individual claim decisions — they frame what the
claim-by-claim review is actually deciding.*

### 0.1 The experiment-claim mismatch

The paper is written as if experiments are done or nearly done. They are not. Five experiments
are described in careful detail in Sections 5.1–5.4; none have been run at scale. The only
empirical evidence currently in hand is a 15-node smoke test on Qwen 1.5B and a 10-node gate
run on Llama 1B confirming prototype functionality. Every performance claim in the paper
(detection rates, OEI/TDS curves, ablation breakdowns, calibration overhead at 20B, cross-model
transfer success) is projected, not measured.

This is not just a hedging problem for individual claims — it is a structural question about
what the paper can honestly assert as of May 6. If the H100 experiments run and produce clean
results, the present-tense claims become true and the paper is strong. If they don't — due to
timeline, access delays, signal quality, or unexpected findings — the mismatch becomes visible
to reviewers in the worst possible way.

**Proposed resolution:** At the meeting, explicitly decide which experiments will definitely run
before May 6 (given H100 access timeline) and scope the paper's claims to match only those.
Convert claims for unrun experiments to future-tense framing now. Do not assume experiments
will run and then scramble to hedge if they don't.

### 0.2 The OEI signal quality issue

The Qwen gate run showed OEI saturating at ~0.996 across all non-root nodes. A diagnostic
metric that returns ~1.0 for every node is not discriminating between alignment-faking and
non-faking states — it is responding to the perturbation magnitude itself. Three interpretations:

1. **α is too large** — perturbation overwhelms the representation regardless of content.
2. **1.5B model is too small** — at this scale, KV-cache interventions produce chaotic rather
   than structured orthogonal redistribution.
3. **OEI is correctly measuring that all perturbed states escape to null spaces** — which would
   be a finding, but requires a baseline comparison to distinguish from interpretation 1 or 2.

The Llama gate run showed OEI = 0.883, which is better. Before the H100 experiments,
**determine the α range where OEI varies meaningfully across nodes** — otherwise the paper's
core diagnostic metric is not functioning as claimed in the abstract and contributions.

**Proposed resolution:** Add an α sweep as a precursor to Experiment 1. Treat OEI calibration
as a prerequisite, not an assumption.

### 0.3 The ρ_R bug affects more than the data

The current ρ_R implementation returns a normalized [0,1] value instead of the raw dot product
per Eq. 4. This flows through to the T_t matrix, the diagnostic state classifier, the MCTS
reward function (Eq. 8), and the OEI/TDS calculations — all of which consume ρ_R as an input.
The telemetry matrix T_t is not currently implemented as specified in the paper. This is not a
data quality issue only; it is an implementation correctness issue. All current gate run results
are produced by a system that deviates from the paper's equations.

**Proposed resolution:** Fix ρ_R (Chunk 1A) before any other experiment work. Re-run the gate
smoke test with the corrected implementation and confirm the signal is qualitatively different
before treating any existing results as paper-valid.

### 0.4 Orthogonal Escape: formalization without validation

The paper's third contribution is the "first formal definition and experimental protocol for
Orthogonal Escape." The Limitations section honestly states it "has not yet been empirically
validated in the specific setting of KV-cache interventions." This is good faith. But the
tension is: if Experiment 2 (OE characterization) does not run before May 6, then the third
contribution is a formalization of a hypothesis — math without confirmation. NeurIPS reviewers
for the E&D track, who are specifically evaluating whether the submission enables meaningful
evaluation, will ask: where is the evidence that this phenomenon exists and that the proposed
metrics capture it?

**Three options for the meeting to decide:**

| Option | What it means | Risk | Strength |
|---|---|---|---|
| **A. Run Experiment 2 before May 6** | OE becomes a validated contribution | Timeline pressure; depends on H100 and signal quality | Paper has all three contributions grounded |
| **B. Reframe OE as "emerging observation"** | Demote to preliminary evidence + future work; describe observations consistent with OE in gate run data | Smaller contribution claim | Honest; still novel; doesn't collapse under reviewer scrutiny |
| **C. Drop OE from contributions, keep as related observation** | Two-contribution paper (T_t + Reversible MCTS); OE in discussion section | Loses the most ambitious claim | Cleanest scope; strongest submission if H100 time is tight |

### 0.5 Scope question for the meeting

Given the May 6 deadline and the constraints above, the team should explicitly answer:

> **"Which of the five experiments will definitely produce results before May 5?"**

The answer to that question determines the paper. Everything else follows from it. A two-experiment
paper that delivers on its claims is stronger than a five-experiment paper where three results are
projected. E&D reviewers have seen overscoped proposals before.

---

## 1. Active `\reviewedit{}` Markup

There are three passages currently wrapped in `\reviewedit{}` (shown in blue in compiled PDF).
These were applied in the previous session based on paper-commenter feedback. The team should
decide whether each edit stands, is modified, or is reverted.

---

### Edit A — LAT Acronym (Section 4.1, line 158)

**Current text:**
> We apply the **\reviewedit{Linear Artificial Tomography (LAT)}** procedure from Zou et al. (2023)

**What the edit does:** Introduces the LAT acronym explicitly at first use.

**Risk level:** LOW.

**Analysis:** Defining an acronym at first use is correct practice — NeurIPS style and general
academic convention both require it. The acronym "LAT" appears later in the paper; without this
definition it is undefined on first encounter. This edit is a clarity improvement with no
scientific or rhetorical implications.

**Recommended decision:** KEEP. Remove the `\reviewedit{}` wrapper (keep the text itself).

| Decision | Owner | Notes |
|---|---|---|
| ☐ Keep · ☐ Modify · ☐ Revert | | |

---

### Edit B — Novelty Hedge on MCTS (Contribution 2, line 73)

**Current text:**
> We introduce **\reviewedit{what is, to our knowledge,}** the first Monte Carlo Tree Search
> algorithm that operates directly in the continuous latent space of a Transformer's key-value
> cache, rather than in the discrete text space used by all prior MCTS-based red-teaming
> approaches [wu2025mcts, damon2025, li2025ace].

**What the edit does:** Adds a hedge — "to our knowledge" — before the "first" novelty claim.
This was applied from paper-commenter feedback as a precaution.

**The problem with keeping it:**
"To our knowledge" is a signal to reviewers that the literature search was not completed before
making the novelty claim. This is worse than either (a) asserting confidently after actually
verifying, or (b) removing the "first" frame entirely. If a reviewer finds prior work that the
authors should have found, a hedge provides no protection — it just makes the authors look less
thorough. Reviewers can and will search for prior work. If they find it, a hedge looks like an
admission of corner-cutting rather than good-faith uncertainty.

**Three paths:**

**Path 1 — Verify, then assert confidently:**
Search arXiv and Semantic Scholar for MCTS operating in transformer KV-cache or continuous
latent space (not text-space prompt mutation). If nothing found → remove the hedge entirely:
> "We introduce **the first** Monte Carlo Tree Search algorithm that operates directly in the
> continuous latent space..."

Risk: reviewer finds something the team missed.

**Path 2 — Drop the priority frame (recommended):**
Remove both the hedge and the "first" claim. The contribution is fully characterized by the
contrast with prior work:
> "We introduce a Monte Carlo Tree Search algorithm that operates directly in the continuous
> latent space of a Transformer's key-value cache, rather than the discrete text space used
> by all prior MCTS-based red-teaming approaches [wu2025mcts, damon2025, li2025ace]."

The statement "rather than the discrete text space used by all prior MCTS-based red-teaming
approaches" is a substantive scientific claim that can be verified against citations. It does
not require a "first" assertion to be impactful. This framing is also more resilient — if prior
work in latent-space MCTS exists in a different domain (e.g., planning, not red-teaming), the
claim is still true as stated because it refers specifically to red-teaming approaches.

**Path 3 — Keep "to our knowledge":**
Acceptable in some venues. Not recommended here given the analysis above.

**Recommended decision:** Path 2. Verify the cited prior works are correctly described as
text-space only, then drop the "first" framing in favor of the contrast statement.

| Decision | Owner | Notes |
|---|---|---|
| ☐ Path 1 · ☐ Path 2 · ☐ Path 3 · ☐ Other | | |

---

### Edit C — GCG Comparison Paragraph (Section 2.4, line 99)

**Current text (entire passage is wrapped in `\reviewedit{}`):**
> Non-MCTS adversarial search methods face a related but distinct limitation.
> Gradient-based attacks such as Greedy Coordinate Gradient (GCG) [zou2023gcg] operate in
> token embedding space and are therefore equally constrained to the discrete combinatorial
> surface: they optimize token sequences via gradient signals but cannot navigate the continuous
> geometry of the model's internal representation space. MCTS provides the exploration
> semantics---branching, rollout, and UCB1 selection---that allow systematic coverage of the
> latent intervention landscape in a way that both prompt-mutation and gradient-based
> approaches **structurally cannot**.

**What the edit does:** Adds a paragraph contrasting GCG (gradient-based discrete attacks)
against MCTS (continuous latent-space exploration) to justify why MCTS is necessary.

**What is factually correct:**
- GCG optimizes in token embedding / discrete token space — this is correct by construction,
  stated in the GCG paper [arXiv:2307.15043].
- GCG cannot directly manipulate KV-cache tensors — correct.
- MCTS as implemented here does operate in continuous KV-cache latent space — confirmed by
  gate run and implementation.

**What is rhetorically heavy:**
"Structurally cannot" overstates the claim. GCG can in principle be extended to operate on
latent representations — the point is that the standard formulation does not, and was not
designed to. "Structurally cannot" invites a reviewer to argue about theoretical extensions.
"Are not designed to navigate" is accurate and defensible without creating that surface.

**Recommended decision:** KEEP the paragraph (it is scientifically useful context), but
replace "structurally cannot" with "are not designed to navigate." Remove `\reviewedit{}`
wrapper once the team agrees on the wording.

| Decision | Owner | Notes |
|---|---|---|
| ☐ Keep as-is · ☐ Keep with toned rhetoric · ☐ Remove paragraph · ☐ Other | | |

---

## 2. Broader Claims Inventory

All strong claims in the paper, regardless of `\reviewedit{}` markup. The team should decide
the posture on each claim before the H100 runs begin, so that the experiments are designed to
confirm the decided claims.

| # | Claim (verbatim or close paraphrase) | Section | Type | Current posture | Recommended action | Team decision |
|---|---|---|---|---|---|---|
| 1 | "the first MCTS algorithm that operates directly in the continuous latent space" | 1.1 / 2.4 | Priority claim | Hedged in 1.1; unhedged in 2.4 ("Our work is the first to...") | Decide Path 1 or 2 from Edit B above; apply consistently to both locations | |
| 2 | "all existing MCTS-based red-teaming operates exclusively in text space" [wu2025mcts, damon2025, li2025ace, agenticred2025] | 2.4 | Comparative | Unhedged; four citations given | Verify each cited paper is accurately described as text-space only; if correct, keep | |
| 3 | "No prior work has used structured search algorithms to explore the intervention space adversarially" | 2.2 | Priority claim | Unhedged | Verify or qualify to "in the KV-cache latent space" to narrow scope | |
| 4 | "the first formal definition and experimental protocol for Orthogonal Escape" | 1.1 | Priority claim | Unhedged | Team: is this defensible? Has anyone else formalized OE-like phenomena? | |
| 5 | O(b^d · M) → O(M + d · K) memory reduction | 1.1 | Mathematical claim | Unhedged | Verify the algebraic derivation in the paper is complete and correct (Theorem 1 section) | |
| 6 | "enables deep MCTS exploration on a single 80GB GPU for models up to 20B parameters" | 1.1 | Empirical capability claim | Unhedged, stated as present fact | Move to experiments section; reframe as projected capability until H100 run confirms | |
| 7 | "score divergences of 0.10–0.20 on normalized scales across independent runs" (LLM-as-Judge) | 2.5 | Quantitative cited claim | Citation [llmasjudge2025limitations] given | Verify the cited paper actually states this figure; if yes, keep with citation | |
| 8 | "FP32 accumulator achieves zero semantic degradation" | 4.2 | Empirical | Unhedged | KEEP — confirmed by gate run (Lipschitz drift = 0.00e+00 over 200 cycles, seed 20260416) ✅ | |
| 9 | "Orthogonal Escape hypothesis has not yet been empirically validated in the specific setting of KV-cache interventions" | Limitations | Honest limitation | Present in draft | KEEP — this is required for reviewer credibility ✅ | |
| 10 | "20B model KV cache ≈ 40GB in bf16" | 4.2 | Empirical specification | Unhedged | KEEP — calculable from model architecture; well-established figure ✅ | |
| 11 | "accumulators add ~50MB overhead for 20B model with S'=10 positions" | 4.2 | Empirical memory claim | Unhedged | Verify calculation; if correct, keep ✅ | |

---

## 3. Claims That Depend on Unrun Experiments

The following sections describe planned experiments (Sections 5.1–5.4 in the paper).
None of these experiments have been run at the scale required for the paper's claims.

**Editorial posture until experiments complete:** Use future-tense framing ("our framework
is designed to X", "we expect to show Y") rather than present-tense assertions. After H100
runs, these become past-tense empirical results. This is the correct and honest approach for
a research proposal paper — reviewers distinguish proposals from results papers, but they
cannot forgive results that don't materialize.

| Experiment | Relevant paper claims | Current status |
|---|---|---|
| Experiment 1: Alignment detection under MCTS | Detection performance numbers; OEI/TDS thresholds that distinguish faking from genuine | Not run at scale; gate run (n=11) only |
| Experiment 2: Orthogonal Escape characterization | OEI/TDS curves; claim that orthogonal redistribution is measurable and systematic | Not run; hypothesis only |
| Experiment 3: Component ablation | Contribution of each telemetry channel (H-Neuron vs. RepE vs. both) | Not run |
| Experiment 4: Calibration overhead | Timing claims at scale; practical feasibility on H100 | Gate run timing (1.8s for n=15) is prototype-only |
| Experiment 5: Cross-model transfer (Procrustes) | Transfer success rates across model families | Not run; 7B+ target model not yet selected |

---

## 4. Decision Record (Fill at Meeting)

Use this table as the authoritative output of the Saturday April 18 review session.

| Item | Decision | Owner | Target revision by |
|---|---|---|---|
| Edit A (LAT acronym) | | | |
| Edit B (MCTS novelty hedge) — path chosen | | | |
| Edit C (GCG paragraph rhetoric) | | | |
| Claim 1: "first MCTS in latent space" framing | | | |
| Claim 2: all prior MCTS in text space — verify citations | | | |
| Claim 3: "no prior work structured search adversarially" | | | |
| Claim 4: "first formal definition of Orthogonal Escape" | | | |
| Claim 5: memory reduction theorem — verify proof | | | |
| Claim 6: "single 80GB GPU for 20B" — reframe to projected? | | | |
| Claim 7: LLM-as-Judge 0.10–0.20 figure — verify citation | | | |
| Experiment framing — apply future tense for unrun experiments | | | |

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

---

## Update Log

| Date | What changed |
|---|---|
| 2026-04-16 | Initial version created from paper audit (commit 041087b) |
