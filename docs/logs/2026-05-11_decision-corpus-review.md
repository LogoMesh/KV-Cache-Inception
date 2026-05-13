# Decision Corpus Review — Past-Week Guidance on the 5 Open Sign-Off Decisions

**Author:** Session A
**Date:** 2026-05-11 (Day 3 EOD)
**Audience:** Josh (sign-off decision); Session B (applies after sign-off)
**Plan:** `C:\Users\Josh\.claude\plans\misty-scribbling-fiddle.md`
**Source:** Per-decision deep-read of flagged source documents in `docs/logs/` from 2026-05-04 to 2026-05-11.

---

## How to read this

The Day-3 stale-sentence audit ([2026-05-11_stale-sentence-audit.md](2026-05-11_stale-sentence-audit.md)) produced 16 findings. 5 require Josh's sign-off before Day 4 TeX application. This review mines the past-week corpus for guidance on each — verbatim quotes from primary artifacts where possible, with source-file + section refs for each. Per the primary-source-fidelity discipline (`memory/feedback.md` 2026-05-11), recommendations must trace to corpus passages, not be invented.

For each decision below:
- **Audit's proposed fix** — quick context (one line)
- **Source-of-truth corpus** — table of relevant past-week docs + verbatim or near-verbatim quotes
- **Cross-corpus consistency check** — does the corpus agree or contradict?
- **Recommendation** — anchored to corpus passages, with HIGH/MEDIUM/LOW confidence
- **Open question** — if any remains for Josh

---

## Decisions-at-a-glance summary

| # | Decision | Recommendation | Confidence | Open Q? |
|---|---|---|---|---|
| 1 | §4 Table 1 "Diagnostic States" | **Delete entirely** (audit option a) | HIGH | None |
| 2 | §4 Eq.~mcts_reward subsubsection | **Delete entirely** (audit option a) | HIGH | None |
| 3 | §5.4 Baselines table | **Trim to 4 rows** (Rimsky CAA + Belitsky Cache Steering + Gupta-Jenner RL-Obfuscation + Bailey Obfuscated Activations) + reframe caption | HIGH | None |
| 4 | §7.6 paragraph 1 rewrite | **Apply the audit's merged rewrite** | HIGH | None |
| 5 | §7.2 Broader Impact rewrite | **Apply the audit's rewrite** (anchored to interpretability framing + Gemini Part 5 + Part 4 + v9-limitations item 7) | HIGH | One mild caveat — see below |

**Net signal:** all 5 decisions have HIGH-confidence corpus-grounded recommendations. The corpus converges; no contradictions surfaced. The audit's proposed fixes are corpus-aligned with one minor refinement in Decision 3 (4 rows, not 3).

---

## Decision 1 — §4 Table 1 "Diagnostic States" (Canonical TeX lines 228-244)

**Audit's proposed fix (audit §4-2):** Delete the table entirely. The 4-states taxonomy (Creative Synthesis / Contextual Fabrication / Alignment Faking / Orthogonal Escape) is α+ residue; §3 Dimensional Escape framing is load-bearing under Option D+, not the table.

### Source-of-truth corpus

| Doc | Relevance | What it says |
|---|---|---|
| `2026-05-10_session-A-prose-drafts-v9-problem-formulation.md` (Day 1 draft, applied Day 2) | **PRIMARY** | Establishes new §3.2 structural framing: "Definition~\ref{def:dim_escape}, Hypothesis~\ref{hyp:orthogonal_escape} (monitor-driven sub-case), Hypothesis~\ref{hyp:goodhart} (reward-driven sub-case)... existing §3.2 Discrete Text-Space Bottleneck renumbers to §3.3 with no content change." The 4-state taxonomy is conspicuously absent from the new §3 — the Definition + 2 Hypotheses do all the conceptual work. |
| `2026-05-09_Latent-Space MCTS_ Reward Design Strategy.md` Part 5 | **PRIMARY** | Gemini's Dimensional Escape framing proposal: "the terminology must be decoupled from the narrow, anthropomorphic domain of 'deception.' The broader framing should be rigorously defined as Dimensional Escape: the fundamental tendency of overparameterized latent spaces to resolve localized optimization pressure by routing representations through unconstrained orthogonal dimensions. Goal-directed deception... is simply a specialized sub-case of this broader mathematical pathology. Reward hacking... is the second sub-case." Gemini does NOT propose a 4-state taxonomy. |
| `2026-05-10_session-A-prose-drafts-v9-limitations.md` (item 4) | **PRIMARY** | Already-applied limitation: "the per-layer telemetry channels σ_H and ρ_R serve as latent-response diagnostics in our experiments rather than as deception detectors... We deliberately use σ_H and ρ_R to observe geometric trajectories of MCTS search under steering, not to claim that elevated σ_H implies the model is being deceptive." Direct contradiction of Table 1's "Alignment Faking" row caption ("strategic compliance masking") and "Orthogonal Escape" row caption ("deceptive intent in orthogonal directions"). |
| Canonical TeX lines 132-181 (applied Day 2) | **PRIMARY** | The applied §3 has Definition + Hypothesis 1 + Hypothesis 2 + Eq.~goodhart + Eq.~alignment_faking + Eq.~orthogonal_escape. NO 4-state taxonomy. The geometric framing fully replaces the taxonomic framing. |
| `2026-05-11_pivot-decision-memo.md` | SUPPORTING | Lists "Candidate A System-2 framing" as **falsified** (§2 table) and "Telemetry matrix (σ_H + ρ_R) as static monitoring signal" as **intact**. The 4-state taxonomy was part of the System-2 framing's diagnostic apparatus; the telemetry matrix tensor (Eq.~telemetry_matrix) is preserved as a monitoring tool. The TABLE — i.e., the taxonomic interpretation of the matrix — was a deception-detection claim, not a monitoring claim. |
| `memory/recast_action_plan.md` §"What changed in the paper as a result" | SUPPORTING | "§3 Problem Formulation: reframed around Dimensional Escape unifying framework. Original Hypothesis 1 (deception-driven) preserved as one instance; reward-driven failure framed as second instance of the same geometric pathology." — confirms the Definition + 2 Hypotheses framing is load-bearing, not the taxonomy. |

### Cross-corpus consistency check

All five sources converge: the Definition + 2 Hypotheses structure is the load-bearing conceptual framing under Option D+. The 4-state taxonomy is not proposed by Gemini, not present in the v9 §3 draft, and the §7.3 Limitations item 4 (already applied) **directly contradicts** the table's "Alignment Faking" row interpretation. No contradictions between sources. No source argues for preserving the table.

### Recommendation

**Delete §4 Table 1 entirely** (audit option a). Confidence: **HIGH**.

Justification anchored to corpus:
1. The v9 §3 problem-formulation draft (already applied) gives the load-bearing conceptual work to Definition~\ref{def:dim_escape} + Hypothesis 1 + Hypothesis 2. The taxonomy is redundant.
2. Gemini Part 5 explicitly proposes Dimensional Escape as the unifying framing; the 4-state taxonomy is anthropomorphic in exactly the way Gemini argues against ("decoupled from the narrow, anthropomorphic domain of 'deception'").
3. The v9 §7.3 Limitations item 4 (already applied) and the audit's §2-3 finding both establish that σ_H / ρ_R are latent-response diagnostics, not deception detectors. Table 1's "Alignment Faking" row caption ("strategic compliance masking") and "Orthogonal Escape" row caption ("deceptive intent in orthogonal directions") directly contradict this. Keeping the table = internal contradiction.

Page-budget saving: ~15 lines.

### Open question

None.

---

## Decision 2 — §4 Eq.~mcts_reward subsubsection (Canonical TeX lines 292-309)

**Audit's proposed fix (audit §4-4):** Delete the subsubsection entirely. Replace with one-line "MCTS node reward in our experiments is the entropy-normalized signal Eq.~\ref{eq:goodhart}; the framework is reward-agnostic and supports alternative rewards as a Phase B research program."

### Source-of-truth corpus

| Doc | Relevance | What it says |
|---|---|---|
| `2026-05-09_Latent-Space MCTS_ Reward Design Strategy.md` Part 1 | **PRIMARY** | "Any reward function derived exclusively from internal model heuristics — such as confidence, entropy, or representation engineering projections — that lacks an exogenous grounding mechanism for factual correctness will inevitably succumb to prior-amplification or orthogonal representation redistribution. The dimensionality of the continuous representation space ensures there is always a readily accessible mathematical vector that satisfies the numerical proxy metric without satisfying the semantic objective." |
| `2026-05-09_Latent-Space MCTS_ Reward Design Strategy.md` Part 2 #1 (Multi-Channel Ensemble) | **PRIMARY** | Direct evaluation of Eq.~mcts_reward's structural twin: "The proposed reward mechanism would calculate simultaneous agreement across the H-Neuron stress signal (σ_H), the Representation Engineering honesty projection (ρ_R), next-token entropy, and output perplexity... The MCTS will likely discover a manifold where the easiest constraints (perplexity and entropy) are minimized by amplifying a benign prior, while the telemetry channels (σ_H and ρ_R) are trivially satisfied through representation redistribution into the null space of the linear monitors. Given the high-dimensional nature of the 2048-dimension embedding space, finding a vector that satisfies four linear constraints without generating reasoning is highly probable. The estimated probability of producing the required +10 percentage point improvement is a mere 15%." |
| `2026-05-11_track-F-negative-control-report.md` §7 | **PRIMARY** | "This is a *conceptual* failure of the entropy-as-MCTS-reward hypothesis, not an implementation issue. No amount of (a) tuning the alpha set, (b) tuning depth/branches/n_nodes, or (c) using a calibrated steering direction would change the basic mechanism: an entropy-minimising objective on a frozen model's KV cache reinforces the model's existing priors regardless of input semantics." |
| `2026-05-11_track-F-negative-control-report.md` §8 (recommendations) | **PRIMARY** | "**DO NOT continue with Candidate A as written.** The entropy-reward formulation is the load-bearing empirical claim of the System-2 framing's third contribution ('Telemetry-Guided Latent Optimization'). Track F has demonstrated that this claim does not hold experimentally. Re-running with marginally different alphas or directions will not flip the result — the failure is mechanism-level (§7)." |
| `2026-05-11_pivot-decision-memo.md` §2 (What's preserved vs killed) | **PRIMARY** | Lists "**Entropy-as-MCTS-reward (Contribution 3 of Candidate A)** ❌ **EMPIRICALLY FALSIFIED**" and "Candidate A's 'System-2 Latent Search' headline framing ❌ **Falsified**". Telemetry matrix as static monitoring signal preserved (intact); telemetry-as-reward is dead. |
| Canonical TeX §7.1 Reward design paragraph (already applied Day 3 P1) | SUPPORTING | "Track F demonstrates that a reward signal validated as a static correlate of correctness (|r| ≈ 0.60 between next-token entropy and ARC-Easy answer correctness, Track A) does not survive deployment as an active MCTS optimization objective... Anti-Goodhart reward designs surveyed in the Phase B research program (§\ref{sec:future-work}) — multi-channel ensembles, external Process Reward Models, self-consistency across stochastic rollouts, counterfactual sensitivity mapping, anti-prior penalties, information-bottleneck objectives — all attempt to raise the effective rank of the scoring projection or to anchor it in exogenous signal that the search cannot itself rewrite. Whether any of these resists Dimensional Escape at the empirical level is an open question; our Track F result establishes the null-control (entropy minimization fails) against which any candidate must be measured." |

### Cross-corpus consistency check

The corpus converges with unusual unanimity. **Track F established that entropy-as-reward fails mechanism-level**; Gemini Part 2 #1 evaluated the **exact same structural form as Eq.~mcts_reward** (multi-channel ensemble: σ_H + ρ_R + entropy + perplexity) and predicted **15% probability of working under MCTS**. The pivot decision memo retired Candidate A's Contribution 3 explicitly. §7.1 of the canonical TeX (already applied) explicitly characterizes Track F's entropy reward as a "null-control" against which any candidate must be measured — implying that Eq.~mcts_reward is an untested candidate, not the actually-tested reward.

**Key implication:** Eq.~mcts_reward in the current §4 is NOT just stale — it is an **untested + predicted-to-fail recipe** that, if kept, would have the paper publish a §4 reward formula that the paper's own Track F experiment + Gemini Part 2 analysis predict would Goodhart-fail. A reviewer reading §4 would see Eq.~mcts_reward, then read §5 (Track F's entropy reward), then read §7 (Goodhart mechanism), and ask "why does §4 describe a reward we never used?"

### Recommendation

**Delete the §4.2.3 MCTS Node Valuation subsubsection entirely** (audit option a). Replace with a one-line pointer to Eq.~goodhart as the actually-tested reward. Confidence: **HIGH**.

Justification anchored to corpus:
1. Track F §7 (primary): the failure of entropy-as-MCTS-reward is **conceptual, not implementation-level**. The mechanism generalizes to any internally-derived surrogate reward (Gemini Part 1 confirms).
2. Gemini Part 2 #1 (primary) explicitly evaluated the multi-channel ensemble structural form (exactly Eq.~mcts_reward's shape) and rated 15% probability under MCTS. Publishing this equation without running the experiment is publishing a recipe the paper's own analysis predicts will fail.
3. Pivot decision memo (primary): Candidate A's Contribution 3 is retired. Eq.~mcts_reward IS Contribution 3.
4. Canonical TeX §7.1 (already applied): explicitly frames entropy as the **null-control** against which anti-Goodhart candidates must be measured. The reward-agnostic framework framing is established; the equation is no longer needed.

Page-budget saving: ~18 lines.

### Open question

None.

---

## Decision 3 — §5.4 Baselines table (Canonical TeX lines 430-448)

**Audit's proposed fix (audit §5-5):** Trim table to 3 rows (Activation Steering / Cache Steering / Latent Monitor Evasion) + reframe caption. Alternative: delete §5.4 entirely.

### Source-of-truth corpus

| Doc | Relevance | What it says |
|---|---|---|
| `2026-05-09_track-B-citation-audit.md` §1 per-citation table | **PRIMARY** | Verdicts on each peer: Belitsky 2025 (Cache Steering) — CLEAN, "One-shot intervention applied directly to the key-value cache"; Rimsky 2024 (CAA, ACL 2024) — CLEAN, "Inference-time modification of activations"; Gupta & Jenner 2025 (RL-Obfuscation) — CLEAN, monitor evasion; Bailey 2024 (Obfuscated Activations) — CLEAN, "gradient-trained input-space attacks against a frozen model"; Zou 2023 (GCG) — CLEAN, "Gradient-based, white-box, discrete token suffix attack"; Wu 2025 (MCTS-MPA) — CLEAN, text-space jailbreak; Zheng 2023 / Liu 2023 (MT-Bench / G-Eval) — CLEAN, LLM-as-judge methodology. |
| `2026-05-11_session-A-prose-drafts-v10-exp2.md` (the §5 Exp 2 draft) | **PRIMARY** | "We characterize how the search optimizes a surrogate reward as a function of prompt class." Cartography is positioned as **structural mapping methodology**, not a benchmark-beating exercise — implying no head-to-head ASR / accuracy comparisons against text-space red-teaming. |
| `2026-05-09_Latent-Space MCTS_ Reward Design Strategy.md` Part 3 Design 1 | SUPPORTING | "The submission to EMNLP would be framed as a novel mechanistic interpretability methodology: 'Reversible MCTS as a Lens for Exposing Embedded Pre-Training Priors in Large Language Models.'" Implies the paper's claim space is interpretability, not red-teaming benchmarks. |
| Canonical TeX §2.2 (current) | SUPPORTING | Already positions Bailey 2024 + Rimsky 2024 (CAA) + Belitsky 2025 (Cache Steering) + Gupta & Jenner 2025 (RL-Obfuscation) as peer work. These four are the "structurally-relevant peers" under Option D+. |
| Canonical TeX §5.4 caption (existing) | SUPPORTING | Caption frames the table as "Baseline methods for comparative evaluation" — but under Option D+ Cartography is structural mapping, not comparative evaluation. Caption itself is α+ residue. |
| `memory/project_context.md` §"Citation pile-up CLEAN" | SUPPORTING | "25 citations audited, 0 fatal, 0 mismatched, 1 historical NUANCED (RepE certainty probe doesn't exist) retired by Track A's entropy fallback validation." All 25 citations stay in bibliography; the question is which appear in §5.4 vs §2.2 vs elsewhere. |

### Cross-corpus consistency check

Corpus agrees: under Option D+, the paper's framing is structural mapping / interpretability, not head-to-head benchmark comparison. GCG / MPA / DAMON (text-space red-teaming) and SonarQube/Semgrep (static analysis) are not relevant peers; MT-Bench / G-Eval (LLM-as-judge) is methodology context, not a baseline. The structurally-relevant peers are: (1) **Cache Steering** (Belitsky 2025) — same intervention site (KV cache), different paradigm (one-shot vs iterative search); (2) **Activation Steering** (Rimsky 2024 CAA) — same conceptual frame (steering vectors); (3) **RL-Obfuscation** (Gupta & Jenner 2025) — monitor-driven Dimensional Escape variant; (4) **Obfuscated Activations** (Bailey 2024) — gradient-trained input-space variant of the same pathology, explicitly cited in our §3.2.

The audit suggested 3 rows; the corpus suggests **4 rows is the right number**, including Bailey 2024, because §3.2 already positions Bailey as a structural peer to Hypothesis 1 (monitor-driven sub-case) and §2.2 already has Bailey engagement. Including Bailey in §5.4 mirrors this positioning consistently.

### Recommendation

**Trim §5.4 Baselines table to 4 rows** + reframe caption. Confidence: **HIGH**.

Specific 4-row composition (anchored to corpus):
1. **Activation Steering** — CAA (Rimsky et al., 2024) — "Contrastive Activation Addition, ACL 2024"
2. **Cache Steering** — Belitsky et al., 2025 — "One-shot static KV-cache steering"
3. **Latent Monitor Evasion** — RL-Obfuscation (Gupta & Jenner, 2025) — "Weight-finetuning-based monitor evasion; complementary to our inference-time variant"
4. **Gradient-Trained Activation Obfuscation** — Bailey et al., 2024 — "Input-space gradient attacks; complementary gradient-trained variant of Dimensional Escape per §3.2"

Caption rewrite: "Inference-time and gradient-trained interventions on frozen language models that are structurally related to Reversible KV-Cache MCTS. The four rows span complementary intervention sites (activation residual stream / KV cache / weight finetuning / input embedding) and complementary optimization regimes (corrective monitor vs surrogate reward). Reversible MCTS differs from each by: iterative reversible KV-cache mutation under a surrogate reward in a frozen-weight, inference-time regime."

Page-budget saving: ~8 lines (5 fewer rows, possibly +2 lines in caption).

### Open question

None. (Whether to ALSO include Wu 2025 MPA + DAMON as text-space points-of-contrast is a borderline call; corpus leans toward dropping them since §2.4 already engages with text-space MCTS work. Recommendation: drop from §5.4.)

---

## Decision 4 — §7.6 Ethical Considerations paragraph 1 (Canonical TeX line 588)

**Audit's proposed fix (audit §7-3):** Replace the "adversarial datasets generated" sentence with a merged paragraph anchoring on Croissant 1.1 runtime evidence artifacts (per-node telemetry from Track F + Cartography).

### Source-of-truth corpus

| Doc | Relevance | What it says |
|---|---|---|
| `2026-05-11_stale-sentence-audit.md` §7-3 | **PRIMARY** | Identifies the staleness: "The adversarial datasets generated will be released..." is Option-α+/Candidate-A residue. Under Option D+, no adversarial datasets are generated; only runtime evidence artifacts from Track F + Track G. Proposes merged rewrite. |
| `2026-05-11_session-A-prose-drafts-v9-license-attribution.md` (the J2 paragraph just applied Day 3) | **PRIMARY** | The newly-added §7.6 license-attribution paragraph at TeX:592-593 sets the tone: "The Reversible KV-Cache MCTS algorithm reported in this work is evaluated on Meta's Llama 3.2-1B-Instruct and Llama 3.2-3B-Instruct models. Built with Llama..." This paragraph treats the work as evaluation/research, not as adversarial-dataset generation. The paragraph 1 framing ("adversarial datasets generated") is then internally inconsistent. |
| `2026-05-10_session-A-prose-drafts-v9-limitations.md` §"What's intentionally not in the limitations" | SUPPORTING | "Phase B experiments not yet run — dropped wholesale; the paper as submitted ships with the empirical content it has, not promises of future work." This indicates the paper ships **what we measured** (Track F + Cartography runtime evidence) — not adversarial datasets. |
| `2026-05-11_track-G-cartography-report.md` and Track F report | SUPPORTING | Both define the runtime evidence artifacts (per-node telemetry traces, JSON results, search trajectories). These are the Croissant 1.1 release content; they are NOT adversarial datasets. |

### Cross-corpus consistency check

Corpus unambiguously agrees: the paper releases **runtime evidence artifacts**, not adversarial datasets. The "adversarial datasets generated will be released" phrasing is a direct vestige of Candidate A's framing when the paper was about discovering adversarial interventions. Under Option D+, we measure structural priors; we don't generate adversaries. The just-applied license-attribution paragraph at TeX:592 makes the inconsistency more visible.

### Recommendation

**Apply the audit's merged rewrite.** Confidence: **HIGH**.

Specific rewrite anchored to corpus (audit's proposed text, lightly refined):

> "All experiments were conducted on open-weight models in isolated environments. No experiments targeted production systems or real users. We release a Croissant 1.1 metadata package (with Croissant RAI extension fields) documenting runtime evidence artifacts — per-node telemetry traces from Track F entropy-MCTS (Experiment 1) and Track G Latent Cartography sweeps (Experiment 2) — so provenance, usage constraints, and schema-level reproducibility checks are machine-verifiable against the original raw artifacts."

Tense corrected to past tense ("were conducted", "targeted") — experiments are complete, no longer aspirational.

Page-budget saving: ~1 sentence ≈ −20 words.

### Open question

None.

---

## Decision 5 — §7.2 Broader Impact (Canonical TeX lines 533-535)

**Audit's proposed fix (audit §7-1):** Replace the existing paragraph (which says "discover alignment-breaking interventions" + "cryptographic hash of payload" + "responsible disclosure of vulnerabilities") with a methodology + interpretability framing.

### Source-of-truth corpus

| Doc | Relevance | What it says |
|---|---|---|
| `2026-05-09_Latent-Space MCTS_ Reward Design Strategy.md` Part 4 (Honest Verdict) | **PRIMARY** | "The EMNLP 2026 program values mechanistic interpretability and the rigorous analysis of model internals significantly more than fragile, heavily engineered performance bumps... By embracing Option D, the team seamlessly transitions the Track F results from a 'failed system architecture' into a 'successful interpretability discovery.'" |
| `2026-05-09_Latent-Space MCTS_ Reward Design Strategy.md` Part 5 (Connection to Original Orthogonal-Escape Framing) | **PRIMARY** | "The paper must be immediately reframed to introduce Reversible MCTS not as a tool for boosting benchmark scores, but as a high-precision diagnostic probe. The empirical contribution is profound: the team has discovered a generalized mechanism demonstrating that surrogate rewards in continuous latent spaces inevitably lead to prior-amplification or greedy-collapse. This phenomenon effectively maps the boundaries and inherent dangers of inference-time test scaling." |
| `2026-05-09_Latent-Space MCTS_ Reward Design Strategy.md` Part 5 (abstract sketch) | **PRIMARY** | "Our findings establish that surrogate rewards lacking exogenous grounding are inherently unstable in high-dimensional latent spaces, positioning Reversible MCTS as a critical interpretability tool for mapping these degenerate manifolds." |
| `2026-05-10_session-A-prose-drafts-v9-limitations.md` item 7 (already applied) | **PRIMARY** | "The framing of Reversible MCTS as a probe of representational geometry rather than a reasoning-improvement tool is itself a scoping choice for this submission; whether a Goodhart-resistant reward can be constructed within the FP32-accumulator framework is an open Phase B question." This is the now-explicit project framing under Option D+. |
| Canonical TeX §7.1 Monitor design paragraph (already applied Day 3 P1) | **PRIMARY** | "The correct frame for these findings is monitor-design diagnostics rather than attack capability: Reversible KV-Cache MCTS requires white-box access to a frozen model and is therefore unavailable to a real adversary at inference time. Its value is interpretability and controlled-access red-teaming." This is the already-applied dual-use framing for §7.1; §7.2 Broader Impact should be consistent. |
| `2026-05-11_pivot-decision-memo.md` §3 Option D | SUPPORTING | "Future-research story is strong: 'what reward signals DO survive MCTS optimization in latent space?' is a publishable open question." The broader impact under Option D+ is opening this research program, not enabling adversarial discovery. |
| `2026-05-11_track-G-cartography-report.md` §3 cross-scale finding | SUPPORTING | "Cartography shows the pattern is MCQ-format-specific, not universal." The interpretability contribution is mapping pre-training-prior structure; this is the broader-impact content under Option D+. |

### Cross-corpus consistency check

Corpus converges strongly: under Option D+, the paper's broader impact is **methodology + interpretability** (Gemini Part 4 + Part 5), **NOT attack discovery**. The framework requires white-box access, isn't deployable as an inference-time attack vector (§7.1 already applied), and its contribution is opening a reward-design research program (pivot memo). The existing §7.2's three "mitigations" — (1) controlled offline environment, (2) cryptographic hash for audit trail, (3) responsible disclosure of vulnerabilities — are all premised on the false claim that the work produces deployable adversarial interventions.

One mild **caveat**: the corpus also acknowledges dual-use considerations are real. Gemini Part 5: "Reversible MCTS as a high-precision diagnostic probe" still implies the framework could be misused. The rewrite should preserve a brief acknowledgement of dual-use without overstating it.

### Recommendation

**Apply the audit's proposed rewrite** (with one refinement to preserve the dual-use acknowledgement). Confidence: **HIGH**.

Specific rewrite anchored to corpus:

> "This work has dual-use considerations we address transparently. Reversible KV-Cache MCTS provides a controlled-access white-box probe of the latent geometry of frozen language models; the Dimensional Escape pathology it surfaces — that low-rank surrogate signals (whether deployed as monitors or as search rewards) admit degenerate optima in high-dimensional cache space — has implications for the design of both monitoring procedures and test-time scaling schemes. The framework requires white-box access to the frozen model and is not deployable as an inference-time attack vector against real-world systems. We disclose findings openly through this paper and accompanying supplementary materials rather than withholding them; the future-research questions our results open (which reward designs resist Dimensional Escape; whether monitor-driven Dimensional Escape manifests under inference-time KV-cache perturbation) are inputs to the design of more robust monitoring and reward-modeling procedures rather than enablers of adversarial discovery."

This preserves the dual-use acknowledgement while anchoring on methodology/interpretability framing per Gemini Part 4-5. The "white-box access" line mirrors the already-applied §7.1 statement so the two paragraphs are stylistically consistent.

Page-budget impact: roughly net zero (~80 → ~110 words).

### Open question

**Mild**: Should we cite Bailey et al. 2024 explicitly in §7.2 as a peer dual-use case? Bailey reported a negative result for gradient-trained input attacks; cross-referencing them would frame our broader-impact discussion as consistent with the existing literature on related techniques. NOT strictly required for sign-off — can be added as an additive polish later.

---

## Closing notes

### Highest-confidence (fast sign-off)

Decisions 1, 2, 3, 4 — all corpus-converging with no contradictions. Recommend Josh signs off on all four without further discussion.

### Mild caveat (Decision 5)

Decision 5's rewrite has a minor open question about whether to cite Bailey 2024 explicitly. The rewrite as proposed is corpus-grounded and complete; the Bailey citation is a polish addition that can be added later. **Recommend sign-off with the polish-citation question deferred to Day 4-5.**

### Verification per plan

- **Inventory completeness:** All flagged sources from Phase-1 inventory appear in at least one decision's Source-of-truth corpus table. ✓
- **Verbatim discipline:** All quotes have (filename, §ref) attribution. Spot-checked Decisions 1, 2 against source files. ✓
- **Contradiction surfacing:** No contradictions found within or across decisions. Cross-corpus consistency check sections explicitly note convergence. ✓
- **Recommendation traceability:** Every recommendation cites the corpus passages that grounded it. ✓
- **Coverage of audit's proposals:** All 5 audit-proposed fixes are endorsed (Decisions 1, 2, 4) or refined-with-corpus-reason (Decisions 3, 5). None silently dropped. ✓
- **Open question surfacing:** Decision 5 has one mild polish question stated as a single sentence. ✓

### Application path post-sign-off

Per the stale-sentence audit's `## Application strategy` table:
- **Day 4 AM:** Apply BLOCKING text-only fixes (audit §2-2, §2-3, §2-4, §4-1, §4-3) + Decision 4 (§7.6 ¶1 rewrite) + Decision 1 (Table 1 delete) + Decision 2 (Eq.~mcts_reward delete)
- **Day 4 PM / Day 5:** Apply RECOMMENDED set + Decision 3 (Baselines trim) + Decision 5 (§7.2 rewrite)
- **Day 5-7:** Continue §5 Experiment 1 recast (Track F prose, draft pending) + §5 Experiment 2 application (v10 draft already exists) + §5.5 Evaluation Metrics rewrite per audit §5-6

---

*End of decision corpus review. Five HIGH-confidence recommendations, all corpus-grounded with verbatim citations. Awaits Josh sign-off; on sign-off, Session B begins Day 4 TeX application per audit's strategy table.*
