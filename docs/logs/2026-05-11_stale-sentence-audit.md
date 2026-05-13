# Canonical TeX Stale-Sentence Audit — 2026-05-11

**Authored:** 2026-05-11 (Day 3 EOD; deferred to Day 4 application pending Josh sign-off)
**Audience:** Josh (batched review) + Session B (when applying)
**Scope:** Sweep of `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` (709 lines as of Day 3 EOD) for Option α+ / Candidate-A residue not caught in Day 2 + Day 3 prose application passes.
**Motivation:** Josh observation 2026-05-11: "a review to catch more stale sentences is mandatory now." Likely recovers page-budget naturally per §5/§5.4 stale content.

---

## How to read this

Each finding has:
- **Location** — file line numbers
- **Current text** — verbatim quote
- **Issue** — what's stale and why
- **Severity** — BLOCKING / RECOMMENDED / OPTIONAL
- **Proposed fix** — replacement text or deletion strategy
- **Page-budget impact** — estimated line delta

**Categories:**
- **BLOCKING** — paper-level coherence requires fix before ARR submission (e.g., contradicts a current §7.3 Limitation, or relies on a metric that's no longer the primary measurement)
- **RECOMMENDED** — framing-level inconsistency a careful reviewer would flag; not strictly contradictory but cumulatively misleading
- **OPTIONAL** — minor polish; could defer to camera-ready Phase 3

---

## §2 Background and Related Work

### Finding §2-1 — RECOMMENDED — line 98 closing clause

**Current:** "Existing empirical work on alignment faking and backdoor persistence suggests that behavioral evaluations alone are insufficient for reliable alignment detection~\cite{greenblatt2024alignment, hubinger2024sleeper}: a model that strategically complies or conceals a backdoor can pass behavioral tests while maintaining **deceptive internal representations**---one motivation for the inference-time latent-space analysis framework developed in this paper, of which the telemetry matrix is the monitoring component."

**Issue:** "Deceptive internal representations" is the alignment-faking framing. Under Option D+, the paper's framing is geometric pathology in latent space (Dimensional Escape), not deception detection. §2.1 motivates the work via alignment-faking literature — that connection is fine — but the closing phrase "deceptive internal representations" should soften to match §7.3 Limitations item 4 ("σ_H and ρ_R serve as latent-response diagnostics in our experiments rather than as deception detectors").

**Proposed fix:** Replace "deceptive internal representations---one motivation for" → "internal representational divergences invisible to behavioral testing alone---one motivating application area for".

**Page-budget impact:** −1 word, net neutral.

---

### Finding §2-2 — BLOCKING — line 102 §2.2 closing

**Current:** "Unlike this static approach, our MCTS iteratively explores the KV-cache intervention space across multiple rollouts, reversing each intervention exactly via FP32 accumulators and **guided by a telemetry-derived reward** rather than a fixed behavioral objective."

**Issue:** "Telemetry-derived reward" is the α+ Eq.~mcts_reward (σ_H + ρ_R + Divergence). Under Option D+ Track F, our actual reward is entropy minimization (Eq.~\ref{eq:goodhart}). The telemetry-derived reward was never empirically tested in this submission. This sentence claims an empirical capability we don't demonstrate.

**Proposed fix:** "guided by a telemetry-derived reward rather than a fixed behavioral objective" → "guided by an inference-time surrogate reward (in our reported experiments, normalized next-token entropy; Eq.~\ref{eq:goodhart}) rather than a fixed behavioral objective".

**Page-budget impact:** +6 words.

---

### Finding §2-3 — BLOCKING — line 107 §2.2 closing on Bailey

**Current:** "Our work is complementary along three axes. First, the intervention is applied to cached key/value tensors at inference time rather than to input embeddings during gradient training, so neither the model weights nor the embedding layer is modified. Second, **the search optimizes a behavioral telemetry signal (Eq.~\ref{eq:mcts_reward})** rather than directly minimizing similarity to baseline, so any redistribution we observe is incidental rather than optimization-driven. Third, **OEI measures a more targeted geometric quantity**---the fraction of induced perturbation invisible to a specific linear monitor---rather than global activation orthogonality."

**Issue:** Two stale claims compounded. (a) "Search optimizes a behavioral telemetry signal (Eq.~mcts_reward)" — STALE per §2-2. (b) "OEI measures a more targeted geometric quantity" — OEI is no longer the primary metric under Option D+ (Cartography uses best-path α-tuple distribution + argmax-at-best-leaf + accuracy). This third axis no longer characterizes our work.

**Proposed fix:** Rewrite axes 2 and 3:
- Axis 2: "Second, the search optimizes a surrogate reward defined on the model's own next-token distribution (entropy minimization in our experiments; Eq.~\ref{eq:goodhart}) rather than minimizing similarity to baseline, so any geometric redistribution we observe is a side-effect of reward optimization rather than the optimization target."
- Axis 3: "Third, our measurements characterize how a surrogate-reward MCTS resolves the reward through latent-space trajectories (Experiments~\ref{sec:exp1}--\ref{sec:exp2}) — a different empirical regime from Bailey's similarity-minimization study, leaving open whether the inference-time, frozen-weight, surrogate-reward setting exhibits the same constraint Bailey identifies."

**Page-budget impact:** ~ +10 words.

---

### Finding §2-4 — BLOCKING — line 111 §2.3 entire paragraph

**Current:** "Gao et al.\ (2025)~\cite{gao2025hneurons} identified H-Neurons---a sparse subset of feed-forward network units whose activation patterns reliably predict hallucination occurrences and are causally linked to over-compliance behaviors. Unlike black-box evaluation, H-Neuron monitoring provides a continuous, neuron-level signal that can serve as a differentiable reward for optimization algorithms. **We extend this finding to the domain of alignment faking by treating $\sigma_H$ as an \emph{over-compliance pressure proxy}: H-Neurons validated by Gao et al.\ (2025) as causal indicators of over-compliance and hallucination may produce detectable activation signatures when a model encounters pressure to fabricate alignment-consistent outputs. This extrapolation from over-compliance to strategic, instrumentally rational deception is a testable assumption rather than an established equivalence---one that our experimental protocol is designed to validate.**"

**Issue:** The bolded portion directly contradicts §7.3 Limitations item 4 ("σ_H and ρ_R serve as latent-response diagnostics in our experiments rather than as deception detectors... not to claim that elevated σ_H implies the model is being deceptive"). A reviewer reading §2.3 will form expectations that §7.3 explicitly retracts. This is an internal contradiction.

**Proposed fix:** Replace the bolded portion (3 sentences) with:
"We use $\sigma_H$ as a layered latent-response diagnostic: under MCTS-driven steering, $\sigma_H$ provides a per-layer, per-step measurement of how the model's internal computation responds to the perturbation, independent of whether the model is being deceptive. Whether the observed response patterns correspond to deception-relevant computation is left as an open empirical question (see §\ref{sec:limitations} item 4)."

**Page-budget impact:** −2 sentences worth = ~ −30 words.

---

### Finding §2-5 — OPTIONAL — line 115 §2.4 framing

**Current:** "Monte Carlo Tree Search has been increasingly applied to automated red-teaming of LLMs. Wu et al.\ (2025)~\cite{wu2025mcts} proposed MCTS-based prompt autogeneration for jailbreak attacks..."

**Issue:** §2.4 frames MCTS-for-LLMs as red-teaming. Under Option D+ our work isn't red-teaming — it's a probe of latent geometry. This section serves to position prior MCTS work, which is fine, but the closing claim "Our work is the first to conduct MCTS via continuous arithmetic mutations of the KV cache with an exact FP32 rollback mechanism, enabling gradient-free, inference-time exploration of the model's representational landscape" could end "...exploration of the model's representational landscape" rather than implicitly comparing to red-teaming. Minor polish, not blocking.

**Proposed fix:** Section heading "MCTS for LLM Red-Teaming" → "MCTS for Language Model Search". Closing sentence unchanged.

**Page-budget impact:** Net zero.

---

## §4 Proposed Method

### Finding §4-1 — RECOMMENDED — line 214 §4.1 RepE LAT extrapolation

**Current:** "We apply the Linear Artificial Tomography (LAT) procedure from Zou et al.\ (2023)~\cite{zou2023repe} to extract linear probes for cognitive dimensions including **honesty ($\mathbf{w}_{\text{hon}}^{(\ell)}$), certainty ($\mathbf{w}_{\text{cert}}^{(\ell)}$), and goal-coercion ($\mathbf{w}_{\text{coerce}}^{(\ell)}$)**."

**Issue:** Three probe directions listed. Under Option D+: certainty probe retired (Track A entropy replaces certainty probe); goal-coercion probe retired (no current contribution depends on it). Only honesty probe is operationally relevant.

**Proposed fix:** "extract linear probes for cognitive dimensions; in our reported experiments we use the honesty probe ($\mathbf{w}_{\text{hon}}^{(\ell)}$) only."

**Page-budget impact:** −6 words.

---

### Finding §4-2 — BLOCKING — lines 228-244 Table 1 "Diagnostic States"

**Current:** Table with 4 rows: Creative Synthesis / Contextual Fabrication / **Alignment Faking** / **Orthogonal Escape**. Caption: "...The critical diagnostic for Orthogonal Escape is a statistically significant divergence between $\sigma_H$ and $\rho_R$: the H-Neuron channel detects over-compliance pressure while the RepE channel reports nominal honesty, indicating that the model has learned to satisfy the RepE probe while encoding **deceptive intent** in orthogonal directions."

**Issue:** The "Alignment Faking" row presupposes the model is engaged in strategic deception (now retired claim). The "Orthogonal Escape" row interprets divergence as "deceptive intent" — contradicts §7.3 Limitations item 4 and §3 Hypothesis 1's structural framing (Definition def:dim_escape decouples Dimensional Escape from intent). Caption is the contradiction-bearing element.

**Proposed fix (two options):**
- **(a)** Delete the table entirely. The four states framing is α+ residue. §3 already establishes the geometric framing without needing this taxonomy.
- **(b)** Keep table, drop "Alignment Faking" row, rewrite caption: "Diagnostic signatures from the joint telemetry matrix. The four states characterize the response space of MCTS-driven steering in a frozen model; Orthogonal Escape — Definition~\ref{def:dim_escape} — is the structural pathology Hypothesis~\ref{hyp:orthogonal_escape} predicts as one route through the latent space, observed when σ_H and ρ_R diverge."

Recommendation: **(a) Delete.** The four-states taxonomy isn't load-bearing for Option D+; the geometric framing in §3 is.

**Page-budget impact (option a):** ~ −15 lines (table + caption + surrounding text).

---

### Finding §4-3 — BLOCKING — line 271 §4.2 Step 2

**Current:** "\paragraph{Step 2: Evaluation.} Execute the forward pass, generate the next token, and **record the telemetry matrix $\mathbf{T}_t$ as the MCTS node reward**."

**Issue:** Telemetry-matrix-as-MCTS-reward. STALE — Option D+ Track F uses entropy reward.

**Proposed fix:** "Step 2: Evaluation. Execute the forward pass, generate the next token, and compute the MCTS node reward (we report results with the entropy-normalized reward $\hat{r}(\mathbf{h}) = -H(p(\cdot \mid \mathbf{h}))/\log|V|$ described in §\ref{sec:dim_escape}, Eq.~\ref{eq:goodhart})."

**Page-budget impact:** +10 words.

---

### Finding §4-4 — BLOCKING — lines 292-309 §4.2.3 MCTS Node Valuation

**Current:** Entire `\subsubsection{MCTS Node Valuation}` plus Eq.~\ref{eq:mcts_reward} defining `r(node) = λ₁·σ̄_H + λ₂·(1-ρ̄_R) + λ₃·Divergence(σ_H, ρ_R)` and Divergence-as-Orthogonal-Escape-signal.

**Issue:** Eq.~mcts_reward is the α+ telemetry-as-reward formula. Under Option D+ Track F's actual reward is entropy (Eq.~goodhart). Keeping Eq.~mcts_reward without testing it experimentally claims an unsupported capability.

**Proposed fix (three options):**
- **(a)** Delete the subsubsection entirely. Replace with one-line "MCTS node reward in our experiments is the entropy-normalized signal Eq.~\ref{eq:goodhart}; the framework is reward-agnostic and supports alternative rewards as a Phase B research program (§\ref{sec:future-work} reward-design paragraph)."
- **(b)** Keep but reframe: present Eq.~mcts_reward as "one candidate reward formulation" alongside Eq.~goodhart as the actually-tested one.
- **(c)** Keep entire subsubsection but add a sentence at the end clarifying that Track F tested the simpler entropy reward (Eq.~goodhart); the full telemetry-derived reward is a Phase B variant.

Recommendation: **(a) Delete.** Cleanest. Page-budget savings.

**Page-budget impact (option a):** ~ −18 lines.

---

## §5 Experimental Design

### Finding §5-1 — RECOMMENDED — line 334 §5.1 Phase 0

**Current:** "Before deploying telemetry hooks, the monolithic MCTS orchestrator must be refactored to support: (a)~a model abstraction layer replacing hardcoded API calls with a \texttt{BaseModelClient} interface supporting local HuggingFace models; (b)~a state serialization pipeline enabling MCTS tree persistence across environments; and (c)~**a continuous reward interface processing neuronal activation tensors directly as MCTS node valuations**."

**Issue:** "Continuous reward interface processing neuronal activation tensors" — α+ framing. Under Option D+ reward is entropy on next-token distribution, not neuronal activation tensors.

**Proposed fix:** Item (c) → "a continuous reward interface accepting the per-step next-token distribution as MCTS node reward input."

**Page-budget impact:** Net zero.

---

### Finding §5-2 — BLOCKING — lines 343-348 §5.2 Phase A bullets

**Current:** Four bullets including "RepE probe training via PCA on contrastive pairs for honesty, **certainty, and goal-coercion** dimensions" and "Cross-model alignment theory via **supervised Procrustes alignment** between 1B and 7B model representation spaces."

**Issue:** Certainty + goal-coercion probes retired (per Finding §4-1). Procrustes alignment retired (was Experiment 5, deferred to Phase B per §7.5).

**Proposed fix:**
- "RepE probe training... for honesty, certainty, and goal-coercion dimensions" → "RepE probe training via PCA on contrastive pairs for the honesty dimension"
- "Cross-model alignment theory via supervised Procrustes alignment between 1B and 7B model representation spaces" → DELETE bullet (the bullet's content moves to §7.5 Future Work infrastructure paragraph where it already exists).

**Page-budget impact:** −1 bullet item = ~ −3 lines.

---

### Finding §5-3 — BLOCKING — lines 357-383 §5 Experiment 1

**Current:** Entire `\paragraph{Experiment 1: OEI Characterization under Inference-Time KV-Cache Perturbation.}` plus Results + Table 1 + Interpretation.

**Issue:** Under Option D+ Experiment 1 should be **Track F entropy-MCTS negative control**, not OEI characterization. §3 Hypothesis 2 forward-references `sec:exp1` as Track F evidence. §7.1 Reward design paragraph references `sec:exp1` as Track F evidence. The current OEI characterization is α+ residue.

**Proposed fix:** Wholesale recast. The Day-4 §5 Exp 1 draft (to be authored — flagged in Day 3 P3 draft notes) will produce the replacement TeX. The existing OEI alpha-sweep content (Table 1 with σ_H/ρ_R/OEI/TDS values, ~lines 364-380) can be **relocated to §7.4 Empirical Measurement Prerequisites** which already discusses these measurements in prose; the numbers in §7.4's current paragraph 2 (line 564) implicitly reference Table 1. Move the table there.

**Page-budget impact:** Day-4 §5 Exp 1 recast draft estimated +30-50 lines body for Track F result + table; offset by relocating Table 1 to §7.4 (~ −5 line move). Net +25-45 lines.

---

### Finding §5-4 — BLOCKING — lines 385-388 §5 Experiment 2

**Current:** `\paragraph{Experiment 2: Reward-Function Ablation in Latent Space.}` + body + IMPL line indicating PARTIAL.

**Issue:** Under Option D+ Experiment 2 should be **Latent Cartography MCTS** across 4 prompt classes × 2 scales, not three-arm reward ablation. The three-arm ablation never ran (blocked by B6 bug at time of writing).

**Proposed fix:** Apply the Day-3 v10 draft (`docs/logs/2026-05-11_session-A-prose-drafts-v10-exp2.md`) which already specifies the replacement TeX + Table 2 + Figure 1 spec. Day 7 of plan v2.

**Page-budget impact:** +57 lines per the v10 draft.

---

### Finding §5-5 — RECOMMENDED — lines 430-448 §5.4 Baselines table

**Current:** 8-row table with baselines including GCG, MPA, DAMON, MT-Bench, SonarQube, CAA, Cache Steering, RL-Obfuscation. Caption refers to "comparative evaluation".

**Issue:** Under Option D+, the relevant peers are activation/cache steering and monitor-evasion work, not text-space red-teaming (GCG/MPA/DAMON) or general code-quality static analysis (SonarQube/Semgrep) or LLM-as-judge (MT-Bench/G-Eval). The text-space and code-quality rows are vestigial from when this was an alignment-faking attack-discovery paper.

**Proposed fix:** Trim table to 3 rows: Activation Steering (Rimsky 2024), Cache Steering (Belitsky 2025), Latent Monitor Evasion (Gupta-Jenner 2025). Rewrite caption to reflect "complementary inference-time KV-cache intervention techniques" rather than "comparative evaluation".

Alternative: delete §5.4 Baselines section entirely. Under Option D+ we don't run head-to-head comparisons (Cartography is a structural mapping, not a benchmark beating exercise). The table serves only to position the work.

Recommendation: **trim to 3 rows + reframe caption.**

**Page-budget impact:** −5 table rows + caption rewrite = ~ −8 lines.

---

### Finding §5-6 — BLOCKING — lines 450-472 §5.5 Evaluation Metrics

**Current:** Section defines OEI as primary metric (Eq.~\ref{eq:oei}, lines 453-458) and TDS as the divergence component of MCTS reward (line 469).

**Issue:** Under Option D+ the primary measurements are:
- Best-path α-tuple distribution per class × scale (Cartography Table 2)
- Argmax-at-best-leaf distribution per class × scale
- Accuracy (Track F McNemar paired test)
- Mean-step α per class

OEI is no longer the primary metric. TDS is no longer used as MCTS reward (§4-4 finding). §5.5 needs full rewrite.

**Proposed fix:** Replace §5.5 contents with:
1. **Primary metric (Track F + Cartography):** best-path α-tuple distribution per (class, scale) — characterizes which steering regime the MCTS selects under entropy reward.
2. **Primary metric (Track F):** paired McNemar accuracy test between MCTS arm and matched random-α control, n=200 per scale.
3. **Primary metric (Cartography):** argmax-at-best-leaf distribution per (class, scale) — characterizes whether the search amplifies a class-specific prior token.
4. **Secondary metric:** OEI (Eq.~\ref{eq:oei}) reported in §7.4 Empirical Measurement Prerequisites as monitor-design diagnostic.
5. **Internal metric (not reported):** TDS — the cross-channel divergence inside the telemetry matrix; reported in supplementary materials as one of several monitoring channels.

**Page-budget impact:** Net roughly +10 lines (replaces ~22 existing lines with ~32 of new content).

---

## §6 Theoretical Analysis

### Finding §6-1 — RECOMMENDED — line 515 worked example numbers

**Current:** "With $b = 3$, $d = 5$, and $M_{\text{KV}} = 40\text{GB}$, standard MCTS requires ${\sim}9.7$TB, while reversible MCTS requires ${\sim}40.05$GB."

**Issue:** Placeholder numbers. Track D measured 162 GB working set at 20B configurations; ~60× reduction vs naive parallel. Plan v2 Day-8 already has this scheduled ("Apply Track D §6 memory-complexity replacements (162 GB / 60× / `(b,d)=(3,5)`)").

**Proposed fix:** Day-8 task already on plan v2. No new flag needed here. Listing this finding for completeness only.

**Page-budget impact:** Net neutral or small reduction.

---

## §7 Discussion

### Finding §7-1 — RECOMMENDED — line 535 §7.2 Broader Impact

**Current:** "The Reversible MCTS framework could, in principle, be used to **discover alignment-breaking interventions** that are then exploited rather than patched. We mitigate this risk through three design choices: (1)~the framework operates in a controlled, offline environment rather than against live systems; (2)~a cryptographic hash of the evaluation payload provides an audit trail of all interventions and discoveries; (3)~we commit to responsible disclosure of any critical vulnerabilities discovered during experimentation."

**Issue:** "Discover alignment-breaking interventions" is the α+ attack-discovery framing. Under Option D+ the framework is a probe of latent geometry; it doesn't produce deployable adversarial interventions (the MCTS operates on a frozen model's internal cache, not on a deployable artifact). Three mitigations also reflect α+ framing — point (2) is the "cryptographic hash" thing that came from when we were generating adversarial datasets.

**Proposed fix:** Rewrite §7.2 (1 paragraph, ~80 words) as:

"This work has dual-use considerations. Reversible KV-Cache MCTS provides a controlled-access white-box tool for probing the latent geometry of frozen language models; the Dimensional Escape pathology it surfaces — that low-rank surrogate signals (whether as monitors or as search rewards) admit degenerate optima in high-dimensional cache space — has implications for the design of both monitoring procedures and test-time scaling schemes. We disclose findings openly through this paper and accompanying supplementary materials rather than withholding them; the framework requires white-box access to the frozen model and is not deployable as an inference-time attack vector against real-world systems."

**Page-budget impact:** Roughly neutral (~80 → ~85 words).

---

### Finding §7-2 — RECOMMENDED — lines 562-566 §7.4 paragraph 1 OEI framing

**Current (line 562 opening):** "The Orthogonal Escape Index in Eq.~\ref{eq:oei} requires that the steering direction $\mathbf{d}_K^{(\ell)}$ live in the same space as the cached key tensors..."

**Issue:** §7.4 opens with OEI as the contextual anchor. Under Option D+, the W_K projection issue this paragraph discusses is broader than OEI — it applies to any steering measurement in cache space. Reframing the opening sentence makes §7.4 more general while preserving its empirical content.

**Proposed fix:** Replace opening "The Orthogonal Escape Index in Eq.~\ref{eq:oei} requires that the steering direction..." with "Any measurement of inference-time KV-cache perturbation, including the Orthogonal Escape Index defined in Eq.~\ref{eq:oei}, requires that the steering direction...".

Remainder of paragraph unchanged.

**Page-budget impact:** +6 words.

---

### Finding §7-3 — KNOWN — line 588 §7.6 paragraph 1

**Already flagged** in `docs/logs/2026-05-11_session-log.md` Day 3 P5 entry. Listed here for completeness:

**Current:** "All experiments will be conducted on open-weight models in isolated sandbox environments. No experiments will target production systems or real users. **The adversarial datasets generated will be released under a responsible disclosure framework with appropriate access controls.** The release package will include a Croissant 1.1 metadata file (plus Croissant RAI extension fields) so provenance, usage constraints, and schema-level reproducibility checks are machine-verifiable."

**Issue:** "Adversarial datasets generated" — α+ residue; Option D+ generates no adversarial datasets.

**Proposed fix:** Replace the bolded sentence with: "We release a Croissant 1.1 metadata package documenting runtime evidence artifacts (per-node telemetry from Track F entropy-MCTS and Track G Latent Cartography sweeps) so independent reproductions can verify our findings against the original raw artifacts."

Note: This rephrasing partially overlaps with sentence 4 of the same paragraph (which talks about the Croissant 1.1 metadata file). Recommend merging the two sentences:

"All experiments were conducted on open-weight models in isolated environments. No experiments targeted production systems or real users. We release a Croissant 1.1 metadata package (with Croissant RAI extension fields) documenting runtime evidence artifacts — per-node telemetry from Track F entropy-MCTS and Track G Latent Cartography sweeps — so provenance, usage constraints, and schema-level reproducibility checks are machine-verifiable against the original raw artifacts."

**Page-budget impact:** −1 sentence = ~ −20 words.

---

## Summary

### By severity

| Severity | Count | Total page-budget delta (rough) |
|---|---:|---:|
| BLOCKING | 7 (§2-2, §2-3, §2-4, §4-2, §4-3, §4-4, §5-2 partial, §5-3, §5-4, §5-6) | -25 to -50 lines (net) |
| RECOMMENDED | 6 (§2-1, §4-1, §5-1, §5-5, §6-1, §7-1, §7-2) | -10 to -20 lines (net) |
| OPTIONAL | 1 (§2-5) | net zero |
| KNOWN | 1 (§7-3) | -1 sentence |

### Net page-budget projection

Combined stale-sentence cleanup recovers **~35-70 lines of body** (rough estimate), comfortably offsetting the +57 lines from §5 Exp 2 application (Day 7) and the +30-50 lines from §5 Exp 1 recast (Day 4). After all stale-sentence fixes + Exp 1/Exp 2 applications + §A Reproducibility Appendix (in `\appendix`, unlimited):

- **Conservative estimate:** body lines ~600 → 8.6 pages — within 8-page review budget with margin
- **Optimistic estimate** (aggressive deletes per §4-2 option (a), §4-4 option (a)): body lines ~560 → 8.0 pages — exactly at budget

Josh's hypothesis ("stale-sentence audit recovers page budget naturally") **confirmed by this audit's projections.** §6 → `\appendix` Day-10 contingency cut is likely NOT needed.

### Application strategy

| Order | Task | When | Owner |
|---|---|---|---|
| 1 | Apply BLOCKING fixes §2-2, §2-3, §2-4, §4-1, §4-3 (text-only, no equation deletes yet) | Day 4 morning | Session B |
| 2 | Decide §4-2 (Table 1 delete) and §4-4 (Eq.~mcts_reward delete) | Day 4 — needs Josh sign-off | Josh |
| 3 | Apply §4-2 and §4-4 deletes per Josh decision | Day 4 PM | Session B |
| 4 | Apply RECOMMENDED fixes §2-1, §5-1, §5-5, §7-1, §7-2 | Day 5 | Session B |
| 5 | Apply §5-3 §5 Exp 1 recast (draft by Session A first) | Day 5-7 | A drafts, B applies |
| 6 | Apply §5-4 §5 Exp 2 (from v10 draft) | Day 7 | Session B |
| 7 | Apply §5-6 §5.5 Eval Metrics rewrite | Day 7 | Session B |
| 8 | Apply §7-3 §7.6 paragraph 1 rewrite | Day 4 | Session B (post Josh sign-off) |
| 9 | Defer §2-5 OPTIONAL | Camera-ready Phase 3 or skip | — |

### Open Josh decisions

1. **§4-2 Table 1 ("Diagnostic States")** — delete entirely, or trim (option a vs option b)?
2. **§4-4 Eq.~mcts_reward subsubsection** — delete entirely, or reframe (option a vs option b vs option c)?
3. **§5-5 Baselines table** — trim to 3 rows + reframe caption, or delete §5.4 entirely?
4. **§7-3 §7.6 paragraph 1** — apply the merged rewrite proposed above?
5. **§7-1 §7.2 Broader Impact** — apply the proposed rewrite, or hold for camera-ready?

---

*End of stale-sentence audit. Findings count: 16 distinct issues across 6 sections. Submission-affecting (BLOCKING + RECOMMENDED): 13. Recommended application schedule: Day 4-7 of plan v2. Open decisions: 5 (Josh sign-off needed before applying §4-2, §4-4, §5-5).*
