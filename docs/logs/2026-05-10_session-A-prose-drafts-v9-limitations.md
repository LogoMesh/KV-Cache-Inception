# Session A Prose Drafts v9 — §7.3 Limitations — 2026-05-10

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits) — Day 2 of ARR-cycle execution plan
**Status:** Day-1 prose draft for the Option D+ pivot. Replaces §7.3 Limitations content (lines 513-515 of canonical TeX, marker `G1 | 2026-04-24 | Claims 12+13+14 added to §7 Limitations`).

## Pivot summary

Under Option α+, §7.3 Limitations was structured around an unvalidated OE-hypothesis and proposed Phase-A experiments. Under Option D+:
- The OE-hypothesis is *validated as a sub-case* of Dimensional Escape (§3.2 Hypothesis~\ref{hyp:orthogonal_escape}), but with a *negative empirical result for the reward-driven case* (Hypothesis~\ref{hyp:goodhart}) demonstrated by Track F.
- OEI is no longer a headline contribution; its limitations don't need same prominence.
- Track F is now positive evidence (the Goodhart mechanism is the finding), not a limitation.
- Cartography adds an Experiment 2 with its own scoping limitations.

Three current limitations DELETE, two REWRITE, three PRESERVE, six new ADD.

## Replacement TeX

**TeX target:** §7.3 Limitations paragraph at canonical TeX:513-515 (single paragraph; opens with "We acknowledge several limitations.").

**REVISION marker:** `[REVISION | §7.3-Limitations-recast-D+ | 2026-05-10 | Wholesale rewrite of §7.3 Limitations under Option D+ framing (Session A v9 draft, applied 2026-05-11). Deletions: (a) OE-hypothesis-not-yet-validated — superseded by Track F's validation-as-sub-case + Hypothesis 2 introduction; (b) OEI-false-negatives — OEI is no longer §1 contribution under Option D+; (c) validation-chain-to-deceptive-intent — Option D+ explicitly decouples from intent, replaces with latent-response-under-steering. Rewrites: (a) H-Neuron framing — recast as latent-response diagnostic, not deception detector; (b) σ_H — paired-with-entropy framing. Preserves: Procrustes uncertainty, RepE 1B-out-of-range, single-Llama-family. Adds 6 new Option D+ limitations: single model family, MCQ-letter-prompting under-statement, single steering direction, 4-prompt-class Cartography scope, entropy-as-falsified-design framing, 200-cycle reversibility empirical bound. Net ~+200 words. Marker `G1 | 2026-04-24 | Claims 12+13+14 added to §7 Limitations` is subsumed. — pending audit]`

**Replacement body** (replaces the single Limitations paragraph at TeX:515):

```latex
We acknowledge several limitations of the work as submitted. First, our empirical study covers only the Llama-3.2 model family at 1B and 3B parameter scales. While we observe Dimensional Escape via two structurally distinct routes at the two scales (1B prior-amplification, 3B greedy-collapse), we make no claim about MoE architectures (e.g., gpt-oss-20b), architectures outside the Llama family (e.g., Qwen, Mistral, Gemma), or scales beyond 3B. Generalization across model families is left to Phase B research; the algorithmic infrastructure of Reversible MCTS is architecture-agnostic by construction, but the Goodhart pathology we characterize may manifest with different specifics (different prior tokens, different collapse modes) in other models.

Second, our headline accuracy measurements (Track F, Experiment~1) use letter-prompted multiple-choice scoring on ARC-Easy. Greedy baseline accuracy with this protocol is $0.295$ at 1B and $0.845$ at 3B, both under-stated by approximately $20$ percentage points relative to published log-probability scoring of the same benchmark. Letter-prompted scoring is the natural protocol for our setting because entropy-MCTS steers the next-letter logit directly, so the comparison between MCTS arms is internally fair, but absolute baselines should not be compared cross-protocol. Log-probability scoring of MCTS-augmented outputs is a separate study deferred to Phase B.

Third, the steering direction used across all reported experiments is a single random unit vector in $d_{\text{model}}$ space, shared across items and across MCTS arms to isolate reward signal as the only variable (the COCONUT defense, §\ref{sec:exp1}). Our Latent Cartography sweeps (Experiment~2) similarly use a fixed direction across the alpha grid to isolate magnitude effects. Single-direction results may not generalize to richer steering bases: orthogonal probe directions, adversarially-chosen vectors, learned steering directions from calibrated honesty probes, or per-layer direction-of-difference vectors are all untested. Section~\ref{sec:measurement-prereqs} discusses the projection-based probe-to-cache lifting required to align direction calibration with cache geometry; this remains validated for $\rho_R$ but uncalibrated for the random direction we use here.

Fourth, the per-layer telemetry channels $\sigma_H$ and $\rho_R$ serve as latent-response diagnostics in our experiments rather than as deception detectors. Following Gao et al.~\cite{gao2025hneurons}, H-Neuron activation associates with hallucination and over-compliance; the extension to strategic deception is an extrapolation, not a settled equivalence. We deliberately use $\sigma_H$ and $\rho_R$ to observe geometric trajectories of MCTS search under steering, not to claim that elevated $\sigma_H$ implies the model is being deceptive. The post-merge per-layer H-Neuron calibration produces non-zero $\sigma_H$ across 14--15 of 17 layers in the alpha sweep (§\ref{sec:measurement-prereqs}), confirming the channel is responsive; whether the responsiveness corresponds to deception-relevant computation remains an open empirical question.

Fifth, Latent Cartography (Experiment~2) covers only four prompt classes (factual recall, logical deduction, moral/preference, creative completion), 100 items per class. Out-of-distribution prompts — adversarial probes, multilingual inputs, code-generation tasks, chain-of-thought-style multi-turn reasoning — are not tested. Cartography conclusions about structural pre-training prior amplification assume in-distribution behavior; transfer to OOD regimes is not characterized.

Sixth, Theorem~\ref{thm:reversibility} establishes zero-drift reversibility of the FP32 accumulator for arbitrary search depth, and we validate this empirically over 200 apply-revert cycles on Llama~3.2-1B and 3B (residual norm $= 0.00$ at machine precision). Our reported experiments operate at search depths 3 (Track F path-sampling) and 3 (Cartography MCTS branching), well within the empirical envelope. Very-deep adversarial searches (depth $\ge 50$, or cycle counts $\gg 200$) are not empirically validated; the theorem implies exact reversibility, but practical implementations may surface allocator-rounding or floating-point edge cases at extreme depth not observed in our regime.

Seventh, the headline empirical finding of this work — that entropy-minimization reward under continuous-latent search succumbs to Dimensional Escape via prior-amplification or greedy-collapse — is a negative result for one specific reward-design choice. We do not claim that *all* surrogate rewards on the cached representation must fail; Section~\ref{sec:exp1} \S7 explicitly notes that anti-Goodhart reward designs (multi-channel ensembles, external Process Reward Models, self-consistency, counterfactual-sensitivity, anti-prior penalty, information-bottleneck rewards, hybrid Best-of-N + MCTS, activation-norm regularization, layer-specific perturbation) constitute a research program for future work. The framing of Reversible MCTS as a probe of representational geometry rather than a reasoning-improvement tool is itself a scoping choice for this submission; whether a Goodhart-resistant reward can be constructed within the FP32-accumulator framework is an open Phase B question.

Eighth, Cross-model steering-vector transfer via Procrustes alignment between model families (e.g., Llama 1B $\leftrightarrow$ Llama 7B, or Llama 3.2 $\leftrightarrow$ Mistral 7B) is an open research question with uncertain guarantees and is not addressed in this submission. The Representation Engineering probe validation of Zou et al.\ (2023)~\cite{zou2023repe} was conducted at $\ge 7$B parameter scales; our use of 1B and 3B scales for primary experiments falls below this validation range, and probe-direction reliability at sub-7B scale is an additional open empirical question that our results do not resolve.
\label{sec:limitations}
```

## Notes for Session B

**Length:** ~720 words vs. the existing ~310 words. Net +410 words. This is substantial growth on §7.3 alone; some can be reclaimed if J1 returns a tight page budget by deferring the 5th, 6th, or 8th limitation to §7.5 Future Work (where they overlap with deferred Phase B items).

**Structure note:** the new §7.3 has 8 numbered limitations ("First" through "Eighth"). The first six are new or substantively rewritten; the seventh is the "entropy-as-design-we-falsify" framing that anchors the negative-result honesty; the eighth preserves the Procrustes + RepE-7B-scale-range limitations from v1 (originally items 2 and 4) combined into one item for compactness.

**Cross-references:**
- `\ref{hyp:orthogonal_escape}` and `\ref{hyp:goodhart}` (Hypothesis 1 + 2 from v9-problem-formulation.md, §3.2) — must exist before §7.3 lands (Day 2 application order: §3 first, then §7.3).
- `\ref{sec:exp1}` and `\ref{sec:exp2}` — added Day 7 when §5 gets the Experiment-1/Experiment-2 labels.
- `\ref{sec:measurement-prereqs}` — exists.
- `\ref{thm:reversibility}` — exists.
- `\cite{gao2025hneurons}`, `\cite{zou2023repe}` — exist.

**`\label{sec:limitations}`** added at end so reviewers can quickly reference; not strictly needed since `\subsection{Limitations}` provides a default ref, but explicit is better for citation precision.

**Cited numbers (verbatim from track reports):**
- "$0.295$ at 1B and $0.845$ at 3B" — Track F §2 (200-item aggregate, not Track C's 100-item subset which reported 0.310/0.840)
- "approximately $20$ percentage points relative to published log-probability scoring" — Track C §2.1 + §6 recommendation
- "14--15 of 17 layers" — post-merge B6 verification 2026-05-07
- "200 apply-revert cycles ... residual norm $= 0.00$" — Track A + Theorem 1 validation, TeX:309

**What's intentionally not in the limitations:**
- Bailey 2024 inference-time-KV-cache regime is NOT a limitation (it's deferred to Phase B per the unified framing, but the §3.2 framing positions it as an open extension rather than a weakness of *this* paper).
- "OEI may produce false negatives" — dropped wholesale; OEI is no longer headline.
- "Phase B experiments not yet run" — dropped wholesale; the paper as submitted ships with the empirical content it has, not promises of future work.

**Day-2 application order alongside other v9 drafts:**
1. abstract (replaces TeX:50-53)
2. §1 contributions (replaces TeX:80-87)
3. §3 problem formulation (replaces TeX:131-154)
4. §7.3 Limitations (this draft — replaces TeX:513-515)
5. title (replaces TeX:39, see v9-title.md)

All five together = one coordinated Session B Day-2 TeX pass per the v2 14-day plan.

---

*Session A v9 §7.3 Limitations draft complete. Apply Day 2 of ARR-cycle execution plan as part of the coordinated abstract → contributions → §3 → §7.3 → title trio (now quintet).*
