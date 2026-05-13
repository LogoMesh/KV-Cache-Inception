# Session A Prose Drafts v9 — Abstract — 2026-05-10

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits) — Day 2 of 14-day execution plan
**Status:** Day-1 prose draft for the Option D+ pivot. Replaces the Option α+ abstract (recast 2026-05-06, marker `abstract-recast-α+`) wholesale.

**Pivot rationale:** Track F (2026-05-09) falsified the entropy-as-MCTS-reward formulation that anchored Candidate A's third contribution. Gemini round 4 (2026-05-10) verdict: no Path X exists in 14 days; Option D is correct. Framing upgrade: replace narrow "alignment-faking detection" / "test-time scaling" headlines with **Dimensional Escape** — a unifying geometric framework covering both monitor-driven and reward-driven manifestations of the same overparameterized-latent-space pathology.

## Replacement TeX

**TeX target:** abstract block at TeX:50–53 (the `[REVISION | abstract-recast-α+ | 2026-05-06 | ...]` marker and the `\begin{abstract}...\end{abstract}` body).

**REVISION marker:** `[REVISION | abstract-recast-D+ | 2026-05-10 | Replaced abstract entirely (Session A v9 draft, applied 2026-05-10). Old hook (Option α+): "Latent-space analysis ... requires search infrastructure ... we discuss measurement-pipeline prerequisites." New hook (Option D+): Dimensional Escape as unifying geometric pathology, with Track F (Experiment 1) as controlled negative control and Latent Cartography (Experiment 2) as positive structural mapping. The "alignment-faking detection" headline is dropped; OEI is no longer abstract-level. Forward refs: Theorem~\ref{thm:reversibility} preserved; new refs to Experiments 1 and 2 (sec:exp1, sec:exp2). The Bailey 2024 complementary-measurement framing is preserved but folded into the unified Dimensional Escape framing rather than positioned as the headline contribution. — pending audit]`

**Replacement body** (replaces lines 51–53 verbatim):

```latex
\begin{abstract}
We introduce \textbf{Reversible Monte Carlo Tree Search in KV-cache latent space}, an inference-time algorithm that mutates a frozen Transformer's cached key/value tensors in place via FP32 accumulators and rolls back exact baseline state after each tree expansion. We prove that the FP32 accumulator guarantees zero-drift reversibility (Theorem~\ref{thm:reversibility}; validated to machine precision over 200 apply--revert cycles on Llama~3.2-1B and Llama~3.2-3B), and we measure a constant-factor memory overhead of $3.04$--$3.06\times$ the base KV cache, independent of branching factor---a $\sim$60$\times$ reduction at $(b, d) = (3, 5)$ relative to naive parallel-cache MCTS on 20B-parameter configurations. Using this framework, we conduct a systematic empirical study of continuous-latent search on Llama~3.2 architectures and expose a fundamental geometric pathology we term \textbf{Dimensional Escape}: under any surrogate reward lacking exogenous grounding, the high-dimensional cache resolves localized optimization pressure by routing representations through unconstrained orthogonal dimensions, satisfying the proxy metric without satisfying the semantic objective. We establish this via two complementary experiments. A controlled negative-control on ARC-Easy ($n=200$ paired items) shows that an entropy-minimization reward---which statically correlates with correctness at $|r| \approx 0.60$---collapses under MCTS optimization at both scales: $\Delta(\text{entropy}-\text{random}) = -0.5$pp at 1B and $+0.5$pp at 3B (paired McNemar $p = 1.0$ both scales), with the 1B search reinforcing the model's first-letter prior and the 3B search collapsing to greedy decoding. Latent Cartography sweeps across four prompt classes (factual recall, logical deduction, moral/preference, creative completion) at six steering magnitudes then map the structural pre-training priors the search converges upon. We show that the reward-driven failure surfaced here is structurally identical to the monitor-driven \emph{Orthogonal Escape} previously hypothesized for alignment faking~\cite{bailey2024obfuscated}: both are sub-cases of the same overparameterized-latent-space pathology. Reversible MCTS is therefore a high-precision diagnostic probe of frozen-model representational geometry rather than a reasoning-improvement tool. Artifacts are released with a Croissant~1.1 metadata package.
\end{abstract}
```

## Notes for Session B

- Word count: ~270 words (vs. ~290 in the existing abstract). Within EMNLP norms.
- Forward references that must resolve in the final TeX:
  - `Theorem~\ref{thm:reversibility}` — already exists at the §6 Memory Complexity / FP32 Accumulator block.
  - `\cite{bailey2024obfuscated}` — already in references.
  - The two empirical pillars are referenced descriptively rather than via `\ref`; concrete `\ref{sec:exp1}` / `\ref{sec:exp2}` are added in the §1 Contributions and §5 sections, not the abstract. This keeps the abstract self-contained.
- The phrase "validated to machine precision over 200 apply--revert cycles on Llama~3.2-1B and Llama~3.2-3B" is verbatim-precise to Track A / earlier Theorem 1 validation; do not loosen.
- "$3.04$--$3.06\times$ the base KV cache" — verbatim from Track D §2 (measured 3.06× at 1B, 3.04× at 3B; the report itself uses the "3.04–3.06×" range throughout). Earlier draft used "$3.04\times$" as a single conservative number but that under-reported the 1B measurement; corrected 2026-05-10 post-verification.
- "$\sim$60$\times$ reduction at $(b, d) = (3, 5)$" — Track D §4 table shows the savings ratio is constant *for fixed (b, d)*: 60× at $(3, 5)$, ~14580× at $(3, 10)$. The qualifier $(b, d) = (3, 5)$ was added post-verification because the unqualified "60×" elsewhere in the paper would be inconsistent with the algorithm's actual depth-scaling behavior.
- The two paired-McNemar numbers (`-0.5$pp at 1B and `+0.5$pp at 3B`, `p=1.0` both scales) are verbatim from Track F §3. Do not round.
- "$|r| \approx 0.60$" is Track A's static-correlation finding. Do not state as a precise decimal — Track A's value was reported as approximately 0.60 across both scales.
- "Reversible MCTS is therefore a high-precision diagnostic probe of frozen-model representational geometry rather than a reasoning-improvement tool" is the sentence that explicitly retires the System-2 / reasoning-improvement framing. Keep the explicit "rather than" — reviewers who arrive expecting a test-time-scaling paper need to see this disavowal early.
- The Croissant 1.1 mention at the end is preserved from the Option α+ abstract; the artifact release is unchanged by the pivot.

## Cross-section impact (for downstream Day-2/3 drafts)

After this lands, the §1 Contributions list (v9-contributions.md, separate file) and §3 Problem Formulation (v9-problem-formulation.md, separate file) need to be applied as a coordinated trio. The abstract above forward-references "Dimensional Escape" as a defined term; §3 is where it gets formally defined. Order of TeX edits should therefore be: abstract → §1 Contributions → §3 Problem Formulation, applied in a single Session B pass.

The §7 Discussion rewrite (Day 2 of execution plan, separate v9-discussion.md draft) replaces the OEI-saturation-discussion paragraphs with Dimensional Escape implications for monitor design AND reward design + Future Work pointer to Phase B anti-Goodhart reward research.

The §5 Experiments rewrite is split: Experiment 1 (Track F negative control) and Experiment 2 (Latent Cartography) are drafted Days 5-6 as Cartography data lands.

---

*Session A v9 abstract draft complete. Apply Day 2 of 14-day execution plan. Pair with v9-contributions.md and v9-problem-formulation.md for a coordinated single-pass TeX edit.*
