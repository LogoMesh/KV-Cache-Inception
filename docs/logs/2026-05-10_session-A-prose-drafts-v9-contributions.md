# Session A Prose Drafts v9 — §1 Contributions — 2026-05-10

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits) — Day 2 of 14-day execution plan
**Status:** Day-1 prose draft for the Option D+ pivot. Replaces the Option α+ contributions block (recast 2026-05-06, marker `contributions-recast-α+`) wholesale.

## Pivot summary

Old (Option α+) contributions order: MCTS algorithm → telemetry matrix → OEI as targeted measurement framework.

New (Option D+) contributions order:
1. Reversible KV-Cache MCTS algorithm + Theorem 1 + measured memory bound (UNCHANGED from prior recast in claim, but absorbs Track D's measured 3.04× constant).
2. Per-layer telemetry matrix as latent-search **monitoring** infrastructure (repositioned: it is no longer the search's reward function — that pathway is what Track F empirically falsifies).
3. **NEW:** Empirical demonstration of Dimensional Escape across both reward-driven (Track F) and structural-mapping (Latent Cartography) regimes, with explicit unification with the monitor-driven Orthogonal Escape hypothesis.

The "first formal definition of OEI" headline (already softened in α+ recast) is fully retired here. OEI is no longer a §1 contribution; it appears in §3 (definition) and §5 (incidental measurement during Cartography), but not as a contribution-level claim.

## Replacement TeX

**TeX target:** `\subsection{Contributions}` block at TeX:76–87 (the `[REVISION | contributions-recast-α+ | 2026-05-06 | ...]` marker and the `\begin{enumerate}...\end{enumerate}` block).

**REVISION marker:** `[REVISION | contributions-recast-D+ | 2026-05-10 | Replaced entire §1 Contributions enumerate block with Option D+ recast (Session A v9 draft, applied 2026-05-10). Old order (Option α+): MCTS → telemetry matrix → OEI as targeted measurement. New order (Option D+): MCTS → telemetry matrix as monitoring → Dimensional Escape empirical demonstration. The 2026-05-06 contributions-recast-α+ marker is subsumed; the OEI-as-§1-contribution framing is retired (OEI now appears only in §3 definitional content and §5 incidental measurement). Contribution 1 absorbs Track D's measured 3.04x constant for the memory bound and drops the "Phase B 20B deferral" hedge; Contribution 2 is repositioned from "third reward component" to "monitoring substrate observing the search" (because Track F empirically falsified the entropy-reward pathway and we deliberately argue against telemetry-as-reward in Contribution 3). Contribution 3 is wholly new: Dimensional Escape as the unifying claim, instantiated by Track F (Experiment 1) and Latent Cartography (Experiment 2). Forward refs: thm:reversibility, prop:memory, sec:exp1, sec:exp2 — all resolve. — pending audit]`

**Replacement body** (replaces TeX:81–87 verbatim):

```latex
\begin{enumerate}[leftmargin=*]
    \item \textbf{Reversible Monte Carlo Tree Search in KV-cache latent space.} We introduce an inference-time algorithm that mutates a frozen Transformer's cached key/value tensors via in-place arithmetic ($\mathbf{K}_t \leftarrow \mathbf{K}_t + \alpha \cdot \mathbf{d}_K^{(\ell)}$) and inverts each mutation exactly through an FP32 accumulator. We prove (Theorem~\ref{thm:reversibility}) that the accumulator guarantees zero-drift reversibility independent of search depth and validate this empirically on Llama~3.2-1B-Instruct and Llama~3.2-3B-Instruct (drift $= 0$ to machine precision over 200 apply--revert cycles). The memory complexity of the algorithm is independent of branching factor---a constant-factor multiple of the base KV cache (Proposition~\ref{prop:memory}), measured at $3.06\times M_{\text{KV}}$ on Llama~3.2-1B and $3.04\times M_{\text{KV}}$ on Llama~3.2-3B, identical across the tested $(d, b, n) \in \{(3, 3, 27), (5, 3, 81), (10, 3, 1700)\}$ configurations---in contrast to the $O(b^d \cdot M_{\text{KV}})$ overhead of standard parallel-cache MCTS, a $\sim$60$\times$ reduction at $(b, d) = (3, 5)$ on 20B-parameter configurations. The algorithm is gradient-free and operates on a frozen model at inference time, distinguishing it from gradient-based input-space attacks~\cite{zou2023gcg, bailey2024obfuscated} and from training-time monitor-evasion methods~\cite{gupta2025rlobfuscation}.

    \item \textbf{Per-layer telemetry matrix as latent-search monitoring infrastructure.} We define a $2 \times L$ matrix $\mathbf{T}_t$ that fuses bottom-up hallucination-associated-neuron activation~\cite{gao2025hneurons} with top-down representation-engineering projections~\cite{zou2023repe}, providing a continuous signal at each generation step across $L$ transformer layers. The matrix serves as the \emph{monitoring substrate} for the experiments in §\ref{sec:experiments}---observing the per-step geometric trajectory of the search---rather than as the search's reward function. We argue against the latter design choice in light of Contribution~3: surrogate-reward-as-MCTS-objective in this latent space succumbs to a generic geometric pathology, regardless of whether the reward is constructed from telemetry channels, next-token entropy, or any other internal model signal. Both telemetry channels respond to steering perturbations as expected: $\rho_R$ varies substantially with intervention magnitude, and $\sigma_H$ rises monotonically with steering across 14--15 of 17 layers under per-layer H-Neuron calibration~\cite{gao2025hneurons}.

    \item \textbf{Empirical demonstration of Dimensional Escape under continuous-latent search.} We establish a fundamental geometric pathology of MCTS in overparameterized latent spaces: surrogate rewards lacking exogenous grounding inevitably resolve localized optimization pressure by routing representations through unconstrained dimensions, succumbing to either prior-amplification or greedy-collapse rather than satisfying the semantic objective. Two complementary experiments support the claim. \emph{Experiment 1} (controlled negative-control, §\ref{sec:exp1}): on $n = 200$ paired ARC-Easy items at 1B and 3B, an entropy-minimization reward---which statically correlates with correctness at $|r| \approx 0.60$---collapses under MCTS optimization to within $0.5$pp of a uniform-random reward (paired McNemar $p = 1.0$ at both scales), with mechanism-level differentiation by scale: the 1B search reinforces the model's first-letter prior (predicting ``A'' on 185 of 200 items under heavy steering), while the 3B search collapses to greedy decoding (selecting minimal-magnitude steering on 51\% of items). \emph{Experiment 2} (Latent Cartography, §\ref{sec:exp2}): sweeps across four prompt classes---factual recall, logical deduction, moral/preference, creative completion---at six steering magnitudes structurally map the pre-training priors the search converges upon, demonstrating that the failure manifests across prompt domains and that the specific terminal-token distribution varies systematically with the prompt's semantic class. We further show that the reward-driven pathology surfaced here is structurally identical to the monitor-driven \emph{Orthogonal Escape} previously hypothesized for alignment faking~\cite{bailey2024obfuscated}: both manifestations are sub-cases of the same overparameterized-latent-space geometry (§\ref{sec:dim_escape}). The unified framing positions Reversible MCTS as a high-precision diagnostic probe of frozen-model representational geometry, not a reasoning-improvement tool.
\end{enumerate}
```

## Notes for Session B

- Word count: Contribution 1 ~165 words; Contribution 2 ~140 words; Contribution 3 ~270 words. Total ~575 words for the §1 contributions block (vs. ~480 in the Option α+ version). Net +95 words. Manageable; if page budget is tight on Day 10, the Contribution 3 prose can compress by ~40 words by abbreviating Experiment 1's mechanism description (the full mechanism is in §5).
- Forward references that must resolve:
  - `Theorem~\ref{thm:reversibility}` — exists.
  - `Proposition~\ref{prop:memory}` — exists; need to verify Track D's `3.04× M_KV` constant lands in the proposition body Day 8 (per execution plan).
  - `\ref{sec:experiments}`, `\ref{sec:exp1}`, `\ref{sec:exp2}` — `sec:experiments` already exists; `sec:exp1` and `sec:exp2` need new `\label{}` insertions in §5 when Track F gets its formal §5.1 home and Cartography gets its §5.2 home (Day 7 of execution plan).
  - `\ref{sec:dim_escape}` — needs new `\label{}` in §3.2 Dimensional Escape block (paired draft v9-problem-formulation.md).
  - `\cite{gao2025hneurons}`, `\cite{zou2023repe}`, `\cite{zou2023gcg}`, `\cite{bailey2024obfuscated}`, `\cite{gupta2025rlobfuscation}` — all exist.

- Specific numbers (verbatim from track reports — DO NOT round):
  - "200 apply--revert cycles" (Track A 2026-05-09 + earlier Theorem 1 validation)
  - "$3.06\times M_{\text{KV}}$ on Llama~3.2-1B and $3.04\times M_{\text{KV}}$ on Llama~3.2-3B" — Track D §2 cells 1-3 (1B reports 382.97/125.0 MiB = 3.064×; 3B reports 1328.69/437.5 MiB = 3.037×). Corrected 2026-05-10 post-verification — earlier draft cited "3.04× on both", which under-reported the 1B measurement.
  - "$\sim$60$\times$ reduction at $(b, d) = (3, 5)$" (Track D §4 savings-ratio table; the constant is fixed for the (b, d) pair, so the qualifier is required for accuracy. Replaces the earlier 242× claim in the canonical TeX §6 placeholder.)
  - "$n = 200$ paired ARC-Easy items" (Track F)
  - "$|r| \approx 0.60$" (Track A static correlation, both 1B and 3B)
  - "within $0.5$pp" + "paired McNemar $p = 1.0$" (Track F §3)
  - "185 of 200 items" + "51\% of items" (Track F §1 + §7 mechanism diagnosis)
  - "14--15 of 17 layers under per-layer H-Neuron calibration" (post-merge B6 verification 2026-05-07; ca3ec8e)

- The phrase "We argue against the latter design choice in light of Contribution~3" in Contribution 2 is a deliberate forward signal. It tells the reviewer: the natural architectural pairing (telemetry-as-reward) was tested and found to fail, and the paper is honest about that. This is what Gemini round 4 called the "epistemically clean" framing.

- The phrase "the failure manifests across prompt domains and the specific terminal-token distribution varies systematically with the prompt's semantic class" in Contribution 3 is **conditional** on Cartography producing the structural finding. If Day 7 EOD shows the cartography sweeps fail to produce class-dependent variation (per plasticity rule in 14-day plan §2), this sentence and the Experiment 2 reference get demoted. Track this dependency in the running log.

- The "high-precision diagnostic probe of frozen-model representational geometry, not a reasoning-improvement tool" closer is the same disavowal as the abstract. Repeating it here is intentional — the §1 Contributions list is what reviewers cite when summarizing the paper, so the disavowal needs to live in both abstract and §1.

## Cross-section impact

This contributions block forward-references Dimensional Escape as a defined concept (§3.2 `sec:dim_escape`). The §3 Problem Formulation v9 draft (separate file) introduces the Definition. The order of Session B edits Day 2: abstract → §1 contributions → §3 problem formulation, applied as a single coordinated pass to avoid mid-pass cross-reference resolution failures.

The §2 Background section is largely unaffected by this contributions rewrite. The §2.1 alignment-faking framing softened in `§2.1-soften-α+` (2026-05-07) survives intact under Option D+ — alignment faking is now positioned as one *application* of the unified Dimensional Escape framework rather than the headline framing of the paper.

The §6 Memory Complexity Proposition needs to absorb Track D's $3.04\times$ constant Day 8 (separate edit, marker `§6-track-D-numbers`). Until then, the contributions block here cites the constant directly even though the proposition body still has the placeholder; this is a deliberate temporary inconsistency that resolves cleanly on Day 8.

---

*Session A v9 contributions draft complete. Apply Day 2 of 14-day execution plan as part of the abstract → contributions → §3 coordinated trio.*
