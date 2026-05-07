# Session A Prose Drafts v2 — 2026-05-06 PM

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits)
**Status:** Three drafts ready for application following morning's six-marker landing. Each piece designed to be applied as in-place TeX replacement or insertion with REVISION markers tagged `[REVISION | <slug> | 2026-05-06]`.

**Background:** Session B's morning session landed all four pieces from `2026-05-06_session-A-prose-drafts.md` (six markers total). Two structural items remained outside that batch and were flagged back via §7: §3 (Problem Formulation) still leads with alignment-faking framing, and §5 Experiment 1 still uses hypothesis-test language. The morning batch was scoped to abstract + §1 + §2.2 + §7.4, so this is expected — not a gap.

In parallel, **Josh's mechanical fix for GAP-C2-06 landed and was verified.** Per-α telemetry now varies monotonically: OEI 0.9909 → 0.9912 → 0.9915 → 0.9924 → 0.9950, ρ_R −1.306 → −0.533, TDS 0.094 → 0.046. Theorem 1 holds (residual_norm = 0.00e+00); 47/47 unit tests pass. Code change at `logomesh/kv_mcts.py` (W_K/W_V projection in `_project_residual_to_kv_shape` + accumulator router); raw post-fix data at `tmp/runs/2026-05-06_b6_post-fix.json`; full report appended to `docs/logs/2026-05-05_diagnostic-report.md`.

A separate finding emerged from the post-fix run: σ_H_mean is exactly 0.0 across all α both pre- and post-fix, independent of the steering pipeline. This is consistent with H-Neuron monitor calibration (hallucination-inducing prompts vs. non-hallucination test prompts producing per-neuron-ReLU clamping to zero) — i.e., working as designed, but the bottom-up channel is silent in our reference experiments. Documented in piece #4-revised below.

**Pieces in this file:**
1. **Piece #4-revised** — replaces §7.4 honest-finding subsection. Tightens the morning's draft now that the fix is in: removes "early version" hedging, reflects the actual W_K projection fix, references §5 for results numbers (rather than restating them), adds the σ_H = 0 finding. Net: ~80 words shorter than the morning version.
2. **Piece #5** — §3 reframe (Problem Formulation). Reframes from "Alignment Faking Detection Problem" to "Inference-Time Latent-Space Analysis," preserving Eq. 1 (now positioned as the alignment-monitoring specialization) and Hypothesis 1 verbatim.
3. **Piece #6** — §5 Experiment 1 reframe with results integration. Replaces the hypothesis-test framing with measurement-and-report framing; numbers from the post-fix run live here.

**Application order**: #4-revised first (it removes the placeholder language so the rest of the paper is internally consistent), then #5 (§3 reframe — independent), then #6 (§5 Experiment 1 — references §7.4 forward).

**Cross-references to verify before applying**:
- `\label{sec:measurement-prereqs}` — already exists from morning's piece #4. Verify still resolves.
- `\ref{thm:reversibility}` — added 2026-05-06 morning. Verify present.
- `\ref{prop:memory}` — added 2026-05-06 morning. Verify present.
- `\ref{eq:oei}` — existing label.
- `\ref{eq:mcts_reward}` — existing label.
- `\ref{eq:hneuron_stress}` — existing label (Eq. 3).
- `\ref{hyp:orthogonal_escape}` — existing label (Hypothesis 1).

---

## 1. Piece #4-revised — §7.4 honest-finding subsection (replacement)

**TeX target:** the subsection added 2026-05-06 morning at TeX:488 (§7.4 with `\label{sec:measurement-prereqs}`). Replace the entire subsection content.

**REVISION marker:** `[REVISION | §7.4-post-fix-update | 2026-05-06]` (in addition to the morning's marker, which can stay)

**Replacement text:**

```latex
\subsection{Empirical Measurement Prerequisites and Initial Observations}
\label{sec:measurement-prereqs}

The Orthogonal Escape Index in Eq.~\ref{eq:oei} requires that the steering direction $\mathbf{d}_K^{(\ell)}$ live in the same space as the cached key tensors. Representation-engineering probes are calibrated against residual-stream activations of dimension $d_{\text{model}}$, while the KV cache stores per-head representations of dimension $\text{kv\_heads} \times d_{\text{head}}$ under grouped-query attention. For Llama~3.2-1B these are 2048 and $8 \times 64$ respectively; an explicit projection between probe space and cache space is required. In our reference implementation we identified, characterized, and resolved a measurement-pipeline failure mode in which the broadcast layer silently zeroed the steering delta on shape incompatibility, producing forward passes that were bit-identical across $\alpha$. The fix projects probe directions through layer-specific key projection weights, $\mathbf{d}_K^{(\ell)} = \mathbf{W}_K^{(\ell)} \mathbf{d}_{\text{probe}}^{(\ell)}$, reshaped to per-head form --- interpretable as the cache state that would result from the corresponding residual-stream perturbation modulo position-dependent rotary embedding. Two alternative approaches are also viable: intervening at the residual stream prior to the K/V projection and re-running the layer's forward computation, or calibrating probes directly in per-head cache space against contrastive cache states. Systematic comparison of the three is left to future work.

Under the projection-based fix, OEI varies monotonically with steering magnitude on Llama~3.2-1B-Instruct (results in §\ref{sec:experiments}). The high baseline value (${\sim}0.99$) is geometrically expected: in $d_{\text{model}} = 2048$ with a rank-1 linear monitor, removing one direction from any perturbation leaves ${\sim}99\%$ of the perturbation's norm in the null space by default. This high-dimensional saturation is consistent with the regime Bailey et al.~\cite{bailey2024obfuscated} characterize when discussing why activation cosine similarity is hard to drive low in a behavior-preserving setting; the small but monotonic OEI spread we observe suggests the metric is responsive to genuine steering structure rather than producing a uniform default. Whether OEI under a richer monitor (rank-$k$ probe matrix, or attention-aggregated probes; cf.~\cite{gupta2025rlobfuscation}) yields a more discriminative signal is an open empirical question and a natural extension of the framework.

The bottom-up H-Neuron stress channel ($\sigma_H$) is reported as zero across all $\alpha$ in our reference experiments. This is independent of the steering pipeline (which demonstrably reaches the cache, per the per-$\alpha$ variation in $\rho_R$, TDS, and OEI reported in §\ref{sec:experiments}). The H-Neuron monitor's calibration relies on contrastive activation analysis between hallucination-inducing and non-hallucination-inducing inputs~\cite{gao2025hneurons}; our test prompts are not hallucination-inducing, and the per-neuron ReLU operation in Eq.~\ref{eq:hneuron_stress} correctly clamps below-baseline activations to zero. Characterizing $\sigma_H$ under hallucination-inducing input is required to verify the bottom-up channel's responsiveness in the inference-time KV-cache regime, and we identify this as a precondition for full telemetry-matrix evaluation. We note that Theorem~\ref{thm:reversibility} and Proposition~\ref{prop:memory} are unaffected by these measurement choices: both are intervention-agnostic.
```

**Notes for Session B:**
- The forward reference `\ref{sec:experiments}` assumes §5 (Experimental Design) has a `\label{sec:experiments}` on the section heading. If not, please add `\label{sec:experiments}` to the §5 heading as part of this batch.
- "We note that Theorem~\ref{thm:reversibility} and Proposition~\ref{prop:memory}..." paragraph is the safest single-paragraph trim if the EMNLP 8-page limit binds after style switch — it's reassurance, not load-bearing content.
- The phrase "we identified, characterized, and resolved" (rather than "we identified and have a candidate fix for") is now safe to use because the post-fix B6 verified the resolution actually works. Do not soften to "candidate fix."

---

## 2. Piece #5 — §3 reframe (Problem Formulation)

**TeX target:** §3.1 currently titled "The Alignment Faking Detection Problem". Replace the section heading and the prose paragraph(s) before Eq. 1, while preserving Eq. 1, the Hypothesis 1 environment, and §3.2 ("The Discrete Text-Space Bottleneck") verbatim.

**REVISION marker:** `[REVISION | §3-reframe-α+ | 2026-05-06]`

**Replacement (subsection heading + prose; Eq. 1 and Hypothesis 1 preserved verbatim, §3.2 unchanged):**

```latex
\subsection{Inference-Time Latent-Space Analysis on Frozen Models}

Let $f_\theta$ be a language model with parameters $\theta$ held fixed throughout this work, and let $\mathbf{h}_t^{(\ell)} \in \mathbb{R}^d$ denote the hidden state at layer $\ell$ and token position $t$. A linear monitor $\mathbf{P}_{\text{monitor}}$ is a projection in $\mathbb{R}^d$ --- for instance, a representation-engineering honesty probe~\cite{zou2023repe} or an aggregation thereof~\cite{gupta2025rlobfuscation}. The central question of this paper is: under controlled perturbations of the model's cached key/value tensors (and hence its hidden states at subsequent positions), how does the geometric relationship between the monitored subspace $\text{Range}(\mathbf{P}_{\text{monitor}})$ and the unmonitored complement $\text{Null}(\mathbf{P}_{\text{monitor}})$ evolve?

This question has multiple downstream applications. In mechanistic interpretability, it characterizes how representations respond to controlled perturbations along monitored directions. In alignment evaluation, it specializes to a robustness check on latent-space monitors: a model exhibits \emph{alignment faking} if it satisfies

\begin{equation}
\text{Behavioral}(f_\theta, x) \in \mathcal{S} \quad \text{while} \quad \exists\, \mathbf{v} \in \text{Null}(\mathbf{P}_{\text{monitor}}): \langle \mathbf{h}_t^{(\ell)}, \mathbf{v} \rangle \neq 0
\label{eq:alignment_faking}
\end{equation}

where $\mathcal{S}$ is the set of safe behavioral outputs and $\text{Null}(\mathbf{P}_{\text{monitor}})$ is the null space of the monitoring projection. That is, the model produces safe outputs while encoding non-trivial information in directions invisible to the monitor. Whether such a configuration is reachable under inference-time, frozen-weight steering --- without weight updates or input-space gradient training --- is an open empirical question. Bailey et al.~\cite{bailey2024obfuscated} characterized the analogous question for gradient-trained input-space attacks and reported a tentative negative result; we formulate the inference-time variant as Hypothesis~\ref{hyp:orthogonal_escape}.

\begin{hypothesis}[Orthogonal Escape under inference-time KV-cache perturbation]\label{hyp:orthogonal_escape}
When corrective pressure is applied along the monitored subspace $\text{Range}(\mathbf{P}_{\text{monitor}})$, a sufficiently capable model under inference-time steering may redistribute its representation into $\text{Null}(\mathbf{P}_{\text{monitor}})$, preserving behavior while evading detection:
\begin{equation}
\mathbf{h}_t^{(\ell)} \xrightarrow{\text{steering}} \mathbf{h}_t^{(\ell)} + \alpha \cdot \mathbf{d}_{\text{steer}} \implies \text{Proj}_{\text{Null}(\mathbf{P})}(\mathbf{h}_{t+k}^{(\ell)}) \neq \mathbf{0}
\label{eq:orthogonal_escape}
\end{equation}
where $\mathbf{d}_{\text{steer}}$ is the steering direction and $\alpha$ is the intervention magnitude.
\end{hypothesis}
```

**Notes for Session B:**
- The subsection heading change is the load-bearing reframe. "The Alignment Faking Detection Problem" → "Inference-Time Latent-Space Analysis on Frozen Models" pivots the entire framing from "this paper is about detecting alignment faking" to "this paper is about a measurement framework, with alignment faking as one specialization."
- Eq. 1 (alignment_faking) is preserved verbatim with its label. Hypothesis 1 is preserved with two textual softens: "When corrective pressure is applied... a sufficiently capable model **under inference-time steering may** redistribute" (was "will redistribute"). This is honest under Option α+ — the empirical evidence in §5 doesn't support the strong "will" claim.
- §3.2 "The Discrete Text-Space Bottleneck" is unchanged — that section motivates MCTS in latent space, which is unaffected by the reframe.
- After applying, verify the §1 Contribution 3 forward pointer ("see §3 for the formal definition") still resolves correctly under the new heading.

---

## 3. Piece #6 — §5 Experiment 1 reframe with results integration

**TeX target:** §5 Experiment 1 paragraph at TeX:381 (per session log; verify with grep). Replace the existing "\paragraph{Experiment 1: Orthogonal Escape Detection.}" paragraph and its IMPL comment.

**REVISION marker:** `[REVISION | Exp1-reframe-with-results-α+ | 2026-05-06]`

**Replacement text:**

```latex
\paragraph{Experiment 1: OEI Characterization under Inference-Time KV-Cache Perturbation.}
We apply layer-specific steering deltas of increasing magnitude $\alpha \in \{0.1, 0.5, 1.0, 2.0, 5.0\}$ along the RepE honesty direction, projected through the per-layer key projection weights into per-head cache space (see §\ref{sec:measurement-prereqs}). The sweep is run on Llama~3.2-1B-Instruct on a single 12\,GB consumer GPU; reversibility on Llama~3.2-3B-Instruct is independently confirmed (Theorem~\ref{thm:reversibility}, residual\_norm $= 0.00 \times 10^{0}$ over 200 cycles, validated 2026-04-16). For each $\alpha$ we measure: (i) the OEI (Eq.~\ref{eq:oei}); (ii) the RepE honesty projection $\rho_R$; (iii) the H-Neuron stress signal $\sigma_H$; (iv) the cross-channel divergence $\text{TDS} = \text{JSD}(\sigma_H, \rho_R)$.

\textbf{Results.} Per-$\alpha$ mean values across the sweep are reported in Table~\ref{tab:exp1-results}. OEI varies monotonically with $\alpha$, from $0.991$ at $\alpha = 0.1$ to $0.995$ at $\alpha = 5.0$ --- a spread of $0.004$ in a metric bounded in $[0, 1]$. The high baseline value is geometrically expected (§\ref{sec:measurement-prereqs}). The RepE honesty channel $\rho_R$ varies more substantially, from $-1.306$ at $\alpha = 0.1$ to $-0.533$ at $\alpha = 5.0$, confirming that the linear monitor channel registers the in-direction component of the steered cache state. Cross-channel divergence TDS \emph{decreases} monotonically with $\alpha$ (from $0.094$ to $0.046$), indicating that the H-Neuron and RepE channels agree more strongly under stronger steering --- the opposite of the signature predicted by the Orthogonal Escape Hypothesis (Hypothesis~\ref{hyp:orthogonal_escape}). The H-Neuron stress channel $\sigma_H$ reports zero across all $\alpha$; this is consistent with its calibration against hallucination-inducing prompts and is discussed further in §\ref{sec:measurement-prereqs}.

\begin{table}[h]
\centering
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
$\alpha$ & 0.1 & 0.5 & 1.0 & 2.0 & 5.0 \\
\midrule
OEI & 0.991 & 0.991 & 0.992 & 0.992 & 0.995 \\
$\rho_R$ & $-1.306$ & $-1.258$ & $-1.198$ & $-1.080$ & $-0.533$ \\
$\sigma_H$ & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 \\
TDS & 0.094 & 0.090 & 0.084 & 0.075 & 0.046 \\
\bottomrule
\end{tabular}
\caption{Per-$\alpha$ mean telemetry under inference-time KV-cache steering on Llama~3.2-1B-Instruct, post-projection fix. OEI shows small but monotonic variation in the high-dimensional saturation regime; $\rho_R$ varies substantially, confirming the steering reaches the cache; TDS \emph{decreases}, indicating the channels agree more under stronger steering --- the opposite of the Orthogonal Escape signature. $\sigma_H = 0$ throughout reflects monitor calibration prerequisites discussed in §\ref{sec:measurement-prereqs}. Reversibility holds end-to-end (residual\_norm $= 0$).}
\label{tab:exp1-results}
\end{table}

\textbf{Interpretation.} The results are consistent with the negative-result regime Bailey et al.~\cite{bailey2024obfuscated} report for gradient-trained input attacks: high-dimensional null-space saturation dominates, and orthogonal escape does not emerge from non-orthogonality-optimizing search at the model scales tested. We interpret this as a measurement under one specific monitor specification (rank-1 RepE honesty probe at the middle layer); whether richer monitors or hallucination-inducing input regimes produce a more discriminative signal is the natural next experimental step, and is not contradicted by these results.
% [IMPL: COMPLETE — run_kv_mcts.py post-fix, alpha-sweep on Llama-3.2-1B-Instruct, output at tmp/runs/2026-05-06_b6_post-fix.json. 3B reversibility confirmed via probe_kv_cache_mutability.py 2026-05-05.]
```

**Notes for Session B:**
- The `\label{tab:exp1-results}` is new. It's referenced in the prose. If a similar label already exists in the file (e.g., a placeholder Table 1 or Table 2), check for collisions before applying.
- The IMPL comment is updated from PARTIAL to COMPLETE for the 1B sweep. The 3B sweep is now legitimately deferred (we have 3B reversibility but not 3B alpha-sweep results) — language reflects that.
- The "Interpretation" paragraph is the most important reframing in this piece: it tells the reader the result is *consistent with Bailey 2024's negative result* rather than a positive result extending it. This is the honest framing under Option α+ given the observed data. Do not soften.
- This paragraph removes the previous claim "the Orthogonal Escape hypothesis predicts that (a) and (c) will improve with increasing α while (d) will simultaneously increase, indicating redistribution of deceptive encodings." That claim is now refuted by the data (TDS decreases) — the new prose says so directly.

---

## Application order recommendation

1. **Piece #4-revised first** — replaces the morning's §7.4 with the post-fix-aware version. Removes "early version" placeholder language so other sections can reference §7.4 honestly.
2. **Piece #5** — §3 reframe. Independent of #4 and #6.
3. **Piece #6** — §5 Experiment 1 with results table. References #4-revised for prerequisites context.

After all three pieces land:
- Run `grep -n "REVISION" docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` to verify the new 2026-05-06 markers (`§7.4-post-fix-update`, `§3-reframe-α+`, `Exp1-reframe-with-results-α+`) alongside the morning's six markers.
- Run `grep -n "label{" docs/NeurIPS/...` to verify cross-reference targets (`thm:reversibility`, `prop:memory`, `sec:measurement-prereqs`, `sec:experiments`, `eq:oei`, `eq:mcts_reward`, `eq:hneuron_stress`, `hyp:orthogonal_escape`, `tab:exp1-results`) all resolve.
- Update §4.2 of the action plan with file:line for each landed change. Mark §3/§5 reframe items as done.

---

## What's still outstanding after this batch

- **Memory Complexity Proposition rewrite** — gated on Max's simplified Exp 3. When numbers land, coordinated update across abstract, §1 Contribution 1, §5 Implementation Notes item iv, and §6 Proposition.
- **EMNLP style template download → package switch → page-count check** — Josh + Session B.
- **§2.1 (Alignment Faking and Deceptive Alignment background)** — currently a substantial paragraph that motivates the original framing. Under Option α+ the alignment-faking framing is one application, not the headline. Brief audit needed: does §2.1 still flow correctly given the §3 reframe? Probably yes — §2.1 is about prior work, which is unchanged — but Session B should read it once after applying #5 to confirm no dangling pointers.
- **README rewrite** — now feasible after #5 and #6 land. Targeting Days 8–14.

---

*Session A drafts v2 complete. The fix-landed-with-results-integrated path is now realized in paper text. Either Bailey-consistent negative observation or the natural framing for richer-monitor follow-up — both are honest given the data.*
