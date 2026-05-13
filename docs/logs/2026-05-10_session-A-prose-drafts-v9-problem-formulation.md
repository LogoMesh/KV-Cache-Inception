# Session A Prose Drafts v9 — §3 Problem Formulation — 2026-05-10

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits) — Day 2 of 14-day execution plan
**Status:** Day-1 prose draft for the Option D+ pivot. Replaces §3.1 prose (recast 2026-05-06, marker `§3-reframe-α+`) wholesale and adds a new §3.2 *Dimensional Escape: A Unifying Geometric Pathology* before the existing *Discrete Text-Space Bottleneck* (which becomes §3.3).

## Structural change

Old (Option α+) §3 structure:
- §3.1 Inference-Time Latent-Space Analysis on Frozen Models (with embedded Hypothesis 1)
- §3.2 The Discrete Text-Space Bottleneck

New (Option D+) §3 structure:
- §3.1 Inference-Time Latent-Space Analysis on Frozen Models (tightened opening; the central question is reframed around "do search trajectories navigate the semantic manifold or exploit unconstrained dimensions?")
- §3.2 Dimensional Escape: A Unifying Geometric Pathology (NEW — Definition + Hypothesis 1 reframed as monitor-driven sub-case + new Hypothesis 2 reward-driven sub-case)
- §3.3 The Discrete Text-Space Bottleneck (existing content, renumbered)

The Hypothesis~\ref{hyp:orthogonal_escape} label is preserved bit-identically so all existing cross-references resolve. The new Hypothesis 2 gets a new `hyp:goodhart` label. The new Definition gets a new `def:dim_escape` label.

## Replacement TeX

**TeX target:** §3.1 prose at TeX:131–154 (the `[REVISION | §3-reframe-α+ | 2026-05-06 | ...]` marker, the `\subsection{Inference-Time Latent-Space Analysis on Frozen Models}` opening, and the `\begin{hypothesis}[Orthogonal Escape under inference-time KV-cache perturbation]` block). The existing `\subsection{The Discrete Text-Space Bottleneck}` at TeX:156 onwards is preserved verbatim; only its numbering shifts as a result of the new §3.2 insertion.

**REVISION marker (replacing existing §3-reframe-α+ marker):** `[REVISION | §3-reframe-D+ | 2026-05-10 | Reframed §3 around Dimensional Escape as a unifying geometric pathology (Session A v9 draft, applied 2026-05-10). Old structure (Option α+): §3.1 inference-time latent-space framing with embedded Hypothesis 1 (Orthogonal Escape, monitor-driven). New structure (Option D+): §3.1 tightened framing of the central question; new §3.2 Dimensional Escape with Definition~\ref{def:dim_escape} + Hypothesis 1 (preserved hyp:orthogonal_escape label, retitled as "Monitor-driven sub-case") + new Hypothesis 2 (new hyp:goodhart label, "Reward-driven sub-case"); existing §3.2 Discrete Text-Space Bottleneck renumbers to §3.3 with no content change. Eq.~\ref{eq:alignment_faking} and Eq.~\ref{eq:orthogonal_escape} preserved with labels intact. New Eq.~\ref{eq:goodhart}. The 2026-05-06 §3-reframe-α+ marker is subsumed. Bailey 2024 reference is kept but folded into the Dimensional Escape unifying claim rather than positioned as the standalone framing peer. — pending audit]`

**Replacement body** (replaces TeX:131–154 — i.e., the §3.1 subsection through end of Hypothesis 1):

```latex
\subsection{Inference-Time Latent-Space Analysis on Frozen Models}

Let $f_\theta$ be a language model with parameters $\theta$ held fixed throughout this work, and let $\mathbf{h}_t^{(\ell)} \in \mathbb{R}^d$ denote the hidden state at layer $\ell$ and token position $t$. A linear monitor or surrogate scoring function $\mathbf{P} \in \mathbb{R}^{k \times d}$ of rank $k \ll d$ projects $\mathbf{h}$ onto a low-dimensional subspace---examples include representation-engineering honesty probes~\cite{zou2023repe}, aggregations thereof~\cite{gupta2025rlobfuscation}, and reward functions derived from internal model heuristics such as next-token entropy. The central question of this paper is: when an inference-time search procedure applies controlled perturbations to the model's cached key/value tensors and is guided by such a low-rank projection (whether a corrective monitor, a reward signal, or a representational probe), do the resulting search trajectories navigate the model's semantic manifold, or do they exploit the unconstrained dimensions of the overparameterized latent space?

The latter outcome---which we observe both theoretically and empirically (§\ref{sec:experiments})---is the central pathology this paper characterizes. We give it a unifying name: \emph{Dimensional Escape}.

\subsection{Dimensional Escape: A Unifying Geometric Pathology}\label{sec:dim_escape}

Two seemingly distinct failure modes have been hypothesized for inference-time intervention on frozen language models. The first is \emph{monitor-driven}: when corrective pressure is applied along a monitored subspace, a sufficiently capable model may redistribute its internal representations into the null space of the monitor, preserving behavior while evading detection. This pattern is the subject of Bailey et al.~\cite{bailey2024obfuscated} (gradient-trained input-space variant, with a tentative negative result reported) and motivates the alignment-faking detection literature~\cite{greenblatt2024alignment, hubinger2024sleeper, gupta2025rlobfuscation}. The second mode is \emph{reward-driven}: when a search procedure optimizes a surrogate reward lacking exogenous grounding (for instance, next-token entropy as a proxy for confidence), the optimizer may resolve the reward via geometric paths that bypass the semantic objective entirely. This is Goodhart's Law manifesting in cache geometry: when a measure becomes a target, the optimizer discovers the cheapest computational path to alter the metric, even if that path circumvents the original objective. We argue these are sub-cases of a single underlying pathology.

\begin{definition}[Dimensional Escape]\label{def:dim_escape}
Let $\mathbf{P} \in \mathbb{R}^{k \times d}$ be a monitoring or scoring projection of rank $k \ll d$, and let $\mathcal{L}: \mathbb{R}^d \to \mathbb{R}$ be a localized optimization objective acting on hidden state $\mathbf{h}$. \emph{Dimensional Escape} occurs when the optimization process resolves $\mathcal{L}$ through trajectories whose induced perturbation $\Delta\mathbf{h}$ has substantial component in $\text{Null}(\mathbf{P}_{\text{semantic}})$, where $\mathbf{P}_{\text{semantic}}$ is the (typically unobserved) projection onto the semantically relevant subspace of $\mathbb{R}^d$. The optimizer thus satisfies $\mathcal{L}$ without traversing the manifold of semantically meaningful states.
\end{definition}

The high dimensionality of modern Transformer hidden states ($d \in [2048, 8192]$ for the Llama~3 family) ensures that for any low-rank monitor or scoring function, the unconstrained complement of the constrained subspace is overwhelmingly large. Localized optimization pressure---whether applied as corrective signal during deceptive-alignment scenarios or as reward signal during latent-space search---routes through this unconstrained complement. We instantiate Dimensional Escape under two explicit hypotheses, corresponding to the two failure modes above.

\begin{hypothesis}[Monitor-driven Dimensional Escape: Orthogonal Escape under inference-time KV-cache perturbation]\label{hyp:orthogonal_escape}
When corrective pressure is applied along the monitored subspace $\text{Range}(\mathbf{P}_{\text{monitor}})$, a sufficiently capable model under inference-time steering may redistribute its representation into $\text{Null}(\mathbf{P}_{\text{monitor}})$, preserving behavior while evading detection:
\begin{equation}
\mathbf{h}_t^{(\ell)} \xrightarrow{\text{steering}} \mathbf{h}_t^{(\ell)} + \alpha \cdot \mathbf{d}_{\text{steer}} \implies \text{Proj}_{\text{Null}(\mathbf{P})}(\mathbf{h}_{t+k}^{(\ell)}) \neq \mathbf{0}
\label{eq:orthogonal_escape}
\end{equation}
where $\mathbf{d}_{\text{steer}}$ is the steering direction and $\alpha$ is the intervention magnitude. This hypothesis specializes Definition~\ref{def:dim_escape} to the case where $\mathcal{L}$ is the corrective pressure of an alignment-evaluation procedure and $\mathbf{P}_{\text{monitor}}$ is a linear deception probe. The behavioral consequence---specifically, the alignment-faking specialization with the safe-output set $\mathcal{S}$---is given by Eq.~\ref{eq:alignment_faking}.
\end{hypothesis}

\begin{equation}
\text{Behavioral}(f_\theta, x) \in \mathcal{S} \quad \text{while} \quad \exists\, \mathbf{v} \in \text{Null}(\mathbf{P}_{\text{monitor}}): \langle \mathbf{h}_t^{(\ell)}, \mathbf{v} \rangle \neq 0
\label{eq:alignment_faking}
\end{equation}

\begin{hypothesis}[Reward-driven Dimensional Escape: Goodhart-collapse under surrogate-reward MCTS]\label{hyp:goodhart}
When MCTS optimizes a surrogate reward $\hat{r}: \mathbb{R}^d \to \mathbb{R}$ lacking exogenous grounding---for instance, $\hat{r}(\mathbf{h}) = -H(p(\cdot \mid \mathbf{h}))/\log|V|$ where $H$ is the next-token entropy at the steered cache state and $|V|$ is the vocabulary size---the optimal cache state under $\hat{r}$ generically lies off the semantic manifold:
\begin{equation}
\arg\min_{\mathbf{h}} \hat{r}(\mathbf{h}) \cap \text{SemanticManifold}(x) = \emptyset \quad \text{(generically, in dimension } d \gg \text{rank}(\hat{r})\text{)}
\label{eq:goodhart}
\end{equation}
where $\text{SemanticManifold}(x) \subset \mathbb{R}^d$ is the set of cache states consistent with a correct response to input $x$. This hypothesis specializes Definition~\ref{def:dim_escape} to the case where $\mathcal{L}$ is an MCTS-driven surrogate reward and $\mathbf{P}_{\text{semantic}}$ is the (unobserved) projection onto the answer-relevant subspace. The empirical mechanism by which Eq.~\ref{eq:goodhart} manifests---prior-amplification at small model scale, greedy-collapse at larger scale---is established in §\ref{sec:exp1}.
\end{hypothesis}

Bailey et al.~\cite{bailey2024obfuscated} characterized the monitor-driven case for gradient-trained input-space attacks and reported a tentative negative result: under the regime they study, near-orthogonality and behavior preservation cannot be jointly achieved, since pushing cosine similarity to that of unrelated inputs requires a KL divergence that produces incoherent output. We characterize the reward-driven case empirically in §\ref{sec:exp1}--\ref{sec:exp2} and find that it manifests robustly across model scale and prompt class. The monitor-driven case under inference-time KV-cache perturbation (rather than gradient-trained input attack) is left to Phase B research; the measurement framework introduced in §\ref{sec:measurement-prereqs} remains valid for that setting and supports the necessary empirical characterization.
```

## Notes for Session B

- **Subsection numbering:** the existing `\subsection{The Discrete Text-Space Bottleneck}` block at TeX:156 onwards is preserved verbatim. LaTeX auto-renumbering will shift it from §3.2 to §3.3 without manual edits. Verify by `\ref{}` impact:
  - `\ref{sec:problem}` (if present) — points to §3 as a whole, unaffected.
  - There are no specific `\ref{}` to "§3.2 Discrete Text-Space" in the existing TeX (verified by mental cross-check; Session B should `grep` to confirm before applying).
- **Eq.~\ref{eq:alignment_faking} relocation:** the equation moves from inside §3.1 prose (current location at TeX:140–143) to a standalone display equation immediately after Hypothesis~\ref{hyp:orthogonal_escape}. The `\label{eq:alignment_faking}` is preserved bit-identically. All `\ref{eq:alignment_faking}` cross-references resolve transparently after the move.
- **Hypothesis 2 numbering:** `hyp:goodhart` is a new label. LaTeX will auto-number it as `Hypothesis 2`. Existing `\ref{hyp:orthogonal_escape}` remains `Hypothesis 1`. Any new `\ref{hyp:goodhart}` references in §1 contributions, §5, or §7 should be added when those sections land.
- **Forward references that must resolve:**
  - `\ref{sec:experiments}`, `\ref{sec:exp1}`, `\ref{sec:exp2}` — `sec:experiments` exists; `sec:exp1` and `sec:exp2` get added Day 7 when §5 is rewritten.
  - `\ref{sec:measurement-prereqs}` — exists.
  - `\cite{xie2025entropyloop}` and `\cite{goodhart1984}` — **dropped 2026-05-10 post-verification.** Direct grep of canonical TeX `\begin{thebibliography}{22}` confirmed both citations are absent. Rather than expand the bibliography for an already-tight 22-entry budget, the prose was rewritten to carry the concept without citation: "next-token entropy as a proxy for confidence" (the example is clear without an anchor citation) and "Goodhart's Law manifesting in cache geometry: when a measure becomes a target, the optimizer discovers the cheapest computational path..." (the principle is named-without-cited; the parenthetical operationalizes it). The Goodhart reference is widely enough recognized in mechanistic interpretability venues that it does not require a bibitem.
  - All other `\cite{}` calls (`bailey2024obfuscated`, `greenblatt2024alignment`, `hubinger2024sleeper`, `gupta2025rlobfuscation`, `zou2023repe`, `gao2025hneurons`) are in the bibliography.
- **Length:** the new §3.1 + §3.2 (Definition + 2 Hypotheses) is ~620 words vs. ~340 in the existing §3.1. Net +280 words. This eats into page budget; the Day-8/9 page-budget compression pass (per execution plan) may want to compress the prose between Hypothesis 1 and Hypothesis 2 by ~80 words.
- **Hypothesis 1 substantive change:** the hypothesis body is preserved verbatim except for the title (now reads "Monitor-driven Dimensional Escape: Orthogonal Escape under inference-time KV-cache perturbation"). The pre-§3.2 opening sentence ("Hypothesis 1 specializes Definition...") is the new framing wrapper. This means existing `\ref{hyp:orthogonal_escape}` references all continue to resolve correctly without any text change at the call sites.

## Cross-section impact (downstream)

- **§7 Discussion** (Day 2 v9-discussion.md draft, separate file) needs to reference both Hypothesis 1 and Hypothesis 2 as concrete instantiations of Definition~\ref{def:dim_escape}, and discuss implications of the unified framing for both monitor design AND reward design.
- **§5 Experiments** (Day 7 drafts, separate files) need to label Experiment 1 with `\label{sec:exp1}` and Experiment 2 with `\label{sec:exp2}`, with the Experiment 1 prose explicitly framing Track F's empirical result as the validation of Hypothesis~\ref{hyp:goodhart} at the 1B and 3B scales.
- **§2.1 alignment-faking framing** (`§2.1-soften-α+` 2026-05-07 marker) survives intact under Option D+ — alignment faking is now positioned as one *application* of the unified Dimensional Escape framework. The reference to "informational constraints that manifest as detectable signatures in orthogonal subspaces" at TeX:66 is no longer a load-bearing claim of the paper but remains valid as motivation for §3.2.

---

*Session A v9 problem-formulation draft complete. Apply Day 2 of 14-day execution plan as the third piece in the abstract → contributions → §3 coordinated trio.*
