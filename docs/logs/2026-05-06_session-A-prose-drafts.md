# Session A Prose Drafts for Option α+ Recast — 2026-05-06

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits)
**Status:** Drafts ready for application. All four pieces below are designed to be applied as in-place TeX replacements with REVISION markers tagged `[REVISION | <slug> | 2026-05-06]`.

**What's in this file:**
1. Replacement abstract (replaces TeX:50–52 in full)
2. Replacement §1 Contributions list (replaces TeX:79–92, the three `\item` entries)
3. Bailey 2024 substantive engagement paragraph (insert into §2.2 after the existing activation-steering paragraph at TeX ~105)
4. New §7 honest-finding subsection (insert into §7 Discussion before §7.4 or wherever Session B determines best fit)

**Application protocol:** Each piece below is meant to land cleanly with no further drafting needed. If Session B finds that any piece doesn't fit the surrounding TeX structurally, flag it back to Session A via §7 of the action plan rather than smoothing it over. Drafts should not be smoothed without consultation — the structural choices in each are deliberate.

**Constraints honored across all four drafts:**
- No promises about empirical OEI / σ_H / ρ_R results we cannot deliver. Memory and reversibility claims are validated; OEI claims are framed as measurement infrastructure with explicit prerequisites.
- No "alignment-faking detection" headline. The framing is algorithmic search infrastructure with an interpretability application.
- Bailey 2024 engaged as peer measurement, not as competitor whose novelty claim we're contesting.
- Threat model (Gemini #1) addressed with monitor-design verbs and explicit white-box-access caveats where relevant.
- The §7 honest-finding paragraph documents GAP-C2-06 specifically without burying it.

---

## 1. Replacement abstract

**TeX target:** lines 50–52 (the `\begin{abstract} ... \end{abstract}` block in full).

**REVISION marker:** `[REVISION | abstract-recast-α+ | 2026-05-06]`

**Replacement text:**

```latex
\begin{abstract}
Latent-space analysis of frozen large language models---including activation steering, representation engineering, and adversarial probing of internal monitors---increasingly requires search infrastructure that scales with the depth of intervention rather than with the branching factor of the search. We introduce \textbf{Reversible Monte Carlo Tree Search in KV-cache latent space}, an inference-time algorithm that mutates a frozen Transformer's cached key/value tensors in place via FP32 accumulators and rolls back exact baseline state at each tree expansion. We prove that the FP32 accumulator guarantees zero-drift reversibility (Theorem~\ref{thm:reversibility}; empirically validated to machine precision over 200 apply--revert cycles on Llama~3.2-1B and Llama~3.2-3B), and we analyze a memory bound that is independent of branching factor---a constant-factor multiple of the base KV cache rather than the exponential blowup of parallel-cache MCTS. To make the algorithm useful for representational analysis, we introduce a \textbf{per-layer telemetry matrix} that fuses bottom-up hallucination-associated-neuron activation tracking~\cite{gao2025hneurons} with top-down representation-engineering projections~\cite{zou2023repe}, providing a continuous, differentiable signal at each generation step. As an application, we propose the \textbf{Orthogonal Escape Index (OEI)}---a targeted geometric measurement of the fraction of induced activation change that lies in the null space of a linear monitor---as a measurement framework complementing Bailey et al.'s~\cite{bailey2024obfuscated} cosine-to-baseline measurement of obfuscated activations under gradient-trained input attacks. We discuss measurement-pipeline prerequisites identified in our reference implementation and outline the empirical characterization required for the inference-time, frozen-weight regime. All artifacts are released with a Croissant 1.1 metadata package for reproducibility.
\end{abstract}
```

**Notes for Session B:**
- The `\ref{thm:reversibility}` assumes the reversibility theorem is already labeled. If the existing TeX uses a different label, substitute. Grep the file for `\begin{theorem}` to find the label.
- The "Llama 3.2-3B" claim corresponds to the 2026-05-05 3B probe pass (`gate_passed: true`, `max_logit_delta_after_revert: 0.0`). Genuine empirical evidence.
- "Constant-factor multiple of the base KV cache" is deliberately unquantified — Max's simplified Exp 3 measurement will set the constant. When that lands, the abstract can be tightened to e.g., "approximately 2–3× the base KV cache."
- "All artifacts" includes: the Croissant metadata file (already mentioned in §7 Ethical Considerations) and the Croissant runtime-evidence package (added 2026-04-25 in §7 Future Work). Both already exist in the paper.

---

## 2. Replacement §1 Contributions list

**TeX target:** lines 79–92 (the three `\item` entries inside `\begin{enumerate}`).

**REVISION marker:** `[REVISION | contributions-recast-α+ | 2026-05-06]`

**Replacement text:**

```latex
\begin{enumerate}[leftmargin=*]
    \item \textbf{Reversible Monte Carlo Tree Search in KV-cache latent space.} We introduce an inference-time algorithm that mutates a frozen Transformer's cached key/value tensors via in-place arithmetic ($\mathbf{K}_t \leftarrow \mathbf{K}_t + \alpha \cdot \mathbf{d}_K^{(\ell)}$) and inverts each mutation exactly through an FP32 accumulator. We prove (Theorem~\ref{thm:reversibility}) that the accumulator guarantees zero-drift reversibility independent of search depth and validate this empirically on Llama~3.2-1B-Instruct and Llama~3.2-3B-Instruct (drift $= 0$ to machine precision over 200 cycles; Llama~3.2-3B feasibility confirmed on a single 12\,GB consumer GPU). The memory complexity of the algorithm is independent of branching factor---a constant-factor multiple of the base KV cache (Proposition~\ref{prop:memory})---in contrast to the $O(b^d \cdot M_{\text{KV}})$ overhead of standard parallel-cache MCTS. The algorithm is gradient-free and operates on a frozen model at inference time, distinguishing it from gradient-based input-space attacks~\cite{zou2023gcg, bailey2024obfuscated} and from training-time monitor-evasion methods~\cite{gupta2025rlobfuscation}.

    \item \textbf{Per-layer telemetry matrix for latent-space analysis.} We define a $2 \times L$ matrix $\mathbf{T}_t$ that fuses bottom-up hallucination-associated-neuron activation~\cite{gao2025hneurons} with top-down representation-engineering projections~\cite{zou2023repe}. The matrix provides a continuous, differentiable signal at each generation step and supports four diagnostic regime classifications (creative synthesis, contextual fabrication, alignment faking, orthogonal escape) keyed to joint patterns across the two channels. The telemetry matrix serves dual roles: as a per-step monitoring signal during search, and as the third reward component in the MCTS node valuation function (Eq.~\ref{eq:mcts_reward}).

    \item \textbf{The Orthogonal Escape Index as a targeted measurement of null-space activation redistribution.} We propose OEI---the fraction of induced activation change $\Delta\mathbf{h}$ that lies in the null space of a linear monitor $\mathbf{P}$---as a geometric quantity for studying inference-time activation redistribution under steering interventions. OEI complements Bailey et al.'s~\cite{bailey2024obfuscated} cosine-to-baseline measurement: where Bailey asks whether gradient-trained input perturbations can achieve global activation orthogonality (reporting a tentative negative result), OEI asks how much of any induced perturbation lands in directions invisible to a specific monitor. We discuss measurement-pipeline prerequisites identified in our reference implementation (\S\ref{sec:measurement-prereqs}) and identify the steps required for empirical characterization in the inference-time, frozen-weight regime.
\end{enumerate}
```

**Notes for Session B:**
- The `\ref{thm:reversibility}` and `\ref{prop:memory}` and `\ref{eq:mcts_reward}` are existing labels (Theorem 1, Memory Proposition, Eq. 8). Verify via grep before applying.
- The `\S\ref{sec:measurement-prereqs}` is a forward reference to the new §7 honest-finding subsection (piece #4 below). Session B will need to add the `\label{sec:measurement-prereqs}` to that subsection when applying piece #4.
- The `\cite{zou2023gcg, bailey2024obfuscated, gupta2025rlobfuscation}` citations all already exist in the bibliography. The new positioning of Bailey alongside GCG (gradient-based input attacks) and RL-Obfuscation (training-time evasion) frames our contribution as a third position in the access/cost frontier.
- Total length is comparable to current Contributions section. Net page-count delta ≈ 0.

---

## 3. Bailey 2024 substantive engagement paragraph (§2.2)

**TeX target:** §2.2 Representation Engineering and Activation Steering. Insert after the existing paragraph that ends at TeX ~107 (the paragraph ending with "...the geometric motivation for this redistribution is formalized in Section~3 (Hypothesis~\ref{hyp:orthogonal_escape})").

**REVISION marker:** `[REVISION | §2.2-Bailey-engagement | 2026-05-06]`

**New paragraph (inserts as standalone, no replacement):**

```latex
Most directly relevant to our framing of OEI, Bailey et al.\ (2024)~\cite{bailey2024obfuscated} systematically study whether adversarial perturbations to model inputs can achieve activation states that are simultaneously behavior-preserving and orthogonal to baseline activations. Their gradient-trained universal embedding suffixes optimize a joint objective: minimize the KL divergence between perturbed and baseline output distributions while minimizing the cosine similarity between perturbed and baseline activations. They report a tentative negative result for the regime of trained input-space attacks against a frozen model: near-orthogonality and behavior preservation cannot be jointly achieved, since pushing cosine similarity to the level of unrelated inputs requires a KL divergence that produces incoherent output. Our work is complementary along three axes. First, the intervention is applied to cached key/value tensors at inference time rather than to input embeddings during gradient training, so neither the model weights nor the embedding layer is modified. Second, the search optimizes a behavioral telemetry signal (Eq.~\ref{eq:mcts_reward}) rather than directly minimizing similarity to baseline, so any redistribution we observe is incidental rather than optimization-driven. Third, OEI measures a more targeted geometric quantity---the fraction of induced perturbation invisible to a specific linear monitor---rather than global activation orthogonality. Whether the inference-time, frozen-weight regime exhibits the same constraint Bailey identifies remains an open question, and is one motivation for the measurement framework developed in this paper.
```

**Notes for Session B:**
- This paragraph is designed to drop in cleanly between the existing "RL-Obfuscation conceptual motivation" paragraph and the §2.3 H-Neurons subsection. If §2.2 has additional paragraphs after the RL-Obfuscation paragraph, place this Bailey paragraph immediately before §2.3.
- The forward reference to `Eq.~\ref{eq:mcts_reward}` is intentional — it's the connection back to "behavioral telemetry signal" (i.e., the MCTS reward in Eq. 8).
- After applying, please update §2.2 to ensure the §3 forward pointer (about Hypothesis 1) still resolves and is positioned sensibly (it should remain at the end of the RL-Obfuscation paragraph, *before* the new Bailey paragraph).

---

## 4. §7 honest-finding subsection (Empirical Measurement Prerequisites)

**TeX target:** §7 Discussion. Insert as a new subsection between the existing §7 Limitations and §7 Future Work (or between §7 Future Work and §7 Ethical Considerations — whichever placement Session B determines reads better in the current §7 ordering).

**REVISION marker:** `[REVISION | §7-measurement-prereqs | 2026-05-06]`

**New subsection (inserts as standalone, no replacement):**

```latex
\subsection{Empirical Measurement Prerequisites for OEI}
\label{sec:measurement-prereqs}

The Orthogonal Escape Index defined in Eq.~\ref{eq:oei} is well-defined geometrically given any non-trivial steering operation, but the choice of $\mathbf{d}_K^{(\ell)}$ in $\mathbf{K}_t \leftarrow \mathbf{K}_t + \alpha \cdot \mathbf{d}_K^{(\ell)}$ is consequential and warrants explicit discussion. Representation-engineering probes are calibrated against residual-stream activations of dimension $d_{\text{model}}$, while the KV cache stores per-head representations of dimension $\text{kv\_heads} \times d_{\text{head}}$. For Llama~3.2 architectures with grouped-query attention, $d_{\text{model}} = 2048$ but $d_{\text{head}} = 64$; a direct application of a $d_{\text{model}}$-dimensional probe direction to a per-head cache is shape-incompatible and requires an explicit projection between probe space and cache space. In our reference implementation, we identified that an early version of the broadcast layer between probe space and cache space silently substituted a zero delta when shapes were incompatible, producing forward passes that were bit-identical across $\alpha$ values and rendering OEI measurement uninformative---a measurement-pipeline failure mode we surface here as part of the contribution.

Resolving this requires one of three approaches. The most lightweight is to project the probe direction through the layer-specific key projection weights to land in per-head cache space: $\mathbf{d}_K^{(\ell)} = \mathbf{W}_K^{(\ell)} \mathbf{d}_{\text{probe}}^{(\ell)}$, reshaped to per-head form. This is interpretable as ``the cache state that would have resulted from the corresponding residual-stream perturbation,'' modulo the position-dependent rotary embedding applied at the attention computation. A second approach is to intervene at the residual stream prior to the K/V projection and re-run the layer's forward computation, which is semantically clean but requires modifying the model's forward graph. A third approach calibrates probes directly in per-head cache space against contrastive cache states, which avoids the residual-stream-to-cache projection question entirely but requires per-layer cache-space contrastive datasets that, to our knowledge, do not exist in standard form for this architecture. The choice among these approaches is itself an empirical question, and characterizing OEI under each is the natural next step for studying inference-time activation redistribution in the frozen-weight regime.

We note that the reversibility guarantee of Theorem~\ref{thm:reversibility} and the memory bound of Proposition~\ref{prop:memory} are unaffected by this choice. Both the algorithm and its theoretical properties are intervention-agnostic: any $\mathbf{d}_K^{(\ell)}$ of the appropriate shape is reversibly applied and bounded in memory. The measurement question concerns the interpretation of the perturbation, not the search infrastructure that delivers it.
```

**Notes for Session B:**
- The label `\label{sec:measurement-prereqs}` matches the forward reference in piece #2 (Contribution 3). Both must land for the cross-reference to resolve.
- This subsection is deliberately ~350 words. If page budget is tight after the EMNLP style switch, the third paragraph (about Theorem 1 + Proposition unaffected) can be tightened to a single sentence — but I would keep the full version unless explicitly forced to trim. The third paragraph is what reassures a reviewer that the measurement-pipeline issue does not contaminate the algorithmic contribution.
- The phrase "an early version of the broadcast layer" is intentionally past tense and non-specific. It's honest (the bug exists in the current code) without inviting reviewers to grep for the exact line. If Josh's mechanical fix attempt today succeeds, this phrasing accommodates "an early version had the bug; the current version projects through $\mathbf{W}_K$." If the fix doesn't land, the phrasing accommodates "an early version had the bug, which we identified during preparation of this submission and which we leave as a precondition for empirical characterization." Either outcome works without rewriting this paragraph.
- The `\ref{thm:reversibility}`, `\ref{prop:memory}`, and `\ref{eq:oei}` are existing labels.

---

## Application order recommendation

If Session B applies these in sequence:

1. **Piece #4 first** (§7 honest-finding subsection). It lands the `\label{sec:measurement-prereqs}` that pieces #1 and #2 reference.
2. **Piece #2** (§1 Contributions list). The forward references resolve once piece #4 is in place.
3. **Piece #3** (§2.2 Bailey paragraph). Standalone; doesn't depend on #4.
4. **Piece #1** (abstract). Last because it's the most visible and any structural changes elsewhere should settle first.

After all four pieces land, run `grep -n "REVISION" docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` to verify all six 2026-05-06 markers (`abstract-recast-α+`, `contributions-recast-α+`, `§2.2-Bailey-engagement`, `§7-measurement-prereqs`, plus any earlier markers from today) are present and resolvable.

---

## What's still queued

After these four pieces land, the following remain (all tracked in action plan §4.2):

- Memory Complexity Proposition rewrite (gated on Max's simplified Exp 3 — Josh runs measurement)
- §1 Contribution 2 reference to "Theorem 1" — already done by Session B 2026-05-05; verify the new §1 Contributions piece #2 here doesn't duplicate
- EMNLP style template download → package switch → page-count check (Josh + Session B)
- Bug-fix attempt outcome → optional supplementary content (Josh + Session A drafts if fix succeeds)
- pre-print v0 to arXiv (gated on Tianyu sign-off + above)

---

*Session A drafts complete. Session B can begin application immediately. Post questions or structural concerns to action plan §7 (Inter-session messages) rather than smoothing in place.*
