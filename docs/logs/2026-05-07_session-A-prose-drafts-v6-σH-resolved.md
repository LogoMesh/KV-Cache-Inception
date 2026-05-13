# Session A Prose Drafts v6 — σ_H Resolved Update — 2026-05-07

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits)
**Status:** Four coordinated updates following the post-merge verification. Reflects the empirical finding that Max's per-layer H-Neuron calibration (merged in commit `ca3ec8e` from `max-toscano@976fce5`) produces non-zero σ_H that varies monotonically with α, resolving the second measurement-pipeline issue documented in §7.4 paragraph 3.

**Trigger:** Post-merge B6 verification 2026-05-07 (`docs/logs/2026-05-07_post-merge-verification-report.md`) confirmed:
- σ_H_mean: `{0.228, 0.237, 0.247, 0.267, 0.330}` across α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}; 14–15/17 layers firing
- ρ_R and OEI bit-identical to the pre-merge post-W_K-fix run
- TDS shifts ~7–13% relative downstream of σ_H now being non-zero
- Theorem 1 holds end-to-end (residual_norm = 0)

**Narrative shift in the paper**: from "two measurement-pipeline issues identified, one resolved, one specified as future work" to **"two measurement-pipeline issues identified, both resolved, here are the verified results."**

**Pieces in this file:**
1. **Piece #9-table** — replace σ_H + TDS rows in Table 1
2. **Piece #9-results-prose** — replace TDS + σ_H sentences in §5 Experiment 1 Results paragraph
3. **Piece #9-impl-notes-cleanup** — remove §5 Implementation Notes item ii (layer aggregation) since it's now resolved
4. **Piece #9-§7.4-paragraph-3-rewrite** — substantive rewrite of the σ_H paragraph in §7.4

**Application order:** #1 → #2 → #3 → #4. Pieces are independent textually but logically grouped.

**Cross-reference impact:** none. All existing labels (`tab:exp1-results`, `sec:measurement-prereqs`, `eq:hneuron_stress`, etc.) preserved.

---

## 1. Piece #9-table — Table 1 σ_H + TDS row replacement

**TeX target:** `\label{tab:exp1-results}` table at TeX:344–358 (per current line numbers; the σ_H row is at TeX:352, TDS at TeX:353; caption at TeX:356).

**REVISION marker:** `[REVISION | Exp1-table-σH-TDS-update | 2026-05-07]`

**Replacement** (replaces the σ_H row, TDS row, and the caption only; OEI and ρ_R rows preserved):

```latex
$\sigma_H$ & 0.228 & 0.237 & 0.247 & 0.267 & 0.330 \\
TDS & 0.084 & 0.080 & 0.076 & 0.068 & 0.040 \\
\bottomrule
\end{tabular}
\caption{Per-$\alpha$ mean telemetry under inference-time KV-cache steering on Llama~3.2-1B-Instruct, with both measurement-pipeline issues resolved (steering projected through layer-specific $\mathbf{W}_K^{(\ell)}$ to per-head cache space; H-Neuron calibration applied per-layer per Eq.~\ref{eq:hneuron_stress}). All four channels respond to steering: $\rho_R$ varies substantially, $\sigma_H$ rises monotonically with 14--15 of 17 layers firing, OEI shows the small monotonic variation expected in the high-dimensional saturation regime (§\ref{sec:measurement-prereqs}), and the cross-channel divergence TDS \emph{decreases} monotonically---the opposite of the signature predicted by the Orthogonal Escape Hypothesis (Hypothesis~\ref{hyp:orthogonal_escape}). Reversibility holds end-to-end (residual\_norm $= 0$).}
\label{tab:exp1-results}
```

**Notes for Session B:**
- The σ_H row is the line currently reading `$\sigma_H$ & 0.000 & 0.000 & 0.000 & 0.000 & 0.000 \\`. Replace in place.
- The TDS row is the line currently reading `TDS & 0.094 & 0.090 & 0.084 & 0.075 & 0.046 \\`. Replace in place.
- The OEI row (currently `OEI & 0.991 & 0.991 & 0.992 & 0.992 & 0.995 \\`) and ρ_R row (currently `$\rho_R$ & $-1.306$ & $-1.258$ & $-1.198$ & $-1.080$ & $-0.533$ \\`) are unchanged — verified bit-identical post-merge.
- Caption is rewritten substantively; replace in place.

---

## 2. Piece #9-results-prose — §5 Experiment 1 Results paragraph σ_H + TDS sentences

**TeX target:** §5 Experiment 1 Results paragraph at TeX:341 (the long results paragraph following Table~\ref{tab:exp1-results}).

**REVISION marker:** `[REVISION | Exp1-results-σH-resolved | 2026-05-07]` (layered alongside `Exp1-reframe-with-results-α+` 2026-05-06 and `Exp1-σH-verification-update` 2026-05-06).

**Replacement** (replaces the existing Results paragraph; OEI and ρ_R sentences preserved verbatim; TDS and σ_H sentences updated):

```latex
\textbf{Results.} Per-$\alpha$ mean values across the sweep are reported in Table~\ref{tab:exp1-results}. OEI varies monotonically with $\alpha$, from $0.991$ at $\alpha = 0.1$ to $0.995$ at $\alpha = 5.0$---a spread of $0.004$ in a metric bounded in $[0, 1]$. The high baseline value is geometrically expected (§\ref{sec:measurement-prereqs}). The RepE honesty channel $\rho_R$ varies more substantially, from $-1.306$ at $\alpha = 0.1$ to $-0.533$ at $\alpha = 5.0$, confirming that the linear monitor channel registers the in-direction component of the steered cache state. The H-Neuron stress channel $\sigma_H$ rises monotonically with $\alpha$ (from $0.228$ to $0.330$), with 14--15 of 17 layers firing across the sweep, confirming that the bottom-up channel responds to the steering perturbation under per-layer H-Neuron calibration as Eq.~\ref{eq:hneuron_stress} specifies. Cross-channel divergence TDS \emph{decreases} monotonically with $\alpha$ (from $0.084$ to $0.040$), indicating that the H-Neuron and RepE channels agree more strongly under stronger steering---the opposite of the signature predicted by the Orthogonal Escape Hypothesis (Hypothesis~\ref{hyp:orthogonal_escape}).
```

**Notes for Session B:**
- Replace the entire Results paragraph (between `\textbf{Results.}` and the `\begin{table}[h]` that follows it). Length is comparable; net ~+15 words.
- Specifically, the σ_H sentence at the *end* of the previous version ("...$\sigma_H$ reports zero across all $\alpha$; controlled verification...") is removed and replaced by the new σ_H sentence in the *middle* of the paragraph (where it sits naturally between the ρ_R reporting and the TDS reporting, in the order σ_H → ρ_R → OEI → TDS or similar).
- The TDS sentence numbers `(from $0.094$ to $0.046$)` updates to `(from $0.084$ to $0.040$)`.
- The Bailey-consistency framing at the end ("anti-OE signature") is preserved — TDS still decreases monotonically with α, just at slightly smaller magnitudes.

**Interpretation paragraph (the paragraph after Results) is unaffected** — the conclusion "results are consistent with the negative-result regime Bailey et al. report" still holds. Both telemetry channels now respond to the steering, but cross-channel divergence still *decreases* under stronger steering, which is anti-OE. Verify Session B doesn't accidentally edit it.

---

## 3. Piece #9-impl-notes-cleanup — §5 Implementation Notes item ii removal

**TeX target:** §5 Implementation Notes paragraph (added 2026-05-05 with marker `§5-impl-notes`). Locate item ii (about layer aggregation / "all L vs middle layers").

**REVISION marker:** `[REVISION | §5-impl-notes-item-ii-resolved | 2026-05-07]`

**Recommended action:** Remove item ii entirely (since per-layer calibration is now applied) **OR** rephrase to reflect resolution. The minimal-change approach is to rephrase:

**Replacement for item ii** (in-place, preserving the surrounding `\item` structure):

```latex
\item \emph{Resolved 2026-05-07:} Layer-aggregate H-Neuron calibration was a Phase~2 approximation; per-layer calibration as Eq.~\ref{eq:hneuron_stress} specifies is now applied. The post-merge $\sigma_H$ measurements in §\ref{sec:experiments} reflect the resolved implementation.
```

OR — if Session B prefers the cleaner approach:

**Removal**: simply delete item ii entirely from the `\begin{itemize}` block. The remaining items (i, iii, iv, v) renumber organically and the paragraph stays coherent.

**Notes for Session B:**
- Either approach works. The removal is cleaner; the rephrasing makes the resolution visible to a reviewer who reads the §5 Implementation Notes from start to finish.
- I lean toward **removal** for page-budget reasons and because the paragraph at this point is genuinely about *unresolved* Phase 2 caveats; resolved items don't need to be listed.
- Items still in the paragraph after removal: (i) RepE PCA vs. mean-diff, (iii) per-layer steering vectors, (iv) sparse vs. full-shape accumulators, (v) Theorem 1 unaffected. Renumber if Session B uses explicit numbering; the `\item` structure just lists them.

---

## 4. Piece #9-§7.4-paragraph-3-rewrite — substantive rewrite of the σ_H paragraph

**TeX target:** §7.4 paragraph 3 (currently at TeX:526 area; carries four layered REVISION markers from 2026-05-06 + 2026-05-07).

**REVISION marker:** `[REVISION | §7.4-σH-resolved | 2026-05-07]` (added alongside the four prior σ_H markers as a chronological audit trail; do not remove the older ones).

**Replacement text** (full paragraph; first two §7.4 paragraphs unchanged; last paragraph "We note that Theorem~..." preserved verbatim):

```latex
The bottom-up H-Neuron stress channel ($\sigma_H$) initially reported zero across all $\alpha$ in our reference experiments. To distinguish whether the constancy reflected a calibration-regime mismatch~\cite{gao2025hneurons} or a deeper measurement issue, we ran a controlled three-prompt comparison at fixed $\alpha = 1.0$, varying only the test prompt across hallucination-style, coercion-style, and neutral regimes (with the coercion-style prompt drawn directly from the same class used as the H-Neuron monitor's contrastive calibration set). $\sigma_H$ remained identically zero across all 17 layers in all three regimes. We additionally tested the per-neuron baseline-subtraction-and-ReLU formulation specified by Eq.~\ref{eq:hneuron_stress}; under this formulation $\sigma_H$ likewise remained zero, ruling out per-neuron normalization as the cause. The constancy resolved when we adopted per-layer H-Neuron calibration: in our initial implementation a single set of top-$K$ H-Neuron indices was identified at the deepest layer and applied uniformly across the layer stack as a Phase~2 approximation, while Eq.~\ref{eq:hneuron_stress} as written specifies per-layer indices and per-layer baselines. With per-layer calibration in place, $\sigma_H$ rises monotonically with $\alpha$ across the same alpha sweep (Table~\ref{tab:exp1-results}), with 14--15 of 17 layers firing. Both measurement-pipeline issues identified during the preparation of this paper---the steering-projection issue addressed by the $\mathbf{W}_K$-projection fix discussed above and the bottom-up calibration issue addressed by per-layer H-Neuron identification---are therefore resolved in the implementation reported here.
```

**Notes for Session B:**
- Length is comparable to the existing paragraph (net ~−20 words; the paragraph gets tighter because it moves from "investigation arc with future-work conclusion" to "investigation arc with resolved conclusion").
- The investigation arc (three-prompt verification + per-neuron-ReLU test) is **preserved** because it's valid methodology regardless of the eventual outcome — it's how we narrowed the bug location.
- The conclusion shifts from "second measurement-pipeline issue, distinct from the steering-projection issue resolved above; resolution is left as future work" to "Both measurement-pipeline issues...are therefore resolved in the implementation reported here."
- The phrase "in our initial implementation a single set of top-$K$ H-Neuron indices was identified at the deepest layer and applied uniformly across the layer stack as a Phase~2 approximation" is the honest framing of what happened. We didn't write the paper wrong — Eq.~\ref{eq:hneuron_stress} was correct; the *implementation* used a Phase 2 approximation that didn't match what the equation specifies. That's the right way to describe it.
- The position/context-discrepancy hypothesis from the previous version is **dropped entirely** — it was a wrong hypothesis (a useful methodological hypothesis, but empirically refuted). The paper text shouldn't preserve it; it would be confusing to a reviewer to mention a hypothesis that turned out to be wrong without further context.
- The fourth paragraph of §7.4 ("We note that Theorem~\ref{thm:reversibility}...") stays exactly as it is — unchanged.
- Layered REVISION markers preserved (do not remove): `§7-measurement-prereqs` (2026-05-06 AM, original-insertion), `§7.4-post-fix-update` (2026-05-06 PM, post-fix-aware rewrite), `§7.4-σH-verification-update` (2026-05-06 EOD, three-prompt result), `§7.4-σH-investigation-update` (2026-05-07 PM, GAP-C1-02 ruled out), `§7.4-σH-resolved` (this patch). Five markers total, telling the resolution arc end-to-end.

---

## What's outstanding after this batch

- **§4.6 row 169**: σ_H tracker. After this patch lands, mark **closed (resolved)** rather than "additive update."
- **§4.5 GAP-C1-XX dense-scoring-path silent failure**: mark **resolved** (per-layer calibration was the actual fix).
- **§4.5 GAP-C1-01 per-layer H-Neuron**: mark **resolved** (Max's commit `976fce5` integrated via `ca3ec8e`).
- **Memory `project_context.md`**: remove the σ_H bullet from "What's identified, not resolved"; leave only Memory Complexity Proposition pending Max's simplified Exp 3.
- **README.md "What's identified as known limitations"**: 2 issues → 1 issue (Memory Complexity placeholder). Update accordingly when piece #9 lands.

The paper as it will read after this batch lands: validated theorem (Theorem 1, 1B + 3B); working algorithm (reversible KV-cache MCTS via FP32 accumulator); telemetry-matrix infrastructure (per-layer per Eq. 3 + RepE per Eq. 4); OEI as targeted measurement complementing Bailey 2024; **first controlled measurement of OEI in the inference-time, frozen-weight regime, with both telemetry channels responding to the steering and observed cross-channel divergence decreasing under stronger steering — anti-OE signature, consistent with Bailey 2024's negative-result regime.** Two measurement-pipeline issues identified, characterized, and resolved during the preparation of this work.

That's a paper.

---

*Session A drafts v6 complete. The post-merge B6 verification turned out to be the high-leverage call exactly as expected — Max's per-layer H-Neuron calibration was the fix that closed the σ_H story, and the paper now reflects "both issues resolved" rather than "one resolved, one identified-but-deferred." Apply at convenience.*
