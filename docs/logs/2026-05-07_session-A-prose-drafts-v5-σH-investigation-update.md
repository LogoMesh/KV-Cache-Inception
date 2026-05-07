# Session A Prose Drafts v5 — σ_H Investigation Update — 2026-05-07

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits)
**Status:** One paragraph patch ready for application following the diagnostic session's σ_H GAP-C1-02 investigation outcome. Triggers an additive update to §7.4 paragraph 3.

**Why this exists:** The diagnostic session ran the optional GAP-C1-02 investigation today (file `docs/logs/2026-05-07_optional-σH-investigation-prompt.md`). Path A confirmed GAP-C1-02 is not applied in the current code. Path B applied the fix tentatively (per-neuron baseline subtraction + ReLU before averaging, exactly per Max's GAP-C1-02 recipe) and ran the three-prompt verification: σ_H remained identically 0.000000 across all 17 layers in all three regimes (hallucination, coercion-class, neutral). ρ_R / OEI / TDS values bit-identical to yesterday's verification, confirming the tentative fix touched only the σ_H pipeline. The bug is therefore downstream of per-neuron normalization.

This narrows the future-work scope substantively: the obvious-looking fix has been tested and ruled out. The most plausible remaining cause is a position/context discrepancy between calibration-time and inference-time hidden states (calibration likely uses last-input-token activations from a prompt-only forward pass; MCTS evaluation reads last-token activations from a prompt+generation forward pass with a steered KV cache).

The §7.4 paragraph 3 from yesterday's drafts v3 patch already says "second measurement-pipeline issue, distinct from and independent of the steering-projection issue resolved above." Today's investigation adds: we tested per-neuron normalization and ruled it out as the cause, and we identify position/context discrepancy as the remaining likely candidate. Net stronger paragraph; ~60 words longer.

---

## Piece #8 — §7.4 paragraph 3 additive update

**TeX target:** §7.4 paragraph 3 (the σ_H paragraph). Current state has marker `[REVISION | §7.4-σH-verification-update | 2026-05-06]` at TeX:524 from yesterday's drafts v3 patch.

**REVISION marker:** `[REVISION | §7.4-σH-investigation-update | 2026-05-07]` (added alongside the prior two markers as a layered audit trail; do not remove `§7-measurement-prereqs` or `§7.4-post-fix-update` or `§7.4-σH-verification-update`).

**Replacement text** (full paragraph; opening sentences and final reassurance preserved verbatim from yesterday's version; one new sentence pair inserted between the verification result and the future-work conclusion):

```latex
The bottom-up H-Neuron stress channel ($\sigma_H$) is reported as zero across all $\alpha$ in our reference experiments. This is independent of the steering pipeline (which demonstrably reaches the cache, per the per-$\alpha$ variation in $\rho_R$, TDS, and OEI in §\ref{sec:experiments}). To distinguish whether the constancy reflects a calibration-regime mismatch~\cite{gao2025hneurons} or a deeper measurement issue, we ran a controlled three-prompt comparison at fixed $\alpha = 1.0$, varying only the test prompt across hallucination-style, coercion-style, and neutral regimes --- with the coercion-style prompt drawn directly from the same class used as the H-Neuron monitor's contrastive calibration set. $\sigma_H$ remained identically zero across all 17 layers in all three regimes, while $\rho_R$ varied substantially (deepest-layer values of $-14.66$, $-7.15$, and $-2.93$ across the three regimes respectively). To narrow the cause, we additionally tested the per-neuron baseline-subtraction-and-ReLU formulation specified by Eq.~\ref{eq:hneuron_stress} (which our reference implementation aggregates rather than applying per-neuron); under this formulation $\sigma_H$ remained identically zero across the same three regimes, indicating the cause is downstream of per-neuron normalization. The remaining most plausible cause is a position/context discrepancy between calibration-time and inference-time hidden states: the H-Neuron monitor is calibrated on last-input-token activations from prompt-only forward passes, while the MCTS evaluation reads last-token activations from prompt-plus-generation forward passes under a steered cache; these are different positions in the sequence with different attention contexts, and the H-Neurons selected by the former calibration may not fire under the latter regime even when input semantics are similar. We identify this as a second measurement-pipeline issue, and validation of the bottom-up channel as a precondition for full telemetry-matrix evaluation; resolution is left as future work. We note that Theorem~\ref{thm:reversibility} and Proposition~\ref{prop:memory} are unaffected by these measurement choices: both are intervention-agnostic.
```

**Notes for Session B:**

- Diff vs. yesterday's version: two new sentences inserted between the deepest-layer $\rho_R$ values sentence and the "We identify this as a second measurement-pipeline issue..." sentence. The rest of the paragraph is identical to yesterday's drafts v3 patch.
- Net +60 words.
- The "position/context discrepancy" hypothesis is specific enough to be useful for future work but doesn't commit us to a particular fix recipe. Reviewers familiar with the H-Neuron monitor design will recognize the issue as a real one (calibration vs. inference activation alignment).
- Layered REVISION markers from yesterday + today: `§7-measurement-prereqs` (2026-05-06 morning, original-insertion record), `§7.4-post-fix-update` (2026-05-06 PM, post-fix-aware rewrite), `§7.4-σH-verification-update` (2026-05-06 EOD, three-prompt verification result), `§7.4-σH-investigation-update` (this patch, GAP-C1-02 investigation result). All four should remain in the file as a chronological audit trail.

---

## What's outstanding after this patch

- §7.4 σ_H paragraph: closed for May 25 (further σ_H investigation is post-EMNLP)
- §4.5 Max-lane row for the dense-scoring-path issue: update to confirm "GAP-C1-02 tentatively tested 2026-05-07; σ_H still zero; cause downstream of per-neuron normalization"
- §4.6 row 169: previously closed yesterday; today's investigation produces an *additive* patch (piece #8), not a re-open. Drafted in this v5 file; awaiting Session B application.

After this patch, the paper's σ_H story is genuinely complete for May 25: identified, characterized, two candidate fixes tested (one applied as the W_K projection that fixed GAP-C2-06; one tentatively applied as the per-neuron ReLU that didn't fix σ_H), remaining cause specified as position/context discrepancy, validation deferred to future work. That is a much stronger limitations paragraph than "we don't know why σ_H is zero."

---

*Session A drafts v5 complete. The optional σ_H investigation was high-leverage exactly as Josh hoped: it converted "we don't know what's wrong" into "we tested the obvious fix and it doesn't help; the cause is in this specific class." The paper is materially stronger as a result.*
