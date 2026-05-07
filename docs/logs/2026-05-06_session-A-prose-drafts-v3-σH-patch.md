# Session A Prose Drafts v3 — σ_H Verification Patch — 2026-05-06 PM

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits)
**Status:** Two coordinated paragraph patches ready for application following the σ_H verification outcome. Triggers §4.6 row 169 of the action plan.

**Why this exists:** The diagnostic session ran the σ_H verification today and reported (§ "σ_H VERIFICATION FOLLOW-UP — 2026-05-06" in `docs/logs/2026-05-05_diagnostic-report.md`):

- σ_H = 0 across all 17 layers under all three test regimes (hallucination-style, coercion-style, neutral).
- Critically: σ_H = 0 on Prompt B, which was drawn verbatim from `_COERCED_EXAMPLES` — the same set used as `hallucinated_examples` during H-Neuron calibration. Neurons selected to fire on coercion-vs-benign should fire on a coercion test prompt by construction.
- ρ_R varied substantially across regimes (deepest layer: −14.66 / −7.15 / −2.93), confirming hidden-state delivery is working. The constancy is localized to the H-Neuron dense-scoring path.

This refutes the calibration-prerequisite framing (H_A) currently in piece #4-revised's third paragraph. The honest framing is now: a second silent-failure mode in the bottom-up channel, distinct from GAP-C2-06.

**Pieces in this file:**
1. **Piece #4-revised paragraph 3 patch** — replaces the third paragraph of §7.4 (the σ_H paragraph specifically; first two paragraphs unchanged).
2. **Piece #6 §5 Experiment 1 patch** — replaces one sentence about σ_H in the Results paragraph; rest of Results paragraph and Interpretation paragraph unchanged.

**Application order:** either order works; the two patches don't interact textually. #1 first is mildly preferred because it lands the substantive content that #2 forward-references.

---

## 1. Piece #4-revised paragraph 3 patch (σ_H paragraph)

**TeX target:** §7.4 third paragraph (the σ_H paragraph), inside the subsection landed at TeX:515 with markers `[REVISION | §7-measurement-prereqs | 2026-05-06]` and `[REVISION | §7.4-post-fix-update | 2026-05-06]`. Locate via grep for "bottom-up H-Neuron stress channel" or "hallucination-inducing".

**REVISION marker:** `[REVISION | §7.4-σH-verification-update | 2026-05-06]`

**Replacement text** (replaces the third paragraph only; first two paragraphs of §7.4 stay as-is):

```latex
The bottom-up H-Neuron stress channel ($\sigma_H$) is reported as zero across all $\alpha$ in our reference experiments. This is independent of the steering pipeline (which demonstrably reaches the cache, per the per-$\alpha$ variation in $\rho_R$, TDS, and OEI in §\ref{sec:experiments}). To distinguish whether the constancy reflects a calibration-regime mismatch~\cite{gao2025hneurons} or a deeper measurement issue, we ran a controlled three-prompt comparison at fixed $\alpha = 1.0$, varying only the test prompt across hallucination-style, coercion-style, and neutral regimes --- with the coercion-style prompt drawn directly from the same class used as the H-Neuron monitor's contrastive calibration set. $\sigma_H$ remained identically zero across all 17 layers in all three regimes, while $\rho_R$ varied substantially (deepest-layer values of $-14.66$, $-7.15$, and $-2.93$ across the three regimes respectively). Constancy of $\sigma_H$ under a test prompt drawn from the calibration's selection class indicates a silent-failure mode in the dense-scoring path itself, distinct from and independent of the steering-projection issue resolved above. We identify this as a second measurement-pipeline issue, and validation of the bottom-up channel as a precondition for full telemetry-matrix evaluation; resolution is left as future work. We note that Theorem~\ref{thm:reversibility} and Proposition~\ref{prop:memory} are unaffected by these measurement choices: both are intervention-agnostic.
```

**Notes for Session B:**
- The Theorem 1 + Proposition reassurance sentence at the end of the paragraph is *moved up* from where it was as a standalone fourth paragraph in piece #4-revised — folding it into this rewrite saves a paragraph break. If the previous version had the reassurance as its own paragraph below this one, please remove the standalone version when you apply this patch.
- The phrase "We identify this as a *second* measurement-pipeline issue" is intentional — it foregrounds that we now have two findings, not one. Do not soften.
- The deepest-layer ρ_R numbers (`-14.66, -7.15, -2.93`) are quoted directly from the verification report and are load-bearing evidence that hidden-state delivery is working. Do not round more aggressively (these are already at 2 decimal places).
- After applying this patch, both `[REVISION | §7-measurement-prereqs | 2026-05-06]` (original-insertion record) and `[REVISION | §7.4-post-fix-update | 2026-05-06]` (afternoon update) and `[REVISION | §7.4-σH-verification-update | 2026-05-06]` (this patch) should all be present in the file as a layered audit trail. Add the new marker; do not remove the older two.

---

## 2. Piece #6 §5 Experiment 1 patch (σ_H sentence)

**TeX target:** §5 Experiment 1 Results paragraph, inside the section landed at TeX:335 with marker `[REVISION | Exp1-reframe-with-results-α+ | 2026-05-06]`. The specific sentence currently reads (or close to):
> "The H-Neuron stress channel $\sigma_H$ reports zero across all $\alpha$; this is consistent with its calibration against hallucination-inducing prompts and is discussed further in §\ref{sec:measurement-prereqs}."

**REVISION marker:** `[REVISION | Exp1-σH-verification-update | 2026-05-06]`

**Replacement sentence:**

```latex
The H-Neuron stress channel $\sigma_H$ reports zero across all $\alpha$; controlled verification (§\ref{sec:measurement-prereqs}) localizes this to a measurement-pipeline issue in the dense-scoring path independent of the steering projection or calibration regime.
```

**Notes for Session B:**
- This is a one-sentence in-place replacement. Other sentences in the Results paragraph (the OEI / ρ_R / TDS reporting) and the Interpretation paragraph (the Bailey-consistent framing) are unchanged.
- The sentence is shorter than the original — net page-count savings of ~5 words. Good for budget pressure.

---

## What's outstanding after this patch

Action plan §4.6 row 169 closes (this patch is the queued coordinated update). §4.6 rows 168 (memory-complexity 4-place rewrite, gated on Max's simplified Exp 3) and 170 (bug-fix supplementary content, optional) remain open.

Section A residual work for tomorrow morning:
- §2.1 audit (background paragraph framing under Option α+); possibly piece #7.
- README rewrite drafts after §2.1 settles.

Sections B residual work after this patch:
- Apply this patch (~10 min).
- Then back to standard holding pattern (Max's measurement; EMNLP template).

---

*This patch closes the σ_H tracker item. The paper now honestly identifies two measurement-pipeline issues — one resolved, one identified — under Option α+. Net stronger paper than single-issue framing.*
