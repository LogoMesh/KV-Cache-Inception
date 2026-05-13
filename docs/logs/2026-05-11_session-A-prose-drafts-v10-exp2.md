# Session A v10 Prose Draft — §5 Experiment 2: Latent Cartography MCTS

**Authored:** 2026-05-11 (Day 3 of ARR-cycle execution plan v2; pulled forward from plan-v2 Day 6)
**Audience:** Session B (TeX application target — §5 Experiments)
**Status:** Day-3 first draft. **All empirical numbers sourced verbatim from primary artifact** — `docs/logs/2026-05-11_track-G-cartography-report.md` §2.1, §2.2, §2.3, §2.4, §3 (re-read 2026-05-11 EOD).
**Scope:** This draft covers §5 **Experiment 2 only** (Latent Cartography MCTS). §5 **Experiment 1 also requires recast** (current TeX has Option α+ "OEI Characterization"; under Option D+ Exp 1 should be Track F entropy-MCTS negative control). Exp 1 recast is a separate Day-4 draft, NOT included here.

---

## 1. Pivot summary

| Element | Was (Option α+, current TeX) | Becomes (Option D+, this draft) |
|---|---|---|
| Title | "Reward-Function Ablation in Latent Space" | "Latent Cartography MCTS across Prompt Classes" |
| Methodology | Three-arm ablation (full reward vs σ_H-only vs random-KV) | Track F-style MCTS path-sampling (depth-3, 27 paths, alpha-set {0.1, 0.5, 1.0}, entropy reward) across 4 prompt classes × 2 scales |
| Sample | n=? at single scale Llama-3.2-1B-Instruct | n = 4×100 = ~385 records per scale × 2 scales = 770 records (C3 drops 15 items lacking ≥3 distractors) |
| Status | PARTIAL (blocked by B6 measurement-pipeline bug at time of writing) | COMPLETE 2026-05-11; all 3/3 success criteria MET at both scales |
| Headline | (Pending bug resolution) | Class-specific path selection generalizes Track F across 4 prompt classes; MCQ direction-inverts across scales (1B mean-step α 0.83-0.89; 3B 0.22-0.28); gold-rank preservation nuance |

§5 Exp 2 in Option D+ is the **positive structural mapping** to Track F's negative control: where Track F isolates the mechanism on one benchmark, Cartography characterizes whether the mechanism is benchmark-specific, format-specific, or scale-specific.

---

## 2. Replacement TeX for §5 Experiment 2

**TeX target:** Canonical TeX lines 385-388 (REVISION marker + `\paragraph{Experiment 2}` + body + IMPL line).

**Replacement body** (replaces TeX:385-388 verbatim):

```latex
% [REVISION | Exp2-recast-cartography-D+ | 2026-05-11 | Wholesale recast of §5 Experiment 2 from Option α+'s "Reward-Function Ablation in Latent Space" (three-arm ablation: full reward vs σ_H-only vs random-KV; never ran due to B6 pipeline bug then resolved 2026-05-06) to Option D+'s "Latent Cartography MCTS across Prompt Classes" (Track G, complete 2026-05-11). New methodology: Track F-style depth-3 MCTS path-sampling with entropy reward, alpha-set {0.1, 0.5, 1.0}, applied to 4 prompt classes (factual recall, ARC-Easy MCQ, TruthfulQA MCQ, HellaSwag continuation) × Llama 3.2-{1B, 3B}-Instruct. Track G report: docs/logs/2026-05-11_track-G-cartography-report.md. Adds Table~\ref{tab:exp2-cartography-paths} (best-path α-tuple distribution per class × scale) and Figure~\ref{fig:exp2-mean-step-alpha} (cross-scale mean-step α bar chart). Adds \label{sec:exp2} for §3 Hypothesis 2 and §7.1/§7.5 forward references. The 2026-05-05 Exp2-recast-reward-ablation marker is subsumed; the three-arm ablation framing is dropped (not run; Phase B research item per §7.5 reward-design-research paragraph). — pending audit]
\paragraph{Experiment 2: Latent Cartography MCTS across Prompt Classes.}\label{sec:exp2}
We characterize how the search optimizes a surrogate reward as a function of prompt class. The Track~F protocol (Experiment~1) is generalized across four prompt classes---factual recall (C1; hand-constructed 100-item set covering capitals, dates, arithmetic, common knowledge), ARC-Easy multiple-choice (C2; first 100 test-split items, overlapping Track~F's set for direct comparison), TruthfulQA mc1 multiple-choice (C3; first 100 validation items reformatted to 4-option), and HellaSwag continuation (C4; first 100 validation items with option list stripped)---on both Llama~3.2-1B-Instruct and Llama~3.2-3B-Instruct. For each item we run depth-$3$ MCTS with $27$ path samples drawn from the alpha-set $\{0.1, 0.5, 1.0\}$, scored by the normalized-entropy reward $\hat{r}(\mathbf{h}) = -H(p(\cdot \mid \mathbf{h})) / \log|V|$ (Eq.~\ref{eq:goodhart}); the best path is re-applied permanently and the next-token distribution at the resulting leaf is recorded. The steering direction is the same random unit vector used in Experiment~1, fixed via \texttt{--seed 42} across all items, classes, and scales.

We measure three structural signatures: (i)~the distribution of best-path $\alpha$-tuples selected by the search per class (does the search prefer heavy steering, minimal steering, or class-mixed depending on prompt class?); (ii)~the top-5 argmax-token distribution at the best-path leaf per class (does the search amplify a class-specific prior token?); and (iii)~the gold-first-token rank in the steered distribution per class (does prior amplification erase the correct answer or merely inflate the prior's mass?). Together these signatures characterize whether the Goodhart pathology established by Track~F (Experiment~1) generalizes across prompt classes and across model scale.

\textbf{Results.} Best-path $\alpha$-tuple distributions are reported in Table~\ref{tab:exp2-cartography-paths}; cross-scale mean-step $\alpha$ comparison in Figure~\ref{fig:exp2-mean-step-alpha}. Three findings stand out:

\textbf{(i) Class-specific best-path selection (Table~\ref{tab:exp2-cartography-paths}).} At Llama-3.2-1B, the search selects heavy steering for MCQ classes (mean-step $\alpha = 0.89$ for C2 ARC-Easy with $42\%$ of items at modal path $(1.0,1.0,1.0)$; $\alpha = 0.83$ for C3 TruthfulQA at $28\%$ modal $(1.0,1.0,1.0)$) but minimal-to-moderate steering for factual recall ($\alpha = 0.34$, $41\%$ modal $(0.1,0.1,0.1)$) and continuation ($\alpha = 0.51$, $21\%$ modal $(0.1,0.1,0.1)$). At Llama-3.2-3B, the pattern inverts for MCQ: minimal steering dominates ($\alpha = 0.22$ for C2 at $49\%$ modal $(0.1,0.1,0.1)$; $\alpha = 0.28$ for C3 at $36\%$ modal $(0.1,0.1,0.1)$), while factual recall shifts to moderate steering ($\alpha = 0.61$). The MCQ classes thus exhibit a direction-flip in search-selection $\alpha$ between scales of magnitude $\Delta \approx 0.67$ for C2 and $\Delta \approx 0.55$ for C3.

\textbf{(ii) Argmax-token concentration at the best-path leaf.} At 1B the MCQ classes collapse to the letter ``A'' at the best-path leaf---$92$ of $100$ items for C2 and $81$ of $85$ valid items for C3, reproducing Track~F's headline ($185$ of $200$ items argmax-``A'' under heavy steering, §\ref{sec:exp1}). At 3B no such collapse occurs: the C2 argmax distribution is balanced across the four letters ($29$ ``A'', $27$ ``B'', $22$ ``D'', $16$ ``C''), and C3 similarly remains diverse ($39$ ``A'', $17$ ``B'', $15$ ``C'', $13$ ``D''). The factual recall and continuation classes show no single argmax attractor at either scale---argmax tokens are diverse (e.g., at 3B, factual recall top-1 tokens span ``8'', ``5'', ``7'', ``6'', ``April'').

\textbf{(iii) Gold-rank preservation despite argmax collapse.} At 1B C2 where heavy steering drives argmax to ``A'' on $92\%$ of items, the gold answer's median rank at the steered leaf remains $1$ ($94\%$ of items have gold in top-5). At 1B C3 the median gold rank is also $1$ with $100\%$ in top-5. The MCTS amplification inflates the prior's probability mass without erasing the answer's representation---the model ``knows'' the answer, but Goodhart-optimization for entropy reduction causes argmax-``A'' selection instead of argmax-gold. This is the geometric mechanism Definition~\ref{def:dim_escape} predicts: the search resolves a low-rank reward through trajectories whose projection onto the answer-relevant subspace is non-zero, yet the unconstrained-complement direction (prior amplification) dominates argmax selection.

\textbf{(iv) Mechanism interpretation per class.} Factual recall (no single first-token prior; items have diverse correct first tokens) gives the search no prior to exploit, hence the selected $\alpha$ is light. MCQ classes (strong first-letter prior at 1B from training-data alphabetical-option-listing patterns) provide an exploitable prior, and the search picks heavy $\alpha$ to amplify it; at 3B the prior is already balanced across A/B/C/D so heavy $\alpha$ would only \emph{raise} entropy, and the search instead selects minimal $\alpha$ to preserve the greedy state. Continuation (high baseline entropy) gives bimodal search behavior at 1B (some items pick heavy $\alpha$ to find a low-entropy ``fallback continuation,'' others pick light $\alpha$); at 3B the more general 3B preference for minimal $\alpha$ dominates. The class-specific best-path-$\alpha$ patterns are interpretable as the search exploiting class-specific reward landscapes---in every class, the search finds the $\alpha$ regime where the entropy minimum lies, but the regime differs by class. This is the structural extension of Track~F: Goodhart in cache geometry is not a single fixed mechanism but a class-conditional one, with the route through the latent space determined by the model's class-specific prior structure.

\textbf{Search reward gain (entropy reduction) is positive but small across all classes.} The best path among the 27 sampled gives between $+0.005$ and $+0.025$ better reward (normalized-entropy units) than the mean across paths. The search is genuinely finding lower-entropy states; the entropy reduction simply does not correspond to better reasoning---consistent with Hypothesis~\ref{hyp:goodhart}.
% [IMPL: COMPLETE — Track G driver at scripts/diagnose_track_g_cartography.py (--mode mcts); raw artifacts at scripts/_track_g_mcts_results_meta-llama_Llama-3.2-{1B,3B}-Instruct.json. Reproduction: see Track G report Appendix A.]
```

---

## 3. New Table 2 — Best-path α-tuple distribution (insert after Experiment 2 paragraph)

```latex
\begin{table}[h]
\centering
\small
\begin{tabular}{@{}l rrr rrr@{}}
\toprule
& \multicolumn{3}{c}{\textbf{Llama-3.2-1B}} & \multicolumn{3}{c}{\textbf{Llama-3.2-3B}} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
\textbf{Class} & Modal path & \% modal & $\bar\alpha$ & Modal path & \% modal & $\bar\alpha$ \\
\midrule
C1 Factual Recall & $(0.1,0.1,0.1)$ & $41\%$ & $0.34$ & mixed & $15\%$ & $0.61$ \\
C2 ARC-Easy MCQ & $(1.0,1.0,1.0)$ & $42\%$ & $0.89$ & $(0.1,0.1,0.1)$ & $49\%$ & $0.22$ \\
C3 TruthfulQA MCQ & $(1.0,1.0,1.0)$ & $28\%$ & $0.83$ & $(0.1,0.1,0.1)$ & $36\%$ & $0.28$ \\
C4 Continuation & $(0.1,0.1,0.1)$ & $21\%$ & $0.51$ & $(0.1,0.1,0.1)$ & $31\%$ & $0.38$ \\
\bottomrule
\end{tabular}
\caption{Latent Cartography best-path $\alpha$-tuple distribution per prompt class and scale. Each item runs $27$ depth-$3$ MCTS paths from $\{0.1, 0.5, 1.0\}$; the best path under entropy reward is selected per item, and the per-class distribution over $3^3 = 27$ possible path tuples is reported. ``Modal path'' is the most-frequent best-path tuple; ``\% modal'' is the share of items selecting it; $\bar\alpha$ is the mean per-step $\alpha$ averaged across items and across the three steps of the best path. MCQ classes (C2, C3) exhibit a direction-flip in $\bar\alpha$ across scales: heavy steering at 1B (the search exploits the first-letter prior), minimal steering at 3B (the balanced baseline distribution provides no prior to exploit). Factual recall (C1) and continuation (C4) show smaller cross-scale differences.}
\label{tab:exp2-cartography-paths}
\end{table}
```

---

## 4. New Figure 1 — Cross-scale mean-step α (insert after Table 2)

**Figure spec** (image file to be produced from raw artifacts before Day 7 application):

- **Source data:** `scripts/_track_g_mcts_results_meta-llama_Llama-3.2-{1B,3B}-Instruct.json`
- **Plot type:** Grouped bar chart, 4 groups (one per class C1-C4), 2 bars per group (1B vs 3B mean-step α)
- **Y-axis:** Mean-step α, range $[0, 1]$
- **Numeric values:** C1: 1B=0.34, 3B=0.61 ($\Delta = +0.27$); C2: 1B=0.89, 3B=0.22 ($\Delta = -0.67$); C3: 1B=0.83, 3B=0.28 ($\Delta = -0.55$); C4: 1B=0.51, 3B=0.38 ($\Delta = -0.13$)
- **Visual hook:** C2 and C3 bars flip direction across scales (1B bars high, 3B bars low) while C1 and C4 do not — MCQ classes uniquely exhibit the direction-inversion
- **Output:** `docs/NeurIPS/figures/exp2-mean-step-alpha.pdf`

**LaTeX include placeholder** (insert after Table 2, before §5.4 Baselines):

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.9\linewidth]{figures/exp2-mean-step-alpha.pdf}
\caption{Cross-scale mean-step $\bar\alpha$ by prompt class. MCQ classes (C2 ARC-Easy, C3 TruthfulQA) show direction-flip across scales: at 1B the entropy-MCTS selects heavy steering ($\bar\alpha = 0.89, 0.83$) to amplify the model's first-letter prior; at 3B the search instead selects minimal steering ($\bar\alpha = 0.22, 0.28$) since the balanced baseline distribution offers no prior to exploit. Factual recall (C1) and continuation (C4) show smaller cross-scale differences. The MCQ direction-flip is the Cartography contribution to Hypothesis~\ref{hyp:goodhart}: Goodhart in cache geometry is class-conditional, with the route through latent space determined by the model's class-specific prior structure.}
\label{fig:exp2-mean-step-alpha}
\end{figure}
```

---

## 5. Notes for Session B

**Application order Day 7 (or pulled forward to Day 4 if §5 Exp 1 recast lands first):**

1. Generate `figures/exp2-mean-step-alpha.pdf` from raw JSON (one-off script, ~10 lines of matplotlib; Josh or Session B writes).
2. Apply Experiment 2 paragraph (REVISION marker + replacement body) to canonical TeX lines 385-388.
3. Insert Table 2 immediately after the Experiment 2 paragraph body (between IMPL line and §5.4 Baselines).
4. Insert Figure 1 immediately after Table 2.
5. Verify `\label{sec:exp2}` resolves §3 Hypothesis 2 forward-reference and §7.1 + §7.5 reward-design forward-references.

**Cross-references introduced:**
- `\label{sec:exp2}` — NEW, resolves three existing forward-references (`\ref{sec:exp2}` at TeX:171 in §3, at TeX:529 in §7.1, at TeX:580 in §7.5, at TeX:584 in §7.5).
- `\label{tab:exp2-cartography-paths}` — NEW, referenced in the body paragraph itself.
- `\label{fig:exp2-mean-step-alpha}` — NEW, referenced in the body paragraph itself.

**Cross-references RELIED ON (already exist):**
- `\ref{def:dim_escape}` — TeX:145 ✓
- `\ref{hyp:goodhart}` — TeX:165 ✓
- `\ref{eq:goodhart}` — TeX:169 ✓
- `\ref{sec:exp1}` — **NOT YET DEFINED** — will be added by the Day-4 §5 Experiment 1 recast draft. Will produce an undefined-reference warning on intermediate compilation; resolved when Exp 1 recast lands.

**Compilation impact estimate:**
- Net change in body lines: replacement paragraph (~50 lines of prose; current Exp 2 is 1 paragraph of ~10 lines + IMPL line) = **net +40 lines of body**
- New Table 2: 12 lines including `\begin{table}` framing
- New Figure 1: 5 lines including `\includegraphics`
- Total: **net +57 lines** added to canonical TeX after Day 7 application

**Page-budget implication:** Currently canonical TeX is 709 lines with body ending ~line 600 (bibliography starts ~line 595). Adding +57 lines of body pushes body to ~657 lines. At ~70 body lines/page in ACL two-column format, the body lands at ~9.4 pages — over the 8-page review budget. **Day 9 page-count check + Day 10 cuts mandatory.** Most likely cut: §6 Memory Complexity proof block (lines 477-518) moves to `\appendix`, freeing ~1 page.

**Stale Exp 1 framing in current TeX** (separate Day-4 task, not in this draft scope):
- Current §5 Exp 1 (TeX:357-383) is "OEI Characterization" — Option α+ remnant. Under Option D+ should be **Track F entropy-MCTS negative control on ARC-Easy** (n=200 × 2 scales × 3 conditions = 1200 records).
- Day-4 draft scope: replacement TeX for §5 Exp 1 paragraph + Table 1 recast (current Table 1 is the OEI/ρ_R/σ_H/TDS alpha-sweep — needs replacement with Track F results table per Track F report §3).
- Current OEI alpha-sweep content can be **relocated to §7.4 Empirical Measurement Prerequisites** (which already discusses these measurements in prose; the table would migrate to support that discussion). Saves needing to invent new home for the OEI alpha-sweep results.

---

## 6. Page-budget commentary

| Section | Current line count | Post-Exp-2 line count | Notes |
|---|---|---|---|
| §5 Experiments subsection | ~120 lines | ~177 lines | +57 from this draft |
| Total body (lines 1-587) | 587 lines | 644 lines | +57 from this draft |
| Body lines/page ACL two-column | ~70 | ~70 | Empirical estimate |
| Body pages estimate | 8.4 | 9.2 | Over 8-page review budget |

If Exp 1 recast also adds content (likely +30 lines for Track F detail + Table 1 recast), Day 9 page count lands at ~9.6 pages. **Day 10 cuts mandatory.** Recommended cut list (in priority order):
1. Move §6 Memory Complexity proof to `\appendix` (saves ~1 page, ~40 lines)
2. Condense §5.4 Baselines table to 3 rows under Option D+ (Cache Steering Belitsky / Activation Steering Rimsky / RL-Obfuscation Gupta-Jenner are the relevant peers; GCG/MPA/DAMON drop since text-space red-teaming is not our framing)
3. If still over: condense §4.2.2 verbose paragraphs

These cuts are reversible at camera-ready (Phase 3, additional page granted).

---

## 7. Sources (primary-source-fidelity discipline)

All numeric values and methodological claims in this draft trace to primary artifacts read directly in this session:

| Claim | Primary source | Verified |
|---|---|---|
| 4 prompt classes definition (C1-C4) | Track G report §1 + §1.5 | ✓ |
| MCTS protocol (depth-3, 27 paths, alpha-set) | Track G report §1 + Track F report §2 | ✓ |
| Best-path α-tuple distribution per class × scale | Track G report §2.1 tables | ✓ |
| Mean-step α numbers (0.34/0.89/0.83/0.51 at 1B; 0.61/0.22/0.28/0.38 at 3B) | Track G report §2.1 tables | ✓ |
| Cross-scale mean-step Δ values (+0.27/−0.67/−0.55/−0.13) | Track G report §3 cross-scale table | ✓ |
| Modal percentages (41%/42%/28%/21% at 1B; 15%/49%/36%/31% at 3B) | Track G report §2.1 tables | ✓ |
| Argmax distributions (1B C2 92/100; 1B C3 81/85; 3B C2 29/27/22/16; 3B C3 39/17/15/13) | Track G report §2.2 tables | ✓ |
| Gold-rank statistics (1B C2 median 1, 94% top-5; 1B C3 median 1, 100% top-5) | Track G report §2.3 tables | ✓ |
| Track F reference numbers (185/200 argmax-A at 1B = 92.5%) | Track F report §1 + §3 (primary, not running-log) | ✓ |
| Search reward gain range (+0.005 to +0.025) | Track G report §2.3 tables | ✓ |
| Mechanism interpretation per class | Track G report §2.4 | ✓ |

No number in this draft was sourced from a running-log summary, a compaction-memory recall, or a citation-of-a-citation chain.

---

*End of v10 Exp 2 draft. Apply Day 7 (or pulled forward to Day 4 alongside §5 Exp 1 recast for combined §5 application). Page-budget impact +57 lines body confirms Day 9 page-count check + Day 10 cut decisions become mandatory.*
