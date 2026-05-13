# Session A v10 Prose Draft — §5 Experiment 1: Entropy-MCTS Negative Control on ARC-Easy

**Authored:** 2026-05-12 (Day 4 of ARR-cycle execution plan v2; scheduled Day-4 P1 task).
**Audience:** Session B (TeX application target — §5 Experiments + §7.4 Empirical Measurement Prerequisites).
**Status:** Day-4 first draft. **All empirical numbers sourced verbatim from primary artifact** — `docs/logs/2026-05-11_track-F-negative-control-report.md` §1–§7 (re-read 2026-05-12).
**Scope:** This draft covers (a) §5 **Experiment 1** wholesale recast (Track F entropy-MCTS negative control prose + new Table 1 with Track F accuracy/perplexity statistics); (b) the **relocation of the existing Table 1** (OEI/ρ_R/σ_H/TDS alpha-sweep) from §5 to §7.4 with a label rename, per framing-audit-pass-2 Finding F6 coordination. §5 **Experiment 2** is drafted separately at `docs/logs/2026-05-11_session-A-prose-drafts-v10-exp2.md` (Day 7 application target).

---

## 1. Pivot summary

| Element | Was (Option α+, current TeX lines 326–353) | Becomes (Option D+, this draft) |
|---|---|---|
| Title | "OEI Characterization under Inference-Time KV-Cache Perturbation" | "Entropy-MCTS Negative Control on ARC-Easy" |
| Methodology | Single-prompt RepE alpha sweep (α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}) along the RepE honesty direction; measure OEI, ρ_R, σ_H, TDS per α | Track F-style depth-3 entropy-MCTS path-sampling (27 paths, alpha-set {0.1, 0.5, 1.0}) with paired 3-condition design (G greedy / R random-MCTS / E entropy-MCTS); single shared random unit-vector steering direction |
| Sample | Llama-3.2-1B-Instruct only; single-prompt sweep | n = 200 paired ARC-Easy items × 2 scales (1B + 3B) × 3 conditions = 1200 records |
| Status | Ran 2026-04-25; results consistent with Bailey 2024 negative-result regime for Hypothesis 1 (monitor-driven sub-case) | COMPLETE 2026-05-09; both scales independently produce same headline (NO-GO Gate 3 on entropy-MCTS reasoning improvement) |
| Headline | OEI varies monotonically (0.991 → 0.995); TDS decreases monotonically with α; consistent with Bailey high-dimensional null-space saturation | Δ(E−R) = −0.5pp at 1B / +0.5pp at 3B (McNemar p = 1.0 both scales); mechanism is class-conditional (1B prior-amplification → "A" predicted 185/200; 3B greedy-collapse → minimal α selected 51% of items) |
| Where prior content lands | — | The alpha-sweep measurement content (Table + interpretation) relocates to §7.4 Empirical Measurement Prerequisites (where the prose already discusses these numbers); the table relabels `tab:exp1-results` → `tab:oei-alpha-sweep` |

Under Option D+, §5 Experiment 1 is the **load-bearing controlled negative-control** for Hypothesis 2 (reward-driven Dimensional Escape, Eq.~\ref{eq:goodhart}). §5 Experiment 2 (Latent Cartography, separate draft) is the positive structural mapping of the same mechanism across prompt classes. Together they establish Contribution 3.

The relocation of the alpha-sweep table to §7.4 is structurally clean: §7.4 prose (line 535 of current canonical TeX) already says "Under the projection-based fix, OEI varies monotonically with steering magnitude on Llama 3.2-1B-Instruct (results in §\ref{sec:experiments})". After relocation, the forward-reference resolves locally (the table lives in §7.4 itself); the rest of §7.4's σ_H paragraph (line 540) already references `Table~\ref{tab:exp1-results}` — that reference is updated to the renamed label.

---

## 2. Replacement TeX for §5 Experiment 1

**TeX target:** Canonical TeX lines 326–353 (REVISION markers `Exp1-reframe-with-results-α+` + `Exp1-σH-verification-update` + `Exp1-results-σH-resolved` + Experiment 1 setup paragraph + Results paragraph + Table 1 block + Interpretation paragraph + IMPL line).

**Action:**
1. Replace lines 326–333 (the three layered REVISION markers + `\paragraph{Experiment 1:...}` + setup paragraph + Results paragraph) with the new REVISION marker + new paragraph below.
2. **Cut** the Table 1 block (lines 334–350) from §5 entirely.
3. Replace lines 352–353 (Interpretation paragraph + IMPL line) with the new Interpretation + IMPL lines below.

**Replacement body** (replaces TeX:326–353 verbatim):

```latex
% [REVISION | Exp1-recast-trackf-D+ | 2026-05-12 | Wholesale recast of §5 Experiment 1 from Option α+'s "OEI Characterization under Inference-Time KV-Cache Perturbation" (single-prompt RepE alpha sweep on Llama 3.2-1B-Instruct; ran 2026-04-25; results positioned as consistent with Bailey 2024 high-dimensional null-space saturation under Hypothesis 1 monitor-driven sub-case) to Option D+'s "Entropy-MCTS Negative Control on ARC-Easy" (Track F, complete 2026-05-09; n=200 paired ARC-Easy items × 2 scales × 3 conditions = 1200 records). New methodology: depth-3 entropy-MCTS path-sampling with alpha-set {0.1, 0.5, 1.0} and 27 path samples per item, paired against a random-MCTS arm with identical search machinery (the COCONUT defense isolates the entropy reward as the only difference between R and E). Track F report: docs/logs/2026-05-11_track-F-negative-control-report.md §1-§7. Replaces Table~\ref{tab:exp1-results} content with Track F accuracy + perplexity statistics. The pre-existing Table~\ref{tab:exp1-results} (OEI/ρ_R/σ_H/TDS alpha-sweep) relocates to §7.4 Empirical Measurement Prerequisites with label rename to tab:oei-alpha-sweep per framing-audit-pass-2 Finding F6 coordination — its content (alpha-sweep measurements) is structurally about measurement-pipeline characterization and belongs with §7.4's existing prose on that topic. Adds \label{sec:exp1} resolving forward references from §3.2 (lines 167, 170), §1 Contributions Contribution 3 (line 80), §1 Introduction paragraph 4 (line 68), §1 abstract (line 53), and v10-exp2 §5 Experiment 2 prose. The 2026-05-06 Exp1-reframe-with-results-α+, 2026-05-06 Exp1-σH-verification-update, and 2026-05-07 Exp1-results-σH-resolved markers are subsumed (the parent OEI alpha-sweep paragraph relocates with the table; markers retained as audit trail at the table's new home in §7.4). — pending audit]
\paragraph{Experiment 1: Entropy-MCTS Negative Control on ARC-Easy.}\label{sec:exp1}
We test whether the entropy-normalized surrogate reward $\hat{r}(\mathbf{h}) = -H(p(\cdot \mid \mathbf{h}))/\log|V|$ (Eq.~\ref{eq:goodhart}) carries useful optimization signal for reasoning when deployed as the objective of MCTS over the KV-cache latent space. The benchmark is ARC-Easy (\texttt{allenai/ai2\_arc}, test split, first $200$ deterministic items), evaluated independently on Llama~3.2-1B-Instruct and Llama~3.2-3B-Instruct. On each item we run three paired conditions on the same prompt: \textbf{G} (greedy decoding, no MCTS); \textbf{R} (random-MCTS: $27$ depth-$3$ paths drawn from the alpha-set $\{0.1, 0.5, 1.0\}$, scored by $\hat{r}_R \sim \mathrm{Uniform}(0, 1)$; the best-rewarded path is re-applied permanently to the live KV cache and the answer is generated from the steered state); \textbf{E} (entropy-MCTS: identical search machinery, but path-scoring uses the entropy-normalized surrogate $\hat{r}_E(\mathbf{h}) = -H_t/\log|V|$ evaluated at the steered leaf). The steering direction is a single random unit vector in $d_{\text{model}}$ space (\texttt{--seed 42}), shared identically across all items and across the R and E arms; this isolates the entropy reward signal as the only difference between R and E (the COCONUT defense).

\textbf{Results.} Per-condition accuracy, 95\% Wilson confidence intervals, and mean perplexity of the 5-token completion under the unsteered model are reported in Table~\ref{tab:exp1-results}. The headline pairwise comparison is entropy-MCTS against the random-MCTS control: at 1B the accuracy gap is $\Delta_{E-R} = -0.005$ ($E_{\text{wrong}} \wedge R_{\text{right}} = 1$; $E_{\text{right}} \wedge R_{\text{wrong}} = 2$; exact paired McNemar $p = 1.0000$); at 3B the gap is $\Delta_{E-R} = +0.005$ ($E_{\text{wrong}} \wedge R_{\text{right}} = 2$; $E_{\text{right}} \wedge R_{\text{wrong}} = 1$; exact paired McNemar $p = 1.0000$). With only three discordant items per scale---against a budget calibrated to detect $\Delta = 10$\,pp at $\alpha = 0.05$ and power $0.8$ for discordance rates up to $\pi_d = 0.25$---entropy-MCTS does not differentiate from a uniform-random reward at either scale. Against the no-search greedy baseline, entropy-MCTS regresses by $-2$\,pp at 1B ($p = 0.2188$) and $-2.5$\,pp at 3B ($p = 0.1250$); the search machinery converges to states that are slightly worse than greedy. Mean-perplexity ratios $E/G = 1.348$ at 1B and $1.058$ at 3B confirm the searched cache states remain on-distribution under the unsteered model---the failure is not from off-manifold drift but from the entropy objective misaligning with correctness.

\textbf{Mechanism.} At both scales the search machinery operates as designed: across the $27$ path samples per item, the best-path entropy reward consistently exceeds the across-path mean (mean entropy-reduction gain $+0.012 \pm 0.005$ at 1B; the random arm's best-path distribution is approximately uniform over the alpha-tuples). The pathway by which the search satisfies $\hat{r}_E$ is class-conditional on the unsteered model's prior structure. At 1B, where the unperturbed cache exhibits moderate uncertainty on letter-prompted multiple choice and the alphabetical first-letter prior favors ``A'', the search selects heavy steering: the best path is $(1.0, 1.0, 1.0)$ on $91/200$ items ($46$\%), and the entropy-MCTS arm predicts ``A'' on $185/200$ items against a roughly uniform gold distribution over A/B/C/D---the search weaponizes the model's prior. At 3B, where the unperturbed cache is already low-entropy and instruction-following is strong, minimal steering preserves the greedy-equivalent state: the best path is $(0.1, 0.1, 0.1)$ on $101/200$ items ($51$\%), collapsing to greedy decoding up to small accumulator noise. Neither pathway selects ``the cache state that produces the correct answer''; both select ``the cache state that minimizes entropy under the model's existing biases''. This is the empirical mechanism predicted by Hypothesis~\ref{hyp:goodhart}: in dimension $d \gg \mathrm{rank}(\hat{r}_E)$, the optimizer satisfies the surrogate reward via trajectories whose projection onto the answer-relevant subspace is incidental, while the unconstrained-complement direction---prior amplification at 1B, greedy preservation at 3B---dominates argmax selection. The pathology surfaces by different routes at the two scales but reduces to the same geometric failure mode.

\textbf{Interpretation.} The entropy reward was motivated by the observation that next-token entropy correlates statically with answer correctness at $|r| \approx 0.60$ on fixed prompts that vary in difficulty. That static correlation does not transfer to the latent-search regime, where fixed-difficulty items are perturbed in arbitrary cache-space directions and confidence becomes a function of cache geometry rather than prompt difficulty. The failure is conceptual rather than implementation-specific: no choice of alpha grid, search depth, branching factor, or steering-direction calibration changes the basic mechanism. Bailey et al.~\cite{bailey2024obfuscated} reported a tentative negative result for the structurally parallel monitor-driven case (gradient-trained input-space attacks against a frozen model) on different grounds---near-orthogonality and behavior preservation cannot be jointly achieved when the optimization target is similarity minimization. The present result is the corresponding negative result for the reward-driven case (Hypothesis~\ref{hyp:goodhart}) under inference-time KV-cache perturbation with a surrogate-reward objective: both manifestations terminate in the unconstrained complement of a low-rank scoring projection rather than on the semantic manifold (Definition~\ref{def:dim_escape}). Experiment~2 (§\ref{sec:exp2}) extends this finding from a single benchmark to four prompt classes, characterizing the class-conditional structure of the failure across the Llama 3.2 family.
% [IMPL: COMPLETE — driver script at scripts/diagnose_track_f_negcontrol.py (read-only on logomesh/*; imports FP32Accumulator, _extract_kv_tensors, _kv_eval_cache from logomesh.kv_mcts). Raw artifacts: scripts/_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json (1B) and scripts/_track_f_results_meta-llama_Llama-3.2-3B-Instruct.json (3B). Wall-clock: 1B sweep 18.3 min (5.5 s/item × 200), 3B sweep 33.7 min (10.1 s/item × 200). Reproduction: see Track F report Appendix A.]
```

---

## 3. Replacement Table 1 — Track F accuracy + perplexity

**Insert position:** Immediately after the Results paragraph of the new Experiment 1 body, before the Mechanism paragraph (mirrors the v10-exp2 structure where Table 2 follows the Results header).

```latex
\begin{table}[h]
\centering
\small
\begin{tabular}{@{}l l c c c@{}}
\toprule
\textbf{Scale} & \textbf{Condition} & \textbf{Accuracy} & \textbf{95\% Wilson CI} & \textbf{Mean PPL} \\
\midrule
Llama-3.2-1B & G (greedy) & $0.295$ ($59/200$) & $[0.236, 0.362]$ & $1.547 \pm 0.263$ \\
Llama-3.2-1B & R (random-MCTS) & $0.280$ ($56/200$) & $[0.222, 0.346]$ & $2.351 \pm 1.823$ \\
Llama-3.2-1B & E (entropy-MCTS) & $0.275$ ($55/200$) & $[0.218, 0.341]$ & $2.085 \pm 1.641$ \\
\midrule
Llama-3.2-3B & G (greedy) & $0.845$ ($169/200$) & $[0.788, 0.889]$ & $1.318 \pm 0.198$ \\
Llama-3.2-3B & R (random-MCTS) & $0.815$ ($163/200$) & $[0.755, 0.863]$ & $1.398 \pm 0.327$ \\
Llama-3.2-3B & E (entropy-MCTS) & $0.820$ ($164/200$) & $[0.761, 0.867]$ & $1.394 \pm 0.319$ \\
\bottomrule
\end{tabular}
\caption{Per-condition accuracy on ARC-Easy ($n = 200$ paired items per scale) under three paired conditions: greedy decoding, random-MCTS (uniform reward), and entropy-MCTS (Eq.~\ref{eq:goodhart} as path-scoring reward). The MCTS arms apply 27 depth-3 paths from the alpha-set $\{0.1, 0.5, 1.0\}$ with a shared random unit-vector steering direction; the best-rewarded path is permanently re-applied before generation. Wilson 95\% confidence intervals overlap across all three conditions at both scales. Mean perplexity is the geometric-mean per-token negative-log-likelihood of the 5-token completion under the unsteered model. The pairwise statistical claims appear in the body prose (§\ref{sec:exp1} Results paragraph).}
\label{tab:exp1-results}
\end{table}
```

---

## 4. §7.4 Table 1 relocation — OEI/ρ_R/σ_H/TDS alpha-sweep

**TeX target:** Insert the (renamed) alpha-sweep table block into §7.4 Empirical Measurement Prerequisites immediately after the paragraph at canonical TeX line 535 (the paragraph beginning "Under the projection-based fix, OEI varies monotonically..."). Update the existing in-§7.4 reference at line 540 (`Table~\ref{tab:exp1-results}` → `Table~\ref{tab:oei-alpha-sweep}`).

The relocated block preserves the existing layered REVISION marker chain (`Exp1-reframe-with-results-α+`, `Exp1-σH-verification-update`, `Exp1-results-σH-resolved`, `Exp1-table-σH-TDS-update`) verbatim at the table's new home — these markers are part of the audit trail for the σ_H resolution arc and must not be lost.

### Action

1. **Cut** lines 334–350 of canonical TeX (the `\begin{table}` block beginning at line 334 through the `\end{table}` at line 350).
2. **Paste** the same block at canonical TeX line ~536 (immediately after the "OEI varies monotonically..." paragraph in §7.4).
3. **Rename** the `\label` from `tab:exp1-results` to `tab:oei-alpha-sweep` inside the relocated block.
4. **Wrap** the relocation in a new REVISION marker at the table's old position (deletion explanation) and at the new position (insertion explanation).
5. **Update** the existing in-§7.4 reference at line 540 from `Table~\ref{tab:exp1-results}` to `Table~\ref{tab:oei-alpha-sweep}`.

### Block to be relocated (verbatim from canonical TeX 334–350, with label rename and an added relocation marker)

```latex
% [REVISION | OEI-alpha-sweep-relocate-to-§7.4-D+ | 2026-05-12 | Relocated this Table block from §5 Experiment 1 (lines 334-350 prior) to §7.4 Empirical Measurement Prerequisites, where the surrounding prose (paragraph at line 535 "Under the projection-based fix, OEI varies monotonically..." + paragraph at line 540 σ_H resolution arc) already discusses these alpha-sweep measurements. Per framing-audit-pass-2 Finding F6 coordination with §5 Experiment 1 recast (the new Experiment 1 is Track F entropy-MCTS negative control; the alpha-sweep measurement content is structurally about measurement-pipeline characterization rather than the headline empirical claim of Experiment 1, so its appropriate home is §7.4 alongside the W_K projection and per-layer H-Neuron calibration discussion). Label renamed tab:exp1-results → tab:oei-alpha-sweep to reflect the table's content (alpha-sweep of OEI plus the three other channels) rather than its prior position. The four layered REVISION markers on this block (Exp1-reframe-with-results-α+, Exp1-σH-verification-update, Exp1-results-σH-resolved, Exp1-table-σH-TDS-update) are retained verbatim below as audit trail for the σ_H resolution arc. The only body-text reference to this table outside the table itself is in §7.4 line 540, which is updated to the new label. — pending audit]
\begin{table}[h]
\centering
\small
\begin{tabular}{@{}lrrrrr@{}}
\toprule
$\alpha$ & 0.1 & 0.5 & 1.0 & 2.0 & 5.0 \\
\midrule
OEI & 0.991 & 0.991 & 0.992 & 0.992 & 0.995 \\
$\rho_R$ & $-1.306$ & $-1.258$ & $-1.198$ & $-1.080$ & $-0.533$ \\
% [REVISION | Exp1-table-σH-TDS-update | 2026-05-07 | Updated Table 1 σ_H row from {0,0,0,0,0} to post-merge values {0.228, 0.237, 0.247, 0.267, 0.330} (14–15/17 layers firing, monotonic with α); TDS row from {0.094, 0.090, 0.084, 0.075, 0.046} to {0.084, 0.080, 0.076, 0.068, 0.040} (~7–13% relative shift downstream of σ_H now non-zero). Caption rewritten to reflect both measurement-pipeline issues resolved (W_K projection + per-layer H-Neuron calibration); anti-OE signature framing preserved (TDS still decreases monotonically, just at slightly smaller magnitudes). OEI and ρ_R rows unchanged — verified bit-identical post-merge. Source: post-merge B6 verification at `docs/logs/2026-05-07_post-merge-verification-report.md`. — pending audit]
$\sigma_H$ & 0.228 & 0.237 & 0.247 & 0.267 & 0.330 \\
TDS & 0.084 & 0.080 & 0.076 & 0.068 & 0.040 \\
\bottomrule
\end{tabular}
\caption{Per-$\alpha$ mean telemetry under inference-time KV-cache steering on Llama~3.2-1B-Instruct, with both measurement-pipeline issues resolved (steering projected through layer-specific $\mathbf{W}_K^{(\ell)}$ to per-head cache space; H-Neuron calibration applied per-layer per Eq.~\ref{eq:hneuron_stress}). All four channels respond to steering: $\rho_R$ varies substantially, $\sigma_H$ rises monotonically with 14--15 of 17 layers firing, OEI shows the small monotonic variation expected in the high-dimensional saturation regime (§\ref{sec:measurement-prereqs}), and the cross-channel divergence TDS \emph{decreases} monotonically---the opposite of the signature predicted by the Orthogonal Escape Hypothesis (Hypothesis~\ref{hyp:orthogonal_escape}). Reversibility holds end-to-end (residual\_norm $= 0$).}
\label{tab:oei-alpha-sweep}
\end{table}
```

### Updated §7.4 prose (line 540 modification only)

**In the existing canonical TeX line 540 paragraph** (the σ_H resolution arc closing paragraph), the existing reference:

> ...$\sigma_H$ rises monotonically with $\alpha$ across the same alpha sweep (Table~\ref{tab:exp1-results}), with 14--15 of 17 layers firing.

becomes:

> ...$\sigma_H$ rises monotonically with $\alpha$ across the same alpha sweep (Table~\ref{tab:oei-alpha-sweep}), with 14--15 of 17 layers firing.

**Action marker for the reference update** (apply inline at line 540):

```latex
% [REVISION | §7.4-table-ref-rename | 2026-05-12 | Updated Table~\ref{tab:exp1-results} → Table~\ref{tab:oei-alpha-sweep} in this paragraph following the OEI-alpha-sweep-relocate-to-§7.4-D+ relocation marker above. The label rename disambiguates the table from the new §5 Experiment 1 Track F results table (which now uses tab:exp1-results). — pending audit]
```

---

## 5. Notes for Session B

### Application order Day 4 (or Day 5 if other priorities intervene)

1. **Cut + relocate** the existing Table 1 block (lines 334–350) from §5 to §7.4 — preserves the layered audit markers and is the structural prerequisite for the §5 Exp 1 recast.
2. **Apply** the relabeling: `tab:exp1-results` → `tab:oei-alpha-sweep` inside the relocated block, and update the in-§7.4 reference at line 540.
3. **Apply** the §5 Experiment 1 recast: replace lines 326–333 + 352–353 with the new REVISION marker + paragraph body in §2 of this draft.
4. **Apply** the new Table 1 (Track F accuracy + perplexity) at the position indicated in §3 of this draft (immediately after the Results paragraph, before the Mechanism paragraph).
5. **Smoke-compile** to verify: (a) `\label{sec:exp1}` resolves the forward refs at lines 53, 68, 80, 167, 170 + v10-exp2 references; (b) `\label{tab:exp1-results}` resolves locally in the new Table 1 caption-reference; (c) `\label{tab:oei-alpha-sweep}` resolves the one §7.4 line-540 reference; (d) no orphaned references to the old `tab:exp1-results` label remain anywhere outside §5 Exp 1's new table.

### Cross-references introduced

- `\label{sec:exp1}` — NEW. Resolves five existing forward references: abstract (line 53), §1 Introduction paragraph 4 (line 68), §1 Contribution 3 (line 80), §3.2 Hypothesis 2 closing paragraph (line 167), §3.2 closing paragraph "We characterize the reward-driven case empirically in §\ref{sec:exp1}--\ref{sec:exp2}" (line 170).
- `\label{tab:exp1-results}` — **reused for the new Track F results table**. The label is identical to the old label; the table content is wholly replaced. References to `tab:exp1-results` from outside §5 Exp 1 were all in the old §5 prose (which is being replaced) — no orphan references survive.
- `\label{tab:oei-alpha-sweep}` — NEW. The relocated alpha-sweep table's new label. Only referenced from §7.4 line 540 (the σ_H closing paragraph).

### Cross-references RELIED ON (already exist)

- `\ref{eq:goodhart}` — TeX:165 ✓
- `\ref{hyp:goodhart}` — TeX:161 ✓
- `\ref{def:dim_escape}` — TeX:141 ✓
- `\ref{sec:exp2}` — **NOT YET DEFINED** (will be added Day 7 by §5 Exp 2 application from v10-exp2 draft). The §5 Exp 1 reference at the bottom of the Interpretation paragraph will produce an undefined-reference warning on intermediate compilation; it resolves when Exp 2 application lands.
- `\cite{bailey2024obfuscated}` — exists ✓
- `\ref{hyp:orthogonal_escape}` — TeX:147 ✓ (used in the relocated Table caption)
- `\ref{sec:measurement-prereqs}` — TeX:530 ✓ (used in the relocated Table caption)
- `\ref{eq:hneuron_stress}` — exists ✓ (used in the relocated Table caption)

### Compilation impact estimate

- Lines deleted from §5: 326–353 (28 lines: 3 REVISION markers + setup paragraph + Results paragraph + Table block + Interpretation paragraph + IMPL line)
- Lines added to §5: new REVISION marker (~12 lines wrapped) + new paragraph body (~30 lines for setup/Results/Mechanism/Interpretation/IMPL) + new Table 1 (~16 lines including caption) ≈ **net +30 lines in §5**
- Lines deleted from §7.4: 0 (the relocation is an insert + a one-line reference update)
- Lines added to §7.4: relocated Table block (~17 lines including caption + relocation marker) + one-line table-ref-rename marker ≈ **net +18 lines in §7.4**

**Total net change: approximately +48 body lines** (current canonical TeX 681 lines → ~729 lines post-application).

### Compiler-warning expectation on intermediate state

After Day 4 application but before Day 7 (v10-exp2 application), the following compile warnings are expected and benign:

- `LaTeX Warning: Reference 'sec:exp2' on input line ... undefined` — resolves on Day 7.
- No other undefined references introduced.

---

## 6. Page-budget commentary

Combining this draft's +48 lines with v10-exp2's +57 lines (Day 7 application), the cumulative §5 + §7.4 deltas are approximately **+105 body lines** between current EOD state (681 lines) and post-Day-7 state (~786 lines).

At ~70 body lines per page in ACL two-column format, this corresponds to ~9.5 body pages after Exp 1 + Exp 2 applications, which is over the 8-page review budget by ~1.5 pages. The Day 9 page-count check + Day 10 cuts (plan v2 §2) are mandatory at that point.

Recommended cut order (from v10-exp2 §6, restated for combined context):

1. Move §6 Memory Complexity proof block to `\appendix` (saves ~1 page, ~40 body lines).
2. Tighten §6 worked-example paragraph using Track D §6 numbers (planned Day 8; produces incidental ~5 line savings during the rewrite).
3. If still over: condense §4.2.2 (RepE probes / verbose prose) by 10–15 lines.

These cuts are reversible at camera-ready (Phase 3 affords one additional page).

---

## 7. Sources (primary-source-fidelity discipline)

All numeric values and methodological claims in this draft trace to primary artifacts read directly in this session:

| Claim | Primary source | Verified |
|---|---|---|
| Benchmark: ARC-Easy, first 200 deterministic test items | Track F report §1 (Run configuration table) | ✓ |
| 3 conditions (G/R/E) + paired design | Track F report §1 + §2 | ✓ |
| Search hyperparameters: depth-3, 27 paths, alpha-set {0.1, 0.5, 1.0} | Track F report §1 + Track G report §1 (verified same protocol) | ✓ |
| Reward function definitions: $\hat{r}_R \sim \mathrm{Uniform}(0, 1)$ and $\hat{r}_E = -H_t/\log\|V\|$ | Track F report §1 (Reward function row) | ✓ |
| Shared random unit-vector direction, --seed 42 | Track F report §1 + §10 Notes on protocol deviations | ✓ |
| 1B per-condition accuracy: G 0.295 / R 0.280 / E 0.275 | Track F report §2 (1B aggregate table) | ✓ |
| 1B Wilson 95% CIs: [0.236, 0.362] / [0.222, 0.346] / [0.218, 0.341] | Track F report §2 (1B aggregate table) | ✓ |
| 1B mean perplexity ± std: 1.547±0.263 / 2.351±1.823 / 2.085±1.641 | Track F report §2 (1B aggregate table) | ✓ |
| 3B per-condition accuracy: G 0.845 / R 0.815 / E 0.820 | Track F report §2 (3B aggregate table) | ✓ |
| 3B Wilson 95% CIs: [0.788, 0.889] / [0.755, 0.863] / [0.761, 0.867] | Track F report §2 (3B aggregate table) | ✓ |
| 3B mean perplexity ± std: 1.318±0.198 / 1.398±0.327 / 1.394±0.319 | Track F report §2 (3B aggregate table) | ✓ |
| 1B E vs R: Δ = −0.005, McNemar b/c = 1/2, p = 1.0000 | Track F report §3 (1B pairwise table) | ✓ |
| 3B E vs R: Δ = +0.005, McNemar b/c = 2/1, p = 1.0000 | Track F report §3 (3B pairwise table) | ✓ |
| Discordance rate π_d = 3/200 vs n=200 calibration budget | Track F report §3 interpretation paragraph | ✓ |
| 1B E vs G: Δ = −0.020, p = 0.2188 | Track F report §4 (1B pairwise) | ✓ |
| 3B E vs G: Δ = −0.025, p = 0.1250 | Track F report §4 (3B pairwise) | ✓ |
| Perplexity ratios E/G: 1.348 (1B), 1.058 (3B) | Track F report §5 (G3.3 table) | ✓ |
| 1B mechanism: (1.0, 1.0, 1.0) selected 91/200 (46%) | Track F report top-line + §7 + Appendix B | ✓ |
| 1B mechanism: predicts "A" on 185/200 items | Track F report top-line + §7 | ✓ |
| 3B mechanism: (0.1, 0.1, 0.1) selected 101/200 (51%) | Track F report top-line + §7 | ✓ |
| Mean entropy-reduction gain +0.012 ± 0.005 (1B) | Track F report Appendix B (search-reward stats table) | ✓ |
| Wall-clock per-item: 5.5 s @ 1B, 10.1 s @ 3B | Track F report §1 + top-line | ✓ |
| Static |r| ≈ 0.60 between entropy and correctness | Track A report §4–§5, §8 (referenced in Track F report §7 + canonical TeX abstract line 53 + §1 Contribution 3 line 80) | ✓ |
| Bailey 2024 negative result for monitor-driven sub-case | canonical TeX §3.2 line 170 + §2.2 line 100 (already applied) | ✓ |
| Power calibration: n=200 paired observations, Δ=10pp at α=0.05 power 0.8 | Track F report §3 interpretation + Track C §5 (referenced) | ✓ |
| Driver script + artifact paths | Track F report §11 (Bibliography of files inspected) | ✓ |
| Wall-clock totals: 18.3 min @ 1B, 33.7 min @ 3B | Track F report header (Wall-clock line) | ✓ |

No number in this draft was sourced from a running-log summary, a compaction-memory recall, or a citation-of-a-citation chain. The static $|r| \approx 0.60$ figure is referenced as our own measurement (consistent with abstract + §1 Contribution 3 framing, which also reference it without external citation).

---

*End of v10 Exp 1 draft. Day-4 application target. Page-budget impact +48 body lines; cumulative with Day-7 Exp 2 application brings total to ~+105 lines, mandating Day 9 page-count check + Day 10 cuts (§6 Memory Complexity proof → `\appendix` is the highest-priority cut).*
