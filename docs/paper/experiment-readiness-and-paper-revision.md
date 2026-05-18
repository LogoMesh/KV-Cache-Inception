# Experiment Readiness and Paper Revision

**Created:** 2026-04-17
**Purpose:** Combined pre-flight checklist + paper claim unlock document.
For each of the five planned experiments: what must be true before running → what it produces → exact paper text that becomes past-tense confirmed once results are in hand.

**How to use:**
- **Before the H100 session:** Open this doc. Work through the Pre-flight gate column. Do not start
  an experiment until every gate in its row is checked.
- **After results come in:** Make exactly the paper edits listed in the Paper unlock column for that
  experiment. No ad-hoc present-tense claims.
- **At the Saturday meeting (Apr 18):** Decide experiment priority order and fill in the Priority
  column. The answer to "which experiments will definitely produce results before May 5?" determines
  what the paper can honestly assert.

**Dependency order:** Experiments are not independent. Experiment 1 produces the telemetry signal
that Experiments 2 and 5 consume. Experiment 3 is independent. Experiment 4 requires a large
dataset that only exists after Experiment 1 or 2 runs at scale. Plan accordingly.

---

## Global Pre-flight Checklist

These must be true before ANY experiment runs on H100. Check once, not per-experiment.

| Gate | Status | Notes |
|---|---|---|
| ρ_R raw dot product fix applied | ✅ Done (commit 4bc8fa9) | `whitebox.py`, `kv_mcts.py` — returns unbounded dot product per Eq. 4 |
| Gate smoke test re-run with corrected ρ_R | ❌ Pending | Run on H100 at session start; confirm ρ_R distribution is qualitatively different from old [0,1] values |
| `classify()` thresholds re-calibrated | ❌ Pending | Use `ClassifyConfig` (added `telemetry_matrix.py`) + `report_telemetry_distribution()` after first H100 run to get empirical ρ_R range; set `alignment_faking_rho_threshold ≈ mean + 0.5·std` |
| H100 access confirmed from Tianyu's lab | ❌ Pending | Required for all Phase B experiments |
| T_t matrix-level export (2×L per node) | ✅ Done | `t_matrix_json` field added — serializes `[[σ_H per layer],[ρ_R per layer]]` as 2×L first-class column |
| LAT probes trained on target model | ❌ Pending | `scripts/train_lat_probes.py`; placeholder 0.5 in export until done |
| Target model (`gpt-oss-20b` or 7B TBD) loaded and verified | ❌ Pending | gpt-oss-20b licence/availability unconfirmed; may need fallback |
| `uv run pytest tests/ -v` green (139 passed) | ✅ Green | Confirm again before H100 session |

**Note on ρ_R recalibration:** The gate smoke test on H100 should include a deliberate diagnostic
pass: run the same prompt at α=0 (baseline) and α=1.0 (perturbed) and check that ρ_R values are
numerically different and that `classify()` produces distinct states. If both return the same
state with the new thresholds, the thresholds need adjustment before any experiment that claims
to detect alignment faking.

---

## Experiment 1 — Orthogonal Escape Detection

*α sweep across steering magnitudes; OEI/TDS curves; core validation of OE hypothesis.*

### Pre-flight gate

| Gate | Status |
|---|---|
| Global checklist complete | — |
| α calibration run: OEI must vary meaningfully across α values | ❌ Pending (Qwen run showed OEI ~0.996 at α=1.0 — not discriminative; determine range where OEI ∈ [0.3, 0.9] before committing to experiment α values) |
| Baseline comparison exists (α=0 run, no steering) | ❌ Pending |
| Honesty steering vector trained for target model | ❌ Pending |

**α calibration note:** The Qwen gate run saturated at OEI ~0.996. Before running Experiment 1,
sweep α ∈ {0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0} on a single prompt and plot OEI vs. α. Find the
range where OEI varies — that is the experiment's α axis. If OEI is monotonically 1.0 across the
full range, the metric is responding to perturbation magnitude, not alignment structure, and the
experiment design needs revision before any paper-valid run.

### What it produces

- OEI and TDS scores per node per α value across steering magnitudes {0.1, 0.5, 1.0, 2.0, 5.0}
- Textual compliance rate per α (behavioral metric)
- σ_H and ρ_R distributions at each α level
- Null-space projection norm `‖Proj_Null(w_hon)(h_t^(l))‖₂` per node
- Expected output shape: table of (α, OEI_mean, OEI_std, TDS_mean, compliance_rate) with ≥3 repeats

### Paper unlock

Once Experiment 1 produces clean, discriminative OEI/TDS curves, the following .tex text can be
confirmed:

**Contribution 3 (line 75–76):**
> *Current (future-tense):*
> "We provide the first formal definition and experimental protocol for Orthogonal Escape---the
> phenomenon whereby applying corrective pressure to a model's latent representations along
> monitored directions forces the model to encode deceptive payloads into orthogonal, unmonitored
> subspaces. We derive testable predictions from this formalization and propose metrics for
> quantifying the degree of orthogonal redistribution."

> *After Experiment 1 (past-tense, confirmed):*
> "We provide the first formal definition and experimental protocol for Orthogonal Escape [...].
> In Experiment 1 (Section 5.1), we confirm the core prediction: at α ∈ {X–Y}, textual
> compliance (ρ_R) increases with steering magnitude while the null-space projection norm
> simultaneously increases (OEI ≥ Z), indicating that deceptive encodings redistribute into
> unmonitored orthogonal subspaces rather than being suppressed."

**Abstract OE claim:**
> *Current:*
> "the formalization of **Orthogonal Escape** [...] together with the first systematic experimental
> protocol for its detection and characterization."

> *After Experiment 1:*
> Add one sentence after the OE formalization: "In our experiments, we confirm this redistribution
> across α ∈ {X–Y} on [model name], observing OEI growth of Z units while behavioral compliance
> is maintained."

**Limitations section (keep honest):**
The Limitations sentence "The Orthogonal Escape hypothesis has not yet been empirically validated
in the specific setting of KV-cache interventions" must be removed or revised once Experiment 1
confirms the hypothesis. If Experiment 1 does NOT confirm it, the sentence stays and Section 0.4
of the claims audit (Option B: reframe OE as emerging observation) should be applied instead.

---

## Experiment 2 — MCTS vs. Text-Space Red-Teaming

*Comparative attack success rates; diversity of discovered vectors; rollout efficiency.*

### Pre-flight gate

| Gate | Status |
|---|---|
| Global checklist complete | — |
| Experiment 1 complete (working telemetry signal established) | ❌ Pending |
| Text-space MCTS baseline implemented (`scripts/run_offline_mcts.py`) | ✅ Exists (Phase A) |
| GCG baseline available (or correctly cited as external result) | ❌ Pending |
| Random-search KV-cache baseline implemented | ❌ Pending |
| Attack success rate metric defined and implemented | ❌ Pending |
| Pairwise cosine distance in intervention space implemented | ❌ Pending |

### What it produces

- Attack success rate comparison: (a) text-space MCTS, (b) KV-cache MCTS (our method),
  (c) GCG, (d) random KV-cache search
- Mean rollouts to first success per method
- Diversity metric: pairwise cosine distance in intervention space
- Expected output shape: 4-row comparison table with ≥3 independent runs per method

### Paper unlock

**Section 2.4 comparative claim (line 99):**
> *Current:*
> "all existing MCTS-based red-teaming operates exclusively in text space"

> *After Experiment 2 (add quantitative support):*
> "all existing MCTS-based red-teaming operates exclusively in text space. In Experiment 2
> (Section 5.2), we find that KV-cache MCTS achieves [X]% higher attack success rate and
> [Y]× greater diversity than text-space MCTS on [model name]."

**Contribution 2 comparative frame (line 73):**
The contrast "rather than in the discrete text space used by all prior MCTS-based red-teaming
approaches" can be strengthened after Experiment 2 to cite the actual performance gap.

**Edit C (GCG paragraph, line 99):**
The rhetorical claim that gradient-based approaches "structurally cannot" navigate continuous
geometry becomes supported by empirical evidence (direct comparison). Replace "structurally cannot"
with "are not designed to navigate" (as recommended in claims audit) regardless of Experiment 2 —
do not wait for this edit.

---

## Experiment 3 — Memory Efficiency

*VRAM profiling; Lipschitz drift; O(M + d·K) claim validation.*

**Note:** Experiment 3 is independent of Experiments 1 and 2. It can run in parallel on H100
as soon as the global pre-flight checklist is complete.

### Pre-flight gate

| Gate | Status |
|---|---|
| Global checklist complete | — |
| 20B model (or target model ≥ 20B) loaded on H100 | ❌ Pending |
| VRAM measurement tooling ready (`torch.cuda.memory_stats()` or equivalent) | ❌ Pending |
| `measure_lipschitz_drift.py` runs correctly at depth d ∈ {3, 5, 10, 20} | ✅ Confirmed (200-cycle gate run passed, drift = 0.00e+00) |

### What it produces

- VRAM consumption curves: branching factor b ∈ {2, 3, 5} × depth d ∈ {3, 5, 10, 20}
  — reversible MCTS vs. standard parallel MCTS
- Wall-clock overhead of rollback mechanism at each (b, d) point
- Lipschitz drift (perplexity deviation) with and without FP32 accumulators over extended
  search trajectories (already confirmed 0.00e+00 at 200 cycles on 1B; re-confirm at 20B scale)
- Expected output shape: VRAM(b, d) comparison table; overhead(b, d) table; drift plot

### Paper unlock

**Contribution 2 capability claim (line 73):**
> *Current (unhedged, stated as present fact):*
> "This enables deep MCTS exploration on a single 80GB GPU for models up to 20B parameters."

> *After Experiment 3 (confirmed or revised):*
> If confirmed: keep the claim; add "(Experiment 3, Section 5.3; peak VRAM at d=10, b=3: [X]GB)".
> If not confirmed at 20B: revise to the largest model scale actually validated.

**Contribution 2 memory reduction claim (line 73):**
> *Current:*
> "reduces the memory complexity of latent-space MCTS from $O(b^d \cdot M)$ to $O(M + d \cdot K)$"

> *This is a mathematical derivation — confirm the algebra in the paper body (Theorem 1 section)
> is complete and correct before Experiment 3. Experiment 3 then provides empirical confirmation.*

**Paper: ~50MB accumulator overhead claim:**
> *Current (unverified, line 11 in claims inventory):*
> "accumulators add ~50MB overhead for 20B model with S'=10 positions"

> *After Experiment 3:* Replace with measured figure from H100 run.

---

## Experiment 4 — Evaluation Reproducibility

*500-artifact LLM-as-Judge comparison; inter-rater agreement; correlation with human judgments.*

**Note:** Experiment 4 requires a dataset of 500 artifacts, which does not yet exist. This
experiment is gated on Experiments 1 or 2 running at scale to generate sufficient records.
Timeline risk: this is the last experiment to be ready and the one most likely to be deferred.

### Pre-flight gate

| Gate | Status |
|---|---|
| Global checklist complete | — |
| 500+ MCTS trace artifacts exist from Phase B runs | ❌ Pending (currently 27 records) |
| GPT-4 API access confirmed | ❌ Pending |
| Claude 3.5 API access confirmed | ❌ Pending |
| Human expert panel (3 reviewers) identified and available | ❌ Pending |
| Cohen's κ inter-rater calculation implemented | ❌ Pending |

### What it produces

- Pass@1 against ground-truth in Docker sandbox (requires ground-truth labels — see data
  requirements doc, `ground_truth_label` field)
- Cosine similarity of semantic intent: generated output vs. task specification
- GPT-4 scores: 5 independent runs, inter-run variance
- Claude 3.5 scores: 5 independent runs, inter-run variance
- Human expert panel: 3 raters, Cohen's κ
- Correlation matrix: all methods vs. human panel
- Expected output shape: 5-method comparison table with variance and κ

### Paper unlock

**Section 2.5 LLM-as-Judge claim (line 103):**
> *Current:*
> "score divergences of $0.10$--$0.20$ on normalized scales across independent runs"
> *(cited as [llmasjudge2025limitations] — verify this citation states this figure)*

> *After Experiment 4:* Add comparison of our own inter-run variance to the cited 0.10–0.20
> figure: "We observe inter-run variance of [X] across 5 GPT-4 runs and [Y] across 5 Claude 3.5
> runs, consistent with/exceeding the reported range in [citation]."

**Evaluation framework claims (Section 5.4):**
All pass@1, cosine similarity, and inter-rater statistics become confirmed once Experiment 4 runs.
Current language ("we evaluate using...") becomes "we evaluated using...".

---

## Experiment 5 — Cross-Model Transfer (Procrustes)

*Adversarial interventions on 1B surrogate; transfer to 7B and 20B via Procrustes-aligned vectors.*

### Pre-flight gate

| Gate | Status |
|---|---|
| Global checklist complete | — |
| Experiment 1 complete (adversarial vectors generated on 1B model) | ❌ Pending |
| 7B target model selected (currently TBD in CLAUDE.md) | ❌ Pending |
| Procrustes alignment implementation in `scripts/train_lat_probes.py` working | ⚠ Exists (Phase A) — validate at 7B |
| Transfer evaluation metric implemented | ❌ Pending |

### What it produces

- Attack success rate on 1B model (from Experiment 1 vectors)
- Transfer success rate to 7B model with Procrustes-aligned steering vectors
- Transfer success rate to 20B model
- Degradation curve: success rate vs. model scale
- Telemetry signature preservation: OEI/TDS similarity between source (1B) and target (7B/20B)
- Expected output shape: 3-row (1B, 7B, 20B) table; degradation plot

### Paper unlock

**Phase A cross-model alignment theory (line 269):**
> *Current:*
> "Cross-model alignment theory via supervised Procrustes alignment between 1B and 7B model
> representation spaces."

> *After Experiment 5:* Replace "theory" with measured transfer results.

**Contribution 2 (implicit cross-model framing):**
> *After Experiment 5:* Add to Contribution 2: "We demonstrate that steering vectors generated
> on a 1B surrogate transfer to [target model] with [X]% of original attack success rate via
> Procrustes-aligned projection, enabling low-cost adversarial evaluation on large models."

---

## Priority Decision (Fill at Saturday Meeting)

| Experiment | Dependencies | Owner | Priority | Will run before May 5? |
|---|---|---|---|---|
| Exp 1 — OE Detection | Global preflight + α calibration | | | |
| Exp 2 — MCTS vs. Text-Space | Exp 1 + baselines | | | |
| Exp 3 — Memory Efficiency | Global preflight only (independent) | | | |
| Exp 4 — Reproducibility | Exp 1 or 2 at scale + 500 artifacts | | | |
| Exp 5 — Cross-Model Transfer | Exp 1 + 7B model | | | |

**The scope question (from claims audit Section 0.5):**
> "Which of the five experiments will definitely produce results before May 5?"

The answer to that question determines the paper. Fill this table at the meeting, then apply the
corresponding scope decision from Section 0 of the claims audit (Option A: 2 contributions + 3
experiments, Option B: dataset-forward framing, or other).

---

## Appendix: Claim-to-Experiment Mapping

Which paper claim is unlocked by which experiment:

| Claim | Experiment that confirms it | Current status |
|---|---|---|
| "enables deep MCTS on single 80GB GPU for 20B parameters" | Exp 3 | ❌ Unrun — reframe to projected |
| "OEI and TDS curves discriminate alignment-faking states" | Exp 1 | ❌ Unrun — OEI saturated in gate run |
| "KV-cache MCTS outperforms text-space MCTS on attack success" | Exp 2 | ❌ Unrun |
| "memory complexity: O(b^d·M) → O(M+d·K)" | Math (verify in paper) + Exp 3 | ⚠ Math verify pending |
| "FP32 accumulator: zero semantic degradation" | Gate run ✅ | ✅ Confirmed (200 cycles, 0.00e+00) |
| "OE hypothesis confirmed: null-space redistribution measured" | Exp 1 | ❌ Unrun — hypothesis only |
| "LLM-as-Judge inter-run variance vs. human correlation" | Exp 4 | ❌ Unrun |
| "cross-model transfer via Procrustes alignment" | Exp 5 | ❌ Unrun |
| "~50MB accumulator overhead for 20B, S'=10" | Exp 3 | ❌ Verify calculation |
| "20B model KV cache ≈ 40GB in bf16" | Math calculation | ✅ Calculable — keep |

---

## Update Log

| Date | What changed |
|---|---|
| 2026-04-17 | Initial version created from session log plan, claims audit, and data requirements doc |
