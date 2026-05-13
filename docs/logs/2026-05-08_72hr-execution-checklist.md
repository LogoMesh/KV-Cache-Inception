# 72-Hour Execution Checklist — Candidate A System-2 Pivot

**Created:** 2026-05-08
**Start time (T+0):** when Josh kicks off Pre-Flight section below
**End time (T+72):** ~2026-05-11 EOD or 2026-05-12 AM
**Buffer remaining post-T+72:** ~14 days to 2026-05-25 EMNLP deadline

---

## 0. Plasticity principle

This checklist is **not** a script to be executed blindly. It is a series of **information-gathering tasks** with **three decision gates** at T+24, T+48, T+72. Each gate evaluates whether evidence accumulated since the previous gate still supports continuing on Candidate A (System-2 Latent Search) or whether a pivot is warranted.

**Gate decisions are GO / NO-GO / PIVOT.** Do not blindly continue past a NO-GO or PIVOT gate.

**Pile-up thresholds** (defined upfront so they can be measured rather than felt):

| Threshold | Trigger | Gate response |
|---|---|---|
| 0–1 contradicting findings | Citation/empirical nuance is normal | CONTINUE (log nuance, adjust framing inline) |
| 2 contradicting findings | Adjust framing materially | YELLOW — review at next gate, not mid-track |
| 3+ contradicting findings | Plan structurally unsound | NO-GO at next gate; consider PIVOT to wildcard / hallucination / workshop |
| 1 fatal finding | Load-bearing claim falsified | IMMEDIATE NO-GO regardless of other findings; stop all tracks pending re-plan |

**Fatal findings are pre-defined**:
1. COCONUT (Hao et al. 2025) actually does training-free latent search — kills our differentiator.
2. Certainty probe at 1B/3B is degenerate (constant or random) AND fallback (per-token entropy) also degenerate — kills the reward signal.
3. No reasoning benchmark exists at 1B/3B with measurable headroom — kills the empirical claim.
4. Memory complexity numbers measured by Track D contradict §6 by >2x — kills the "memory bound" headline.
5. Three or more cited papers in Track B return "actually claims something stronger than we say" — kills our incremental-novelty claim across multiple anchors at once.

---

## 1. Track structure (parallel where dependencies allow)

| Track | Owner | Function | Output file |
|---|---|---|---|
| **A** | Diagnostic Claude session #1 | Certainty probe validation at 1B/3B | `docs/logs/2026-05-09_track-A-certainty-probe-report.md` |
| **B** | Diagnostic Claude session #2 | Citation verification (load-bearing claims only, against local PDF/source) | `docs/logs/2026-05-09_track-B-citation-audit.md` |
| **C** | Diagnostic Claude session #3 | Benchmark calibration: baseline accuracy on 4–6 candidate reasoning benchmarks | `docs/logs/2026-05-10_track-C-benchmark-calibration.md` |
| **D** | Diagnostic Claude session #4 | VRAM measurement: actual constants for §6 Memory Complexity Proposition | `docs/logs/2026-05-10_track-D-vram-measurements.md` |
| **E** | Session A (this session) | Synthesis at each gate; paper-text impact; gate decisions | `docs/logs/2026-05-11_track-E-gate-decisions.md` |
| **F** | Diagnostic Claude session #5 | Negative-control random-perturbation MCTS run (gates COCONUT differentiation) | `docs/logs/2026-05-11_track-F-negative-control-report.md` |

**Dependency map** (→ means "depends on"):
- Track C ← Track B-#1 (COCONUT verification); benchmark choice may shift if COCONUT differentiation collapses
- Track F ← Track A (certainty probe must work before negative control is meaningful); Track C (benchmark must be selected first)
- Track E (gates) ← all other tracks at corresponding hours

Tracks A, B, D run in parallel from T+0 with no upstream dependencies.

---

## 2. Pre-flight (T-2 hours before T+0)

Do these BEFORE starting the 72-hour clock. They prevent silent failures during diagnostic sessions.

| # | Action | Owner | Expected duration |
|---|---|---|---|
| P1 | Verify GPU available and `meta-llama/Llama-3.2-1B-Instruct` + `Llama-3.2-3B-Instruct` checkpoints loadable | Josh | 15 min |
| P2 | Verify `whitebox.py` has `PerLayerCertaintyProjector` (or equivalent); if not, surface the gap explicitly | Josh | 10 min — `grep -n "Certainty\|certainty" logomesh/whitebox.py` |
| P3 | Verify `external/Logomesh/` submodule still mounted; commit `.gitmodules` change if you want it tracked, or `git submodule deinit` if you want to back it out | Josh | 5 min |
| P4 | Confirm `docs/NeurIPS/Citations/` directory and the 21 arXiv source bundles are accessible to a fresh Claude session (no permission issues) | Josh | 5 min |
| P5 | Decide whether to commit the v7+v8 drafts files and premortem-v2 TeX before T+0, or leave staged. Recommendation: commit them, so the 72-hour state is reproducible from a single git ref | Josh | 5 min |
| P6 | Set T+0 timestamp explicitly in this file (replace placeholder below) | Josh | 1 min |

**T+0 timestamp**: `__________________________` (Josh fills in)

---

## 3. Hour-by-hour task matrix

| Hour | Track | Task | Status gate |
|---|---|---|---|
| T+0  | A | Spawn Diagnostic Session #1 (prompt §5.1) | — |
| T+0  | B | Spawn Diagnostic Session #2 (prompt §5.2) | — |
| T+0  | D | Spawn Diagnostic Session #4 (prompt §5.4) | — |
| T+0  | E | Notify Session A that tracks A/B/D are dispatched | — |
| T+4  | A | Track A interim status check; abort early if `whitebox.py` certainty probe doesn't even load | quick check |
| T+8  | B | Track B status: at least 3 of 8 priority citations verified by now | quick check |
| T+12 | A | Track A first signal-magnitude readout (does certainty vary across prompts?) | quick check |
| T+18 | B | Track B priority batch (8 citations) complete | hard checkpoint |
| T+18 | A | Track A correlation-with-correctness readout | hard checkpoint |
| T+18 | D | Track D first VRAM measurement (b=3, d=3) | hard checkpoint |
| T+24 | E | **GATE 1** — Synthesis of A/B/D outputs; GO/NO-GO/PIVOT decision (§4.1) | **GATE** |
| T+24 | C | If GATE 1 = GO: spawn Diagnostic Session #3 (prompt §5.3) | conditional |
| T+30 | C | Track C first benchmark baselines (greedy at 1B/3B on 4 benchmarks) | quick check |
| T+36 | D | Track D second VRAM measurement (b=3, d=5) | quick check |
| T+40 | C | Track C narrows to top 2 benchmark candidates | quick check |
| T+44 | D | Track D third measurement (b=3, d=10) and proposition rewrite | hard checkpoint |
| T+48 | E | **GATE 2** — Synthesis of C/D outputs + cumulative A/B updates; GO/NO-GO/PIVOT (§4.2) | **GATE** |
| T+48 | F | If GATE 2 = GO: spawn Diagnostic Session #5 (prompt §5.5) | conditional |
| T+54 | F | Track F first cycle: random-perturb baseline numbers ready | quick check |
| T+60 | F | Track F certainty-guided MCTS comparison numbers ready | quick check |
| T+66 | F | Track F final: certainty-guided vs random comparison with statistical-significance check | hard checkpoint |
| T+66 | E | Session A drafts paper-text impact summary (what changes in TeX as a result of this 72-hour run) | hard checkpoint |
| T+72 | E | **GATE 3** — Final GO/NO-GO/PIVOT decision; if GO, write 14-day execution plan (§4.3) | **GATE** |

**Notes on the matrix:**
- "Quick check" = Josh skims the diagnostic-session output; no formal review.
- "Hard checkpoint" = Josh reads the relevant track's output file and either acknowledges or flags a concern. Concerns are noted in `track-E-gate-decisions.md` running notes.
- Diagnostic sessions running in parallel may exceed their wall-clock budgets. The matrix above is **target-time**, not a hard deadline. Slip ≤ 4h on any task is acceptable; slip > 4h triggers a check-in.
- **Tracks B and D use no GPU.** They can run on CPU, including in parallel with other tracks that DO need GPU. Track A (certainty probe) and Track C (benchmark calibration) will compete for GPU; sequence them or partition the GPU if possible.

---

## 4. Decision gates

### 4.1 Gate 1 (T+24) — "Is the foundation viable?"

**Inputs**:
- `track-A-certainty-probe-report.md` (full read)
- `track-B-citation-audit.md` (priority batch — 8 load-bearing citations)
- `track-D-vram-measurements.md` (first measurement only; full data lands at T+44)

**Pass criteria** (ALL must be true for GO):

| # | Criterion | Falsifier |
|---|---|---|
| G1.1 | Certainty signal varies meaningfully across prompts at 1B (signal range > noise floor by 3x) | Constant or random certainty values |
| G1.2 | Certainty correlates with correctness on factual recall (Pearson r > 0.15, or Spearman > 0.15) | r < 0.10 with no fallback resolved |
| G1.3 | Track B priority batch (8 citations) returns ≤ 2 "nuanced/mismatched" verdicts | 3+ nuanced/mismatched |
| G1.4 | Track B has NO fatal findings (especially: COCONUT does NOT do training-free latent search) | COCONUT or other fatal verdict |
| G1.5 | Track D first measurement broadly consistent with §6 numbers (within 50%) | Off by >2x |

**If GO**: spawn Track C; continue per matrix.

**If YELLOW (1–2 weak criteria)**: Session A drafts a "framing adjustment" note before T+30. Continue but with adjusted scope.

**If NO-GO (3+ weak criteria, OR any fatal finding)**: STOP. Session A produces a "pivot decision memo" within 4h evaluating: (a) wildcard (KV-cache working memory), (b) round-2 hallucination pivot, (c) NeurIPS workshop / journal pivot. Josh decides direction by T+30.

### 4.2 Gate 2 (T+48) — "Is the empirical bet realistic?"

**Inputs**:
- `track-C-benchmark-calibration.md`
- `track-D-vram-measurements.md` (full data: 3 measurements + proposition rewrite)
- Cumulative A/B updates if any nuance has emerged

**Pass criteria** (ALL must be true for GO):

| # | Criterion | Falsifier |
|---|---|---|
| G2.1 | At least 1 benchmark with 1B greedy baseline ≥ 30% AND headroom ≥ 10 absolute points to ceiling | All candidates < 30% baseline OR no headroom |
| G2.2 | At 3B, the same or different benchmark with baseline ≥ 40% AND headroom ≥ 10 points | None at 3B either |
| G2.3 | Statistical significance reachable on ≤ 200 examples per condition (effect size detectable) | Need 1000+ examples for power |
| G2.4 | VRAM measurement supports memory-bound headline (constant factor 2–3·M_KV at consumer scale) | Off by >2x or scaling-dependent in unexpected ways |
| G2.5 | No new contradicting findings in Track B that emerged late (e.g., a citation that came back nuanced after T+24) | 1+ late-arriving nuance |

**If GO**: spawn Track F; commit to benchmark choice in writing.

**If YELLOW**: Session A drafts framing adjustment AND alternate-benchmark fallback before T+54.

**If NO-GO**: PIVOT. Most likely target = Wildcard (KV-cache working memory consolidation) since it's less benchmark-sensitive at 1B. Session A produces wildcard-pivot plan within 6h. Josh decides within 12h.

### 4.3 Gate 3 (T+72) — "Is the empirical claim survivable under adversarial review?"

**Inputs**:
- `track-F-negative-control-report.md`
- Session A's paper-text impact summary
- All cumulative track findings

**Pass criteria** (ALL must be true for GO):

| # | Criterion | Falsifier |
|---|---|---|
| G3.1 | Certainty-guided MCTS beats random-perturbation MCTS on the chosen benchmark by ≥ 2 absolute points | Comparable or random wins |
| G3.2 | Difference is statistically significant at p < 0.05 (or comparable Bayesian readout) | Not significant |
| G3.3 | Certainty-guided MCTS does not catastrophically destroy fluency (perplexity within 2x baseline) | Perplexity blowup |
| G3.4 | The paper-text impact (Session A's summary) is achievable in 14 days | Requires more than 14 days |
| G3.5 | No fatal late-arriving findings from Tracks A/B (cumulative) | 1+ fatal |

**If GO**: commit to Candidate A. Session A drafts 14-day execution plan (T+72 to 2026-05-25). Submit on track.

**If YELLOW**: GO with a "scope downgrade" — drop the comparison-against-Best-of-N claim or similar; ship a smaller paper.

**If NO-GO**: HARD PIVOT. Three options remain — Wildcard, Hallucination, Workshop. Session A produces decision matrix; Josh decides within 24h. We've still got 14 days; a hallucination pivot can ship in that window.

---

## 5. Diagnostic Session prompts

Each session should be spawned as a fresh Claude (Opus 4.7 or 4.6) instance with the prompt below as its initial message. Each session produces ONE output file at the listed path. **Documentation/logging is not optional** — every prompt explicitly requires it.

### 5.1 Diagnostic Session #1 — Track A: Certainty Probe Validation

```
You are a diagnostic Claude session. Your single task is to validate whether the
RepE certainty probe at logomesh/whitebox.py produces a usable signal at the
1B and 3B Llama scales, and to record everything you do in a documented report.

CONTEXT FILES (read in order):
1. CLAUDE.md (project overview)
2. logomesh/whitebox.py (the probe implementation)
3. logomesh/local_model.py (model interface)
4. tests/test_whitebox.py (existing tests)
5. docs/logs/2026-05-08_EMNLP Framing for KV-Cache MCTS.md §5 (why this validation matters)

TASK:
Step 1 — Inventory: identify whether `PerLayerCertaintyProjector` (or equivalent
certainty extraction) exists in whitebox.py. If not, identify the closest
analogue and assess what would be needed to construct one. Save findings to the
output file.

Step 2 — Calibration test: if a certainty projector exists, calibrate it on
a small contrastive set (10 known-confident vs 10 known-uncertain prompts).
Confidence proxies you may use: factual recall ("What is the capital of France?"
→ confident; "What was the population of Lyon in 1342?" → uncertain). Document
the calibration set in the report verbatim.

Step 3 — Signal magnitude check: at 1B, run 50 prompts spanning factual /
ambiguous / nonsensical regimes. Record per-prompt certainty signal (max,
mean across layers). Compute: signal range, std deviation, ratio of range to
noise floor. Save raw numbers to the report.

Step 4 — Correlation with correctness: on a held-out factual recall set
(20–50 questions with ground truth answers), record: model's greedy answer,
correctness (binary), certainty signal at last token. Compute Pearson r and
Spearman rho between certainty and correctness. Save raw data + correlation
to the report.

Step 5 — Repeat at 3B. Document any differences from 1B.

Step 6 — Fallback assessment: if certainty signal degenerate (constant,
random, or r < 0.10), validate per-token entropy as fallback. Same protocol:
calibrate, measure signal range, correlate with correctness.

OUTPUT FILE: docs/logs/2026-05-09_track-A-certainty-probe-report.md

REQUIRED REPORT SECTIONS:
1. Inventory finding (does the probe exist? form?)
2. Calibration set (verbatim prompts)
3. Signal magnitude at 1B (raw numbers + summary stats)
4. Correlation with correctness at 1B (raw data + r/rho)
5. Same for 3B
6. Fallback assessment (entropy as proxy)
7. Pass/fail verdict per Gate 1 criteria G1.1 and G1.2 in the master checklist
8. Plasticity recommendation: if certainty fails AND entropy works → use entropy;
   if both fail → flag as fatal (Gate 1 fatal finding #2)
9. Bibliography of any code paths inspected (file:line references)

DO NOT:
- Edit logomesh/* code (read-only on this track)
- Run anything that takes > 30 minutes wall clock
- Defer documentation; if you discover something, write it down immediately
- Report "looks fine" without numerical evidence
```

### 5.2 Diagnostic Session #2 — Track B: Citation Verification

```
You are a diagnostic Claude session. Your task is to verify that the
load-bearing citations in our paper (under the new Candidate A "System-2
Latent Search" framing) accurately reflect what the cited papers actually
claim. The user has downloaded every cited paper to docs/NeurIPS/Citations/.
Your job is to read each one and produce an audit report.

CONTEXT FILES:
1. docs/logs/2026-05-08_EMNLP Framing for KV-Cache MCTS.md (the proposed framing)
2. docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2-premortem-v2.tex
   (current paper state)
3. docs/NeurIPS/Citations/ (PDFs and arXiv source bundles)
4. docs/NeurIPS/Citations/_extracted/ (some pre-extracted text)

PRIORITY CITATIONS (verify these first, in this order):
1. Hao et al. 2025 — "Training Large Language Models to Reason in a
   Continuous Latent Space" (COCONUT). Likely arXiv-2412.06769. Required
   verification: does COCONUT actually train the model? Does it operate in
   KV-cache space or residual stream? Is exact reversibility a property
   COCONUT claims? Our framing claims we are training-free and reversible —
   confirm we differentiate cleanly. **THIS IS A FATAL-FINDING CHECK.**

2. Snell et al. 2024 — "Scaling LLM Test-Time Compute Optimally..."
   (likely NOT in Citations folder; flag as missing if so). What's the
   actual claim about test-time compute? Discrete or continuous? Sample-
   based or search-based? Our paper would frame it as foundation for
   inference-time search — confirm fit.

3. Zou et al. 2023 — RepE (arXiv-2310.01405). What probes are validated?
   For honesty specifically. Does the paper provide a "certainty"
   formulation, or is that derived? Our paper claims we are using their
   probes — confirm whether certainty is in their framework or a stretch.

4. Gao et al. 2025 — H-Neurons (arXiv-2512.01797 likely). What's the actual
   causal claim about H-Neurons? Hallucination only, or broader (over-
   compliance, sycophancy)? Our paper now uses σ_H as a "stress" signal in
   a System-2 framing — confirm or flag as nuanced.

5. Hasani et al. 2025 — "Mechanistic Interpretability of Large-Scale
   Counting in LLMs through a System-2 Strategy" (arXiv-2601.02989).
   What is their actual System-2 claim? Search-based, or behavioral? How
   do they connect mechanistic interpretability to System-2?

6. Stoehr et al. 2024 — "Activation Scaling for Steering and Interpreting
   Language Models" (EMNLP Findings 2024, arXiv-2410.04444 likely).
   What's the steering method? Static vector add, or something more
   dynamic? What's our differentiator from them?

7. Bailey et al. 2024 — "Obfuscated Activations Bypass LLM Latent-Space
   Defenses" (arXiv-2412.09565). Already cited extensively. Under the new
   System-2 framing, does the Bailey relationship still hold (i.e.,
   "complementary measurement"), or does the framing shift change what's
   actually relevant?

8. Yu et al. 2025 — MUSE (EMNLP main 2025, aclanthology 2025.emnlp-
   main.1080). Likely 2025.emnlp-main.323.pdf or coling-71.txt — verify
   identification. What's the multi-turn MCTS algorithm? Does it operate
   on text or activations? How does our continuous-latent-search differ?

ADDITIONAL CITATIONS (if time permits, lower priority):
9. SHARP (Wu et al. 2025), SteerVLM (Sivakumar et al. 2025), Greenblatt
   et al. 2024 (alignment faking), Hubinger et al. 2024 (sleeper agents),
   Gupta & Jenner 2025 (RL-Obfuscation), Zheng et al. 2023 (LLM-as-Judge),
   Liu et al. 2023 (G-Eval), Hao et al. 2024 (COCONUT predecessor if
   different from #1), and any others present in Citations/.

PROTOCOL FOR EACH CITATION:
1. Locate the paper file. If not present, mark "MISSING — needs download".
2. Read abstract + introduction + main claim sections. ~15 min per paper.
3. Extract the precise claim our paper builds on, verbatim from the cited
   paper, with section reference.
4. Compare to how our paper currently frames the citation (premortem-v2.tex).
5. Verdict: CLEAN (matches cleanly), NUANCED (cited claim is more
   limited/different than our usage), or MISMATCHED (citation does not
   support our usage at all).
6. If NUANCED or MISMATCHED, draft a one-sentence revision to our
   paper's relevant text that would resolve the issue.

OUTPUT FILE: docs/logs/2026-05-09_track-B-citation-audit.md

REQUIRED REPORT SECTIONS:
1. Per-citation table: Name | arXiv ID | File location | Verdict | Notes
2. Detailed entry for each citation with verbatim quote + comparison +
   suggested revision (if NUANCED or MISMATCHED)
3. Summary of fatal findings (if any) — ESPECIALLY whether COCONUT does
   training-free latent search (Gate 1 fatal #1)
4. Summary of pile-up: count of NUANCED + MISMATCHED across all
   verified citations (per master checklist plasticity rule)
5. List of MISSING citations that need to be downloaded before the
   paper can be safely revised
6. List of citations we should ADD (per Gemini round-2 zeitgeist
   characterization: SAE survey, RouteSAE, MUSE if not already, Stoehr,
   Hao et al. 2025 COCONUT, Snell et al. 2024) — note which are
   mandatory vs nice-to-have

DO NOT:
- Edit the .tex file directly (Session A handles paper text)
- Skip a citation because "it looks fine" — read it
- Conclude CLEAN without a verbatim quote
- Defer documentation
```

### 5.3 Diagnostic Session #3 — Track C: Benchmark Calibration

```
You are a diagnostic Claude session. Your single task is to determine which
reasoning benchmark, at the 1B and 3B Llama scales, gives us the cleanest
empirical bed for showing that certainty-guided KV-cache MCTS improves over
greedy decoding. Document everything.

CONTEXT FILES (read first):
1. docs/logs/2026-05-08_72hr-execution-checklist.md (master checklist; this section)
2. docs/logs/2026-05-08_EMNLP Framing for KV-Cache MCTS.md (Candidate A
   experimental design)
3. docs/logs/2026-05-09_track-A-certainty-probe-report.md (will exist by
   the time you start; tells you what reward signal we're working with)
4. CLAUDE.md (model setup, hardware envelope)

CANDIDATE BENCHMARKS (test all, in order):
1. ARC-Easy (commonsense QA) — 1B baseline expected ~50–60%
2. SciQ (science MCQ) — 1B baseline expected ~50–70%
3. HellaSwag (sentence completion) — 1B baseline expected ~40–50%
4. AQuA-RAT subset (numerical reasoning, MCQ format) — 1B baseline ~20–30%
5. ProofWriter (logical entailment) — 1B baseline ~30–50%
6. LogiQA-Easy (logical reasoning MCQ) — 1B baseline ~25–35%

PROTOCOL:
For each benchmark:
1. Load 100 examples from the standard test split.
2. Run greedy decoding at 1B and 3B; record accuracy.
3. Identify the questions on which the 1B model is *almost right* —
   plausible alternative answers, high entropy, etc. These are the
   headroom questions.
4. Estimate the headroom: if a perfect search-guided method got the
   high-entropy questions right, what would accuracy be?
5. Compute statistical-power requirement: how many examples needed to
   detect a 5-point absolute improvement at p < 0.05?

PASS THRESHOLDS PER BENCHMARK:
- 1B baseline ≥ 30%: viable
- Headroom ≥ 10 absolute points: viable
- Power requirement ≤ 200 examples: viable

OUTPUT FILE: docs/logs/2026-05-10_track-C-benchmark-calibration.md

REQUIRED REPORT SECTIONS:
1. Per-benchmark table: Name | 1B greedy acc | 3B greedy acc | Estimated
   headroom | Power requirement | Verdict
2. Top-2 recommended benchmarks with reasoning
3. Specific subset of test examples to use for the actual experiment
   (~200 examples per condition, balanced)
4. Time/compute estimate for the full experiment under each top-2 choice
5. Pass/fail verdict per Gate 2 criteria G2.1, G2.2, G2.3
6. Plasticity recommendation: if NO benchmark passes thresholds → flag
   as Gate 2 fatal #3 in master checklist; recommend wildcard pivot

DO NOT:
- Run the full experiment yet (this is calibration only)
- Use more than 100 examples per benchmark for the calibration sweep
- Pick a benchmark before finishing the sweep, even if early signal looks good
- Defer documentation
```

### 5.4 Diagnostic Session #4 — Track D: VRAM Measurement (formerly Max's lane)

```
You are a diagnostic Claude session. Your single task is to measure
actual VRAM usage of the Reversible KV-Cache MCTS at three configurations
and produce data sufficient to rewrite the §6 Memory Complexity Proposition
numerical example, replacing the current placeholder constants. Document
everything.

CONTEXT FILES:
1. docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2-premortem-v2.tex
   (current §6 numbers — note 40.05 GB / 9.7 TB headline numbers are
   placeholders pending direct measurement)
2. docs/logs/2026-05-05_recast-action-plan.md §4.6 row 168 (the
   coordinated-update item this resolves)
3. logomesh/kv_mcts.py (the algorithm being measured)
4. scripts/run_kv_mcts.py (the runner)
5. CLAUDE.md (hardware envelope)

PROTOCOL:
1. Run `scripts/run_kv_mcts.py --model meta-llama/Llama-3.2-1B-Instruct
   --branches 3 --depth 3 --nodes 27`. Record peak VRAM via
   `nvidia-smi --query-gpu=memory.used`. Run 3 times, take median.
2. Repeat at depth 5, branches 3, nodes 81.
3. Repeat at depth 10, branches 3, nodes ~1700 (if VRAM permits; if
   OOM, document the OOM and note the practical depth ceiling).
4. For each config, separately measure:
   - Base model + KV cache (no MCTS): baseline
   - With MCTS but accumulator disabled (random-perturb mode): mid
   - With MCTS + accumulator enabled (full reversible): full
5. Compute: constant factor (full / baseline). The §6 proposition
   currently asserts O(M_KV + d·K_acc); the measured constant should
   be 2–3·M_KV per the §5 Implementation Notes item (iv).

OUTPUT FILE: docs/logs/2026-05-10_track-D-vram-measurements.md

REQUIRED REPORT SECTIONS:
1. Hardware reference (GPU model, driver, CUDA version)
2. Per-config table: branches | depth | nodes | baseline VRAM | mid VRAM
   | full VRAM | constant factor (full / baseline)
3. Replacement text for the §6 numerical example (the "9.7 TB vs 40.05 GB"
   sentence) using actual measured numbers at the largest config that
   ran successfully
4. Comparison to standard parallel-cache MCTS theoretical memory
   (b^d · M_KV) — show the actual savings ratio
5. If OOM occurred: document the practical depth/branching ceiling on
   the test hardware
6. Pass/fail verdict per Gate 2 criterion G2.4
7. Plasticity recommendation: if measured constant > 5·M_KV (i.e., much
   worse than the 2–3·M_KV claim) → flag as Gate 2 fatal #4 in master
   checklist

DO NOT:
- Run a full experiment (this is memory measurement only)
- Skip the OOM measurement if encountered (it's data)
- Edit logomesh/* or §6 directly (Session A handles text)
- Defer documentation
```

### 5.5 Diagnostic Session #5 — Track F: Negative Control (random-perturbation MCTS)

```
You are a diagnostic Claude session spawned only after Gate 2 = GO.
Your single task is to determine whether certainty-guided MCTS beats
random-perturbation MCTS on the chosen benchmark cleanly enough to
defend the Candidate A framing under adversarial review. This is the
"are we sure this is reasoning, or just numerical noise?" check.

CONTEXT FILES:
1. docs/logs/2026-05-08_72hr-execution-checklist.md (master checklist;
   §4.3 Gate 3 criteria — these are what you must satisfy)
2. docs/logs/2026-05-09_track-A-certainty-probe-report.md (signal we're
   using; whether it's certainty or fallback entropy)
3. docs/logs/2026-05-10_track-C-benchmark-calibration.md (chosen
   benchmark + ~200-example subset)
4. logomesh/kv_mcts.py + logomesh/telemetry_matrix.py (the algorithm
   + reward function)
5. scripts/run_kv_mcts.py (runner)

PROTOCOL:
1. From Track C output, take the chosen benchmark and the recommended
   ~200-example subset.
2. Run THREE conditions on each example (greedy decoding as anchor,
   plus two MCTS variants):
   a. Greedy baseline (model.generate, no MCTS)
   b. Random-perturbation MCTS: same MCTS engine, reward = uniform
      random in [0, 1] per node (i.e., search exists but reward
      provides no signal)
   c. Certainty-guided MCTS: same MCTS engine, reward = certainty
      (or fallback per Track A)
3. Configuration: branches=3, depth=3 (modest; we want signal not
   compute saturation). Wallclock per example budget: ≤30 sec.
4. Record per-example: condition, final answer, correctness, perplexity
   of the generated answer, certainty/reward signal at terminal node.
5. Compute aggregate: accuracy by condition, perplexity by condition,
   per-pair statistical significance (paired t-test or McNemar's,
   accuracy is binary).

OUTPUT FILE: docs/logs/2026-05-11_track-F-negative-control-report.md

REQUIRED REPORT SECTIONS:
1. Run configuration (benchmark, subset size, MCTS config)
2. Per-condition aggregate table: condition | accuracy | mean perplexity
   | std perplexity | wallclock per example
3. Pairwise comparison: certainty-guided vs random-perturbation
   - Absolute accuracy difference
   - Statistical significance (test name, p-value)
   - Effect size (Cohen's h or similar for proportions)
4. Pairwise comparison: certainty-guided vs greedy
   - Same as above
5. Failure-case examples: 5 cases where certainty-guided got the
   answer right but random failed; 5 cases where both failed (for
   the paper's qualitative section)
6. Pass/fail verdict per Gate 3 criteria G3.1, G3.2, G3.3
7. Plasticity recommendation: if certainty does not beat random by ≥2
   absolute points OR not significant → flag as Gate 3 NO-GO; pivot
   options listed in master checklist §4.3

DO NOT:
- Run more than 200 examples per condition (200 is the budget)
- Use a different benchmark than Track C recommended
- Skip the perplexity measurement (it's how we defend against the
  "you broke the model" critique in §5)
- Defer documentation
- Pad the writeup to make a weak result look stronger
```

---

## 6. Plasticity decision tree (concrete behaviors)

```
At each gate, count contradicting findings since the previous gate:

  IF count == 0 → CONTINUE (no log entry needed beyond track outputs)
  IF count == 1 → CONTINUE; Session A logs nuance + framing-adjustment in
                  track-E-gate-decisions.md
  IF count == 2 → YELLOW. Continue but add explicit risk-tracking note
                  to track-E-gate-decisions.md. Re-evaluate at next gate.
  IF count >= 3 → NO-GO at this gate. Stop spawning further tracks.
                  Session A produces pivot decision memo within 4h.
                  Pivot options: (a) Wildcard, (b) Hallucination, (c) Workshop.

  IF any FATAL finding (per master checklist §0):
    → IMMEDIATE NO-GO regardless of other counts.
    → Stop ALL tracks pending re-plan.
    → Session A produces fatal-finding analysis within 4h.

PIVOT DECISIONS — what each option costs at each gate:

  At T+24 NO-GO:
    (a) Wildcard: 13 days remaining; full pivot doable; lose Gemini-
        favored framing but keep most infrastructure.
    (b) Hallucination: 13 days; easiest pivot; lose novelty.
    (c) Workshop: 6 months for journal alternative.

  At T+48 NO-GO:
    (a) Wildcard: 11 days; tight but doable; save what's been built.
    (b) Hallucination: 11 days; still feasible.
    (c) Workshop: same as above.

  At T+72 NO-GO:
    (a) Wildcard: 9 days; very tight; risk of incomplete experiments.
    (b) Hallucination: 9 days; doable if scope is kept minimal.
    (c) Workshop: same; this becomes the highest-probability path at
        T+72 NO-GO.
```

---

## 7. Output file inventory (what should exist post-T+72)

| Path | Created by | Created at | Required for |
|---|---|---|---|
| `docs/logs/2026-05-09_track-A-certainty-probe-report.md` | Diag #1 | T+18 | Gate 1 |
| `docs/logs/2026-05-09_track-B-citation-audit.md` | Diag #2 | T+18 | Gate 1 |
| `docs/logs/2026-05-10_track-C-benchmark-calibration.md` | Diag #3 | T+44 | Gate 2 |
| `docs/logs/2026-05-10_track-D-vram-measurements.md` | Diag #4 | T+44 | Gate 2 |
| `docs/logs/2026-05-11_track-E-gate-decisions.md` | Session A | running | All gates |
| `docs/logs/2026-05-11_track-F-negative-control-report.md` | Diag #5 | T+66 | Gate 3 |
| `docs/logs/2026-05-11_post-T72-execution-plan.md` | Session A | T+72 (if GO) | 14-day execution |
| `docs/logs/2026-05-11_pivot-decision-memo.md` | Session A | T+24/48/72 (if NO-GO) | Pivot path |

---

## 8. Session A's running log

Session A maintains `docs/logs/2026-05-11_track-E-gate-decisions.md` from T+0 onward, appending entries every time a track output lands or a contradicting finding accumulates. Format:

```
### T+H — [Track] — [Finding]
- Source: [output file:section]
- Classification: [CLEAN | NUANCED | MISMATCHED | FATAL]
- Cumulative count this gate: [N]
- Plasticity action: [none | adjust framing | YELLOW | NO-GO trigger]
- Notes: [free-form]
```

This file IS the source of truth for the gate decisions. If it is not being maintained in real time, the gates cannot be evaluated reliably.

---

## 9. What this checklist does NOT cover

For audit-trail clarity:
- TeX edits to the canonical paper (Session B's role; deferred until after Gate 3 GO)
- README updates (Session A; after Gate 3 GO)
- EMNLP style switch (separate path; Josh's lane; gated on .sty file download)
- Tianyu Shi consultation re: research-program path (separate; can happen in parallel)
- Pre-print v0 to arXiv (pre-existing item; gated on Tianyu sign-off)
- Submission portal verification (separate; Josh's lane)

These are not in the 72-hour matrix because they don't gate the GO/NO-GO decision. They become priority items at T+72 if GATE 3 = GO.

---

## 10. End-state if T+72 = GO

Session A produces `docs/logs/2026-05-11_post-T72-execution-plan.md` with:
- Day 4–8 (6 days): Full experimental run on chosen benchmark; 3 conditions; 200 examples; statistical analysis
- Day 9–12 (4 days): TeX rewrite — abstract, §1 contributions, §3 problem formulation, §5 experiment 1, §6 memory complexity (with measured numbers from Track D), §7 limitations + future work
- Day 13–14 (2 days): Citations integration, page-budget under EMNLP style (.sty must be downloaded by Day 13), final REVISION marker audit
- Day 15–17 (3 days): Buffer + submission-portal verification + Tianyu sign-off + co-author review
- Day 18: Submit (May 25)

## 11. End-state if T+72 = NO-GO

Session A produces `docs/logs/2026-05-11_pivot-decision-memo.md` with:
- Summary of why GATE 3 failed (which criteria, which evidence)
- Three pivot options with updated 9-day execution plans for each
- Probability estimates per pivot (post-72-hour-data-update)
- Recommendation with reasoning
- Josh's decision-by deadline: T+96

---

*End of 72-hour execution checklist. This file is the durable artifact governing T+0 through T+72. Modifications must be flagged at the top with date + reason. Discipline matters; the gates are the gates.*
