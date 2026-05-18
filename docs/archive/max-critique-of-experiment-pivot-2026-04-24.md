# Review of the April 24 Experiment Scope Pivot

**Date:** 2026-04-24
**Author:** Max Toscano
**Purpose:** Side-by-side review of Josh's proposed submission strategy against my independent analysis. Identifies where we agree, where we diverge, what can go wrong in each plan, and what I recommend.

---

## 1. The Two Plans at a Glance

Josh and I independently asked our agents what to cut from the paper to make May 6 realistic. We converged on several points and diverged on others.

| Decision | Josh's Plan | Max's Plan |
|---|---|---|
| Drop H100 / gpt-oss-20b | Yes | Yes |
| Cut Experiment 4 (Eval Reproducibility) | Yes | Yes |
| Cut Experiment 5 (Cross-Model Transfer) | Yes | Yes |
| Cut Section 8 (Timeline) | Yes | Yes |
| Reframe Experiment 2 (B7 Option 2) | Yes, latent signal quality | Yes, KV-MCTS vs random KV search (same idea) |
| Cut Experiment 3 (Memory Efficiency) | **Yes, entirely** | **No -- keep a simplified version** |
| Models | **Llama 3.2 1B + 3B** | **Llama 3.2 1B only (3B unvalidated)** |
| Venue | **Main Track, May 6** | **NeurIPS Workshop (July-Aug) or ICLR 2027 (Oct)** |
| DRAFT-HOLD handling | Leave as internal LaTeX comments | **Convert every HOLD to visible text or fix** |

---

## 2. Where We Agree (and Why These Decisions Are Correct)

### Drop the H100 Dependency

The original critical path chained seven sequential dependencies with zero slack. Any single failure -- MoE monitoring unimplemented, H100 allocation not confirmed, MXFP4 precision mismatch -- would kill the submission. Removing gpt-oss-20b and H100 from the submission path is the single most important de-risking decision.

Both plans agree on this. No debate.

### Cut Experiments 4 and 5

**Experiment 4** requires 500 artifacts, 5 evaluation methods (GPT-4 judge, Claude judge, human panel, pass@1, cosine similarity), and Cohen's kappa. No script exists. No evaluation framework exists (`logomesh/evaluation_framework.py` was listed as a prerequisite in EXPERIMENTS.md Chunk 1D but was never created). This is a 2-3 month project being described in a paper due in 12 days.

**Experiment 5** requires `logomesh/procrustes.py` (not implemented), a 7B target model (not selected), paired hidden-state collection across model families, and the H100 hardware we're cutting. Zero lines of experiment code exist.

Both plans agree these are dead weight. Keeping them in Section 5 as described-but-unrun experiments signals to reviewers that the paper is a proposal, not completed research.

### Reframe Experiment 2 (B7 Option 2)

The original Experiment 2 promised ASR comparison across four methods (text-space MCTS, KV-cache MCTS, GCG, random KV search). The structural problem:

- Our MCTS evaluates **one token's hidden state** per node via `generate_one_step()`, then rolls back
- GCG optimizes a discrete adversarial suffix over the **full model**, generating **complete responses**
- MPA/DAMON generate **full multi-turn dialogues** per search node

Comparing ASR across these in the same table is a category error. A reviewer familiar with any of these baselines would flag it immediately. Both plans agree: scope down to what the code actually measures (latent signal quality), drop the GCG/MPA/DAMON comparison.

### Cut Section 8 (Timeline)

A research proposal timeline ("Phase 0: Weeks 1-3...") is a red flag in a submitted paper. It tells reviewers this is a plan, not finished work. Both plans agree: remove it, save ~0.5 pages.

---

## 3. Where We Diverge

### 3.1 Experiment 3 (Memory Efficiency)

**Josh cuts it entirely.** His reasoning: the code uses full-shape accumulators (~4x M_KV), not the sparse accumulators (~50 MB) the paper claims. Running Experiment 3 would produce numbers that contradict the Proposition. Better to not measure than to publish contradictory data.

**I keep a simplified version.** My reasoning: the Proposition is the paper's second formal claim. Theorem 1 (reversibility) is proven and validated. The Proposition (memory complexity) is not. Cutting Experiment 3 means the flagship efficiency claim -- "O(M_KV + d*K_acc) vs O(b^d * M_KV)" -- is an unverified assertion with known-wrong constants.

The paper's Introduction says "we verify empirically in Experiment 3." If Experiment 3 moves to Future Work, that forward reference becomes a broken promise.

**My simplified version:** Don't run the full b={2,3,5} x d={3,5,10,20} grid. Just profile VRAM at three points (b=3 at d=3, d=5, d=10). Compare measured values against theoretical b^d * M_KV. The constants will be ~4x instead of ~1x -- that's fine. Correct them in the paper. The core insight (no b^d scaling) still holds and now has empirical backing. This takes 30 minutes to run.

**What can go wrong with Josh's approach:** A reviewer reads the Proposition claiming 40.05 GB, checks the supplementary code, sees full-shape accumulators that cost ~120-160 GB, and finds no empirical validation. The paper's strongest formal claim appears fabricated.

**What can go wrong with my approach:** The measured numbers (4x M_KV) look worse than the claimed numbers (1.001x M_KV). A reviewer might focus on the discrepancy rather than the core result. But this is better than the alternative -- hiding a known discrepancy that supplementary code review will expose.

### 3.2 Model Selection (1B+3B vs 1B Only)

**Josh says Llama 3.2 1B + 3B, both on RTX 3060.** This gives a two-model comparison and a scaling narrative (does the signal hold at 3x parameters?).

**I say 1B only until 3B is tested.** Nobody has run the 3B model with the KV-MCTS pipeline. The VRAM math:

| Component | VRAM |
|---|---|
| 3B model weights (fp16) | ~6.0 GB |
| KV cache + clones + FP32 accumulators | ~1.7 GB |
| PyTorch overhead + fragmentation | ~1.5 GB |
| **Total** | **~9.2 GB** |
| **RTX 3060** | **12 GB** |
| **Headroom** | **~2.8 GB** |

It fits on paper but:

- `local_model.py` has no quantization support -- loads full fp16, no `load_in_4bit`
- If sequence length grows past 512 during MCTS, KV cache grows and eats headroom
- `_kv_eval_cache()` deep-copies the full DynamicCache per evaluation step -- 150+ transient allocations creating GC pressure
- No graceful OOM handling in `run_kv_mcts.py` -- crash with no output
- CLAUDE.md hardware table lists only "TinyLlama / Llama-3.2-1B-Instruct (~2GB VRAM)" for RTX 3060

**What can go wrong with Josh's approach:** He rewrites the paper around "1B vs 3B comparison." Then 3B OOMs during the MCTS run 3 days before submission. He has to re-rewrite the paper with single-model results under extreme time pressure.

**What can go wrong with my approach:** The paper has results on only one tiny model. Reviewers question whether any signal is meaningful at 1B scale. The paper's own Limitations section admits Llama-3.2-1B falls below RepE's validated range.

**The fix is trivial and settles the question immediately:**

```bash
uv run python scripts/probe_kv_cache_mutability.py \
    --model meta-llama/Llama-3.2-3B-Instruct --device auto
```

This takes 5 minutes. If it passes, use 3B. If it OOMs, don't. Run this before rewriting anything.

### 3.3 Submission Venue

**Josh targets NeurIPS 2026 Main Track (May 6).** His reasoning: the methodology is novel, the team has been working toward this deadline, and there's a "Concept & Feasibility" contribution type for early-stage work.

**I recommend NeurIPS 2026 Workshop (July-Aug) as primary target, with ICLR 2027 (October) for the full paper.** My reasoning: the current evidence base is insufficient for Main Track's ~25% acceptance rate.

After all cuts, the submission contains:

| Evidence | Status |
|---|---|
| Theorem 1 (200-cycle drift = 0) | Done |
| Experiment 1 (OEI alpha-sweep) | Blocked on 3 code fixes + B6 diagnostic |
| Experiment 2 (latent signal quality) | Paper rewrite only |

NeurIPS Main Track reviewers expect controlled baselines (Table 2 has 8 baselines with zero results), scale beyond toy models (1B is not a frontier model), and statistical rigor (one alpha-sweep is a pilot study). A reviewer seeing one theorem + one alpha-sweep on a 1B model will likely score 2-3 (Reject to Borderline Reject).

**What can go wrong with Josh's approach:** The paper gets rejected. Reviewers write comments like "interesting methodology, insufficient validation." The team spent 12 days preparing a submission that yields nothing except reviewer feedback they could have anticipated. Meanwhile, a NeurIPS Workshop submission that would have been accepted was never attempted.

**What can go wrong with my approach:** We don't submit to the most prestigious venue. Someone else publishes MCTS-in-latent-space before us (though arXiv preprint mitigates this). The team loses momentum from missing the original deadline. Prof. Tianyu Shi may have expectations around the NeurIPS Main Track submission.

### 3.4 DRAFT-HOLD Block Handling

**Josh leaves them as internal LaTeX comments** with deferral decisions documented in session logs.

**I say convert every HOLD to visible text or fix the discrepancy.** DRAFT-HOLD blocks are `%` comments in the TeX -- they don't render in the compiled PDF. What a reviewer sees:

- "LAT procedure from Zou et al." -- paper says PCA; code uses difference-in-means
- Per-layer notation `d_K^(l)` -- paper implies per-layer steering; code broadcasts one vector
- "Approximately 50 MB accumulator overhead" -- code allocates ~4x M_KV
- "40.05 GB reversible vs 9.7 TB standard" -- actual reversible is ~120-160 GB

If a reviewer reads the supplementary code (NeurIPS reviewers are encouraged to), they find discrepancies between the paper and implementation with no explanation in the visible text. The internal hedging is invisible to them.

**What can go wrong with Josh's approach:** A diligent reviewer finds a claim-code mismatch, interprets it as carelessness or worse, and downgrades the paper. The careful DRAFT-HOLD reasoning never reaches them.

**What can go wrong with my approach:** Visible caveats and corrections make the paper look less polished. Qualifying claims like "Phase 2 approximation: single-layer steering broadcast" in the body text uses precious page budget and signals incomplete work. But this is strictly better than a reviewer discovering the mismatch independently.

---

## 4. Risks Unique to Josh's Plan

### 4.1 The OEI Diagnostic (B6) Has No Fallback

Josh correctly identifies B6 as a gate: verify OEI varies across alpha values before committing to Experiment 1. But the proposal doesn't say what happens if B6 fails.

Current OEI evidence:

| Model | OEI Value | Formula Version | Interpretation |
|---|---|---|---|
| Qwen 1.5B | 0.996 | Old (ratio, unbounded) | Saturated. Meaningless. |
| Llama 1B | 0.883 | Old (ratio, unbounded) | One data point. Inconclusive. |
| Any model | Never tested | New (delta, [0,1]) | Unknown. Formula changed April 21. |

The OEI formula was recently changed from a ratio (range [0, infinity)) to delta-based (range [0, 1]). The 0.883 data point was collected under the old formula. Nobody has validated the new formula on any model.

If OEI is flat across alpha on 1B and 3B, Experiment 1 produces meaningless curves, there's nothing to put in Results, and the submission has no empirical story. The entire plan collapses with days left before the deadline.

### 4.2 The Paper Still Uses E&D Track Formatting

`\usepackage[eandd]{neurips_2026}` at line 8 of the TeX file. Josh's proposal pivots to Main Track but this blocker isn't flagged. Submitting to Main Track with E&D formatting causes desk rejection. One-line fix, but someone has to do it.

### 4.3 Remaining Evidence Is Very Thin

Even if everything in Josh's plan works perfectly:

- B6 passes (OEI discriminates across alpha)
- 3B runs on RTX 3060 without OOM
- B3/B4/B5 code fixes land cleanly
- Experiment 1 produces interpretable curves

The submission still has: one theorem, one alpha-sweep on 1B+3B, one reframed ablation study, zero baselines with results, zero results tables, and a 1B model the paper itself admits is below the validated range for the techniques being used. Main Track reviewers scoring this against papers with extensive 7B-70B experiments and controlled baseline comparisons will likely assess it as insufficient.

---

## 5. Risks Unique to Max's Plan

### 5.1 Workshop May Feel Like a Downgrade

The team has been targeting NeurIPS Main Track since the project's inception. Pivoting to a workshop may feel like admitting failure, especially to Prof. Tianyu Shi (academic advisor). This is a team morale and relationship risk, not a technical one.

### 5.2 Priority Establishment Delay

If we defer Main Track to ICLR 2027 (October), the KV-cache MCTS idea sits unpublished for 6+ months. During that time, other groups working on latent-space search (COCONUT, CRSM, Okazaki-RAG) could independently develop similar approaches. An arXiv preprint mitigates this but doesn't carry the weight of a peer-reviewed venue.

### 5.3 Simplified Experiment 3 Shows Embarrassing Constants

If I run memory profiling with the current code, the numbers show ~4x M_KV overhead, not the ~1x M_KV the Proposition claims. Even after correcting the paper, the narrative changes from "negligible overhead" to "significant overhead but still much better than exponential." A reviewer might fixate on the ~4x constant rather than the O(1) vs O(b^d) complexity class difference.

### 5.4 1B-Only Results Are Inherently Weak

Without 3B, the paper has results on a single model that the Limitations section admits is too small for the techniques being applied. This undermines credibility regardless of venue. If 3B runs, both plans benefit equally.

### 5.5 Workshop Deadlines Are Not Guaranteed

NeurIPS 2026 workshop CFPs typically open June-July with deadlines August-September. If no appropriate workshop accepts the paper's topic (AI safety + mechanistic interpretability + red-teaming), there may not be a workshop slot available, pushing the next opportunity to ICLR 2027 anyway.

---

## 6. What I Recommend

### Priority 1: Run the Two Diagnostic Tests (Today)

Before committing to ANY plan, answer two empirical questions:

**Does 3B fit on the 3060?**
```bash
uv run python scripts/probe_kv_cache_mutability.py \
    --model meta-llama/Llama-3.2-3B-Instruct --device auto
```

**Does OEI discriminate across alpha values?**
```bash
uv run python scripts/run_kv_mcts.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --nodes 25 --depth 3 --branches 5 \
    --alpha-values 0.1 0.5 1.0 2.0 5.0 \
    --output ./tmp/runs/mcts_oei_diagnostic.json
```

These two tests take under 2 hours combined and determine whether Josh's plan is even feasible.

### Priority 2: Finish My Code Fixes (This Week)

Regardless of venue or strategy:

| Fix | File | Impact |
|---|---|---|
| GAP-C1-01 (per-layer H-Neuron ID) | `hneuron_monitor.py` | Already implemented and tests passing |
| GAP-C1-02 (per-neuron ReLU) | `hneuron_monitor.py` | Makes sigma_H more sensitive to sparse spikes |
| GAP-C1-03 (separate calibration sets) | `scripts/run_kv_mcts.py` | Ensures sigma_H and rho_R measure different things |
| GAP-C1-11 (step counter) | `logomesh/kv_mcts.py` | Metadata correctness |

These fix the telemetry signal that Experiment 1 measures. They apply at any model scale and any venue.

### Priority 3: Decide on Strategy Based on Diagnostic Results

| B6 Result | 3B Result | Recommended Strategy |
|---|---|---|
| OEI discriminates | 3B fits | Josh's plan is viable. Main Track is a long shot but attemptable. Run Experiments 1+2 on 1B+3B. |
| OEI discriminates | 3B OOMs | Modified plan. Main Track with 1B-only results. Weak but honest. Workshop is safer. |
| OEI flat | 3B fits | Major pivot. Experiment 1 is not testable. Submit as Concept & Feasibility with theory only, or defer to workshop. |
| OEI flat | 3B OOMs | Defer. Post arXiv preprint for priority. Target ICLR 2027 with proper experiments on 7B+ with H100. |

### Priority 4: If Submitting to Main Track, Add These to Josh's Plan

1. **Keep simplified Experiment 3** -- three VRAM measurements, 30 minutes, validates the Proposition
2. **Fix DRAFT-HOLD blocks** -- convert to visible Limitations or correct the claims
3. **Change LaTeX from `eandd` to Main Track**
4. **Frame as "Concept & Feasibility" contribution type** -- explicitly tells reviewers to evaluate the idea, not the scale of validation
5. **Post arXiv preprint simultaneously** -- establishes priority regardless of acceptance outcome

---

## 7. Decision Points for Josh

1. Can we test 3B on the 3060 today before rewriting the paper around it?
2. What is the fallback if OEI is flat across alpha values?
3. Will you consider a NeurIPS Workshop as a parallel or alternative target?
4. Can DRAFT-HOLD blocks become visible caveats in the compiled PDF?
5. Should we post an arXiv preprint now to establish priority regardless of venue?
6. When is the LaTeX package being changed from `eandd` to Main Track?
