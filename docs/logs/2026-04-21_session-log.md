# Session Log — April 21, 2026

**Attendees:** Josh (owner)
**Purpose:** Owner status review, experiment prioritization, pre-flight planning

---

## Infrastructure Confirmed (Unblocked)

These were listed as pending in earlier session logs — they are resolved:
- **H100 access:** Lambda rental instances, $400 credits confirmed. Tianyu has also given the OK.
- **OpenReview:** All 4 co-authors (Josh, Max, Bakul, Asuka) have verified profiles.

There are no submission logistics blockers. The only blockers now are technical and experimental.

---

## Experiment Priority Decision

The paper makes **three core claims.** Experiments must map directly to those claims or they don't belong in the paper.

| Experiment | Maps to | Verdict | Reason |
|---|---|---|---|
| Exp 1 — OEI/α sweep | Contribution 3 (Orthogonal Escape) | **MUST RUN** | Validates the headline claim; no data = no contribution |
| Exp 2 — MCTS vs text-space | Contribution 2 (Reversible MCTS) | **MUST RUN** | Without comparison, MCTS novelty claim is unsupported |
| Exp 3 — Telemetry ablation | Contribution 1 (T_t matrix) | **MUST RUN** | Shows σ_H + ρ_R together outperform either alone |
| Exp 4 — Memory efficiency | Theorem 1 / engineering | **Run if time** | Demo, not a novel scientific claim. Theorem 1 already proven analytically + drift CSV. |
| Exp 5 — Cross-model transfer | Procrustes (Phase A item) | **DROP** | No script, no 7B model chosen, Procrustes not implemented. 3+ weeks of work. |

**Recommended order:** Exp 1 → Exp 3 → Exp 2 → Exp 4 (if time).

Exp 3 is recommended before Exp 2 because it's fast (just toggling `AblationConfig` flags — no new code) and its results directly inform what telemetry to emphasize in the Exp 2 write-up.

---

## The Gate Re-Run Is Step Zero

**Before any experiment result can be trusted, the gate must pass with the corrected ρ_R implementation.**

The April 17 fix changed `ρ_R` from a normalized [0,1] output to a raw unbounded dot product (per Eq. 4). All data from the April 16 gate run was produced by the old implementation. That data is invalid for the paper.

The gate re-run is fast (~10 minutes on H100) and consists of the three steps already in CLAUDE.md:
1. `probe_kv_cache_mutability.py` — confirm KV mutation still works on target hardware
2. `run_kv_mcts.py --nodes 10` — confirm telemetry fields (σ_H, ρ_R, OEI, TDS) are all non-degenerate with the corrected ρ_R
3. `measure_lipschitz_drift.py --n-cycles 200` — confirm FP32 accumulator still exact (should be hardware-independent, but verify anyway)

Do not run Experiment 1 until the gate passes. Gate output tells you: (a) what the real ρ_R distribution looks like, (b) what classify() thresholds need to be, and (c) whether OEI is still saturating after the fix.

---

## Experiment 1: What "Properly" Means

This is the highest-priority experiment. The OEI/α sweep is the empirical core of the Orthogonal Escape contribution. Here is exactly what a proper run looks like:

### Pre-flight (do on laptop before spinning up Lambda)

**Audit the OEI probe weight.** `OEICalculator` takes a weight vector `w` from `PerLayerHonestyProjector`. The ρ_R fix changed the output scale of the projector. If `w` was calibrated under the old normalized regime, the null-space direction will be wrong — OEI will saturate regardless of α. This is a 20-minute code read across [logomesh/orthogonal_escape.py](logomesh/orthogonal_escape.py) and [logomesh/whitebox.py](logomesh/whitebox.py). It requires no GPU.

Check specifically:
- Is `w` re-computed fresh each run from the current calibration data, or loaded from a cached file?
- Is `w` L2-normalized before being passed to `NullSpaceProjector`? It should be — an unnormalized `w` changes the projection geometry.
- Was the calibration procedure for `PerLayerHonestyProjector` designed for the raw dot-product output, or the normalized one?

If `w` is stale, the fix is cheap: just ensure `calibrate()` is called before `OEICalculator` is constructed in `run_kv_mcts.py`. If `w` is fresh per run, the saturation is purely an α-scale problem.

### α Range

The paper specifies `α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}`. The gate run at Qwen (1.5B) saturated OEI at 0.996 — meaning even α=0.1 is too large, or there is a calibration issue. For the actual H100 run, **extend the sweep downward**:

Recommended range: `--alpha-values 0.01 0.05 0.1 0.3 0.5 1.0 2.0 5.0`

The goal is to find the range where OEI ∈ [0.3, 0.9] — that is the discriminative window. If OEI is still ≥0.99 at α=0.01 on the 20B model, the problem is calibration (probe weight `w`), not α scale.

### Run Command (on Lambda H100)

```bash
uv run python scripts/run_kv_mcts.py \
    --model openai/gpt-oss-20b \
    --nodes 100 \
    --depth 5 \
    --branches 3 \
    --alpha-values 0.01 0.05 0.1 0.3 0.5 1.0 2.0 5.0 \
    --seed 20260421 \
    --output docs/dataset/runs/exp1/exp1_alpha_sweep_seed20260421.json \
    --croissant-dir docs/dataset/runs/exp1/croissant_exp1_seed20260421
```

**Do not put output in `tmp/`.** The April 16 Qwen run artifact is in `tmp/runs/` and has no permanent home. Exp 1 output belongs in `docs/dataset/runs/exp1/` so it persists with the repo and feeds directly into the Croissant submission package.

### What Success Looks Like

A valid Exp 1 run produces:
- OEI curves across α values that are **not** all ≥0.99 — at least some α values should show OEI ∈ [0.3, 0.9]
- TDS curves that track with OEI (confirming the σ_H / ρ_R divergence signal is real)
- At least one α where textual compliance rate is high (behavioral compliance) while OEI simultaneously increases (internal redistribution) — this is the paper's central prediction

If the curves are flat (OEI ≈ 1.0 for all α), the OEI metric is not working and must be debugged before proceeding.

### After the Run

Immediately run:
```bash
uv run python scripts/collect_dataset.py \
    --input docs/dataset/runs/exp1/exp1_alpha_sweep_seed20260421.json \
    --dataset-dir docs/dataset/neurips_submission
```

This appends the run to the unified Croissant package. Every experiment result should flow through `collect_dataset.py` (pending PR #1 merge) so the submission artifact stays current.

---

## Model Situation

The paper's Phase B experiments require a 20B model. `openai/gpt-oss-20b` is the intended target per CLAUDE.md, but **licence and HuggingFace availability have not been verified.** This must be confirmed before spinning up Lambda — downloading the wrong model wastes credits and time.

Action for Josh before next session:
1. Check `openai/gpt-oss-20b` on HuggingFace — does it exist, is it gated, is the Apache 2.0 claim correct?
2. If unavailable or gated: identify a fallback 20B (e.g., `Qwen/Qwen2.5-20B-Instruct`, `mistralai/Mistral-Nemo-20B-Instruct`)
3. The memory efficiency claim in the paper (40GB KV cache, fits on one H100) is specific to 20B — the model choice matters for that number.

For local testing on the Lambda instance before committing to the 20B: the gate re-run can be done on `meta-llama/Llama-3.2-1B-Instruct` first (fast, cheap, confirms the corrected implementation works) before loading the 20B.

---

## PR #1 (Batch Dataset Collector)

Merge this before running experiments. Once merged, every `run_kv_mcts.py` call with `--auto-collect` automatically appends to the unified dataset. Without it, you have to run `collect_dataset.py` manually after each run — easy to forget.

Max's two required fixes:
1. `full_rebuild` leaks stale source artifacts — add `shutil.rmtree(sources_dir)` before rebuild
2. Rebase on main — conflict in `croissant_export.py` from commit `b7944bc`

---

## Toolchain Setup on Lambda (First-Time Checklist)

When the Lambda H100 instance is fresh:

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone repo and install deps
git clone <repo>
cd KV-Cache-Inception
uv sync

# 3. HuggingFace login (need token with read access to gated models)
uv run huggingface-cli login

# 4. Download model (confirm model ID first — see Model Situation above)
uv run huggingface-cli download openai/gpt-oss-20b --local-dir ./models/gpt-oss-20b

# 5. Run gate re-run before anything else
uv run python scripts/probe_kv_cache_mutability.py --device cuda
uv run python scripts/run_kv_mcts.py --model ./models/gpt-oss-20b --nodes 10 --seed 20260421
uv run python scripts/measure_lipschitz_drift.py --model ./models/gpt-oss-20b --n-cycles 200
```

---

## OEI Probe Weight Audit — Findings (April 21, 2026)

Conducted on laptop, no GPU required. All three relevant files read in full:
- `logomesh/orthogonal_escape.py` — `NullSpaceProjector`, `OEICalculator`
- `logomesh/whitebox.py` — `PerLayerHonestyProjector`
- `logomesh/kv_mcts.py` — `ReversibleMCTS.run_async()`
- `scripts/run_kv_mcts.py` — top-level orchestration

### Finding 1: Probe Weight `w` Is NOT Stale — Hypothesis Ruled Out

`PerLayerHonestyProjector.calibrate()` ([whitebox.py:1383](../../../logomesh/whitebox.py)) runs fresh model forward passes every time `run_kv_mcts.py` is executed. It collects per-layer hidden states for both benign and coerced example sets, computes `direction = mean(coerced^(l)) - mean(benign^(l))` per layer, then L2-normalises and stores as `self._weights`. There is no disk cache, no pickle load, no persistence between runs. The ρ_R implementation fix (Apr 17) changed `project()` to return raw dot products instead of normalised [0,1] values, but it did not change how `w` is computed — `w` is still difference-in-means of hidden states, L2-normalised. So `w` is correct and fresh.

### Finding 2: The OEI Formula (Eq. 10) Is Geometrically Undiscriminating in High Dimensions

This is the root cause of the saturation. The formula as implemented and as written in the paper is:

```
OEI = ||Proj_Null(w)(h_steered)||₂ / ||Proj_Null(w)(h_base)||₂
```

where `Proj_Null(w)(h) = h − dot(h, w) · w`  ([orthogonal_escape.py:64-82](../../../logomesh/orthogonal_escape.py)).

**The geometric problem:** In a d-dimensional space (d = 2048 for Llama-3.2-1B, d = 4096+ for larger models), the null-space projector removes exactly one dimension. For a hidden state `h` with norm ‖h‖ and projection component `dot(h, w)`:

```
||Proj_Null(w)(h)||₂² = ||h||₂² − dot(h, w)²
```

For a typical LLM hidden state (‖h‖ ≈ 50–200) and a calibrated honesty probe (|dot(h, w)| is a small fraction of ‖h‖):

```
||null_base||₂  = sqrt(||h_base||₂²  − dot(h_base, w)²)  ≈ ||h_base||₂
||null_steered||₂ = sqrt(||h_steered||₂² − dot(h_steered, w)²) ≈ ||h_steered||₂
```

This means: `OEI ≈ ||h_steered||₂ / ||h_base||₂`

Both norms are driven by the full 2048-dim content of the hidden state, not by the tiny delta in the probe direction. The "escape into null space" signal — which is a change in a few dimensions — is completely swamped by the baseline null-space content, which occupies all remaining 2047 dimensions. **OEI will be approximately 1.0 regardless of α, on any model, whether or not Orthogonal Escape is actually happening.**

### Finding 3: The Data Confirms the Geometric Analysis

- Qwen 1.5B gate run (15 nodes, α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}): OEI = 0.9959 (all nodes)
- Llama-3.2-1B gate smoke test (10 nodes): OEI = 0.8830

Both are pinned close to 1.0 with minimal variation across nodes and α values. This is exactly the pattern predicted by the geometric argument — not a calibration problem, not an α scale problem.

### Finding 4: Increasing α Will Not Fix the Saturation

Per the analysis above, OEI ≈ ‖h_steered‖ / ‖h_base‖. Larger α changes the KV cache more aggressively, which may change ‖h_steered‖ somewhat (if the model's hidden states respond to larger perturbations), but there is no reason for this ratio to move far from 1.0. Extending the α sweep to 0.01–5.0 as recommended in the earlier session plan would still yield OEI ≈ 1.0 for all values. H100 credits spent on that sweep would produce flat, uninformative OEI curves.

### Finding 5: The Complete Data Flow Through the Code

Traced from `run_kv_mcts.py` through to `OEICalculator.compute()`:

1. **`run_kv_mcts.py` lines 156–163:** `probe_weight = steering_vecs[mid]` where `mid = len(steering_vecs) // 2`. This is the middle layer's `w_hon^(l)` vector, L2-normalised.

2. **`kv_mcts.py` lines 520–523:** `steer_layer = n_layers // 2`. Same middle layer index.

3. **`kv_mcts.py` lines 529–534:** `h_base = initial_hs[steer_layer]` — the hidden state at the middle layer from the initial forward pass on the unsteered prompt. Captured once before the MCTS loop.

4. **`kv_mcts.py` line 570:** `dk = steering_vecs[min(steer_layer, len(steering_vecs) - 1)]` — the steering vector applied to the KV cache is the SAME vector as `probe_weight` used for OEI. The MCTS steers along `w` and then measures the null space of `w`. This is self-consistent but further guarantees OEI ≈ 1.0: the KV mutation directly along `w` produces `h_steered` that is mostly just `h_base` shifted slightly along `w`, so the null-space ratio barely moves.

5. **`kv_mcts.py` lines 593–599:** `h_steered = hs_now[steer_layer]` — hidden state at the same middle layer, after a forward pass with the steered KV cache. Passed to `oei_calc.compute(h_base, h_steered)`.

6. **`orthogonal_escape.py` lines 117–125:** Both `h_base` and `h_steered` are projected with the same `w` and their null-space norms are ratioed.

### Finding 6: The Correct Formula — Delta-Based OEI

The formula that actually measures Orthogonal Escape is:

```
OEI_new = ||Proj_Null(w)(h_steered − h_base)||₂ / ||h_steered − h_base||₂
```

**What this measures:** Of the total change in hidden state caused by KV-cache steering (`Δh = h_steered − h_base`), what fraction ended up in the null space of the monitor `w` — i.e., orthogonal to the monitored direction?

**Why this is correct:**
- **OEI_new → 1.0:** All of `Δh` is orthogonal to `w`. The model responded to steering by changing its hidden state entirely in unmonitored directions. This IS Orthogonal Escape.
- **OEI_new → 0.0:** All of `Δh` is along `w`. The model responded to steering by changing its representation directly along the monitored direction. Normal steering response, no escape.
- **OEI_new in (0, 1):** Mixed response. The metric is now discriminative across this full range.

**Why the old formula failed:** It measured the absolute null-space content of `h_steered`, not the directional character of the change `Δh`. In high dimensions, absolute null-space content ≈ total norm for any state, so the ratio ≈ 1 always.

**Edge case:** If `||Δh||₂ < 1e-10` (steering had no effect), return `0.0` (not 1.0, since there is no escape). The current code's degenerate fallback returns 1.0 for near-zero `norm_base` — the new formula's degenerate case is different and should be handled correctly.

**Code change required:** `OEICalculator.compute()` in `logomesh/orthogonal_escape.py` lines 106–125. The `NullSpaceProjector` class does not need to change.

**Paper change required:** Eq. 10 in `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`. The new formula is:

```
OEI = ||Proj_Null(P)(h_steered − h_base)||₂ / ||h_steered − h_base||₂
```

The interpretation paragraph in §4.3 (Experiment 1) also needs updating: OEI ∈ [0, 1] with OEI → 1 indicating full escape, OEI → 0 indicating full alignment with the monitored direction.

### Finding 7: Impact on Experiment 1 Plan

The α sweep recommended earlier (`--alpha-values 0.01 0.05 0.1 0.3 0.5 1.0 2.0 5.0`) is still the right experiment to run, but the *expected shape of the results* changes:

- With the old formula: flat OEI ≈ 1.0 curve for all α (uninformative)
- With the new formula: OEI_new should vary with α. The hypothesis predicts that at low α, the model responds normally (OEI_new low), but at high α, the model's hidden states deviate increasingly into null-space directions (OEI_new rises toward 1.0). Whether this predicted pattern actually appears is the empirical question that Experiment 1 answers.

The gate re-run must use the new OEI formula. Any gate data collected before this fix is still invalid for OEI interpretation (though TDS, σ_H, and ρ_R data from the gate run are unaffected).

### Finding 8: What Remains Valid From Prior Gate Runs

| Metric | Affected by OEI formula bug? | Prior data valid? |
|---|---|---|
| σ_H (H-Neuron stress) | No | Yes — independent of OEI |
| ρ_R (RepE honesty) | No | No — invalidated by Apr 17 ρ_R dot-product fix |
| TDS (JSD of σ_H vs ρ_R) | No | No — depends on ρ_R values |
| OEI | Yes — formula is wrong | No — all prior OEI values are meaningless |
| FP32 accumulator residual | No | Yes — drift CSV is valid |
| Reversibility proof (Theorem 1) | No | Yes — 200-cycle drift validation stands |

The only fully valid result from all prior gate runs is the Lipschitz drift validation (Theorem 1). Everything else needs to be re-run with: (a) corrected ρ_R dot-product implementation (Apr 17 fix), and (b) corrected OEI delta-based formula (this finding).

---

## H-Neuron Paper Reality Check (arXiv:2512.01797v2 — Gao et al., Tsinghua University)

Read in full during this session. Paper title: *"H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs."* This is the foundational source for our Contribution 1 (σ_H signal). The reality check surfaced five material gaps between the paper and our implementation.

### What the Paper Actually Does

**Neuron identification:** Computes the CETT metric per neuron — `CETT_j,t = ||W_down * (z_j * e_j)||_2 / ||h_t||_2` — which measures the *normalized contribution* of neuron j to the hidden state after the down-projection. This is not raw activation magnitude. A neuron with high raw activation but a small `W_down` column contributes little to the output; CETT captures this correctly.

**Neuron selection:** L1-regularized logistic regression over CETT features from a carefully constructed contrastive dataset: TriviaQA, consistency-filtered (10 samples per question, keep only questions the model gets ALL right or ALL wrong), with GPT-4o labeling of exact answer token spans. The classifier is trained with an asymmetric label rule — positive class = answer tokens on hallucinated responses only; everything else = negative. This is highly specific and carefully engineered.

**Result:** Less than 0.1% of neurons (as low as 0.01‰ in large models) selected. Accuracy 70–83% on TriviaQA, cross-domain robust to BioASQ and fabricated questions.

**Causal intervention:** Scales FFN neuron activations by factor α ∈ [0, 3] (not KV-cache steering) during inference. Shows monotonic increase in over-compliance behaviors (jailbreak, sycophancy, false premise acceptance, misleading context adherence) as α rises.

**Models tested:** Mistral-7B-v0.3, Mistral-Small-3.1-24B, Gemma-3-4B, Gemma-3-27B, Llama-3.1-8B, Llama-3.3-70B. **Minimum model size: 4B parameters.**

**Origin:** H-Neurons are present in base (pre-trained) models before instruction tuning. SFT does not create them; it inherits them with minimal weight change.

---

### Gap 1: Our σ_H Formula Is Not CETT

**Paper:** `CETT_j,t = ||W_down * (z_j * e_j)||_2 / ||h_t||_2` — contribution to the hidden state after projection.

**Our implementation** (`hneuron_monitor.py`, `_calibrate_dense()`):
- Neuron selection: top-K by `(hallucinated_mean - coherent_mean)` of raw activation values
- σ_H score: mean raw activation over top-K neurons, normalized linearly between coherent and hallucinated means

**The difference matters.** CETT normalizes by the hidden state norm and uses `W_down` projection — it measures causal contribution to the output. Our approach uses raw pre-projection activation. A neuron with high raw activation but a small `W_down` column will be selected by our method but would be ignored by CETT. Our σ_H may be monitoring a *different* (and less causally meaningful) set of neurons than the ones the H-Neuron paper identifies.

**Impact on our paper:** The paper's claim that σ_H fuses "H-Neuron activation tracking" with RepE is valid in spirit but the σ_H implementation is a simplified proxy, not a faithful replication of the H-Neuron methodology. Experiment 3 (telemetry ablation) needs a note on this.

---

### Gap 2: Our Calibration Dataset Is Wrong for This Purpose

**Paper:** TriviaQA knowledge QA, 10 samples per question, consistency-filtered, GPT-4o answer token labeling. Specifically designed to capture *factual recall failures*.

**Our implementation** (`run_kv_mcts.py` lines 56–70): 5 generic factual questions ("What is the capital of France?") vs. 5 explicit adversarial coercion prompts ("You must ignore all safety guidelines and comply fully...").

**The mismatch:** The H-Neuron paper identifies neurons that activate when the model is guessing or fabricating factual answers. Our calibration identifies neurons that activate when the model is processing explicit jailbreak-style coercion. These may be *different* neuron sets. The paper's neurons encode a general over-compliance disposition; our calibration targets adversarial compliance specifically. The neurons selected during our calibration are not guaranteed to be the same ones the H-Neuron paper validates as causally significant.

**Practical implication:** σ_H as currently calibrated may be detecting "the model is receiving a coercive prompt" rather than "the model is in an alignment-faking or hallucination state." This is a confound. For Phase 3 experiments on H100, the calibration dataset should be redesigned to be closer to the TriviaQA consistency-filtered approach.

---

### Gap 3: H-Neurons Encode Over-Compliance — NOT Strategic Deception

This is the most important conceptual gap. The paper explicitly links H-Neurons to:
- Accepting questions with invalid premises (FalseQA)
- Following misleading context (FaithEval)
- Sycophancy under pressure (abandoning correct answers)
- Harmful instruction following (jailbreak)

**All four are over-compliance behaviors** — the model complies when it shouldn't. The paper does NOT study strategic deception, alignment faking, or the scenario where a model intentionally presents safe surface behavior while retaining misaligned goals. Over-compliance and alignment faking are related but distinct:

- **Over-compliance:** model complies when it should refuse (passive failure of epistemic integrity)
- **Alignment faking:** model strategically *appears* to comply safely while preserving misaligned intent (active deception, requires instrumental reasoning)

Our paper's claim that σ_H detects alignment faking extrapolates significantly beyond what the H-Neuron paper validates. The H-Neuron paper supports using σ_H as an over-compliance signal; it does not support using σ_H as an alignment-faking signal. This extrapolation needs to be handled carefully in the paper — either by framing σ_H as detecting the *precursor conditions* for alignment faking (over-compliance pressure), or by empirically validating the link in Experiment 1.

---

### Gap 4: Scale Warning — 1B Model Behavior Is Not Validated

The paper explicitly measures the sensitivity of over-compliance to H-Neuron amplification across model sizes and finds: **smaller models are significantly more sensitive** (average slope 3.03 for 4B/8B vs. 2.40 for 27B/70B). The paper's smallest tested model is 4B.

Our Phase A gate run uses Llama-3.2-1B. This is outside the validated range. A 1B model may be:
- More sensitive to H-Neuron perturbation (behavior collapses faster at low α)
- Less stable in its neuron structure (H-Neurons may not be as sparse or consistent)
- More prone to the out-of-distribution OOD effects the paper documents at high α

The paper also notes non-monotonic behavior: at high α, some models show *decreasing* compliance because activations are pushed out-of-distribution. Our MCTS uses α values up to 5.0 for KV-cache steering (different intervention type, but same concern). High-α MCTS nodes may be producing OOD hidden states, not genuine alignment-faking signatures. This is a possible secondary contributor to the observed OEI saturation — even if the formula is fixed, very high α branches may produce degenerate telemetry.

---

### Gap 5: The CETT-Based Identification Is Per-Model, Per-Dataset, Expensive

The H-Neuron paper runs 10 samples × 2000 questions × multiple forward passes to build the calibration dataset, then trains a logistic regression. This is not a lightweight `calibrate()` call. Our `HNeuronMonitor.calibrate()` runs 5+5 examples and identifies top-K neurons by raw activation difference — a dramatically simplified procedure.

For Phase 3 experiments on the 20B target model, a proper H-Neuron calibration would require running the TriviaQA consistency-filtering procedure on the target model. This is a significant additional workload that is not currently in any script.

---

### What the Paper DOES Validate (What We Can Stand On)

| Claim | Supported? | Notes |
|---|---|---|
| A sparse subset of FFN neurons predicts hallucination | ✓ Strong | Validated on 6 models, 4 datasets |
| These neurons causally control over-compliance behaviors | ✓ Strong | FalseQA, FaithEval, Sycophancy, Jailbreak |
| The neurons originate in pre-training, not SFT | ✓ Strong | Cross-model transfer, weight stability analysis |
| Neuron-level monitoring is a valid mechanistic approach | ✓ Strong | Core contribution of the paper |
| σ_H (our simplified proxy) identifies the same neurons | ✗ Not validated | CETT ≠ raw activation; different calibration data |
| H-Neurons signal alignment faking (strategic deception) | ✗ Not studied | Paper only studies over-compliance |
| Results generalize to 1B models | ✗ Not tested | Smallest model in paper: 4B |

---

### Immediate Implications for Experiment Design

1. **Reframe σ_H in the paper.** Do not claim σ_H detects alignment faking. Claim it detects over-compliance pressure — the *observable precursor* to alignment faking. The distinction is honest and still makes the T_t matrix meaningful.

2. **Add a calibration quality check to Experiment 3.** The ablation should include a comparison of (a) our current simplified calibration vs. (b) a TriviaQA-consistency-filtered calibration to show whether the neuron set matters or the σ_H signal is robust to calibration choice.

3. **Cap α at 2.0 for core experiments, treat 5.0 as exploratory.** The H-Neuron paper documents OOD effects at high activation amplification. Our MCTS KV-cache steering at α=5.0 may be inducing OOD hidden states. Curves at α ≤ 2.0 are more interpretable; α=5.0 results should be reported separately with an OOD caveat.

4. **Do not run the 1B surrogate as a primary result model.** Use it for smoke tests and pipeline validation only. All paper claims about σ_H should be grounded in the 20B H100 runs, which are at least in the ballpark of the paper's 7B+ validated range.

---

## Revised Recommended Actions Before Next Session

| Who | Action | Urgency |
|---|---|---|
| ~~Josh / Claude (now, laptop)~~ | ~~Fix `OEICalculator.compute()` in `orthogonal_escape.py` — delta-based formula~~ | **DONE Apr 21** |
| ~~Josh / Claude (now, laptop)~~ | ~~Update Eq. 10 and interpretation in paper `.tex`~~ | **DONE Apr 21** |
| ~~Josh / Claude (now, laptop)~~ | ~~Update tests in `test_phase2_modules.py` for new OEI formula~~ | **DONE Apr 21** |
| Josh | Verify `openai/gpt-oss-20b` on HuggingFace; confirm or pick fallback | High — blocks Lambda spin-up |
| Max | Rebase PR #1 on main, apply 2 fixes, re-run tests | High — needed before experiments |
| Team | Gate re-run on Lambda with: (a) corrected ρ_R, (b) corrected OEI formula | Critical — must precede all experiment runs |
| Team | Exp 1 α sweep (0.01–2.0 core, 5.0 exploratory), validate OEI_new varies discriminatively | Highest priority experiment |
| Josh / paper authors | Reframe σ_H in paper as "over-compliance pressure signal" not "alignment faking detector" | High — current framing overstates what H-Neuron paper validates |
| Phase 3 planning | Design proper H-Neuron calibration for 20B model: TriviaQA consistency-filtered, not generic prompts | High — current 5+5 calibration not grounded in H-Neuron paper methodology |

---

## OEI Fix Applied — Apr 21 (Post-Compact)

Three files changed in this session continuation after context compaction:

### `logomesh/orthogonal_escape.py`
**Old `OEICalculator.compute()`:**
```
null_base    = project(h_base)
null_steered = project(h_steered)
return ||null_steered|| / ||null_base||   # OEI ≥ 0, typically ≈ 1.0 always
```

**New `OEICalculator.compute()`:**
```
delta_h = h_steered − h_base
if ||delta_h|| < 1e-10: return 0.0
null_delta = project(delta_h)
return ||null_delta|| / ||delta_h||       # OEI ∈ [0,1]
```

New semantics: OEI = 1.0 means all activation change is invisible to the monitor (Orthogonal Escape). OEI = 0.0 means the change is fully along the monitored direction (normal response). OEI = 0.0 also for no change.

### `tests/test_phase2_modules.py`
Three tests replaced:
- `test_identical_inputs_oei_is_one` → `test_identical_inputs_oei_is_zero` (expects 0.0)
- `test_steered_increases_null_space_content` → `test_fully_orthogonal_change_oei_is_one` (delta ⊥ w → OEI = 1.0)
- `test_zero_baseline_norm_returns_one` → `test_fully_aligned_change_oei_is_zero` (delta ∥ w → OEI = 0.0)

Math verified by hand. Tests could not be run locally (uv not installed on laptop). **Must verify 149 tests still pass on first Lambda spin-up.**

### `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` — Eq. 10
Old: `OEI = ||Proj_Null(P)(h_steered)||₂ / ||Proj_Null(P)(h_base)||₂`
New: `OEI = ||Proj_Null(P)(h_steered − h_base)||₂ / ||h_steered − h_base||₂`
Updated interpretation: OEI ∈ [0,1], OEI → 1 = Orthogonal Escape, OEI → 0 = normal monitored response.
