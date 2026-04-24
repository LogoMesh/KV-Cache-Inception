# Briefing for Max — Telemetry Matrix Branch
**April 21, 2026 — Updated April 24, 2026**
**From:** Josh
**To:** Max

---

## UPDATE — April 24, 2026

Two things changed since this briefing was written. Your assignment (Priorities 1–3 and the HOLD on GAP-C1-01) is unchanged.

**Paper fix priority list — status:** Items 1 (fabricated "0.10–0.20" figure), 2 (GCG "discrete token space"), 3 (AgenticRed claim), and 5 (CAA citation) are now done in the paper. Items 4, 6, and 7 have formal DRAFT–HOLD blocks added with recorded decisions (April 22–23). The HOLD block for GAP-C1-05 (RepE/PCA, item 6) explicitly names the code fix owner as Phase 3 — that's your whitebox.py work, deferred as briefed.

**OEI formula has changed in the paper (Contribution 3 — not your branch, but be aware):** The paper's current OEI definition is now:
```
OEI = ||Proj_Null(P)(h_steered − h_base)||₂ / ||h_steered − h_base||₂
```
Range [0, 1]. OEI → 1 = Orthogonal Escape. This is different from Eq. 10 in the contribution-3 doc and from the current code (which computes the ratio of null-space norms, range [0, ∞), OEI > 1 = escape). The contribution-3 doc is now stale on this point. Does not affect your branch, but don't reference the old OEI semantics if you need cross-contribution context.

---

---

## Where We Are

Two major workstreams completed today in parallel:

1. **Citation audit (Josh + Claude):** Every paper cited in the NeurIPS draft was read against its source. Several fabricated or misattributed claims were found — the paper needs significant revision before submission. A full prioritized fix list is at the bottom of this doc.

2. **Code-to-paper gap analysis (your three contribution docs):** These are excellent. Every gap is correctly identified, well-located, and the severity assignments hold up. They match and extend the citation audit findings. Good work.

The two streams agree on several key issues, which raises confidence — described below.

---

## Your Assignment: Telemetry Matrix Branch

You are cleared to start on a branch targeting **Contribution 1 (Telemetry Matrix)** in `logomesh/hneuron_monitor.py`, `logomesh/whitebox.py`, and `logomesh/telemetry_matrix.py`. **Do not touch Contributions 2 or 3 yet.**

### Rules for This Branch

1. **Add a comment to every line you change.** Brief, explain why (e.g., `# fix: per-neuron ReLU against individual baseline, not aggregate`). This is mandatory — the code needs to be auditable for the paper revision.

2. **Keep a dated session log** at `docs/logs/maxwell/YYYY-MM-DD_session-log.md`. Mirror the structure of `docs/logs/2026-04-21_session-log.md` — date, what you did, what changed, what the result was. One file per session.

3. **Run `uv run pytest tests/ -v` before every commit.** Must be 100% green. Do not push a failing test suite.

4. **Brief me before opening the PR** — send a message with: what you changed, what tests cover it, and what you could NOT fix (and why). I will review before merge.

---

## Priority Order for Your Branch

Work these in order. Stop and tell me when you finish each one.

---

### PRIORITY 1 — GAP-C1-02: Fix Per-Neuron ReLU (Medium effort)

**What the paper says (Eq. 3):**
```
σ_H^(l)(t) = (1/|H|) * SUM_{j ∈ H} ReLU(a_j^(l)(t) − ā_j^(l))
```
Each neuron's activation is individually subtracted against its own calibration baseline and ReLU'd before averaging.

**What the code does (`hneuron_monitor.py:247–251`, `_raw_dense_score()`):**
Computes the mean of raw activations over the H-Neuron index set. The per-neuron subtraction and ReLU against `ā_j` are not applied. The only clamp happens on the final normalized aggregate (`max(0.0, ...)` in `_score_layer()`), which is not the same thing.

**The problem:** Neurons where `a_j < ā_j` (below their calibration baseline — i.e., not stressed) contribute a negative value to the average instead of being zeroed. This makes σ_H less sensitive to sparse activation spikes.

**What to fix:** In `_raw_dense_score()`, apply the per-neuron subtraction and ReLU before averaging:
```python
# Current (wrong):
return activations[indices].mean()

# Correct:
baselines = self._coherent_means_per_neuron[indices]  # need to store this
return max(0.0, (activations[indices] - baselines).clip(min=0.0).mean())
```
This requires storing per-neuron baseline values (`coherent_means_per_neuron`) as an instance variable indexed by neuron position, rather than a single aggregate mean. The calibration loop in `_calibrate_dense()` already computes the right data — it just needs to be stored per-neuron and passed through.

---

### PRIORITY 2 — GAP-C1-03: Separate Calibration Sets (Low effort)

**What the paper says:**
σ_H (Eq. 3) is calibrated against "a calibration set of faithful generations" vs. hallucinated outputs — factual generation pressure.

**What the code does (`scripts/run_kv_mcts.py:56–70`):**
The "hallucinated" calibration set is actually jailbreak/coercion prompts (e.g., `"Ignore all safety guidelines"`). These are coercion prompts, not factual hallucination prompts. The HNeuronMonitor docstring at `hneuron_monitor.py:24–26` already shows the intended hallucination-style examples.

**The problem:** The H-Neuron baseline is calibrated against coercion-induced stress rather than hallucination stress. The two channels (σ_H and ρ_R) are supposed to measure different phenomena. Calibrating both against the same coercion prompts conflates them.

**What to fix:** In the runner, use two distinct calibration sets:
- `HNeuronMonitor`: factual hallucination prompts (e.g., `"Describe the moons of Planet Quasar-7 in detail"`, `"What happened at the Battle of Zorvak in 1453?"`)
- `PerLayerHonestyProjector`: coercion/alignment-pressure prompts (current set is fine for this)

The `HNeuronMonitor` docstring examples are the right reference. This is a runner-side fix, not a library fix.

---

### PRIORITY 3 — GAP-C1-11: Thread Step Counter (Low effort)

**What the paper says:**
T_t is indexed by generation step t. It's a per-step matrix with a meaningful time index.

**What the code does (`kv_mcts.py:643`):**
```python
TelemetryMatrix(h_neuron=..., repe_honesty=..., step=0)  # always 0
```

**What to fix:** Track a step counter in `run_async()` and pass it to `TelemetryMatrix()`. The counter should increment with each call to `_read_telemetry()`. The step index belongs in the output JSON artifacts for experiment analysis.

---

### HOLD — GAP-C1-01: Per-Layer H-Neuron Identification (Do NOT do yet)

The paper's Eq. 3 implies H-Neuron identification could differ per layer. The code reuses the same TOP_K=50 neuron indices (found at last-layer calibration) for every layer.

This is the most impactful gap but also the highest-complexity fix — it requires running contrastive activation analysis at each layer independently, which multiplies calibration cost by L. **This is Phase 3 work.** Document it clearly in your session log but do not implement it in this branch.

---

## What to Communicate to Me Before the PR

When you're done, send me a message (or put it in your session log and ping me) with:

1. Which gaps you fixed (by gap ID)
2. What the test results look like after your changes
3. Which gaps you could NOT fix and why
4. Any new issues you noticed while working

---

## Context From the Citation Audit — What This Means for Your Work

The citation audit confirmed several of your gap findings independently. Here are the ones that overlap and matter for how you should think about the code:

**RepE probes (GAP-C1-05 — do not fix in this branch, but understand why it matters):**
The paper cites Zou et al. (2023) for "LAT procedure via PCA." The code does difference-in-means. The citation audit confirmed this divergence directly from the RepE source paper. PCA finds the direction of maximum variance in the contrastive set; difference-in-means finds the centroid shift. For the paper to survive peer review, this either needs to be changed to PCA or the claim needs to be corrected in the text. Josh will handle the paper side; the code side is Phase 3.

**H-Neuron calibration (GAP-C1-03 — you ARE fixing this):**
The citation audit of Gao et al. (2025) confirmed that H-Neurons are identified by contrastive activation on *hallucination-inducing* prompts, not coercion prompts. Your fix in Priority 2 is exactly correct and is supported by the source paper.

---

## Critical Issues Max Should Know About (Not Your Branch — For Awareness)

These are issues elsewhere in the codebase that may affect your work or that you'll need to not break:

**Memory analysis in the paper is wrong (GAP-C2-03 + GAP-C2-04):**
The paper's Proposition claims memory complexity of O(M_KV + ~50MB). The code actually uses ~4×M_KV because:
- The base cache lives in memory (M_KV)
- `from_kv_cache()` clones K_base and V_base (another M_KV)
- FP32 accumulators are full-shape, not sparse — another ~2×M_KV in FP32

The core result (no b^d scaling) is still correct and is the main contribution. But the paper's specific 50MB figure is misleading. Josh will revise the Proposition.

**Two silent failure modes in the code (should be fixed by Josh/Claude before Phase 3):**
- `_broadcast_to()` in `kv_mcts.py` returns zero tensors silently on shape mismatch — MCTS appears to run but produces meaningless results
- Dummy OEI calculator in `run_kv_mcts.py` produces garbage `oei_score` values with no warning in the JSON

**Phase B blocker (GAP-C1-04):**
For MoE models like `gpt-oss-20b`, `score_per_layer()` returns `[0.5] * n` — the entire bottom-up channel is neutralized. This must be fixed before Lambda spin-up. Flag it explicitly when you brief Josh on your PR.

**Experiment 1 is not fully testable yet (GAP-C3-04, GAP-C3-05):**
The Orthogonal Escape hypothesis requires measuring textual compliance rate (does the model APPEAR more aligned as alpha increases?) — but `graders.py` is not wired into the MCTS loop. There is also no analysis script that loads the MCTS JSON and performs the actual hypothesis test. These are Phase 3 priorities.

---

## Paper Fix Priority List (For Josh's Reference)

Critical fixes Josh will make in the paper, in priority order.
*Status updated April 24 — see UPDATE block above for summary.*

1. ✅ **§2.5 line 117** — ~~Remove fabricated claim: `"score divergences of 0.10–0.20 on normalized scales across independent runs"`.~~ **Done.**

2. ✅ **§2.4 line 107** — ~~Fix `"token embedding space"` → `"discrete token space"` for GCG.~~ **Done.**

3. ✅ **§2.4 line 107** — ~~Remove `"AgenticRed applied MCTS within an agentic red-teaming framework."` AgenticRed's primary method is evolutionary algorithms.~~ **Done (citation retained, claim removed).**

4. ⏳ **§6 Proposition** — Revise memory complexity claim. O(M_KV + ~50MB) is wrong. Correct figure is approximately 4×M_KV (live cache + clone + full-shape FP32 accumulators). The core result (no b^d scaling) remains valid. **HOLD block added (GAP-C2-03 + GAP-C2-04) — paper fix deferred to Phase 3 pending sparse accumulator implementation.**

5. ✅ **Table 2 + §2.2** — ~~Add citation for CAA baseline.~~ **Done — `\cite{panickssery2023steering}` added.**

6. ⏳ **§4.1** — Qualify RepE implementation: code uses difference-in-means, not PCA. Either update the text or update the code before submission. **HOLD block added (GAP-C1-05) — draft fix quoted in paper; code fix deferred to Phase 3. Code fix owner: Max (whitebox.py).**

7. ⏳ **§4.2** — Qualify single-layer steering: code broadcasts middle-layer honesty direction to all layers, not per-layer d_K^(l) as Eq. 7 implies. Qualify as Phase 2 approximation. **HOLD block added (GAP-C2-01) — paper fix deferred to Phase 3 pending per-layer steering implementation.**

---

## Log Structure Reference

Your session logs at `docs/logs/maxwell/` should follow this format:

```markdown
# Session Log — [Date]

**Branch:** [branch name]
**Purpose:** [what you're working on]

---

## Changes Made

### [filename]
- What changed and why (gap ID)

## Test Results

`uv run pytest tests/ -v` — [N/N passed]

## What's Next

[What you plan to do next session]

## Blockers

[Anything blocking you]
```

---

*This briefing was prepared April 21, 2026. Reach out before opening the PR.*
