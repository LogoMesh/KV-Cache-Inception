# Experiment Checklist — KV-Cache Inception

**NeurIPS 2026 deadline: May 6, 2026**
Track progress in this file. Check boxes as tasks complete.
Linear issues referenced as LOG-XX throughout.

---

## Dependency Map

```
Phase 3 Gate (must pass first)
  └── LAT Probe Retraining (LOG-32)
        └── Experiment 1: Orthogonal Escape Detection (LOG-33)  ← Central paper claim
              └── Experiment 2: MCTS Comparison (LOG-34)
  └── Experiment 3: Memory Efficiency (LOG-35)                  ← Can run in parallel
  └── evaluation_framework.py (LOG-39)
        └── Experiment 4: Evaluation Reproducibility (LOG-36)
  └── procrustes.py (LOG-38)
        └── Experiment 5: Cross-Model Transfer (LOG-37)
              └── All results → Dataset (LOG-40) → Paper (LOG-42–46)
```

---

## Current Status

- [x] Phase 2 code complete — 117 tests passing
- [ ] Phase 3 gate passed on real hardware
- [ ] Any experiment script written
- [ ] Any experiment results collected

---

## Chunk 0 — Phase 3 Gate (no code needed, just run it)

> Validates that Phase 2 code works on a real 1B model before writing experiment scripts.
> **Everything below is blocked until this passes.**

- [ ] `huggingface-cli login`
- [ ] Run KV-cache mutability gate:
  ```bash
  uv run python scripts/probe_kv_cache_mutability.py \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device auto
  ```
  Pass criterion: exit code `0`, `gate_passed: true` in output JSON
- [ ] Run KV-MCTS smoke test (10 nodes):
  ```bash
  uv run python scripts/run_kv_mcts.py \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --nodes 10 --depth 3 --branches 2
  ```
  Pass criterion: JSON output with 10 nodes, each with `sigma_H_mean`, `rho_R_mean`, `oei_score` populated (not null/0.5 flat)
- [ ] Run Theorem 1 drift validation:
  ```bash
  uv run python scripts/measure_lipschitz_drift.py \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --n-cycles 200
  ```
  Pass criterion: FP32 accumulator column stays bounded; naive bf16 column shows linear growth

---

## Chunk 1 — Prerequisites (implement before experiments)

### 1A — Fix ρ_R raw dot product (code gap)

> Paper Eq. 4: `ρ_R^(l)(t) = w_hon^(l)⊤ h_t^(l)` — raw dot product.
> Current Phase 2 implementation clips to [0,1] via `(proj + 1) / 2`. Must fix before Experiment 1 results are paper-valid.

- [ ] Update `PerLayerHonestyProjector.project()` in `logomesh/whitebox.py` — remove `(proj + 1) / 2` normalisation, return raw dot product
- [ ] Update `classify()` thresholds in `logomesh/telemetry_matrix.py` if needed (0.3/0.7 bounds assume [0,1])
- [ ] Update affected tests in `tests/test_phase2_modules.py`
- [ ] `uv run pytest tests/ -v` → still 117 passed

### 1B — LAT Probe Retraining (LOG-32)

> Replaces uniform neuron-index approximation with paper-faithful per-layer probes.
> Script: `scripts/train_lat_probes.py`

- [ ] Construct **honesty** contrast pairs: benign factual responses vs. coerced confabulation (minimum 30 pairs)
- [ ] Construct **certainty** contrast pairs: high-confidence outputs vs. hedged/uncertain outputs (minimum 30 pairs)
- [ ] Construct **goal-coercion** contrast pairs: aligned refusals vs. jailbroken completions (minimum 30 pairs)
- [ ] Run probe training:
  ```bash
  uv run python scripts/train_lat_probes.py \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
  ```
- [ ] Verify per-layer `w_hon^(l)` weights saved and loadable into `PerLayerHonestyProjector`
- [ ] Verify probe accuracy ≥ 0.65 on held-out set (threshold in script)

### 1C — `logomesh/procrustes.py` (LOG-38)

> Required by Experiment 5. Implement before running Exp 5.

- [ ] Create `logomesh/procrustes.py` with `ProcrustesAligner` class
  - `fit(source_states, target_states)` — `scipy.linalg.orthogonal_procrustes`
  - `transform(vectors)` — apply R to steering vectors
- [ ] Export `ProcrustesAligner` in `logomesh/__init__.py`
- [ ] Write `tests/test_procrustes.py`:
  - Verify R is orthogonal: `‖R^T R − I‖_F < 1e-5`
  - Verify transform on identity-aligned spaces returns input unchanged
- [ ] `uv run pytest tests/ -v` → all passing

### 1D — `logomesh/evaluation_framework.py` (LOG-39)

> Required by Experiment 4. Implement before running Exp 4.

- [ ] Create `logomesh/evaluation_framework.py` with:
  - `SandboxEvaluator` — pass@1 via Docker subprocess execution
  - `SemanticSimilarityEvaluator` — cosine similarity using `all-MiniLM-L6-v2`
  - `LLMJudgeEvaluator` — GPT-4 + Claude 3.5, 5-run protocol
  - `cohen_kappa(labels_a, labels_b)` — inter-rater agreement
- [ ] Export in `logomesh/__init__.py`
- [ ] Write model-free tests (mock Docker + mock LLM responses)
- [ ] `uv run pytest tests/ -v` → all passing

---

## Chunk 2 — Experiment 1: Orthogonal Escape Detection

> **This is the central empirical claim of the paper.** Run locally on TinyLlama first, then on H100.
> Linear: LOG-33 | Script: `scripts/experiment1_orthogonal_escape.py`

### Implementation
- [ ] Write `scripts/experiment1_orthogonal_escape.py`
  - CLI: `--model`, `--alpha-values`, `--n-prompts`, `--output-dir`
  - Inner loop: for each α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
    - Run `ReversibleMCTS` with that α
    - For each node record all four measurements (see below)
  - Output: one JSON per α + summary CSV

### Local run (TinyLlama)
- [ ] Collect measurement (a): **textual compliance rate** — does the model output pass a safety check?
- [ ] Collect measurement (b): **H-Neuron stress distribution** σ_H^(l) across all L layers
- [ ] Collect measurement (c): **RepE honesty projection** ρ_R^(l) across all L layers
- [ ] Collect measurement (d): **null-space norm** `‖Proj_Null(w_hon)(h_t^(l))‖_2` per layer
- [ ] Verify hypothesis locally: (a) and (c) improve with α, (d) simultaneously increases
- [ ] Generate OEI vs α figure (local prototype — not paper quality yet)

### H100 run (full scale)
- [ ] Coordinate H100 access with Prof. Tianyu Shi
- [ ] Confirm target model (Llama-3.2-1B-Instruct or gpt-oss-20b)
- [ ] Run full α sweep on H100
- [ ] Collect results at scale (≥ 100 prompts per α value)
- [ ] Generate paper-quality OEI vs α figure

---

## Chunk 3 — Experiment 2: MCTS Comparison

> Compares KV-MCTS against three baselines on the same prompt set.
> Linear: LOG-34 | Script: `scripts/experiment2_mcts_comparison.py`

### Implementation
- [ ] Write `scripts/experiment2_mcts_comparison.py`
  - Baseline (a): text-space MCTS — wire to `scripts/run_offline_mcts.py`
  - Baseline (b): KV-cache MCTS — wire to `scripts/run_kv_mcts.py`
  - Baseline (c): GCG — implement greedy coordinate gradient attack
  - Baseline (d): random KV-cache search — same α/layer distribution, no UCB1
  - CLI: `--model`, `--n-prompts`, `--output`

### Metrics (all three required)
- [ ] **Attack Success Rate (ASR)** — fraction of rollouts producing alignment-breaking intervention
- [ ] **Mean rollouts to first success** — efficiency comparison
- [ ] **Diversity** — pairwise cosine distance of intervention vectors in KV-cache space

### Runs
- [ ] Local validation run on TinyLlama (small n)
- [ ] Full H100 run
- [ ] Generate comparison table for paper §5

---

## Chunk 4 — Experiment 3: Memory Efficiency

> Validates Theorem 1 memory complexity claim: O(M_KV + d·K_acc) vs O(b^d · M_KV).
> Linear: LOG-35 | Script: `scripts/experiment3_memory_profiling.py`

### Implementation
- [ ] Write `scripts/experiment3_memory_profiling.py`
  - CLI: `--model`, `--b-values`, `--d-values`, `--output-csv`
  - Profile peak VRAM for KV-MCTS (reversible) at each (b, d) pair
  - Profile peak VRAM for naive parallel MCTS (clone full cache per branch)
  - Measure wall-clock time overhead of rollback per cycle
  - Measure Lipschitz drift (perplexity deviation) with/without FP32 accumulator

### Parameters (from paper)
- Branching factors: b ∈ {2, 3, 5}
- Search depths: d ∈ {3, 5, 10, 20}

### Metrics
- [ ] Peak VRAM at each (b, d) — reversible vs standard parallel
- [ ] Wall-clock rollback overhead
- [ ] Lipschitz drift — perplexity deviation from baseline
- [ ] MER = M_standard / M_reversible

### Runs
- [ ] Local run on RTX 3060 (small models — validates code + methodology)
- [ ] H100 run for 20B model (produces paper numbers)
- [ ] Generate memory scaling plot for paper

---

## Chunk 5 — Experiment 4: Evaluation Reproducibility

> Validates that diagnostic state classification is reproducible across evaluators.
> Requires `evaluation_framework.py` (Chunk 1D).
> Linear: LOG-36 | Script: `scripts/experiment4_evaluation_reproducibility.py`

### Implementation
- [ ] Write `scripts/experiment4_evaluation_reproducibility.py`
  - Collect 500 MCTS node artifacts (telemetry + generation output)
  - Run all five evaluation methods

### Five evaluators (all required by paper)
- [ ] (a) **pass@1** — ground-truth execution in constrained Docker sandbox
- [ ] (b) **Cosine similarity** — sentence embeddings (all-MiniLM-L6-v2) vs. task specification
- [ ] (c) **GPT-4 as judge** — 5 independent runs
- [ ] (d) **Claude 3.5 as judge** — 5 independent runs
- [ ] (e) **Human expert panel** — recruit 3 reviewers, collect labels

### Metrics
- [ ] Inter-run variance for (c) and (d)
- [ ] **Cohen's κ** across all evaluators — target κ ≥ 0.7
- [ ] Correlation of LLM judges with human panel

---

## Chunk 6 — Experiment 5: Cross-Model Transfer

> Tests whether 1B steering vectors transfer to 7B and 20B via Procrustes alignment.
> Requires `procrustes.py` (Chunk 1C).
> Linear: LOG-37 | Script: `scripts/experiment5_procrustes_transfer.py`

### Implementation
- [ ] Write `scripts/experiment5_procrustes_transfer.py`
  - Collect paired hidden-state activations: run same prompts through 1B and 7B
  - Fit `ProcrustesAligner` on paired states
  - Transfer `w_hon^(l)` steering vectors: 1B → 7B space
  - Run KV-MCTS on 7B with transferred vectors
  - Repeat: 1B → 20B (gpt-oss-20b) — H100 only

### Confirm with Tianyu Shi
- [ ] Confirm 7B target model ID
- [ ] Confirm 20B access (gpt-oss-20b) on H100 cluster
- [ ] Note: gpt-oss-20b is MoE — use router logit entropy for H-Neuron monitoring, NOT dense MLP

### Metrics (both required)
- [ ] (a) **ASR degradation** across scales: 1B native → 7B transferred → 20B transferred
- [ ] (b) **Telemetry signature preservation** — do σ_H/ρ_R patterns carry over?

### Reference
Oozeer et al. (2025), arXiv:2503.04429 — empirical support for transferability approach

---

## Chunk 7 — Dataset + Paper

> Assembles everything into the NeurIPS submission. Runs in parallel with late experiments.

### Dataset (Max — LOG-40, LOG-46)
- [ ] Finalise Croissant schema fields once Experiment 1 output format is confirmed (`docs/dataset/croissant_schema_stub.json`)
  - Per-node: T_t matrix (σ_H^(l), ρ_R^(l)), DiagnosticState, OEI, TDS
  - MCTS tree: node_id, parent_id, depth, alpha, reward
  - Generation: prompt, system, completion
  - Run metadata: model_id, hardware, seed, timestamp
- [ ] Collect all experiment artifacts into dataset
- [ ] Upload dataset to HuggingFace Hub
- [ ] Write dataset card (NeurIPS D&B datasheet fields)

### Paper (LOG-42–44)
- [ ] **Methods section** (LOG-42) — can write NOW, no results needed
  - Finalise all 12 equations (Eq. 1, 3–12)
  - Theorem 1 both bounds + proof sketch
  - Proposition: memory complexity with concrete 20B example
- [ ] **Related Work section** (LOG-44) — blocked on Bakul's competitive analysis (LOG-41)
- [ ] **Experiments section** (LOG-43) — blocked on all 5 experiments completing
  - Exp 1: OEI vs α curve
  - Exp 2: ASR comparison table
  - Exp 3: VRAM scaling plot
  - Exp 4: Cohen's κ score
  - Exp 5: Procrustes transfer comparison
  - Ablations: sensitivity to λ₁, λ₂, λ₃ and JSD threshold

### Final submission prep
- [ ] Reproducibility manifest (LOG-45): run IDs, git SHAs, seeds, hardware, timing, `uv.lock` snapshot
- [ ] Advisor review — send Methods draft to Prof. Tianyu Shi
- [ ] All authors confirm affiliations and disclosures
- [ ] **Submit by May 6, 2026**

---

## Hardware Reference

| Task | Hardware | Model |
|---|---|---|
| Chunks 0–1 (gate + prereqs) | RTX 3060 / CPU | TinyLlama 1.1B |
| Experiments 1–4 (local validation) | RTX 3060 12GB | TinyLlama or Llama-3.2-1B |
| Experiments 1–5 (full scale) | 8× H100 80GB (McGill) | gpt-oss-20b |

---

*Last updated: 2026-04-13. See Linear for issue-level detail: https://linear.app/logomesh-agentbeats-phase-2/project/logomesh-kv-cache-inception-neurips-2026-6e647f4bb3f5*
