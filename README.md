# KV-Cache Inception

**Reversible Monte Carlo Tree Search in Latent Space for Detecting and Stress-Testing Alignment Faking in Large Language Models**

*NeurIPS 2026 Datasets & Benchmarks Track submission*

**Paper Outline Here:** *https://www.overleaf.com/read/gcfjswwqwkhq#1ac1ec*

**In-Repo Outline Here:** *docs\NeurIPS\04.18.2026-NeurIPS-Research-Proposal-2.tex*

---

## Abstract

Current approaches to evaluating the safety and alignment of large language models (LLMs) rely predominantly on black-box behavioral testing. These methods are fundamentally blind to *alignment faking* — wherein a model strategically complies with safety constraints during evaluation while preserving misaligned capabilities in its latent representations.

We propose **KV-Cache Inception**, a framework combining three novel contributions: 
(1) a **multi-dimensional micro-telemetry matrix** fusing H-Neuron activation tracking with Representation Engineering (RepE) projections to provide a continuous signal for distinguishing genuine reasoning from compliance-driven fabrication; 
(2) a **Reversible MCTS algorithm** conducting Monte Carlo Tree Search directly in the model's KV-cache latent space using a mathematically reversible FP32 accumulator rollback mechanism; and 
(3) the formalization of **Orthogonal Escape** — the phenomenon whereby corrective pressure applied to monitored latent subspaces forces deceptive encodings into unmonitored orthogonal null spaces.

---

## Three Contributions

1. **Micro-Telemetry Matrix** `T_t ∈ ℝ^{2×L}` — per-layer, per-step fusion of H-Neuron stress `σ_H` and RepE honesty `ρ_R`. Defines four diagnostic states (Creative Synthesis, Contextual Fabrication, Alignment Faking, Orthogonal Escape).

2. **Reversible MCTS in KV-Cache Space** — first MCTS operating in continuous latent space rather than discrete token space. Memory: `O(M_KV + d·K_acc)` vs. `O(b^d · M_KV)` for standard parallel MCTS.

3. **Orthogonal Escape** — formal definition, OEI metric, TDS metric, and experimental protocol (Experiment 1: RepE steering magnitude sweep at α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}).

---

## Script Reference

### Entry-point scripts (`scripts/`)

| Script | Paper section | What it does | Hardware | Status |
|---|---|---|---|---|
| `probe_kv_cache_mutability.py` | §4.2 precondition | KV-cache in-place mutation + reversibility gate | Any (CPU/GPU) | ✅ Complete |
| `measure_lipschitz_drift.py` | §4.2 Theorem 1 | FP32 accumulator drift validation (n cycles) | Any | ✅ Complete |
| `run_kv_mcts.py` | §4.2–4.3, Exp 1 | Reversible MCTS + T_t + OEI + TDS; Exp 1 α sweep | RTX (1B) / H100 (20B) | ✅ Implemented; H100 scale pending |
| `run_offline_mcts.py` | §5 Phase A/B, Exp 2 baseline | Offline text-space MCTS on defender simulacrum | RTX (1B) / H100 (20B) | ✅ Implemented; H100 scale pending |
| `train_lat_probes.py` | §5 Phase A prereq | LAT probe training for RepE ρ_R calibration | RTX (1B) | ✅ Implemented |
| `export_kv_mcts_to_croissant.py` | §6 dataset | Single artifact → Croissant 1.1 + RAI 1.0 package | Any | ✅ Complete |
| `collect_dataset.py` | §6 dataset | Batch N artifacts → NeurIPS submission package | Any | ✅ Complete |

### Core library (`logomesh/`)

| Module | Paper equation | Role |
|---|---|---|
| `telemetry_matrix.py` | Eq. 3, 8; Table 1 | `T_t ∈ ℝ^{2×L}`, `DiagnosticState`, MCTS reward |
| `kv_mcts.py` | Eq. 5–6; §4.2 | `ReversibleMCTS`, `FP32Accumulator` (Theorem 1) |
| `orthogonal_escape.py` | Eq. 10 | `OEICalculator`, `TDSCalculator` |
| `hneuron_monitor.py` | Eq. 3 row 0 | H-Neuron stress σ_H per layer (bottom-up) |
| `whitebox.py` | Eq. 3 row 1, Eq. 4 | RepE honesty ρ_R per layer — `PerLayerHonestyProjector` |
| `local_model.py` | Infrastructure | HuggingFace wrapper with KV-cache telemetry access |
| `croissant_export.py` | §6 | Croissant 1.1 + RAI 1.0 export and schema validation |

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run tests (no model required — must be 100% green before any experiment)
uv run pytest tests/ -v
```

### What you can run today (consumer GPU, RTX 3060+)

Run these in order. First run auto-downloads TinyLlama (~1GB, ~2GB VRAM).

```bash
# 1. Validate KV-cache mutability on your hardware (paper §4.2 precondition)
#    Must pass before running run_kv_mcts.py. Exit 0 = passed.
uv run python scripts/probe_kv_cache_mutability.py --device auto

# 2. Validate Theorem 1: FP32 accumulator drift is bounded independent of rollback count
#    Paper §4.2, Theorem 1. Expected result: accumulator_inf_norm stays at 0.00e+00.
uv run python scripts/measure_lipschitz_drift.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --n-cycles 200

# 3. Reversible MCTS smoke test — Experiment 1 α sweep on 1B surrogate (paper §4.2–4.3, §5)
#    Outputs a JSON artifact with T_t telemetry, OEI scores, and TDS per node.
uv run python scripts/run_kv_mcts.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --nodes 50 --depth 5 --branches 3

# 4. Offline MCTS baseline — Experiment 2 comparator (paper §5 Phase A/B)
#    Download model first: huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ./models/llama-3.2-1b
uv run python scripts/run_offline_mcts.py \
    --model ./models/llama-3.2-1b --episodes 20
```

**Notes:**
- `--device auto` resolves: `cuda` → `mps` → `cpu`.
- For gated HuggingFace models, run `huggingface-cli login` first.
- `probe_kv_cache_mutability.py` exit code 0 = gate passed; non-zero = failed.

### What requires H100 access (not yet run)

- Experiments 1–2 at scale (requires `openai/gpt-oss-20b`, ~16GB VRAM floor)
- Experiments 3–5: scripts not yet built (see Experiment Status table below)

### Dataset packaging (after H100 runs)

```bash
# Export a single run artifact to Croissant format
uv run python scripts/export_kv_mcts_to_croissant.py \
  --input ./tmp/mcts_output.json --output ./tmp/croissant

# Or emit Croissant directly from a run
uv run python scripts/run_kv_mcts.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --nodes 10 \
  --output ./tmp/mcts_output.json --croissant-dir ./tmp/croissant

# Batch-assemble all artifacts into the NeurIPS submission package
uv run python scripts/collect_dataset.py \
  --input-dir ./tmp/runs --output-dir ./docs/dataset
```

---

## Phase Roadmap

**Naming note:** This README uses development milestone numbering (Phase 1–4). The paper (§5) uses scientific execution sequence (Phase 0 / A / B).

| README | Paper | Description |
|---|---|---|
| Phase 1 | Phase 0 | Repo cleanup, `BaseModelClient` interface, modular architecture |
| Phase 2 | Phase A + Phase B setup | Reversible KV-MCTS, T_t telemetry, OEI, FP32 reversibility |
| Phase 3 | Phase B | Experiments 1–5 on H100 |
| Phase 4 | (implicit in Phase B) | Research dataset (Croissant), paper writing |

| Phase | Status |
|---|---|
| 1 — Repo cleanup | ✅ Complete |
| 2 — Reversible KV-MCTS + telemetry | ✅ Complete |
| 3 — Experiments (Phase B, H100) | 🔲 In progress |
| 4 — Dataset + paper | 🔲 Not started |

### Experiment Status

| Experiment | What it tests | Script | Hardware | Status |
|---|---|---|---|---|
| 1 — Orthogonal Escape Detection | α sweep: compliance rate vs. OEI / ρ_R divergence | `run_kv_mcts.py --alpha-values` | H100 (20B) | 1B prototype ✅; H100 scale ⏳ |
| 2 — MCTS vs. Text-Space | Attack success: latent MCTS vs. text MCTS vs. GCG vs. random | `run_kv_mcts.py` + `run_offline_mcts.py` baseline | H100 (20B) | 1B prototype ✅; H100 scale ⏳ |
| 3 — Memory Efficiency | VRAM: reversible MCTS vs. parallel MCTS at b×d | **Not yet built** | H100 | ❌ No script |
| 4 — Evaluation Reproducibility | Inter-rater agreement (Cohen's κ) across judge types | **Not yet built** | Any | ❌ No script |
| 5 — Cross-Model Transfer | Procrustes transfer of steering vectors: 1B → 7B → 20B | **Not yet built** | H100 | ❌ No script |

---

## Citation

```bibtex
@article{logomesh2026kvcache,
  title   = {KV-Cache Inception: Reversible Monte Carlo Tree Search in Latent Space
             for Detecting and Stress-Testing Alignment Faking in Large Language Models},
  author  = {[Authors]},
  journal = {NeurIPS 2026 Datasets and Benchmarks Track},
  year    = {2026}
}
```
