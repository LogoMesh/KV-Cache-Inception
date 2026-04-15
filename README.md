# KV-Cache Inception

**Reversible Monte Carlo Tree Search in Latent Space for Detecting and Stress-Testing Alignment Faking in Large Language Models**

*NeurIPS 2026 Datasets & Benchmarks Track submission*

**Paper Outline Here:** *https://www.overleaf.com/read/gcfjswwqwkhq#1ac1ec*

---

## Abstract

Current approaches to evaluating the safety and alignment of large language models (LLMs) rely predominantly on black-box behavioral testing. These methods are fundamentally blind to *alignment faking* — wherein a model strategically complies with safety constraints during evaluation while preserving misaligned capabilities in its latent representations.

We propose **KV-Cache Inception**, a framework combining three novel contributions: (1) a **multi-dimensional micro-telemetry matrix** fusing H-Neuron activation tracking with Representation Engineering (RepE) projections to provide a continuous signal for distinguishing genuine reasoning from compliance-driven fabrication; (2) a **Reversible MCTS algorithm** conducting Monte Carlo Tree Search directly in the model's KV-cache latent space using a mathematically reversible FP32 accumulator rollback mechanism; and (3) the formalization of **Orthogonal Escape** — the phenomenon whereby corrective pressure applied to monitored latent subspaces forces deceptive encodings into unmonitored orthogonal null spaces.

---

## Three Contributions

1. **Micro-Telemetry Matrix** `T_t ∈ ℝ^{2×L}` — per-layer, per-step fusion of H-Neuron stress `σ_H` and RepE honesty `ρ_R`. Defines four diagnostic states (Creative Synthesis, Contextual Fabrication, Alignment Faking, Orthogonal Escape).

2. **Reversible MCTS in KV-Cache Space** — first MCTS operating in continuous latent space rather than discrete token space. Memory: `O(M_KV + d·K_acc)` vs. `O(b^d · M_KV)` for standard parallel MCTS.

3. **Orthogonal Escape** — formal definition, OEI metric, TDS metric, and experimental protocol (Experiment 1: RepE steering magnitude sweep at α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}).

---

## Repo Structure

```
logomesh/                       # Core research package
  oracle.py                     # BaseModelClient ABC + OpenAIModelClient
  local_model.py                # LocalLlamaOracle — HF transformers, hidden states, KV cache API
  hneuron_monitor.py            # H-Neuron stress σ_H (bottom-up channel); score_per_layer()
  whitebox.py                   # RepE probes + PerLayerHonestyProjector ρ_R (top-down channel)
  telemetry_matrix.py           # T_t ∈ ℝ^{2×L}, DiagnosticState, compute_node_reward (Eq. 8)
  orthogonal_escape.py          # NullSpaceProjector, OEICalculator (Eq. 10), TDSCalculator
  kv_mcts.py                    # ReversibleMCTS, FP32Accumulator (Theorem 1), KVCacheNode
  search_policy.py              # UCB1 bandit (node selection)
  payload_library.py            # PayloadEntry + PayloadLibrary
  evidence_store.py             # Structured per-run logging
  graders.py                    # PluginGrader, RuleBasedGrader
  ablation.py                   # AblationConfig — experiment toggles
  threat_model.py               # ThreatModel, GoalTaxonomy

scripts/
  probe_kv_cache_mutability.py  # Phase 2 gate: validates in-place KV mutation + reversibility
  run_kv_mcts.py                # Phase 2 runner: Reversible MCTS with T_t, OEI, TDS output
  measure_lipschitz_drift.py    # Theorem 1 validation: FP32 accumulator drift vs naive bf16
  run_offline_mcts.py           # Phase A offline text-generation MCTS (baseline)
  train_lat_probes.py           # LAT probe training pipeline

tests/
  test_sage.py                  # logomesh module unit tests
  test_whitebox.py              # RepE / WhiteBoxEvaluator tests
  test_local_model_interface.py # Phase 2 LocalLlamaOracle interface tests
  test_phase2_modules.py        # TelemetryMatrix, OEI, TDS, FP32Accumulator, MCTS smoke tests

docs/
  NeurIPS/                      # Paper drafts (canonical: 04.02.2026-NeurIPS-Research-Proposal.tex)
  reviews/                      # Gap analysis and transition audit
  dataset/                      # Croissant schema stub (Phase 4)
```

---

## Quickstart

```bash
# Install dependencies
uv sync

# Run tests (no model required)
uv run pytest tests/ -v

# Phase 2 KV-MCTS (auto-downloads TinyLlama ~2GB)
uv run python scripts/run_kv_mcts.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --nodes 50 --depth 5 --branches 3

# Theorem 1 drift validation
uv run python scripts/measure_lipschitz_drift.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --n-cycles 200

# Phase A offline MCTS (download model first)
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ./models/llama-3.2-1b
uv run python scripts/run_offline_mcts.py --model ./models/llama-3.2-1b --episodes 100
```

## Phase 2 Gate (Portable)

Run the KV-cache mutability gate before building KV-MCTS internals.

Cross-platform probe (auto model download/cache + auto device):

```bash
uv run python scripts/probe_kv_cache_mutability.py --device auto
```

Use a local model folder if present:

```bash
uv run python scripts/probe_kv_cache_mutability.py --model ./models/llama-3.2-1b --device auto --use-chat-template
```

Use a specific HuggingFace model id (works on Windows/macOS/Linux):

```bash
uv run python scripts/probe_kv_cache_mutability.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device auto
```

H100-specific run (prefer bf16):

```bash
uv run python scripts/probe_kv_cache_mutability.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda --dtype bfloat16 --use-chat-template
```

Notes:
- `--model` accepts either a local path or a HuggingFace model id.
- `--device auto` resolves in order: `cuda` -> `mps` -> `cpu`.
- For gated model ids (for example Llama family), run `huggingface-cli login` first.
- Exit code `0` means gate passed; non-zero means mutable/revert checks did not pass or model access failed.

---

## Phase Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Repo cleanup — `logomesh/` package, research-oriented structure | ✅ Complete |
| 2 | Reversible KV-MCTS — `kv_mcts.py`, `telemetry_matrix.py`, `orthogonal_escape.py`, per-layer telemetry | ✅ Complete |
| 3 | Experiments 1–5 + Procrustes transfer + evaluation framework | 🔲 Next |
| 4 | Research dataset (Croissant format) + paper writing | 🔲 Not started |

---

## Team

- **Josh** — Team Lead / Data Scientist
- **Prof. Tianyu Shi** — Academic Advisor (McGill University)
- **Max** — Data Pipeline / Croissant format
- **Bakul** — Competitive Analysis / Org


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
