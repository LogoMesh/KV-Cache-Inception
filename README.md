# LogoMesh — KV-Cache Inception

**Reversible Monte Carlo Tree Search in Latent Space for Detecting and Stress-Testing Alignment Faking in Large Language Models**

*NeurIPS 2026 Datasets & Benchmarks Track submission*

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
logomesh/                 # Core research package
  oracle.py               # BaseModelClient ABC + OpenAIModelClient
  local_model.py          # LocalLlamaOracle — HF transformers + telemetry
  hneuron_monitor.py      # H-Neuron monitor (bottom-up channel)
  whitebox.py             # RepE probes / WhiteBoxEvaluator (top-down channel)
  search_policy.py        # UCB1 bandit (base for Phase 2 KV-MCTS)
  payload_library.py      # PayloadEntry + PayloadLibrary
  evidence_store.py       # Structured per-run logging
  graders.py              # PluginGrader, RuleBasedGrader
  ablation.py             # AblationConfig — experiment toggles
  threat_model.py         # ThreatModel, GoalTaxonomy

scripts/
  run_offline_mcts.py     # Phase A offline MCTS runner
  train_lat_probes.py     # LAT probe training pipeline

tests/
  test_sage.py            # logomesh unit tests
  test_whitebox.py        # RepE / WhiteBoxEvaluator tests

docs/
  NeurIPS/                # Paper drafts
  reviews/                # Gap analysis
  dataset/                # Croissant schema (Phase 4)
```

---

## Quickstart

```bash
# Install dependencies
uv sync

# Download Phase A model
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ./models/llama-3.2-1b

# Run tests
uv run pytest tests/ -v

# Phase A offline MCTS (requires model download above)
uv run python scripts/run_offline_mcts.py --model ./models/llama-3.2-1b --episodes 100
```

---

## Phase Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Repo cleanup — `logomesh/` package, research-oriented structure | ✅ Complete |
| 2 | Reversible KV-MCTS (`kv_mcts.py`, `telemetry_matrix.py`, `orthogonal_escape.py`) | 🔲 Next |
| 3 | Experiments 1–5 + Procrustes transfer + evaluation framework | 🔲 Not started |
| 4 | Research dataset (Croissant format) + paper writing | 🔲 Not started |

---

## Team

- **Josh** — Team Lead / Data Scientist
- **Prof. Tianyu Shi** — Academic Advisor (McGill University)
- **Max** — Data Pipeline / Croissant format
- **Bakul** — Competitive Analysis / Org
- **Asuka** — Formal Methods / Ablation Studies

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
