# LogoMesh — Research Session Guide

> **Read this file at the start of every session.**
> Then check `docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex` for the current paper state
> and `docs/reviews/phase-a-gaps-2026-04-11.md` for the verified gap analysis.

---

## Project

**KV-Cache Inception: Reversible Monte Carlo Tree Search in Latent Space for Detecting and
Stress-Testing Alignment Faking in Large Language Models**

NeurIPS 2026 Datasets & Benchmarks track. Submission deadline: **May 6, 2026**.

Academic advisor: Prof. Tianyu Shi (McGill). Contributors: Josh, Max, Bakul, Asuka.

---

## Three Paper Contributions

1. **Multi-Dimensional Micro-Telemetry Matrix** — H-Neuron activation (bottom-up) fused with
   RepE linear probes (top-down) into a per-layer, per-step matrix `T_t ∈ ℝ^{2×L}`. Defines
   four diagnostic states: Creative Synthesis, Contextual Fabrication, Alignment Faking,
   Orthogonal Escape.

2. **Reversible MCTS in KV-Cache Latent Space** — MCTS that mutates KV-cache tensors in-place
   (`K_t ← K_t + α·d_K`) then reverses exactly via FP32 accumulators. Reduces memory from
   `O(b^d · M_KV)` to `O(M_KV + d·K_acc)`. Enables deep MCTS on a single H100 for 20B models.

3. **Orthogonal Escape Formalization** — Formal definition + experimental protocol for deceptive
   encodings redistributing into null spaces of linear monitors under corrective pressure.
   Key metrics: OEI (Orthogonal Escape Index) and TDS (Telemetry Divergence Score).

---

## Repo Structure

```
logomesh/                   — core research package
  oracle.py                 — BaseModelClient ABC + OpenAIModelClient
  local_model.py            — LocalLlamaOracle: HF transformers, hidden states, KV cache API
                              generate_one_step(), get_kv_cache(), set_kv_cache()
  hneuron_monitor.py        — H-Neuron stress σ_H; score() scalar + score_per_layer() → [L]
  whitebox.py               — RepE probes + PerLayerHonestyProjector ρ_R^(l) + steering vecs
  telemetry_matrix.py       — TelemetryMatrix T_t ∈ ℝ^{2×L}, DiagnosticState, compute_node_reward
  orthogonal_escape.py      — NullSpaceProjector, OEICalculator (Eq. 10), TDSCalculator
  kv_mcts.py                — ReversibleMCTS, FP32Accumulator (Theorem 1), KVCacheNode, MCTSConfig
  search_policy.py          — UCB1 bandit (node selection reuse)
  payload_library.py        — PayloadEntry + PayloadLibrary (Phase 4: extend to research dataset)
  croissant_export.py       — Croissant 1.1 + RAI export helpers + schema-shape validation
  evidence_store.py         — structured per-run logging
  graders.py                — PluginGrader, RuleBasedGrader, CompositeGrader
  ablation.py               — AblationConfig (experiment toggles)
  threat_model.py           — ThreatModel, GoalTaxonomy, AttackSurface

scripts/
  probe_kv_cache_mutability.py — Phase 2 gate: validates in-place mutation + reversibility
  run_kv_mcts.py               — Phase 2 runner: ReversibleMCTS with T_t, OEI, TDS JSON output
  export_kv_mcts_to_croissant.py — Runtime artifact -> Croissant package exporter
  measure_lipschitz_drift.py   — Theorem 1 validation: FP32 accumulator vs naive bf16 drift
  run_offline_mcts.py          — Phase A text-generation MCTS (baseline)
  train_lat_probes.py          — LAT probe training (Phase 3: retrain for paper semantics)

tests/
  test_sage.py                 — logomesh module unit tests (no LLM calls)
  test_whitebox.py             — RepE / WhiteBoxEvaluator tests
  test_local_model_interface.py — Phase 2 LocalLlamaOracle KV cache interface tests
  test_phase2_modules.py       — TelemetryMatrix, OEI, TDS, FP32Accumulator, MCTS smoke

docs/
  NeurIPS/                  — paper drafts (canonical: 04.02.2026-NeurIPS-Research-Proposal.tex)
  reviews/                  — gap analysis + transition audit
  logs/                     — session logs
  dataset/                  — Croissant schema stub (Phase 4)
```

---

## Phase Status

| Phase | Description | Status |
|---|---|---|
| 1 | Repo cleanup — `logomesh/` package, `BaseModelClient` interface, deleted competition code | ✅ Complete |
| A | Local 1B model offline MCTS foundations (H-Neurons, LAT probes, payload library) | ✅ Foundations built |
| 2 | Reversible KV-MCTS — `kv_mcts.py`, `telemetry_matrix.py`, `orthogonal_escape.py`, per-layer telemetry | ✅ Complete |
| 3 | Experiment infrastructure (5 experiment scripts, Procrustes, evaluation framework) | 🔲 Next |
| 4 | Research dataset (Croissant), paper writing | 🔲 Not started |

**Phase 3 gate: PASSED (2026-04-16) on `meta-llama/Llama-3.2-1B-Instruct`.**
All three Chunk 0 steps completed: KV-cache mutability probe ✓, 10-node MCTS smoke test ✓,
Lipschitz drift validation (200 cycles, FP32 accumulator exact at 0.00e+00) ✓.
Full record: `docs/logs/2026-04-16_session-log.md`.

---

## Hardware

| Environment | Hardware | Models |
|---|---|---|
| Phase 2 dev | RTX 3060 12GB | TinyLlama / Llama-3.2-1B-Instruct (~2GB VRAM) |
| Phase 3 experiments | Tianyu Shi's lab — 8× H100 80GB | gpt-oss-20b (MoE, ~16GB floor) |

---

## Key Models

| Model | Use | Access |
|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | Phase 2 KV-MCTS prototype | HuggingFace |
| `[7B TBD]` | Procrustes transfer target (Experiment 5) | TBD |
| `openai/gpt-oss-20b` | Phase 3 scaling (MoE, H100 only) | HuggingFace (Apache 2.0) — **TBC: not yet accessed** |

**gpt-oss-20b gotcha:** MoE (32 experts, 4 active/token). Use router logit entropy for
H-Neuron monitoring — do NOT use dense MLP neuron monitoring. See `hneuron_monitor.py`.
**Note:** Model availability and Apache 2.0 licence claim have not been verified against HuggingFace. Confirm before Phase 3 planning.

---

## Hard Rules

1. **No competition framing** in any code, comments, or docs. The paper is standalone.
2. **Run tests before every commit:** `uv run pytest tests/ -v` — must be 100% green.
3. Phase 2 changes are **additive only** — never remove existing `logomesh/` module interfaces.
4. The paper's subject of monitoring is the **model under study itself** (detecting alignment
   faking in its own representations), not an external proxy model.

---

## Key Paper Equations (Quick Reference)

**Telemetry matrix** (Eq. 3):
`T_t = [[σ_H^(1)..σ_H^(L)], [ρ_R^(1)..ρ_R^(L)]]`

**MCTS reward** (Eq. 8):
`r(node) = λ₁·σ̄_H + λ₂·(1 - ρ̄_R) + λ₃·Divergence(σ_H, ρ_R)`

**OEI** (Eq. 10):
`OEI = ‖Proj_Null(P)(h_steered)‖₂ / ‖Proj_Null(P)(h_base)‖₂`

**Reversibility** (Eq. 6):
`A^(ℓ) ← A^(ℓ) + α·d^(ℓ)` (FP32), `K_t ← K_base + cast_bf16(A^(ℓ))`

---

## Dev Commands

```bash
uv run pytest tests/ -v                           # must be 100% green before every commit
uv sync                                            # install/update deps

# Phase 2 runners (auto-download TinyLlama ~2GB on first run)
uv run python scripts/run_kv_mcts.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --nodes 50 --depth 5 --branches 3
uv run python scripts/measure_lipschitz_drift.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --n-cycles 200
uv run python scripts/probe_kv_cache_mutability.py --device auto

# Phase A model download (for run_offline_mcts.py)
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ./models/llama-3.2-1b
```
