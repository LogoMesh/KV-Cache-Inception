# Reversible Monte Carlo Tree Search in Latent Space

**A Memory-Bounded Algorithm for Inference-Time Latent-Space Analysis of Frozen Language Models**

*Submission target: **EMNLP 2026 Findings** (deadline 2026-05-25)*

**Canonical paper draft:** [`docs/paper/kv-mcts-dimensional-escape.tex`](docs/paper/kv-mcts-dimensional-escape.tex)
**Read-only Overleaf view:** *https://www.overleaf.com/read/gcfjswwqwkhq#1ac1ec*

> **Venue note**: submission target was pivoted from NeurIPS 2026 (D&B then Main Track) to EMNLP 2026 Findings via ARR May 2026 on 2026-04-25 after a venue-pivot briefing.

---

## Abstract

Latent-space analysis of frozen large language models — including activation steering, representation engineering, and adversarial probing of internal monitors — increasingly requires search infrastructure that scales with the depth of intervention rather than with the branching factor of the search.

We introduce **Reversible Monte Carlo Tree Search in KV-cache latent space**, an inference-time algorithm that mutates a frozen Transformer's cached key/value tensors in place via FP32 accumulators and rolls back exact baseline state at each tree expansion. We prove that the FP32 accumulator guarantees zero-drift reversibility (Theorem 1; empirically validated to machine precision over 200 apply–revert cycles on Llama 3.2-1B and 3B), and we analyze a memory bound that is independent of branching factor — a constant-factor multiple of the base KV cache rather than the exponential blowup of parallel-cache MCTS.

To make the algorithm useful for representational analysis, we introduce a **per-layer telemetry matrix** that fuses bottom-up hallucination-associated-neuron activation tracking with top-down representation-engineering projections, providing a continuous, differentiable signal at each generation step.

As an application, we propose the **Orthogonal Escape Index (OEI)** — a targeted geometric measurement of the fraction of induced activation change that lies in the null space of a linear monitor — as a measurement framework complementing Bailey et al. (2024)'s cosine-to-baseline measurement of obfuscated activations under gradient-trained input attacks. We discuss measurement-pipeline prerequisites identified in our reference implementation and report initial observations consistent with Bailey's negative-result regime.

---

## Three Contributions

1. **Reversible MCTS in KV-cache latent space.** An inference-time algorithm that mutates cached K/V tensors via in-place arithmetic (`K_t ← K_t + α · d_K^(ℓ)`) and inverts each mutation exactly through an FP32 accumulator. **Theorem 1** guarantees zero-drift reversibility independent of search depth. Memory complexity is independent of branching factor — a constant-factor multiple of the base KV cache (vs. `O(b^d · M_KV)` for standard parallel-cache MCTS). Empirically validated to machine precision on Llama 3.2-1B-Instruct and Llama 3.2-3B-Instruct; reversibility on both confirmed via direct probe.

2. **Per-layer telemetry matrix for latent-space analysis.** A `2 × L` matrix `T_t` fusing bottom-up hallucination-associated-neuron activation (per Gao et al., 2025) with top-down representation-engineering projections (per Zou et al., 2023). Provides a continuous, differentiable signal at each generation step. Supports four diagnostic regime classifications (creative synthesis, contextual fabrication, alignment faking, orthogonal escape) keyed to joint patterns across the two channels.

3. **Orthogonal Escape Index (OEI) as a targeted null-space measurement.** OEI = `‖Proj_Null(P)(Δh)‖₂ / ‖Δh‖₂` — the fraction of induced activation change lying in the null space of a specific linear monitor `P`. Complements Bailey et al. (2024)'s global cosine-to-baseline measurement: where Bailey asks whether gradient-trained perturbations can achieve activation orthogonality (reporting a tentative negative result), OEI asks how much of any induced perturbation lands in directions invisible to a specific monitor.

---

## What's validated empirically

| Claim | Evidence | Where |
|---|---|---|
| FP32-accumulator reversibility (Theorem 1), 1B + 3B | drift = 0.00e+00 over 200 apply–revert cycles | `scripts/measure_lipschitz_drift.py` (2026-04-16); 3B probe `scripts/probe_kv_cache_mutability.py` (2026-05-05) |
| Reversible KV-cache mutability on consumer GPU | `gate_passed: true`, both 1B and 3B fit on 12 GB | `tmp/runs/2026-04-16/...` and 2026-05-05 3B probe |
| OEI measurement under inference-time KV-cache steering (Experiment 1) | Per-α monotonic variation: OEI 0.991 → 0.995 across α ∈ {0.1, 0.5, 1.0, 2.0, 5.0} on Llama 3.2-1B; ρ_R varies substantially (−1.31 → −0.53); TDS *decreases* with α (anti-OE signature) | `tmp/runs/2026-05-06_b6_post-fix.json`; reported in paper §5 Experiment 1 |
| Result interpretation | Consistent with Bailey 2024's negative-result regime — no orthogonal escape emerges from non-orthogonality-optimizing search at the model scales tested | paper §5 Interpretation paragraph |

---

## What's identified as known limitations

The paper's §7.4 ("Empirical Measurement Prerequisites and Initial Observations") documents two measurement-pipeline issues identified during this work:

1. **Steering-projection issue (resolved).** RepE probes are calibrated against residual-stream activations of dimension `d_model`, while the KV cache stores per-head representations of dimension `kv_heads × d_head`. An early version of the broadcast layer between probe space and cache space silently substituted a zero delta when shapes were incompatible, producing forward passes that were bit-identical across α. **Fix:** project probe directions through layer-specific key projection weights (`d_K^(ℓ) = W_K^(ℓ) d_probe^(ℓ)`, reshaped per head). Verified: per-α telemetry now varies monotonically.

2. **H-Neuron dense-scoring issue (identified, future work).** σ_H reports zero across all α regardless of input regime. A controlled three-prompt verification (hallucination-style, coercion-class, neutral) confirmed this. The per-neuron-ReLU formulation specified by the paper's σ_H equation was tentatively tested and σ_H still zero — the cause is downstream of per-neuron normalization. Most plausible remaining cause: position/context discrepancy between calibration-time hidden states (last-input-token, prompt-only forward pass) and inference-time hidden states (last-token, prompt+generation forward pass under steered cache). Resolution is post-EMNLP work.

Both issues are documented honestly in the paper as part of the contribution.

---

## Repository structure

```
logomesh/                       — core research package
  oracle.py                     — BaseModelClient ABC + OpenAIModelClient
  local_model.py                — LocalLlamaOracle: HF transformers, hidden states, KV cache API
  hneuron_monitor.py            — H-Neuron stress σ_H (bottom-up channel)
  whitebox.py                   — RepE probes + PerLayerHonestyProjector ρ_R
  telemetry_matrix.py           — TelemetryMatrix T_t ∈ ℝ^{2×L}, DiagnosticState, MCTS reward
  orthogonal_escape.py          — NullSpaceProjector, OEICalculator, TDSCalculator
  kv_mcts.py                    — ReversibleMCTS, FP32Accumulator (Theorem 1), W_K projection
  search_policy.py              — UCB1 bandit (node selection)
  payload_library.py            — PayloadEntry + PayloadLibrary
  croissant_export.py           — Croissant 1.1 + RAI export
  evidence_store.py             — structured per-run logging
  graders.py                    — grader interfaces
  ablation.py                   — AblationConfig (experiment toggles)
  threat_model.py               — ThreatModel, GoalTaxonomy, AttackSurface

scripts/
  probe_kv_cache_mutability.py  — KV-cache mutability + reversibility gate
  measure_lipschitz_drift.py    — Theorem 1 empirical validation
  run_kv_mcts.py                — Reversible MCTS runner (Experiment 1)
  run_offline_mcts.py           — Phase A text-generation MCTS (baseline)
  train_lat_probes.py           — LAT probe training (offline scorer path)
  export_kv_mcts_to_croissant.py — Runtime artifact → Croissant package
  collect_dataset.py            — Batch artifacts → dataset package

tests/
  test_sage.py                  — logomesh module unit tests (no LLM calls)
  test_whitebox.py              — RepE / PerLayerHonestyProjector tests
  test_local_model_interface.py — LocalLlamaOracle KV cache interface tests
  test_phase2_modules.py        — TelemetryMatrix, OEI, TDS, FP32Accumulator, MCTS smoke

docs/
  paper/                        — paper drafts (canonical: kv-mcts-dimensional-escape.tex)
  reviews/                      — citation audit + DRAFT-HOLD index
  logs/                         — session logs, pre-mortem findings, action plans
  dataset/                      — Croissant 1.1 schema artifacts
```

> **Note on `docs/logs/`**: this directory contains ~120 development-process records (session logs, pre-mortems, draft prose, audit reports, prompts). They preserve the project's decision history but are **not required reading** for understanding or reproducing the paper. New collaborators should start with the paper TeX, `CLAUDE.md`, this README, and the per-track reports under `docs/paper/` and the published reports referenced from the canonical TeX's Appendix~A.

---

## Quick start

```bash
# Install
uv sync

# Tests (no model required; must be 100% green before running experiments)
uv run pytest tests/ -v
```

### Reproduce the validated empirical claims (consumer GPU)

```bash
# 1. Validate KV-cache mutability + reversibility on your hardware
uv run python scripts/probe_kv_cache_mutability.py \
    --model meta-llama/Llama-3.2-1B-Instruct --device auto

# Optional: also probe 3B (~9.2 GB VRAM, fits on RTX 3060 12GB)
uv run python scripts/probe_kv_cache_mutability.py \
    --model meta-llama/Llama-3.2-3B-Instruct --device auto

# 2. Theorem 1 empirical validation
uv run python scripts/measure_lipschitz_drift.py \
    --model meta-llama/Llama-3.2-1B-Instruct --n-cycles 200

# 3. Experiment 1: alpha-sweep with current measurement infrastructure
uv run python scripts/run_kv_mcts.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --nodes 25 --depth 3 --branches 5 \
    --alpha-values 0.1 0.5 1.0 2.0 5.0 \
    --output ./tmp/runs/exp1.json
```

Notes:
- `--device auto` resolves: `cuda` → `mps` → `cpu`.
- For gated HuggingFace models, run `huggingface-cli login` first.
- Probe exit code 0 = gate passed.

### Out of scope for this submission

- gpt-oss-20b (21B MoE, Apache 2.0). Kept as Phase B target post-EMNLP.
- Procrustes 1B↔7B alignment.
- Sparse FP32 accumulators (current implementation uses full-shape; constant-factor memory bound holds in either form).
- True per-layer steering (current implementation uses a per-layer-projected single residual-stream direction).
- σ_H validation under hallucination-inducing input across the full alpha sweep (gated on the dense-scoring-path investigation; documented as future work in §7.4).

---

## Phase status

| Phase | Status |
|---|---|
| 1 — Repo cleanup, modular architecture | ✅ Complete |
| 2 — Reversible KV-MCTS, FP32 accumulator, telemetry | ✅ Complete; W_K projection fix landed 2026-05-06 |
| 3 — Experiment 1 alpha-sweep on 1B + 3B feasibility | ✅ 1B sweep complete with results in §5; 3B reversibility confirmed |
| 4 — Croissant dataset + paper writing | ⏳ In progress (paper draft on Option α+ recast) |

---

## Documentation pointers

For new sessions or new collaborators, read in this order:

1. **`docs/logs/2026-05-05_recast-action-plan.md`** — live coordination doc. Six pre-mortem decisions table (5/6 locked + 1 awaiting external sign-off), task status board with owner tags, inter-session messages.
2. **`docs/logs/2026-05-03_premortem-findings.md`** — six failure modes + addendum. Strategic context for the paper recast.
3. **Daily session logs** — `docs/logs/2026-05-05_session-log.md` through `2026-05-07_session-log.md`. Append-only narrative records.
4. **Diagnostic report** — `docs/logs/2026-05-05_diagnostic-report.md`. B6 + 3B probe + bug-localization + σ_H verification + GAP-C1-02 investigation.
5. **Canonical TeX** — `docs/paper/kv-mcts-dimensional-escape.tex`. 18 REVISION markers from 2026-05-06 + 2026-05-07 capture the full Option α+ recast.

---

## For prospective collaborators

The paper is in active preparation for EMNLP Findings 2026 (deadline 2026-05-25). Specific kinds of review feedback would be most useful:

- **Algorithm/systems review** — Theorem 1, the FP32-accumulator reversibility argument, the W_K projection fix for the steering-projection issue, the constant-factor memory analysis. Code: `logomesh/kv_mcts.py`, `tests/test_phase2_modules.py`.
- **Empirical/methodology review** — Experiment 1 results (paper §5), the Bailey-consistent interpretation of the OEI measurements, the σ_H finding documented in §7.4. Data: `tmp/runs/2026-05-06_b6_post-fix.json`.
- **Writing/framing review** — paper PDF + sense of the recast direction. Read `docs/logs/2026-05-03_premortem-findings.md` for strategic context, then the paper itself.
- **Domain review** (interpretability, alignment monitoring) — Bailey 2024 engagement at §2.2, Hypothesis 1 at §3, the position/context discrepancy hypothesis at §7.4.

The paper as currently drafted tells a coherent honest Option α+ story: validated theorem, working algorithm, two measurement-pipeline issues identified (one resolved, one specified), Bailey-complementing measurement framework, no overclaiming. Reviewer feedback is most useful when it is specific about which of these threads benefits from outside expertise.

---

## Hard rules for contributors

1. `uv run pytest tests/ -v` must be 100% green before any commit.
2. Phase 2 changes are additive only — never remove existing `logomesh/` interfaces.
3. Every TeX or code change gets a REVISION marker (`% [REVISION | <tag> | YYYY-MM-DD | ...]`).
4. The subject of monitoring is the model under study itself, not an external proxy.

---

## Citation

```bibtex
@inproceedings{logomesh2026reversible,
  title     = {Reversible {M}onte {C}arlo Tree Search in Latent Space:
               A Memory-Bounded Algorithm for Inference-Time Latent-Space Analysis
               of Frozen Language Models},
  author    = {[Authors]},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026}
}
```
