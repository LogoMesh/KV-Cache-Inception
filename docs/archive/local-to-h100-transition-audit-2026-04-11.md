# Local to H100 Transition Audit

Date: 2026-04-11

## Scope

This audit evaluates whether the current repository and paper plan support a smooth transition from local development machines (Windows/macOS/Linux, consumer GPUs) to H100 execution.

In-scope evidence:
- Active repository files only.
- No Claude transcript artifacts used as evidence.

Primary evidence files:
- [../../docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex](../../docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex)
- [../../CLAUDE.md](../../CLAUDE.md)
- [../../README.md](../../README.md)
- [../../pyproject.toml](../../pyproject.toml)
- [../../uv.lock](../../uv.lock)
- [../../logomesh/local_model.py](../../logomesh/local_model.py)
- [../../logomesh/hneuron_monitor.py](../../logomesh/hneuron_monitor.py)
- [../../logomesh/whitebox.py](../../logomesh/whitebox.py)
- [../../scripts/probe_kv_cache_mutability.py](../../scripts/probe_kv_cache_mutability.py)
- [../../scripts/run_offline_mcts.py](../../scripts/run_offline_mcts.py)
- [../../scripts/train_lat_probes.py](../../scripts/train_lat_probes.py)
- [../../docs/dataset/croissant_schema_stub.json](../../docs/dataset/croissant_schema_stub.json)

## Executive Readiness Verdict

Overall transition readiness: BLOCKED

Readiness summary:
- Cross-platform KV gate script: PARTIAL READY
- End-to-end offline runner on active tree: BLOCKED
- Paper algorithm parity (Phase 2 core): NOT STARTED
- H100 scaling and experiment stack: NOT STARTED
- Dataset/reproducibility packaging for Datasets and Benchmarks: PARTIAL

Why blocked:
1. Current runner/training entrypoints instantiate null plugins and are not runnable without additional wiring.
2. Core paper modules (KV-MCTS, telemetry matrix, orthogonal escape metrics) are still absent.
3. Dependency strategy is pinned to CUDA index, which is brittle for non-CUDA machines.

## What Is Already Strong

1. Base package and abstractions exist.
- Base model interface exists in [../../logomesh/oracle.py](../../logomesh/oracle.py).
- Local model has additive step and cache interfaces in [../../logomesh/local_model.py](../../logomesh/local_model.py).

2. KV mutability gate is portable and practical.
- Portable gate section exists in [../../README.md](../../README.md) at line 74.
- Auto device fallback logic exists in [../../scripts/probe_kv_cache_mutability.py](../../scripts/probe_kv_cache_mutability.py) at lines 66-74.
- Auto dtype selection includes bf16 support in [../../scripts/probe_kv_cache_mutability.py](../../scripts/probe_kv_cache_mutability.py) at lines 78-91.

3. Telemetry foundations are present.
- H-neuron monitor supports dense and MoE scoring paths in [../../logomesh/hneuron_monitor.py](../../logomesh/hneuron_monitor.py).
- White-box scaffolding and evaluator wiring exist in [../../logomesh/whitebox.py](../../logomesh/whitebox.py).

4. Tests exist for core module behavior.
- Additive local model interface tests exist in [../../tests/test_local_model_interface.py](../../tests/test_local_model_interface.py).
- Broad module tests exist in [../../tests/test_sage.py](../../tests/test_sage.py) and [../../tests/test_whitebox.py](../../tests/test_whitebox.py).

## Critical Blockers

### C1. Offline and training scripts are currently non-runnable due to null plugin wiring

Evidence:
- All scenario configs currently set plugin_class to None in [../../scripts/run_offline_mcts.py](../../scripts/run_offline_mcts.py) at lines 86, 117, 142, 179, 209.
- Runtime path still instantiates plugin_class directly in [../../scripts/run_offline_mcts.py](../../scripts/run_offline_mcts.py) at lines 501-503.
- Training script does the same in [../../scripts/train_lat_probes.py](../../scripts/train_lat_probes.py) at lines 302-304.

Impact:
- Active local-to-H100 operational path is blocked before scaling concerns are even reached.

Required fix:
- Restore real plugin adapters (or replace plugin dependency with research-native configs) before any platform transition work can be validated end-to-end.

### C2. Local model loading path is not fully portable for model source and device strategy

Evidence:
- Loader currently requires local path existence in [../../logomesh/local_model.py](../../logomesh/local_model.py) at line 44.
- Load auto-select only chooses cuda or cpu in [../../logomesh/local_model.py](../../logomesh/local_model.py) at line 108 (no mps auto path).
- Loader passes dtype and device_map in [../../logomesh/local_model.py](../../logomesh/local_model.py) at lines 59-60.

Impact:
- Local macOS developer path is weaker than the gate script behavior.
- Operational behavior may diverge between probe script and main runtime path.

Required fix:
- Align local_model loading semantics with probe script model/device/dtype resolution.

### C3. Dependency strategy is CUDA-index pinned, creating cross-platform install risk

Evidence:
- Torch source index override in [../../pyproject.toml](../../pyproject.toml) at line 33.
- Explicit custom index in [../../pyproject.toml](../../pyproject.toml) at lines 35-38.
- Lock metadata also pins torch to CUDA index in [../../uv.lock](../../uv.lock) at line 364.

Impact:
- Windows/macOS/Linux users without CUDA may hit install friction or inconsistent dependency behavior.

Required fix:
- Split environment profiles (cpu, cuda, h100) or use optional dependency groups instead of one CUDA-pinned default.

### C4. Paper core algorithm modules are absent

Evidence:
- Missing files:
  - [../../logomesh/kv_mcts.py](../../logomesh/kv_mcts.py)
  - [../../logomesh/telemetry_matrix.py](../../logomesh/telemetry_matrix.py)
  - [../../logomesh/orthogonal_escape.py](../../logomesh/orthogonal_escape.py)
- Paper requires these capabilities across method and experiments in [../../docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex](../../docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex) at lines 180, 266, 306, 313.

Impact:
- Cannot execute the transition target architecture on any hardware tier yet.

Required fix:
- Implement paper core modules before H100 scale-out work.

## High-Priority Transition Gaps

### H1. Telemetry output shape mismatch vs paper requirement

Evidence:
- Paper requires per-step, per-layer telemetry matrix T_t in [../../docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex](../../docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex) around lines 151-177.
- Current monitor score API returns scalar in [../../logomesh/hneuron_monitor.py](../../logomesh/hneuron_monitor.py) at line 158.
- Offline runner consumes scalar score directly in [../../scripts/run_offline_mcts.py](../../scripts/run_offline_mcts.py) at line 406.

Impact:
- Current telemetry cannot support paper-level diagnostic state classification and metrics.

### H2. Reward semantics mismatch with paper node reward

Evidence:
- Current runner fallback and reward composition paths in [../../scripts/run_offline_mcts.py](../../scripts/run_offline_mcts.py) at lines 408 and 447.
- Paper defines telemetry-driven node reward in [../../docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex](../../docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex) around line 230.

Impact:
- Search objective remains legacy/offline-competition style rather than paper objective.

### H3. White-box evaluator has real orchestration but several core analyzers are stubs

Evidence:
- Probe wiring exists in [../../logomesh/whitebox.py](../../logomesh/whitebox.py) at line 1018.
- Stubbed internals return placeholder outputs:
  - [../../logomesh/whitebox.py](../../logomesh/whitebox.py) at line 1168 (_run_gcg_optimization)
  - [../../logomesh/whitebox.py](../../logomesh/whitebox.py) at line 1183 (_analyze_attention_heads)
  - [../../logomesh/whitebox.py](../../logomesh/whitebox.py) at line 1190 (_sample_loss_landscape)

Impact:
- H100 experiments relying on these probes are not research-complete yet.

### H4. Experiment script surface required by paper is missing

Evidence:
- Paper explicitly specifies Experiments 1-5 in [../../docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex](../../docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex) at lines 266-278.
- No script files matching experiment*.py currently under scripts.

Impact:
- No reproducible execution path for the planned H100 experiment program.

## Medium Gaps

### M1. MoE telemetry path assumptions are not fully validated in runtime path

Evidence:
- Local model comments indicate router logits expected for MoE in [../../logomesh/local_model.py](../../logomesh/local_model.py) around line 206.
- CLAUDE guidance requires router entropy path for gpt-oss-20b in [../../CLAUDE.md](../../CLAUDE.md) at line 101.

Impact:
- gpt-oss-20b monitoring path may need explicit runtime validation hooks before H100 launch.

### M2. Dataset schema exists but remains a stub with TODO work

Evidence:
- Croissant intervention schema exists in [../../docs/dataset/croissant_schema_stub.json](../../docs/dataset/croissant_schema_stub.json) at lines 15, 68, 74, 116.
- TODO block remains in [../../docs/dataset/croissant_schema_stub.json](../../docs/dataset/croissant_schema_stub.json) at line 124.

Impact:
- Dataset and Benchmarks track packaging is not yet publication-ready.

### M3. Reproducibility controls are partial and distributed across modules

Evidence:
- Train script uses fixed seeds in [../../scripts/train_lat_probes.py](../../scripts/train_lat_probes.py) at lines 250 and 253.
- Whitebox config includes random_seed in [../../logomesh/whitebox.py](../../logomesh/whitebox.py) around line 974.

Impact:
- Need a unified reproducibility policy (global seeds, deterministic settings, artifact hashing, run manifests) for local-to-H100 consistency.

## Required Transition Sequence (Do Not Reorder)

### Phase 0: Unblock runtime path (hard gate)

Exit criteria:
1. run_offline_mcts and train_lat_probes execute without null-plugin failures.
2. local_model can load local path or HuggingFace id with consistent device/dtype handling.
3. Dependency install profiles are valid for cpu, mps, and cuda environments.

### Phase 1: Cross-platform single-node hardening

Targets:
- Windows + CUDA developer machine.
- macOS + MPS developer machine.
- Linux + CPU fallback machine.

Exit criteria:
1. KV mutability gate passes on each platform class where supported.
2. Core scripts run in dry-run/smoke mode with deterministic config and stable outputs.

### Phase 2: Paper-core implementation readiness

Targets:
- Implement and validate KV-MCTS, telemetry_matrix, orthogonal_escape modules.
- Wire per-step, per-layer telemetry outputs and paper reward function.

Exit criteria:
1. Smoke test executes reversible latent-space search and rollback cycle.
2. T_t, OEI, TDS are produced and logged from active code paths.

### Phase 3: H100 single-node readiness

Targets:
- Run Experiment 1 and Experiment 3 subsets on H100.
- Validate MoE telemetry path on gpt-oss-20b runtime.

Exit criteria:
1. Stable memory profile matches reversible MCTS expectations.
2. Drift and rollback checks pass over extended cycles.

### Phase 4: H100 program scale and publication pipeline

Targets:
- Implement remaining experiment scripts and evaluation framework.
- Complete Croissant schema and reproducibility/audit trails.

Exit criteria:
1. Experiment 1-5 script suite runs with reproducible manifests.
2. Dataset packaging is complete for Datasets and Benchmarks submission quality.

## Platform Validation Matrix

### A. Environment setup checks

1. Windows CUDA check
- Command: uv sync
- Expected: clean install with CUDA-compatible torch profile.

2. macOS MPS check
- Command: uv sync
- Expected: clean install without CUDA-only dependency conflict.

3. Linux CPU check
- Command: uv sync
- Expected: clean install without GPU requirement failures.

### B. Runtime checks

1. KV gate check
- Command: uv run python scripts/probe_kv_cache_mutability.py --device auto
- Expected: structured JSON output with gate_passed true on supported model/hardware.

2. Offline runner check
- Command: uv run python scripts/run_offline_mcts.py --help
- Expected: script loads and parses args; runtime smoke mode does not fail on plugin initialization.

3. Probe training check
- Command: uv run python scripts/train_lat_probes.py --help
- Expected: script loads and parses args; runtime smoke mode does not fail on plugin initialization.

### C. H100 checks

1. bf16 gate check
- Command: uv run python scripts/probe_kv_cache_mutability.py --model meta-llama/Llama-3.2-1B-Instruct --device cuda --dtype bfloat16 --use-chat-template
- Expected: gate_passed true with reversible restoration.

2. MoE telemetry check
- Expected: router-logit path is active and logged when using gpt-oss-20b.

## Suggested Owner/Effort Breakdown

1. Runtime unblock (plugins + script execution): Owner Josh, Effort M
2. local_model portability alignment: Owner Josh, Effort M
3. Dependency profile split for cpu/mps/cuda/h100: Owner Max or Josh, Effort M
4. KV-MCTS core modules: Owner Josh + Asuka, Effort L
5. Telemetry matrix + OEI/TDS wiring: Owner Josh + Asuka, Effort L
6. Experiment scripts 1-5 and evaluation framework: Owner team, Effort L
7. Croissant completion and publication pipeline: Owner Max, Effort M

## Final Assessment

The project has good structural foundations and a strong portable KV gate, but it is not yet in a runnable state for a smooth local-to-H100 transition. The first milestone is not H100 scaling itself; it is restoring runnable end-to-end script paths and aligning runtime portability with the gate script behavior. After that, paper-core algorithm implementation is the true critical path.
