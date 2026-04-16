# Session Log — 2026-04-16

Session focus: Execute the first substantive KV-MCTS run with Croissant artifact export, then record reproducibility and validation evidence.

---

## Run Kickoff

1. Objective:
   - Perform a substantive (10+ node) `scripts/run_kv_mcts.py` run.
   - Export a Croissant package from the produced runtime artifact.
   - Capture command, seed, artifact paths, and key metrics for teammate/status updates.

2. Planned command shape:
   - `uv run python scripts/run_kv_mcts.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --nodes <N> --depth <D> --branches <B> --seed <S> --output <json> --croissant-dir <dir>`

3. Status:
   - Completed.

---

## Execution Attempts

1. Attempt A (TinyLlama online path):
    - Command:
       - `c:/Users/Josh/Documents/LogoMesh-NeurIPS/.venv/Scripts/python.exe scripts/run_kv_mcts.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --nodes 15 --depth 3 --branches 3 --seed 20260416 --output ./tmp/runs/2026-04-16/mcts_tinyllama_substantive_seed20260416.json --croissant-dir ./tmp/runs/2026-04-16/croissant_tinyllama_substantive_seed20260416`
    - Outcome:
       - Did not complete. Model resolution required HuggingFace network access; run terminated with `KeyboardInterrupt` during download path.

2. Attempt B (Qwen offline path, first try):
    - Command:
       - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 c:/Users/Josh/Documents/LogoMesh-NeurIPS/.venv/Scripts/python.exe scripts/run_kv_mcts.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --nodes 15 --depth 3 --branches 3 --seed 20260416 --output ./tmp/runs/2026-04-16/mcts_qwen_substantive_seed20260416.json --croissant-dir ./tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416`
    - Outcome:
       - Did not complete. Polling via new terminal commands interrupted the active foreground process (`KeyboardInterrupt`).

3. Attempt C (Qwen offline path, async mode):
    - Command:
       - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 c:/Users/Josh/Documents/LogoMesh-NeurIPS/.venv/Scripts/python.exe scripts/run_kv_mcts.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --nodes 15 --depth 3 --branches 3 --seed 20260416 --output ./tmp/runs/2026-04-16/mcts_qwen_substantive_seed20260416.json --croissant-dir ./tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416`
    - Execution mode:
       - Running in persistent async terminal (`ID=5c269174-9054-474a-958a-52c2986a498f`) to avoid interruption.
    - Outcome:
       - Completed successfully.

---

## Substantive Run Completion Record

1. Successful command:
   - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 c:/Users/Josh/Documents/LogoMesh-NeurIPS/.venv/Scripts/python.exe scripts/run_kv_mcts.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --nodes 15 --depth 3 --branches 3 --seed 20260416 --output ./tmp/runs/2026-04-16/mcts_qwen_substantive_seed20260416.json --croissant-dir ./tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416`

2. Runtime outcomes:
   - `PerLayerHonestyProjector calibrated: 29 layers, d=1536`
   - `MCTS complete: 15 nodes expanded`
   - `MCTS done in 1.8s. 16 nodes visited`
   - `Results saved -> tmp/runs/2026-04-16/mcts_qwen_substantive_seed20260416.json`
   - `Croissant package saved -> tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/metadata.json`
   - `Croissant records exported: 16`

3. Key metrics (from artifact JSON):
   - `model_id`: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
   - `n_nodes_visited`: `16`
   - `elapsed_seconds`: `1.8`
   - Top-node `mean_reward`: `0.432811`
   - Top-node `oei_score` (depth 1): `0.995935`
   - Top-node telemetry sample: `sigma_H_mean=1.0`, `rho_R_mean=0.714173`, `tds=0.024966`

4. Reproducibility metadata:
   - `seed`: `20260416`
   - `git_sha`: `203465dcff8a164cac3a55dbdbe5e2b2d0d699d6`
   - `git_dirty`: `true`
   - `run_started_utc`: `2026-04-16T07:07:42.201026Z`
   - `run_finished_utc`: `2026-04-16T07:10:07.370592Z`
   - `python_version`: `3.12.9`

5. Croissant package inventory:
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/interventions.csv`
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/metadata.json`
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/source_run.json`

6. Notes:
   - TinyLlama direct Hub path was not used for the final substantive run because this environment required online fetch at execution time.
   - The completed substantive run is therefore recorded against cached Qwen in strict offline mode.

---

## Architecture Review Brief Plan

Objective: prepare a focused  architecture brief, before scaling beyond the first substantive run.

### Message History:
[
redacted for privacy
]

### Suggested Docx Brief Outline (What else to Provide?)

1. Proposed repository architecture:
   - core modules and interfaces,
   - experiment runner flow,
   - evidence capture + Croissant export path.
2. First experiment set:
   - first substantive run configuration (already completed),
   - next 2--3 experiment increments to run immediately after review.
3. Review asks:
   - architecture risks,
   - missing abstractions,
   - experiment-order improvements ahead of first scaled runs.

### Run Artifacts to Reference in the Message and Docx

1. Substantive run JSON artifact:
   - `tmp/runs/2026-04-16/mcts_qwen_substantive_seed20260416.json`
2. Croissant package directory:
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/`
3. Croissant metadata:
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/metadata.json`
4. Croissant tabular export:
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/interventions.csv`
5. Source run snapshot in package:
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/source_run.json`
6. Execution log reference:
   - `docs/logs/2026-04-16_session-log.md`

1. Send the docx.
2. Include the artifact paths above in a compact bullet list.
3. Add one-paragraph summary of what feedback is most needed before first scaled execution.

---

## Phase 3 Gate Attempt — Llama-3.2-1B-Instruct

HuggingFace access to `meta-llama/Llama-3.2-1B-Instruct` confirmed (Meta Llama 3.2 gating group, approved 2026-03-20).

### Gate sequence (EXPERIMENTS.md Chunk 0):

1. Step 1 — KV-cache mutability probe:
   - Command: `uv run python scripts/probe_kv_cache_mutability.py --model meta-llama/Llama-3.2-1B-Instruct --device auto`
   - Status: **PASSED**
   - Output: `gate_passed: true`, `mutable_in_place: true`, `reversible_with_copy_restore: true`
   - `max_logit_delta_after_mutation: 0.171875`, `max_logit_delta_after_revert: 0.0`
   - Device: `cuda`, dtype: `bfloat16`, model source: `huggingface_id_or_cache`

2. Step 2 — KV-MCTS smoke test (10 nodes):
   - Command: `uv run python scripts/run_kv_mcts.py --model meta-llama/Llama-3.2-1B-Instruct --nodes 10 --depth 3 --branches 2 --seed 20260416 --output ./tmp/runs/2026-04-16/mcts_llama_gate_seed20260416.json`
   - Status: **PASSED**
   - Output: `MCTS complete: 10 nodes expanded`, `11 nodes visited`, `elapsed=0.6s`
   - `steer_layer=8`, `residual_norm=0.00e+00` (exact reversal)
   - Top-node telemetry: `sigma_H=0.0032`, `rho_R=0.3622`, `TDS=0.0084`, `OEI=0.8830`, `reward=0.2144`
   - Artifact: `tmp/runs/2026-04-16/mcts_llama_gate_seed20260416.json`
   - Note: telemetry values are non-degenerate vs Qwen (sigma_H not saturated, OEI not 0.996), but still identical across all top-5 nodes — ρ_R fix (Chunk 1A) still required before Experiment 1 results are paper-valid.

3. Step 3 — Lipschitz drift validation:
   - Command: `uv run python scripts/measure_lipschitz_drift.py --model meta-llama/Llama-3.2-1B-Instruct --n-cycles 200`
   - Status: **PASSED** (after two bug fixes to the script — see below)
   - Results (200 cycles, alpha=0.5):

     | Cycle | Naive bf16 `||K_n-K_0||_inf` | FP32 accumulator |
     |-------|------------------------------|-----------------|
     | 1     | 1.56e-02                     | 0.00e+00        |
     | 50    | 1.28e+00                     | 0.00e+00        |
     | 100   | 2.84e+00                     | 0.00e+00        |
     | 150   | 4.41e+00                     | 0.00e+00        |
     | 200   | 5.97e+00                     | 0.00e+00        |

   - Naive drift grows linearly (~+1.56 per 50 cycles). FP32 accumulator exact at zero throughout.
   - Artifact: `drift_results.csv`
   - Script fixes applied:
     - Bug 1: `dk_t = torch.from_numpy(dk_raw).float()` created a CPU tensor; added `.to(k_naive.device)` for CUDA compatibility.
     - Bug 2: Naive path did independent add/sub pairs per iteration, reaching a fixed point instead of accumulating drift. Redesigned to apply N adds then N subtracts cumulatively — faithfully simulates reversal error after N cycles.
     - Bug 3: Windows cp1252 UnicodeEncodeError on `‖` and `ε` in summary print; replaced with ASCII equivalents.

### Gate Verdict: PASSED (all 3 steps)

All EXPERIMENTS.md Chunk 0 criteria met on `meta-llama/Llama-3.2-1B-Instruct`:
- KV-cache mutable in-place and reversible with exact zero residual ✓
- MCTS smoke test: 10 nodes, non-degenerate telemetry (sigma_H not saturated vs Qwen) ✓
- Lipschitz drift: FP32 accumulator exact (0.00e+00) across 200 cycles; naive bf16 grows linearly ✓

Remaining blocker before Experiment 1 paper-valid results: ρ_R raw dot product fix (EXPERIMENTS.md Chunk 1A).

### Follow-on:
- `docs/NeurIPS/2026-04-16_Architecture-Brief-for-Aladdin.docx` updated to reflect gate outcomes.

---

## Architecture Brief

- Created: `docs/NeurIPS/2026-04-16_Architecture-Brief-for-Aladdin.docx`
- Contents: 5 sections — repo architecture, first Qwen run summary (with flat-telemetry disclosure), known gaps, next increments, review asks for Aladdin.
- Status: **Draft — pending gate results before send.**

---

## Assistant-Driven Changes Summary

1. Files created/updated in this session:
   - Created `docs/logs/2026-04-16_session-log.md`.
   - Updated `docs/CLAUDE_CONTEXT_BRIEF.md` to index today's log as current active handoff record.
   - Created `docs/NeurIPS/2026-04-16_Architecture-Brief-for-Aladdin.docx` (architecture brief for Aladdin).

2. Validation outcomes captured:
   - Substantive runtime execution completed successfully (offline Qwen path).
   - Artifact checks confirmed output JSON + Croissant package files.
   - HuggingFace authentication confirmed (joshhickson, Llama-3.2-1B-Instruct access verified).
   - Phase 3 gate (Llama-3.2-1B-Instruct) in progress — results pending.

3. Commit status:
   - No commit created in this run-only/logging step.
