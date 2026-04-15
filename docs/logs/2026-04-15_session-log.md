# Session Log — 2026-04-15

Session focus: Prepare the repository for first experiments with a controlled, reviewable sequence.

---

## Planned Steps (from readiness review)

1. Step 1 — Runner exit reliability hardening:
   - Target file: `scripts/run_kv_mcts.py`
   - Goal: prevent Windows terminal Unicode encode failures in top-node summary output so successful runs exit with code 0.

2. Step 2 — Experiment reproducibility controls:
   - Target file: `scripts/run_kv_mcts.py`
   - Goal: add explicit seed handling (Python, NumPy, Torch) and persist run metadata (seed, git sha, model, config) into output JSON.

---

## Status

- Step 1: Completed.
- Baseline commit after Step 1: Completed (`d794ad2`).
- Step 2: Completed (code + validation).
- Step 3 (tmp hygiene before push): Completed (`ee31085`).

---

## Step 3 Completion Record

1. Added `tmp/` to `.gitignore` to keep transient runtime artifacts out of commits.
2. Commit:
   - Hash: `ee31085`
   - Message: `chore: ignore tmp runtime artifacts`

---

## Step 1 Completion Record

### Code change

1. Updated `scripts/run_kv_mcts.py` to avoid Windows terminal Unicode print failures:
   - Changed logger line `Results saved → ...` to ASCII `Results saved -> ...`.
   - Changed top-node summary labels from Unicode (`σ̄_H`, `ρ̄_R`) to ASCII (`sigma_H`, `rho_R`).

### Validation run

1. Command (offline cached Qwen, minimal settings):
   - `python scripts/run_kv_mcts.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --nodes 1 --depth 1 --branches 1 --output ./tmp/mcts_qwen_offline_n1_step1check.json`
   - Environment: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`.

2. Result:
   - Exit code: `0`.
   - JSON output written successfully to `tmp/mcts_qwen_offline_n1_step1check.json`.
   - Top-node summary printed without UnicodeEncodeError.

### Hold point

1. Baseline commit created before Step 2 implementation:
   - Commit: `d794ad2`
   - Message: `phase3 prep: baseline commit after step1 runner reliability`

---

## Step 2 Completion Record

### Code changes

1. Updated `scripts/run_kv_mcts.py` with explicit reproducibility controls:
   - Added `--seed` CLI argument (default `42`).
   - Added `_configure_reproducibility(seed)` to seed Python `random`, NumPy, and Torch.
   - Enabled deterministic cuDNN settings (`deterministic=True`, `benchmark=False`) when available.

2. Added run metadata persistence to output JSON (`run_metadata` block):
   - `seed`
   - `git_sha`
   - `git_dirty`
   - `model`
   - `config`
   - `run_started_utc`
   - `run_finished_utc`
   - `python_version`
   - `command`

### Validation

1. Determinism and metadata helper checks via direct Python execution:
   - Imported `_configure_reproducibility`, `_git_commit_sha`, `_git_is_dirty` from `scripts/run_kv_mcts.py`.
   - Re-seeded twice with `1337` and verified identical sampled values across Python/NumPy/Torch.
   - Verified git SHA and dirty-state helpers both returned valid values.
   - Result: pass (`seed_reproducible True`, `git_sha_present True`, `git_dirty_known True`).

2. Full test gate:
   - Command: `uv run pytest tests/ -v`
   - Result: `130 passed`.

3. Stable minimal offline runtime check (cached Qwen, seeded):
   - Command: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python scripts/run_kv_mcts.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --nodes 1 --depth 1 --branches 1 --seed 1337 --output ./tmp/mcts_qwen_offline_n1_step2check.json`
   - Result: completed successfully; output written to `tmp/mcts_qwen_offline_n1_step2check.json`.
   - Runtime summary:
     - `PerLayerHonestyProjector calibrated: 29 layers, d=1536`
     - `MCTS complete: 1 nodes expanded`
     - `Results saved -> tmp\\mcts_qwen_offline_n1_step2check.json`

4. Artifact metadata verification:
   - `seed: 1337`
   - `git_sha`: present
   - `git_dirty: False`
   - `model: Qwen/Qwen2.5-Coder-1.5B-Instruct`
   - `config`: present
   - `run_started_utc`, `run_finished_utc`, `python_version`, `command`: present
