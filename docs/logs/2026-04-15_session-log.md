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

## Croissant Alignment TODOs (Requested Hold Before Execution)

Status: On hold by request; do not execute until explicitly resumed.
Historical note: this hold was later lifted in the same session. See "Croissant Alignment Execution (Resumed)" below for completion records.

- [ ] TODO 1: Add explicit Croissant evidence-format language to the canonical paper draft (`docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex`) so collection and release format are unambiguous.
- [ ] TODO 2: Add a Croissant exporter path that maps runtime artifacts (for example `run_kv_mcts.py` JSON outputs and evidence records) into the schema defined in `docs/dataset/croissant_schema_stub.json`.
- [ ] TODO 3: Add validation plumbing so runs can produce both raw JSON and Croissant-ready outputs with schema-shape checks for key fields.

---

## Croissant Submodule Consideration (Pre-implementation Snapshot)

Historical note: this section records the pre-addition decision pass. Current state is covered in the resumed execution section below.

Question: should `https://github.com/mlcommons/croissant.git` be pulled as a git submodule for local learning/reference?

Findings:
1. Upstream is reachable and active; current resolved HEAD during this check: `828034a45d5c536789c7f6311d4c4a68f7804129`.
2. Repository appears to serve as both spec/examples and implementation (`python/mlcroissant`) with broad integrations.
3. At time of this check, this workspace had no submodules configured (`.gitmodules` absent).

Recommendation:
1. Reasonable as an optional reference submodule if we want offline local spec/examples while implementing TODOs 1–3.
2. Keep it isolated as a non-runtime dependency (for example `external/croissant`), pinned to a commit, and avoid coupling production code imports directly to submodule source.
3. Alternative (lighter): rely on published package/docs (`mlcroissant`) and vendor only minimal schema examples.

Decision status:
- Initially pending explicit go-ahead before adding submodule to this repository.
- Current state: submodule now present at `external/croissant`.

---

## Status

- Step 1: Completed.
- Baseline commit after Step 1: Completed (`d794ad2`).
- Step 2: Completed (code + validation).
- Step 3 (tmp hygiene before push): Completed (`ee31085`).
- Step 4 (Croissant alignment implementation): Completed (code + validation, commit pending).

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

---

## Croissant Alignment Execution (Resumed)

Resume context: Croissant hold was lifted and implementation proceeded with standards-first validation against Croissant 1.1 + Croissant RAI vocabulary and examples.

### TODO resolution

- [x] TODO 1: Added explicit Croissant evidence-format language to canonical paper draft (`docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex`).
- [x] TODO 2: Added exporter path from runtime artifact JSON to Croissant package structure.
- [x] TODO 3: Added schema-shape validation plumbing and optional strict validation mode.

### Code and data changes

1. Added `logomesh/croissant_export.py`:
   - Runtime artifact flattening to tabular records.
   - Croissant metadata generation with Croissant 1.1 + RAI context.
   - Internal CSV/metadata shape checks and optional strict validator hook.
2. Added `scripts/export_kv_mcts_to_croissant.py`:
   - Post-hoc CLI conversion from existing run JSON to Croissant package.
3. Updated `scripts/run_kv_mcts.py`:
   - Optional Croissant export flags for in-run dual output (raw JSON + Croissant package).
   - Explicit export status/error handling.
4. Updated `docs/dataset/croissant_schema_stub.json`:
   - Converted planning stub into executable template shape with concrete `distribution` + `recordSet` mappings.
5. Added example dataset backing files for the schema template:
   - `docs/dataset/data/interventions.csv`
   - `docs/dataset/data/source_run.json`
6. Added test coverage in `tests/test_croissant_export.py`.
7. Updated handoff/docs references (`README.md`, `CLAUDE.md`, `docs/CLAUDE_CONTEXT_BRIEF.md`) for exporter workflow visibility.

### Validation results

1. Targeted exporter tests:
   - Command: `uv run pytest tests/test_croissant_export.py -v`
   - Result: `3 passed`.
2. Full test gate:
   - Command: `uv run pytest tests/ -v`
   - Result: `133 passed in 35.82s`.

### Commit state

1. Croissant implementation batch is currently uncommitted in working tree.
2. Commit hash: pending (will be recorded here once commit is created).
