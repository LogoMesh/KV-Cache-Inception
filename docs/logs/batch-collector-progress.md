# Batch Dataset Collector — Progress Summary

**Date:** 2026-04-16 (updated 2026-04-18)
**Owner:** Max (Data Pipeline — LOG-40, LOG-46)
**Branch:** `feature/batch-collector-incremental`
**Status:** COMPLETE — all pipeline code implemented, tested (9/9 unit tests, 144/144 full suite), and verified end-to-end on local hardware. No further pipeline code changes required unless Phase 3 experiment output format changes.

---

## What This Solves

The Croissant export pipeline (`logomesh/croissant_export.py`) already converts **individual** MCTS run artifacts into self-contained Croissant packages. But each run produces an isolated 3-file package. There is nothing that merges multiple experiment runs into one unified dataset for NeurIPS submission.

The batch collector script (`scripts/collect_dataset.py`) is the missing piece:

```
N run artifact JSONs  -->  scripts/collect_dataset.py  -->  1 merged Croissant dataset
```

With **incremental append mode**, each experiment run automatically folds its artifact into the growing dataset via `--auto-collect` on `run_kv_mcts.py`. No manual merge step — the submission-ready package stays up to date after every run. SHA-256 checksums prevent duplicates, and a full rebuild from raw artifacts is always available as a fallback.

---

## Output Package Layout

```
<output_dir>/
  metadata.json                 # Croissant 1.1 + RAI 1.0 JSON-LD (whole dataset)
  data/
    interventions.csv           # All MCTS nodes from all runs, merged
    sources/
      source_run_001.json       # Copied run artifacts (sequential naming)
      source_run_002.json
      ...
      manifest.json             # Maps seq names -> original filenames + SHA-256 + rich run detail
```

---

## Completed Work

### 1. `build_batch_metadata_document()` added to `logomesh/croissant_export.py` (2026-04-16)

New public function (additive only — no existing code changed). Generates Croissant 1.1 + RAI 1.0 metadata for a batch of merged run artifacts.

**Signature:**

```python
def build_batch_metadata_document(
    *,
    csv_file_name: str,
    csv_sha256: str,
    source_artifacts: list[dict[str, str]],
    dataset_name: str = "KV-Cache Inception Runtime Interventions (Batch)",
    dataset_description: str = "...",
    version: str = "0.1.0",
    date_published: str | None = None,
) -> dict[str, Any]:
```

**How it works:**
- Calls existing `build_metadata_document()` to get a valid skeleton (context, fields, RAI annotations, recordSet)
- Replaces `distribution` list with: the CSV FileObject entry + one `cr:FileObject` per source artifact (`@id`: `source-run-001`, `source-run-002`, etc.)
- Field mappings all reference `interventions-csv`, so multi-source distribution entries are transparent
- Validated: `validate_metadata_shape()` handles multi-entry distribution correctly

### 2. `scripts/collect_dataset.py` — CLI batch collector (2026-04-17)

Full CLI batch collector with incremental append mode. All functions implemented and tested.

**Internal functions:**

| Function | Purpose |
|---|---|
| `discover_artifacts()` | Walk directory, find valid run artifact JSONs, skip invalid |
| `flatten_and_deduplicate()` | Call `build_records_from_run_artifact()` per artifact, merge, check ID collisions |
| `apply_labels()` | Join sidecar CSV (artifact_id, textual_compliance, ground_truth_label) onto records |
| `copy_source_artifacts()` | Copy JSONs with sequential naming, write manifest.json with rich run detail |
| `collect_dataset()` | Main orchestration: discover -> flatten -> label -> write CSV -> copy sources -> validate -> metadata -> validate -> write |
| `load_manifest()` | Read existing manifest.json, return set of already-processed SHA-256 hashes |
| `filter_new_artifacts()` | Compare discovered artifacts against manifest hashes, return only unprocessed ones |
| `append_csv_rows()` | Open existing interventions.csv in append mode, write new rows with stable column order |

**Reuses from `logomesh/croissant_export.py`:**
- `build_records_from_run_artifact()` — per-artifact flattening (28 columns)
- `write_records_csv()` — CSV with stable column order
- `write_metadata_json()` — JSON-LD output
- `validate_csv_shape()` / `validate_metadata_shape()` — structural validation
- `validate_with_mlcroissant()` — optional strict validation
- `_sha256_file()` — integrity checksums
- `REQUIRED_COLUMNS` — column constants

**CLI:**
```bash
uv run python scripts/collect_dataset.py \
    --input-dir ./runs \
    --output-dir ./dataset \
    [--glob "mcts_output*.json"] \
    [--no-recursive] \
    [--experiment-id run_kv_mcts] \
    [--labels ./labels.csv] \
    [--dataset-name "..."] \
    [--version 0.1.0] \
    [--strict] \
    [--full-rebuild]
```

**Auto-detection (safe by default):**
- If no dataset exists → full build from scratch
- If a dataset already exists (manifest.json found) → automatically appends only new artifacts (SHA-256 dedup)
- If `--full-rebuild` is passed → discards existing dataset and rebuilds from scratch

No `--append` flag needed — the script does the right thing automatically.

### 3. `--auto-collect` flag on `scripts/run_kv_mcts.py` (2026-04-17)

After each MCTS run writes its artifact JSON, if `--auto-collect` is passed, it calls `collect_dataset()` automatically. `--dataset-dir` controls where the package lands (default `./dataset`). For this project, use `--dataset-dir ./docs/dataset` to keep everything in one place.

```bash
uv run python scripts/run_kv_mcts.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --nodes 10 --depth 3 --branches 2 \
    --auto-collect --dataset-dir ./docs/dataset
```

### 4. `tests/test_collect_dataset.py` — 10 unit tests (2026-04-17, updated 2026-04-18)

All 10 tests passing. 145/145 tests green across the full suite.

| Test | What it verifies | Status |
|---|---|---|
| `test_discover_artifacts_finds_matching_files` | 3 valid + 1 non-matching -> finds 3 | PASS |
| `test_discover_artifacts_skips_invalid_json` | 1 valid + 1 missing all_nodes -> finds 1 | PASS |
| `test_flatten_and_deduplicate_merges_records` | 2 artifacts x 2 nodes -> 4 merged records | PASS |
| `test_apply_labels_joins_correctly` | 1 matching + 1 orphaned label -> correct join + warning | PASS |
| `test_collect_dataset_end_to_end` | Full pipeline: 2 artifacts in -> merged package out | PASS |
| `test_build_batch_metadata_document_validates` | 3 source artifacts -> validate_metadata_shape() clean | PASS |
| `test_auto_append_skips_duplicates` | Same artifacts twice, auto-detects and adds nothing | PASS |
| `test_auto_append_adds_new_artifact` | Build 1, then run again with 2nd artifact -> auto-appends | PASS |
| `test_full_rebuild_overwrites` | `--full-rebuild` discards existing dataset and rebuilds clean | PASS |
| `test_manifest_rich_metadata` | TinyLlama vs gpt-oss-20b entries distinguishable by model, seed, node count | PASS |

### 5. Rich manifest metadata for run identification (2026-04-17)

Each `manifest.json` entry stores detailed run metadata so test runs are immediately distinguishable from real experiment runs:

```json
{
  "seq": 1,
  "seq_name": "source_run_001.json",
  "original_filename": "mcts_output_1776545805.json",
  "sha256": "72b9af55...",
  "run_detail": {
    "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "seed": 42,
    "node_count": 11,
    "run_started_utc": "2026-04-18T20:56:54.099720Z",
    "run_finished_utc": "2026-04-18T21:02:44.835619Z",
    "git_sha": "5f9427bdd712124dbd7a8d7fdca257a835a8a962",
    "git_dirty": true,
    "command": "scripts/run_kv_mcts.py --nodes 10 --depth 3 --branches 2",
    "python_version": "3.13.13",
    "n_nodes_config": 10,
    "max_depth": 3,
    "branching_factor": 2,
    "alpha_values": [0.1, 0.5, 1.0, 2.0, 5.0],
    "system_prompt": "You are a helpful AI assistant.",
    "user_prompt": "Explain the concept of alignment faking in AI systems."
  }
}
```

At a glance you can tell: model (TinyLlama test vs gpt-oss-20b real), scale (10 nodes vs 500), seed, exact command, and timestamp.

---

## End-to-End Verification (2026-04-18)

Verified on local machine (RTX 3060, TinyLlama, Python 3.13).

### Test 1: Manual collector run

```
Run:     uv run --python 3.13 python scripts/run_kv_mcts.py --nodes 10 --depth 3 --branches 2
Result:  mcts_output_1776545805.json produced (11 nodes, seed=42)

Collect: uv run --python 3.13 python scripts/collect_dataset.py --input-dir . --output-dir ./dataset-test
Result:  11 rows in interventions.csv, 1 source in manifest, metadata.json valid
```

### Test 2: Auto-collect with dedup

```
Run:     uv run --python 3.13 python scripts/run_kv_mcts.py --nodes 10 --depth 3 --auto-collect --dataset-dir ./dataset-test
Result:  New artifact produced. Auto-collect discovered 3 artifacts total:
         - Skipped mcts_output_1776545805.json (already in manifest)
         - Flattened 11 records from mcts_output_1776546841.json (seed=99, earlier run)
         - Flattened 11 records from mcts_output_1776547266.json (seed=42, this run)
         - Appended 22 rows to interventions.csv (33 total)
         - Manifest updated: 3 total entries (source_run_001 through source_run_003)
         - Metadata regenerated
```

**Confirmed behaviors:**
- Auto-collect triggers automatically after experiment completes
- Already-processed artifacts are skipped (SHA-256 dedup works)
- New artifacts are appended without rewriting existing data
- Manifest grows incrementally with rich run detail per entry
- Metadata regenerated with correct source list and CSV hash
- All runs land in one unified Croissant dataset

---

## Pipeline Status: COMPLETE

All data processing pipeline code is implemented, tested, and verified. No further changes needed unless Phase 3 experiment output format changes.

| Item | Status |
|---|---|
| Batch collector pipeline (`scripts/collect_dataset.py`) | Done |
| Batch metadata generation (`build_batch_metadata_document()`) | Done |
| Auto-collect integration (`--auto-collect` on `run_kv_mcts.py`) | Done |
| Auto-detect append vs full build (safe by default) | Done |
| Rich manifest metadata for run identification | Done |
| Label join (`--labels` flag) | Done (code ready, awaiting annotation data) |
| Unit tests (10/10) | Done — 145/145 full suite green |
| Local e2e verification | Done — TinyLlama on RTX 3060 |

---

## Downstream Tasks (Phase 4 — not pipeline code)

These items depend on the pipeline but are separate workstreams. None require changes to `collect_dataset.py` or `croissant_export.py`.

| Item | Status | Blocked On | Notes |
|---|---|---|---|
| Phase 3 real experiments | Not started | H100 access (Tianyu's lab) | Pipeline will ingest results via `--auto-collect` |
| Ground truth annotation CSV | Not started | Phase 3 experiment data | `--labels` join is ready, just needs the CSV |
| Croissant schema finalization | Stub | Exp 1 output format confirmation | `docs/dataset/croissant_schema_stub.json` |
| `mlcroissant` strict validation | Not tested | `pip install mlcroissant` | `--strict` flag is wired up, one-time verification |
| HuggingFace upload script | Not started | Final dataset package | Package layout is already HF-compatible |
| Dataset card (README) | Not started | Final dataset contents | RAI annotations in metadata cover partial ground |

---

## Key Design Decisions

1. **Post-process metadata, don't modify `build_metadata_document()`** — the existing function has a clean single-source contract and is tested. The batch function wraps it and replaces the distribution list.

2. **Sequential source artifact naming** (`source_run_001.json`, etc.) with `manifest.json` for traceability back to original filenames.

3. **Label join is integrated but optional** (`--labels` flag) — avoids a separate step someone could forget.

4. **ID collisions warn, don't drop rows** — `artifact_id` collisions are near-impossible (node_ids are UUIDs), and silently dropping research dataset rows would be worse than a warning.

5. **Forward slashes always** in `run_artifact_path` column values, regardless of OS.

6. **Safe by default** — the collector auto-detects whether to build or append based on whether a dataset already exists. No `--append` flag needed. `--full-rebuild` is the only destructive option and must be explicitly passed.

7. **Auto-collect is opt-in** — `run_kv_mcts.py` only triggers the collector when `--auto-collect` is passed. Default behavior is unchanged.

---

## Command Reference

**Run experiment only (no dataset collection):**
```bash
uv run --python 3.13 python scripts/run_kv_mcts.py --nodes 10 --depth 3 --branches 2
```
Produces a JSON artifact file. Nothing else.

**Run experiment + auto-collect into dataset:**
```bash
uv run --python 3.13 python scripts/run_kv_mcts.py --nodes 10 --depth 3 --auto-collect --dataset-dir ./docs/dataset
```
Runs the experiment, then automatically adds results into `docs/dataset/`. Creates the dataset if it doesn't exist, appends if it does.

**Collect manually (auto-detects build vs append):**
```bash
uv run --python 3.13 python scripts/collect_dataset.py --input-dir ./runs --output-dir ./docs/dataset
```
Scans input directory for artifacts. If dataset already exists, appends only new ones. If not, builds from scratch.

**Force a full rebuild (wipe and start over):**
```bash
uv run --python 3.13 python scripts/collect_dataset.py --input-dir ./runs --output-dir ./docs/dataset --full-rebuild
```
Ignores existing dataset and rebuilds everything from raw artifacts.

**Collect with labels:**
```bash
uv run --python 3.13 python scripts/collect_dataset.py --input-dir ./runs --output-dir ./docs/dataset --labels ./labels.csv
```
Same as manual collect, but joins ground truth labels onto the records.

> **Note:** `docs/dataset/` previously contained a single-run Croissant stub (`croissant_schema_stub.json`, `source_run.json`). The batch collector replaces that with the multi-run package. The old stub files should be removed before the first real collection run to avoid confusion.

---

## Testing Plan (Reference)

#### Level 1: Unit tests (no model, no GPU — instant)

```bash
uv run pytest tests/test_collect_dataset.py -v     # new tests pass
uv run pytest tests/test_croissant_export.py -v   # existing tests still pass
uv run pytest tests/ -v                             # full suite green
```

#### Level 2: Dry run with existing artifacts (no model needed)

```bash
uv run python scripts/collect_dataset.py \
    --input-dir ./runs --output-dir ./dataset-test
```

#### Level 3: End-to-end with auto-collect on TinyLlama (RTX 3060)

```bash
uv run python scripts/run_kv_mcts.py --nodes 10 --depth 3 --auto-collect --dataset-dir ./dataset-e2e
```

Run multiple times, verify CSV grows, manifest grows, duplicates skipped, metadata regenerated.
