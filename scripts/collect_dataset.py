#!/usr/bin/env python3
"""
Batch dataset collector — merge multiple MCTS run artifacts into one Croissant package.

Scans an input directory for run artifact JSONs produced by run_kv_mcts.py,
flattens them into a single interventions.csv, copies source artifacts with
sequential naming, and generates Croissant 1.1 + RAI 1.0 metadata.

Safe by default: if the output directory already contains a dataset (manifest.json
exists), new artifacts are appended incrementally. A full rebuild from scratch
requires the explicit --full-rebuild flag.

Output package layout:
    <output_dir>/
      metadata.json
      data/
        interventions.csv
        sources/
          source_run_001.json
          source_run_002.json
          ...
          manifest.json

Usage:
    # First run (no existing dataset) — full build:
    uv run python scripts/collect_dataset.py \\
        --input-dir ./runs --output-dir ./dataset

    # Subsequent runs — auto-detects existing dataset, appends new artifacts:
    uv run python scripts/collect_dataset.py \\
        --input-dir ./runs --output-dir ./dataset

    # Explicit full rebuild (destructive — overwrites existing dataset):
    uv run python scripts/collect_dataset.py \\
        --input-dir ./runs --output-dir ./dataset --full-rebuild

    # With optional labels:
    uv run python scripts/collect_dataset.py \\
        --input-dir ./runs --output-dir ./dataset --labels ./labels.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from logomesh.croissant_export import (
    REQUIRED_COLUMNS,
    _sha256_file,
    build_batch_metadata_document,
    build_records_from_run_artifact,
    validate_csv_shape,
    validate_metadata_shape,
    write_metadata_json,
    write_records_csv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("collect_dataset")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover_artifacts(
    input_dir: Path,
    *,
    glob_pattern: str = "mcts_output*.json",
    recursive: bool = True,
) -> list[Path]:
    """Find valid run artifact JSONs in *input_dir*.

    A file is considered valid if it is parseable JSON and contains an
    ``all_nodes`` key with a non-empty list.
    """
    pattern = f"**/{glob_pattern}" if recursive else glob_pattern
    candidates = sorted(input_dir.glob(pattern))

    valid: list[Path] = []
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", path, exc)
            continue

        nodes = data.get("all_nodes")
        if not isinstance(nodes, list) or len(nodes) == 0:
            logger.warning("Skipping %s: missing or empty all_nodes", path)
            continue

        valid.append(path)

    logger.info("Discovered %d valid artifact(s) from %d candidate(s)", len(valid), len(candidates))
    return valid


# ---------------------------------------------------------------------------
# Manifest (checkpoint for incremental append)
# ---------------------------------------------------------------------------


def _sha256_of_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _extract_run_detail(artifact_data: dict[str, Any]) -> dict[str, Any]:
    """Pull rich metadata from a run artifact for manifest storage.

    This is what makes test runs distinguishable from real runs at a glance:
    model_id, seed, node count, timestamps, experiment config, and hardware.
    """
    run_meta = artifact_data.get("run_metadata") or {}
    config = run_meta.get("config") or artifact_data.get("config") or {}
    nodes = artifact_data.get("all_nodes") or []

    return {
        "model_id": artifact_data.get("model_id", "unknown"),
        "seed": run_meta.get("seed"),
        "node_count": len(nodes) if isinstance(nodes, list) else 0,
        "run_started_utc": run_meta.get("run_started_utc"),
        "run_finished_utc": run_meta.get("run_finished_utc"),
        "git_sha": run_meta.get("git_sha"),
        "git_dirty": run_meta.get("git_dirty"),
        "command": run_meta.get("command"),
        "python_version": run_meta.get("python_version"),
        "n_nodes_config": config.get("n_nodes"),
        "max_depth": config.get("max_depth"),
        "branching_factor": config.get("branching_factor"),
        "alpha_values": config.get("alpha_values"),
        "system_prompt": artifact_data.get("system_prompt"),
        "user_prompt": artifact_data.get("user_prompt"),
    }


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load existing manifest.json or return an empty structure."""
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {"entries": []}


def _manifest_processed_hashes(manifest: dict[str, Any]) -> set[str]:
    """Return the set of SHA-256 hashes already in the manifest."""
    return {entry["sha256"] for entry in manifest.get("entries", []) if "sha256" in entry}


def filter_new_artifacts(
    artifacts: list[Path],
    manifest: dict[str, Any],
) -> list[Path]:
    """Return only artifacts whose SHA-256 is not already in the manifest."""
    existing = _manifest_processed_hashes(manifest)
    new = []
    for path in artifacts:
        sha = _sha256_of_file(path)
        if sha in existing:
            logger.info("Skipping already-processed artifact: %s", path.name)
        else:
            new.append(path)
    return new


def _next_seq_number(manifest: dict[str, Any]) -> int:
    """Determine the next sequential source number from the manifest."""
    entries = manifest.get("entries", [])
    if not entries:
        return 1
    max_seq = max(entry.get("seq", 0) for entry in entries)
    return max_seq + 1


# ---------------------------------------------------------------------------
# Flatten & deduplicate
# ---------------------------------------------------------------------------


def flatten_and_deduplicate(
    artifacts: list[Path],
    *,
    experiment_id: str = "run_kv_mcts",
    existing_ids: set[str] | None = None,
) -> tuple[list[dict[str, str]], list[tuple[Path, dict[str, Any]]]]:
    """Flatten all artifacts into records and return (records, artifact_data_pairs).

    Checks for artifact_id collisions against *existing_ids* (from a previous
    CSV in append mode) and across the new batch. Warns on collision but does
    NOT drop rows.

    Returns:
        records: merged list of flattened row dicts
        artifact_data_pairs: list of (path, parsed_json) for downstream use
    """
    if existing_ids is None:
        existing_ids = set()

    all_records: list[dict[str, str]] = []
    artifact_data_pairs: list[tuple[Path, dict[str, Any]]] = []
    seen_ids: set[str] = set(existing_ids)

    for path in artifacts:
        data = json.loads(path.read_text(encoding="utf-8"))
        artifact_data_pairs.append((path, data))

        # Use forward slashes always, per design decision #5
        artifact_rel_path = path.name.replace("\\", "/")

        records = build_records_from_run_artifact(
            data,
            run_artifact_path=artifact_rel_path,
            experiment_id=experiment_id,
        )

        for record in records:
            aid = record.get("artifact_id", "")
            if aid in seen_ids:
                logger.warning(
                    "artifact_id collision: %s (from %s) — keeping row, not dropping",
                    aid,
                    path.name,
                )
            seen_ids.add(aid)

        all_records.extend(records)
        logger.info("Flattened %d records from %s", len(records), path.name)

    return all_records, artifact_data_pairs


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def apply_labels(
    records: list[dict[str, str]],
    labels_path: Path,
) -> list[dict[str, str]]:
    """Join a sidecar label CSV onto records by artifact_id.

    The label CSV must have columns: artifact_id, textual_compliance, ground_truth_label.
    Unmatched labels produce a warning. Unmatched records keep empty label fields.
    """
    label_map: dict[str, dict[str, str]] = {}
    with labels_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            aid = row.get("artifact_id", "").strip()
            if aid:
                label_map[aid] = {
                    "textual_compliance": row.get("textual_compliance", "").strip(),
                    "ground_truth_label": row.get("ground_truth_label", "").strip(),
                }

    matched = 0
    for record in records:
        aid = record.get("artifact_id", "")
        if aid in label_map:
            record["textual_compliance"] = label_map[aid]["textual_compliance"]
            record["ground_truth_label"] = label_map[aid]["ground_truth_label"]
            matched += 1

    orphaned = set(label_map.keys()) - {r.get("artifact_id", "") for r in records}
    if orphaned:
        logger.warning(
            "%d label(s) did not match any record: %s",
            len(orphaned),
            ", ".join(sorted(orphaned)[:5]),
        )

    logger.info("Labels joined: %d matched, %d orphaned", matched, len(orphaned))
    return records


# ---------------------------------------------------------------------------
# Source artifact copying
# ---------------------------------------------------------------------------


def copy_source_artifacts(
    artifact_data_pairs: list[tuple[Path, dict[str, Any]]],
    sources_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Copy artifact JSONs with sequential naming and update the manifest.

    Each manifest entry stores rich run detail so you can tell at a glance
    whether an artifact came from a quick TinyLlama test or a real H100 run.
    """
    sources_dir.mkdir(parents=True, exist_ok=True)
    seq = _next_seq_number(manifest)

    for path, data in artifact_data_pairs:
        seq_name = f"source_run_{seq:03d}.json"
        dest = sources_dir / seq_name
        shutil.copy2(path, dest)

        sha = _sha256_of_file(dest)
        run_detail = _extract_run_detail(data)

        entry = {
            "seq": seq,
            "seq_name": seq_name,
            "original_filename": path.name,
            "sha256": sha,
            "run_detail": run_detail,
        }
        manifest.setdefault("entries", []).append(entry)
        logger.info(
            "Copied %s -> %s (model=%s, nodes=%d, seed=%s)",
            path.name,
            seq_name,
            run_detail.get("model_id", "?"),
            run_detail.get("node_count", 0),
            run_detail.get("seed", "?"),
        )
        seq += 1

    return manifest


# ---------------------------------------------------------------------------
# CSV append
# ---------------------------------------------------------------------------


def append_csv_rows(
    records: list[dict[str, str]],
    csv_path: Path,
) -> None:
    """Append new rows to an existing CSV without rewriting the header."""
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        for record in records:
            writer.writerow({key: record.get(key, "") for key in REQUIRED_COLUMNS})


def _read_existing_artifact_ids(csv_path: Path) -> set[str]:
    """Read all artifact_id values from an existing CSV for collision checking."""
    ids: set[str] = set()
    if not csv_path.exists():
        return ids
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            aid = row.get("artifact_id", "").strip()
            if aid:
                ids.add(aid)
    return ids


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def collect_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    full_rebuild: bool = False,
    glob_pattern: str = "mcts_output*.json",
    recursive: bool = True,
    experiment_id: str = "run_kv_mcts",
    labels_path: Path | None = None,
    dataset_name: str = "KV-Cache Inception Runtime Interventions (Batch)",
    version: str = "0.1.0",
    strict: bool = False,
) -> dict[str, Any]:
    """Discover, flatten, merge, and package run artifacts into a Croissant dataset.

    Safe by default: if *output_dir* already contains a manifest, new artifacts
    are appended incrementally. Pass ``full_rebuild=True`` to discard the
    existing dataset and rebuild from scratch.

    Returns a summary dict with record counts, paths, and validation results.
    """
    data_dir = output_dir / "data"
    sources_dir = data_dir / "sources"
    csv_path = data_dir / "interventions.csv"
    manifest_path = sources_dir / "manifest.json"
    metadata_path = output_dir / "metadata.json"

    # --- Discover ---
    all_artifacts = discover_artifacts(input_dir, glob_pattern=glob_pattern, recursive=recursive)
    if not all_artifacts:
        logger.warning("No valid artifacts found in %s", input_dir)
        return {"record_count": 0, "new_artifacts": 0, "status": "no_artifacts"}

    # --- Auto-detect: append if dataset exists, full build otherwise ---
    dataset_exists = manifest_path.exists() and not full_rebuild
    manifest: dict[str, Any]
    existing_ids: set[str]

    if dataset_exists:
        manifest = load_manifest(manifest_path)
        artifacts_to_process = filter_new_artifacts(all_artifacts, manifest)
        existing_ids = _read_existing_artifact_ids(csv_path)
        if not artifacts_to_process:
            logger.info("No new artifacts to process — dataset is up to date")
            return {"record_count": 0, "new_artifacts": 0, "status": "up_to_date"}
        logger.info("Existing dataset detected — appending %d new artifact(s)", len(artifacts_to_process))
    else:
        # Full build — start clean
        manifest = {"entries": []}
        artifacts_to_process = all_artifacts
        existing_ids = set()
        if full_rebuild and manifest_path.exists():
            logger.info("Full rebuild requested — discarding existing dataset")

    # --- Flatten ---
    records, artifact_data_pairs = flatten_and_deduplicate(
        artifacts_to_process,
        experiment_id=experiment_id,
        existing_ids=existing_ids,
    )

    if not records:
        logger.warning("Flattening produced no records")
        return {"record_count": 0, "new_artifacts": len(artifacts_to_process), "status": "no_records"}

    # --- Labels (optional) ---
    if labels_path is not None:
        records = apply_labels(records, labels_path)

    # --- Write / append CSV ---
    if dataset_exists and csv_path.exists():
        append_csv_rows(records, csv_path)
        logger.info("Appended %d rows to %s", len(records), csv_path)
    else:
        data_dir.mkdir(parents=True, exist_ok=True)
        write_records_csv(records, csv_path)
        logger.info("Wrote %d rows to %s", len(records), csv_path)

    # --- Validate CSV ---
    csv_errors = validate_csv_shape(csv_path)
    if csv_errors:
        raise ValueError("CSV validation failed: " + " | ".join(csv_errors))

    # --- Copy source artifacts ---
    manifest = copy_source_artifacts(artifact_data_pairs, sources_dir, manifest)

    # --- Write manifest ---
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Manifest updated: %d total entries", len(manifest.get("entries", [])))

    # --- Build metadata ---
    csv_sha = _sha256_file(csv_path)
    source_artifacts_for_metadata = [
        {
            "name": f"data/sources/{entry['seq_name']}",
            "sha256": entry["sha256"],
        }
        for entry in manifest["entries"]
    ]

    metadata = build_batch_metadata_document(
        csv_file_name="data/interventions.csv",
        csv_sha256=csv_sha,
        source_artifacts=source_artifacts_for_metadata,
        dataset_name=dataset_name,
        version=version,
    )

    # --- Validate metadata ---
    metadata_errors = validate_metadata_shape(metadata)
    if metadata_errors:
        raise ValueError("Metadata validation failed: " + " | ".join(metadata_errors))

    # --- Write metadata ---
    write_metadata_json(metadata, metadata_path)
    logger.info("Metadata written to %s", metadata_path)

    # --- Optional strict validation ---
    strict_rc = None
    strict_output = ""
    if strict:
        from logomesh.croissant_export import validate_with_mlcroissant

        strict_rc, strict_output = validate_with_mlcroissant(metadata_path)
        if strict_rc != 0:
            raise RuntimeError(f"Strict mlcroissant validation failed (exit={strict_rc}): {strict_output}")

    total_entries = len(manifest.get("entries", []))
    return {
        "record_count": len(records),
        "new_artifacts": len(artifacts_to_process),
        "total_source_artifacts": total_entries,
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path),
        "manifest_path": str(manifest_path),
        "strict_validation_return_code": strict_rc,
        "strict_validation_output": strict_output,
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True, type=Path, help="Directory containing run artifact JSONs")
    p.add_argument("--output-dir", required=True, type=Path, help="Output directory for the Croissant package")
    p.add_argument("--full-rebuild", action="store_true", help="Discard existing dataset and rebuild from scratch (default: auto-append if dataset exists)")
    p.add_argument("--glob", default="mcts_output*.json", help="Glob pattern for artifact files (default: mcts_output*.json)")
    p.add_argument("--no-recursive", action="store_true", help="Don't search subdirectories")
    p.add_argument("--experiment-id", default="run_kv_mcts", help="Experiment identifier for records")
    p.add_argument("--labels", type=Path, default=None, help="Sidecar CSV with artifact_id, textual_compliance, ground_truth_label")
    p.add_argument("--dataset-name", default="KV-Cache Inception Runtime Interventions (Batch)", help="Dataset name in metadata")
    p.add_argument("--version", default="0.1.0", help="Dataset version")
    p.add_argument("--strict", action="store_true", help="Run strict mlcroissant CLI validation")
    args = p.parse_args()

    result = collect_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        full_rebuild=args.full_rebuild,
        glob_pattern=args.glob,
        recursive=not args.no_recursive,
        experiment_id=args.experiment_id,
        labels_path=args.labels,
        dataset_name=args.dataset_name,
        version=args.version,
        strict=args.strict,
    )

    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result.get("status") in ("ok", "up_to_date", "no_artifacts") else 1


if __name__ == "__main__":
    raise SystemExit(main())
