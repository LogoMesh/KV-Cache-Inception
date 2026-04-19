"""Tests for the batch dataset collector (scripts/collect_dataset.py)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.collect_dataset import (
    apply_labels,
    collect_dataset,
    copy_source_artifacts,
    discover_artifacts,
    filter_new_artifacts,
    flatten_and_deduplicate,
    load_manifest,
)
from logomesh.croissant_export import (
    REQUIRED_COLUMNS,
    build_batch_metadata_document,
    validate_metadata_shape,
)


def _make_artifact(
    tmp_path: Path,
    name: str,
    *,
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    seed: int = 42,
    node_count: int = 2,
) -> Path:
    """Write a minimal valid run artifact JSON and return its path."""
    nodes = []
    for i in range(node_count):
        nodes.append(
            {
                "node_id": f"{name}-node-{i}",
                "parent_id": None if i == 0 else f"{name}-node-{i - 1}",
                "depth": i,
                "alpha": 0.5,
                "layer": 10,
                "visit_count": 1,
                "mean_reward": 0.1 * (i + 1),
                "oei_score": None,
            }
        )

    artifact = {
        "model_id": model_id,
        "config": {
            "n_nodes": node_count,
            "max_depth": 5,
            "branching_factor": 3,
            "alpha_values": [0.1, 0.5, 1.0],
        },
        "system_prompt": "You are a helpful AI assistant.",
        "user_prompt": "Explain alignment faking.",
        "run_metadata": {
            "seed": seed,
            "git_sha": "abc123deadbeef",
            "git_dirty": False,
            "run_started_utc": "2026-04-16T10:00:00Z",
            "run_finished_utc": "2026-04-16T10:00:05Z",
            "command": f"scripts/run_kv_mcts.py --nodes {node_count}",
            "python_version": "3.12.0",
        },
        "all_nodes": nodes,
    }

    path = tmp_path / f"mcts_output_{name}.json"
    path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_discover_artifacts_finds_matching_files(tmp_path: Path):
    """3 valid artifacts + 1 non-matching filename -> discovers 3."""
    _make_artifact(tmp_path, "run1")
    _make_artifact(tmp_path, "run2")
    _make_artifact(tmp_path, "run3")
    # Non-matching filename
    other = tmp_path / "not_an_artifact.json"
    other.write_text(json.dumps({"all_nodes": [{"node_id": "x"}]}), encoding="utf-8")

    found = discover_artifacts(tmp_path, glob_pattern="mcts_output*.json")
    assert len(found) == 3


def test_discover_artifacts_skips_invalid_json(tmp_path: Path):
    """1 valid + 1 missing all_nodes -> discovers 1."""
    _make_artifact(tmp_path, "good")
    bad = tmp_path / "mcts_output_bad.json"
    bad.write_text(json.dumps({"no_nodes_here": True}), encoding="utf-8")

    found = discover_artifacts(tmp_path, glob_pattern="mcts_output*.json")
    assert len(found) == 1


# ---------------------------------------------------------------------------
# Flatten & dedup
# ---------------------------------------------------------------------------


def test_flatten_and_deduplicate_merges_records(tmp_path: Path):
    """2 artifacts x 2 nodes each -> 4 merged records."""
    a1 = _make_artifact(tmp_path, "alpha", node_count=2)
    a2 = _make_artifact(tmp_path, "beta", node_count=2)

    records, pairs = flatten_and_deduplicate([a1, a2])
    assert len(records) == 4
    assert len(pairs) == 2
    # All records should have all required columns
    for record in records:
        for col in REQUIRED_COLUMNS:
            assert col in record


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_apply_labels_joins_correctly(tmp_path: Path):
    """1 matching label + 1 orphaned label -> correct join + warning."""
    a = _make_artifact(tmp_path, "labeled", node_count=1)
    records, _ = flatten_and_deduplicate([a])
    assert len(records) == 1

    aid = records[0]["artifact_id"]

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "artifact_id,textual_compliance,ground_truth_label\n"
        f"{aid},compliant,safe\n"
        "orphan-id-999,noncompliant,unsafe\n",
        encoding="utf-8",
    )

    records = apply_labels(records, labels_csv)
    assert records[0]["textual_compliance"] == "compliant"
    assert records[0]["ground_truth_label"] == "safe"


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def test_collect_dataset_end_to_end(tmp_path: Path):
    """Full pipeline: 2 artifacts in -> merged package out."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_artifact(runs_dir, "run1", node_count=2)
    _make_artifact(runs_dir, "run2", node_count=3)

    output_dir = tmp_path / "dataset"
    result = collect_dataset(input_dir=runs_dir, output_dir=output_dir)

    assert result["status"] == "ok"
    assert result["record_count"] == 5
    assert result["total_source_artifacts"] == 2

    # CSV exists and has correct row count
    csv_path = output_dir / "data" / "interventions.csv"
    assert csv_path.exists()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5

    # Metadata exists
    metadata_path = output_dir / "metadata.json"
    assert metadata_path.exists()

    # Manifest exists with 2 entries
    manifest_path = output_dir / "data" / "sources" / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 2

    # Source files exist
    assert (output_dir / "data" / "sources" / "source_run_001.json").exists()
    assert (output_dir / "data" / "sources" / "source_run_002.json").exists()


# ---------------------------------------------------------------------------
# Batch metadata validation
# ---------------------------------------------------------------------------


def test_build_batch_metadata_document_validates():
    """3 source artifacts -> validate_metadata_shape() clean."""
    sources = [
        {"name": "data/sources/source_run_001.json", "sha256": "a" * 64},
        {"name": "data/sources/source_run_002.json", "sha256": "b" * 64},
        {"name": "data/sources/source_run_003.json", "sha256": "c" * 64},
    ]
    metadata = build_batch_metadata_document(
        csv_file_name="data/interventions.csv",
        csv_sha256="d" * 64,
        source_artifacts=sources,
    )
    errors = validate_metadata_shape(metadata)
    assert errors == []


# ---------------------------------------------------------------------------
# Incremental append
# ---------------------------------------------------------------------------


def test_auto_append_skips_duplicates(tmp_path: Path):
    """Run collect twice on the same artifacts — second run auto-detects and adds nothing."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_artifact(runs_dir, "run1", node_count=2)

    output_dir = tmp_path / "dataset"

    # First run — no dataset exists, full build
    r1 = collect_dataset(input_dir=runs_dir, output_dir=output_dir)
    assert r1["status"] == "ok"
    assert r1["record_count"] == 2

    # Second run — no flags, auto-detects existing dataset, skips duplicates
    r2 = collect_dataset(input_dir=runs_dir, output_dir=output_dir)
    assert r2["status"] == "up_to_date"
    assert r2["new_artifacts"] == 0

    # CSV still has only 2 rows
    csv_path = output_dir / "data" / "interventions.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_auto_append_adds_new_artifact(tmp_path: Path):
    """Build with 1 artifact, then run again with a second — auto-appends."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_artifact(runs_dir, "run1", node_count=2)

    output_dir = tmp_path / "dataset"

    # First run
    r1 = collect_dataset(input_dir=runs_dir, output_dir=output_dir)
    assert r1["record_count"] == 2
    assert r1["total_source_artifacts"] == 1

    # Add a second artifact
    _make_artifact(runs_dir, "run2", node_count=3)

    # Second run — no flags, auto-detects and appends
    r2 = collect_dataset(input_dir=runs_dir, output_dir=output_dir)
    assert r2["status"] == "ok"
    assert r2["record_count"] == 3
    assert r2["total_source_artifacts"] == 2

    # CSV now has 5 rows total
    csv_path = output_dir / "data" / "interventions.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5

    # Manifest has 2 entries
    manifest_path = output_dir / "data" / "sources" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 2
    assert manifest["entries"][0]["seq_name"] == "source_run_001.json"
    assert manifest["entries"][1]["seq_name"] == "source_run_002.json"


def test_full_rebuild_overwrites(tmp_path: Path):
    """--full-rebuild discards existing dataset and rebuilds from scratch.

    Specifically tests that stale source artifact files are removed from the
    sources/ directory, not just ignored by the manifest.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    a1 = _make_artifact(runs_dir, "run1", node_count=2)
    a2 = _make_artifact(runs_dir, "run2", node_count=3)

    output_dir = tmp_path / "dataset"

    # First run — build with both artifacts (source_run_001 + source_run_002)
    r1 = collect_dataset(input_dir=runs_dir, output_dir=output_dir)
    assert r1["record_count"] == 5
    assert r1["total_source_artifacts"] == 2
    assert (output_dir / "data" / "sources" / "source_run_002.json").exists()

    # Remove run2 from the input directory so the rebuild only sees run1
    a2.unlink()

    # Full rebuild — should discard source_run_002.json from sources/
    r2 = collect_dataset(input_dir=runs_dir, output_dir=output_dir, full_rebuild=True)
    assert r2["status"] == "ok"
    assert r2["record_count"] == 2
    assert r2["total_source_artifacts"] == 1

    # CSV has exactly 2 rows (rebuilt from run1 only)
    csv_path = output_dir / "data" / "interventions.csv"
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2

    # Manifest has 1 entry
    manifest_path = output_dir / "data" / "sources" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["entries"]) == 1

    # Sources directory contains only the one rebuilt artifact — no stale files
    sources_dir = output_dir / "data" / "sources"
    source_files = sorted(sources_dir.glob("source_run_*.json"))
    assert len(source_files) == 1
    assert source_files[0].name == "source_run_001.json"


# ---------------------------------------------------------------------------
# Manifest rich metadata
# ---------------------------------------------------------------------------


def test_manifest_rich_metadata(tmp_path: Path):
    """Manifest entries contain enough detail to distinguish test vs real runs."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    _make_artifact(
        runs_dir, "tiny_test",
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        seed=42,
        node_count=2,
    )
    _make_artifact(
        runs_dir, "real_exp",
        model_id="openai/gpt-oss-20b",
        seed=1337,
        node_count=50,
    )

    output_dir = tmp_path / "dataset"
    collect_dataset(input_dir=runs_dir, output_dir=output_dir)

    manifest_path = output_dir / "data" / "sources" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    entries = manifest["entries"]
    assert len(entries) == 2

    # Each entry has run_detail with distinguishing fields
    for entry in entries:
        detail = entry["run_detail"]
        assert "model_id" in detail
        assert "seed" in detail
        assert "node_count" in detail
        assert "run_started_utc" in detail
        assert "git_sha" in detail
        assert "command" in detail
        assert "system_prompt" in detail

    # Can tell the two apart
    models = {e["run_detail"]["model_id"] for e in entries}
    assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in models
    assert "openai/gpt-oss-20b" in models

    seeds = {e["run_detail"]["seed"] for e in entries}
    assert 42 in seeds
    assert 1337 in seeds
