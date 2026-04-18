"""Tests for Croissant runtime export utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from logomesh.croissant_export import (
    REQUIRED_COLUMNS,
    build_metadata_document,
    export_run_artifact_to_croissant,
    validate_csv_shape,
    validate_metadata_shape,
)


def _sample_run_artifact() -> dict:
    return {
        "model_id": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "run_metadata": {
            "seed": 1337,
            "git_sha": "deadbeefcafebabe",
            "git_dirty": False,
            "run_started_utc": "2026-04-15T00:00:00Z",
            "run_finished_utc": "2026-04-15T00:00:05Z",
            "command": "scripts/run_kv_mcts.py --nodes 1",
        },
        "all_nodes": [
            {
                "node_id": "root-node",
                "parent_id": None,
                "depth": 0,
                "alpha": 0.0,
                "layer": -1,
                "visit_count": 1,
                "mean_reward": 0.1,
                "oei_score": None,
            },
            {
                "node_id": "child-node",
                "parent_id": "root-node",
                "depth": 1,
                "alpha": 0.5,
                "layer": 10,
                "visit_count": 1,
                "mean_reward": 0.42,
                "oei_score": 1.12,
                "telemetry": {
                    "sigma_H_mean": 0.22,
                    "rho_R_mean": 0.85,
                    "tds": 0.05,
                    "h_neuron": [0.2, 0.25, 0.21],
                    "repe_honesty": [0.84, 0.86, 0.85],
                },
            },
        ],
    }


def test_export_run_artifact_to_croissant_creates_package(tmp_path: Path):
    input_artifact = tmp_path / "mcts_output.json"
    input_artifact.write_text(json.dumps(_sample_run_artifact(), indent=2), encoding="utf-8")

    output_dir = tmp_path / "croissant"
    info = export_run_artifact_to_croissant(run_artifact_path=input_artifact, output_dir=output_dir, strict=False)

    assert info["record_count"] == 2

    csv_path = output_dir / "interventions.csv"
    metadata_path = output_dir / "metadata.json"
    source_run_path = output_dir / "source_run.json"

    assert csv_path.exists()
    assert metadata_path.exists()
    assert source_run_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["datePublished"]
    assert metadata["inLanguage"] == "en"
    assert isinstance(metadata.get("keywords"), list)
    assert metadata.get("publisher", {}).get("name") == "LogoMesh Research Team"
    assert metadata.get("recordSet", [{}])[0].get("key") == [{"@id": "interventions/artifact_id"}]

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or []) == REQUIRED_COLUMNS
        rows = list(reader)

    assert len(rows) == 2

    child_rows = [row for row in rows if row.get("node_id") == "child-node"]
    assert len(child_rows) == 1
    child = child_rows[0]
    assert child["diagnostic_state"] in {
        "creative_synthesis",
        "contextual_fabrication",
        "alignment_faking",
        "orthogonal_escape",
    }
    assert child["audit_hash"]

    # t_matrix_json must be a 2×L matrix matching the two per-layer vectors
    t_matrix = json.loads(child["t_matrix_json"])
    assert isinstance(t_matrix, list) and len(t_matrix) == 2
    assert len(t_matrix[0]) == len(t_matrix[1])  # both rows same length (L layers)
    assert t_matrix[0] == json.loads(child["h_neuron_json"])
    assert t_matrix[1] == json.loads(child["repe_honesty_json"])


def test_validate_csv_shape_reports_missing_columns(tmp_path: Path):
    broken_csv = tmp_path / "broken.csv"
    broken_csv.write_text("only_one_column\nvalue\n", encoding="utf-8")

    errors = validate_csv_shape(broken_csv)
    assert errors
    assert "missing required columns" in errors[0]


def test_validate_metadata_shape_reports_missing_required_key():
    metadata = build_metadata_document(
        csv_file_name="interventions.csv",
        csv_sha256="abc123",
        source_artifact_name="source_run.json",
        source_artifact_sha256="def456",
    )

    metadata.pop("recordSet")
    errors = validate_metadata_shape(metadata)
    assert errors
    assert any("recordSet" in error for error in errors)


def test_validate_metadata_shape_accepts_dct_conforms_to_alias():
    metadata = build_metadata_document(
        csv_file_name="interventions.csv",
        csv_sha256="a" * 64,
        source_artifact_name="source_run.json",
        source_artifact_sha256="b" * 64,
    )
    metadata["dct:conformsTo"] = metadata.pop("conformsTo")

    errors = validate_metadata_shape(metadata)

    assert errors == []


def test_validate_metadata_shape_reports_invalid_distribution_sha256():
    metadata = build_metadata_document(
        csv_file_name="interventions.csv",
        csv_sha256="a" * 64,
        source_artifact_name="source_run.json",
        source_artifact_sha256="b" * 64,
    )
    metadata["distribution"][0]["sha256"] = "not-a-valid-sha"

    errors = validate_metadata_shape(metadata)

    assert errors
    assert any("invalid sha256 format" in error for error in errors)
