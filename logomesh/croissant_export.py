"""
Croissant export utilities for KV-MCTS runtime artifacts.

This module converts `scripts/run_kv_mcts.py` JSON outputs into a self-contained
Croissant package:

- `interventions.csv` with one row per MCTS node
- `source_run.json` as the raw runtime artifact snapshot
- `metadata.json` Croissant 1.1 + RAI 1.0 JSON-LD metadata

It also provides lightweight schema-shape checks and optional strict validation
through the `mlcroissant` CLI when available.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .telemetry_matrix import TelemetryMatrix, classify

CROISSANT_SPEC_URI = "http://mlcommons.org/croissant/1.1"
RAI_SPEC_URI = "http://mlcommons.org/croissant/RAI/1.0"

CROISSANT_CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "containedIn": "cr:containedIn",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}


@dataclass(frozen=True)
class CroissantFieldSpec:
    """Column-level metadata for Croissant field generation."""

    name: str
    data_type: str
    description: str


FIELD_SPECS: tuple[CroissantFieldSpec, ...] = (
    CroissantFieldSpec("artifact_id", "sc:Text", "Stable intervention record identifier."),
    CroissantFieldSpec("run_artifact_path", "sc:Text", "Relative path to the raw run artifact JSON snapshot."),
    CroissantFieldSpec("model_id", "sc:Text", "Model identifier used for generation."),
    CroissantFieldSpec("seed", "sc:Integer", "Global random seed for the run."),
    CroissantFieldSpec("git_sha", "sc:Text", "Git commit SHA at run time."),
    CroissantFieldSpec("git_dirty", "sc:Text", "Whether the git worktree was dirty at run time."),
    CroissantFieldSpec("run_started_utc", "sc:DateTime", "UTC timestamp when run started."),
    CroissantFieldSpec("run_finished_utc", "sc:DateTime", "UTC timestamp when run finished."),
    CroissantFieldSpec("command", "sc:Text", "Command line used to produce the run artifact."),
    CroissantFieldSpec("node_id", "sc:Text", "MCTS node identifier."),
    CroissantFieldSpec("parent_id", "sc:Text", "Parent node identifier, if present."),
    CroissantFieldSpec("depth", "sc:Integer", "Tree depth for this MCTS node."),
    CroissantFieldSpec("alpha", "sc:Float", "Steering magnitude applied at this node."),
    CroissantFieldSpec("intervention_layer", "sc:Integer", "Layer index steered for this node."),
    CroissantFieldSpec("visit_count", "sc:Integer", "Number of visits for this node."),
    CroissantFieldSpec("mean_reward", "sc:Float", "Mean reward for this node."),
    CroissantFieldSpec("oei_score", "sc:Float", "Orthogonal Escape Index (OEI) for this node, if available."),
    CroissantFieldSpec("sigma_h_mean", "sc:Float", "Mean H-neuron stress across layers."),
    CroissantFieldSpec("rho_r_mean", "sc:Float", "Mean RepE honesty projection across layers."),
    CroissantFieldSpec("tds_score", "sc:Float", "Telemetry divergence score (JSD between channels)."),
    CroissantFieldSpec("h_neuron_json", "sc:Text", "JSON-serialized per-layer H-neuron vector."),
    CroissantFieldSpec("repe_honesty_json", "sc:Text", "JSON-serialized per-layer RepE honesty vector."),
    CroissantFieldSpec("diagnostic_state", "sc:Text", "Diagnostic state classification from telemetry_matrix.classify()."),
    CroissantFieldSpec("textual_compliance", "sc:Text", "Reserved field for textual compliance label (optional in current phase)."),
    CroissantFieldSpec("ground_truth_label", "sc:Text", "Reserved field for independent ground-truth label (optional in current phase)."),
    CroissantFieldSpec("experiment_id", "sc:Text", "Experiment identifier (for example run_kv_mcts)."),
    CroissantFieldSpec("hardware_context", "sc:Text", "Runtime hardware or platform context string."),
    CroissantFieldSpec("audit_hash", "sc:Text", "SHA-256 audit hash derived from key node attributes."),
)

REQUIRED_COLUMNS: tuple[str, ...] = tuple(field.name for field in FIELD_SPECS)


def _is_hex_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _json_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=True)


def _coerce_scalar(value: Any) -> str:
    """Convert a value to a CSV-safe scalar string, preserving blanks for nulls."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return _json_compact(value)
    return str(value)


def _diagnostic_state_from_telemetry(telemetry: dict[str, Any], depth: int) -> str:
    h_neuron = telemetry.get("h_neuron")
    repe_honesty = telemetry.get("repe_honesty")
    if not isinstance(h_neuron, list) or not isinstance(repe_honesty, list):
        return "unknown"
    if len(h_neuron) == 0 or len(h_neuron) != len(repe_honesty):
        return "unknown"

    try:
        matrix = TelemetryMatrix(
            h_neuron=np.asarray(h_neuron, dtype=np.float32),
            repe_honesty=np.asarray(repe_honesty, dtype=np.float32),
            step=depth,
        )
    except Exception:
        return "unknown"

    try:
        return classify(matrix).value
    except Exception:
        return "unknown"


def _build_audit_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_records_from_run_artifact(
    run_artifact: dict[str, Any],
    *,
    run_artifact_path: str,
    experiment_id: str = "run_kv_mcts",
) -> list[dict[str, str]]:
    """Flatten one run artifact into tabular records (one row per node)."""
    run_metadata = run_artifact.get("run_metadata") or {}
    nodes = run_artifact.get("all_nodes") or []
    if not isinstance(nodes, list):
        return []

    model_id = run_artifact.get("model_id")
    hardware_context = platform.platform()

    records: list[dict[str, str]] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue

        telemetry = node.get("telemetry") if isinstance(node.get("telemetry"), dict) else {}
        depth = int(node.get("depth") or 0)
        node_id = node.get("node_id")
        git_sha = run_metadata.get("git_sha")

        h_neuron = telemetry.get("h_neuron") if isinstance(telemetry.get("h_neuron"), list) else []
        repe_honesty = telemetry.get("repe_honesty") if isinstance(telemetry.get("repe_honesty"), list) else []
        diagnostic_state = _diagnostic_state_from_telemetry(telemetry, depth)

        if isinstance(git_sha, str) and git_sha:
            artifact_prefix = git_sha[:8]
        else:
            artifact_prefix = "nogit"

        if isinstance(node_id, str) and node_id:
            artifact_suffix = node_id.split("-")[0]
        else:
            artifact_suffix = f"n{index:05d}"

        artifact_id = f"{artifact_prefix}-{artifact_suffix}"

        hash_payload = {
            "artifact_id": artifact_id,
            "model_id": model_id,
            "node_id": node_id,
            "alpha": node.get("alpha"),
            "intervention_layer": node.get("layer"),
            "oei_score": node.get("oei_score"),
            "mean_reward": node.get("mean_reward"),
            "tds_score": telemetry.get("tds"),
        }
        audit_hash = _build_audit_hash(hash_payload)

        record = {
            "artifact_id": artifact_id,
            "run_artifact_path": run_artifact_path,
            "model_id": model_id,
            "seed": run_metadata.get("seed"),
            "git_sha": git_sha,
            "git_dirty": run_metadata.get("git_dirty"),
            "run_started_utc": run_metadata.get("run_started_utc"),
            "run_finished_utc": run_metadata.get("run_finished_utc"),
            "command": run_metadata.get("command"),
            "node_id": node_id,
            "parent_id": node.get("parent_id"),
            "depth": node.get("depth"),
            "alpha": node.get("alpha"),
            "intervention_layer": node.get("layer"),
            "visit_count": node.get("visit_count"),
            "mean_reward": node.get("mean_reward"),
            "oei_score": node.get("oei_score"),
            "sigma_h_mean": telemetry.get("sigma_H_mean"),
            "rho_r_mean": telemetry.get("rho_R_mean"),
            "tds_score": telemetry.get("tds"),
            "h_neuron_json": h_neuron,
            "repe_honesty_json": repe_honesty,
            "diagnostic_state": diagnostic_state,
            "textual_compliance": "",
            "ground_truth_label": "",
            "experiment_id": experiment_id,
            "hardware_context": hardware_context,
            "audit_hash": audit_hash,
        }
        records.append({key: _coerce_scalar(value) for key, value in record.items()})

    return records


def write_records_csv(records: list[dict[str, str]], output_csv_path: Path) -> None:
    """Persist flattened records as CSV with stable field order."""
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in REQUIRED_COLUMNS})


def build_metadata_document(
    *,
    csv_file_name: str,
    csv_sha256: str,
    source_artifact_name: str,
    source_artifact_sha256: str,
    dataset_name: str = "KV-Cache Inception Runtime Interventions",
    dataset_description: str = (
        "Runtime evidence export from Reversible MCTS in KV-cache latent space. "
        "Each record corresponds to one MCTS node with telemetry channels, "
        "diagnostic classification, OEI/TDS metrics, and reproducibility provenance."
    ),
    version: str = "0.1.0",
    date_published: str | None = None,
) -> dict[str, Any]:
    """Create Croissant 1.1 + RAI 1.0 metadata JSON-LD document."""
    now_iso = _utc_now_iso()
    published_iso = date_published or now_iso
    creator = {
        "@type": "sc:Organization",
        "name": "LogoMesh Research Team",
    }

    fields = []
    for spec in FIELD_SPECS:
        fields.append(
            {
                "@type": "cr:Field",
                "@id": f"interventions/{spec.name}",
                "name": spec.name,
                "description": spec.description,
                "dataType": spec.data_type,
                "source": {
                    "fileObject": {"@id": "interventions-csv"},
                    "extract": {"column": spec.name},
                },
            }
        )

    return {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "name": dataset_name,
        "description": dataset_description,
        "conformsTo": [CROISSANT_SPEC_URI, RAI_SPEC_URI],
        "license": "https://opensource.org/licenses/MIT",
        "url": "https://github.com/Logomesh/kv-cache-inception",
        "version": version,
        "sdVersion": "1.1",
        "datePublished": published_iso,
        "dateCreated": now_iso,
        "dateModified": now_iso,
        "inLanguage": "en",
        "keywords": [
            "alignment-faking detection",
            "kv-cache intervention",
            "reversible mcts",
            "latent telemetry",
        ],
        "creator": creator,
        "publisher": creator,
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "interventions-csv",
                "name": csv_file_name,
                "contentUrl": csv_file_name,
                "encodingFormat": "text/csv",
                "sha256": csv_sha256,
            },
            {
                "@type": "cr:FileObject",
                "@id": "source-run-json",
                "name": source_artifact_name,
                "contentUrl": source_artifact_name,
                "encodingFormat": "application/json",
                "sha256": source_artifact_sha256,
            },
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "interventions",
                "name": "interventions",
                "description": (
                    "One record per MCTS node exported from run_kv_mcts runtime "
                    "artifacts, including telemetry, reward, OEI/TDS, and provenance."
                ),
                "key": [{"@id": "interventions/artifact_id"}],
                "field": fields,
            }
        ],
        "rai:dataUseCases": (
            "Safety diagnostics and stress testing of alignment-faking behavior in "
            "language model latent representations."
        ),
        "rai:dataLimitations": (
            "Current export reflects model-internal telemetry and search traces only; "
            "independent human ground-truth labels are reserved fields for later phases."
        ),
        "rai:dataBiases": (
            "Signal quality depends on probe calibration prompts and model family; "
            "cross-family comparability requires additional calibration and transfer studies."
        ),
        "rai:personalSensitiveInformation": (
            "No intentional personal data collection. Any prompts or outputs should be "
            "screened via downstream release governance before publication."
        ),
        "rai:dataReleaseMaintenancePlan": (
            "Versioned with reproducibility metadata (seed, git SHA, command, timestamps) "
            "and deterministic audit hashes per intervention record."
        ),
    }


def write_metadata_json(metadata: dict[str, Any], metadata_path: Path) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def validate_csv_shape(csv_path: Path, required_columns: tuple[str, ...] = REQUIRED_COLUMNS) -> list[str]:
    """Validate CSV header and basic shape for Croissant field extraction."""
    errors: list[str] = []

    if not csv_path.exists():
        return [f"CSV file does not exist: {csv_path}"]

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            errors.append(f"CSV missing required columns: {', '.join(missing)}")

        row_count = sum(1 for _ in reader)
        if row_count == 0:
            errors.append("CSV has no data rows; expected at least one intervention record.")

    return errors


def validate_metadata_shape(metadata: dict[str, Any]) -> list[str]:
    """Validate core Croissant top-level shape and field source mappings."""
    errors: list[str] = []

    required_top_level = ("@context", "@type", "distribution", "recordSet", "name", "description", "datePublished")
    for key in required_top_level:
        if key not in metadata:
            errors.append(f"Metadata missing required top-level key: {key}")

    conforms_to = metadata.get("conformsTo")
    if conforms_to is None:
        conforms_to = metadata.get("dct:conformsTo")

    if conforms_to is None:
        errors.append("Metadata missing required top-level key: conformsTo")

    if isinstance(conforms_to, str):
        conforms_values = [conforms_to]
    elif isinstance(conforms_to, list):
        conforms_values = conforms_to
    else:
        conforms_values = []

    if CROISSANT_SPEC_URI not in conforms_values:
        errors.append(f"Metadata conformsTo must include {CROISSANT_SPEC_URI}")
    if RAI_SPEC_URI not in conforms_values:
        errors.append(f"Metadata conformsTo must include {RAI_SPEC_URI}")

    distribution = metadata.get("distribution")
    record_set = metadata.get("recordSet")
    if not isinstance(distribution, list) or not distribution:
        errors.append("Metadata distribution must be a non-empty list.")

    if not isinstance(record_set, list) or not record_set:
        errors.append("Metadata recordSet must be a non-empty list.")
        return errors

    known_distribution_ids = {
        item.get("@id") for item in distribution if isinstance(item, dict) and isinstance(item.get("@id"), str)
    }

    first_record_set = record_set[0] if isinstance(record_set[0], dict) else {}
    key_spec = first_record_set.get("key")
    if key_spec is None:
        errors.append("First recordSet entry should define key for record uniqueness.")

    fields = first_record_set.get("field")
    if not isinstance(fields, list) or not fields:
        errors.append("First recordSet entry must include a non-empty field list.")
        return errors

    for item in distribution:
        if isinstance(item, dict) and "sha256" in item and not _is_hex_sha256(item.get("sha256")):
            errors.append(f"distribution item {item.get('@id', '<unknown>')} has invalid sha256 format.")

    extracted_columns: list[str] = []
    field_ids: set[str] = set()
    for field in fields:
        if not isinstance(field, dict):
            errors.append("recordSet.field entries must be objects.")
            continue

        field_id = field.get("@id")
        if isinstance(field_id, str) and field_id:
            field_ids.add(field_id)

        source = field.get("source")
        if not isinstance(source, dict):
            errors.append(f"Field {field.get('name', '<unknown>')} missing source mapping.")
            continue

        file_object = source.get("fileObject")
        extract = source.get("extract")
        if not isinstance(file_object, dict) or "@id" not in file_object:
            errors.append(f"Field {field.get('name', '<unknown>')} source.fileObject.@id missing.")
            continue

        if file_object.get("@id") not in known_distribution_ids:
            errors.append(
                f"Field {field.get('name', '<unknown>')} references unknown fileObject id: {file_object.get('@id')}"
            )

        if not isinstance(extract, dict) or "column" not in extract:
            errors.append(f"Field {field.get('name', '<unknown>')} source.extract.column missing.")
            continue

        extracted_columns.append(str(extract.get("column")))

    missing_required = [column for column in REQUIRED_COLUMNS if column not in extracted_columns]
    if missing_required:
        errors.append(
            "Metadata field mappings missing required extracted columns: " + ", ".join(missing_required)
        )

    if key_spec is not None:
        key_ids: list[str] = []
        if isinstance(key_spec, list):
            for key_ref in key_spec:
                if isinstance(key_ref, dict) and isinstance(key_ref.get("@id"), str):
                    key_ids.append(key_ref["@id"])
                elif isinstance(key_ref, str):
                    key_ids.append(key_ref)
        elif isinstance(key_spec, dict) and isinstance(key_spec.get("@id"), str):
            key_ids = [key_spec["@id"]]
        elif isinstance(key_spec, str):
            key_ids = [key_spec]

        if not key_ids:
            errors.append("recordSet.key must reference one or more field @id values.")
        else:
            unknown_key_ids = [key_id for key_id in key_ids if key_id not in field_ids]
            if unknown_key_ids:
                errors.append(
                    "recordSet.key references unknown field ids: " + ", ".join(unknown_key_ids)
                )

    return errors


def validate_with_mlcroissant(metadata_path: Path) -> tuple[int, str]:
    """Run optional strict validation with mlcroissant CLI if installed."""
    executable = shutil.which("mlcroissant")
    if executable is None:
        return 127, "mlcroissant CLI not found in PATH"

    process = subprocess.run(
        [executable, "validate", "--jsonld", str(metadata_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    output = (process.stdout or "") + (process.stderr or "")
    return process.returncode, output.strip()


def export_run_artifact_to_croissant(
    *,
    run_artifact_path: Path,
    output_dir: Path,
    strict: bool = False,
    dataset_name: str = "KV-Cache Inception Runtime Interventions",
    dataset_description: str = (
        "Runtime evidence export from Reversible MCTS in KV-cache latent space. "
        "Each record corresponds to one MCTS node with telemetry channels, "
        "diagnostic classification, OEI/TDS metrics, and reproducibility provenance."
    ),
) -> dict[str, Any]:
    """Create a self-contained Croissant package from one run artifact JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    run_artifact_path = run_artifact_path.resolve()
    if not run_artifact_path.exists():
        raise FileNotFoundError(f"Run artifact not found: {run_artifact_path}")

    artifact_data = json.loads(run_artifact_path.read_text(encoding="utf-8"))

    packaged_run_artifact = output_dir / "source_run.json"
    shutil.copy2(run_artifact_path, packaged_run_artifact)

    records = build_records_from_run_artifact(
        artifact_data,
        run_artifact_path=packaged_run_artifact.name,
        experiment_id="run_kv_mcts",
    )

    csv_path = output_dir / "interventions.csv"
    write_records_csv(records, csv_path)

    csv_errors = validate_csv_shape(csv_path)
    if csv_errors:
        raise ValueError("CSV validation failed: " + " | ".join(csv_errors))

    metadata = build_metadata_document(
        csv_file_name=csv_path.name,
        csv_sha256=_sha256_file(csv_path),
        source_artifact_name=packaged_run_artifact.name,
        source_artifact_sha256=_sha256_file(packaged_run_artifact),
        dataset_name=dataset_name,
        dataset_description=dataset_description,
    )

    metadata_errors = validate_metadata_shape(metadata)
    if metadata_errors:
        raise ValueError("Metadata validation failed: " + " | ".join(metadata_errors))

    metadata_path = output_dir / "metadata.json"
    write_metadata_json(metadata, metadata_path)

    strict_validation_output = ""
    strict_validation_return_code = None
    if strict:
        strict_validation_return_code, strict_validation_output = validate_with_mlcroissant(metadata_path)
        if strict_validation_return_code != 0:
            raise RuntimeError(
                "Strict mlcroissant validation failed "
                f"(exit={strict_validation_return_code}): {strict_validation_output}"
            )

    return {
        "record_count": len(records),
        "csv_path": str(csv_path),
        "metadata_path": str(metadata_path),
        "source_run_path": str(packaged_run_artifact),
        "strict_validation_return_code": strict_validation_return_code,
        "strict_validation_output": strict_validation_output,
    }
