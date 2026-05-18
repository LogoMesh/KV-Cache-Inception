"""Build the Croissant 1.1 supplementary package for the ARR May 2026 paper.

Reads primary-source experiment JSONs under ``scripts/_track_*.json``, flattens
Track F (entropy-MCTS negative control, Experiment 1) and Track G
(Latent Cartography, Experiment 2) into row-wise CSVs, copies all raw track
JSONs plus ``uv.lock`` into ``docs/dataset/data/``, and emits a
``docs/dataset/croissant.json`` Croissant 1.1 + RAI 1.0 JSON-LD metadata
document describing the bundle.

Design notes are in ``docs/logs/2026-05-17_croissant-build-plan.md`` (Phase 1).

Hard rules upheld:
- No fabricated field values; every cell traces to a primary-source JSON path.
- Loud failure (``ValueError``) on any structural invariant violation.
- Idempotent within the output directory (safe to re-run).
- License + URL are live-repo (Option D+); anonymization for the
  anonymous.4open.science upload happens in Phase 4 cleanup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Reuse the existing JSON-LD @context block (Croissant 1.1 + RAI vocab); the
# context itself is spec-aligned and reusable across Phase 2 (Phase-2-era
# Run-KV-MCTS export) and Phase 3 (Track F + G supplementary). The records and
# RAI strings around it are re-authored below to reflect Option D+ framing.
from logomesh.croissant_export import (
    CROISSANT_CONTEXT,
    CROISSANT_SPEC_URI,
    RAI_SPEC_URI,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "dataset"

# --- Source JSON paths (relative to REPO_ROOT) ---------------------------------

TRACK_F_SOURCES: tuple[tuple[str, Path], ...] = (
    ("1B", REPO_ROOT / "scripts" / "_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json"),
    ("3B", REPO_ROOT / "scripts" / "_track_f_results_meta-llama_Llama-3.2-3B-Instruct.json"),
)

TRACK_G_MCTS_SOURCES: tuple[tuple[str, Path], ...] = (
    ("1B", REPO_ROOT / "scripts" / "_track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json"),
    ("3B", REPO_ROOT / "scripts" / "_track_g_mcts_results_meta-llama_Llama-3.2-3B-Instruct.json"),
)

TRACK_G_INIT_SOURCES: tuple[tuple[str, Path], ...] = (
    ("1B", REPO_ROOT / "scripts" / "_track_g_results_meta-llama_Llama-3.2-1B-Instruct.json"),
    ("3B", REPO_ROOT / "scripts" / "_track_g_results_meta-llama_Llama-3.2-3B-Instruct.json"),
)

TRACK_A_SOURCES: tuple[tuple[str, Path], ...] = (
    ("1B", REPO_ROOT / "scripts" / "_track_a_results_meta-llama_Llama-3.2-1B-Instruct.json"),
    ("3B", REPO_ROOT / "scripts" / "_track_a_results_meta-llama_Llama-3.2-3B-Instruct.json"),
)

TRACK_C_SOURCES: tuple[tuple[str, Path], ...] = (
    ("1B", REPO_ROOT / "scripts" / "_track_c_results_meta-llama_Llama-3.2-1B-Instruct.json"),
    ("3B", REPO_ROOT / "scripts" / "_track_c_results_meta-llama_Llama-3.2-3B-Instruct.json"),
)

TRACK_D_SOURCES: tuple[tuple[str, Path], ...] = (
    ("1B", REPO_ROOT / "scripts" / "_track_d_vram.json"),
    ("3B", REPO_ROOT / "scripts" / "_track_d_vram_3b.json"),
)

TRACK_G_C1_DEFINITION: Path = REPO_ROOT / "scripts" / "_track_g_c1_factual_recall.json"
UV_LOCK_SOURCE: Path = REPO_ROOT / "uv.lock"

# --- Class-code → human name mapping (Track G) ---------------------------------
# Per docs/logs/2026-05-11_track-G-cartography-report.md and the C1 definition
# JSON. C2/C3/C4 names are confirmed against the paper abstract S6 line 56
# "(factual recall, ARC-Easy/TruthfulQA MCQ, HellaSwag continuation)".

TRACK_G_CLASS_NAMES: dict[str, str] = {
    "C1": "factual_recall",
    "C2": "arc_easy_mcq",
    "C3": "truthfulqa_mcq",
    "C4": "hellaswag_continuation",
}

EXPECTED_TRACK_G_CLASSES: tuple[str, ...] = ("C1", "C2", "C3", "C4")
EXPECTED_TRACK_F_CONDITIONS: tuple[str, ...] = ("G", "R", "E")


# --- Track F field schema -------------------------------------------------------

@dataclass(frozen=True)
class FieldSpec:
    """Column-level metadata for Croissant field generation."""

    name: str
    data_type: str
    description: str


TRACK_F_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("scale", "sc:Text", "Model scale: '1B' or '3B' (meta-llama/Llama-3.2-{1B,3B}-Instruct)."),
    FieldSpec("item_idx", "sc:Integer", "Item index 0-199 within the 200-item ARC-Easy subset (deterministic indices, no shuffling)."),
    FieldSpec("source_id", "sc:Text", "Source ARC-Easy item identifier (e.g., 'Mercury_417466')."),
    FieldSpec("gold_letter", "sc:Text", "Gold answer letter, one of A/B/C/D."),
    FieldSpec("n_options", "sc:Integer", "Number of multiple-choice options (3 or 4 depending on item)."),
    FieldSpec("condition", "sc:Text", "Decoding condition: 'G' (greedy, no MCTS), 'R' (random-MCTS), 'E' (entropy-MCTS)."),
    FieldSpec("pred_letter", "sc:Text", "Predicted answer letter from the 5-token model.generate output."),
    FieldSpec("correct", "sc:Integer", "1 if pred_letter equals gold_letter (case-insensitive), else 0."),
    FieldSpec("answer_text", "sc:Text", "Decoded text from the model.generate output (max 5 new tokens)."),
    FieldSpec("first_token_entropy_nats", "sc:Float", "Shannon entropy of the first-token distribution under the active condition (nats)."),
    FieldSpec("per_token_entropy_mean_nats", "sc:Float", "Mean per-token Shannon entropy across generated tokens (nats)."),
    FieldSpec("perplexity_5tok", "sc:Float", "Perplexity over the 5-token completion."),
    FieldSpec("search_status", "sc:Text", "MCTS status: 'ok' (search ran), 'n/a' (greedy, no search)."),
    FieldSpec("mean_search_reward", "sc:Float", "Mean reward across all 27 enumerated paths under the active condition (empty for greedy)."),
    FieldSpec("best_search_reward", "sc:Float", "Reward of the best-rewarded path applied to the live KV cache (empty for greedy)."),
    FieldSpec("mean_terminal_entropy_nats", "sc:Float", "Mean terminal-step entropy across all 27 paths (nats; empty for greedy)."),
    FieldSpec("best_path_json", "sc:Text", "JSON-encoded best path: a 3-element list of alpha values from {0.1, 0.5, 1.0} (empty for greedy)."),
    FieldSpec("n_paths_evaluated", "sc:Integer", "Number of paths evaluated by MCTS (27 = depth-3 enumeration of {0.1,0.5,1.0}^3; empty for greedy)."),
    FieldSpec("n_apply_failures", "sc:Integer", "Number of paths that failed to apply (empty for greedy; empirically 0 for all R/E records)."),
)

# --- Track G MCTS field schema --------------------------------------------------

TRACK_G_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("scale", "sc:Text", "Model scale: '1B' or '3B' (meta-llama/Llama-3.2-{1B,3B}-Instruct)."),
    FieldSpec("class_code", "sc:Text", "Prompt class code: C1 (factual recall), C2 (ARC-Easy MCQ), C3 (TruthfulQA mc1 MCQ), C4 (HellaSwag continuation)."),
    FieldSpec("class_name", "sc:Text", "Human-readable class name (factual_recall, arc_easy_mcq, truthfulqa_mcq, hellaswag_continuation)."),
    FieldSpec("item_idx", "sc:Integer", "Item index within the class (0-99 for C1/C2/C4; 0-84 for C3 at both scales, since 15 TruthfulQA mc1 items are dropped at the dataset-construction step for having fewer than 3 distractors)."),
    FieldSpec("subclass", "sc:Text", "Per-class subclass label: 'capital'/'date'/'arithmetic'/'knowledge' for C1; 'logical_deduction' for C2; 'moral_preference' for C3; 'creative_completion' for C4."),
    FieldSpec("source_id", "sc:Text", "Stable record identifier composed as '<class>_<idx>_<subclass>'."),
    FieldSpec("eval_mode", "sc:Text", "Evaluation mode used for the class (always 'first_token_str' for Track G MCTS-primary)."),
    FieldSpec("prompt_preview", "sc:Text", "First ~80 characters of the prompt template applied (chat-template not included for brevity)."),
    FieldSpec("gold", "sc:Text", "Gold answer string for the item."),
    FieldSpec("search_status", "sc:Text", "MCTS status: 'ok' (search ran), error code on failure."),
    FieldSpec("best_path_json", "sc:Text", "JSON-encoded best path: a 3-element list of alpha values from {0.1, 0.5, 1.0}."),
    FieldSpec("n_paths_evaluated", "sc:Integer", "Number of paths evaluated by MCTS (always 27 for depth-3 enumeration)."),
    FieldSpec("n_apply_failures", "sc:Integer", "Number of paths that failed to apply."),
    FieldSpec("mean_search_reward", "sc:Float", "Mean reward across all 27 enumerated paths."),
    FieldSpec("best_search_reward", "sc:Float", "Reward of the best-rewarded path."),
    FieldSpec("mean_terminal_entropy_nats", "sc:Float", "Mean terminal-step entropy across all 27 paths (nats)."),
    FieldSpec("best_terminal_entropy_nats", "sc:Float", "Terminal-step entropy of the best-rewarded path (nats)."),
    FieldSpec("argmax_token_str", "sc:Text", "Decoded argmax first-token string under the best-path-steered KV cache."),
    FieldSpec("argmax_token_id", "sc:Integer", "Argmax first-token vocabulary ID under the best-path-steered KV cache."),
    FieldSpec("next_token_entropy_nats", "sc:Float", "Entropy of the next-token distribution at the generation step under best-path steering (nats)."),
    FieldSpec("top5_json", "sc:Text", "JSON-encoded top-5 next-token distribution: list of [token_str, probability] pairs."),
    FieldSpec("gold_first_token_rank", "sc:Integer", "Rank of the gold first token in the full vocab distribution (0 = top; -1 if rank is outside top-K returned by the search)."),
    FieldSpec("gold_first_token_prob", "sc:Float", "Probability mass on the gold first token under best-path steering."),
    FieldSpec("wallclock_sec", "sc:Float", "Wall-clock seconds for this item's MCTS expansion (path enumeration + reward evaluation + apply/rollback)."),
)


# --- IO helpers ----------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required source JSON does not exist: {path}")
    try:
        # parse_constant=lambda c: float('nan') if c == 'NaN' else ... — accept JSON NaN literal
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, parse_constant=_parse_json_constant)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse source JSON at {path}: {exc}") from exc


def _parse_json_constant(token: str) -> float:
    """Accept JSON's non-standard NaN / Infinity tokens that Python's json emits."""
    if token == "NaN":
        return math.nan
    if token == "Infinity":
        return math.inf
    if token == "-Infinity":
        return -math.inf
    raise ValueError(f"Unrecognized JSON constant: {token}")


def _coerce_cell(value: Any) -> str:
    """Convert a Python value to a CSV-safe scalar string.

    None / NaN -> empty string (the column's dataType remains sc:Float; readers
    treat empty cells as 'not applicable').
    bool -> 'true' / 'false'.
    int / float -> str(value); NaN handled above.
    list / dict -> compact JSON (no whitespace).
    str -> verbatim (CSV writer handles quoting).
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return repr(value)  # round-tripping via repr() avoids loss
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    return str(value)


# --- Track F flattener ---------------------------------------------------------

def _flatten_track_f(sources: tuple[tuple[str, Path], ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scale, path in sources:
        data = _load_json(path)
        _assert_track_f_invariants(scale, path, data)
        items = data["items"]
        for item in items:
            for cond in EXPECTED_TRACK_F_CONDITIONS:
                cond_block = item.get(cond)
                if not isinstance(cond_block, dict):
                    raise ValueError(
                        f"Track F {scale} item idx={item.get('idx')} missing condition block {cond}: {path}"
                    )
                rows.append({
                    "scale": scale,
                    "item_idx": item.get("idx"),
                    "source_id": item.get("source_id"),
                    "gold_letter": item.get("gold_letter"),
                    "n_options": item.get("n_options"),
                    "condition": cond,
                    "pred_letter": cond_block.get("pred_letter"),
                    "correct": cond_block.get("correct"),
                    "answer_text": cond_block.get("answer_text"),
                    "first_token_entropy_nats": cond_block.get("first_token_entropy_nats"),
                    "per_token_entropy_mean_nats": cond_block.get("per_token_entropy_mean_nats"),
                    "perplexity_5tok": cond_block.get("perplexity_5tok"),
                    "search_status": cond_block.get("search_status"),
                    "mean_search_reward": cond_block.get("mean_search_reward"),
                    "best_search_reward": cond_block.get("best_search_reward"),
                    "mean_terminal_entropy_nats": cond_block.get("mean_terminal_entropy_nats"),
                    "best_path_json": cond_block.get("best_path"),
                    "n_paths_evaluated": cond_block.get("n_paths_evaluated"),
                    "n_apply_failures": cond_block.get("n_apply_failures"),
                })
    return rows


def _assert_track_f_invariants(scale: str, path: Path, data: dict[str, Any]) -> None:
    if data.get("n_items") != 200:
        raise ValueError(f"Track F {scale} expected n_items=200, got {data.get('n_items')}: {path}")
    items = data.get("items")
    if not isinstance(items, list) or len(items) != 200:
        raise ValueError(f"Track F {scale} expected 200 items, got len={len(items) if isinstance(items, list) else 'N/A'}: {path}")
    config = data.get("config") or {}
    if config.get("seed") != 42:
        raise ValueError(f"Track F {scale} expected seed=42, got {config.get('seed')}: {path}")
    if config.get("depth") != 3 or config.get("n_nodes") != 27:
        raise ValueError(
            f"Track F {scale} expected depth=3, n_nodes=27, got depth={config.get('depth')}, "
            f"n_nodes={config.get('n_nodes')}: {path}"
        )


# --- Track G flattener ---------------------------------------------------------

def _flatten_track_g_mcts(sources: tuple[tuple[str, Path], ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scale, path in sources:
        data = _load_json(path)
        _assert_track_g_invariants(scale, path, data)
        for record in data["records"]:
            cls = record.get("class")
            if cls not in TRACK_G_CLASS_NAMES:
                raise ValueError(f"Track G {scale} record has unknown class={cls}: {path}")
            rows.append({
                "scale": scale,
                "class_code": cls,
                "class_name": TRACK_G_CLASS_NAMES[cls],
                "item_idx": record.get("item_idx"),
                "subclass": record.get("subclass"),
                "source_id": record.get("source_id"),
                "eval_mode": record.get("eval_mode"),
                "prompt_preview": record.get("prompt_preview"),
                "gold": record.get("gold"),
                "search_status": record.get("search_status"),
                "best_path_json": record.get("best_path"),
                "n_paths_evaluated": record.get("n_paths_evaluated"),
                "n_apply_failures": record.get("n_apply_failures"),
                "mean_search_reward": record.get("mean_search_reward"),
                "best_search_reward": record.get("best_search_reward"),
                "mean_terminal_entropy_nats": record.get("mean_terminal_entropy_nats"),
                "best_terminal_entropy_nats": record.get("best_terminal_entropy_nats"),
                "argmax_token_str": record.get("argmax_token_str"),
                "argmax_token_id": record.get("argmax_token_id"),
                "next_token_entropy_nats": record.get("next_token_entropy_nats"),
                "top5_json": record.get("top5"),
                "gold_first_token_rank": record.get("gold_first_token_rank"),
                "gold_first_token_prob": record.get("gold_first_token_prob"),
                "wallclock_sec": record.get("wallclock_sec"),
            })
    return rows


def _assert_track_g_invariants(scale: str, path: Path, data: dict[str, Any]) -> None:
    if data.get("mode") != "mcts":
        raise ValueError(f"Track G MCTS {scale} expected mode='mcts', got {data.get('mode')}: {path}")
    if data.get("seed") != 42:
        raise ValueError(f"Track G MCTS {scale} expected seed=42, got {data.get('seed')}: {path}")
    classes = tuple(data.get("classes") or ())
    if classes != EXPECTED_TRACK_G_CLASSES:
        raise ValueError(f"Track G MCTS {scale} expected classes={EXPECTED_TRACK_G_CLASSES}, got {classes}: {path}")
    if data.get("depth") != 3 or data.get("n_nodes") != 27:
        raise ValueError(
            f"Track G MCTS {scale} expected depth=3, n_nodes=27, got depth={data.get('depth')}, "
            f"n_nodes={data.get('n_nodes')}: {path}"
        )
    records = data.get("records")
    if not isinstance(records, list) or len(records) < 385:
        raise ValueError(
            f"Track G MCTS {scale} expected >=385 records, got len={len(records) if isinstance(records, list) else 'N/A'}: {path}"
        )


# --- CSV writer ----------------------------------------------------------------

def _write_csv(rows: list[dict[str, Any]], fields: tuple[FieldSpec, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f.name for f in fields]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _coerce_cell(row.get(col)) for col in columns})


# --- File-copy helper ----------------------------------------------------------

def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and src.exists() and _sha256_file(src) == _sha256_file(dst):
        return  # Idempotent: skip copy when checksums match
    shutil.copy2(src, dst)


# --- Inline Croissant 1.1 shape validator --------------------------------------
# The shape-check in logomesh.croissant_export is hard-wired to a Phase-2 28-column
# schema (one row per MCTS node). This function checks the structural rules of
# Croissant 1.1 that apply regardless of recordSet field names. The mlcroissant
# CLI (run via --strict) provides the full spec-conformant validation.

_HEX_CHARS = set("0123456789abcdefABCDEF")


def _is_hex_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX_CHARS for c in value)


def _validate_shape(metadata: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top = (
        "@context", "@type", "conformsTo", "description", "license", "name",
        "url", "creator", "datePublished", "distribution", "recordSet",
    )
    for key in required_top:
        if key not in metadata:
            errors.append(f"missing required top-level key: {key}")
    if metadata.get("@type") != "sc:Dataset":
        errors.append(f"@type must be 'sc:Dataset', got {metadata.get('@type')!r}")
    conforms_to = metadata.get("conformsTo")
    if isinstance(conforms_to, str):
        conforms_list = [conforms_to]
    elif isinstance(conforms_to, list):
        conforms_list = conforms_to
    else:
        conforms_list = []
    if CROISSANT_SPEC_URI not in conforms_list:
        errors.append(f"conformsTo must include {CROISSANT_SPEC_URI}")

    distribution = metadata.get("distribution")
    if not isinstance(distribution, list) or not distribution:
        errors.append("distribution must be a non-empty list")
        distribution = []
    known_distribution_ids: set[str] = set()
    for item in distribution:
        if not isinstance(item, dict):
            errors.append("distribution entry must be an object")
            continue
        item_id = item.get("@id")
        if not isinstance(item_id, str) or not item_id:
            errors.append("distribution entry missing @id")
            continue
        known_distribution_ids.add(item_id)
        if "sha256" in item and not _is_hex_sha256(item.get("sha256")):
            errors.append(f"distribution entry {item_id} has invalid sha256 format")

    record_set = metadata.get("recordSet")
    if not isinstance(record_set, list) or not record_set:
        errors.append("recordSet must be a non-empty list")
        return errors

    for rs_index, rs in enumerate(record_set):
        if not isinstance(rs, dict):
            errors.append(f"recordSet[{rs_index}] must be an object")
            continue
        rs_id = rs.get("@id") or f"<recordSet[{rs_index}]>"
        for req in ("@type", "@id", "name", "description", "field"):
            if req not in rs:
                errors.append(f"recordSet {rs_id} missing required key: {req}")
        fields = rs.get("field")
        if not isinstance(fields, list) or not fields:
            errors.append(f"recordSet {rs_id} must have non-empty field[]")
            continue
        field_ids: set[str] = set()
        for f_index, field in enumerate(fields):
            if not isinstance(field, dict):
                errors.append(f"recordSet {rs_id} field[{f_index}] must be an object")
                continue
            f_id = field.get("@id") or f"<field[{f_index}]>"
            field_ids.add(field.get("@id", ""))
            for req in ("@type", "@id", "name", "dataType", "source"):
                if req not in field:
                    errors.append(f"recordSet {rs_id} field {f_id} missing required key: {req}")
            source = field.get("source")
            if not isinstance(source, dict):
                continue
            file_object = source.get("fileObject")
            if not isinstance(file_object, dict) or "@id" not in file_object:
                errors.append(f"recordSet {rs_id} field {f_id} source.fileObject.@id missing")
                continue
            if file_object.get("@id") not in known_distribution_ids:
                errors.append(
                    f"recordSet {rs_id} field {f_id} references unknown fileObject @id: {file_object.get('@id')}"
                )
            extract = source.get("extract")
            if not isinstance(extract, dict) or "column" not in extract:
                errors.append(f"recordSet {rs_id} field {f_id} source.extract.column missing")

        key_spec = rs.get("key")
        if key_spec is not None:
            key_ids: list[str] = []
            if isinstance(key_spec, list):
                for k in key_spec:
                    if isinstance(k, dict) and isinstance(k.get("@id"), str):
                        key_ids.append(k["@id"])
                    elif isinstance(k, str):
                        key_ids.append(k)
            elif isinstance(key_spec, dict) and isinstance(key_spec.get("@id"), str):
                key_ids = [key_spec["@id"]]
            elif isinstance(key_spec, str):
                key_ids = [key_spec]
            unknown = [k for k in key_ids if k not in field_ids]
            if unknown:
                errors.append(
                    f"recordSet {rs_id} key references unknown field @id(s): {', '.join(unknown)}"
                )

    return errors


# --- Croissant metadata builder ------------------------------------------------

def _field_dict(record_set_id: str, spec: FieldSpec, file_object_id: str) -> dict[str, Any]:
    """Construct a single cr:Field entry for a CSV-backed recordSet."""
    return {
        "@type": "cr:Field",
        "@id": f"{record_set_id}/{spec.name}",
        "name": spec.name,
        "description": spec.description,
        "dataType": spec.data_type,
        "source": {
            "fileObject": {"@id": file_object_id},
            "extract": {"column": spec.name},
        },
    }


def _build_metadata(
    *,
    track_f_csv_sha256: str,
    track_g_csv_sha256: str,
    file_objects: list[dict[str, Any]],
    build_date: str,
) -> dict[str, Any]:
    dataset_name = "KV-Cache Inception: Reversible KV-Cache MCTS Runtime Evidence"

    dataset_description = (
        "Runtime evidence and per-item records from Reversible KV-Cache Monte Carlo Tree Search (MCTS) "
        "on frozen Llama 3.2-{1B,3B}-Instruct, used to characterize the Dimensional Escape pathology "
        "(an instance of Goodhart's Law in high-dimensional latent-space search) under an "
        "entropy-normalized surrogate reward. Includes two headline experiments: Track F "
        "(entropy-MCTS negative control vs random-MCTS vs greedy baseline on 200 ARC-Easy items; "
        "n=200 paired McNemar, p=1.0 at both scales) and Track G (Latent Cartography across four "
        "prompt classes -- hand-constructed factual recall, ARC-Easy MCQ, HellaSwag continuation, "
        "TruthfulQA MCQ). Supporting tracks A, C, D provide static entropy-correctness correlation, "
        "benchmark calibration, and VRAM-measurement records that anchor the paper's reproducibility "
        "appendix Section A. Released as supplementary material for the ARR May 2026 / EMNLP 2026 "
        "paper 'Probing Dimensional Escape in Frozen Language Models via Reversible KV-Cache MCTS.'"
    )

    creator = {"@type": "sc:Organization", "name": "LogoMesh Research Team"}

    track_f_fields = [_field_dict("track_f_negcontrol", spec, "track-f-csv") for spec in TRACK_F_FIELDS]
    track_g_fields = [_field_dict("track_g_cartography_mcts", spec, "track-g-csv") for spec in TRACK_G_FIELDS]

    cite_as = (
        "@inproceedings{kvcache_inception_2026,\n"
        "  title = {Probing Dimensional Escape in Frozen Language Models via Reversible KV-Cache MCTS},\n"
        "  author = {[Authors]},\n"
        "  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing "
        "(via ACL Rolling Review May 2026)},\n"
        "  year = {2026}\n"
        "}"
    )

    rai_block: dict[str, str] = {
        "rai:dataUseCases": (
            "Interpretability research on inference-time KV-cache search behavior in frozen language models. "
            "The data documents a negative-result characterization of Dimensional Escape (an instance of "
            "Goodhart's Law in high-dimensional latent-space search) under an entropy-normalized surrogate "
            "reward across two model scales. Use cases include: reproducing the paper's Track F "
            "(entropy-MCTS negative control) and Track G (Latent Cartography) headline findings; "
            "benchmarking subsequent reward designs against the dataset's pre-registered failure mode; "
            "extending the Cartography sweep to additional prompt classes or model families."
        ),
        "rai:dataLimitations": (
            "Records are derived from two Llama 3.2 scales (1B and 3B Instruct variants) on a small sample "
            "of hand-curated and public-benchmark items (Track F: 200 ARC-Easy items per scale; Track G: "
            "~100 items per prompt class per scale, with class C3 (TruthfulQA mc1 MCQ) yielding 85 records "
            "at both scales because 15 of the first 100 validation-set items have fewer than 3 distractors "
            "and are dropped at the dataset-construction step). "
            "No human-annotated ground-truth labels beyond the public benchmarks' provided gold answers. "
            "Top-5 token records and best-path entries are model-internal artifacts; cross-family "
            "comparability requires re-calibration of the steering protocol. Wall-clock seconds are "
            "captured per-item where available but cover the MCTS expansion loop only, not model-loading "
            "or evaluation overhead."
        ),
        "rai:dataBiases": (
            "Findings are conditional on the specific steering direction (a single random unit vector in "
            "d_model space, seeded by --seed 42, shared across items and across the random-MCTS and "
            "entropy-MCTS arms) and on the Llama 3.2 instruction-tuned models. Behavioral patterns "
            "(first-letter prior amplification at 1B, greedy-collapse at 3B) are properties of this "
            "model family under this steering protocol and should not be generalized to other model "
            "families, other steering directions, or non-instruction-tuned base models without "
            "independent verification."
        ),
        "rai:personalSensitiveInformation": (
            "No personal data. Prompts are drawn from public benchmarks (ARC-Easy, HellaSwag, TruthfulQA, "
            "SciQ, AQuA-RAT, ProofWriter, LogiQA) or hand-constructed factual-recall items (world "
            "capitals, historical dates, single-digit arithmetic, common-knowledge facts). "
            "Model-generated answers are 5-token continuations against multiple-choice options or "
            "first-token completions; no PII generation paths."
        ),
        "rai:dataReleaseMaintenancePlan": (
            "Versioned snapshot for ACL Rolling Review May 2026 submission (EMNLP 2026 commitment). "
            "Provenance metadata baked into each record: random seed (42), HuggingFace model "
            "identifiers (meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct), "
            "PyTorch version (2.6.0+cu124), CUDA version (12.4). HuggingFace checkpoint commit SHAs "
            "are filled at submission via huggingface-cli per the paper's Appendix A.2. Phase B "
            "(post-acceptance) extensions will add additional models and prompt classes; the package "
            "version field will increment accordingly."
        ),
    }

    metadata: dict[str, Any] = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "name": dataset_name,
        "description": dataset_description,
        "conformsTo": [CROISSANT_SPEC_URI, RAI_SPEC_URI],
        "license": "https://opensource.org/licenses/MIT",
        "url": "https://github.com/LogoMesh/KV-Cache-Inception",
        "creator": creator,
        "publisher": creator,
        "datePublished": build_date,
        "dateCreated": build_date,
        "dateModified": build_date,
        "version": "1.0.0",
        "sdVersion": "1.1",
        "inLanguage": "en",
        "keywords": [
            "dimensional escape",
            "reversible kv-cache mcts",
            "frozen language model interpretability",
            "latent geometry probing",
            "goodhart's law",
            "negative result",
        ],
        "citeAs": cite_as,
        "distribution": file_objects,
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "track_f_negcontrol",
                "name": "track_f_negcontrol",
                "description": (
                    "Per-item-per-condition records from Track F (Experiment 1: entropy-MCTS negative control). "
                    "200 ARC-Easy items at each of two model scales (1B, 3B), each evaluated under three "
                    "decoding conditions: G (greedy, no MCTS), R (random-MCTS: paths scored with "
                    "Uniform(0,1) reward), E (entropy-MCTS: paths scored with -H_t / log|V| from Eq. 1). "
                    "All MCTS arms enumerate the 27 depth-3 paths from {0.1, 0.5, 1.0} and apply the "
                    "best-rewarded path permanently before the 5-token answer continuation."
                ),
                "key": [
                    {"@id": "track_f_negcontrol/scale"},
                    {"@id": "track_f_negcontrol/item_idx"},
                    {"@id": "track_f_negcontrol/condition"},
                ],
                "field": track_f_fields,
            },
            {
                "@type": "cr:RecordSet",
                "@id": "track_g_cartography_mcts",
                "name": "track_g_cartography_mcts",
                "description": (
                    "Per-item records from Track G (Experiment 2: Latent Cartography MCTS-primary variant). "
                    "Four prompt classes (C1=factual_recall, C2=arc_easy_mcq, C3=truthfulqa_mcq, "
                    "C4=hellaswag_continuation), 100 items per class at each of two model scales (with class C3 "
                    "TruthfulQA mc1 yielding 85 records at both scales because 15 of the first 100 validation-set "
                    "items have fewer than 3 distractors and are dropped at the dataset-construction step). "
                    "Each record reports the best-path entropy-MCTS expansion's terminal token, "
                    "next-token distribution top-5, and gold-token rank/probability."
                ),
                "key": [
                    {"@id": "track_g_cartography_mcts/scale"},
                    {"@id": "track_g_cartography_mcts/class_code"},
                    {"@id": "track_g_cartography_mcts/item_idx"},
                ],
                "field": track_g_fields,
            },
        ],
        **rai_block,
    }
    return metadata


def _build_file_objects(
    *,
    track_f_csv_path: Path,
    track_g_csv_path: Path,
    raw_copies: list[tuple[str, str, Path, str]],
    uv_lock_path: Path,
) -> list[dict[str, Any]]:
    """Build the distribution[] list.

    raw_copies: list of (file_object_id, encoding, dst_path, description) tuples
    where dst_path is the path the file ALREADY exists at (under output_dir/data/...).
    """
    objects: list[dict[str, Any]] = [
        {
            "@type": "cr:FileObject",
            "@id": "track-f-csv",
            "name": "track_f_negcontrol.csv",
            "contentUrl": "data/track_f_negcontrol.csv",
            "encodingFormat": "text/csv",
            "sha256": _sha256_file(track_f_csv_path),
            "description": (
                "Flattened Track F records (1200 rows = 200 items x 3 conditions x 2 scales). "
                "Source-of-truth backing the track_f_negcontrol recordSet."
            ),
        },
        {
            "@type": "cr:FileObject",
            "@id": "track-g-csv",
            "name": "track_g_cartography_mcts.csv",
            "contentUrl": "data/track_g_cartography_mcts.csv",
            "encodingFormat": "text/csv",
            "sha256": _sha256_file(track_g_csv_path),
            "description": (
                "Flattened Track G MCTS-primary records (770 rows = 385 per scale x 2 scales; "
                "C3 has 85 at 3B due to HellaSwag tokenization drops). Source-of-truth backing "
                "the track_g_cartography_mcts recordSet."
            ),
        },
    ]
    for file_id, encoding, dst_path, description in raw_copies:
        if not dst_path.exists():
            raise FileNotFoundError(f"Expected packaged copy at {dst_path} but it is missing.")
        objects.append({
            "@type": "cr:FileObject",
            "@id": file_id,
            "name": dst_path.name,
            "contentUrl": f"data/raw/{dst_path.name}",
            "encodingFormat": encoding,
            "sha256": _sha256_file(dst_path),
            "description": description,
        })
    # uv.lock under data/ (not data/raw/) to make the dependency-pin file easy to find
    objects.append({
        "@type": "cr:FileObject",
        "@id": "uv-lock",
        "name": uv_lock_path.name,
        "contentUrl": f"data/{uv_lock_path.name}",
        "encodingFormat": "text/plain",
        "sha256": _sha256_file(uv_lock_path),
        "description": (
            "Pinned dependency manifest (uv.lock) listing the exact resolved versions of every "
            "library used to produce the records. Reproducibility-critical: paper Appendix A.1 "
            "cites this file."
        ),
    })
    return objects


# --- Build orchestration -------------------------------------------------------

def build_package(output_dir: Path, *, strict: bool, build_date: str) -> dict[str, Any]:
    """Top-level orchestration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    raw_dir = data_dir / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/7] Loading + flattening Track F (Experiment 1) records...")
    track_f_rows = _flatten_track_f(TRACK_F_SOURCES)
    if len(track_f_rows) != 1200:
        raise ValueError(f"Track F flatten produced {len(track_f_rows)} rows, expected 1200")

    print(f"[2/7] Loading + flattening Track G MCTS-primary (Experiment 2) records...")
    track_g_rows = _flatten_track_g_mcts(TRACK_G_MCTS_SOURCES)
    expected_min_g = 770  # 385 * 2; C3=85 at 3B is accounted for in the 385 per-scale invariant
    if len(track_g_rows) < expected_min_g:
        raise ValueError(f"Track G MCTS flatten produced {len(track_g_rows)} rows, expected >= {expected_min_g}")

    print(f"[3/7] Writing flattened CSVs to {data_dir}...")
    track_f_csv_path = data_dir / "track_f_negcontrol.csv"
    track_g_csv_path = data_dir / "track_g_cartography_mcts.csv"
    _write_csv(track_f_rows, TRACK_F_FIELDS, track_f_csv_path)
    _write_csv(track_g_rows, TRACK_G_FIELDS, track_g_csv_path)

    print(f"[4/7] Copying raw track JSONs to {raw_dir}...")
    raw_copies: list[tuple[str, str, Path, str]] = []
    raw_copy_specs: list[tuple[str, str, Path, str]] = [
        ("track-f-raw-1b", "application/json", TRACK_F_SOURCES[0][1],
         "Track F raw output, Llama 3.2-1B-Instruct (200 items x G/R/E conditions; full provenance)."),
        ("track-f-raw-3b", "application/json", TRACK_F_SOURCES[1][1],
         "Track F raw output, Llama 3.2-3B-Instruct (200 items x G/R/E conditions; full provenance)."),
        ("track-g-mcts-raw-1b", "application/json", TRACK_G_MCTS_SOURCES[0][1],
         "Track G MCTS-primary raw output, Llama 3.2-1B-Instruct (4 classes; C3 TruthfulQA mc1 yields 85 records due to <3-distractor filter, others=100; 385 records total)."),
        ("track-g-mcts-raw-3b", "application/json", TRACK_G_MCTS_SOURCES[1][1],
         "Track G MCTS-primary raw output, Llama 3.2-3B-Instruct (4 classes; C3 TruthfulQA mc1 yields 85 records due to <3-distractor filter, others=100; 385 records total)."),
        ("track-g-init-raw-1b", "application/json", TRACK_G_INIT_SOURCES[0][1],
         "Track G initial parameter-sweep variant, 1B (HISTORICAL; supplemented by MCTS-primary above per 2026-05-11 protocol-fidelity re-run)."),
        ("track-g-init-raw-3b", "application/json", TRACK_G_INIT_SOURCES[1][1],
         "Track G initial parameter-sweep variant, 3B (HISTORICAL; supplemented by MCTS-primary above per 2026-05-11 protocol-fidelity re-run)."),
        ("track-g-c1-definition", "application/json", TRACK_G_C1_DEFINITION,
         "Dataset definition for Track G class C1 (factual recall): 100 hand-constructed items (25 capitals + 25 dates + 25 arithmetic + 25 common-knowledge)."),
        ("track-a-raw-1b", "application/json", TRACK_A_SOURCES[0][1],
         "Track A raw output, 1B: static entropy-correctness correlation on 85 factual-recall items (25 easy + 30 hard + 30 ultra)."),
        ("track-a-raw-3b", "application/json", TRACK_A_SOURCES[1][1],
         "Track A raw output, 3B: static entropy-correctness correlation on 85 factual-recall items (25 easy + 30 hard + 30 ultra)."),
        ("track-c-raw-1b", "application/json", TRACK_C_SOURCES[0][1],
         "Track C raw output, 1B: benchmark calibration sweep across 6 benchmarks (ARC-Easy, SciQ, HellaSwag, AQuA-RAT, ProofWriter, LogiQA), 100 items each."),
        ("track-c-raw-3b", "application/json", TRACK_C_SOURCES[1][1],
         "Track C raw output, 3B: benchmark calibration sweep across 6 benchmarks (ARC-Easy, SciQ, HellaSwag, AQuA-RAT, ProofWriter, LogiQA), 100 items each."),
        ("track-d-raw-1b", "application/json", TRACK_D_SOURCES[0][1],
         "Track D raw output, 1B: VRAM measurement across 3 (depth, branches, nodes) configurations on RTX 3060."),
        ("track-d-raw-3b", "application/json", TRACK_D_SOURCES[1][1],
         "Track D raw output, 3B: VRAM measurement across 3 (depth, branches, nodes) configurations on RTX 3060."),
    ]
    for file_id, encoding, src_path, description in raw_copy_specs:
        if not src_path.exists():
            raise FileNotFoundError(f"Required raw source missing: {src_path}")
        dst_path = raw_dir / src_path.name
        _copy_file(src_path, dst_path)
        raw_copies.append((file_id, encoding, dst_path, description))

    print(f"[5/7] Copying uv.lock to {data_dir}...")
    uv_lock_dst = data_dir / "uv.lock"
    _copy_file(UV_LOCK_SOURCE, uv_lock_dst)

    print(f"[6/7] Building Croissant 1.1 + RAI 1.0 metadata JSON-LD...")
    file_objects = _build_file_objects(
        track_f_csv_path=track_f_csv_path,
        track_g_csv_path=track_g_csv_path,
        raw_copies=raw_copies,
        uv_lock_path=uv_lock_dst,
    )
    metadata = _build_metadata(
        track_f_csv_sha256=_sha256_file(track_f_csv_path),
        track_g_csv_sha256=_sha256_file(track_g_csv_path),
        file_objects=file_objects,
        build_date=build_date,
    )

    shape_errors = _validate_shape(metadata)
    if shape_errors:
        raise RuntimeError("Croissant metadata shape validation failed: " + " | ".join(shape_errors))

    croissant_path = output_dir / "croissant.json"
    croissant_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"[7/7] Writing MANIFEST.txt...")
    manifest_path = output_dir / "MANIFEST.txt"
    _write_manifest(
        manifest_path,
        croissant_path=croissant_path,
        track_f_csv_path=track_f_csv_path,
        track_g_csv_path=track_g_csv_path,
        raw_copies=raw_copies,
        uv_lock_path=uv_lock_dst,
        track_f_row_count=len(track_f_rows),
        track_g_row_count=len(track_g_rows),
        build_date=build_date,
    )

    strict_result: dict[str, Any] = {"return_code": None, "output": ""}
    if strict:
        print("Running strict mlcroissant validation...")
        cli_path = shutil.which("mlcroissant") or shutil.which(
            "mlcroissant.exe",
            path=str(Path(sys.executable).parent),
        )
        if cli_path is None:
            raise RuntimeError(
                "Strict validation requested but mlcroissant CLI not found in PATH. "
                "Install via `uv pip install mlcroissant`."
            )
        proc = subprocess.run(
            [cli_path, "validate", "--jsonld", str(croissant_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            cwd=str(REPO_ROOT),
        )
        strict_result = {
            "return_code": proc.returncode,
            "output": (proc.stdout + proc.stderr).strip(),
        }
        if proc.returncode != 0:
            raise RuntimeError(
                "Strict mlcroissant validation failed "
                f"(exit={proc.returncode}):\n{strict_result['output']}"
            )

    summary = {
        "croissant_path": str(croissant_path),
        "track_f_csv_path": str(track_f_csv_path),
        "track_g_csv_path": str(track_g_csv_path),
        "track_f_row_count": len(track_f_rows),
        "track_g_row_count": len(track_g_rows),
        "raw_copy_count": len(raw_copies),
        "manifest_path": str(manifest_path),
        "strict_validation": strict_result,
    }
    return summary


def _write_manifest(
    manifest_path: Path,
    *,
    croissant_path: Path,
    track_f_csv_path: Path,
    track_g_csv_path: Path,
    raw_copies: list[tuple[str, str, Path, str]],
    uv_lock_path: Path,
    track_f_row_count: int,
    track_g_row_count: int,
    build_date: str,
) -> None:
    lines = [
        "KV-Cache Inception: Reversible KV-Cache MCTS Runtime Evidence",
        f"Built: {build_date}",
        "",
        "Files (with sha256):",
    ]
    files_to_list = [
        (croissant_path, "Croissant 1.1 + RAI 1.0 metadata JSON-LD"),
        (track_f_csv_path, f"Track F flattened CSV ({track_f_row_count} rows)"),
        (track_g_csv_path, f"Track G MCTS-primary flattened CSV ({track_g_row_count} rows)"),
    ]
    for _, _, dst_path, description in raw_copies:
        files_to_list.append((dst_path, description))
    files_to_list.append((uv_lock_path, "Pinned dependency manifest"))

    for path, description in files_to_list:
        rel = path.relative_to(REPO_ROOT)
        size = path.stat().st_size
        sha = _sha256_file(path)
        lines.append(f"  {rel}  ({size:,} bytes, sha256={sha})")
        lines.append(f"    {description}")
        lines.append("")
    manifest_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output package root (default: {DEFAULT_OUTPUT_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Run mlcroissant CLI validation after build (fails loudly on non-zero exit).",
    )
    parser.add_argument(
        "--build-date",
        type=str,
        default="2026-05-17",
        help="ISO date string baked into datePublished/dateCreated/dateModified (default: 2026-05-17).",
    )
    args = parser.parse_args(argv)
    summary = build_package(args.output_dir, strict=args.strict, build_date=args.build_date)
    print()
    print("=" * 78)
    print("BUILD SUMMARY")
    print("=" * 78)
    print(f"Croissant package:   {summary['croissant_path']}")
    print(f"Track F CSV:         {summary['track_f_csv_path']} ({summary['track_f_row_count']} rows)")
    print(f"Track G CSV:         {summary['track_g_csv_path']} ({summary['track_g_row_count']} rows)")
    print(f"Raw JSON copies:     {summary['raw_copy_count']}")
    print(f"Manifest:            {summary['manifest_path']}")
    strict = summary["strict_validation"]
    if strict["return_code"] is None:
        print("Strict validation:   skipped (use --strict to enable)")
    else:
        print(f"Strict validation:   exit={strict['return_code']} ({'OK' if strict['return_code'] == 0 else 'FAILED'})")
        if strict["output"]:
            print(f"  output: {strict['output'][:200]}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
