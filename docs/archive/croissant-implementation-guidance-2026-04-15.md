# Croissant 1.1 / RAI 1.0 Implementation Guidance

Date: 2026-04-15
Scope: KV-Cache Inception runtime artifact packaging (`run_kv_mcts` outputs)

## Purpose

This document defines repository-specific, standards-grounded guidance for exporting and validating Croissant metadata so releases remain reproducible and externally auditable for NeurIPS.

## Evidence Base (Verified)

Primary references inspected in the submodule:

- `external/croissant/docs/croissant-spec-1.1.md`
- `external/croissant/docs/croissant-rai-spec.md`
- `external/croissant/python/mlcroissant/mlcroissant/_src/core/context.py`
- `external/croissant/python/mlcroissant/mlcroissant/_src/structure_graph/nodes/metadata.py`
- `external/croissant/python/mlcroissant/mlcroissant/_src/structure_graph/nodes/record_set.py`

Current implementation audited:

- `logomesh/croissant_export.py`
- `scripts/export_kv_mcts_to_croissant.py`
- `scripts/run_kv_mcts.py`
- `docs/dataset/croissant_schema_stub.json`
- `tests/test_croissant_export.py`

## Standards Decisions for This Repository

1. `conformsTo` alias usage:
- Keep top-level `conformsTo` in emitted JSON-LD.
- Keep `@context` mapping `"conformsTo": "dct:conformsTo"`.
- Rationale: this follows Croissant JSON-LD alias conventions and matches mlcroissant test fixtures.

2. Required dataset-level metadata:
- Always include `datePublished` in exported metadata.
- Also include `name`, `description`, `license`, `url`, `creator`, `conformsTo`, `distribution`, and `recordSet`.

3. Recommended dataset-level metadata (treated as repository baseline):
- `inLanguage`
- `keywords`
- `publisher`
- `sdVersion`

4. Record identity policy:
- Define `recordSet.key` and bind it to `interventions/artifact_id`.
- Rationale: record uniqueness remains explicit under joins and downstream consumers.

5. RAI conformance policy:
- Include baseline RAI fields in all generated metadata:
  - `rai:dataUseCases`
  - `rai:dataLimitations`
  - `rai:dataBiases`
  - `rai:personalSensitiveInformation`
  - `rai:dataReleaseMaintenancePlan`

## Exporter Requirements (Implementation Contract)

`logomesh/croissant_export.py` must enforce:

1. Metadata includes both Croissant and RAI spec URIs:
- `http://mlcommons.org/croissant/1.1`
- `http://mlcommons.org/croissant/RAI/1.0`

2. `distribution[*].sha256` values are valid hex SHA-256 strings.

3. `recordSet.field[*].source.fileObject.@id` references existing `distribution` IDs.

4. `recordSet.field[*].source.extract.column` maps exactly to exporter CSV columns.

5. `recordSet.key` points only to existing field `@id` values.

## Validation Model

Two-layer validation is required:

1. Internal shape validation (always on):
- `validate_csv_shape(...)`
- `validate_metadata_shape(...)`

2. External strict validation (optional gate):
- `validate_with_mlcroissant(...)`
- Enabled via `--strict` in `scripts/export_kv_mcts_to_croissant.py`
- Enabled via `--strict-croissant` in `scripts/run_kv_mcts.py`

If `mlcroissant` CLI is unavailable, strict validation should fail only when explicitly requested, and should report a clear dependency message.

## Test Requirements

`tests/test_croissant_export.py` should continue covering at least:

1. End-to-end package creation from run artifact.
2. Required CSV column presence.
3. Required metadata-key failures.
4. Alias support for `dct:conformsTo` and `conformsTo`.
5. Invalid SHA-256 detection in distribution entries.

Repository gate remains:

- `uv run pytest tests/ -v`

## Current Status (as of 2026-04-15)

Implemented and validated:

1. Added required `datePublished` to generated metadata.
2. Added repository-baseline recommended fields: `inLanguage`, `keywords`, `publisher`, `sdVersion`.
3. Added `recordSet.key` for `interventions/artifact_id`.
4. Strengthened metadata-shape validation for:
- spec URI coverage
- distribution SHA-256 formatting
- `recordSet.key` references
5. Updated schema stub to mirror exporter defaults.
6. Expanded tests to lock these guarantees.

Validation result:

- `uv run pytest tests/test_croissant_export.py -v` -> 5 passed
- `uv run pytest tests/ -v` -> 135 passed
- strict exporter mode currently reports `mlcroissant CLI not found in PATH` in this environment; install dependency before enforcing strict gate in CI

## Remaining Hardening (Recommended)

1. Add strict-validator CI gate in an environment where `mlcroissant` CLI is installed.
2. Add `citeAs` once canonical citation text is frozen for submission artifacts.
3. Add one golden fixture test that snapshots generated `metadata.json` for schema-drift detection.

## Operational Checklist

Before any release of Croissant artifacts:

1. Generate package from a real run artifact.
2. Run unit and full test suites.
3. Run strict validator mode when available.
4. Verify `record_count` and schema fields in output package.
5. Record command, git SHA, and validation outcomes in session log.
