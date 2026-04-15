#!/usr/bin/env python3
"""
Export a run_kv_mcts JSON artifact to a self-contained Croissant package.

Produces, under the output directory:
- metadata.json      (Croissant 1.1 + RAI 1.0 JSON-LD)
- interventions.csv  (one row per MCTS node)
- source_run.json    (copied raw runtime artifact)

Usage:
    uv run python scripts/export_kv_mcts_to_croissant.py \
        --input ./tmp/mcts_qwen_offline_n1_step2check.json \
        --output ./tmp/croissant_export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from logomesh.croissant_export import export_run_artifact_to_croissant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to run_kv_mcts JSON artifact.")
    parser.add_argument("--output", required=True, help="Directory for generated Croissant package files.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Run strict mlcroissant CLI validation (requires mlcroissant installed in PATH).",
    )
    args = parser.parse_args()

    export_info = export_run_artifact_to_croissant(
        run_artifact_path=Path(args.input),
        output_dir=Path(args.output),
        strict=args.strict,
    )

    print(json.dumps(export_info, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
