"""Track G — Dataset access probe.

Verifies that the three Cartography datasets load cleanly under current
`datasets` version before committing to the full driver script. Per
`docs/logs/2026-05-10_latent-cartography-sweep-protocol.md` §3:

  C2 (calibration / continuity with Track F) — ARC-Easy test split, first 100
  C3 (moral/preference)                      — TruthfulQA mc1 split, first 100
  C4 (creative completion)                   — HellaSwag validation split, first 100

For each: try load, print success/failure, print first 2 items' field names + a
short rendering of the first item. ARC-Easy is also probed as a positive
control since Track C + Track F have already used it successfully.

If TruthfulQA or HellaSwag fail to load (e.g., legacy-script issue under
`datasets >= 4.0` like LogiQA had per Track C), surface the error message
verbatim so we can pick substitutes.

Read-only, no GPU, no `logomesh/*` imports. Runs in ~30s on a cold cache,
<1s on a warm cache.

Usage:
    uv run python scripts/probe_track_g_datasets.py
"""
from __future__ import annotations

import sys
import traceback


def probe(name: str, loader_fn) -> None:
    print(f"\n=== {name} ===")
    try:
        ds = loader_fn()
        print(f"OK — loaded.")
        print(f"  split-keys: {list(ds.keys()) if hasattr(ds, 'keys') else 'single-split'}")
        # If it's a DatasetDict, pick the first split for inspection.
        if hasattr(ds, 'keys'):
            first_split_key = list(ds.keys())[0]
            split = ds[first_split_key]
            print(f"  inspecting split: {first_split_key} (n={len(split)})")
        else:
            split = ds
            print(f"  inspecting single split (n={len(split)})")
        if len(split) == 0:
            print("  WARN: empty split")
            return
        first = split[0]
        print(f"  fields: {list(first.keys())}")
        print(f"  first item keys + types:")
        for k, v in first.items():
            tv = type(v).__name__
            if isinstance(v, str):
                preview = v[:120].replace("\n", " ") + ("…" if len(v) > 120 else "")
                print(f"    {k!r}: {tv} — {preview!r}")
            elif isinstance(v, (list, tuple)):
                inner = type(v[0]).__name__ if v else "?"
                print(f"    {k!r}: {tv}[{inner}] (len={len(v)}) — {str(v)[:120]}")
            elif isinstance(v, dict):
                print(f"    {k!r}: dict, keys={list(v.keys())}")
                for kk, vv in v.items():
                    tvv = type(vv).__name__
                    preview = str(vv)[:80].replace("\n", " ")
                    print(f"      {kk!r}: {tvv} — {preview!r}")
            else:
                print(f"    {k!r}: {tv} — {v!r}")
    except Exception as e:
        print(f"FAIL — {type(e).__name__}: {e}")
        # Print a 5-line traceback head so we can diagnose without scrolling.
        tb = traceback.format_exc()
        for line in tb.splitlines()[:8]:
            print(f"  {line}")


def main() -> int:
    from datasets import load_dataset

    # C2 control — should always work (Track C + Track F used this).
    probe("C2 calibration: ARC-Easy", lambda: load_dataset("allenai/ai2_arc", "ARC-Easy"))

    # C3 unknown — TruthfulQA mc1.
    probe("C3 moral/preference: TruthfulQA mc1", lambda: load_dataset("truthful_qa", "multiple_choice"))

    # C4 unknown — HellaSwag.
    probe("C4 creative completion: HellaSwag", lambda: load_dataset("Rowan/hellaswag"))

    print("\n=== Done. Read each section above and report back which loaded OK. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
