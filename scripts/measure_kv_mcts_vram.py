"""Track D — Reversible KV-Cache MCTS VRAM measurement (memory-only).

Per `docs/logs/2026-05-08_diagnostic-prompt-track-D.md`. Measures peak VRAM in
three modes per (depth, branches, nodes) configuration:

  baseline — model + KV cache live in memory (no MCTS state).
  mid      — baseline + bf16 baseline-clones of K and V (≈ "MCTS with
             accumulator disabled / random-perturb" — what naive
             "snapshot before perturb, subtract after" rollback would hold).
  full     — baseline + FP32Accumulator.from_kv_cache(...) (algorithm's
             actual per-search storage: K_base + V_base in bf16 + K_accum +
             V_accum in fp32).

Methodology rationale:
  - We DO NOT run the full ReversibleMCTS rollout (per prompt: "memory
    measurement only"). The FP32Accumulator is allocated once at root and
    rollback restores in-place, so per-search memory is independent of the
    number of expansions. The only depth/branch dependency would be transient
    activation buffers from `model(use_cache=True)` calls — those are bounded
    by the longest single forward pass, not by the search size.
  - Each mode is measured in isolation: we record `baseline_floor` once
    (model + activations from one warm-up forward pass), then for each mode we
    reset the CUDA peak counter, allocate the mode's structures, record the
    peak, and tear down before the next mode.
  - We report the memory in two normalisations:
      delta_over_baseline_mib    — (peak_mode − baseline_mib).
      delta_in_kv_units          — delta / kv_cache_mib (the §6-relevant
                                    constant factor in M_KV units).
  - The seed-prompt length is configurable (default 4000 tokens) so the KV
    cache is large enough to dominate measurement noise on a 1B Llama
    (KV at 4k tokens ≈ 130 MiB on Llama-3.2-1B).

Read-only on logomesh/* — imports `FP32Accumulator`, `_extract_kv_tensors`,
and `LocalLlamaOracle` only.
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import statistics
import subprocess
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from logomesh.kv_mcts import FP32Accumulator, _extract_kv_tensors  # type: ignore
from logomesh.local_model import LocalLlamaOracle  # type: ignore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("track_d")

DEFAULT_SEED_TOKENS = 4000


def nvsmi_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().splitlines()
        return int(out[0])
    except Exception:
        return None


def _bytes_to_mib(b: int) -> float:
    return b / (1024 ** 2)


def _kv_total_bytes(past_kv) -> int:
    layers = _extract_kv_tensors(past_kv)
    total = 0
    for k, v in layers:
        total += k.element_size() * k.numel()
        total += v.element_size() * v.numel()
    return total


def _kv_shapes_summary(past_kv) -> dict:
    layers = _extract_kv_tensors(past_kv)
    if not layers:
        return {}
    k0, v0 = layers[0]
    return {
        "n_layers": len(layers),
        "k_shape": list(k0.shape),
        "v_shape": list(v0.shape),
        "k_dtype": str(k0.dtype),
        "v_dtype": str(v0.dtype),
    }


def _peak_alloc_mib() -> float:
    return _bytes_to_mib(torch.cuda.max_memory_allocated())


def _curr_alloc_mib() -> float:
    """Current allocated memory (drops as tensors are freed)."""
    return _bytes_to_mib(torch.cuda.memory_allocated())


def _hard_reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _build_long_prompt(oracle: LocalLlamaOracle, target_tokens: int) -> tuple[torch.Tensor, int]:
    """Tokenize a long context that fills target_tokens KV positions."""
    tok = oracle._tokenizer
    # Repeat a content-rich paragraph until we exceed the target.
    seed = (
        "We measure the peak GPU memory of a Reversible KV-Cache Monte Carlo Tree "
        "Search algorithm operating in the latent space of a frozen language model. "
        "The accumulator stores per-layer FP32 deltas which permit exact rollback "
        "after each candidate steering perturbation. The forward pass is captured "
        "with use_cache=True so the live KV cache reflects the prompt prefix. "
    )
    text = ""
    while True:
        text += seed
        ids = tok(text, return_tensors="pt").input_ids
        if ids.shape[1] >= target_tokens:
            ids = ids[:, :target_tokens]
            break
    return ids.to(oracle._device), int(ids.shape[1])


def _seed_kv_cache(oracle: LocalLlamaOracle, seed_tokens: int):
    """Run one forward pass with `seed_tokens` of context to populate past_key_values."""
    ids, n_tokens = _build_long_prompt(oracle, seed_tokens)
    with torch.no_grad():
        out = oracle._model(input_ids=ids, use_cache=True)
    return out.past_key_values, n_tokens


def measure_one_repeat(oracle: LocalLlamaOracle, seed_tokens: int) -> dict:
    """Measure steady-state allocation per mode, plus the forward-pass peak.

    We track BOTH:
      - steady-state (memory_allocated) AFTER forward completes and intermediate
        activations are freed — this is the algorithm's persistent footprint.
      - per-step peak (max_memory_allocated) DURING forward — this is the
        transient activation surge from one model(use_cache=True) call. The
        algorithm experiences this peak each step regardless of mode.
    """
    # 0) Hard reset to clear any prior peak.
    _hard_reset()

    # 1) Forward pass — populate KV cache. Records the activation peak.
    past_kv, n_tokens = _seed_kv_cache(oracle, seed_tokens)
    torch.cuda.synchronize()
    fwd_peak = _peak_alloc_mib()  # transient peak during forward
    # Reset peak so subsequent allocations are visible above the steady floor.
    torch.cuda.reset_peak_memory_stats()
    baseline_steady = _curr_alloc_mib()
    kv_bytes = _kv_total_bytes(past_kv)
    kv_mib = _bytes_to_mib(kv_bytes)
    shapes = _kv_shapes_summary(past_kv)

    # 2) Mid mode: allocate bf16 K_clone + V_clone (naive snapshot rollback).
    layers = _extract_kv_tensors(past_kv)
    k_clone = [k.clone() for k, _ in layers]
    v_clone = [v.clone() for _, v in layers]
    torch.cuda.synchronize()
    mid_steady = _curr_alloc_mib()
    # Free clones before measuring full
    del k_clone, v_clone
    gc.collect()
    torch.cuda.empty_cache()

    # 3) Full mode: allocate FP32Accumulator.
    accumulator = FP32Accumulator.from_kv_cache(past_kv, model=oracle._model)
    torch.cuda.synchronize()
    full_steady = _curr_alloc_mib()

    k_base_bytes = sum(t.element_size() * t.numel() for t in accumulator.k_base)
    v_base_bytes = sum(t.element_size() * t.numel() for t in accumulator.v_base)
    k_accum_bytes = sum(t.element_size() * t.numel() for t in accumulator.k_accum)
    v_accum_bytes = sum(t.element_size() * t.numel() for t in accumulator.v_accum)
    accumulator_total_bytes = k_base_bytes + v_base_bytes + k_accum_bytes + v_accum_bytes

    # 4) Per-step peak under "full" mode: simulate one steered forward pass with
    #    the accumulator alive. This is the worst-case in-flight VRAM during search.
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        # Single-token continuation re-using past_kv, mirroring run_async() loop.
        ids_step = torch.tensor([[0]], device=oracle._device, dtype=torch.long)
        _ = oracle._model(input_ids=ids_step, past_key_values=past_kv, use_cache=True)
    torch.cuda.synchronize()
    full_step_peak = _peak_alloc_mib()

    del accumulator, past_kv
    oracle.clear_cache()
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "n_tokens": n_tokens,
        "kv_cache_mib": kv_mib,
        "kv_shapes": shapes,
        "fwd_peak_mib": fwd_peak,
        "baseline_steady_mib": baseline_steady,
        "mid_steady_mib": mid_steady,
        "full_steady_mib": full_steady,
        "full_step_peak_mib": full_step_peak,
        "k_base_mib": _bytes_to_mib(k_base_bytes),
        "v_base_mib": _bytes_to_mib(v_base_bytes),
        "k_accum_mib": _bytes_to_mib(k_accum_bytes),
        "v_accum_mib": _bytes_to_mib(v_accum_bytes),
        "accumulator_total_mib": _bytes_to_mib(accumulator_total_bytes),
    }


def measure_one_config(
    oracle: LocalLlamaOracle, *, depth: int, branches: int, nodes: int,
    n_repeats: int, seed_tokens: int,
) -> dict:
    """For a (depth, branches, nodes) cell. The algorithm's per-search memory is
    INDEPENDENT of (depth, branches, nodes) because the FP32 accumulator is
    allocated once at root and rollback restores in-place. We re-measure at
    each cell anyway as the protocol asks, and document the invariance."""
    log.info("=== config: depth=%d branches=%d nodes=%d ===", depth, branches, nodes)
    repeats = []
    for r in range(n_repeats):
        log.info("repeat %d/%d", r + 1, n_repeats)
        rec = measure_one_repeat(oracle, seed_tokens=seed_tokens)
        repeats.append(rec)

    def _med(key):
        return statistics.median(rec[key] for rec in repeats)

    median_baseline = _med("baseline_steady_mib")
    median_mid = _med("mid_steady_mib")
    median_full = _med("full_steady_mib")
    median_full_step_peak = _med("full_step_peak_mib")
    median_fwd_peak = _med("fwd_peak_mib")
    median_kv = _med("kv_cache_mib")
    median_accum = _med("accumulator_total_mib")

    delta_mid = median_mid - median_baseline
    delta_full = median_full - median_baseline

    return {
        "depth": depth,
        "branches": branches,
        "nodes": nodes,
        "n_repeats": n_repeats,
        "seed_tokens": seed_tokens,
        "kv_shapes": repeats[0]["kv_shapes"],
        "kv_cache_mib_median": median_kv,
        "fwd_peak_mib_runs": [r["fwd_peak_mib"] for r in repeats],
        "baseline_steady_mib_runs": [r["baseline_steady_mib"] for r in repeats],
        "mid_steady_mib_runs": [r["mid_steady_mib"] for r in repeats],
        "full_steady_mib_runs": [r["full_steady_mib"] for r in repeats],
        "full_step_peak_mib_runs": [r["full_step_peak_mib"] for r in repeats],
        "accumulator_total_mib_runs": [r["accumulator_total_mib"] for r in repeats],
        "fwd_peak_mib_median": median_fwd_peak,
        "baseline_steady_mib_median": median_baseline,
        "mid_steady_mib_median": median_mid,
        "full_steady_mib_median": median_full,
        "full_step_peak_mib_median": median_full_step_peak,
        "accumulator_total_mib_median": median_accum,
        "delta_mid_over_baseline_mib": delta_mid,
        "delta_full_over_baseline_mib": delta_full,
        "delta_mid_in_kv_units": delta_mid / median_kv if median_kv > 0 else float("nan"),
        "delta_full_in_kv_units": delta_full / median_kv if median_kv > 0 else float("nan"),
        "constant_factor_full_over_baseline": median_full / median_baseline if median_baseline > 0 else float("nan"),
        "k_base_mib_median": statistics.median(r["k_base_mib"] for r in repeats),
        "v_base_mib_median": statistics.median(r["v_base_mib"] for r in repeats),
        "k_accum_mib_median": statistics.median(r["k_accum_mib"] for r in repeats),
        "v_accum_mib_median": statistics.median(r["v_accum_mib"] for r in repeats),
        "nvsmi_used_mib_post_full": nvsmi_used_mib(),
    }


async def main_async(args):
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device != "cuda":
        log.error("CUDA required — no GPU detected. Aborting.")
        return

    log.info("loading oracle: %s", args.model)
    oracle = LocalLlamaOracle.load(args.model, device=device)
    log.info("model loaded: %s", oracle.model_id)

    configs = [
        {"depth": 3, "branches": 3, "nodes": 27},
        {"depth": 5, "branches": 3, "nodes": 81},
        {"depth": 10, "branches": 3, "nodes": 1700},
    ]

    results = {
        "model_id": oracle.model_id,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_mib": _bytes_to_mib(torch.cuda.get_device_properties(0).total_memory),
        "n_repeats": args.repeats,
        "seed_tokens": args.seed_tokens,
        "configs": [],
        "notes": [
            "FP32Accumulator is allocated once at root; per-search VRAM is independent of (depth, branches, nodes).",
            "Mid mode = bf16 baseline-clone of K and V (naive snapshot rollback strategy, no FP32 accumulator).",
            "Full mode = FP32Accumulator.from_kv_cache(): K_base+V_base (bf16) + K_accum+V_accum (fp32).",
            "Each cell is measured n_repeats times after a hard CUDA reset; medians reported.",
        ],
    }

    for cfg in configs:
        try:
            cell = measure_one_config(
                oracle, depth=cfg["depth"], branches=cfg["branches"], nodes=cfg["nodes"],
                n_repeats=args.repeats, seed_tokens=args.seed_tokens,
            )
            results["configs"].append(cell)
            log.info(
                "  KV=%.1f MiB | baseline=%.1f, mid=%.1f (Δ %+.1f, %.2f×M_KV) | "
                "full=%.1f (Δ %+.1f, %.2f×M_KV) | accum-tensor-bytes=%.1f MiB | "
                "fwd_peak=%.1f, full_step_peak=%.1f",
                cell["kv_cache_mib_median"],
                cell["baseline_steady_mib_median"],
                cell["mid_steady_mib_median"], cell["delta_mid_over_baseline_mib"], cell["delta_mid_in_kv_units"],
                cell["full_steady_mib_median"], cell["delta_full_over_baseline_mib"], cell["delta_full_in_kv_units"],
                cell["accumulator_total_mib_median"],
                cell["fwd_peak_mib_median"],
                cell["full_step_peak_mib_median"],
            )
        except torch.cuda.OutOfMemoryError as e:
            log.error("OOM on config %s: %s", cfg, e)
            results["configs"].append({**cfg, "status": "OOM", "error": str(e)})
            torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info("wrote %s", out)

    print("\nSUMMARY:")
    print(json.dumps({"model": results["model_id"], "configs": [
        {k: c.get(k) for k in ("depth", "branches", "nodes",
                                "kv_cache_mib_median",
                                "baseline_steady_mib_median",
                                "mid_steady_mib_median",
                                "full_steady_mib_median",
                                "full_step_peak_mib_median",
                                "fwd_peak_mib_median",
                                "accumulator_total_mib_median",
                                "delta_full_over_baseline_mib",
                                "delta_full_in_kv_units",
                                "constant_factor_full_over_baseline",
                                "nvsmi_used_mib_post_full")}
        for c in results["configs"]
    ]}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--device", default="auto")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed-tokens", type=int, default=DEFAULT_SEED_TOKENS,
                   help="Number of prompt tokens to seed the KV cache (default 4000)")
    p.add_argument("--out", default="scripts/_track_d_vram.json")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
