#!/usr/bin/env python3
"""Phase 2 gate: validate in-place KV-cache mutability on HuggingFace models.

This probe verifies the required precondition for Reversible MCTS:
  1) A KV-cache tensor can be mutated in place.
  2) That mutation changes downstream logits.
  3) Restoring the original tensor restores baseline logits.

Expected usage:
  uv run python scripts/probe_kv_cache_mutability.py \
      --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --device auto

Or with a local folder:
  uv run python scripts/probe_kv_cache_mutability.py \
      --model ./models/llama-3.2-1b \
      --device auto
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def _clone_cache(past_key_values: Any):
    """Clone past_key_values while preserving tensor values and structure."""
    import torch

    if isinstance(past_key_values, (tuple, list)):
        cloned_layers = []
        for layer in past_key_values:
            if not isinstance(layer, (tuple, list)):
                raise TypeError("Unexpected layer format in past_key_values")
            cloned_layers.append(
                tuple(t.clone() if torch.is_tensor(t) else t for t in layer)
            )
        return tuple(cloned_layers)

    # Dynamic cache classes in newer transformers are safest to clone deeply.
    return copy.deepcopy(past_key_values)


def _get_first_key_tensor(past_key_values: Any):
    """Return the first layer key tensor for in-place mutation testing."""
    if isinstance(past_key_values, (tuple, list)):
        if not past_key_values:
            raise ValueError("Empty past_key_values")
        layer0 = past_key_values[0]
        if not isinstance(layer0, (tuple, list)) or not layer0:
            raise ValueError("Unexpected layer0 format in past_key_values")
        return layer0[0], "legacy_tuple"

    # Transformers <=5.2 style DynamicCache
    if hasattr(past_key_values, "key_cache"):
        key_cache = getattr(past_key_values, "key_cache")
        if not key_cache:
            raise ValueError("Empty key_cache on dynamic cache")
        return key_cache[0], "dynamic_cache"

    # Transformers 5.3 style DynamicCache with list-valued layers
    if hasattr(past_key_values, "layers"):
        layers = getattr(past_key_values, "layers")
        if not layers:
            raise ValueError("Empty layers on dynamic cache")

        layer0 = layers[0]
        if hasattr(layer0, "keys"):
            k = layer0.keys
            if k is None:
                raise ValueError("Layer 0 keys are not initialised")
            return k, "dynamic_cache_layers"

        if isinstance(layer0, (tuple, list)) and layer0:
            return layer0[0], "dynamic_cache_layers_tuple"

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def _resolve_device(requested: str):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(requested: str, device: str):
    import torch

    if requested == "float32":
        return torch.float32, "float32"
    if requested == "float16":
        return torch.float16, "float16"
    if requested == "bfloat16":
        return torch.bfloat16, "bfloat16"

    # auto mode
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"
    if device == "mps":
        return torch.float16, "float16"
    return torch.float32, "float32"


def _resolve_model_ref(model_arg: str) -> tuple[str, bool]:
    """Return (model_reference, is_local_path)."""
    as_path = Path(model_arg)
    if as_path.exists():
        return str(as_path), True
    return model_arg, False


def run_probe(args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from logomesh.kv_mcts import _kv_eval_cache

    model_ref, is_local = _resolve_model_ref(args.model)
    device = _resolve_device(args.device)
    torch_dtype, dtype_label = _resolve_dtype(args.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(model_ref)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    prompt = args.prompt
    if args.use_chat_template:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        initial = model(**inputs, use_cache=True)

    past_key_values = initial.past_key_values
    if past_key_values is None:
        raise RuntimeError("Model forward did not return past_key_values")

    key_tensor, cache_type = _get_first_key_tensor(past_key_values)
    if not torch.is_tensor(key_tensor):
        raise TypeError("Resolved key tensor is not a torch.Tensor")

    # Clone cache twice so baseline and mutated runs are comparable.
    baseline_cache = _clone_cache(past_key_values)
    mutated_cache = _clone_cache(past_key_values)

    probe_input_ids = inputs["input_ids"][:, -1:]

    # Use tuple snapshots for all model calls so DynamicCache.update() cannot
    # replace key_cache[l] entries between mutation and restore steps.
    with torch.no_grad():
        baseline = model(
            input_ids=probe_input_ids,
            past_key_values=_kv_eval_cache(baseline_cache),
            use_cache=True,
        )
    baseline_logits = baseline.logits[:, -1, :].float()

    mutated_key_tensor, _ = _get_first_key_tensor(mutated_cache)
    original_key = mutated_key_tensor.detach().clone()
    with torch.no_grad():
        mutated_key_tensor.add_(args.alpha)
        # Snapshot AFTER mutation so the model sees the mutated tensor
        mutated = model(
            input_ids=probe_input_ids,
            past_key_values=_kv_eval_cache(mutated_cache),
            use_cache=True,
        )
    mutated_logits = mutated.logits[:, -1, :].float()

    delta = (mutated_logits - baseline_logits).abs()
    max_delta = float(delta.max().item())
    mean_delta = float(delta.mean().item())

    with torch.no_grad():
        # Restore original key tensor in the cache object (still valid — not replaced)
        mutated_key_tensor.copy_(original_key)
        # Snapshot AFTER restore so the model sees the original tensor
        reverted = model(
            input_ids=probe_input_ids,
            past_key_values=_kv_eval_cache(mutated_cache),
            use_cache=True,
        )
    reverted_logits = reverted.logits[:, -1, :].float()
    revert_max_delta = float((reverted_logits - baseline_logits).abs().max().item())

    mutable = max_delta > args.min_delta
    reversible = revert_max_delta <= args.revert_tol

    return {
        "model": model_ref,
        "model_source": "local_path" if is_local else "huggingface_id_or_cache",
        "device": device,
        "dtype": dtype_label,
        "cache_type": cache_type,
        "alpha": args.alpha,
        "min_delta": args.min_delta,
        "revert_tol": args.revert_tol,
        "max_logit_delta_after_mutation": max_delta,
        "mean_logit_delta_after_mutation": mean_delta,
        "max_logit_delta_after_revert": revert_max_delta,
        "mutable_in_place": mutable,
        "reversible_with_copy_restore": reversible,
        "gate_passed": mutable and reversible,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model reference: local directory path or HuggingFace model id",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Torch device selection",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype selection (auto adapts per device)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain why entropy matters in language modeling.",
        help="Prompt used to seed KV cache",
    )
    parser.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Apply chat template before tokenization",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-3,
        help="In-place delta applied to first-layer key tensor",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-7,
        help="Minimum logit delta considered a real mutation effect",
    )
    parser.add_argument(
        "--revert-tol",
        type=float,
        default=1e-5,
        help="Max logit delta allowed after restore",
    )
    args = parser.parse_args()

    try:
        result = run_probe(args)
    except Exception as exc:
        suggestion = (
            "If using a gated model id, run `huggingface-cli login` first. "
            "If using a local path, verify the directory exists and contains model files."
        )
        print(
            json.dumps(
                {
                    "gate_passed": False,
                    "error": str(exc),
                    "suggestion": suggestion,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))

    if result["gate_passed"]:
        return 0
    if not result["mutable_in_place"]:
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
