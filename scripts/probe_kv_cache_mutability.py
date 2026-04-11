#!/usr/bin/env python3
"""Phase 2 gate: validate in-place KV-cache mutability on local HuggingFace models.

This probe verifies the required precondition for Reversible MCTS:
  1) A KV-cache tensor can be mutated in place.
  2) That mutation changes downstream logits.
  3) Restoring the original tensor restores baseline logits.

Expected usage:
  uv run python scripts/probe_kv_cache_mutability.py \
      --model ./models/llama-3.2-1b \
      --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _clone_legacy_cache(past_key_values: Any):
    """Clone tuple/list-form past_key_values recursively."""
    import torch

    if not isinstance(past_key_values, (tuple, list)):
        raise TypeError("Only tuple/list past_key_values are supported by this probe")
    cloned_layers = []
    for layer in past_key_values:
        if not isinstance(layer, (tuple, list)):
            raise TypeError("Unexpected layer format in past_key_values")
        cloned_layers.append(
            tuple(t.clone() if torch.is_tensor(t) else t for t in layer)
        )
    return tuple(cloned_layers)


def _get_first_key_tensor(past_key_values: Any):
    """Return the first layer key tensor for in-place mutation testing."""
    if isinstance(past_key_values, (tuple, list)):
        if not past_key_values:
            raise ValueError("Empty past_key_values")
        layer0 = past_key_values[0]
        if not isinstance(layer0, (tuple, list)) or not layer0:
            raise ValueError("Unexpected layer0 format in past_key_values")
        return layer0[0], "legacy_tuple"

    if hasattr(past_key_values, "key_cache"):
        key_cache = getattr(past_key_values, "key_cache")
        if not key_cache:
            raise ValueError("Empty key_cache on dynamic cache")
        return key_cache[0], "dynamic_cache"

    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def run_probe(args: argparse.Namespace) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}. "
            "Download first with huggingface-cli."
        )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_dtype = torch.float16 if device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch_dtype,
        device_map=device,
    )
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
    baseline_cache = _clone_legacy_cache(past_key_values)
    mutated_cache = _clone_legacy_cache(past_key_values)

    probe_input_ids = inputs["input_ids"][:, -1:]
    with torch.no_grad():
        baseline = model(
            input_ids=probe_input_ids,
            past_key_values=baseline_cache,
            use_cache=True,
        )
    baseline_logits = baseline.logits[:, -1, :].float()

    mutated_key_tensor, _ = _get_first_key_tensor(mutated_cache)
    original_key = mutated_key_tensor.detach().clone()
    with torch.no_grad():
        mutated_key_tensor.add_(args.alpha)
        mutated = model(
            input_ids=probe_input_ids,
            past_key_values=mutated_cache,
            use_cache=True,
        )
    mutated_logits = mutated.logits[:, -1, :].float()

    delta = (mutated_logits - baseline_logits).abs()
    max_delta = float(delta.max().item())
    mean_delta = float(delta.mean().item())

    with torch.no_grad():
        mutated_key_tensor.copy_(original_key)
        reverted = model(
            input_ids=probe_input_ids,
            past_key_values=mutated_cache,
            use_cache=True,
        )
    reverted_logits = reverted.logits[:, -1, :].float()
    revert_max_delta = float((reverted_logits - baseline_logits).abs().max().item())

    mutable = max_delta > args.min_delta
    reversible = revert_max_delta <= args.revert_tol

    return {
        "model": str(model_path),
        "device": device,
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
    parser.add_argument("--model", required=True, help="Local model directory")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device selection",
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
        print(
            json.dumps(
                {
                    "gate_passed": False,
                    "error": str(exc),
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
