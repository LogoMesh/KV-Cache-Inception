"""Track F — Negative-control / COCONUT-defense headline test on ARC-Easy.

Per `docs/logs/2026-05-08_diagnostic-prompt-track-F.md`. Three conditions per
ARC-Easy item:

  G  — greedy baseline (no MCTS).
  R  — random-perturb MCTS: search machinery exists; reward = uniform(0,1).
  E  — entropy-guided MCTS: search machinery exists; reward = -H_t/log|V|.

Same depth-3 alpha-composition search machinery in R and E; only the reward
signal differs. The COCONUT defense is: if the search machinery alone helps,
R should beat G. If the entropy reward signal carries the load, E should beat
R. The headline statistical claim is `acc(E) - acc(R) >= 10pp` at p<0.05 by
paired McNemar.

Protocol deviation (documented in the report):
  The protocol calls for `branches=3, depth=3, n_nodes=27` and "invoke
  kv_mcts.py MCTS engine". The existing `ReversibleMCTS` is tightly coupled
  to HNeuronMonitor + PerLayerHonestyProjector + telemetry calibration and
  always rolls back its perturbations, so it cannot directly produce a
  "best-path-applied" answer. We implement a focused depth-3 path-sampling
  search that uses the same FP32Accumulator primitive (read-only import from
  logomesh.kv_mcts), which preserves the COCONUT defense (random vs entropy
  on the same machinery + same direction) while remaining read-only on
  logomesh/*. Each "path" is a sequence of 3 alphas applied sequentially;
  we sample 27 such paths (matching n_nodes=27), evaluate each by rolling
  out one token under the path's cumulative steering, then commit the
  best-rewarded path's cumulative steering before generating the answer.

Read-only on logomesh/*. Outputs:
  - scripts/_track_f_results_<modelid>.json
  - one-line per-item progress to stdout.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from logomesh.kv_mcts import (  # type: ignore
    FP32Accumulator,
    _extract_kv_tensors,
    _kv_eval_cache,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("track_f")

SYSTEM_PROMPT = "You are a careful, concise reasoner. Answer multiple-choice questions with a single capital letter."
LETTERS = ["A", "B", "C", "D", "E"]
N_ITEMS = 200
MAX_NEW = 5
ALPHA_SET = (0.1, 0.5, 1.0)   # branches=3
DEPTH = 3
N_NODES = 27                   # 27 path samples


# ---------------------------------------------------------------------------
# ARC-Easy loader (same protocol as Track C / locked by Track C §3)
# ---------------------------------------------------------------------------

def load_arc_easy(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
    split = ds["test"]
    out = []
    for ex in split.select(range(min(n, len(split)))):
        choices = ex["choices"]
        gold = ex["answerKey"].strip()
        if gold.isdigit():
            gold = LETTERS[int(gold) - 1]
        if gold not in LETTERS:
            continue
        out.append({
            "question": ex["question"],
            "options": choices["text"],
            "labels": choices["label"],
            "gold_letter": gold,
            "source_id": ex.get("id", ""),
        })
    return out


def format_prompt(ex: dict) -> str:
    lines = [ex["question"], ""]
    for letter, opt in zip(ex["labels"], ex["options"]):
        lines.append(f"{letter}. {opt}")
    lines.append("")
    lines.append(f"Answer with the single capital letter ({'/'.join(ex['labels'])}):")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inference primitives
# ---------------------------------------------------------------------------

def chat_input_ids(tok, ex: dict, device: str) -> torch.Tensor:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_prompt(ex)},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(prompt, return_tensors="pt").input_ids.to(device)


def generate_answer(model, tok, input_ids: torch.Tensor, past_kv=None, max_new: int = MAX_NEW) -> dict:
    """Greedy generate from current state. If past_kv supplied, only the LAST token of input_ids is fed."""
    with torch.no_grad():
        if past_kv is None:
            gen = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
            new_ids = gen.sequences[0][input_ids.shape[1]:]
            scores = gen.scores
        else:
            # Resume from past_kv with only the last token; iterate manually
            cur_kv = past_kv
            cur_ids = input_ids[:, -1:].clone()
            new_ids_list = []
            scores = []
            for _ in range(max_new):
                with torch.no_grad():
                    out = model(input_ids=cur_ids, past_key_values=cur_kv, use_cache=True)
                logits = out.logits[0, -1, :]
                scores.append(logits.unsqueeze(0))
                next_id = int(torch.argmax(logits).item())
                new_ids_list.append(next_id)
                if next_id == tok.eos_token_id:
                    break
                cur_kv = out.past_key_values
                cur_ids = torch.tensor([[next_id]], device=input_ids.device)
            new_ids = torch.tensor(new_ids_list, device=input_ids.device)

    answer_text = tok.decode(new_ids, skip_special_tokens=True).strip()
    pred_letter = None
    for ch in answer_text:
        if ch.upper() in LETTERS:
            pred_letter = ch.upper()
            break

    # First-token entropy of the answer
    first_logits = scores[0][0].float()
    log_probs = torch.log_softmax(first_logits, dim=-1)
    probs = log_probs.exp()
    H1 = -(probs * log_probs).sum().item()

    # Per-token avg entropy across all generated tokens
    H_per_token = []
    for s in scores:
        l = s[0].float()
        lp = torch.log_softmax(l, dim=-1)
        p = lp.exp()
        H_per_token.append(-(p * lp).sum().item())

    return {
        "answer_text": answer_text,
        "pred_letter": pred_letter,
        "first_token_entropy_nats": H1,
        "per_token_entropy_mean_nats": float(np.mean(H_per_token)) if H_per_token else float("nan"),
        "n_generated": int(len(new_ids)),
        "new_ids": new_ids.cpu().tolist(),
    }


def perplexity_of_sequence(model, tok, prompt_ids: torch.Tensor, answer_ids: list[int], device: str) -> float:
    """PPL of `answer_ids` conditioned on `prompt_ids`, as scored by the unsteered model.

    Returns exp(mean per-token NLL) over the answer tokens.
    """
    if not answer_ids:
        return float("nan")
    full = torch.cat([prompt_ids[0], torch.tensor(answer_ids, device=device)]).unsqueeze(0)
    with torch.no_grad():
        out = model(input_ids=full)
    logits = out.logits[0]
    n_prompt = prompt_ids.shape[1]
    nll_sum = 0.0
    for i, gold_id in enumerate(answer_ids):
        # logits at position (n_prompt + i - 1) predict token at position (n_prompt + i)
        pos = n_prompt + i - 1
        log_probs = torch.log_softmax(logits[pos].float(), dim=-1)
        nll_sum += -float(log_probs[gold_id].item())
    return math.exp(nll_sum / len(answer_ids))


# ---------------------------------------------------------------------------
# Search loop (depth-3 alpha-composition path sampling, n_nodes=27)
# ---------------------------------------------------------------------------

def _entropy_of_logits(logits: torch.Tensor) -> tuple[float, float]:
    """Return (entropy_nats, normalised_certainty_reward = -H/log|V|)."""
    l = logits.float()
    lp = torch.log_softmax(l, dim=-1)
    p = lp.exp()
    H = -(p * lp).sum().item()
    V = l.shape[-1]
    return H, -H / math.log(V)


def run_search(
    model, tok, ex, device, *, reward_mode: str, steering_dir: np.ndarray,
    rng: random.Random,
) -> dict:
    """Run depth-3 path-sampling search. Apply best path to the live KV cache.
    Returns dict with answer + diagnostics.

    reward_mode: "random" or "entropy".
    """
    input_ids = chat_input_ids(tok, ex, device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past_kv = out.past_key_values

    # Build accumulator (one-time per item)
    accumulator = FP32Accumulator.from_kv_cache(past_kv, model=model)
    n_layers = len(_extract_kv_tensors(past_kv))
    dk_per_layer = [steering_dir.astype(np.float32)] * n_layers

    # Sample N_NODES paths of length DEPTH; record per-path mean reward
    paths = []
    rewards = []
    n_apply_failures = 0

    for _ in range(N_NODES):
        path = [rng.choice(ALPHA_SET) for _ in range(DEPTH)]
        # Apply path
        applied_alphas = []
        ok = True
        for a in path:
            if not accumulator.apply(past_kv, a, dk_per_layer):
                ok = False
                n_apply_failures += 1
                break
            applied_alphas.append(a)

        if not ok:
            # Roll back what we did apply, skip this path
            for a in reversed(applied_alphas):
                accumulator.rollback(past_kv, a, dk_per_layer)
            continue

        # Rollout: one forward step under the steered cache to read the next-token entropy
        with torch.no_grad():
            step = model(
                input_ids=input_ids[:, -1:],
                past_key_values=_kv_eval_cache(past_kv),
                use_cache=True,
            )
        next_logits = step.logits[0, -1, :]
        H, neg_norm_H = _entropy_of_logits(next_logits)

        if reward_mode == "random":
            r = rng.random()
        elif reward_mode == "entropy":
            r = neg_norm_H  # closer to 0 = more confident; we want to MAXIMIZE this
        else:
            raise ValueError(f"unknown reward_mode {reward_mode}")

        paths.append(path)
        rewards.append({"reward": r, "entropy_nats": H, "neg_norm_entropy": neg_norm_H})

        # Roll back the path (in reverse order of application)
        for a in reversed(path):
            accumulator.rollback(past_kv, a, dk_per_layer)

    if not paths:
        # No usable path — fall through to greedy
        del accumulator
        gen = generate_answer(model, tok, input_ids)
        gen["search_status"] = "no-paths-applicable"
        gen["mean_search_reward"] = float("nan")
        gen["best_path"] = None
        gen["n_apply_failures"] = n_apply_failures
        # Perplexity from unsteered model (same as greedy)
        gen["perplexity_5tok"] = perplexity_of_sequence(model, tok, input_ids, gen["new_ids"], device)
        return gen

    # Pick best path by reward
    best_idx = int(np.argmax([r["reward"] for r in rewards]))
    best_path = paths[best_idx]

    # Re-apply the best path PERMANENTLY
    for a in best_path:
        accumulator.apply(past_kv, a, dk_per_layer)

    # Generate the answer from the steered cache
    gen = generate_answer(model, tok, input_ids, past_kv=past_kv)

    # Compute perplexity of the generated tokens under the unsteered model
    pplx = perplexity_of_sequence(model, tok, input_ids, gen["new_ids"], device)

    gen.update({
        "search_status": "ok",
        "mean_search_reward": float(np.mean([r["reward"] for r in rewards])),
        "best_search_reward": float(rewards[best_idx]["reward"]),
        "mean_terminal_entropy_nats": float(np.mean([r["entropy_nats"] for r in rewards])),
        "best_path": list(best_path),
        "n_paths_evaluated": len(paths),
        "n_apply_failures": n_apply_failures,
        "perplexity_5tok": pplx,
    })

    # Cleanup
    del accumulator
    return gen


def run_greedy(model, tok, ex, device) -> dict:
    input_ids = chat_input_ids(tok, ex, device)
    gen = generate_answer(model, tok, input_ids)
    gen["perplexity_5tok"] = perplexity_of_sequence(model, tok, input_ids, gen["new_ids"], device)
    gen["search_status"] = "n/a"
    gen["mean_search_reward"] = float("nan")
    gen["best_path"] = None
    return gen


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_steering_dir(model, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    d_model = model.config.hidden_size
    v = rng.randn(d_model).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v


def per_item_seed(item_idx: int, condition: str, model_seed: int) -> int:
    h = hashlib.md5(f"{item_idx}:{condition}:{model_seed}".encode()).hexdigest()
    return int(h[:8], 16)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--n", type=int, default=N_ITEMS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true", help="Run only 5 items for smoke test")
    args = p.parse_args()

    if args.smoke:
        args.n = 5

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out = Path(args.out or f"scripts/_track_f_results_{args.model.replace('/', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    log.info("device=%s; loading %s", device, args.model)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map=device)
    model.eval()
    log.info("model loaded in %.1fs", time.time() - t0)

    steering_dir = make_steering_dir(model, args.seed)
    log.info("steering direction: ||d||=%.4f, d_model=%d", np.linalg.norm(steering_dir), len(steering_dir))

    log.info("loading ARC-Easy first %d items", args.n)
    items = load_arc_easy(args.n)
    log.info("loaded %d items", len(items))

    results = {
        "model_id": args.model,
        "device": device,
        "n_items": len(items),
        "config": {
            "alpha_set": list(ALPHA_SET),
            "depth": DEPTH,
            "n_nodes": N_NODES,
            "max_new_tokens": MAX_NEW,
            "seed": args.seed,
        },
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if device == "cuda" else None,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "items": [],
    }

    t_run = time.time()
    for i, ex in enumerate(items):
        t_item = time.time()
        # GREEDY
        g_rec = run_greedy(model, tok, ex, device)
        g_rec["correct"] = int(g_rec["pred_letter"] == ex["gold_letter"])

        # RANDOM-MCTS
        r_rng = random.Random(per_item_seed(i, "R", args.seed))
        r_rec = run_search(model, tok, ex, device,
                            reward_mode="random", steering_dir=steering_dir, rng=r_rng)
        r_rec["correct"] = int(r_rec["pred_letter"] == ex["gold_letter"])

        # ENTROPY-MCTS
        e_rng = random.Random(per_item_seed(i, "E", args.seed))
        e_rec = run_search(model, tok, ex, device,
                            reward_mode="entropy", steering_dir=steering_dir, rng=e_rng)
        e_rec["correct"] = int(e_rec["pred_letter"] == ex["gold_letter"])

        item_secs = time.time() - t_item
        rec = {
            "idx": i,
            "source_id": ex.get("source_id", ""),
            "gold_letter": ex["gold_letter"],
            "n_options": len(ex["labels"]),
            "G": g_rec,
            "R": r_rec,
            "E": e_rec,
            "wall_seconds": item_secs,
        }
        results["items"].append(rec)

        if (i + 1) % 10 == 0 or i < 5 or args.smoke:
            G_acc = sum(it["G"]["correct"] for it in results["items"]) / len(results["items"])
            R_acc = sum(it["R"]["correct"] for it in results["items"]) / len(results["items"])
            E_acc = sum(it["E"]["correct"] for it in results["items"]) / len(results["items"])
            log.info(
                "item %d/%d (%.1fs)  G=%s R=%s E=%s gold=%s | running acc G=%.3f R=%.3f E=%.3f",
                i + 1, len(items), item_secs,
                g_rec["pred_letter"], r_rec["pred_letter"], e_rec["pred_letter"], ex["gold_letter"],
                G_acc, R_acc, E_acc,
            )

        # Persist after every 25 items in case of interruption
        if (i + 1) % 25 == 0 or i == len(items) - 1:
            with open(out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)

    total_secs = time.time() - t_run
    results["wallclock_seconds"] = total_secs

    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("done in %.1fs (avg %.2fs/item) → %s", total_secs, total_secs / max(len(items), 1), out)

    # Final summary
    G_acc = sum(it["G"]["correct"] for it in results["items"]) / len(results["items"])
    R_acc = sum(it["R"]["correct"] for it in results["items"]) / len(results["items"])
    E_acc = sum(it["E"]["correct"] for it in results["items"]) / len(results["items"])
    print(f"\nSUMMARY ({args.model}, n={len(items)}):")
    print(f"  G (greedy)        accuracy = {G_acc:.3f}")
    print(f"  R (random-MCTS)   accuracy = {R_acc:.3f}")
    print(f"  E (entropy-MCTS)  accuracy = {E_acc:.3f}")
    print(f"  delta(E - R) = {E_acc - R_acc:+.3f}")
    print(f"  delta(E - G) = {E_acc - G_acc:+.3f}")


if __name__ == "__main__":
    main()
