"""Track G — Latent Cartography sweep across 4 prompt classes × 6 steering magnitudes.

Per `docs/logs/2026-05-10_latent-cartography-sweep-protocol.md`. Maps the
structural pre-training priors that emerge when Reversible KV-Cache MCTS
applies entropy-minimization steering pressure across diverse prompt classes.
Becomes §5 Experiment 2 of the paper, paired with Track F's negative-control
as Experiment 1.

Four prompt classes:
  C1 — Factual recall (hand-constructed 100-item set in `_track_g_c1_factual_recall.json`)
  C2 — Logical deduction (ARC-Easy test split, first 100; continuity with Track F)
  C3 — Moral/preference (TruthfulQA mc1 validation split, first 100, reformatted to 4-option MCQ)
  C4 — Creative completion (HellaSwag validation split, first 100, option list stripped)

Alpha grid: {0.0, 0.1, 0.5, 1.0, 2.0, 5.0}.
Per (item, alpha): apply 3 sequential steerings of magnitude alpha (cumulative
effective ~3α), measure top-5 next-token distribution at the leaf.

PROTOCOL DEVIATION FROM SWEEP PROTOCOL §6: the protocol description references
"best_path_alphas" and 27-path sampling. Cartography here uses a *pure
parameter sweep* per (item, alpha) — single 3-step trajectory at fixed alpha,
no path sampling. Rationale:
  (a) Track F already established that under entropy-minimization the search
      collapses to the heavy-alpha path 91/200 at 1B and the minimal-alpha
      path 101/200 at 3B; path-selection patterns are already characterized.
  (b) Cartography's job is structural mapping across prompt classes, not
      path-selection. The Cartography heatmap is the top-5 token distribution
      as a function of (class, alpha) — directly observable from a single
      trajectory per (item, alpha).
  (c) Pure sweep is ~4x faster (1 trajectory vs 27), reducing per-item time
      from ~5s to ~1s at 1B.

Success criteria (per sweep protocol §8) revised: criterion C-1 (best-path
alpha-tuple variation across classes) is N/A under pure sweep — dropped.
Criteria C-2 (terminal-token concentration varies by class) and C-3
(mechanism plausibility) still apply. Threshold lowered from "2 of 3 hold"
to "2 of 2 hold" (stricter).

Read-only on logomesh/*. Outputs:
  - scripts/_track_g_results_<modelid>.json (per-item records, persisted every 50)
  - one-line per-item progress to stdout
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
log = logging.getLogger("track_g")

SYSTEM_PROMPT_MCQ = "You are a careful, concise reasoner. Answer multiple-choice questions with a single capital letter."
SYSTEM_PROMPT_QA = "You are a careful, concise reasoner. Answer with a single short response."
LETTERS = ["A", "B", "C", "D", "E"]
N_ITEMS_PER_CLASS = 100
ALPHA_GRID = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)  # sweep-mode per-step magnitudes
DEPTH = 3
TOP_K = 5  # Cartography heatmap top-K token distribution

# MCTS mode (matches Track F protocol so cross-track comparison holds)
MCTS_ALPHA_SET = (0.1, 0.5, 1.0)
MCTS_N_NODES = 27

C1_FILE = _REPO / "scripts" / "_track_g_c1_factual_recall.json"


# ---------------------------------------------------------------------------
# Per-class loaders
# ---------------------------------------------------------------------------

def load_c1_factual_recall(n: int) -> list[dict]:
    with C1_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["items"][:n]
    out = []
    for it in items:
        out.append({
            "class": "C1",
            "item_idx": it["idx"],
            "subclass": it["subclass"],
            "prompt_user": f"Q: {it['question']} A:",
            "gold": it["gold"],
            "system_prompt": SYSTEM_PROMPT_QA,
            "eval_mode": "first_token_str",
            "source_id": f"C1_{it['idx']}_{it['subclass']}",
        })
    return out


def load_c2_arc_easy(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
    split = ds["test"]
    out = []
    for i, ex in enumerate(split.select(range(min(n, len(split))))):
        choices = ex["choices"]
        gold = ex["answerKey"].strip()
        if gold.isdigit():
            gold = LETTERS[int(gold) - 1]
        if gold not in LETTERS:
            continue
        lines = [ex["question"], ""]
        for letter, opt in zip(choices["label"], choices["text"]):
            lines.append(f"{letter}. {opt}")
        lines.append("")
        lines.append(f"Answer with the single capital letter ({'/'.join(choices['label'])}):")
        out.append({
            "class": "C2",
            "item_idx": i,
            "subclass": "logical_deduction",
            "prompt_user": "\n".join(lines),
            "gold": gold,
            "system_prompt": SYSTEM_PROMPT_MCQ,
            "eval_mode": "first_letter",
            "source_id": ex.get("id", f"C2_{i}"),
        })
    return out


def load_c3_truthful_qa(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice")
    split = ds["validation"]
    out = []
    for i, ex in enumerate(split.select(range(min(n, len(split))))):
        mc1 = ex["mc1_targets"]
        choices = mc1["choices"]
        labels = mc1["labels"]
        # mc1: exactly one correct choice (label=1)
        try:
            correct_idx = labels.index(1)
        except ValueError:
            continue
        correct_text = choices[correct_idx]
        distractors = [c for j, c in enumerate(choices) if labels[j] == 0]
        if len(distractors) < 3:
            continue
        # Place the correct answer at deterministic position via hash(question) mod 4
        target_pos = int(hashlib.md5(ex["question"].encode()).hexdigest(), 16) % 4
        four = list(distractors[:3])
        four.insert(target_pos, correct_text)
        # Render
        opts_letters = LETTERS[:4]
        lines = [ex["question"], ""]
        for letter, opt in zip(opts_letters, four):
            lines.append(f"{letter}. {opt}")
        lines.append("")
        lines.append(f"Answer with the single capital letter ({'/'.join(opts_letters)}):")
        out.append({
            "class": "C3",
            "item_idx": i,
            "subclass": "moral_preference",
            "prompt_user": "\n".join(lines),
            "gold": opts_letters[target_pos],
            "system_prompt": SYSTEM_PROMPT_MCQ,
            "eval_mode": "first_letter",
            "source_id": f"C3_{i}",
        })
    return out


def load_c4_hellaswag(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag")
    split = ds["validation"]
    out = []
    for i, ex in enumerate(split.select(range(min(n, len(split))))):
        ctx = ex.get("ctx") or (ex.get("ctx_a", "") + " " + ex.get("ctx_b", "")).strip()
        if not ctx:
            continue
        endings = ex.get("endings", [])
        label = ex.get("label", "")
        try:
            gold_idx = int(label) if label != "" else 0
        except (ValueError, TypeError):
            gold_idx = 0
        gold_text = endings[gold_idx] if 0 <= gold_idx < len(endings) else ""
        out.append({
            "class": "C4",
            "item_idx": i,
            "subclass": "creative_completion",
            "prompt_user": ctx,
            "gold": gold_text.strip(),
            "system_prompt": SYSTEM_PROMPT_QA,
            "eval_mode": "open_continuation",
            "source_id": ex.get("source_id", f"C4_{i}"),
        })
    return out


# ---------------------------------------------------------------------------
# Inference primitives
# ---------------------------------------------------------------------------

def chat_input_ids(tok, ex: dict, device: str) -> torch.Tensor:
    msgs = [
        {"role": "system", "content": ex["system_prompt"]},
        {"role": "user", "content": ex["prompt_user"]},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(prompt, return_tensors="pt").input_ids.to(device)


def next_token_distribution(model, tok, past_kv, last_token_id: int, device: str, top_k: int) -> dict:
    """One forward step under the (possibly steered) cache; return top-k token distribution + entropy."""
    cur_ids = torch.tensor([[last_token_id]], device=device)
    with torch.no_grad():
        step = model(
            input_ids=cur_ids,
            past_key_values=_kv_eval_cache(past_kv),
            use_cache=True,
        )
    logits = step.logits[0, -1, :].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    H = -(probs * log_probs).sum().item()
    # Top-k
    top_p, top_i = torch.topk(probs, k=top_k)
    top_tokens = []
    for p_val, idx in zip(top_p.tolist(), top_i.tolist()):
        try:
            tok_str = tok.decode([idx])
        except Exception:
            tok_str = f"<id={idx}>"
        top_tokens.append([tok_str, float(p_val)])
    # Argmax token (the model's "answer first token")
    argmax_id = int(top_i[0].item())
    argmax_str = top_tokens[0][0]
    return {
        "entropy_nats": H,
        "top5": top_tokens,
        "argmax_token_str": argmax_str,
        "argmax_token_id": argmax_id,
    }


def gold_first_token_id(tok, gold_text: str) -> int | None:
    """Tokenize gold_text and return the first token id (the one the model 'should' produce first)."""
    if not gold_text:
        return None
    # Use add_special_tokens=False so we get just the content tokens.
    ids = tok(gold_text, add_special_tokens=False).input_ids
    if not ids:
        return None
    return int(ids[0])


def gold_surprisal(top5_with_full_dist, tok, gold_text: str) -> float | None:
    """Compute surprisal -log p(gold_first_token | steered cache). Returns None if gold is empty."""
    # This requires the full distribution, not just top-5. We can compute from the top5 if gold is in top5,
    # or fall back to None otherwise. For C4 we'll compute it from a separate forward pass — see eval_item.
    raise NotImplementedError("Use eval_item's full-distribution computation")


# ---------------------------------------------------------------------------
# Cartography sweep loop (per item, per alpha — pure parameter sweep)
# ---------------------------------------------------------------------------

def sweep_item_one_alpha(
    model, tok, ex, device,
    *, alpha: float, steering_dir: np.ndarray, top_k: int,
) -> dict:
    """Apply 3 sequential steerings of magnitude alpha; record top-k token distribution at the leaf.

    For alpha == 0.0: skip steering entirely; measure baseline distribution.
    """
    input_ids = chat_input_ids(tok, ex, device)
    # Build base KV cache via prefill
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past_kv = out.past_key_values
    last_token_id = int(input_ids[0, -1].item())

    if alpha == 0.0:
        # Baseline measurement — no steering applied.
        dist = next_token_distribution(model, tok, past_kv, last_token_id, device, top_k)
        # Also compute gold first-token surprisal for evaluation context
        gold_id = gold_first_token_id(tok, ex.get("gold", ""))
        gold_rank = None
        gold_p = None
        if gold_id is not None:
            # Rerun to get full distribution and find rank of gold
            with torch.no_grad():
                step = model(
                    input_ids=torch.tensor([[last_token_id]], device=device),
                    past_key_values=_kv_eval_cache(past_kv),
                    use_cache=True,
                )
            logits = step.logits[0, -1, :].float()
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_ids = torch.argsort(log_probs, descending=True)
            # rank
            try:
                gold_rank = int((sorted_ids == gold_id).nonzero(as_tuple=False)[0, 0].item())
                gold_p = float(log_probs[gold_id].exp().item())
            except IndexError:
                gold_rank = None
                gold_p = None
        return {
            "alpha": 0.0,
            "applied_steps": 0,
            **dist,
            "gold_first_token_rank": gold_rank,
            "gold_first_token_prob": gold_p,
        }

    # alpha > 0.0: apply 3 sequential steerings of magnitude alpha.
    accumulator = FP32Accumulator.from_kv_cache(past_kv, model=model)
    n_layers = len(_extract_kv_tensors(past_kv))
    dk_per_layer = [steering_dir.astype(np.float32)] * n_layers

    applied = 0
    ok = True
    for _ in range(DEPTH):
        if not accumulator.apply(past_kv, alpha, dk_per_layer):
            ok = False
            break
        applied += 1

    # Capture distribution at the steered leaf (whether or not full depth applied)
    dist = next_token_distribution(model, tok, past_kv, last_token_id, device, top_k)

    # Gold first-token rank under the steered cache
    gold_id = gold_first_token_id(tok, ex.get("gold", ""))
    gold_rank = None
    gold_p = None
    if gold_id is not None:
        with torch.no_grad():
            step = model(
                input_ids=torch.tensor([[last_token_id]], device=device),
                past_key_values=_kv_eval_cache(past_kv),
                use_cache=True,
            )
        logits = step.logits[0, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        sorted_ids = torch.argsort(log_probs, descending=True)
        try:
            gold_rank = int((sorted_ids == gold_id).nonzero(as_tuple=False)[0, 0].item())
            gold_p = float(log_probs[gold_id].exp().item())
        except IndexError:
            gold_rank = None
            gold_p = None

    # Roll back to baseline (in reverse order, same magnitude)
    for _ in range(applied):
        accumulator.rollback(past_kv, alpha, dk_per_layer)

    # Cleanup
    del accumulator

    return {
        "alpha": alpha,
        "applied_steps": applied,
        "applied_ok": ok,
        **dist,
        "gold_first_token_rank": gold_rank,
        "gold_first_token_prob": gold_p,
    }


# ---------------------------------------------------------------------------
# MCTS mode — Track F-style path-sampling search with entropy reward
# ---------------------------------------------------------------------------

def _entropy_of_logits(logits: torch.Tensor) -> tuple[float, float]:
    """Return (entropy_nats, normalised_certainty_reward = -H/log|V|)."""
    l = logits.float()
    lp = torch.log_softmax(l, dim=-1)
    p = lp.exp()
    H = -(p * lp).sum().item()
    V = l.shape[-1]
    return H, -H / math.log(V)


def mcts_item(
    model, tok, ex, device,
    *, steering_dir: np.ndarray, alpha_set: tuple, n_nodes: int, depth: int,
    rng: random.Random, top_k: int,
) -> dict:
    """Track F-style depth-D path-sampling search with entropy reward.

    For each item:
      - Build accumulator once.
      - Sample n_nodes paths of length depth from alpha_set.
      - Apply each path step by step, evaluate entropy at the leaf, roll back.
      - Pick the best path by reward (max -H/log|V|), re-apply permanently.
      - Measure top-k token distribution + gold-first-token rank at the steered leaf.
    """
    input_ids = chat_input_ids(tok, ex, device)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past_kv = out.past_key_values
    last_token_id = int(input_ids[0, -1].item())

    accumulator = FP32Accumulator.from_kv_cache(past_kv, model=model)
    n_layers = len(_extract_kv_tensors(past_kv))
    dk_per_layer = [steering_dir.astype(np.float32)] * n_layers

    paths = []
    rewards = []
    n_apply_failures = 0

    for _ in range(n_nodes):
        path = [rng.choice(alpha_set) for _ in range(depth)]
        applied_alphas = []
        ok = True
        for a in path:
            if not accumulator.apply(past_kv, a, dk_per_layer):
                ok = False
                n_apply_failures += 1
                break
            applied_alphas.append(a)

        if not ok:
            for a in reversed(applied_alphas):
                accumulator.rollback(past_kv, a, dk_per_layer)
            continue

        # Rollout: one forward step under the steered cache to read entropy
        with torch.no_grad():
            step = model(
                input_ids=torch.tensor([[last_token_id]], device=device),
                past_key_values=_kv_eval_cache(past_kv),
                use_cache=True,
            )
        next_logits = step.logits[0, -1, :]
        H, neg_norm_H = _entropy_of_logits(next_logits)

        paths.append(path)
        rewards.append({"reward": neg_norm_H, "entropy_nats": H})

        for a in reversed(path):
            accumulator.rollback(past_kv, a, dk_per_layer)

    if not paths:
        del accumulator
        # Fallback: measure unsteered distribution
        dist = next_token_distribution(model, tok, past_kv, last_token_id, device, top_k)
        return {
            "search_status": "no-paths-applicable",
            "best_path": None,
            "n_paths_evaluated": 0,
            "n_apply_failures": n_apply_failures,
            "mean_search_reward": float("nan"),
            "best_search_reward": float("nan"),
            "mean_terminal_entropy_nats": float("nan"),
            **dist,
            "gold_first_token_rank": None,
            "gold_first_token_prob": None,
        }

    best_idx = int(np.argmax([r["reward"] for r in rewards]))
    best_path = paths[best_idx]

    # Re-apply best path PERMANENTLY
    for a in best_path:
        accumulator.apply(past_kv, a, dk_per_layer)

    # Measure at the (now-permanently-)steered leaf
    dist = next_token_distribution(model, tok, past_kv, last_token_id, device, top_k)

    # Gold-rank under the steered cache
    gold_id = gold_first_token_id(tok, ex.get("gold", ""))
    gold_rank = None
    gold_p = None
    if gold_id is not None:
        with torch.no_grad():
            step = model(
                input_ids=torch.tensor([[last_token_id]], device=device),
                past_key_values=_kv_eval_cache(past_kv),
                use_cache=True,
            )
        logits = step.logits[0, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        sorted_ids = torch.argsort(log_probs, descending=True)
        try:
            gold_rank = int((sorted_ids == gold_id).nonzero(as_tuple=False)[0, 0].item())
            gold_p = float(log_probs[gold_id].exp().item())
        except IndexError:
            gold_rank = None
            gold_p = None

    del accumulator

    return {
        "search_status": "ok",
        "best_path": list(best_path),
        "n_paths_evaluated": len(paths),
        "n_apply_failures": n_apply_failures,
        "mean_search_reward": float(np.mean([r["reward"] for r in rewards])),
        "best_search_reward": float(rewards[best_idx]["reward"]),
        "mean_terminal_entropy_nats": float(np.mean([r["entropy_nats"] for r in rewards])),
        "best_terminal_entropy_nats": float(rewards[best_idx]["entropy_nats"]),
        **dist,
        "gold_first_token_rank": gold_rank,
        "gold_first_token_prob": gold_p,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_steering_dir(model, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    d_model = model.config.hidden_size
    v = rng.randn(d_model).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--n", type=int, default=N_ITEMS_PER_CLASS, help="Items per class")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true", help="Run only 5 items per class for smoke test")
    p.add_argument("--classes", default="C1,C2,C3,C4", help="Comma-separated subset of {C1,C2,C3,C4}")
    p.add_argument("--mode", default="sweep", choices=["sweep", "mcts"],
                   help="sweep = fixed-magnitude parameter sweep across alpha grid (ablation); "
                        "mcts = Track F-style path-sampling with entropy reward (primary, matches Hypothesis 2 framing)")
    args = p.parse_args()

    if args.smoke:
        args.n = 5
        alpha_grid = (0.0, 1.0, 5.0)
    else:
        alpha_grid = ALPHA_GRID

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    model_safe = args.model.replace("/", "_")
    mode_tag = "_mcts" if args.mode == "mcts" else ""
    out_path = Path(args.out) if args.out else _REPO / "scripts" / f"_track_g{mode_tag}_results_{model_safe}.json"

    log.info("Loading tokenizer + model: %s on %s", args.model, device)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
    model.eval()

    # Steering direction shared across all items and classes.
    steering_dir = make_steering_dir(model, args.seed)
    log.info("Steering direction: random unit vector in d_model=%d, seed=%d", model.config.hidden_size, args.seed)

    classes_requested = [c.strip() for c in args.classes.split(",") if c.strip()]
    loaders = {
        "C1": load_c1_factual_recall,
        "C2": load_c2_arc_easy,
        "C3": load_c3_truthful_qa,
        "C4": load_c4_hellaswag,
    }
    if not all(c in loaders for c in classes_requested):
        raise ValueError(f"unknown class in {classes_requested}; valid: {list(loaders.keys())}")

    log.info("Classes to run: %s; alpha grid: %s; items per class: %d", classes_requested, alpha_grid, args.n)

    all_items: list[dict] = []
    for cls in classes_requested:
        log.info("Loading class %s ...", cls)
        items = loaders[cls](args.n)
        log.info("  %s: loaded %d items", cls, len(items))
        all_items.extend(items)

    if args.mode == "sweep":
        total_evals = len(all_items) * len(alpha_grid)
        log.info("MODE=sweep — items×alphas: %d × %d = %d evals", len(all_items), len(alpha_grid), total_evals)
    else:
        total_evals = len(all_items)
        log.info("MODE=mcts — items: %d (alpha_set=%s, n_nodes=%d, depth=%d)",
                 len(all_items), MCTS_ALPHA_SET, MCTS_N_NODES, DEPTH)

    records = []
    t0 = time.time()
    eval_count = 0
    last_persist = 0
    mcts_rng = random.Random(args.seed)

    for ex in all_items:
        if args.mode == "mcts":
            # ONE record per item (best-path-applied measurement).
            t_item = time.time()
            try:
                res = mcts_item(
                    model, tok, ex, device,
                    steering_dir=steering_dir,
                    alpha_set=MCTS_ALPHA_SET, n_nodes=MCTS_N_NODES, depth=DEPTH,
                    rng=mcts_rng, top_k=TOP_K,
                )
                rec = {
                    "class": ex["class"],
                    "item_idx": ex["item_idx"],
                    "subclass": ex["subclass"],
                    "source_id": ex["source_id"],
                    "eval_mode": ex["eval_mode"],
                    "prompt_preview": ex["prompt_user"][:120].replace("\n", " "),
                    "gold": ex.get("gold", ""),
                    "search_status": res.get("search_status"),
                    "best_path": res.get("best_path"),
                    "n_paths_evaluated": res.get("n_paths_evaluated"),
                    "n_apply_failures": res.get("n_apply_failures"),
                    "mean_search_reward": res.get("mean_search_reward"),
                    "best_search_reward": res.get("best_search_reward"),
                    "mean_terminal_entropy_nats": res.get("mean_terminal_entropy_nats"),
                    "best_terminal_entropy_nats": res.get("best_terminal_entropy_nats"),
                    "argmax_token_str": res.get("argmax_token_str"),
                    "argmax_token_id": res.get("argmax_token_id"),
                    "next_token_entropy_nats": res.get("entropy_nats"),
                    "top5": res.get("top5"),
                    "gold_first_token_rank": res.get("gold_first_token_rank"),
                    "gold_first_token_prob": res.get("gold_first_token_prob"),
                    "wallclock_sec": time.time() - t_item,
                }
                records.append(rec)
            except Exception as e:
                log.exception("Item %s/%d MCTS failed: %s", ex["class"], ex["item_idx"], e)
                records.append({
                    "class": ex["class"],
                    "item_idx": ex["item_idx"],
                    "error": f"{type(e).__name__}: {e}",
                })
            eval_count += 1
            if eval_count % 10 == 0 or eval_count == total_evals:
                elapsed = time.time() - t0
                rate = eval_count / elapsed
                eta = (total_evals - eval_count) / rate if rate > 0 else 0
                log.info("Progress %d/%d (%.1f%%) — elapsed %.0fs — rate %.2f item/s — ETA %.0fs",
                         eval_count, total_evals, 100 * eval_count / total_evals, elapsed, rate, eta)
            if eval_count - last_persist >= 50:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump({
                        "model": args.model,
                        "mode": args.mode,
                        "seed": args.seed,
                        "alpha_set": list(MCTS_ALPHA_SET),
                        "n_nodes": MCTS_N_NODES,
                        "depth": DEPTH,
                        "n_items_per_class": args.n,
                        "classes": classes_requested,
                        "elapsed_sec": time.time() - t0,
                        "records": records,
                    }, f, indent=2)
                last_persist = eval_count
            continue

        # SWEEP mode (existing): for each item, iterate over the alpha grid.
        for alpha in alpha_grid:
            t_item = time.time()
            try:
                res = sweep_item_one_alpha(
                    model, tok, ex, device,
                    alpha=alpha, steering_dir=steering_dir, top_k=TOP_K,
                )
                rec = {
                    "class": ex["class"],
                    "item_idx": ex["item_idx"],
                    "subclass": ex["subclass"],
                    "source_id": ex["source_id"],
                    "eval_mode": ex["eval_mode"],
                    "alpha": alpha,
                    "prompt_preview": ex["prompt_user"][:120].replace("\n", " "),
                    "gold": ex.get("gold", ""),
                    "applied_steps": res.get("applied_steps", 0),
                    "applied_ok": res.get("applied_ok", True),
                    "argmax_token_str": res.get("argmax_token_str"),
                    "argmax_token_id": res.get("argmax_token_id"),
                    "next_token_entropy_nats": res.get("entropy_nats"),
                    "top5": res.get("top5"),
                    "gold_first_token_rank": res.get("gold_first_token_rank"),
                    "gold_first_token_prob": res.get("gold_first_token_prob"),
                    "wallclock_sec": time.time() - t_item,
                }
                records.append(rec)
            except Exception as e:
                log.exception("Item %s/%d alpha %.2f failed: %s", ex["class"], ex["item_idx"], alpha, e)
                records.append({
                    "class": ex["class"],
                    "item_idx": ex["item_idx"],
                    "alpha": alpha,
                    "error": f"{type(e).__name__}: {e}",
                })

            eval_count += 1
            if eval_count % 25 == 0 or eval_count == total_evals:
                elapsed = time.time() - t0
                rate = eval_count / elapsed
                eta = (total_evals - eval_count) / rate if rate > 0 else 0
                log.info("Progress %d/%d (%.1f%%) — elapsed %.0fs — rate %.2f eval/s — ETA %.0fs",
                         eval_count, total_evals, 100 * eval_count / total_evals, elapsed, rate, eta)

            # Persist every 50 evals
            if eval_count - last_persist >= 50:
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump({
                        "model": args.model,
                        "seed": args.seed,
                        "alpha_grid": list(alpha_grid),
                        "n_items_per_class": args.n,
                        "classes": classes_requested,
                        "elapsed_sec": time.time() - t0,
                        "records": records,
                    }, f, indent=2)
                last_persist = eval_count

    # Final persist (mode-aware schema)
    payload = {
        "model": args.model,
        "mode": args.mode,
        "seed": args.seed,
        "n_items_per_class": args.n,
        "classes": classes_requested,
        "elapsed_sec": time.time() - t0,
        "records": records,
    }
    if args.mode == "sweep":
        payload["alpha_grid"] = list(alpha_grid)
    else:
        payload["alpha_set"] = list(MCTS_ALPHA_SET)
        payload["n_nodes"] = MCTS_N_NODES
        payload["depth"] = DEPTH
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    log.info("Done. Wrote %d records to %s. Wall: %.1fs", len(records), out_path, time.time() - t0)

    # Brief summary
    print("\n=== SUMMARY ===")
    print(f"Model: {args.model}")
    print(f"Wall: {time.time() - t0:.1f}s")
    print(f"Records: {len(records)}")
    for cls in classes_requested:
        cls_recs = [r for r in records if r.get("class") == cls and "error" not in r]
        print(f"  {cls}: {len(cls_recs)} records")


if __name__ == "__main__":
    sys.exit(main() or 0)
