"""Track C — Benchmark calibration sweep.

Per `docs/logs/2026-05-08_diagnostic-prompt-track-C.md`. Greedy decoding only,
no MCTS, no steering. Six candidate reasoning benchmarks at 1B and 3B Llama.

For each (benchmark, model):
  - Load up to N=100 examples (deterministic slice of the test/validation split).
  - Chat-template prompt the question with labeled options.
  - Greedy `model.generate(... output_scores=True, max_new_tokens=5)`.
  - First-non-whitespace generated token is the model's letter answer.
  - Compute: accuracy, first-token entropy in nats (the validated Track A
    reward signal), top-3 next-token candidates (for "near-miss" headroom).

For each (benchmark, model) we report:
  - accuracy
  - mean / median first-token entropy on correct vs incorrect
  - headroom_top_quartile: |wrong items with first-token entropy in top quartile
    of all-items entropy| / |total| — an UPPER BOUND on what perfect search
    over high-uncertainty wrongs could recover.
  - headroom_near_miss: |wrong items where the gold letter is in the top-3 of
    the next-token distribution| / |total| — a TIGHTER UPPER BOUND.
  - mcnemar_n_for_5pp: required paired n to detect a 5-point absolute
    accuracy improvement at p<0.05 power 0.8 (proportion-of-discordant-pairs
    formulation).

Read-only on `logomesh/*`. Outputs:
  - scripts/_track_c_results_<modelid>.json
  - one-line SUMMARY per benchmark to stdout.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("track_c")

SYSTEM_PROMPT = "You are a careful, concise reasoner. Answer multiple-choice questions with a single capital letter."

LETTERS = ["A", "B", "C", "D", "E"]
N_PER_BENCHMARK = 100
MAX_NEW = 5  # we only need the first letter, max 5 tokens guards against verbose prefixes


# ---------------------------------------------------------------------------
# Benchmark loaders.
# Each loader returns a list of dicts: {question, options (list of strings),
#                                       gold_letter ("A".."E"), source_id}
# Failure modes return [] and the caller marks the benchmark MISSING.
# ---------------------------------------------------------------------------

def _try_load(*names_and_configs):
    """Attempt several (name, config) pairs; return the first that loads."""
    from datasets import load_dataset
    for entry in names_and_configs:
        if isinstance(entry, str):
            args = (entry,)
            kwargs = {}
        else:
            args = entry[:1] if len(entry) == 1 else entry[:2]
            kwargs = entry[2] if len(entry) > 2 else {}
        try:
            ds = load_dataset(*args, **kwargs)
            log.info("  loaded: %s %s", args, kwargs)
            return ds
        except Exception as e:
            log.info("  failed: %s %s — %s", args, kwargs, e.__class__.__name__)
    return None


def load_arc_easy(n: int) -> list[dict]:
    ds = _try_load(("allenai/ai2_arc", "ARC-Easy"), ("ai2_arc", "ARC-Easy"))
    if ds is None:
        return []
    split = ds["test"] if "test" in ds else ds["validation"]
    out = []
    for ex in split.select(range(min(n, len(split)))):
        choices = ex["choices"]  # {"text": [...], "label": [...]}
        gold = ex["answerKey"].strip()
        # Some ARC items have label like "1"/"2"/... — normalise to letter
        if gold.isdigit():
            gold = LETTERS[int(gold) - 1]
        out.append({
            "question": ex["question"],
            "options": choices["text"],
            "labels": choices["label"],  # letters or digits as given in dataset
            "gold_letter": gold if gold in LETTERS else None,
            "source_id": ex.get("id", ""),
        })
    return [x for x in out if x["gold_letter"] is not None]


def load_sciq(n: int) -> list[dict]:
    ds = _try_load("sciq", "allenai/sciq")
    if ds is None:
        return []
    split = ds["test"] if "test" in ds else ds["validation"]
    out = []
    for ex in split.select(range(min(n, len(split)))):
        # SciQ has correct_answer + 3 distractors; we randomise to A/B/C/D
        opts = [ex["distractor1"], ex["distractor2"], ex["distractor3"], ex["correct_answer"]]
        # Deterministic shuffle keyed on the index within selection
        idx = ex.get("question", "")
        # Simple deterministic permutation: rotate by hash(question) mod 4
        rot = (hash(idx) % 4 + 4) % 4
        opts = opts[rot:] + opts[:rot]
        gold_pos = opts.index(ex["correct_answer"])
        out.append({
            "question": ex["question"],
            "options": opts,
            "labels": LETTERS[:4],
            "gold_letter": LETTERS[gold_pos],
            "source_id": "",
        })
    return out


def load_hellaswag(n: int) -> list[dict]:
    ds = _try_load("hellaswag", "Rowan/hellaswag")
    if ds is None:
        return []
    split = ds["validation"] if "validation" in ds else ds["test"]
    out = []
    for ex in split.select(range(min(n, len(split)))):
        ctx = (ex["activity_label"] + ": " + ex["ctx"]).strip()
        endings = ex["endings"]
        gold_idx = int(ex["label"]) if ex["label"] != "" else -1
        if gold_idx < 0:
            continue
        out.append({
            "question": "Choose the most plausible continuation:\n\n" + ctx,
            "options": endings,
            "labels": LETTERS[:4],
            "gold_letter": LETTERS[gold_idx],
            "source_id": ex.get("ind", ""),
        })
    return out


def load_aqua_rat(n: int) -> list[dict]:
    ds = _try_load(("aqua_rat", "raw"), "aqua_rat", "deepmind/aqua_rat")
    if ds is None:
        return []
    split = ds["test"] if "test" in ds else ds["validation"]
    out = []
    for ex in split.select(range(min(n, len(split)))):
        # AQuA "options" is a list of strings like "A)2" "B)3" etc.
        opts_raw = ex["options"]
        opts_text = []
        for o in opts_raw:
            # strip leading "A)", "B)", etc.
            stripped = o.split(")", 1)[-1].strip() if ")" in o else o
            opts_text.append(stripped)
        gold = ex["correct"].strip().upper()
        if gold not in LETTERS[:5]:
            continue
        out.append({
            "question": ex["question"],
            "options": opts_text,
            "labels": LETTERS[:len(opts_text)],
            "gold_letter": gold,
            "source_id": "",
        })
    return out


def load_proofwriter(n: int) -> list[dict]:
    """ProofWriter — logical entailment. `tasksource/proofwriter` has fields
    `theory`, `question`, `answer` (True/False). We map True/False to A/B."""
    ds = _try_load(
        ("tasksource/proofwriter",),
        ("renma/ProofWriter",),
    )
    if ds is None:
        return []
    split = ds["validation"] if "validation" in ds else (ds["test"] if "test" in ds else next(iter(ds.values())))
    out = []
    # Take a deterministic shuffled slice to balance answer labels (default split is heavily ordered)
    indices = list(range(0, min(len(split), n * 30), 30))[:n * 5]
    for ix in indices:
        if len(out) >= n:
            break
        ex = split[ix]
        q = ex.get("question") or ex.get("hypothesis") or ex.get("query") or ""
        ctx = ex.get("theory") or ex.get("context") or ex.get("premise") or ""
        ans = ex.get("answer") if "answer" in ex else ex.get("label")
        if isinstance(ans, bool):
            gold = "A" if ans else "B"
        elif isinstance(ans, str):
            a = ans.lower().strip()
            if a in {"true", "yes", "entailment"}:
                gold = "A"
            elif a in {"false", "no", "not_entailment", "non-entailment", "neutral", "contradiction", "unknown"}:
                gold = "B"
            else:
                continue
        else:
            continue
        if not q or not ctx:
            continue
        prompt = f"Context:\n{ctx}\n\nClaim: {q}\n\nIs the claim entailed by the context?"
        out.append({
            "question": prompt,
            "options": ["True (entailed)", "False (not entailed)"],
            "labels": ["A", "B"],
            "gold_letter": gold,
            "source_id": str(ex.get("id", "")),
        })
    return out


def load_logiqa(n: int) -> list[dict]:
    """LogiQA replacement: BBH `logical_deduction_three_objects` (BIG-Bench Hard).

    The canonical LogiQA HF mirrors all rely on legacy dataset scripts that
    were deprecated by HuggingFace `datasets` >= 4.0; none of `lucasmccabe/logiqa`,
    `EleutherAI/logiqa`, `lighteval/logiqa_harness`, `hails/logiqa`, or
    `baber/logiqa2` load on PyTorch 2.6 / datasets 5.x.

    BBH `logical_deduction_three_objects` is a well-known logical-reasoning MCQ
    benchmark (250 test items; Suzgun et al. 2022) and is a defensible substitute:
    same task family (multi-step deductive reasoning), same MCQ format, on a HF
    mirror that loads cleanly. Documented as a substitution in the report.

    The `input` field already includes the question, options, and an
    `Options:\\n(A) ...\\n(B) ...\\n(C) ...\\nA:` style suffix. The `target`
    field is a string like `(A)` — we strip the parens to get the gold letter.
    """
    ds = _try_load(("lukaemon/bbh", "logical_deduction_three_objects"))
    if ds is None:
        return []
    split = ds["test"]
    out = []
    for ex in split.select(range(min(n, len(split)))):
        target = (ex.get("target") or "").strip()
        gold = target.replace("(", "").replace(")", "").strip().upper()
        if gold not in LETTERS:
            continue
        # Input includes options inline; we present the input verbatim but reformat
        # the trailing "A:" prompt our own way for entropy-on-letter consistency.
        text = ex["input"].rstrip()
        # Strip trailing "A:" if present (BBH formatting)
        if text.endswith("A:"):
            text = text[:-2].rstrip()
        out.append({
            "question": text,
            "options": ["A", "B", "C"],   # placeholders — real options are inline in `text`
            "labels": ["A", "B", "C"],
            "gold_letter": gold,
            "source_id": "",
            "inline_options": True,  # signal to format_prompt: don't re-list options
        })
    return out


BENCHMARKS = [
    ("ARC-Easy", load_arc_easy),
    ("SciQ", load_sciq),
    ("HellaSwag", load_hellaswag),
    ("AQuA-RAT", load_aqua_rat),
    ("ProofWriter", load_proofwriter),
    ("LogiQA", load_logiqa),
]


# ---------------------------------------------------------------------------
# Inference + scoring
# ---------------------------------------------------------------------------

def format_prompt(ex: dict) -> str:
    """Render the MCQ as 'Question + lettered options + Answer:' for the chat template.

    If ex["inline_options"] is True (e.g. BBH), the question already contains
    its own option listing; we skip our own listing to avoid double-printing.
    """
    if ex.get("inline_options"):
        lines = [ex["question"], "", f"Answer with the single capital letter ({'/'.join(ex['labels'])}):"]
    else:
        lines = [ex["question"], ""]
        for letter, opt in zip(ex["labels"], ex["options"]):
            lines.append(f"{letter}. {opt}")
        lines.append("")
        lines.append(f"Answer with the single capital letter ({'/'.join(ex['labels'])}):")
    return "\n".join(lines)


def score_one(model, tok, ex: dict, device: str) -> dict:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_prompt(ex)},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    new_ids = gen.sequences[0][inputs.input_ids.shape[1]:]
    answer_text = tok.decode(new_ids, skip_special_tokens=True).strip()
    # Extract the first letter A-E from the model's answer
    pred_letter = None
    for ch in answer_text:
        cu = ch.upper()
        if cu in ex["labels"]:
            pred_letter = cu
            break

    # Entropy on the FIRST generated token (validated Track A signal)
    first_logits = gen.scores[0][0].float()
    log_probs = torch.log_softmax(first_logits, dim=-1)
    probs = log_probs.exp()
    entropy_nats = -(probs * log_probs).sum().item()

    # Top-3 letters by next-token probability for "near-miss" headroom
    # Tokenize each candidate letter and rank by mass
    letter_probs = {}
    for L in ex["labels"]:
        # Try both " A" and "A" — chat templates often have a leading space
        ids_with_space = tok.encode(f" {L}", add_special_tokens=False)
        ids_no_space = tok.encode(L, add_special_tokens=False)
        p_max = 0.0
        for ids in (ids_with_space, ids_no_space):
            if ids:
                p_max = max(p_max, float(probs[ids[0]].item()))
        letter_probs[L] = p_max
    ranked = sorted(letter_probs.items(), key=lambda kv: kv[1], reverse=True)
    top3 = [r[0] for r in ranked[:3]]
    gold_in_top3 = ex["gold_letter"] in top3
    gold_rank = next((i for i, (L, _) in enumerate(ranked) if L == ex["gold_letter"]), -1)

    return {
        "question_preview": ex["question"][:120],
        "options": ex["options"],
        "labels": ex["labels"],
        "gold_letter": ex["gold_letter"],
        "answer_text": answer_text,
        "pred_letter": pred_letter,
        "correct": int(pred_letter == ex["gold_letter"]),
        "first_token_entropy_nats": entropy_nats,
        "letter_probs": letter_probs,
        "top3_letters": top3,
        "gold_in_top3": int(gold_in_top3),
        "gold_rank_among_letters": gold_rank,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _mcnemar_n_for_delta(p_acc: float, delta: float = 0.05, alpha: float = 0.05, power: float = 0.8) -> int:
    """Approx paired-sample n to detect absolute accuracy improvement of `delta`
    at significance `alpha` and power `power`. Uses a coarse two-proportions
    z-test back-of-envelope (the McNemar exact form depends on the discordance
    rate which we don't know yet); reports an UPPER BOUND on n.
    Formula: n ≈ (z_{α/2} + z_{β})² · 2 · p̄(1−p̄) / Δ²
    where p̄ = p_acc + delta/2.
    """
    if delta <= 0 or p_acc < 0 or p_acc > 1:
        return 10**9
    from math import sqrt
    z_alpha2 = 1.959964
    z_beta = 0.841621
    pbar = max(0.0, min(1.0, p_acc + delta / 2))
    var = 2 * pbar * (1 - pbar)
    n = ((z_alpha2 + z_beta) ** 2) * var / (delta ** 2)
    return int(math.ceil(n))


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    accs = [r["correct"] for r in records]
    Hs = [r["first_token_entropy_nats"] for r in records]
    n = len(records)
    n_correct = sum(accs)
    acc = n_correct / n
    H_correct = [r["first_token_entropy_nats"] for r in records if r["correct"]]
    H_wrong = [r["first_token_entropy_nats"] for r in records if not r["correct"]]
    q75 = _quantile(Hs, 0.75)
    # Headroom upper bound: of wrong items, how many have entropy >= top-quartile threshold
    high_entropy_wrong = sum(1 for r in records if not r["correct"] and r["first_token_entropy_nats"] >= q75)
    # Headroom near-miss: of wrong items, how many had gold in top-3 next-token
    near_miss_wrong = sum(1 for r in records if not r["correct"] and r["gold_in_top3"])
    return {
        "n": n,
        "n_correct": n_correct,
        "accuracy": acc,
        "entropy_mean_nats": sum(Hs) / n,
        "entropy_correct_mean": (sum(H_correct) / len(H_correct)) if H_correct else float("nan"),
        "entropy_wrong_mean": (sum(H_wrong) / len(H_wrong)) if H_wrong else float("nan"),
        "entropy_q75": q75,
        "headroom_top_quartile_count": high_entropy_wrong,
        "headroom_top_quartile_frac": high_entropy_wrong / n,
        "headroom_near_miss_count": near_miss_wrong,
        "headroom_near_miss_frac": near_miss_wrong / n,
        "mcnemar_n_for_5pp_p05_pwr80": _mcnemar_n_for_delta(acc, delta=0.05),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_one_model(model_id: str, device: str, n_per: int, out_path: Path):
    log.info("loading model: %s", model_id)
    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()
    log.info("model loaded in %.1fs", time.time() - t_load)

    results = {"model_id": model_id, "device": device, "n_per_benchmark": n_per, "benchmarks": {}}

    for bm_name, loader in BENCHMARKS:
        log.info("=== benchmark: %s ===", bm_name)
        t0 = time.time()
        try:
            examples = loader(n_per)
        except Exception as e:
            log.error("%s loader crashed: %s", bm_name, e)
            results["benchmarks"][bm_name] = {"status": "MISSING", "reason": f"loader crash: {e!r}"}
            continue
        if not examples:
            log.warning("%s — no examples loaded; marking MISSING", bm_name)
            results["benchmarks"][bm_name] = {"status": "MISSING", "reason": "no examples loaded (loader returned [] or upstream load failed)"}
            continue
        log.info("%s — running %d examples", bm_name, len(examples))
        records = []
        for i, ex in enumerate(examples):
            try:
                rec = score_one(model, tok, ex, device)
                records.append(rec)
            except Exception as e:
                log.warning("  item %d failed: %s", i, e)
        agg = aggregate(records)
        agg["wall_seconds"] = time.time() - t0
        agg["records"] = records
        agg["status"] = "ok"
        results["benchmarks"][bm_name] = agg
        log.info(
            "%s → acc=%.3f n=%d  H̄=%.3f  H̄_wrong=%.3f  q75=%.3f  hr_q75=%.3f  hr_top3=%.3f  n_for_5pp=%d  (%.1fs)",
            bm_name, agg.get("accuracy", float("nan")), agg.get("n", 0),
            agg.get("entropy_mean_nats", float("nan")),
            agg.get("entropy_wrong_mean", float("nan")),
            agg.get("entropy_q75", float("nan")),
            agg.get("headroom_top_quartile_frac", float("nan")),
            agg.get("headroom_near_miss_frac", float("nan")),
            agg.get("mcnemar_n_for_5pp_p05_pwr80", 0),
            agg.get("wall_seconds", 0),
        )
        # Save partial after each benchmark so a crash doesn't lose data
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

    # Tear down
    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--n", type=int, default=N_PER_BENCHMARK)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    out = Path(args.out or f"scripts/_track_c_results_{args.model.replace('/', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("output: %s", out)

    run_one_model(args.model, device, args.n, out)
    log.info("done")


if __name__ == "__main__":
    main()
