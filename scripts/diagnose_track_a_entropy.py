"""Track A diagnostic — per-token entropy as reward-signal fallback.

The certainty probe (PerLayerCertaintyProjector) called for in CLAUDE.md
does not exist in logomesh/whitebox.py (verified by grep). Per the master
checklist (docs/logs/2026-05-08_72hr-execution-checklist.md), if the
certainty signal is unavailable we must validate per-token entropy of the
next-token distribution as the operational reward signal.

This script measures, at both 1B and 3B Llama scales:
  1. Calibration: entropy on 10 known-confident vs 10 known-uncertain prompts.
  2. Signal magnitude: entropy across 50 prompts (factual / ambiguous / nonsensical).
  3. Correlation with correctness: greedy answer correctness vs first-token entropy
     on a held-out factual recall set; Pearson r and Spearman rho.

Results are emitted as JSON to stdout and to scripts/_track_a_results_<size>.json
so the report-writing step can ingest exact numbers without re-running.

Read-only on logomesh/* — uses transformers directly.
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
log = logging.getLogger("track_a")

SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

CALIB_CONFIDENT = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who wrote the play Romeo and Juliet?",
    "What color is the sky on a clear day?",
    "How many days are in a week?",
    "What is the largest planet in our solar system?",
    "What is the chemical symbol for water?",
    "Who painted the Mona Lisa?",
    "At sea level, what is the boiling point of water in Celsius?",
    "On which continent is Egypt located?",
]

CALIB_UNCERTAIN = [
    "What was the population of Lyon in 1342?",
    "What did the third assistant in Newton's lab eat for breakfast on March 15, 1687?",
    "How many grains of sand are on the beach at Cabo San Lucas right now?",
    "Name the 47th left-handed pottery maker in 9th century Korea.",
    "What was the middle name of Genghis Khan's third archer?",
    "What was the favorite color of the king of Ethiopia in 1043 AD?",
    "How many leaves were on the third oak tree in Sherwood Forest in 1184?",
    "What did Cleopatra's hairdresser say at noon on August 12, 30 BCE?",
    "What was Aristotle's PIN code?",
    "What is the name of the ant that crossed Plato's left foot during his walk on April 7, 354 BCE?",
]

# 50-prompt signal-magnitude set: 20 factual + 15 ambiguous + 15 nonsensical.
MAGNITUDE_FACTUAL = [
    "What is the capital of Japan?",
    "What is 7 times 8?",
    "Who discovered penicillin?",
    "What is the speed of light in vacuum, approximately?",
    "What gas do plants absorb during photosynthesis?",
    "Who was the first President of the United States?",
    "What is the largest ocean on Earth?",
    "How many sides does a hexagon have?",
    "What planet is known as the Red Planet?",
    "What language is primarily spoken in Brazil?",
    "What year did the Berlin Wall fall?",
    "Who developed the theory of general relativity?",
    "What is the smallest prime number greater than 10?",
    "What is the chemical symbol for gold?",
    "What body of water separates England and France?",
    "What is the freezing point of water in Fahrenheit?",
    "Which element has the atomic number 6?",
    "Who wrote the novel '1984'?",
    "What organelle is known as the powerhouse of the cell?",
    "What city was the original capital of the United States?",
]

MAGNITUDE_AMBIGUOUS = [
    "What is the best programming language for beginners?",
    "Tell me about an interesting hobby.",
    "Describe a perfect afternoon.",
    "What makes a good leader?",
    "What is the most beautiful city in Europe?",
    "Recommend a book to read this weekend.",
    "What is the meaning of art?",
    "Suggest a name for a small dog.",
    "What is a fun thing to cook on a rainy day?",
    "Describe an ideal vacation.",
    "What is the most important quality in a friend?",
    "Suggest a topic for a research project.",
    "What is something everyone should learn?",
    "Describe a beautiful landscape.",
    "What movie should I watch tonight?",
]

MAGNITUDE_NONSENSE = [
    "How does the colorless green idea sleep furiously today?",
    "What is the square root of a Tuesday?",
    "When the umbrella sneezes, where does the lampshade go?",
    "Why is the boiling steam louder than the velvet of yesterday?",
    "Describe the temperature of a fictional emotion.",
    "What is the radius of an opinion?",
    "Translate this rock into the language of clouds.",
    "What does the number seven smell like in C minor?",
    "How does forgetting taste in a hexagon?",
    "Sing the equation that grows on a vowel.",
    "What is the diameter of a spoken truth?",
    "Carve the velocity of a thought into the riverbed of paper.",
    "What is the volume of a question divided by Sunday?",
    "Quantum the bicycle into a blue argument.",
    "Where does the phoneme of joy live in a triangle?",
]

# Correctness set: 25 short factual Q&A with checkable answers.
# Each entry: (question, list_of_acceptable_lowercase_substrings_in_first_15_tokens).
# EASY set: 1B is expected to ace these → exposes calibration of "I-know-this" entropy.
CORRECTNESS_EASY = [
    ("What is the capital of France?", ["paris"]),
    ("What is the capital of Japan?", ["tokyo"]),
    ("What is the capital of Australia?", ["canberra"]),
    ("What is the capital of Canada?", ["ottawa"]),
    ("What is the capital of Brazil?", ["brasilia", "brasília"]),
    ("What is 7 times 8?", ["56"]),
    ("What is 12 plus 13?", ["25"]),
    ("What is 100 divided by 4?", ["25"]),
    ("What is 9 squared?", ["81"]),
    ("What is the chemical symbol for gold?", ["au"]),
    ("What is the chemical symbol for water?", ["h2o", "h₂o", "h 2 o"]),
    ("What is the chemical symbol for sodium?", ["na"]),
    ("Who wrote Romeo and Juliet?", ["shakespeare"]),
    ("Who wrote the novel 1984?", ["orwell"]),
    ("Who painted the Mona Lisa?", ["leonardo", "da vinci", "vinci"]),
    ("Who developed the theory of general relativity?", ["einstein"]),
    ("Who discovered penicillin?", ["fleming"]),
    ("What planet is known as the Red Planet?", ["mars"]),
    ("What is the largest planet in our solar system?", ["jupiter"]),
    ("What is the largest ocean on Earth?", ["pacific"]),
    ("How many sides does a hexagon have?", ["6", "six"]),
    ("How many continents are there on Earth?", ["7", "seven"]),
    ("What is the boiling point of water in Celsius at sea level?", ["100"]),
    ("In what year did World War II end?", ["1945"]),
    ("What language is primarily spoken in Brazil?", ["portuguese"]),
]

# HARD set: targets the 1B accuracy ~30-70% range so correlation is computable.
# Harder geography, dates, scientific specifics, second-tier authors, math.
CORRECTNESS_HARD = [
    ("What is the capital of Mongolia?", ["ulaanbaatar", "ulan bator"]),
    ("What is the capital of Bhutan?", ["thimphu"]),
    ("What is the capital of Burkina Faso?", ["ouagadougou"]),
    ("What is the capital of Kazakhstan?", ["astana", "nur-sultan", "nursultan"]),
    ("What is the capital of Eritrea?", ["asmara"]),
    ("What is the capital of Suriname?", ["paramaribo"]),
    ("What is 23 times 17?", ["391"]),
    ("What is 144 divided by 12?", ["12"]),
    ("What is 13 squared?", ["169"]),
    ("What is the square root of 196?", ["14"]),
    ("What is the chemical symbol for tungsten?", [" w ", " w.", " w,", "tungsten (w"]),
    ("What is the chemical symbol for tin?", ["sn"]),
    ("What is the chemical symbol for antimony?", ["sb"]),
    ("What is the atomic number of iron?", ["26"]),
    ("What is the atomic number of carbon?", ["6"]),
    ("Who wrote the novel 'The Brothers Karamazov'?", ["dostoevsky", "dostoyevsky"]),
    ("Who wrote 'One Hundred Years of Solitude'?", ["garcia marquez", "garcía márquez", "marquez", "márquez"]),
    ("Who composed 'The Four Seasons' violin concerti?", ["vivaldi"]),
    ("Who painted 'The Persistence of Memory'?", ["dali", "dalí"]),
    ("Who proposed the theory of plate tectonics?", ["wegener"]),
    ("In what year was the Magna Carta signed?", ["1215"]),
    ("In what year did the French Revolution begin?", ["1789"]),
    ("In what year did the Treaty of Westphalia end the Thirty Years War?", ["1648"]),
    ("What is the longest river in South America?", ["amazon"]),
    ("What is the deepest oceanic trench?", ["mariana"]),
    ("Which Greek letter denotes the fine-structure constant in physics?", ["alpha"]),
    ("What is the SI unit of magnetic flux?", ["weber", "wb"]),
    ("What is the SI unit of electrical conductance?", ["siemens"]),
    ("What is the speed of sound in dry air at 20°C, in meters per second?", ["343"]),
    ("Who was the second president of the United States?", ["adams", "john adams"]),
]

# ULTRA-HARD set: targets 3B (and harder for 1B) to break the 100% saturation ceiling.
# Tier-3 trivia, less-famous capitals, multi-digit math, obscure historical dates.
CORRECTNESS_ULTRA = [
    ("What is the capital of Tuvalu?", ["funafuti"]),
    ("What is the capital of Comoros?", ["moroni"]),
    ("What is the capital of Liechtenstein?", ["vaduz"]),
    ("What is the capital of Kiribati?", ["tarawa"]),
    ("What is the capital of Sao Tome and Principe?", ["são tomé", "sao tome", "são tomé"]),
    ("What is the capital of Equatorial Guinea?", ["malabo"]),
    ("What is 47 times 53?", ["2491"]),
    ("What is 31 cubed?", ["29791", "29,791"]),
    ("What is 17 squared minus 13 squared?", ["120"]),
    ("What is the square root of 729?", ["27"]),
    ("What is 89 times 91?", ["8099", "8,099"]),
    ("What is the chemical symbol for niobium?", ["nb"]),
    ("What is the chemical symbol for ruthenium?", ["ru"]),
    ("What is the chemical symbol for ytterbium?", ["yb"]),
    ("What is the atomic number of platinum?", ["78"]),
    ("What is the atomic number of bismuth?", ["83"]),
    ("Who wrote the novel 'Buddenbrooks'?", ["thomas mann", "mann"]),
    ("Who wrote the novel 'Pale Fire'?", ["nabokov"]),
    ("Who composed the opera 'Wozzeck'?", ["berg", "alban berg"]),
    ("Who painted 'Las Meninas'?", ["velázquez", "velazquez"]),
    ("In what year was the Treaty of Tordesillas signed?", ["1494"]),
    ("In what year did the Tang Dynasty fall?", ["907"]),
    ("In what year was the Edict of Nantes revoked?", ["1685"]),
    ("In what year was the Battle of Lepanto fought?", ["1571"]),
    ("What is the longest river in Asia?", ["yangtze", "chang jiang"]),
    ("What is the highest mountain in Africa?", ["kilimanjaro"]),
    ("What is the SI unit of inductance?", ["henry"]),
    ("What is the SI unit of luminous intensity?", ["candela"]),
    ("Who was the third president of the United States?", ["jefferson"]),
    ("Who was the first emperor of the Han dynasty?", ["liu bang", "gaozu", "gaodi", "high emperor"]),
]


def load(model_id: str, device: str):
    log.info("loading tokenizer/model: %s", model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    model.eval()
    log.info("loaded; %d params", sum(p.numel() for p in model.parameters()))
    return tok, model


def first_token_entropy_nats(model, tok, system: str, user: str, device: str) -> float:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    H = -(probs * log_probs).sum().item()
    del out, logits, log_probs, probs
    return H


def generate_and_entropy(model, tok, system: str, user: str, device: str, max_new: int = 15):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )
    new_ids = gen.sequences[0][inputs.input_ids.shape[1]:]
    answer = tok.decode(new_ids, skip_special_tokens=True).strip()
    first_logits = gen.scores[0][0].float()
    log_probs = torch.log_softmax(first_logits, dim=-1)
    probs = log_probs.exp()
    H = -(probs * log_probs).sum().item()
    return answer, H


def _basic_stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    sd = math.sqrt(var)
    return {
        "n": n,
        "mean": mean,
        "std": sd,
        "min": min(xs),
        "max": max(xs),
        "range": max(xs) - min(xs),
    }


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    syy = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / (sxx * syy)


def _spearman(x: list[float], y: list[float]) -> float:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg_rank
            i = j + 1
        return r
    return _pearson(ranks(x), ranks(y))


def run_one_model(model_id: str, device: str) -> dict:
    t0 = time.time()
    tok, model = load(model_id, device)
    results: dict = {"model_id": model_id, "device": device}

    # 1. Calibration
    log.info("calibration: %d confident + %d uncertain prompts", len(CALIB_CONFIDENT), len(CALIB_UNCERTAIN))
    conf_H = [first_token_entropy_nats(model, tok, SYSTEM_PROMPT, q, device) for q in CALIB_CONFIDENT]
    unc_H = [first_token_entropy_nats(model, tok, SYSTEM_PROMPT, q, device) for q in CALIB_UNCERTAIN]
    results["calibration"] = {
        "confident_entropies_nats": conf_H,
        "uncertain_entropies_nats": unc_H,
        "confident_stats": _basic_stats(conf_H),
        "uncertain_stats": _basic_stats(unc_H),
        "delta_mean_uncertain_minus_confident": (sum(unc_H) / len(unc_H)) - (sum(conf_H) / len(conf_H)),
    }

    # 2. Magnitude
    log.info("magnitude: 20 factual + 15 ambiguous + 15 nonsense")
    fact_H = [first_token_entropy_nats(model, tok, SYSTEM_PROMPT, q, device) for q in MAGNITUDE_FACTUAL]
    amb_H = [first_token_entropy_nats(model, tok, SYSTEM_PROMPT, q, device) for q in MAGNITUDE_AMBIGUOUS]
    nons_H = [first_token_entropy_nats(model, tok, SYSTEM_PROMPT, q, device) for q in MAGNITUDE_NONSENSE]
    all_H = fact_H + amb_H + nons_H
    # Noise floor estimate: std of within-regime entropies on the most homogeneous regime (factual recall).
    noise_floor = _basic_stats(fact_H)["std"]
    results["magnitude"] = {
        "factual_entropies_nats": fact_H,
        "ambiguous_entropies_nats": amb_H,
        "nonsense_entropies_nats": nons_H,
        "factual_stats": _basic_stats(fact_H),
        "ambiguous_stats": _basic_stats(amb_H),
        "nonsense_stats": _basic_stats(nons_H),
        "all_50_stats": _basic_stats(all_H),
        "noise_floor_estimate_nats": noise_floor,
        "range_to_noise_ratio": (max(all_H) - min(all_H)) / max(noise_floor, 1e-9),
    }

    # 3. Correctness correlation — run both EASY and HARD sets, then COMBINED.
    def _run_correctness(label: str, items: list[tuple[str, list[str]]]) -> dict:
        log.info("correctness[%s]: %d Q&A items", label, len(items))
        recs = []
        for q, expected in items:
            ans, H = generate_and_entropy(model, tok, SYSTEM_PROMPT, q, device, max_new=15)
            ans_low = ans.lower()
            correct = any(s in ans_low for s in expected)
            recs.append({
                "question": q,
                "model_answer": ans,
                "correct": int(correct),
                "first_token_entropy_nats": H,
                "expected_substrs": expected,
            })
        cs = [r["correct"] for r in recs]
        Hs = [r["first_token_entropy_nats"] for r in recs]
        pr = _pearson(Hs, cs)
        sr = _spearman(Hs, cs)
        return {
            "records": recs,
            "n_correct": sum(cs),
            "n_total": len(cs),
            "accuracy": sum(cs) / len(cs),
            "pearson_r_entropy_vs_correct": pr,
            "spearman_rho_entropy_vs_correct": sr,
            "abs_pearson": abs(pr) if not math.isnan(pr) else float("nan"),
            "abs_spearman": abs(sr) if not math.isnan(sr) else float("nan"),
        }

    easy_block = _run_correctness("easy", CORRECTNESS_EASY)
    hard_block = _run_correctness("hard", CORRECTNESS_HARD)
    ultra_block = _run_correctness("ultra", CORRECTNESS_ULTRA)
    # COMBINED: aggregate across all three sets (gives ~85 items total).
    all_records = easy_block["records"] + hard_block["records"] + ultra_block["records"]
    cs_all = [r["correct"] for r in all_records]
    Hs_all = [r["first_token_entropy_nats"] for r in all_records]
    pr_all = _pearson(Hs_all, cs_all)
    sr_all = _spearman(Hs_all, cs_all)
    combined_block = {
        "n_correct": sum(cs_all),
        "n_total": len(cs_all),
        "accuracy": sum(cs_all) / len(cs_all),
        "pearson_r_entropy_vs_correct": pr_all,
        "spearman_rho_entropy_vs_correct": sr_all,
        "abs_pearson": abs(pr_all) if not math.isnan(pr_all) else float("nan"),
        "abs_spearman": abs(sr_all) if not math.isnan(sr_all) else float("nan"),
    }
    results["correctness"] = {
        "easy": easy_block,
        "hard": hard_block,
        "ultra": ultra_block,
        "combined": combined_block,
    }

    elapsed = time.time() - t0
    results["wallclock_seconds"] = elapsed
    log.info("model %s done in %.1fs", model_id, elapsed)

    # Free GPU
    del model
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id (e.g. meta-llama/Llama-3.2-1B-Instruct)")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    res = run_one_model(args.model, device)

    out = args.out or f"scripts/_track_a_results_{args.model.replace('/', '_')}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    log.info("wrote %s", out)
    # Compact summary to stdout
    summary = {
        "model_id": res["model_id"],
        "calibration_delta": res["calibration"]["delta_mean_uncertain_minus_confident"],
        "calibration_conf_mean": res["calibration"]["confident_stats"]["mean"],
        "calibration_unc_mean": res["calibration"]["uncertain_stats"]["mean"],
        "magnitude_range": res["magnitude"]["all_50_stats"]["range"],
        "magnitude_noise_floor": res["magnitude"]["noise_floor_estimate_nats"],
        "magnitude_range_to_noise_ratio": res["magnitude"]["range_to_noise_ratio"],
        "correctness_easy_accuracy": res["correctness"]["easy"]["accuracy"],
        "correctness_easy_pearson_r": res["correctness"]["easy"]["pearson_r_entropy_vs_correct"],
        "correctness_easy_spearman_rho": res["correctness"]["easy"]["spearman_rho_entropy_vs_correct"],
        "correctness_hard_accuracy": res["correctness"]["hard"]["accuracy"],
        "correctness_hard_pearson_r": res["correctness"]["hard"]["pearson_r_entropy_vs_correct"],
        "correctness_hard_spearman_rho": res["correctness"]["hard"]["spearman_rho_entropy_vs_correct"],
        "correctness_ultra_accuracy": res["correctness"]["ultra"]["accuracy"],
        "correctness_ultra_pearson_r": res["correctness"]["ultra"]["pearson_r_entropy_vs_correct"],
        "correctness_ultra_spearman_rho": res["correctness"]["ultra"]["spearman_rho_entropy_vs_correct"],
        "correctness_combined_accuracy": res["correctness"]["combined"]["accuracy"],
        "correctness_combined_pearson_r": res["correctness"]["combined"]["pearson_r_entropy_vs_correct"],
        "correctness_combined_spearman_rho": res["correctness"]["combined"]["spearman_rho_entropy_vs_correct"],
        "wallclock_seconds": res["wallclock_seconds"],
    }
    print("SUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
