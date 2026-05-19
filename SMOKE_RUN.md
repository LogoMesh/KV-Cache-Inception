# Smoke Run — Reproducing & Verifying the Paper's Headline Claims

**Audience:** New contributors who want to independently verify the paper's empirical claims before submission. Read this entire file before running anything.

**Premise of this document:** Treat every numerical claim in the paper as **unverified until you have personally regenerated it from raw model output**. The point of this smoke run is *not* "make the pipeline finish without crashing" — it is "confirm the headline numbers Tianyu and his student are about to put their names on are real, are reproducible, and match what's already cached in `docs/dataset/data/raw/`."

If any headline claim fails to reproduce within tolerance, **stop and flag it**. ARR submission is on May 25; rebuttal is harder than retraction.

---

## 0. Pre-flight: claim-vs-cached-JSON consistency

Before you spend any GPU time, the doc author cross-checked every headline numeric claim against the raw JSON cached in `docs/dataset/data/raw/`. Status at this doc's authoring time (2026-05-18):

| Paper claim | Status against cached JSON |
|---|---|
| Track F: n=200 paired items per scale | ✅ Verified |
| Track F: 1B Δ_{E−R} = −0.005, 3B Δ_{E−R} = +0.005 | ✅ Verified |
| Track F: McNemar p = 1.0 at both scales (n_discordant = 3) | ✅ Verified |
| Track F: 1B "A" prediction count = 185/200 | ✅ Verified (exact) |
| Track F: 3B minimal-magnitude steering on 51% of items | ✅ Verified (101/200 = 50.5%, rounds to 51%) |
| Track F: Perplexity ratio E/G = 1.348 (1B), 1.058 (3B) | ✅ Verified (paper uses *ratio of means*; if you compute mean of ratios you'll get 1.328 / 1.062 — both valid, but cite the paper's method) |
| Track G: 100 records per non-C3 class per scale; C3 = 85 | ✅ Verified |
| Track G: Mean-step ᾱ per (class, scale) — all 8 cells | ✅ Verified within rounding (max abs delta 0.005) |
| Track D: 3.04--3.06× M_KV constant factor (steady-state) | ✅ Verified (1B = 3.064, 3B = 3.037; identical across all 3 (depth, branches) configs at each scale) |
| **Track D step-peak transient** | **⚠️ Not in headline.** Same JSON's `full_step_peak_mib_median` is 3,118 MiB at 1B depth=3, giving transient peak = (3118−2495)/125 = **4.98 × M_KV** during the in-place apply step. The paper headlines the steady-state 3.04× floor; the actual transient peak is ~5×. See §6.5. |
| **Track A: \|r\| ≈ 0.60 entropy↔correctness** | **⚠️ Partial / overstated.** Actual combined-bucket Pearson at 1B is −0.469 (\|r\|=0.47); at 3B is −0.508 (\|r\|=0.51). The 0.60 figure is true for *specific buckets* (1B hard, 3B ultra) but not for the combined 85-item correlation that the paper text implies. See §6.3. |

So if you only have time for two things: **verify Track A (overstated correlation) and run a depth=10 MCTS end-to-end to validate Track D's depth-independence claim empirically** — those are the two suspect-or-thinly-supported claims as of 2026-05-19. Everything else cross-checks against the cached JSON within tolerance. See §9 for the depth=10 plan.

Re-running the full sweep (next sections) tests whether the cached JSON itself is right — i.e., whether your fresh re-run regenerates these same numbers from raw model outputs. If your re-run agrees with the cached JSON, that confirms the cached data is real. If your re-run *disagrees*, that's a separate kind of red flag (either seed instability or cached-data tampering).

---

## 1. Hardware required

This codebase is **NVIDIA CUDA only**. Apple Silicon / MPS / CPU is not a supported execution path for the latent-search experiments. Specifically:

- `logomesh/kv_mcts.py` uses CUDA-specific tensor operations (KV-cache in-place mutation, FP32 accumulator on GPU)
- `transformers` model loads default to `device_map="auto"` which assumes CUDA
- `torch.cuda.memory_allocated()` is the primitive used by Track D (memory-bound measurement)

**Verified working configurations:**

| Setup | GPU | VRAM | Comments |
|---|---|---:|---|
| Consumer | RTX 3060 | 12 GB | What the original Track F/G runs used (driver 591.44; Win 10 / WSL) |
| Cloud (preferred for verification) | Lambda Cloud H100 / A100 | 40-80 GB | Faster; comfortable margin for full sweeps |
| Cloud (cheaper) | Lambda Cloud A10 / RTX 6000 | 24 GB | Adequate for 1B + 3B Llama 3.2 |

**Minimum VRAM:** ~12 GB. Llama 3.2-3B at fp16 is ~6 GB resident + KV cache + activations + accumulators. Running both 1B and 3B in the same Python session has been tested at 12 GB consumer scale.

**Disk:** Allow ~25 GB for the two HuggingFace model checkpoints + intermediate artifacts.

**Wall time budget** (rough, on RTX 3060):
- Setup (uv sync, HF model download): 15-30 min (download time dominates)
- Track A 1B + 3B: ~5 min
- Track C 1B + 3B: ~19 min
- Track D 1B + 3B: ~5 min (VRAM measurement only)
- Track F 1B sweep (200 items): ~18 min
- Track F 3B sweep (200 items): ~34 min
- Track G MCTS 1B + 3B (385 records each): ~47 min

For a full re-run: **~2-2.5 hours of GPU wall on consumer hardware**, ~30-60 min on H100.

For a fast smoke-only path: **~10 min** (Track A + Track F `--smoke` 5-item).

---

## 2. Hugging Face access (do this first; can take days)

### 2.1 Model choice is load-bearing for the verification

The two models the paper uses are:

- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`

**Every numerical claim in the paper is conditional on these specific checkpoints.** If you substitute another small open-weight model (e.g., TinyLlama-1.1B, Qwen 2.5-1.5B, Mistral-7B, SmolLM-1.7B) you can verify that:

- ✅ The pipeline runs end-to-end without crashing
- ✅ The FP32 accumulator reversibility holds (Theorem 1's zero-drift claim is architecture-independent)
- ✅ The memory bound is roughly N×M_KV for some scale-dependent N (the structural argument applies to any frozen Transformer)

But you **cannot** verify with substitutes:

- ❌ The specific Track F numbers (Δ_{E−R} = ±0.005, McNemar p = 1.0, 185/200 "A" predictions) — these depend on Llama 3.2's specific tokenizer prior and first-letter-prior structure
- ❌ The specific Track G ᾱ values per class (0.89/0.22 etc.) — class-conditional behavior is model-specific
- ❌ The 3.04-3.06× M_KV constant — the bf16/fp32 ratio is universal but the precise figure depends on K_proj/V_proj dimensions, which differ per architecture (Llama 3.2 uses grouped-query attention with 8 KV heads; many models differ)
- ❌ The Goodhart pathology direction-flip across 1B→3B — that's a scale-dependent property of *this* model family

**Bottom line:** if your goal is the paper-claim verification protocol in §6, you need meta-llama/Llama-3.2 access. A substitute-model run validates the algorithm but does not reproduce Table 1 or Table 2.

### 2.2 Getting access

The Llama 3.2 models are **gated** — Meta requires manual approval per HuggingFace account.

**Steps:**

1. Create a HuggingFace account: https://huggingface.co/join
2. Visit each model page and accept Meta's Llama 3.2 Community License:
   - https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
   - https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Wait for approval. Meta's manual review typically takes a few days to two weeks. There is no fast-track; plan accordingly.
4. Create an access token at https://huggingface.co/settings/tokens (read scope is sufficient).
5. Run `huggingface-cli login` and paste the token.

If you do not have Meta-approval yet and the smoke run is time-critical, ask Josh whether he can share an access token from an already-approved account. (Token sharing has security implications; the only legitimate uses are short-lived per-session inference, never CI or shared infrastructure.)

---

## 3. Environment setup

The repository uses `uv` for dependency management. The locked manifest is `uv.lock`.

```bash
# Install uv (https://docs.astral.sh/uv/) if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/LogoMesh/KV-Cache-Inception.git
cd KV-Cache-Inception

# Sync dependencies (creates a .venv with pinned versions)
uv sync
```

**Critical version pins from `uv.lock` (do not let `uv pip install` upgrade these):**

- Python 3.11
- PyTorch 2.6.0+cu124 (the `+cu124` tag is load-bearing — must match the CUDA toolkit on the system)
- transformers 5.3.0 (chat-template behavior in 5.x is reproducibility-critical; do not downgrade)
- accelerate 1.13.0 (the `device_map="auto"` flow in 1.x is what the drivers expect)
- numpy 2.3.3 (the legacy-`RandomState` steering-direction generation in `scripts/diagnose_track_f_negcontrol.py` requires numpy 2.x ABI)
- huggingface_hub ≥1.7.2

If your CUDA toolkit isn't 12.4, `uv sync` will likely succeed but at first GPU call PyTorch will raise. Install matching CUDA, or pin a different PyTorch wheel.

**Verify GPU is visible:**

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expect: True NVIDIA <whatever>
```

**Run the test suite (smoke check that imports work):**

```bash
uv run pytest tests/ -v
# Expect: 179 passed
```

Tests do not exercise model loading or CUDA — they verify the pure-Python logic of `kv_mcts.py`, `whitebox.py`, etc. A green test suite is necessary but not sufficient evidence that the pipeline reproduces.

---

## 4. The minimum smoke path (~10 min wall)

The goal here is "the pipeline executes end-to-end at one scale and produces output structurally identical to what's already in `docs/dataset/data/raw/`." It does NOT verify headline numbers — that's section 6.

```bash
# Track F at 1B with the --smoke flag (5 items instead of 200; ~3 min on RTX 3060)
uv run python scripts/diagnose_track_f_negcontrol.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --seed 42 \
    --smoke

# This writes scripts/_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json
# (overwriting any previous run at the same path — back it up first if you want
# to keep the cached one for comparison)
```

**Pass criteria:**
- Exit code 0
- JSON file written at the expected path
- Top-level keys present: `model_id`, `device`, `n_items` (= 5 under `--smoke`), `config`, `torch_version`, `cuda_version`, `gpu_name`, `items` (list of length 5), `wallclock_seconds`
- Each item in `items` has three sub-blocks: `G` (greedy), `R` (random-MCTS), `E` (entropy-MCTS); each sub-block has `pred_letter`, `correct`, etc.

If the smoke run crashes, common causes:
- HuggingFace gated-model rejection → check `huggingface-cli whoami` and license-accept status
- CUDA OOM → confirm 12 GB free VRAM; close other processes
- numpy `RandomState` AttributeError → numpy version mismatch (must be 2.x)

---

## 5. The full reproduction protocol

If the smoke run passes, run the full track sweep. Order matters mildly (Track A is cheapest; do it first to confirm setup with a small wall budget):

```bash
# Track A — static entropy↔correctness correlation (~5 min total)
uv run python scripts/diagnose_track_a_entropy.py \
    --model meta-llama/Llama-3.2-1B-Instruct --seed 42
uv run python scripts/diagnose_track_a_entropy.py \
    --model meta-llama/Llama-3.2-3B-Instruct --seed 42

# Track D — VRAM measurement (~5 min total; no model inference, just shape measurement)
uv run python scripts/measure_kv_mcts_vram.py \
    --model meta-llama/Llama-3.2-1B-Instruct --seed 42
uv run python scripts/measure_kv_mcts_vram.py \
    --model meta-llama/Llama-3.2-3B-Instruct --seed 42

# Track C — benchmark calibration (~19 min total)
uv run python scripts/diagnose_track_c_benchmarks.py \
    --model meta-llama/Llama-3.2-1B-Instruct --seed 42
uv run python scripts/diagnose_track_c_benchmarks.py \
    --model meta-llama/Llama-3.2-3B-Instruct --seed 42

# Track F — entropy-MCTS negative control (~52 min total — the most expensive)
uv run python scripts/diagnose_track_f_negcontrol.py \
    --model meta-llama/Llama-3.2-1B-Instruct --seed 42
uv run python scripts/diagnose_track_f_negcontrol.py \
    --model meta-llama/Llama-3.2-3B-Instruct --seed 42

# Track G MCTS — Latent Cartography (~47 min total)
uv run python scripts/diagnose_track_g_cartography.py \
    --model meta-llama/Llama-3.2-1B-Instruct --mode mcts --seed 42
uv run python scripts/diagnose_track_g_cartography.py \
    --model meta-llama/Llama-3.2-3B-Instruct --mode mcts --seed 42
```

All scripts default to `--n=200`, `--device=auto`, and write to `scripts/_track_<X>_results_<modelid>.json` next to the driver. Defaults are spelled out in each driver's `argparse` block.

**Outputs to keep:**
- The 13 JSON files under `scripts/_track_*.json` (the raw artifacts the paper cites)
- The driver stdout (often has the pairwise McNemar p-values printed inline at end-of-run)

---

## 6. Verification protocol — assume the paper is wrong until proven otherwise

This is the actual point of the smoke run. The headline claims below are taken verbatim from the paper at `docs/paper/kv-mcts-dimensional-escape.tex` (canonical) and `docs/paper/05-18-2026_review/Joshua_Hickson___ARR___EMNLP__Copy_/sections/` (modular restructure — same content). **Each claim either reproduces from your re-run or it doesn't. There is no middle ground.**

For each claim, the cross-reference is to: (a) the section in the paper, (b) the source JSON in `docs/dataset/data/raw/` that the paper cites, (c) a re-derivation command. If your re-run produces a value materially different from the cached JSON's value, **that itself is also a red flag** — it means either the seed isn't deterministic or the cached JSON was tampered with.

### 6.1 Track F headline (Experiment 1, §5 of the paper)

| Claim | Source (cached) | Re-derive |
|---|---|---|
| **n = 200 paired ARC-Easy items per scale** | `data/raw/_track_f_results_meta-llama_Llama-3.2-{1B,3B}-Instruct.json` → top-level `n_items` | `jq '.n_items' data/raw/_track_f_results_*.json` (in repo root: `docs/dataset/data/raw/`) |
| **1B Δ_{E−R} = −0.005 (E loses to random by 0.5pp)** | Same JSON → `items[].E.correct` mean − `items[].R.correct` mean | `python -c "import json; d=json.load(open('docs/dataset/data/raw/_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json')); E=sum(x['E']['correct'] for x in d['items'])/200; R=sum(x['R']['correct'] for x in d['items'])/200; print(f'E={E}, R={R}, Δ={E-R:+.3f}')"` |
| **3B Δ_{E−R} = +0.005** | Same with 3B | Same command, swap 1B→3B |
| **McNemar p = 1.0 at both scales (3 discordant pairs each)** | Same JSON, recompute paired McNemar | See §6.4 below for the McNemar re-derivation snippet |
| **1B entropy-MCTS predicts "A" on 185/200 items** | Same JSON → `items[].E.pred_letter == 'A'` count | `python -c "import json; d=json.load(open('docs/dataset/data/raw/_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json')); print(sum(1 for x in d['items'] if x['E']['pred_letter']=='A'))"` |
| **3B entropy-MCTS picks minimal-magnitude path on 51% of items** | Same JSON → 3B `items[].E.best_path == [0.1, 0.1, 0.1]` count | `python -c "import json; d=json.load(open('docs/dataset/data/raw/_track_f_results_meta-llama_Llama-3.2-3B-Instruct.json')); print(sum(1 for x in d['items'] if x['E']['best_path']==[0.1,0.1,0.1])/200)"` |
| **Perplexity ratio E/G = 1.348 (1B), 1.058 (3B)** | Same JSON → mean of `items[].E.perplexity_5tok / items[].G.perplexity_5tok` | similar pattern |

**Hallucination red flags:**
- Δ_{E−R} far from ±0.005 (say |Δ| > 0.05) means *either* the cached JSON is wrong or the search produces a real signal — both rewrite the paper
- "A" prediction count materially below 185/200 at 1B means the prior-amplification mechanism the paper builds Hypothesis 2 around is overstated
- McNemar p ≠ 1.0 (say p < 0.5) means the search reward actually differentiates from random — the paper's main negative result would collapse

### 6.2 Track G headline (Experiment 2, §5 of the paper)

| Claim | Source | Re-derive |
|---|---|---|
| **C1 (factual recall): 100 records per scale; mean-step ᾱ = 0.34 / 0.61** | `data/raw/_track_g_mcts_results_*.json` → `records[]` filtered by `class == 'C1'`; per-record `best_path` is 3-tuple, mean is `sum/3/n` | `python -c "import json; d=json.load(open('docs/dataset/data/raw/_track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json')); C1=[r for r in d['records'] if r['class']=='C1']; alpha=[sum(r['best_path'])/3 for r in C1]; print(f'n={len(C1)} mean_alpha={sum(alpha)/len(alpha):.2f}')"` |
| **C2 (ARC-Easy MCQ): ᾱ = 0.89 / 0.22 (1B/3B) — direction flip** | Same JSON filtered to `class == 'C2'` | Same pattern |
| **C3 (TruthfulQA MCQ): 85 records (not 100, due to <3-distractor filter)** | Same JSON filtered to `class == 'C3'`; `len()` should be 85 at both scales | `python -c "import json; d=json.load(open('docs/dataset/data/raw/_track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json')); print(len([r for r in d['records'] if r['class']=='C3']))"` |
| **C4 (HellaSwag continuation): ᾱ = 0.51 / 0.38** | Same with C4 | Same |
| **Cross-scale flip magnitude Δ ≈ 0.67 (C2), 0.55 (C3)** | Computed from above: ᾱ_1B − ᾱ_3B for C2, C3 | Subtraction |

**Hallucination red flags:**
- C3 record count ≠ 85 at either scale → paper's "TruthfulQA mc1 <3-distractor filter" claim is suspect
- ᾱ values for C2 or C3 that don't show the cross-scale direction flip → the paper's class-conditional Goodhart mechanism is overstated
- The whole Cartography (Experiment 2) story rests on the MCQ direction-flip; any cell where 1B and 3B agree directionally is a major problem

### 6.3 Track A headline (paper §5 Interpretation, §A.4)

| Claim | Source | Re-derive |
|---|---|---|
| **\|r\| ≈ 0.60 Pearson between next-token entropy and correctness** | `data/raw/_track_a_results_*.json` → `correctness.combined.pearson_r_entropy_vs_correct` | `python -c "import json; d=json.load(open('docs/dataset/data/raw/_track_a_results_meta-llama_Llama-3.2-1B-Instruct.json')); print(d['correctness']['combined']['pearson_r_entropy_vs_correct'])"` |
| **85 items: 25 easy + 30 hard + 30 ultra** | `data/raw/_track_a_results_*.json` → `correctness.{easy,hard,ultra}.n_total` | `python -c "import json; d=json.load(open('docs/dataset/data/raw/_track_a_results_meta-llama_Llama-3.2-1B-Instruct.json')); print({k: d['correctness'][k]['n_total'] for k in ['easy','hard','ultra']})"` |

**⚠️ Verified discrepancy (this doc's authoring time):** Per the raw JSON the paper cites, the actual combined-bucket Pearson values are:

| Bucket | 1B \|r\| | 3B \|r\| |
|---|---:|---:|
| easy (n=25) | NaN (acc=1.00 — no variance) | NaN |
| hard (n=30) | **0.602** | NaN (acc=1.00 at 3B) |
| ultra (n=30) | 0.357 | **0.614** |
| combined (n=85) | **0.469** | **0.508** |

The paper's "|r| ≈ 0.60" claim is true for *specific buckets* (1B hard, 3B ultra) but **NOT for the combined 85-item correlation, which is 0.47-0.51**. The paper's phrasing ("on fixed prompts of varying difficulty") implies the combined value — but the combined value is closer to 0.5 than 0.6. **Flag this to Josh / Tianyu for clarification** before submission: either the citation needs a bucket-specific qualifier ("|r| ≈ 0.60 on the hard bucket / ultra bucket") or the headline value needs adjusting to ~0.5.

**Hallucination red flag:** if your fresh re-run produces |r| far outside the 0.40-0.65 range, even the bucket-level claim collapses.

### 6.4 Theorem 1 reversibility (paper §4, §A appendix)

| Claim | Source | Re-derive |
|---|---|---|
| **Zero drift over 200 apply-revert cycles at machine precision** | Implementation in `logomesh/kv_mcts.py` (`FP32Accumulator`); test in `tests/test_phase2_modules.py` | `uv run pytest tests/test_phase2_modules.py -v -k "FP32" -k "rollback"` |

If any FP32-accumulator-related test fails, the paper's reversibility proof is empirically unsupported.

### 6.5 Track D memory bound (paper §6, Proposition 1)

| Claim | Source | Re-derive |
|---|---|---|
| **3.04 — 3.06× M_KV constant factor, independent of branching factor** | `data/raw/_track_d_vram*.json` → `configs[].delta_full_in_kv_units` | `python -c "import json; d=json.load(open('docs/dataset/data/raw/_track_d_vram.json')); [print(c['delta_full_in_kv_units']) for c in d['configs']]"` |
| **Three (depth, branches, nodes) configurations tested** | Same JSON → `configs[]` length should be 3 | `python -c "import json; d=json.load(open('docs/dataset/data/raw/_track_d_vram.json')); print(len(d['configs']))"` |

**✅ Verified consistent (this doc's authoring time):** Cached values:
- 1B: 3.06375 × M_KV across all three (depth, branches) configurations (3,3), (5,3), (10,3)
- 3B: 3.037 × M_KV across the same three configurations

The paper's "3.04--3.06×" range correctly spans the 3B (≈3.04) and 1B (≈3.06) values. Depth/branching-independence is exactly confirmed (same multiplier across all three configs at each scale).

**Hallucination red flag:** if your re-run shows the three configs producing different multipliers at the same scale (say 3.0, 4.5, 8.0), the depth/branching-independence claim is wrong.

#### How the 3.04× was actually measured

The driver is `scripts/measure_kv_mcts_vram.py`. Read the docstring at lines 1-32 for the canonical methodology statement; below is a paraphrase of what it does.

**Setup:** Load Llama-3.2-{1B,3B}-Instruct on a single GPU, run one forward pass on a 4000-token seed prompt to populate the KV cache, then allocate three different "memory states" in isolation and record each one's steady-state peak.

**Three modes per (depth, branches, nodes) config:**

| Mode | What's resident in VRAM |
|---|---|
| `baseline` | model weights + the KV cache from the warm-up forward pass (no MCTS structures) |
| `mid` | baseline + bf16 clones of K and V (naive "snapshot-before-perturb, subtract-after" rollback approach) |
| `full` | baseline + `FP32Accumulator.from_kv_cache(...)` — what Reversible KV-Cache MCTS actually holds per active search |

**Measurement primitive:** `torch.cuda.memory_allocated()` for the steady-state value, cross-checked against `nvidia-smi --query-gpu=memory.used` at the largest config to confirm PyTorch's allocator view matches the OS view. Three repeats per cell, median taken.

**Concrete numbers (1B, depth=3 config, from `data/raw/_track_d_vram.json` → `configs[0]`):**

```
kv_cache_mib_median         = 125.0 MiB     (the KV cache itself)
baseline_steady_mib_median  = 2,495 MiB     (model + KV)
full_steady_mib_median      = 2,878 MiB     (model + KV + FP32 accumulator)
delta_full_over_baseline    = 2,878 − 2,495 = 383 MiB
delta_full_in_kv_units      = 383 / 125 = 3.064
```

So **3.04× M_KV** is literally `(peak memory with FP32 accumulator − baseline memory) / (KV cache size)`. The 1B value rounds to 3.06; 3B rounds to 3.04 (mild difference because some constant model-side overhead is amortized differently at the 3B size).

**Why the multiplier is 3-ish, structurally:** the FP32 accumulator stores 4 tensors alongside the bf16 KV cache (verified in `k_base_mib_median`, `v_base_mib_median`, `k_accum_mib_median`, `v_accum_mib_median` fields):

| Tensor | Dtype | Size for 1B/4000-token cache |
|---|---|---:|
| K_base | bf16 | 62.5 MiB |
| V_base | bf16 | 62.5 MiB |
| K_accum | **fp32** | 125 MiB |
| V_accum | **fp32** | 125 MiB |
| **Total accumulator overhead** | — | **375 MiB** ≈ 3 × M_KV |

The 3.04-3.06× factor is basically `2 × (1 + 0.5)` because fp32 accumulators are 2× the bf16 KV cache size, and the bf16 base-clones add another 0.5× each. The 0.04-0.06 extra is PyTorch allocator fragmentation/overhead.

**Why all three (depth, branches, nodes) configs show the same multiplier:** `delta_full_in_kv_units` is **identical** (3.06375 at 1B; 3.037 at 3B) across (d=3,b=3,n=27), (d=5,b=3,n=81), (d=10,b=3,n=1700). That's not a coincidence — it's the empirical demonstration of Proposition 1's depth-independence claim. The accumulator is allocated once at the search root and rollback restores in-place; there's no per-node accumulator state, so measured peak memory doesn't change with depth/branches.

**⚠️ The step-peak transient is materially higher than the headline.** The same JSON also captures `full_step_peak_mib_median` — peak memory during a single apply operation (the in-place `K_t ← K_t + α·d_K` op, which briefly allocates a temporary for `α·d_K` before the in-place add).

At 1B depth=3:
```
full_step_peak_mib_median  = 3,118 MiB
baseline_steady            = 2,495 MiB
step_peak − baseline       = 623 MiB
step_peak_in_kv_units      = 623 / 125 = 4.98 × M_KV
```

So the *actual peak memory during a step* is ~5 × M_KV, not 3.04 ×. The paper headlines the steady-state floor (3.04×). For hardware sizing, the relevant number is the step-peak (~5×), because that's what your GPU needs to hold for the transient. **Anyone sizing an H100 budget from the paper's 3.04× alone could OOM at the apply transient at large scale.**

This isn't necessarily wrong reporting — the steady-state ratio is what proves depth-independence, and the step-peak is bounded by a single layer's transient (also depth-independent). But the paper should arguably report both, and a skeptical reviewer may push on it.

**Caveats the smoke runner should know:**

1. **The measurement does NOT run an actual MCTS search.** The driver allocates the accumulator structures, measures, then tears down. The argument that "peak memory is independent of search size" is a structural/theoretical claim backed by the code organization — not an empirical observation of memory during a real 1700-node depth-10 search. A skeptical reviewer could push back. The driver docstring explicitly acknowledges this: "We DO NOT run the full ReversibleMCTS rollout (per prompt: 'memory measurement only')."

   **Why this isn't necessarily devastating:** the structural argument is sound. Reversible KV-Cache MCTS is built so the FP32Accumulator is allocated *once* at the search root, each expansion is an in-place add to the pre-allocated K tensor, each rollback is an in-place subtract — no tensors are created during the search itself. If you can prove every operation is in-place on pre-allocated tensors, then peak memory IS the post-allocation steady state. Track D measures that post-allocation state for the real algorithm's actual data structures (it imports `FP32Accumulator` from the live `logomesh/kv_mcts.py`, the same class `ReversibleMCTS` uses, which is unit-tested in `tests/test_phase2_modules.py`). Additionally, Tracks F and G between them ran ~1,170 actual MCTS searches end-to-end without OOM on a 12 GB consumer GPU — weak-form empirical confirmation that the structural argument holds in practice at depth-3.

   **Why a reviewer might still push:** depth-independence is demonstrated structurally + by allocating three identical static states three times. The paper claims it holds at depth=10 with 1,700 nodes; that scale has never been empirically run end-to-end. PyTorch's allocator fragmentation behavior over 1,700 forward passes is untested. This is the strongest attack on Proposition 1's empirical surface. **See §9 for the recommended remedy: running one depth=10, 1,700-node search end-to-end.**

2. **The "162 GB at 20B" projection** the paper cites is **not measured** — it's extrapolated from the 1B/3B M_KV scaling (KV cache size ∝ #layers × hidden_dim × seq_len × 2 bytes for bf16), with the 3.04× factor then applied. Not measured directly because the 20B run doesn't fit on the RTX 3060 used for the empirical sweep.

3. **The "naive parallel-cache MCTS" comparison** (`O(b^d · M_KV)` baseline that the paper says Reversible-MCTS beats) is a **theoretical worst-case** — the driver does not actually run a naive parallel-cache implementation to measure its memory. The paper's claim is "if you naively branched the cache at every expansion, you'd need O(b^d · M_KV)" — geometrically obvious but not empirically demonstrated.

For verification purposes: the 3.04-3.06× number itself is real and reproducible (steady-state); the 162 GB / 60× claims that build on it are projections from that empirical anchor; and the step-peak ~5× is real and currently un-headlined.

### 6.6 The McNemar re-derivation snippet (for §6.1)

The paper's pairwise headline is "paired McNemar exact p = 1.0 (3 discordant pairs each scale)." Re-derive:

```python
import json
from scipy.stats import binom

for scale in ['1B', '3B']:
    d = json.load(open(f'docs/dataset/data/raw/_track_f_results_meta-llama_Llama-3.2-{scale}-Instruct.json'))
    # Build paired contingency: items where E and R disagree
    b = sum(1 for x in d['items'] if x['E']['correct'] == 1 and x['R']['correct'] == 0)
    c = sum(1 for x in d['items'] if x['E']['correct'] == 0 and x['R']['correct'] == 1)
    # McNemar exact (two-sided binomial on b vs b+c)
    n = b + c
    # Two-sided p: 2 * min(P(X<=b), P(X>=b)) where X ~ Bin(n, 0.5), capped at 1
    p_low = binom.cdf(min(b, c), n, 0.5)
    p = min(2 * p_low, 1.0)
    print(f'{scale}: b={b}, c={c}, n_discordant={n}, McNemar exact p = {p:.4f}')
```

Expected output (verified against `docs/dataset/data/raw/` JSON at this doc's authoring time):
```
1B: b=1, c=2, n_discordant=3, McNemar exact p = 1.0000
3B: b=2, c=1, n_discordant=3, McNemar exact p = 1.0000
```

Internal consistency check: `b - c` should equal `(E_correct - R_correct)` in absolute count. At 1B this is −1 (which maps to Δ_{E−R} = −1/200 = −0.005). At 3B it's +1 (Δ_{E−R} = +0.005). If your `b - c` doesn't match `E_correct - R_correct`, you have a code bug — fix before continuing.

If your `b`/`c` counts disagree with the values above, the cached JSON has different content than what was used to produce the paper. Both possibilities (your re-run differs, or the cached JSON differs) warrant escalation.

---

## 7. Comparing your re-run vs cached artifacts

Both runs should produce semantically identical statistics, up to:
- Numerical noise in non-deterministic ops (rarely matters; the steering RNG is seeded with `--seed 42` and `hashlib.md5` is used for per-item draws — see `scripts/diagnose_track_f_negcontrol.py` for details)
- HuggingFace model revision drift (if the model card was re-uploaded since the cached run, model weights may differ at machine-precision level — see the next section)

If your re-run differs *materially* from the cached values, treat the cached JSON as suspect and the paper claim as unverified.

### A note on HF checkpoint reproducibility

The paper §A.2 declares that "exact HuggingFace checkpoint commit SHAs used in this paper (one per scale) are recorded in the anonymized supplementary materials package." At the time this doc is written, those SHA values are not yet captured (placeholder text in the supplementary `HF_CHECKPOINT_SHAS.md`). If you want to pin to whatever commit the original Track F/G runs used, ask Josh — he will need to retrieve them via `huggingface-cli` against the cache that produced the artifacts. Without pinning, your re-run uses whatever is currently on `main` for each model repo, which may or may not match.

For a smoke run, this drift is usually tolerable — but for full verification of the headline numbers, pin to the same SHAs.

---

## 8. Reporting results

When the run is complete, share the following with Josh:

1. **The 13 generated JSON files** under `scripts/_track_*.json` (compress to a `.tar.gz` if convenient)
2. **A pass/fail table** for the section-6 claims, with your re-derived values
3. **Driver stdout logs** for any track that erred or produced surprising output
4. **Hardware + software config** you used (output of `nvidia-smi`, `pip show torch transformers numpy`)
5. **Any red flags** — discrepancies between your re-run and the cached JSONs, or between either and the paper's headline numbers

The goal at the end of the smoke run is one of:

- ✅ **Reproduced.** All §6 claims hold within tolerance. The paper's empirical surface is verified. Submission can proceed.
- ⚠️ **Reproduced with caveats.** Most §6 claims hold; one or two need clarification or adjustment. Patch the paper before submission.
- ❌ **Did not reproduce.** A headline claim materially failed. Stop submission; investigate; rewrite if needed.

Submission deadline is **2026-05-25** (ARR May). Treat your smoke-run result as the final go/no-go gate.

---

## 9. Meaningfully stronger run — pre-submission upgrade actions

If GPU time and Meta-approved HuggingFace access allow, three actions would materially strengthen the paper's empirical surface beyond what's in the cached `docs/dataset/data/raw/` JSONs. These are pre-submission upgrades — if any of them lands by 2026-05-24, the paper can cite the empirical observation directly in §6 / §A.4 rather than relying on the structural argument alone.

### Hardware summary

| Action | Min VRAM | Recommended | Wall (RTX 3060) | Wall (H100) | Lambda $ cost |
|---|---:|---:|---:|---:|---:|
| 9.1 (depth=10 end-to-end) | 8 GB | 12 GB | ~15 min | ~5 min | ~$0.10 on A10 |
| 9.2 (paper-text edit) | — (no GPU) | — | 30 min LaTeX | — | $0 |
| 9.3 (Track A re-measure at 300 items) | 8 GB | 12 GB | ~15 min | ~5 min | ~$0.10 on A10 |
| **All three** | **8 GB** | **12 GB** | **~30 min GPU + 30 min paper edit** | **~10 min GPU + 30 min paper edit** | **<$0.50** |

None of these need expensive cloud hardware — any consumer NVIDIA card with 8 GB VRAM running CUDA suffices. On Lambda Cloud, an A10 (24 GB, ~$0.50/hr) clears all three for under a dollar; an H100 ($2-3/hr) is overkill at these scales and would only save ~20 minutes wall.

Peak VRAM at 1B Llama-3.2 is ~3.1 GB (matches the cached `full_step_peak_mib_median`). At 3B Llama-3.2 it's ~6-7 GB. The 12 GB recommendation is for headroom; the actual workloads are far below it.

### Action 9.1 — Run one depth=10 MCTS end-to-end and observe peak memory

**Why:** Track D's "depth-independent 3.04× M_KV" claim is currently demonstrated *structurally* (three identical static accumulator allocations measured three times) and *circumstantially* (the depth-3 searches in Tracks F and G completed without OOM). The paper claims the bound holds at depth=10 with 1,700 nodes (the Track D config `(d=10, b=3, n=1700)`). That depth has not been empirically run end-to-end. A reviewer's strongest attack is exactly this gap: "you measured three copies of the same depth-3 static state and called it depth-independent."

**What:** Add ~30 lines to `scripts/measure_kv_mcts_vram.py` (or write a new `scripts/measure_kv_mcts_depth10.py`) that:

1. Loads `meta-llama/Llama-3.2-1B-Instruct` and seeds a 4,000-token prompt (same setup as Track D)
2. Allocates `FP32Accumulator.from_kv_cache(...)` at root
3. Runs **1,700 actual `apply` / `rollback` cycles** at depth ∈ {1, ..., 10} steering levels, on real KV tensors
4. Polls `torch.cuda.max_memory_allocated()` every ~100 cycles (and after the loop) to record the running peak
5. Asserts the observed peak does not materially exceed the static `full_step_peak_mib_median` (≈3,118 MiB at 1B) — i.e., that no per-node memory leak occurs over 1,700 iterations

**Pass criteria:**
- 1,700 cycles complete without OOM
- Observed peak ≤ 1.05 × the static `full_step_peak_mib` (5% allocator-fragmentation tolerance)
- `K_base, V_base, K_accum, V_accum` survive the cycles with byte-for-byte invariance at the start of each rollback (the Theorem 1 zero-drift assertion holds at scale)

**Wall:** ~15 minutes on RTX 3060 (1B model, with 1-token forward pass per node). ~5 minutes on H100. Could also be run as apply/rollback math only (no forward pass) in ~1-2 minutes if you only want to test memory invariance of the accumulator math itself; the realistic test with forward passes per node is the one that exposes any allocator-fragmentation growth during a deep search and is the meaningful upgrade.

**Resulting paper claim:** "We empirically validate depth-independence by running 1,700 apply/rollback cycles end-to-end on Llama-3.2-1B; observed peak memory is X.XX × M_KV, within 5% of the structurally predicted bound." This converts the strongest reviewer attack from "you didn't measure it" into "we measured it; here's the number."

### Action 9.2 — Report the step-peak transient alongside the steady-state in §6 / Proposition 1

**Why:** The paper currently headlines the steady-state factor (3.04-3.06× M_KV) but the same Track D JSON also captures `full_step_peak_mib_median` showing a transient peak of ~5 × M_KV during the in-place apply. The transient is what an H100 or any GPU actually needs to hold; the steady-state is the resting floor after each apply completes. A reviewer sizing the 20B projection from 3.04× alone would underestimate by ~60%.

**What:** Add a single column to Track D's reporting table (or one paragraph in §6 / §A.4 of the paper) reporting:

| Scale | Steady-state | Step-peak (transient) | Step-peak − steady-state |
|---|---:|---:|---:|
| 1B | 3.06 × M_KV | 4.98 × M_KV | 1.92 × M_KV (one layer's `α·d_K` temporary) |
| 3B | 3.04 × M_KV | TBD (re-read JSON) | TBD |

Reading the existing JSON's `full_step_peak_mib_runs` field is sufficient — no new GPU work required. ~30 minutes of paper-text edit.

**Resulting paper claim:** "The bound is 3.04 × M_KV at steady state, with transient peaks during in-place apply reaching ~5 × M_KV (bounded by a single layer's `α·d_K` allocation, also depth-independent). Hardware sizing should plan for the transient." Honest and complete — defuses the "you understated peak memory" attack pre-emptively.

### Optional Action 9.3 — Re-measure Track A combined-bucket Pearson at a longer item budget

**Why:** The Track A combined Pearson is 0.47-0.51 in the cached JSON, but the paper text claims "|r| ≈ 0.60" (which is only true for specific buckets). One way to "rescue" the headline claim would be to verify whether a larger sample reaches 0.60 — the current 85-item budget yields wide confidence intervals and the true |r| could be higher.

**What:** Re-run `scripts/diagnose_track_a_entropy.py` with an expanded factual-recall item set (e.g., 300 items instead of 85), and check whether combined |r| stabilizes around 0.5 or rises toward 0.6.

**Wall:** ~15 minutes on RTX 3060 for 1B + 3B at 300 items each (linear extrapolation from the cached 85-item Track A run that took 116 sec / 134 sec per scale).

**Outcome:** Either confirms the paper's claim (and the cached 85-item value was just a small-sample fluctuation) or confirms the 0.5 value, in which case the paper text needs a one-word edit (`≈ 0.60` → `≈ 0.50`) or a bucket-specific qualifier.

### How these connect to the smoke run

These three actions are *not* part of the verification protocol in §6 — they're the next-step upgrades after §6 verification passes. The smoke run as written is sufficient to declare "the paper's existing claims reproduce." Actions 9.1-9.3 are about *strengthening* the paper's empirical surface to head off reviewer attacks pre-submission. Prioritize them in order if GPU budget is limited: 9.1 > 9.2 > 9.3.
