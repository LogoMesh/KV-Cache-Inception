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
| Track D: 3.04--3.06× M_KV constant factor | ✅ Verified (1B = 3.064, 3B = 3.037; identical across all 3 (depth, branches) configs at each scale) |
| **Track A: \|r\| ≈ 0.60 entropy↔correctness** | **⚠️ Partial / overstated.** Actual combined-bucket Pearson at 1B is −0.469 (\|r\|=0.47); at 3B is −0.508 (\|r\|=0.51). The 0.60 figure is true for *specific buckets* (1B hard, 3B ultra) but not for the combined 85-item correlation that the paper text implies. See §6.3. |

So if you only have time for one thing: **verify Track A — that's the one suspect claim** as of 2026-05-18. Everything else cross-checks against the cached JSON within tolerance.

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

The two models the paper uses are **gated** — Meta requires manual approval per HuggingFace account:

- `meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`

(Note: an earlier informal mention of "TinyLlama" was a misnomer — the actual models are Meta's Llama 3.2 family, not the TinyLlama-1.1B project.)

**Steps:**

1. Create a HuggingFace account: https://huggingface.co/join
2. Visit each model page and accept Meta's Llama 3.2 Community License:
   - https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
   - https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
3. Wait for approval. Meta's manual review typically takes a few days to two weeks. There is no fast-track; plan accordingly.
4. Create an access token at https://huggingface.co/settings/tokens (read scope is sufficient).
5. Run `huggingface-cli login` and paste the token.

If you do not have Meta-approval yet and the smoke run is time-critical, ask Josh whether he can share an access-token from an already-approved account. (Token sharing has security implications; the only legitimate uses are short-lived per-session inference, never CI or shared infrastructure.)

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
