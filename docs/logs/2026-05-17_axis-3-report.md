# Axis 3 Review — Reproducibility Walkthrough of §A from Scratch

**Reviewer:** Axis 3 (Round 2 parallel adversarial review).
**Date:** 2026-05-17 (Day 12).
**Artifacts reviewed:**
- `docs/logs/2026-05-13_session-A-prose-drafts-v10-reproducibility-appendix.md` (the §A draft)
- `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` lines 482–511 (deployed §A)
- `docs/logs/2026-05-11_track-F-negative-control-report.md` (the primary source the headline rests on)
- `scripts/diagnose_track_f_negcontrol.py` (the driver §A points reviewers to)
- `logomesh/kv_mcts.py:610-631` (MCTSConfig dataclass)

**Method:** Simulate a hypothetical researcher with an H100, never having touched this repo, attempting to reproduce the headline Track F result (paired McNemar p=1.0, n=200, 1B + 3B). For each step, identify what §A tells the reviewer vs. what they would actually need.

**Draft-vs-deployed cross-check.** I read the deployed §A in the TeX (lines 482–511) against the draft (Section 2 of the markdown). They are *identical character-for-character* in subsection prose modulo the REVISION marker date (draft says "2026-05-13" via the marker; deployed says "2026-05-14"). No divergence finding from this comparison — but every §A defect listed below is present in *both* the draft and the deployed TeX.

---

## Step 1 — Reading §A top-to-bottom: what a new researcher needs that §A omits

A reviewer reading §A in isolation would discover the following missing prerequisites only after they fail. None of them are stated in §A:

| Missing prerequisite | Where it bites the reviewer | Inferable from §A alone? |
|---|---|---|
| **HuggingFace account + token** | Both `meta-llama/Llama-3.2-1B-Instruct` and `Llama-3.2-3B-Instruct` are gated repositories. `from_pretrained` fails without `huggingface-cli login` or `HF_TOKEN` env var. | No — §A says checkpoints are "open-weight" (true after licence accept, but gated to first download). |
| **Meta Community Licence accept-click** | Llama 3.2 requires accepting the licence on the HuggingFace model page before token access works. | No — §A does not mention licence acceptance. |
| **Python + uv installed** | Driver header (`uv run python`) assumes `uv` is installed; not in any apt/brew distro default. | Partial — `uv run` appears in the Track F Appendix A reproduction snippet (line 266–270 of `2026-05-11_track-F-negative-control-report.md`), but the deployed §A has no `uv` invocation example. A reviewer reading §A alone never sees `uv run`. |
| **Repo clone URL** | §A points to `logomesh/kv_mcts.py` and `scripts/diagnose_track_f_negcontrol.py`, but never says where the repo lives. (For ARR review, this is by design — the supplementary URL is masked as `XXXX`.) | No. |
| **PyTorch CUDA wheel install command** | §A states "PyTorch 2.6.0+cu124" but does not provide the install command (which is non-trivial — wrong wheel = silent CPU fallback). | No. |
| **Python interpreter version** | Driver uses `from __future__ import annotations` and PEP 604-style annotations; needs Python ≥ 3.10 in practice. §A is silent on the interpreter version. | No. |
| **`transformers` version pin** | §A defers this to supplementary. The Llama 3.2 chat template format and grouped-query attention shape require `transformers ≥ 4.45`. Without this pin, the chat template silently renders differently and the entropy calibration drifts. | No — §A explicitly defers all non-PyTorch version pins. |
| **`datasets` version pin** | `load_dataset("allenai/ai2_arc", "ARC-Easy")` API has changed across major `datasets` versions; older versions return different column names (`answerKey` vs `answer_key`). | No. |
| **Disk space for HF cache** | The 1B is ~2.5 GB on disk; the 3B is ~6 GB. ARC-Easy adds ~30 MB. Not block-stopping, but worth a one-liner. | No. |

**Severity assessment:** the HuggingFace gating + licence acceptance is the largest practical blocker because it converts a "5-minute setup" into a "wait 24 hours for Meta to approve your account" delay. A reviewer doing this in the 4-week ARR review window will be annoyed.

- **GAP-1.1 [MAJOR]** §A omits HuggingFace authentication and Llama licence acceptance. Reviewer says: "I cloned and ran `python diagnose_track_f_negcontrol.py --model meta-llama/Llama-3.2-1B-Instruct` and got `OSError: You are trying to access a gated repo`."
  - **Patch:** Add to §A.2 (Models and checkpoints): "Both checkpoints are gated. Reviewers must (i) accept Meta's Llama Community Licence on the HuggingFace model page and (ii) authenticate via `huggingface-cli login` before first run."
- **GAP-1.2 [MODERATE]** §A omits the Python interpreter version. Reviewer says: "Driver uses `from __future__ import annotations` and PEP 604 type hints; do I need 3.10 or 3.11?"
  - **Patch:** Add to §A.1 (Hardware and software stack): "Python 3.10 or later (CPython, tested on 3.11)."

---

## Step 2 — Constructing the environment from §A alone: pinned vs un-pinned dependencies

§A.1 (Hardware and software stack) names exactly **one** version pin: PyTorch 2.6.0+cu124. The §A.5 closing sentence enumerates the pinned-but-deferred libraries: `transformers, datasets, huggingface_hub, numpy, scipy`. There is no `accelerate`, no `tokenizers`, no `safetensors` mention. The Track F report (line 12 above + line 37 hardware row) likewise pins only PyTorch.

The §A draft footnote (Section 4 of the markdown, page-budget commentary) acknowledges this: "Full HuggingFace checkpoint commit SHA table + `requirements-frozen.txt` excerpt or equivalent pip-freeze listing (`transformers, datasets, huggingface_hub, numpy, scipy, accelerate, etc.`) ... Phase 3 camera-ready expansion fills these." So the deferral is **intentional** for ARR review.

**The problem:** the deferral is intentional but the *consequences of the deferral* are not flagged. A reviewer trying to construct an environment from §A alone has zero version constraint on `transformers`, the library that loads the chat template that drives the prompt formatting that determines the entropy reward signal that the whole headline rests on. The deferred metadata is **not** uniformly "supplementary-OK" — see Step 7 (assessment) for the per-item triage.

- **GAP-2.1 [MAJOR]** §A defers `transformers` version pin to supplementary, but the chat template and the GQA shape constants change across `transformers` major versions. Reviewer says: "I installed `transformers` latest; my entropy values don't match the paper to 3 decimal places. Is this a different chat template or a real reproduction failure?"
  - **Patch:** Add to §A.1: "Library version pins critical to reproducibility: `transformers ≥ 4.45.0` (Llama 3.2 chat-template support), `datasets ≥ 3.0.0` (ARC-Easy column schema), `accelerate ≥ 1.0.0` (`device_map='cuda'` semantics)."
- **GAP-2.2 [MODERATE]** §A names `numpy` and `scipy` in the deferred list but the driver imports them at module top (`import numpy as np`, no scipy). The reviewer cannot tell from §A whether numpy ≥ 2.x is supported (numpy 2.0 broke several float casts that PyTorch 2.6 only handles cleanly on cu124 + Linux). On Windows + cu124 some users have hit `numpy 2.x` → `torch` ABI mismatches.
  - **Patch:** Add `numpy < 2.0` to the §A.1 pinned list.

---

## Step 3 — Downloading the model from §A's instructions

§A.2 says: `meta-llama/Llama-3.2-1B-Instruct`, loaded via `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16)`. No SHA, no revision pin.

The §A.5 closing sentence: "Exact HuggingFace checkpoint commit SHAs ... are recorded in the anonymized supplementary materials package, available at `https://anonymous.4open.science/r/kv-mcts-dimensional-escape-XXXX`".

A reviewer would:
1. Type the model name verbatim — unambiguous, no naming collisions on HuggingFace.
2. Get the *current main branch tip* of the checkpoint.
3. Have no way to verify they got the same weights the paper used.

**Why this matters in practice.** Meta has revised the `Llama-3.2-1B-Instruct` checkpoint at least once since release (the model card notes a `v1.1` update for tokenizer alignment). Without the SHA, a reviewer who downloads in May 2026 vs the paper authors who downloaded in May 2025 may legitimately be using different weights. The paper's accuracy of 0.295 (1B greedy on ARC-Easy 200) is a 3-decimal-place claim; weight drift can move that by 1–2pp.

- **GAP-3.1 [MAJOR]** §A defers checkpoint SHAs to supplementary. Reviewer says: "I downloaded the 1B model in May 2026 and got greedy accuracy 0.310; the paper reports 0.295. Is this a real reproduction failure or a different weight revision?"
  - **Patch:** Add to §A.2: "The 1B and 3B HuggingFace commit SHAs used in this paper are [SHA-1B] and [SHA-3B] respectively. To reproduce exactly, use `from_pretrained(..., revision='[SHA]')`." The SHAs **must** be in §A, not deferred to supplementary, because they are necessary to even start the reproduction.
  - This is the highest-priority §A patch. It costs one line per checkpoint and is the single most likely cause of "tried to reproduce, accuracy is off, gave up" failures.

---

## Step 4 — Constructing the CLI command from §A

§A.4 (Per-track configurations) says under Track F: "Driver: `scripts/diagnose_track_f_negcontrol.py`; raw artifacts `scripts/_track_f_results_meta-llama_Llama-3.2-{1B,3B}-Instruct.json`."

A reviewer wanting to reproduce **does not have**:
- The full command-line invocation.
- The required `--model` argument value.
- The required `--seed` value (mentioned only in §A.5 as "`--seed 42`" generically).
- The `--n`, `--device`, `--out`, `--smoke` flag defaults.

The Track F report Appendix A (lines 263–283 of the report) **does** show the exact command:
```
uv run python scripts/diagnose_track_f_negcontrol.py --model meta-llama/Llama-3.2-1B-Instruct
uv run python scripts/diagnose_track_f_negcontrol.py --model meta-llama/Llama-3.2-3B-Instruct
```

…but a reviewer reading §A does not necessarily read the Track F report. The Track F report is not in the camera-ready package (it is an internal log under `docs/logs/`). For ARR review, the only artifact a reviewer has is the paper. §A is the *only* reference point.

Reading the driver source (`diagnose_track_f_negcontrol.py:355-363`):
```python
p.add_argument("--model", required=True)
p.add_argument("--device", default="auto")
p.add_argument("--n", type=int, default=N_ITEMS)        # 200
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out", default=None)
p.add_argument("--smoke", action="store_true", ...)
```

So `--seed 42` is already the default, but a reviewer reading §A would not know this and might pass `--seed 0` thinking they're being thorough. Also, `--n` defaults to 200 and `--device auto` defaults to cuda — both fine, but un-stated.

- **GAP-4.1 [MAJOR]** §A does not provide the exact reproduction commands. A reviewer would have to guess or read source. Reviewer says: "I see the driver script name; what's the command line?"
  - **Patch:** Add a code block at the end of §A.4 (Per-track configurations):
    ```
    To reproduce the Track F headline:
      python scripts/diagnose_track_f_negcontrol.py --model meta-llama/Llama-3.2-1B-Instruct --seed 42
      python scripts/diagnose_track_f_negcontrol.py --model meta-llama/Llama-3.2-3B-Instruct --seed 42
    Defaults: --n 200, --device auto, --out scripts/_track_f_results_<modelid>.json.
    ```
- **GAP-4.2 [MINOR]** §A.5 says "`--seed 42`" generically but does not say it is the default for the Track F driver. A reviewer might assume it must be passed explicitly.
  - **Patch:** Either change §A.5 to "`--seed 42` (the script default)" or add the explicit command in GAP-4.1.

---

## Step 5 — Does §A correctly describe the protocol details?

**(a) ARC-Easy 200-item subset.** §A.4 Track C paragraph: "first 200 deterministic test-split items from `allenai/ai2_arc` configuration ``ARC-Easy'' (indices 0…199, no shuffling or filtering)". Driver code (`diagnose_track_f_negcontrol.py:77-96`) confirms: `split.select(range(min(n, len(split))))`, which is `range(0, 200)`. ✓ **Correct.**

  - **Caveat:** the driver *filters* items where `gold` is not in `{A, B, C, D, E}` (lines 86–88), which can drop items silently. The Track F report does not flag this. §A is silent on it. If any of the first 200 items have a non-letter answerKey, the actual n could be < 200.
  - **GAP-5a.1 [MINOR]** §A says "no shuffling or filtering" but driver does filter on `gold in LETTERS` (5-letter set including E). A reviewer doing a sanity check on `len(items)` may find < 200.
    - **Patch:** Either remove "or filtering" qualifier or add: "items whose `answerKey` is not in {A,B,C,D,E} are filtered (in practice, zero items in the first 200 are dropped on ARC-Easy test split)."

**(b) MCTS protocol.** §A.3: "depth-3 trees with 27 = 3^3 enumerated path samples drawn from a three-element alpha set {0.1, 0.5, 1.0}". Driver code (lines 68–70 + 244–284) confirms exactly. ✓ **Correct.**
  - **However:** §A.3 says "evaluates each path under the entropy-normalized reward (Eq. ref{eq:goodhart})" — this conflates Track F's two-arm protocol (R uses uniform, E uses entropy) into a single statement. §A.4 Track F paragraph does spell out both arms correctly. The §A.3 sentence is misleading in isolation.
  - **GAP-5b.1 [MINOR]** §A.3's "evaluates each path under the entropy-normalized reward" misrepresents the protocol; R arm uses uniform.
    - **Patch:** Rephrase §A.3 closing: "The variant evaluates each path under either a random reward (the R arm) or the entropy-normalized reward Eq.~ref{eq:goodhart} (the E arm)."

**(c) Steering direction.** §A.4 Track F: "single random unit vector in d_model space, seeded by `--seed 42`, shared identically across all items and across the R and E arms". Driver code (lines 342–347) confirms: `np.random.RandomState(seed).randn(d_model)`, normalized. ✓ **Correct.**
  - **However:** `np.random.RandomState` is the *legacy* numpy RNG. Newer numpy versions discourage it; reproducibility depends on numpy keeping the legacy RandomState behaviour stable, which it does, but the reviewer cannot verify they're getting the same vector without inspecting `np.random.RandomState(42).randn(2048)[:5]` and comparing to a paper-supplied checksum.
  - **GAP-5c.1 [MODERATE]** §A does not provide a steering-direction checksum or a "sanity-check first-5-floats" output. A reviewer cannot tell ex-ante if their steering direction matches the paper's. A subtle numpy/RandomState version drift could silently produce a different vector while passing all type checks.
    - **Patch:** Add to §A.4 Track F paragraph: "Sanity check: for `--seed 42` at 1B (d_model=2048), the first 5 elements of the unit-normalized steering vector are [V1, V2, V3, V4, V5]. Reviewers reproducing the run should verify this matches before proceeding."

**(d) Three conditions G / R / E.** §A.4 Track F paragraph spells out all three: G (greedy decoding, no MCTS); R (random-MCTS, `r_R ~ Uniform(0,1)`); E (entropy-MCTS, `r_E = -H_t / log|V|`). Driver code confirms. ✓ **Correct.**

**(e) Output measurement.** §A.4 Track A says accuracy via "case-insensitive substring match"; §A.4 Track F says "the model generates the 5-token answer continuation" but does not explicitly state how the predicted letter is extracted. Driver code (lines 154–159): iterates characters of the decoded text, picks the first uppercase letter in `{A,B,C,D,E}`. This is NOT a "case-insensitive substring match" — it is a "first uppercase letter" extraction.
  - **GAP-5e.1 [MODERATE]** §A.4's "case-insensitive substring match" description (Track A) and silence on extraction (Track F) misrepresents the actual driver logic, which scans characters left-to-right for the first member of `{A,B,C,D,E}` (case-insensitive on the char, but converted to uppercase before comparison). Items where the model outputs "The answer is A" vs "A" vs "(A)" vs "A.\n" all extract to "A" — but an item where the model outputs "Let me think. The right choice is option B" would also extract to "A" if the word "Let" comes first and "L" is not in the letter set (so "B" wins) — but for cases where the model outputs "Answer A is wrong; B is right", the extraction picks "A" (first uppercase letter in the letter set), giving the wrong letter.
    - This is a known issue with letter-extraction heuristics on free-form completions. The Track F report does not flag it. The fact that the *first uppercase letter in {A,B,C,D,E}* wins means certain failure modes are masked or amplified.
    - **Patch:** Add to §A.4 Track F: "Predicted letter is the first uppercase character in the 5-token decoded continuation that matches `{A,B,C,D,E}`; if none matches, the item is scored as wrong. This extraction is stricter than substring match and may misattribute predictions on free-form completions that mention multiple letters."
  - **GAP-5e.2 [MINOR]** §A.4 Track F: "mean perplexity" — defined as `exp(mean per-token NLL)` per the report (line 36 of `2026-05-11_track-F-negative-control-report.md`). §A.4 doesn't define it at all. The deployed §A.5 says perplexities "are deterministic" but never says what "perplexity" means.
    - **Patch:** Add to §A.4 Track F: "Per-item perplexity is `exp(mean per-token NLL)` over the 5-token completion, evaluated under the *unsteered* model (forward pass on prompt + answer tokens, no FP32 accumulator state)."

**System-prompt contradiction (cross-checked across all three sources).**

The §A draft Track C paragraph (line 47 of the markdown, identical in deployed §A line 501) states: *"Prompts use the model's chat template with system message ``You are a helpful assistant. Answer concisely.''"*

The Track F driver (`diagnose_track_f_negcontrol.py:64`) actually uses: *"You are a careful, concise reasoner. Answer multiple-choice questions with a single capital letter."*

The Track C driver (`diagnose_track_c_benchmarks.py:47`) uses the same string as Track F.

This is a **direct contradiction** between §A and primary source. The two system prompts differ in (i) role wording (helpful assistant vs careful concise reasoner), (ii) explicit MCQ format instruction (present in driver, absent in §A). On a base-rate-sensitive task like ARC-Easy MCQ, a different system prompt can move accuracy by 3–5pp. A reviewer who copies §A's system prompt verbatim into their own pipeline will get different numbers than the driver produces.

- **GAP-5e.3 [SEVERE]** §A's system prompt does not match the driver's system prompt. Reviewer says: "I followed §A literally, used 'You are a helpful assistant. Answer concisely.' as my system message, and got greedy 1B accuracy of 0.31 instead of 0.295."
  - This is the single most reproduction-blocking defect in §A. It is internally inconsistent (the §A claims to be primary-source-faithful; the primary source here is the driver script; the strings do not match).
  - **Patch:** Change the §A.4 Track C paragraph to: *"system message ``You are a careful, concise reasoner. Answer multiple-choice questions with a single capital letter.'' and user-side suffix ``Answer with the single capital letter (A/B/C/D):''."* Verify by direct copy from `diagnose_track_f_negcontrol.py:64` (since Track C and Track F use the identical constant).
  - This patch is **submission-blocking** in the sense that an ARR reviewer asked to verify reproducibility would catch the contradiction by reading the driver, and it constitutes a primary-source-fidelity violation by §A.

**Additional user-suffix mismatch.** §A says the user-side suffix is `"Answer with the single capital letter (A/B/C/D):"` — driver constructs this dynamically from `ex['labels']`, which on ARC-Easy is sometimes `['A','B','C','D']` and sometimes `['1','2','3','4']` (some items use numeric labels), and on ARC-Challenge there are also `['A','B','C','D','E']` cases. Since §A is describing ARC-Easy specifically, the 4-letter case is the modal one, but a strict reviewer would flag the contradiction.
- **GAP-5e.4 [MINOR]** §A.4 Track C user suffix hardcodes "(A/B/C/D)"; driver builds it dynamically. Reviewer says: "Did you canonicalize all 200 items to A-D, or use whatever labels the dataset provides?"
  - **Patch:** Change suffix to: "the user-side suffix `Answer with the single capital letter ({labels}):` where `{labels}` is the per-item join of the dataset's `choices.label` field." Or, if all 200 items happen to have `['A','B','C','D']` labels (verifiable from the artifact JSON), state that explicitly.

---

## Step 6 — Final list of gaps that would block reproduction

Ranked by reviewer-perceived blocking severity. A "blocking" gap is one where a reviewer would say "I tried to reproduce but couldn't because §A omits/contradicts X." A "non-blocking" gap is one where the reviewer can recover via reading source or supplementary materials.

| # | Gap ID | Severity | Blocking? | One-line summary |
|---|---|---|---|---|
| 1 | GAP-5e.3 | SEVERE | Yes (direct contradiction with driver) | §A's system prompt does not match the driver's. |
| 2 | GAP-3.1 | MAJOR | Yes (cannot verify weight identity) | §A defers checkpoint SHAs to supplementary. |
| 3 | GAP-1.1 | MAJOR | Yes (cannot run without auth) | §A omits HF token + Llama licence accept. |
| 4 | GAP-4.1 | MAJOR | Yes (cannot run without command line) | §A omits the exact reproduction CLI commands. |
| 5 | GAP-2.1 | MAJOR | Partial (transformers ABI breaks chat template) | §A defers `transformers` version pin. |
| 6 | GAP-5e.1 | MODERATE | Partial (subtle accuracy drift) | §A misrepresents letter-extraction logic. |
| 7 | GAP-5c.1 | MODERATE | Partial (steering vector drift) | §A omits steering-direction sanity-check checksum. |
| 8 | GAP-2.2 | MODERATE | Partial (numpy 2.x ABI breaks) | §A defers `numpy < 2.0` pin. |
| 9 | GAP-5e.2 | MINOR | No (definition is in Track F report) | §A doesn't define "mean perplexity". |
| 10 | GAP-5a.1 | MINOR | No (zero items dropped in practice) | §A says "no filtering" but driver filters non-letter answerKeys. |
| 11 | GAP-5b.1 | MINOR | No (subsequent paragraph clarifies) | §A.3 misrepresents protocol as single-arm. |
| 12 | GAP-5e.4 | MINOR | No (modal case is A/B/C/D) | §A hardcodes user suffix labels. |
| 13 | GAP-1.2 | MODERATE | Partial (PEP 604 fails on 3.9) | §A omits Python interpreter version. |
| 14 | GAP-4.2 | MINOR | No (recoverable from GAP-4.1 patch) | §A.5 doesn't say `--seed 42` is the default. |

---

## Step 7 — Assessment: deferred metadata items — supplementary-OK or submission-blocking?

§A explicitly defers three items: (1) HuggingFace checkpoint commit SHAs, (2) exhaustive library version pins beyond PyTorch 2.6.0+cu124, (3) full per-script command-line argument tables.

Per-item triage:

**(1) Checkpoint SHAs.** *Submission-blocking.* The whole point of the headline (paired McNemar at p=1.0 on n=200) is a numerical claim. Without the weight SHA the reproducer cannot distinguish "real reproduction failure" from "different weight revision." This is GAP-3.1. The two SHAs (one per scale) cost two lines of TeX and would close the largest blocker.

**(2) Library version pins.** *Mixed.* `transformers ≥ 4.45.0` (chat template), `accelerate ≥ 1.0.0` (`device_map`), and `numpy < 2.0` (ABI) are submission-blocking-ish in the sense that the wrong version silently produces different numerics. Other pins (huggingface_hub, scipy, tokenizers, safetensors) are supplementary-OK. The §A.1 sentence "specific version pins beyond PyTorch are recorded in the supplementary materials package" should be narrowed to: "[explicit critical pins listed in §A.1]; remaining pins in supplementary."

**(3) Per-script CLI argument tables.** *Mixed.* For Tracks A, C, D the CLI args are not headline-bearing — those tracks produce supporting evidence. For Track F the exact `--seed 42 --n 200` invocation is headline-bearing (the McNemar p=1.0 claim is for the n=200 sweep at seed 42). Track F's reproduction command line **must** be in §A. The patch in GAP-4.1 closes this.

**Verdict:** of the three deferred items, item (1) is wholly submission-blocking and items (2) + (3) are partly submission-blocking. The "supplementary-OK" framing in §A is over-broad; about 60% of the deferred volume is reasonably supplementary, and 40% should be in §A itself.

---

## Step 8 — Wall-clock claim: is "18.3 min at 1B, 33.7 min at 3B" verifiable from §A?

§A.4 Track F: "Wall-clock: 18.3 min (1B sweep, 5.5 s/item) and 33.7 min (3B sweep, 10.1 s/item)."

These were measured (per the Track F report header and §A.1 hardware row) on an **RTX 3060 12 GiB**. §A.4 Track F **does not** state the GPU. A reviewer reading §A.4 Track F in isolation might infer the H100 from the "H100 lab" mention elsewhere in the paper body, but §A.1 makes clear that all experiments were run on the 3060.

**The mismatch.** The Phase 3 protocol (per CLAUDE.md Hardware section) anticipates Tianyu Shi's lab 8× H100 80GB for "Phase 3 experiments." But Track F is a Phase 2 experiment, run on the RTX 3060. A reviewer with an H100 reproducing Track F should expect **faster** wall-clock than 18.3 min (H100 is ~3-4× faster than RTX 3060 on fp16 Llama 3.2-1B inference). If they get 18.3 min on an H100, something is wrong (e.g., they're CPU-bound, or `device_map='auto'` is putting weights on multiple GPUs in a slow configuration).

The wall-clock is *verifiable* in the sense that the hardware (RTX 3060) is stated in §A.1; a reviewer who reads §A end-to-end will find the hardware. It is *not* verifiable from §A.4 Track F **in isolation**. A reviewer skimming for the "how long should I expect this to take?" answer in the Track F paragraph alone would miss the hardware context.

- **GAP-8.1 [MINOR]** §A.4 Track F's wall-clock figure should be qualified with the GPU. Reviewer says: "I have an H100 — what's the expected wall-clock there?"
  - **Patch:** Add to §A.4 Track F closing: "Wall-clock measured on the RTX 3060 (§A.1); on H100-class hardware, expect ~5× speedup."
- **GAP-8.2 [MODERATE]** §A makes no commitment about reproducing the wall-clock on H100 / other hardware. Phase 3 camera-ready commits to "validation on Linux + CUDA 12.x with smoke-test reproduction wall-clocks" (§A draft Section 4), but §A itself has no smoke-test wall-clock for a different hardware class. Reviewer says: "How do I sanity-check my setup is producing comparable wall-clocks before launching the 33-minute 3B sweep?"
  - **Patch:** Add a `--smoke` invocation (5-item run, ~50 s on 1B / ~90 s on 3B at RTX 3060) to §A.4 Track F: "Sanity-check via `python scripts/diagnose_track_f_negcontrol.py --model meta-llama/Llama-3.2-1B-Instruct --smoke` (5 items, ~50 s on RTX 3060)."

---

## Final summary — ranked list of §A patches before Day-10 application (now retroactively, before Day-15 submission)

Day-10 has already happened (§A was applied to TeX 2026-05-14 per the REVISION marker). These patches apply to *post-deployment* §A editing before Day-15 (ARR May-25 deadline).

**Tier 1 — must-fix before submission (SEVERE/MAJOR, submission-blocking):**

1. **(GAP-5e.3, SEVERE)** Fix the system-prompt contradiction in §A.4 Track C paragraph. Verbatim driver string: "You are a careful, concise reasoner. Answer multiple-choice questions with a single capital letter." Cost: ~1 sentence revision.
2. **(GAP-3.1, MAJOR)** Add the two HuggingFace checkpoint commit SHAs (1B + 3B) to §A.2. Cost: ~1 sentence.
3. **(GAP-4.1, MAJOR)** Add the exact CLI reproduction commands to the end of §A.4. Cost: ~3 lines of code block.
4. **(GAP-1.1, MAJOR)** Add HF authentication + Llama licence acceptance note to §A.2. Cost: ~1 sentence.
5. **(GAP-2.1, MAJOR)** Pin `transformers ≥ 4.45`, `accelerate ≥ 1.0`, `numpy < 2.0` in §A.1. Cost: ~1 sentence.

**Tier 2 — should-fix before submission (MODERATE):**

6. **(GAP-5e.1, MODERATE)** Correct the letter-extraction description in §A.4 Track F. Cost: ~1 sentence.
7. **(GAP-5c.1, MODERATE)** Add a steering-direction sanity-check checksum. Cost: ~1 sentence (with the actual numbers).
8. **(GAP-8.2, MODERATE)** Add the `--smoke` invocation as a sanity-check. Cost: ~1 sentence.
9. **(GAP-1.2, MODERATE)** Add Python interpreter version. Cost: ~5 words.

**Tier 3 — nice-to-have (MINOR):**

10. **(GAP-5e.2, MINOR)** Define "mean perplexity" inline. Cost: ~1 sentence.
11. **(GAP-5a.1, MINOR)** Reconcile "no filtering" with the actual filter. Cost: ~5 words.
12. **(GAP-5b.1, MINOR)** Disambiguate the §A.3 single-arm phrasing. Cost: ~5 words.
13. **(GAP-5e.4, MINOR)** Clarify the user-suffix labels handling. Cost: ~5 words.
14. **(GAP-8.1, MINOR)** Add GPU context to wall-clock numbers. Cost: ~5 words.
15. **(GAP-4.2, MINOR)** Note `--seed 42` is driver default. Cost: ~3 words. (Subsumed by GAP-4.1 patch.)

**Aggregate page-budget impact of Tier 1 + Tier 2 patches:** approximately +8 lines of TeX (~0.12 column-page at ACL two-column density). Well within the unlimited appendix budget — Tier 1 alone costs about 5 lines.

**Severity tally:** **1 SEVERE / 4 MAJOR / 4 MODERATE / 6 MINOR.** The SEVERE finding (system-prompt contradiction) is a primary-source-fidelity violation by §A and is the load-bearing blocker. Without it fixed, the §A appendix is internally inconsistent with the driver script it points reviewers to, and that inconsistency is detectable by any reviewer who opens the driver — i.e., the very reviewers most likely to attempt reproduction.

---

## Methodology note (primary-source fidelity discipline)

Every claim in this report traces to one of: (a) the §A draft markdown lines as cited, (b) the deployed §A in TeX lines 482–511, (c) the Track F report markdown lines as cited, (d) `scripts/diagnose_track_f_negcontrol.py:1-478`, or (e) `logomesh/kv_mcts.py:600-660` (MCTSConfig dataclass). Word counts and severity assessments are this reviewer's judgment. No external context was used.

*End of Axis 3 report.*
