# Track A — Certainty/Uncertainty Reward-Signal Diagnostic Report

**Date authored:** 2026-05-08 (one day ahead of the 2026-05-09 filename target — the diagnostic completed early).
**Author:** Diagnostic Claude session #1.
**Output location:** `docs/logs/2026-05-09_track-A-certainty-probe-report.md` (per master checklist §3).
**Models exercised:** `meta-llama/Llama-3.2-1B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`, both fp16 on RTX 3060 12 GB.
**Wall-clock budget:** ≤ 30 minutes per the prompt; actual end-to-end (loads + 4 runs) ≈ 13 min.
**Driver script:** `scripts/diagnose_track_a_entropy.py` (read-only on `logomesh/*` — uses `transformers` directly).
**Raw per-prompt artifacts:** `scripts/_track_a_results_meta-llama_Llama-3.2-1B-Instruct.json`, `scripts/_track_a_results_meta-llama_Llama-3.2-3B-Instruct.json`.

---

## 1. Inventory finding — does a certainty probe exist?

**No.** A `PerLayerCertaintyProjector` (or any equivalent representation-level certainty extractor) is **not implemented** in `logomesh/whitebox.py` or anywhere else in `logomesh/`.

Verification commands (both produce zero matches in `whitebox.py`):

```
grep -nE "[Cc]ertainty|[Cc]onfidence" logomesh/whitebox.py    # 0 matches
```

Repo-wide hits for `certainty|confidence|entropy` outside `whitebox.py`:

| Match site | What it is | Relevance to a certainty probe? |
|---|---|---|
| `logomesh/graders.py` (multiple) | `confidence: float` field on `GradeResult`; static labels (1.0, 0.9, 0.3) for whether a grader's verdict is reliable. | **No.** This is grader-pipeline self-reporting, not a model-internal certainty signal. |
| `logomesh/hneuron_monitor.py` (Lines 11-12, 91-93, 388-521) | MoE *router-logit* entropy used as the H-Neuron stress proxy on `gpt-oss-20b`. `_compute_router_entropy` returns mean entropy normalized by `log(num_experts)`. | **Partially relevant — but limited to MoE router logits.** Not the next-token output distribution we need; not available on dense models (Llama-3.2-1B/3B return empty `router_logits`, see `local_model.py:222-235, 308-314`). |
| `logomesh/whitebox.py` (Lines 118, 339, 430, 605, 660, 674) | "attention_entropy" on `AttentionProbe` (per-head) and cross-entropy loss usage. | **No.** Attention-pattern entropy, not output-distribution certainty. |
| `logomesh/evidence_store.py:30` | `self_eval_confidence: float = -1.0` placeholder. | **No.** Logging field, never populated by a probe. |
| `logomesh/local_model.py:231` | Comment about MoE entropy path selection. | **No.** Routing comment. |

The closest implemented analogue to a certainty projector is `PerLayerHonestyProjector` (`logomesh/whitebox.py:1343-1490`). Its `calibrate(...)` method fits per-layer weights via difference-in-means over (benign, coerced) example pairs, then `project(hidden_states)` returns a raw dot product `ρ_R^(l)(t) = w_hon^(l)⊤ h_t^(l)` per layer (Eq. 4 of the paper). To turn this into a certainty probe one would need to:

1. Replace the (benign, coerced) calibration pairs with a (high-confidence, low-confidence) contrastive set — i.e., paired examples where one elicits a known-confident answer and the other does not.
2. Decide the layer aggregation (max, mean, last-layer-only) for a scalar certainty score.
3. Validate that the resulting direction generalises beyond the calibration set.

**Pre-flight P2 corroboration:** the master checklist (`docs/logs/2026-05-08_72hr-execution-checklist.md:61`) specifies P2 as `grep -n "Certainty\|certainty" logomesh/whitebox.py` — this returns zero matches in the present working tree.

**Track B citation audit corroboration** (`docs/logs/2026-05-09_track-B-citation-audit.md`): Zou et al. 2023 (RepE) validates probes for honesty, truthfulness, hallucination, utility, jailbreaking, emotion, and power-seeking — *not* certainty. Even if we built such a probe, we could not claim it is covered by the published RepE validation.

**Conclusion for Step 1:** the certainty probe at the representation level cannot be exercised because it does not exist. We pivot the bulk of this diagnostic to Step 6 (per-token entropy) and document Steps 2–5 as a non-existence audit trail.

---

## 2. Calibration set (verbatim)

A "certainty probe" cannot be calibrated because it does not exist. The same calibration prompts are nevertheless used as the *input* to the entropy fallback, so we record them verbatim here.

**System prompt (used for every prompt in this diagnostic):**

> You are a helpful assistant. Answer concisely.

**10 known-confident prompts (canonical factual recall):**

```
1. What is the capital of France?
2. What is 2 + 2?
3. Who wrote the play Romeo and Juliet?
4. What color is the sky on a clear day?
5. How many days are in a week?
6. What is the largest planet in our solar system?
7. What is the chemical symbol for water?
8. Who painted the Mona Lisa?
9. At sea level, what is the boiling point of water in Celsius?
10. On which continent is Egypt located?
```

**10 known-uncertain prompts (impossible / unknowable trivia):**

```
1. What was the population of Lyon in 1342?
2. What did the third assistant in Newton's lab eat for breakfast on March 15, 1687?
3. How many grains of sand are on the beach at Cabo San Lucas right now?
4. Name the 47th left-handed pottery maker in 9th century Korea.
5. What was the middle name of Genghis Khan's third archer?
6. What was the favorite color of the king of Ethiopia in 1043 AD?
7. How many leaves were on the third oak tree in Sherwood Forest in 1184?
8. What did Cleopatra's hairdresser say at noon on August 12, 30 BCE?
9. What was Aristotle's PIN code?
10. What is the name of the ant that crossed Plato's left foot during his walk on April 7, 354 BCE?
```

These mirror the proxies suggested by the prompt ("What is the capital of France?" → confident; "What was the population of Lyon in 1342?" → uncertain). Verbatim source: `scripts/diagnose_track_a_entropy.py` lines 26–48.

The 50-prompt magnitude set (20 factual / 15 ambiguous / 15 nonsense) and the 25 + 30 + 30 correctness items (easy / hard / ultra) are also verbatim in the same script (lines 50–212). They are not re-printed here for length; cross-reference by line number.

---

## 3. Signal magnitude at 1B

**Quantity measured:** Shannon entropy of the next-token distribution at the assistant-position immediately after the prompt, in nats. Computed as `H = -Σ p_i log p_i` over `softmax(logits[-1, :])` from a single fp16 forward pass on the chat-templated prompt.

**Calibration entropies (10 confident vs 10 uncertain, nats):**

| | confident | uncertain |
|---|---|---|
| values | 0.615, 0.145, 1.312, 1.088, 0.273, 0.802, 0.808, 0.851, 2.358, 0.545 | 1.814, 1.239, 0.920, 0.949, 0.826, 1.166, 1.050, 1.009, 0.848, 1.593 |
| n | 10 | 10 |
| mean | 0.880 | 1.141 |
| std | 0.593 | 0.310 |
| range | 2.213 | 0.988 |
| Δmean (uncertain − confident) | — | **+0.261 nats** (uncertain has *higher* entropy, as expected) |

**50-prompt magnitude entropies (nats):**

| Regime | n | mean | std | min | max | range |
|---|---:|---:|---:|---:|---:|---:|
| factual | 20 | 0.925 | 0.870 | 0.016 | 2.858 | 2.843 |
| ambiguous | 15 | 1.639 | 1.099 | 0.192 | 4.014 | 3.821 |
| nonsense | 15 | 2.379 | 0.662 | 1.332 | 3.584 | 2.252 |
| **all 50** | **50** | **1.576** | **1.078** | **0.016** | **4.014** | **3.998** |

**Noise floor estimate** (std of the most homogeneous regime, factual): **0.870 nats**.
**Range / noise ratio** (G1.1 metric): **3.998 / 0.870 ≈ 4.60×.** ✅

The mean-entropy ordering is monotonically increasing across the three regimes (factual < ambiguous < nonsense), which is the qualitatively expected behaviour for a content-uncertainty signal.

Raw per-prompt arrays: `scripts/_track_a_results_meta-llama_Llama-3.2-1B-Instruct.json` lines 5–137.

---

## 4. Correlation with correctness at 1B

**Protocol:** for each correctness item, the model is asked the question via `model.generate(...)` with `do_sample=False`, `max_new_tokens=15`, `output_scores=True`. The first-step score tensor `gen.scores[0]` is the logits over the next token immediately after the prompt; entropy is computed from those logits. The full 15-token greedy completion is decoded and a binary `correct` label is assigned by case-insensitive substring match against a hand-written acceptable-answer list.

Three sets were used (the easy set was added in pre-flight; hard + ultra were added when the easy set saturated at 100% accuracy and made correlation undefined):

| Set | n | accuracy | Pearson r | Spearman ρ |
|---|---:|---:|---:|---:|
| easy | 25 | 1.000 | nan (no variance) | nan |
| **hard** | **30** | **0.867** | **−0.602** | **−0.498** |
| ultra | 30 | 0.800 | −0.357 | −0.356 |
| combined (hard + ultra + easy) | 85 | 0.882 | −0.469 | −0.424 |

The sign is negative as predicted: higher first-token entropy ⇒ more likely to be incorrect.

|r| on the *hard* set is **0.602**, vs the G1.2 threshold of 0.15 — clearing the bar by **4×**. On the *combined* set (which has more statistical power), |r| = 0.469. All non-degenerate sets exceed the 0.15 threshold by a wide margin.

**Spot-check of the four 1B "hard" wrong answers** (all multi-digit math or close arithmetic confusion):

```
"What is 47 times 53?"               H=2.640  →  "47 * 53 = 2501"   (correct: 2491)
"What is 31 cubed?"                  H=1.240  →  cut off mid-computation at 15 tokens
"What is 17 squared minus 13 squared?" H=0.959  →  cut off mid-computation
"What is 89 times 91?"               H=2.113  →  "89 * 91 = 8009"   (correct: 8099)
"Who wrote 'Pale Fire'?"             H=1.623  →  "...Tom..." (correct: Nabokov)
"Who painted 'Las Meninas'?"         H=1.768  →  cut off mid-Velázquez (token-budget artifact)
```

The arithmetic failures are notable: the model produces close-but-wrong digit sequences with elevated entropy. The two truncation cases (31 cubed, 17² − 13²) are partly a `max_new_tokens=15` artifact, not a genuine entropy/correctness mismatch.

Raw records: `scripts/_track_a_results_meta-llama_Llama-3.2-1B-Instruct.json` lines 139–end.

---

## 5. Same protocol at 3B

**Calibration entropies (10 confident vs 10 uncertain, nats):**

| | confident | uncertain |
|---|---|---|
| values | 0.573, 0.292, 1.078, 0.220, 0.107, 0.703, 0.582, 0.677, 1.136, 0.486 | 1.584, 0.354, 0.835, 0.057, 0.105, 0.453, 0.256, 0.611, 0.063, 0.744 |
| n | 10 | 10 |
| mean | 0.585 | 0.506 |
| std | 0.320 | 0.446 |
| Δmean (uncertain − confident) | — | **−0.079 nats** (sign *flips* relative to 1B) |

**Important diagnostic finding at 3B:** the calibration delta inverts. Several "uncertain" prompts elicit entropy near 0.05 (essentially deterministic). Spot-checking the 3B answers, the model has learned a confident response mode for impossible trivia — either a deterministic refusal pattern ("I don't have a way to determine that…") or a confident hallucination. So the *intent* of the calibration set ("model should hallucinate or signal uncertainty here") is broken at 3B by a learned RLHF behaviour, not by a missing signal.

This finding does **not** invalidate the entropy-as-reward-signal hypothesis. What matters operationally is whether entropy correlates with *correctness on factual recall* (Step 4 / G1.2), not with synthetic uncertain-prompt labels. See §5b below.

**5a. 50-prompt magnitude entropies at 3B (nats):**

| Regime | n | mean | std | min | max | range |
|---|---:|---:|---:|---:|---:|---:|
| factual | 20 | 0.483 | 0.358 | 0.007 | 1.362 | 1.355 |
| ambiguous | 15 | 1.473 | 0.797 | 0.101 | 2.838 | 2.737 |
| nonsense | 15 | 1.736 | 0.531 | 0.912 | 2.792 | 1.880 |
| **all 50** | **50** | **1.156** | **0.799** | **0.007** | **2.838** | **2.831** |

**Noise floor estimate:** 0.358 nats.
**Range / noise ratio (G1.1):** **2.831 / 0.358 ≈ 7.90×.** ✅ (Higher than 1B's 4.60×.)

The factual-< ambiguous-< nonsense monotonic ordering is preserved at 3B and the regime separations widen (factual mean drops from 0.93 to 0.48; ambiguous mean stays ≈ 1.5). Cleaner separation, tighter factual-recall confidence, more dynamic range relative to noise.

**5b. Correlation with correctness at 3B (same protocol as §4):**

| Set | n | accuracy | Pearson r | Spearman ρ |
|---|---:|---:|---:|---:|
| easy | 25 | 1.000 | nan | nan |
| hard | 30 | 1.000 | nan (3B aces the "hard" set as well) | nan |
| **ultra** | **30** | **0.900** | **−0.614** | **−0.456** |
| combined (easy + hard + ultra) | 85 | 0.965 | −0.508 | −0.291 |

|r| on the *ultra* set is **0.614** vs threshold 0.15 — clearing the bar by **>4×**. The combined-set Spearman ρ at 3B is lower (0.291) than at 1B (0.424) primarily because the easy-and-hard items are nearly all correct at 3B, contributing zero variance — they dilute the rank correlation without adding signal.

**3B ultra-set wrong answers** (all 3 are multi-digit arithmetic, same failure mode as 1B):

```
"What is 47 times 53?"               H=1.943  →  "47 * 53 = 2481"   (correct: 2491)
"What is 17 squared minus 13 squared?" H=0.944  →  truncated mid-computation
"What is 89 times 91?"               H=2.542  →  "7899" (correct: 8099)
```

Raw records: `scripts/_track_a_results_meta-llama_Llama-3.2-3B-Instruct.json`.

**Differences between 1B and 3B:**

1. **Cleaner regime separation at 3B.** Factual-recall mean entropy drops from 0.925 → 0.483 nats; ambiguous and nonsense means are similar to 1B. The signal-to-noise ratio improves (4.60× → 7.90×).
2. **Calibration delta inverts.** Synthetic "uncertain" prompts no longer produce higher entropy at 3B — the model has learned to confidently produce a refusal or a hallucination on impossible trivia. The reward-signal usability claim therefore depends on *factual-recall correctness correlation* (which is fine), not on the synthetic confident-vs-uncertain delta (which is unreliable at 3B).
3. **Stronger correlation per item at 3B's hardest set.** On 3B's ultra set, |r| = 0.614 vs 1B's hard-set |r| = 0.602. Roughly equal at the head; 3B has lower combined-set accuracy variance because it answers the easy and hard sets perfectly.
4. **Same systematic failure mode.** Both scales fail identically on multi-digit multiplication and 15-token-truncated arithmetic, with elevated entropy on most failures. Increasing `max_new_tokens` for arithmetic items would likely sharpen the correlation further but is out of scope for this diagnostic.

---

## 6. Fallback assessment — entropy as the operational reward signal

Because no certainty probe exists at the representation level (§1) and Zou et al. 2023 RepE does not validate one (Track B), per-token next-token entropy is the operational substitute. Recapping the case for it:

- **Computability.** Already available end-to-end through `LocalLlamaOracle.generate_one_step(...)` — it returns `logits` at the last position (`local_model.py:372, 396-402`). No new code is required in `logomesh/`; a 4-line entropy formula on the returned logits is sufficient.
- **Magnitude / G1.1.** Range/noise ratios of 4.60× (1B) and 7.90× (3B) — both clear the 3× threshold comfortably.
- **Correctness correlation / G1.2.** |Pearson r| on the most-informative correctness set: 0.602 (1B hard) and 0.614 (3B ultra). Both clear the 0.15 threshold by ≥ 4×. Sign is negative as predicted.
- **Regime ordering.** Factual < ambiguous < nonsense monotonically, at both scales. This is exactly the qualitative behaviour one would want a reward signal to exhibit if "high entropy ↔ low certainty" is the desired interpretation.
- **Known limitation.** First-token entropy on the *first answer token* is sensitive to surface formatting choices (e.g., the model deciding between "Paris" vs "The capital of France is Paris."). Two of the highest-entropy *correct* 1B answers had values 1.27 and 1.47 nats simply because the model hesitates between equally-valid answer-prefix templates. For the MCTS reward signal this means we should consider:
  - aggregating entropy over the first *k* generated tokens (smooths surface-form noise), or
  - entropy averaged over only the *content-bearing* tokens, or
  - using a normalised entropy (per-token H / log(vocab)).
  These refinements are out of scope for Gate 1; they belong in the experimental design phase.
- **Known limitation.** The synthetic uncertain calibration set is not a reliable signal at 3B (Δmean inverts to −0.08 nats). Using "model gives high entropy on impossible trivia" as a *negative control* will fail at 3B and probably also at scale; instead, the negative control should be "model gives low entropy on questions it answered correctly" — which is the mirror of the validated correlation in §4-5.

**Verdict on entropy as fallback:** validated at 1B and 3B against both Gate-1 criteria, with caveats for surface-form sensitivity and a 3B-specific RLHF-learned response-mode for impossible trivia. The latter affects synthetic calibration but does **not** affect the operational use-case (rewarding low-entropy correct trajectories during MCTS rollouts on factual or reasoning benchmarks).

---

## 7. Pass/fail verdict per Gate 1 criteria (G1.1 and G1.2)

Criteria are stated in `docs/logs/2026-05-08_72hr-execution-checklist.md:120-121`:

| # | Criterion | Falsifier | 1B result | 3B result | Verdict |
|---|---|---|---|---|---|
| **G1.1** | Certainty signal varies meaningfully across prompts at 1B (range > noise floor by 3×) | Constant or random | range/noise = **4.60×** | range/noise = **7.90×** | **PASS at both scales** (fallback signal: entropy) |
| **G1.2** | Certainty correlates with correctness on factual recall (\|Pearson r\| > 0.15 OR \|Spearman ρ\| > 0.15) | r < 0.10 with no fallback | hard-set \|r\| = **0.602**, \|ρ\| = **0.498**; combined \|r\| = **0.469**, \|ρ\| = **0.424** | ultra-set \|r\| = **0.614**, \|ρ\| = **0.456**; combined \|r\| = **0.508**, \|ρ\| = **0.291** | **PASS at both scales** (fallback signal: entropy) |

**Both Gate 1 criteria pass at both 1B and 3B**, on the explicit understanding that the validated signal is per-token entropy of the next-token distribution, not a representation-level certainty probe. The certainty probe falsifier ("Constant or random certainty values") cannot strictly be evaluated for an unbuilt probe, but the entropy fallback that the master checklist explicitly lists as the authorised substitute (`docs/logs/2026-05-08_72hr-execution-checklist.md:27`) clears both bars.

---

## 8. Plasticity recommendation

**Use entropy.** Concretely:

1. **Reward signal for Reversible KV-MCTS.** Replace any `ρ_R` (RepE certainty) term in `compute_node_reward` (`logomesh/telemetry_matrix.py`) with a function of next-token entropy at the rolled-out leaf. The signal is already wired through `LocalLlamaOracle.generate_one_step(...)` via the returned `logits`; no new probe-training step is required. Suggested concrete formula (kept simple for Phase 3):
   - `H_t = -Σ softmax(logits[-1,:]) log softmax(logits[-1,:])`
   - `certainty_reward = -H_t / log(|V|)` (per-token, normalised to roughly [-1, 0]; vocab-size denominator makes it comparable across tokenizers).
2. **Calibration data.** Drop the synthetic confident/uncertain pairing as the calibration objective; calibrate against factual-recall *correctness* directly (Spearman ρ between entropy and correctness on a held-out QA set is the single number to track). The 30-item ultra correctness set in this script is a usable starter.
3. **Aggregation across tokens.** For a multi-token rollout, use mean per-token entropy over content-bearing tokens (not the very first token, which is dominated by templating choices). This is one extra line in the rollout loop.
4. **Both bars are fatal-finding-2 false alarms.** The master checklist's fatal finding #2 reads "Certainty probe at 1B/3B is degenerate **AND** fallback (per-token entropy) also degenerate". The first half is vacuously true (the probe doesn't exist), but the second half is *false* by §3-§5 above. Do **not** flag fatal #2.

If certainty probe construction is later attempted (§1 paragraph "PerLayerCertaintyProjector would need…"), it should be benchmarked **against** entropy as the established baseline, not in place of it.

---

## 9. Bibliography of code paths inspected

Citations are `path:line` pairs. Lines refer to the working tree at the start of this session (commit `45c9330` per `git log -1`).

**Probe inventory (§1):**
- `logomesh/whitebox.py:1-1490` — full file. Probes implemented: `GradientSaliencyProbe` (Lines 335-477), `GCGAttackProbe` (483-588), `AttentionAnalysisProbe` (594-714), `LossLandscapeProbe` (720-867), `RepresentationEngineeringProbe` (873-963), `PerLayerHonestyProjector` (1343-1490). No certainty class.
- `logomesh/whitebox.py:1196-1336` — `_extract_refusal_direction` — closest analogue for a difference-in-means linear probe; implemented for refusal direction, not certainty.
- `logomesh/whitebox.py:1343-1490` — `PerLayerHonestyProjector` — the per-layer projector class that a certainty version would parallel.
- `logomesh/graders.py:19-168` — confidence as a *grader-output* field (irrelevant to probe-level certainty).
- `logomesh/hneuron_monitor.py:11-12, 91-93, 388-521` — MoE router-logit entropy code for `gpt-oss-20b`. Reusable as a precedent for a per-layer entropy aggregator but only fires when `output_router_logits=True` (MoE path).
- `logomesh/evidence_store.py:30` — `self_eval_confidence: float = -1.0` placeholder.
- `logomesh/local_model.py:222-235` — `get_router_logits()` returns empty list for dense Llama (so MoE-entropy path is unreachable at 1B/3B).

**Operational entropy access (§§3-5, §8):**
- `logomesh/local_model.py:168-202` — `generate_one_step(...)` async signature.
- `logomesh/local_model.py:324-402` — `_generate_one_step_sync(...)`. Note line 372 (`logits = outputs.logits[:, -1, :]`) and the returned dict on line 396 (`"logits": logits.detach().cpu()`). Per-token entropy is one softmax/dot-product away from this returned tensor.
- `logomesh/local_model.py:255-322` — full `_generate_sync` for completeness; not used by the entropy probe.

**Equation references (paper):**
- `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` — Eqs. 3, 4, 8, 10 (referenced via `CLAUDE.md` "Key Paper Equations" block — not re-read in this session because the diagnostic is upstream of any paper edit).

**Master checklist & supporting docs:**
- `docs/logs/2026-05-08_72hr-execution-checklist.md:25-28` — fatal-finding definitions, especially #2.
- `docs/logs/2026-05-08_72hr-execution-checklist.md:60-61` — pre-flight P2 (the grep that motivated this diagnostic's pivot).
- `docs/logs/2026-05-08_72hr-execution-checklist.md:109-122` — Gate 1 criteria G1.1, G1.2.
- `docs/logs/2026-05-08_72hr-execution-checklist.md:186-240` — the prompt template that authored Track A.
- `docs/logs/2026-05-08_EMNLP Framing for KV-Cache MCTS.md:84` — the framing claim that "the System-2 pivot simply requires swapping the top-down RepE target from 'Honesty' to 'Certainty'", which §1 of this report shows is not yet a pre-existing capability in the code.
- `docs/logs/2026-05-09_track-B-citation-audit.md` — referenced for the Zou et al. 2023 RepE certainty-probe non-coverage finding cited in §1.

**Diagnostic artefacts produced this session (read-only on `logomesh/*`):**
- `scripts/diagnose_track_a_entropy.py` — driver script (~280 lines, transformers-only, no `logomesh` imports).
- `scripts/_track_a_results_meta-llama_Llama-3.2-1B-Instruct.json` — full per-prompt 1B record set.
- `scripts/_track_a_results_meta-llama_Llama-3.2-3B-Instruct.json` — full per-prompt 3B record set.

---

## Appendix A — Reproduction

```
uv run python scripts/diagnose_track_a_entropy.py --model meta-llama/Llama-3.2-1B-Instruct
uv run python scripts/diagnose_track_a_entropy.py --model meta-llama/Llama-3.2-3B-Instruct
```

Each call: ≈ 100–210 s wall on RTX 3060 (one-time HF download for 3B if cache cold). Outputs JSON record + a `SUMMARY:` line to stdout. Deterministic except for non-determinism in `model.generate(..., do_sample=False)` (greedy decoding is deterministic given fixed seeds; HF transformers does not set a manual seed inside this diagnostic).
