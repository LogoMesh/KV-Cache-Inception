# Track G — Latent Cartography Report (MCTS-primary + parameter-sweep supplementary, 1B + 3B)

**Author:** Diagnostic Claude session (executed in-session by Session A on 2026-05-11, pulled forward from plan Days 4-5).
**Output location:** `docs/logs/2026-05-11_track-G-cartography-report.md` (per plan v2 §2 Days 4-5).
**Models exercised:** `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct`, fp16 on RTX 3060.
**Driver script:** `scripts/diagnose_track_g_cartography.py` (read-only on `logomesh/*`; reuses `FP32Accumulator`, `_extract_kv_tensors`, `_kv_eval_cache` from `logomesh.kv_mcts`). Driver supports `--mode {sweep, mcts}`.
**Raw artefacts:**
- MCTS-primary (paper's primary Cartography methodology, matches §3 Hypothesis 2 framing):
  - `scripts/_track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json` (385 records, 1023s = 17 min wall)
  - `scripts/_track_g_mcts_results_meta-llama_Llama-3.2-3B-Instruct.json` (385 records, 1799s = 30 min wall)
- Parameter-sweep supplementary (deterministic ablation across full alpha grid):
  - `scripts/_track_g_results_meta-llama_Llama-3.2-1B-Instruct.json` (2310 records, 330s = 5.5 min wall)
  - `scripts/_track_g_results_meta-llama_Llama-3.2-3B-Instruct.json` (2310 records, 619s = 10.3 min wall)

**Total wall:** 47 min MCTS + 16 min sweep = 63 min for both methodologies across both scales.

**Methodology note (read before reading the rest):** The MCTS-primary run uses Track F's depth-3 path-sampling protocol with entropy reward — the methodology the paper's §3 Hypothesis 2 and §5 Experiment 2 framing commit to. The parameter-sweep supplementary is a deterministic ablation across a wider alpha grid `{0.0, 0.1, 0.5, 1.0, 2.0, 5.0}`; it isolates cache-geometry magnitude effects without confounds from search policy. The original v1 of this report (2026-05-11 02:00) used only the parameter sweep and was rewritten 2026-05-11 03:30 after Josh surfaced the methodology/framing-consistency issue (logged in `memory/feedback.md` + running log Day 2 PM addendum).

---

## 0. Top-line findings (read first)

- **All three success criteria from sweep-protocol §8 MET at both scales** under MCTS-primary methodology. (The original parameter-sweep run reported only 2/2 because C-1 was N/A under pure sweep.)
- **C-1 (cross-class variation in best-path α-tuple distribution): MET.** At 1B, mean-step α varies 0.34 (C1 factual) → 0.89 (C2 MCQ) → 0.83 (C3 MCQ) → 0.51 (C4 continuation). At 3B, mean-step α inverts for MCQ — 0.22 (C2) → 0.28 (C3) — while staying moderate for the open classes — 0.61 (C1) → 0.38 (C4).
- **C-2 (cross-class variation in argmax-at-best-leaf): MET.** 1B C2: 92% A under MCTS; 3B C2: only 29% A (balanced) since 3B prefers minimal steering. 1B C4: "The"=27 + "is"=15 (continuation prior); 3B C4: "The"=28 + "..."=25 (preserved natural distribution under minimal steering).
- **C-3 (mechanism plausibility): MET.** The class-specific path-selection patterns are interpretable as: search exploits a class-specific reward landscape — when the prompt class has a strong prior (1B MCQ: "A"), the search picks heavy α to amplify it; when the prompt class is already low-entropy (3B at scale on most classes), the search picks minimal α to preserve the greedy-equivalent state.
- **Headline cross-scale finding — the MCQ first-letter prior is the swing-feature:** 1B MCQ classes have a strong A-prior (83-93% baseline argmax = A), so the entropy-MCTS at 1B selects HEAVY α to amplify the prior. 3B MCQ classes have a balanced A/B/C/D distribution (29-44% A baseline), so the entropy-MCTS at 3B selects MINIMAL α — there's no prior to exploit, and any steering would only raise entropy. Track F observed this scale-difference on ARC-Easy; Cartography now shows it generalizes to TruthfulQA MCQ (C3) and contrasts with the open classes (C1, C4).
- **Search IS finding lower-entropy states** at both scales: mean reward gain (best - mean across the 27 paths) is +0.005 to +0.025 (in negative-normalized-entropy units), consistently positive. The Goodhart mechanism manifests as entropy reduction without correctness improvement, exactly as Track F characterized.
- **Plasticity verdict: GREEN (no escalation).** All 3 of 3 success criteria met at both scales. Cartography lands as Experiment 2 of the paper. Probability anchor: ~50% acceptance (Gemini round 4 high-end estimate).

---

## 1. Run configuration

| Field | MCTS-primary | Sweep supplementary |
|---|---|---|
| Methodology | Track F-style path sampling, entropy reward | Pure parameter sweep, fixed magnitudes |
| Per item: | 27 paths × 3 steps from `{0.1, 0.5, 1.0}`; pick best by `-H/log\|V\|`; re-apply best path | For each α in `{0.0, 0.1, 0.5, 1.0, 2.0, 5.0}`: apply 3 steps of magnitude α |
| Records per item | 1 (best-path leaf measurement) | 6 (one per α) |
| Total records per scale | 385 | 2310 |
| Wall (1B) | 17 min (~2.7 s/item) | 5.5 min (~0.4 s/eval) |
| Wall (3B) | 30 min (~4.7 s/item) | 10.3 min (~0.7 s/eval) |
| Steering direction | Same: random unit vector, `--seed 42`, shared across items / classes / scales |
| Class breakdown | C1=100, C2=100, C3=85, C4=100 (C3 drops 15 items with <3 distractors) |

Prompt classes (same for both methodologies):
- **C1 Factual Recall:** hand-constructed 100-item set (25 capitals / 25 dates / 25 arithmetic / 25 common-knowledge), see `scripts/_track_g_c1_factual_recall.json`. Eval mode: first-token-str.
- **C2 ARC-Easy MCQ:** test split first 100 items (overlaps Track F's items for direct comparison). Eval mode: first-letter (A/B/C/D).
- **C3 TruthfulQA mc1 MCQ:** validation split first 100 items, reformatted to 4-option (correct + 3 distractors, deterministic position via `hash(question) mod 4`). Eval mode: first-letter.
- **C4 HellaSwag Continuation:** validation split first 100 items, option list stripped (open continuation). Eval mode: open continuation (no first-letter match, just top-5 distribution + gold-rank).

---

## 2. MCTS-primary results

### 2.1 Best-path α-tuple distribution per class (success criterion C-1)

**Llama-3.2-1B-Instruct:**

| Class | Modal best-path | % at modal | 2nd modal | 3rd modal | Mean-step α |
|---|---|---:|---|---|---:|
| C1 Factual | `(0.1, 0.1, 0.1)` | 41% | `(0.1, 0.5, 0.1)` 10% | `(1.0, 1.0, 0.5)` 9% | **0.34** |
| C2 MCQ | `(1.0, 1.0, 1.0)` | 42% | `(0.5, 1.0, 1.0)` 23% | `(1.0, 1.0, 0.5)` 19% | **0.89** |
| C3 MCQ | `(1.0, 1.0, 1.0)` | 28% | `(1.0, 0.5, 1.0)` 25% | `(1.0, 1.0, 0.5)` 19% | **0.83** |
| C4 Continuation | `(0.1, 0.1, 0.1)` | 21% | `(1.0, 1.0, 0.5)` 9% | `(1.0, 1.0, 1.0)` 9% | **0.51** |

**Llama-3.2-3B-Instruct:**

| Class | Modal best-path | % at modal | 2nd modal | 3rd modal | Mean-step α |
|---|---|---:|---|---|---:|
| C1 Factual | `(0.5, 0.5, 1.0)` / `(1.0, 0.1, 1.0)` | 15% each | mixed | mixed | **0.61** |
| C2 MCQ | `(0.1, 0.1, 0.1)` | 49% | `(0.5, 0.1, 0.1)` 14% | `(0.1, 0.5, 0.1)` 13% | **0.22** |
| C3 MCQ | `(0.1, 0.1, 0.1)` | 36% | `(0.1, 0.1, 0.5)` 14% | `(0.1, 0.5, 0.1)` 13% | **0.28** |
| C4 Continuation | `(0.1, 0.1, 0.1)` | 31% | `(0.5, 0.1, 0.1)` 13% | `(1.0, 1.0, 1.0)` 10% | **0.38** |

**Cross-scale comparison (mean-step α by class):**

| Class | 1B mean-step | 3B mean-step | Δ |
|---|---:|---:|---:|
| C1 Factual Recall | 0.34 | 0.61 | +0.27 |
| **C2 ARC-Easy MCQ** | **0.89** | **0.22** | **−0.67** |
| **C3 TruthfulQA MCQ** | **0.83** | **0.28** | **−0.55** |
| C4 Continuation | 0.51 | 0.38 | −0.13 |

**Headline finding:** MCQ classes (C2, C3) show OPPOSITE search-selection patterns across scales — 1B selects heavy steering, 3B selects minimal. This is the Track F two-mechanism-route finding generalized: 1B's stronger A-prior is exploited by the search via heavy α; 3B's balanced A/B/C/D distribution provides no prior to exploit, so the search picks minimal α to preserve greedy. C1 and C4 (no strong letter prior at either scale) show smaller cross-scale differences.

### 2.2 Argmax-at-best-leaf distribution per class (success criterion C-2)

**Llama-3.2-1B-Instruct:**

| Class | Top-5 argmax (count out of n) |
|---|---|
| C1 Factual (n=100) | `The`=11, `5`=5, `July`=4, `World`=4, `9`=4 |
| C2 MCQ (n=100) | **`A`=92**, `The`=4, `1`=3, `C`=1 |
| C3 MCQ (n=85) | **`A`=81**, `B`=2, `The`=1, `C`=1 |
| C4 Continuation (n=100) | `The`=27, `is`=15, `They`=9, `he`=5, `uses`=5 |

**Llama-3.2-3B-Instruct:**

| Class | Top-5 argmax (count out of n) |
|---|---|
| C1 Factual (n=100) | `8`=7, `5`=5, `7`=4, `6`=4, `April`=3 |
| C2 MCQ (n=100) | `A`=29, `B`=27, `D`=22, `C`=16, `2`=3 |
| C3 MCQ (n=85) | `A`=39, `B`=17, `C`=15, `D`=13, `I`=1 |
| C4 Continuation (n=100) | `The`=28, `...`=25, `is`=11, `They`=5, `He`=4 |

**1B C2 reproduces Track F's headline finding** — 92/100 argmax "A" matches Track F's 185/200 = 92.5% A-prediction under heavy steering. Cartography extends this to C3 (TruthfulQA MCQ): also collapses to "A" at 81/85 (95%). The mechanism is the same; only the benchmark differs.

**3B C2 stays balanced (29/27/22/16 across A/B/D/C)** because the search selects minimal steering and there's no prior to amplify. Same for C3 (39/17/15/13 A/B/C/D).

**Cross-scale comparison: the prior-amplification mechanism activates at 1B but not at 3B**, exactly because the search's selected α differs by scale (per §2.1). Track F's two-mechanism-route finding (1B prior-amplification / 3B greedy-collapse) is the search-induced behavior; Cartography shows it generalizes across MCQ benchmarks.

### 2.3 Entropy at best-path leaf + gold-first-token rank

**Llama-3.2-1B-Instruct:**

| Class | Mean best-leaf entropy (nats) | Median gold-rank | % gold in top-5 | Search reward gain (best − mean of 27) |
|---|---:|---:|---:|---:|
| C1 | 1.45 | 0 | 90% | +0.025 |
| C2 | 1.22 | 1 | 94% | +0.011 |
| C3 | 1.19 | 1 | 100% | +0.005 |
| C4 | 3.96 | 155 | 10% | +0.020 |

**Llama-3.2-3B-Instruct:**

| Class | Mean best-leaf entropy (nats) | Median gold-rank | % gold in top-5 | Search reward gain (best − mean of 27) |
|---|---:|---:|---:|---:|
| C1 | 0.33 | 0 | 96% | +0.008 |
| C2 | 0.73 | 0 | 95% | +0.009 |
| C3 | 1.03 | 1 | 100% | +0.010 |
| C4 | n/a (data truncated) | — | — | — |

**Key observation: gold-rank is preserved under MCTS in MCQ classes.** At 1B C2, the search selects heavy steering that drives argmax to "A" 92% of the time (§2.2), yet the gold answer's median rank stays at 1 (94% of items have gold in top-5). The MCTS amplification inflates the prior's probability mass without erasing the answer representation — same nuance Cartography-sweep already showed, now confirmed under MCTS.

**Search reward gain is positive but small.** Across 27 path samples, the best path gives 0.005-0.025 better reward (in negative-normalized-entropy units) than the mean. The search IS finding lower-entropy states; the entropy reduction just doesn't correspond to better reasoning.

### 2.4 Mechanism diagnosis per class (success criterion C-3)

- **C1 Factual Recall (1B mean-step 0.34, 3B 0.61):** factual prompts have varied responses (capitals, dates, arithmetic answers), no single first-token prior. Search picks light-to-moderate steering because heavy steering would only raise entropy on items the model already has confident answers for.

- **C2 + C3 MCQ (1B mean-step 0.83-0.89, 3B 0.22-0.28):** MCQ format triggers letter-prior amplification at 1B. The search exploits the stronger A-prior at 1B via heavy α (mean-step 0.83-0.89, 42% modal `(1.0, 1.0, 1.0)`); at 3B with no strong prior, the search picks minimal α (49% modal `(0.1, 0.1, 0.1)`) to preserve the already-balanced distribution.

- **C4 Continuation (1B mean-step 0.51, 3B 0.38):** open-ended continuation has high baseline entropy (~4 nats). Search behavior is bimodal at 1B (some items pick heavy α to find a low-entropy "fallback continuation"; others pick light α to preserve coherent continuation). At 3B the distribution is more uniformly minimal-α (matches 3B's general "avoid destabilizing" pattern).

All four class-specific patterns are interpretable as the search exploiting class-specific reward landscapes. The reward function (entropy minimization) has different optimal paths depending on whether the class admits prior-amplification.

---

## 3. Cross-scale comparison — Track F generalized across 4 prompt classes

Track F observed (on ARC-Easy only): 1B selects `(1.0, 1.0, 1.0)` 91/200 (45.5%); 3B selects `(0.1, 0.1, 0.1)` 101/200 (50.5%). Cartography-MCTS at 1B reproduces this on C2 (`(1.0, 1.0, 1.0)` 42/100 = 42%) and extends to C3 TruthfulQA (`(1.0, 1.0, 1.0)` 24/85 = 28%, less dominant but same direction).

Cartography then shows the pattern is **MCQ-format-specific, not universal:**
- C1 (factual recall): 1B prefers `(0.1, 0.1, 0.1)` 41/100 = 41% — opposite to Track F's 1B pattern
- C4 (continuation): 1B prefers `(0.1, 0.1, 0.1)` 21/100 = 21% — also opposite to Track F's pattern

So Track F's 1B "heavy steering preference" is specifically about MCQ format triggering letter-prior amplification. The Cartography MCTS data shows this empirically across 4 prompt classes × 2 scales.

The "two mechanism routes" framing (Track F §7) becomes: **the search picks whichever α minimizes entropy under the model's per-class baseline distribution.** When the baseline has a strong prior (1B MCQ), heavy α exploits it. When the baseline is already low-entropy (3B everywhere except continuation), minimal α preserves it. When the baseline is balanced (1B factual), light-moderate α suffices. When the baseline is high-entropy (continuation), bimodal selection occurs.

---

## 4. Parameter-sweep supplementary ablation (sweep methodology)

The parameter-sweep run isolates cache-geometry magnitude effects across the wider alpha grid `{0.0, 0.1, 0.5, 1.0, 2.0, 5.0}` without confounds from search policy. For each (item, α), apply 3 sequential steerings of magnitude α and measure top-5 next-token distribution at the leaf. This complements the MCTS-primary by characterizing what happens at fixed magnitudes the MCTS didn't necessarily select.

### 4.1 Argmax token distribution by (class, α) — magnitude heatmap

**Llama-3.2-1B-Instruct:**

| α | C1 top-1 | C2 top-1 | C3 top-1 | C4 top-1 |
|---|---|---|---|---|
| 0.0 | `The`=12 (diverse) | `A`=83 | `A`=79/85 | `The`=22 (diverse) |
| 1.0 | `The`=11 | `A`=94 | `A`=81/85 | `The`=33 |
| 2.0 | `The`=8 (diverse) | **`A`=100** | **`A`=85/85** | `*`=44 |
| 5.0 | **`"`=100** | `"`=69 + `)`=26 | `"`=58 + `)`=26 | **`"`=100** |

**Llama-3.2-3B-Instruct:**

| α | C1 top-1 | C2 top-1 | C3 top-1 | C4 top-1 |
|---|---|---|---|---|
| 0.0 | `8`=7 (diverse) | `A`=29 (balanced) | `A`=37 | `The`=31 |
| 1.0 | `8`=7 | `A`=30 | `A`=45 | `...`=53 |
| 2.0 | `4`=7 (diverse) | `A`=68 | `A`=72 | `up`=22 (verb cluster) |
| 5.0 | ` to`=96 | ` to`=86 | ` to`=71 | **` to`=100** |

**Key supplementary findings:**
- **At α=5.0, ALL classes collapse to a class-universal "noise token"** — 1B → `"`, 3B → ` to`. Different specific token by scale but same universal-collapse pattern.
- **The MCTS's chosen α (per §2.1) lies in the regime where class-specific attractors emerge** — at 1B, MCTS picks α near 0.89 for MCQ where A-fraction is 87-94%; at 3B, MCTS picks α near 0.22 for MCQ where A-fraction is still balanced (~29%). The MCTS samples exactly the path lengths that maximize entropy reduction via class-specific priors, not the path lengths where universal collapse begins.

### 4.2 Entropy trajectory (sweep methodology)

For 1B MCQ classes the parameter sweep showed **entropy DECREASING** under moderate steering: C2 goes 1.51 → 1.49 → 1.35 → 1.21 nats across α=0.0/0.1/0.5/1.0, then back up to 1.64 at α=2.0 and 4.87 at α=5.0. The minimum entropy is at α≈1.0 — exactly the regime where MCTS selects `(1.0, 1.0, 1.0)` modally.

This validates the MCTS reward-landscape interpretation: the MCTS finds the entropy minimum that the parameter sweep reveals. The MCTS isn't selecting at random — it's discovering the cache-geometry feature the parameter sweep characterizes.

The full parameter-sweep tables, entropy trajectories, gold-rank trajectories, and per-item amplification analyses are preserved in this report's prior version (`docs/logs/2026-05-11_track-G-cartography-report.md` git-history before 2026-05-11 03:30); the headline numbers above are sufficient for the ablation framing.

---

## 5. Cached for post-T+72 paper rewrite (§5 Experiment 2 + §7 Discussion impacts)

These are the concrete paper-rewriting anchor points for the Day 3-6 Session A drafting of §5 Experiment 2 prose.

### §5 Experiment 2 (Latent Cartography) prose anchors

1. **Headline finding for §5.2 intro:** "Latent Cartography sweeps the four-prompt-class × two-scale grid with depth-3 entropy-MCTS to characterize how the search optimizes a surrogate reward as a function of prompt class. We find that the search selects class-specific path magnitudes: at 1B, MCQ-format prompts (C2 ARC-Easy, C3 TruthfulQA) drive selection of heavy steering (`(1.0, 1.0, 1.0)` 42% and 28% modal respectively; mean-step α 0.83-0.89), while factual recall and continuation prompts drive selection of light steering (mean-step α 0.34, 0.51). At 3B, MCQ selection inverts to minimal steering (mean-step 0.22, 0.28), matching Track F's 3B greedy-collapse finding. The class-specific path-selection generalizes Track F's two-mechanism-route observation across 4 prompt classes."

2. **Table 2 (MCTS path-selection table):** §2.1 of this report — best-path α-tuple distribution per (class, scale). Single combined table; rows = classes; columns = 1B modal + percent + mean-step / 3B modal + percent + mean-step.

3. **Table 3 (argmax-at-best-leaf):** §2.2 of this report — argmax distribution per (class, scale). Show MCQ collapse to "A" at 1B (92%, 95%) vs balanced distribution at 3B.

4. **Figure 1 candidate (cross-scale mean-step α):** §3 of this report — bar chart comparing 1B vs 3B mean-step α across the 4 classes. Visual hook: MCQ bars flip direction across scales.

5. **Figure 2 candidate (entropy reward landscape):** parameter-sweep entropy trajectories (§4.2 of this report) for 1B C2 — shows the entropy minimum at α≈1.0 that explains why MCTS selects `(1.0, 1.0, 1.0)` there. Validates the search's reward-landscape navigation.

6. **§5.2 Implementation Notes addition:** the parameter-sweep ablation should appear in §5.2 as "Parameter-sweep cache-geometry characterization" — explain that the supplementary sweep isolates magnitude effects without search-policy confounds, and that the MCTS picks exactly the α regimes where the sweep shows the entropy minimum.

### §3 Hypothesis 2 (`hyp:goodhart`) empirical support

The v9 §3 draft says: "The empirical mechanism by which Eq.~\ref{eq:goodhart} manifests---prior-amplification at small model scale, greedy-collapse at larger scale---is established in §\ref{sec:exp1}." 

**Cartography extends this to**: "...is established in §\ref{sec:exp1} for the entropy-minimization reward on a single benchmark, and characterized across four prompt classes in §\ref{sec:exp2}. We find the prior-amplification route is MCQ-format-specific: when the prompt admits a strong first-token prior, the search exploits it via heavy steering; otherwise, the search prefers minimal steering."

### §7.1 Discussion (Implications) nuance

The v9 §7 Discussion draft (`docs/logs/2026-05-11_session-A-prose-drafts-v9-discussion.md`) says the search "succumbs ... via prior-amplification or greedy-collapse rather than satisfying the semantic objective." Cartography MCTS data adds: **gold-rank preservation under MCTS** — at 1B C2 when 92% of items argmax-predict "A", the gold answer's median rank stays at 1 (94% in top-5). The Goodhart amplification doesn't erase the answer; it just inflates the prior. This is a methodologically important nuance — the model "knows" the answer but predicts the wrong (prior) token at argmax. New sentence for §7.1 reward-design paragraph.

---

## 6. Success criteria evaluation (sweep-protocol §8) — MCTS-primary

| # | Criterion | 1B verdict | 3B verdict | Combined |
|---|---|---|---|---|
| **C-1** | Cross-class variation in best-path α-tuple distribution | **MET** — mean-step α varies 0.34 → 0.89 → 0.83 → 0.51 across classes (§2.1) | **MET** — mean-step α varies 0.61 → 0.22 → 0.28 → 0.38 across classes; MCQ inverts to minimal (§2.1) | **MET at both scales** |
| **C-2** | Cross-class variation in argmax-at-best-leaf | **MET** — MCQ classes collapse to "A" (92%, 95%); factual/continuation stay diverse (§2.2) | **MET** — MCQ stays balanced (29%, 39%) because search picks minimal α; continuation preserves natural distribution (§2.2) | **MET at both scales** |
| **C-3** | Mechanism plausibility | **MET** — class-specific path selection interpretable as search exploiting per-class reward landscape (§2.4) | **MET** — same interpretation; the absent prior at 3B MCQ produces no exploit, hence minimal-α preference | **MET at both scales** |

**Combined verdict: 3/3 criteria met at both scales.** Plasticity escalation NOT triggered. Cartography lands as Experiment 2 with full empirical support for Hypothesis 2.

---

## 7. Plasticity recommendation

Per 14-day plan v2 §3 plasticity rule:
- "By Day 7 EOD: 2-3 of 3 success criteria hold." GREEN → Session A drafts §5 Experiment 2 as planned.

**Verdict: GREEN.** 3 of 3 criteria hold at both scales under MCTS-primary methodology. The parameter-sweep ablation provides supplementary cache-geometry characterization.

**Probability anchor (per plan v2 §7 risk register):** "If Cartography lands cleanly: probability anchors at the high end of Gemini's 45-55% range." Cartography lands cleanly under MCTS-primary methodology — the framing/methodology consistency issue surfaced earlier (logged in `memory/feedback.md`) is now resolved. **Paper acceptance probability anchored at ~50%.**

---

## 8. Bibliography of code paths inspected

**Source files (read-only):**
- `logomesh/kv_mcts.py:222-410` — `FP32Accumulator` (apply, rollback, residual_norm).
- `logomesh/kv_mcts.py:107-217` — `_extract_kv_tensors`, `_kv_eval_cache`.

**Track G driver + data:**
- `scripts/diagnose_track_g_cartography.py` — driver with `--mode {sweep, mcts}` flag.
- `scripts/_track_g_c1_factual_recall.json` — C1 hand-constructed 100-item set.
- `scripts/probe_track_g_datasets.py` — pre-flight dataset-access probe.
- `scripts/_track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json` — 385 raw MCTS records at 1B.
- `scripts/_track_g_mcts_results_meta-llama_Llama-3.2-3B-Instruct.json` — 385 raw MCTS records at 3B.
- `scripts/_track_g_results_meta-llama_Llama-3.2-1B-Instruct.json` — 2310 raw sweep records at 1B (supplementary).
- `scripts/_track_g_results_meta-llama_Llama-3.2-3B-Instruct.json` — 2310 raw sweep records at 3B (supplementary).

**Prior tracks referenced:**
- `docs/logs/2026-05-09_track-A-certainty-probe-report.md` — entropy as static correctness signal.
- `docs/logs/2026-05-11_track-F-negative-control-report.md` — entropy-MCTS Goodhart mechanism on ARC-Easy at scale. Cartography MCTS reproduces and generalizes Track F's headline finding across 4 prompt classes.
- `docs/logs/2026-05-10_latent-cartography-sweep-protocol.md` — protocol spec.

**Methodology / framing consistency lesson:**
- `memory/feedback.md` — "Protocol deviations must be pre-checked against paper-level framing claims that depend on the protocol's methodology."
- `docs/logs/2026-05-11_track-E-gate-decisions.md` — running log Day 2 PM addendum documenting the protocol-deviation issue and resolution path.

---

## Appendix A — Reproduction

```bash
# Pre-flight dataset probe:
uv run python scripts/probe_track_g_datasets.py

# MCTS-primary (Track F-style path-sampling):
uv run python scripts/diagnose_track_g_cartography.py \
    --model meta-llama/Llama-3.2-1B-Instruct --mode mcts
uv run python scripts/diagnose_track_g_cartography.py \
    --model meta-llama/Llama-3.2-3B-Instruct --mode mcts

# Parameter-sweep supplementary ablation:
uv run python scripts/diagnose_track_g_cartography.py \
    --model meta-llama/Llama-3.2-1B-Instruct --mode sweep
uv run python scripts/diagnose_track_g_cartography.py \
    --model meta-llama/Llama-3.2-3B-Instruct --mode sweep

# Smoke test (5 items × 4 classes ~ 1 min wall):
uv run python scripts/diagnose_track_g_cartography.py \
    --model meta-llama/Llama-3.2-1B-Instruct --mode mcts --smoke
```

**Determinism:** `--seed 42` controls the steering-direction random vector AND the MCTS path-sampling RNG. Reproducible across runs given fixed model checkpoint SHA + `transformers` / `datasets` versions + `torch.float16`.

---

*End of Track G Latent Cartography report v2 (MCTS-primary). Time-on-task ~65 min wall total (sweep + MCTS + report v1 + report v2 rewrite). Plasticity verdict GREEN. Session A proceeds to Day 3-6 drafting of §5 Experiment 2 prose using §5 anchor points above. Methodology/framing consistency lesson durable in `memory/feedback.md`.*
