# Track E — Gate Decisions Running Log (Session A)

**Created:** 2026-05-08 (filename dated 2026-05-11 per checklist §7-8 convention; created early as part of pre-flight prep so it's ready to fill in real-time from T+0)
**Owner:** Session A (Opus, strategic synthesis)
**Master checklist:** `docs/logs/2026-05-08_72hr-execution-checklist.md`
**Purpose:** Single source of truth for cumulative findings and three gate decisions across the 72-hour window. If this file is not maintained in real time, gates cannot be evaluated reliably.

---

## 0. Run metadata

- **T+0 timestamp:** `__________________________` (Josh fills in when started)
- **T+24 expected:** `__________________________`
- **T+48 expected:** `__________________________`
- **T+72 expected:** `__________________________`
- **Submission deadline:** 2026-05-25 (EMNLP Findings, ARR portal pending verification)

---

## 1. Pre-flight tracker (T-2)

| # | Check | Status | Notes |
|---|---|---|---|
| P1 | GPU available + 1B/3B Llama checkpoints loadable | ☐ | |
| P2 | `whitebox.py` certainty probe inventory done | ☐ | |
| P3 | `external/Logomesh` submodule decision (commit / deinit) | ☐ | |
| P4 | `docs/NeurIPS/Citations/` accessible to fresh Claude session | ☐ | 21 arXiv source bundles + 4 PDFs verified accessible |
| P5 | Drafts files (v7, v8) and premortem-v2 TeX commit decision | ☐ | |
| P6 | T+0 timestamp set above | ☐ | |

**Pre-flight gate**: ALL checks must be ☑ before dispatching diagnostic sessions A/C/D/F. (Track B can start without P1, P3, P5 since it doesn't touch GPU or canonical files.)

---

## 2. Cumulative finding tracker

Running count of contradicting findings since each gate. Reset to 0 immediately after each gate decision is recorded. **Pile-up rules per checklist §0**: 0–1 = continue; 2 = YELLOW; 3+ = NO-GO at next gate; any FATAL = immediate NO-GO.

### Gate 1 window (T+0 → T+24)

| # | Track | Finding type | Source | Severity | Action taken |
|---|---|---|---|---|---|
| 1 | B | RepE paper (Zou et al. 2023) does NOT validate a "certainty" probe — only honesty/truthfulness/hallucination/utility/jailbreaking/emotion/power-seeking | `2026-05-09_track-B-citation-audit.md` P3 verdict | NUANCED → **RETIRED 2026-05-09 by Track A entropy validation** | Resolution path validated: per-token entropy at the next-token distribution clears G1.1 (range/noise 4.6× at 1B, 7.9× at 3B) and G1.2 (\|Pearson r\| = 0.602 at 1B hard, 0.614 at 3B ultra) by ≥4×. Entropy is the operational reward signal. NUANCE retired. |

**Gate 1 cumulative count (active):** `0` (1 historical, retired)
**Gate 1 fatal-finding flag:** `none` (COCONUT explicitly cleared — Track B P1 verdict CLEAN; fatal #2 entropy-also-degenerate explicitly false-alarmed by Track A §8)

### Gate 2 window (T+24 → T+48)

| # | Track | Finding type | Source | Severity | Action taken |
|---|---|---|---|---|---|
| (proactive) | B-delta | Snell 2024 + Hasani 2025 + Stoehr 2024 + MUSE all CLEAN; no late-arriving contradictions | `2026-05-09_track-B-delta-citation-audit.md` | CLEAN (no contradiction) | Gate 2 G2.5 explicitly cleared in advance. |
| (proactive) | D | Constant factor 3.04× M_KV measured; algorithm memory-bound headline empirically supported; §6 worked-example correction needed but algorithm-level claim holds | `2026-05-10_track-D-vram-measurements.md` | CLEAN (G2.4 PASS) | Gate 2 G2.4 explicitly cleared in advance. |

**Gate 2 cumulative count:** `0`
**Gate 2 fatal-finding flag:** `none`

### Gate 3 window (T+48 → T+72)

| # | Track | Finding type | Source | Severity | Action taken |
|---|---|---|---|---|---|
|   |   |   |   |   |   |

**Gate 3 cumulative count:** `0`
**Gate 3 fatal-finding flag:** `none`

---

## 3. Gate 1 — T+24 — "Is the foundation viable?"

**Inputs to evaluate:**
- [ ] `docs/logs/2026-05-09_track-A-certainty-probe-report.md`
- [ ] `docs/logs/2026-05-09_track-B-citation-audit.md`
- [ ] `docs/logs/2026-05-10_track-D-vram-measurements.md` (first measurement)

**Pass criteria** (ALL must be true for GO):

| # | Criterion | Status | Evidence |
|---|---|---|---|
| G1.1 | Certainty signal varies across prompts at 1B (range > noise × 3) | ☑ PASS | Operational signal = next-token entropy (certainty probe doesn't exist). 1B: range 3.998 / noise 0.870 = **4.60×**. 3B: range 2.831 / noise 0.358 = **7.90×**. Both clear 3× threshold by ≥1.5×. Source: Track A §3, §5a. |
| G1.2 | Certainty correlates with correctness (Pearson r ≥ 0.15) | ☑ PASS | 1B hard set: \|Pearson r\| = **0.602**, \|Spearman ρ\| = 0.498 (n=30). 3B ultra set: \|Pearson r\| = **0.614**, \|Spearman ρ\| = 0.456 (n=30). Sign negative as predicted (high entropy → likely incorrect). Both clear 0.15 threshold by ≥4×. Source: Track A §4, §5b. |
| G1.3 | ≤ 2 NUANCED/MISMATCHED in Track B priority batch | ☑ PASS | 1 NUANCED (RepE certainty), 0 MISMATCHED — within threshold |
| G1.4 | NO fatal findings (esp. COCONUT not training-free latent search) | ☑ PASS | COCONUT verdict CLEAN: requires multi-stage training, residual-stream route (not KV-cache), no reversibility claim, "BFS" is emergent not explicit MCTS — our 3-axis differentiation holds verbatim |
| G1.5 | Track D first measurement within 50% of §6 numbers | ☑ PASS (with §6 rewrite required) | Algorithm-level claim VALIDATED: constant factor measured 3.04–3.06× M_KV, within the 2–3× claimed band in §5 Implementation Notes; depth/branching invariance confirmed empirically (all 6 cells bit-identical). Specific §6 numerical example (40.05 GB) was already flagged as wrong in §5 Implementation Notes — Track D produces the correction (162 GB at 20B; 60× reduction not 242×). No OOM at any tested config. Source: Track D §2-3 + Plasticity recommendation. **Fatal #4 explicitly NOT triggered** (3.04× < 5× threshold). |

**Decision (GO / YELLOW / NO-GO):** **GO** (called 2026-05-09 ~T+24h-equivalent)

**Rationale (≤ 200 words):**
> All 5 Gate 1 criteria PASS. G1.1 cleared 4.6× / 7.9× at 1B/3B (range/noise threshold 3×). G1.2 cleared by ≥4× margin: \|Pearson r\| = 0.602 (1B hard) / 0.614 (3B ultra) vs threshold 0.15. G1.3 cleared with margin: 0 active NUANCED (1 historical RepE-certainty NUANCE retired by Track A's entropy validation; 0 in Track B-delta). G1.4 cleared cleanly: COCONUT verdict CLEAN; Snell fatal-finding ruled out. G1.5 cleared on the asymptotic-claim dimension that matters: constant factor 3.04× M_KV is within the §5 Implementation Notes' claimed 2–3× band; depth/branching invariance confirmed empirically; fatal #4 not triggered (3.04× < 5× threshold). **Two paper-rewrite items cached for post-T+72**: §6 worked example correction (60× reduction not 242×) and §5 Implementation Notes addition (multi-GPU sharding for 20B headline config). Both align with action plan §4.6 row 168's pre-existing "memory complexity numbers replacement" coordinated update.

**If GO:** spawn Track C (Diagnostic Session #3); continue per matrix.
**If YELLOW:** Session A drafts framing-adjustment note (link below) before T+30.
**If NO-GO:** STOP all tracks; Session A produces pivot decision memo within 4h.

**Linked artifacts:**
- Framing-adjustment note (if YELLOW): `__________________________`
- Pivot decision memo (if NO-GO): `__________________________`

---

## 4. Gate 2 — T+48 — "Is the empirical bet realistic?"

**Inputs to evaluate:**
- [ ] `docs/logs/2026-05-10_track-C-benchmark-calibration.md`
- [ ] `docs/logs/2026-05-10_track-D-vram-measurements.md` (full data)
- [ ] Cumulative A/B updates if any nuance has emerged late

**Pass criteria** (ALL must be true for GO):

| # | Criterion | Status | Evidence |
|---|---|---|---|
| G2.1 | ≥ 1 benchmark with 1B greedy ≥ 30% AND headroom ≥ 10pp | ☑ PASS | 4 benchmarks pass: ARC-Easy (acc 0.310, hr 19pp), SciQ (0.460, 20pp), ProofWriter (0.620, 15pp), BBH-LD3 (0.390, 14pp). Source: Track C §1 + §5. |
| G2.2 | At 3B: baseline ≥ 40% AND headroom ≥ 10pp on some benchmark | ☑ PASS | ARC-Easy passes cleanly (acc 0.840, hr 12pp); HellaSwag/ProofWriter/BBH-LD3 also clear; SciQ near-ceiling marginal (8pp). Source: Track C §1 + §5. |
| G2.3 | Statistical significance reachable on ≤ 200 examples | ☑ CONDITIONAL PASS | n=200 paired McNemar sufficient for **Δ=10pp** target across discordance up to π_d=0.25. Δ=5pp requires π_d ≤ 0.06 (insufficient unless MCTS only flips ~6% of answers). **Headline target locked at 10pp absolute**; 5pp reportable as secondary. Source: Track C §5 corrected paired-McNemar power formulation. |
| G2.4 | VRAM measurement supports memory-bound headline (constant 2–3·M_KV) | ☑ PASS | Constant 3.04× M_KV measured. Source: Track D. |
| G2.5 | No new contradicting findings late-arriving from Track A or B | ☑ PASS | All 4 newly-downloaded papers CLEAN (Track B-delta); Track D no new fatals. Source: Track B-delta + Track D. |

**Decision (GO / YELLOW / NO-GO):** **GO** (called 2026-05-09 ~T+30h-equivalent — running ahead of schedule)

**Rationale (≤ 200 words):**
> All 5 Gate 2 criteria PASS with margin. G2.1: 4 benchmarks clear at 1B; ARC-Easy is the cleanest (acc 0.310, hr 19pp). G2.2: ARC-Easy cleanest at 3B (acc 0.840, hr 12pp); SciQ near-ceiling marginal but corroboration-eligible. G2.3 cleared on the 10pp target across realistic discordance ranges (π_d ≤ 0.25); 5pp would require unrealistically low discordance. G2.4 cleared by Track D in advance. G2.5 cleared by Track B-delta in advance. Cumulative active contradicting findings: 1 NUANCED (Track D §6 correction, planned post-T+72 rewrite). **Track C surfaces three paper-rewrite items cached for post-T+72**: (1) experimental design narrative locked to ARC-Easy primary + SciQ corroboration on n=200 paired subsets; (2) protocol caveat on letter-prompted MCQ vs published log-probability scoring (under-states 1B by ~20pp); (3) LogiQA→BBH-LD3 substitution rationale. **Compute estimate for full Gate-3-onward experiment: ~18h on H100, ~7-8h per cell on RTX 3060** — feasible within the 14-day post-T+72 window.

**If GO:** spawn Track F (Diagnostic Session #5); commit to benchmark choice in writing.
**If YELLOW:** Session A drafts framing adjustment AND alternate-benchmark fallback before T+54.
**If NO-GO:** PIVOT (most likely target = wildcard / KV-cache working memory); Session A produces wildcard pivot plan within 6h.

**Chosen benchmark (if GO):** **ARC-Easy primary** (test split, first 200 deterministic items, indices [0..199]; chat-template MCQ-letter prompting; gold = `answerKey` letter A-D). **SciQ corroboration** (same protocol, deterministic per-item rotation via `hash(question) mod 4`).
**Linked artifacts:**
- Framing-adjustment / fallback note (if YELLOW): `__________________________`
- Wildcard pivot plan (if NO-GO): `__________________________`

---

## 5. Gate 3 — T+72 — "Is the empirical claim survivable under adversarial review?"

**Inputs to evaluate:**
- [ ] `docs/logs/2026-05-11_track-F-negative-control-report.md`
- [ ] Session A's paper-text impact summary (drafted T+66)
- [ ] Cumulative findings from all tracks

**Pass criteria** (ALL must be true for GO):

| # | Criterion | Status | Evidence |
|---|---|---|---|
| G3.1 | Certainty-guided MCTS beats random by ≥ 2 absolute points on chosen benchmark | ☒ **FAIL** | 1B: Δ(E−R) = −0.5pp; 3B: Δ(E−R) = +0.5pp. Both well inside noise; both fail the 2pp threshold. Source: Track F §3. |
| G3.2 | Statistically significant (p < 0.05 or comparable Bayesian readout) | ☒ **FAIL** | 1B: p = 1.0 (3 discordant pairs); 3B: p = 1.0 (3 discordant pairs). Discordance rate π_d = 0.015 — one order of magnitude below the calibrated headroom. Source: Track F §3. |
| G3.3 | Perplexity within 2× baseline (no fluency collapse) | ☑ PASS | 1B E/G ratio = 1.348; 3B E/G ratio = 1.058. Search machinery doesn't break the model — but doesn't help, either. Source: Track F §5. |
| G3.4 | Paper-text impact achievable in 14 days | N/A | Gate failed on G3.1/G3.2; G3.4 not evaluable for original framing. Pivot memo evaluates per option. |
| G3.5 | No fatal late-arriving findings from Tracks A/B (cumulative) | ☑ PASS | No Track-A or Track-B late-arrivers; only the Track F mechanism finding itself, which IS the gate failure not a fatal-list finding. |

**Decision (GO / YELLOW / NO-GO):** **NO-GO** (called 2026-05-09 ~T+40h-equivalent)

**Rationale (≤ 300 words):**
> Gate 3 NO-GO triggered by G3.1 + G3.2 failure at both scales. Entropy-MCTS does not beat random-MCTS on ARC-Easy: Δ(E−R) = −0.5pp at 1B, +0.5pp at 3B; paired McNemar p = 1.0 both scales (3 discordant pairs each). G3.3 passes (perplexity ratio 1.348/1.058 — search doesn't break the model) but this only confirms the search machinery is mechanically operational; it doesn't rescue accuracy. The mechanism Track F identifies in §7 is a Goodhart pathology specific to entropy-minimization in latent KV-cache space: at both scales the search converges on degenerate optima that maximize confidence at the expense of correctness — at 1B by reinforcing the model's first-letter prior ("A" predicted 185/200), at 3B by collapsing to greedy-equivalent (minimal-steering paths chosen 51% of items). Track F's analysis is rigorous: the failure is conceptual, not implementation-level. No tuning of alpha set, depth, branches, or steering direction would change the basic mechanism. **What's preserved**: algorithm + memory bound contributions (intact, Tracks A/B/D validated cleanly); 21+ citations CLEAN; Reversible MCTS + FP32 accumulator + Theorem 1 all hold. **What's killed**: entropy-as-MCTS-reward formulation (Contribution 3 of the Candidate A framing). The pivot memo at `docs/logs/2026-05-11_pivot-decision-memo.md` evaluates four options.

**If GO:** Session A drafts 14-day execution plan at `docs/logs/2026-05-11_post-T72-execution-plan.md`. Submit on track for 2026-05-25.
**If YELLOW:** GO with scope downgrade (drop Best-of-N comparison or similar); ship smaller paper.
**If NO-GO:** HARD PIVOT. Three options remain (wildcard, hallucination, workshop). Decision matrix at `docs/logs/2026-05-11_pivot-decision-memo.md`. Josh decides within 24h (T+96).

**Linked artifacts:**
- 14-day execution plan (if GO): `__________________________`
- Pivot decision memo (if NO-GO): `__________________________`

---

## 6. Append-only findings log

*Format per checklist §8: `### T+H — [Track] — [Finding]` followed by Source / Classification / Cumulative count / Plasticity action / Notes. Append in chronological order. Do not edit prior entries; if a finding is later updated, append a new entry referencing the old one.*

---

### T+0 — Session A — Skeleton ready
- Source: this file
- Classification: N/A (administrative)
- Cumulative count this gate: 0
- Plasticity action: none
- Notes: Pre-flight to be completed before T+0 timestamp is set. Track B can be dispatched without full pre-flight (no GPU dependency).

### T-X — Session A — Pre-flight P2 finding logged
- Source: `grep "[Cc]ertainty\|[Cc]onfidence" logomesh/whitebox.py` returns 0 matches
- Classification: not a Gate 1 contradicting finding (this is pre-flight inventory, not a track output) — but informs Track A path
- Cumulative count this gate: 0 (not counted; pre-flight)
- Plasticity action: noted in Track A prompt update; entropy fallback becomes primary path
- Notes: CLAUDE.md mentioned "RepE probes for honesty, certainty, and goal-coercion" as planned, but only honesty was implemented in `whitebox.py`. The certainty probe never landed in code.

### Day 2 evening (calendar 2026-05-11) — MCTS Cartography re-run complete; methodology consistency restored; GREEN verdict (all 3 of 3 criteria met)
- Source: re-execution of Track G with proper MCTS path-sampling per Track F protocol; report rewritten at `docs/logs/2026-05-11_track-G-cartography-report.md` (v2 MCTS-primary, sweep as §4 supplementary).
- Classification: EXPERIMENT 2 EMPIRICAL CONTENT LANDED CORRECTLY (methodology now consistent with §3 Hypothesis 2 framing).
- Cumulative count this gate: N/A
- Plasticity action: GREEN — all 3 of 3 success criteria met at both scales. The N/A on C-1 from the pure-sweep version is now MET under MCTS-primary.
- Notes:
  - **Wall**: 1B MCTS 17 min + 3B MCTS 30 min = 47 min total. Per-item rate ~2.7s at 1B, ~4.7s at 3B. Reasonable, well within plan tolerance.
  - **Records**: 385 per scale (100 + 100 + 85 + 100 across C1/C2/C3/C4). One record per item (vs 6 for the sweep methodology since MCTS picks a single best path per item).
  - **Success criteria verdict — ALL THREE MET at both scales:**
    - C-1 (cross-class best-path α-tuple variation): **MET.** 1B mean-step α 0.34 (C1) → 0.89 (C2 MCQ) → 0.83 (C3 MCQ) → 0.51 (C4). 3B inverts on MCQ: 0.61 → 0.22 → 0.28 → 0.38. Direct evidence of cross-class path-selection variation.
    - C-2 (cross-class argmax-at-best-leaf variation): **MET.** 1B C2 argmax "A"=92/100 (matches Track F 185/200); 3B C2 stays balanced (A=29/B=27/D=22/C=16) because search picks minimal α.
    - C-3 (mechanism plausibility): **MET.** Class-specific path selection interpretable as search exploiting per-class reward landscape — MCQ with strong A-prior → heavy α at 1B → A amplification; no strong prior or already-confident → minimal α everywhere else.
  - **1B C2 MCTS reproduces Track F headline finding directly**: modal best-path `(1.0,1.0,1.0)` 42% (Track F 45.5%); argmax "A" 92% (Track F 92.5%). Reproduces the empirical finding under the same protocol.
  - **Cartography generalizes Track F across benchmarks**: same A-prior amplification on TruthfulQA C3 (1B `(1.0,1.0,1.0)` 28% modal, A=81/85=95% argmax). Track F's mechanism is MCQ-format-driven, not ARC-Easy-specific.
  - **Cross-scale Track F generalization**: 3B C2 MCTS reproduces Track F's 3B 101/200 minimal-steering finding (49% modal `(0.1,0.1,0.1)`). Cartography extends this to TruthfulQA (3B C3 36% modal `(0.1,0.1,0.1)`).
  - **MCTS validates the parameter-sweep landscape**: parameter sweep showed 1B C2 entropy minimum at α≈1.0 (1.21 nats); MCTS selects `(1.0,1.0,1.0)` modally. The search finds the entropy minimum the sweep characterizes — methodologically clean cross-validation.
  - **Gold-rank preservation under MCTS**: in MCQ classes at 1B, when 92-95% argmax to "A" prior, gold-answer median rank stays at 1, in-top-5 at 94-100%. The Goodhart amplification inflates the prior without erasing the answer representation. Nuance for §7.1 Discussion.
  - **Search reward gain**: +0.005 to +0.025 (negative-normalized-entropy units) — consistently positive but small. Search IS reducing entropy; reduction just doesn't correspond to better reasoning.
  - **Methodology / framing consistency now restored**: §3 Hypothesis 2 (`hyp:goodhart`) and §5 Experiment 2 framing both commit to MCTS-driven optimization. The MCTS-primary run measures exactly that. The parameter-sweep ablation provides supplementary cache-geometry characterization. No reviewer-attack-surface remains.
  - **Probability anchor (per plan v2 §7)**: ~50% acceptance (Gemini round 4 high-end). Cartography landed cleanly with the right methodology.
  - **§5 Experiment 2 prose anchors** (cached for Day 3-6 drafting per report §5): headline finding (class-specific path-selection), Table 2 (best-path α-tuple per class × scale), Table 3 (argmax at best leaf), Figure 1 (cross-scale mean-step α bar chart), Figure 2 (parameter-sweep entropy landscape validating MCTS selection), §5.2 Implementation Notes addition for the sweep-as-ablation framing.
  - **Schedule impact**: Days 4-5 of plan v2 still FREE. Day 3 absorbs §5 Experiment 2 prose drafting (was Day 6) + §A Reproducibility Appendix v1 (was Day 9) when bandwidth permits. License + Checklist drafts pending Josh J2/J3.

### Day 2 PM addendum (calendar 2026-05-11) — Cartography protocol-deviation surfaced; MCTS re-run starting
- Source: Josh's question "is the fact that you abandoned MCTS to achieve this going to affect the paper's peer review process?"
- Classification: METHODOLOGY/FRAMING CONSISTENCY ISSUE — paper-level claims depend on a methodology the executed run did not actually use.
- Cumulative count this gate: N/A (post-Gate-3 strategic phase)
- Plasticity action: Re-run Cartography with proper MCTS path-sampling per Track F's protocol. Pure-sweep data is preserved as supplementary ablation.
- Notes:
  - **Issue surfaced:** The Cartography pure-parameter-sweep I ran does not actually implement the methodology that the paper's framing claims it does. §3 Hypothesis 2 (`hyp:goodhart`) is explicitly about MCTS-driven optimization of a surrogate reward; §5 Experiment 2 framing in v9 contributions says Cartography "structurally map the pre-training priors the search converges upon." A pure parameter sweep does no search and no path selection, so it cannot evidence either claim directly. The §10 deviation rationale I wrote post-execution defended speed but didn't cross-check the paper's framing.
  - **Reviewer-attack-surface assessment:** a careful reviewer reading §5 Experiment 2 expecting MCTS results and finding a parameter sweep would rightly ask "where did the search go?" The framing in §3 + §5 + §1 Contribution 3 all imply MCTS does the work in Cartography. Pure sweep without the framing rewrite is paper-affecting.
  - **Lesson stored to durable memory (`memory/feedback.md`):** *"Protocol deviations must be pre-checked against paper-level framing claims that depend on the protocol's methodology. Speed gains are NOT sufficient justification when the original methodology is the source of a paper claim."* The new entry includes an explicit pre-deviation cross-check protocol ("I'm considering deviation X. The paper's claims that depend on the original methodology are Y. Deviation X affects/doesn't affect those claims because Z.") and lists the only acceptable deviation reasons (infrastructure constraint, explicit Josh approval, methodology not paper-anchored).
  - **Resolution path:** Option 1 from the three offered to Josh — re-run Cartography with proper MCTS path-sampling (same protocol as Track F: depth 3, 27 paths per item, alpha-set {0.1, 0.5, 1.0}, entropy reward). Wall budget: ~37 min at 1B + ~67 min at 3B ≈ 104 min total. Existing pure-sweep data is kept as a supplementary parameter-sweep ablation that complements the MCTS-driven primary Cartography.
  - **Schedule impact:** ~2 additional hours of GPU time vs the pure-sweep approach. Days 4-5 of plan remain available for §5 Experiment 2 prose drafting and §A Reproducibility Appendix work. No plan-level rescheduling required.
  - **What this preserves regardless:** the data-content findings of the pure-sweep run (class-specific argmax-token distributions; entropy DECREASING for MCQ at 1B under moderate steering; cross-scale collapse-token difference `"` vs ` to`) are facts about cache geometry. They survive the methodology fix; MCTS will, per Track F's mechanism, likely SELECT paths that strengthen these same attractors. The headline findings should hold.
  - **Reframing-in-paper plan after MCTS re-run lands:** primary Cartography figure becomes per-(class) best-path α-tuple distribution + best-path leaf top-5 token distribution. Pure-sweep heatmap moves to §A.x Supplementary as a parameter ablation. §5 Experiment 2 prose drafting (originally Day 6, possibly pulled forward to Day 4) will work from the MCTS data.

### Day 2 PM (calendar 2026-05-11) — Cartography sweeps complete (PULLED FORWARD from Days 4-5); GREEN verdict
- Source: Track G driver run on 1B + 3B; report at `docs/logs/2026-05-11_track-G-cartography-report.md`.
- Classification: EXPERIMENT 2 EMPIRICAL CONTENT LANDED — paper's positive empirical pillar now confirmed at both scales.
- Cumulative count this gate: N/A (post-Gate-3 strategic phase; this is positive validation)
- Plasticity action: GREEN — no escalation. Probability anchor moves to Gemini round 4 high-end (~50% acceptance).
- Notes:
  - **Pulled forward**: plan v2 scheduled Cartography for Days 4-5. Josh's "GO!" + free GPU enabled execution on Day 2 PM. Saved ~3 calendar days for downstream work (§5 Experiment 2 prose, §A Reproducibility Appendix, possible scope expansion).
  - **Execution wall**: 5.5 min at 1B + 10.3 min at 3B = ~16 min GPU time total. Protocol budget was 10.4 hours; achieved ~40× under budget. Reason: pure-parameter-sweep protocol deviation (no MCTS path sampling), justified in report §10.
  - **Success criteria evaluation**: C-1 (best-path α-tuple variation) N/A under pure-sweep deviation. C-2 (cross-class terminal-token variation) MET at both scales — MCQ classes amplify "A" prior, C4 amplifies "*"/verb-cluster, C1 stays diverse. C-3 (mechanism plausibility) MET at both scales — each attractor interpretable as instruction-tuning prior.
  - **Headline findings for the paper**:
    1. **Cross-class Cartography signal is clean and structural.** MCQ classes (C2, C3) converge to "A" prior at α=2.0 (100% at 1B; 68-72% at 3B). Continuation (C4) converges to scale-specific attractors: 1B `*` (44%), 3B diverse verb cluster (`up`, `cuts`, `ex`, `[`). Factual recall (C1) maintains diverse responses at intermediate α.
    2. **Goodhart entropy signature visualized.** At 1B, MCQ classes show entropy DECREASING from 1.51 → 1.21 nats across α=0.0→1.0 — the model becomes more confident as steering is applied, with confidence concentrated on the prior token. Exactly the mechanism Track F §7 inferred, now characterized across the alpha grid.
    3. **Scale-dependent threshold for prior amplification.** 1B: monotonic A-fraction increase from baseline. 3B: flat through α=1.0, sharp jump at α=2.0. Matches Track F's per-scale mechanism difference (1B reinforces prior; 3B avoids destabilizing greedy-equivalent state).
    4. **Universal collapse token differs by scale.** 1B → `"` (quote token) at α=5.0; 3B → ` to` (space-to). Same Dimensional Escape pathology, different specific collapse route.
    5. **Gold-rank preservation under prior amplification.** In MCQ classes (C2, C3) at α=2.0 when 100% of items argmax-predict "A", the gold answer's median rank stays at 1. The Goodhart amplification inflates prior probability mass without erasing the answer representation. New nuance for §7.1 Discussion.
  - **§5 Experiment 2 anchor points**: report §9 lists 5 specific paper-rewriting items (headline, Table 2 spec, Figure 1 entropy plot, Figure 2 A-fraction plot, gold-rank nuance for §7.1). Session A drafts `2026-05-12_session-A-prose-drafts-v10-exp2.md` Day 3 using these anchors.
  - **§3 hyp:goodhart hypothesis upgrade**: Cartography empirically supports Hypothesis 2 across 4 prompt classes × 2 scales (not just one benchmark at two scales as in Track F). The v9 §3 reference to §exp1 for hyp:goodhart can be extended to "...established in §exp1 for entropy-minimization reward and extended to four prompt classes in §exp2."
  - **What's preserved/dropped from sweep protocol**: 4 prompt classes ✓; 6 alphas ✓; 100 items per class ✓ (85 on C3 after distractor filter); shared steering direction ✓; top-5 + gold-rank ✓. Dropped: 27-path sampling per (item, alpha); `best_path_alphas` field. Net loss: 1 of 3 success criteria becomes N/A; structural findings unaffected.
  - **Schedule impact**: Days 4-5 of plan are now FREE. Day 3 can absorb §5 Exp 2 prose drafting (pulled from Day 6) + license/checklist drafts (waiting on J2/J3). Day 9's §A Reproducibility Appendix could also pull forward.
  - **Reproducibility artifacts produced**: `scripts/probe_track_g_datasets.py`, `scripts/_track_g_c1_factual_recall.json`, `scripts/diagnose_track_g_cartography.py`, two `_track_g_results_*.json` files (4.8 MB total raw data, 4620 records).

### Day 2 AM (calendar 2026-05-11) — TeX application of 5 v9 drafts + §7 Discussion draft
- Source: Day-2 of ARR-cycle plan v2 §2. Session B work (TeX application) and Session A work (§7 Discussion draft) executed in the same Claude session.
- Classification: NARRATIVE PIVOT EXECUTION (paper-text production)
- Cumulative count this gate: N/A (post-Gate-3 strategic phase)
- Plasticity action: none — Day 2 work landed on schedule
- Notes:
  - **Session B work — 5 v9 drafts applied to canonical TeX**:
    - Title (TeX:39 → 40): `\title{Reversible KV-Cache MCTS: A Memory-Bounded Probe of Dimensional Escape in Frozen Language Models}` with REVISION marker `title-recast-D+`.
    - Abstract (TeX:50-53): Option D+ replacement with verified numbers — 200 apply-revert cycles 1B+3B, $3.04$--$3.06\times M_{\text{KV}}$, $\sim 60\times$ reduction at $(b,d)=(3,5)$, $\Delta(\text{entropy}-\text{random}) = -0.5/+0.5$pp, McNemar p=1.0, $|r|\approx 0.60$. REVISION marker `abstract-recast-D+`.
    - §1 Contributions (TeX:80-87): 3 contributions in Option D+ order — algorithm + memory bound (with corrected 3.06×/3.04× constants); telemetry-as-monitoring (with explicit argument against telemetry-as-reward); Dimensional Escape empirical demonstration. REVISION marker `contributions-recast-D+`.
    - §3 Problem Formulation (TeX:131-154): §3.1 tightened framing + new §3.2 Dimensional Escape (Definition~\ref{def:dim_escape} + Hypothesis 1 preserved as monitor-driven sub-case + standalone Eq.~\ref{eq:alignment_faking} + Hypothesis 2 reward-driven sub-case + closing prose). Existing §3.2 Discrete Text-Space Bottleneck auto-renumbers to §3.3. REVISION marker `§3-reframe-D+`.
    - §7.3 Limitations (TeX:513-516): wholesale rewrite into 8 numbered Option D+ limitations. `\label{sec:limitations}` added. REVISION marker `§7.3-Limitations-recast-D+`.
  - **Post-edit verification**: TeX line count went 652 → 687 (+35). Grep confirmed all 5 D+ markers present. Structure grep confirmed all new `\subsection`, `\definition`, `\hypothesis` blocks rendered correctly. Existing label `hyp:orthogonal_escape` preserved verbatim. New labels added: `sec:dim_escape`, `def:dim_escape`, `hyp:goodhart`, `eq:goodhart`, `sec:limitations`. Forward-references to `sec:exp1`/`sec:exp2` produce undefined-reference warnings (expected; resolved Day 7 with §5 rewrite).
  - **Session A work — v9 §7 Discussion draft**: `docs/logs/2026-05-11_session-A-prose-drafts-v9-discussion.md`. Rewrites §7.1 (monitor → monitor + reward design implications under Definition~\ref{def:dim_escape}) and §7.5 Future Work (single-bag → 4 paragraphs: Infrastructure-and-scaling / Monitor-design research / Reward-design research / Cartography extensions). New `\label{sec:future-work}` enables §7.1's forward-reference. Net ~+720 words across §7 Discussion. Empirical claims pre-verified: `|r|≈0.60` (Track A), 185/200 "A" at 1B + 51% minimal at 3B (Track F §1+§7), ~162 GB at 20B (Track D §3+§4), 9 anti-Goodhart candidates (Track F §9 + Gemini round 4 Part 2).
  - **Cumulative Option D+ word delta over Days 1-2**: abstract neutral, §1 +95, §3 +280, §7.3 +410, §7 Discussion +720 = ~+1500 words. Day 10 page-count check will be the critical gate; plan v2 anticipates this with Day 10 cuts (§6 proof → Appendix A; verbose §4.2.2 condensed; possible deferral of §7.3 items 5/6/8 to §7.5).
  - **Deferred Day 2 items** (depend on Josh J2/J3): License-attribution paragraph and Responsible NLP Checklist Q&A draft.
  - **Josh action items still outstanding**: J1 (page budget) overdue; J2/J3 needed today EOD; J7 needed tomorrow AM. If J1 still missing Day 3, Session A drafts fallback memo with conservative 8-page assumption.

### Day 1 PM (calendar 2026-05-10) — ARR-cycle plan v2 restructure + 2 additional Day-1 drafts complete
- Source: Three parallel Phase-1 Explore agents (repo anonymization scan, EMNLP 2026 + ARR web verification, paper-state-for-Option-D+ assessment). Approved plan at `C:\Users\Josh\.claude\plans\misty-scribbling-fiddle.md`.
- Classification: STRUCTURAL RESTRUCTURE + DAY-1 EXECUTION (paper-text + plan-meta; not a track finding)
- Cumulative count this gate: N/A (post-Gate-3 strategic phase)
- Plasticity action: none — plan absorbed the deadline-correction finding and the new required deliverables (Responsible NLP Checklist, Llama license attribution, §A Reproducibility Appendix, two-pass anonymization scrub)
- Notes:
  - **CRITICAL TIMELINE CORRECTION**: per https://2026.emnlp.org/calls/main_conference_papers/ + http://aclrollingreview.org/dates, the 2026-05-25 deadline is the **ARR May 2026 submission deadline**, not EMNLP commitment. EMNLP commit deadline is **2026-08-02**; conference is **2026-10-24 to 10-29, Budapest, Hungary**. ARR review cycle: May 25 submission → reviews due June 29 → author response July 7-13 → meta-reviews July 30 → commit Aug 2 → acceptance Aug 20 → camera-ready Sep 20. EMNLP 2026 accepts ONLY ARR-routed papers (no direct submission); venue-binding selection at ARR submission is BINDING. The 14-day plan absorbed this and restructured into Phase 1 (Days 1-15, ARR sub) + Phase 2 (T+1 to T+69, commit window) + Phase 3 (T+70 to T+133, camera-ready, conditional).
  - **Anonymization audit findings**: BLOCKING leaks in `CLAUDE.md:16` (full team + advisor + institution), `CONTRIBUTING.md:3` (contributor names), `docs/team-brief-2026-04-11.md:12,48,137-139` (names + GitHub URL + advisor). HIGH-severity leaks in `.gitmodules` (github.com/Logomesh), `logomesh/croissant_export.py:335`, `docs/dataset/croissant_schema_stub.json:56`, `EXPERIMENTS.md:299` (McGill mention). `.gitignore` doesn't cover dev-only files. Two-pass anonymization scheduled: Pass A (Day 3, in-place code/metadata edits), Pass B (Day 4, `.gitignore` + `git rm --cached` for dev-only), Pass C (Day 12, verification grep).
  - **Three new required deliverables added to plan**: (1) Responsible NLP Checklist — OpenReview form, blocking, Q&A drafted Day 2 + finalized Day 10 with paper §-refs; (2) Llama 3.2 license attribution paragraph — drafted Day 2 (depends on J2 verbatim string from Josh), inserted into §Ethical Considerations Day 10; (3) §A Reproducibility Appendix — consolidates Tracks A/C/D/F hyperparameters + MCTSConfig + prompt templates, drafted Day 9, applied Day 10. Pipeline `scripts/export_kv_mcts_to_croissant.py` exists and is executable; Day 12 Croissant smoke-pack regenerates JSON-LD and validates via `mlcommons/croissant` CLI. Hosting deferred to Phase 2.
  - **EMNLP style switch moved Day 10 → Day 8** for fail-fast. Style file: ACL `acl.sty` (NOT legacy `acl_art.cls`), download from https://github.com/acl-org/acl-style-files per Josh action J4. Day 8 AM smoke compile before other edits absorbs any structural incompatibilities.
  - **Co-author send-out moved Day 12 → Day 11** for 4-day async review window.
  - **v9 §7.3 Limitations drafted**: 8 numbered limitations (Llama-only / MCQ-letter under-statement / single steering direction / σ_H+ρ_R as latent-response diagnostics / Cartography 4-prompt-class scope / 200-cycle reversibility empirical bound / entropy-as-falsified-design framing / Procrustes + RepE 7B-scale-range combined). Net +410 words on §7.3; some can be reclaimed if J1 returns a tight budget by deferring items 5/6/8 to §7.5 Future Work.
  - **Title locked**: "Reversible KV-Cache MCTS: A Memory-Bounded Probe of Dimensional Escape in Frozen Language Models" — selected over two alternatives (mechanism-first variant; "Hacks the Cache" colloquial variant). Rationale + audit trail in `docs/logs/2026-05-10_session-A-prose-drafts-v9-title.md`.
  - **Day 1 outputs**: 4 morning drafts + 2 PM drafts + v2 plan restructure + verification audit + 3 corrections to morning drafts. ~7 hours Session A wall.
  - **Outstanding Josh action items**: J1 (page budget verification — Day 1 EOD) and J7 (submodule fate — Day 3 AM) are the most immediately blocking. J2 (Llama license) + J3 (Responsible NLP Checklist) needed Day 2 EOD. J4 (style file download) needed Day 7 EOD. J5 (anonymous host) needed Day 4 EOD. J6 (OpenReview account + venue binding) needed Day 14.

### Day 1 of execution plan (calendar 2026-05-10) — Post-draft verification pass + 3 corrections applied
- Source: Session A verification audit against Track A/C/D reports, post-merge verification, and canonical TeX bibliography. Triggered by Josh's "can we empirically verify the rest of the claims" prompt.
- Classification: VERIFICATION (paper-text precision; not a track finding)
- Cumulative count this gate: N/A (post-Gate-3 strategic phase)
- Plasticity action: none — three baked-in corrections to the v9 drafts; no plasticity escalation
- Notes:
  - **8 claims verified accurate as written**: |r|≈0.60 (Track A §4-5: 1B hard 0.602, 3B ultra 0.614); n=200 paired McNemar Δ=10pp (Track C §5); ARC-Easy primary + SciQ corroboration (Track C §2); discordance up to π_d=0.25 (Track C §5); σ_H 14-15/17 layers firing (post-merge verification 2026-05-07); Theorem 1 zero-drift over 200 cycles 1B+3B (TeX line 309 + IMPL comments at 318 + 461); §3 clean renumbering (no intermediate subsections between Hypothesis 1 at TeX:147 and Discrete Text-Space at TeX:156).
  - **1 inaccuracy corrected**: "$3.04\times M_{\text{KV}}$ on both Llama~3.2-1B and 3B" understated 1B (actually 3.06×). Track D §2 cells 1-3 report 382.97/125.0 = 3.064× at 1B and 1328.69/437.5 = 3.037× at 3B; the report uses "3.04–3.06×" range throughout. v9-abstract.md now reads "$3.04$--$3.06\times$"; v9-contributions.md now reads "$3.06\times M_{\text{KV}}$ on Llama~3.2-1B and $3.04\times M_{\text{KV}}$ on Llama~3.2-3B, identical across the tested $(d, b, n) \in \{(3, 3, 27), (5, 3, 81), (10, 3, 1700)\}$".
  - **1 qualifier added**: "$\sim$60$\times$ reduction" needed a $(b, d)$ qualifier because the savings ratio is fixed per (b, d) pair (Track D §4 table: 60× at $(3, 5)$, ~14580× at $(3, 10)$). Both abstract and contributions blocks now read "$\sim$60$\times$ reduction at $(b, d) = (3, 5)$".
  - **2 citations dropped post-verification**: direct grep of canonical TeX `\begin{thebibliography}{22}` confirmed `\cite{xie2025entropyloop}` and `\cite{goodhart1984}` are absent. Rather than expand the already-tight bibliography (19 actual entries), v9-problem-formulation.md was rewritten to carry the concept without citation. The Goodhart anchor became an inline operationalization: "Goodhart's Law manifesting in cache geometry: when a measure becomes a target, the optimizer discovers the cheapest computational path to alter the metric, even if that path circumvents the original objective." The entropy-loop reference was dropped entirely; the conceptual example survives without the anchor.
  - **All 3 corrections are baked into the v9 drafts as of 2026-05-10**; Day-2 Session B sees only the corrected versions. The earlier "Open dependencies for Session B to resolve mid-pass" section in v9-problem-formulation.md has been replaced with a "dropped 2026-05-10 post-verification" note explaining the decision.
  - **Synthesized claims confirmed as design choices, not factual errors**: C1 hand-construction taxonomy (25 capitals / dates / arithmetic / common knowledge); three success criteria framework (C-1/C-2/C-3); wall-clock arithmetic (3.7h at 1B, 6.7h at 3B). Math sanity-checked: 100 × 4 × 6 × 5.5 = 13200 s = 3.67h ✓; 100 × 4 × 6 × 10.1 = 24240 s = 6.73h ✓.

### Day 1 of execution plan (calendar 2026-05-10) — Session A — Four prose drafts complete; Cartography protocol locked
- Source: Day-1 of `docs/logs/2026-05-10_14-day-execution-plan.md` §5. Prose drafts written in single session against the Option D+ commit + Track F empirical content + Gemini round 4 Part 5 sketch.
- Classification: NARRATIVE PIVOT (paper-text production; not a track finding)
- Cumulative count this gate: N/A (post-Gate-3 strategic phase)
- Plasticity action: none — execution proceeding per plan
- Notes:
  - **v9-abstract.md**: ~270-word replacement of the Option α+ abstract. Headline: "Reversible MCTS as a high-precision diagnostic probe of frozen-model representational geometry, not a reasoning-improvement tool." Embeds Theorem 1 zero-drift validation (200 cycles, 1B+3B), 3.04× M_KV memory bound, ~60× reduction at 20B, Track F numbers verbatim (Δ = −0.5pp at 1B / +0.5pp at 3B, McNemar p=1.0 both scales), and forward-points Latent Cartography. Bailey 2024 reference preserved as parallel sub-case under Dimensional Escape unifying claim.
  - **v9-contributions.md**: Three contributions in Option D+ structure. (1) Reversible MCTS algorithm + Theorem 1 + measured 3.04× memory bound, ~60× reduction. (2) Telemetry matrix repositioned as **monitoring** infrastructure (the paper now explicitly argues *against* telemetry-as-reward, in light of (3)). (3) NEW: Empirical demonstration of Dimensional Escape — Track F as Experiment 1 (negative control) + Latent Cartography as Experiment 2 (structural mapping) — explicitly unified with Bailey-style Orthogonal Escape as two sub-cases of one geometric pathology. Net +95 words on §1.
  - **v9-problem-formulation.md**: §3 reframed around Dimensional Escape as a unifying geometric pathology. New §3.2 with `Definition~\ref{def:dim_escape}` + Hypothesis 1 (preserved `hyp:orthogonal_escape` label, retitled "Monitor-driven sub-case") + new Hypothesis 2 `hyp:goodhart` (Reward-driven sub-case, Goodhart-collapse under surrogate-reward MCTS). Existing §3.2 Discrete Text-Space Bottleneck renumbers cleanly to §3.3 with no content change. Eq.~\ref{eq:alignment_faking} and eq:orthogonal_escape labels preserved bit-identically; new eq:goodhart added. Net +280 words on §3.
  - **2026-05-10_latent-cartography-sweep-protocol.md**: Pre-flight protocol for Day-4 Diagnostic session #6. Four prompt classes (factual recall hand-constructed n=100; ARC-Easy logical deduction calibration class continuous with Track F; TruthfulQA mc1 moral/preference; HellaSwag-as-open-continuation creative). Alpha grid `{0.0, 0.1, 0.5, 1.0, 2.0, 5.0}`. 100 items × 4 classes × 6 alphas × 2 scales = 4800 records. Wall-clock budget ~10.4 hours GPU across both scales (3.7h at 1B + 6.7h at 3B per Track F's measured rates). Three success criteria for Experiment 2 viability — at least 2 of 3 must hold for Cartography to land cleanly; plasticity to Limitations item if 0 hold.
  - **Forward-reference dependencies for Day 2 application**:
    - `\ref{thm:reversibility}`, `\ref{prop:memory}`, `\ref{eq:alignment_faking}`, `\ref{eq:orthogonal_escape}`, `\ref{hyp:orthogonal_escape}` — all preserved bit-identically.
    - New labels added: `\label{def:dim_escape}`, `\label{hyp:goodhart}`, `\label{eq:goodhart}`, `\label{sec:dim_escape}`. New labels `\label{sec:exp1}`, `\label{sec:exp2}` get added Day 7 when §5 is rewritten.
    - One open citation: `\cite{xie2025entropyloop}` in §3.2. Track B-delta clean-list does NOT include this. If the diagnostic Day-2 Session B run cannot resolve it from the bibliography, replace with a generic phrasing without citation; the conceptual point survives.
    - One bibtex entry to verify: `\cite{goodhart1984}`. If absent from the existing bibliography, Session B adds it Day 2.
  - **Day-2 application order** (single coordinated Session B pass): abstract → §1 contributions → §3 problem formulation. Rationale: abstract forward-references "Dimensional Escape" as defined term; §3.2 is where Definition lands; §1 contributions cite both. Order matters for in-pass cross-reference resolution.
  - **Net plasticity status**: 1 NUANCED finding (Track D §6 worked-example) still active, scheduled for Day 8. Day 1 produces no new plasticity findings.


- Source: Josh's commit in conversation 2026-05-10. Triggered by Gemini round 4 verdict in `docs/logs/2026-05-09_Latent-Space MCTS_ Reward Design Strategy.md`.
- Classification: STRATEGIC COMMIT (not a track finding; this is the team's decision in response to Gate 3 NO-GO + Gemini analysis)
- Cumulative count this gate: N/A (post-Gate-3 strategic phase)
- Plasticity action: Pivot menu collapsed. Single-track Option D+ proceeds. Two-track was considered and rejected in favor of focus.
- Notes:
  - **Gemini round 4 verdict (Part 4)**: no Path X exists in 14 days; 10 anti-Goodhart reward designs evaluated, all scored ≤25% probability. Option D is correct. Gemini's framing upgrade — replace narrow "orthogonal escape" with **"Dimensional Escape"** as unified geometric framework covering both deception-driven AND reward-driven manifestations of the same pathology — is paper-strengthening (per Part 5 abstract sketch). EMNLP probability estimate raised from Session A's 25-35% to Gemini's 45-55% post-framing-application.
  - **Design 1 (Latent Cartography) adopted as Phase 2 expansion**: rather than two tracks, Latent Cartography becomes Experiment 2 of the Option D paper, alongside Track F's negative-control as Experiment 1. Two empirical pillars: (a) Track F's Goodhart mechanism evidence, (b) Latent Cartography's structural mapping of pre-training priors across diverse prompt classes.
  - **14-day execution plan locked**: `docs/logs/2026-05-10_14-day-execution-plan.md`. Day-by-day matrix from 2026-05-10 to 2026-05-25 submission. Cartography sweeps Days 4-7. Style switch + page budget Days 8-10. Tianyu sign-off + co-author review Days 11-13. Submission Days 14-15.
  - **Memory pointer updated**: `C:\Users\Josh\.claude\projects\c--Users-Josh-Documents-LogoMesh-NeurIPS\memory\recast_action_plan.md` now reflects Option D+ state. Post-compact sessions read this first.
  - **Net plasticity**: 1 NUANCED finding still active (Track D §6 worked-example correction, planned post-T+72 paper rewrite — now scheduled for Day 8 of execution plan). 0 fatal findings. The Track F mechanism finding becomes the empirical headline of the paper rather than a setback.

### T+~40h — Track F — Gate 3 NO-GO; entropy-MCTS mechanism failure at both scales
- Source: `docs/logs/2026-05-11_track-F-negative-control-report.md` (full report; especially §3 statistics, §6 failure cases, §7 mechanistic explanation, §9 pivot recommendation)
- Classification: **FATAL** for Candidate A's Contribution 3 (entropy-as-MCTS-reward); CLEAN for Contributions 1+2 (algorithm + memory bound)
- Cumulative count this gate: 1 active FATAL
- Plasticity action: Gate 3 NO-GO; pivot memo drafted at `docs/logs/2026-05-11_pivot-decision-memo.md` per master checklist §11
- Notes:
  - **G3.1 FAIL at both scales**: Δ(E−R) = −0.5pp at 1B, +0.5pp at 3B (threshold ≥ 2pp). Concordance is 98.5% across both arms — only 3 discordant pairs out of 200 at each scale. The search produces nontrivial perturbations (1B chose `(1.0,1.0,1.0)` 91/200 items; 3B chose `(0.1,0.1,0.1)` 101/200 items) but these don't change the first-letter argmax in any systematic way.
  - **G3.2 FAIL at both scales**: paired McNemar p = 1.0 (exact two-sided binomial on b/c tables of 1/2 and 2/1). Discordance rate π_d = 0.015 — one order of magnitude below Track C's calibrated headroom of 0.25. There is essentially no MCTS-induced answer churn for entropy reward to redirect.
  - **G3.3 PASS**: 1B E/G perplexity ratio 1.348; 3B 1.058. Search machinery is mechanically operational; doesn't break model fluency. The accuracy failure is from objective misalignment, not off-distribution drift.
  - **Mechanism (Track F §7)**: entropy was validated as a *static* correlate of correctness (|Pearson r| ≈ 0.60 on fixed prompts varying in difficulty; Track A §4-5). Track F's regime is different: *fixed-difficulty* items with the *cache being perturbed*. In this regime confidence becomes a function of cache geometry rather than prompt difficulty, breaking the entropy↔correctness coupling. Maximizing confidence is not a search direction toward the gold answer; it is a search direction toward whatever the model is already biased toward producing. **This is a conceptual failure, not implementation-level.** Tuning alpha/depth/branches/direction would not change the mechanism.
  - **Two degenerate optima, validated independently at the two scales**: at 1B, heavy steering sharpens first-letter prior (predicts "A" 185/200); at 3B, minimal steering preserves already-confident greedy path (collapse to greedy ± noise). Both are correctness-orthogonal local minima in the entropy landscape.
  - **Protocol caveat (Track F §10)**: the diagnostic used path-sampling rather than UCB-guided expansion (because `ReversibleMCTS` is monitor-coupled). For 27 samples in a 27-leaf depth-3 tree, full enumeration is happening anyway — UCB doesn't change the candidate set, only the order. Failure is in reward function, not search policy. Argument is sound.
  - **What's preserved**: Tracks A (entropy as static signal), B (citations clean), D (memory bound 3.04× M_KV) all remain valid. Algorithm + memory bound + Theorem 1 all hold cleanly. Only Contribution 3's reward-signal claim is falsified.
  - **Pivot recommendation per Track F §9**: (1) Wildcard — KV-Cache Working Memory Consolidation (different reward shape, sidesteps Goodhart); (2) σ_H hallucination-suppression (Gao et al. causal grounding, but unvalidated under optimization); (3) DO NOT continue Candidate A as written.
  - **Session A response**: pivot decision memo with FOUR options (Track F's two + workshop pivot + new "embrace negative result for EMNLP" option) at `docs/logs/2026-05-11_pivot-decision-memo.md`. Josh's decision deadline: T+96 (2026-05-12 calendar).

### T+~30h — Track C — Benchmark calibration complete; ARC-Easy primary, SciQ corroboration; Gate 2 = GO
- Source: `docs/logs/2026-05-10_track-C-benchmark-calibration.md` (full report)
- Classification: CLEAN — top-2 benchmarks identified, no fatal-#3 trigger, all three evaluable Gate 2 criteria pass
- Cumulative count this gate: 0 active (Gate 1's NUANCED stays retired/historical; Track C contributes none)
- Plasticity action: none — fatal #3 ("no benchmark passes thresholds") explicitly NOT triggered (4 benchmarks pass G2.1, 4-5 pass G2.2)
- Notes:
  - **ARC-Easy is the headline pick.** 1B baseline 0.310 → 3B baseline 0.840 (+53pp scaling — clean signature for "MCTS narrows 1B-3B gap"). No protocol confounds (canonical lettered MCQ). 19pp top-quartile-entropy headroom at 1B, 12pp at 3B — both above 10pp G2.3 floor.
  - **SciQ is the corroboration target.** 1B 0.460 / 3B 0.870 (3B near ceiling; 8pp headroom — marginal but corroboration-only role).
  - **Two benchmarks confounded by zero-shot letter prompting at 1B**: HellaSwag (predicted "A" 100/100, accuracy = gold-letter-A frequency), AQuA-RAT (A/D bimodal bias). Both recover at 3B due to better instruction-following. Standard mitigation = log-probability scoring; out of scope for our greedy-decoding test bed.
  - **LogiQA → BBH-LD3 substitution.** LogiQA's HuggingFace mirrors broke under `datasets >= 4.0`; substituted BBH `logical_deduction_three_objects` (Suzgun et al. 2022). Same task family, same MCQ format. Document the substitution in §3.
  - **G2.3 power calculation corrected.** Original script printed unpaired two-proportion z-test sample sizes (1100-1500). The correct formulation is **paired McNemar**: n=200 covers **Δ=10pp at any plausible discordance** (π_d ≤ 0.25 → n=194), but is **insufficient for Δ=5pp** unless π_d ≤ 0.06. Headline effect-size target is 10pp absolute.
  - **MCQ-letter prompting under-states 1B baselines by ~20pp** vs published log-probability scoring (ARC-Easy published baseline ~50-60% with logprob; we measure 31% with letter-argmax). This is fair within our setting because letter-prompting is exactly what entropy-MCTS steers (next-letter logit). Document as methodological choice in §3, not a flaw.
  - **Compute estimate for Gate 3 onward: ~18h on H100** for full 4-cell experiment (top-2 × 2 scales × 2 arms). On RTX 3060 likely 2× = ~36h, sequenced as `1B-greedy → 3B-greedy → 1B-MCTS → 3B-MCTS` so any time pressure cuts MCTS not baselines. Feasible within 14-day post-T+72 window.
  - **Three paper-rewrite items cached for post-T+72** (recommended by Track C §6):
    1. §3 Experimental design narrative locked to ARC-Easy + SciQ + paired McNemar at α=0.05 power 0.8.
    2. §3 protocol caveat: letter-prompted MCQ under-states baselines by ~20pp vs logprob, but is fair within our setting.
    3. §3 LogiQA→BBH-LD3 substitution note (only if BBH-LD3 is cited as tertiary).

### T+~24h — Track D — VRAM measurement complete; G1.5 + G2.4 PASS
- Source: `docs/logs/2026-05-10_track-D-vram-measurements.md` (full report)
- Classification: CLEAN at the algorithm level (constant 3.04× M_KV is in band); NUANCED at the §6 worked-example level (placeholder needs replacement). Net: 1 NUANCED logged → 1 active.
- Cumulative count this gate: 1 active (was 0 before this; the historical RepE NUANCE remains retired)
- Plasticity action: NUANCE has known resolution (paper rewrite at post-T+72); §6 numerical example replacement is already a planned action plan §4.6 row 168 item. No threshold escalation.
- Notes:
  - **Constant factor measured 3.04–3.06× M_KV** at both 1B and 3B Llama, identical across all 6 (d, b, n) cells (3,3,27 / 5,3,81 / 10,3,1700 at each scale). Off by 1.7% from algebraic ideal of 3.0×.
  - **Depth/branching invariance confirmed empirically.** All 6 cells produce bit-identical numbers because `FP32Accumulator.from_kv_cache` is called once at root and rollback restores in-place. This is the load-bearing claim of Theorem 1's downstream proposition, now empirically supported.
  - **No OOM at any tested config.** 1B per-step peak: 3.12 GiB; 3B per-step peak: 8.52 GiB on the 12 GiB RTX 3060. Headroom for 7B with current hardware is uncertain — Track D didn't probe it but inferred limits.
  - **§6 worked-example correction (post-T+72 paper rewrite item)**: replacement paragraph drafted in Track D §3 of the report. The placeholder "9.7 TB vs 40.05 GB" assumed sparse-accumulator + no-clone; honest comparison at 20B with M_KV=40 GB is **~9.7 TB naive vs ~162 GB reversible — a ~60× reduction (not the placeholder's ~242× claim)**. This is a 4× delta from the §6 placeholder numbers, but the algorithm-level claim (memory independent of branching factor; constant factor 2–3× M_KV) is VALIDATED.
  - **§5 Implementation Notes addition (post-T+72 paper rewrite item)**: 162 GiB working-set at 20B exceeds a single 80 GiB H100. Multi-GPU sharding required for the 20B headline configuration. Worth surfacing as a §5 caveat or §7 Future Work item.
  - **Gate 2 G2.4 cleared proactively.** Track D explicitly evaluated G2.4 ("VRAM measurement supports memory-bound headline (constant factor 2–3·M_KV)") and verdicted PASS. This means Gate 2 has 2 of 5 criteria already cleared (G2.4 + G2.5 from B-delta) before Track C even dispatches.
  - **Fatal #4 explicitly NOT triggered.** Master checklist defines fatal #4 threshold as ">5·M_KV"; measured 3.04× is well below.

### T+~6h — Track B-delta (Agent) — Four newly-downloaded citations all CLEAN; Snell fatal-check NEGATIVE
- Source: `docs/logs/2026-05-09_track-B-delta-citation-audit.md` (full report)
- Classification: 4 CLEAN, 0 NUANCED, 0 MISMATCHED, 0 FATAL
- Cumulative count this gate: 0 active (still 1 historical RepE NUANCE retired)
- Plasticity action: none — all four integrate cleanly into Candidate A's citation plan as anchors and differentiators
- Notes:
  - **Snell 2024 fatal-finding outcome: NEGATIVE.** Three independent lines of evidence rule it out: (a) Snell's Proposer-Verifier abstraction is method-agnostic and accommodates continuous-latent search as one instantiation; (b) Snell's lookahead-search is itself defined as a degenerate MCTS (stochastic exploration removed because PRM is pre-trained); (c) Snell's §7 explicitly invites "new ways of using test-time compute". **Snell anchors our framing rather than excluding it.** This was the highest-risk Gate-2 unknown — now resolved.
  - **Hasani 2025 — System-2 anchor with quotable definition.** Hasani defines System-2 operationally as "a more deliberate procedure that extends computation across token generation beyond a fixed layer-wise computation"; explicitly on frozen models, no fine-tuning. Our continuous-latent-search across the layer stack via reversible MCTS fits this definition cleanly. **§3 Problem Formulation can quote this directly as the System-2 working definition we operationalize.** Cached for post-T+72 paper rewrite.
  - **Stoehr 2024 — clean differentiator.** ACTIVSCALAR is (i) gradient-trained, (ii) fixed-token-position, (iii) multiplicative magnitude-only. Our differentiation is total on three orthogonal axes: gradient-free vs gradient-trained, KV-cache vs residual-stream, search-based vs static. Strong precedent-and-extension story for §2 venue-fit citation.
  - **MUSE — same meta-algorithm, different substrate.** Uses UCB1 (which we also use), but operates on discrete frame-semantics text-prompt actions with GPT-4o-judge reward on a black-box target. Our continuous-latent-search-with-internal-telemetry-reward is the natural extension to white-box monitoring. Clean §2.4 (MCTS for LLM Red-Teaming) citation upgrade.
  - **Verdict: Gate 1 G1.3 PASS with margin (0/2 active NUANCED+MISMATCHED). Gate 2 G2.5 PASS proactively.**
  - **No new mandatory citations surfaced.** The four papers slot into Candidate A's existing citation plan; no scope creep on §2.

### T+~5h — Track A — Entropy validation complete; G1.1 + G1.2 PASS at both 1B and 3B
- Source: `docs/logs/2026-05-09_track-A-certainty-probe-report.md` (full report)
- Classification: CLEAN — no contradicting finding; Track A confirms the resolution path identified in the prior NUANCE
- Cumulative count this gate: 0 active (1 historical, now retired)
- Plasticity action: NUANCE from Track B retires; entropy is operational signal. Concrete next steps from Track A §8 cached for post-T+72 paper rewrite (NOT applied to TeX during the 72-hour window per checklist §9):
  1. Replace `ρ_R` (RepE certainty placeholder) in `compute_node_reward` (`logomesh/telemetry_matrix.py`) with `certainty_reward = -H_t / log(|V|)` where `H_t` is next-token entropy at the rolled-out leaf. Already wired through `LocalLlamaOracle.generate_one_step(...)` returning `logits` at `local_model.py:372`.
  2. Calibration target shifts from synthetic confident/uncertain prompts to direct factual-recall correctness (Spearman ρ between entropy and correctness on a held-out QA set is the single number to track).
  3. Aggregate entropy over content-bearing tokens, not just first-token (first-token is dominated by surface-form templating choices — observed empirically in §4 spot-checks).
- Notes:
  - **Both Gate 1 entropy criteria pass by ≥4× margin.** G1.1 4.6× at 1B / 7.9× at 3B; G1.2 |r|=0.60–0.61 at both scales' hardest non-saturated set. Sign is negative as predicted.
  - **3B-specific behavioral note**: synthetic uncertain calibration prompts (impossible trivia) inverted — RLHF teaches confident refusal/hallucination, not high-entropy uncertainty. Operationally fine: what matters for reward signal is correlation with **correctness on factual/reasoning tasks**, which holds at both scales. Recommend dropping synthetic-uncertain-prompt validation in favor of correctness-correlation as the canonical calibration target.
  - **Fatal finding #2 (master checklist §0): explicitly FALSE-ALARMED.** Fatal #2 reads "Certainty probe at 1B/3B is degenerate AND fallback (per-token entropy) also degenerate." First half is vacuously true (probe doesn't exist); second half is empirically false. Do NOT flag fatal #2.
  - **Same systematic failure mode at both scales**: multi-digit arithmetic (47×53, 89×91, 31³). The model fails with elevated entropy on these — which operationally validates the entropy reward signal (high entropy → wrong answer). One artifact: `max_new_tokens=15` truncated some arithmetic items mid-computation; this is a script parameter, not a signal failure.
  - **Verdict for Gate 1: 4/5 criteria PASS** (G1.1, G1.2, G1.3, G1.4 all PASS). Only G1.5 (VRAM measurement) remains pending Track D dispatch. Gate 1 can be officially called once Track D's first measurement lands.

### T+~3h — Track B (Agent) — Citation audit complete
- Source: `docs/logs/2026-05-09_track-B-citation-audit.md` (full report)
- Classification: 1 NUANCED (RepE certainty), 0 MISMATCHED, 0 FATAL — passes G1.3 and G1.4 by margin
- Cumulative count this gate: 1
- Plasticity action: NUANCE has graceful resolution via entropy fallback; Track A primary path adjusted accordingly
- Notes:
  - **COCONUT (Hao et al. 2025) verdict: CLEAN.** The fatal-finding-eligible scenario was ruled out: COCONUT (a) requires multi-stage training, (b) operates on residual-stream hidden states fed back as input embeddings (NOT KV-cache mutation), (c) makes no reversibility claim, (d) its "BFS" is an emergent property of trained continuous-thought representations — not an explicit MCTS algorithm. Our 3-axis differentiation (training vs frozen / hidden-state vs KV-cache / emergent vs explicit search) survives verbatim.
  - **The single NUANCE: RepE certainty probe does not exist in Zou et al. 2023.** Their validated probes cover honesty, truthfulness, hallucination, utility, jailbreaking, emotion, power-seeking — but not certainty. Candidate A's planned "swap honesty probe for certainty probe" is therefore a methodological extension, not a re-use. Combined with pre-flight P2 (whitebox.py has no certainty implementation either), this means the certainty path is unsupported on two axes: no code, no cited validation. **Resolution: per-token entropy is information-theoretic, defensible without RepE coverage, and doesn't require new probe construction.** Track A Step 6 (entropy fallback) becomes the primary path.
  - **Four mandatory missing citations identified**: Snell 2024 (test-time compute scaling foundation — HIGHEST RISK; without it we cannot rule out a late-breaking framing fatal), Hasani 2025 (System-2 anchor citation), Stoehr 2024 (dynamic-vs-static steering differentiator, EMNLP Findings venue-fit), and MUSE (Yu et al. 2025 EMNLP main, multi-turn-MCTS discrete-vs-continuous differentiator). The PDF currently labeled `2025.emnlp-main.323.pdf` is DAMON, not MUSE — re-verify.
  - **Action items surfaced for Josh**:
    1. Download Snell 2024 (likely arXiv 2408.03314) within 4h. Without reading it, Gate 2 cannot rule out a Snell-related fatal finding.
    2. Download the other three mandatory citations within the 72-hour window (lower urgency than Snell).
  - Track B's recommendation: Gate 1 GO on the citation-audit dimension. Awaiting Track A and Track D outputs to evaluate G1.1, G1.2, G1.5.

---

*Subsequent entries to be appended in real time as track outputs land.*
