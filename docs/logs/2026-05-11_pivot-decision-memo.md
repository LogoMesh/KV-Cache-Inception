# Pivot Decision Memo — Gate 3 NO-GO

**Created:** 2026-05-09 (~T+40h-equivalent into the 72-hour matrix)
**Decision deadline:** T+96 (2026-05-12 calendar) per master checklist §11
**Author:** Session A (Opus, strategic synthesis)
**Status:** Awaiting Josh's decision

---

## 1. Why Gate 3 failed

Track F at 1B and 3B independently produced the same result: **entropy-MCTS does not beat random-MCTS by any meaningful margin on ARC-Easy.**

| Criterion | 1B | 3B | Verdict |
|---|---:|---:|---|
| G3.1 (Δ(E−R) ≥ 2pp) | −0.5pp | +0.5pp | FAIL both scales |
| G3.2 (paired McNemar p < 0.05) | p = 1.0 (3 discordant) | p = 1.0 (3 discordant) | FAIL both scales |
| G3.3 (perplexity E/G ≤ 2.0) | 1.348 | 1.058 | PASS both scales |

**Mechanistic diagnosis (Track F §7)**: entropy validated as a *static* correlate of correctness (|Pearson r| ≈ 0.60 on fixed prompts varying in difficulty) but fails as an *MCTS optimization target* because the reward has degenerate optima. At 1B, heavy steering reinforces the model's first-letter prior ("A" predicted 185/200). At 3B, minimal steering preserves greedy. Neither pathway corresponds to "find the cache state that produces the correct answer" — both are correctness-orthogonal confidence-maximizing pathways. **This is conceptual, not implementation-level.** Tuning alpha/depth/branches/direction would not change the mechanism.

This is a textbook Goodhart's-Law manifestation: a reward correlated with the target in observation does not imply the reward is causally tied to the target under optimization pressure.

## 2. What's preserved vs killed

| Element | Status |
|---|---|
| Reversible KV-Cache MCTS algorithm | ✅ Intact — Track D validated |
| FP32 Accumulator + zero-drift reversibility (Theorem 1) | ✅ Intact — empirically validated 200 cycles |
| Memory bound (3.04× M_KV measured; ~60× reduction vs naive parallel) | ✅ Intact — Track D measured |
| Telemetry matrix (σ_H + ρ_R) as static monitoring signal | ✅ Intact — Track A validated |
| 21+ citation pile-up CLEAN | ✅ Intact — Tracks B and B-delta validated |
| **Entropy-as-MCTS-reward (Contribution 3 of Candidate A)** | ❌ **EMPIRICALLY FALSIFIED** |
| Candidate A's "System-2 Latent Search" headline framing | ❌ **Falsified** — the headline empirical claim is the entropy-search-improves-reasoning result that didn't materialize |

## 3. Four pivot options

### Option A — Wildcard: KV-Cache Working Memory Consolidation

**Summary**: Apply Reversible MCTS to the prefill phase of long-context documents. Reward = retention of critical semantic information (measured by top-layer RepE probes) under selective forgetting. Same algorithm, same memory bound, different reward shape.

**Why it might work**: The reward (probe-measured information retention) is a different shape from entropy minimization. Goodhart pathology may not apply because the reward is not a confidence signal.

**Why it might not**: The wildcard's reward IS still a probe — same kind of correlation-not-causation-of-target situation. Optimizing the probe could still find degenerate paths. We have NO empirical evidence that this reward survives optimization any better than entropy did.

**Execution plan (14 days remaining)**:
- Days 1–3: Validate the new reward signal — Track A-equivalent (does probe agreement correlate with downstream task accuracy under perturbation?)
- Days 4–5: Validate optimization survival — Track F-equivalent (does the search beat random?)
- Days 6–10: If both validations pass, run full experiment on long-context QA at 1B/3B (NarrativeQA / RULER / similar)
- Days 11–14: Paper rewrite (§1, §3, §5, §6, §7 all change)

**Probability of EMNLP acceptance**: **15–25%**
- High validation risk: two diagnostic gates have to pass (Track A + Track F equivalents) before commitment
- Long-context benchmarks at 1B/3B may be capability-bottlenecked themselves
- Same Goodhart risk as Candidate A; we'd be guessing whether this reward avoids it

**Failure mode**: validation fails on Day 5 → no time for fallback → emergency pivot to workshop with whatever's salvageable.

### Option B — σ_H Hallucination-Suppression

**Summary**: Use MCTS to minimize σ_H (H-Neuron stress) on factual benchmarks. Reward grounded in Gao et al. (2025) causal claim about hallucination.

**Why it might work**: σ_H is causally tied to hallucination (per Gao et al.). Causal grounding might make Goodhart less severe.

**Why it might not**: Same fundamental mechanism as entropy — σ_H is a signal optimized via MCTS in latent space. Track F's finding generalizes: ANY signal correlated-with-correctness might have degenerate optima under optimization. We haven't validated σ_H under optimization. Worse: Track A surfaced σ_H as fragile (required Max's per-layer calibration fix; could collapse again).

**Execution plan (14 days remaining)**:
- Days 1–3: Validate σ_H-vs-correctness correlation (Track A-equivalent, didn't run for σ_H)
- Days 4–5: Validate optimization survival (Track F-equivalent)
- Days 6–10: If both pass, run on TruthfulQA / FEVER at 1B/3B
- Days 11–14: Paper rewrite

**Probability of EMNLP acceptance**: **10–20%**
- Same risk profile as A
- Worse priors: σ_H has known fragility AND we just learned signal-correlation-doesn't-survive-optimization
- Round 2 pre-mortem rated hallucination pivot at 20–25%; that estimate didn't account for the new Goodhart concern

**Failure mode**: same as A, but with worse base-rate.

### Option C — Workshop / Journal Pivot

**Summary**: Drop the EMNLP target. Submit algorithm + memory bound contributions to a more methodology-friendly venue: NeurIPS workshop (deadlines vary, typically July–September), ICLR systems track (October), or a journal (months-long cycle).

**Why it works**: The algorithm + Theorem 1 + memory bound are genuinely novel and standalone. NeurIPS / ICLR systems venues actively want this kind of contribution. Track F's mechanism finding becomes a Discussion / Limitations item.

**Why it's hard**: lower prestige than EMNLP main/Findings. Loses the May 25 deadline momentum (psychological cost). Workshop deadlines are 2–4 months out, so no immediate submission possibility.

**Execution plan**:
- Identify target workshop (NeurIPS 2026 workshops, deadlines TBD; or ICLR 2027 systems track Oct 2026; or journal — TMLR / JMLR)
- Polish current paper around algorithm + memory bound + Track F as honest finding
- Submit to whichever venue fits
- Total time: 4–8 weeks at relaxed pace

**Probability of acceptance**: **50–70%** (workshop) / 40–60% (journal)
- Algorithm contribution is real; workshops are more receptive
- Theorem 1 + memory bound are publishable on their own
- Lower competition than EMNLP

**Failure mode**: low risk of rejection; main risk is "this is just algorithm-centric work, not interpretability" reviewer at a workshop that values empirical findings.

### Option D — Embrace the Negative Result for EMNLP (NEW; not in original pivot menu)

**Summary**: Reframe the EMNLP submission. Title becomes "Reversible KV-Cache MCTS in Latent Space: A Memory-Bounded Algorithm and a Goodhart Failure Analysis on Frozen Models." Contributions: (1) Reversible KV-cache MCTS algorithm + Theorem 1, (2) Memory bound, (3) **Empirical demonstration that surface-level reward signals fail under optimization in latent KV-cache space, with a mechanistic Goodhart explanation validated at two model scales by independent degenerate optima.**

**Why it's strong**:
- Track F's empirical finding IS the contribution. We don't need new experiments. We have the data.
- Two-scale validation with independent degenerate optima is unusual evidence for a Goodhart claim. Not "we tested one reward and it failed"; rather, "we tested at 1B and 3B and the failure manifested via DIFFERENT mechanisms — first-letter prior reinforcement at 1B, greedy collapse at 3B — converging on the same conclusion." That's mechanism-level evidence.
- Solves Gemini Round 2's biggest critiques cleanly:
  - FM 1 (algorithmic foregrounding without NLP empirics) → Track F provides ARC-Easy + paired-McNemar empirics
  - FM 5 (no textual evaluation) → Track F has perplexity, accuracy, qualitative failure cases
  - FM 4 (developer diary) → Track F is a deliberate scientific finding, not bug-fixing narrative
  - FM 2 (1B/3B too small) → if the claim is "the technique fails at these scales," scale-triviality is part of the contribution
- Restores the "future-research story" requested in the original Gemini synthesis prompt: this paper opens the question "what reward signals DO survive MCTS optimization in latent space?" Plenty of follow-on work.

**Why it's risky**:
- Negative-result papers are harder to land at EMNLP (per Round 2 FM 1)
- Reviewer may say "you only tested one reward signal" — counter: the mechanism is general; Section 7 proves the failure is conceptual
- Reviewer may say "Goodhart is well-known" — counter: its specific manifestation in latent-space MCTS via two independent degenerate optima at different scales is novel

**Execution plan (14 days remaining)**:
- Days 1–4: Paper rewrite — abstract + §1 contributions + §3 problem formulation + §5 experiment 1 (becomes the Track F result) + §6 memory complexity (apply Track D corrections) + §7 discussion (mechanism + future work for reward design)
- Days 5–8: Citation integration + EMNLP style switch + page-budget compression
- Days 9–11: Co-author review + Tianyu sign-off
- Days 12–14: Submission prep + final REVISION marker audit + submit

**No new experiments needed.** This is the key advantage: 14 days is comfortable for paper rewrite alone.

**Probability of EMNLP acceptance**: **25–35%**
- Higher than Round 2's "post-mitigation 20–25%" estimate because Track F provides empirical NLP grounding the prior framing lacked
- Lower than Option C workshop because EMNLP is more competitive
- Specific risk: reviewer who reads "negative result" as "this paper doesn't deliver"; mitigated by §7 mechanism analysis

**Failure mode**: rejection from EMNLP → submit to workshop / journal (Option C) post-rejection. Track F + algorithm contributions still ship.

## 4. Comparison matrix

| Option | EMNLP probability | New experiments needed | Time pressure | Goodhart risk | Future-research seed |
|---|---:|---|---|---|---|
| A (Wildcard) | 15–25% | 2 validation gates + full run | Tight | High (unvalidated) | Strong if it works |
| B (σ_H) | 10–20% | 2 validation gates + full run | Tight | High (similar to A) | Modest |
| C (Workshop) | 50–70% (different venue) | None | Relaxed | N/A | Strong (algo) |
| **D (Embrace negative result)** | **25–35%** | **None** | Comfortable | N/A (negative result) | **Strong** (opens reward-design question) |

## 5. My recommendation

**Option D, with Option C as the rejection-fallback path.**

Reasoning:

1. **Track F's data IS a complete empirical contribution.** Two-scale validation with independent degenerate optima at each scale is unusual evidence for a Goodhart claim — strong enough to anchor a paper. We don't need to bet on a new reward signal that might also fail.

2. **No new experiments needed = maximum buffer for writing.** 14 days for paper rewrite alone is comfortable. Options A and B compress 14 days into 2 validation gates + full experimental run + paper rewrite — too tight if anything goes wrong.

3. **Better expected value than A or B.** A and B's 15–25% × commit-cost-of-bet is similar to D's 25–35% × no-new-bet-cost. D is strictly Pareto-better.

4. **The Goodhart finding is genuinely novel for this venue.** Mechanistic interpretability + RL/optimization-failure intersection is under-served at EMNLP. Track F provides exactly the kind of "method tested, failure characterized, mechanism explained" empirical contribution EMNLP values.

5. **Future-research story is strong**: "what reward signals DO survive MCTS optimization in latent space?" is a publishable open question. The team can pursue follow-on work without depending on the EMNLP submission to land.

6. **Option C is the natural rejection-fallback.** Even if EMNLP rejects D, the same content fits NeurIPS workshop / ICLR systems / TMLR cleanly. Negative-result + algorithmic contribution is a stronger workshop submission than algorithm-only.

**What I'd NOT recommend**: A or B without Track F-equivalent validation FIRST. Both repeat the methodological mistake Track F just exposed: assuming a signal that's correlated-with-correctness will survive optimization. We learned this is wrong; we shouldn't bet again on the same pattern.

## 6. Counter-considerations (in case my recommendation is wrong)

**If I'm wrong about Option D**:
- EMNLP reviewers may have stronger negative-result allergy than I'm modeling. Round 1+2 pre-mortems didn't surface this directly because we were pitching positive results. A focused pre-mortem of "submit a negative-result paper to EMNLP Findings" might surface different failure modes.
- The 25–35% estimate is uncertainty-laden. Could be 15% in reality.

**If I'm wrong about Option A**:
- The wildcard's reward shape may genuinely avoid Goodhart in ways I'm not seeing. RepE probe-based rewards are different in kind from entropy-based rewards (probes are low-rank linear projections; entropy is a softmax-derived scalar). Track F's mechanism may not generalize.

**If I'm wrong about Option C**:
- The user may have stronger emotional/professional reasons for EMNLP than I'm weighting. Workshop pivot might cost more in non-technical terms than the probability gap justifies.

## 7. Decision needed from Josh

By T+96 (calendar 2026-05-12):
- Pick A, B, C, or D
- If A or B: Session A immediately drafts Track A-equivalent + Track F-equivalent prompts for the new reward signal; you spawn diagnostic sessions
- If C: Session A drafts a workshop/journal target identification memo; we shift mode to relaxed-pace polish
- If D: Session A immediately drafts the rewritten abstract + §1 contributions + §3 problem formulation; we have 14 days for full paper rewrite

I'll execute whichever you pick. No hedging on my recommendation — D is what I think gives the best expected value for the 14 days remaining, but your call.

---

*End of pivot decision memo. Track F's mechanism finding is preserved as a publishable artifact regardless of which option is chosen.*
