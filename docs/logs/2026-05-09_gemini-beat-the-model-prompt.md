# Discovery Prompt: Beating the Model — Reward Design for Goodhart-Resistant Latent-Space MCTS

## Setup — what you're being asked

A research team has built a Reversible Monte Carlo Tree Search algorithm operating in the KV-cache latent space of frozen Llama-3.2-1B and 3B language models. The infrastructure works empirically. But the most natural test of whether the search produces measurable reasoning improvement on a standard NLP benchmark has just *failed cleanly* in a way that surfaces a fundamental obstacle — what the team is now calling a Goodhart's-Law manifestation in cache geometry.

They have **14 days to the EMNLP Findings 2026 deadline (2026-05-25, ARR portal)**. They want to know:

1. Is there a reward design that *survives* the obstacle and produces a positive result on a reasoning benchmark — making EMNLP submission viable on the original "improve reasoning via continuous-latent search" framing?
2. If not, is the right move to embrace the obstacle as the empirical contribution itself ("we discovered a Goodhart mechanism specific to latent-space MCTS on frozen LLMs") and ship that to EMNLP?
3. Or, third, drop EMNLP entirely and pivot to a workshop / journal venue?

Your job is to figure out which, with rigor and honesty.

---

## What you have access to (files uploaded with this prompt)

The user is uploading the following files alongside this prompt. Reference them by filename:

1. **`04.18.2026-NeurIPS-Research-Proposal-2-premortem-v2.tex`** — the latest paper draft. Reflects the "Candidate A" System-2 framing that Track F's failure has now falsified.
2. **`2026-05-11_track-F-negative-control-report.md`** — *the central document.* Read in full. §3 has the headline statistics, §6 has failure cases, §7 has the mechanistic Goodhart explanation, §9 has the team's pivot recommendation, §10 documents protocol caveats.
3. **`2026-05-11_pivot-decision-memo.md`** — the four-option pivot menu the team is currently weighing.
4. **`2026-05-09_track-A-certainty-probe-report.md`** — Track A. Establishes that entropy correlates with correctness STATICALLY (|Pearson r| ≈ 0.60). This was the validated foundation that motivated the doomed Track F design.
5. **`2026-05-08_EMNLP Framing for KV-Cache MCTS.md`** — the original "Candidate A" framing doc whose Contribution 3 Track F just empirically falsified. Has the System-2 reasoning paradigm pitch and the wildcard pivot (Part 8).
6. **`2026-05-08_72hr-execution-checklist.md`** — the master execution plan. Especially §0 (plasticity rules) and §4.3 (Gate 3 criteria). Establishes that the team commits to gate decisions based on falsifiable evidence, not vibes.
7. **`2026-05-07_gemini-pre-mortem-EMNLP-findings.md`** + **`2026-05-08_gemini-pre-mortem-EMNLP-findings.md`** — two prior Gemini pre-mortem rounds. Treat as validated diagnoses; don't re-derive their findings.
8. **`2026-05-09_track-B-citation-audit.md`** + **`2026-05-09_track-B-delta-citation-audit.md`** + **`2026-05-10_track-C-benchmark-calibration.md`** + **`2026-05-10_track-D-vram-measurements.md`** — supporting empirical evidence for Tracks B/C/D.

If a file you want is not in the upload list and the team's reasoning depends on something you can't verify from the uploads, **flag it as a constraint** rather than guessing.

---

## Critical facts at a glance (so you don't have to dig)

**The infrastructure that works:**
- Reversible Monte Carlo Tree Search in KV-cache latent space. Frozen LLM; in-place arithmetic mutations to cached K/V tensors; exact rollback via FP32 accumulator (Theorem 1: zero drift over 200 cycles, validated on Llama-3.2-1B and 3B).
- Memory bound: 3.04× M_KV measured (Track D). ~60× reduction vs naive parallel-cache MCTS at 20B configurations. Independent of branching factor.
- Per-layer telemetry matrix fusing σ_H (H-Neuron stress, Gao et al. 2025; causally tied to hallucination) + ρ_R (RepE honesty projection, Zou et al. 2023). Both validated as static signals.
- Per-token next-token entropy: |Pearson r| ≈ 0.60 with correctness on factual recall, both 1B and 3B, validated by Track A on 30+ items per scale.
- 21+ citations CLEAN per Tracks B + B-delta. No fatal findings. One historical NUANCED (RepE certainty probe doesn't exist) retired by Track A's entropy fallback validation.

**The Track F failure (the central thing):**
- Test design: ARC-Easy 200 paired items at 1B and 3B. Three conditions per item:
  - **G**: greedy decoding (anchor)
  - **R**: random-perturbation MCTS (uninformative reward — uniform[0,1] per node)
  - **E**: entropy-guided MCTS (reward = `−H_t / log|V|`, the formula validated by Track A as a static correlate)
  - All three arms use identical search machinery, same seed, same steering direction. Only the reward function differs between R and E. This is a clean COCONUT-style negative-control design.
- Configuration: branches=3, depth=3, n_nodes=27 (full enumeration of the depth-3 tree).
- Result, both scales:
  - **G3.1 (Δ ≥ 2pp absolute, E vs R): FAIL.** 1B: Δ = −0.5pp. 3B: Δ = +0.5pp. Both well inside noise.
  - **G3.2 (paired McNemar p < 0.05): FAIL.** 1B: p = 1.0 (3 discordant pairs). 3B: p = 1.0 (3 discordant pairs). Concordance is 98.5% across both arms — only 3 of 200 items differ between random-MCTS and entropy-MCTS. Discordance rate π_d = 0.015, *one order of magnitude* below the calibrated power-budget headroom of 0.25.
  - **G3.3 (perplexity ratio E/G ≤ 2.0): PASS.** 1B: 1.348. 3B: 1.058. Search machinery is mechanically operational; doesn't break model fluency. The accuracy failure is from objective misalignment, not off-distribution drift.

**The mechanism diagnosis (Track F §7), validated independently at two scales by different routes:**
- The reward says "minimize entropy." The cheapest path to low entropy isn't "find the right answer"; it's "amplify what the model is already biased toward."
- **At 1B**: the model's strongest reflex on letter-MCQ is "predict A" (alphabetic prior). Heavy steering sharpens this. The search picks `(α=1.0, 1.0, 1.0)` cumulative steering 91/200 times (46%). Entropy-MCTS predicts "A" on 185/200 items. *The search has weaponized the model's prior.*
- **At 3B**: the unperturbed cache is already low-entropy due to better instruction-following. Heavy steering destabilizes it. The search picks `(α=0.1, 0.1, 0.1)` minimal steering 101/200 times (50.5%). *The search collapses to greedy ± noise.*
- In neither regime does the search's "lower entropy" objective select for "more likely correct." Maximizing confidence is not a search direction toward the gold answer; it is a search direction toward whatever the model is already biased toward producing.
- **Conceptual, not implementation.** No tuning of alpha set, depth, branches, or steering direction would change the basic mechanism. This is a Goodhart's-Law manifestation: a reward correlated with correctness in observation does not survive optimization pressure because there are correctness-orthogonal degenerate optima the search finds first.

**Static-vs-dynamic regime gap (what Track F revealed about Track A):**
- Track A's |Pearson r| ≈ 0.60 was measured on *fixed prompts that vary in difficulty* (some are easy, some are hard; harder prompts have higher entropy and are more often wrong).
- Track F operates on *fixed-difficulty items where the cache is being perturbed in arbitrary directions*.
- These are different regimes. The first measures "do you tend to be wrong when you're uncertain?" — yes, naturally. The second measures "if I make you more confident by manipulating your internal state, are you more likely to be right?" — no, because confidence-manipulation finds degenerate optima.

---

## The connection to "orthogonal escape" (potentially paper-strengthening)

The team's original paper hypothesis (before Candidate A) was the **Orthogonal Escape Hypothesis**: under corrective pressure (monitoring), a deceptive model redistributes its representations into the null space of the monitor. This is a goal-directed-deception mechanism.

What Track F surfaced is structurally similar but *not* about deception. The latent space has paths satisfying a measured metric (entropy) without satisfying the underlying goal (correctness). Both are "the latent geometry has alternative paths the optimization finds first."

**Question for your analysis (Part 5 below)**: could the paper legitimately claim it surfaces TWO instances of the same geometric pathology (one deception-driven, one reward-driven), positioning the algorithm as a general probe of cache-space orthogonal escape? Or is that overreach?

---

## Operating assumptions you must adopt

1. **Track F's mechanism diagnosis is correct.** Do not re-litigate. Do not argue it's a hyperparameter issue. Do not assume tuning rescues it. The team's diagnostic was rigorous, the mechanism is conceptual, the failure manifested by *different routes* at the two scales — that's mechanism-level evidence, not noise.
2. **The team has a working algorithm + memory bound + Theorem 1 contribution that is intact regardless of Track F.** Whatever you propose preserves these.
3. **The 14-day timeline is real.** Any proposed reward signal must be (a) validated for non-degeneracy [Track A-equivalent: ~3 days], (b) validated for optimization-survival [Track F-equivalent: ~2 days], (c) deployed in the headline experiment [~3-5 days], and (d) written into a paper [~5 days]. **Total: 13-15 days.** If you propose a plan that requires more than 14 days, say so.
4. **The user wants to "beat the model".** Stated goal: produce a positive result on reasoning improvement that survives optimization pressure. **NOT** "explain why nothing works"; **NOT** automatically "embrace the negative result." Find a path if one exists.
5. **You are explicitly invited to think outside the box.** The team's Session A surfaced ideas (ensemble agreement, external verifier, self-consistency, counterfactual sensitivity, layer-specific perturbation, anti-prior penalty, information-bottleneck reward, adversarial probing, Best-of-N+MCTS hybrid). Treat these as starting points, not constraints. If you have a wholly different direction nobody has considered, surface it.
6. **Be honest about probability.** If after deep analysis the answer is "no reward design plausibly beats this in 14 days, recommend Option D (embrace the negative result for EMNLP)," **say that plainly**. The user has explicitly authorized this answer. False optimism is worse than honest pessimism.

---

## Deliverables

A **strategy report** organized as follows:

### Part 1: Mechanism analysis (1 page)

Pressure-test Track F's §7 diagnosis. Specifically:
- Is **Goodhart's Law** the right diagnostic frame here, or is something else going on (e.g., representation collapse, instruction-following overfit at 1B, multi-modal entropy basins in the unperturbed cache, cumulative-steering nonlinearity)?
- Are there latent-space MCTS settings where entropy DOES survive optimization? Specific layer subsets, calibrated steering directions, specific token positions, specific depth/branching combinations?
- Is the failure regime-specific (only on letter-prompted MCQ at 1B/3B) or fundamental (any reward correlated with correctness, any benchmark, any frozen LLM)? **Defend your answer with reasoning, not assertion.**

### Part 2: Anti-Goodhart reward designs (3-5 pages)

For each candidate reward design, evaluate using this template:

- **Mechanism**: what is the reward function and *specifically why* might it survive Goodhart pressure where entropy didn't?
- **Validation cost** (Track A-equivalent): how many days to confirm the signal is non-degenerate?
- **Optimization-survival cost** (Track F-equivalent): how many days to confirm it survives MCTS optimization?
- **Full experiment cost**: how many days for the headline experiment on ARC-Easy or alternative?
- **Total time fit in 14 days**: pass / fail.
- **Failure mode**: if Track F-equivalent shows this also fails, what's the most likely Goodhart pathway?
- **Probability** of producing a +10pp improvement (the Track C-locked headline target): your estimate, defended.

Cover at minimum these candidates plus your own:

1. **Multi-channel ensemble agreement** — reward = simultaneous agreement across entropy + σ_H + ρ_R + perplexity (or other independent signals). Goodhart presumably harder if degenerate paths must satisfy all signals simultaneously.
2. **External judge model** — reward from a separate small LLM verifier or trained scorer. Search can't steer the judge's weights. Snell et al. 2024's Process Reward Model approach is one instantiation.
3. **Self-consistency** — reward = answer agreement across multiple stochastic rollouts from the perturbed state (Wang et al. self-consistency CoT). Real reasoning converges; degenerate confidence diverges under temperature.
4. **Counterfactual sensitivity** — reward = answer variance under opposite-direction perturbations. Maps the "fragility surface" of items the model is uncertain about; identifies items where the answer is unstable.
5. **Layer-specific perturbation** — perturb only middle layers (per Zou et al. 2023 RepE finding) or only late layers (output-aligned). Map where reasoning-relevant computation happens. Might naturally avoid prior-amplification.
6. **Anti-prior penalty** — reward = entropy minimization MINUS a penalty for amplifying the model's prior token distribution. Explicitly fight the failure mode Track F surfaced. (Risk: penalty term could itself become Goodhart-able.)
7. **Information-bottleneck reward** — reward = mutual information between prompt and final answer through the perturbed cache. Forces the search to find caches that *use* the prompt rather than ignore it.
8. **Inverted goal — adversarial probing** — search for the cheapest perturbation that *flips* the answer rather than amplifies confidence. Maps vulnerability instead of improving accuracy. Different positive-result claim entirely (interpretability rather than reasoning improvement). Track F's mechanism is *useful* in this framing, not an obstacle.
9. **Best-of-N + MCTS hybrid** — generate N greedy rollouts, apply MCTS perturbations to each, pick the best. Combines verifier-style safety with continuous search.
10. **Anything else you think of.** (Strongly invited.)

Tabulate the comparison at the end.

### Part 3: Outside-the-box experimental designs (1-2 pages)

The user explicitly asked for outside-the-box experiments. Surface 2-3 experimental designs that:
- Use the existing infrastructure (Reversible MCTS, FP32 accumulator, telemetry matrix, 1B/3B Llama, ARC-Easy or analog)
- Avoid the Goodhart pathology Track F surfaced
- Could plausibly land at EMNLP

These need not be "different reward design"; they could be different experimental questions entirely. For example:

- **MCTS as interpretability tool, not improvement tool** — map the structure of the reasoning manifold at 1B vs 3B by characterizing where the search converges and what those convergence points reveal about model organization.
- **MCTS as a dataset generator** — produce paired (prompt, perturbed-cache, output) tuples that could distill into a smaller fine-tuned model. Knowledge transfer angle.
- **MCTS as a probe of model priors** — Track F's "failure" actually reveals strong empirical content about *what 1B/3B Llama models are biased toward*. The 1B "predicts A 185/200 times under heavy steering" is a finding about the prior structure, not just a methodological setback. Could this be the paper itself?
- **MCTS for monitor stress-testing** — go back to the paper's original §7.1 framing (revised under recast as "Implications for Latent-Space Monitor Design"), but use Track F's empirical content as the evidence: the search finds cache states that fool linear probes (entropy is a linear-monitor proxy) at predictable rates. This is empirical orthogonal-escape evidence under reward-driven search.

### Part 4: Honest verdict (1 page)

Three options exist, plus your own potential addition:

- **Path X (yours)**: a specific reward design + benchmark + experimental plan that you assess as having ≥ 30% probability of producing a positive result in 14 days. Recommend one specifically.
- **Option D (embrace the negative result)**: ship the Goodhart-mechanism paper. The team's current default fallback. Estimate probability of EMNLP acceptance and compare to Path X.
- **Option C (workshop / journal pivot)**: drop EMNLP. Algorithm + memory bound stand alone at NeurIPS workshop / ICLR systems / TMLR. Estimate probability.

If Path X exists with > Option D's probability after honest analysis, recommend Path X with a 14-day execution plan. If not, recommend D and explain why no path X exists in the available time.

**Anti-rubber-stamping rule**: if your honest answer after deep analysis is "no reward design plausibly beats Track F's mechanism in 14 days, the team's Option D is correct," **say so plainly**. The user has explicitly authorized this answer.

### Part 5: Connection to the original orthogonal-escape framing (1 page)

The paper's original hypothesis was about deceptive models redistributing representations to evade *monitors* under corrective pressure. Track F surfaced a structurally similar phenomenon under *reward-optimization* pressure: the search finds cache states satisfying a measured metric without satisfying the underlying goal.

Evaluate:
- Is "orthogonal escape" the right framing for the broader phenomenon, or is it too specific to deception?
- Could the paper legitimately claim it surfaces TWO instances of the same geometric pathology (deception-driven AND reward-driven orthogonal escape), with the algorithm positioned as a general probe of cache-space orthogonal escape rather than as a reasoning-improvement tool?
- Would EMNLP reviewers buy that claim, or read it as overreach? Specifically: would the reviewer cohort that handles mechanistic interpretability + alignment + activation-steering at EMNLP/ACL recognize this as a coherent unifying framing, or as a post-hoc rescue of a failed contribution?
- If the team pursued this framing under Option D, what would the strongest version look like? Sketch the abstract + §1 contribution structure.

### Part 6: Wildcard recommendation (1 paragraph)

One concrete recommendation that none of the prior four Gemini sessions, two Claude sessions, the user, or the team's Session A surfaced. Could be conservative (a small reframe) or adventurous (a structural reconceptualization). Justify in 3-4 sentences.

---

## Anti-anchoring rules

- **Track F's mechanism diagnosis is correct.** Build on it; don't re-litigate.
- **The pivot menu is not exhaustive.** If the right answer is a fifth option, name it.
- **Path X must be costed honestly.** A 14-day plan that requires 4 days of validation + 4 of experiment + 4 of writeup leaves 2 days of buffer. If anything goes wrong, the plan fails. Surface that risk in your evaluation, not in fine print.
- **"Beat the model" is the user's stated goal but not necessarily the right goal.** If the right goal is "produce the best possible paper for EMNLP given Track F's finding," the answer might be Option D not Path X. Be explicit about which question you're answering.
- **No invented strengths.** If the team's existing infrastructure has a property you'd want for Path X but doesn't have, surface that as a constraint, don't paper over it.
- **Cite specific papers** for any anti-Goodhart technique you propose. Where possible, give title + venue + year + arXiv ID. If you cite a method that doesn't exist as you describe it, the team will catch it during their citation-verification track and your credibility will degrade.
- **Probability estimates are estimates, not facts.** Use ranges. Don't claim 47% precision. But don't hide behind ranges either — commit to a most-likely value with the range as honest uncertainty.

---

## Output format

Markdown. ~6-10 pages depending on Path X complexity. Use tables for the reward-design evaluations in Part 2. Cite specific papers with title + venue + year + arXiv ID. URLs welcome.

---

## A note on the team's epistemics so you can calibrate

The team has run two prior Gemini deep-research pre-mortem rounds and one Gemini deep-research synthesis round (total 3 prior Gemini sessions on this paper) plus extensive Claude Opus diagnostic and synthesis work. They have a 72-hour falsification matrix with three GO/NO-GO/PIVOT gates and explicit pile-up thresholds. They just called Gate 3 = NO-GO based on Track F's evidence and are now considering pivot options with the discipline of someone who has already disconfirmed their first three hypotheses. **They will respect a pessimistic answer** if you defend it rigorously. They will discount an optimistic answer that ignores the evidence in Track F. Pitch your output for that audience.

---

The team has been running ahead of schedule, has 14 days remaining, and has built infrastructure that empirically works. They want to know whether the model can be beaten in this regime, or whether the right play is to embrace what Track F discovered. Take the question seriously and answer it honestly.
