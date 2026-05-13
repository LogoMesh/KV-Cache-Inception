You are a diagnostic Claude session spawned after Gate 2 = GO (called 2026-05-09). Your single task is to determine whether entropy-guided MCTS beats random-perturbation MCTS on ARC-Easy cleanly enough to defend the Candidate A framing under adversarial review. This is the "are we sure this is reasoning, or just numerical noise?" check.

**LOCKED-IN CONFIGURATION (per Tracks A and C; do not deviate):**
- **Benchmark:** ARC-Easy primary (test split, first 200 deterministic items, indices [0..199]). SciQ corroboration if compute budget permits (same protocol; deterministic per-item rotation via `hash(question) mod 4`).
- **Reward signal:** entropy-based, `certainty_reward = -H_t / log(|V|)` where `H_t` is the next-token entropy of the rolled-out leaf, normalised by `log(vocab_size)`. The certainty probe does NOT exist (Track A §1 + pre-flight P2 confirmed); RepE certainty was never validated by Zou et al. (Track B audit). Entropy is the validated operational signal — |Pearson r| ≈ 0.60 vs correctness at both 1B and 3B (Track A §4-§5).
- **Aggregation:** mean per-token entropy over content-bearing tokens (skip the very first token which is dominated by surface-form templating; Track A §6 limitation).
- **Headline target effect size:** Δ = **10pp absolute** (paired McNemar α=0.05 power 0.8 reachable on n=200 paired observations across discordance up to π_d ≤ 0.25; Track C §5 corrected formulation). Δ = 5pp is reportable as a secondary, lower-confidence claim if discordance is small.
- **MCQ prompting protocol:** chat-template with system + user; user contains question + lettered options (A/B/C/D); "Answer with the single capital letter (A/B/C/D):" suffix. Same protocol Track C used. Letter-prompting under-states 1B baselines by ~20pp vs logprob scoring, but is fair within our setting because that's what MCTS actually steers.

CONTEXT FILES:
1. docs/logs/2026-05-08_72hr-execution-checklist.md (master checklist; §4.3 Gate 3 criteria — these are what you must satisfy)
2. docs/logs/2026-05-09_track-A-certainty-probe-report.md (entropy signal validation; §8 has the concrete reward formula)
3. docs/logs/2026-05-10_track-C-benchmark-calibration.md (ARC-Easy chosen + 200-example subset + paired McNemar power calc)
4. docs/logs/2026-05-10_track-D-vram-measurements.md (VRAM envelope — fits on RTX 3060 at 1B and 3B for d≤10, b≤3)
5. logomesh/kv_mcts.py + logomesh/telemetry_matrix.py (the algorithm + reward function)
6. scripts/run_kv_mcts.py (runner)
7. scripts/diagnose_track_a_entropy.py + scripts/diagnose_track_c_benchmarks.py (reference for chat-template + entropy + ARC-Easy loading patterns)

PROTOCOL:
1. **Benchmark:** ARC-Easy first 200 deterministic test items (indices [0..199]), per Track C §3. SciQ corroboration is OPTIONAL — only run if total wallclock budget permits.
2. **THREE conditions** per example, all on the same item with the same chat-template prompt:
   a. **Greedy baseline:** `model.generate(do_sample=False, max_new_tokens=5, output_scores=True)`. No MCTS. Anchor for both pairwise comparisons.
   b. **Random-perturbation MCTS:** invoke `kv_mcts.py` MCTS engine, but replace the reward function with `reward = uniform(0, 1)` per node (search exists, reward provides no signal). This is the COCONUT-defense control — it isolates "does the search machinery itself help" from "does the reward signal help".
   c. **Entropy-guided MCTS:** invoke same MCTS engine with reward = `-H_t / log(|V|)` per Track A §8 (mean entropy over content-bearing tokens of the rolled-out leaf, vocab-normalised).
3. **Configuration:** `branches=3, depth=3, n_nodes=27` (modest — signal-finding not compute saturation). Per-example MCTS wallclock budget: ≤30 sec at 1B / ≤60 sec at 3B on RTX 3060 (Track D measured ~3.12 GiB at 1B / ~8.52 GiB at 3B; well within 12 GiB envelope).
4. **Per-example records:** condition, final answer (decoded letter), correctness (binary, case-insensitive substring match against gold letter), perplexity of the 5-token completion, per-token entropy at the terminal node, mean reward across MCTS rollouts (random arm: should be ~0.5 ± noise; entropy arm: should reflect actual entropy distribution).
5. **Aggregate stats:**
   a. Per-condition accuracy + 95% CI (Wilson score interval).
   b. Per-condition mean perplexity ± std.
   c. **Paired McNemar test** for pairwise accuracy comparisons (entropy-MCTS vs random-MCTS, entropy-MCTS vs greedy). Use exact McNemar for small discordant counts; use χ² approximation otherwise. Report b/c table, p-value, effect size (Cohen's h for proportions).
   d. Cohen's h ≥ 0.10 corresponds to ~5pp absolute on baseline 0.31; report this for interpretability alongside p-value.
6. **Both 1B and 3B** if budget permits. If forced to choose, prioritise 1B (more headroom for MCTS to demonstrate; 3B is near-ceiling on ARC-Easy at 0.840).

OUTPUT FILE: docs/logs/2026-05-11_track-F-negative-control-report.md

REQUIRED REPORT SECTIONS:
1. Run configuration (benchmark, subset size, MCTS config, model scale, hardware)
2. Per-condition aggregate table: condition | accuracy | 95% CI | mean perplexity ± std | wallclock per example
3. Pairwise comparison: **entropy-MCTS vs random-MCTS** (this is the COCONUT-defense — the headline statistical claim)
   - Absolute accuracy difference (Δ)
   - Paired McNemar test: b/c table, p-value, exact or χ² flag
   - Effect size (Cohen's h for proportions)
   - Verdict on G3.1 (Δ ≥ 2pp) and G3.2 (p < 0.05)
4. Pairwise comparison: entropy-MCTS vs greedy (secondary)
   - Same statistics
5. **Perplexity comparison: entropy-MCTS vs greedy.** Compute ratio (mean_pplx_mcts / mean_pplx_greedy). G3.3 PASS if ratio ≤ 2.0.
6. Failure-case examples: 5 cases where entropy-guided got the answer right but random failed (defends the reward-signal claim); 5 cases where entropy-guided got it right and greedy failed (defends the search-helps claim). For each, include the question, gold answer, condition outputs, and the entropy values that drove the search.
7. Pass/fail verdict per Gate 3 criteria G3.1, G3.2, G3.3 with explicit numbers.
8. Plasticity recommendation:
   - If entropy beats random by Δ ≥ 10pp absolute and p < 0.05: HEADLINE result, full Gate 3 GO
   - If entropy beats random by Δ ≥ 2pp absolute and p < 0.05 but Δ < 10pp: SCOPE-DOWNGRADE Gate 3 (per master checklist §4.3 YELLOW)
   - If entropy does not beat random by Δ ≥ 2pp OR not significant: Gate 3 NO-GO; recommend pivot to wildcard or hallucination per master checklist §4.3
9. Notes on any anomalies (e.g., per-example wallclock outliers, MCTS terminating without producing a different answer than greedy, etc.)

DO NOT:
- Run more than 200 examples per condition (200 is the budget; matches Track C subset)
- Use a different benchmark than ARC-Easy (Track C locked this)
- Skip the perplexity measurement (it's how we defend against the "you broke the model" critique in §5)
- Use a probe-based reward (no probe exists — entropy is the locked signal)
- Defer documentation
- Pad the writeup to make a weak result look stronger
- Confuse paired vs unpaired statistical tests — McNemar is the right test for our paired design (Track C §5 was explicit about this)
