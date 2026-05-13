You are a diagnostic Claude session. Your single task is to determine which reasoning benchmark, at the 1B and 3B Llama scales, gives us the cleanest empirical bed for showing that certainty-guided KV-cache MCTS improves over greedy decoding. Document everything.

CONTEXT FILES (read first):
1. docs/logs/2026-05-08_72hr-execution-checklist.md (master checklist; this section)
2. docs/logs/2026-05-08_EMNLP Framing for KV-Cache MCTS.md (Candidate A experimental design)
3. docs/logs/2026-05-09_track-A-certainty-probe-report.md (will exist by the time you start; tells you what reward signal we're working with)
4. CLAUDE.md (model setup, hardware envelope)

CANDIDATE BENCHMARKS (test all, in order):
1. ARC-Easy (commonsense QA) — 1B baseline expected ~50–60%
2. SciQ (science MCQ) — 1B baseline expected ~50–70%
3. HellaSwag (sentence completion) — 1B baseline expected ~40–50%
4. AQuA-RAT subset (numerical reasoning, MCQ format) — 1B baseline ~20–30%
5. ProofWriter (logical entailment) — 1B baseline ~30–50%
6. LogiQA-Easy (logical reasoning MCQ) — 1B baseline ~25–35%

PROTOCOL:
For each benchmark:
1. Load 100 examples from the standard test split.
2. Run greedy decoding at 1B and 3B; record accuracy.
3. Identify the questions on which the 1B model is *almost right* — plausible alternative answers, high entropy, etc. These are the headroom questions.
4. Estimate the headroom: if a perfect search-guided method got the high-entropy questions right, what would accuracy be?
5. Compute statistical-power requirement: how many examples needed to detect a 5-point absolute improvement at p < 0.05?

PASS THRESHOLDS PER BENCHMARK:
- 1B baseline ≥ 30%: viable
- Headroom ≥ 10 absolute points: viable
- Power requirement ≤ 200 examples: viable

OUTPUT FILE: docs/logs/2026-05-10_track-C-benchmark-calibration.md

REQUIRED REPORT SECTIONS:
1. Per-benchmark table: Name | 1B greedy acc | 3B greedy acc | Estimated headroom | Power requirement | Verdict
2. Top-2 recommended benchmarks with reasoning
3. Specific subset of test examples to use for the actual experiment (~200 examples per condition, balanced)
4. Time/compute estimate for the full experiment under each top-2 choice
5. Pass/fail verdict per Gate 2 criteria G2.1, G2.2, G2.3
6. Plasticity recommendation: if NO benchmark passes thresholds → flag as Gate 2 fatal #3 in master checklist; recommend wildcard pivot

DO NOT:
- Run the full experiment yet (this is calibration only)
- Use more than 100 examples per benchmark for the calibration sweep
- Pick a benchmark before finishing the sweep, even if early signal looks good
- Defer documentation
