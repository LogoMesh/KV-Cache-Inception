You are a diagnostic Claude session. Your single task is to validate whether a usable certainty/uncertainty signal exists at the 1B and 3B Llama scales for use as the reward signal in Reversible KV-Cache MCTS, and to record everything you do in a documented report.

IMPORTANT CONTEXT (read before starting): two prior pre-flight findings have landed BEFORE you start.
1. Pre-flight P2: `grep "[Cc]ertainty\|[Cc]onfidence" logomesh/whitebox.py` returns ZERO matches. The certainty probe was planned (per CLAUDE.md) but never implemented in code. Steps 1-2 below will confirm this and pivot to Step 6 (entropy fallback) as the primary path.
2. Track B citation audit: Zou et al. 2023 RepE paper does NOT validate a certainty probe. Their validated probes cover honesty, truthfulness, hallucination, utility, jailbreaking, emotion, power-seeking — not certainty. So even if we built a "certainty" probe, we couldn't claim it's covered by the Zou et al. validation.

Therefore: **per-token entropy of the next-token distribution is expected to be the operational reward signal.** Steps 1-5 should still be performed quickly for the audit trail (so the report documents the non-existence of certainty probe), but the bulk of your time should go to Step 6 (entropy validation at 1B/3B).

CONTEXT FILES (read in order):
1. CLAUDE.md (project overview)
2. logomesh/whitebox.py (the probe implementation)
3. logomesh/local_model.py (model interface)
4. tests/test_whitebox.py (existing tests)
5. docs/logs/2026-05-08_EMNLP Framing for KV-Cache MCTS.md §5 (why this validation matters)

TASK:
Step 1 — Inventory: identify whether `PerLayerCertaintyProjector` (or equivalent certainty extraction) exists in whitebox.py. If not, identify the closest analogue and assess what would be needed to construct one. Save findings to the output file.

Step 2 — Calibration test: if a certainty projector exists, calibrate it on a small contrastive set (10 known-confident vs 10 known-uncertain prompts). Confidence proxies you may use: factual recall ("What is the capital of France?" → confident; "What was the population of Lyon in 1342?" → uncertain). Document the calibration set in the report verbatim.

Step 3 — Signal magnitude check: at 1B, run 50 prompts spanning factual / ambiguous / nonsensical regimes. Record per-prompt certainty signal (max, mean across layers). Compute: signal range, std deviation, ratio of range to noise floor. Save raw numbers to the report.

Step 4 — Correlation with correctness: on a held-out factual recall set (20–50 questions with ground truth answers), record: model's greedy answer, correctness (binary), certainty signal at last token. Compute Pearson r and Spearman rho between certainty and correctness. Save raw data + correlation to the report.

Step 5 — Repeat at 3B. Document any differences from 1B.

Step 6 — Fallback assessment: if certainty signal degenerate (constant, random, or r < 0.10), validate per-token entropy as fallback. Same protocol: calibrate, measure signal range, correlate with correctness.

OUTPUT FILE: docs/logs/2026-05-09_track-A-certainty-probe-report.md

REQUIRED REPORT SECTIONS:
1. Inventory finding (does the probe exist? form?)
2. Calibration set (verbatim prompts)
3. Signal magnitude at 1B (raw numbers + summary stats)
4. Correlation with correctness at 1B (raw data + r/rho)
5. Same for 3B
6. Fallback assessment (entropy as proxy)
7. Pass/fail verdict per Gate 1 criteria G1.1 and G1.2 in the master checklist (docs/logs/2026-05-08_72hr-execution-checklist.md)
8. Plasticity recommendation: if certainty fails AND entropy works → use entropy; if both fail → flag as fatal (Gate 1 fatal finding #2)
9. Bibliography of any code paths inspected (file:line references)

DO NOT:
- Edit logomesh/* code (read-only on this track)
- Run anything that takes > 30 minutes wall clock
- Defer documentation; if you discover something, write it down immediately
- Report "looks fine" without numerical evidence
