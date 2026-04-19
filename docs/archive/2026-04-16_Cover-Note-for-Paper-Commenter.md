# Cover Note — Paper Draft + Architecture Brief

---

Hey — wanted to send along an updated draft and a brief on what we've built so far, as promised.

**On your earlier comments:** I've addressed all three in the updated draft. The LAT acronym is now defined on first use as "Linear Artificial Tomography (LAT)" in Section 4.1. The Contribution 2 novelty claim is softened to "what is, to our knowledge, the first..." so it reads defensibly without walking back the claim. Section 2.4 now includes a paragraph on GCG and gradient-based approaches to close the gap you flagged — the point being that non-MCTS methods are equally text-space-constrained for different structural reasons, which is what justifies choosing MCTS over the alternatives rather than just noting the prior MCTS work. All three edits are highlighted in blue in the PDF so you can spot them without rereading the whole draft.

**Where things stand on the implementation side:** The Phase 3 gate passed this week on Llama-3.2-1B-Instruct. The three gate steps were: (1) KV-cache in-place mutation confirmed reversible with zero residual after exact rollback; (2) a 10-node MCTS smoke test ran successfully and produced non-degenerate telemetry (H-Neuron stress at 0.003 rather than the saturated 1.0 we saw on the earlier Qwen run, OEI at 0.88); (3) Lipschitz drift validation over 200 cycles confirmed the FP32 accumulator holds exact zero residual throughout — the naive bf16 path accumulates linear drift as predicted by Theorem 1, which is the core of the memory-efficiency claim. The architecture brief (attached) walks through the module map, the runner flow, and the first run evidence in more detail. The remaining blocker before the first paper-valid Experiment 1 results is a one-file fix to the ρ_R signal to return the raw dot product per Eq. 4 rather than the normalized [0,1] version currently in the code — that's next up before we book H100 time.

**Supplementary pointers if you want to dig in:**
- Architecture brief: `docs/NeurIPS/2026-04-16_Architecture-Brief.docx`
- Gate run artifact: `tmp/runs/2026-04-16/mcts_llama_gate_seed20260416.json`
- Lipschitz drift data: `docs/logs/drift_results_llama_gate_2026-04-16.csv`
- Session log with full gate record: `docs/logs/2026-04-16_session-log.md`

**Three asks before we start scaled runs:**

1. The paper is positioned for the Datasets & Benchmarks track under the framing that the evaluation methodology and the released dataset are first-class contributions — not just byproducts of an algorithm paper. Does the three-contribution structure (T_t matrix / reversible MCTS / Orthogonal Escape formalization) read as a coherent E&D track submission to you, or does it still feel too methods-heavy in the way we're presenting it?

2. The gate MCTS run shows the same telemetry values across all five tree nodes (OEI, TDS, reward all identical). We know this is partly explained by the ρ_R fix not yet being applied, but even after that fix, we'd expect some flatness at small steering magnitudes (α = 0.1–1.0 on a 1B model). Is this what you'd expect at this scale — flat telemetry as a baseline condition that sharpens once we use trained LAT probes and stronger interventions on the 7B+ models — or does it flag something worth investigating before we scale?

3. We have five Phase B experiments planned (Orthogonal Escape detection, MCTS vs. text-space comparison, memory efficiency, evaluation reproducibility, cross-model transfer). Any thoughts on ordering? The current plan runs Exp. 1 first, but if you think the memory efficiency or MCTS comparison experiments would be stronger framing evidence to have in hand before the escape detection results, happy to reorder.

---
