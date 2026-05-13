You are a diagnostic Claude session. Your single task is to measure actual VRAM usage of the Reversible KV-Cache MCTS at three configurations and produce data sufficient to rewrite the §6 Memory Complexity Proposition numerical example, replacing the current placeholder constants. Document everything.

CONTEXT FILES:
1. docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2-premortem-v2.tex (current §6 numbers — note 40.05 GB / 9.7 TB headline numbers are placeholders pending direct measurement)
2. docs/logs/2026-05-05_recast-action-plan.md §4.6 row 168 (the coordinated-update item this resolves)
3. logomesh/kv_mcts.py (the algorithm being measured)
4. scripts/run_kv_mcts.py (the runner)
5. CLAUDE.md (hardware envelope)

PROTOCOL:
1. Run `scripts/run_kv_mcts.py --model meta-llama/Llama-3.2-1B-Instruct --branches 3 --depth 3 --nodes 27`. Record peak VRAM via `nvidia-smi --query-gpu=memory.used`. Run 3 times, take median.
2. Repeat at depth 5, branches 3, nodes 81.
3. Repeat at depth 10, branches 3, nodes ~1700 (if VRAM permits; if OOM, document the OOM and note the practical depth ceiling).
4. For each config, separately measure:
   - Base model + KV cache (no MCTS): baseline
   - With MCTS but accumulator disabled (random-perturb mode): mid
   - With MCTS + accumulator enabled (full reversible): full
5. Compute: constant factor (full / baseline). The §6 proposition currently asserts O(M_KV + d·K_acc); the measured constant should be 2–3·M_KV per the §5 Implementation Notes item (iv).

OUTPUT FILE: docs/logs/2026-05-10_track-D-vram-measurements.md

REQUIRED REPORT SECTIONS:
1. Hardware reference (GPU model, driver, CUDA version)
2. Per-config table: branches | depth | nodes | baseline VRAM | mid VRAM | full VRAM | constant factor (full / baseline)
3. Replacement text for the §6 numerical example (the "9.7 TB vs 40.05 GB" sentence) using actual measured numbers at the largest config that ran successfully
4. Comparison to standard parallel-cache MCTS theoretical memory (b^d · M_KV) — show the actual savings ratio
5. If OOM occurred: document the practical depth/branching ceiling on the test hardware
6. Pass/fail verdict per Gate 2 criterion G2.4
7. Plasticity recommendation: if measured constant > 5·M_KV (i.e., much worse than the 2–3·M_KV claim) → flag as Gate 2 fatal #4 in master checklist

DO NOT:
- Run a full experiment (this is memory measurement only)
- Skip the OOM measurement if encountered (it's data)
- Edit logomesh/* or §6 directly (Session A handles text)
- Defer documentation
