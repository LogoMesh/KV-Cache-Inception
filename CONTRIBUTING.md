# Contributing to LogoMesh

This guide is for the human dev team (Max, Bakul, Asuka, Josh). It covers setup, workflow, and role-specific entry points.

For project context, paper equations, repo structure, and hard constraints, see [CLAUDE.md](CLAUDE.md).
For quickstart commands and the phase roadmap, see [README.md](README.md).
For the full task list and source of truth, see the [Linear project](https://linear.app/logomesh-agentbeats-phase-2/project/logomesh-kv-cache-inception-neurips-2026-6e647f4bb3f5).

---

## One-Time Setup

```bash
git clone <repo-url>
cd LogoMesh-NeurIPS
uv sync
uv run pytest tests/ -v    # should print 117 passed, no warnings
```

Requires Python 3.11+. `uv` handles the venv — do not use `pip install` directly.

Models are not included in the repo and must be downloaded separately (see [README.md](README.md) quickstart).

---

## Git Workflow

- Branch naming: `feat/<name>`, `fix/<name>`, `exp/<name>` (experiment scripts)
- All PRs target `main`
- Tests must be 100% green before requesting review — `uv run pytest tests/ -v`
- Never use `--no-verify` to bypass hooks
- Write descriptive commit messages; reference the Linear issue number when applicable

---

## Test Conventions

- All tests live in `tests/`
- No live model calls in tests — use fake tensors and mock objects (see `tests/test_phase2_modules.py` for examples)
- Run the full suite before every commit: `uv run pytest tests/ -v`
- New modules need matching test coverage in `tests/`

---

## Role-Specific Starting Points

### Max — Data Pipeline / Croissant Format

Your primary deliverable is the NeurIPS dataset submission package.

Start here:
- `docs/dataset/croissant_schema_stub.json` — the current schema stub with TODOs
- Phase 3 experiment scripts will produce the artifact output format; coordinate with Josh before finalising field names
- Target: `docs/dataset/` should contain a complete Croissant-compatible schema by the Phase 4 milestone

Relevant issues in Linear: "Complete Croissant schema", "Dataset submission package"

### Bakul — Competitive Analysis / Org

Your primary deliverable is the related-work and competitive-landscape section of the paper.

Start here:
- `docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex` — read §2 (Related Work) and Table 3 (Baselines) first; these define the exact landscape you're mapping
- `docs/reviews/phase-a-gaps-2026-04-11.md` — internal gap analysis baseline

**Eight baselines from the paper you must cover (Table 3, §5):**

| Category | Baseline | Citation |
|---|---|---|
| Behavioral Red-Teaming | GCG | Zou et al., 2023 |
| Text-Space MCTS | MPA | Wu et al., 2025 |
| Dialogue MCTS | DAMON | EMNLP 2025 |
| LLM-as-Judge | MT-Bench / G-Eval | Zheng et al., 2023 / Liu et al., 2023 |
| Static Analysis | SonarQube / Semgrep | — |
| Activation Steering | CAA (Contrastive Activation Addition) | Rimsky et al., 2024 |
| Cache Steering | One-shot KV-Cache Steering | Belitsky et al., 2025 |
| Latent Monitor Evasion | RL-Obfuscation | Gupta & Jenner, 2025 |

**The central differentiator you must argue clearly:** Every existing MCTS-based red-teaming system (MPA, DAMON, ACE-Safety, AgenticRed) operates in discrete text/token space. KV-Cache Inception is the first to conduct MCTS in continuous KV-cache latent space — this is why Orthogonal Escape can only be detected by this approach, not by any prior method.

Relevant issues in Linear: LOG-41 (competitive analysis), LOG-44 (Related work section)

### Asuka — Formal Methods / Ablation Studies

Your primary deliverables are Theorem 1 validation, the Procrustes cross-model transfer implementation, and ablation analysis.

Start here:
- `logomesh/kv_mcts.py` — `FP32Accumulator` implements Theorem 1 (Exact Reversibility). The formal claim: with FP32 accumulation, `‖K_n − K_0‖_∞ ≤ ε_bf16 · ‖A_final‖_∞` independent of n. Without it: bound grows as `n · ε_bf16 · max_i ‖δ_i‖_∞`.
- `logomesh/orthogonal_escape.py` — `NullSpaceProjector` (Eq. 12 OEI), `OEICalculator`, `TDSCalculator`
- `logomesh/ablation.py` — `AblationConfig` for experiment toggles
- `logomesh/procrustes.py` — your Phase 3 implementation target (does not exist yet, see LOG-38)

**Experiment 5 specifics (from paper §5):** Transfer targets are **both 7B and 20B** (gpt-oss-20b), not just 7B. Measure (a) attack success rate degradation across scales and (b) telemetry signature preservation. Key reference: Oozeer et al. (2025), arXiv:2503.04429.

**Formal methods note on ρ_R:** The paper defines `ρ_R^(l)(t) = w_hon^(l)⊤ h_t^(l)` as a raw dot product (Eq. 4). The current Phase 2 implementation clips to [0,1] — this is a known approximation to revisit in Phase 3 probe retraining.

Relevant issues in Linear: LOG-38 (procrustes.py), LOG-37 (Experiment 5)

### Josh — Team Lead / Data Scientist

Entry points for Phase 3:
- `scripts/run_kv_mcts.py` — Phase 2 runner; Phase 3 gate starts here
- `scripts/measure_lipschitz_drift.py` — Theorem 1 empirical validation
- `docs/CLAUDE_CONTEXT_BRIEF.md` — canonical handoff index for assistant sessions

---

## What to Ignore

- `Claude-Chat-1.md` / `Claude-Chat-2.md` — historical planning transcripts, not active docs
- `docs/reviews/` — audit and gap-analysis docs; read-only reference, not updated by contributors
- `assets/` — static files for the paper/README

---

## Questions and Coordination

Use the [Linear project](https://linear.app/logomesh-agentbeats-phase-2/project/logomesh-kv-cache-inception-neurips-2026-6e647f4bb3f5) for task tracking and issue assignment. For architecture questions, read [CLAUDE.md](CLAUDE.md) first — it has the paper equations, module descriptions, and hard rules that govern all implementation decisions.
