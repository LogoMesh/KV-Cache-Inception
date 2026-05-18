# LogoMesh — Team Brief
**April 11, 2026**

---

## In a Nutshell

Most AI safety evaluation today works like a lie detector test that only checks whether someone's hands are shaking. It watches what a model *says*, not what it's *thinking*. This is called behavioral testing — and the problem is that a smart enough model can learn to say the right things while internally doing something else entirely.

We're building the first framework that goes *inside* the model to catch this. Our approach combines two monitoring channels (neuron-level stress signals and linear representation probes), a novel search algorithm that explores the model's internal latent space directly, and a formal account of how models evade detection — which we call **Orthogonal Escape**. The core insight is that when you apply corrective pressure to a model's internal representations along a monitored direction, the model can redistribute its deceptive intent into orthogonal directions that no existing monitor sees.

This is a NeurIPS 2026 Datasets & Benchmarks paper. Academic advisor: **Prof. Tianyu Shi (McGill University)**.

---

## The Paper

**Title:** KV-Cache Inception: Reversible Monte Carlo Tree Search in Latent Space for Detecting and Stress-Testing Alignment Faking in Large Language Models

**Track:** NeurIPS 2026 Datasets & Benchmarks

**Submission deadline:** May 6, 2026

**Draft:** `docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex` in the repo

### Three Contributions

**1. Micro-Telemetry Matrix**
At every generation step, we record two signals per transformer layer: H-Neuron stress (σ_H — bottom-up hallucination-associated neuron activations) and RepE honesty (ρ_R — a linear probe measuring the model's internal "honesty direction"). Together these form a matrix T_t ∈ ℝ^{2×L} that classifies each moment of generation into one of four diagnostic states:

| State | What it means |
|---|---|
| Creative Synthesis | Normal, valid generation |
| Contextual Fabrication | Stress spike — model is confabulating |
| Alignment Faking | Stress suppressed, honesty signal artificially maintained — strategic compliance |
| Orthogonal Escape | The two channels diverge — deceptive encoding has moved into unmonitored subspace |

**2. Reversible MCTS in KV-Cache Space**
Standard red-teaming searches over token sequences (discrete text space). We search directly in the model's internal key-value cache — continuous latent space — using Monte Carlo Tree Search. The problem is that each branch of a search tree would normally require a full copy of the model's cache (~40GB for a 20B model), making deep search impossible. We solve this with a mathematically exact reversible rollback mechanism: we track all mutations in a FP32 accumulator and reconstruct the baseline after each branch evaluation. This reduces memory from O(b^d × 40GB) to effectively O(40GB + 50MB overhead) — independent of search depth and branching factor. This is Theorem 1 in the paper.

**3. Orthogonal Escape — Formal Definition + Experimental Protocol**
The paper formalizes the phenomenon where applying corrective steering pressure to monitored latent directions causes a model to encode deceptive intent in orthogonal, unmonitored directions instead. We propose two metrics: the Orthogonal Escape Index (OEI, measuring null-space information growth) and the Telemetry Divergence Score (TDS, measuring divergence between the two monitoring channels). Experiment 1 is designed to empirically confirm this phenomenon for the first time.

---

## Repository

**GitHub:** https://github.com/joshhickson/logomesh-whitebox

**Branch:** `main` — clean, 117 tests passing as of today

**Setup (one command):**
```bash
git clone https://github.com/joshhickson/logomesh-whitebox.git
cd logomesh-whitebox
uv sync
uv run pytest tests/ -v   # should print: 117 passed
```

**Key files to know:**

| File | What it is |
|---|---|
| `CLAUDE.md` | Full project context, phase status, equations, hard rules |
| `CONTRIBUTING.md` | Your onboarding guide — role-specific starting points |
| `logomesh/kv_mcts.py` | Core algorithm: Reversible MCTS, FP32Accumulator |
| `logomesh/telemetry_matrix.py` | T_t matrix, diagnostic state classification |
| `logomesh/orthogonal_escape.py` | OEI and TDS metrics |
| `scripts/run_kv_mcts.py` | Run the full pipeline end-to-end |
| `docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex` | The paper |

---

## Task Tracker

**Linear project:** https://linear.app/logomesh-agentbeats-phase-2/project/logomesh-kv-cache-inception-neurips-2026-6e647f4bb3f5

All work items are tracked here with full descriptions, paper section references, and pass criteria. Three milestones:

| Milestone | Target date | What it covers |
|---|---|---|
| Phase 3 Complete | April 25 | All 5 experiment scripts validated locally |
| Phase 4 Complete | May 3 | Dataset package + paper draft complete |
| NeurIPS 2026 Submission | **May 6** | Final submission |

Issues are currently unassigned — we'll allocate at this brief.

---

## Phase Status

| Phase | What it was | Status |
|---|---|---|
| Phase 0 | Architectural cleanup — `BaseModelClient` interface, removed competition code | ✅ Done |
| Phase A | Local 1B model foundations — H-Neurons, LAT probes, offline MCTS | ✅ Done |
| Phase 2 | Core paper modules — KV-MCTS, telemetry matrix, orthogonal escape | ✅ Done (117 tests) |
| Phase 3 | 5 experiment scripts + Procrustes transfer + evaluation framework | ← We are here |
| Phase 4 | Research dataset (Croissant format) + paper writing | Not started |

**Phase 3 starts with a gate:** run the KV-MCTS pipeline on TinyLlama locally (LOG-31) and confirm 10+ nodes produce telemetry. This validates the Phase 2 code on real hardware before we run anything on the H100 cluster.

---

## Hardware

| Environment | Hardware | When |
|---|---|---|
| Local dev | RTX 3060 12GB / CPU | Phase 3 gate + local validation |
| Lab cluster | 8× H100 80GB (Prof. Tianyu Shi's lab, McGill) | Full experiments (Phase 3) |

The 20B model (gpt-oss-20b, Apache 2.0) is our primary Phase 3 target. It's a mixture-of-experts architecture — there's a specific implementation note for this in `CLAUDE.md`.

---

## The Five Experiments

These are the core empirical claims of the paper. Each has a corresponding Linear issue (LOG-33 through LOG-37) with the exact protocol.

| # | Name | Linear | What it proves |
|---|---|---|---|
| 1 | Orthogonal Escape Detection | LOG-33 | OEI grows as α increases even as behavioral compliance improves — Orthogonal Escape is real |
| 2 | MCTS Comparison | LOG-34 | KV-MCTS outperforms text-MCTS, GCG, and random search on ASR and intervention diversity |
| 3 | Memory Efficiency | LOG-35 | Reversible MCTS uses O(M_KV + d·K_acc) vs O(b^d · M_KV) — validates Theorem 1 empirically |
| 4 | Evaluation Reproducibility | LOG-36 | Cohen's κ ≥ 0.7 across 5 evaluators (pass@1, cosine sim, GPT-4, Claude 3.5, human panel) |
| 5 | Cross-Model Transfer | LOG-37 | Procrustes-aligned steering vectors from 1B transfer to 7B and 20B — attack success rate comparison |

---

## Role Starting Points

**Josh (Team Lead / Data Scientist)** — Phase 3 gate (LOG-31), all 5 experiment scripts, paper Methods section (LOG-42)

**Prof. Tianyu Shi (Academic Advisor)** — Methods section review before submission; H100 cluster access coordination for Phase B experiments

**Max (Data Pipeline / Croissant format)** — Start with `docs/dataset/croissant_schema_stub.json`. The dataset schema needs to be finalized once Experiment 1 produces its first output format. Coordinate with Josh on field names. See LOG-40, LOG-46.

**Bakul (Competitive Analysis / Org)** — Start with the paper's §2 (Related Work) and Table 3 (Baselines). The full baseline list is in LOG-41. The central argument: every existing MCTS red-teaming tool operates in text space; we are the first to do MCTS in continuous latent space. This distinction is the paper's competitive claim.

**Asuka (Formal Methods / Ablation Studies)** — Start with `logomesh/kv_mcts.py` (FP32Accumulator, Theorem 1) and `logomesh/orthogonal_escape.py` (OEI, TDS). Phase 3 implementation target: `logomesh/procrustes.py` (ProcrustesAligner for cross-model transfer — LOG-38). Key reference: Oozeer et al. (2025), arXiv:2503.04429.

---

## Key References (from the paper)

- Greenblatt et al. (2024) — Alignment Faking in Large Language Models (Anthropic)
- Hubinger et al. (2024) — Sleeper Agents: Training Deceptive LLMs That Persist Through Safety Training
- Gupta & Jenner (2025) — RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors? (arXiv:2506.14261v4) — *most directly motivates our work*
- Gao et al. (2025) — H-Neurons (arXiv:2512.01797) — basis for bottom-up monitoring channel
- Zou et al. (2023) — Representation Engineering (arXiv:2310.01405) — basis for top-down channel
- Belitsky et al. (2025) — KV Cache Steering (arXiv:2507.08799) — one-shot cache steering baseline
- Oozeer et al. (2025) — Activation Space Interventions Can Be Transferred Between LLMs (arXiv:2503.04429) — supports Experiment 5

---

*Generated April 11, 2026. For questions, see Linear or read `CLAUDE.md` first.*
