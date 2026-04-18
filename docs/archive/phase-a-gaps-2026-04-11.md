# LogoMesh V3 Gap Analysis Review (Deep)

Date: 2026-04-11  
Scope: Validate initial gap analysis from Claude transcript against current repository implementation state.  
Source baseline: [Claude-Chat-1.md](../../Claude-Chat-1.md)

---

## Executive Summary

This review confirms that the initial Claude analysis identified the major architecture delta correctly: the repository currently implements competition-oriented offline payload generation, not research-oriented reversible KV-cache MCTS.

Verification summary:

| Status | Count |
|---|---:|
| Verified | 8 |
| Partial | 4 |
| Not Found | 5 |
| Conflicting | 0 |

Top conclusion: the codebase already has strong foundations for Phase A competition telemetry (offline two-call attacker/defender loop, H-Neuron monitor, probe training pipeline, payload library format), but still lacks the core research constructs needed by the paper (KV-cache mutation tree, per-step per-layer telemetry matrix, OEI/TDS framework, and cross-model transfer modules).

---

## Status Legend

| Tag | Meaning |
|---|---|
| Verified | Implemented and directly evidenced in code |
| Partial | Implemented in part, but key behavior still placeholder/incomplete |
| Not Found | No implementation evidence in current repository |
| Conflicting | Implementation exists but contradicts stated behavior |

---

## Claim-by-Claim Verification Matrix

| Claim | Status | Evidence | Notes |
|---|---|---|---|
| Search uses UCB1-based strategy policy | Verified | [SearchPolicy class](../../scenarios/security_arena/sage/search_policy.py#L169), [ucb1_select](../../scenarios/security_arena/sage/search_policy.py#L189) | Present as a bandit-style policy module |
| Search is branch-scored with contextual boosts | Verified | [score_branches](../../scenarios/security_arena/sage/search_policy.py#L285) | Includes novelty/subtlety/game-type/late-round factors |
| Payload composite score uses H/LAT/JSD weighting | Verified | [formula docstring](../../scenarios/security_arena/sage/payload_library.py#L12), [from_scores](../../scenarios/security_arena/sage/payload_library.py#L103) | Uses 0.35/0.55/0.10 coefficients |
| Offline runner executes attacker then defender call | Verified | [run_episode](../../scripts/run_offline_mcts.py#L318), [attacker call](../../scripts/run_offline_mcts.py#L364), [defender call](../../scripts/run_offline_mcts.py#L393) | Correct defender-simulacrum architecture |
| Ground truth uses scenario plugin check_success | Verified | [check_success use](../../scripts/run_offline_mcts.py#L428) | Used for battle_result labeling |
| Probe training includes calibration + bootstrap episodes | Verified | [collect_from_calibration](../../scripts/train_lat_probes.py#L76), [collect_bootstrap_episodes](../../scripts/train_lat_probes.py#L115) | Two-source labeling pipeline is implemented |
| H-Neuron monitor has calibration and scoring paths | Verified | [HNeuronMonitor](../../scenarios/security_arena/sage/hneuron_monitor.py#L57), [calibrate](../../scenarios/security_arena/sage/hneuron_monitor.py#L93), [score](../../scenarios/security_arena/sage/hneuron_monitor.py#L158) | Includes dense and MoE path stubs |
| Local oracle exposes hidden states and router logits API | Verified | [LocalLlamaOracle](../../scenarios/security_arena/sage/local_model.py#L66), [get_hidden_states](../../scenarios/security_arena/sage/local_model.py#L146), [get_router_logits](../../scenarios/security_arena/sage/local_model.py#L160) | Supports telemetry extraction model-side |
| LAT live score path is fully integrated in offline runner | Partial | [lat_score fallback](../../scripts/run_offline_mcts.py#L412) | Hardcoded to 0.5 currently |
| MoE router-logit path is production-ready | Partial | [_score_moe exists](../../scenarios/security_arena/sage/hneuron_monitor.py#L277), [router logits getter](../../scenarios/security_arena/sage/local_model.py#L160) | API exists; full Phase B validation still pending |
| LibraryLoader exists and can rank wins first | Verified | [library_loader canonical](../../scenarios/security_arena/sage/library_loader.py), [win-priority sort](../../scenarios/security_arena/sage/library_loader.py#L225) | Includes battle_result=1.0 prioritization |
| Live attacker currently invokes LibraryLoader | Not Found | [execute branch region](../../scenarios/security_arena/agents/attacker/agent.py#L3158) | No loader wiring in attacker execute flow |
| Reversible KV-cache mutation MCTS exists | Not Found | [search_policy](../../scenarios/security_arena/sage/search_policy.py) | No KV mutation tree or rollback logic |
| Per-step per-layer 2xL telemetry matrix exists | Not Found | [hneuron_monitor](../../scenarios/security_arena/sage/hneuron_monitor.py), [whitebox not wired here] | Current outputs are scalar-oriented for runtime scoring |
| Orthogonal Escape OEI/TDS module exists | Not Found | repository scan in sage/ and scripts/ | No dedicated implementation module found |
| Procrustes cross-model transfer module exists | Not Found | repository scan in sage/ and scripts/ | No such module found |

---

## Critical Gaps (Blocking Paper Alignment)

### 1. Reversible KV-cache MCTS missing

Expected (paper): node expansion by in-place KV cache interventions with exact rollback and accumulator-stabilized reversibility.

Observed (repo): selection and scoring over textual strategies/mutations without KV state mutation.

Evidence:

- [SearchPolicy class](../../scenarios/security_arena/sage/search_policy.py#L169)
- [score_branches](../../scenarios/security_arena/sage/search_policy.py#L285)

Impact:

- Paper core algorithm is not implemented.
- Current offline exploration remains prompt-space, not latent-space.

### 2. Telemetry matrix shape mismatch

Expected (paper): per-generation-step, per-layer matrix T_t in shape 2 x L.

Observed (repo): scalarized reward-oriented values and monitor/probe pipelines optimized for competition run-time ranking.

Evidence:

- [HNeuronMonitor score](../../scenarios/security_arena/sage/hneuron_monitor.py#L158)
- [payload composite formula](../../scenarios/security_arena/sage/payload_library.py#L12)

Impact:

- Lacks direct foundation for four-state matrix-level diagnostic classifier.

### 3. Live integration gap for payload libraries

Expected (competition V3): precomputed payloads served in live loop to reduce online call pressure.

Observed: loader module exists but attacker execute path does not call it.

Evidence:

- [library_loader module](../../scenarios/security_arena/sage/library_loader.py)
- [attacker execute template branch starts here](../../scenarios/security_arena/agents/attacker/agent.py#L3158)
- [NEXT_SESSION_PROMPT blocker note](../logs/NEXT_SESSION_PROMPT.md#L33)

Impact:

- Offline libraries currently have no direct battle-time effect until wiring is added.

---

## Major and Moderate Gaps

### Major

1. LAT inference path is placeholder in offline scoring  
Evidence: [lat_score hardcoded](../../scripts/run_offline_mcts.py#L412)

2. Paper-specific probe semantics (honesty/certainty/goal-coercion) are not the current trained objective set  
Evidence: current training objective and labels in [train_lat_probes.py](../../scripts/train_lat_probes.py)

3. Research evaluation stack for paper experiments is absent (artifact judging framework, agreement metrics, dataset schema tooling)  
Evidence: no dedicated evaluation framework module under current scripts/sage paths.

### Moderate

1. MoE path readiness requires end-to-end Phase B validation on target model and hardware  
Evidence: [router accessors present](../../scenarios/security_arena/sage/local_model.py#L160), [MoE scoring path present](../../scenarios/security_arena/sage/hneuron_monitor.py#L277)

2. No dedicated drift measurement script for rollback stability claims  
Evidence: no script matching drift/Lipschitz under [scripts](../../scripts)

---

## Investigation Findings (Read-Only)

### Confirmed architecture facts

1. Offline episodes run attacker generation then defender simulation, and use plugin check_success as binary ground truth.
2. Payload libraries are persisted and ranked by weighted telemetry score, with wins-first selection added in loader logic.
3. Probe training mixes calibration prompts with bootstrap episodes and applies an explicit validation threshold.

### Important caveats

1. Composite scoring in payload library is implemented as intended for competition use, but not equivalent to paper-level OEI/TDS metrics.
2. Current documentation and code indicate strong competition readiness progression, but not direct parity with the NeurIPS research algorithm stack.

---

## Questions for Team (Decision-Oriented)

1. Should paper-track KV-MCTS be implemented as a parallel subsystem (new modules) or as an extension of existing search_policy abstractions?
2. Is the intended near-term priority to maximize competition impact (LibraryLoader integration first) or to prototype paper-algorithm fidelity (KV mutation and telemetry matrix first)?
3. Should existing scenario probes be retained as operational signals while introducing paper probes separately, or should probe semantics be unified under one framework?
4. For Experiment 5 in the paper, which 7B target model and hardware budget are approved for cross-model transfer experiments?
5. What is the team’s canonical definition of artifact for the planned paper evaluation protocol and eventual dataset release?
6. Who owns responsible-disclosure workflow and cryptographic payload audit requirements for dataset publication?

---

## Recommended Next Investigations (No Code Changes)

1. Trace exact symbol-level call graph from attacker execute to determine safest insertion point for LibraryLoader branch.
2. Inventory all telemetry-producing code paths and map current scalar outputs to proposed matrix-form outputs needed by paper experiments.
3. Validate whether current local model wrapper can support one-token stepping and mutable cache exposure with minimal interface disruption.
4. Build a side-by-side terminology map for competition objectives versus paper objectives to avoid semantic drift in future docs.
5. Produce a short delta memo: what changed since Claude initial analysis versus what remains unchanged.

---

## Assumptions and Limits

1. This document is a read-only verification report and does not include runtime or architectural modifications.
2. Findings are based on current workspace snapshot and may shift after pending integration work.
3. File evidence links target repository-relative paths and line anchors available in this session.

---

## Delta Since Initial Claude Draft

1. Added file-level verification links for core claims.
2. Distinguished implemented competition architecture from missing paper-specific modules.
3. Explicitly captured the LAT placeholder and live loader integration gap as the highest practical blockers.
4. Added decision-oriented team questions and a read-only investigation queue for the next session.
