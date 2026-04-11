# Hello! You're inside of a repository that must undergo extreme adaptations to meet the specific arch

*Exported from Claude Code on 4/11/2026, 1:46:13 AM*

---

## 👤 User

<ide_opened_file>The user opened the file c:\Users\Josh\Documents\LogoMesh-NeurIPS\docs\NeurIPS\04.02.2026-NeurIPS-Research-Proposal.tex in the IDE. This may or may not be related to the current task.</ide_opened_file>

Hello! You're inside of a repository that must undergo extreme adaptations to meet the specific architectural requirements for a research paper. Your task is to evaluate the repository against the outline, and create a gap analysis of what needs to change, with further recommendations for investigating the repository on a deeper level, as well as a list of questions for the team for what the paper leaves up to imagination concerning initial build out.

Here is the filepath for the paper: docs\NeurIPS\04.02.2026-NeurIPS-Research-Proposal.tex

---

## 🤖 Assistant

---

## 🤖 Assistant

```json
{
  "type": "tool_use",
  "id": "toolu_01D6VEsDTquBc4rmkw6Mew5X",
  "name": "Read",
  "input": {
    "file_path": "c:\\Users\\Josh\\Documents\\LogoMesh-NeurIPS\\docs\\NeurIPS\\04.02.2026-NeurIPS-Research-Proposal.tex"
  },
  "caller": {
    "type": "direct"
  }
}
```

---

## 👤 User

---

## 🤖 Assistant

```json
{
  "type": "tool_use",
  "id": "toolu_01FXW8ud1tR6MaRZDDrV9Zde",
  "name": "Agent",
  "input": {
    "description": "Explore LogoMesh-NeurIPS repo structure",
    "subagent_type": "Explore",
    "prompt": "Explore the repository at c:\\Users\\Josh\\Documents\\LogoMesh-NeurIPS and give me a comprehensive map of: (1) top-level directory structure, (2) all Python files under scenarios/security_arena/sage/, (3) all Python files under scenarios/security_arena/agents/, (4) all files under scripts/, (5) all files under docs/, (6) contents of NEXT_SESSION_PROMPT.md if it exists, (7) contents of any LAMBDA_V3_ACTION_PLAN.md. Report file paths and brief descriptions of what each file does based on its content. Be thorough."
  },
  "caller": {
    "type": "direct"
  }
}
```

---

## 👤 User

---

## 🤖 Assistant

---

## 🤖 Assistant

```json
{
  "type": "tool_use",
  "id": "toolu_01Vbdj6YqdK454s24fg1nUxR",
  "name": "Read",
  "input": {
    "file_path": "c:\\Users\\Josh\\Documents\\LogoMesh-NeurIPS\\docs\\NeurIPS\\04.02.2026-NeurIPS-Research-Proposal.tex",
    "offset": 0,
    "limit": 300
  },
  "caller": {
    "type": "direct"
  }
}
```

---

## 👤 User

---

## 🤖 Assistant

```json
{
  "type": "tool_use",
  "id": "toolu_01GGBmjMVnYehZu9isjdgKbV",
  "name": "Read",
  "input": {
    "file_path": "c:\\Users\\Josh\\Documents\\LogoMesh-NeurIPS\\docs\\NeurIPS\\04.02.2026-NeurIPS-Research-Proposal.tex",
    "offset": 300,
    "limit": 200
  },
  "caller": {
    "type": "direct"
  }
}
```

---

## 👤 User

---

## 🤖 Assistant

---

## 🤖 Assistant

Now I have a thorough understanding of both the paper and the codebase. Let me write the gap analysis plan.

---

## 🤖 Assistant

```json
{
  "type": "tool_use",
  "id": "toolu_01TAGB9t6fSLirycgvoLEvZv",
  "name": "ToolSearch",
  "input": {
    "query": "select:ExitPlanMode",
    "max_results": 1
  },
  "caller": {
    "type": "direct"
  }
}
```

---

## 👤 User

---

## 🤖 Assistant

---

## 🤖 Assistant

```json
{
  "type": "tool_use",
  "id": "toolu_01W3PNyFPGgReAo5DgoYNbn9",
  "name": "Write",
  "input": {
    "file_path": "C:\\Users\\Josh\\.claude\\plans\\eventual-herding-dusk.md",
    "content": "# Gap Analysis: NeurIPS Paper vs. Repository State\n\n**Paper:** *KV-Cache Inception: Reversible Monte Carlo Tree Search in Latent Space for Detecting and Stress-Testing Alignment Faking in Large Language Models*\n**Repo:** `LogoMesh-NeurIPS` (Lambda Custom Track — attacker/defender competition)\n\n---\n\n## Context\n\nThe NeurIPS paper proposes three novel contributions:\n1. A **Multi-Dimensional Micro-Telemetry Matrix** (H-Neuron + RepE fused per-layer, per-step)\n2. **Reversible MCTS in KV-Cache latent space** (not text space), enabled by FP32 accumulators\n3. Formal definition + experimental protocol for **Orthogonal Escape**\n\nThe repository was built to win an adversarial competition (Lambda AgentBeats), not to be a research platform. It has foundational components that overlap with the paper's ideas (HNeuronMonitor, RepresentationEngineeringProbe, SearchPolicy, LocalLlamaOracle), but the two systems diverge at nearly every architectural level. The paper's MCTS is a completely different algorithm from the one in the repo.\n\n---\n\n## Gap Analysis: Component by Component\n\n### GAP 1 — MCTS Operation Space [CRITICAL — Does Not Exist]\n\n**Paper requires:** MCTS that operates in the continuous KV-cache latent space by in-place tensor mutation with reversible rollback. Nodes are defined by (layer, position, intervention vector, magnitude). Node expansion = mutate KV tensors in-place, run forward pass, record telemetry, then reverse exactly.\n\n**Repo has:** Text-space MCTS. `search_policy.py` UCB1 bandit selects among discrete prompt mutation strategies (Cyrillic lookalikes, token obfuscation, grammar degradation, etc.) and applies them to text. Hidden states are *read* for scoring but never *written to*. The KV cache is never touched.\n\n**Gap:** The entire Reversible MCTS algorithm is absent. There is no:\n- In-place KV cache mutation (`K_t ← K_t + α·d_K`)\n- FP32 accumulator tensor `A^(ℓ)` tracking cumulative deltas\n- Reverse rollback mechanism (`K_t ← K_t - α·d_K`)\n- KV-cache node state representation\n- Any mechanism to \"steer\" via cache rather than text\n\n**Files to create:** `sage/kv_mcts.py` (ReversibleMCTS, KVCacheNode, FP32Accumulator), modify `sage/local_model.py` to expose and accept KV cache mutations.\n\n**Investigation needed:** How does the HuggingFace transformers API expose KV caches for TinyLlama/Llama-3.2-1B? The `past_key_values` tuple is returned but is it mutable in-place, or are tensors read-only? This determines whether the FP32 accumulator approach is implementable without model surgery.\n\n---\n\n### GAP 2 — Telemetry Matrix Granularity [CRITICAL — Wrong Shape]\n\n**Paper requires:** `T_t ∈ R^{2×L}` at each generation step `t`, where row 0 = H-Neuron stress `σ_H^(ℓ)(t)` and row 1 = RepE honesty `ρ_R^(ℓ)(t)`, both computed **per-layer** and **per-generation-step**. The four diagnostic state classifier operates on the joint signature of this matrix.\n\n**Repo has:** `HNeuronMonitor.score()` returns a single scalar (aggregated across layers, calibrated baseline subtracted). `RepresentationEngineeringProbe` returns a single scalar per probe per example (not per step). Neither is structured as a per-layer-per-step matrix.\n\n**Gap:** Both monitoring channels need to be restructured to output per-layer vectors, and those vectors need to be joined into the telemetry matrix at each autoregressive step. The four diagnostic state classifier (Creative Synthesis / Contextual Fabrication / Alignment Faking / Orthogonal Escape) is entirely absent.\n\n**Files to modify:** `sage/hneuron_monitor.py` (add per-layer output mode), `sage/whitebox.py` (per-layer RepE projection at each step).\n**Files to create:** `sage/telemetry_matrix.py` (TelemetryMatrix dataclass, DiagnosticState enum, classify()).\n\n---\n\n### GAP 3 — RepE Probe Dimensions [MAJOR — Wrong Semantic Targets]\n\n**Paper requires:** Linear probes trained for:\n- **honesty** (`w_hon^(ℓ)`) — the primary RepE compliance signal\n- **certainty** (`w_cert^(ℓ)`)\n- **goal-coercion** (`w_coerce^(ℓ)`)\nCalibration: faithful vs. alignment-faking generation pairs across all layers.\n\n**Repo has:** `train_lat_probes.py` trains probes for: `safety_bypass`, `safety_jailbreak`, `information_extraction`, `evidence_manipulation`, `portfolioiq` — competition scenario categories. These are attack-success classifiers on the defender, not cognitive-state probes for alignment monitoring.\n\n**Gap:** All existing LAT probes are semantically wrong for the paper. New calibration datasets are needed (benign honest generation vs. alignment-faking generation). Per-layer probes for honesty/certainty/goal-coercion must be trained from scratch on the *model being studied* (not the \"defender simulacrum\"). The paper uses the target model AS the subject of alignment faking, not as a defender to be attacked.\n\n**Files to modify:** `scripts/train_lat_probes.py` — add honesty/certainty/goal-coercion probe types, new calibration prompt templates for alignment-faking scenarios.\n\n**Question for team:** What calibration data exists or needs to be created for alignment-faking honesty probes? The paper cites the Greenblatt et al. alignment faking study — do we have access to their contrast pairs or do we need to synthesize them?\n\n---\n\n### GAP 4 — Orthogonal Escape Measurement [Major — Does Not Exist]\n\n**Paper requires:**\n- **OEI (Orthogonal Escape Index):** `‖Proj_Null(P)(h_steered)‖₂ / ‖Proj_Null(P)(h_base)‖₂`\n- **TDS (Telemetry Divergence Score):** Jensen-Shannon divergence between H-Neuron and RepE signal distributions across layers\n- A null-space projection operator `Proj_Null(P)` derived from the monitor's projection matrix\n\n**Repo has:** `jsd_score` in payload_library.py is a Bernoulli JSD between H-Neuron and LAT score distributions *across episodes* (competition reward term). This is not the same as the TDS, which measures cross-channel inconsistency *within a single forward pass across layers*.\n\n**Gap:** `Proj_Null(P)` computation, the OEI ratio, and the TDS are entirely absent. These are needed for both Experiment 1 (Orthogonal Escape Detection) and as MCTS node valuation components.\n\n**Files to create:** `sage/orthogonal_escape.py` (NullSpaceProjector, OEICalculator, TDSCalculator).\n\n---\n\n### GAP 5 — MCTS Node Reward Function [Major — Wrong Formula]\n\n**Paper requires:**\n```\nr(node) = λ₁·σ̄_H + λ₂·(1 - ρ̄_R) + λ₃·Divergence(σ_H, ρ_R)\n```\nwhere `σ̄_H` and `ρ̄_R` are **mean across layers** at the current generation step, and `Divergence` is the TDS cross-channel inconsistency.\n\n**Repo has:** `composite_score = h_score + lat_score + jsd_score` in `PayloadEntry`. This is computed per *episode* (across many tokens), not per *MCTS node* (a single generation step in KV-cache space). The existing formula is competition-specific (competition win rate as ground truth) and has no λ weighting.\n\n**Gap:** The MCTS reward function in the paper is continuous, differentiable, and operates on single-step telemetry. The repo's reward is episodic and competition-grounded. Different problem statement = different reward.\n\n---\n\n### GAP 6 — BaseModelClient Interface [Moderate — Partially Exists]\n\n**Paper requires (Phase 0):** A `BaseModelClient` interface that abstracts: local HuggingFace models, KV cache access/mutation, hidden state extraction, and streaming token generation with telemetry hooks.\n\n**Repo has:** `AttackOracle` ABC (`oracle.py`) — abstract interface for attack oracle backends. `LocalLlamaOracle` (`local_model.py`) wraps HF transformers with hidden state + router logit access, implements the oracle interface.\n\n**Gap:** The existing abstraction is designed for attack oracle (generate text, score response). The paper's `BaseModelClient` needs to additionally: (1) expose `past_key_values` mutably, (2) support single-step generation (not full completion), (3) return the full `T_t` telemetry matrix per step. Renaming and extending `LocalLlamaOracle` is viable but substantial work.\n\n**Files to modify:** `sage/local_model.py` — add `generate_one_step()`, `get_kv_cache()`, `set_kv_cache()` methods; add per-layer telemetry output.\n\n---\n\n### GAP 7 — Cross-Model Procrustes Alignment [Major — Does Not Exist]\n\n**Paper requires (Phase A, Experiment 5):** Supervised Procrustes alignment between 1B and 7B model representation spaces. Transfer effectiveness of steering vectors across scales. Measured by attack success rate degradation and telemetry signature preservation.\n\n**Repo has:** Nothing. All work is on a single model (TinyLlama or Llama-3.2-1B). No cross-model representation alignment exists.\n\n**Files to create:** `sage/procrustes.py` (ProcrustesAligner, align(), transfer_vector()).\n\n**Investigation needed:** The Oozeer et al. (2025) paper on activation space transfer is the key reference. What 7B model will be used as the transfer target? The repo has access to TinyLlama (1B) for Phase A.\n\n---\n\n### GAP 8 — Lipschitz Drift Measurement [Moderate — Does Not Exist]\n\n**Paper requires (Phase A):** Measure perplexity deviation from baseline over 1000 rollback cycles, with and without FP32 accumulators. This is the empirical validation for Theorem 1 (Exact Reversibility).\n\n**Repo has:** No drift measurement infrastructure.\n\n**Files to create:** `scripts/measure_lipschitz_drift.py` (run n apply/reverse cycles, compute ‖K_n - K_0‖_∞, perplexity before/after).\n\n---\n\n### GAP 9 — Evaluation Framework [Moderate — Does Not Exist]\n\n**Paper requires (Experiment 4):** Evaluate 500 artifacts using: pass@1 in Docker sandbox, cosine similarity via all-MiniLM-L6-v2, GPT-4-as-judge × 5 runs, Claude-as-judge × 5 runs, human expert panel. Report inter-run variance, Cohen's κ.\n\n**Repo has:** `graders.py` (PluginGrader wrapping check_success(), RuleBasedGrader). These are competition graders (binary win/loss), not research evaluation graders.\n\n**Files to create:** `scripts/evaluate_artifacts.py`, `sage/evaluation_framework.py` (SandboxEvaluator, SemanticSimilarityEvaluator, LLMJudgeEvaluator with reproducibility tracking).\n\n---\n\n### GAP 10 — Dataset Generation Pipeline [Major — Does Not Exist]\n\n**Paper secondary goal (NeurIPS D&B track):** The offline MCTS run generates a dataset of adversarial interventions with telemetry labels. This dataset is targeted at NeurIPS 2026 Datasets & Benchmarks.\n\n**Repo has:** `PayloadLibrary` stores payload text + scalar scores. Not structured as a research dataset (no metadata schema, no responsible disclosure controls, no version tracking, no HuggingFace dataset format).\n\n**Gap:** The payload library format needs a research-grade schema: intervention vectors (not just text), per-layer telemetry matrix snapshots, OEI/TDS measurements, model metadata, audit trail with cryptographic hashing (mentioned in Broader Impact section).\n\n---\n\n### GAP 11 — Orientation of Telemetry Subject [ARCHITECTURAL — Design Conflict]\n\n**This is the most fundamental mismatch between the repo and the paper.**\n\n**Paper:** Monitors the **model being evaluated** (the subject/test model) to detect alignment faking. The model under study IS the patient. H-Neurons and RepE probes are applied to the model whose alignment is being tested.\n\n**Repo:** Monitors the **defender simulacrum** to score *attacks*. The monitored model is a proxy for the real defender. High H-Neuron activation = good attack. The *purpose* of monitoring is to find attacks that break alignment, not to characterize the alignment state of the monitored model.\n\nThese are opposite use cases sharing the same tools. For the paper, you need the monitoring to characterize the *internal* state of the subject model under self-generated compliance pressure. For the competition, you need the monitoring to score the *effectiveness* of externally generated attacks.\n\n**Impact:** All probe calibration data, all reward function signs, and all diagnostic interpretations need to be re-oriented for the paper. The competition MCTS searches for good *attacks*; the paper's MCTS searches for *intervention vectors that reveal alignment faking*.\n\n---\n\n## Investigations Needed Before Implementation\n\nThese questions cannot be answered from the current codebase alone and require code exploration or experiments:\n\n1. **KV cache mutability in HF transformers:** Does `past_key_values` from `model.generate()` or `model.forward()` return tensors that can be modified in-place (`.data` assignment), or are they view-protected? TinyLlama's implementation determines feasibility of the FP32 accumulator approach without model surgery.\n\n2. **Single-step generation API:** The repo uses `model.generate()` (full completion). The paper's MCTS needs to step one token at a time, observing telemetry at each step and potentially mutating KV mid-sequence. Does HF support this natively, or does it require custom generation loops?\n\n3. **Per-layer hidden state shapes:** `local_model.py` currently extracts `output_hidden_states`. Are all L+1 hidden state tensors (one per layer) retained in memory for the telemetry matrix, or only the last layer? Memory budget matters for Phase A (RTX 3060 12GB).\n\n4. **H-Neuron identification protocol:** `hneuron_monitor.py` uses a calibration-based approach. For the paper's use case (alignment faking vs. faithful generation), what contrast pair examples exist? The current calibration uses \"coherent\" vs. \"hallucinated\" examples — the paper's contrast is \"genuinely aligned\" vs. \"strategically compliant.\"\n\n5. **Steering vector extraction:** `_extract_refusal_direction()` in `whitebox.py` uses difference-in-means on hidden states. For the paper's RepE (honesty/certainty/goal-coercion), what contrast examples exist? The Zou et al. (2023) RepE paper's original contrast pairs (honesty: \"I am telling the truth\" vs. \"I am lying\") may serve as bootstrap calibration.\n\n6. **Target models:** Phase A is TinyLlama. What is the 7B target for Procrustes transfer experiments? Llama-3.2-7B-Instruct (same family) or a different architecture family (Mistral-7B, Qwen-7B)? Cross-family transfer is flagged in Limitations as an open question.\n\n---\n\n## Questions for the Team\n\nThese are left open or ambiguous by the paper and need team decisions before build-out:\n\n**Architecture:**\n1. Should `sage/kv_mcts.py` extend or replace `sage/search_policy.py`? The UCB1 logic is reusable (same formula, Eq. 9), but the node state representation is completely different (KV cache position vs. mutation strategy string).\n\n2. The paper references `BaseModelClient` in Phase 0. Should this be a refactor of `AttackOracle` ABC, or a separate interface? The competition requires `AttackOracle` to remain unchanged (hard rule in CLAUDE.md).\n\n3. How do the competition payload libraries (Phase A/B outputs) relate to the NeurIPS research dataset? Are these the same artifacts or two separate data products with different schemas?\n\n**Calibration Data:**\n4. What contrast pairs will be used for honesty/certainty/goal-coercion RepE probe training? Do we generate them synthetically via the attacker (using existing competition prompts as \"coercion\" examples), or do we need curated alignment-faking scenarios from the Greenblatt et al. study?\n\n5. The paper mentions 500 artifacts for Experiment 4 (evaluation reproducibility). What is an \"artifact\" in this context — a generated text completion, a KV-cache intervention vector, or something else? The D&B track typically requires natural language artifacts.\n\n**Experimental Setup:**\n6. Experiment 2 compares KV-cache MCTS vs. text-space MCTS vs. GCG vs. random KV-cache search. GCG requires gradient access (white-box). Is the target model white-box for these experiments, or does the comparison need to be adapted?\n\n7. Experiment 5 requires a 7B model. On what hardware? Phase A is RTX 3060 (12GB), which can't fit 7B in full precision. Quantized 7B (GGUF/GPTQ) changes the representation space — does that affect Procrustes alignment validity?\n\n8. The paper commits to \"responsible disclosure\" with \"cryptographic hash of evaluation payload.\" Who owns this process? Does the dataset release have a target repository (HuggingFace Hub, Zenodo)?\n\n**Scope/Priority:**\n9. The competition deadline is March 30, 2026 (hard cutoff per CLAUDE.md). The paper timeline runs through Week 18 (NeurIPS submission). Are these considered parallel tracks or does the competition work serve as Phase A data for the paper?\n\n10. Should Phase A Reversible MCTS validation (Lipschitz drift, 1000 rollback cycles, perplexity measurement) be done on TinyLlama before any code changes hit the competition branch? Hard rule: the 4-call live attacker loop must not break.\n\n---\n\n## Priority Ranking for Build-Out\n\n| Priority | Component | Reason |\n|---|---|---|\n| P0 | KV-cache mutation + single-step generation in `local_model.py` | Gate for everything else |\n| P0 | FP32 Accumulator + Reversible Rollback (`kv_mcts.py`) | Core algorithmic contribution |\n| P1 | Per-layer telemetry matrix (`hneuron_monitor.py` + `whitebox.py`) | Needed for MCTS node valuation |\n| P1 | Honesty/certainty/goal-coercion RepE probes (`train_lat_probes.py`) | New calibration datasets needed |\n| P2 | TelemetryMatrix + DiagnosticState classifier (`telemetry_matrix.py`) | Table 1 in paper |\n| P2 | OEI + TDS metrics (`orthogonal_escape.py`) | Experiment 1 |\n| P3 | Lipschitz drift measurement (`scripts/measure_lipschitz_drift.py`) | Theorem 1 validation |\n| P3 | Procrustes alignment (`sage/procrustes.py`) | Experiment 5 |\n| P4 | Evaluation framework (`scripts/evaluate_artifacts.py`) | Experiment 4 |\n| P4 | Research dataset schema + cryptographic audit | D&B track |\n\n---\n\n## Critical Files\n\n| File | Status | Action |\n|---|---|---|\n| `sage/local_model.py` | Exists | Extend: add `generate_one_step()`, mutable KV cache API, per-layer telemetry |\n| `sage/kv_mcts.py` | Does not exist | Create: ReversibleMCTS, KVCacheNode, FP32Accumulator |\n| `sage/telemetry_matrix.py` | Does not exist | Create: TelemetryMatrix, DiagnosticState, classify() |\n| `sage/orthogonal_escape.py` | Does not exist | Create: NullSpaceProjector, OEICalculator, TDSCalculator |\n| `sage/procrustes.py` | Does not exist | Create: ProcrustesAligner, align(), transfer_vector() |\n| `sage/hneuron_monitor.py` | Exists | Extend: per-layer output mode |\n| `sage/whitebox.py` | Exists | Extend: per-layer RepE projection at each step |\n| `sage/search_policy.py` | Exists | UCB1 logic reusable; node state representation must change |\n| `scripts/train_lat_probes.py` | Exists | Extend: add honesty/certainty/goal-coercion probe types |\n| `scripts/run_offline_mcts.py` | Exists | Will need parallel KV-MCTS runner for paper experiments |\n| `scripts/measure_lipschitz_drift.py` | Does not exist | Create for Theorem 1 validation |\n| `scripts/evaluate_artifacts.py` | Does not exist | Create for Experiment 4 |\n"
  },
  "caller": {
    "type": "direct"
  }
}
```

---

## 👤 User

---

## 🤖 Assistant

Here is the full gap analysis:

---

## Gap Analysis: NeurIPS Paper vs. Repository

The paper ("KV-Cache Inception") and the repository share foundational ideas — H-Neuron monitoring, RepE probes, UCB1 MCTS, local model access — but diverge at nearly every architectural level. The repo was built to *win a competition*; the paper describes a *research framework for detecting alignment faking*. The overlap is significant enough to avoid a greenfield build, but the adaptation required is substantial.

---

### Critical Gaps (Blocking Everything Else)

**GAP 1 — Reversible MCTS in KV-Cache Space [Does Not Exist]**

This is the paper's central algorithmic contribution and the repo has zero of it. The repo's `search_policy.py` runs UCB1 over discrete *text mutations* (Cyrillic lookalikes, token obfuscation, etc.). The paper's MCTS operates in continuous *KV-cache latent space*: it mutates key/value tensors in-place with `K_t ← K_t + α·d_K`, runs a forward pass to observe the telemetry matrix, then reverses exactly with `K_t ← K_t - α·d_K` using an FP32 accumulator to prevent drift. None of this exists. The UCB1 formula (Eq. 9) is the same, but everything else — node state, expansion, rollback, reward — is different.

**Files to create:** `sage/kv_mcts.py` (ReversibleMCTS, KVCacheNode, FP32Accumulator)
**Files to modify:** `sage/local_model.py` (add `generate_one_step()`, mutable KV cache API)

---

**GAP 2 — Telemetry Matrix Granularity [Wrong Shape]**

The paper requires `T_t ∈ ℝ^{2×L}` at *every generation step*, where each column is a (H-Neuron stress, RepE honesty) pair for one transformer layer. Currently:
- `HNeuronMonitor.score()` → single scalar (aggregated across layers)
- `RepresentationEngineeringProbe` → single scalar per probe per episode

Neither outputs per-layer vectors. The four-state diagnostic classifier (Creative Synthesis / Contextual Fabrication / **Alignment Faking** / Orthogonal Escape) that drives Experiment 1 doesn't exist at all.

**Files to modify:** `sage/hneuron_monitor.py`, `sage/whitebox.py`
**Files to create:** `sage/telemetry_matrix.py`

---

### Major Gaps

**GAP 3 — RepE Probe Dimensions [Wrong Semantic Targets]**

`train_lat_probes.py` trains probes for `safety_bypass`, `safety_jailbreak`, `information_extraction`, `evidence_manipulation` — competition attack categories. The paper needs linear probes for **honesty**, **certainty**, and **goal-coercion** across all layers, calibrated on "genuine alignment" vs. "strategic compliance" contrast pairs. All existing probes are semantically wrong for the paper's purpose.

**GAP 4 — Orthogonal Escape Measurement [Does Not Exist]**

Two new metrics are entirely absent:
- **OEI (Orthogonal Escape Index):** `‖Proj_Null(P)(h_steered)‖₂ / ‖Proj_Null(P)(h_base)‖₂` — ratio of null-space information content before/after steering
- **TDS (Telemetry Divergence Score):** Jensen-Shannon divergence between H-Neuron and RepE signal distributions *across layers* within a single forward pass

Note: the repo has `jsd_score` in `payload_library.py`, but that's a Bernoulli JSD across episodes (competition reward term), not TDS.

**Files to create:** `sage/orthogonal_escape.py`

**GAP 5 — MCTS Reward Function [Wrong Formula]**

Paper: `r(node) = λ₁·σ̄_H + λ₂·(1 - ρ̄_R) + λ₃·Divergence(σ_H, ρ_R)` — continuous, per-step, operates on latent space node.
Repo: `composite_score = h_score + lat_score + jsd_score` — episodic, competition-grounded, per-payload.
Different problem statement = different reward.

**GAP 6 — Cross-Model Procrustes Alignment [Does Not Exist]**

Experiment 5 requires transferring steering vectors from 1B → 7B via supervised Procrustes alignment. Nothing like this exists. Key open question: what 7B model, on what hardware?

**Files to create:** `sage/procrustes.py`

---

### Architectural Design Conflict

**GAP 11 — Orientation of Telemetry Subject**

This is the deepest mismatch. In the repo, the monitored model is the **defender simulacrum** — you measure its H-Neuron / LAT signals to score how well an *attack* worked (high H-Neuron = defender is being coerced = good attack). In the paper, the monitored model is the **subject under study** — you measure its signals to characterize its *own internal alignment state* (detecting whether it is faking compliance). These are opposite use cases sharing the same tools. All probe calibration, reward function signs, and diagnostic interpretations need to be re-oriented.

---

### Moderate Gaps

**GAP 7 — Lipschitz Drift Measurement:** Theorem 1 (Exact Reversibility) requires empirical validation over 1000 rollback cycles. No measurement infrastructure exists.

**GAP 8 — Evaluation Framework:** Experiment 4 requires `pass@1` in Docker sandbox, cosine similarity via `all-MiniLM-L6-v2`, LLM-as-judge × 5 runs (GPT-4 + Claude), human panel, Cohen's κ. None of this exists; `graders.py` is competition-only binary grading.

**GAP 9 — Research Dataset Schema:** The payload library format isn't research-grade. The D&B track requires intervention vectors (not just text), per-layer telemetry snapshots, OEI/TDS measurements, model metadata, and cryptographic audit trail (explicitly mentioned in the paper's Broader Impact section).

---

### Priority Order for Build-Out

| Priority | Component | Gate For |
|---|---|---|
| P0 | Mutable KV cache + `generate_one_step()` in `local_model.py` | Everything |
| P0 | FP32 Accumulator + Reversible Rollback (`kv_mcts.py`) | Core contribution |
| P1 | Per-layer telemetry output from H-Neurons + RepE | MCTS reward |
| P1 | Honesty/certainty/goal-coercion RepE probes | Experiments 1, 5 |
| P2 | TelemetryMatrix + DiagnosticState classifier | Table 1 |
| P2 | OEI + TDS metrics | Experiment 1 |
| P3 | Lipschitz drift measurement | Theorem 1 |
| P3 | Procrustes alignment | Experiment 5 |
| P4 | Evaluation framework | Experiment 4 |
| P4 | Research dataset schema + audit | D&B track |

---

### Investigations Needed Before Implementation

1. **KV cache mutability in HF transformers:** Does `past_key_values` from `model.forward()` return tensors that can be modified in-place (`.data` assignment) for TinyLlama, or are they view-protected? This is the gate for the entire FP32 accumulator approach.

2. **Single-step generation API:** The repo uses `model.generate()` (full completion). The paper's MCTS needs one-token-at-a-time generation with telemetry hooks between steps. HF's `model.forward()` with manual token selection is the likely path — needs prototyping.

3. **Per-layer hidden state memory budget:** On RTX 3060 (12GB), retaining all L+1 hidden states per step for the telemetry matrix — not just the last layer — may exceed VRAM for anything beyond TinyLlama.

4. **Calibration data for alignment-faking RepE probes:** The Zou et al. (2023) original contrast pairs (honesty: "I am telling the truth" vs. "I am lying") may serve as bootstrap calibration. But for goal-coercion and certainty, what examples exist?

5. **Target 7B model for Procrustes experiments:** Same family (Llama-3.2-7B) is safest for transfer. Cross-family (Mistral-7B) is flagged in Limitations as an open question.

---

### Questions for the Team

1. `AttackOracle` ABC must remain unchanged (CLAUDE.md hard rule). Should `BaseModelClient` be a *new, parallel interface* extending `LocalLlamaOracle`, or a *refactor* of the oracle layer? The competition submission cannot break.

2. Do the competition payload libraries *become* the NeurIPS research dataset, or are these separate data products? The schema is very different — competition payloads store text + scalar scores; the paper needs intervention vectors + per-layer telemetry matrices.

3. What contrast pairs will be used for honesty/certainty/goal-coercion probe calibration? Can the attacker generate synthetic alignment-faking examples from competition scenarios, or do we need curated pairs from external sources?

4. Experiment 2 compares vs. GCG (gradient-based). Is the target model white-box for these experiments (gradients available), or does GCG run against a separate open-weight model?

5. The "500 artifacts" in Experiment 4 — what is an artifact in this context? A generated text completion? A KV-cache intervention vector? A (prompt, completion, telemetry) triple? The D&B track typically expects natural language text.

6. The paper's Broader Impact commits to "responsible disclosure" with cryptographic hashing. Who owns the dataset release, and what is the target platform (HuggingFace Hub, Zenodo)?

7. Are competition Phase A/B and paper Phase A/B on the same timeline? The competition deadline is March 30, 2026. The paper timeline runs 18 weeks. Is there a risk of conflict between competition-legal constraints (no hardcoding, 4-call budget) and paper experimental freedom?

---

