# Session Log — 2026-03-20

## Entry Point
New session. Set up repo structure, established V3 strategy, completed Phase 0.

## What Was Done

### Repo Setup
- Pushed LogoMesh-Dev backup to `joshhickson/LogoMesh-MVP-Tests` before any changes
- Archived `fastapi/tests/` → `fastapi-archive-2026-03-20/` (gitignored, local only)
- Removed `fastapi/` directory; committed `.gitignore` update
- Cloned `sszz01/logomesh-lambda` into workspace root

### Strategy Documentation (in LogoMesh-Dev/lambda-docs/)
- `LAMBDA_V3_ACTION_PLAN.md` — full implementation plan
- `LAMBDA_V3_DISCOVERY_CHECKLIST.md` — gap map (Q1 answered: gpt-oss-20b confirmed on HuggingFace)
- `LAMBDA_V3_TIMELINE.md` — day-by-day execution plan
- `SESSION_LOG.md` — early session log (superseded by this docs/logs/ structure)

### Phase 0: Architectural Decoupling — COMPLETE

**Files created:**
- `scenarios/security_arena/sage/oracle.py` — `AttackOracle` ABC + `OpenAIOracle`
- `scenarios/security_arena/agents/attacker/sage/oracle.py` — mirror copy

**Files modified:**
- `scenarios/security_arena/sage/__init__.py` — export `AttackOracle`, `OpenAIOracle`
- `scenarios/security_arena/sage/ablation.py` — added `use_payload_library`, `payload_library_path`
- `scenarios/security_arena/sage/search_policy.py` — added `save()`, `load()`, `run_offline_episode()` stub; added `json`, `Path` imports
- `scenarios/security_arena/agents/attacker/agent.py` — added oracle import + param; `_call_llm` delegates to `self._oracle.generate()`
- (mirror edits to `attacker/sage/` copies)

**Test result:** 198/198 passed.

## Decisions Made

See `docs/decisions/` for formal ADRs. Summary:

- `security_arena/sage/` is canonical (runtime + tests). `attacker/sage/` is a maintained copy with different defaults. Both must be updated.
- gpt-oss-20b H-Neuron signal = **router logit entropy** (MoE, not dense MLP)
- `whitebox.py` `RepresentationEngineeringProbe` is the LAT probe home — no separate `lat_monitor.py`
- Josh's RTX 3060 12GB: Phase A uses Llama-3.2-1B only. gpt-oss-20b is H100-only.
- Phase B H100 restructured: 150-ep calibration run first → train gpt-oss-20b LAT probes → 2000-ep MCTS

## Known Issues / Carried Forward
- `attacker/sage/` and `security_arena/sage/` defaults diverge intentionally (`enable_self_eval`, `enable_refinement`). Document clearly in CLAUDE.md — do not "fix" this divergence.
- `run_offline_episode()` is a stub. Full implementation is Phase A Day 5 work in `scripts/run_offline_mcts.py`.
- `_extract_refusal_direction()` in `whitebox.py` is a stub. Phase A Day 4 work.

## Next Session Entry Point

**Phase A — Day 3: Local Model Setup**

Prerequisites Josh must do before next session starts:
1. Download Llama-3.2-1B: `huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ./models/llama-3.2-1b`
2. Confirm model directory exists at `logomesh-lambda/models/llama-3.2-1b/`

Options for next session (align with Josh before touching anything):

**Option A — Full Phase A start (if model is downloaded):**
Add `transformers`, `torch`, `accelerate`, `scikit-learn` to `pyproject.toml`; build `sage/local_model.py` (`LocalLlamaOracle`); build `sage/hneuron_monitor.py`; implement `_extract_refusal_direction()` in `whitebox.py`.

**Option B — Phase A prep only (if model not yet downloaded):**
Add deps to `pyproject.toml`, run `uv sync`, write `sage/local_model.py` interface/skeleton, then block on model download before smoke test.

**Option C — Divert to ADRs / discovery checklist questions:**
Answer open questions from `LAMBDA_V3_DISCOVERY_CHECKLIST.md` Part 3 with teammate input before continuing Phase A.

> Align with Josh before touching anything.
