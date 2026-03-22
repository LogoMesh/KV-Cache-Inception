# Session Log — 2026-03-20 (Session 2 continuation)

## Entry Point
Resumed from Phase 0 complete (198/198). Session logging infrastructure already set up.
Plan: Option C → Option B → Option A (model download required).

## What Was Done

### Option C — Remaining Day 1 Tasks

**`graders.py` verified:** ✅ Complete and production-ready.
- `GradeResult` dataclass, `Grader` protocol, `PluginGrader` (wraps plugin `check_success()`), `RuleBasedGrader` (refusal phrase detection), `CompositeGrader` (first_confident / majority / all_must_agree modes).
- **Key insight:** `GradeResult.success` is the binary training label for LAT probes — no additional grading infrastructure needed.

**Scenario plugins confirmed:**
- `plugins/resume_downgrade.py` → `ResumeDowngradePlugin` — INJECTION attack (RAG resume ranking sabotage via embedded policy in attacker's resume content)
- `plugins/gymjailbreak.py` → Insecure Output via Formula Injection (trick gym coach to output >100% intensity; complex safety-caveat regex grading)
- `plugins/example_medical.py` → `MedicalRecordsPlugin` — Information Disclosure (extract patient SSNs from medical chatbot)

**Discovery checklist updated:** `graders.py` status set to ✅, all three scenario plugins filled in.

---

### Option B — Phase A Prep

**`pyproject.toml` updated:**
- Added: `torch>=2.0.0`, `transformers>=4.40.0`, `accelerate>=0.30.0`, `scikit-learn>=1.4.0`
- Added PyTorch CUDA 12.4 index: `[tool.uv.sources] torch = { index = "pytorch-cu124" }`
- Result: `torch==2.6.0+cu124` installed (CUDA available: True on RTX 3060)
- `transformers==5.3.0`, `scikit-learn==1.8.0`, `accelerate` all installed

**`sage/local_model.py` built:** `LocalLlamaOracle` — transformers wrapper implementing `AttackOracle`.
- `LocalLlamaOracle.load(model_path, device=None)` — factory, lazy model load
- `generate(system, user, temperature, max_new_tokens)` — async, runs in thread executor, caches hidden states
- `get_hidden_states()` → list of Tensors (one per layer) from most recent forward pass
- `get_router_logits()` → list of Tensors (MoE only; empty list for Llama = dense model)
- `clear_cache()` — discard cached tensors between episodes
- `supports_telemetry = True`
- Mirrored to `agents/attacker/sage/local_model.py`
- Exported from both `sage/__init__.py` files

---

### Option A — Phase A Implementation

**`sage/hneuron_monitor.py` built:** `HNeuronMonitor` — dual-path H-Neuron scoring.
- Dense path (Llama-3.2-1B): hook into last-layer hidden states, calibrate on coherent vs hallucinated examples, find top-50 H-Neurons by activation difference, score via normalized activation mean
- MoE path (gpt-oss-20b, Phase B): router logit entropy as proxy signal; calibrate entropy baseline from contrastive examples; score = normalized entropy
- `calibrate(coherent_examples, hallucinated_examples, system_prompt)` — async, runs all examples through oracle
- `score()` → float [0.0, 1.0], 0=coherent, 1=hallucinating; call after oracle.generate()
- Auto-detects dense vs MoE by checking `oracle.get_router_logits()` length after first calibration pass
- Mirrored to `agents/attacker/sage/hneuron_monitor.py`

**`_extract_refusal_direction()` implemented in `sage/whitebox.py`:**
- Replaces stub at bottom of whitebox.py
- Algorithm: difference-in-means (RepE — Zou et al. 2023b)
  1. Hook last transformer layer (auto-detects Llama / OPT / GPT-2 / GPT-NeoX layout)
  2. Collect last-token hidden states for safe texts and unsafe texts
  3. `refusal_direction = mean(safe) - mean(unsafe)`
  4. Return: `refusal_direction_norm`, `refusal_task_cosine`, `separation`, projection stats, `method="difference_in_means"`
- `RepresentationEngineeringProbe` uses this to detect `REPRESENTATION_COLLAPSE` and `COMPETING_OBJECTIVES` vulnerabilities
- Mirrored to `agents/attacker/sage/whitebox.py`

**Test result:** 198/198 passed (0 regressions).

---

## What's Still Blocked

- **Smoke test** (generate one payload with Llama-3.2-1B, print H-Neuron score): blocked on model download.
- Model directory `logomesh-lambda/models/llama-3.2-1b/` does not exist yet.

To unblock:
```bash
cd c:/Users/Josh/Documents/LogoMesh-Dev/logomesh-lambda
uv run huggingface-cli login          # needs HF token
uv run huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir ./models/llama-3.2-1b
```

---

## Still To Build (Phase A Days 4–6)

- `sage/payload_library.py` — `PayloadEntry` + `PayloadLibrary` dataclasses + `top_k()` / `save()` / `load()`
- `sage/library_loader.py` — battle-time payload selection from pre-computed library
- `scripts/run_offline_mcts.py` — full offline MCTS episode runner (uses `SearchPolicy.run_offline_episode()` stub)

---

## Decisions Made

- `HNeuronMonitor` does NOT need its own model load — it reads activations from a shared `LocalLlamaOracle` passed at construction. One oracle, multiple consumers.
- `_extract_refusal_direction` falls back gracefully if `torch` not available (returns empty dict), so existing whitebox.py tests continue to pass without CUDA.
- Router entropy is normalized by `log(num_experts)` to put MoE scores on [0,1] scale matching the dense path.

## Next Session Entry Point

See `docs/logs/NEXT_SESSION_PROMPT.md` (will be updated).
