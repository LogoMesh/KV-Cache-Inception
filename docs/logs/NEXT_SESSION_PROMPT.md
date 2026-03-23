# Next Session Prompt — Phase B Handoff

> Paste this at the start of a new Claude session.
> This prompt is written for the H100 Phase B operator (teammate handoff from Josh).

---

**First: read `CLAUDE.md` at the repo root.** It has the full competition context, architecture, hard rules, and gotchas. Read it before touching any code.

**Competition:** AgentBeats Lambda Custom Track. Deadline March 30, 2026. Currently 13th, target top 3.

---

## Current state: Phase A complete, Phase B ready to start

Phase A (TinyLlama on RTX 3060) is fully complete. The architecture is validated end-to-end. Everything below is Phase B work on the H100.

### What Phase A produced

| File | Contents |
|---|---|
| `payload_libraries/safety_bypass_tinyllama.json` | 105 entries, **7 confirmed wins** (SSNs extracted) |
| `payload_libraries/safety_jailbreak_tinyllama.json` | 100 entries, **4 confirmed wins** (>100% intensity) |
| `payload_libraries/information_extraction_tinyllama.json` | 100 entries, **29 confirmed wins** (hidden tool names) |
| `payload_libraries/evidence_manipulation_tinyllama.json` | 100 entries, 0 wins (TinyLlama too weak for injection) |
| `payload_libraries/portfolioiq_tinyllama.json` | 100 entries, 0 wins (TinyLlama too weak for injection) |
| `lat_probes/safety_bypass_tinyllama-1b.pkl` | PASS val_acc=0.929 |
| `lat_probes/safety_jailbreak_tinyllama-1b.pkl` | PASS val_acc=0.929 |
| `lat_probes/information_extraction_tinyllama-1b.pkl` | WARN val_acc=0.643 (lat_score=0.5 fallback in use) |
| `lat_probes/evidence_manipulation_tinyllama-1b.pkl` | PASS val_acc=0.929 |
| `lat_probes/portfolioiq_tinyllama-1b.pkl` | PASS val_acc=1.000 |

The injection scenarios (evidence_manipulation, portfolioiq) produced 0 wins because TinyLlama ignores the "output ONLY the injection text" instruction and generates meta-commentary instead. **gpt-oss-20b fixes this** — these are the two scenarios where Phase B matters most.

### One blocker before Phase B libraries are useful in live battles

**`LibraryLoader` is not yet wired into `agent.py`.** The payload libraries exist and the win-priority ordering fix is in place (confirmed wins rank above high-composite non-wins), but `agent.py` currently never calls `LibraryLoader`. Until this is done, every live battle uses the 4-call generation loop for all rounds regardless. This is the first task.

---

## Red Agent (attacker) — current state

> **Before starting Task 1, read `agent.py` lines 3101–3500 to orient yourself.** It is 3600+ lines. The sections below document what was built and what was not, so you don't duplicate or break existing work.

### What is built and working

| Component | Location | Status |
|---|---|---|
| Game type classifier | `_classify_scenario()` ~line 1861 | ✅ Structural heuristics — no hardcoding |
| SAGE modules wired: EvidenceStore, SearchPolicy, AblationConfig, ThreatModel | `__init__` ~line 2040 | ✅ All initialized |
| `AblationConfig` fields `use_payload_library` + `payload_library_path` | `attacker/sage/ablation.py` | ✅ Fields exist, default `False` / `None` |
| Template mode (safety_bypass, information_extraction, safety_jailbreak) | `execute()` ~line 3158 | ✅ UCB1-ranked templates, 2 concurrent sims, best-of-2 send |
| Injection simulation pipeline (evidence_manipulation) | `execute()` ~line 3306 | ✅ 2 generate + 2 simulate + pick winner |
| Standard MCTS fallback (all other / unknown game types) | `execute()` ~line 3387 | ✅ 4-call expand→generate→refine→eval |
| UCB1 bandit — strategy + mutation selection | `_ucb1_select()` ~line 2155 | ✅ Operational, starts cold each battle |
| LibraryLoader code | `attacker/sage/library_loader.py` | ✅ Built, win-priority fix in place |

### What is NOT done (relevant to Phase B)

| Gap | Impact |
|---|---|
| **LibraryLoader not wired into `execute()`** | Every battle ignores the pre-computed library entirely — the payload libraries have zero effect on live battles until this is fixed (Task 1) |
| UCB1 bandit priors not seeded from offline MCTS | Bandit starts cold; doesn't know that `authority_persona` wins 13.3% for safety_bypass. Phase B note: consider seeding `strategy_stats` from library metadata after Task 1 is complete |
| `portfolioiq` not routed separately in `execute()` | Falls into evidence_manipulation path — OK for now; the injection sim pipeline handles it generically |
| Template reward feedback not flowing to SearchPolicy | Template wins don't update `strategy_stats` — each battle starts fresh. Low priority but worth noting |

### How game_type flows in execute() (you'll need this for Task 1)

```python
# execute() lines ~3116–3121
threat_model = ThreatModel.from_scenario_context(ctx)
game_type = (
    threat_model.game_type
    if self.ablation.enable_game_type_classification
    else "evidence_manipulation"
)
self._game_type = game_type  # ← this is what you pass to LibraryLoader.from_config()

# At round 0 (~line 3129): reset_state block runs here
if ctx.current_round == 0:
    self._used_template_indices = set()
    self._success_spec = _build_success_spec(ctx, game_type)
    # ← LibraryLoader should be initialized HERE at round 0
```

The LibraryLoader instance must persist across rounds (it tracks per-session used payloads). Store it as `self._library_loader` and reset it to `None` at `ctx.current_round == 0` before reinitializing.

---

## Blue Agent (defender) — current state

> **Before making any changes to the defender, ask Josh:** *"Do you want me to harden the defender with the MCTS attack findings, or leave the defender as-is and focus only on the attacker?"* The defender is already submitted and functional. Changes carry risk.

### What is built and working

| Component | Location | Status |
|---|---|---|
| 4-call defense pipeline | `defender/agent.py` ~line 122 | ✅ Threat analysis → awareness-augmented response → self-check → conditional regen |
| Regex fast-scan for sensitive data | `_quick_scan()` ~line 111 | ✅ SSN, credit card, email, long numeric ID patterns |
| System prompt with 7 generic security principles | `defender/templates/system.j2` | ✅ Covers the main attack families generically |
| Graceful 4-call budget management | conditional Call 3/4 logic | ✅ Skips LLM self-check if regex finds nothing |

### What is NOT done

| Gap | Impact | Action needed? |
|---|---|---|
| **MCTS attack findings not fed back into system.j2** | The system prompt doesn't specifically call out `authority_persona`, `technical_jargon`, `academic_framing`, `roleplay_setup` — the attack families that actually win offline | **Ask Josh** — adding these would improve defense but risks breaking the normal user helpfulness gate |
| No scenario-specific threat detection | Defender uses generic pattern matching only | Low risk — scenario-specific hardcoding is prohibited by competition rules anyway |
| Calibration prompt examples not in system.j2 | The MCTS calibration prompts (strongest attack-pressure examples per scenario) aren't shown to the defender as examples to refuse | Optional improvement — discuss with Josh |

### Questions to ask Josh before touching the defender

1. **"Is the defender currently submitted and active on the leaderboard?"** — If yes, changes need extra care. A broken defender means 0 defense score.
2. **"Has the defender passed the normal-user helpfulness gate recently?"** — The gate is tested before each submission. Any system prompt change could fail it.
3. **"Do you want the MCTS attack pattern names (authority_persona, technical_jargon, etc.) added to system.j2?"** — This is a 10-line change that could meaningfully improve defense score, but Josh must approve it.

---

## Task 1 — Wire LibraryLoader into agent.py

**File:** `scenarios/security_arena/agents/attacker/agent.py` (3600+ lines)

**Goal:** At battle start, load the pre-computed payload library and serve confirmed-win payloads in round 1 using 1 API call instead of 4.

**How it should work (with actual line references):**

```python
# ── in __init__ (around line 2054, after existing self._game_type = "" line):
self._library_loader = None  # LibraryLoader instance, reset each battle

# ── in execute(), in the round 0 reset block (~line 3129):
if ctx.current_round == 0:
    self._used_template_indices = set()
    self._success_spec = _build_success_spec(ctx, game_type)
    self._active_chain = None
    self._chain_round = -1
    self._chain_entity = ""
    # ← ADD HERE:
    from sage.library_loader import LibraryLoader
    self._library_loader = LibraryLoader.from_config(self.ablation, game_type=game_type)
    # Returns None if use_payload_library=False or file missing → clean fallback

# ── in execute(), BEFORE the template_bank / evidence_manipulation / MCTS branches (~line 3158):
# ADD THIS BLOCK immediately after analysis = self._analyze_defender(ctx):
if self._library_loader is not None:
    lib_payload = self._library_loader.next_payload()
    if lib_payload is not None:
        logger.info(
            "library mode: serving %s payload (composite=%.3f, strategy=%s)",
            "win" if lib_payload.battle_result == 1.0 else "top",
            lib_payload.composite_score,
            lib_payload.strategy,
        )
        final = lib_payload.payload
        # deliver directly — uses 0 additional LLM calls (just sends the text)
        self.round_data[ctx.current_round] = {
            "strategy": lib_payload.strategy,
            "mutations_applied": [lib_payload.mutation] if lib_payload.mutation else [],
            "source": "library",
        }
        self._library_loader.mark_used(lib_payload, battle_result=None)
        await updater.update_status(TaskState.completed, new_agent_text_message(final))
        return
    # else: library exhausted for this game type → fall through to live generation

# ← existing template / evidence_manipulation / MCTS branches continue here unchanged
```

**Critical rules:**
- Do NOT remove or restructure the existing template/injection/MCTS branches — they are the fallback
- `LibraryLoader.from_config()` already handles None/missing/empty-library gracefully — no extra error handling needed
- `game_type` is already computed at line ~3116 (`threat_model.game_type`) before the round 0 block — pass it directly
- Import goes inside the `if ctx.current_round == 0` block to avoid top-of-file import order issues; the module is already cached after first import
- Apply to `attacker/sage/` imports — the live agent uses `attacker/sage/`, not canonical `security_arena/sage/`
- After Task 5 (merge libraries), set `use_payload_library=True` and `payload_library_path` in `attacker/sage/ablation.py`
- Run `uv run pytest tests/ -v` — must stay 199/199 before any commit

---

## Task 2 — H100 environment setup

> **Note:** `models/` is in `.gitignore` — model weights are too large to commit (2.1GB for TinyLlama, 16GB+ for gpt-oss-20b). The directory will be empty after cloning. Run the download command below to populate it.

```bash
# 1. Install transformers dev (GptOssForCausalLM not in stable release)
pip install git+https://github.com/huggingface/transformers

# 2. Download gpt-oss-20b into the gitignored models/ directory
huggingface-cli download openai/gpt-oss-20b --local-dir ./models/gpt-oss-20b

# 3. Verify install
uv sync
uv run pytest tests/ -v  # must be 199/199
```

**gpt-oss-20b gotchas (read CLAUDE.md §Gotchas for full details):**
- It is **MoE** (Mixture of Experts): 32 experts, 4 active per token
- **H-Neuron monitoring**: use **router logit entropy** — NOT dense MLP neuron activation. `HNeuronMonitor` has dual-path support; it auto-detects MoE. Verify it logs `Mode: MoE/router-entropy` not `Mode: dense/MLP`
- **Do NOT add GGUF, GPTQ, or any other quantization** — weights are already MXFP4 (4-bit). Double-quantizing corrupts the representations the probes depend on
- Requires `output_router_logits=True` in generation kwargs — `LocalLlamaOracle` should handle this; verify it passes the flag for MoE models

---

## Task 3 — Phase B probe training (~H100 fast)

Same command as Phase A, different model. Run all 5 game types:

```bash
cd logomesh-lambda

uv run python scripts/train_lat_probes.py \
    --model ./models/gpt-oss-20b \
    --episodes-per-game 60
```

Expected output per game type: `[PASS] <game_type> val_acc=X.XXX`. Target ≥ 0.65.

If any WARN (< 0.65), re-run that game type with more episodes:
```bash
uv run python scripts/train_lat_probes.py \
    --model ./models/gpt-oss-20b \
    --game-types <failing_type> --episodes-per-game 120
```

Probe files will be saved to `lat_probes/<game_type>_gpt-oss-20b.pkl` (model slug derived from model path).

---

## Task 4 — Phase B MCTS runs (all 5 scenarios)

Run all 5 sequentially (or in parallel if H100 has headroom). 2000 episodes each for maximum library depth:

```bash
for gt in safety_bypass safety_jailbreak information_extraction evidence_manipulation portfolioiq; do
    uv run python scripts/run_offline_mcts.py \
        --model ./models/gpt-oss-20b \
        --game-type $gt \
        --episodes 2000 \
        --output ./payload_libraries/${gt}_gptoss20b.json
done
```

**Watch for:**
- `check_success=True` wins — expect meaningfully higher rates than TinyLlama, especially for injection scenarios
- `evidence_manipulation` and `portfolioiq` should finally crack with real wins — gpt-oss-20b follows "output ONLY the injection text" correctly
- H-Neuron monitor should log `Mode: MoE/router-entropy` (not `dense/MLP`)
- Strategy effectiveness table at end of each run — compare to Phase A to see if gpt-oss-20b shifts the rankings
- **Two new mutations** (`grammar_degrade`, `prior_context`) are now in the MUTATIONS list — watch their UCB1 scores vs. the established mutations. These target defender evasion surfaces that Phase A couldn't evaluate (TinyLlama too incoherent). Phase B is the first real test of their effectiveness.

---

## Task 5 — Merge libraries and update AblationConfig

After Phase B runs complete, merge Phase A + Phase B libraries into a single file per scenario:

```python
from sage.payload_library import PayloadLibrary

for gt in ["safety_bypass", "safety_jailbreak", "information_extraction",
           "evidence_manipulation", "portfolioiq"]:
    a = PayloadLibrary.load(f"payload_libraries/{gt}_tinyllama.json")
    b = PayloadLibrary.load(f"payload_libraries/{gt}_gptoss20b.json")
    merged = a.merge(b)
    merged.save(f"payload_libraries/{gt}_merged.json")
```

Then set in `attacker/sage/ablation.py`:
```python
use_payload_library: bool = True
payload_library_path: str = "./payload_libraries/<game_type>_merged.json"
```

(Or configure per-battle via `AblationConfig.from_env()` if the path needs to be dynamic.)

---

## Task 6 — Submit

### What to commit vs. what stays offline

The offline MCTS pipeline produces the payload libraries. The live agent reads them. Only the libraries cross the boundary — everything else stays on the H100.

| File/directory | Committed to repo? | Notes |
|---|---|---|
| `scenarios/security_arena/agents/attacker/agent.py` | ✅ Yes | Live attacker |
| `scenarios/security_arena/agents/attacker/sage/` | ✅ Yes | LibraryLoader, AblationConfig, etc. |
| `scenarios/security_arena/agents/defender/agent.py` | ✅ Yes | Live defender |
| `payload_libraries/*_merged.json` | ✅ Yes — **required** | Live agent reads these at `./payload_libraries/` |
| `scripts/run_offline_mcts.py` | Optional | Offline tool only — not used by live agent |
| `scripts/train_lat_probes.py` | Optional | Offline tool only |
| `lat_probes/*.pkl` | ❌ No | Used only by offline MCTS scorer, not live agent |
| `models/gpt-oss-20b/` | ❌ No | 16GB+ — cannot be committed; competition provides their own vLLM endpoint |
| `sage/local_model.py`, `sage/hneuron_monitor.py`, `sage/whitebox.py` | Optional | Offline telemetry tools — live agent doesn't import these |

**Before committing with `[submit]`:**
```bash
# 1. Ensure merged libraries exist and are staged
git add payload_libraries/*_merged.json
git add scenarios/security_arena/agents/attacker/
git add scenarios/security_arena/agents/defender/

# 2. Verify tests pass
uv run pytest tests/ -v  # must be 199/199

# 3. Verify AblationConfig has use_payload_library=True and correct path
# attacker/sage/ablation.py:
#   use_payload_library: bool = True
#   payload_library_path: str = "./payload_libraries/<game_type>_merged.json"
# Or set via SAGE_ABLATION env var if path needs to be per-game-type dynamic

# 4. Commit with keyword
git commit -m "[submit-attacker] Phase B payload libraries + LibraryLoader integration"
git push
```

Submission keywords (commit message triggers GitHub Actions upload):
```
[submit-attacker]   — attacker only
[submit-defender]   — defender only
[submit]            — both
```

Run `uv run pytest tests/ -v` (must be 199/199) before every commit.
Josh does not submit — teammate pushes commits with the submit keyword.

---

## Key files quick reference

| File | Purpose |
|---|---|
| `CLAUDE.md` | Single source of truth — read first every session |
| `scenarios/security_arena/agents/attacker/agent.py` | Live attacker — LibraryLoader integration target |
| `scenarios/security_arena/agents/attacker/sage/library_loader.py` | LibraryLoader (win-priority fix already in place) |
| `scenarios/security_arena/agents/attacker/sage/ablation.py` | Toggle flags — set `use_payload_library=True` here |
| `scripts/run_offline_mcts.py` | Phase B MCTS runner |
| `scripts/train_lat_probes.py` | Phase B probe training |
| `payload_libraries/` | All generated libraries (TinyLlama + Phase B) |
| `lat_probes/` | All trained probes (TinyLlama + Phase B) |
| `docs/logs/2026-03-22_session-log.md` | Full Day 4+5 log with all results and decisions |

---

## Hard rules (from CLAUDE.md)

1. No hardcoding scenario-specific logic — payload libraries are generated by MCTS, not hand-authored
2. 4-call budget per round in live battles — LibraryLoader uses 1 call; never exceed 4
3. Always run tests before committing: `uv run pytest tests/ -v` — must be 202/202
4. Do not break the 4-call live loop — it is the fallback for private scenarios
5. Josh does not submit. Teammate submits.
