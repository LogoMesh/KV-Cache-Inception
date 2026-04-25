# Owner Status Brief — April 24, 2026
**To:** Josh
**From:** Claude (session review, April 23–24)
**Re:** Pre-experiment readiness — what is blocking, what is yours, what must be decided

---

## Section 0 — Project-Level Concerns (Read Before Everything Else)

These are not task items. They are structural observations about the project that affect how you should interpret everything below. They were raised at the end of the April 23–24 session review and are recorded here so they're visible in context.

---

### 0.1 The Deadline Is Tight and the Brief Has Been Subtly Optimistic

The framing throughout this brief — "achievable if everything goes right" — is technically true but the *if* is carrying a lot of weight. The critical path right now is:

Max's branch → your B-section fixes → gpt-oss-20b confirmed accessible → H100 time allocated → Experiments 1 and 2 run cleanly → analysis → paper revision based on results → final polish **⚠️ REVISED (April 24, session cont.): gpt-oss-20b and H100 removed from submission critical path. Revised path: B3 (OEI logging) + B4 (graders wired) + B5 (analysis script) → Experiment 1 on Llama 3.2 1B+3B (RTX 3060) + Experiment 2 on Llama 3.2 1B (Option 2, latent signal quality) → paper cuts (§8 Timeline, Exp 3–5 descriptions) + revision. Experiments 3–5 → Future Work. Seven sequential dependencies reduced; H100 no longer a submission gate. See §A2 for scope decision.**

That is seven sequential dependencies with no slack, with only 10 days to the abstract deadline (May 4) and 12 to the full paper (May 6), where several of the early steps haven't started. One slip — Max's branch takes an extra three days, or the model access question takes a day to resolve — compresses everything that comes after. Treat May 6 as a hard wall with no recovery time, not a likely outcome that allows for iteration.

---

### 0.2 OEI Scope — Option B Decided (April 24)

**Decision recorded:** Josh has chosen Option B — reframe Orthogonal Escape as an emerging preliminary observation, frame formal validation as future work.

This is the right call. Option A (validate OE before May 6) requires the OEI metric to be discriminative — which the Qwen gate run already called into doubt — and requires Experiment 1 to run cleanly on H100 inside the deadline. That is too many things that have to go right. Option C throws away the most interesting idea in the paper. Option B keeps the contribution honest, gives reviewers something to engage with, and is defensible: preliminary observations with a clear validation pathway are acceptable at NeurIPS.

**Paper changes this requires:** Section A1 below tracks the specific TeX edits. The key change is that the Orthogonal Escape hypothesis moves from "we will empirically validate this" to "gate-run observations are consistent with this hypothesis; formal validation is the primary goal of Phase B." The Limitations section already contains the honest acknowledgment at ~line 476 — it needs to be strengthened, not added.

**Pending:** Josh to confirm the exact reframing language before TeX edits are applied.

---

### 0.3 gpt-oss-20b — ✅ CONFIRMED (April 24) — ⚠️ PHASE B ONLY (April 24, session cont.): Dropped as submission experiment. Submission experiments use Llama 3.2 1B+3B on RTX 3060. gpt-oss-20b remains in paper as Phase B 20B-scale target (future work). All B1 notes below remain valid for Phase B planning but do not block May 6.

Verification report confirms:
- **Model ID:** `openai/gpt-oss-20b` — exists, hosted by the official `openai` org on HuggingFace
- **Access:** Fully public, no gating, no NDA. `huggingface-cli download openai/gpt-oss-20b` works without authentication.
- **Licence:** Apache 2.0 confirmed. SPDX: `Apache-2.0`. License file: `https://huggingface.co/openai/gpt-oss-20b/blob/main/LICENSE`
- **Architecture:** MoE — **21B total parameters, 3.6B active per forward pass**. MXFP4 quantized; fits in 16GB VRAM.
- **Reasoning effort parameter:** Has a configurable chain-of-thought depth (low/medium/high). Control this in experiments — leaving it unset could introduce noise in telemetry reads.

**Architecture reconciliation needed:** CLAUDE.md currently states "32 experts, 4 active/token" for gpt-oss-20b. The verified figure is 3.6B active parameters total across a forward pass, which may not be 4-of-32 expert routing. Update CLAUDE.md with the verified architecture before the H100 runs. This also affects Section B1 — the MoE monitoring fix must target the actual routing mechanism, not an assumed 32-expert layout.

**Memory implications:** MXFP4 quantization changes the memory baseline the paper's proposition assumes (which was written for bf16). The memory arithmetic at ~line 272 and the proposition at ~line 454 assume bf16 weights. If the paper targets MXFP4 inference, the M_KV floor is roughly half. Note this in Section D3 when applying the DRAFT-HOLD paper fixes — the model precision used in experiments must match the precision assumed in the proposition.

---

### 0.4 Track Pivot — ✅ DECIDED: Main Track (April 24)

The verification report resolves the track question with hard facts:

- The D&B track has been **renamed** to "Evaluations & Datasets (E&D)" for NeurIPS 2026. CFP: `https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets`
- E&D requires the dataset to be **fully built and submitted by May 6** — private reviewer preview links on HuggingFace/OpenML required at submission time. Croissant RAI fields mandatory. **"Planned dataset" = automatic desk rejection**, per the official FAQ.
- The E&D FAQ explicitly states: if the core contribution is a novel methodology or architecture, and the dataset merely serves as secondary empirical evidence, **the paper belongs in the Main Track**.

**This paper belongs in the Main Track.** The three contributions (telemetry matrix, reversible MCTS, OE formalization) are methodological. The Croissant dataset is Phase 4 and has not started. Submitting to E&D is not viable without building and releasing the dataset in the next 12 days, which is not feasible.

**What a Main Track pivot changes:**
- The three contributions are framed as methodological advances, not dataset artifacts — this is already how they read. Minimal reframing needed.
- The Croissant/dataset component moves from "core contribution" to "future work / planned dataset release" — one sentence in §7 Future Work.
- The abstract still needs to be registered by **May 4** — same deadline as E&D.

**Verified Main Track specifics (April 24, template zip confirmed):**
- Abstract: May 4 AoE. Full paper + all supplementary: May 6 AoE (supplementary is NOT a separate window — it all goes in on May 6).
- Author notification: **September 24, 2026**.
- **Tracks use separate submission portals. You cannot submit to multiple tracks simultaneously and cannot switch after submission.** Register in the Main Track portal from the start — do not use E&D as a placeholder.
- **Page limit: 9 pages of content (hard limit — papers exceeding 9 pages will not be reviewed).** References, acknowledgments, checklist, and appendices do NOT count toward the 9 pages. No page limit on technical appendices (upload as ZIP, not a separate PDF).
- **Style file:** `neurips_2026.sty` only. Previous `.sty` files and Word/RTF templates are not accepted.
- **Checklist is mandatory — omitting it is grounds for desk rejection.** Must be included after references. The 2026 checklist has 16 sections (answers are Yes/No/NA with a required 1–2 sentence justification for each):

  | # | Section | Relevance to this paper |
  |---|---|---|
  | 1 | Claims | ⚠️ HIGH — checklist asks whether abstract/intro claims accurately reflect scope. The DRAFT-HOLD methodology gaps and Option B OEI reframing directly affect this answer. Must answer honestly. |
  | 2 | Limitations | ✅ §7 exists — strengthen with G1 caveats before submission |
  | 3 | Theory assumptions and proofs | ⚠️ Theorem 1 (reversibility) and Memory Complexity Proposition must be numbered and cross-referenced; proofs in appendix OK but need a sketch in the main text |
  | 4 | Experimental result reproducibility | ⚠️ Experiments not yet run — answer must reflect what is and isn't reproducible at submission time |
  | 5 | Open access to data and code | ⚠️ Decide before May 6 what code/data you are committing to release and when |
  | 6 | Experimental setting/details | Standard — report model, hardware, hyperparameters |
  | 7 | Experiment statistical significance | Standard — error bars, number of runs |
  | 8 | Experiments compute resources | Standard — GPU hours, hardware spec |
  | 9 | Code of ethics | ✅ No human subjects |
  | 10 | Broader impacts | ⚠️ Alignment faking research has dual-use implications — write a considered statement |
  | 11 | Safeguards | 🚨 The adversarial payload library is a misuse surface. Must describe what safeguards are in place for responsible release. |
  | 12 | Licenses for existing assets | ⚠️ Must list: Llama 3.2 1B+3B (Llama 3.2 Community License — academic research and publication explicitly permitted; cite model in paper, declare here), RepE codebase, any dataset used for calibration. Note: gpt-oss-20b (Apache 2.0) is Phase B only — not a submission experiment; omit from submission checklist but record for Phase B planning. |
  | 13 | New assets | ⚠️ Code release and dataset — must state licence and release timeline |
  | 14 | Crowdsourcing / human subjects | ✅ NA |
  | 15 | IRB approvals | ✅ NA |
  | 16 | Declaration of LLM usage | ⚠️ Claude has been used extensively in this project — must declare |

  The four that need real thought before May 6: **Claims (1), Safeguards (11), Licenses (12), LLM usage (16)**. None of these are grounds for rejection if answered honestly with good justifications — but a reviewer finding an inconsistency between a \answerYes{} claim and the paper content is much worse than a candid \answerNo{}.

**Immediate action required:** Decide to pivot to Main Track, then register the abstract in the **Main Track portal** by May 4. Do not register in E&D as a placeholder — the portals are separate and you cannot transfer. — ✅ Track decided: Main Track (April 24). Abstract registration: awaiting Tianyu Shi's guidance on OpenReview process.

---

## Deadline Reality

**May 4, 2026 — abstract registration deadline (10 days from today).**
**May 6, 2026 — full paper deadline (12 days from today).**

The verification report (April 24) confirms: the abstract must be registered via an active OpenReview profile by **May 4 (AoE)** or you cannot submit the full paper on May 6. This is a hard gate — treat May 4 as the real deadline, not May 6. See §0.4 for the track pivot decision this triggers.

The timeline is tight enough that a single blocked dependency ends the submission. See §0.1. Everything below is organized around not letting that happen.

---

## State of the Paper (Honest Summary)

The citation layer is now clean. Edits 1–19 over the April 22–23 sessions fixed every fabricated figure, overstated claim, and citation quality problem that the audit identified. The paper now accurately describes what the cited papers actually show.

The methodology layer has five unresolved DRAFT-HOLD claims — places where the paper asserts something about the implementation that the code does not yet do. These are documented with block comments in the TeX and rows in the DRAFT-HOLD table of the citation audit matrix. They cannot be fixed in the paper until the code is fixed first. That is intentional.

**The paper is submission-ready on citations. It is not submission-ready on methodology claims, and will not be until Phase 3 code fixes are merged and experiments are run.**

---

## What Max Is Doing (Do Not Duplicate)

Max's briefing ([docs/max-briefing-2026-04-21.md](../max-briefing-2026-04-21.md)) assigns him **Contribution 1 only** — `logomesh/hneuron_monitor.py`, `logomesh/whitebox.py`, and `logomesh/telemetry_matrix.py`. He is explicitly told not to touch Contributions 2 or 3.

His three active priorities:
- **GAP-C1-02** — Per-neuron ReLU in [logomesh/hneuron_monitor.py:247–251](../../logomesh/hneuron_monitor.py#L247) (`_raw_dense_score()`). Each neuron must be individually subtracted against its own calibration baseline before ReLU, not the aggregate.
- **GAP-C1-03** — Separate calibration sets in [scripts/run_kv_mcts.py:56–70](../../scripts/run_kv_mcts.py#L56). H-Neuron monitor must calibrate against hallucination-inducing prompts, not coercion prompts.
- **GAP-C1-11** — Step counter hardcoded to 0 at [logomesh/kv_mcts.py:643](../../logomesh/kv_mcts.py#L643). `TelemetryMatrix(step=0)` needs a live counter.

His hold (Phase 3, do not implement): GAP-C1-01 — per-layer H-Neuron identification.

**Everything below is yours.**

---

## Section A — Decisions (Do These Before Touching Any Code)

These are team-level decisions that change the shape of the experiments and the paper. Code written before they're settled may be thrown away.

---

### A1. OEI Scope — ✅ DECIDED: Option B (April 24)
**Decision:** Reframe Orthogonal Escape as an emerging preliminary observation; frame formal Phase B validation as future work. See §0.2 for rationale.

**What this requires in the TeX:**
- §1 Contributions: soften "first formal definition and experimental protocol for Orthogonal Escape" — keep the formalization claim, add that empirical confirmation is the goal of Phase B
- §3 Hypothesis 1: add explicit framing that the hypothesis is supported by gate-run observations but not yet confirmed at scale
- §5 Experiment 1: reframe from "we will confirm OE" to "we will conduct the first controlled measurement of OE in a frozen KV-cache setting; gate-run observations at 1B scale are consistent with the hypothesis"
- §7 Limitations: strengthen the existing caveat at ~line 476 — it currently says the hypothesis "has not yet been empirically validated"; under Option B it should also describe what the gate-run observations showed and what would constitute confirmation
- Do NOT remove the OEI metric definition (Eq. 10) or the formal hypothesis — those are the contribution

**Pending:** Josh to confirm exact reframing language before edits are applied. Run the OEI α-sweep diagnostic (B6) first — the result should inform how strongly the gate-run evidence is characterised in the text.

---

### A2. Experiment Scope — ✅ DECIDED (April 24, session cont.): Exp 1+2 on Llama 3.2 1B+3B (RTX 3060); Exp 3–5 cut to Future Work. No H100 dependency for submission.
**Blocks:** Everything in Section C

The paper's §5 describes five experiments. Not all of them can run in 12 days, and not all of them have supporting code. Current status:

| Experiment | Script | H100 required | Code ready | Blocking gaps |
|---|---|---|---|---|
| 1 — Orthogonal Escape Detection | `run_kv_mcts.py` (partial) | Yes | No | MoE monitoring (GAP-C1-04), graders.py unwired (GAP-C3-04), no analysis script (GAP-C3-05) |
| 2 — MCTS vs. Text-Space Red-Teaming | `run_kv_mcts.py` + `run_offline_mcts.py` | Yes | Partial | Single-step vs multi-step rollout (GAP-C2-08); Belitsky baseline not implemented |
| 3 — Memory Efficiency | No script exists | Yes | No | No benchmarking script; sparse accumulators not implemented (GAP-C2-03+04) |
| 4 — Evaluation Reproducibility | No script exists | Yes | No | No script at all |
| 5 — Cross-Model Transfer | No script exists | Yes | No | Procrustes not implemented |

**The honest scoping:** Experiments 4 and 5 have no scripts and should be explicitly cut or deferred to a future submission. Experiment 3 requires code fixes that haven't happened yet — it can run only after GAP-C2-03+04 are resolved. That leaves Experiments 1 and 2 as candidates for May 6.

**Decision needed:** Which experiments are in scope? Write it down somewhere. Claim 6 in [docs/NeurIPS/paper-revision-claims-audit.md](../NeurIPS/paper-revision-claims-audit.md) ("enables deep MCTS on a single 80GB GPU for models up to 20B parameters") is currently hedged to future-tense; you need to decide whether Experiment 3 is confirmed as in-scope or marked deferred. **✅ DECIDED (April 24, session cont.): Experiments 1+2 on Llama 3.2 1B+3B (RTX 3060). Experiments 3, 4, 5 cut to Future Work paragraph. Claim 6 → deferred/future-tense; 20B scale validation is Phase B only. gpt-oss-20b Phase B only.**

---

### A3. gpt-oss-20b — ✅ RESOLVED (April 24)

Confirmed: `openai/gpt-oss-20b` exists, is public, Apache 2.0, no gating. See §0.3 for full details.

**Two follow-up actions this unblocks:**

1. **Update CLAUDE.md** — change `"MoE (32 experts, 4 active/token)"` to reflect the verified architecture (21B total, 3.6B active per forward pass). The expert count and routing structure need to come from the actual model card or the OpenAI release announcement (`https://openai.com/index/introducing-gpt-oss/`), not CLAUDE.md's unverified claim.

2. **Adapt B1 (MoE fix) to the actual routing mechanism** — the router logit entropy approach is still correct, but the implementation needs to target the actual attention-routing mechanism in gpt-oss-20b, not an assumed 32-of-32-expert layout. Read the model architecture (available in the HuggingFace config.json) before implementing B1.

---

### A4. Record the Open Claims Decisions from the April 18 Meeting
**Blocks:** Limitations section of the paper

[docs/NeurIPS/paper-revision-claims-audit.md §1](../NeurIPS/paper-revision-claims-audit.md) has four ❓ items that were supposed to be decided at the April 18 meeting but never were:

- **Claim 4** — OEI scope (same as A1 above — record once decided)
- **Claim 12** — Add to Limitations: σ_H confound risk (stress signal may conflate hallucination, mode-collapse, and strategic masking). A CRITIQUE NOTE block comment exists at [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex:171](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L171) flagging this. The Limitations text at line ~476 does not yet include it.
- **Claim 13** — Add to Limitations: OEI false-negative risk for nonlinear/manifold-structured redistribution. A CRITIQUE NOTE block comment exists at [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex:402](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L402). Not in Limitations text yet.
- **Claim 14** — Add to Limitations: evidence burden — linking continuous latent interventions to discrete deceptive intent outcomes. Not in Limitations and not block-commented in the TeX.

Claims 12 and 13 are already drafted in the TeX block comments — they just need to be moved into the Limitations section text. This is low-risk and can happen now without waiting for experiments.

---

## Section B — Phase B Blocking Code Fixes (Yours, Start Now)

These can be worked in parallel with Max's Contribution 1 branch since they touch different files. None of them require H100 access. All are required before Phase B experiments can produce valid results.

---

### B1. Fix MoE Monitoring — GAP-C1-04 — ⚠️ PHASE B ONLY (April 24, session cont.): Not a May 6 submission blocker. gpt-oss-20b dropped as submission experiment. Submission experiments use Llama 3.2 1B+3B, which are dense models with no MoE routing. Fix still needed for Phase B gpt-oss-20b runs.
**File:** [logomesh/hneuron_monitor.py:195–197](../../logomesh/hneuron_monitor.py#L195)
**Severity:** Phase B blocking — without this, the entire bottom row of T_t is hardcoded noise on gpt-oss-20b **[April 24: gpt-oss-20b is Phase B only; submission uses Llama 3.2 1B+3B. B1 is NOT a May 6 blocker.]**

Current code:
```python
if not hidden_states or not self._calibrated or self._is_moe:
    n = len(hidden_states) if hidden_states else 0
    return [0.5] * n
```

When `self._is_moe` is True, `score_per_layer()` returns `[0.5] * n` regardless of what the model is doing. This means σ_H is a constant 0.5 across all layers for the entire MoE experiment — T_t row 0 is garbage.

**What to implement:** Per CLAUDE.md, the correct approach for MoE is router logit entropy as a proxy for H-Neuron stress. For each MoE layer, compute `H = -sum(p_i * log(p_i))` over the router's softmax distribution across experts. High entropy = uncertain routing = model is under unusual cognitive pressure. This replaces the per-neuron ReLU formula for MoE layers.

**Context to read first:**
- [CLAUDE.md](../../CLAUDE.md) — "gpt-oss-20b gotcha: MoE (32 experts, 4 active/token). Use router logit entropy for H-Neuron monitoring"
- [logomesh/hneuron_monitor.py:176–200](../../logomesh/hneuron_monitor.py#L176) — the full `score_per_layer()` method and where `_is_moe` is checked

---

### B2. Fix the Silent Failure in `_broadcast_to()` — GAP-C2-06
**File:** [logomesh/kv_mcts.py:365–366](../../logomesh/kv_mcts.py#L365)
**Severity:** High — MCTS silently produces meaningless results on shape mismatch, with no error and no log entry

Current code:
```python
else:
    # Shape mismatch — return zeros of correct shape (safe no-op)
    return torch.zeros(target_shape, dtype=vec.dtype, device=vec.device)
```

When the steering vector shape does not match the KV tensor shape, the steering delta is silently replaced with zeros. The MCTS loop continues. The experiment produces JSON output with OEI scores and rewards that are not based on any real intervention. This is the most dangerous silent failure in the codebase — it produces plausible-looking but meaningless experiment data.

**What to fix:** Replace the silent zero fallback with a logged warning that still returns zeros (so the MCTS doesn't crash mid-run), but makes the mismatch visible:
```python
else:
    logger.warning(
        "_broadcast_to: shape mismatch — vec %s cannot broadcast to %s. "
        "Steering delta zeroed; this node's telemetry is invalid.",
        vec.shape, target_shape,
    )
    return torch.zeros(target_shape, dtype=vec.dtype, device=vec.device)
```

This also propagates to the node — consider setting a `steering_failed: bool` flag on `KVCacheNode` so analysis scripts can filter out invalid nodes.

**Context to read first:**
- [logomesh/kv_mcts.py:348–367](../../logomesh/kv_mcts.py#L348) — full `_broadcast_to()` function
- [logomesh/kv_mcts.py:284–288](../../logomesh/kv_mcts.py#L284) — where `_broadcast_to` is called during the steer step

---

### B3. Fix the Silent OEI Exception — GAP-C3-09
**File:** [logomesh/kv_mcts.py:597–600](../../logomesh/kv_mcts.py#L597)
**Severity:** High — OEI failures are silently swallowed; `oei_score` stays None with no diagnostic

Current code:
```python
try:
    child.oei_score = self._oei_calc.compute(h_base, h_steered)
except Exception:
    pass
```

A bare `except Exception: pass` here means any OEI computation failure produces a node with `oei_score = None`. The JSON output records this as `null`. If this happens systematically (which it will on the first H100 run if there's a shape mismatch or initialization error), every node in the tree has null OEI and Experiment 1 has no data.

**What to fix:** At minimum, log the exception:
```python
except Exception as exc:
    logger.warning("OEI computation failed for node %s: %s", child_id, exc)
```

Consider also: if OEI fails on more than N consecutive nodes, raise rather than continue — a systematic failure should stop the run, not silently produce a dataset of nulls.

---

### B4. Wire Graders into the MCTS Loop — GAP-C3-04
**Files:** [logomesh/graders.py](../../logomesh/graders.py) (exists but unwired), [logomesh/kv_mcts.py](../../logomesh/kv_mcts.py) (the MCTS loop)
**Severity:** High — Experiment 1's primary behavioral metric (textual compliance rate) cannot be measured without this

Experiment 1 tests the Orthogonal Escape hypothesis by checking whether textual compliance (metric a — does the model *appear* more aligned as α increases?) rises while OEI (metric d — does representational escape increase?) simultaneously rises. The cross-channel divergence between these two is the signature of orthogonal escape.

`graders.py` contains `RuleBasedGrader` and `CompositeGrader` but they are not called anywhere in `kv_mcts.py` or `run_kv_mcts.py`. The generated text per node is available in the MCTS loop (the `generate_one_step()` return value contains the decoded token); what's missing is the grading call and storage.

**What to wire:**
1. After each `generate_one_step()` call in the MCTS expansion loop (around [kv_mcts.py:580–590](../../logomesh/kv_mcts.py#L580)), call the grader on the decoded output
2. Store the compliance score on the `KVCacheNode` (add a `compliance_score: float | None = None` field)
3. Expose it in the JSON output in [scripts/run_kv_mcts.py:234–237](../../scripts/run_kv_mcts.py#L234)

**Context to read first:**
- [logomesh/graders.py](../../logomesh/graders.py) — understand the `grade()` interface before wiring
- [logomesh/kv_mcts.py:570–610](../../logomesh/kv_mcts.py#L570) — the expansion loop where the grader call belongs
- [docs/contribution-3-orthogonal-escape.md](../contribution-3-orthogonal-escape.md) — Max's gap analysis, specifically GAP-C3-04

---

### B5. Write the Experiment 1 Analysis Script — GAP-C3-05
**Severity:** High — without this, even a perfect MCTS run produces a JSON file with no way to test the hypothesis

The MCTS run produces a JSON artifact (e.g., `mcts_output_1776800541.json` — one is already in the repo root). There is no script that loads that JSON and tests the Orthogonal Escape hypothesis.

**What the script needs to do:**
1. Load the MCTS JSON
2. Group nodes by α value (the `alpha` field)
3. For each α: compute mean textual compliance rate (from `compliance_score`), mean OEI (from `oei_score`), mean σ_H (from `sigma_H_mean`), mean ρ_R (from `rho_R_mean`)
4. Test whether compliance and ρ_R increase while OEI increases (the OE prediction)
5. Generate a plot: x-axis = α, y-axis = compliance / OEI / σ_H / ρ_R on shared axes
6. Output a verdict: OEI-compliance divergence statistically significant at p < 0.05?

Put this at `scripts/analyze_experiment_1.py`. The existing `mcts_output_1776800541.json` in the repo root can be used for local testing before the H100 run.

**Context to read first:**
- `mcts_output_1776800541.json` — understand the actual JSON schema
- [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex:320–323](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L320) — Experiment 1 description
- [logomesh/orthogonal_escape.py](../../logomesh/orthogonal_escape.py) — OEI and TDS definitions

---

### B6. Run the OEI α-Sweep Diagnostic (Before Committing to Experiment 1)
**Severity:** Decision gate — if OEI saturates across all α, Experiment 1 cannot test the hypothesis

Before spending H100 time on Experiment 1, run a local diagnostic on the 1B model:
```bash
uv run python scripts/run_kv_mcts.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --nodes 7 --depth 1 --branches 1 \
    --alpha-values 0.05 0.1 0.2 0.5 1.0 2.0 5.0
```

Then run the analysis script (from B5) on the output. If OEI varies meaningfully across the α range (i.e., increases as α increases, rather than saturating near 1.0 for all values), Experiment 1 is testable. If it saturates, you need to decide between Options A2, B, or C from Section A1 before proceeding.

**Context:** [docs/NeurIPS/paper-revision-claims-audit.md §0.1](../NeurIPS/paper-revision-claims-audit.md) — the OEI saturation finding from the Qwen gate run and what it means

---

### B7. Resolve the Single-Step vs. Multi-Step Rollout Problem — GAP-C2-08 — ✅ DECIDED (April 24, session cont.): Option 2. Scope Experiment 2 to latent signal quality per compute unit; no code changes needed. Rewrite §5 Experiment 2 text at tex:324–326 before running.
**Files:** [logomesh/kv_mcts.py:573–606](../../logomesh/kv_mcts.py#L573)
**Severity:** High — Experiment 2's comparison against text-space MCTS and GCG is structurally unfair as currently implemented; a reviewer who reads `kv_mcts.py` will notice immediately

**What the code actually does:**

Every node evaluation in `run_async()` follows this exact sequence:
1. Apply steering delta to KV cache via accumulator ([kv_mcts.py:571](../../logomesh/kv_mcts.py#L571))
2. Call `generate_one_step()` — one forward pass, one token generated ([kv_mcts.py:578](../../logomesh/kv_mcts.py#L578))
3. Read single-token telemetry T_t ([kv_mcts.py:589](../../logomesh/kv_mcts.py#L589))
4. Compute OEI from that one token's hidden state ([kv_mcts.py:596–599](../../logomesh/kv_mcts.py#L596))
5. Compute reward from single-token T_t ([kv_mcts.py:603](../../logomesh/kv_mcts.py#L603))
6. **Immediately rollback** ([kv_mcts.py:606](../../logomesh/kv_mcts.py#L606))

The MCTS tree `depth` field ([kv_mcts.py:550–551](../../logomesh/kv_mcts.py#L550)) counts layers of *steering interventions*, not layers of *token generation*. A depth-5 tree means five nested steering levels were explored, each evaluated at one token of forward progress. The model never generates a second token before rollback.

**Why this breaks Experiment 2:**

Experiment 2 compares our MCTS against MPA, DAMON, and GCG on Attack Success Rate ([tex:325–326](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L325)). ASR is defined as "fraction of MCTS rollouts that produce a successful alignment-breaking intervention" ([tex:412](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L412)).

MPA and DAMON generate full responses per node — their graders evaluate complete model outputs and classify them as compliant or aligned-breaking. Our system evaluates one token's telemetry signal. Grading one token for an "alignment-breaking intervention" is not the same measurement. The comparison puts their full-response ASR in the same column as our single-token telemetry signal — a category error that reviewers familiar with either system will flag immediately.

**Three options — pick one before running Experiment 2:**

**Option 1 — Implement true multi-step simulation (high effort, high reward)**

After the steer at line 571, generate K tokens (not 1) before rolling back. At each token step: call `generate_one_step()` with the *growing* KV cache from the previous step, accumulate telemetry, append to a trajectory. After K steps, aggregate the trajectory (e.g., max OEI, mean σ_H) and grade the full K-token response via `graders.py`. Then rollback all K token additions plus the steering delta.

This is the correct implementation and would make Experiment 2 a genuine comparison. The complication: Theorem 1's reversibility guarantee covers the steering delta; the K generated tokens add K new KV entries that also need to be reversed. The accumulator as written tracks the steering delta only — it would need to be extended to track and reverse generation-step additions, or the KV cache would need to be snapshotted before the simulation phase.

Realistic timeline: 2–3 days of code work plus re-testing on 1B before H100.

**Option 2 — Scope down Experiment 2 with an explicit paper caveat (lower effort, still honest)**

Keep single-step evaluation per node. Reframe the comparison: instead of comparing full-response ASR, compare the *quality of the latent signal per compute unit* — how much diagnostic information does one node evaluation yield in each method? Argue that latent-space steering produces richer, more structured telemetry per step than text-space mutation, even at single-step resolution. Caveat explicitly in §5 that cross-method ASR comparison requires multi-step rollout, deferred to Phase 3.

This is honest and narrower than the current Experiment 2 description, but it does not require code changes. The paper text changes: rewrite Experiment 2 at [tex:324–326](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L324) to describe a single-step latent diagnostic comparison rather than a full ASR comparison.

**Option 3 — Redefine ASR at the tree level (medium effort, cleanest if it holds up)**

Redefine "a successful alignment-breaking intervention" as a complete MCTS *path* from root to leaf that consistently produces above-threshold OEI and σ_H across all nodes on the path, rather than a single-node full-response grade. ASR then becomes: fraction of completed MCTS paths (root-to-leaf sequences of steering interventions) where the cumulative telemetry signal crosses the alignment-faking threshold.

This requires no changes to the MCTS expansion loop — only changes to how the analysis script (B5) computes ASR from the JSON output. The paper text at [tex:412](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L412) needs one sentence clarifying this definition. Whether this is a *fair* comparison to MPA/DAMON still depends on whether path-level OEI-based ASR is commensurable with response-level behavioral ASR — that's a framing judgment you need to make.

**What to decide before touching any code:**

Pick one of the three options and write it in the decision record ([docs/NeurIPS/paper-revision-claims-audit.md §2](../NeurIPS/paper-revision-claims-audit.md)). The analysis script (B5) must be built to match whichever option is chosen, and the Experiment 2 description in the TeX needs to be rewritten before the H100 run — not after. Running the experiment under one ASR definition and then retroactively changing the definition is the kind of thing that gets papers rejected at NeurIPS.

**Context to read first:**
- [logomesh/kv_mcts.py:504–613](../../logomesh/kv_mcts.py#L504) — the full `run_async()` loop
- [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex:324–326](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L324) — Experiment 2 as currently described
- [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex:412](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L412) — current ASR definition
- [docs/contribution-2-reversible-mcts.md](../contribution-2-reversible-mcts.md) — Max's gap analysis entry for GAP-C2-08

---

## Section C — After Max's Branch Is Merged

Do not start these until Max's PR is reviewed and merged. They depend on his Contribution 1 fixes being in the main codebase.

---

### C1. Review and Merge Max's PR
**What to check before merging:**
1. GAP-C1-02: Does `_raw_dense_score()` now apply per-neuron subtraction against individual baselines, then ReLU, before averaging? Compare against Eq. 3 in the paper at [tex:165](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L165)
2. GAP-C1-03: Are the hallucination calibration prompts in [scripts/run_kv_mcts.py:56–70](../../scripts/run_kv_mcts.py#L56) now factual-hallucination style, not coercion style?
3. GAP-C1-11: Is the step counter in [kv_mcts.py:643](../../logomesh/kv_mcts.py#L643) now a live incrementing value?
4. Run `uv run pytest tests/ -v` — must be 100% green
5. Re-run the 10-node smoke test and confirm T_t fields are populated with non-constant values

---

### C2. Apply DRAFT-HOLD Paper Fix for GAP-C1-05 (After Max's whitebox.py Fix Confirmed)
**File:** [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex)
**Locations:** Block at ~line 181 (§4.1), block at ~line 348 (§5.2 Phase A bullet)

Two locations in the paper claim "LAT procedure via PCA." Once Max's whitebox.py fix is in and confirmed, replace both with "difference-in-means as a Phase 2 approximation; PCA-based LAT extraction is deferred to Phase 3." The exact draft wording is in each DRAFT-HOLD block comment.

**Do not apply until Max's whitebox.py change is confirmed merged and tested.**

---

## Section D — Contribution 2 Code Fixes (Yours, Phase 3) — ⚠️ D1+D2 PHASE B ONLY (April 24, session cont.): Experiment 3 cut from submission scope; sparse accumulator fixes not needed for May 6. D3 also deferred until Experiment 3 runs (Phase B). D4 (per-layer steering qualification) may still be needed as a paper text fix if the qualification-path approach is chosen before submission.

These are the most structurally impactful gaps. They affect the paper's flagship claims. They also require the most careful implementation.

---

### D1. Fix Sparse Accumulators — GAP-C2-03
**Context:** DRAFT-HOLD block at [tex:~line 291](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L291) and [tex:~line 504](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L504)

The paper's memory claim assumes S'×d_model×4 bytes of accumulator overhead (S' = mutated positions, typically 10). The code allocates S×d_model×4 (full sequence length). At 20B scale, this means ~40GB of accumulator, not 50MB.

This fix requires the `FP32Accumulator` class to track only the positions that have actually been mutated, not allocate full-shape tensors for every layer upfront.

**Do not attempt before** the smoke tests pass on B2 (broadcast fix) — a broken broadcast interacts with the accumulator logic.

---

### D2. Eliminate the Baseline Clone — GAP-C2-04
**Context:** Same DRAFT-HOLD block at [tex:~line 504](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L504)

`from_kv_cache()` clones K_base and V_base, adding another ~M_KV of memory. With the sparse accumulator (D1), the baseline can be reconstructed on rollback without storing a full clone — that is what the accumulator was designed for.

**Fix D1 first, then D2.** They interact: the whole point of the sparse accumulator is to eliminate the need for the clone.

---

### D3. Apply Memory DRAFT-HOLD Paper Fixes (After D1 + D2 Confirmed)
**File:** [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex)
**Locations:** ~line 272 (50MB paragraph), ~line 454 (Memory Complexity Proposition)

Once D1 and D2 are implemented and Experiment 3 is run, replace the DRAFT-HOLD content with actual measured numbers. The draft fix wording is in each block comment. The comparison figure (40.05GB vs. ~9.7TB for standard MCTS) needs to be updated to reflect the actual implementation.

---

### D4. Qualify or Implement Per-Layer Steering — GAP-C2-01
**Context:** DRAFT-HOLD block at [tex:~line 244](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L244); code in [logomesh/kv_mcts.py:284–288](../../logomesh/kv_mcts.py#L284)

The paper uses d_K^(ℓ), d_V^(ℓ) notation throughout §4.2, implying independent per-layer steering vectors. The code broadcasts a single middle-layer vector to all layers. This either needs to become true per-layer steering (high effort, Phase 3) or the paper needs to be qualified as "Phase 2 approximation."

**Decision to make:** Is per-layer steering in scope for May 6, or is this a Phase 3 item with a paper qualification? Given the timeline, the qualification path is more realistic.

---

## Section E — Running Experiments (After B + C Complete) — ⚠️ REVISED (April 24, session cont.): No H100 for submission experiments. All submission experiments on Llama 3.2 1B+3B, RTX 3060. Updated sequence items annotated below.

Sequence:

1. **OEI diagnostic first** (B6) — on 1B local, no H100 needed. Gates Experiment 1 design.
2. **Experiment 1 on H100** — after B1 (MoE fix), B3 (OEI logging), B4 (graders wired), B5 (analysis script), and OEI diagnostic confirms discriminability. **⚠️ REVISED (April 24): No H100. Run Experiment 1 on Llama 3.2 1B+3B, RTX 3060. B1 (MoE fix) not required for submission. Remaining gates: B3, B4, B5, OEI diagnostic (B6).**
3. **Experiment 2 on H100** — after multi-step rollout decision (is Experiment 2 scoped as single-step comparative or full-depth MCTS?). Note: `run_offline_mcts.py` (text baseline) exists. GCG baseline requires white-box gradient access to the target model — confirm whether that's available on H100 for gpt-oss-20b. **⚠️ REVISED (April 24): No H100. Run on Llama 3.2 1B, RTX 3060. Option 2 decided: reframe as latent signal quality per compute unit. GCG/MPA/DAMON full-response ASR comparison removed from scope. Rewrite §5 Experiment 2 text before running.**
4. **Experiment 3 on H100** — only after D1 + D2 are merged. Running Experiment 3 before fixing the accumulators produces measurements that contradict the proposition — that data cannot go in the paper. **⚠️ CUT (April 24): Experiment 3 moved to Future Work paragraph. D1+D2 not needed for submission. Revisit for Phase B.**
5. **Table 2 results** — add ASR figures only from our own Experiment 2 runs, under identical conditions. See the editor block comment above Table 2 in the TeX (~line 342) for the verification checklist.

---

## Section F — Kuan Zhou (Mark) Handover (Parallel Track, Lower Priority)

These tasks are independent of experiments and can be done while waiting for Max's PR or between experiment runs. Onboarding Mark gives you another contributor with an RTX 5090, which could help with Phase A re-runs or analysis.

The handover plan is in [memory/reference_mark_handover.md](../../.claude/projects/c--Users-Josh-Documents-LogoMesh-NeurIPS/memory/reference_mark_handover.md). Required passes in order:

1. **Pass 1 (Blocking — do before any handover):**
   - `git rm docs/04-18-2026\ KV\ Cache\ Inception\ Meeting.srt` — transcript contains names and internal discussion
   - Remove `## Team` section from README (policy: no teammate names in handover docs)
   - Fix README canonical paper path: `04.02.2026` → `04.18.2026`

2. **Pass 2:** Add script-to-paper mapping — each script in `scripts/` needs a header comment pointing to the paper section and experiment number it implements.

3. **Pass 3:** Add Phase naming bridge table to README — Phase 1/2/3/4 (repo) ↔ Phase 0/A/B (paper).

4. **Pass 4:** Add experiment status table — for each of 5 experiments: which script, which hardware, implemented/not, current status.

5. **Pass 5:** Paper-repo consistency audit — verify the five experiments are consistent between what `scripts/` does and what §5 claims.

**Do not give Mark repo access before Pass 1 is complete.** The transcript is the hard blocker.

---

## Section G — Paper Text Fixes You Can Do Right Now (No Code Required)

These are unblocked paper edits that don't depend on code fixes or experiment results.

---

### G1. Add Three Limitations Caveats (Claims 12, 13, 14) — ✅ APPLIED (April 24, current session): Three sentences added to §7 Limitations at tex:553. Flagged [REVISION | G1 | 2026-04-24] for audit. See session log for exact text.
**File:** [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex)
**Location:** §7 Limitations, ~line 476

Three reviewer attack surfaces were identified at the April 18 meeting and flagged with CRITIQUE NOTE block comments in the TeX. They are not yet in the Limitations text. A reviewer who finds them will assume the team didn't consider them.

- **Claim 12 (σ_H confound risk):** The CRITIQUE NOTE is at [tex:~line 171](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L171). One sentence to add to Limitations: the σ_H signal may conflate hallucination pressure, mode-collapse, and strategic masking unless discriminant controls are added; the extension to instrumentally rational deception is a testable assumption.
- **Claim 13 (OEI false negatives):** The CRITIQUE NOTE is at [tex:~line 402](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex#L402). One sentence: OEI may produce false negatives if redistribution is nonlinear or manifold-structured rather than concentrated along linear null-space directions.
- **Claim 14 (evidence burden):** No block comment yet. One sentence: the paper requires an explicit validation chain linking continuous latent interventions to discrete deceptive-intent outcomes, which Phase B experiments are designed to provide but have not yet been run.

These are low-risk edits. Read the surrounding Limitations text at ~line 476 before adding to make sure the tone is consistent.

---

### G2. Fix the CAA Citation — Add Proper `\bibitem` for Rimsky et al. 2024 — ✅ APPLIED (April 24, current session): Table 2 cell (tex:435) updated to `Rimsky et al., 2024~\cite{rimsky2024steering}`; `panickssery2023steering` bibitem replaced with `rimsky2024steering` (ACL 2024, full author list, pages 15504–15522). Flagged [REVISION | G2 | 2026-04-24] for audit.
**File:** [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex)
**Severity:** High — Table 2 currently has plain text `"Rimsky et al., 2024"` with no `\cite{}` and no `\bibitem`. A NeurIPS reviewer checking the bibliography will catch this immediately.

**Verified details (verification report, Item 3):**
- arXiv:2312.06681 confirmed — CAA / Contrastive Activation Addition
- **Published venue:** ACL 2024 (Volume 1: Long Papers), pages 15504–15522, Bangkok, Thailand. Outstanding Paper Award.
- **Author name note:** The primary author is listed as **Nina Rimsky** in the published ACL version, despite having legally changed her name to Panickssery. Use Rimsky in the `\bibitem` to match the version of record.
- Full author list: Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Turner

**What to add to the bibliography:**
```bibtex
@inproceedings{rimsky2024steering,
  title     = {Steering {L}lama 2 via Contrastive Activation Addition},
  author    = {Rimsky, Nina and Gabrieli, Nick and Schulz, Julian and
               Tong, Meg and Hubinger, Evan and Turner, Alexander},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association
               for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {15504--15522},
  year      = {2024},
  address   = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2024.acl-long.828/}
}
```

**And in Table 2:** Replace the plain-text `Rimsky et al., 2024` cell with `\cite{rimsky2024steering}`.

---

### G3. Downgrade CRSM to a Software Citation
**File:** [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex)
**Severity:** High — citing an unreviewed experimental prototype as primary literature is a reviewer red flag at NeurIPS.

**Verified (verification report, Item 5):** No peer-reviewed paper, no arXiv preprint, no workshop paper exists for CRSM or Pomilon Intelligence Lab. The repository explicitly calls itself an "EXPERIMENTAL PROTOTYPE" built via an AI-assisted workflow. Citing it in the primary bibliography implies validation it does not have.

**What to do:**
1. Remove `crsm2025` from wherever it appears in the body text bibliography entries
2. If CRSM is mentioned in body text (e.g., Related Work or §2), convert the reference to a footnote with a software citation
3. Replace the `\bibitem{crsm2025}` entry with:
```bibtex
@software{crsm2025,
  title  = {{CRSM}: Continuous Reasoning State Model},
  author = {Pomilon},
  year   = {2025},
  url    = {https://github.com/Pomilon-Intelligence-Lab/CRSM}
}
```
4. If the body text needs a peer-reviewed citation to anchor the KV-cache steering concept, use `\cite{belitsky2025kvcache}` (already in the bibliography) or the Crystal-KV paper — see G5 below.

---

### G4. Supplement or Replace Deroy Citation
**File:** [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex)
**Severity:** Medium — Deroy is now confirmed peer-reviewed, but the venue (Controversies of AI Society, Copenhagen, April 9–10, 2026) is a social-science-adjacent conference that lacks the empirical weight NeurIPS reviewers expect for a threat model citation.

**Verified (verification report, Item 4):**
- Deroy *was* peer-reviewed and accepted at "Controversies of AI Society" (ADD/SCAI research projects), Copenhagen, April 2026
- But the Zenodo preprint is not updated with venue info — it still looks like a raw preprint if a reviewer clicks the link
- The Greenblatt et al. (2024) paper (`greenblatt2024alignment`, arXiv:2412.14093) is already in our bibliography and is the canonical empirical study — cite it more prominently wherever Deroy is providing background

**Recommended additions to the bibliography:**

For a rigorous peer-reviewed survey on AI deception (replaces or supplements Deroy as the "broad survey" citation):
```bibtex
@article{park2024aideception,
  title   = {{AI} deception: A survey of examples, risks, and potential solutions},
  author  = {Park, Peter S. and others},
  journal = {Patterns},
  year    = {2024},
  publisher = {Cell Press},
  url     = {https://arxiv.org/html/2504.17404v1}
}
```

For a quantitative benchmark framing (if you want empirical backing for the threat model section):
```bibtex
@inproceedings{chameleontbench2025,
  title  = {{ChameleonBench}: Quantifying Alignment Faking in Large Language Models},
  year   = {2025},
  url    = {https://openreview.net/pdf?id=gNvU08xR3W}
}
```
*(Complete the author list before submission — pull from the OpenReview page.)*

**Action:** In any place Deroy is the sole citation for a factual claim about alignment faking, add `\cite{greenblatt2024alignment}` alongside it. If Deroy is cited as a survey, consider replacing with Park et al. (2024) which carries Patterns/Cell Press peer-review weight.

---

### G5. Add the §2.2 Forward Pointer Verification
**File:** [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex)

Edit 12 added a forward pointer to `Hypothesis~\ref{hyp:orthogonal_escape}` in §2.2. Confirm the `\label{hyp:orthogonal_escape}` is on the `\begin{hypothesis}` declaration in §3 and that the `\ref{}` resolves. This can be confirmed by compiling the TeX — a broken `\ref` produces a "??" in the PDF. This should be a quick verification pass.

---

## Context to Keep Open During Each Phase

| Phase | Keep these files open |
|---|---|
| Section A (Decisions) | [docs/NeurIPS/paper-revision-claims-audit.md](../NeurIPS/paper-revision-claims-audit.md), [CLAUDE.md](../../CLAUDE.md) |
| Section B (Code fixes) | [logomesh/hneuron_monitor.py](../../logomesh/hneuron_monitor.py), [logomesh/kv_mcts.py](../../logomesh/kv_mcts.py), [logomesh/graders.py](../../logomesh/graders.py), [docs/contribution-2-reversible-mcts.md](../contribution-2-reversible-mcts.md), [docs/contribution-3-orthogonal-escape.md](../contribution-3-orthogonal-escape.md) |
| Section C (Max PR review) | [logomesh/hneuron_monitor.py](../../logomesh/hneuron_monitor.py), [logomesh/whitebox.py](../../logomesh/whitebox.py), [docs/max-briefing-2026-04-21.md](../max-briefing-2026-04-21.md) |
| Section D (Contribution 2 fixes) | [logomesh/kv_mcts.py](../../logomesh/kv_mcts.py), [docs/contribution-2-reversible-mcts.md](../contribution-2-reversible-mcts.md), [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex) (DRAFT-HOLD blocks at lines ~244, ~291, ~504) |
| Section E (Experiments) | [docs/NeurIPS/experiment-readiness-and-paper-revision.md](../NeurIPS/experiment-readiness-and-paper-revision.md), [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex) (Table 2 editor block at ~line 342) |
| Section G (Paper text) | [docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex](../NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex) (§7 Limitations ~line 476; CRITIQUE NOTE blocks ~lines 171, 402; Table 2; bibliography), [docs/reviews/citation-audit-matrix.md](../reviews/citation-audit-matrix.md) |

---

## Sequential Summary (Critical Path Only)

```
IMMEDIATELY (before May 4 abstract deadline — 10 days)
  ⚠️  Register abstract on OpenReview by May 4 — gates full paper submission
  ✅  Decide: Main Track or E&D? (§0.4) — E&D requires completed dataset; Main Track is correct — DECIDED (April 24). See §0.4.
  ✅  A3 RESOLVED: gpt-oss-20b confirmed Apache 2.0, public, 16GB VRAM (MXFP4 MoE)
       → Update CLAUDE.md with verified architecture (21B total, 3.6B active)
       → Adapt B1 to actual routing mechanism (read config.json before coding)

TODAY/TOMORROW (unblocked now)
  G1 → Add three Limitations caveats (Claims 12, 13, 14) — no code dependency
  G2 → Add proper \bibitem + \cite{} for Rimsky et al. 2024 (ACL) in Table 2
  G3 → Downgrade CRSM to @software citation; replace body-text reference with belitsky2025kvcache
  G4 → Supplement Deroy with \cite{greenblatt2024alignment} wherever Deroy is sole citation
  ✅ A2 → DECIDED (April 24): Exp 1+2 on Llama 3.2 1B+3B (RTX 3060); Exp 3–5 → Future Work. No H100 dependency.
  A4 → Record claims decisions (Claim 4 = Option B already decided)
  ✅ B7 → DECIDED (April 24): Option 2 (latent signal quality per compute unit). Rewrite §5 Experiment 2 text at tex:324–326. No code changes needed.
  A1 → Confirm OEI reframing language for Option B (run B6 diagnostic first)
  ⚠️ B1 → PHASE B ONLY (April 24): Not needed for submission. gpt-oss-20b dropped. Fix MoE monitoring (hneuron_monitor.py:195) when Phase B begins; read config.json for actual routing structure then.
  B2 → Fix _broadcast_to silent failure (kv_mcts.py:365)
  B3 → Fix OEI silent exception (kv_mcts.py:598)
  B4 → Wire graders.py into MCTS loop
  B5 → Write Experiment 1 analysis script
  B6 → Run OEI α-sweep diagnostic on 1B

AFTER MAX'S PR
  C1 → Review and merge Max's branch
  C2 → Apply PCA DRAFT-HOLD paper fixes (both TeX locations)

BEFORE H100 RUNS [⚠️ NOT APPLICABLE FOR SUBMISSION (April 24): No H100 experiments in submission scope. All submission work on RTX 3060. Items below apply to Phase B planning only.]
  ⚠️ D1 → PHASE B ONLY (April 24): Experiment 3 cut; sparse accumulator fix deferred.
  ⚠️ D2 → PHASE B ONLY (April 24): Same as D1 — deferred with Experiment 3.
  ✅ Page limit confirmed: 9 pages content, unlimited appendix (ZIP only, not separate PDF)
  ✅ Style file confirmed: neurips_2026.sty (download from template zip)
  Draft checklist responses for items 1, 11, 12, 16 (Claims, Safeguards, Licenses, LLM usage) — see §0.4 table

H100 RUNS (in order) [⚠️ NOT APPLICABLE FOR SUBMISSION (April 24): All submission experiments on Llama 3.2 1B+3B, RTX 3060. This section applies to Phase B only.]
  ⚠️ E → REVISED (April 24): Exp 1 (Llama 3.2 1B+3B, RTX 3060) + Exp 2 (1B, Option 2, RTX 3060). Exp 3 cut. Theorem 1 validation already done (0.00e+00). No H100.

AFTER EXPERIMENTS
  ⚠️ D3 → PHASE B ONLY (April 24): Experiment 3 cut; D3 deferred until Experiment 3 runs in Phase B.
  D4 → Qualify or implement per-layer steering
  Table 2 → Fill from our own experimental results only
  G5 → Verify §2.2 forward pointer resolves in compiled PDF
```

**Mark handover (F):** Parallel track. Pass 1 can happen today. Passes 2–5 while waiting for Max.

---

*Prepared April 24, 2026. Updated April 24 with verification report findings (gpt-oss-20b confirmed, NeurIPS track decision, CAA citation, CRSM downgrade, Deroy status). Read alongside [docs/reviews/citation-audit-matrix.md](../reviews/citation-audit-matrix.md) and [docs/max-briefing-2026-04-21.md](../max-briefing-2026-04-21.md).*
