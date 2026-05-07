# Merge Briefing — Max's `max/experiment-pivot-critique` → main
**Date:** 2026-05-07
**From:** Josh
**To:** Max
**Re:** What was merged, what we kept as-is, what was modified in conflict resolution, post-merge verification, and what's next.

---

## TL;DR

- **All 10 of your fixes merged.** 7 gaps resolved + 3 bug fixes from your two commits (`976fce5` + `8df1225`).
- **One file (`logomesh/kv_mcts.py`) merged manually** — combined your two-pass accumulator hardening with my W_K projection fix from `main`. Both improvements landed; neither was dropped.
- **5 dataset JSON files excluded** — they were from runs before the W_K projection fix and show the GAP-C2-06 silent-zero artifact (bit-identical OEI=0.965 across α). They documented the bug; not the post-fix experimental results. Can be regenerated post-merge if you want them in the dataset package.
- **Tests: 179 pass** (was 47 before your work + my W_K fix; you added 132 net).
- **Post-merge verification on Llama 3.2-1B**: your **per-layer H-Neuron calibration is the fix that flipped σ_H from constant-zero to varying-with-α**. This is the single biggest empirical result of the week.

---

## 1. Gap-by-gap matrix

| Gap | What you did | File(s) | Conflict? | Status |
|---|---|---|---|---|
| **GAP-C1-01** per-layer H-Neuron identification | Implemented `_h_neuron_indices_per_layer`, `_coherent_mean_per_layer`, `_hallucinated_mean_per_layer`; rewrote `_calibrate_dense` to rank H-Neurons per layer; `score_per_layer` now uses per-layer baselines | `hneuron_monitor.py` | None | ✅ Merged as-is |
| **GAP-C1-06** rho_R rescaled from raw dot product | (Per your contribution-1 doc update marking it resolved) | `whitebox.py` / contribution-1 | None | ✅ Merged as-is |
| **GAP-C1-13** test coverage gaps for integration + edge cases | +35 tests including hand-calculated formula verification, two-pass safety tests, per-layer H-Neuron tests | `tests/test_phase2_modules.py` | None | ✅ Merged as-is |
| **GAP-C2-06** silent broadcast zero-fallback | `_broadcast_to` returns `(tensor, success_bool)`; logs warning on shape mismatch; two-pass pre-validation in `apply`/`rollback` (atomic — returns False if any layer fails) | `kv_mcts.py` | YES — see §2 below | ✅ Merged with manual resolution |
| **GAP-C2-10** drift validation default | `--n-cycles` 200 → 1000 (matching paper §5.1); explicit return-value checks in the script | `scripts/measure_lipschitz_drift.py` | None | ✅ Merged as-is |
| **GAP-C2-12** root node has no baseline telemetry | Added root-node baseline telemetry; OEI logging | `scripts/run_kv_mcts.py` | None | ✅ Merged as-is |
| **GAP-C3-09** silent OEI exception | Replaced bare `except` with proper logging + `oei_valid` JSON flag | `kv_mcts.py` (also exposed via runner) | (resolved as part of the kv_mcts merge below) | ✅ Merged |
| **BUG-1** blocked counter incrementing unconditionally | Only increment `blocked` when `reward < 0.5` | `search_policy.py` | None | ✅ Merged as-is |
| **BUG-2** unbounded ρ̄_R in reward formula | Apply sigmoid normalization to ρ̄_R before `(1 - ρ̄_R)` so reward stays in [0,1]. Raw ρ_R values in TelemetryMatrix unchanged | `telemetry_matrix.py` | None | ✅ Merged as-is |
| **BUG-3** bare `except Exception` in template error path | Narrowed to `(ValueError, TypeError, KeyError, jinja2.TemplateError)` with `logger.warning`; applied to both `_generate_sync` and `_generate_one_step_sync` | `local_model.py` | None | ✅ Merged as-is |

Plus your supporting infrastructure landed:
- **CLAUDE.md** updates: quality gate, paper-to-code verification rules
- **docs/contribution-{1,2,3}.md** updates: gap-status notes reflecting your fixes
- **Two new session logs**: `docs/logs/2026-04-21_session-log_maxwell.md`, `docs/logs/2026-05-05_session-log_maxwell.md`

---

## 2. The one conflict — `logomesh/kv_mcts.py`

Both your branch and `main` had touched this file. Your work hardened the FP32 accumulator with two-pass pre-validation; `main` had a different fix for GAP-C2-06 (the W_K projection — see §3 below for context). Three conflict hunks: two in `apply`/`rollback`, one in the body of `_broadcast_to`.

**Resolution: kept both improvements.** The composition rule was to make `_shape_match_kv` (the W_K-projecting wrapper) return your `(tensor, success_bool)` signature so it plugs cleanly into your two-pass design.

```python
# Final shape after merge:
def _shape_match_kv(vec, target_tensor, model, layer_idx, proj_kind) -> tuple[Any, bool]:
    if (model is not None
        and vec.dim() == 1
        and len(target_tensor.shape) > 1
        and vec.shape[0] != target_tensor.shape[-1]):
        # Project residual-stream direction through W_K^(ℓ) / W_V^(ℓ)
        projected = _project_residual_to_kv_shape(...)
        return projected, True
    return _broadcast_to(vec, target_tensor.shape)  # your version, returns (vec, bool)
```

And in `apply` / `rollback`:
```python
dk, dk_ok = _shape_match_kv(dk, k_live, self.model, l_idx, "k")
dv, dv_ok = _shape_match_kv(dv, v_live, self.model, l_idx, "v")
if not dk_ok or not dv_ok:
    return False  # your atomic-abort design preserved
broadcast_results.append((dk, dv))

# ... pass 2 applies all mutations only after all broadcasts validated
```

**No code from your branch was modified other than this composition glue.** Your two-pass safety design is preserved end-to-end.

---

## 3. Context on GAP-C2-06 — what the W_K projection fix is

This was the silent-failure mode I worked through this week. Short version:

The steering vector `dk` is shape `(d_model,) = (2048,)` because RepE probes are calibrated on residual-stream activations. The KV cache is shape `(batch, kv_heads=8, seq, head_dim=64)` under grouped-query attention. Last-dim mismatch (2048 ≠ 64) hit `_broadcast_to`'s zero-fallback. Every α produced an identical forward pass; OEI was bit-identical across α; we couldn't actually measure anything.

**Your warning + `(vec, bool)` return surfaces the bug** (instead of silent zero). My W_K projection actually **fixes the steering**: project `dk` through the layer's `W_K` weights to get a per-head cache-space delta. Both improvements compose — your two-pass design now uses my projection as its primary path and falls back to your warning on truly-incompatible shapes (e.g., when `model=None`).

So both of us "resolved C2-06" but at different layers: yours is defensive hygiene (visibility), mine is the substantive fix. Together: clean.

This is in commit `55e8556` on main, fully written up in `docs/logs/2026-05-05_diagnostic-report.md` and the §7.4 paragraph of the paper.

---

## 4. Files excluded from the merge

| File | Why |
|---|---|
| `docs/dataset/data/sources/source_run_001.json` through `005.json` (5 × 1310 lines) | Pre-W_K-fix runs. The data shows `oei_score: 0.965016` bit-identical across α and `sigma_h_mean: 0.0` — the GAP-C2-06 silent-zero signature. Useful as historical documentation of the bug, but not the post-fix experimental results. |
| `docs/dataset/data/sources/manifest.json` | Croissant manifest referencing the 5 source runs |
| `docs/dataset/metadata.json` | Croissant metadata wrapping the 5 source runs |
| `docs/dataset/data/interventions.csv` | Same buggy run rows; reverted to pre-cherry-pick state |

**Replacing them**: with the merge landed, a clean re-run of the dataset script on the post-merge code would produce real Croissant artifacts. The runner is in `scripts/run_kv_mcts.py` and `scripts/export_kv_mcts_to_croissant.py`. Recommend running fresh dataset extraction once we lock the experimental scope for May 25.

---

## 5. Verification

### Tests

```
$ uv run pytest tests/ -v
============================ 179 passed in 20.85s =============================
```

All 179 tests green on the merged tree (your 132 new + the 47 we had before). No regressions.

### Post-merge B6 alpha-sweep on Llama 3.2-1B

Same parameters as the 2026-05-06 post-W_K-fix run for direct comparison.

| α | σ_H_mean | σ_H non-zero layers | ρ_R_mean | OEI | TDS |
|---|---|---|---|---|---|
| 0.1 | **0.228** | 14/17 | −1.306 | 0.991 | 0.084 |
| 0.5 | **0.237** | 14/17 | −1.258 | 0.991 | 0.080 |
| 1.0 | **0.247** | 14/17 | −1.198 | 0.992 | 0.076 |
| 2.0 | **0.267** | 14/17 | −1.080 | 0.992 | 0.068 |
| 5.0 | **0.330** | 15/17 | −0.533 | 0.995 | 0.040 |

**ρ_R and OEI bit-identical to the post-W_K-fix run.** Confirms your hardening didn't perturb the steering pipeline.

**σ_H column completely changed.** Pre-merge: `{0, 0, 0, 0, 0}`. Post-merge: `{0.228, 0.237, 0.247, 0.267, 0.330}` — monotonically rising with α, 14–15 of 17 layers firing. **This is your per-layer H-Neuron calibration paying off.**

Why this matters: I had spent two days documenting σ_H = 0 as a separate measurement-pipeline issue (§7.4 of the paper), with a position/context-discrepancy hypothesis as the remaining most-plausible cause. **Your per-layer fix demonstrated the actual cause was the layer-aggregate Phase 2 approximation. That hypothesis chain in §7.4 was a useful methodology but the wrong candidate cause; per-layer H-Neuron calibration (Paper Eq. 3 as actually written) was the fix.**

`residual_norm = 0.00e+00` after the sweep — Theorem 1 holds end-to-end on the merged code.

Full verification report: `docs/logs/2026-05-07_post-merge-verification-report.md`.
Appended to: `docs/logs/2026-05-05_diagnostic-report.md` ("POST-MERGE VERIFICATION — 2026-05-07" section).

---

## 6. What's next

### Paper text updates (Session A drafting today, applying tomorrow)

The σ_H result requires four paper updates under the existing Option α+ recast:
1. **§5 Experiment 1 Results table** — replace σ_H column `{0,0,0,0,0}` with the new monotonic values; replace TDS column with the new (slightly smaller) values.
2. **§5 Interpretation paragraph** — drop "σ_H reports zero throughout" framing; add "σ_H rises monotonically, indicating bottom-up channel responsiveness under per-layer calibration"; preserve Bailey-consistent framing (TDS still decreases with α — anti-OE signature holds).
3. **§7.4 paragraph 3** — substantial rewrite. Investigation arc preserved (three-prompt verification + GAP-C1-02 test were valid methodology); conclusion changes from "left as future work" to "resolved by adopting per-layer H-Neuron calibration as Paper Eq. 3 specifies."
4. **§5 Implementation Notes item ii** — remove or mark resolved (layer aggregation is no longer a Phase 2 approximation).

The narrative shift is from "two issues, one resolved, one identified-but-deferred" to **"two measurement-pipeline issues, both resolved, here are the verified results."**

### Outstanding for May 25 (not blocking, but on the radar)

- **Memory Complexity Proposition** rewrite: still gated on the simplified Exp 3 (3 VRAM measurements at b=3, d∈{3,5,10}, ~30 min runtime) — your §3.1 idea from your April 24 critique. When you have time after finals, that would close the last paper-side empirical gap. The §6 Proposition currently has placeholder constants (`~constant-factor multiple of M_KV`) that get replaced with real numbers from your measurement.
- **EMNLP 2026 style template** download + page-count check: my task; deferred to mid-week.
- **arXiv pre-print v0**: gated on Tianyu's author-list sign-off.
- **Three-prompt σ_H verification under post-merge code**: deferred to a future session. The alpha sweep alone conclusively answered the σ_H question, so this is supplementary content if pursued — useful for verifying σ_H also varies meaningfully with input regime (hallucination vs. coercion-class vs. neutral) but not load-bearing for May 25.

---

## 7. Open questions for you

1. **GAP-C1-03** (separate calibration sets — hallucination vs. coercion): your per-layer fix produces non-zero σ_H even with the existing coercion-style calibration set. Is C1-03 still in your queue, or did the per-layer fix make the calibration-content question moot? My read: per-layer fix makes σ_H respond to the steering perturbation regardless of calibration content, but proper hallucination-style calibration (per the docstring intent) is probably still cleaner. Your call on whether to pursue it post-EMNLP.

2. **GAP-C1-11** (step counter hardcoded to 0 in `_read_telemetry`): not in your two commits' resolution lists; still open?

3. **The 5 source_run JSONs**: do you want a fresh post-merge run produced as their replacement (for the Croissant package)? I can dispatch the diagnostic session to produce 5 new runs after the paper text settles.

4. **MoE monitoring (GAP-C1-04)**: still Phase B / post-EMNLP, gpt-oss-20b path? I haven't touched it; it's documented as Phase B in CLAUDE.md.

---

## 8. Acknowledgments

Two specific calls worth naming:

- **The per-layer H-Neuron calibration fix.** I had circled around σ_H = 0 for two days, ran a controlled three-prompt verification, ran a GAP-C1-02 investigation, generated a "position/context discrepancy" hypothesis as the remaining most-plausible cause, and was prepared to ship the paper with σ_H documented as a known limitation. Your fix demonstrated the actual cause was the Phase 2 layer-aggregate approximation. Adopting Paper Eq. 3 literally (per-layer H-Neuron sets) was the answer the whole time. Good catch.

- **Two-pass apply/rollback safety.** The atomic-mutation design protects against partial-state corruption that would otherwise quietly contaminate downstream telemetry. This is exactly the kind of defensive engineering the paper's empirical claims need. The composition with the W_K projection (via `_shape_match_kv` returning `(tensor, success_bool)`) was clean because of the safety design you set up.

Plus the +35 tests, including the hand-calculated formula verification — that's the kind of test methodology that catches future regressions cleanly. Genuinely strong work.

---

## 9. Pointers for catching up on context

If you want to read the recast arc (the paper went through a major reframe between your April 24 critique and now):

| File | What it is |
|---|---|
| `docs/logs/2026-05-05_recast-action-plan.md` | Live coordination doc — six pre-mortem decisions, task status, inter-session messages |
| `docs/logs/2026-05-03_premortem-findings.md` | Strategic context for the recast — the decision document that produced "Option α+" (algorithm headline, not alignment-faking-detection) |
| `docs/logs/2026-05-05_session-log.md` through `2026-05-07_session-log.md` | Daily narrative records of the recast work |
| `docs/logs/2026-05-05_diagnostic-report.md` | All diagnostic-session reports (B6 + 3B probe + bug-localization + σ_H verification + GAP-C1-02 investigation + post-merge verification) |
| `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` | The paper itself, post-recast (15 REVISION markers from May 6–7 capture the full Option α+ recast) |
| `README.md` | Public-facing summary, also recast |

The TL;DR: paper title is now "Reversible Monte Carlo Tree Search in Latent Space: A Memory-Bounded Algorithm for Inference-Time Latent-Space Analysis of Frozen Language Models." Algorithm + Theorem 1 are the headline; OE / OEI machinery is one application section complementing Bailey et al. (2024). Submission target is **EMNLP 2026 Findings (deadline 2026-05-25)** — not NeurIPS Main Track.

Take your time with finals. None of this is urgent on your side.

— Josh
