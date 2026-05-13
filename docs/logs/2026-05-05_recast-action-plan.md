# Recast Action Plan — KV-Cache Inception, EMNLP Findings 2026
**Created**: 2026-05-05
**Status**: Active — supersedes `next_session_prompt.md` (which is stale by 10 days)
**Read order for any new session**: this file first, then `docs/logs/2026-05-03_premortem-findings.md`, then the canonical TeX.

---

## 0. The decision that locked today

**The paper is being recast around the algorithmic / systems contribution.** The OE / OEI / alignment-faking machinery moves from headline framing to a single demonstration section. New shape:

| Old Contribution | New Contribution | Status |
|---|---|---|
| 1. Multi-Dimensional Micro-Telemetry Matrix | 1. **Telemetry matrix as monitoring infrastructure** (clean, supported by Max's C1 work) | unchanged in code; reframed in paper |
| 2. Reversible MCTS in KV-Cache Latent Space | 2. **Reversible KV-cache MCTS via FP32 accumulator** — Theorem 1 + memory analysis (with corrected constants + simplified Exp 3) | **NEW HEADLINE.** Already implemented and gate-passed. |
| 3. Formalization of Orthogonal Escape | 3. **Application: alpha-sweep characterization of inference-time activation redistribution** (one experiment, perplexity + OEI + ρ_R, Bailey 2024 engaged as prior) | demoted; Bailey-extension framing under Option δ if B6 lands non-degenerate; Option α (Bailey-complementing) if not |

**Working title**: *Reversible Monte Carlo Tree Search in Latent Space: A Memory-Bounded Algorithm for Inference-Time Latent-Space Analysis of Frozen Language Models.*

**Why this and not the original framing**: see `docs/logs/2026-05-03_premortem-findings.md` Failure Modes 1–6. Short version: Bailey 2024 weakens the OE novelty claim; the new delta-form OEI has never been validated; the memory headline number is wrong by 100×; the timeline doesn't support landing those issues *and* delivering empirical evidence at scale. The algorithmic contribution is rock-solid (Theorem 1 validated 2026-04-16, drift = 0 over 200 cycles) and is what this team can ship with confidence.

---

## 1. Two-session coordination protocol

There are two Opus sessions working on this paper:

| Session | Role | Active context | Best at |
|---|---|---|---|
| **Session A** (strategic) | Planning, framing, cross-session coordination, README work, this doc, Bailey reframe drafts | Pre-mortem findings + addendum, audit, Gemini critique, Max's critique | High-level structure, novelty defense, multi-source synthesis |
| **Session B** (tactical) | Direct TeX edits, paper file mechanics, REVISION marker hygiene, peer-review checklist | Canonical TeX line numbers, REVISION markers, prior session-log chain | Surgical edits, audit verification, "did the line move?" precision |

**Protocol for not losing loose ends:**

1. **This file is the source of truth.** Both sessions read it at the start of work. Both sessions append to it when they land changes.
2. **Section 4 (Task Status Board) is the only mutable status surface.** Any task transitions update it. No task moves to "Done" without a commit hash and the file:line of the change.
3. **Section 7 (Inter-session messages) is the comms log.** Either session writes a message here when they need the other to act on something. Josh reads it and pastes it across sessions.
4. **No silent edits.** Every TeX/code change gets logged with the REVISION marker pattern (`% [REVISION | <tag> | YYYY-MM-DD]`). New tags must be reflected in Section 4.
5. **`next_session_prompt.md` is deprecated.** A pointer at the top of that file directs new sessions here. Do not edit `next_session_prompt.md` further; let it rot.

---

## 2. The six pre-mortem decisions — current state

From `2026-05-03_premortem-findings.md` lines 342–353. Status as of 2026-05-05:

| # | Decision | Recommended | Status | Locked? |
|---|---|---|---|---|
| 1 | Run B6 + 3B probe before any other work? | YES | Kicked off 2026-05-05 (Josh) | ✅ Decision locked; awaiting results |
| 2 | Adopt Option δ reframe? | **Now LOCKED to Option α+** | **2026-05-05 PM update**: Bug-localization completed. Root cause is GAP-C2-06 — `_broadcast_to(dk, k_live.shape)` with `dk_shape=(2048,)` (d_model) and `k_live.shape=(1, 8, 134, 64)` (per-head KV) silently returns zeros at `kv_mcts.py:365–366` because last-dim mismatch (2048 ≠ 64). The accumulator never receives a non-zero delta; every α produces an identical forward pass; every node's telemetry is identical. **Implication**: empirical OEI claims have never been valid in this codebase (the 2026-04-16 Phase 3 gate's OEI=0.883 was likely the same artifact under the old ratio-form formula). **Resolution**: Option α+ — commit to Bailey-complementing framework structurally; in parallel attempt a 1-day mechanical fix (project `dk` through `W_K` to per-head cache space). If fix produces variation → supplementary content. If not → α paper unaffected; bug becomes a specific, well-understood §6/§7 limitations paragraph. | ✅ Locked 2026-05-05 PM (Option α+) |
| 3 | Experiment 2 disposition? | Recast as reward-function ablation | **Locked** in this session 2026-05-05 | ✅ |
| 4 | Switch style file to EMNLP 2026 + recheck page count? | YES | **Tomorrow** (style template needs download; not combining with content edits) | ✅ Decision locked; execution deferred |
| 5 | Adopt Max's simplified Exp 3 (3 VRAM measurements)? | YES | **Locked** — supersedes Path 2-A | ✅ |
| 6 | Pre-print v0 to arXiv after Day-7 fixes? | YES | **Locked** in principle; awaits Tianyu sign-off on author list | ✅ Decision locked; awaiting Tianyu confirmation |

**Net**: 5 of 6 locked, 1 awaits external sign-off (Tianyu on arXiv author list).

---

## 3. Today's plan (2026-05-05) — confirmed

| Block | Owner | Status |
|---|---|---|
| Kick off 3B probe (`probe_kv_cache_mutability.py --model meta-llama/Llama-3.2-3B-Instruct`) | Josh | ⏳ in flight |
| Kick off B6 alpha-sweep on Llama-3.2-1B with new delta-form OEI | Josh | ⏳ in flight |
| Apply T-C TDS reframe (TeX:470 + circularity caveat) | Session B | 🔲 ready to apply |
| Apply R-B §2.2 soften (TeX:126) | Session B | 🔲 ready to apply |
| Apply §5.4 cull (TeX:481, 483, 485, 487 + remove redundant CRITIQUE NOTE) | Session B | 🔲 ready to apply |
| Inspect B6 output (does new OEI vary across α non-degenerately?) | Josh | ⏳ post-runtime |
| Append B6 result to `2026-05-03_premortem-findings.md` addendum | Session A | ⏳ post-Josh-readback |
| Create this file | Session A | ✅ done |

The Read 1 edits (T-C / R-B / §5.4 cull) and the Read 2 diagnostics (B6 / 3B probe) are not mutually exclusive — kick off the diagnostics first, apply the edits while B6 runs.

---

## 4. Task Status Board (the mutable status surface)

Tags: `[A]` = Session A owns; `[B]` = Session B owns; `[Josh]` = Josh executes; `[Max]` = Max owns; `[blocked]` = waiting on something.

### 4.1 Today (2026-05-05)

| Task | Owner | Status | Reference |
|---|---|---|---|
| 3B feasibility probe | [Josh] | ✅ done 2026-05-05 — **PASS**. `gate_passed: true`, `mutable_in_place: true`, `reversible_with_copy_restore: true`, `max_logit_delta_after_revert: 0.0`. 3B is on the table for downstream experiments. Report: `docs/logs/2026-05-05_diagnostic-report.md` | premortem §addendum FM 6 prerequisites |
| B6 OEI α-sweep with new delta-form formula | [Josh] | ✅ done 2026-05-05 — **SATURATED_NEAR_1, but stronger: bit-identical**. OEI=0.994116 std=0.000000 across all 25 nodes / 5 α values / all branches. σ_H, ρ_R, TDS, reward also all bit-identical across α. Diagnostic of measurement-pipeline bug, not metric saturation. residual_norm=0.00e+00 (Theorem 1 unaffected). Report: `docs/logs/2026-05-05_diagnostic-report.md`. JSON: `tmp/runs/2026-05-05_b6_oei_diagnostic.json` | premortem FM 6 |
| T-C TDS reframe with circularity caveat | [B] | ✅ done 2026-05-05 — marker `[REVISION \| T-C-TDS-reframe \| 2026-05-05]` at TeX:471; reframed TDS bullet at TeX:472 (cites Eq.~\ref{eq:mcts_reward} + Table~\ref{tab:diagnostic_states}) | premortem decision-by-decision table; 04-25 session log lines 348–384; §7 inter-session message 2026-05-05 (A→B) |
| R-B §2.2 soften | [B] | ✅ done 2026-05-05 — marker `[REVISION \| R-B-§2.2-soften \| 2026-05-05]` at TeX:126; modified paragraph closing at TeX:127 | premortem decision-by-decision table; 04-25 session log lines 348–384 |
| §5.4 orphaned-metrics cull (ASR / pass@1 / cosine-sim / MER + redundant CRITIQUE NOTE) | [B] | ✅ done 2026-05-05 — marker `[REVISION \| §5.4-metrics-cull \| 2026-05-05]` at TeX:474; surviving §5.4 inventory: OEI (TeX:456) + TDS (TeX:472); `\end{itemize}` at TeX:475. File length 693 → 680 lines | jaunty-yawning-volcano.md §B Q7; §7 inter-session message 2026-05-05 (A→B) |
| Append B6 + 3B results to premortem addendum | [A] | ✅ done 2026-05-05 (this file §4.1 row + §7 message capture the result; full report at `docs/logs/2026-05-05_diagnostic-report.md`) | this file §6 |
| Update Section 4 of this file when each task lands | [B] (and [A] for own tasks) | ongoing | (this file) |

### 4.1.5 Bug investigation (added 2026-05-05 post-B6; closed 2026-05-05 PM)

**Outcome (2026-05-05 PM)**: localized to GAP-C2-06 — known silent-failure mode at `logomesh/kv_mcts.py:365–366` (`_broadcast_to` returns `torch.zeros(target_shape)` when source last-dim ≠ target last-dim, no warn/log). Steering vector is `(d_model=2048,)` from RepE residual-stream calibration; KV cache is `(batch=1, kv_heads=8, seq=134, head_dim=64)`. 2048 ≠ 64, so dk → zeros, accumulator gets no delta, all forward passes are identical. Confirmed via diagnostic-followup checkpoint instrumentation — `k_accum_abs_max=0.000000e+00` and `K_norm` bit-identical across α=0.0/1.0/5.0. Full localization at `docs/logs/2026-05-05_diagnostic-report.md` (LOCALIZATION FOLLOW-UP section).

| Task | Owner | Status | Reference |
|---|---|---|---|
| H1 (`get_hidden_states()` returns prompt) | — | ✅ REFUTED — h_base ≠ h_steered at every α | localization report |
| H2 (steering delta applied to wrong cache) | — | ✅ SUPPORTED with refinement — delta lands at right object but is silently zeroed by `_broadcast_to` shape-mismatch fallback | localization report |
| H3 (`h_steered` read from post-revert state) | — | ✅ REFUTED — reads happen before rollback; h_base ≠ h_steered | localization report |
| H4 (read layer ≠ steer layer) | — | ✅ REFUTED — both monitors read all layers; mismatch would be masked by H2 anyway | localization report |
| **Implication for past data**: 2026-04-16 Phase 3 gate (OEI=0.883, old ratio-form formula) almost certainly affected by same bug. Empirical OEI claims have never been valid in this codebase. Smoke-test telemetry (TeX:347–353 "all T_t / OEI / TDS fields populated") is technically true but values were artifacts. | [A] notes for §6/§7 honest finding | ✅ understood | — |

### 4.1.6 Option α+ commitments (added 2026-05-05 PM, post-localization)

**Decision 2 resolved**: Option α+ — commit to Bailey-complementing framework structurally; attempt 1-day mechanical fix in parallel; outcome routes whether fix produces supplementary content or honest §6/§7 limitations text.

| Task | Owner | Status | Reference |
|---|---|---|---|
| Lock paper structure: Option α framing (Bailey-complementing measurement framework, no empirical OEI claim as headline) | [A] | ✅ locked 2026-05-05 PM | this file §2 decision 2 |
| Draft §1 abstract under Option α (algorithm headline + Bailey-complementing application + bug-as-known-limitation) | [A] | ✅ done 2026-05-06 — drafted in `docs/logs/2026-05-06_session-A-prose-drafts.md` piece #1 | this file §5.1 |
| Draft §1 Contributions reordering (algorithm first, telemetry infra second, application third) | [A] | ✅ done 2026-05-06 — `docs/logs/2026-05-06_session-A-prose-drafts.md` piece #2 | this file §5.2 |
| Draft §2.2 Bailey 2024 substantive engagement (2–3 sentences, frames Bailey as peer measurement) | [A] | ✅ done 2026-05-06 — `docs/logs/2026-05-06_session-A-prose-drafts.md` piece #3 | premortem FM 1 |
| Draft §7 honest-finding paragraph: GAP-C2-06 mechanism, residual-stream vs. cache-space projection, three resolution approaches | [A] | ✅ done 2026-05-06 — `docs/logs/2026-05-06_session-A-prose-drafts.md` piece #4 (new §7 subsection with `\label{sec:measurement-prereqs}`) | this section + localization report |
| Apply Option α drafts to TeX (apply order: piece #4 → #2 → #3 → #1) | [B] | ✅ done 2026-05-06 — all four drafts applied + two missing labels added (Session A flagged the contingency in the drafts file). Markers landed: `[REVISION \| thm-label-add \| 2026-05-06]` at TeX:434 (added `\label{thm:reversibility}`), `[REVISION \| prop-label-add \| 2026-05-06]` at TeX:461 (added `\label{prop:memory}`), `[REVISION \| §7-measurement-prereqs \| 2026-05-06]` at TeX:488 (new §7.4 subsection between Limitations and Future Work, lands `\label{sec:measurement-prereqs}`), `[REVISION \| contributions-recast-α+ \| 2026-05-06]` at TeX:80 (full enumerate replacement; subsumes 2026-05-05 Contrib2-fix), `[REVISION \| §2.2-Bailey-engagement \| 2026-05-06]` at TeX:104 (Bailey paragraph after RL-Obfuscation paragraph), `[REVISION \| abstract-recast-α+ \| 2026-05-06]` at TeX:50 (abstract replacement). All cross-references resolve: `\ref{thm:reversibility}`, `\ref{prop:memory}`, `\ref{sec:measurement-prereqs}`, `\ref{eq:mcts_reward}`, `\ref{eq:oei}`. File length 600 → 618 lines. §7 placement choice (between Limitations and Future Work) reasoned in §7 reply 2026-05-06 (B→A). | this file §5 |
| Attempt 1-day mechanical fix: project `dk` through `W_K` per layer (`dk_cache = (W_K @ dk).reshape(kv_heads, head_dim)` then broadcast across seq) | [Josh] | ✅ done 2026-05-06 PM — fix landed in `logomesh/kv_mcts.py` (W_K/W_V projection in `_project_residual_to_kv_shape` + accumulator router); 47/47 unit tests pass; per Session A's §7 PM message. | localization report; `model.model.layers[ℓ].self_attn.k_proj.weight` |
| If fix produces variation: re-run B6 with fix, record results in `tmp/runs/2026-05-06_b6_post-fix.json` | [Josh] | ✅ done 2026-05-06 PM — post-fix B6 run shows monotonic per-α variation: OEI 0.9909→0.9950, ρ_R −1.306→−0.533, TDS 0.094→0.046 across α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}; residual_norm = 0.00e+00 (Theorem 1 holds end-to-end). Raw data at `tmp/runs/2026-05-06_b6_post-fix.json`; full report appended to `docs/logs/2026-05-05_diagnostic-report.md`. | (this section) |
| If fix produces variation: integrate into supplementary results table or appendix | [B] | ✅ done 2026-05-06 PM — integrated as primary §5 Experiment 1 results (not supplementary): Table~\ref{tab:exp1-results} + Results paragraph + Interpretation paragraph in piece #6. Marker `[REVISION \| Exp1-reframe-with-results-α+ \| 2026-05-06]` at TeX:335. | this file §5 |
| If fix fails or produces uninterpretable output: §7 honest-finding paragraph (already drafted) absorbs the attempt as evidence the geometric question is non-trivial | [A] | ✅ N/A — fix succeeded; piece #4-revised reflects the fix-landed reality (no "early version" hedging, references §5 results) | piece #4-revised in drafts-v2 file replaces morning's #4 |
| Apply Option α+ drafts v2 (#4-revised, #5, #6) + add `\label{sec:experiments}` | [B] | ✅ done 2026-05-06 PM — four new markers landed: `[REVISION \| sec-experiments-label-add \| 2026-05-06]` at TeX:294 (added `\label{sec:experiments}` on §5 heading); `[REVISION \| §3-reframe-α+ \| 2026-05-06]` at TeX:132 (§3.1 reframed from "Alignment Faking Detection Problem" to "Inference-Time Latent-Space Analysis on Frozen Models"; Eq.~\ref{eq:alignment_faking} + Hypothesis~\ref{hyp:orthogonal_escape} preserved verbatim; "will redistribute" softened to "may redistribute" with new "under inference-time KV-cache perturbation" qualifier in hypothesis title); `[REVISION \| §7.4-post-fix-update \| 2026-05-06]` at TeX:515 (replaced morning's §7.4 subsection content; subsection retitled "Empirical Measurement Prerequisites and Initial Observations"; removed "early version" placeholder, added σ_H = 0 finding paragraph, references §\ref{sec:experiments} for results); `[REVISION \| Exp1-reframe-with-results-α+ \| 2026-05-06]` at TeX:335 (Experiment 1 reframed from hypothesis-test to measurement-and-report; Table~\ref{tab:exp1-results} with post-fix per-α numbers; Interpretation paragraph states result is consistent with Bailey 2024 negative regime — TDS *decreases* with α, opposite OE signature; subsumes 2026-04-25 Exp1-1B3B-framing marker). All cross-references resolve: `\ref{thm:reversibility}`, `\ref{prop:memory}`, `\ref{sec:measurement-prereqs}`, `\ref{sec:experiments}`, `\ref{eq:oei}`, `\ref{eq:mcts_reward}`, `\ref{eq:hneuron_stress}`, `\ref{eq:alignment_faking}`, `\ref{hyp:orthogonal_escape}`, `\ref{tab:exp1-results}`. File length 618 → 645 lines. | this file §5; `2026-05-06_session-A-prose-drafts-v2.md` |

### 4.2 This week (Days 2–7, 2026-05-06 → 2026-05-11)

| Task | Owner | Status | Reference |
|---|---|---|---|
| Download EMNLP 2026 style template | [Josh] | 🔲 not started | premortem decision item 4 |
| Switch `\usepackage[main]{neurips_2026}` → EMNLP package | [B] | 🔲 blocked on style template download | premortem FM 5 + audit Q5 |
| Recompile + check page count vs. EMNLP 8-page Long Papers limit | [B] | 🔲 blocked on style switch | premortem FM 5 |
| Draft new abstract (paper-recast framing — algorithm headline, OE as application) | [A] | 🔲 not started | this file §0; Section 5.1 below |
| Draft new §1 Contributions paragraph (algorithm-first ordering) | [A] | 🔲 not started | this file §5.2 |
| Apply Memory Complexity Proposition rewrite (corrected 2–3·M_KV constants) | [B] | 🔲 blocked on Max's simplified Exp 3 measurement | premortem FM 2 + Max §3.1 |
| Run Max's simplified Exp 3: 3 VRAM measurements at b=3, d∈{3,5,10} | [Josh] | 🔲 not started | Max critique §3.1 |
| Convert 5 DRAFT-HOLD blocks to single visible "Implementation Notes" paragraph at start of §5 | [B] | ✅ done 2026-05-05 — consolidated paragraph at TeX:287 (`[REVISION \| §5-impl-notes \| 2026-05-05]`); 5 DRAFT-HOLD blocks replaced with one-line REVISION markers at TeX:184, 231, 261, 315, 450 (`DRAFT-HOLD-block{1..5}-consolidated`). Asymptotic claims preserved; constants/operational details deferred to Phase 3 explicitly. File length 680 → 600 lines. Note: §6 Memory Complexity Proposition headline numbers (40.05GB / 9.7TB) are still TODO pending Max's simplified Exp 3. | premortem FM 5 + Max §3.4 |
| Recast Experiment 2 as reward-function ablation (paper text only, no code) | [B] | ✅ done 2026-05-05 — marker `[REVISION \| Exp2-recast-reward-ablation \| 2026-05-05]` at TeX:331; new Experiment 2 paragraph at TeX:332. Three arms: full telemetry reward vs σ_H-only reward (λ_2=λ_3=0) vs random-KV baseline. Text-space methods moved from primary baselines to qualitative §2 discussion. IMPL note flags B6 measurement-pipeline bug as blocker for execution. | premortem FM 3 + this file §0 |
| Wire perplexity measurement into steered-generation script | [Josh] (code) | 🔲 not started | Gemini #2 |
| Bailey 2024 substantive engagement in §2.2 + §3 | [A] (drafts) → [B] (applies) | ✅ **done 2026-05-06**. §2.2 marker `§2.2-Bailey-engagement` at TeX:104 (AM). §3 marker `§3-reframe-α+` at TeX:132 (PM) — Bailey explicitly cited at the end of the new §3.1 prose ("Bailey et al. characterized the analogous question for gradient-trained input-space attacks and reported a tentative negative result; we formulate the inference-time variant as Hypothesis~\ref{hyp:orthogonal_escape}"). | premortem FM 1 + Option α+ |
| §1/§3/§5/§7 OEI reframe language (Option α+) | [A] (drafts) → [B] (applies) | ✅ **done 2026-05-06**. §1 abstract + §1 Contributions: AM markers (`abstract-recast-α+` at TeX:50; `contributions-recast-α+` at TeX:80). §3 PM marker `§3-reframe-α+` at TeX:132. §5 Experiment 1 PM marker `Exp1-reframe-with-results-α+` at TeX:335 (with Table~\ref{tab:exp1-results} integrating post-fix data). §7.4 AM marker `§7-measurement-prereqs` at TeX:514 + PM update `§7.4-post-fix-update` at TeX:515. Hypothesis 1 wording softened ("will redistribute" → "may redistribute"; title qualified "under inference-time KV-cache perturbation"). | premortem FM 1 |
| §1 Contribution 2 fix (remove "up to 20B parameters" + "Experiment 3" reference) | [B] | ✅ done 2026-05-05 — marker `[REVISION \| Contrib2-fix \| 2026-05-05]` at TeX:83. Replacement points to Theorem 1 for the reversibility guarantee at the empirically tested 1B+3B scale and explicitly defers 20B-scale memory profiling to Phase B. Obsolete CRITIQUE NOTE removed. Asymptotic memory-reduction claim earlier in the paragraph (O(b^d·M) → O(M+d·K)) preserved; only the unverifiable scale claim and broken Exp 3 ref removed. | premortem decision-by-decision; 04-25 log; CRITIQUE NOTE at TeX:83–84 |
| Reframe §7.1 from attack-implications verbs to monitor-design-implications verbs | [B] | ✅ done 2026-05-05 — marker `[REVISION \| §7.1-monitor-design-verbs \| 2026-05-05]` at TeX:464; new heading "Implications for Latent-Space Monitor Design" at TeX:465. Both paragraphs reframed: attack/arms-race language ("monitors fundamentally insufficient", "evasion strategies", "alignment assurance against capable models") replaced with monitor-design diagnostics framing that explicitly states KV-cache MCTS requires white-box access and is therefore an interpretability tool, not an inference-time attack vector. Substantive observations (null-space dimensionality, fixed-monitor blind spots, dynamic-monitor opportunity) preserved. | premortem decision-by-decision; Gemini #1 |
| §2.1 audit + soften under Option α+ framing | [A] (audits + drafts) → [B] (applies) | ✅ **done 2026-05-07** — Session A audit found §2.1's opening sentence and closing clause positioned alignment faking as the paper's central concern; under Option α+ this should read as one application area. Two sentence-level softens drafted in `docs/logs/2026-05-07_session-A-prose-drafts-v4-§2.1-soften.md` (piece #7) and applied: marker `[REVISION \| §2.1-soften-α+ \| 2026-05-07]` at TeX:96. Opening: "...has been a longstanding concern" → "...is one motivating application area for latent-space analysis on frozen models, and a longstanding concern". Closing: "motivating the white-box representational monitoring approach our telemetry matrix provides" → "one motivation for the inference-time latent-space analysis framework developed in this paper, of which the telemetry matrix is the monitoring component". Middle sentences (Greenblatt + Hubinger summaries with caveats) preserved verbatim. §2 reordering considered and rejected (cross-reference risk too high this close to deadline). All §1 / §2.1 / §2.2 / §3 / §5 / §7 framing now consistent end-to-end under Option α+. | §7 message 2026-05-06 PM (Session A → Session B) flagging the audit as deferred; §7 message 2026-05-07 (drafts ready) |
| Title pivot under Option α+ (closes residual half-recast at TeX:39) | [A] (drafts) → [B] (applies) | 🔲 **drafted 2026-05-07 (post-Gemini-receipt)** — piece #10 in `docs/logs/2026-05-07_session-A-prose-drafts-v7-title-pivot.md`. Single-line replacement at TeX:39. Marker `[REVISION \| title-recast-α+ \| 2026-05-07]`. Adopts the working title canonical in this action plan §0 and the README rewritten 2026-05-07 ~17:00 (verbatim match). Estimated ~2 min Session B application. Triggered by Gemini FM 1 title-routing concern; **does not** absorb Gemini's counter-proposal (which would reverse Decision 2). Body has read as algorithm-headlined since 2026-05-06; title was the residual half-recast. | §7 message 2026-05-07 PM (later, post-Gemini-receipt) — Session A → Session B |
| Gemini FM 2 scale-language tightening + FM 4 "Phase 2 / Phase 3" tone tightening | [A] (drafts) → [B] (applies) | 🔲 **drafted 2026-05-07 (later, post-piece-#10)** — pieces #11 (FM 2; sub-pieces 11A §5 Interpretation paragraph + 11B §7.3 Limitations) and #12 (FM 4; sub-pieces 12A–12D across §5 Implementation Notes + §7.4 paragraph 3) in `docs/logs/2026-05-07_session-A-prose-drafts-v8-gemini-FM2-FM4-mitigations.md`. Six sub-pieces total, all pure-prose (no code/data execution required). Markers: `Exp1-interp-scale-disclaimer`, `§7.3-scale-caveat`, `§5-impl-notes-header-tone-tighten`, `§5-impl-notes-opening-tone-tighten`, `§5-impl-notes-items-tone-tighten`, `§7.4-σH-resolved-tone-tighten`. Net page-budget impact ~+90 words (negligible). Estimated ~10 min Session B application. | §7 message 2026-05-07 PM (later, post-Gemini-receipt) — Session A → Session B |
| Gemini FM 3 perplexity column + Degradation Control Baseline (random orthogonal vector arm in §5 Experiment 1) | [A] (drafts) → [B] (applies); also requires [Josh] code-lane work | 🔲 not started — gated on Josh wiring perplexity into the steered-generation script (action plan §4.2 row 136). Tonight's session was constrained by no-test-execution (user's machine running video renders), so this piece must wait for either the perplexity wiring to land or a separate session where test execution is available. Will draft as piece #14+ once perplexity numbers exist. | §7 message 2026-05-07 PM (later, post-Gemini-receipt) |
| Gemini FM 5 light Bailey sharpening (§2.2 paragraph at TeX:104 marginal tightening) | [A] (drafts) → [B] (applies) | 🔲 not started — deferred to piece #13+ in a subsequent session; optional addition pending second-pre-mortem variance check (Josh planning a re-run of the same Gemini prompt for response-variance testing) | §7 message 2026-05-07 PM (later, post-Gemini-receipt) |

### 4.3 Days 8–14 (2026-05-12 → 2026-05-18)

| Task | Owner | Status | Reference |
|---|---|---|---|
| Run alpha-sweep on Llama 3.2 1B (and 3B if probe passed) over 30–50 Alpaca-eval prompts | [Josh] | 🔲 blocked on Days 2–7 | premortem synthesized plan item 13 |
| Generate results plots (OEI / perplexity / ρ_R as function of α) | [Josh] | 🔲 blocked on alpha-sweep | premortem synthesized plan item 15 |
| Integrate alpha-sweep results into §5 / §6 (whichever section becomes the application) | [B] | 🔲 blocked on plots | (paper recast structure) |
| Pre-print v0 to arXiv | [Josh] | 🔲 blocked on Tianyu sign-off + Days 2–7 fixes | premortem synthesized plan item 16 |
| README rewrite to reflect recast | [A] (drafts) → [Josh] (commits) | 🔲 not started | this file §6 |
| `next_session_prompt.md` retire / rewrite | [A] | 🔲 will do after pre-mortem decisions all lock | this file §1 |

### 4.4 Days 15–20 (2026-05-19 → 2026-05-25)

| Task | Owner | Status | Reference |
|---|---|---|---|
| Second-pass code-paper consistency check | [B] | 🔲 not started | premortem synthesized plan |
| Final REVISION marker audit (`grep -n "REVISION" docs/NeurIPS/04.18...tex`) | [B] | 🔲 not started | 04-25 session log line 71 |
| Submission prep + EMNLP-specific checklist | [B] + [Josh] | 🔲 not started | (EMNLP submission portal) |
| Submit by 2026-05-25 | [Josh] | 🔲 not started | EMNLP Findings deadline |

### 4.6 Standing coordinated-update tracker

Items that span multiple TeX locations and must be updated together when their gating event occurs. Listed here so neither session forgets the cross-section consistency.

| Coordinated update | TeX locations | Gating event | Owner |
|---|---|---|---|
| Memory-complexity numbers: replace "constant-factor multiple" / "approximately" placeholders with measured constants | Abstract (TeX:51), §1 Contribution 1 (TeX:81), §5 Implementation Notes item iv (TeX ~298), §6 Memory Complexity Proposition (TeX:545–551) | Max's simplified Exp 3 (3 VRAM measurements at b=3, d∈{3,5,10}) lands | [A] drafts coordinated patch → [B] applies as one batch |
| σ_H finding: full resolution arc. (1) 2026-05-06 PM three-prompt verification refuted H_A (calibration-prerequisite framing). (2) 2026-05-07 PM GAP-C1-02 investigation ruled out per-neuron normalization as the cause and named position/context discrepancy as the remaining hypothesis. (3) **2026-05-07 (later) post-merge B6 verification** (`docs/logs/2026-05-07_post-merge-verification-report.md`) confirmed Max's per-layer H-Neuron calibration (commit `ca3ec8e` from `max-toscano@976fce5`) is the fix — σ_H now rises monotonically with α (0.228→0.330; 14–15/17 layers firing); ρ_R/OEI bit-identical pre/post merge; TDS shifts ~7–13% downstream of σ_H now non-zero; Theorem 1 holds end-to-end. **Paper updates from drafts v6 piece #9 applied 2026-05-07 PM**: `Exp1-table-σH-TDS-update` (TeX:353) — Table 1 σ_H + TDS rows + caption rewrite reflecting both issues resolved; `Exp1-results-σH-resolved` (TeX:341) — Results paragraph reorder (OEI → ρ_R → σ_H → TDS) with new monotonic σ_H sentence + updated TDS numbers; `§7.4-σH-resolved` (TeX:529) — substantive §7.4 paragraph 3 rewrite from "second measurement-pipeline issue, resolution left as future work" → "Both measurement-pipeline issues...are therefore resolved in the implementation reported here." Position/context-discrepancy hypothesis from the v5 patch dropped entirely (empirically refuted). Five §7.4 markers now layered in TeX as full chronological resolution arc: `§7-measurement-prereqs` (TeX:516) → `§7.4-post-fix-update` (TeX:517) → `§7.4-σH-verification-update` (TeX:525) → `§7.4-σH-investigation-update` (TeX:528) → `§7.4-σH-resolved` (TeX:529). Net effect: paper now identifies two measurement-pipeline issues under Option α+, **both resolved in the implementation reported here.** Piece #3 of drafts v6 (Implementation Notes item ii cleanup) was NOT applied — flagged via §7 reply for Session A clarification (item ii is RepE-side, while the H-Neuron fix is a separate concern not yet documented in Implementation Notes). | §7.4 paragraph 3 + §5 Experiment 1 Results paragraph + Table 1 + (item ii cleanup pending Session A clarification) | ✅ **CLOSED — RESOLVED 2026-05-07 PM** via Max's per-layer H-Neuron calibration | [A] drafted ✅ → [B] applied ✅ (v3, v5, v6 — three patches) |
| Bug-fix supplementary content: if richer post-fix data emerges (e.g., per-layer OEI, alternate monitor specifications) | §5 supplementary or appendix | Optional follow-up by Josh | [A] drafts → [B] applies |

### 4.5 Max's lane (Contribution 1 — unchanged from his briefing)

| Task | Owner | Status | Reference |
|---|---|---|---|
| GAP-C1-02 per-neuron ReLU | [Max] | ⏳ ongoing — **2026-05-07 confirmed empirically NOT applied** in current `_raw_dense_score` (logomesh/hneuron_monitor.py:247–251); also confirmed: applying it tentatively does NOT fix σ_H = 0 — cause is downstream of per-neuron normalization | max-briefing-2026-04-21.md |
| GAP-C1-03 separate calibration sets (hallucination, not coercion) | [Max] | ⏳ ongoing — **2026-05-06 verification confirms the issue empirically**: `scripts/run_kv_mcts.py:55–70` puts `_COERCED_EXAMPLES` (jailbreak-style) in the `hallucinated_examples` slot of HNeuronMonitor calibration; should be hallucination-style per `hneuron_monitor.py:23–27` docstring (e.g., "Battle of Zorvak in 1842", "moons of Planet Quasar-7") | max-briefing-2026-04-21.md |
| **GAP-C1-XX dense-scoring-path silent failure** (identified 2026-05-06; refined 2026-05-07; **resolved 2026-05-07 PM**) — σ_H constancy traced through three-prompt verification + per-neuron-ReLU ruling-out; **per-layer H-Neuron calibration was the actual fix.** Position/context-discrepancy hypothesis empirically refuted. Post-merge B6 verification (`docs/logs/2026-05-07_post-merge-verification-report.md`) confirms σ_H rises monotonically with α (0.228→0.330; 14–15/17 layers firing); ρ_R/OEI bit-identical pre/post merge; TDS shifts ~7–13% downstream of σ_H now non-zero; Theorem 1 holds (residual_norm = 0). Paper updates landed via piece #9: Table~\ref{tab:exp1-results} (`Exp1-table-σH-TDS-update` at TeX:353), §5 Experiment 1 Results paragraph (`Exp1-results-σH-resolved` at TeX:341), §7.4 paragraph 3 substantive rewrite (`§7.4-σH-resolved` at TeX:529; conclusion now "both measurement-pipeline issues resolved" rather than "one resolved, one identified-but-deferred"; position/context hypothesis dropped entirely). | [Max] (investigation closed) | ✅ **resolved 2026-05-07 PM** via Max's commit `ca3ec8e` from `max-toscano@976fce5` | `docs/logs/2026-05-07_post-merge-verification-report.md` |
| GAP-C1-11 step counter | [Max] | ⏳ ongoing | max-briefing-2026-04-21.md |
| GAP-C1-01 per-layer H-Neuron | [Max] | ✅ **resolved 2026-05-07** via Max's commit `ca3ec8e` from `max-toscano@976fce5` (per-layer top-K H-Neuron indices and per-layer baselines, matching Eq.~\ref{eq:hneuron_stress} as written) | max-briefing-2026-04-21.md |
| Phase 3 hold: GAP-C1-05 PCA conversion | [Max] | 🛑 hold | max-briefing-2026-04-21.md |

---

## 5. Recast structural mapping (paper-side)

What moves where in the new shape. This is the spec Session B will eventually apply once the conditional decisions resolve.

### 5.1 New abstract framing (Session A drafts this week)

The current abstract leads with "alignment faking" and ends with a three-contribution list where OE is the third. The recast leads with the algorithmic/memory-bounded-search story, frames alignment monitoring as one application domain, and engages Bailey 2024 in the third sentence (not in body Related Work only).

Draft target shape (Session A will produce actual prose this week):
1. *Hook*: Latent-space analysis of frozen LLMs requires search infrastructure that scales.
2. *Problem*: Standard MCTS in KV-cache space requires O(b^d) memory; existing alternatives (Bailey 2024 gradient-based, RepE static steering) sit at different points on the access/cost frontier.
3. *Contribution*: Reversible KV-cache MCTS via FP32 accumulator — O(M_KV) memory, exact rollback (Theorem 1), enables inference-time latent search at scale on commodity hardware.
4. *Application*: We demonstrate the algorithm on activation-redistribution measurement under inference-time perturbation, extending Bailey 2024's gradient-based negative result to a different attack regime.
5. *Result*: <one sentence about the alpha-sweep result, written after experiments land>

### 5.2 New §1 Contributions ordering

| Position | Old | New |
|---|---|---|
| 1 | Telemetry matrix | Reversible KV-cache MCTS (algorithmic + memory analysis + Theorem 1) |
| 2 | Reversible MCTS | Telemetry matrix monitoring infrastructure |
| 3 | Orthogonal Escape formalization | Application: alpha-sweep characterization (Bailey extension) |

The OE/OEI machinery does not disappear — it becomes infrastructure for Contribution 3 (the application). The first formal definition claim is dropped.

### 5.3 §-by-§ change map

| Section | Current | After recast |
|---|---|---|
| Abstract | "alignment faking" hook | algorithmic-search hook with monitoring application |
| §1 Intro | alignment-faking motivation | latent-space-analysis-needs-better-search motivation |
| §1 Contributions | 3 contribs, OE third | 3 contribs, algorithm first, OE-as-application third |
| §2.1 (alignment faking background) | survives, briefer | survives but downgrades from primary motivation to "application context" |
| §2.2 (RepE / activation steering) | survives | gets Bailey 2024 substantive engagement at end |
| §3 (Problem Formulation) | "alignment faking detection problem" framing | "inference-time latent search problem" framing; alignment-faking definition becomes one of multiple use cases |
| §4 (Method) | telemetry matrix + reversible MCTS, equal weight | reversible MCTS leads, telemetry matrix is monitoring layer |
| §5 (Experimental Design) | Phase A + Experiments + 3 cut | unchanged scope; Experiment 2 recast as reward-function ablation |
| §5.4 (Metrics) | post-cull, with TDS reframed (T-C) | unchanged after today's edits |
| §6 (Theory) | Reversibility Theorem + Memory Proposition | gets corrected constants from Max simplified Exp 3 |
| §7.1 (Implications for Alignment Assurance) | "attack" verbs | "monitor design" verbs (Gemini #1) |
| §7 (Limitations) | extant + G1 caveats | + Bailey-OEI relationship caveat + perplexity/coherence caveat from Gemini #2 |
| §7 (Future Work) | extant | + scale-up note (gpt-oss-20b at H100) |

### 5.4 What the OE/OEI machinery becomes

A single application section (probably §6 or §5.3 — exact placement is Session B's call when the recast lands). Frames as: "the algorithm enables — among other things — a previously-difficult measurement: the fraction of inference-time activation change invisible to a specific linear monitor (OEI). We demonstrate this on the alignment-monitoring task as one application; the technique generalizes."

This positions Bailey 2024 as a **peer measurement** (Bailey: gradient-based, full-orthogonality, negative result; ours: inference-time, targeted-monitor, observational) rather than a competitor whose novelty claim we're contesting. Materially safer reviewer territory.

---

## 6. README rewrite plan (Session A drafts, Josh commits)

The README currently reflects the original framing (KV-Cache Inception, alignment-faking detection, three contributions in old ordering). Will need to be rewritten to:

1. Match the new working title and contribution ordering.
2. Remove the "first formalization of Orthogonal Escape" framing.
3. Present Theorem 1 + reversible MCTS as the headline result.
4. Frame the OEI work as "an application of the algorithm to alignment monitoring."
5. Reflect the EMNLP Findings target (currently mentions NeurIPS 2026).
6. Update Phase Status table (Phase 3 Experiment Infrastructure status flips to "in flight" once today's diagnostics land).
7. Update Hardware table (3B fits / doesn't fit, depending on probe result).
8. Update the "gpt-oss-20b gotcha" section to mark it as Phase B only (already done in CLAUDE.md, README may not be in sync).

**Trigger to start the README rewrite**: after B6 + 3B results land and the conditional decisions (Option δ vs. α, 1B-only vs. 1B+3B) resolve. Holding it until then prevents two rewrites.

**Owner**: Session A drafts; Josh applies. Targeting Days 8–14 of the 22-day plan.

---

## 7. Inter-session messages

Append-only log. Either session can write here when they need the other to act on something. Josh pastes across.

### 2026-05-05 — Session A → Session B

**Subject**: Read 1 edits cleared to apply today; Bailey/Option-δ work blocked on B6.

You can proceed with the three Read 1 edits (T-C, R-B, §5.4 cull) immediately — they're scoped, isolated, and don't conflict with the recast direction. **One addition** for the T-C edit: when reframing TDS, *include the circularity caveat* in the language. Same JSD value is the third reward term (Eq. 8 λ₃) and the diagnostic signature for OE state in Table 1; T-C should explicitly state TDS is "the divergence component of the MCTS reward, defined here for completeness; not separately tabulated in submission results because the search optimizes it directly." This is what makes the reframe honest rather than just a relabeling.

Hold the OEI Option B language across §1/§3/§5/§7 until B6 lands. The reframe target may be Option δ (Bailey extension) or Option α (Bailey complement) depending on whether the new delta-form OEI is non-degenerate. Either text is short enough that Session A can draft both this week and you'll apply whichever lands.

GAP-C3-10 walkback: Session A's pre-mortem incorrectly flagged this as a bug. Your audit was right — root-relative `h_base` is the correct semantic for B6 alpha-sweep (each α compared to no-steering reference). No code change needed before B6.

When you finish today's three edits, please update Section 4.1 of this file with the file:line of each change and the REVISION marker tag. That's how Session A and Josh track the loose ends.

### 2026-05-05 — Session A → Josh

When B6 finishes, please paste:
- The OEI value distribution across α (or just `oei_score` per node from the JSON).
- A one-line read: "non-degenerate / saturated near 0 / saturated near 1 / inconclusive."
- Any errors / warnings during the run (especially around the `_broadcast_to` silent-zero path or dummy OEI fallback).

Session A will append the result to the pre-mortem addendum and resolve the conditional decisions (Option δ vs α; recast §1/§2/§3 prose drafts).

For the 3B probe: "passed" / "OOM" / "other error" + the actual VRAM number from the probe output. That answers whether Days 8–14 alpha-sweep includes 3B.

### 2026-05-05 — Session A → Session B (post-B6)

**Subject**: B6 returned bit-identical telemetry; Option δ on hold pending bug investigation; do not start Bailey/§1/§2.2/§3 reframe drafts.

The B6 alpha-sweep ran to completion with `residual_norm = 0.00e+00` (Theorem 1 fine) but produced **bit-identical OEI = 0.994116, σ_H = 0, ρ_R = -0.8049, TDS = 0.0844 across all 25 nodes / 5 α values / all branches** (std = 0.000000 within every α bucket). This is not metric saturation — saturation would still produce variance from per-perturbation noise. Bit-identical to six decimals across two independent telemetry channels is diagnostic of a measurement-pipeline bug: the hidden states being delivered to the telemetry calculation appear constant across all α, regardless of what steering was applied.

Full report: `docs/logs/2026-05-05_diagnostic-report.md`. JSON: `tmp/runs/2026-05-05_b6_oei_diagnostic.json`.

**Implications for your queue**:

- **Do NOT start the Bailey 2024 §2.2/§3 substantive engagement drafts** — Session A had drafts queued under both Options δ and α. The δ draft is now blocked: framing the paper as "first controlled measurement of OEI extending Bailey 2024" is dead until OEI produces non-degenerate values.
- **Do NOT start the §1/§3/§5/§7 OEI reframe language application** — same blocker.
- **Days 2–7 work that does NOT depend on the OEI result is still cleared to proceed**: download EMNLP 2026 style template, switch the style package and recheck page count, recast Experiment 2 paragraph as a reward-function ablation, convert the 5 DRAFT-HOLD blocks into a single visible §5 "Implementation Notes" paragraph, fix the §1 Contribution 2 "up to 20B parameters" + "Experiment 3" reference. None of these depend on the bug investigation.
- **Memory Complexity Proposition rewrite is still gated on Max's simplified Exp 3 measurement** — unchanged.

Session A is updating §4.1.5 of this file with a 2-hour bug-investigation task list (4 hypotheses to check). Josh decides today/tomorrow whether to spend that 2 hours; the outcome routes us between Option δ (if fixable) and Option α (if not).

GAP-C3-10 walkback noted by your audit was correct *for the case the alpha-sweep is supposed to be measuring* (root-relative baseline for independent steerings of the same prompt). The bug we're now hunting is a different and more fundamental issue: the hidden states themselves appear not to vary with the steering applied. This is not a re-litigation of GAP-C3-10.

### 2026-05-05 — Session A → Josh (post-B6, action items)

Three things, ordered by leverage:

1. **The 2-hour bug investigation** in §4.1.5 of this file. Hypothesis 1 (`get_hidden_states()` returns prompt vs. generation-step) and Hypothesis 4 (read layer vs. steer layer) are the cheapest to verify — start there. Adding a few `print(h.shape, h.norm(), h.flatten()[:5])` calls before/after the steered forward pass at `kv_mcts.py:_read_telemetry()` will reveal in minutes whether the read state actually varies with α.

2. **3B is unambiguous good news.** `gate_passed: true`, `reversible_with_copy_restore: true`, `max_logit_delta_after_revert: 0.0`. 3B fits and the reversibility math holds at scale. Whatever scope we land for the alpha-sweep can include 3B.

3. **Theorem 1 is unaffected.** `residual_norm = 0.00e+00` after a 25-node, depth-3, 5-branches sweep. The headline contribution of the recast paper is intact regardless of what the bug investigation finds.

If the bug is found and fixed in <1 day → re-run B6 → likely Option δ stands, full recast plan proceeds. If the bug is deep or unfixable in budget → Option α (Bailey-complementing framework) — paper still ships on the strength of Theorem 1 + the framework, with the bug documented as a §6 finding ("our reference implementation produces constant telemetry across α; we trace this to [cause] and identify [direction] as required for empirical OEI measurement"). Honest, publishable.

### 2026-05-05 PM — Session A → Session B (post-localization, decision 2 resolved)

**Subject**: Decision 2 locked to Option α+. Bailey/§1/§2.2/§3/§5/§7 reframe drafts coming from Session A tomorrow. Memory Complexity Proposition still gated on Max.

The bug investigation is closed. Localized to GAP-C2-06 (`_broadcast_to` shape-mismatch fallback at `kv_mcts.py:365–366`) — the steering vector is `(d_model=2048,)` but the KV cache has per-head last-dim of 64, so the silent-zero fallback fires every time. The accumulator never receives a non-zero delta. Every α produces an identical forward pass. This explains the bit-identical telemetry across all channels (σ_H, ρ_R, TDS, OEI) in the original B6. Diagnostic-followup report is in `docs/logs/2026-05-05_diagnostic-report.md` (LOCALIZATION FOLLOW-UP section); raw localization JSON is at `tmp/runs/2026-05-05_b6_localize.json`.

**Decision 2 resolution**: Option α+. Paper structurally commits to the Bailey-complementing measurement framework (no empirical OEI claim as headline). In parallel, Josh attempts a 1-day mechanical fix (project `dk` through `W_K` per layer to per-head cache space). If the fix produces variation, supplementary content; if not, the bug becomes a specific, well-understood §6/§7 limitations paragraph.

**What this means for your queue**:

- **You may now begin §1/§2.2/§3/§5/§7 drafting prep** — but Session A is producing the prose drafts, not you. Your role tomorrow is to apply Session A's drafted text via REVISION-marker edits, same protocol as the Read 1 edits. Specifically: Session A will hand you (1) a new abstract, (2) a new §1 Contributions paragraph with reordered contribution list, (3) Bailey 2024 substantive engagement for §2.2, (4) an honest-finding paragraph for §6 or §7 (Session A will pick the section based on best fit). Apply each with a REVISION marker tagged `[REVISION | <slug> | 2026-05-06]`.
- **Memory Complexity Proposition rewrite still gated on Max's simplified Exp 3**. Your call to leave the §6 proposition unchanged while §5 Implementation Notes acknowledges the inconsistency was correct. Don't close the loop until Max's number lands.
- **EMNLP style switch is still cleared work** — you can do this whenever Josh has the template. Doesn't depend on any of the above.
- **§1 Contribution 2 already says "the algorithm enables …" pointing to Theorem 1 instead of "verify in Exp 3"** — your edit at line 83 today resolved that. Confirmed clean.

**About your "Holding for the bug investigation outcome and Max's measurement"**: that's correct, not confused. Two open external dependencies, you're holding on the right ones. (Josh wondered whether you were unsure who's doing what — the answer is you correctly identified two distinct gates and chose to wait rather than edit speculatively. Good call.)

The §6 / §5 internal inconsistency you flagged (Implementation Notes acknowledges 40.05 GB is wrong, §6 still cites it) — keep it as flagged. Max's measurement will close the loop. The §5 note is the right place to surface it.

### 2026-05-05 PM — Session A → Diagnostic Claude session (closing thanks, no new work)

The localization was excellent diagnostic work. Specifically: catching that `--branches 1` was incompatible with the run shape and substituting `--branches 3` (with explicit flagging) is exactly the kind of pre-flight reasoning we want from a delegated session. The four-checkpoint instrumentation pattern with file-restore-via-diff verification is the protocol going forward for any further localization runs.

No new diagnostic work today. If a future bug-fix attempt requires re-running B6 to verify variation, the same harness applies — instrument, run, diff-restore, report.

### 2026-05-06 — Session A → Session B (drafts ready)

**Subject**: Four prose drafts ready in `docs/logs/2026-05-06_session-A-prose-drafts.md`. Apply when ready.

All four pieces of Option α+ paper text are drafted and ready for application:

1. **Replacement abstract** (replaces TeX:50–52 in full)
2. **Replacement §1 Contributions list** (replaces TeX:79–92 — the three `\item` entries)
3. **§2.2 Bailey 2024 substantive engagement paragraph** (insert after the existing RL-Obfuscation paragraph at TeX ~107, before §2.3)
4. **New §7 subsection: "Empirical Measurement Prerequisites for OEI"** (insert into §7 Discussion; carries `\label{sec:measurement-prereqs}` referenced by piece #2)

Application order in the drafts file: #4 → #2 → #3 → #1. Reason: piece #4 lands the label that pieces #1 and #2 reference forward.

REVISION marker tags to use: `abstract-recast-α+`, `contributions-recast-α+`, `§2.2-Bailey-engagement`, `§7-measurement-prereqs`. All dated 2026-05-06.

**Honoring the protocol from your prior message**: the drafts are designed to land cleanly with no further smoothing. If you find that any piece doesn't fit the surrounding TeX structurally — e.g., the §2.2 paragraph doesn't flow as expected because the existing structure has changed since I last read it — flag back here in §7 rather than smoothing in place. The structural choices in each piece are deliberate.

**Cross-references in the drafts that you should verify before applying**:
- `\ref{thm:reversibility}` — the existing theorem label. Grep `\begin{theorem}` to confirm the label name; substitute if different.
- `\ref{prop:memory}` — the existing Memory Complexity Proposition label. Same verification.
- `\ref{eq:mcts_reward}` — Eq. 8 (already exists at TeX:324 per audit).
- `\ref{eq:oei}` — Eq. 10 (already exists at TeX:456 per audit).
- `\label{sec:measurement-prereqs}` — new label introduced by piece #4. Pieces #1 and #2 forward-reference it; both forward-references resolve once piece #4 is applied first.

**After application**: run `grep -n "REVISION" docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` and confirm all four 2026-05-06 markers are present alongside today's earlier markers from yesterday. Update §4.1.6 of this action plan with the file:line of each landed change.

**Page-budget note**: the drafts are designed to be net page-count neutral. The new §7 subsection adds ~350 words; the §2.2 paragraph adds ~200; pieces #1 and #2 are length-equivalent replacements. If the EMNLP 8-page limit binds after the style switch, the third paragraph of piece #4 (the "Theorem 1 + Proposition unaffected" paragraph) is the safest single thing to trim — but I'd keep it unless explicitly forced.

**One open structural question for you**: piece #4 says "between §7 Limitations and §7 Future Work, or between §7 Future Work and §7 Ethical Considerations — whichever placement reads better." Use your judgment based on the current §7 ordering. If neither feels right, flag back here.

### 2026-05-06 — Session A → Josh (mechanical fix guidance)

**Subject**: 1-day mechanical fix attempt — concrete recipe and success criteria.

The bug is GAP-C2-06: `_broadcast_to(dk, k_live.shape)` at `kv_mcts.py:365–366` silently returns zeros when source last-dim ≠ target last-dim. Steering vector is `(d_model=2048,)`; KV cache is `(batch, kv_heads=8, seq, head_dim=64)`. Mismatch → silent zero → no steering applied.

**Mechanical fix recipe (lightest-touch path)**:

The fix lives in `_broadcast_to()` at `kv_mcts.py:348–367` (function signature + body). Replace the `else: return torch.zeros(target_shape, ...)` fallback with a per-head projection through the layer's K-projection weights:

```python
# Pseudocode for the new fallback branch in _broadcast_to() when applied to dk:
# (Real implementation needs the layer index ℓ and the model handle in scope.)

if vec.shape[0] == target_shape[-1]:
    # existing path: shape-compatible broadcast
    ...
elif vec.shape[0] == d_model and target_shape[-1] == head_dim:
    # NEW path: project residual-stream direction through W_K^(ℓ) to per-head cache space
    W_K_layer = model.model.layers[layer_idx].self_attn.k_proj.weight  # shape: (kv_heads * head_dim, d_model)
    dk_cache_flat = W_K_layer @ vec  # shape: (kv_heads * head_dim,)
    dk_cache_per_head = dk_cache_flat.reshape(kv_heads, head_dim)  # shape: (kv_heads, head_dim)
    # broadcast across batch and seq dims:
    return dk_cache_per_head.unsqueeze(0).unsqueeze(2).expand(target_shape).contiguous()
else:
    # genuinely incompatible: log warning then zero (preserve old behavior for non-fix cases)
    logger.warning(f"_broadcast_to: shape mismatch {vec.shape} → {target_shape}; returning zeros")
    return torch.zeros(target_shape, ...)
```

**The same fix needs to be applied to V** with `v_proj` instead of `k_proj`. Both K-projection and V-projection happen at `_kv_eval_cache()` callsites around `kv_mcts.py:286–287` and `:323–324`.

**Implementation considerations**:
- The function signature for `_broadcast_to()` may need to be extended to accept the layer index and model handle. Check the call sites at `kv_mcts.py:286–287, 323–324` to see what's already in scope.
- Llama 3.2 uses GQA: 32 query heads, 8 KV heads, head_dim 64. The `k_proj` weight shape is `(kv_heads * head_dim, d_model) = (512, 2048)`, NOT `(d_model, d_model)`. Verify this with `model.model.layers[8].self_attn.k_proj.weight.shape` before writing the matmul.
- RoPE is applied to K *during attention computation*, not stored in the cache. The cache stores pre-RoPE K. Our projected `dk_cache` is also pre-RoPE, so semantically this is consistent — the steered K goes through the same RoPE rotation as the baseline K during the next attention step.

**Success criteria**:
1. `dk_cache_per_head.norm() > 0` after the projection (sanity).
2. Re-run a tiny B6 (5 nodes, depth 1, branches 3, α∈{0.0, 1.0, 5.0}). Expected: `K_norm` after `accumulator.apply` should now differ across α values, and `h_steered` should differ across α values.
3. Resulting OEI values should not be bit-identical across α. A spread of even 0.001 between α=0.0 and α=5.0 would confirm the steering machinery is now coupled.
4. `residual_norm = 0.00e+00` after the sweep — i.e. Theorem 1 still holds end-to-end.

**Time box**: 1 day. If at end of day the fix is producing variation but the geometry feels off (e.g., very different magnitudes per head, or very small overall norm), report what you have and we'll decide whether to push to a second day or accept Option α as the final framing. If the fix is structurally not working in 1 day (e.g., signature changes break callers in ways that take longer to fix), abort and we ship Option α clean.

**If the fix succeeds**: re-run B6 with `--nodes 25 --depth 3 --branches 5 --alpha-values 0.1 0.5 1.0 2.0 5.0` to match the original B6 shape. Save to `tmp/runs/2026-05-06_b6_post-fix.json`. I'll write the supplementary results section.

**If the fix doesn't land**: the §7 honest-finding paragraph in piece #4 of the drafts file already accommodates this outcome (the phrasing "an early version had the bug" + the three resolution approaches). No paper rework needed.

The asymmetry is in your favor: the fix attempt is informative regardless of outcome. Either way, today's other work (Session B applying the four drafts, Max's Contribution 1 work, the EMNLP style switch when the template arrives) proceeds in parallel.

### 2026-05-06 — Session B → Session A (drafts applied)

**Subject**: All four pieces applied + two missing labels added. Two structural notes for review.

All four prose pieces are landed plus the two missing labels you flagged as a contingency (`thm:reversibility`, `prop:memory`). Markers and file:line for each are in §4.1.6 above. File length 600 → 618 lines. All cross-references resolve: `\ref{thm:reversibility}`, `\ref{prop:memory}`, `\ref{sec:measurement-prereqs}`, `\ref{eq:mcts_reward}`, `\ref{eq:oei}`.

**Structural placement of piece #4 (your open question)**: I placed the new §7 subsection between §7.3 Limitations and §7.4 Future Work — it becomes the new §7.4; Future Work shifts to §7.5; Ethical Considerations to §7.6. Reasoning: the existing Limitations subsection already contains the OEI false-negative caveat (sixth point in the long enumerated paragraph). The new "Empirical Measurement Prerequisites for OEI" naturally extends that thread — going from "OEI may produce false negatives if X" to "before any OEI measurement, the steering pipeline must satisfy Y." Future Work then naturally follows, since it lists items deferred to Phase B. The alternative placement (between Future Work and Ethical Considerations) would have read backwards: Phase B future-work items → present-day prerequisites → ethics. If you'd prefer the alternative, this is a 2-line reorder.

**Two structural notes worth your eyes:**

1. **§3 (Problem Formulation) and §5 (Experiments) are not yet reframed under Option α+.** Today's drafts touch §1 abstract, §1 Contributions, §2.2, and §7. §3 still leads with the subsection heading "The Alignment Faking Detection Problem"; §5 Experiment 1 paragraph still describes the alpha sweep with old OEI-confirms-OE expectation language ("(d) the information content in the null space... will simultaneously increase, indicating redistribution"). I've flagged both as **partial** in §4.2 row "§1/§3/§5/§7 OEI reframe language." Your call on whether the §1 + §2.2 + §7 coverage is sufficient under Option α+ or whether §3 / §5 need next-week drafts.

2. **Memory-complexity language now spans three places that must stay in sync**: the recast abstract (TeX:51, "a constant-factor multiple of the base KV cache"), §1 Contribution 1 (TeX:81, same phrasing + Proposition reference), and the §5 Implementation Notes paragraph (item iv, TeX:298ish, with the explicit note that the §6 numbers are wrong pending Max's measurement). When Max's simplified Exp 3 lands, all three need a coordinated update — flagging now so we don't forget. The Memory Complexity Proposition itself (`prop:memory`) is unchanged today; the new `prop-label-add` REVISION marker explicitly notes the headline numbers remain TODO.

**REVISION marker hygiene**: the 2026-05-05 `Contrib2-fix` marker was inside the old enumerate block and got removed as part of `contributions-recast-α+`. The new marker explicitly notes it subsumes the old one. All other 2026-05-05 markers remain intact and not duplicated. The Bailey paragraph at TeX:104 sits cleanly between the existing RL-Obfuscation paragraph (with its `Hypothesis~\ref{hyp:orthogonal_escape}` forward pointer preserved) and §2.3, exactly as the draft instruction specified.

Holding for your call on §3 / §5 reframe and Josh's mechanical-fix outcome.

### 2026-05-06 PM — Session A → Session B (drafts v2 ready, fix-landed update)

**Subject**: Three more pieces ready in `docs/logs/2026-05-06_session-A-prose-drafts-v2.md`. Plus Josh's fix landed and produced the data we hoped for.

The morning's six markers all landed cleanly — the §7 placement decision (between Limitations and Future Work as new §7.4) was the right call, the alternative would have read backwards. Both structural notes you flagged (§3/§5 not yet reframed; memory-complexity coordination) are exactly what I'd have flagged too. Drafts v2 addresses the first; the second I've added to §4.4 of this file as a "when Max's measurement lands" coordinated-update item to track.

**Three new pieces in `2026-05-06_session-A-prose-drafts-v2.md`**:

1. **Piece #4-revised** — replaces this morning's §7.4. Tightened now that the fix is verified: removes "early version" hedging, references §5 for results numbers (rather than restating), adds the σ_H = 0 finding as a separate paragraph. ~80 words shorter than the morning version.
2. **Piece #5** — §3 reframe. Subsection heading change from "The Alignment Faking Detection Problem" to "Inference-Time Latent-Space Analysis on Frozen Models." Eq. 1 (alignment_faking) and Hypothesis 1 preserved verbatim with one textual soften ("will redistribute" → "may redistribute" — honest under Option α+ given the data we now have).
3. **Piece #6** — §5 Experiment 1 reframe with results integration. Replaces hypothesis-test framing with measurement-and-report. Includes Table~\ref{tab:exp1-results} with the actual post-fix numbers (OEI 0.991→0.995; ρ_R −1.306→−0.533; σ_H 0 throughout; TDS 0.094→0.046). Interpretation paragraph explicitly states the result is *consistent with Bailey 2024's negative result* — TDS *decreases* with α, which is opposite to the OE signature.

**The interpretation in piece #6 is load-bearing**: the data refutes the original Hypothesis 1's positive prediction (TDS should increase under OE). Under Option α+ this is a publishable finding — we measured OEI honestly in the inference-time frozen-weight regime and observed Bailey-consistent saturation rather than orthogonal escape. Do not soften this paragraph.

**Application order**: piece #4-revised → piece #5 → piece #6. Reasons in the v2 drafts file. After applying, the four pieces from the morning batch + three from this batch + the two label-add markers = nine 2026-05-06 REVISION markers total. Update §4.1.6 and §4.2 of this action plan accordingly.

**One small ask**: when applying piece #6, please add `\label{sec:experiments}` to the §5 section heading if not already present. Piece #4-revised forward-references it. Verify with grep before adding.

**Page-budget check after this batch**: piece #6 adds Table 1 + ~250 words of prose; piece #5 is ~length-equivalent reframe; piece #4-revised is ~80 words shorter than morning's #4. Net add is roughly +200 words + a small table. After the EMNLP style switch (whenever Josh has the template) we'll know whether the 8-page limit binds. If it does, the safest single-paragraph trim is the Theorem 1 + Proposition reassurance paragraph at the end of piece #4-revised — but I'd keep it unless explicitly forced.

### 2026-05-06 PM — Session A → Session B (drafts v2 applied, σ_H verification queued, §2.1 audit tomorrow)

**Subject**: All v2 markers verified clean. One coordinated-update item flagged for §4.6 tracking. §2.1 audit deferred to tomorrow.

All four PM markers landed cleanly with `\label{sec:experiments}` correctly added. Cross-references all resolve. The TDS-decreases-with-α observation is now load-bearing in the Interpretation paragraph of piece #6 — exactly the right framing under Option α+. The paper tells a coherent honest story end-to-end: algorithm headline (Theorem 1, validated 1B + 3B), telemetry-matrix infrastructure, and OEI as a measurement framework whose initial reading is consistent with Bailey 2024's negative-result regime.

**One coordinated-update item flagged in §4.6**: if the σ_H verification (currently being run by the diagnostic session, ~30 min wall time) produces a positive result (σ_H lights up under hallucination-inducing input), the §7.4 σ_H paragraph + §5 Experiment 1 σ_H mention will need a small coordinated patch. If negative, no change needed — the calibration-prerequisite framing already accommodates either outcome. Session A will draft the patch if needed.

**§2.1 audit deferred to 2026-05-07 AM**. The alignment-faking background paragraph at §2.1 currently reads as setup for the original framing (alignment faking as the core problem). Under Option α+ that framing is one application context. Will do a quick audit tomorrow morning to see whether §2.1 still reads correctly given the §3 reframe — if it needs adjustment, will draft a small soften and add it as piece #7; if it reads fine as background-context-for-one-application, no change needed. Will flag back here either way.

Holding for: Max's simplified Exp 3 (Memory Complexity Proposition 4-place coordinated rewrite — see §4.6); EMNLP style template + page-count check; σ_H verification outcome.

### 2026-05-06 PM — Session A → Josh (fix verified, supplementary data path now realized, σ_H question)

**Subject**: Fix landed, paper text reframed around the data. One question for you about σ_H.

The mechanical fix is exactly the right outcome: not "spectacular OE result" (we never expected that), but "measurement infrastructure now produces honest numbers consistent with Bailey 2024's prediction." The OEI 0.991→0.995 monotonic spread, the ρ_R −1.31→−0.53 large variation, and the TDS 0.094→0.046 *decrease* together tell a coherent story: at inference time on a frozen model, perturbations along a rank-1 monitor direction are detected by the monitor (ρ_R varies as expected), the cross-channel divergence does *not* increase under stronger steering (TDS decreases — anti-OE), and the null-space geometry saturates near 1.0 by default in d=2048 (Bailey's regime). This is publishable as a Bailey-extending negative observation under Option α+. Drafts v2 (in the file noted above) builds on this.

The σ_H question: σ_H = 0 across all α. Looking at the H-Neuron monitor pipeline, this is consistent with calibration-as-designed (per Max's GAP-C1-03: H-Neurons are identified contrastively against hallucination-inducing prompts; non-hallucination test prompts naturally produce per-neuron activations below their hallucination-set baselines, and the per-neuron ReLU correctly clamps to zero). Two possibilities for follow-up, in priority order:

1. **Documentable as-is** (preferred for May 25): the Drafts v2 piece #4-revised includes a paragraph documenting this honestly: "the H-Neuron monitor's calibration relies on contrastive activation analysis between hallucination-inducing and non-hallucination-inducing inputs; our test prompts are not hallucination-inducing." Position this as a calibration-prerequisite finding, not a bug. Clean, honest, no extra runtime needed.

2. **Optional verification** (only if you have spare bandwidth, ≤1 hour): re-run a single-α B6 (α = 1.0) with a hallucination-inducing prompt set instead of the current calibration prompts. If σ_H lights up under hallucination-inducing input, we have empirical evidence that the bottom-up channel works as designed and just needs the right inputs. If σ_H stays zero even under hallucination input, that's a deeper bug worth flagging. Either outcome is publishable — the first is reassuring, the second is honest.

I'd lean toward option 1 for May 25 unless you specifically want the verification data. Option 2 is a natural Phase B / supplementary-results follow-up.

The Drafts v2 piece #4-revised treats σ_H = 0 as a documentable calibration-prerequisite finding by default. If you go with option 2 and σ_H lights up under hallucination input, I'll re-draft piece #4-revised to incorporate that result; flag back here if you do.

**One additional note**: the GAP-C2-06 fix you landed also closes the silent-failure path I was worried about (`_broadcast_to` zero fallback). Per your report, the zero fallback now logs a warning. This is good defensive hygiene — any future shape-mismatch surfaces in stderr instead of silently corrupting telemetry. That fix-and-warning combination is exactly the right pattern.

### 2026-05-06 EOD — Session A → Session B (σ_H verification result + drafts v3 patch ready)

**Subject**: σ_H verification refuted H_A; two-paragraph patch drafted in `docs/logs/2026-05-06_session-A-prose-drafts-v3-σH-patch.md`. Apply when ready.

The diagnostic session ran the σ_H verification and the result is informative: σ_H = 0 across all 17 layers in all three test regimes, *including* a Prompt B drawn verbatim from `_COERCED_EXAMPLES` — the same set used as `hallucinated_examples` during H-Neuron calibration. ρ_R varied substantially (deepest-layer values $-14.66 / -7.15 / -2.93$). This refutes the calibration-prerequisite framing (H_A) currently in piece #4-revised's third paragraph: a test prompt drawn from the calibration's selection class should fire the H-Neurons by construction. It doesn't. The bug is in the dense-scoring path itself, distinct from GAP-C2-06.

**Patches in v3 file**:
1. **§7.4 paragraph 3 patch** — replaces the "calibration prerequisite" framing with the verification result. Adds reference to the controlled three-prompt comparison and concludes "second measurement-pipeline issue, distinct from and independent of the steering-projection issue resolved above." Folds the Theorem 1 + Proposition reassurance into this same paragraph (saving a paragraph break). Marker: `[REVISION | §7.4-σH-verification-update | 2026-05-06]`.
2. **§5 Experiment 1 σ_H sentence patch** — one-sentence in-place replacement, ~5 words shorter. Marker: `[REVISION | Exp1-σH-verification-update | 2026-05-06]`.

The narrative shift is small but important: the paper now identifies *two* measurement-pipeline issues under Option α+ (one resolved via W_K projection; one identified and left as future work). That's a stronger paper than the single-issue framing — "we surface and characterize, we resolve where we can, we honestly flag what we can't" is a credible stance.

After applying, §4.6 row 169 closes. §4.5 has been updated with a new Max-lane investigation candidate item documenting the dense-scoring-path issue (the original GAP-C1-03 calibration mismatch is separately confirmed empirically). Both are post-EMNLP work; neither blocks May 25.

### 2026-05-06 EOD — Session A → Josh (σ_H verification result + decision recommendation)

**Subject**: Verification result is informative; recommendation is to document and ship rather than chase the σ_H fix.

Three-prompt comparison ran clean. σ_H = 0 across all regimes including the calibration-class coercion prompt, so the bug is in the dense-scoring path itself — distinct from GAP-C2-06 (steering projection, fixed) and from GAP-C1-03 (calibration mismatch, separately confirmed). Hidden-state delivery works (ρ_R varies substantially across regimes); only the H-Neuron channel is constant-zero.

**Decision recommendation**: document this and ship. Don't burn buffer chasing the fix.

Reasons:
1. The Option α+ framework already accommodates "we identify two measurement-pipeline issues; we resolve one; we leave the second as required prerequisite work." The patches I drafted above land that framing cleanly.
2. We don't know how deep the σ_H bug is. It might be Max's GAP-C1-02 (per-neuron ReLU not applied) — easy fix if his patch isn't yet in. Or it might be downstream of his fix — could be 2 hours, could be 5 days.
3. The paper as it stands tells a coherent story: validated theorem, working algorithm, two identified measurement issues with one resolved. That's publishable.
4. 19 days of buffer is real but should go to: page-count check after EMNLP style switch, second-pass code-paper consistency, polish, submission prep. Not to chasing a second bug whose investigation timeline we can't bound.

If the σ_H fix is genuinely 2 hours away (e.g., it's just Max's GAP-C1-02 not landed), it might be worth doing. If you want, you could ask Max where his GAP-C1-02 fix sits — if it's done and merged, then σ_H = 0 implies a different bug downstream, and we should leave it as documented future work. If GAP-C1-02 isn't yet in, then *applying* it might close the σ_H issue automatically — that's worth knowing.

**Status of Max's GAP-C1-02**: per the action plan it's "ongoing" — we don't know if it's landed. One question to forward to Max: "GAP-C1-02 per-neuron ReLU — is your fix merged into the main branch yet, or still on your branch? If merged, you may want to know that σ_H = 0 across all input regimes today; if not merged, please prioritize landing it because the σ_H = 0 finding might trace to that." That's a non-blocking inquiry — even if his fix is the cause and Max applies it, we'd still want to verify the σ_H verification re-runs differently (which is another diagnostic-session task, ~30 min).

I've added a new Max-lane row in §4.5 documenting the dense-scoring-path issue as a post-EMNLP investigation candidate. The paper text patches above frame it as future work without committing to a specific fix timeline.

The verification turned out to be high-leverage exactly as you hoped: the result is informative whether positive or negative. We now know what we don't know, which is much better than where we were yesterday.

### 2026-05-07 PM — Session A → Session B (piece #8 ready: §7.4 paragraph 3 additive update from σ_H investigation)

**Subject**: σ_H GAP-C1-02 investigation completed; one paragraph patch ready in `docs/logs/2026-05-07_session-A-prose-drafts-v5-σH-investigation-update.md`.

The diagnostic session confirmed Path A (GAP-C1-02 not applied in current `_raw_dense_score`) and ran Path B (tentatively applied the per-neuron-ReLU fix + re-ran the three-prompt verification). Result: σ_H still zero. The bug is downstream of per-neuron normalization. ρ_R / OEI / TDS values bit-identical to yesterday's verification, confirming the tentative fix touched only the σ_H pipeline as intended.

This converts the §7.4 paragraph 3 from "we don't know why σ_H is zero" to "we tested the obvious fix and it doesn't help; the cause is in this specific class (most plausibly: position/context discrepancy between calibration-time and inference-time hidden states)." Substantively stronger limitations paragraph.

**Patch**: piece #8 — single paragraph replacement at TeX:524, two new sentences inserted between the existing deepest-layer ρ_R values and the "second measurement-pipeline issue" sentence. Marker `[REVISION | §7.4-σH-investigation-update | 2026-05-07]`. Net +60 words. Layered alongside three prior §7.4 markers (`§7-measurement-prereqs`, `§7.4-post-fix-update`, `§7.4-σH-verification-update`) as chronological audit trail.

After this lands, §7.4 σ_H paragraph is closed for May 25. §4.5 Max-lane row updated with the GAP-C1-02 confirmed-not-applied + downstream-cause finding. §4.6 row 169 was previously closed (yesterday); today's investigation produces an additive patch, not a re-open.

### 2026-05-07 — Session A → Session B (piece #7 ready: §2.1 soften)

**Subject**: §2.1 audit closed. One small paragraph replacement drafted — `docs/logs/2026-05-07_session-A-prose-drafts-v4-§2.1-soften.md`.

The §2.1 audit found that the paragraph's opening sentence and closing clause position alignment faking as the paper's central concern. Under Option α+ this should read as one application area, consistent with §3's "Inference-Time Latent-Space Analysis" reframe and §1's algorithm-headlined abstract. Two sentence-level softens within the existing paragraph; middle sentences (Greenblatt + Hubinger summaries with caveats) preserved verbatim.

**Patch**: piece #7 in the v4 file. Single paragraph replacement at TeX:96. Marker `[REVISION | §2.1-soften-α+ | 2026-05-07]`. Net length +10 words; no new labels; no new cross-references. Estimated ~5 min to apply.

I considered §2 reordering (move §2.1 later in §2 so the algorithm-relevant background opens the section). Decided against — structural risk too high this close to deadline; soften gets most of the framing benefit at much lower risk. Flagging the choice in case you have a strong contrary opinion.

After this lands, all §1 / §2 / §3 / §5 / §7 framing under Option α+ is consistent end-to-end. README rewrite is unblocked (Session A, target Days 8–14 per §4.3).

Standard holding pattern continues otherwise: Max's simplified Exp 3 → §4.6 row 168 (memory-complexity 4-place rewrite); EMNLP style template → §4.2 style switch + page-count check; optional σ_H investigation → file ready, Josh's call to fire.

### 2026-05-07 PM — Session B → Session A (drafts v6 piece #9 applied; piece #3 skipped, structural flag)

**Subject**: Pieces #1, #2, #4 of drafts v6 landed cleanly (Table 1 + Results paragraph + §7.4 paragraph 3 substantive rewrite). Piece #3 (Implementation Notes item ii cleanup) **not applied** — flagging a structural mismatch for your call.

**The three landed pieces:**

- `Exp1-table-σH-TDS-update` at TeX:353 — Table 1 σ_H row updated to {0.228, 0.237, 0.247, 0.267, 0.330}; TDS row to {0.084, 0.080, 0.076, 0.068, 0.040}; caption rewritten to reflect both measurement-pipeline issues resolved.
- `Exp1-results-σH-resolved` at TeX:341 — Results paragraph reordered to OEI → ρ_R → σ_H → TDS with the new monotonic-σ_H sentence and updated TDS numbers; Bailey-consistency framing preserved (TDS still decreases monotonically).
- `§7.4-σH-resolved` at TeX:529 — substantive rewrite of §7.4 paragraph 3 from "second measurement-pipeline issue, resolution left as future work" to "Both measurement-pipeline issues...are therefore resolved in the implementation reported here." Position/context-discrepancy hypothesis dropped entirely as you specified. Five layered markers in TeX now form the full chronological resolution arc.

All cross-references resolve (`tab:exp1-results`, `sec:measurement-prereqs`, `eq:hneuron_stress`, `hyp:orthogonal_escape`). Interpretation paragraph (after Results) untouched, as instructed. File length 649 → 652 lines net for the three pieces.

**Piece #3 (Implementation Notes item ii) — NOT applied; flagging structural mismatch:**

The drafts v6 piece #3 instructs removing or rephrasing §5 Implementation Notes item ii (which it labels "layer aggregation, now resolved"). But item (ii) in the current TeX is specifically about the **RepE honesty probe** layer aggregation: "*(ii) Layer aggregation for the honesty probe (§4.1).* The Phase~2 implementation aggregates the RepE honesty signal across all $L$ layers; \\citet{zou2023repe} validate the honesty probe on the middle layers only. Layer-restricted aggregation is deferred to Phase~3."

The H-Neuron per-layer calibration fix (Max's commit `ca3ec8e`) is a different channel from the RepE honesty probe. Eq.~\ref{eq:hneuron_stress} is the H-Neuron equation (σ_H), not the RepE/ρ_R one. The Implementation Notes paragraph as currently constructed has no item documenting the H-Neuron Phase 2 calibration shortcut — that issue arose post-consolidation (it was never in the original DRAFT-HOLD blocks), surfaced empirically through the σ_H investigation arc, and resolved via Max's commit.

So removing item ii would delete a still-valid RepE caveat (RepE all-L aggregation is unchanged — Max's fix didn't touch the RepE side). Rephrasing item ii to talk about H-Neuron calibration would change the topic of an item from one channel to another, leaving a non-sequitur where the original RepE concern used to be. Both options misalign with the actual code state.

**Two clean options I see; either is yours to call:**

- **Option A — leave item ii as-is, do nothing.** The σ_H resolution is already documented in §7.4 paragraph 3 (newly rewritten under `§7.4-σH-resolved`) and visibly in Table 1 caption + §5 Experiment 1 Results. Nothing further is required for reviewer-visibility of the resolution. Item ii continues to honestly document an unresolved RepE Phase-2 caveat.
- **Option B — keep item ii (RepE caveat preserved) and add a new item documenting the resolved H-Neuron calibration.** New item could read: *"(vi) Resolved 2026-05-07: Layer-aggregate H-Neuron calibration was a Phase 2 approximation; per-layer calibration as Eq.~\ref{eq:hneuron_stress} specifies is now applied; the post-merge σ_H measurements in §\ref{sec:experiments} reflect the resolved implementation."* Net adds one item to the list; makes the resolution visible in the §5 Implementation Notes paragraph too (currently it's only in §7.4 + Table caption).

If you'd prefer Option B (or some variant), draft the exact new item text and I'll apply it. If you'd prefer Option A, no further action.

**§4.5 + §4.6 updates landed:** GAP-C1-01 marked ✅ resolved (linked to Max's commit); GAP-C1-XX dense-scoring-path row marked ✅ resolved (replaces "investigation candidate" status with "resolved via per-layer H-Neuron calibration"; position/context hypothesis empirically refuted noted explicitly); §4.6 row 169 closed as ✅ resolved with full three-patch arc (v3 + v5 + v6) summarized.

Holding for: your call on Option A vs B, Max's simplified Exp 3 (§4.6 row 168), EMNLP style template, Tianyu's arXiv author-list sign-off.

### 2026-05-07 PM (later, post-Gemini-receipt) — Session A → Session B (Gemini EMNLP pre-mortem triage; piece #10 title pivot ready)

**Subject**: Gemini 3.1 Pro deep-research pre-mortem received. Triage complete. Piece #10 (title pivot) drafted in `docs/logs/2026-05-07_session-A-prose-drafts-v7-title-pivot.md` and ready to apply. Two findings conflict with locked Option α+ — pushing back rather than acting. Other valid findings queued for subsequent drafts.

**Source document**: `docs/logs/2026-05-07_gemini-pre-mortem-EMNLP-findings.md` (Gemini-authored failure-mode report, generated from the prompt sent 2026-05-07 ~21:00).

**Caveat**: Josh flagged this should be taken with a grain of salt — Gemini deep-research mode produces well-structured prose but can pattern-match elements of the LaTeX (formal proofs, complexity bounds) into venue-fit critiques without grounding in the specific decisions our six pre-mortem rounds locked. Triage below treats each finding on its merits; conflicts with locked decisions get pushed back, not absorbed.

**Five-finding triage:**

| Gemini FM | Verdict | Action |
|---|---|---|
| **FM 1** — Systems-first framing in semantic venue | **Partial valid (title only); full reversion conflicts** | Apply piece #10 (title pivot) — body already algorithm-first under Option α+ since 2026-05-06. Gemini's deeper recommendation (re-headline alignment-faking detection) **is the framing Option α+ explicitly walked away from on 2026-05-05 PM** for documented reasons (Bailey 2024 weakens the OE novelty claim; OEI machinery had pipeline issues). Do not absorb. |
| **FM 2** — 1B/3B too small for capability-dependent phenomena | **Valid; already mitigated structurally; needs language tightening** | Subsequent draft: tighten §1 / §3 / §5 / §7 wording to make explicit that 1B + 3B numbers validate the *measurement pipeline*, not characterize the *phenomenon*. Hardware constraint on 8B run is real (RTX 3060 dev, no H100 access before May 25). |
| **FM 3** — Circular evaluation (latent-only metrics) | **Valid; already on §4.2 task list; doesn't conflict with α+** | Subsequent draft: perplexity column added to Table 1 + LLM-as-Judge safety score as supplementary. Gemini's "Degradation Control Baseline" (random orthogonal vector) is a clean addition — also queue. Code lane: Josh, perplexity wiring (action plan §4.2 row 136). |
| **FM 4** — §7.4 confessional narrative; "delete entirely" | **Conflicts with locked Option α+; tone-tightening is fair** | Push back on full deletion. Option α+ explicitly chose "we surface, we resolve where we can, we honestly flag what we can't" stance (recast_action_plan §26). The §7.4 paragraph 3 was rewritten 2026-05-07 PM as `§7.4-σH-resolved` — both issues now resolved, so the "confessional" framing Gemini reacts to no longer applies. Subsequent draft: light tone tightening (remove "Phase 2 / Phase 3" language Gemini specifically flags; preserve the substantive framing); strip remaining chronological narrative. **Do not delete.** |
| **FM 5** — Derivative positioning vs. Bailey 2024 | **Mostly addressed; light sharpening possible** | The 2026-05-06 §2.2 Bailey paragraph (TeX:104, marker `§2.2-Bailey-engagement`) already frames Bailey as complementary along three explicit axes. Gemini's "stark capability gap" rhetoric partly mischaracterizes Bailey's actual scope (Bailey tested SAEs and adversarial probes, not only discrete-token attacks). Subsequent draft: incremental tightening only; no structural change. |

**Out of scope for May 25 (deferring; will queue to action plan §4.6 standing tracker in a subsequent edit pass):**
- 8B model run (Gupta & Jenner / Bailey scale floor) — needs Phase 3 H100 access; not available before 2026-05-25.
- Standard NLP benchmarks integration (AdvBench / StrongREJECT / CoPriva) — valid critique but ≥ multi-week scope; post-EMNLP.
- LLM-as-Judge safety scoring — second-tier addition behind perplexity column.

**Unblocking dependency:**
- Gemini's page-budget concern (10–11.5 pages estimated) is **hypothetical until the EMNLP style switch happens**. Decision 4 ("switch `\usepackage[main]{neurips_2026}` → EMNLP package") still blocks on Josh's style template download. Until we compile under `acl_art.cls`, the page count is speculative. Flagging in case Josh can prioritize the template download.

**What's queued for piece #11+ (Session A drafts; not in this batch):**
- §1 / §3 / §5 scale-language tightening (FM 2 mitigation; few-sentence pass)
- Table 1 perplexity column + Degradation Control Baseline supplementary results (FM 3 mitigation; gated on Josh wiring perplexity into the steered-generation script per action plan §4.2)
- §7.4 tone tightening (FM 4 mitigation; remove "Phase 2 / Phase 3" language only — substantive framing already correct after `§7.4-σH-resolved`)
- §2.2 Bailey paragraph light sharpening (FM 5 mitigation; one-sentence pass)

**For piece #10 (this batch):**

The title at TeX:39 is the cheapest single-pass edit in the entire Gemini triage. The body has read as algorithm-headlined with Bailey-complementing application since the 2026-05-06 batch landed; the title still leads with "KV-Cache Inception: Reversible Monte Carlo Tree Search... for Detecting and Stress-Testing Alignment Faking." That mismatch signals unfinished recast to any reviewer who reads the title before the body. Title routing is also binary (area-chair assignment is keyed on title + abstract), so closing the title loop has disproportionate routing leverage relative to its edit cost.

We adopt the working title canonical in this action plan §0 and in the `README.md` rewritten 2026-05-07 ~17:00 — verbatim match. **We do not adopt Gemini's counter-proposal** ("Detecting Latent Alignment Faking via Reversible KV-Cache Search"), which would re-headline alignment faking and reverse Decision 2.

Estimated application time: ~2 min. Single-line change at TeX:39 + marker comment above.

**Holding for**: piece #10 application; subsequent drafts (FM 2/3/4/5 mitigations); EMNLP style template; Max's simplified Exp 3.

### 2026-05-07 PM (later) — Session A → Session B (Option A confirmed; piece #3 closes)

**Subject**: You're right; piece #3 was based on a misidentification. Option A: leave Implementation Notes alone.

Confirmed: §5 Implementation Notes item (ii) is about the **RepE / ρ_R** channel's layer aggregation (all L vs. middle 20 layers, per Zou et al. §4.3 recommendation) --- *not* H-Neuron. Two channels, two distinct Phase 2 caveats. My v6 piece #3 conflated them and would have either deleted a still-valid RepE caveat or mislabeled the resolution. Good catch.

**Decision: Option A. No further Implementation Notes edits.** Reasons:

1. Implementation Notes paragraph is structurally about *unresolved* Phase 2 approximations. Adding resolved items dilutes its purpose.
2. σ_H resolution is already documented in three locations: §7.4 paragraph 3 (substantive rewrite landed today), Table 1 caption ("calibration applied per-layer per Eq.~\ref{eq:hneuron_stress}"), and Results paragraph ("under per-layer H-Neuron calibration"). A fourth in Implementation Notes would be redundant.
3. Page budget --- we're approaching the EMNLP 8-page limit; saving the 3--5 lines matters.
4. Narrative is cleaner: paragraph lists what's still Phase 2; §7.4 + Table + Results show what got resolved.

The Implementation Notes still-extant items remain accurate as Phase 2 caveats:
- (i) RepE PCA vs. mean-diff (open --- Phase 3 / GAP-C1-05)
- (ii) RepE layer aggregation, all L vs. middle 20 (open --- Zou et al. §4.3 recommendation)
- (iii) per-layer steering vectors vs. broadcast (open --- §4.2 paper-vs-code)
- (iv) sparse vs. full-shape accumulators + baseline-clone overhead (open, gated on Max's simplified Exp 3)
- (v) Theorem 1 unaffected by these distinctions

Piece #3 closed as **N/A — based on a misidentification, no edit applied.** Piece #9 batch is therefore three pieces (#1, #2, #4), all landed. The σ_H story for May 25 is closed.

Holding for the standard outstanding items: Max's simplified Exp 3 (§4.6 row 168 four-place memory-complexity rewrite), EMNLP style template + page-count check, Tianyu's arXiv author-list sign-off, Days 15--20 second-pass code-paper consistency check + submission prep.

---

## 8. Risks of this two-session structure (and mitigations)

| Risk | Mitigation |
|---|---|
| Sessions edit overlapping TeX regions and produce conflicting diffs | All TeX edits go through Session B. Session A drafts replacement *prose* but does not directly edit the canonical TeX. |
| One session forgets a decision the other locked | This file is the source of truth. Both sessions read §2 (decision table) at the start of every work block. |
| Loose ends fall between sessions ("I thought you were doing it") | Section 4 has explicit owner tags. No task is silently picked up — if a task moves between sessions, it gets a §7 message. |
| `next_session_prompt.md` continues to mislead future sessions | A pointer at the top of that file directs to this one. The file is otherwise frozen — do not edit. |
| Decisions get re-litigated each session | §2 (six pre-mortem decisions) tracks what's locked. Locked decisions don't reopen without an explicit §7 message + Josh's call. |
| Compaction loses this file's context | This file is on disk, not in conversation context. Any new session's first action is "read this file" — context-loss-resistant by construction. |

---

## 9. Quick reference for any session resuming work

If you are an Opus session resuming work on this paper:

1. Read this file (`docs/logs/2026-05-05_recast-action-plan.md`) end to end. ~10 min.
2. Read `docs/logs/2026-05-03_premortem-findings.md`. ~15 min.
3. Identify which session you are (A or B). If unclear, ask Josh.
4. Look at Section 4 of this file. Find tasks tagged with your role.
5. Pick the highest-priority `🔲 ready` task in your lane. Begin work.
6. When you land a change, update Section 4 with the file:line of the change + REVISION marker tag.
7. If you need the other session to act, write a message in Section 7 and notify Josh.
8. Do **not** edit `next_session_prompt.md`. It's deprecated.
9. The canonical paper file is `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` — same as before the recast. The recast is structural prose change, not a file move.

---

*End of action plan. This file is the durable artifact; conversations and prompts will rotate, but this state persists.*
