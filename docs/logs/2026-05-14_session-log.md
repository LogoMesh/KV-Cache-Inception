# Session Log — 2026-05-14 (Day 5 of plan v2 execution)

**Session type:** Session A (Opus) + Session B (TeX application within same Claude session)
**Day of ARR-cycle execution plan:** Day 5 (calendar 2026-05-14 = plan v2 §2 Day 5; internal day-number and plan-calendar are aligned today — the 1-day-ahead offset that held through Day 4 has resolved now that Day 4's work spanned calendar 2026-05-12 + 2026-05-13)
**Submission deadline countdown:** T-11 calendar days to 2026-05-25 ARR submission (T-80 to EMNLP commitment 2026-08-02)

**Continuity:** Day-4 EOD state is recorded in [`2026-05-12_session-log.md`](2026-05-12_session-log.md) ("Day 4 extension" + "Day-4 P2" sections). TeX state at Day-4 rollover: **704 lines**, D+-aligned through §7 except §5 Exp 2 + §5.5 Metrics + §6 worked-example numbers (Day 7 + Day 8 scheduled). Day-5 work picks up from clean Day-4 baseline; no blockers.

---

## Day 5 entry — Day-9 residue pull-forward: `zou2023repe` `\citet` numbers-mode prose fix

**Date:** 2026-05-14 (Day 5 of plan v2)
**Owner:** Session A + Session B (same Claude session)
**Task:** Pull forward Day-9 residue item from plan v2 §2 — rewrite the two `\citet{zou2023repe}` usages at canonical TeX lines 292 + 294 (§5 Implementation Notes items i + ii) to follow the line-214 pattern `Zou et al.\ (2023)~\cite{zou2023repe}`, eliminating the numbers-mode natbib rendering artifact (`\citet{}` renders as bare `[N]` under the Day-4 `natbib-numbers-mode-D+` quick patch).
**Status:** ✅ COMPLETE — TeX edit applied, G-smoke-compile passed, commit landed.
**Decision basis:** Josh selected Option D from the Day-5 prompt's A–E menu after the Phase-1 state summary.

### Process

Per Phase-4 trip-wire in the Day-5 prompt ("If the rewrite at lines 292+294 conflicts with surrounding prose, e.g., already says 'Zou et al.\ (2023)' in adjacent text, flag and stop — re-verify before patching"):

1. Read canonical TeX lines 285–300 in full before editing. Confirmed lines 292 and 294 are bare `\citet{zou2023repe}` usages; no "Zou et al." or "Zou (2023)" appears in surrounding prose (items iii, iv, v use no Zou-citation). Trip-wire **cleared**.
2. Applied the two-line surgery as a single `Edit` operation, wrapping it in a new REVISION marker (`zou2023repe-citet-numbers-mode-fix-D+`, 2026-05-14) that documents the rationale and flags the Day-8 ACL-switch revisit condition (under `acl.sty` + `acl_natbib.bst` author-year mode, `\citet{}` renders correctly and the cosmetic rewrite becomes redundant-but-harmless).

### Edit 1 — canonical TeX line 292 (§5 Implementation Notes item i)

**Before:** `The text describes the Linear Artificial Tomography procedure of \citet{zou2023repe} using PCA on contrastive pairs.`
**After:** `The text describes the Linear Artificial Tomography procedure of Zou et al.\ (2023)~\cite{zou2023repe} using PCA on contrastive pairs.`

### Edit 2 — canonical TeX line 294 (§5 Implementation Notes item ii)

**Before:** `...the RepE honesty signal across all $L$ layers; \citet{zou2023repe} validate the honesty probe on the middle layers only.`
**After:** `...the RepE honesty signal across all $L$ layers; Zou et al.\ (2023)~\cite{zou2023repe} validate the honesty probe on the middle layers only.`

### REVISION marker

A single new marker covers both edits, inserted immediately before item (i):

```
% [REVISION | zou2023repe-citet-numbers-mode-fix-D+ | 2026-05-14 | Replaced two `\citet{zou2023repe}` textual citations at items (i) and (ii) with the prose-and-cite pattern "Zou et al.\ (2023)~\cite{zou2023repe}" used at line 214. Under the numbers-mode natbib quick-patch applied Day 4 (natbib-numbers-mode-D+ at line 8), `\citet{}` renders as a bare numeric `[N]` instead of "Zou et al.\ (2023)", producing awkward prose ("LAT procedure of [N]" and "[N] validate..."). The two-line surgery removes the rendering artifact while preserving the citation. REMOVE OR REVISIT at Day 8 ACL style switch: `acl.sty` + `acl_natbib.bst` reload natbib in author-year mode, at which point `\citet{}` renders correctly; the cosmetic `Zou et al.\ (2023)~\cite{}` rewrite is then redundant but not harmful (the form at line 214 is stable across both modes). — pending audit]
```

### G-smoke-compile gate

Per plan v2 §9, ran `pdflatex` twice on the canonical TeX.

| Pass | Exit | PDF | Pages | Bytes | Errors | Warnings |
|---|---|---|---|---|---|---|
| 1 | 0 | produced | 18 | 383,177 | 0 | 10 (7× sec:exp2, 1× eq:mcts_reward, 1× h→ht float, 1× "There were undefined references" meta) |
| 2 | 0 | produced | 18 | 383,177 | 0 | identical to pass 1 |

**Day-4-EOD-to-Day-5 warning delta:** Day-4 EOD had 12 warnings (8 undefined refs + 2× natbib `Author undefined for citation 'zou2023repe'` + 1× h→ht float + 1× "There were undefined references" meta). Day-5 EOD has 10 warnings — the 2× `Author undefined` natbib warnings are retired by this fix; everything else identical. PDF byte-count and page-count unchanged.

**Gate verdict:** PASS. Acceptable undefined refs (`sec:exp2`, `eq:mcts_reward`) only; no new undefined refs introduced; no `Error:` lines.

### TeX state

| Metric | Pre-edit (Day-4 EOD) | Post-edit (Day-5 EOD) | Delta |
|---|---|---|---|
| Total lines | 704 | **705** | +1 (single REVISION marker comment line; the two body edits are in-place rewrites with net 0 line delta) |
| §5 Implementation Notes items (i)+(ii) prose | `\citet{zou2023repe}` (renders as bare numeric `[N]`) | `Zou et al.\ (2023)~\cite{zou2023repe}` (renders consistently) | ✅ cosmetic-correct under numbers-mode natbib |
| Day-4-EOD `Author undefined for citation 'zou2023repe'` natbib warnings | 2× present | 0× present | ✅ retired |
| §5–§7 D+ alignment | as Day 4 EOD | unchanged | — |

### Git commit

`19f9eb5` — `docs: Day 5 — zou2023repe \citet numbers-mode prose fix at §5 items (i)+(ii)`. Scope: 1 file changed (canonical TeX), +3 / −2 lines (the +1 REVISION marker spans 1 comment line; the in-place edits to lines 292 + 294 net to +2 / −2 since each line gets a longer prose-and-cite form). No build artifacts, no log files, no submodule churn included.

### Plan v2 §2 reconciliation

Plan v2 §2 places this exact fix in the Day-9 sub-row residue list (2026-05-18): *"(i) `natbib Warning: Author undefined for citation 'zou2023repe'`... two-line surgery rewriting lines 292 and 294 to follow line 214's pattern."* Today's work pulls that residue item forward by **4 days** (Day 5 vs scheduled Day 9). The pull-forward pattern continues from prior days:
- Day 2: pulled forward Cartography Track G from Days 4–5.
- Day 3: pulled forward §5 Exp 2 prose draft from Day 6.
- Day 4 P2: pulled forward §A Reproducibility Appendix v1 from Day 9.
- Day 4 (Day-6 satellite): pulled forward Figure 1 generation + Track G audit pass from Day 6.
- **Day 5 (today): pulled forward Day-9 zou2023repe residue fix from Day 9.**

This is the sixth pull-forward in five elapsed days, consistent with the team's documented ~50% time-budget observation. Per plan v2 §9's plasticity rule, the day's edits compile cleanly so they promote to the next day's baseline.

### Day-9 residue list status

Plan v2 §2 Day-9 sub-row had two residue items filed Day-4: (i) the zou2023repe `\citet` fix — **CLOSED today**; (ii) the conditional check at Day-8 ACL switch that `acl_natbib.bst` re-resolves `\citet{}` correctly under author-year mode. Item (ii) is unchanged by today's work but the new REVISION marker at TeX line 291 documents the Day-8 revisit explicitly ("REMOVE OR REVISIT at Day 8 ACL style switch...").

### Memory updates made in-session

None. The plan-v2 §2 Day-9 sub-row is the source of truth for the residue list; no memory write needed for a residue closure.

### Day 5 wall time

| Activity | Wall |
|---|---|
| Phase 1 read (CLAUDE.md + MEMORY index + 5 protocol-rule entries + recast_action_plan + project_context + Day-4 session log + Track G audit pass + canonical TeX 704 lines + v10-exp2 draft + §A v1 draft + v10-exp1 draft + 14-day plan v2) + Phase-1 state summary | ~40 min |
| Phase-4 trip-wire check (read lines 285–300) + Edit application + REVISION marker authoring | ~5 min |
| G-smoke-compile (2 pdflatex passes + warning grep) | ~3 min |
| Commit + session log authoring | ~10 min |
| **Day 5 total** | **~58 min** |

---

*End of Day 5 (first session). Day-5 wall to this point: ~58 min (Phase 1 read-heavy + 1 small TeX edit + smoke-compile + commit + log). One commit landed (`19f9eb5`). Day-9 residue list trimmed from 2 items to 1 (Day-8 ACL conditional check remains). EOD plan for Day 6 superseded by the resume session below (the VSCode-extension stream error EOD prompted a same-day fresh-session resume to close out the §5 Exp 2 work that was queued for Day 7).*

---

## Day 5 resume — §5 Experiment 2 Cartography recast (post-stream-error close-out)

**Date:** 2026-05-14 (same calendar day; fresh Claude session after VSCode-extension `Unhandled case: [object Object]` client-side stream error EOD of the first Day-5 session)
**Owner:** Session A + Session B (same Claude session, fresh context; resumed via [`2026-05-14_day-5-resume-prompt.md`](2026-05-14_day-5-resume-prompt.md))
**Task:** Close out Day 5 by verifying the §5 Experiment 2 Cartography recast Edit applied to canonical TeX in the prior session (uncommitted on disk at session start), running G-smoke-compile, committing, and appending this session log. This is the 7th pull-forward of the sprint (plan v2 §2 had §5 Exp 2 scheduled Day 7; applied Day 5 = 2-day pull-forward, building on the Day-3 v10-exp2 draft + Day-6 Figure 1 pull-forward + Day-6 Track G audit pass).
**Status:** ✅ COMPLETE — Edit verified, G-smoke-compile passed, commit landed (`895ecca`), this entry appended.
**Decision basis:** Edit already applied; resume task is execute-the-close-out, no decision menu.

### Phase 1 verification (resume readback)

Per resume-prompt Phase 1, 9 artifacts read for orientation. Edit verified verbatim against the resume-prompt expectations:

| Element | Expected | Actual | Verdict |
|---|---|---|---|
| REVISION marker tag | `option-D+-exp2-cartography-2026-05-14` | line 360 verbatim | ✅ |
| Paragraph title | "Experiment 2: Latent Cartography MCTS across Prompt Classes" | line 361 verbatim | ✅ |
| Footnote text | "The upper bound is driven by HellaSwag continuation at the 3B scale (path-mean reward gain $+0.037$); the other seven cells span $+0.005$ to $+0.025$." | line 376 verbatim | ✅ |
| §5-Exp3-5-cut marker preserved after | yes | line 405 unchanged | ✅ |
| Figure file exists | `docs/NeurIPS/figures/exp2-mean-step-alpha.pdf` | 21 KB, generated 2026-05-13 | ✅ |
| `\label{sec:exp2}` defined | yes | line 361 | ✅ |
| `\label{tab:exp2-cartography-paths}` defined | yes | line 395 | ✅ |
| `\label{fig:exp2-mean-step-alpha}` defined | yes | line 402 | ✅ |
| TeX line count | 780-795 (resume-prompt estimate) | **745** | 🟡 below estimate; content structurally complete (paragraph-per-line LaTeX formatting; v10-exp2 draft itself estimated +57 lines body, actual +40 — both estimates were heuristic, not blocking) |
| Edit unstaged | yes | `+44 / -4` diff stat | ✅ |
| Prior `19f9eb5` in history | yes | confirmed | ✅ |

Flagged the 745-vs-780-795 line-count discrepancy to Josh in Phase-1 readback; Josh approved go-ahead with that knowledge. All structural elements verbatim-match; the discrepancy was a heuristic estimate, not missing content.

### Edit details (recap from prior session's work)

**Source draft:** [`docs/logs/2026-05-11_session-A-prose-drafts-v10-exp2.md`](2026-05-11_session-A-prose-drafts-v10-exp2.md) (Day 3 draft, 2-day pull-forward from Day 6).

**Section-level edits applied (single Edit operation in prior session):**
1. Deleted old `% [REVISION | Exp2-recast-reward-ablation | 2026-05-05 | ...]` marker (Option α+ three-arm ablation framing — never ran due to B6 measurement-pipeline bug).
2. Deleted old `\paragraph{Experiment 2: Reward-Function Ablation in Latent Space.}` block including the body paragraph (which contained the last surviving `\ref{eq:mcts_reward}` usage; see warning inventory below) and IMPL line.
3. Inserted new REVISION marker `option-D+-exp2-cartography-2026-05-14`.
4. Inserted new `\paragraph{Experiment 2: Latent Cartography MCTS across Prompt Classes.}\label{sec:exp2}` with 6 prose paragraphs: setup (item enumeration of C1–C4 classes + MCTS protocol), three-signature framework, Results lead-in, findings (i)–(iv) (class-specific best-path selection / argmax-token concentration / gold-rank preservation / mechanism interpretation per class), closing paragraph on search reward gain with BLOCKING-fix footnote inline.
5. Inserted new IMPL line referencing `scripts/diagnose_track_g_cartography.py`.
6. Inserted new Table 2 block (`\label{tab:exp2-cartography-paths}`) — best-path α-tuple distribution per class × scale (4 rows × 6 cols).
7. Inserted new Figure 1 block (`\label{fig:exp2-mean-step-alpha}`) — `\includegraphics{figures/exp2-mean-step-alpha.pdf}` cross-scale mean-step α bar chart.

The 2026-04-24 §5-Exp3-5-cut marker is preserved at line 405, immediately after the new Figure 1 block, marking the boundary to §5.4 Baselines.

### BLOCKING fix wording (Track G audit pass Family D, applied inline)

Track G audit pass 2026-05-13 reclassified the original Track-G report search-reward-gain range "+0.005 to +0.025" as scope-incomplete: the 3B C4 HellaSwag-continuation cell was marked "data truncated" in the report but exists in the raw JSON at $+0.0367$, which rounds to $+0.037$. The recommended-option fix (per `recast_action_plan.md`) extends the body-prose range to "+0.005 to +0.037" and footnotes the upper-bound driver:

> The best path among the 27 sampled gives between $+0.005$ and $+0.037$ better reward (normalized-entropy units) than the mean across paths.\footnote{The upper bound is driven by HellaSwag continuation at the 3B scale (path-mean reward gain $+0.037$); the other seven cells span $+0.005$ to $+0.025$.}

The footnote text is verbatim as approved (option b per Josh sign-off). It names the asymmetric driver cell honestly without inflating the range claim for the other 7 cells.

### Reference resolution

The new `\label{sec:exp2}` resolves 7 forward references that were undefined at Day-5 EOD (zou2023repe-fix session):

| Source location | Reference | Status |
|---|---|---|
| §1 Introduction paragraph 4 (line ~70) | `\ref{sec:exp2}` | ✅ resolves |
| §1 Contribution 3 (line ~82) | `\ref{sec:exp2}` | ✅ resolves |
| §2.2 Bailey complementarity (line ~135) | `\ref{sec:exp2}` | ✅ resolves |
| §3.2 Hypothesis 2 closing (line ~172) | `\ref{sec:exp2}` | ✅ resolves |
| §5 Exp 1 Interpretation closing (line ~357) | `\ref{sec:exp2}` | ✅ resolves |
| §7.5 monitor-design-research paragraph (line ~569) | `\ref{sec:exp2}` | ✅ resolves |
| §7.5 reward-design-research paragraph (line ~580 area) | `\ref{sec:exp2}` | ✅ resolves |

All 7 of the Day-5-EOD-inventoried `sec:exp2` undefined-reference warnings are retired. The new labels `tab:exp2-cartography-paths` and `fig:exp2-mean-step-alpha` are self-referenced within the new §5 Exp 2 body paragraphs; no collisions with pre-existing labels.

### G-smoke-compile gate

Per plan v2 §9, ran `pdflatex` twice on the canonical TeX from inside `docs/NeurIPS/`. MiKTeX at `C:\Users\Josh\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe` with `-interaction=nonstopmode -halt-on-error`.

| Pass | Exit | PDF | Pages | Bytes | Errors |
|---|---|---|---|---|---|
| 1 | 0 | produced | 20 | 430,992 | 0 |
| 2 | 0 | produced | 20 | 432,417 | 0 |

Pass 2 byte-count slightly larger than pass 1: cross-references resolve and Table 2 + Figure 1 numbers fill in on the second pass. Page count went from 18 (Day-5 zou2023repe-fix EOD) → 20 (post-§5-Exp-2 today), accounting for the new Table 2 + Figure 1 + ~50 lines of body prose.

### Warning inventory verification

| Warning class | Day-5 zou2023repe-fix EOD | Day-5 §5-Exp-2 (today) | Verdict |
|---|---|---|---|
| `sec:exp2` undefined | 7× | **0×** | ✅ resolved by new `\label{sec:exp2}` |
| `eq:mcts_reward` undefined | 1× | **0×** | ✅ **incidental retirement** — the old `\paragraph{Experiment 2: Reward-Function Ablation in Latent Space.}` body contained the last surviving `\ref{eq:mcts_reward}` usage; replacement deletes it. Day-7 §5.5 rewrite scope reduced by 1 item (the broken-ref cleanup is already complete; only the metric-set rewrite remains). |
| "There were undefined references" meta | 1× | **0×** | ✅ no undefined refs remain |
| `'h' → 'ht'` float (cosmetic) | 1× | 5× | acceptable — Table 2 + Figure 1 + cumulative carries add float warnings; all cosmetic per resume-prompt explicit allow |
| `zou2023repe` Author undefined natbib | 0× (retired by `19f9eb5`) | 0× | ✅ stable |
| `Error:` lines | 0 | 0 | ✅ |

**Total warning lines:** 10 (Day-5 zou2023repe-fix EOD) → **5** (today, all cosmetic). NET DOWN by 5 lines; structural warnings fully retired.

**Gate verdict:** PASS. Beats the resume-prompt expectation of 3-4 lines with `eq:mcts_reward` remaining (the incidental retirement makes this cleaner than projected).

### Git commit

`895ecca` — `docs: Day 5 — §5 Exp 2 Cartography recast + Table 2 + Figure 1 + BLOCKING fix`. Scope: 1 file changed (canonical TeX), +44 / −4 lines. No build artifacts, no `.claude/`, no submodule churn, no premortem-v2 variant, no session-log churn (this log update is appended now, not bundled — per the Day-4/5 pattern of committing session logs alongside the next day's work).

### Plan v2 §2 reconciliation

§5 Exp 2 was scheduled for Day 7 in plan v2 §2. Today's application is a **2-day pull-forward**, the 7th pull-forward of the sprint:

1. Day 2: Cartography Track G from Days 4–5.
2. Day 3: §5 Exp 2 prose draft from Day 6.
3. Day 4 P2: §A Reproducibility Appendix v1 from Day 9.
4. Day 4 (Day-6 satellite): Figure 1 generation + Track G audit pass from Day 6.
5. Day 5 (zou2023repe-fix session): Day-9 residue zou2023repe `\citet` numbers-mode fix from Day 9.
6. **Day 5 (resume session, today): §5 Exp 2 Cartography recast from Day 7.**
7. Day 5 (resume session, today, incidental): `eq:mcts_reward` undefined-ref cleanup from Day 7 §5.5 rewrite scope.

The pattern continues to track the team's documented ~50% time-budget observation (see `memory/feedback.md` "Time estimates are systematically conservative"). Per plan v2 §9's plasticity rule, today's Edit compiles cleanly so it promotes to the next day's baseline.

### Day-9 residue list status

Unchanged from Day-5 morning: 1 item remains (Day-8 ACL conditional check that `acl_natbib.bst` re-resolves `\citet{}` correctly under author-year mode). The §5 Exp 2 work today doesn't touch this item; it neither closes it nor adds residue.

### TeX state

| Metric | Pre-Exp-2 (Day-5 zou2023repe EOD) | Post-Exp-2 (Day-5 resume EOD) | Delta |
|---|---|---|---|
| Total lines | 705 | **745** | +40 (paragraph-per-line formatting; v10-exp2 draft estimated +57, resume-prompt estimated +80) |
| §5 Experiment 2 framing | Option α+ "Reward-Function Ablation" (placeholder; never ran) | Option D+ "Latent Cartography MCTS across Prompt Classes" (Track G, complete 2026-05-11) | ✅ recast |
| §5 Tables | Table 1 (Track F results) | Table 1 (Track F) + **Table 2 (Cartography α-tuple distribution)** | +1 |
| §5 Figures | none | **Figure 1 (cross-scale mean-step α)** | +1 |
| PDF pages | 18 | 20 | +2 |
| `sec:exp2` forward refs (undefined) | 7 | 0 | ✅ resolved |
| `eq:mcts_reward` forward refs (undefined) | 1 | 0 | ✅ incidental |
| §5–§7 D+ alignment | through §7 except §5 Exp 2, §5.5 Metrics, §6 worked-example numbers | through §7 except **§5.5 Metrics (Day 7) + §6 worked-example numbers (Day 8)** | ✅ §5 Exp 2 closed |

After today's commit, §5 + §7 paper-body D+ alignment for Phase 1 is complete except §5.5 Evaluation Metrics (Day 7 rewrite per plan v2 §2) and §6 worked-example numbers (Day 8 Track-D 162-GB / 60× replacement). Days 6, 9, 11, 12, 13 + 14 retain their plan-v2 schedules.

### Day 5 resume wall time

| Activity | Wall |
|---|---|
| Phase 1 read (9 artifacts: CLAUDE.md + MEMORY index + recast_action_plan + feedback.md + 2× session logs + Track G audit pass + v10-exp2 draft + canonical TeX lines 358-460) + Phase-1 readback authoring | ~25 min |
| Step A sanity check (git diff + figure file + line count flag) | ~3 min |
| Step B G-smoke-compile (pdflatex 2 passes) | ~4 min |
| Step C warning inventory verification + grep cross-checks | ~3 min |
| Step D commit | ~1 min |
| Step E session log append (this section) | ~15 min |
| Step F memory cleanup | ~3 min |
| **Day 5 resume total** | **~54 min** |

---

## Open items for Day 6 (post-resume, supersedes original Day-5 EOD planning above)

Plan v2 §2 Day 6 nominal task: *"Complete §5 Exp 2 draft + Table 2 + Figure 1."* As of Day-5 EOD post-resume, **the nominal Day 6 task is fully closed** — Exp 2 paragraph + Table 2 + Figure 1 all landed in canonical TeX in commit `895ecca`. Day 6 therefore has substantial slack:

- **A. ~~v10-exp2 polish + Day-7 fix-spec sufficiency check~~ — CLOSED today.** Exp 2 with BLOCKING fix landed and compiles clean; no polish or fix-spec sufficiency check remains.
- **B. Pull forward Day-6 Figure 1 re-derivation as a third-pass raw-JSON check.** The figure exists and is now referenced from the compiled TeX; a third-pass JSON-vs-figure-value check (no rounding drift between draft and rendered figure) is still doable. ~30 min. Doesn't touch canonical TeX.
- **C. Pull forward Day-8 ACL style switch dry-run.** Higher-leverage but heavier (~1–2 hr); 25% P-failure flagged in plan-v2 §8 risk register — pulling forward de-risks 3 days early. Starts a session, not a tail.
- **D. ~~Day-9 zou2023repe residue~~ — CLOSED Day 5 (zou2023repe-fix session).**
- **E. Bailey 2024 §7.2 polish.** ~20 min. Deferred indefinitely per Josh; reopening requires explicit direction.
- **F. (NEW) §5.5 Evaluation Metrics rewrite (plan v2 §2 Day 7).** With Exp 2 done, §5.5 is the next paper-body D+-alignment item. ~60–90 min. Pulls Day 7 forward by 1 day. Note: the `eq:mcts_reward` broken-ref cleanup that was part of Day-7 §5.5 scope was incidentally retired today, so Day-7 §5.5 is now metric-set-rewrite-only.
- **G. (NEW) §A Reproducibility Appendix application (plan v2 §2 Day 10).** §A v1 draft was authored Day 4 P2 and standalone-smoke-compiled clean; Session B inserts at end of canonical TeX before `\end{document}` with REVISION marker `reproducibility-appendix-v1-D+`. ~30 min. Pulls Day 10 forward by 4 days.

Decision is Josh's call at Day-6 kickoff. The pull-forward pattern is healthy but not load-bearing for the May 25 timeline — Day 6 could equally well be light. After today's resume, plan v2 §2 has at least 3 days of pre-built work available for pull-forward (Items F, G, C). Items F + G together close all remaining §5–§7 paper-body D+-alignment work.

**TeX state for Day-6 baseline:** 745 lines (was 705 at Day-5 zou2023repe-fix EOD), D+-aligned through §7 except the two remaining pockets (§5.5 Evaluation Metrics awaits Day-7 audit rewrite; §6 worked-example numbers awaits Day-8 Track-D 162-GB/60× replacement). G-smoke-compile clean; 5 fewer warnings than Day-5 zou2023repe-fix EOD (all 8 structural warnings retired; only 5 cosmetic h→ht float warnings remain). PDF 20 pages / 432 KB.

---

## Day 5 — Track-selection decision + Deep-research synthesis (post-§5-Exp-2-close-out)

**Trigger:** Josh paused the resume close-out after Step F to apply an EMNLP 2026 subject-area track decision (the J6c row, missing from plan v2 §6 J-table).

**Decision:** Lock subject-area track to **"Interpretability and Analysis of Models for NLP"** (track #13 of 29; verbatim per call-page primary-source verification 2026-05-14).

### Artifacts created/updated this session

- J6c row added to [`docs/logs/2026-05-10_14-day-execution-plan.md`](2026-05-10_14-day-execution-plan.md) §6 J-item table (immediately after J6b, before strategic decisions block).
- `memory/project_context.md` "Plan (v2 ARR-cycle restructure)" venue/deadline paragraph: inline-appended "Subject-area track LOCKED 2026-05-13" note per the inline-only-notes rule from `memory/feedback.md` (no new lines added).
- [`docs/logs/2026-05-13_session-log.md`](2026-05-13_session-log.md) new file: "Track-selection decision" audit-trail entry with decision, reasoning, runner-up rejection, primary-source verification table, G-track gate result (canonical TeX grepped for `submission track` / `subject area` / `subject-area`: 0 hits, clean).
- [`docs/logs/2026-05-14_track-selection-deep-research-prompts.md`](2026-05-14_track-selection-deep-research-prompts.md) new file: 4 standalone Deep Research prompts (base-rate + 3 paper-shape-fit per candidate track #6/#13/#17) + synthesis protocol with decision matrix.

### Day-6 pull-forwards completed during synthesis-wait window (~30 min idle window filled productively)

- **Item B (Figure 1 raw-JSON third-pass verification): PASS.** Ran `scripts/figure_exp2_mean_step_alpha.py`; all 8 mean-step α values rendered in Figure 1 / Table 2 / body prose match raw JSON exactly (0 discrepancies). C3/C4 cross-scale Δ rounding-divergence replicates Track G audit pass's known finding (round-then-subtract -0.55/-0.13 vs subtract-then-round -0.54/-0.12; deferred optional polish per `recast_action_plan.md`, not blocking).
- **Item G prep (§A Reproducibility Appendix application — no TeX touch):** all 7 cross-references the appendix relies on (`sec:experiments`, `sec:exp1`, `sec:exp2`, `prop:memory`, `eq:goodhart`, `tab:exp1-results`, `tab:oei-alpha-sweep`) verified defined in post-`895ecca` TeX. ZERO undefined-ref warnings expected post-application. No `sec:reproducibility` label collision (0 existing hits). Estimated 10-15 min Day-6 application time vs ~30 min original Day-10 estimate.

### Deep Research synthesis (Gemini 3.1 Pro, 4 parallel sessions)

Reports archived at [`docs/logs/gemini research reports/`](gemini%20research%20reports/):

| File | Coverage | Sample N | Verdict |
|---|---|---|---|
| `EMNLP Track Acceptance Rates Analysis.md` | Prompt 1 base rates | EMNLP 2024 + 2025 + ARR cycles | Track-level rates not published anywhere; aggregate 39.50% Main+Findings at EMNLP 2025; Interpretability named one of 3 highest-volume tracks at EACL 2024 |
| `Analyzing Interpretability Track Submissions.md` | Prompt 2 #13 fit | 26 sampled accepted papers | **STRONG fit** |
| `Analyzing NLP Efficiency Track Papers.md` | Prompt 3 #6 fit | 30 sampled accepted papers | **STRONG fit** (recommends #6 as PRIMARY, not secondary — disagrees with current J6c) |
| `Assessing Negative Results in ML-for-NLP.md` | Prompt 4 #17 fit | 25 sampled accepted papers | **PLAUSIBLE fit** (Best Paper ACL 2024 was a primary negative result; reviewer-luck risk remains high) |

### Decision matrix application

Per synthesis protocol step 3: two STRONG fits (#13 + #6) → invoke tie-breaker ("favor the one whose fit-paragraph cites more precedent papers shaped like ours"). Three convergent signals:

1. **Precedent count:** #13 has 8-11 directly comparable papers (L\*, Finite State Automata, BC-Prover, CUTE, Finding Blind Spots, Lexical Recall, Experimental Contexts, RAVEL, TRACE, CausalGym, Latent Concept-based Explanation); #6 has 5-8 (CODI, Decoding in Latent Spaces, TokenSkip, Extracting Prompts by Inverting, AutoEvolve, Multimodal Jailbreak, Back Attention, MYTE). → **#13 wins.**
2. **CFP keyword specificity:** #13's ARR keywords include "probing", "feature attribution", "knowledge tracing/discovering/inducing", "data shortcuts/artifacts" — each maps to a specific paper facet; #6's keywords are generic ("LLM efficiency", "NLP in resource-constrained settings"). → **#13 wins.**
3. **Paper self-positioning:** title is "Memory-Bounded **Probe** of Dimensional Escape" — probe is the noun, memory-bounded is the adjective. Empirical content (Track F + Track G) is interpretability-shaped; the memory bound is enabling infrastructure. → **#13 wins.**

### Artifact decision

**No artifact updates needed.** Decision matches existing J6c row + project_context.md + 2026-05-13 session log entry (synthesis protocol step 4: "update artifacts only if decision changes").

### Honest disagreements surfaced for Day-11 co-author cover note

1. **Prompt 3 explicitly recommends #6 as PRIMARY** (not secondary): formal mathematical analysis appears in only 10% of accepted #6 papers, so the reversibility theorem would be uniquely-positioned there. Credible counter-case that fails the tie-breaker but deserves disclosure to Tianyu/Max/Alaa.
2. **Prompt 4 partially overturned the original #17 dismissal:** the "ML-for-NLP track expects benchmark gains" reasoning was directionally right (reviewer-luck risk is real) but quantitatively wrong (~25-30% of accepted papers have primary negative results; one won Best Paper). #17 remains higher-variance than #13/#6 but is not the dead end the pre-research call framed.

### Cover-note language for Day 11 (2026-05-20 send-out)

> Primary track: **#13 Interpretability and Analysis of Models for NLP** (locked 2026-05-13). Decision basis: 4-report Deep Research synthesis 2026-05-14, 81 sampled accepted papers across EMNLP/ACL 2024-2025. Tie-breaker between #13 STRONG fit and #6 Efficient Methods STRONG fit favored #13 on (a) precedent count, (b) CFP keyword specificity, (c) title self-positioning as a "Probe." If ARR portal allows secondary track, add **#6 Efficient Methods** as secondary hedge. Flag for sign-off: one of the 4 reports recommended #6 as primary; team override based on tie-breaker.

### Day-14 verification gate (carried)

Verify ARR portal supports secondary-track selection. If yes → add #6 secondary. If no → single primary is #13 (current state stands).

### Day 5 track-selection + synthesis wall time

| Activity | Wall |
|---|---|
| J6c row authoring + WebFetch verification + project_context.md edit + 2026-05-13 session log creation | ~30 min |
| Track-pick honest assessment + base-rate-question clarification | ~10 min |
| Deep-research prompts file authoring | ~15 min |
| Day-6 pull-forwards (Item B verification + Item G prep) during synthesis wait | ~20 min |
| 4-report synthesis + decision matrix + this log entry | ~25 min |
| **Track-selection + synthesis total** | **~100 min** |

---

*End of Day 5 (combined zou2023repe-fix session + §5 Exp 2 Cartography recast resume session + track-selection decision + Deep-research synthesis). Day-5 grand total: ~58 min + ~54 min + ~100 min = ~212 min (~3.5 hours). Two commits landed (`19f9eb5` zou2023repe `\citet` fix + `895ecca` §5 Exp 2 Cartography recast + Table 2 + Figure 1 + BLOCKING fix); uncommitted artifacts include this session log + J6c row + 2026-05-13 audit entry + track-selection prompts file (all to bundle with the next paper-content commit per the Day-4/5 pattern). Day-9 residue list: 1 item remaining (Day-8 ACL conditional check). Day-5 absorbed plan-v2 §2 Day 7's §5 Exp 2 task + incidentally absorbed the `eq:mcts_reward` cleanup portion of Day-7 §5.5 scope. §5 + §7 paper-body D+ alignment complete except §5.5 Metrics (Day 7) and §6 worked-example numbers (Day 8). EMNLP subject-area track locked (J6c). Day-6 pull-forwards Items B + G partially landed (verification + prep, no TeX touch). Parallel-review-plan drafting queued next per Josh's earlier request. Ready for Day-6 direction at Josh's call.*
