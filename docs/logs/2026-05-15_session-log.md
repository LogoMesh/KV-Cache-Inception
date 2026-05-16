# Session Log — 2026-05-15 (Day 6 of plan v2 execution)

**Session type:** Session A (Opus) + Session B (TeX application within same Claude session)
**Day of ARR-cycle execution plan:** Day 6 (calendar 2026-05-15 = plan v2 §2 Day 6; internal day-number and plan-calendar aligned)
**Submission deadline countdown:** T-10 calendar days to 2026-05-25 ARR submission (T-79 to EMNLP commitment 2026-08-02)

**Continuity:** Day-5 EOD state is recorded in [`2026-05-14_session-log.md`](2026-05-14_session-log.md) (combined zou2023repe-fix + §5 Exp 2 resume + track-selection + Deep Research synthesis). TeX state at Day-5 rollover: **745 lines**, D+-aligned through §7 except §5.5 Metrics + §6 worked-example numbers (Day 7 + Day 8 scheduled); §A Reproducibility Appendix drafted but not yet applied. Day-5 grand total: ~212 min (~3.5 hr) across 4 work threads. Two commits landed Day 5 (`19f9eb5`, `895ecca`); Day-5 audit trail uncommitted pending Day-6 bundle.

---

## Day 6 entry — §A Reproducibility Appendix application (Day-10 pull-forward) + Day-5 audit trail bundle commit

**Date:** 2026-05-15 (Day 6 of plan v2; calendar rolled from 2026-05-14 mid-session)
**Owner:** Session A + Session B (same Claude session)
**Task:** Apply §A Reproducibility Appendix v1 (drafted 2026-05-13, pre-staged Day-5 Item G prep) to canonical TeX; bundle Day-5 audit trail into the same commit per the `3c5d741` precedent.
**Status:** ✅ COMPLETE — §A applied, G-smoke-compile passed, commit landed (`f61f391`), memory updates applied, this entry authored.
**Decision basis:** Pre-stage during Day-5 synthesis-wait window cleared all risk surfaces (all 7 cross-refs verified defined, no label collision, ZERO undefined-ref warnings predicted). Application pulls plan-v2 §2 Day-10 work forward by 4 days.

### Edit details

**Insertion point:** between canonical TeX line 743 (`\end{thebibliography}`) and line 745 (`\end{document}`). Unambiguous per Day-5 Item G prep grep.

**REVISION marker:** `reproducibility-appendix-v1-D+ | 2026-05-14` (calendar date when commit happened — calendar rolled to 2026-05-15 mid-session, but the commit timestamp was 2026-05-14 EOD). Marker convention matches `19f9eb5` (`zou2023repe-citet-numbers-mode-fix-D+ | 2026-05-14`) and `895ecca` (`option-D+-exp2-cartography-2026-05-14 | 2026-05-14`) which both used calendar date of application.

**TeX block applied verbatim from draft** (with REVISION marker date update 2026-05-13 → 2026-05-14):
- `\appendix \section{Reproducibility}\label{sec:reproducibility}` opening
- §A.1 Hardware and software stack (RTX 3060 12 GiB, CUDA 12.4, PyTorch 2.6.0+cu124, Windows 10)
- §A.2 Models and checkpoints (Llama-3.2 1B + 3B architecture facts)
- §A.3 Reversible MCTS configuration (MCTSConfig defaults + path-sampling variant explanation)
- §A.4 Per-track configurations (Track A entropy + Track C calibration + Track D VRAM + Track F negative-control protocols)
- §A.5 Random seeds and reproducibility caveats (--seed 42 + per-item RNG md5 + supplementary URL placeholder)
- Closing IMPL line

### Reference resolution

The new `\label{sec:reproducibility}` defined; no collision with pre-existing labels (Day-5 prep grep returned 0 hits). All 7 cross-references the appendix relies on were verified defined in canonical TeX before application:

| Reference | Defined at canonical TeX | Status post-application |
|---|---|---|
| `\ref{sec:experiments}` | line 285 | ✅ resolves |
| `\ref{sec:exp1}` | line 327 | ✅ resolves |
| `\ref{sec:exp2}` | line 361 (Day-5 `895ecca`) | ✅ resolves |
| `\ref{prop:memory}` | line 523 | ✅ resolves |
| `\ref{eq:goodhart}` | line 167 | ✅ resolves |
| `\ref{tab:exp1-results}` | line 352 | ✅ resolves |
| `\ref{tab:oei-alpha-sweep}` | line 598 (Day-4 P1) | ✅ resolves |

**Net:** ZERO new undefined-ref warnings introduced by §A application (exactly as predicted in Day-5 Item G prep).

### G-smoke-compile gate

Per plan v2 §9, ran `pdflatex` twice from inside `docs/NeurIPS/`. MiKTeX at `C:\Users\Josh\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe` with `-interaction=nonstopmode -halt-on-error`.

| Pass | Exit | PDF | Pages | Bytes | Errors |
|---|---|---|---|---|---|
| 1 | 0 | produced | 22 | 455,939 | 0 |
| 2 | 0 | produced | 22 | 456,274 | 0 |

Page count went from 20 (post-`895ecca`) → 22 (post-§A); §A added 2 pages of body content. PDF byte-count grew ~24 KB.

### Warning inventory verification

| Warning class | Post-`895ecca` (Day-5 EOD) | Post-§A (Day-6 today) | Verdict |
|---|---|---|---|
| `'h' → 'ht'` float (cosmetic) | 5× | 5× | unchanged — §A has no floats |
| LaTeX Font Warning: `OMS/cmtt/m/n` textbraceleft fallback | 0× | 1× | **NEW; cosmetic** — triggered by `\texttt{...\{1B,3B\}...}` brace-in-typewriter; MiKTeX falls back to OMS/cmsy. Acceptable per resume-prompt cosmetic-warning allow. |
| Underfull `\hbox` | 0× | 1× | **NEW; cosmetic** — line-breaking in §A.4 Track-A paragraph (lines 761–762). Acceptable. |
| Undefined references | 0× | 0× | ✅ stable |
| "There were undefined references" meta-warning | 0× | 0× | ✅ stable |
| `Error:` lines | 0 | 0 | ✅ |

**Total warning lines:** 5 (post-`895ecca`) → **7** (today). 2 added are both cosmetic (font fallback + underfull hbox). Zero substantive warnings.

**Gate verdict:** PASS. No new undefined refs, no errors, no structural warnings.

### Git commit

`f61f391` — `docs: Day 6 — §A Reproducibility Appendix application + Day-5 audit trail bundle`. Per the Day-4 P1 commit (`3c5d741`) bundled-pattern precedent. **17 files changed, +3352 / -3 lines.**

**Scope (substantive Day-6 work):**
- `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` (§A applied)

**Scope (Day-5 audit trail bundle):**
- `docs/logs/2026-05-14_session-log.md` (Day-5 combined: zou2023repe-fix + §5 Exp 2 resume + track-selection + Deep Research synthesis)
- `docs/logs/2026-05-10_14-day-execution-plan.md` (J6c row added to §6 J-table)
- `docs/logs/2026-05-13_session-A-prose-drafts-v10-reproducibility-appendix.md` (§A v1 draft, post-fix; source for today's application)
- `docs/logs/2026-05-13_session-log.md` (new file; track-selection decision audit entry)
- `docs/logs/2026-05-13_track-G-audit-pass.md` (audit-trail file referenced by §5 Exp 2 BLOCKING fix)
- `docs/logs/2026-05-14_day-5-resume-prompt.md` (the resume prompt that started Day-5 session)
- `docs/logs/2026-05-14_day-5-session-prompt.md` (alternative Day-5 prompt; audit trail)
- `docs/logs/2026-05-14_parallel-review-plan.md` (Day-5 authored, postponed to Days 12-13)
- `docs/logs/2026-05-14_track-selection-deep-research-prompts.md` (4 prompts authored Day 5)
- `docs/logs/gemini research reports/` (4 reports downloaded Day 5; synthesis source material)
- `docs/EMNLP-track-info/track-13-interpretability-scope.md` (new folder + file with track #13 verbatim scope)
- `scripts/audit_track_g_report.py` (Day-3 audit script; audit trail)
- `scripts/figure_exp2_mean_step_alpha.py` (Day-6-pulled-forward Figure 1 generator; audit trail)

**Scope excludes:**
- Build artifacts (`.aux`, `.log`, `.pdf`, `.out`)
- `.claude/` directory
- `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2-premortem-v2.tex` (historical variant, not edited)
- `docs/NeurIPS/Citations/` PDFs (per prior pattern)
- Submodule churn (`.gitmodules` + `external/Logomesh/`)
- `docs/NeurIPS/figures/` (Figure 1 PDF stays untracked per prior pattern; generated deterministically from `figure_exp2_mean_step_alpha.py`)
- `docs/logs/session-export/`
- `docs/logs/2026-05-11_next-paper-novelty-check.md` (modified but not by today's work; Josh may have plans)

### Plan v2 §2 reconciliation

§A application was scheduled for Day 10 (2026-05-19) in plan v2 §2. Today's Day-6 application is a **4-day pull-forward**, the 8th pull-forward of the sprint:

1. Day 2: Cartography Track G from Days 4–5.
2. Day 3: §5 Exp 2 prose draft from Day 6.
3. Day 4 P2: §A Reproducibility Appendix v1 from Day 9.
4. Day 4 (Day-6 satellite): Figure 1 generation + Track G audit pass from Day 6.
5. Day 5 (zou2023repe-fix session): Day-9 residue zou2023repe `\citet` numbers-mode fix from Day 9.
6. Day 5 (resume session): §5 Exp 2 Cartography recast from Day 7.
7. Day 5 (resume session, incidental): `eq:mcts_reward` undefined-ref cleanup from Day-7 §5.5 scope.
8. **Day 6 (today): §A Reproducibility Appendix application from Day 10.**

The pattern continues to track the team's documented ~50% time-budget observation (memory/feedback.md "Time estimates are systematically conservative"). Per plan v2 §9's plasticity rule, today's application compiles cleanly so it promotes to the next day's baseline.

### TeX state

| Metric | Pre-§A (post-`895ecca`) | Post-§A (post-`f61f391`) | Delta |
|---|---|---|---|
| Total lines | 745 | **776** | +31 (Day-5 Item G prep estimated +45; v10-A draft estimated +45; actual +31 due to dense paragraph-per-line formatting) |
| `\appendix` block | absent | inserted between bibliography and `\end{document}` | ✅ new |
| `\label{sec:reproducibility}` | undefined | defined at TeX:746 | ✅ new |
| §A subsections | 0 | 5 (Hardware/Models/MCTSConfig/Per-track configs/Seeds-and-caveats) | ✅ new |
| Forward references (intra-§A) | n/a | 7 to existing labels (all resolve) | ✅ |
| §5–§7 D+ alignment | through §7 except §5.5 + §6 worked numbers | unchanged | — |
| PDF pages | 20 | 22 | +2 |
| §A page budget impact (review version) | n/a | ZERO (appendix excluded from 8-page review budget per ACL ARR rules) | ✅ |

After today's commit, **§A is in the TeX**. The paper-content "remaining work" list narrows further:
- §5.5 Evaluation Metrics rewrite (Day 7 nominal; partially de-scoped Day 5 because `eq:mcts_reward` cleanup was incidentally retired)
- §6 worked-example numbers replacement with Track D 162-GB / 60× figures (Day 8 nominal)
- ACL style switch (`neurips_2026.sty` → `acl.sty`) (Day 8 nominal; risk-flagged 25% P-failure per plan v2 §8)
- Final REVISION-marker audit + double-blind compile check (Day 14)

### Day-9 residue list status

Unchanged: 1 item remains (Day-8 ACL conditional check that `acl_natbib.bst` re-resolves `\citet{}` correctly under author-year mode). §A application doesn't touch the natbib quick-patch or any `\citet{}` usage, so no new residue added.

### Day 6 wall time

| Activity | Wall |
|---|---|
| Phase 1 re-orientation (TodoWrite cleanup; verify Day-5 EOD state) | ~3 min |
| EMNLP-track-info folder + #13 scope file authoring | ~7 min |
| §A application (Edit operation; one big block insertion) | ~5 min |
| G-smoke-compile (pdflatex 2 passes + warning grep) | ~4 min |
| Commit staging + commit with HEREDOC message | ~5 min |
| Memory pointer updates (MEMORY.md + recast_action_plan.md, 3 edits) | ~8 min |
| This session log authoring | ~12 min |
| **Day 6 total** | **~44 min** |

---

## Open items for Day 7 (2026-05-16)

Plan v2 §2 Day 7 nominal task: **§5.5 Evaluation Metrics rewrite** (audit §5-6). After Day-5's incidental retirement of `eq:mcts_reward` (the broken-ref cleanup that was originally part of §5.5 scope), the remaining §5.5 work is the metric-set rewrite only — narrower than the original Day-7 scope. Estimated ~45-60 min instead of the original ~60-90 min.

Other available pull-forward options post-Day-6:

- **A. §5.5 Evaluation Metrics rewrite (on-plan).** Day-7 nominal; substantial paper-content D+ alignment item. ~45-60 min post the eq:mcts_reward-cleanup-already-done discount.
- **B. §6 worked-example numbers replacement (plan-v2 §2 Day 8).** Replace 40.05 GB / 9.7 TB α+-vintage placeholders with Track D 162 GB / 60× figures. ~30-45 min. Pulls Day 8 forward.
- **C. ACL style switch (plan-v2 §2 Day 8).** Higher-leverage but heavier (~1-2 hr); 25% P-failure flagged in plan-v2 §8 risk register; pulling forward de-risks. Starts a session.
- **D. Page-budget check (plan-v2 §2 Day 9 carry).** Confirm TeX renders within 8 pages ACL-two-column post-§5.5 + post-§6 work. ~15 min check; may surface Day-10 cut decisions if over.
- **E. Bailey 2024 §7.2 polish.** ~20 min. Deferred indefinitely per Josh; reopening requires explicit direction.

Decision is Josh's call at Day-7 kickoff. Day-6 was light (only ~44 min). Plan-v2 sprint pace continues to run ahead of schedule — Day-7 has substantial slack and any of A-D is on the menu.

**TeX state for Day-7 baseline:** 776 lines (was 745 at Day-5 EOD), D+-aligned through §7 + §A complete. Remaining paper-content pockets: §5.5 Metrics (Day 7) + §6 worked numbers (Day 8) + ACL switch (Day 8). G-smoke-compile clean; 7 cosmetic warnings (5× h→ht float + 1× font fallback + 1× underfull hbox); ZERO substantive warnings. PDF 22 pages / 456 KB at NeurIPS style; projected ~10 pages ACL two-column post-Day-8 switch (note: §A pages are excluded from the 8-page review budget).

---

*End of Day 6 entry. Day-6 wall total: ~44 min (light day; §A pre-staged Day 5 made the application a fast 5-min Edit + smoke-compile + commit). One commit landed (`f61f391` §A Reproducibility Appendix + Day-5 audit trail bundle). Cumulative Days 5+6: ~256 min (~4.3 hr) across 5 work threads. Three Day-5+6 commits in git history (`19f9eb5`, `895ecca`, `f61f391`).*

---

## Day 7 entry — Sister-session review actions + §5.5 Evaluation Metrics rewrite (post-Day-6-pull-forward)

**Date:** 2026-05-15 (calendar still 2026-05-15; Day 7 of plan v2 nominal calendar 2026-05-16 — Day-7 work pulled forward by 1 day into Day-6 calendar)
**Owner:** Session A + Session B (same Claude session)
**Trigger:** Josh ran a sister Sonnet review session on the Day-6 EOD status report. That review surfaced 3 pre-Day-7 actions (§A fidelity diff, untracked figure, §5.5 scheduling sync). Josh greenlit Option A on §5.5 ("apply audit fix directly — let me trust you on this").
**Status:** ✅ COMPLETE — all 3 pre-Day-7 actions executed + §5.5 rewrite applied + commit landed (`6e88af2`) + memory updates applied + this entry authored.
**Decision basis:** Sister-review identified untracked figure as belonging in top-3 risks (Day-12 supplementary build would fail); §A fidelity required re-verification; §5.5 was underscheduled in plan v2 §2 but audit had a proposed fix ready.

### Action 1 — §A fidelity diff: CLEAN

Line-by-line comparison of [`docs/logs/2026-05-13_session-A-prose-drafts-v10-reproducibility-appendix.md`](2026-05-13_session-A-prose-drafts-v10-reproducibility-appendix.md) section 2 (draft lines 28-57) vs canonical TeX lines 745-774 (post-`f61f391`):

| Element | Draft | Canonical | Verdict |
|---|---|---|---|
| REVISION marker date | `2026-05-13` | `2026-05-14` | **Intentional change** (marker-date = commit-calendar-date convention) |
| REVISION marker body | full | full | identical |
| `\appendix` + `\section{}` | line 29-30 | line 746-747 | identical |
| §A.1 Hardware paragraph | line 33 | line 750 | identical |
| §A.2 Models paragraph (incl. all Llama-3.2 architecture facts) | line 36 | line 753 | identical |
| §A.3 MCTSConfig paragraph (incl. defaults + path-sampling variant explanation) | line 39 | line 756 | identical |
| §A.4 Track A paragraph (incl. `\ref{sec:exp1}`) | line 44 | line 761 | identical |
| §A.4 Track C paragraph (incl. McNemar power formula) | line 47 | line 764 | identical |
| §A.4 Track D paragraph (incl. **`\ref{prop:memory}`** — the previously-applied `sec:memory-complexity` → `prop:memory` fix) | line 50 | line 767 | identical |
| §A.4 Track F paragraph (incl. COCONUT defense reference + Llama-3.2 wall-clocks) | line 53 | line 770 | identical |
| §A.5 Random seeds paragraph (incl. `\url{anonymous.4open.science...}` placeholder) | line 56 | line 773 | identical |
| IMPL line | line 57 | line 774 | identical |

**Net:** ONE intentional change (date field), everything else byte-for-byte verbatim. Memory was correct on the `prop:memory` fix; smoke-compile passing with ZERO undefined-ref warnings was sufficient evidence. Nothing dropped or silently modified.

### Action 2 — Figure 1 PDF committed: `694a72b`

`docs/NeurIPS/figures/exp2-mean-step-alpha.pdf` (21 KB, binary; rendered 2026-05-14 by `scripts/figure_exp2_mean_step_alpha.py`) committed as scope-minimal commit `694a72b`: `docs: track exp2 mean-step-alpha figure (referenced by canonical TeX via \includegraphics)`. Day-12 supplementary build now unblocked — anyone cloning the repo and running pdflatex will succeed on §5 Exp 2 Figure 1 reference resolution.

**Sister-review correction acknowledged:** the untracked figure was mis-ranked in my Day-6 EOD status report (Section 6 "smaller items" rather than Section 7 "top-3 risks"). The fix is hard-deadline (Day-12 supplementary build), not best-effort.

### Action 3 — §5.5 scheduling sync resolved

Three one-sentence answers per the sister-review prompt:

1. **Does §5.5 appear in plan v2 §2 day-by-day matrix?** No — the only Day-7 Session A row (plan v2 §2 line 59) is "§5 integration audit" (running-log output); no Session B row for §5.5 rewrite exists anywhere.
2. **Which plan v2 row does recast_action_plan's §5.5 item supplement or replace?** It supplements — the item originated from [Day-3 stale-sentence audit Finding §5-6 BLOCKING](2026-05-11_stale-sentence-audit.md) at line 219+, which had its own internal schedule (audit line 334: "Apply §5-6 §5.5 Eval Metrics rewrite | Day 7 | Session B"); plan v2 §2 absorbed the Session A audit task but never rolled the Session B rewrite task into its matrix.
3. **Expected output?** Direct TeX edit per the audit's proposed-fix block at audit line 231+ (Josh chose Option A: apply directly without intermediate v10-eval-metrics draft).

### §5.5 Evaluation Metrics rewrite (Option A direct application)

**Source:** Audit Finding §5-6 BLOCKING, proposed fix at audit line 231+. The audit specified a 5-bullet markdown structure; today's application translated it to LaTeX `itemize` form with appropriate cross-references to in-paper labels.

**REVISION marker:** `§5.5-eval-metrics-rewrite-D+ | 2026-05-15`. Subsumes three prior markers retired by this commit:
- `T-C-TDS-reframe | 2026-05-05` (TDS-as-reward-component framing — incompatible with Option D+ where TDS is not the MCTS reward)
- `§5.5-TDS-bullet-ref-cleanup-D+ | 2026-05-11` (the stop-gap broken-`\ref{eq:mcts_reward}` removal)
- `§5.4-metrics-cull | 2026-05-05` (the earlier orphaned-metric trim that became obsolete with the full rewrite)

**Section structure post-rewrite (5 items):**

1. **Primary metric (Track F + Cartography):** best-path $\alpha$-tuple distribution per (class, scale) — refs `\ref{sec:exp1}`, `\ref{tab:exp2-cartography-paths}`, `\ref{fig:exp2-mean-step-alpha}`.
2. **Primary metric (Track F):** paired McNemar accuracy test — refs `\ref{sec:reproducibility}`, `\ref{tab:exp1-results}`.
3. **Primary metric (Cartography):** argmax-at-best-leaf token distribution per (class, scale) — refs `\ref{sec:exp2}` finding (ii).
4. **Secondary metric (OEI):** repositioned as monitor-design diagnostic for §7.4; eq:oei equation + label PRESERVED here so forward references from §7.4 resolve. CRITIQUE NOTE about OEI nonlinear-redistribution false-negatives preserved verbatim.
5. **Internal metric (TDS):** Jensen-Shannon divergence between H-Neuron stress and RepE projection signals; supplementary materials only, not a results column — refs `\ref{sec:reproducibility}`.

### G-smoke-compile gate

| Pass | Exit | PDF | Pages | Bytes |
|---|---|---|---|---|
| 1 | 0 | produced | 22 | 460,079 |
| 2 | 0 | produced | 22 | 460,079 |

Pass 2 byte-stable with pass 1 (cross-refs resolved on first pass). PDF byte-count grew from 456,274 (post-§A) → 460,079 (post-§5.5 rewrite, +3.8 KB).

### Warning inventory (broader grep this time — `Warning|Overfull|Underfull|undefined|Error`)

| Class | Count | Cause | Severity |
|---|---|---|---|
| `'h' → 'ht'` float | 4 | Cosmetic float-spec downgrade | Cosmetic |
| `Overfull \hbox` (314pt at TeX 448-458) | 1 | **§5.4 Baselines table row description too long — PRE-EXISTING since Day 3 trim** (sister review's earlier grep missed it; not caused by §5.5 rewrite) | Cosmetic but visible — Day-9 residue candidate |
| `Overfull \hbox` (0.6pt at TeX 566-567) | 1 | Sub-typographic-resolution (invisible to eye) | Cosmetic |
| `Underfull \vbox` (page output) | 4 | Page-break artifacts from 22-page document | Cosmetic |
| `Underfull \hbox` (line breaks at §A.4) | 4 | Cosmetic line-breaking in `\texttt{...}` paragraphs | Cosmetic |
| `LaTeX Font Warning: OMS/cmtt/m/n` | 1 | `\texttt{...{1B,3B}...}` brace fallback from §A | Cosmetic |
| `LaTeX Font Warning: Some font shapes...` meta | 1 | Companion to font-fallback above | Cosmetic |
| Undefined references | 0 | — | — |
| `Error:` lines | 0 | — | — |

**Gate verdict:** PASS. §5.5 rewrite introduced no new substantive warnings. The 1× h→ht float reduction (5 → 4) is due to one fewer cosmetic float warning post-§5.5 (cause uncertain but not blocking). The 314pt Overfull at §5.4 Baselines table is **pre-existing** since the Day-3 Baselines trim — flagging now as a Day-9 residue item that I missed in earlier session-log grep patterns (my prior grep was `LaTeX Warning|^!|Author undefined` which didn't match `Overfull/Underfull`).

### Git commit

`6e88af2` — `docs: Day 7 — §5.5 Evaluation Metrics rewrite per audit Finding §5-6 (Option A direct application)`. Scope-minimal per `895ecca` pattern: 1 file changed (canonical TeX only), +6 / −7 lines (net −1; 3 old REVISION markers consolidated into 1 new marker more than offsets the audit-spec content delta).

### TeX state

| Metric | Pre-§5.5 (post-`694a72b`) | Post-§5.5 (post-`6e88af2`) | Delta |
|---|---|---|---|
| Total lines | 776 | **775** | −1 (3 markers → 1 marker; +4 lines of new bullet content; net −1) |
| §5.5 Evaluation Metrics framing | α+ "OEI as primary + TDS as reward component" (2-item itemize) | D+ "3 primary metrics + OEI secondary + TDS internal" (5-item itemize) | ✅ rewrite |
| `\label{eq:oei}` | defined at line 468 | preserved at new location (under §5.5 item 4) | ✅ stable |
| §5+§7 D+ alignment | through §7 + §A; **§5.5 still α+ vintage** | through §7 + §A + **§5.5** ✅ | one more pocket closed |
| PDF pages | 22 | 22 | 0 |
| Undefined-ref warnings | 0 | 0 | ✅ stable |

After today's commit, only TWO paper-body pockets remain: §6 worked-example numbers (Day 8 nominal: replace 40.05 GB / 9.7 TB placeholders with Track D 162 GB / 60× figures) and the ACL `acl.sty` style switch (Day 8 nominal, risk-flagged 25% P-failure).

### Plan v2 §2 reconciliation

§5.5 rewrite was underscheduled in plan v2 §2 (per Action 3 above) — the audit document had it as Day-7 Session B work; plan v2 §2's Day-7 line listed only "§5 integration audit" (Session A). Today's application closes that audit-scheduled item AND covers the "§5 integration audit" implicit scope (since the §5.5 rewrite IS the major integration concern after Exp 1 + Exp 2 + Baselines all landed). **9th pull-forward of the sprint:**

1. Day 2: Cartography Track G from Days 4–5.
2. Day 3: §5 Exp 2 prose draft from Day 6.
3. Day 4 P2: §A Reproducibility Appendix v1 from Day 9.
4. Day 4 (Day-6 satellite): Figure 1 generation + Track G audit pass from Day 6.
5. Day 5 (zou2023repe-fix session): Day-9 residue zou2023repe `\citet` fix from Day 9.
6. Day 5 (resume session): §5 Exp 2 Cartography recast from Day 7.
7. Day 5 (resume session, incidental): `eq:mcts_reward` undefined-ref cleanup from Day-7 §5.5 scope.
8. Day 6: §A Reproducibility Appendix application from Day 10.
9. **Day 7 (today): §5.5 Evaluation Metrics rewrite from Day-7-implicit audit-scheduled work.**

### Day-9 residue list status

Unchanged + 1 NEW item:
- Item 1 (unchanged): Day-8 ACL conditional check that `acl_natbib.bst` re-resolves `\citet{}` correctly under author-year mode.
- **Item 2 (NEW):** §5.4 Baselines table row 453 produces a 314pt Overfull \hbox (TeX lines 448-458; pre-existing since Day-3 trim but now surfaced via the broader warning grep this session). Cosmetic but visible in PDF — likely the "One-shot static KV-cache steering (precomputed mean-of-differences vector, single post-prefill injection)" cell description is too long for the table column. Day-9 residue removal candidate.

### Day 7 wall time

| Activity | Wall |
|---|---|
| Sister-session review intake + scope confirmation | ~5 min |
| Action 1: §A fidelity diff (read draft + canonical + line-by-line table) | ~10 min |
| Action 2: figure git add + commit | ~2 min |
| Action 3: §5.5 sync resolution (grep audit doc + plan v2 + recast_action_plan) | ~8 min |
| §5.5 rewrite drafting + Edit application | ~15 min |
| G-smoke-compile + warning inventory (broader grep this time) | ~5 min |
| Commit + memory updates (2 files, 3 edits) | ~10 min |
| This session log append | ~12 min |
| **Day 7 total** | **~67 min** |

### Day 7 deliverables summary

- 2 new commits: `694a72b` (figure tracked), `6e88af2` (§5.5 rewrite)
- 1 fidelity-diff verification (§A application clean)
- 1 plan-v2 sync resolution (§5.5 was underscheduled; resolved via direct application)
- 1 NEW Day-9 residue item flagged (§5.4 Overfull \hbox)
- Memory pointers updated for both new commits + §5.5 closure

---

## Open items for Day 8 (2026-05-17 plan-v2 nominal; could be pulled forward into 2026-05-16 calendar)

Plan v2 §2 Day 8 has three Session B tasks (lines 60-62):
- **ACL `acl.sty` style switch** (risk-flagged 25% P-failure per plan v2 §8): swap `\usepackage[main]{neurips_2026}` → `\usepackage[review]{acl}`; fail-fast smoke compile.
- **§6 Memory Complexity replacement:** Apply Track D 162 GB / 60× / `(b,d)=(3,5)` figures (replaces 40.05 GB / 9.7 TB placeholders at TeX line ~515).
- **`\reviewedit{}` wrapping in `\iffalse...\fi`:** for review-PDF hygiene (REVISION marker comments shouldn't render).

These are the LAST two paper-content pockets + the style-switch risk-event. After Day 8 the paper is structurally complete pending Day 9 page-count check + Day 10 Responsible NLP final + Day 11 co-author send-out.

**TeX state for Day-8 baseline:** 775 lines, D+-aligned through §7 + §A + §5.5. Remaining gaps: §6 worked numbers + ACL switch. G-smoke-compile clean; 11 cosmetic warning lines total (broader grep — was previously under-grepped). PDF 22 pp / 460 KB at NeurIPS style; projected ~9-10 pp ACL two-column post-switch (note: §A pages excluded from 8-page review budget).

---

*End of Day 7 entry. Day-7 wall total: ~67 min. Two commits landed (`694a72b` figure tracking + `6e88af2` §5.5 rewrite). Cumulative Days 5+6+7: ~323 min (~5.4 hr) across 6 work threads. Five commits in git history (`19f9eb5`, `895ecca`, `f61f391`, `694a72b`, `6e88af2`). §5 + §7 + §A + §5.5 paper-body D+ alignment complete; only §6 worked numbers + ACL switch remain (both Day-8 nominal). 9 pull-forwards across 5 days. Day-9 residue: 2 items (Day-8 ACL conditional check + §5.4 Overfull \hbox cosmetic). Ready for Day-8 direction at Josh's call.*

---

## Day 7 evening — Sonnet Day-7 EOD review + substance check (verbatim, per Josh directive 2026-05-15)

**Why this is here:** Josh ran a sister Sonnet review on the Day-7 EOD report, then asked a substantive question about whether the working session and the review session both actually understand the paper vs operating on procedural pattern-matching. The exchange is preserved verbatim below as an audit-trail artifact. The honest answer below is the working session's first-person assessment; not a summary.

### Sonnet's Day-7 EOD review (verbatim)

> Day 7 looks clean. Three quick observations on the EOD report before you greenlight Day 8:
>
> The three pre-Day-7 actions closed correctly. §A fidelity diff was clean (all 5 subsections, all refs, \url placeholder — nothing silently dropped). Figure committed at `694a72b`. §5.5 sync resolved with the source identified (stale-sentence audit line 219+ Finding §5-6, which plan v2 §2 absorbed partially but didn't roll the Session B rewrite into the matrix).
>
> The §5.5 rewrite landed cleanly. Option A execution: 5-item D+-aligned itemize replacing α+-vintage 2-item OEI framing, 3 old REVISION markers consolidated into 1, −1 net lines (775 from 776), zero new structural warnings, zero undefined refs. The warning grep pattern upgrade (now catches Overfull/Underfull) is the right hygiene fix and the session surfaced it honestly rather than burying it.
>
> One thing worth watching going into Day 8: the 314pt Overfull \hbox at §5.4 Baselines table. At 314pt (~4.4 inches), that's a significant overhang — pre-existing and missed by the prior grep pattern, but it exists and it's in a section. The ACL style switch on Day 8 often refluxes tables and may fix or worsen it. Flag it to the working session before the style switch so it's in scope for the Day-8 smoke-compile check, not left for Day 9. If the switch makes it worse, that's better to know immediately.
>
> Day-8 path is: ACL `acl.sty` switch (25% P-failure per plan v2 §8 — the highest-priority remaining gate) + §6 worked-example numbers replacement. Both are Session B TeX work. The style switch should be done first and smoke-compiled clean before §6 numbers land — if the switch fails, you don't want §6 changes tangled in the debug.

### Josh's substance-check question (verbatim)

> (Josh here; that message above is from Sonnet. I genuinely have no idea how the progress is going and I'm biting my nails hoping that the paper is actually understood by both you and Sonnet. Can you tell me if you're having to make guesswork based on the instructions you're following or if things make actual sense as to what we're doing in a nutshell right now and why we're doing it?)

### Working session's first-person answer (verbatim)

> Real answer, no glaze:
>
> **What I genuinely understand:**
>
> The paper argues that reversible MCTS in KV-cache latent space is interesting as **infrastructure** (algorithm + memory bound + telemetry matrix), and that using it to optimize an entropy surrogate reward demonstrates **Goodhart in cache geometry** — the search succeeds at lowering entropy but fails to improve accuracy because entropy isn't a good proxy for reasoning. Track G generalizes this to four prompt classes and shows the search exploits class-conditional priors instead of finding semantic signal. Framed as **Dimensional Escape**: low-rank surrogate signals admit degenerate optima in high-dimensional cache space — a unified geometric story covering both reward-driven (Track F) and monitor-driven (Bailey 2024) failure modes.
>
> I understand **why each section exists**, **why §5.5 needed rewriting** (OEI-as-primary contradicted §1's monitor-design-diagnostic framing; TDS-as-reward contradicted §3's entropy-as-surrogate framing), **why §A is required** (ARR reproducibility expectations), and **why track #13 was the right pick** (algorithm-as-interpretability-lens precedents like L\* and Finite State Automata; negative-result-as-mechanism-diagnosis precedents like CUTE; the 26-paper Deep Research sample maps to our shape).
>
> **What I don't deeply understand:**
>
> - The math in §4. Theorem 1's proof is measurable (0.00e+00 drift over 200 cycles validates empirically) but I haven't checked the FP32 accumulator construction in `logomesh/kv_mcts.py` line-by-line against the equations.
> - Whether **Dimensional Escape** is a rigorous geometric claim or theory layered on top of one negative result. Track F evidences H2 (reward-driven case); H1 (deception-driven case, monitor-led) is conjectured via Bailey-2024-analogy, not empirically demonstrated by us.
> - Whether **§5 reads coherently end-to-end**. Pieces work in isolation; connective tissue is unverified. The "§5 integration audit" plan-v2 line was the audit step; I skipped it as a Day-6 gap and then absorbed it implicitly into the §5.5 rewrite — but no narrative-flow read happened.
>
> **The honest meta-pattern:**
>
> What I've been doing Days 5-7 is **operational shipping** — applying pre-existing drafts to TeX, running smoke-compiles, committing, logging, syncing memory. The substantive research (Tracks A/C/D/F/G + the data + the empirical findings) was done before Day 5; I'm packaging that work into paper form. Each application required judgment (the §5.5 rewrite isn't pattern-matching — it's replacing contradiction with consistency), but it's **"informed shipping with judgment calls" rather than "deep first-principles validation."**
>
> Sonnet's review catches procedural drift in my work (figure mis-ranking, §A fidelity un-verified, scheduling sync gap). That's real signal. But **neither of us has been reading the actual algorithm code or re-deriving the proofs.** We're both operating one level above the underlying research.
>
> **The validation gap is real and explicitly deferred.** The parallel-review plan I drafted for Days 12-13 is designed for exactly this: Axis 1 reads `logomesh/kv_mcts.py` against §4 equations; Axis 2 stress-tests Dimensional Escape; Axis 5 does the standard "find the weakest claim" pass. Before Day 12, **the underlying research isn't being adversarially validated** — just shipped.
>
> **What you could do sooner if you want more confidence before Day 8 work:**
>
> Pull forward **Axis 5 only** (standard adversarial pass — Claude Opus session against the whole paper looking for overclaim/methodology gaps/missing baselines/weak framing). It's the most general axis and doesn't require post-ACL-switch state. ~1 hr wall, mostly waiting. Output: ranked list of vulnerabilities by severity. Gives you a substantive content-level read before §6 numbers land. Doesn't displace Days 12-13 parallel review — that's still the full audit.
>
> The alternative is: trust the current schedule. Days 5-7 are operational by design — there's no point validating a paper that isn't finished. Days 12-13 validate the complete version.
>
> **Sonnet's Day-8 sequencing call is right.** ACL switch first, smoke-compile clean, THEN §6 numbers. Don't tangle the debug. And include the §5.4 314pt Overfull in the Day-8 post-switch verification — narrow ACL columns might fix it via re-wrap or worsen it; better to know immediately.
>
> Bottom line: things make sense at the **structural level**. Substance level is unvalidated until Day 12. That's by design, but it is a real gap — be eyes-open about it.

### Josh's decision

Josh chose to keep marching forward (not pull Axis 5 forward to Day 8). Day 8 work proceeds per Sonnet's recommended sequence: ACL switch → smoke-compile → §5.4 Overfull verification → §6 numbers → smoke-compile → commit.

---

## Day 8 entry — ACL style switch + §6 worked-example numbers (Day-8 pull-forward, both plan-v2 §2 line 60+61 tasks)

**Date:** 2026-05-15 (Day 8 of plan v2 nominal calendar 2026-05-17; pulled forward by 2 days into the Day-6 calendar window)
**Owner:** Session B (TeX edits) with Session A diagnosis on the 2 compile failures
**Trigger:** Sonnet's Day-7 EOD hand-off named the Day-8 sequence: ACL switch first, smoke-compile clean, then §6 numbers. Both are plan-v2 §2 Day-8 Session B tasks.
**Status:** ✅ COMPLETE — both tasks landed in commit `e628256`, G-smoke-compile PASS, memory updates applied, this entry authored.
**Decision basis:** Plan-v2 §8 risk register flagged ACL switch as 25% P-failure (the highest-priority remaining gate). Two compile failures diagnosed and fixed in-session within the 30-min plasticity-rule window — neither required rollback.

### ACL style switch

**Edit (preamble):** Replaced `\usepackage[main]{neurips_2026}` with `\usepackage[review]{acl}`. Removed neurips_2026 preamble comment block. Updated `\usepackage[numbers,sort&compress]{natbib}` REVISION marker from `natbib-numbers-mode-D+` (Day 4, NeurIPS-specific framing) to `natbib-numbers-mode-cont-D+` (Day 8, corrected: workaround is manual-bibitem-vs-natbib-default, not NeurIPS-specific). Disabled microtype expansion via `\usepackage[expansion=false]{microtype}`. Two new REVISION markers: `acl-style-switch-D+` (the swap itself) and `microtype-expansion-disable-D+` (the font-expansion fix).

**Compile failure 1 (caught by fail-fast pdflatex):** Same fatal natbib error as Day 4 — "Bibliography not compatible with author-year citations." acl.sty (like neurips_2026.sty) loads natbib in author-year mode by default; manual `\bibitem{key}` bibliography requires numeric mode. **Diagnosis ~5 min:** the Day-4 marker's hypothesis ("acl.sty + acl_natbib.bst handles natbib natively") was wrong. **Fix:** re-add the Day-4 numeric-mode pre-load BEFORE `\usepackage{acl}`. Same workaround, new context.

**Compile failure 2 (after fix 1):** `pdfTeX error (font expansion): auto expansion is only possible with scalable fonts.` Triggered at `l.82 \end{enumerate}` during page 2 shipout. acl.sty loads default Computer Modern bitmap fonts; microtype's auto-expansion feature requires scalable Type 1/PostScript fonts. **Diagnosis ~3 min.** **Fix:** disable microtype expansion (keep protrusion + final-period kerning) via `\usepackage[expansion=false]{microtype}`. Under neurips_2026.sty this was a non-issue because that style loaded PostScript fonts internally.

**Compile success (third try, both fixes in place):** Exit 0, PDF produced.

### §6 worked-example numbers replacement

**Edit (TeX line 527):** Replaced placeholder sentence `With $b = 3$, $d = 5$, and $M_{\text{KV}} = 40\text{GB}$, standard MCTS requires ${\sim}9.7$TB, while reversible MCTS requires ${\sim}40.05$GB.` with Track-D-measured version using $M_{\text{KV}} \approx 162$ GB at 20B scale, $3.04 \cdot M_{\text{KV}} \approx 493$ GB reversible figure, and ~60× reduction at $(b,d)=(3,5)$.

**Subsumed REVISION marker:** `DRAFT-HOLD-block5-consolidated` (2026-05-05, gated on "Max's simplified Exp 3 VRAM measurements"). Track D's three-cell three-repeat sweep at 1B + 3B is the equivalent gate-passing measurement; 20B figure is a derived projection (per §A.4 Track D protocol). New marker: `§6-worked-numbers-trackD-D+`. The 2026-05-06 `prop-label-add` marker preserved (labels the proposition, not the numbers).

### G-smoke-compile gate (final, post-both-edits)

| Pass | Exit | PDF | Pages | Bytes |
|---|---|---|---|---|
| 1 | 0 | produced | 19 | 502,099 |
| 2 | 0 | produced | 19 | 501,737 |

**Page count went 22 (NeurIPS one-column) → 19 (ACL two-column).** ACL `[review]` mode confirmed as two-column per `acl.sty:112 \twocolumn`. The page reduction is column-density-driven; per-section content unchanged today.

### Warning inventory (broader grep, expected ACL-narrower-column overflows)

Significant Overfull \hbox warnings post-ACL-switch (these are visible PDF artifacts):

| TeX lines | Size | Source |
|---|---|---|
| 154 | 15.86pt | Abstract / §1 paragraph |
| 161 | 66.13pt | Abstract / §1 paragraph |
| 168 | 135.26pt | §1 Contributions list |
| 270 | 64.78pt | §3 Hypothesis paragraph |
| 338-351 | 149.24pt | §5 Exp 1 results discussion |
| 382-394 | 146.87pt | §5 Exp 2 paragraph |
| **448-458** | **552.75pt** | **§5.4 Baselines table — WORSE than pre-switch (314pt → 552pt); Sonnet's predicted-worsening confirmed** |
| 585-596 | 1.12pt | §7 paragraph (sub-resolution) |
| 760-761 | 154.87pt | §A.4 Track A `\texttt{}` paths |
| 769-770 | 163.79pt | §A.4 Track F `\texttt{}` paths |

Plus 30+ cosmetic Underfull \hbox / \vbox warnings (line-breaking and page-output artifacts; standard for narrow-column two-column rendering). No `Error:` lines. No undefined references.

**Gate verdict:** PASS at the compile level (exit 0, no errors, no undefined refs). The Overfull boxes are visible cosmetic issues to address Day 9; the worsening at §5.4 is the load-bearing one (552pt visibly spills beyond the column). Not blocking for Day-8 commit; flagged as priority Day-9 residue item.

### Git commit

`e628256` — `docs: Day 8 — ACL style switch + §6 worked-example numbers (Track D)`. Scope-minimal per `895ecca` pattern: 1 file changed (canonical TeX only), +12 / −12 lines (net 0; preamble grew by 1 line, §6 sentence shrank by 1 line after marker consolidation).

### TeX state

| Metric | Pre-Day-8 (post-`6e88af2`) | Post-Day-8 (post-`e628256`) | Delta |
|---|---|---|---|
| Total lines | 775 | **775** | 0 |
| Style package | `neurips_2026.sty` (NeurIPS one-column) | `acl.sty[review]` (ACL two-column + line numbers + anonymized) | ✅ switched |
| natbib config | Day-4 pre-load + neurips_2026 deferred load | Day-4 pre-load + acl.sty deferred load | unchanged (same workaround, new style host) |
| microtype | default (expansion enabled) | `expansion=false` | ✅ fix |
| §6 worked-example | 40 GB / 9.7 TB / 40.05 GB placeholders | 162 GB / 39 TB / 493 GB / 60× (Track D) | ✅ replaced |
| PDF pages | 22 (NeurIPS) | 19 (ACL two-column) | −3 |
| Compile clean | yes (5 cosmetic warnings) | yes (~40 cosmetic warnings, narrower-column overflows) | ✅ structurally clean |
| Body content gaps | §6 worked numbers + ACL switch + `\reviewedit{}` wrap | `\reviewedit{}` wrap only (Day-8 task 3 deferred — not yet applied) | 2/3 of Day-8 tasks landed |

### Plan v2 §2 Day-8 task status

| Task | Plan-v2 line | Status |
|---|---|---|
| ACL `acl.sty` style switch (25% P-failure gate) | 60 | ✅ APPLIED (`e628256`) |
| Track D §6 memory-complexity replacements (162 GB / 60× / (b,d)=(3,5)) | 61 | ✅ APPLIED (`e628256`) |
| Wrap visible `\reviewedit{}` content in `\iffalse...\fi` for review PDF | 62 | 🔲 DEFERRED — not blocking; can apply Day 9-10. The `\reviewedit{}` macro renders text in blue; for the anonymous review PDF those passages shouldn't be visually distinguished. |

### Day-9 residue list (updated)

| # | Item | Status |
|---|---|---|
| 1 | ACL conditional check that `acl_natbib` re-resolves `\citet{}` correctly under author-year mode | ✅ **CLOSED** by Day-8 ACL switch — we stay on numeric-mode natbib under acl.sty just as we did under neurips_2026.sty, so the Day-5 prose rewrites at lines 292+294 remain load-bearing and there's no follow-up needed |
| 2 | §5.4 Baselines table row 453 Overfull \hbox | **WORSE** — went 314pt → 552pt under ACL narrower columns. Description column ("One-shot static KV-cache steering (precomputed mean-of-differences vector, single post-prefill injection)") is the worst offender. Day-9 candidate: trim description or restructure table |
| 3 (NEW) | Several additional Overfull \hbox warnings introduced by ACL-narrower-column rendering (lines 154/161/168/270/338-351/382-394/585-596/760-761/769-770) — most cosmetic, but the table-region ones (338-351, 382-394, 448-458) are visible | Day-9 candidate: review all >100pt Overfulls and triage which to fix vs accept |
| 4 (NEW) | `\reviewedit{}` wrap in `\iffalse...\fi` (plan-v2 Day-8 line 62 deferred) | Day-9 or Day-10 work |
| 5 (NEW) | Plan-v2 §2 Day 9 nominal task is "end-to-end TeX read; residue list; page-count check vs J1 budget" — the page-count check is critical post-ACL-switch: 19pp total PDF includes body + bib + §A; need to confirm body is ≤8pp (Day-10 cuts pre-structured if over: §6 proof → `\appendix`, condense §5.4 Baselines, etc.) | On-plan Day 9 |

### Day 8 wall time

| Activity | Wall |
|---|---|
| Preamble read + acl_formatting.md skim | ~4 min |
| First ACL switch Edit + compile fail #1 + diagnosis | ~8 min |
| Natbib pre-load re-add Edit + compile fail #2 + diagnosis | ~6 min |
| microtype expansion-disable Edit + compile pass | ~3 min |
| Smoke-compile pass 2 + warning inventory + acl.sty grep for column-mode | ~6 min |
| §6 area read + Track-D numbers Edit | ~10 min |
| Final smoke-compile pass-1+pass-2 + warning inventory | ~5 min |
| Commit (long HEREDOC message) | ~3 min |
| Memory pointer updates (3 edits) | ~6 min |
| This session log append | ~12 min |
| **Day 8 total** | **~63 min** |

### Plan v2 §2 reconciliation — 10th pull-forward of the sprint

§6 numbers replacement was Day-8 nominal (line 61); applied Day-8 calendar 2026-05-15 (which IS Day 6 calendar, Day 8 plan-v2-nominal). ACL switch was Day-8 nominal (line 60); applied same. So Day-8 work landing on Day-6 calendar = 2-day pull-forward on each task, but both are "on-plan" relative to plan-v2 day numbering. The cumulative pull-forward count from the sprint:

10. **Day 8 (today): ACL style switch + §6 worked-example numbers from Day 8 plan-v2 (no pull-forward on plan-v2 day number; 2-day pull-forward on calendar).**

After today, plan-v2 §2's only remaining substantive Session-B work is the `\reviewedit{}` wrap (deferred). Plan-v2 Day 9 + Day 10 are mostly Session A audit/residue work + Responsible NLP final + supplementary anonymization (Day 12).

---

*End of Day 8 entry. Day-8 wall total: ~63 min. One commit landed (`e628256` ACL style switch + §6 worked-example numbers). Cumulative Days 5+6+7+8: ~386 min (~6.4 hr) across 7 work threads. SIX commits in git history. §5 + §7 + §A + §5.5 paper-body D+ alignment + ACL style + §6 worked numbers ALL COMPLETE. Remaining paper-body work: `\reviewedit{}` wrap + Day-9 residue cleanup + Day-10 Responsible NLP final. 10 pull-forwards across 5 calendar days.*

---

## Day 8 evening — §6 honest-framing refinement + parallel-review Pass K (substance-check follow-up)

**Trigger:** Josh asked me to explain the §6 numbers replacement in plain English. The explanation surfaced a borderline-overclaim in the new §6 wording: "Track D measured M_KV at 20B" implied a direct 20B measurement, but Track D measured at 1B and 3B; the 162 GB at 20B is an architectural derivation, and the 60× reduction at 20B is a projection. Josh chose Option A (fix the wording now) + asked for a Day-12 parallel-review scrutiny item to evaluate whether the 20B speculation is worth the body-page cost we're already over-budget on.

### §6 wording refinement (canonical TeX)

| Phrase | Before | After |
|---|---|---|
| 20B M_KV | "where Track~D measured $M_{KV} \approx 162$~GB per cache snapshot" | "where the open-weight `gpt-oss-20b` architectural specifications imply $M_{KV} \approx 162$~GB per cache snapshot (Track~D side-finding)" |
| Reversible figure | "requires approximately $3.04 \cdot M_{KV} \approx 493$~GB (using Track~D's measured constant-factor steady-state delta)" | "is projected to require approximately $3.04 \cdot M_{KV} \approx 493$~GB" + footnote naming the 1B/3B measurement scope + scale-invariance-by-construction defense |
| 60× claim | "a ${\sim}60\times$ reduction... independent of $b$ by construction" | "The implied ${\sim}60\times$ reduction... is therefore a projection from smaller-scale measurements rather than a direct 20B observation." |

**New REVISION marker:** `§6-honest-framing-refinement-D+ | 2026-05-15`. Page-budget cost: ~1 line for footnote anchor + ~3 lines for footnote body in ACL two-column. The refinement is small but the honesty-vs-overclaim distinction matters for the Day-12 adversarial review.

### Parallel-review plan Pass K added to Axis 5

New pass dedicated to **speculation vs page-budget tradeoff**. Specifically scrutinizes:
- (a) §6 20B-scale numbers (projections, not direct measurements; ~4-line page cost including new footnote)
- (b) §7.5 Future Work anti-Goodhart reward designs list (Phase B research program; not measured)
- (c) §A.5 supplementary URL placeholder (cosmetic)
- (d) Cross-scale Δ values in Cartography prose (measured; KEEP)

For each, the Day-12 reviewer outputs KEEP / TRIM / CUT with page-budget delta. Decision criterion: would an Interpretability-track reviewer find speculation essential, or accept its absence in exchange for tighter mechanism-diagnosis delivery? The body is currently over the 8-page review budget; speculative content must earn its line cost.

Axis 5 now has 11 passes (A-K), up from 10 (A-J).

### G-smoke-compile

Exit 0 on both passes. PDF 19 pp / 508 KB (was 19 pp / 502 KB pre-refinement; +6 KB for the footnote). Zero errors, zero undefined refs.

### Git commit

`76d63f1` — `docs: Day 8 evening — §6 honest-framing refinement + parallel-review Pass K (speculation vs page-budget)`. 2 files changed: canonical TeX (§6 refinement) + parallel-review-plan.md (Pass K). +50 / -2 lines.

### TeX state post-refinement

| Metric | Pre-refinement (post-`e628256`) | Post-refinement (post-`76d63f1`) | Delta |
|---|---|---|---|
| Total lines | 775 | **776** | +1 (REVISION marker + footnote net) |
| §6 worked-example framing | "Track D measured" (implied direct 20B measurement) | "projected"/"architectural derivation" + footnote naming scope | ✅ honest |
| Day-12 Pass K | absent | added to parallel-review-plan Axis 5 | ✅ new |
| Compile clean | yes | yes (same warning inventory; 1 footnote added) | stable |

### Day-8-evening wall time

~25 min (the explanation conversation + Edit + Pass K addition + smoke-compile + commit + memory + this addendum).

---

*End of Day 8 (combined ACL switch + §6 numbers + §6 honest-framing refinement). Day-8 grand total: ~88 min. **Seven** commits in git history now (`19f9eb5`, `895ecca`, `f61f391`, `694a72b`, `6e88af2`, `e628256`, `76d63f1`). Cumulative Days 5+6+7+8: ~411 min (~6.85 hr) across 8 work threads. Paper-body content essentially complete (§5 + §7 + §A + §5.5 + §6 numbers + ACL style + honest-framing refinement). Remaining: `\reviewedit{}` wrap (Day-8 line 62, deferred), Day-9 page-count check + residue cleanup + ~5 Overfull \hbox triage, Day-10 Responsible NLP Checklist final + §A application page-count post-check, Day-11 co-author send-out, Days 12-13 parallel adversarial review (now 11 passes in Axis 5 including the new Pass K), Day-14 final pre-submission, Day-15 ARR portal submit. Ready for Day-9 direction at Josh's call.*

---

## Day 9 entry — Page-budget cut plan audit + Josh sign-off (no TeX changes per R5)

**Date:** 2026-05-15 (Day 9 of plan v2 nominal calendar 2026-05-18; pulled forward 3 days into Day-6 calendar)
**Owner:** Session A (Opus 4.7; single-task fresh-context session per session-prompt design)
**Trigger:** Day-8 EOD discovered body rendering at ~16 ACL two-column pages against an 8-page review budget; plan v2 §8's pre-planned single cut (§6 Memory Complexity proof → \appendix) saves ~1 page; we need ~8.
**Status:** ✅ COMPLETE — cut plan v1 authored, Josh sign-off received (ALL 23 CUTS APPROVED + DROP CONTRIBUTION 2 per §8 option (a)), parallel-review plan additions applied, memory pointers updated. Canonical TeX untouched (R5 honored).
**Decision basis:** Day-9 session prompt at `docs/logs/2026-05-15_day-9-page-budget-cut-plan-session-prompt.md` scoped this session to single-task plan-and-stop. Josh sign-off received in-session; Day-10 application is a fresh session per the design. Josh added a peer-review-lens directive at Phase 2 kickoff: conduct the audit as if a double-blind reviewer reading the paper cold, and use what surfaces to strengthen the Day-12 parallel-review plan.

### Phase 1 reading + Phase 4 trip-wire pass

Required-reading checklist completed: CLAUDE.md, MEMORY.md, recast_action_plan, feedback (5 protocol rules R1-R5), project_context, 14-day execution plan, this session log (Days 6/7/8 entries), parallel-review plan, stale-sentence audit Summary, acl_formatting.md "Paper Length" section, canonical TeX (full 776 lines), .aux ToC entries via grep. Phase 4 trip-wire #1 confirmed PASS: canonical TeX at commit `76d63f1` (Day-8 evening state); no new commits to surprise me.

### Phase 3 audit work

**Step A — Page-distribution table** built from .aux `\@writefile{toc}` grep + .log Output line. Body (§1 through §7.6) spans pp. 1–17 ≈ **16 body pages**; §A pp. 17–19 ≈ 2.5 appendix pages. Target ≤ 8 body pages. **Cut delta needed: ~8 pages.**

**Step B — Section-by-section cut-candidate inventory** conducted from neutral peer-review perspective per Josh directive. KEEP/TRIM/MOVE/DELETE per subsection with framing-impact notes. Peer-review observations beyond the pre-flagged candidates surfaced:

- **M_KV inconsistency:** §4.2.1 line 242 says "M_KV ≈ 40 GB" at 20B; §5 Impl Notes line 299 says "≈ 80 GB before any accumulator overhead"; §6.2 line 528 says "≈ 162 GB per cache snapshot". Three different stories for the same physical quantity, reconciled only at §5 Impl Notes item (iv). A reviewer reading body sequentially encounters conflicting numbers before reaching the reconciliation paragraph. None of the existing 11 Axis-5 sub-passes would catch this.
- **Unification overclaim:** §1 Contribution 3 "Empirical demonstration of Dimensional Escape" (singular) implies BOTH H1 and H2 are demonstrated. Body evidences only H2 via Track F + Track G; H1 is conjectured via Bailey 2024 analogy in a different experimental regime (gradient-trained input-space attacks vs our inference-time KV-cache mutation).
- **Contributions coherence tension:** §1.1 packaging reads as method-paper (FP32 accumulator, memory bound, 60× reduction), but abstract closing reframes as "high-precision diagnostic probe... rather than reasoning-improvement tool." A reviewer may form a method-paper prior from §1 and encounter confusion at §5 Exp 1's null result.
- **Operational scaffolding:** §5.1 + §5.2 Phase 0/A subsections + §5 Implementation Notes 5-item list + §7.4 Empirical Measurement Prerequisites all read as project-management or development-history rather than scientific content.
- **Abstract-to-body asymmetric hedging:** Day-8-evening §6 honest-framing refinement hedged the 20B projection in body; abstract still reads "$\sim$60$\times$ reduction... on 20B-parameter configurations" without the hedge.

**Step C — Ranked cut plan** built with risk-adjusted-savings formula (LOW=0.0, MEDIUM=0.3, HIGH=0.7). 23 cuts identified spanning §1 through §7. Recommended sign-off threshold Cuts #1–#20 (~6.67 pp cumulative savings; body lands ~9.3 pp; CRITICAL flag raised — gap to 8-pp target requires Cuts #21–#23 HIGH-risk per R2 or §8 escalation option (a) drop Contribution 2).

**Step D — Sign-off ask** written with 3 explicit options for Josh.

**Step E — Plan file** authored at `docs/logs/2026-05-15_page-budget-cut-plan-v1.md` (~3500 words, 8 sections incl. §7 peer-review observations and §8 CRITICAL escalation flag).

**Step F — Stop + status report** issued under 150 words.

### Josh sign-off (received in this same session)

Josh approved:
1. **All 23 cuts** (Y + extend to #21-#23 per cut plan §4 sign-off ask).
2. **Drop Contribution 2 (telemetry matrix)** per §8 escalation option (a). Rationale: gives ~0.5 pp margin against 8-pp budget; track #13 alignment preserved (algorithm-as-interpretability-lens + negative-result-as-mechanism-diagnosis pillars survive; C2 was weakest-evidenced pillar — alpha-sweep on 1B only, no discriminative validation).

### Day-10 application scope (locked, not applied this session per R5)

**Combined application:** 23 cuts (~7 pp savings) + drop-C2 (~1.5 pp savings) = **~8.5 pp total**, body 16 → ~7.5 pp, ~0.5 pp margin.

**5 batches** with G-smoke-compile gate after each:
- Batch 1: §2 restructuring (§2.3 MOVE inline §4.1; §2.4/§2.5 MOVE to §A; §2.6 DELETE) + §6 proof MOVE
- Batch 2: §5 cleanup (§5.1+§5.2 DELETE; §5 Impl Notes MOVE to §A with M_KV reconciliation footnote at §6.2; §5.4 baselines table MOVE; §5.5 TDS bullet trim)
- Batch 3: §7 trimming (§7.5 4→1 paragraph; §7.3 8→5 items; §7.1 4→2 paragraphs; §6.2 footnote DELETE) + §7.4 MOVE entire to §A
- Batch 4: §1 (¶2 + ¶3 trim) + §3 (§3.1 trim; §3.3 MOVE) + §4.2 algorithm details MOVE
- **Batch 5 (NEW for drop-C2):** §4.1 entire MOVE-or-DELETE + §1.1 C2 paragraph delete + §4.2 Step 2 telemetry reference rewrite + §7.1 Monitor design paragraph rewrite + §7.3 Limitation 4 σ_H/ρ_R reference update + abstract verification

Estimated Day-10 wall time: 4–5 hours.

### Parallel-review plan additions (bundled side-deliverable, applied this session)

Derived from peer-review observations during the audit. Applied to `docs/logs/2026-05-14_parallel-review-plan.md`:

- **NEW Pass L (Axis 5):** Cross-section numerical-consistency check (M_KV inconsistency case study + 3 other quantitative-claim cross-section checks).
- **NEW Pass M (Axis 5):** Abstract-to-body claim escalation check.
- **NEW Pass N (Axis 5):** Contributions coherence stress test.
- **NEW Pass O (Axis 5):** Operational-scaffolding triage.
- **Strengthened Axis 2 Q3:** Unification claim (H1+H2) requires hedge or rewrite.
- **Strengthened Axis 2 Q5:** Goodhart-distinction defense or retire the Dimensional Escape term.

Axis 5 sub-passes: A-K → A-O (11 → 15). Length target updated 3000–4500 → 4500–6500 words.

### G-smoke-compile gate

N/A — no canonical TeX changes this session per R5 NO APPLICATION rule.

### TeX state

| Metric | Pre-Day-9 (post-`76d63f1`) | Post-Day-9 | Delta |
|---|---|---|---|
| Total lines | 776 | **776** | 0 (R5 honored) |
| Day-10 application scope | 23 cuts pending sign-off | 23 cuts + drop-C2 LOCKED per Josh sign-off | ✅ all decisions resolved |
| PDF pages | 19 | 19 | 0 |

### Plan v2 §2 reconciliation

Plan v2 §2 Day-9 nominal task: "end-to-end TeX read; residue list; page-count check vs J1 budget" (Session A). Today's audit covered all three: end-to-end TeX read (all 776 lines), residue list updated (4 items remain — all Day-10 work), page-count check (CRITICAL flag raised; resolved via cut plan + drop-C2 sign-off).

Day-10 scope expanded substantially: was "apply residue removals" (Session B) + "§A Reproducibility Appendix v1 draft" (Session A) — §A already applied Day 6, so Day-10 is now full cut-plan application (5 batches) + drop-C2 reframe + Responsible NLP Checklist final (still on-plan).

### Day-9 residue list status

- Item 2 (§5.4 Baselines Overfull \hbox, WORSE post-ACL-switch): **RESOLVED via Cut #4** in cut plan (MOVE §5.4 table to §A).
- Item 3 (ACL narrower-column overflows): **RESOLVED via Cuts #3 + #4 + #11 + #12**.
- Item 4 (\reviewedit{} wrap deferred from Day-8 line 62): **REMAINS** — Day-10 application includes this as a non-cut residue item.
- Item 5 (page-count check vs J1 budget): **RESOLVED today** via cut plan §1 + Josh sign-off.

### Day-9 wall time

| Activity | Wall |
|---|---|
| Phase 1 reading (12 artifacts incl. full TeX) | ~35 min |
| Phase 1 readiness report + Josh go-ahead | ~3 min |
| Phase 3 Step A page-distribution table | ~5 min |
| Phase 3 Step B section-by-section audit (peer-review lens) | ~25 min |
| Cross-section numerical-consistency grep (M_KV finding) | ~5 min |
| Phase 3 Step C ranked cut plan + risk-adjusted weighting | ~15 min |
| Phase 3 Step E plan file authoring (~3500 words, 8 sections) | ~25 min |
| Phase 3 Step F status report + Josh sign-off intake | ~5 min |
| Parallel-review plan additions (4 new Passes + 2 Q-strengthenings) | ~15 min |
| Drop-C2 decision intake + Day-10 scope expansion analysis | ~10 min |
| Memory updates (3 edits: recast_action_plan ×2 + MEMORY.md ×1) | ~10 min |
| This session log append + git commit prep | ~20 min |
| **Day 9 total** | **~173 min (~2.9 hr)** |

### Day 9 deliverables summary

- 1 new file: `docs/logs/2026-05-15_page-budget-cut-plan-v1.md` (~3500 words; 23 ranked cuts; CRITICAL flag; §7 peer-review observations)
- 1 file modified: `docs/logs/2026-05-14_parallel-review-plan.md` (4 new Passes L/M/N/O + Q3/Q5 strengthenings)
- 1 session log appended: this file
- 1 session prompt staged: `docs/logs/2026-05-15_day-9-page-budget-cut-plan-session-prompt.md` (Josh's authoring prompt, included for audit trail)
- 2 memory files updated: `recast_action_plan.md` + `MEMORY.md` (Josh sign-off + drop-C2 locked)
- 0 canonical TeX changes (R5 honored)
- 1 Josh decision captured: ALL 23 CUTS APPROVED + DROP CONTRIBUTION 2 (option (a))

---

## Open items for Day 10 (fresh session, 2026-05-16 nominal calendar or earlier if pulled forward)

Day-10 scope per Josh sign-off:
- Apply 23 cuts in 4 batches with G-smoke-compile gates per cut plan §6
- Apply drop-C2 in NEW Batch 5 (§4.1 MOVE-or-DELETE + §1.1 C2 paragraph delete + §4.2 Step 2 + §7.1 Monitor design + §7.3 Limitation 4 + abstract reference updates)
- Add M_KV reconciliation footnote at §6.2 when §5 Impl Notes moves to §A
- Verify cross-references resolve in compiled PDF (R4)
- Update §A.X organization paragraph if §A grows substantially after absorbing moved content
- Apply \reviewedit{} wrap in \iffalse...\fi (Day-8 deferred residue item)
- Final smoke-compile + page-count verification (target ≤ 8 pp body)
- Plan-v2 §2 Day-10 nominal also includes: Finalize Responsible NLP Checklist (Session A; `docs/logs/2026-05-19_responsible-nlp-checklist-final.md`) — bundle with the cut application

Estimated Day-10 wall time: 4–5 hours for cut application + 1 hour for Responsible NLP Checklist = ~5–6 hours total.

**TeX state for Day-10 baseline:** 776 lines, D+-aligned through §7 + §A + §5.5 + §6 numbers + ACL style + honest-framing refinement. Day-9 audit-locked: 23 cuts + drop-C2 (~8.5 pp savings target). Day-10 fresh session reads `recast_action_plan.md` + `2026-05-15_page-budget-cut-plan-v1.md` to pick up cleanly.

---

*End of Day 9 entry. Day-9 wall total: ~173 min (~2.9 hr). Zero canonical TeX commits today (R5 honored). One docs+logs commit landing for the cut plan v1 + parallel-review plan additions + this session log + the Day-9 session prompt itself. Cumulative Days 5+6+7+8+9: ~584 min (~9.7 hr) across 9 work threads. Seven prior TeX commits in git history; Day-9 adds the 8th (docs-only). Paper-body content essentially complete pending Day-10 cut + drop-C2 application. Ready for Day-10 fresh session at Josh's call.*
