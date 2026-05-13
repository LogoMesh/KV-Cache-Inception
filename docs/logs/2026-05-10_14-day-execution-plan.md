# ARR-Cycle Execution Plan — Option D+ with Latent Cartography Phase 2

**File version:** v2 (restructured 2026-05-10 PM)
**v1 → v2 trigger:** Day-1 verification pass discovered (a) 2026-05-25 is the ARR May submission deadline, NOT the EMNLP commitment deadline (EMNLP commit is 2026-08-02; conference 2026-10-24 to 10-29 in Budapest); (b) BLOCKING identifying-information leaks in tracked files; (c) three required deliverables missing from v1 (Responsible NLP Checklist, Llama license attribution, §A Reproducibility Appendix). Full rationale in plan file `C:\Users\Josh\.claude\plans\misty-scribbling-fiddle.md`.

**v1 reference:** the original Days 1–15 day-by-day matrix is preserved in §10 as the v1 appendix, marked superseded. Plasticity rules from v1 §2 are preserved here in §3 with v2 additions; v1 §3 "preserved across the pivot" content is preserved here in §4; v1 §4 "DROPPED items" is preserved here in §5; v1 risk register is updated in §8.

**Submission deadline (ARR cycle):** 2026-05-25
**EMNLP commitment deadline:** 2026-08-02
**Camera-ready deadline (if accepted):** 2026-09-20
**Conference:** 2026-10-24 to 10-29, Budapest, Hungary

---

## 0. Strategic commit (carried forward from v1)

Team committed 2026-05-10 to **single-track Option D+ with Latent Cartography (Design 1) as Phase 2 empirical expansion**. Reasoning in `docs/logs/2026-05-11_pivot-decision-memo.md` + `docs/logs/2026-05-09_Latent-Space MCTS_ Reward Design Strategy.md` (Gemini round 4). Two-track was considered and rejected in favor of single-track focus + richer empirical content.

**Paper has two empirical pillars:**
1. **Track F (already complete)** — controlled negative-control evidence of Goodhart pathology in entropy-MCTS at 1B and 3B. Demonstrates that the failure manifests by *different routes* at the two scales, mechanism-level.
2. **Latent Cartography sweeps (to be run, Days 4-5)** — diverse prompt classes × MCTS pressure × 1B/3B. Maps the structural pre-training priors revealed under continuous-latent search. Positive empirical mapping.

**Unified framing:** "Dimensional Escape" — generalizes the original orthogonal escape hypothesis to cover both monitor-driven and reward-driven manifestations of the same geometric pathology. Per Gemini Part 5 abstract sketch.

---

## 1. Phase structure

| Phase | Window | Scope |
|---|---|---|
| **Phase 1** | 2026-05-10 → 2026-05-25 (T-15 → T-0) | ARR May submission. Hard deadline. Restructured day-by-day matrix in §2 below. |
| **Phase 2** | 2026-05-26 → 2026-08-02 (T+1 → T+69) | Review period + author response + EMNLP commitment. Deferred items: Croissant 1.1 hosting, generalization sweeps to 11B or other model family, full reproducibility-appendix expansion, anti-Goodhart Phase B research. See §7. |
| **Phase 3** | 2026-08-20 → 2026-09-20 (conditional on acceptance) | Camera-ready window. De-anonymization, final polish, public Croissant release. See §7. |

---

## 2. Phase 1 day-by-day matrix

Owner key: **A** = Session A (strategic), **B** = Session B (tactical TeX), **D6** = Diagnostic session #6, **J** = Josh.

| Day | Date | Owner | Task | Output |
|---|---|---|---|---|
| 1 | 05-10 | A | ✅ Done v9 abstract / contributions / §3 / Cartography protocol | 4 files in `docs/logs/` |
| 1 | 05-10 | A | ✅ Done v9 §7.3 Limitations Option D+ rewrite | `docs/logs/2026-05-10_session-A-prose-drafts-v9-limitations.md` |
| 1 | 05-10 | A | ✅ Done Title-lock memo (drop alignment-faking framing) | `docs/logs/2026-05-10_session-A-prose-drafts-v9-title.md` |
| 1 | 05-10 | A | ✅ Done v2 restructure of this plan file | this file |
| 1 | 05-10 | J | **Josh kickoff block** (verifications J1, J2, J3, J7 — see §6) | append to `docs/logs/2026-05-10_session-log.md` |
| 2 | 05-11 | B | Apply v9 abstract / §1 / §3 / §7.3 / title to canonical TeX | edits in `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` |
| 2 | 05-11 | A | v9 §7 Discussion (Dimensional Escape implications, Future Work splits into monitor-design + reward-design) | `docs/logs/2026-05-11_session-A-prose-drafts-v9-discussion.md` |
| 2 | 05-11 | A | License-attribution paragraph + Responsible NLP Checklist Q&A draft | `docs/logs/2026-05-11_session-A-prose-drafts-v9-license-attribution.md` + `docs/logs/2026-05-11_session-A-responsible-nlp-checklist-draft.md` |
| 3 | 05-12 | B | ✅ Done Apply v9 §7 Discussion to TeX (§7.1 + §7.5; lines 522-584) | TeX edits applied 2026-05-12 |
| 3 | 05-12 | A | §5 Experiment 2 prose drafting (PULLED FORWARD from Day 6; Cartography MCTS+sweep complete Day 2) | `docs/logs/2026-05-12_session-A-prose-drafts-v10-exp2.md` |
| 3 | 05-12 | A | Day-3 cross-ref audit + anonymization-scope verification (see §5.1 below) | running-log append |
| 4 | 05-13 | A | §A Reproducibility Appendix v1 (PULLED FORWARD from Day 9) | `docs/logs/2026-05-13_session-A-reproducibility-appendix-v1.md` |
| 5 | 05-14 | A | §5 Experiment 2 polish + Table 2 + Figure 1/2 specs from Track G report §5 anchors | drafts file |
| 5 | 05-14 | A | Begin §5 Experiment 2 draft as data lands | `docs/logs/2026-05-14_session-A-prose-drafts-v10-exp2.md` |
| 6 | 05-15 | A | Complete §5 Exp 2 draft + Table 2 + Figure 1 | drafts file |
| 7 | 05-16 | B | Apply §5 Exp 2 to TeX (marker `option-D+-exp2-cartography-2026-05-16`) | TeX edits |
| 7 | 05-16 | A | §5 integration audit | running log |
| **8** | 05-17 | B | **EMNLP style switch** (moved up from v1 Day 10) — swap `\usepackage[main]{neurips_2026}` → ACL `acl.sty`; fail-fast smoke compile | TeX compiles under ACL style |
| 8 | 05-17 | B | Apply Track D §6 memory-complexity replacements (162 GB / 60× / `(b,d)=(3,5)`) | TeX edits |
| 8 | 05-17 | B | Wrap visible `\reviewedit{}` content in `\iffalse...\fi` for review PDF | TeX edits |
| 9 | 05-18 | A | End-to-end TeX read; residue list; page-count check vs J1 budget | residue list |
| 9 | 05-18 | A | **Day-9 residue list pre-loaded items (filed 2026-05-12 from Day-4 P1 smoke-compile; confirmed 2026-05-13)**: (i) `natbib Warning: Author undefined for citation 'zou2023repe'` (×2 in pass-2 log) is NOT log noise — grep `\citet{zou2023repe}` confirmed 2 textual-citation usages at canonical TeX lines 292 and 294 (§5 Implementation Notes items i and ii). Under numbers-mode natbib, these render as bare numeric `[N]` instead of "Zou et al.\ (2023)", producing awkward prose ("LAT procedure of [N]" and "[N] validate..."). **FIX:** rewrite lines 292 and 294 to follow line 214's pattern ("Zou et al.\ (2023)~\cite{zou2023repe}") with author surname in prose and `\cite{}` for the bracketed reference. Two-line surgery. (ii) Day 8 ACL switch will reload natbib via `acl.sty` + `acl_natbib.bst` — verify the `\citet` rendering re-resolves under acl_natbib.bst's `\bibitem` format expectations and decide whether the line-292/294 prose rewrites are still needed post-switch. | residue list addendum |
| 9 | 05-18 | B | Apply residue removals; integrate Track B-delta citations (Snell / Hasani / Stoehr / MUSE) | TeX + bibtex |
| 9 | 05-18 | A | **§A Reproducibility Appendix v1 draft** (consolidates Tracks A/C/D/F + MCTSConfig) | `docs/logs/2026-05-18_session-A-reproducibility-appendix-v1.md` |
| 10 | 05-19 | B | Apply §A Reproducibility Appendix as `\appendix \section{Reproducibility}`; insert license-attribution paragraph in §Ethical Considerations | TeX edits |
| 10 | 05-19 | B | Page-count check post-appendix; cut decisions if over budget | page-count log |
| 10 | 05-19 | A | **Finalize Responsible NLP Checklist answers** (bound to actual paper §-refs) | `docs/logs/2026-05-19_responsible-nlp-checklist-final.md` |
| 11 | 05-20 | A | Final REVISION-marker audit (`grep "pending audit"` → resolve all) | running log |
| 11 | 05-20 | J | **Send paper to Tianyu Shi + co-authors** for sign-off (moved up from v1 Day 12) — 4-day async window | external |
| 12 | 05-21 | B | **Build anonymized supplementary package** (curated subset; per §5.1 below) — copy `logomesh/`, `scripts/`, `tests/`, `pyproject.toml`, anonymized `README.md` to staging dir; apply URL/identifier anonymization in the COPY only; do NOT edit live repo. Stage to anonymous.4open.science per J5. | `tmp/supplementary-staged/` + `docs/logs/2026-05-21_supplementary-build.md` |
| 12 | 05-21 | A | **Anonymization verification grep** on the staged supplementary directory (per §9 G-anon-B). Verify compiled PDF anonymization via `pdftotext` over the review PDF. | `docs/logs/2026-05-21_anonymization-verification.md` |
| 12 | 05-21 | B | **Croissant 1.1 smoke-pack** (generated from anonymized supplementary copy) — regenerate JSON-LD; validate with `mlcommons/croissant` CLI. | `docs/logs/2026-05-21_croissant-smoke.md` |
| 13 | 05-22 | A | Triage Tianyu + co-author feedback → must-act vs nice-to-have | triage log |
| 13 | 05-22 | B | Apply must-act revisions | TeX |
| 14 | 05-23 | B | Final review-version compile; `\iffalse`-wrap verification (REVISION markers absent from PDF text) | submission PDF |
| 14 | 05-23 | J | Verify ARR/OpenReview account; confirm EMNLP binding selection; final double-blind check | submission readiness |
| 15 | 05-24 | J | **Submit to ARR May 2026 cycle several hours before 2026-05-25 deadline** | submission receipt at `docs/logs/2026-05-24_arr-submission-receipt.md` |

---

## 3. Plasticity rules

Inherits the v1 plasticity discipline with v2 additions.

| Trigger | Action |
|---|---|
| Latent Cartography sweeps fail to produce coherent structural finding by Day 7 EOD | YELLOW — Session A drafts a "Cartography minimal contribution" version (1-2 prompt classes); if Day 9 still no signal → NO-GO, Cartography demotes to Limitations item, paper ships as pure Option D |
| EMNLP style switch (Day 8) reveals page count irreducibly > 9 pages | YELLOW — additional cuts needed; §6 Memory Complexity full proof moves to Appendix A |
| Tianyu signs off late or surfaces a structural concern past Day 13 | YELLOW — Josh decides incorporate now vs. log for Phase 2 author response window |
| Track-F-equivalent failure surfaces in Cartography (no class-dependent variation) | NO-GO on Cartography Experiment 2 — drop and ship pure Option D paper; probability drops to 25-35% Session A baseline |
| Citation pile-up new fatal finding | NO-GO on submission; pivot to workshop/journal (Option C) with same content |
| **NEW v2 — Day 8 style switch fails (TeX constructs incompatible with `acl.sty`)** | YELLOW — Day 8 PM compile-error triage; if not resolved by Day 9 morning, restructure to compatible idioms (likely 1-2 packages need replacement) |
| **NEW v2 — Anonymization grep Pass C finds new leaks beyond Pass A's known list** | YELLOW — Day 13 slack absorbs known-pattern fixes; if Day 13 EOD still has BLOCKING leaks, escalate to Day 14 emergency cleanup; G-double-blind gate blocks submission until clear |
| **NEW v2 — Responsible NLP Checklist forces a "No" answer that risks desk rejection** | YELLOW — Day 2 surfaces issue → Day 9 §-level mitigation (e.g., new sentence in Limitations addressing the bar); if mitigation not possible, log as known weakness in author-response Phase 2 |
| **NEW v2 — Page-budget verification (J1) returns Findings limit < 8 pages** | YELLOW — mandatory deeper cuts; §6 proof → Appendix A; verbose §4.2.2 condensed; consider moving Reproducibility Appendix to supplementary materials rather than `\appendix` |
| **NEW v2 (2026-05-12) — G-smoke-compile fails after a TeX-edit day's changes** | YELLOW — the failure is diagnosed within 30 min; if not resolved in that window, the day's edits are rolled back. Failed edits do not promote to the next day's baseline state until compile is clean. The plasticity rule prevents compile errors from compounding across consecutive edit days. |

---

## 4. What's preserved across the pivot (don't re-validate)

- **Algorithm + memory bound + Theorem 1**: empirically validated in Tracks A/D. No re-run needed.
- **Static signal validations**: σ_H (post-merge B6 verification 14-15/17 layers), ρ_R (bit-identical pre/post merge), entropy (Track A |r| ≈ 0.60 both scales) — all valid as monitoring outputs.
- **Citation pile-up**: 25 citations CLEAN per Tracks B + B-delta. No re-audit.
- **Track D VRAM measurements**: 3.04–3.06× M_KV measured. Used directly in §6 rewrite.
- **Track F empirical content**: 200 items × 3 conditions × 2 scales = 1200 records. The §5 Experiment 1 dataset.

## 5. What gets DROPPED from the prior framing

- "System-2 reasoning" headline language. The paper is no longer a reasoning-improvement paper.
- "Test-Time Scaling in Continuous Latent Space" title direction. Replaced by v9 title (`docs/logs/2026-05-10_session-A-prose-drafts-v9-title.md`).
- Any claim that entropy-guided MCTS produces accuracy gains. Falsified by Track F.
- Bailey 2024 "complementary measurement" peer framing — repositioned as monitor-driven sub-case of unified Dimensional Escape rather than headline peer.
- The four candidate framings in `2026-05-08_EMNLP Framing for KV-Cache MCTS.md` Part 2 — all retired by Option D+ commitment. Latent Cartography (Part 3 Design 1) is the only Part-3 idea retained.
- **NEW v2 dropped**: "Day 10 style switch" — pulled forward to Day 8 for fail-fast.
- **NEW v2 dropped**: hard-deadline framing — replaced with ARR-cycle + EMNLP commitment + camera-ready phased framing.
- **NEW v2 dropped 2026-05-12**: Day-3 Anonymization Pass A (4 in-place edits to live repo) and Day-4 Anonymization Pass B (`.gitignore` + `git rm --cached`). Primary-source verification 2026-05-12 against [EMNLP 2026 main call](https://2026.emnlp.org/calls/main_conference_papers/) + [ARR CFP](https://aclrollingreview.org/cfp) + [ARR anonymity policy](https://aclrollingreview.org/anonymity) established that the main GitHub repo does NOT need to be anonymized under the post-Feb-2024 ACL policy (non-anonymous preprints are explicitly allowed during peer review). Anonymization applies only to (a) the submitted PDF and (b) the supplementary materials package uploaded at submission. See §5.1 below for the corrected scope.

---

## 5.1 Anonymization scope — primary-source corrected (2026-05-12)

**The plan v2 (2026-05-10) treated anonymization as a repo-cleaning exercise. Primary-source verification on 2026-05-12 corrected this scope.**

### What the actual EMNLP / ARR rules require (verbatim from primary sources)

| Surface | Anonymization required? | Source |
|---|---|---|
| **Submitted paper PDF** — `\author{}`, acknowledgements, body text, in-text URLs | YES | ARR CFP: "Papers must exclude author names and affiliations. Self-references revealing identity are prohibited." |
| **Bibliography in the paper** (for our own prior works) | YES — third-person | ARR CFP: write "Smith previously showed (Smith, 1991)" not "We previously showed" |
| **Supplementary materials package** uploaded with the paper | YES | ARR CFP: "Supplementary materials require anonymization as well." |
| **Code repository link in supplementary** | YES — anonymous host | ARR CFP: "Software repositories must be properly anonymized (e.g., Anonymous GitHub)"; "Links to file hosting services that can track downloads, such as Dropbox, are not allowed" |
| **Main GitHub repo** (`joshhickson/LogoMesh-NeurIPS`, public) | **NO** | Post-Feb-2024 ACL Anonymity Policy: "Authors are free to post and discuss non-anonymous preprints at any time" |
| **arXiv preprint** (if posted) | NO | Same |
| **Local files** (`CLAUDE.md`, `team-briefs`, `max-briefings`) on GitHub | **NO** | Not part of the submission package |
| **`.gitmodules` `external/Logomesh`** in main repo | **NO** | Same — only relevant if included in supplementary copy |
| **`logomesh/croissant_export.py:335`** GitHub URL in main repo | **NO** | Same — anonymize in supplementary copy at Day 12 build, not main repo |
| **`EXPERIMENTS.md` McGill mention** in main repo | **NO** | Same |

### Day-3 grep evidence (canonical TeX as of 2026-05-12)

Grep over `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` for `(?i)(logomesh|mcgill|joshhickson|hickson|tianyu|toscano|bakul|asuka|maxwell|linear\.app)` + `\bShi\b|\bMax\b` + `github\.com|@gmail|http[s]?://...` returned:

- **Renders in PDF (visible to reviewers):** ONLY `\author{[Authors]}` placeholder at line 42. ✅ Already anonymized at the rendered-PDF level.
- **Does not render in PDF (LaTeX `%` comments only):** mentions of "logomesh/", "Tianyu Shi", "Max's lane", "Max's simplified Exp 3" in REVISION markers and CRITIQUE notes. These do not appear in the compiled PDF.

**Conclusion:** The PDF that ARR reviewers see is already anonymized. The comments are only relevant if the raw `.tex` source is uploaded as supplementary material; standard practice is to upload only the compiled PDF + curated code/data package, not the raw `.tex`.

### Corrected Day 12 workflow

Replace the deleted Day-3 Pass A and Day-4 Pass B with a single Day-12 supplementary-package build:

1. Copy curated subset to `tmp/supplementary-staged/` (per the "Released-supplementary-materials policy" table further in this section — INCLUDE `logomesh/`, `scripts/`, `tests/`, `pyproject.toml`, anonymized `README.md`, `docs/dataset/` post-anonymization, Croissant smoke-pack JSON-LD; EXCLUDE `docs/NeurIPS/` paper drafts, `docs/logs/` session logs, `Claude-Chat-*.md`, `CLAUDE.md`, team briefs, `docs/reviews/`, `external/Logomesh/`).
2. Apply URL anonymization in the COPY only: `github.com/Logomesh/kv-cache-inception` → anonymous.4open.science URL (per J5); `McGill` → "institutional cluster" in copied `EXPERIMENTS.md`; `.gitmodules` reference dropped from copy.
3. Verify via `G-anon-B` grep gate on `tmp/supplementary-staged/` only.
4. Upload to anonymous.4open.science.

The live repo is untouched. De-anonymization at camera-ready (Phase 3) is then a no-op — the repo is already public under real identity.

### Hosting decision (still open)

J5 unchanged: anonymous.4open.science recommended. ARR CFP requires anonymous hosting with no download tracking.

---

## 6. Josh action items (URL-verified)

Each item below has a deadline; all must complete by Day 8 latest (style switch); J2 and J3 needed by Day 10.

| # | Action | URL to verify | Deadline | Records to | Blocking? |
|---|---|---|---|---|---|
| **J1** | ✅ Done 2026-05-11. Page budget per [ACL formatting.md](../NeurIPS/acl_formatting.md) + [EMNLP 2026 call](https://2026.emnlp.org/calls/main_conference_papers/): **REVIEW version: 8 pages of content for long papers + UNLIMITED references; review versions MUST NOT include acknowledgements**. Final version (post-acceptance): 9 pages + unlimited acks + unlimited references. Appendix is unlimited (`\appendix` is outside the page count). **Limitations and Ethical Considerations COUNT toward the 8-page review budget** (not excluded). Findings track not mentioned in EMNLP 2026 call — Findings is a venue commitment chosen during the EMNLP commit phase (Aug 2), not at ARR submission. | https://2026.emnlp.org/calls/main_conference_papers/ + acl_formatting.md | done | drives Day 9-10 cuts |
| **J2** | ✅ Done 2026-05-11. Llama 3.2 license attribution string verbatim captured. Three binding requirements: (a) "Built with Llama" prominently displayed; (b) notice text `Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.`; (c) Acceptable Use Policy URL https://llama.com/llama3_2/use-policy/. Draft at `docs/logs/2026-05-11_session-A-prose-drafts-v9-license-attribution.md`. | https://www.llama.com/llama3_2/license/ | done | unblocks §7.6 paragraph |
| **J3** | ✅ Done 2026-05-11. All 17 Responsible NLP Checklist questions captured verbatim. Section A (A1-A2 limitations + risks), B (B1-B6 artifacts), C (C1-C4 computational experiments), D (D1-D5 human annotators — all N/A for us), E (E1 AI assistants — YES, Claude + Gemini disclosed). First-pass Q&A draft at `docs/logs/2026-05-11_session-A-responsible-nlp-checklist-draft.md`. Final version Day 10. | https://aclrollingreview.org/responsibleNLPresearch/ | done | unblocks Day 10 finalization |
| **J4** | ✅ Done 2026-05-11. Downloaded to `docs/NeurIPS/`: `acl.sty` (11.6 KB), `acl_natbib.bst` (45 KB), `acl_formatting.md` (formatting reference). Note: only `acl.sty` + `acl_natbib.bst` are required; `acl.bst` (older) was not downloaded since the .sty file's `\bibliographystyle{acl_natbib}` points to the natbib version. The .sty file uses `\usepackage{acl}` with `[review]` option for double-blind submission. | https://github.com/acl-org/acl-style-files | done | unblocks Day 8 style switch |
| **J5** | ✅ Decided 2026-05-11: **anonymous.4open.science**. Rationale: ARR CFP names Anonymous GitHub as one valid example, but anonymous.4open.science is the established academic standard for ACL/EMNLP supplementary code uploads, supports unlimited repo size, doesn't track downloads, and offers the standard URL pattern `https://anonymous.4open.science/r/<repo-slug>` that is unambiguous in supplementary references. URL placeholder for Day-12 supplementary build: `https://anonymous.4open.science/r/kv-mcts-dimensional-escape-XXXX` (final slug assigned at upload time). | https://anonymous.4open.science | done | unblocks Day 12 supplementary URL anonymization |
| **J6a** | ✅ Done 2026-05-11: Josh + Tianyu Shi OpenReview accounts confirmed exist. Max + Alaa messaged 2026-05-11 to verify their accounts; responses pending. | https://openreview.net | Day 14 (was) | done | — |
| **J6b** | Commit to EMNLP-binding venue at ARR submission (Day 15 click-through in OpenReview portal) | https://openreview.net | Day 15 (2026-05-24) | submission receipt | Yes — last action before submit |
| **J7** | ~~Decide `external/Logomesh` submodule fate~~ — RESOLVED 2026-05-12 as N/A per §5.1 (submodule in main repo doesn't affect submission; will be excluded from supplementary copy at Day 12 build) | n/a | n/a | n/a | **No longer blocking** |

**Strategic decisions Josh needs to lock at kickoff** (covered in plan file §10):

- ~~**D1**~~ — RESOLVED 2026-05-12 as moot per §5.1. The `external/Logomesh` submodule is in the main repo, which doesn't need to be anonymized. It will be excluded from the Day-12 supplementary package by curation, not by repo editing.
- **D2** — Confirm EMNLP-binding venue selection (recommended). ARR portal makes this binding at submission.

---

## 7. Phase 2 + Phase 3 deferred items

| Window | Items |
|---|---|
| **Phase 2 review period (05-26 → 06-29)** | (a) Croissant 1.1 hosting plan + per-node-telemetry backfill of Track F artifacts (e.g., Zenodo DOI, HuggingFace Datasets); (b) extend Cartography to a 5th prompt class if Phase 1 results are weak; (c) Llama 3.2 11B generalization sweep to address "single model family" Limitation directly. |
| **Phase 2 author response (07-07 → 07-13)** | Standard ARR author response cycle. Text-only "Official Comments" on reviews. No new empirical work. |
| **Phase 2 commit decision (07-30 → 08-02)** | EMNLP commitment binding. If accept: lock venue. If reject from ARR or wish to recycle: workshop/journal pivot. |
| **Phase 3 camera-ready (08-20 → 09-20, conditional)** | (d) Full §A Reproducibility Appendix expansion to 3-4 pages; (e) full de-anonymization (restore `\author{}`, GitHub URL, advisor); (f) Croissant package publication on Zenodo or HuggingFace Datasets; (g) anti-Goodhart reward research program write-up as §Future Work expansion. |

Items consciously **deferred from Phase 1 to Phase 2** because not ARR-required:
- Croissant **hosting** (export script is enough for ARR; hosting is a camera-ready bar).
- 11B generalization to address "single model family" Limitation directly.
- Full 3-4 page reproducibility appendix expansion (1-page v1 is enough for ARR review).
- Anti-Goodhart-reward-design follow-up alluded to in §7 Future Work.

---

## 8. Risk register (v2)

| Risk | P | Impact | Mitigation |
|---|---|---|---|
| Day-8 ACL style switch fails | 25% | High — cascades | Pulled forward from v1 Day 10. J4 delivers `acl.sty` by Day 7. Smoke compile Day 8 AM before other edits. |
| Anonymization grep Day 12 finds leaks in the staged supplementary | 15% | Medium | Per §5.1, scope is the staged supplementary package only (not the main repo). Anonymization is applied during the Day-12 build step itself, so leaks should be caught at build time, not at verification. Day 13 slack absorbs any residue. |
| Co-author review surfaces structural concern late | 20% | Medium | Pulled co-author send-out from v1 Day 12 → Day 11. 4-day async window. Plasticity: only must-act items applied. |
| Responsible NLP Checklist forces "No" answer | 10% | High | Day 2 draft surfaces issue; Day 9 has room for §-level mitigation. |
| Llama license language changes between draft and submission | 5% | Low | J2 captures verbatim string and snapshot URL on Day 2. |
| anonymous.4open.science outage on submission day | 10% | Medium | Pre-stage upload by Day 12; verify URL Day 14. |
| ~~`external/Logomesh` reveals identity even after URL change~~ | n/a | n/a | RESOLVED 2026-05-12 — `external/Logomesh` lives in the main repo which doesn't need to be anonymized. Excluded from the Day-12 supplementary package by curation. |
| Croissant validator rejects generated JSON-LD | 15% | Low — Phase 2 hosting work | Day-12 smoke isolates; failure deferred to Phase 2. |
| Cartography sweeps produce no structural variation by Day 7 | 25% | High | Unchanged plasticity rule from v1 (YELLOW reduced-claim → NO-GO Limitations). |
| Page budget verification (J1) returns Findings limit < 8 pages | 10% | Medium | Day 10 cuts pre-structured; §6 proof → Appendix becomes mandatory. |
| GPU access disrupted (renders, hardware) Days 4-5 | 15% | Medium | Cartography sweeps total ~10.4 hours; can compress to weekend block. JSON-checkpoint-every-25-items design (per protocol §6) means crash-resumability. |

---

## 9. Verification gates (per artifact)

| Gate | Trigger | Test | Pass criterion |
|---|---|---|---|
| **G-smoke-compile** | End of every TeX-edit day in §2 (Days 4, 7, 8, 9, 10, 11, 13, 14) | Run `pdflatex` twice over `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` (second pass resolves cross-refs). Capture the `.log` file. | Compiles without `Error:` lines and produces a PDF. The only acceptable `Warning: Reference ... undefined` is `sec:exp2` and `eq:mcts_reward` until Day 7 application lands; after Day 7 EOD, zero undefined refs. No new undefined refs introduced by the day's edits. |
| **G-style** | Day 8 EOD | `pdflatex` builds with `acl.sty` | Compiles; page count ≤ J1 budget |
| ~~**G-anon-A**~~ | n/a | RETIRED 2026-05-12 — per §5.1, the main repo doesn't need to be anonymized. G-anon-A applied to the live repo and was scope-mismatched. |
| **G-anon-B** | Day 12 | Grep over `tmp/supplementary-staged/` (the staged supplementary package only — NOT the main repo) for `(?i)(josh\|max\\b\|bakul\|asuka\|tianyu\|shi\|mcgill\|logomesh)`, `github\\.com/(joshhickson\|Logomesh)`, `linear\\.app/logomesh`, email patterns, "advisor". Also: `pdftotext` over the review PDF for the same patterns. | Only false positives in supplementary; PDF has zero identifying-string matches in rendered text (every match audit-signed) |
| **G-revision** | Day 11 | `grep -n "pending audit"` over canonical TeX | Zero matches |
| **G-checklist** | Day 10 | Every "Yes" answer in Responsible NLP Checklist cites a TeX §-ref that exists | All citations resolve |
| **G-croissant** | Day 12 | `python -m mlcroissant validate <generated JSON>` | Valid Croissant 1.1 |
| **G-license** | Day 10 | Llama attribution string (per J2) appears verbatim in §Ethical Considerations | Present |
| **G-double-blind** | Day 14 | TeX has empty `\author{}`; no acknowledgments; no GitHub URL in compiled PDF | All three confirmed by grep over aux + `pdftotext` |
| **G-submission** | Day 15 | OpenReview submission accepted; venue=EMNLP; double-blind=Y | Confirmation receipt logged |

**G-anon-B is the load-bearing gate** for submission readiness. Pass C is verification, not discovery — if it finds new leaks beyond Pass A's known list, Day 13 slack absorbs cleanup.

**G-smoke-compile is the load-bearing gate for edit integrity** (added 2026-05-12 after Day-4 Session A surfaced a project-long pre-existing fatal natbib error that had gone uncaught — the first project pdflatex run since at least Day 2; actual onset of the natbib mismatch is unknown). Compile-level verification at the end of every TeX-edit day catches reference errors, label collisions, package-load errors, and bibliography-format mismatches at the day they're introduced rather than at Day 8 ACL-switch or Day 14 final-compile. Day 11 is included because Day-11's `G-revision` task (resolving "pending audit" markers) is TeX-touching even if non-substantive. Plasticity rule: see §3.

---

## 10. v1 reference (superseded — preserved for audit trail)

The v1 day-by-day matrix below is preserved verbatim from the v1 file. Use the v2 matrix in §2 above for current planning.

### v1 Days 1-3 (2026-05-10 → 2026-05-12) — Narrative pivot

| Day | Track | Task | Output |
|---|---|---|---|
| 1 | A | Draft new abstract using Gemini Part 5 sketch | `docs/logs/2026-05-10_session-A-prose-drafts-v9-abstract.md` |
| 1 | A | Draft new §1 Contributions list | `docs/logs/2026-05-10_session-A-prose-drafts-v9-contributions.md` |
| 1 | A | Draft new §3 Problem Formulation | `docs/logs/2026-05-10_session-A-prose-drafts-v9-problem-formulation.md` |
| 1 | A | Draft Latent Cartography sweep protocol | `docs/logs/2026-05-10_latent-cartography-sweep-protocol.md` |
| 2 | B | Apply v9 abstract / contributions / §3 to canonical TeX | TeX edits |
| 2 | A | Draft new §7 Discussion | `docs/logs/2026-05-10_session-A-prose-drafts-v9-discussion.md` |
| 3 | B | Apply §7 / §7.4 drafts | TeX edits |

### v1 Days 4-7 (2026-05-13 → 2026-05-16) — Latent Cartography sweeps

[as in v2 §2, unchanged]

### v1 Days 8-10 (2026-05-17 → 2026-05-19) — §6 + §1 polish + page budget

[v2 moved style switch from Day 10 → Day 8; otherwise similar]

### v1 Days 11-15 (2026-05-20 → 2026-05-24)

[v2 added Anonymization Pass C and Croissant smoke pack on Day 12; otherwise similar]

---

## 11. Session A's running notes (v2)

This space is for Session A to track day-of-day progress under v2. Append entries below with timestamps as work lands.

### 2026-05-10 (Day 1, late PM)

v9 prose drafts complete (abstract, contributions, problem-formulation, Cartography sweep protocol). Post-draft verification audit ran on Track A/C/D reports + post-merge verification + canonical TeX bibliography; 8 claims verified, 1 inaccuracy corrected (3.06× at 1B / 3.04× at 3B vs single 3.04×), 1 qualifier added ($(b,d) = (3, 5)$ on the 60× claim), 2 citations dropped post-bibliography-grep. v9 §7.3 Limitations rewrite and title-lock memo drafted as additional Day-1 work. This plan file restructured v1 → v2 reflecting the ARR-cycle deadline correction and the three newly-required deliverables (Responsible NLP Checklist, Llama attribution, §A Reproducibility Appendix). Josh kickoff block (J1, J2, J3, J7) outstanding; deadlines: J1 + J7 by Day 1 EOD, J2 + J3 by Day 2 EOD.

---

*End of v2 ARR-Cycle Execution Plan. This file is the durable artifact governing 2026-05-10 through 2026-05-25 submission and the Phase 2 commitment window through 2026-08-02. Modifications must be flagged at the top with date + reason. Discipline matters.*
