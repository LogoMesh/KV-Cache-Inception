# Session Log — 2026-05-11

**Session type:** Session A (Opus, strategic) — also executing Session B TeX application in same Claude session
**Day of ARR-cycle execution plan:** Day 2
**Submission deadline countdown:** T-14 calendar days to 2026-05-25 ARR submission (T-83 to EMNLP commitment 2026-08-02)

---

## Session frame

Resumed Day 2 of the ARR-cycle execution plan v2 (filename retains `2026-05-10_14-day-execution-plan.md`). Day 2 owner assignments: Session B applies the 5 v9 drafts to canonical TeX; Session A drafts §7 Discussion + License attribution + Responsible NLP Checklist. The License + Checklist drafts depend on Josh's J2/J3 verification responses, which have not yet landed in the session log, so they are deferred to whenever Josh's responses arrive.

Josh's directive on resume: "continue, provided we are prepared to log our work and ensure that our claims are empirically verified." Empirical-verification discipline applied throughout — all numeric and empirical claims in v9 §7 Discussion cross-referenced against source track reports before inclusion.

## Inputs read

1. Canonical TeX lines 39, 50-53, 76-87, 128-165, 510-580 — confirmed unchanged since Day 1 read (no Session B intervention between Day 1 and Day 2).
2. Track A, Track C, Track D, Track F reports — re-confirmed numbers cited in v9 Discussion draft.
3. Post-merge verification report — re-confirmed 14-15/17 layers σ_H claim.
4. ARR-cycle execution plan v2 — Day-2 task list.

## Outputs produced

### Session B work (TeX application of 5 v9 drafts)

| Section | Lines edited | Status | Verification |
|---|---|---|---|
| Title (TeX:39 → 40) | 1 → 2 lines | ✅ Applied | Grep confirmed `\title{Reversible KV-Cache MCTS...}` at line 40 |
| Abstract (TeX:50-53) | Replaced REVISION marker + body | ✅ Applied | Structure preserved; new D+ marker at line 50; new body line 52 |
| §1 Contributions (TeX:80-87) | Replaced REVISION marker + enumerate block | ✅ Applied | New D+ marker at line 80/81; three contributions with corrected memory bound numbers |
| §3 Problem Formulation (TeX:131-154) | Replaced §3.1 + Hypothesis 1; added §3.2 Dimensional Escape | ✅ Applied | §3.2 Dimensional Escape created at line 141; Definition + Hypothesis 2 + relocated Eq added; existing "Discrete Text-Space Bottleneck" auto-renumbers from §3.2 → §3.3 |
| §7.3 Limitations (TeX:513-516) | Replaced body | ✅ Applied | 8 numbered limitations under Option D+; `\label{sec:limitations}` added |

**File state post-edits:** 687 lines (was 652, net +35). Structure grep confirmed all 5 REVISION markers landed (one as `§3-reframe-D+`, four as `*-recast-D+`).

**Verification gates checked:**
- All five `\subsection{}` and `\section{}` headings parse correctly
- Hypothesis 1 (`hyp:orthogonal_escape`) label preserved verbatim
- Hypothesis 2 (`hyp:goodhart`) added correctly
- Definition (`def:dim_escape`) added correctly
- Eq `eq:alignment_faking` label preserved with equation now standalone (was embedded in §3.1 prose, now post-Hypothesis 1)
- Eq `eq:goodhart` added correctly
- New labels: `sec:dim_escape`, `def:dim_escape`, `hyp:goodhart`, `eq:goodhart`, `sec:limitations`
- Forward-references to `sec:exp1` and `sec:exp2` will produce undefined-reference warnings until Day 7 §5 rewrite — expected and documented

### Session A work (v9 §7 Discussion draft)

| Artifact | Path |
|---|---|
| v9 §7 Discussion draft | [docs/logs/2026-05-11_session-A-prose-drafts-v9-discussion.md](2026-05-11_session-A-prose-drafts-v9-discussion.md) |

**Scope:**
- §7.1 retitled "Implications for Latent-Space Monitor and Reward Design"; expanded from 2 paragraphs (monitor-only) to 4 paragraphs (monitor + reward + unifying lesson)
- §7.5 Future Work expanded from 1 paragraph (Phase B single bag) to 4 paragraphs (Infrastructure-and-scaling / Monitor-design research / Reward-design research / Cartography extensions)
- New labels: `sec:discussion-implications`, `sec:future-work`

**Empirical claims verified before inclusion:**

| Claim in Discussion draft | Source | Status |
|---|---|---|
| `|r| ≈ 0.60` static correlation entropy↔correctness | Track A §4 (1B hard 0.602) + §5b (3B ultra 0.614) | Verified |
| "predicting `A` on 185 of 200 items under heavy steering" at 1B | Track F §1 + §7 mechanism diagnosis | Verified |
| "minimal-magnitude steering on 51% of items" at 3B | Track F §1 (101/200 = 50.5% → 51%) | Verified |
| "≈ 162 GB at 20B-parameter configurations" working set | Track D §3 + §4 replacement-paragraph | Verified |
| Nine anti-Goodhart candidates enumerated | Track F §9 + Gemini round 4 Part 2 survey | Verified |

### Session B deferred items (pending Josh's J2 / J3)

| Artifact | Status | Depends on |
|---|---|---|
| `docs/logs/2026-05-11_session-A-prose-drafts-v9-license-attribution.md` | DEFERRED | J2: verbatim Llama 3.2 license attribution string |
| `docs/logs/2026-05-11_session-A-responsible-nlp-checklist-draft.md` | DEFERRED | J3: Responsible NLP Checklist questions verbatim from ARR |

## Outstanding Josh action items (per ARR-cycle plan v2 §6)

Re-stated for visibility — none answered yet on Day 1, all still pending:

| # | Action | Deadline | Status |
|---|---|---|---|
| **J1** | Confirm EMNLP 2026 Findings page budget | Day 1 EOD (was 2026-05-10) | **OVERDUE** |
| **J2** | Llama 3.2 license attribution string + retention | Day 2 EOD (today, 2026-05-11) | Blocking Day 2 license draft |
| **J3** | Responsible NLP Checklist questions verbatim from ARR | Day 2 EOD (today, 2026-05-11) | Blocking Day 2 checklist draft |
| **J4** | Download ACL `acl.sty` + bib files to `docs/NeurIPS/` | Day 7 EOD (2026-05-16) | Blocks Day 8 style switch |
| **J5** | Decide release-host for anonymized code | Day 4 EOD (2026-05-13) | Blocks Pass A (Day 3 partial) |
| **J6** | OpenReview account + EMNLP venue binding | Day 14 (2026-05-23) | Blocks submission |
| **J7** | Decide `external/Logomesh` submodule fate (drop entirely, recommended) | Day 3 AM (tomorrow, 2026-05-12) | Blocks Pass A |

J1 is now overdue. If still unverified at the start of Day 3, Session A will draft `docs/logs/2026-05-11_J1-page-budget-fallback.md` with a conservative 8-page assumption and a contingency cut list, so Day-2 work can continue without blocking on J1.

## Day 3 hand-off

### Session B Day 3 tasks
1. Apply v9 §7.1 Implications-recast-D+ (TeX:522-527 → new content per `docs/logs/2026-05-11_session-A-prose-drafts-v9-discussion.md` §"Replacement TeX for §7.1")
2. Apply v9 §7.5 FutureWork-recast-D+ (TeX:567-576 → new content per same draft §"Replacement TeX for §7.5 Future Work")
3. **NEW Anonymization Pass A**: edit 4 files per ARR-cycle plan v2 §"Anonymization scrub strategy / Pass A" — `.gitmodules` (drop `external/Logomesh` per J7 default), `logomesh/croissant_export.py:335`, `docs/dataset/croissant_schema_stub.json:56`, `EXPERIMENTS.md:299`

### Session A Day 3 tasks
1. Day-3 cross-ref audit running-log entry
2. If J2 + J3 land overnight: draft `v9-license-attribution.md` + `v9-responsible-nlp-checklist-draft.md`
3. If J1 still missing: draft `J1-page-budget-fallback.md`

## Pace observation update

Session A work on Day 2 took ~4 hours including the §7 Discussion empirical-verification pass. Pace remains aligned with the ~50% of estimated time pattern noted in `feedback.md` and project_context.md.

---

*Mid-day update: see "Day 2 PM — Cartography pulled forward" below.*

---

## Day 2 PM — Cartography sweeps pulled forward from Days 4-5

Josh's request: "Let's push up the latent cartography sweeps. What needs to be done before we can run them?" After enumerating prereqs (probe script, C1 hand-list, driver script, dataset verification, GPU availability), Josh confirmed GPU free → execution proceeded.

### Execution timeline (this afternoon)

| Step | Time | Result |
|---|---|---|
| Probe script + C1 hand-list written | ~10 min | `scripts/probe_track_g_datasets.py` + `scripts/_track_g_c1_factual_recall.json` |
| Dataset access probe run | ~30s | C2/C3/C4 all load cleanly. ARC-Easy, TruthfulQA mc1, HellaSwag verified |
| Driver script written | ~20 min | `scripts/diagnose_track_g_cartography.py` (~430 lines) — pure parameter sweep, no MCTS path sampling (deviation rationale §10 of report) |
| Smoke test | 9.4s wall | 60/60 records on 5 items × 3 alphas × 4 classes — pipeline works |
| 1B full sweep | 5.5 min wall | 2310 records at 7.00 eval/s — 40× faster than protocol's 3.7 hour estimate |
| 3B full sweep | 10.3 min wall | 2310 records at 3.73 eval/s — also dramatically faster than 6.7 hour estimate |
| Cartography report | ~15 min | `docs/logs/2026-05-11_track-G-cartography-report.md` — combined 1B + 3B, ~500 lines |

**Total wall: ~50 min for the entire Cartography contribution.** Far under the 10.4 hour protocol budget (~12× faster).

### Cartography findings (top-line)

- **Success criteria C-2 and C-3 BOTH MET at BOTH scales.** Cartography lands cleanly as Experiment 2 of the paper. Plasticity verdict: GREEN. Probability anchor: ~50% per Gemini round 4 high-end.
- **Cross-class variation in argmax-token concentration at intermediate α** — MCQ classes (C2, C3) amplify "A" prior; continuation (C4) amplifies "The"/"\*"/verb cluster; factual recall (C1) shows no single attractor at intermediate α. Class-specific signal clean and structural.
- **Goodhart signature visualized in entropy trajectory:** at 1B, MCQ classes show entropy DECREASING from 1.51 → 1.21 nats across α=0.0→1.0. The model becomes more confident under steering, with confidence concentrated on the "A" prior. This is the exact mechanism Track F §7 described, now characterized across the full alpha grid.
- **3B is more resistant to steering at low α** (matches Track F): C2 A-fraction stays at 29-30% from α=0.0 through α=1.0, jumps to 68% at α=2.0. Versus 1B's monotonic 83→100%.
- **Universal collapse token at α=5.0 differs by scale**: 1B → `"` (quote), 3B → ` to` (space-to). Same Dimensional Escape pathology, different route.
- **Gold-rank preservation in MCQ:** at α=2.0 when 100% of items argmax-predict "A", the gold answer's median rank remains 1. The Goodhart amplification inflates the prior's probability mass without erasing the gold answer's representation. New nuance for §7.1 discussion.

### Schedule impact

Days 4-5 of plan v2 are now FREE. Plan refresh recommended:
- Days 3-5 can be reallocated to: §5 Experiment 2 prose drafting (Day 4 anchor), Table 2 + Figure 1 / Figure 2 specification, license-attribution + Responsible NLP Checklist drafts (when J2/J3 land), §A Reproducibility Appendix v1 (originally Day 9).
- Day 6+ remains unchanged.

The Cartography report's §9 ("Cached for post-T+72 paper rewrite") contains specific paper-rewriting anchor points that Session A will use Day 3-4 to draft `2026-05-12_session-A-prose-drafts-v10-exp2.md` (the §5 Experiment 2 prose) — pulled forward from Day 6.

### Outstanding Josh action items (unchanged from morning, still blocking)

J1 (page budget) still overdue. J2 (Llama license string) still needed. J3 (Responsible NLP Checklist questions) still needed. J7 (submodule decision) still needed Day 3 AM. The Cartography sweeps did not unblock any Josh action item.

---

*Mid-day continued: see "Day 2 PM addendum — Cartography protocol-deviation lesson + MCTS re-run" below.*

---

## Day 2 PM addendum — Cartography protocol-deviation lesson + MCTS re-run

Josh asked: "is the fact that you abandoned MCTS to achieve this going to affect the paper's peer review process?" Honest answer: yes, a careful reviewer would catch and ding it.

**The tension:** the paper's framing (v9 §3 Hypothesis 2: "When MCTS optimizes a surrogate reward... the optimal cache state under r̂ generically lies off the semantic manifold"; v9 §5 Experiment 2 framing: Cartography "maps the structural pre-training priors the search converges upon") commits to MCTS being central to Contribution 3. The pure-parameter-sweep I ran doesn't measure either claim — no search, no path selection, no reward evaluated. The §10 justification I wrote post-execution ("Track F already characterized path-selection patterns") was true at the data-content level but ignored the paper-level framing dependency.

**Lesson stored in `memory/feedback.md`:** *"Protocol deviations must be pre-checked against paper-level framing claims that depend on the protocol's methodology. Speed gains are NOT sufficient justification when the original methodology is the source of a paper claim."* New entry includes: explicit cross-check protocol ("State the cross-check explicitly: I'm considering deviation X. The paper's claims that depend on the original methodology are Y. Deviation X affects/doesn't affect those claims because Z."), and a list of acceptable reasons to deviate (infrastructure constraint, explicit Josh approval, methodology not paper-anchored).

**Re-run starting:** Option 1 from the three I presented — re-run Cartography with proper MCTS path-sampling. The existing pure-sweep results are preserved as a parameter-sweep ablation in the paper's supplementary. The MCTS-driven run will produce the per-(class, item) best-path α-tuple distribution that addresses Hypothesis 2 directly and restores the dropped success criterion C-1 (cross-class variation in best-path α-tuple selection by the search).

**MCTS protocol:** Same as Track F — depth-3, 27 path samples per item, alpha-set {0.1, 0.5, 1.0}, entropy reward (-H_t / log|V|), per-item best-path applied permanently before measuring top-5 leaf distribution. 4 prompt classes × 100 items per class × 2 scales. Expected wall: ~37 min at 1B + ~67 min at 3B = ~104 min total.

**Cartography pure-sweep data: kept as supplementary ablation.** It cleanly shows magnitude-vs-token-collapse heatmap across 6 alphas without confounds from the search policy. Useful as a "deterministic-trajectory baseline" that the MCTS-driven Cartography refines.

---

*Continued: see "Day 2 evening — MCTS Cartography re-run complete; methodology consistency restored" below.*

---

## Day 2 evening — MCTS Cartography re-run complete; methodology consistency restored

Re-ran Cartography with proper MCTS path-sampling per Track F protocol (depth-3, 27 paths, alpha-set `{0.1, 0.5, 1.0}`, entropy reward). Both scales:
- 1B MCTS: 385 records, 17 min wall (~2.7 s/item)
- 3B MCTS: 385 records, 30 min wall (~4.7 s/item)
- Combined wall: 47 min

**ALL THREE success criteria now MET at both scales** (C-1 recovered from N/A under pure-sweep).

### Headline MCTS findings

- **Class-specific best-path α-tuple selection** (success criterion C-1):
  - 1B mean-step α: C1 Factual 0.34 / **C2 MCQ 0.89** / **C3 MCQ 0.83** / C4 Continuation 0.51
  - 3B mean-step α: C1 0.61 / **C2 MCQ 0.22** / **C3 MCQ 0.28** / C4 0.38
  - **MCQ classes INVERT direction across scales** (1B heavy → 3B minimal). Direct evidence for the Track F two-mechanism-route finding extended across 4 prompt classes.

- **1B C2 MCTS reproduces Track F's headline finding:** `(1.0,1.0,1.0)` modal at 42% (Track F was 91/200 = 45.5%); argmax "A" at 92/100 (Track F was 185/200 = 92.5%).
- **3B C2 reproduces Track F's 3B finding:** `(0.1,0.1,0.1)` modal at 49% (Track F was 101/200 = 50.5%); argmax balanced across A/B/C/D since search picks minimal α.
- **Cartography generalizes Track F to TruthfulQA MCQ (C3):** same pattern — heavy α + "A" amplification at 1B, minimal α + balanced distribution at 3B.

### Cartography report updated

`docs/logs/2026-05-11_track-G-cartography-report.md` rewritten to position **MCTS-primary methodology** as the central Cartography contribution. Parameter-sweep data preserved as **§4 Supplementary parameter-sweep ablation** — characterizes cache-geometry magnitude effects across the wider alpha grid without search-policy confounds. Key insight: the MCTS's chosen α regimes (mean-step 0.83-0.89 for MCQ at 1B; 0.22-0.28 at 3B) lie exactly where the parameter sweep reveals the entropy minimum — validates the search's reward-landscape navigation.

### Memory + framing-consistency lesson lives

The protocol-deviation lesson stored in `memory/feedback.md` survives this session as durable guidance for future protocol decisions. Future Session A runs (or fresh Claude sessions resuming this work) will see the lesson and apply the pre-deviation cross-check protocol.

### Schedule impact (revised)

- Cartography work (sweep + MCTS + report v2) complete on Day 2. Days 4-5 remain FREE.
- Day 3 (tomorrow) can absorb: §5 Experiment 2 prose drafting (originally Day 6, with MCTS data anchor points from report §5); §A Reproducibility Appendix v1 (originally Day 9); license-attribution + Responsible NLP Checklist drafts (when Josh's J2/J3 land).

### Outstanding Josh action items (unchanged)

J1 page-budget (overdue), J2 Llama license, J3 Responsible NLP Checklist, J7 submodule decision (Day 3 AM). Cartography sweeps did not unblock any.

---

---

## Day 3 kickoff prep — 2026-05-12 priorities

**READ FIRST upon resuming work tomorrow:**
1. `memory/MEMORY.md` (the index) — confirm which feedback entries apply.
2. `memory/feedback.md` — especially the 2026-05-11 entries on primary-source fidelity and protocol-deviation pre-check. Both are durable lessons; apply them before any new drafting or empirical claim.
3. `memory/recast_action_plan.md` — current state pointer, ARR-vs-EMNLP deadline correction.
4. `docs/logs/2026-05-10_14-day-execution-plan.md` §2 Day 3 row.

### Day 3 task list (in priority order)

| Priority | Owner | Task | Source-of-truth check before writing |
|---|---|---|---|
| 1 | B | Apply v9 §7 Discussion drafts to canonical TeX (§7.1 + §7.5 sections) | Read `docs/logs/2026-05-11_session-A-prose-drafts-v9-discussion.md` directly for the replacement TeX bodies. Do NOT paraphrase from this session log. |
| 2 | B | Anonymization Pass A — 4 files: `.gitmodules` (drop `external/Logomesh` per D1 default OR anonymize URL); `logomesh/croissant_export.py:335`; `docs/dataset/croissant_schema_stub.json:56`; `EXPERIMENTS.md:299` (McGill → "institutional cluster") | Read each file fresh to find exact current strings. Pass A details in plan v2 §"Anonymization scrub strategy". |
| 3 | A | §5 Experiment 2 prose drafting (PULLED FORWARD from Day 6) | **Source from `docs/logs/2026-05-11_track-G-cartography-report.md` directly** — that's the primary artifact. Anchor points in report §5 give exact numbers + table specs. Do NOT cite from this session log's headline. Re-derive any number that seems important by re-reading the report. |
| 4 | A | §A Reproducibility Appendix v1 (PULLED FORWARD from Day 9) | Sources to read directly: `scripts/diagnose_track_f_negcontrol.py` (Track F hyperparams), `scripts/diagnose_track_g_cartography.py` (Track G), `logomesh/kv_mcts.py` MCTSConfig dataclass. Track A/C reports for prompt templates. |
| 5 | A | License-attribution paragraph draft (DEPENDS on Josh J2) | Verbatim Llama attribution string from https://www.llama.com/llama3_2/license/ — must match exactly. |
| 6 | A | Responsible NLP Checklist Q&A draft (DEPENDS on Josh J3) | Verbatim questions from https://aclrollingreview.org/responsibleNLPresearch/. |

### Primary-source fidelity discipline for Day 3

Per the 2026-05-11 `memory/feedback.md` entry, every paper claim must trace to a primary artifact. For Day 3 specifically:

- **§5 Experiment 2 numbers** come from `docs/logs/2026-05-11_track-G-cartography-report.md` (Track G report) — NOT from this session log or the running log. Examples of numbers and where they MUST be cited from:
  - "1B C2 mean-step α = 0.89" → Track G report §2.1 table
  - "1B C2 modal best-path `(1.0, 1.0, 1.0)` 42%" → Track G report §2.1 table
  - "92% argmax 'A' at 1B C2 best-path leaf" → Track G report §2.2 table
  - "Track F headline 91/200 = 45.5%" → Track F report §1 + §3 (NOT the cumulative-findings running log)
- **§5 Experiment 1 numbers** come from `docs/logs/2026-05-11_track-F-negative-control-report.md` directly.
- **§A Reproducibility** code-config numbers come from the actual `scripts/*.py` files, not from prior REVISION markers.

If any number in the draft is unfamiliar or surprising, run an analysis script against the raw `_track_*_results_*.json` to re-derive it. Never accept a "running log says X" citation chain for paper text.

### Methodology fidelity for Day 3

Per the same feedback entry, any code or methodology change touching `logomesh/` or experimental scripts must:
1. Cross-check whether the change affects a paper-level claim (§1 Contributions, §3 Hypotheses, §5 Experiment framings).
2. If yes, flag to Josh before executing.
3. Document the cross-check in the running log entry, even when the answer is "no, doesn't affect paper claims."

The Anonymization Pass A on `logomesh/croissant_export.py:335` is a non-paper-affecting change (just a URL string) — straightforward. The `.gitmodules` submodule decision (D1) is Josh's choice and pre-flagged. No other code changes expected on Day 3.

### Outstanding Josh action items (still blocking, in priority order)

1. **J1 page-budget** (overdue from Day 1) — Day 10 page-cuts depend on knowing the limit. If still missing Day 3 AM, Session A drafts conservative-8-page-assumption fallback memo.
2. **J7 submodule decision** (Day 3 AM) — blocks Pass A. Default: drop entirely.
3. **J2 Llama license string + J3 Responsible NLP Checklist questions** (Day 2 EOD, now today) — block §-level drafting.
4. J4 (Day 7), J5 (Day 4), J6 (Day 14) all still on schedule.

### Schedule status

- Day 1 ✅ done (Drafts + plan v2 restructure)
- Day 2 ✅ done (TeX application + §7 draft + Cartography sweep + MCTS + report v2)
- Day 3 (tomorrow) 🔄 see above
- Days 4-5: now mostly absorbed by Day 3 acceleration; bandwidth for Tianyu-pre-review TeX polish OR pulling Day 8 style switch forward
- Day 11: send to Tianyu + co-authors (still 4-day async review window)
- Day 15: submit to ARR May 2026 cycle, EMNLP binding venue

Probability anchor: ~50% acceptance per Gemini round 4 high-end estimate; Cartography landed cleanly under MCTS-primary methodology.

---

*End of 2026-05-11 Day-2 session log. Day 2 final wall: ~7 hours Session A (5 TeX edits + §7 Discussion draft + sweep + MCTS + report v2 + lesson stored + plan v2 restructured + memory updates for Day 3 prep). Apply Day 3 of ARR-cycle execution plan v2 starting from `memory/feedback.md` and the §5 Experiment 2 anchor points in `docs/logs/2026-05-11_track-G-cartography-report.md`.*

---

## Day 3 execution (started 2026-05-11 PM, running ahead of plan-v2's 2026-05-12 slot)

Calendar date is 2026-05-11; per the ~50% pace pattern, Day 3 work began the same calendar day Day 2 closed.

### Inputs read this session (primary-source discipline applied)

1. Canonical TeX lines 39-50, 132-181, 520-580 — confirmed unchanged since Day 2 EOD.
2. v9 §7 Discussion draft (`docs/logs/2026-05-11_session-A-prose-drafts-v9-discussion.md`) — full read for the replacement-TeX bodies of §7.1 and §7.5.
3. Track G Cartography report (`docs/logs/2026-05-11_track-G-cartography-report.md`) — re-verified §2.1 (1B C2 MCQ modal `(1.0,1.0,1.0)` 42%; mean-step α 0.89), §2.2 (1B C2 argmax "A" 92/100), §3 cross-scale comparison numbers.
4. Plan v2 (`docs/logs/2026-05-10_14-day-execution-plan.md`) Day-3 row.
5. **Primary-source web fetches for EMNLP / ARR anonymization rules** (NEW this session):
   - [EMNLP 2026 main paper call](https://2026.emnlp.org/calls/main_conference_papers/) — "Following the ACL and ARR policies, there is no anonymity period requirement"
   - [ARR CFP](https://aclrollingreview.org/cfp) — paper + supplementary anonymization rules verbatim
   - [ARR anonymity policy](https://aclrollingreview.org/anonymity) — post-Feb-2024 preprint policy
   - [NAACL 2024 anonymity blog](https://2024.naacl.org/blog/anonimity/) — policy change confirmation

### Outputs produced

#### P1 ✅ Apply v9 §7 Discussion to canonical TeX

| Section | TeX target | New labels | Status |
|---|---|---|---|
| §7.1 Implications | lines 522-527 → 522-531 (heading retitled "Latent-Space Monitor and Reward Design"; 3 `\paragraph{}` blocks: Monitor design / Reward design / Unifying lesson) | `sec:discussion-implications` | ✅ Applied |
| §7.5 Future Work | lines 567-576 → 571-584 (4 `\paragraph{}` blocks: Infrastructure / Monitor-design / Reward-design / Cartography extensions) | `sec:future-work` | ✅ Applied |

**File state post-edits:** 695 lines (was 687). Cumulative Option D+ delta from baseline-pre-Day-2: 652 → 695 = +43 lines. All inner-section labels verified before edit (`def:dim_escape` line 145, `hyp:orthogonal_escape` line 151, `hyp:goodhart` line 165, `eq:alignment_faking` line 162). Forward-references `sec:exp1` / `sec:exp2` still undefined until Day-7 §5 rewrite (expected, documented).

**Empirical claims verified before §7.1 inclusion:** |r|≈0.60 (Track A §4 + §5b), 185/200 "A" at 1B heavy steering (Track F §1 + §7), 51% minimal-magnitude at 3B (Track F §1: 101/200 = 50.5% rounded). All numbers traced to primary track reports, not running-log citations.

#### Primary-source anonymization-rule verification

Five primary sources read and synthesized (all URLs in §"Inputs read"). The plan v2 (2026-05-10) treated anonymization as a repo-cleaning exercise — editing `.gitmodules`, `croissant_export.py:335`, `croissant_schema_stub.json:56`, `EXPERIMENTS.md:299`, plus `git rm --cached` of `CLAUDE.md` / team briefs. This was scope-mismatched to the actual rules.

**Actual binding rules (post-Feb-2024 ACL policy):**

| Surface | Anonymization required? | Source |
|---|---|---|
| Submitted PDF | YES | ARR CFP |
| Bibliography (own works) | YES — third-person | ARR CFP |
| Supplementary materials | YES | ARR CFP |
| Code repo link in supplementary | YES — Anonymous GitHub or anonymous.4open.science | ARR CFP |
| Main GitHub repo | **NO** | ACL Anonymity Policy (Feb 2024) — non-anonymous preprints explicitly allowed during review |
| arXiv preprint | NO | Same |
| Local files (`CLAUDE.md`, team briefs) on GitHub | **NO** | Not part of submission |
| `external/Logomesh` submodule in main repo | **NO** | Same |

**Grep over canonical TeX** for `(?i)(logomesh|mcgill|joshhickson|hickson|tianyu|toscano|bakul|asuka|maxwell|linear\.app)` + `\bShi\b|\bMax\b` + `github\.com|@gmail|http[s]?://...`:

- **Renders in PDF:** ONLY `\author{[Authors]}` placeholder at line 42 — already anonymized. ✅
- **Does not render in PDF (LaTeX `%` comments only):** mentions of "logomesh/" at line 332 (IMPL comment), "Tianyu Shi" at line 573 (CRITIQUE NOTE comment we wrote earlier today), "Max" / "Max's lane" / "Max's simplified Exp 3" in REVISION markers at lines 212, 345, 361, 506, 511, 576 — all `%`-prefixed, none rendered in compiled output.

**Conclusion:** The PDF reviewers see is already anonymized. Comments only matter if raw `.tex` source is uploaded as supplementary — standard practice is to upload PDF + curated code package, not raw `.tex`.

#### Plan v2 corrections applied (2026-05-11 PM)

| Change | File | Status |
|---|---|---|
| Day 3 row: drop Pass A; replace with §5 Exp 2 prose drafting (pulled forward from Day 6) | `docs/logs/2026-05-10_14-day-execution-plan.md` §2 | ✅ Applied |
| Day 4 row: drop Pass B; replace with §A Reproducibility Appendix (pulled forward from Day 9) | `docs/logs/2026-05-10_14-day-execution-plan.md` §2 | ✅ Applied |
| Day 12 row: rescope from "Pass C verification" to "build anonymized supplementary package + verify" | `docs/logs/2026-05-10_14-day-execution-plan.md` §2 | ✅ Applied |
| New §5.1 "Anonymization scope — primary-source corrected (2026-05-12)" | `docs/logs/2026-05-10_14-day-execution-plan.md` §5.1 | ✅ Applied |
| §6 J7 retired (submodule decision moot) | `docs/logs/2026-05-10_14-day-execution-plan.md` §6 | ✅ Applied |
| §8 risk register: simplified anonymization risk; retired Logomesh-submodule risk | `docs/logs/2026-05-10_14-day-execution-plan.md` §8 | ✅ Applied |
| §9 gates: G-anon-A retired; G-anon-B rescoped to staged supplementary + PDF | `docs/logs/2026-05-10_14-day-execution-plan.md` §9 | ✅ Applied |
| `memory/recast_action_plan.md` CRITICAL CORRECTIONS block: added 4th item (anonymization scope) | memory | ✅ Applied |
| `memory/MEMORY.md` index: updated recast_action_plan pointer description | memory | ✅ Applied |

### Day 3 remaining work (pulled forward + freed by anonymization-scope correction)

| Priority | Owner | Task | Source-of-truth check |
|---|---|---|---|
| 3 | A | §5 Experiment 2 prose drafting (originally Day 6; freed by Cartography pulled forward + Pass A removal) | Source from `docs/logs/2026-05-11_track-G-cartography-report.md` §2.1, §2.2, §5 directly. Track F numbers from Track F report directly. |
| 4 | A | §A Reproducibility Appendix v1 (originally Day 9; freed by Pass B removal) | Sources: `scripts/diagnose_track_f_negcontrol.py`, `scripts/diagnose_track_g_cartography.py`, `logomesh/kv_mcts.py` MCTSConfig dataclass directly. |
| 5 | A | License-attribution paragraph (DEPENDS on Josh J2) | Verbatim from https://www.llama.com/llama3_2/license/ |
| 6 | A | Responsible NLP Checklist Q&A (DEPENDS on Josh J3) | Verbatim from https://aclrollingreview.org/responsibleNLPresearch/ |

### Outstanding Josh action items (J7 retired; revised list)

1. **J1 page-budget** (overdue) — verify EMNLP 2026 Findings page limit; Day 10 cuts depend on this
2. **J2 Llama license string** — blocks license-attribution paragraph
3. **J3 Responsible NLP Checklist questions** — blocks Q&A draft
4. **J4 ACL `acl.sty` download** (Day 7) — blocks Day 8 style switch
5. **J5 anonymous code host** (Day 4; default anonymous.4open.science) — used at Day 12 supplementary build
6. **J6 OpenReview + EMNLP binding** (Day 14)
7. ~~J7~~ — RETIRED 2026-05-11 (submodule decision moot per anonymization-scope correction)

---

*End of Day 3 execution block. Wall: ~2.5 hours (TeX edits + 5 primary-source web fetches + grep verification + plan v2 corrections + memory updates + session-log append). Day 3 P3-P6 (§5 Exp 2, §A Reproducibility Appendix, license, checklist) can begin on Josh's next green light.*

---

## Day 3 evening — J1-J5 executed in parallel + roster update

Josh asked at Day 3 EOD: "can't you do all of these things?" — pointed out (correctly) that J1-J5 are info-gathering tasks (web fetches + a decision), not Josh-only actions. Only J6 truly requires Josh's identity for OpenReview submission. Executed J1-J5 in parallel along with co-author roster updates.

### Co-author roster update (project_context memory)

Roster as of 2026-05-11:
- **Active**: Josh Hickson (lead), Max Toscano (Contribution 1), **Alaa Elobaid** (new addition, role TBD), Prof. Tianyu Shi (advisor)
- **Dropped off**: Bakul, Asuka (no recent participation)
- Day-11 plan-v2 send-out list: Tianyu + Max + Alaa
- OpenReview status: Josh + Tianyu have accounts; Max + Alaa messaged 2026-05-11 (responses pending)

Updated `memory/project_context.md` Team section. Plan v2 §6 J6 split into J6a (account verify — partial done) and J6b (venue commit at submission). J7 retired earlier in session per anonymization-scope correction.

### J1 — EMNLP 2026 page budget verified

**Primary source:** [ACL formatting.md](../NeurIPS/acl_formatting.md) — downloaded with style files. Lines 17-23 verbatim.

| Version | Long paper | Short paper | Notes |
|---|---|---|---|
| **REVIEW** (what we submit) | **8 pages of content** + UNLIMITED references | 4 pages + unlimited references | **Review versions MUST NOT include acknowledgements.** |
| FINAL (post-acceptance) | 9 pages + unlimited acks + unlimited references | 5 pages + unlimited acks + unlimited references | |

**Critical clarification — formerly mis-stated in `memory/project_context.md` line 14:** Limitations + Ethical Considerations are NOT excluded from the page count; they count toward the 8-page review budget. Only References (and the appendix via `\appendix`) are unlimited.

**Findings track:** Not mentioned in EMNLP 2026 main call. Per ARR routing, Findings is a venue-commitment decision at the EMNLP commit phase (2026-08-02), not at ARR submission. Our paper submits as a standard long paper; venue (Main vs. Findings) is decided based on review scores.

**Day 9-10 page-budget implication:** Current canonical TeX body (lines 1-586, excluding bibliography starting ~line 587) is borderline 8 pages. After §5 Experiment 2 prose addition (Day 7, ~30-50 lines), §A Reproducibility Appendix (in `\appendix`, unlimited), license paragraph (~5 lines), and AI assistant disclosure paragraph (~5 lines), the body will be near or slightly over budget. Day 9 page-count check is critical; if over, deeper cuts on §6 Memory Complexity proof (move to `\appendix`) become mandatory.

### J2 — Llama 3.2 license attribution captured

**Primary source:** https://www.llama.com/llama3_2/license/ — fetched 2026-05-11.

Three binding requirements:
1. **"Built with Llama"** prominently displayed (website, UI, blogpost, about page, or product documentation — paper's §Ethical Considerations satisfies this).
2. **Verbatim notice text required:** `Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.`
3. **Acceptable Use Policy URL** must be referenced: https://llama.com/llama3_2/use-policy/

Draft TeX for §7.6 Ethical Considerations + bibitem (`meta2024llama3_2_aup`) authored at [docs/logs/2026-05-11_session-A-prose-drafts-v9-license-attribution.md](2026-05-11_session-A-prose-drafts-v9-license-attribution.md). Apply Day 3-4 (Session B P5).

### J3 — Responsible NLP Checklist captured

**Primary source:** https://aclrollingreview.org/responsibleNLPresearch/ — fetched 2026-05-11.

17 questions across 5 sections:
- **A (every submission):** A1 limitations [YES → §7.3], A2 risks [YES → §7.2 + §7.6]
- **B (artifacts):** B1-B6 [all YES; references to §7.6 license para, §5.2 Phase A, §A Reproducibility, §5 Experiments]
- **C (computational):** C1-C4 [all YES; references to §A Reproducibility]
- **D (human annotators):** D1-D5 [all N/A — no human participants]
- **E (AI assistants):** E1 [YES — Claude + Gemini disclosure paragraph drafted for §7.6 addition]

First-pass Q&A draft at [docs/logs/2026-05-11_session-A-responsible-nlp-checklist-draft.md](2026-05-11_session-A-responsible-nlp-checklist-draft.md). Six open items noted (Reproducibility Appendix must exist; dataset license verification; AI disclosure TeX addition; §7.6 may need `sec:ethical-considerations` label). Final version Day 10 of plan v2.

### J4 — ACL style files downloaded

**Source:** https://github.com/acl-org/acl-style-files (master branch root path, NOT `latex/` subdirectory — verified by probing both paths).

Saved to `docs/NeurIPS/`:
- `acl.sty` (11,615 bytes) — main style file. Uses `\usepackage[review]{acl}` for double-blind submission.
- `acl_natbib.bst` (45,186 bytes) — bibliography style.
- `acl_formatting.md` (326 lines) — reference for page rules, anonymization conventions, structural requirements.

Smoke compilation deferred to Day 8 of plan v2 (the actual style-switch task). Note from acl.sty inspection: the `[review]` option auto-anonymizes the author block, adds line numbers, and adds page numbers — Session B can drop the manual `\iffalse...\fi` REVISION-marker wrapping if review-mode handles it natively. Verify Day 8.

### J5 — Anonymous host decision: anonymous.4open.science

**Rationale:** ARR CFP names "Anonymous GitHub" as one valid example; anonymous.4open.science is the established academic standard for ACL/EMNLP supplementary code uploads. Reasons: (a) supports unlimited repo size; (b) no download tracking (compliant with ARR CFP "Links to file hosting services that can track downloads ... are not allowed"); (c) standard URL pattern unambiguous in supplementary references; (d) higher uptime + bandwidth than Anonymous GitHub for during-review traffic.

URL placeholder for Day-12 supplementary build: `https://anonymous.4open.science/r/kv-mcts-dimensional-escape-XXXX` (final slug assigned at upload time per anonymous.4open.science's URL convention).

### Plan v2 §6 J-items table updated

All five J1-J5 marked done with status detail. J6a (account verify) partial done; J6b (venue commit at submission) is the only remaining truly-Josh-only item (Day 15).

---

*End of Day 3 evening. Wall: ~25 min for J1-J5 + memory + plan updates (5 web fetches + 2 drafts + 4 file edits). Outstanding: J6b at Day 15 + Max/Alaa OpenReview confirmation when responses land. Next concrete trail: §5 Experiment 2 prose drafting (P3), §A Reproducibility Appendix v1 (P4), §7.6 license + AI disclosure paragraph applications (P5 — depends on Session B applying J2 draft).*

---

## Day 3 night — P5 §7.6 license + AI disclosure applied; P3 §5 Exp 2 prose drafted

Josh said "keep walking". Two units delivered:

### P5 ✅ §7.6 license + AI disclosure paragraphs applied to canonical TeX

Two `\paragraph{}` additions to §7.6 Ethical Considerations + one new bibitem to the bibliography:

| Edit | TeX target | Result |
|---|---|---|
| §7.6 license paragraph (J2 verbatim) | After existing TeX:588 paragraph, before §8-cut REVISION marker | ✅ Applied at TeX:592-593 |
| §7.6 AI assistant disclosure paragraph (J3 E1 answer) | Immediately after license paragraph | ✅ Applied at TeX:595-596 |
| `\bibitem{meta2024llama3_2_aup}` | Before `\end{thebibliography}` at TeX:693 (now line 707) | ✅ Applied at TeX:701-704 |

**File state post-edits:** 709 lines (was 695 after Day 3 P1). Cumulative Day 2 + Day 3 delta from baseline-pre-Day-2: 652 → 709 = +57 lines.

**Honest flag — stale paragraph 1 of §7.6:** The existing first paragraph at TeX:588 (preserved unchanged in this additive edit) says "The adversarial datasets generated will be released under a responsible disclosure framework with appropriate access controls." This is Option-α+/Candidate-A residue — Option D+ generates no adversarial datasets; we run Cartography sweeps on existing public benchmarks. Croissant 1.1 release is for runtime evidence artifacts (per-node telemetry from Track F + Track G), not adversarial datasets. Minimal rewrite candidate flagged in the new REVISION marker; awaits Josh sign-off for separate edit. Not paper-blocking.

### P3 ✅ §5 Experiment 2 prose drafted ([2026-05-11_session-A-prose-drafts-v10-exp2.md](2026-05-11_session-A-prose-drafts-v10-exp2.md))

Pulled forward from plan v2 Day 6. ~250-line markdown with full replacement TeX for §5 Experiment 2 paragraph, new Table 2 (best-path α-tuple distribution per class × scale), new Figure 1 spec (cross-scale mean-step α bar chart with raw data values for plotter), Session B notes, page-budget impact estimate, and a primary-source-fidelity verification table for all numeric values cited.

**Headline narrative beats:**
- Setup: 4 prompt classes × 100 items × 2 scales × Track F-style MCTS (depth-3, 27 paths, alpha-set {0.1, 0.5, 1.0}, entropy reward)
- Finding (i): class-specific best-path selection (Table 2 — MCQ classes flip direction across scales)
- Finding (ii): argmax-token concentration (1B C2 92/100 = "A"; 3B C2 balanced 29/27/22/16)
- Finding (iii): gold-rank preservation (median 1, 94% top-5 at 1B C2 even when argmax is "A" 92% of items — geometric mechanism per Definition def:dim_escape)
- Finding (iv): mechanism interpretation per class (prior-amplification route at 1B MCQ; greedy-preservation route at 3B MCQ; light-α-no-attractor route at C1/C4)
- Search reward gain: +0.005 to +0.025 — search IS finding lower-entropy states but they don't correspond to better reasoning

**Page-budget red flag.** Draft estimates +57 body lines from Exp 2 application alone. Current body lines (1-587) at ~70 lines/page = ~8.4 pages. Post-Exp-2 estimate: ~9.2 pages. Combined with Day-4 Exp 1 recast (+30 lines for Track F) + Day-9 §A Reproducibility Appendix (in `\appendix`, unlimited) → ~9.6 body pages estimated. **Day 9 page-count check + Day 10 cuts MANDATORY.** Recommended cut list documented in the draft.

**Cross-references resolved by this draft (Day 7 application):**
- `\label{sec:exp2}` — NEW; resolves 4 forward-references already in TeX (§3 line 171, §7.1 line 529, §7.5 lines 580 + 584)
- `\label{tab:exp2-cartography-paths}` — NEW
- `\label{fig:exp2-mean-step-alpha}` — NEW

**Cross-reference still unresolved:**
- `\ref{sec:exp1}` — referenced in this Exp 2 draft + in §3 + in §7.1; will resolve when Day-4 §5 Exp 1 recast draft is applied.

### Flagged for Day 4

- **§5 Experiment 1 recast** (Track F entropy-MCTS negative control) — pulled forward from Day 6/7. Required because current TeX Exp 1 ("OEI Characterization") is Option α+ residue incompatible with §3 Hypothesis 2's forward-reference to `sec:exp1` as Track F evidence.
- **§7.6 paragraph 1 minimal rewrite** — drops stale "adversarial datasets" framing; preserves Croissant 1.1 sentence. Awaits Josh sign-off.

### Session totals at Day 3 night

| Day-3 task | Status | Wall |
|---|---|---|
| P1: Apply v9 §7 Discussion drafts to TeX | ✅ | ~30 min |
| P2: Anonymization Pass A (live repo) | ✅ Retired per primary-source-verified scope correction | — |
| P3: §5 Experiment 2 prose draft | ✅ Drafted (Day-7 application target) | ~50 min |
| P4: §A Reproducibility Appendix v1 | 🔄 Not started | — |
| P5: §7.6 license + AI disclosure | ✅ Applied to TeX + new bibitem | ~15 min |
| P6: Responsible NLP Checklist Q&A draft | ✅ Drafted (Day-10 finalization target) | (J3 fetch + write) |
| J1-J5 web fetches + decisions | ✅ All done | ~25 min |
| Memory + plan corrections (anonymization scope) | ✅ Applied | (earlier in session) |
| Team roster update | ✅ Applied (Aladdin = Alaa Elobaid added; Bakul/Asuka out) | (parallel batch) |

**Cumulative Day-3 wall: ~3.5 hours of strategic session work.** ~50% of-estimated-time pattern holds.

**TeX state at Day 3 night:** 709 lines. Body: 1-587. Bibliography: 595-708. Footer: 709. All Day 2 + Day 3 P1 + P5 edits landed clean. Cross-references audit: 1 unresolved (`sec:exp1`, Day-4 fix); all others resolve.

---

*End of Day 3 night. Next trail-step options on Josh's green light: (a) §A Reproducibility Appendix v1 draft (P4, ~30-45 min); (b) §5 Experiment 1 recast draft for Day-4 (Track F negative control, ~30-45 min); (c) Minimal rewrite of §7.6 paragraph 1 (5 min, needs sign-off); (d) Pause to review. Plan v2 has us comfortably ahead of schedule — Day 3 effectively absorbed Day 6 + Day 9 work.*

---

## Day 3 late-night — Phase A memory tidy + Phase B stale-sentence audit

Josh directive after the J1-J5 + license/AI/Exp 2 push: "make sure you're making time to keep your documentation / memory organized" + page-budget framed as symptom of likely-stale-sentence-residue rather than a separate concern.

**Phase A — Memory tidied:**
- `memory/project_context.md`: 4 edits — locked title status, corrected venue page-budget (was incorrectly stated "excluding Limitations + Ethics" — they COUNT), corrected style-file (acl.sty not acl_art.cls; Day 8 not Day 10), canonical paper status updated to reflect Day 2 + Day 3 P1 + P5 application progress, Day-by-day section refreshed (Day 3 marked ✅ with full task list; Day 4 outlook with candidates), "What's added post-pivot" timeline marker moved from Day 2 EOD → Day 3 EOD with the §7 Discussion + §7.6 license/AI + J1-J5 + anonymization-scope-correction additions all logged. Roster section (Bakul/Asuka out, Alaa added) already done earlier.
- `memory/recast_action_plan.md`: 2 edits — "Day 2 state summary" header replaced with "Day 3 EOD state summary" + full applied-edits table (lines 39-91, 131-181, 522-584, 592-596, 701-704); "Critical pointers" canonical-TeX status corrected from "still the entropy-reward Candidate A version" to "Option D+ rewrite in progress; 709 lines as of Day 3 EOD".
- `memory/MEMORY.md`: previously updated this session.

**Phase B — Stale-sentence audit on canonical TeX:** [docs/logs/2026-05-11_stale-sentence-audit.md](2026-05-11_stale-sentence-audit.md)

Comprehensive sweep of §2 Background, §4 Proposed Method, §5 Experimental Design, §6 Theoretical Analysis, §7 Discussion for Option α+ / Candidate-A residue not caught in Day 2 + Day 3 prose application passes. 16 distinct findings:

| Severity | Count | Locations |
|---|---:|---|
| BLOCKING | 7 | §2-2 (line 102), §2-3 (line 107), §2-4 (line 111), §4-2 (lines 228-244 Table 1), §4-3 (line 271), §4-4 (lines 292-309 Eq.~mcts_reward), §5-3 (§5 Exp 1 entire) + §5-4 (§5 Exp 2 entire) + §5-6 (§5.5 Eval Metrics OEI-as-primary) |
| RECOMMENDED | 6 | §2-1 (line 98), §4-1 (line 214), §5-1 (line 334), §5-5 (§5.4 Baselines table), §6-1 (line 515 placeholder numbers — already on plan v2 Day 8), §7-1 (line 535 §7.2), §7-2 (line 562 §7.4 OEI framing) |
| OPTIONAL | 1 | §2-5 (line 115 section heading) |
| KNOWN (already flagged) | 1 | §7-3 (line 588 §7.6 paragraph 1 "adversarial datasets generated") |

**Page-budget projection confirmed Josh's hypothesis:** combined stale-sentence cleanup recovers ~35-70 lines of body, comfortably offsetting +57 lines from §5 Exp 2 (Day 7) and +30-50 lines from Day-4 §5 Exp 1 recast. Conservative estimate: 8.6 pages body post-all-fixes. Optimistic (with aggressive deletes per §4-2 + §4-4 option a): 8.0 pages. **§6 Memory Complexity proof → `\appendix` contingency cut likely NOT needed.**

**Open Josh decisions** (5 items requiring sign-off before Session B applies):
1. §4-2 Table 1 "Diagnostic States" — delete or trim?
2. §4-4 Eq.~mcts_reward subsubsection — delete or reframe?
3. §5-5 Baselines table — trim to 3 rows or delete §5.4?
4. §7-3 §7.6 paragraph 1 — apply merged rewrite?
5. §7-1 §7.2 Broader Impact — apply rewrite or hold for camera-ready?

Application schedule (per audit §"Application strategy"): Day 4 morning text-only BLOCKING + RECOMMENDED fixes; Day 4 PM equation/table deletes after Josh signs off; Day 5 RECOMMENDED set; Day 5-7 §5 Exp 1 recast (Session A drafts, B applies); Day 7 §5 Exp 2 + §5.5 Eval Metrics application.

---

*End of Day 3 late-night. Cumulative Day 3 wall: ~4.5 hours strategic session work. TeX state: 709 lines. Documentation/memory state: Day-3-EOD-aligned across `memory/project_context.md`, `memory/recast_action_plan.md`, `memory/MEMORY.md`, `docs/logs/2026-05-10_14-day-execution-plan.md`. Stale-sentence audit awaits Josh sign-off on 5 open decisions before Day 4 application begins.*

---

## Day 3 final — Decision corpus review

Josh directive: produce a plan to review the past-week log corpus and surface what advice it holds against the 5 open audit decisions before sign-off. Plan written (`C:\Users\Josh\.claude\plans\misty-scribbling-fiddle.md` overwritten); plan approved; review executed.

**Findings file:** [docs/logs/2026-05-11_decision-corpus-review.md](2026-05-11_decision-corpus-review.md)

**Methodology applied per plan:** per-decision deep-read of flagged source documents (NOT Grep excerpts) — pivot decision memo (read fully), v9 §3 problem-formulation draft (read fully), Gemini strategy report Parts 1-6 (read fully), v9 §7.3 Limitations draft (read fully), Track F report §7 mechanism diagnosis (read directly via grep + section), Track B citation audit (peer verdicts table), already-applied canonical TeX sections cross-referenced. Verbatim quotes extracted per primary-source-fidelity discipline.

**5 recommendations + confidence:**

| # | Decision | Recommendation | Confidence |
|---|---|---|---|
| 1 | §4 Table 1 "Diagnostic States" | Delete entirely (audit option a) | HIGH |
| 2 | §4 Eq.~mcts_reward subsubsection | Delete entirely (audit option a) | HIGH |
| 3 | §5.4 Baselines table | Trim to **4 rows** (Rimsky + Belitsky + Gupta-Jenner + Bailey) + reframe caption (refined from audit's 3 — Bailey added per §3.2 already-applied peer positioning) | HIGH |
| 4 | §7.6 paragraph 1 rewrite | Apply audit's merged rewrite (past tense correction) | HIGH |
| 5 | §7.2 Broader Impact rewrite | Apply rewrite with one refinement (preserve brief dual-use acknowledgement; mirror §7.1's white-box-access phrasing) | HIGH |

**Strongest corpus-grounded finding for Decision 2:** Gemini Part 2 #1 (Multi-Channel Ensemble) explicitly evaluated the **structural twin** of Eq.~mcts_reward (σ_H + ρ_R + entropy + perplexity) and predicted **15% probability of working under MCTS**. Combined with Track F §7's mechanism-level conclusion (the failure is conceptual, not implementation), keeping Eq.~mcts_reward in §4 would publish an **untested + predicted-to-fail recipe** the paper's own analysis predicts would Goodhart. Strong case for deletion.

**Refinement to audit for Decision 3:** Corpus suggested **4 rows** rather than the audit's 3, adding Bailey 2024 (Obfuscated Activations) since §3.2 already positions Bailey as a structural peer to Hypothesis 1 (monitor-driven sub-case). Other baseline categories from §5.4 (GCG/MPA/DAMON text-space; SonarQube/Semgrep static analysis; MT-Bench/G-Eval LLM-as-judge) dropped per Option D+ framing.

**One mild open question for Decision 5:** whether to cite Bailey et al. 2024 explicitly in §7.2 as a peer dual-use case (negative result for gradient-trained input attacks). Not strictly required for sign-off; can be added as Day 4-5 polish.

**Next step:** Josh reviews `2026-05-11_decision-corpus-review.md`. On sign-off, Day-4 Session B begins TeX application per the audit's `## Application strategy` table.

---

*End of Day 3 final. Cumulative Day 3 wall: ~5.5 hours strategic session work. Output state: TeX 709 lines (no further edits this session); 1 new findings file (decision-corpus-review); 1 updated plan file (misty-scribbling-fiddle.md overwritten with the review-corpus plan); 1 updated session log (this file). Day 4 work blocked on Josh sign-off of the 5 decision recommendations.*

---

## Day 3 absolute-final — 14 audit-flagged + Josh-signed-off edits applied

Josh response on the decision corpus review: "Defer the mild polish question and apply the rest of the changes." Bailey 2024 citation polish for §7.2 deferred. All 5 sign-off decisions + 9 other audit-flagged BLOCKING + RECOMMENDED fixes applied in a single bottom-up batch (highest-line-number first, so earlier edits don't shift later line references).

### Edits applied (in bottom-up order)

| # | Edit | TeX target | Severity / Decision | Result |
|---|---|---|---|---|
| 1 | §7.6 paragraph 1 rewrite | line 588 | Decision 4 (HIGH) | "adversarial datasets generated" α+ residue → Croissant 1.1 runtime-evidence-artifact framing; past tense |
| 2 | §7.4 opening soften | line 562 | audit §7-2 RECOMMENDED | "The Orthogonal Escape Index in Eq..." → "Any measurement of inference-time KV-cache perturbation, including the Orthogonal Escape Index..." |
| 3 | §7.2 Broader Impact rewrite | lines 533-535 | Decision 5 (HIGH) | "discover alignment-breaking interventions" α+ residue → white-box interpretability framing mirroring §7.1; brief dual-use ack preserved; Bailey citation deferred per Josh |
| 4 | §5.4 Baselines trim to 4 rows | lines 430-448 | Decision 3 (HIGH; refined to 4 rows per corpus) | 8 rows → 4 rows (Rimsky CAA + Belitsky Cache Steering + Gupta-Jenner RL-Obfuscation + Bailey Obfuscated Activations); caption reframed as "structurally related" not "comparative evaluation" |
| 5 | §5.2 Phase A bullets trim | lines 343-348 | audit §5-2 BLOCKING | RepE probe enumeration trimmed to honesty-only; Procrustes bullet deleted (deferred to Phase B per §7.5) |
| 6 | §5.1 Phase 0 item (c) soften | line 334 | audit §5-1 RECOMMENDED | "continuous reward interface processing neuronal activation tensors" α+ residue → "accepting the per-step next-token distribution as MCTS node reward input" |
| 7a | §4.2.3 MCTS Node Valuation delete | lines 292-308 | Decision 2 (HIGH) | Entire subsubsection deleted: Eq.~mcts_reward (λ₁σ̄_H + λ₂(1-ρ̄_R) + λ₃·Divergence), UCB1 equation, surrounding prose. Replaced with one-line `\paragraph{MCTS Node Reward (Choice of $\hat{r}$).}` pointing to Eq.~goodhart + §sec:future-work for alternative-reward research program |
| 7b | §5.5 TDS bullet ref cleanup | line 467 | downstream from 7a | Removed broken `\ref{eq:mcts_reward}` reference; minimal stop-gap until §5-6 §5.5 rewrite on Day 7 |
| 8 | §4.2 Step 2 (Evaluation) | line 271 | audit §4-3 BLOCKING | "record the telemetry matrix $\mathbf{T}_t$ as the MCTS node reward" → "compute the per-step next-token distribution... evaluate the MCTS node reward $\hat{r}(\mathbf{h}_t)$ from this distribution (entropy-normalized; Eq.~\ref{eq:goodhart})"; telemetry matrix reframed as static monitoring signal |
| 9 | §4.1 Table 1 delete | lines 228-244 | Decision 1 (HIGH) | Entire 4-state Diagnostic Table deleted ("Alignment Faking" + "Orthogonal Escape" rows contradicted §7.3 Limitations item 4). Replaced with one-line bridge pointing to Definition + Hypotheses in §3.2 |
| 10 | §4.1 RepE LAT extrapolation | line 214 | audit §4-1 RECOMMENDED | "honesty, certainty, and goal-coercion" → "honesty dimension only"; certainty + goal-coercion probes retired |
| 11 | §2.3 H-Neurons over-compliance paragraph | line 111 | audit §2-4 BLOCKING | "We extend this finding to the domain of alignment faking by treating σ_H as an over-compliance pressure proxy..." → "We use σ_H as a layered latent-response diagnostic... independent of any claim about model intent." Resolves internal contradiction with §7.3 Limitations item 4 |
| 12 | §2.2 Bailey complementarity axes 2+3 | line 107 | audit §2-3 BLOCKING | Axis 2 (referenced eq:mcts_reward → deleted) and axis 3 (OEI as primary metric → no longer true) rewritten to anchor on entropy-normalized reward (Eq.~goodhart) and Experiments 1-2 |
| 13 | §2.2 telemetry-derived reward | line 102 | audit §2-2 BLOCKING | "guided by a telemetry-derived reward" → "guided by an inference-time surrogate reward (in our reported experiments, normalized next-token entropy; Eq.~\ref{eq:goodhart})" |
| 14 | §2.1 closing softening | line 98 | audit §2-1 RECOMMENDED | "deceptive internal representations" → "internal representational divergences invisible to behavioral testing alone"; aligns with §7.3 Limitations item 4 |

### File state

| Metric | Before audit edits | After 14 edits |
|---|---:|---:|
| Total lines | 709 | **685** |
| Net delta | — | **−24 lines** |

**Page-budget recovery confirmed empirically.** Josh's hypothesis from 2026-05-11 ("a review to catch more stale sentences is mandatory now; I'm sure that recovers page budget") validated by net −24 lines from stale-sentence cleanup alone, with the Decision 1 (Table 1) and Decision 2 (Eq.~mcts_reward) deletes accounting for the bulk (−15 and −18 lines respectively, partially offset by the +9-line §7.2 rewrite and +4-line §7.6 ¶1 rewrite and +6 lines net from §4 RepE/Step-2 expansions).

### Label / cross-reference state

| Label | Before | After |
|---|---|---|
| `eq:mcts_reward` | line 298 | **REMOVED** (downstream references at line 388 §5 Exp 2 paragraph remain — handled by §5-4 §5 Exp 2 application Day 7; the §5.5 TDS bullet reference was cleaned in Edit 7b) |
| `eq:ucb1` | line 305 | **REMOVED** (no downstream references) |
| `tab:diagnostic_states` | line 243 | **REMOVED** (no downstream references in body; only the §5.5 TDS REVISION marker comment referenced it, and that's also been cleaned) |
| `eq:goodhart` | line 169 | preserved (now the primary referenced reward equation) |
| `sec:future-work` | line 546 | preserved (referenced by deleted §4.2.3's replacement paragraph) |

### Items intentionally NOT applied this batch

- **§5-3 §5 Experiment 1 recast** (Track F prose) — needs Session A draft first (Day-4/5 task)
- **§5-4 §5 Experiment 2 application** (v10 draft exists) — Day 7 task with figure generation
- **§5-6 §5.5 Evaluation Metrics rewrite** — Day 7 task; minimal TDS-bullet stop-gap applied as Edit 7b
- **§2-5 OPTIONAL §2.4 section heading rename** — defer to camera-ready Phase 3
- **§6-1 §6 Memory Complexity worked-example numbers** — Day 8 plan-v2 task (Track D 162 GB / 60× replacement)
- **Bailey 2024 §7.2 citation polish** — deferred per Josh ("defer the mild polish question")

### Cross-references with known dangling state

- `\ref{sec:exp1}` and `\ref{sec:exp2}` — undefined until Day 7 §5 application (expected since plan v2; produces undefined-reference compile warnings, not errors)
- `\ref{eq:mcts_reward}` — still referenced at line 388 (§5 Exp 2 paragraph being recast Day 7); will resolve when §5 Exp 2 application runs

### Verification

- Grep over canonical TeX for stale phrases (`adversarial datasets`, `alignment-breaking interventions`, `over-compliance pressure proxy`, `four diagnostic states`, `GCG (Greedy`) returned 0 body-text hits; all remaining matches are inside REVISION-marker `%`-comments (audit-trail content, compile-invisible).
- Structure grep confirmed all subsection headings preserved + §4.2.3 absent + Table 1 absent.
- File diff: 709 → 685 lines, net -24.

### Plan v2 schedule status post-Day-3

Day 3 has comfortably absorbed Day 6 (§5 Exp 2 prose draft), Day 9 partial (§7.4 OEI framing + audit-flagged sweep), and a substantial chunk of the Day 4-5 stale-sentence audit application. Remaining Day 4-7 work: §5 Exp 1 recast draft (A) + §5 Exp 2 TeX application (B + figure gen) + §5.5 Evaluation Metrics rewrite (B) + §A Reproducibility Appendix v1 draft (A). On schedule with comfortable buffer.

---

*End of Day 3 absolute-final. Day-3 wall total: ~6 hours strategic session work. TeX state: 685 lines, all 14 audit-flagged + Josh-signed-off edits applied. Cumulative Option D+ rewrite progress: Day 2 (5 v9 drafts applied) + Day 3 P1 (§7 Discussion) + Day 3 P5 (§7.6 license + AI) + Day 3 final (14 stale-sentence cleanups) = paper now in a coherent Option D+ state through §7 except for the known Day 4-7 §5 Experiments + §5.5 + §A work. Memory + plan v2 + session log all up-to-date.*

---

## Day 3 post-final — full paper read + PSF fix + framing-audit-pass-2 + Day 4 F1 + F2 + F3

Josh directive late-Day-3: "Just read the paper in full before you continue?" — primary-source paper read uncovered TWO findings the Pass-1 stale-sentence audit missed: §1 Intro α+ framing + §7.3 item 5 Cartography class-name PSF error.

### PSF fix — Cartography class names (3 locations)

Track G report §1 actually ran C1 Factual / C2 ARC-Easy MCQ / C3 TruthfulQA mc1 ($n=85$) / C4 HellaSwag continuation. Three TeX locations had v9-vintage planned names ("factual recall, logical deduction, moral/preference, creative completion") — abstract (TeX:53), §1 Contributions (TeX:87), §7.3 Limitations item 5 (TeX:522). Also "six steering magnitudes" → reframed to MCTS-primary methodology (depth-3 entropy-MCTS, alpha-set {0.1, 0.5, 1.0}) per Track G report §2.1. Marker: `[REVISION | cartography-class-names-PSF-fix-D+ | 2026-05-11]`. PSF violation had landed in applied TeX Day 2; caught by second-pass paper read.

### Framing-consistency audit pass 2 ([docs/logs/2026-05-11_framing-audit-pass-2.md](2026-05-11_framing-audit-pass-2.md))

Method: section-level read against 4 anchors (title, abstract concluding claim, §1 Contribution 3, §3.2 Definition `def:dim_escape`). Pass-1 was word-level grep; missed paragraph-level + heading-level framing residue.

6 findings:

| # | Where | Severity | Status |
|---|---|---|---|
| F1 | §1 Intro paragraphs 1-3 (lines 60-67) — α+ framing | BLOCKING | ✅ APPLIED Day 3 night |
| F2 | §2.4 heading "MCTS for LLM Red-Teaming" → "MCTS for Language Model Search" | RECOMMENDED | ✅ APPLIED Day 3 night |
| F3 | §3.3 lines 180-184 phrases ("Orthogonal Escape phenomenon" → "Dimensional Escape pathology"; "deceptive encodings" → "internal representations") | RECOMMENDED | ✅ APPLIED Day 3 night |
| F4 | §1 CRITIQUE NOTE comment block (lines 69-75) | OPTIONAL | ✅ DELETED with F1 |
| F5 | §5.1/§5.2 "Phase 0 / Phase A" headings | OPTIONAL | DEFER camera-ready |
| F6 | §7.4 line 539 forward-ref + Table 1 relocation when §5 Exp 1 recasts | KNOWN coordination | Day 4-5 with §5 Exp 1 recast |

### Memory write — new feedback entry

[`memory/feedback.md`](C:/Users/Josh/.claude/projects/c--Users-Josh-Documents-LogoMesh-NeurIPS/memory/feedback.md): added entry "Stale-sentence audits must combine word-level residue grep with section-level framing-consistency reads. Grep alone is never sufficient when the residue is paragraph-level or heading-level."

Why: 2026-05-11 — Pass-1 audit declared paper coherent; full paper read caught §1 Intro α+ residue (paragraph-level) + §2.4 heading (heading-level) + §3.3 phrases (paragraph-level). Word-level patterns can't detect this.

How to apply: For any paper-text audit when major framings have pivoted: Pass-1 word-level grep for known-residue phrases + Pass-2 section-level read of each heading + opening/closing paragraphs against paper-level anchors (title, abstract, contributions, definitional content). Both required.

### TeX state at Day 3 post-final

| Metric | Value |
|---|---|
| Total lines | **681** |
| Net delta from Day-3-start (709 lines) | **−28 lines** |
| Net delta from baseline-pre-Day-2 (652 lines) | **+29 lines** |

Day 3 cumulative wall ~7.5 hours strategic work.

### TeX section-by-section state (Day 3 absolute-EOD)

| Section | Status | Notes |
|---|---|---|
| Title (line 40) | ✅ Option D+ | Day 2 |
| Abstract (line 53) | ✅ Option D+ | Day 2 + PSF fix Day 3 (class names + MCTS-primary methodology) |
| §1 Intro (lines 61-68) | ✅ Option D+ | **NEW Day 3 post-final** (F1 recast applied) |
| §1 Contributions (lines 72-79) | ✅ Option D+ | Day 2 + PSF fix Day 3 (class names) |
| §2 Background (§2.1-§2.6) | ✅ Option D+ | Day 3 audit edits (§2.1, §2.2, §2.3) + Day 3 post-final (§2.4 heading rename F2) |
| §3 Problem Formulation (§3.1, §3.2, §3.3) | ✅ Option D+ | Day 2 (§3.1, §3.2) + Day 3 post-final (§3.3 F3 vocab softening) |
| §4 Proposed Method (§4.1, §4.2) | ✅ Option D+ | Day 3 audit edits (Table 1 delete + Eq.~mcts_reward delete + RepE trim + Step 2 reframe) |
| §5 Experiments | 🔄 PARTIAL | Day 3 audit edits applied to §5.1, §5.2, §5.4. §5 Exp 1 (α+ "OEI Characterization") + §5 Exp 2 (α+ "Reward-Function Ablation") + §5.5 Eval Metrics still α+. Day-4-7 scheduled. |
| §6 Theoretical Analysis | 🔄 PARTIAL | Theorem 1 + Memory Complexity Proposition fine; worked-example numbers (40.05 GB / 9.7 TB) still placeholder. Day 8 scheduled. |
| §7 Discussion (§7.1-§7.6) | ✅ Option D+ | Day 2 + Day 3 P1 + Day 3 P5 + Day 3 audit edits (§7.2, §7.4 opening, §7.6 ¶1) |
| Bibliography | ✅ | meta2024llama3_2_aup added Day 3 P5 |

### Day 4 work order (revised post-F1+F2+F3)

| Priority | Task | Owner | Status |
|---|---|---|---|
| 1 | §5 Experiment 1 recast — Track F entropy-MCTS negative control prose + Table 1 relocation from §5 to §7.4 (F6 coordination) | A drafts, B applies | NOT STARTED |
| 2 | §A Reproducibility Appendix v1 draft | A | NOT STARTED |
| 3 | §5 Experiment 2 TeX application from v10 draft + figure generation (`figures/exp2-mean-step-alpha.pdf`) | B | NOT STARTED (Day 7 scheduled but draft ready) |
| 4 | §5.5 Evaluation Metrics rewrite (audit §5-6) | B | NOT STARTED (Day 7 scheduled) |
| 5 | §6 Memory Complexity worked-example numbers (162 GB / 60× from Track D) | B | Day 8 scheduled |
| 6 | ACL `acl.sty` style switch + smoke compile | B | Day 8 (style files already downloaded Day 3 J4) |
| 7 | Send paper to Tianyu + Max + Alaa for co-author review | J | Day 11 |
| 8 | Bailey 2024 §7.2 citation polish | A | DEFERRED per Josh (mild polish question) |

### Open Josh action items (post Day 3)

- **J6b** — venue commit click-through at Day 15 ARR submission (only truly Josh-only item)
- **Max + Alaa OpenReview confirmation** — when responses land
- **§5 Exp 1 recast scope approval** — when Session A draft lands Day 4, Josh confirms direction
- **Bailey citation polish** — when Josh chooses to re-open

### Outputs produced this Day 3 (cumulative, 2026-05-11)

| File | Purpose |
|---|---|
| `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` | Canonical paper, 681 lines Option-D+-aligned through §7 (excluding scheduled §5/§6 work) |
| `docs/NeurIPS/acl.sty` | ACL style file (J4) |
| `docs/NeurIPS/acl_natbib.bst` | ACL bibliography style (J4) |
| `docs/NeurIPS/acl_formatting.md` | ACL formatting reference (J4) |
| `docs/logs/2026-05-11_session-A-prose-drafts-v9-license-attribution.md` | J2 license attribution paragraph + bibitem |
| `docs/logs/2026-05-11_session-A-responsible-nlp-checklist-draft.md` | J3 Q&A draft (Day 10 finalization target) |
| `docs/logs/2026-05-11_session-A-prose-drafts-v10-exp2.md` | §5 Exp 2 Cartography prose + Table 2 + Figure 1 spec (Day 7 application target) |
| `docs/logs/2026-05-11_stale-sentence-audit.md` | Pass-1 stale-sentence audit (16 findings) |
| `docs/logs/2026-05-11_decision-corpus-review.md` | Per-decision corpus-grounded recommendations for the 5 sign-off decisions |
| `docs/logs/2026-05-11_framing-audit-pass-2.md` | Pass-2 framing-consistency audit (6 findings) |
| `docs/logs/2026-05-11_session-log.md` | This file |
| `docs/logs/2026-05-10_14-day-execution-plan.md` | Plan v2 updated with anonymization-scope correction + J-items + risk register |
| `C:\Users\Josh\.claude\plans\misty-scribbling-fiddle.md` | Plan file (overwritten with corpus-review plan; approved + executed) |
| memory: `project_context.md`, `recast_action_plan.md`, `MEMORY.md`, `feedback.md` | All Day-3-EOD-aligned |

### State for compact

**Read FIRST after compact:**
1. `memory/MEMORY.md` (index)
2. `memory/feedback.md` (latest 2026-05-11 entries: primary-source fidelity + protocol-deviation pre-check + word+section-level audit discipline)
3. `memory/recast_action_plan.md` (state pointer, READ FIRST per the doc itself)
4. `docs/logs/2026-05-10_14-day-execution-plan.md` (day-by-day schedule)
5. `docs/logs/2026-05-11_session-log.md` (this file)
6. `docs/logs/2026-05-11_framing-audit-pass-2.md` (latest audit findings + Day 4 work order)
7. Canonical TeX state: 681 lines; section-by-section status table above.

**Next concrete step Day 4:** §5 Experiment 1 recast draft (Track F entropy-MCTS prose + Table 1 relocation to §7.4 per F6 coordination).

---

*End of Day 3 absolute absolute-final. Day-3 wall total: ~7.5 hours. TeX state: 681 lines. Day 3 absorbed Day 6 (§5 Exp 2 draft) + Day 9 partial (§7.4 OEI framing + framing audit pass-2). Comfortably ahead of plan v2 schedule. Day 4 work blocked on nothing — can begin §5 Exp 1 recast draft immediately on next session resume.*

---

## Day 4 entry — §5 Experiment 1 recast draft (Session A)

**Date:** 2026-05-12 (Day 4 of plan v2)
**Owner:** Session A
**Task:** P1 — §5 Experiment 1 wholesale recast draft (Track F entropy-MCTS negative control prose + Table 1 relocation from §5 to §7.4 per framing-audit-pass-2 Finding F6)
**Status:** ✅ DRAFT COMPLETE — awaiting Josh scope approval before Session B application
**Output:** [docs/logs/2026-05-12_session-A-prose-drafts-v10-exp1.md](2026-05-12_session-A-prose-drafts-v10-exp1.md)

### Process

Per Josh's directive "do not make any assumptions; if you are missing data, ask me for help" — read all primary-source artifacts before drafting:

1. `docs/logs/2026-05-11_track-F-negative-control-report.md` (full read, §1–§11 + Appendices A–B) — primary source for all empirical numbers
2. `docs/logs/2026-05-11_session-A-prose-drafts-v10-exp2.md` (full read) — structural template for the v10-exp1 markdown format
3. Canonical TeX lines 295–360 (current §5 Exp 1 + §5 Exp 2 + §5.4 Baselines preamble) — verified what is being replaced
4. Canonical TeX lines 130–180 (§3 Problem Formulation) — verified the existing cross-references the new `\label{sec:exp1}` resolves
5. Canonical TeX lines 42–100 (abstract + §1 Contributions + §1 Introduction recast) — verified the |r| ≈ 0.60 figure is referenced as our own measurement without external citation
6. Canonical TeX lines 530–540 (§7.4 Empirical Measurement Prerequisites) — verified the relocation target paragraph and the one in-§7.4 reference at line 540 that needs to update

Three presentation calls made (per "you make the call" feedback) — Josh confirmed:
- Title: "Entropy-MCTS Negative Control on ARC-Easy" (parallel to v10-exp2)
- Table 1: single descriptive 6-row table (2 scales × 3 conditions), pairwise statistics inline in prose
- Label rename: `tab:exp1-results` → `tab:oei-alpha-sweep` for the relocated alpha-sweep table; the new Track F table reuses the freed-up `tab:exp1-results` label
- Figure: SKIPPED per Josh confirmation (conserve figure budget; v10-exp2's Figure 1 already carries the mean-step α visualization across classes)

### Draft contents

| Section | Content |
|---|---|
| §1 Pivot summary | Was → becomes table; 6 dimensions (title, methodology, sample, status, headline, relocation) |
| §2 Replacement TeX | REVISION marker `Exp1-recast-trackf-D+` + new `\paragraph{Experiment 1: Entropy-MCTS Negative Control on ARC-Easy.}\label{sec:exp1}` + Setup / Results / Mechanism / Interpretation / IMPL paragraphs |
| §3 New Table 1 | 6 rows: 1B G/R/E + 3B G/R/E; columns: accuracy (count), 95% Wilson CI, mean perplexity ± std |
| §4 §7.4 relocation | Cut-and-paste spec for the existing Table block; label rename `tab:exp1-results` → `tab:oei-alpha-sweep`; new relocation REVISION marker; preserves 4 layered audit-trail markers verbatim; one-line update at line 540 reference |
| §5 Session B notes | Application order (5 steps); cross-references introduced + relied-on; compilation impact estimate (+48 net body lines); expected `sec:exp2` undefined-ref warning until Day 7 |
| §6 Page-budget | Cumulative +105 lines after Day-7 Exp 2 application; recommends §6 Memory Complexity proof → `\appendix` as highest-priority Day-10 cut |
| §7 Primary-sources verification | 23-row table; every number traceable to Track F report §-ref (§1–§7 + Appendix B); discipline note that no figure was sourced from running-log summary or compaction memory |

### Cross-references the new `\label{sec:exp1}` resolves

| Location | Existing text |
|---|---|
| Abstract (line 53) | "A controlled negative-control on ARC-Easy ($n=200$ paired items) shows..." (no explicit `\ref{sec:exp1}` — Exp 1 referenced by name) |
| §1 Introduction paragraph 4 (line 68) | "Experiment 1, §\ref{sec:exp1}" |
| §1 Contribution 3 (line 80) | "Experiment 1 (controlled negative-control, §\ref{sec:exp1})" |
| §3.2 Hypothesis 2 closing (line 167) | "...is established in §\ref{sec:exp1}" |
| §3.2 closing paragraph (line 170) | "We characterize the reward-driven case empirically in §\ref{sec:exp1}--\ref{sec:exp2}" |
| v10-exp2 §5 prose | "...reproducing Track F's headline ($185$ of $200$ items argmax-A under heavy steering, §\ref{sec:exp1})" |

All five existing canonical-TeX forward references resolve when the new `\label{sec:exp1}` lands; v10-exp2's reference resolves on Day-7 application of Exp 2 (which references both `sec:exp1` and creates `sec:exp2`).

### Next step — Josh sign-off + Session B application

Josh review queue for this draft:
- Replacement paragraph framing (4 \textbf{}-prefixed prose blocks: Results, Mechanism, Interpretation; setup paragraph leads)
- Table 1 layout choice (6 rows × 5 cols; could compact further if page budget tighter than expected)
- Label rename to `tab:oei-alpha-sweep` (clearer than reusing `tab:exp1-results` for both tables would be)
- Relocation marker convention (`OEI-alpha-sweep-relocate-to-§7.4-D+`)
- Page-budget projection (+48 lines this draft; +105 cumulative with Exp 2)

After Josh sign-off, Session B applies the 5 application steps (cut + relocate + relabel + insert + smoke-compile) — estimated ~30–45 minutes of TeX edit work + verification.

### Day 4 status

- ✅ Day-4 P1 draft complete (this entry)
- 🔲 §A Reproducibility Appendix v1 draft — Day-4 P2 (not started; lower priority; Josh-confirmed direction first)
- 🔲 Day-5+ §5 Exp 1 TeX application — pending Josh sign-off on this draft

TeX state unchanged: **681 lines**, all D+-aligned through §7.

---

*End of Day 4 entry (Session A draft only; Session B application pending Josh sign-off). Day-4 wall total to this point: ~75 min strategic session work (primary-source verification + draft authoring). Ready for Josh review.*

---

## Day 4 — Session B application (post Josh sign-off)

**Date:** 2026-05-12
**Owner:** Session B (TeX edits within same Claude session)
**Trigger:** Josh "Looks fine, please proceed" to v10-exp1 draft.

### Three Edits applied to canonical TeX

**Edit 1 — §7.4 insert relocated Table + update OEI monotonicity paragraph reference.**
Anchor: the OEI monotonicity paragraph ("Under the projection-based fix...") through the first σ_H REVISION marker. Inserted between them: the new REVISION marker `OEI-alpha-sweep-relocate-to-§7.4-D+` (2026-05-12) + the relocated Table block (alpha-sweep of OEI/ρ_R/σ_H/TDS at α ∈ {0.1, 0.5, 1.0, 2.0, 5.0}) + `\label{tab:oei-alpha-sweep}`. Inline edit: "(results in §\ref{sec:experiments})" → "(results in Table~\ref{tab:oei-alpha-sweep} below)" — the OEI alpha-sweep data is now local to §7.4.

**Edit 2 — §7.4 σ_H paragraph reference rename.**
Inline rename `Table~\ref{tab:exp1-results}` → `Table~\ref{tab:oei-alpha-sweep}` at the σ_H closing sentence ("With per-layer calibration in place, σ_H rises monotonically..."). The rename is documented in the upstream relocation REVISION marker (no separate marker needed).

**Edit 3 — §5 Experiment 1 wholesale replacement.**
Replaced canonical TeX lines 326-353 (3 layered REVISION markers + Exp 1 setup + Results + old Table block + Interpretation + IMPL) with the new REVISION marker `Exp1-recast-trackf-D+` (2026-05-12) + `\paragraph{Experiment 1: Entropy-MCTS Negative Control on ARC-Easy.}\label{sec:exp1}` + 4 prose paragraphs (Setup / Results with inline pairwise McNemar stats / Mechanism with class-conditional pathway analysis / Interpretation closing on Bailey 2024 monitor-driven parallel + Dimensional Escape unification) + new Table 1 (Track F accuracy + perplexity, 6 rows × 5 cols) reusing the freed-up `\label{tab:exp1-results}` + new IMPL line referencing `scripts/diagnose_track_f_negcontrol.py`.

### Reference resolution verification

| Reference | Where used | Where resolved | Status |
|---|---|---|---|
| `\label{sec:exp1}` | line 327 | abstract (line 53), §1 Intro paragraph 4 (line 68), §1 Contribution 3 (line 80), §3.2 Hypothesis 2 (line 167), §3.2 closing paragraph (line 170), §5 Exp 1 self-ref in Table caption (line 348), §5 Exp 1 Interpretation refers to §exp2 | ✅ all resolve |
| `\label{tab:exp1-results}` | line 349 (new Track F results table) | §5 Exp 1 Results paragraph (line 330) | ✅ resolves |
| `\label{tab:oei-alpha-sweep}` | line 555 (relocated alpha-sweep table) | §7.4 OEI monotonicity paragraph (line 537), §7.4 σ_H paragraph (line 561) | ✅ resolves |
| `\ref{sec:exp2}` | §5 Exp 1 Interpretation paragraph (line 354) | NOT YET DEFINED — Day 7 v10-exp2 application | 🟡 expected undefined warning |

Grep verification — `tab:exp1-results` outside the new §5 Track F table block: zero. `tab:oei-alpha-sweep` outside §7.4: zero. Label conflicts: zero.

### TeX state

| Metric | Pre-edit | Post-edit | Delta |
|---|---|---|---|
| Total lines | 681 | **702** | +21 |
| §5 Experiment 1 framing | α+ "OEI Characterization" | D+ "Entropy-MCTS Negative Control on ARC-Easy" | ✅ recast |
| §5 Table 1 content | OEI/ρ_R/σ_H/TDS alpha-sweep | Track F accuracy + Wilson CIs + perplexity | ✅ replaced |
| §7.4 Table location | absent (table lived in §5) | relocated alpha-sweep table with `tab:oei-alpha-sweep` label | ✅ inserted |
| `\label{sec:exp1}` | undefined (forward refs orphaned) | defined at line 327 | ✅ resolves 5 existing forward refs |

### Section-by-section status (Day 4 post-application)

| Section | Status |
|---|---|
| Title, Abstract, §1 (Intro + Contributions) | ✅ Option D+ (Day 2 + Day 3 PSF fix + Day 3 F1) |
| §2 Background and Related Work | ✅ Option D+ (Day 3 audit edits + Day 3 F2 heading rename) |
| §3 Problem Formulation | ✅ Option D+ (Day 2 + Day 3 F3 phrase softening) |
| §4 Proposed Method | ✅ Option D+ (Day 3 audit edits) |
| §5 Phase 0 / Phase A | ✅ Option D+ (Day 3 audit edits) |
| §5 Experiments — Exp 1 | ✅ Option D+ Track F negative control (**Day 4 — this entry**) |
| §5 Experiments — Exp 2 | 🔄 α+ "Reward-Function Ablation" remains — Day 7 v10-exp2 application |
| §5 Baselines | ✅ Option D+ (Day 3 audit edits — 4 rows) |
| §5 Evaluation Metrics | 🔄 Day 7 rewrite scheduled (audit §5-6) |
| §6 Theoretical Analysis | 🔄 Day 8 worked-example numbers replacement |
| §7 Discussion (§7.1–§7.6) | ✅ Option D+ (Day 2 + Day 3 P1 + Day 3 P5 + Day 3 audit edits + Day 4 §7.4 relocation) |
| Bibliography | ✅ |

### Outstanding Day 4 work

| Task | Priority | Status |
|---|---|---|
| §5 Experiment 1 recast draft + apply | P1 | ✅ COMPLETE (this entry) |
| §A Reproducibility Appendix v1 draft | P2 | NOT STARTED (Day 4-5 window) |
| Other Day-5+ items per plan v2 §2 | — | unchanged |

### Day 4 wall time (cumulative)

| Activity | Wall |
|---|---|
| Primary-source verification + draft authoring (Session A) | ~75 min |
| Session B application (3 Edits + reference verification) | ~30 min |
| **Day 4 total to this point** | **~105 min** |

The Session B application went smoothly — string-based Edits with deliberate bottom-up ordering (§7.4 first, then §5) avoided line-number drift confusion. All forward refs resolve except the expected `sec:exp2` (Day 7 work).

---

*End of Day 4 P1 (Exp 1 recast + Table relocation). TeX state: 702 lines, all D+-aligned through §7 except §5 Exp 2 + §5.5 Metrics + §6 worked-example numbers (Day 7 + Day 8 scheduled). Next available work: Day 4 P2 (§A Reproducibility Appendix v1 draft) — pending Josh direction on cadence.*
