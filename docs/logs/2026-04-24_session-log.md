# Session Log — April 24, 2026

**Attendees:** Josh (owner)
**Purpose:** External verification of open blockers; brief finalization; push to main

---

## Context

This session did not apply TeX edits. It was a verification and planning session: processing the results of a web browsing agent dispatched to resolve five open questions, updating the owner status brief with findings, and preparing the repo for the next morning's work sprint. The brief (`docs/logs/2026-04-24_owner-status-brief.md`) is the primary artifact of this session.

---

## Decisions Recorded

### OEI Scope — Option B (April 24)
**Decision:** Josh has chosen Option B — reframe Orthogonal Escape as an emerging preliminary observation, not a validated third contribution. Formal Phase B validation is framed as future work.

This closes the open A1 item from the brief. Paper text changes (§1, §3, §5 Experiment 1, §7 Limitations) are deferred until Josh confirms exact reframing language and the OEI α-sweep diagnostic (B6) is run first.

---

## Verification Report — Five Items Resolved

A web browsing agent was dispatched with a self-contained prompt covering five open questions. Full report saved at `docs/decisions/NeurIPS Paper Verification Request.txt`.

### Item 1 — gpt-oss-20b (✅ CONFIRMED)
- Model ID `openai/gpt-oss-20b` exists on HuggingFace under the official `openai` org
- Fully public, no gating, no NDA required
- Licence: Apache 2.0 (SPDX: `Apache-2.0`)
- Architecture: MoE — **21B total parameters, 3.6B active per forward pass**. MXFP4 quantized; fits in 16GB VRAM.
- Has a configurable reasoning effort parameter (low/medium/high) — must be controlled in experiments
- **Follow-up required:** CLAUDE.md states "32 experts, 4 active/token" — unverified. Update before H100 runs by reading `config.json` from the HuggingFace repo.
- **Memory implication:** MXFP4 halves the M_KV floor vs. bf16. Memory Complexity Proposition (DRAFT-HOLD, ~line 454) must use the precision actually used in experiments.

### Item 2 — NeurIPS 2026 Track (🚨 CRITICAL FINDINGS)
- D&B track renamed to **"Evaluations & Datasets (E&D)"** for 2026
- E&D requires dataset fully built and submitted by May 6 — planned dataset = automatic desk rejection
- E&D FAQ explicitly states: methods/architecture papers belong in the **Main Track**
- **Decision required:** Pivot to Main Track. Paper contributions are methodological; Croissant dataset is Phase 4 (not started). E&D is not viable.
- **Abstract deadline: May 4 (AoE)** — gates full paper submission on May 6. Separate OpenReview portals; cannot switch tracks after registering.
- **Template zip downloaded and verified:**
  - Page limit: **9 pages of content** (hard — papers over limit not reviewed)
  - References, acknowledgments, checklist, appendix do NOT count toward 9 pages
  - Appendix: no page limit, ZIP only (no separate PDF)
  - Style file: `neurips_2026.sty` only
  - **Checklist mandatory** (16 sections, desk rejection if missing). Four flagged for this paper: Claims (1), Safeguards (11), Licenses (12), LLM usage (16)

### Item 3 — CAA Citation (✅ RESOLVED)
- arXiv:2312.06681 confirmed: "Steering Llama 2 via Contrastive Activation Addition"
- **Published venue:** ACL 2024 (Volume 1: Long Papers), pages 15504–15522. Outstanding Paper Award.
- **Primary author:** Nina Rimsky (published name) = Nina Panickssery (legal name change). Use Rimsky to match version of record.
- Full author list: Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Turner
- **Paper fix required (G2):** Table 2 has plain-text `"Rimsky et al., 2024"` with no `\cite{}` and no `\bibitem`. Add proper `\bibitem{rimsky2024steering}` and replace cell with `\cite{rimsky2024steering}`.

### Item 4 — Deroy 2026 (⚠️ PARTIALLY RESOLVED)
- Deroy IS peer-reviewed — accepted at "Controversies of AI Society" conference (Copenhagen, April 9–10, 2026, ADD/SCAI research projects)
- BUT: venue is social-science-adjacent, not a hard-ML conference — may still be questioned at NeurIPS
- Zenodo preprint not updated with venue info — still appears as a raw preprint to anyone who clicks the link
- **Paper fix (G4):** Wherever Deroy is the sole citation for an alignment faking claim, add `\cite{greenblatt2024alignment}` alongside it. Consider replacing the broad-survey role with Park et al. (2024), Patterns/Cell Press: "AI deception: A survey of examples, risks, and potential solutions"
- ChameleonBench (OpenReview, peer-reviewed) also available as an empirical supplement

### Item 5 — CRSM (✅ RESOLVED)
- Confirmed: no peer-reviewed paper, no arXiv preprint, no workshop paper for CRSM or Pomilon Intelligence Lab
- Repository explicitly self-describes as "EXPERIMENTAL PROTOTYPE" built via AI-assisted workflow
- **Paper fix (G3):** Remove from primary bibliography. Downgrade to `@software` citation in a footnote. Replace any body-text peer-review anchor with `\cite{belitsky2025kvcache}` (already in bibliography).

---

## NeurIPS 2026 Template — Key Facts

Downloaded from `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`.

Contains: `neurips_2026.tex`, `neurips_2026.sty`, `checklist.tex`

**Checklist sections (all mandatory, answers Yes/No/NA with 1–2 sentence justification):**

| # | Section | Status for this paper |
|---|---|---|
| 1 | Claims | ⚠️ DRAFT-HOLD gaps + Option B reframing affect this answer |
| 2 | Limitations | ✅ §7 exists — add G1 caveats |
| 3 | Theory assumptions and proofs | ⚠️ Theorem 1 + Memory Proposition need appendix sketch |
| 4 | Experimental result reproducibility | ⚠️ Experiments not yet run |
| 5 | Open access to data and code | ⚠️ Decide release scope before May 6 |
| 6 | Experimental setting/details | Standard |
| 7 | Experiment statistical significance | Standard |
| 8 | Experiments compute resources | Standard |
| 9 | Code of ethics | ✅ NA |
| 10 | Broader impacts | ⚠️ Alignment faking dual-use — write a considered statement |
| 11 | Safeguards | 🚨 Adversarial payload library is a misuse surface |
| 12 | Licenses for existing assets | ⚠️ Llama 3.2, gpt-oss-20b (Apache 2.0), RepE codebase |
| 13 | New assets | ⚠️ State code/dataset licence and release timeline |
| 14 | Crowdsourcing / human subjects | ✅ NA |
| 15 | IRB approvals | ✅ NA |
| 16 | Declaration of LLM usage | ⚠️ Claude used extensively — must declare |

---

## Brief Updated

`docs/logs/2026-04-24_owner-status-brief.md` was updated throughout this session to reflect:
- §0.3: gpt-oss-20b marked ✅ CONFIRMED with architecture details and MXFP4 note
- §0.4: Track pivot marked 🚨 DECISION REQUIRED with full verified Main Track specs, checklist table, portal warning
- §0.1: Deadline updated to reflect May 4 abstract wall (10 days) + May 6 full paper (12 days)
- §A3: Marked ✅ RESOLVED; two follow-up actions noted (update CLAUDE.md, read config.json before B1)
- §G: Added G2 (CAA citation fix), G3 (CRSM downgrade), G4 (Deroy supplement); renumbered G sections 1–5 by priority
- Sequential summary: abstract registration moved to top; resolved items marked; G items updated
- B7: Moved from orphaned position after Section E into Section B where it belongs

---

## What's Queued for Next Session

Start from the top of the brief's sequential summary:

| Priority | Task | Notes |
|---|---|---|
| 🚨 IMMEDIATE | Register abstract on OpenReview — Main Track portal | Must be done by May 4 (AoE) |
| 🚨 IMMEDIATE | Confirm track pivot to Main Track | Decide before registering |
| A2 | Confirm experiment scope (which of 5 are in for May 6) | Gates all H100 planning |
| A4 / B7 | Decide B7 Option 1/2/3 (single-step vs multi-step rollout) | Gates Experiment 2 design |
| G1 | Add three Limitations caveats (Claims 12, 13, 14) | Unblocked, no code needed |
| G2 | Fix CAA `\bibitem` (Rimsky et al. 2024, ACL) in Table 2 | Unblocked |
| G3 | Downgrade CRSM to `@software` citation | Unblocked |
| G4 | Supplement Deroy with Greenblatt wherever sole citation | Unblocked |
| B1 | Fix MoE monitoring (`hneuron_monitor.py:195`) | Read config.json first |
| B2–B5 | Silent failures, graders, analysis script | Can begin now |
| B6 | OEI α-sweep diagnostic on 1B | Gates Experiment 1 design and Option B wording |

---

## Blockers

None for paper text or planning. H100 experiments blocked on B1–B7 + Max's PR.

---

## Revised Experiment Strategy — April 24 (Continuation)

**Context:** Josh met with Max and agreed that experiments must be trimmed aggressively to make the May 6 deadline. The paper is currently 10 pages — one over the hard 9-page limit — before any experimental results are added. The owner-status-brief was written assuming the full methodology stays intact; it needs to be updated to reflect the actual submission plan.

### Problem with the original plan

The original plan had seven sequential dependencies with no slack (brief §0.1, line 18). gpt-oss-20b as the primary experiment target is the highest-risk dependency: B1 (MoE monitoring fix) not implemented, H100 allocation not confirmed, MXFP4 precision discrepancy unresolved, routing architecture unverified. Any single slip ends the submission. The 1B → 7B → 20B scaling narrative requires H100 for every data point above 1B.

### Proposed revised plan (agreed April 24)

**Models:** Llama 3.2 1B and 3B only. Both run on the local RTX 3060 12GB. No H100 dependency for submission experiments.

**License confirmed:** Llama 3.2 Community License explicitly permits academic research and publication. Attribution required: cite the model in the paper, declare the license in checklist item 12. The 700M MAU commercial threshold is irrelevant for a research group.

**Experiments in scope for May 6:**

| Experiment | Model | Hardware | Status |
|---|---|---|---|
| Theorem 1 validation (reversibility) | Llama 3.2 1B | RTX 3060 | ✅ Already done — `measure_lipschitz_drift.py`, 200 cycles, 0.00e+00 |
| Experiment 1 — OEI α-sweep | Llama 3.2 1B + 3B | RTX 3060 | Blocked on B3 (OEI logging), B4 (graders), B5 (analysis script) |
| Experiment 2 — Latent signal quality (scoped) | Llama 3.2 1B | RTX 3060 | Requires B7 Option 2 decision; no additional code needed |

**Experiments cut from §5 (move to Future Work paragraph):**
- Experiment 3 (Memory Efficiency) — requires sparse accumulator fix (GAP-C2-03+04), no script
- Experiment 4 (Evaluation Reproducibility) — no script, no plan
- Experiment 5 (Cross-Model Transfer) — no script, Procrustes not implemented

**Paper sections to cut:**
- §8 Timeline and Milestones — research proposal artifact, not a paper section; cut entirely (~0.5 page)
- Experiments 3–5 from §5 — move to Future Work (~0.5–1 page saved)
- Scaling narrative (1B → 7B → 20B) — replaced by 1B → 3B local story
- Croissant/dataset content — one sentence in Future Work

**gpt-oss-20b status:** Remains in the paper as the intended Phase B target (20B scale validation), but not as a submission-time experiment dependency. Described as future work.

**B7 resolution:** Experiment 2 takes Option 2 — reframe from full-response ASR comparison to latent signal quality per compute unit. Sidesteps the single-step vs. full-response category error without requiring code changes. Paper text at tex:324–326 needs rewriting before the run.

**Expected page count after cuts:** ~7–8 pages, leaving room for Experiment 1 results table and Theorem 1 empirical summary.

### What must be updated next

1. `docs/logs/2026-04-24_owner-status-brief.md` — restructure experiment scope, remove H100 as submission dependency, update model targets, mark gpt-oss-20b as Phase B only
2. `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` — cut §8, reframe §5, rewrite Experiment 2, add 1B→3B framing, update Future Work

These are the two briefing files Josh flagged. The brief will be harder to update (structural changes to many sections); the TeX will require careful section-by-section revision.

---

## Session Continuation — Max Briefing Update (April 24)

### Context

Josh confirmed the revised experiment strategy came out of a meeting with Max. Their main concern: the number of code gaps Max found made it unclear whether they could fix everything AND run all experiments in time. Josh and Max agreed to each independently ask their models what experiments to cut, then compare answers. Josh's session produced the proposal documented above.

Josh also clarified that Max is working from `docs/contribution-1-telemetry-matrix.md` — but the owner-status-brief covers everything in that file, so it doesn't need a separate update.

### Files Read This Continuation

- `docs/logs/2026-04-24_owner-status-brief.md` — full read (lines 1–683). Confirmed full picture of experiment scope, blocking gaps, Max's assignment, and sequential critical path.
- `docs/max-briefing-2026-04-21.md` — full read. Max's assignment is Contribution 1 only: `hneuron_monitor.py`, `whitebox.py`, `telemetry_matrix.py`. Three active priorities: GAP-C1-02 (per-neuron ReLU), GAP-C1-03 (calibration sets), GAP-C1-11 (step counter). One critical issues note referencing gpt-oss-20b (GAP-C1-04 — MoE monitoring fix) that needed updating. One paper fix entry referencing the old CAA citation key (`panickssery2023steering`) that G2 will replace.

### Max Briefing Changes Applied

Josh's instruction: "your original briefing document will be updated (new lines at the top of the document, plus notes throughout so all you will have to do is ask your model to reread the same brief)." The proposal is labeled as PROPOSED pending Max's review — Josh confirmed this explicitly: "the briefing will label my experiment cut idea as proposed so that you and your agent can scrutinize it before deciding if its the right call."

Three types of changes made to `docs/max-briefing-2026-04-21.md`:

**1. New PROPOSED section at the top of the document (before the existing April 24 UPDATE block):**
- Heading: `PROPOSED — April 24, 2026 (Session Continuation): Experiment Scope Pivot`
- Blockquote at top marking it as a proposal pending Max's review, not a finalized decision
- "In a nutshell" (3 sentences): why gpt-oss-20b is high-risk as a submission target, what the proposed re-scope is (1B+3B local), and what gpt-oss-20b's new role is (Phase B)
- One-line confirmation that Max's three priorities (GAP-C1-02, 03, 11) are unchanged
- Full experiment scope table (Theorem 1 validation ✅ done, Experiment 1 blocked on B3/B4/B5, Experiment 2 scoped to Option 2)
- Experiments cut list (3, 4, 5 → Future Work paragraph)
- Paper sections being cut (§8 Timeline, Exp 3–5 from §5, scaling narrative, Croissant content)
- B7 Option 2 explanation: why single-token telemetry vs. full-response ASR is a category error, and why Option 2 (scope to latent signal quality) is the correct reframing with no code changes needed
- Llama 3.2 Community License confirmation: academic research and publication explicitly permitted; declare in checklist item 12

**2. Inline note in Critical Issues / GAP-C1-04:**
Added blockquote note after the existing GAP-C1-04 description clarifying that MoE monitoring is now Phase B work, not a May 6 blocker. The submission experiments run on Llama 3.2 1B and 3B. Max should still flag it in his PR brief for Phase B planning.

**3. Inline note in paper fix priority list / item 5 (CAA citation):**
Added note that G2 will replace `\cite{panickssery2023steering}` with `\cite{rimsky2024steering}` (ACL 2024, Nina Rimsky). No action needed from Max — just don't reference the old key in new code comments.

### What Remains for Next Session

- `docs/logs/2026-04-24_owner-status-brief.md` — the harder update. Many sections reference H100, gpt-oss-20b, and the 5-experiment plan. Sections requiring changes: §0.1 (critical path), §0.3 (gpt-oss-20b status), §A2 (experiment scope decision), §B1 (MoE monitoring priority), §B7 (mark decided — Option 2), §D (sparse accumulators → Phase B), §E (experiment sequence), Sequential Summary (major restructure), checklist table item 12 (Llama 3.2 Community License).
- `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` — paper revisions (G1–G4 text fixes, §8 cut, §5 reframe, Experiment 2 rewrite).
- G1–G4 paper text fixes are still unblocked and can begin once the brief is updated.

### Max Briefing Final State (committed b55571a)

Three rounds of edits applied to `docs/max-briefing-2026-04-21.md`:
1. PROPOSED experiment scope pivot section added at top (labeled as proposal pending Max's review)
2. Inline notes: GAP-C1-04 now Phase B only; G2 incoming citation key change
3. Correction: inline note in "Where We Are" section clarifying contribution-3 OEI section is stale (paper changed Eq. 10 to delta-based formula), while contribution-1 and contribution-2 remain accurate gap analyses.

---

## Session Continuation — Brief Update + Paper Text Edits (April 24, new session post-compact)

### Context

New Claude session started from the compacted handoff. Josh's instruction: review the full session log before touching anything; then proceed carefully, one edit at a time.

Files read at session start: `docs/logs/2026-04-24_session-log.md` (full), `docs/logs/2026-04-24_owner-status-brief.md` (full, 683 lines).

---

### Owner-Status-Brief Update (commit b15c4e9)

Applied 23 inline-only notes to `docs/logs/2026-04-24_owner-status-brief.md` reflecting all decisions from the revised experiment strategy. Per the inline-only rule, every note was appended to an existing line — no new lines added. Sections updated:

| Section | Change applied |
|---|---|
| §0.1 (line 18) | Critical path revised: gpt-oss-20b and H100 removed as submission gates; new path is B3+B4+B5 → Exp 1 (1B+3B, RTX 3060) + Exp 2 (1B, Option 2) |
| §0.3 (line 36) | Heading marked Phase B Only; submission uses Llama 3.2 1B+3B |
| §0.4 item 12 (line 87) | License corrected to "Llama 3.2 Community License"; gpt-oss-20b noted as Phase B only |
| §A2 (lines 155, 170) | Heading and "Decision needed" paragraph both marked ✅ DECIDED with full scope |
| §B1 (lines 206, 208) | Heading and Severity line both marked Phase B Only |
| §B7 (line 342) | Marked ✅ DECIDED: Option 2 (latent signal quality per compute unit) |
| §D (line 424) | D1+D2+D3 Phase B Only; D4 still potentially needed as paper text fix |
| §E (lines 467, 472, 473, 474) | Section heading + three H100 experiment steps annotated/revised |
| Sequential Summary (9 lines) | A2, B7, B1, D1, D2, D3, H100 blocks, E sequence — all annotated with decisions |

---

### G1 — Three Limitations Caveats Applied

**File:** `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`
**Location:** §7 Limitations, line 553 (end of paragraph) + line 554 (revision marker)
**Status:** Applied, flagged `[REVISION | G1 | 2026-04-24]` for audit

Three sentences appended to the end of the existing Limitations paragraph, after the existing "Fourth…" sentence. The existing text was not modified — only extended. The three additions:

- **Claim 12 (σ_H confound risk):** "Fifth, $\sigma_H$ may conflate hallucination pressure, mode-collapse, and strategic masking under corrective pressure unless discriminant controls are applied; the extension from hallucination association to instrumentally rational deception is a testable assumption rather than a settled equivalence."
- **Claim 13 (OEI false negatives):** "Sixth, OEI may produce false negatives if deceptive redistribution is nonlinear or manifold-structured rather than concentrated along linear null-space directions, in which case $\text{OEI}\to 0$ while the model still evades the linear monitor."
- **Claim 14 (evidence burden):** "Finally, the proposed framework requires an explicit validation chain linking continuous latent interventions to discrete deceptive-intent outcomes; Phase~B experiments are designed to provide this evidence but have not been run at time of submission."

**Note on Claim 12 and existing text:** The existing "Third" item in Limitations already partially covers the σ_H/deception assumption ("open theoretical liability"). The new "Fifth" sentence adds the specific failure-mode enumeration (hallucination, mode-collapse, masking) and the discriminant-controls framing, which the existing sentence does not include. The audit should verify there is no redundancy that weakens either sentence.

**Revision marker in TeX:** `% [REVISION | G1 | 2026-04-24 | Claims 12+13+14 added to §7 Limitations — pending audit]`
To find all revision markers: `grep -n "REVISION" docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`

---

### G2 — CAA Citation Fix Applied

**File:** `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`
**Locations:** Table 2 cell (line 435) + bibliography (line 676)
**Status:** Applied, flagged `[REVISION | G2 | 2026-04-24]` for audit

**What was found vs. what the brief described:** The brief said Table 2 had plain text with no `\cite{}`. The actual file had `\cite{panickssery2023steering}` already present, but with wrong author ordering (Panickssery & Rimsky), wrong year (2023 arXiv, not 2024 ACL), and a stale bibitem pointing to the arXiv preprint.

**Changes applied:**

1. **Table 2 cell (line 435):** `Panickssery \& Rimsky et al., 2023~\cite{panickssery2023steering}` → `Rimsky et al., 2024~\cite{rimsky2024steering}`

2. **Bibliography (line 676):** Replaced entire `panickssery2023steering` entry with:
   - Key: `rimsky2024steering`
   - First author: N. Rimsky (published name, version of record)
   - Full author list: N. Rimsky, N. Gabrieli, J. Schulz, M. Tong, E. Hubinger, A. Turner
   - Venue: ACL 2024, Volume 1 Long Papers, pages 15504–15522, Bangkok, Thailand
   - (Previously pointed to arXiv:2312.06681, 2023 — now points to published venue)

**Revision marker in TeX:** `% [REVISION | G2 | 2026-04-24 | panickssery2023steering replaced with rimsky2024steering (ACL 2024) — pending audit]`
