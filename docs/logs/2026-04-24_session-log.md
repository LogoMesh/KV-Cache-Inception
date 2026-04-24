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
