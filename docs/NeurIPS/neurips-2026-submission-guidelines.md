# NeurIPS 2026 Submission Guidelines

**Last verified:** 2026-04-16 (from official NeurIPS Main Track Handbook + neurips_2026.sty)
**Next re-verify:** Week of April 28 — fetch live NeurIPS pages before abstract deadline
**Status key:** ✅ CONFIRMED (from official sources) · ⚠ UNVERIFIED (claimed, not confirmed) · ❓ UNKNOWN (not yet found)

**Authoritative sources (in priority order):**
1. Live NeurIPS 2026 pages: https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets
2. Downloaded handbook: `docs/NeurIPS/Main Track Handbook 2026_files/`
3. Official style file: `docs/NeurIPS/neurips_2026.sty` (from official ZIP 2026-04-16)
4. Official checklist: `docs/NeurIPS/checklist.tex`

**Do NOT treat as authoritative:**
The AI-generated guides (`NeurIPS Submission Guide (KV-CAche-Inception).md` and `NeurIPS Submission Guide (general).md`) contain confirmed URL errors and unverified platform claims. Use this document instead.

---

## 1. Track and Framing

✅ **Track name:** Evaluations & Datasets (E&D) — renamed from "Datasets & Benchmarks" to
signal that evaluation methodology is a first-class scientific contribution.

✅ **In-scope work:**
- Red-teaming and adversarial evaluation protocols
- Novel benchmark and evaluation datasets
- Evaluation methodology critique and analysis
- Dataset auditing, benchmark analysis
- Reproducibility and documentation standards

✅ **KV-Cache Inception qualifies on:** red-teaming methodology + novel dataset of alignment
stress-test traces.

✅ **Review model:** Double-blind by default.
- Opt-out to single-blind via `\usepackage[eandd, nonanonymous]{neurips_2026}` — only permitted
  if dataset anonymization is technically impossible.
- Camera-ready (after acceptance): `\usepackage[eandd, final]{neurips_2026}`

---

## 2. Deadlines

✅ All confirmed from official handbook.

| Milestone | Date | Notes |
|---|---|---|
| OpenReview portal opens | April 15, 2026 | Already open |
| Abstract submission | May 4, 2026, 11:59 PM AoE | AoE = UTC−12 |
| Full paper + data artifacts | May 6, 2026, 11:59 PM AoE | Data is primary, not supplementary |
| Author notification | September 24, 2026 | |
| ❓ Camera-ready | Not yet published | Check NeurIPS site after notification |

**Internal deadlines (recommended):**
- Paper: May 4 (2 days before hard deadline)
- Data artifacts: May 5 (1 day before)
- OpenReview profile check: **now** (see Section 3)

---

## 3. OpenReview Admin

✅ **Co-author profiles:** All co-authors (Josh, Max, Bakul, Asuka) must have active OpenReview
profiles before the May 4 abstract deadline.

⚠ **Non-institutional emails (@gmail.com):** Activation takes up to 2 weeks moderation.
Two weeks from April 18 = May 2. This is extremely tight — **all team members should verify
or create their profiles this weekend.**

✅ **What profiles need:**
- Full Education & Career history
- Conflicts of interest listed
- Publication history (DBLP integration recommended)

✅ **Author list freezes after May 6.** No additions or removals after full paper submission.

**Action items:**
- [ ] Josh — verify OpenReview profile is active
- [ ] Max — verify OpenReview profile is active
- [ ] Bakul — verify OpenReview profile is active
- [ ] Asuka — verify OpenReview profile is active
- [ ] Decide: who owns the submission portal account?

---

## 4. Formatting

✅ **LaTeX preamble (submission):**
```latex
\documentclass{article}
\usepackage[eandd]{neurips_2026}
```

✅ **Camera-ready:**
```latex
\usepackage[eandd, final]{neurips_2026}
```

✅ **Current paper status:** `docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex` already
uses correct preamble (applied 2026-04-16).

✅ **Rules:**
- No custom geometry, margin hacks, or font-size overrides
- 9-page main content limit (camera-ready gets +1 page)
- Type area: 5.5 inches wide × 9 inches tall; 1.5 inch left margin
- Fonts: Type 1 or Embedded TrueType only
- PDF size limit: 50MB

✅ **PDF assembly order (mandatory):**
1. Main content (max 9 pages)
2. References (unlimited)
3. Appendices (optional; must not contain critical material)
4. NeurIPS checklist (mandatory; template in `docs/NeurIPS/checklist.tex`)

---

## 5. Anonymization

✅ **Rules:**
- No identifiable author metadata in submission PDF or any submission artifact
- Self-citations: third-person neutral style only ("The authors of [X] showed...")
- No personal GitHub links in manuscript
- Data hosting URLs must be anonymized (e.g., anonymous HuggingFace repo or Dataverse link)
- Code supplements must be anonymized

⚠ **UNVERIFIED:** Exact protocol for anonymizing a HuggingFace-hosted dataset during review
period. No official guidance found on the NeurIPS E&D page specifically. Options: anonymous
HuggingFace org, private-during-review then public, or Dataverse embargo. Team needs to
decide and verify before creating the hosting repository.

---

## 6. Code Supplement

✅ **Rules:**
- Separate 100MB ZIP file (not embedded in main PDF)
- Must include: training code, evaluation code, dependency specification
- Should be executable in isolation (Docker / VM / isolated cloud)
- If not executable, explain why (proprietary hardware, non-open libraries)
- Must be fully anonymized at submission time

**Action item:** Decide whether we submit a code supplement and who owns the anonymization pass.

---

## 7. Paper Checklist

✅ **Mandatory** for E&D track. Must be placed at the absolute end of the PDF (after appendices).

✅ **Template:** `docs/NeurIPS/checklist.tex` (extracted from official NeurIPS 2026 ZIP).

✅ **"No" answers are acceptable** — rigorous justification is expected and sufficient.
Reviewers use checklist answers as a primary evaluation tool, not a pass/fail gate.

❓ **UNKNOWN:** Whether the E&D track has a distinct checklist from the main track.
The downloaded `checklist.tex` appears to be the main track version. Verify on the E&D
call-for-papers page before final submission.

---

## 8. Accessibility

✅ **Required:**
- No color-only encoding for any critical distinction in figures
- Colorblind-safe palettes; minimum 4.5:1 contrast ratio for text, 3:1 for graphics
- All figures must remain legible after two-column paper scaling
- All figures must remain interpretable after grayscale printing

**Note for experiments:** Any figures generated from H100 runs (confusion matrices, OEI curves,
telemetry heatmaps) must use accessible color schemes from the start.

---

## 9. Ethics and Dual-Use

✅ **Required in paper:**
- Broader impact statement addressing: deception risk, fraud vectors, harassment potential
- Environmental impact / compute footprint transparency

✅ **For dual-use artifacts (our intervention traces and attack payloads):**
- Controlled access gateway required post-acceptance
- Explicit intended-use and non-intended-use language in paper and dataset metadata

✅ **Ethics review process:**
- Ethics reviewers can flag papers for additional review
- Ethics reviewers do NOT have authority to reject — escalation goes to program chairs
- Flagging is not rejection; rigorous ethics language in the paper is the correct mitigation

---

## 10. Open Questions (Team Decides at Meeting)

| Question | Owner | Decision |
|---|---|---|
| Are all four co-author OpenReview profiles active? | All | |
| Who owns the submission portal account? | Josh | |
| Hosting platform for dataset? (see data requirements doc) | Team | |
| Anonymization strategy for dataset URL? | Team | |
| Code supplement: yes or no? | Team | |
| If yes: who owns code anonymization pass? | TBD | |
| Is the E&D checklist distinct from main track checklist? | Josh | Verify on neurips.cc |

---

## Update Log

| Date | What changed | Source |
|---|---|---|
| 2026-04-16 | Initial version created from official handbook + sty | Handbook txt, neurips_2026.sty |
