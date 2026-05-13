# Framing-Consistency Audit Pass 2 — Canonical TeX

**Author:** Session A
**Date:** 2026-05-11 (Day 3 EOD; following PSF fix for §7.3 item 5 class names + abstract + §1 Contributions)
**Audience:** Josh (sign-off + Day 4 work guidance); Session B (when applying)
**Trigger:** Josh directive 2026-05-11 — "The stale-sentence audit missed paragraph-level framing residue (§1 Intro, §3.3). Run a second audit pass before Day 4 work begins — explicit section-level framing-consistency check, not just word-level α+/Candidate-A residue grep."

---

## Method (different from Pass-1 stale-sentence audit)

Pass-1 audit was word-level: grep for α+/Candidate-A phrases ("adversarial datasets", "alignment-breaking interventions", "over-compliance pressure proxy"). It found 16 word-level findings but missed paragraph-level framing residue.

Pass-2 audit reads each section heading + opening paragraph + closing paragraph against the paper-level Option D+ thesis as expressed in four anchors:

| Anchor | Verbatim claim |
|---|---|
| **A. Title** (line 40) | "Reversible KV-Cache MCTS: A Memory-Bounded Probe of Dimensional Escape in Frozen Language Models" |
| **B. Abstract concluding claim** (line 53) | "Reversible MCTS is therefore a high-precision diagnostic probe of frozen-model representational geometry rather than a reasoning-improvement tool." |
| **C. §1 Contribution 3** (line 87) | "Empirical demonstration of Dimensional Escape under continuous-latent search... a fundamental geometric pathology of MCTS in overparameterized latent spaces." |
| **D. §3.2 Definition `def:dim_escape`** (line 147) | "Dimensional Escape occurs when the optimization process resolves $\mathcal{L}$ through trajectories whose induced perturbation $\Delta\mathbf{h}$ has substantial component in $\text{Null}(\mathbf{P}_{\text{semantic}})$..." |

For each section, I flag: (a) framing inconsistencies — does it pitch the work as something other than the four anchors say? (b) implicit framing presumptions that contradict Option D+; (c) place-holder framings not updated post-pivot.

---

## Findings

### Finding F1 — §1 Introduction lines 60-67 — BLOCKING

Already flagged in 2026-05-11 paper-read message. Listed here for completeness + suggested scope of recast.

**The §1 opening 3 paragraphs frame the entire paper as alignment-faking detection.** Reading top-to-bottom, a reviewer sees the abstract pitch Dimensional Escape + diagnostic probe (anchor B), then immediately encounters §1 paragraphs pitching alignment-evaluation framework development (α+ Candidate-A framing). The §1 Contributions block 30 lines later then snaps back to Option D+.

Specific α+ phrases:

| Line | Phrase | Why α+ |
|---|---|---|
| 61 | "alignment evaluation one of the most consequential open problems in machine learning" | Frames work as alignment evaluation. Anchor B says "high-precision diagnostic probe... rather than reasoning-improvement tool" — neither alignment evaluation nor reasoning improvement is the headline goal. |
| 65 | "The question is no longer whether models can fake alignment, but how to develop evaluation frameworks that can detect and stress-test this capability at the mechanistic level." | Frames work as detection-framework development; anchor C says "empirical demonstration of Dimensional Escape" |
| 67 | "Rather than observing a model's internal states and hoping to detect deception, we systematically perturb those states... while a model may learn to mask deceptive intent... informational constraints that manifest as detectable signatures in orthogonal subspaces" | "deceptive intent" + "detectable signatures" — directly contradicts §7.3 Limitations item 4 (latent-response diagnostics, not deception detectors) |

**Suggested recast scope:** Replace paragraphs 1-3 (lines 61, 63, 65) and the line-67 paragraph entirely. Lead with: (i) test-time scaling + inference-time interventions on frozen LMs as the open research surface, (ii) the geometric question of whether such interventions navigate the semantic manifold or exploit unconstrained latent dimensions, (iii) we find they generally don't, and we characterize the pathology. The existing Greenblatt/Hubinger/Gupta-Jenner citations can stay but as "monitor-driven Dimensional Escape was hypothesized in this prior work; we test the parallel reward-driven case here" rather than as motivation for "we detect alignment faking."

**Also**: CRITIQUE NOTE at TeX:69-75 (% comment) still references "OEI/TDS during KV-cache steering correspond to behaviorally deceptive outputs" — stale, but comment-only. Either delete with the §1 recast or leave as audit-trail.

**Severity:** BLOCKING — reviewer-readable internal inconsistency.

---

### Finding F2 — §2.4 subsection heading "MCTS for LLM Red-Teaming" — RECOMMENDED

The subsection heading at TeX:115 reads "MCTS for LLM Red-Teaming". The body content discusses prior MCTS work (Wu 2025 MPA, DAMON, Li 2025 ACE-Safety) which is positioned in the red-teaming literature — that part is fine. But the heading frames OUR paper's relationship to this prior work as "MCTS for red-teaming." Under Option D+, our work is NOT red-teaming — it's a probe of Dimensional Escape.

Pass-1 audit had this as §2-5 OPTIONAL (deferred to camera-ready). Pass-2 reclassifies as **RECOMMENDED** under framing-consistency.

**Suggested fix:** Rename heading to "MCTS for Language Model Search" or "MCTS in Language Model Optimization". Body content unchanged.

**Severity:** RECOMMENDED (heading-level, low-touch).

---

### Finding F3 — §3.3 lines 180-184 — RECOMMENDED

Already flagged in 2026-05-11 paper-read message.

**Current text contains two α+-vintage phrases:**

Line 182: "We argue that the **Orthogonal Escape phenomenon** is intrinsically a geometric problem in continuous latent space. The redistribution of **deceptive encodings** into the null space of a linear monitor is a continuous vector operation..."

Under §3.2's Dimensional Escape unified framing:
- "Orthogonal Escape phenomenon" should be "Dimensional Escape pathology" (or scope to monitor-driven sub-case explicitly)
- "deceptive encodings" should be "internal representational divergences" (the same softening applied to §2.1 line 98 today)

**Suggested fix:**

> "We argue that the Dimensional Escape pathology---in both monitor-driven and reward-driven manifestations---is intrinsically a geometric problem in continuous latent space. The redistribution of internal representations into the null space of a low-rank scoring projection is a continuous vector operation that cannot be meaningfully probed by discrete token-level perturbations. Detecting and characterizing this redistribution requires an algorithm that can navigate the continuous geometry of the model's representation space directly."

**Severity:** RECOMMENDED — internal inconsistency with §3.2 + §7.3 item 4.

---

### Finding F4 — §1 Introduction CRITIQUE NOTE at TeX:69-75 — OPTIONAL

The CRITIQUE NOTE at lines 69-75 (`%` comment block) is α+ residue:

> "The empirical burden is high: the paper will need an explicit validation chain linking continuous latent interventions to discrete deceptive intent outcomes. Reviewers will ask: does an increase in OEI/TDS during KV-cache steering correspond to a measurable increase in behaviorally deceptive outputs?"

Comment-only — doesn't compile. But under Option D+ we no longer claim a "validation chain to deceptive intent" (Hypothesis 2 is reward-driven Dimensional Escape, intent-decoupled). The CRITIQUE NOTE would mislead a human reader of the .tex source.

**Suggested fix:** Bundle with §1 Intro recast (F1) — delete the CRITIQUE NOTE comment block.

**Severity:** OPTIONAL — comment cleanup.

---

### Finding F5 — §5.1 "Phase 0: Architectural Decoupling" heading — OPTIONAL

Heading is project-chronicle vintage ("Phase 0" / "Phase A" naming from internal sprint planning, applied 2026-04-16 era). A paper reader doesn't have the "Phase 0 / Phase A / Phase B" naming convention as context. Section is structurally OK but the heading reads as project-chronicle rather than paper structure.

**Suggested fix (camera-ready Phase 3):** Rename "Phase 0: Architectural Decoupling" → "Implementation Decoupling" or merge with §5.2 under "Implementation Substrate". Same for "Phase A: Local Prototyping on Lightweight Surrogates" → "Local Prototyping on Llama 3.2-1B".

**Severity:** OPTIONAL — readability polish, defer.

---

### Finding F6 — §7.4 line 539 forward-reference will be stale post-Day-4 — KNOWN COORDINATION

Line 539: "Under the projection-based fix, OEI varies monotonically with steering magnitude on Llama~3.2-1B-Instruct (results in §\ref{sec:experiments})."

After Day-4 §5 Exp 1 recast (audit §5-3), §5 Experiment 1 becomes Track F entropy-MCTS negative control. The OEI alpha-sweep table (current Table 1 at TeX:339-355) needs to relocate — naturally into §7.4 itself, where the prose already discusses these numbers (§7.4 paragraphs 2-4 reference σ_H 0.228→0.330, ρ_R variations, etc.).

**Suggested fix:** Day-4 §5 Exp 1 recast task should include relocating the OEI/ρ_R/σ_H/TDS alpha-sweep table from §5 Experiment 1 into §7.4 Empirical Measurement Prerequisites. Audit §5-3 already noted this coordination but didn't make it explicit at the line-level.

**Severity:** KNOWN coordination (Day-4 task).

---

## Summary

| # | Location | Type | Severity | Action |
|---|---|---|---|---|
| F1 | §1 Introduction paragraphs 1-3 (lines 60-67) | Section-level α+ framing | BLOCKING | Day 4 — full paragraph recast |
| F2 | §2.4 heading "MCTS for LLM Red-Teaming" | Heading-level framing | RECOMMENDED | Day 4-5 — heading rename |
| F3 | §3.3 lines 180-184 phrases | Paragraph-level α+ vocabulary | RECOMMENDED | Day 4 — bundle with F1 recast |
| F4 | §1 CRITIQUE NOTE comment (lines 69-75) | Comment-only stale | OPTIONAL | Day 4 — delete with F1 |
| F5 | §5.1 / §5.2 Phase 0 / Phase A headings | Project-chronicle vintage | OPTIONAL | Camera-ready Phase 3 |
| F6 | §7.4 line 539 forward-ref + Table 1 relocation | Cross-section coordination | KNOWN | Day 4-5 with §5 Exp 1 recast |

### Pass-2 vs Pass-1 comparison

| Audit pass | Method | New findings |
|---|---|---|
| Pass-1 (stale-sentence audit) | Word-level α+/Candidate-A residue grep | 16 findings; missed paragraph-level + heading-level framing |
| Pass-2 (this) | Section-level framing-consistency vs 4 anchors (title, abstract, §1 Contribution 3, §3.2 Definition) | 6 findings; 1 BLOCKING, 2 RECOMMENDED, 2 OPTIONAL, 1 KNOWN coordination |

### Methodology lesson (memory candidate)

Pass-1 missed F1, F2, F3 because the audit method was word-level grep. Pass-2 caught them by reading each section's framing against paper-level anchors. **For future paper audits: pair word-level + section-level passes.** Word-level catches lexical residue; section-level catches paragraph-level framing residue.

This is a candidate addition to `memory/feedback.md`: "Stale-sentence audits must include section-level framing-consistency in addition to word-level residue grep, especially when major paper framings have pivoted (α+ → D+ here)."

### Recommended Day-4 work order (revised post-Pass-2)

1. Apply F1 §1 Introduction recast (BLOCKING; draft + apply in sequence)
2. Apply F3 §3.3 lines 180-184 softening (RECOMMENDED; bundle with F1)
3. Apply F2 §2.4 heading rename (RECOMMENDED; trivial)
4. Apply F4 §1 CRITIQUE NOTE delete (with F1 application)
5. Apply F6 §7.4 + §5 Exp 1 recast coordination — Session A drafts §5 Exp 1 (Track F prose) + relocates Table 1 from §5 to §7.4
6. Defer F5 to camera-ready Phase 3

### Open question

None requiring Josh sign-off — F1 BLOCKING needs your scope approval on the §1 Intro recast (the actual prose), but the diagnosis is unambiguous. F2-F4 are clear refinements to apply.

---

*End of framing-consistency audit pass 2. 6 findings, 1 BLOCKING. Methodology gap surfaced for `memory/feedback.md` candidate addition.*
