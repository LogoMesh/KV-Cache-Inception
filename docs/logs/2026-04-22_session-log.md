# Session Log — April 22, 2026

**Attendees:** Josh (owner)
**Purpose:** Paper revision — citation audit fixes, P1-A through P2-A

---

## Context

This session begins the paper revision phase following the April 21 citation audit. All primary citations were audited in that session; today we work top-to-bottom through the audit matrix applying required paper-text fixes. Source papers for P1-B and P2-A were re-verified from the downloaded TeX sources before any edits were made. P1-A (gao2025hneurons, arXiv:2512.01797) was not downloaded; edits are based on the prior-session audit findings (marked "Not needed — already read" in the matrix, findings confirmed reliable).

All edits are to: `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`

---

## Edits Applied

### Edit 1 — P2-A · §1 Introduction · Generalization qualifier
**Gap:** Our paper stated "models trained to evade a single monitor can generalize to evade unseen monitors" without the qualifier the source paper explicitly states: max-pooling and attention-based monitors are robust and NOT evadable by this attack. Omitting this half of the paired result misrepresents the paper's contribution.

**Source:** Gupta & Jenner (2025), `eval_policy.tex` RQ2: *"RL-Obfuscation on one type of static probe generalises to other types of static probes as well, unless using maxpool probes or attention-based probes."*

**Fix:** Added `"of the same aggregation class---though the paper reports that max-pooling and attention-based monitors remain robust to this attack"` before the citation.

---

### Edit 2 — P1-B · §2.2 · Removed "compliance" from linearly decodable list
**Gap:** "compliance" is not a concept studied in Zou et al. (2023). The paper studies: truthfulness, utility, probability, morality, emotion, lying, power-seeking. The word "compliance" appears exactly once in the paper, in an emotion-control context. Claiming it is linearly decodable per Zou et al. is unsupported.

**Source:** `sections/3-RepE.tex` line 21: *"We begin by extracting various concepts, including truthfulness, utility, probability, morality, and emotion, as well as functions which denote processes, such as lying and power-seeking."*

**Fix:** `"honesty, truthfulness, and compliance"` → `"honesty and truthfulness"`

---

### Edit 3 — P2-A · §2.2 · RL-Obfuscation mechanism and motivation reframe
**Gap (3a):** "via RL" undersells — the paper uses RL weight finetuning (GRPO across epochs), not inference-time perturbation. Our KV-cache steering leaves weights frozen. The connection is analogical, not mechanistic.
**Gap (3b):** "direct empirical motivation" overstates — the primary evasion mechanism in the paper is behavioral output-policy change (~70% of recall drop), with representational redistribution secondary (~30%). Our paper presented the representational finding as the clean story.
**Gap (3c):** Token-level vulnerability was cited without the paper's paired robustness finding (max-pool/attention probes remain robust).

**Source:**
- `training_details.tex`: *"we use Group Relative Policy Optimisation (GRPO)... we use Llama 2 7B and train it to evade monitors"*
- `eval_policy.tex` RQ1: *"the change in policy contributes to approximately 70% of the drop in recall"*
- Abstract: *"token-level monitors are highly vulnerable to this attack while more holistic monitors, such as max-pooling or attention-based probes, remain robust"*

**Fix:** Replaced the two-sentence block with a reframed paragraph: specifies RL finetuning (weight updates), adds the 70% behavioral / ~30% representational split, adds the robustness qualifier for max-pool/attention monitors, and explicitly distinguishes RL-Obfuscation's mechanism from our inference-time frozen-model approach.

---

### Edit 4 — P1-A · §2.3 · σ_H reframed as over-compliance pressure proxy
**Gap:** "We extend this finding to the domain of alignment faking" framed the H-Neuron → strategic deception link as established. Gao et al. (2025) validate H-Neurons for over-compliance and hallucination only. The extension to instrumentally rational strategic deception is a testable assumption.

**Source:** Prior-session audit (arXiv:2512.01797v2), Finding 3: *"Paper validates over-compliance, NOT strategic deception — we overstate the link to alignment faking."*

**Fix:** Replaced the sentence with two sentences that: (a) name σ_H explicitly as an "over-compliance pressure proxy," (b) frame the extrapolation to strategic deception as a testable assumption that the experimental protocol is designed to validate.

---

### Edit 5 — P1-B · §4.1 · Goal-coercion and certainty caveat added
**Gap:** Goal-coercion and certainty probe directions are not validated in Zou et al. (2023). "Goal-coercion" does not appear anywhere in the paper. Presenting these as LAT-validated probe dimensions is unsupported.

**Source:** `sections/3-RepE.tex` — Explore agent confirmed "goal-coercion" is entirely absent from arXiv-2310.01405v4.

**Fix:** Added one sentence immediately after the probe list: *"Note that Zou et al.\ (2023) validate LAT for honesty and truthfulness; the goal-coercion and certainty probe directions are extrapolations of the LAT methodology to novel concept dimensions not directly studied in the source paper."*

---

### Edit 6 — P1-B · §7 Limitations · RepE 1B scale caveat added
**Gap:** The paper's Limitations section mentioned H-Neuron scale issues (smallest validated model is 4B) but said nothing about RepE's scale. Zou et al. (2023) validate only on 7B–70B models. Our 1B surrogate falls outside this range.

**Source:** Explore agent on arXiv-2310.01405v4: *"The smallest LLM tested is 7B (LLaMA-2-7B). There is NO mention of any sub-7B model anywhere in the paper."*

**Fix:** Added sentence to the Third limitation clause: *"Additionally, Zou et al.\ (2023) validated RepE on models of 7B parameters and above; our 1B surrogate falls outside this validated range, and probe reliability at sub-7B scale is an open empirical question."*

---

## Verification

All 6 edits were read back after application. Surrounding context is intact in each case. No TeX structure was disturbed.

---

## What's Next

Continue downward through the audit matrix from P2-B:

| Entry | Citation | Status | Paper fixes needed |
|---|---|---|---|
| P2-B | `greenblatt2024alignment` | ⚠️ 4 gaps | Add caveats: situational awareness required, goals were benign (HHH), scale limit (Opus/3.5 Sonnet only), scratchpad dependence |
| P2-C | `hubinger2024sleeper` | ⚠️ 4 gaps | Add scale qualifier (persistence weakest in small models), clarify deliberately inserted ≠ emergent, strengthen adversarial training characterization |
| P2-D | `belitsky2025kvcache` | ⚠️ 4 gaps | Distinguish static one-shot vs. MCTS iterative in Table 2; add domain qualifier for stability claim |
| P2-E | `llmasjudge2025limitations` | 🚨 CRITICAL | Remove fabricated "0.10–0.20 on normalized scales" figure |

P2-E is the critical fix — it should be prioritized even if it's not next in strict matrix order.

---

## Blockers

None for paper revision. Max's code branch (telemetry matrix) is in progress independently. Gate re-run on H100 still pending before experiments can start.

---

## Peer-Review Hardening Discussion (April 22, continuation)

After applying Edits 1–6, the revision approach was re-examined with peer review explicitly in mind. Key findings and decisions:

### What changed about the approach

**Spot-checks are insufficient for IMPLEMENT citations.** P1-A and P1-B are both IMPLEMENT category — the paper derives formulas and procedures directly from these sources. A NeurIPS reviewer specializing in H-Neurons or RepE will read those sections against the originals. Full source reads of the downloaded TeX are mandatory for all IMPLEMENT entries before applying paper fixes.

**The matrix contains inaccuracies.** The P1-A entry (`gao2025hneurons`) was marked "Not needed (already read)" in the PDF column, implying the source paper was not downloaded. Josh confirmed it IS available at `docs/NeurIPS/Citations/arXiv-2512.01797v2/`. Josh also confirmed he downloaded every paper referenced in the matrix to the Citations folder. A full audit of the Citations folder against the matrix is needed to find other discrepancies.

**Three new issues surfaced from the peer-review lens:**
1. §4.1 states "We apply the LAT procedure from Zou et al." but the code (`whitebox.py`) uses difference-in-means, not PCA — the paper's primary method. A reviewer familiar with RepE will catch this.
2. §2.2's RL-Obfuscation motivation was reframed as "by analogy rather than direct mechanism" (Edit 3). This is accurate but leaves the Orthogonal Escape hypothesis less motivated in §2.2. A forward pointer to §3 (where the geometric motivation is formalized) is needed.
3. Citation quality issues (Deroy/Zenodo, CRSM/GitHub, Okazaki-RAG/personal Gmail) are Medium in the matrix but become High or Critical for NeurIPS submission — reviewers will flag these.

### Decisions made

**Implementation gaps: hold paper fixes, add conspicuous DRAFT—HOLD notes.**
For any gap where the code is wrong and the paper text should reflect the corrected code, do NOT apply the paper fix yet. Instead: (a) add a highly visible block comment in the TeX so neither humans nor AI will miss it, and (b) add a dedicated DRAFT—HOLD section to the matrix. The paper fix wording is drafted and stored in both places, but not applied until the code owner confirms the fix is in.

This applies to:
- GAP-C1-05 (P1-B): Code uses mean-diff; paper should say PCA (once whitebox.py is fixed)
- P1-B Finding 4: Code aggregates all layers; paper should qualify to middle layers only (once kv_mcts.py is fixed)
- §4.2 single-layer steering: Code broadcasts one middle-layer vector; paper implies per-layer d_K^(l) (Phase 3 fix)

**Severity elevations for peer review:**
- P4-A (deroy2026structured): citation quality → **Critical** (Zenodo preprint, single author, no peer review — NeurIPS will flag this)
- P4-D (crsm2025): citation quality → **High**
- P4-E (okazakirag2025): citation quality → **High**
- P1-B Finding 1 (PCA vs mean-diff): → **Critical** for peer review (IMPLEMENT citation)

### Plan for remainder of session

1. ~~Update session log~~ ← this entry
2. ~~Audit Citations/ folder against matrix — find all availability discrepancies~~ ← completed
3. ~~Update citation-audit-matrix.md: add DRAFT—HOLD section, fix P1-A availability, re-rate severities, add §2.2 forward pointer as required fix, add IMPLEMENT citation policy note~~ ← completed
4. ~~Add DRAFT—HOLD block comment in TeX (§4.1, near line 180)~~ ← completed
5. ~~Full read of P1-A source (arXiv-2512.01797v2) — verify all 5 matrix findings~~ ← completed
6. ~~Close session log~~ ← this entry

---

## Session Close (April 22)

### Steps completed

**Citations folder audit:** Three papers were confirmed present that the matrix had marked as unavailable or unreviewed:
- `arXiv-2512.01797v2/` — gao2025hneurons (P1-A): was marked "Not needed (already read)"; now marked as available + full read required
- `arXiv-2306.05685v4/` — zheng2023judging (P4-I): availability column corrected
- `arXiv-2303.16634v3/` — liu2023geval (P4-J): availability column corrected

**Citation audit matrix corrections applied:**
- Added IMPLEMENT citation policy note (full source reads mandatory for peer review)
- Added DRAFT—HOLD section with 3 implementation gap entries (PCA/mean-diff, layer aggregation, single-layer steering)
- Fixed P1-A, P4-I, P4-J availability columns
- Elevated P4-A (Deroy) citation quality: High → Critical
- Elevated P4-D (CRSM) citation quality: Medium → High
- Added §2.2 forward pointer as new required fix in P2-A entry

**TeX DRAFT—HOLD block comment added:** Highly visible box comment inserted before §4.1 LAT paragraph (near line 180) documenting the PCA/mean-diff and layer aggregation gaps with draft fix wording and code fix ownership.

**P1-A full source read (arXiv-2512.01797v2/neurips_2024.tex):**
All 5 prior audit findings confirmed against source. No corrections needed. Key source quotes logged in matrix. One additional low-severity gap noted: code uses fixed TOP_K=50 vs. paper's data-adaptive sparse logistic regression selection.

### What's queued for next session

Continue paper edits from P2-B downward, using the corrected matrix as the guide:

| Entry | Citation | Key fixes |
|---|---|---|
| P2-B | `greenblatt2024alignment` | Add: situational awareness required, goals were benign (HHH), scale limit (Opus/Sonnet only), scratchpad dependence |
| P2-C | `hubinger2024sleeper` | Add: scale qualifier (small models' backdoors ARE removed), deliberately inserted ≠ emergent, adversarial training hid rather than removed backdoors |
| P2-D | `belitsky2025kvcache` | Distinguish static one-shot vs. MCTS iterative in Table 2; add domain qualifier; §2.2 paradigm separation sentence |
| P2-E | `llmasjudge2025limitations` | 🚨 CRITICAL: Remove fabricated "0.10–0.20 on normalized scales" figure |
| P2-A (new) | §2.2 forward pointer | Add forward pointer to §3 after RL-Obfuscation motivation was reframed as "by analogy" |

P2-E remains the highest-priority fix despite not being next in strict order. Matrix now contains the correct state for all entries up through P2-A.

---

## Post-Session Matrix Audit (April 24, retroactive)

Audit of citation-audit-matrix.md against this session log identified one discrepancy: Edits 2, 5, and 6 applied the three P1-B paper-text fixes (remove "compliance," add goal-coercion caveat, add RepE 1B scale caveat to §7 Limitations) but none of them were marked ✅ in the P1-B Required Fixes table. The P1-B status line and summary table row also still read "5 GAPS FOUND" as if nothing had been applied.

**Corrections made (April 24):**
- P1-B status line: updated to reflect 3 text fixes applied + 2 code tasks remain (DRAFT-HOLD)
- P1-B Required Fixes table: Edits 2, 5, 6 marked ✅ APPLIED; code task rows labelled DRAFT-HOLD with owners
- P1-B summary table row: updated from "⚠️ 5 GAPS FOUND" to reflect current state
- Orphan summary table rows (`contextual2025`, `oozeer2025transfer`): corrected from "⚠️ CHECK" to "✅ REMOVED 2026-04-23" (April 23 gap caught in same pass)
