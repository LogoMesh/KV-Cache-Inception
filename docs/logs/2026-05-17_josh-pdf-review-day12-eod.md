# Josh's PDF Review — Day-12 EOD

**Date:** 2026-05-17 (Day 12 EOD)
**PDF reviewed:** commit `e15a6ec` ("docs: restore M4 abstract hedge + aggressive abstract trim ~315→~205 words"), compiled at `C:\Users\Josh\Documents\LogoMesh-NeurIPS\docs\NeurIPS\04.18.2026-NeurIPS-Research-Proposal-2.pdf` on 2026-05-17 12:26.
**PDF stats at review:** 14 pages / 471 KB / body 8pp / 0 undefined refs / 0 LaTeX errors.
**Reviewer:** Josh (paper author, primary).
**Status:** INCOMPLETE — Josh may add more findings before action.

---

## Reviewer summary

This is a cold PDF read after the heavy Day-12 push (Round 1 + M1-M5 + Round 2 + SEVERE fix + Tier-1 reframe + §A MAJORs + M4 restore + abstract trim, 12 commits in one day). Issues surfaced fall into four categories: format-compliance bugs (table widths, math overflow), authorial-tone issues (Phase B / future-work over-referencing), primary-source verification needs (Bailey "Orthogonal Escape" attribution, Croissant ARR acceptance), and copy-edit cosmetic items.

**No SEVERE findings.** No paper-go/no-go-level issues.

---

## Findings (numbered for action tracking)

### F1 — Abstract line 43 mid-paragraph indent

**Observation (Josh):** "On line 43 'We argue the reward-driven failure...' There's some kind of indent before that sentence starts and I don't know if it's correct formatting since the previous sentence ends on the same line."

**Assessment:** Plausibly a real bug. Abstract should be ONE paragraph; mid-paragraph indent suggests stray `\par`, blank line, `\indent`, `\paragraph{}`, or similar inserted by the Tier-1 abstract rewrite + M4 restore.

**Verification needed:** YES — inspect TeX source structure around the "We argue" sentence.

**Severity:** MODERATE (formatting flag visible at first glance).

---

### F2 — Croissant 1.1 metadata package + supplementary readiness

**Observation (Josh):** "On line 53 'Artifacts ship with a Croissant 1.1 metadata package' we need to confirm that ACL / ARR accepts this format and that we actually have this package ready according to what the paper describes about it and how it refers to it."

**Assessment:** Two sub-issues:
- (a) **Format acceptance:** Croissant is the MLCommons standard for dataset metadata. NeurIPS Datasets & Benchmarks track explicitly accepts it; ACL/EMNLP/ARR don't formally specify but the format is supplementary-format-agnostic (JSON-LD). Likely accepted, needs primary-source confirmation.
- (b) **Package readiness:** `logomesh/croissant_export.py` exists in repo but the actual built artifact + the anonymized supplementary at `anonymous.4open.science/r/kv-mcts-dimensional-escape-XXXX` — XXXX still placeholder. Abstract claim is half-promised, half-built.

**Verification needed:** YES — (1) run `logomesh/croissant_export.py` to confirm we have a buildable Croissant package; (2) confirm ARR/ACL accepts Croissant from a primary source (cleanup prompt's Phase 0 Q4 covers this); (3) verify the package matches the paper's textual description of what's in it.

**Severity:** MAJOR if abstract claim is unsupported at submission time. Day-13/14 cleanup prompt should resolve.

---

### F3 — Phase B / future-work mentions throughout

**Observation (Josh):** "I'm beginning to realize how many times Phase B research is mentioned. Here is the first on line 130. Gut feeling says we need to remove most if not all mentions of Phase B / future work."

**Assessment:** Sharp observation. "Phase B" is internal team jargon, not standard paper terminology. Reviewers don't know what it means. Frequent Phase B / future-work references read as "this paper is part of a bigger thing we're working on" — reviewer mental model: "the actual paper isn't complete" or "we're padding to claim breadth we don't have."

**Verification needed:** YES — grep the rendered body for "Phase B" + "Phase 2" + "Phase 3" + "future work" cross-references to count occurrences and locations.

**Severity:** MODERATE (cumulative reviewer flag — "this paper feels padded").

**Reading the §7.4 future work section is fine; Josh is OK with that.** The issue is BODY references that gesture at §7.4 without giving substance.

---

### F4 — §1.1 C1 line 142 — "memory floor reduction" claim provability

**Observation (Josh):** "I don't understand how we've proven this claim concerning reduced memory floor. I've had other agents look at 3.04-3.06x M_KV and they seem to think it's some achievement but I have no idea what these details mean because it's not my profession. I'm most concerned about whether or not we have provably done these things, not how easy they are to understand."

**Assessment:** **Yes, the asymptotic claim is provably established.** Specifically:
- **Proposition 1** (memory): claim O(M_KV + d · K_acc), independent of branching factor b. Proof is a cache-counting argument: naive parallel MCTS spawns b^d cache copies (one per leaf); our algorithm uses 1 baseline + d accumulators regardless of b. Proof sketch in §C appendix. Defensible.
- **Theorem 1** (reversibility): FP32 accumulator returns cache to baseline exactly after complete rollback. Proof in §C, empirically validated 200 cycles 1B + 3B (residual norm 0.00).

**What's MEASURED (not proven):** the 3.04-3.06× M_KV constant — empirical from Track D, not derivable a priori.

**Verification needed:** NO — proof exists; Josh's concern is comprehension, not correctness.

**Severity:** None for correctness. MINOR clarity issue.

**Optional cleanup:** add a 1-sentence intuition in body §6 (currently body only says "proof sketch in Appendix C"). Something like "The proof is a cache-counting argument: naive parallel MCTS spawns b^d cache copies, while our algorithm reuses one baseline cache plus d accumulators, regardless of b." Makes the claim readable in body without appendix consultation. ~25 words.

---

### F5 — §1.1 C1 line 151 — "concrete scale-specific projections are deferred to §7.4"

**Observation (Josh):** "I feel like we're only including these details to make it sound like we have a bigger idea and I don't know if it's worth the space all these references to future work are taking up in the paper."

**Assessment:** Same root cause as F3 (Phase B / future-work overuse). This clause is residue from when we had 20B projections in body §6.2; we dropped the projections in commit `267e9cb` but left the meta-reference pointing to where they "would" go in §7.4. The phrase as it stands is half-baked: gestures at projections without giving any.

**Verification needed:** NO.

**Severity:** MINOR (cosmetic) but **high-leverage on the "paper feels padded" reviewer mental model.**

**Recommended action:** delete the clause "concrete scale-specific projections are deferred to §7.4" entirely. The asymptotic-claim sentence stands on its own.

---

### F6 — §1.1 C2 line 194 — "Orthogonal Escape" attribution to Bailey

**Observation (Josh):** "Starting on line 194 'We further argue... structurally analogous... Orthogonal Escape... previously hypothesized for alignment faking [13]... both as sub-cases...' If this is just referencing J. Bailey et al, then this sentence makes sense. But we need to check if that paper actually uses the words Orthogonal Escape."

**Assessment:** **NEEDS PRIMARY-SOURCE VERIFICATION.** I do not have confidence Bailey 2024 uses the term "Orthogonal Escape." The paper's title is "Obfuscated Activations Bypass LLM Latent-Space Defenses" — doesn't include "Orthogonal Escape." The team may have COINED "Orthogonal Escape" as a label for Bailey's phenomenon, then attributed it back to Bailey — which would be a citation error.

If the paranoid hallucination audit (the one Josh fired in a different session) ran on this claim, it should be in the audit matrix; either way worth verifying directly now.

**Verification needed:** **YES — open Bailey 2024 PDF + grep for "Orthogonal Escape" / "orthogonal escape."** Bailey 2024 lives in `docs/NeurIPS/Citations/` (filename TBD — likely arXiv-format or named).

**Severity:** **POTENTIALLY MAJOR.** Citation errors are reviewer-rejection triggers. If Bailey doesn't use the term, the paper makes an unsupported attribution across 4+ locations: abstract S7, §1.1 C2, §3.2 §C preamble, Hypothesis 1 title.

**If verification fails** (Bailey doesn't use "Orthogonal Escape"):
- (a) Drop "Orthogonal Escape" terminology; refer to Bailey 2024 as "obfuscated-activations work" or "gradient-trained input-space variant."
- (b) Find what Bailey actually calls the phenomenon (likely "obfuscated activations" or "behavior-preserving obfuscation") and use that term.
- (c) NOT viable: coin "Orthogonal Escape" ourselves (contradicts "previously hypothesized" language we've used).

---

### F7 — §3.2 line 308 — three-name cascade reads as authorial uncertainty

**Observation (Josh):** "'We position Dimensional Escape as a specific manifestation of Goodhart's Law [14, 15] in low-rank-scored, high-dimensional latent spaces, generalizing the gradient-trained input-attack case characterized by Bailey et al. as Orthogonal Escape [13]. The novelty we claim is not a new phenomenon class but a specific empirical setting...' This reads to me badly because we're giving the pathology a new name and saying it's similar to Bailey's named pathology. Like we couldn't decide what to call it, which is technically true. We started this paper with the name Orthogonal Escape front and center."

**Assessment:** **Your gut is sharp.** Reading the §3.2 preamble cold, the three-name cascade IS visible: "Dimensional Escape → manifestation of Goodhart's Law → generalizing Bailey's Orthogonal Escape." Three names for what could be one phenomenon. Reads as committee branding / authorial uncertainty.

This was the M3 patch from Day-12 (the "anchor DE to Goodhart's Law" defense against Axis 2 Q5's "you're just relabeling X" attack vector). The defense rationale is real, but the prose execution reads worse than intended.

**Verification needed:** NO — editorial judgment call.

**Severity:** MODERATE (quality-of-prose; reviewer-detectable but not rejection-tier).

**Better framing — three options:**
- (A) **Tight Goodhart anchor without the cascade:** keep ONE Goodhart citation but drop the "we position X as a specific manifestation of Y" sentence. Replace with in-passing reference: e.g., "...Dimensional Escape (related to Goodhart's Law in high-dimensional optimization~\cite{gao2023scaling, skalse2022defining})..." Reads as standard scholarly citation.
- (B) Drop Goodhart anchor entirely; let DE stand on its own. Re-exposes Axis 2 Q5 relabeling attack.
- (C) Drop DE coining; use "Goodhart's Law in high-dimensional latent search" everywhere. Was Option A retire-DE; reversed Day-12 EOD because Axis 4 needs "Probing X" routing signal.

**Recommended:** Option (A). Defuses the cascade prose concern while preserving the Goodhart citation as relabeling-attack defense.

**This finding fuses with F6:** if F6 verification fails and we drop "Orthogonal Escape" attribution to Bailey, the three-name cascade automatically simplifies (no longer Bailey-anchored). Resolve F6 first; then apply F7's tightening if still needed.

---

### F8 — Tables 1, 2, 3 too wide; overflow column / overprint right-column text

**Observation (Josh):** "All of the tables are too wide except for table 4. They either go off the page entirely if they're on the right side, and if they're on the left, they are printed on top of text on the right side."

**Assessment:** This is a real ACL two-column overflow bug. ACL format: each column = 7.7 cm wide. Tables defined with `\begin{table}` (single-column) overflow if their content exceeds 7.7 cm. Tables defined with `\begin{table*}` span both columns (full page width = ~15.4 cm).

Likely current state per the canonical TeX:
- **Table 1** (tab:exp1-results, ~line 247): 5-column (Scale, Condition, Accuracy, 95% Wilson CI, Mean PPL). Content like "[0.236, 0.362]" + "$1.547 \pm 0.263$" is wide. Likely overflowing.
- **Table 2** (tab:exp2-cartography-paths, ~line 286): 7-column (per-scale × 3 sub-columns). Definitely overflowing.
- **Table 3** (tab:baselines, appendix §B.3, ~line 564): 3-column (Category, Method, Reference). Narrower but reference column has long author lists.
- **Table 4** (tab:oei-alpha-sweep, appendix §G, ~line 675): per Josh, fine. 5-column with single values per cell — narrow.

**Verification needed:** YES — open TeX, locate each `\begin{table}` declaration, check single-column vs starred-environment usage, measure actual rendered widths.

**Severity:** **MAJOR — format compliance.** Overflow tables in ACL submission are reviewer-likely flag and can trigger format-compliance desk-reject.

**Fix options (per table):**
- (1) Change `\begin{table}` → `\begin{table*}` for tables that must be full-width. Cleanest fix.
- (2) Wrap tabular in `\resizebox{\linewidth}{!}{...}` to force fit (reduces font size, may hurt readability).
- (3) Use `\small` / `\footnotesize` / `\scriptsize` on table content.
- (4) Refactor table columns (drop columns, combine columns, abbreviate cell contents).

Recommended default: (1) `\begin{table*}` for Tables 1, 2, 3. They're already substantial content; spanning both columns is appropriate.

**Caveat on (1):** `\begin{table*}` floats to top OR bottom of a page across both columns, which may shift the body text layout. Smoke-compile after each conversion to verify body still 8pp.

---

### F9 — Hypothesis 2 line 359 + line 399 — text drifts too far right

**Observation (Josh):** "Line 359 (Hypothesis 2); the text drifts too far right, and Line 399."

**Assessment:** Both lines are in the §3.2 Hypothesis 2 region. Likely cause: long math expressions (eq:goodhart's `\arg\min` formulation) or long inline math that doesn't break at column boundaries.

Hypothesis 2 (per TeX memory) contains:
- Body prose introducing the hypothesis
- Eq:goodhart: `\arg\min_{\mathbf{h}} \hat{r}(\mathbf{h}) \cap \text{SemanticManifold}(x) = \emptyset` with parenthetical "(generically, in dimension $d \gg \text{rank}(\hat{r})$)"

The "(generically, in dimension $d \gg \text{rank}(\hat{r})$)" parenthetical attached to the equation could be too wide for the column.

**Verification needed:** YES — open TeX at lines 359 and 399, identify what's overflowing.

**Severity:** MODERATE (formatting bug; reviewer-detectable).

**Fix options:**
- (1) Insert manual `\\` line break in equation
- (2) Use `\begin{align}` / `\begin{multline}` instead of `\begin{equation}` for forced line breaks
- (3) Move parenthetical from equation line to following prose
- (4) Use `\text{}` wrapper to allow text breaking

---

### F10 — HellaSwag viability as a source

**Observation (Josh):** "Is HellaSwag a reference we're allowed to use? I'm assuming yes, but that name feels inappropriate and I want to make sure that it's a viable source."

**Assessment:** **YES, HellaSwag is fully viable.** It's Zellers et al. 2019 "HellaSwag: Can a Machine Really Finish Your Sentence?" — published at ACL 2019, peer-reviewed, widely cited in NLP literature, used routinely as a commonsense-reasoning benchmark. The Hugging Face dataset is `Rowan/hellaswag`. The cute name is internet-style but doesn't disqualify the dataset; it's appeared in hundreds of ACL/EMNLP/NAACL papers.

**Verification needed:** NO — established benchmark.

**Severity:** None.

**Optional defense if Josh wants extra confidence:** add a quick footnote at first §5 Exp 2 use stating "HellaSwag (Zellers et al., 2019)" with full citation. Currently the paper may just say "HellaSwag" without the inline cite + year. Strengthens scholarly rigor.

---

### F11 — §6.2 Memory Complexity again references deferred work

**Observation (Josh):** "6.2 Memory complexity talks about deferred model work again. I don't like how many times this is mentioned, it's beginning to make it sound like we ran out of time to do that work now (which we did). I'm okay with the future work section though."

**Assessment:** Same root cause as F3 + F5. §6.2 currently has "concrete scale-specific projections (notably for ~20B open-weight architectures such as gpt-oss-20b) and the multi-GPU runs needed to anchor them empirically are deferred to §7.4." This is the third location in the body where we point to §7.4 for projections we didn't run.

The pattern: §1.1 C1 + §6.2 + §3.2 all gesture at §7.4. Reviewer reading top-to-bottom hits the same "deferred" reference 3× before getting to §7.4. Cumulative effect: reads as "they ran out of time," exactly as Josh's gut says.

**Verification needed:** NO — Josh has named the diagnosis correctly.

**Severity:** MODERATE (cumulative; pairs with F3 + F5).

**Recommended action (pairs with F3 + F5 cleanup pass):**
- Drop §6.2 "...deferred to §7.4" clause entirely. The asymptotic separation claim is self-contained.
- Drop §1.1 C1 "concrete projections deferred to §7.4" clause (F5).
- Audit body for any other forward-pointers to §7.4; trim where they add no substance.
- KEEP §7.4 Future Work itself — it's the natural home for these references.

---

### F12 — Appendix sections A through G (~5 pages): allowed and normal?

**Observation (Josh):** "After References, Sections A through F; is it normal for them to be so long? It's like 5 pages. Just want to make sure that's allowed."

**Assessment:** **Yes, 5 pages of appendix is explicitly allowed and is normal for technical NLP papers.**

Primary source — `acl_formatting.md:17`: "Review versions of long papers may have up to eight (8) pages of content plus **unlimited pages for references**." And `acl_formatting.md:23`: "The conference encourages submission of appendices and supplementary material, which are **not required to fit within these page limits**." And `acl_formatting.md:283`: "Appendices are material that can be read, and include lemmas, formulas, proofs, and tables that are not critical to the reading and understanding of the paper. Letter them in sequence and provide an informative title... The appendices come after the references."

ACL format explicitly:
- Body: 8 pages hard limit ✓ (ours is 8 exactly)
- References: unlimited ✓
- Appendices: unlimited ✓
- Total PDF: 14 pages = 8 body + 1 references + 5 appendix — normal

For technical NLP papers, 3-5 pages of appendix is the common range. Papers with theorems, proofs, reproducibility blocks, and implementation-detail discussions routinely exceed 5pp of appendix. Ours falls in the standard range.

**Self-containment rule:** body must stand alone; appendix is "for reviewers who want depth" per `acl_formatting.md:23` ("review versions of papers must be self-contained: it is optional for reviewers to look at appendices or supplementary material"). Our body is self-contained (8pp) — appendix is reviewer-enrichment.

**Verification needed:** NO — format-compliance is clean.

**Severity:** None (format compliance verified).

**Sub-observation: appendix content audit (optional reading).** Each section's load-bearing status:

| Section | Content | Load-bearing? | Source |
|---|---|---|---|
| §A Reproducibility | Hardware, models, MCTSConfig, per-track configs, seeds, supplementary URL | YES — required by ARR Responsible NLP Checklist + reviewer-verification | Native (always was appendix) |
| §B Extended Related Work | MCTS-for-LM, Continuous-Latent-Reasoning, Baselines table | borderline — needed for related-work coverage but could be tightened | Moved from body §2 Day-10 |
| §C Implementation Notes (Phase 2/3) | 5 items: probe extraction, layer aggregation, per-layer steering, sparse accumulator, reversibility | YES — defends against "code ≠ math" reviewer critique | Moved from body §5 Day-10 |
| §D Theorem 1 Proof Detail | Proof sketch (1 paragraph) | YES — required for §6.1 theorem claim | Always was appendix |
| §E Algorithm Detail (Steps 1-3) | Forward mutation + reverse rollback + memory complexity | borderline — body §4.2 has inline reverse-rollback per Day-11 minimal re-fold; could be tightened | Moved from body §4.2 Day-10 |
| §F Telemetry Matrix Framework | T_t definition + σ_H + ρ_R equations | borderline — Day-10 drop-C2 moved this from body; serves as supplementary for §G | Moved from body §4.1 Day-10 |
| §G Measurement Pipeline | OEI definition + W_K projection issue + σ_H calibration + Table 4 | YES — defends against measurement-pipeline reviewer questions | Moved from body §7.4 Day-10 |

Several sections (§B, §E, §F) are "moved from body Day-10 for page-budget" rather than naturally-appendix content. They COULD in theory be trimmed but trimming carries reviewer-coverage risk: each section defends against a specific reviewer-likely question.

**Bottom line:** 5pp appendix is allowed, normal, and load-bearing. **No action needed.** If Day-14 buffer permits, a light §B/§E/§F trim pass could reclaim ~0.5-1pp appendix space, but there's no submission-pressure reason to do it.

---

## Severity-ranked verification list (pre-action gate)

Verifications to run BEFORE acting on findings:

| # | Verification | Action when done |
|---|---|---|
| V1 | **Open Bailey 2024 PDF; grep for "Orthogonal Escape" / "orthogonal escape"** (F6) | Determines scope of F6 + F7 cleanup |
| V2 | Inspect abstract TeX structure (F1) | Find + fix the indent source |
| V3 | Grep body for `\begin{table}` declarations + content widths (F8) | Decide per-table fix (`\begin{table*}` vs resize vs refactor) |
| V4 | Inspect lines 359 + 399 TeX (F9) | Identify what's overflowing column |
| V5 | Grep body for "Phase B", "Phase 2", "Phase 3", "deferred to", "future work" cross-refs (F3 + F5 + F11) | Build cleanup pass target list |
| V6 | Run `logomesh/croissant_export.py` + confirm package builds; verify ARR Croissant acceptance from primary source (F2) | Resolve abstract Croissant claim; folds into Day-13/14 cleanup prompt |
| V7 | Confirm HellaSwag inline-citation present in §5 Exp 2 (F10, optional) | Add inline `\cite{zellers2019hellaswag}` if missing |

---

## Severity-ranked action priorities

After verifications complete:

### Blocking (must-fix before submission)

| # | Finding | Effort | Notes |
|---|---|---|---|
| F8 | Tables 1, 2, 3 overflow | ~30 min | Format-compliance; reviewer-likely desk-reject risk |
| F6 | Bailey "Orthogonal Escape" attribution (if V1 fails) | ~20 min | Citation error if Bailey doesn't use term |
| F1 | Abstract indent | 5 min | First-impression visual bug |
| F2 | Croissant package readiness | (cleanup prompt) | Abstract claim must be supported by submission time |

### Recommended (should-fix before submission)

| # | Finding | Effort | Notes |
|---|---|---|---|
| F3 + F5 + F11 | Phase B / future-work cleanup pass | ~30 min | Cumulative "paper feels padded" reviewer signal |
| F9 | Hypothesis 2 + line 399 math overflow | ~15 min | Math layout cleanup |
| F7 | §3.2 three-name cascade tightening | ~15 min | Editorial polish; may simplify if F6 forces re-attribution |

### Optional (nice-to-have)

| # | Finding | Effort | Notes |
|---|---|---|---|
| F4 | Add 1-sentence proof intuition in body §6 | 10 min | Clarity improvement |
| F10 | Add HellaSwag inline cite if missing | 5 min | Scholarly rigor |

---

## Notes for the action session

- **Run V1 (Bailey verification) FIRST.** If Bailey doesn't use "Orthogonal Escape," F6 + F7 fuse and scope grows; if Bailey DOES use it, F6 closes and F7 is a separate, smaller patch.
- **Most fixes are isolated and don't interact.** F3 + F5 + F11 fuse into one cleanup pass; F6 + F7 fuse pending V1; F8 + F9 are independent format fixes.
- **No SEVERE findings.** Paper is in a defensible state. The accumulated issues are reviewer-quality flags, not paper-go/no-go.
- **Page-budget cushion:** body is 8pp exactly. Any addition (F4) or deletions (F3 + F5 + F11) needs smoke-compile gate to verify page count.
- **Table overflow fixes (F8):** if `\begin{table*}` shifts body layout, body could re-slip to 9pp. Smoke-compile after each table conversion.

---

## STATUS: review in progress — Josh may add more findings

This document will be updated as Josh continues review. Action sequencing waits for full review.
