# Day-9 Page-Budget Cut Plan v1 — Authored 2026-05-15 (Opus 4.7, Session A)

**Date:** 2026-05-15 (Day 9 of plan v2 execution; calendar 2026-05-15)
**Owner:** Session A (audit + ranking author)
**Status:** AWAITING JOSH SIGN-OFF before Day-10 application
**Decision basis:** Day-8 EOD ACL switch surfaced body rendering at ~16 ACL two-column pages against an 8-page review budget (per [acl_formatting.md:17](../NeurIPS/acl_formatting.md):17). Plan v2 §8's pre-planned cut (§6 proof → \appendix) saves ~1 page; we need ~8.
**TeX baseline:** Commit `76d63f1` (Day-8 evening, §6 honest-framing refinement). 776 source lines. .aux ToC + .log file used as primary sources for page anchors.
**Peer-review lens:** Per Josh's directive 2026-05-15, this audit was conducted from a neutral double-blind-reviewer perspective (not from a "preserve what we built" perspective). Peer-review observations that surfaced during the audit are recorded in §7 as additions for the Day-12 parallel-review plan.

---

## §1. Current page distribution (primary source: .aux ToC + .log)

Page-start anchors below come from `04.18.2026-NeurIPS-Research-Proposal-2.aux` `\@writefile{toc}` entries. "Est. pages" = page-of-next-section minus page-of-this-section (the section roughly spans those pages; coarse where multiple sub-sections share a page).

| Section | Page start | Est. pages | Span |
|---|---:|---:|---|
| §1 Introduction | 1 | 2 | pp. 1–2 |
| §1.1 Contributions | 2 | 1 | within p. 2 |
| §2 Background and Related Work | 3 | 2 | pp. 3–5 (§3 starts mid-p. 5) |
| §2.1 Alignment Faking | 3 | <1 | within p. 3 |
| §2.2 Representation Engineering | 3 | 1 | pp. 3–4 |
| §2.3 H-Neurons | 4 | <1 | within p. 4 |
| §2.4 MCTS for Language Model Search | 4 | 1 | pp. 4–5 |
| §2.5 Continuous Latent Reasoning | 5 | <1 | within p. 5 |
| §2.6 LLM-as-a-Judge | 5 | <1 | within p. 5 |
| §3 Problem Formulation | 5 | 1.5 | pp. 5–6 |
| §3.1 Inference-Time Latent-Space Analysis | 5 | <1 | within p. 5 |
| §3.2 Dimensional Escape (Def + 2 Hypotheses + Eqs) | 5 | 1 | pp. 5–6 |
| §3.3 Discrete Text-Space Bottleneck | 6 | 1 | pp. 6–7 |
| §4 Proposed Method | 7 | 1.5 | pp. 7–8 (§5 starts mid-p. 8) |
| §4.1 Telemetry Matrix | 7 | <1 | within p. 7 |
| §4.2 Reversible MCTS (incl. §4.2.1 + §4.2.2 + 6 paragraphs) | 7 | 1.5 | pp. 7–8 |
| §5 Experimental Design | 8 | 3.5 | pp. 8–12 |
| §5 Implementation Notes (5-item list) | 8 | <1 | within p. 8 |
| §5.1 Phase 0 | 8 | <1 | within p. 8 |
| §5.2 Phase A | 8 | <1 | pp. 8–9 |
| §5.3 Experiments (Exp 1 + Exp 2; tables + figure) | 9 | 2.5 | pp. 9–11 |
| §5.4 Baselines (table only; no body prose comparing) | 11 | <1 | within p. 11 |
| §5.5 Evaluation Metrics | 11 | 1 | pp. 11–12 |
| §6 Theoretical Analysis | 12 | 1 | pp. 12–13 |
| §6.1 Reversibility Guarantee (Theorem + proof sketch) | 12 | <1 | within p. 12 |
| §6.2 Memory Complexity (Proposition + worked example + footnote) | 12 | <1 | pp. 12–13 |
| §7 Discussion | 13 | 4 | pp. 13–17 |
| §7.1 Implications (4 paragraphs) | 13 | <1 | within p. 13 |
| §7.2 Broader Impact | 13 | 1 | pp. 13–14 |
| §7.3 Limitations (8 numbered items) | 14 | 1 | pp. 14–15 |
| §7.4 Empirical Measurement Prerequisites (incl. tab:oei-alpha-sweep) | 15 | 1 | pp. 15–16 |
| §7.5 Future Work (4 paragraphs) | 16 | <1 | within p. 16 |
| §7.6 Ethical Considerations (3 paragraphs: general + license + AI disclosure) | 16 | 1 | pp. 16–17 (§A starts mid-p. 17) |
| **Body total (§1 through §7.6)** | — | **≈16** | **pp. 1–17 partial** |
| §A Reproducibility (5 subsections; APPENDIX — unlimited pages) | 17 | 2.5 | pp. 17–19 |

**Body page count: approximately 16 pages.** Target: ≤8 pages. **Cut delta needed: ≈8 pages.** PDF total per `.log` Output line: 19 pages (502 KB).

Confidence note (R1 fidelity): the .aux gives section-start page anchors directly. Within-section page lengths are inferred from anchor deltas; longer sections (§5, §7) have multiple sub-sections sharing pages, so per-subsection estimates are coarse (±0.2 page).

---

## §2. Section-by-section cut-candidate inventory

Action codes: **KEEP** (load-bearing), **TRIM** (internal compression possible), **MOVE** (relocate wholesale to §A appendix; appendix pages are unlimited at review per [acl_formatting.md:23](../NeurIPS/acl_formatting.md:23)), **DELETE** (remove entirely).

Risk weighting: LOW (cosmetic / clearly cuttable), MEDIUM (touches framing-adjacent content), HIGH (touches contribution claims or empirical pillars — flag per Phase 2 R2).

### §1 Introduction (pp. 1–2)
- **¶1 (line 64) Framing question** — KEEP. Sets the central geometric question; load-bearing for track #13 framing.
- **¶2 (line 66) Prior application-domain hypotheses** — TRIM. Currently 7 lines covering alignment-faking literature + Gupta-Jenner + Bailey. Could compress to 3 sentences. Save ~0.2 page. Risk: LOW (overlap with §2.1, §2.2).
- **¶3 (line 68) Algorithm contribution summary** — TRIM. Currently 4 lines re-stating Theorem 1 + Memory Proposition pointers. Could be 2 sentences. Save ~0.1 page. Risk: LOW.
- **¶4 (line 70) Two-experiment summary** — KEEP. Load-bearing for Contribution 3.
- **§1.1 Contributions list** — KEEP at structure level; HIGH RISK to cut per R2. Each contribution paragraph is 6–10 lines; Contribution 3 (line 82) is verbose but framing-load-bearing. TRIM to ~25% shorter is feasible but HIGH RISK. Save ~0.2 page if trimmed.

### §2 Background and Related Work (pp. 3–5)
**Peer-review observation:** §2 reads as a survey. Track #13 norms (per the 26-paper Deep Research sample) tolerate compact related-work sections, often deferred to §A. Six subsections is unusually broad for 8-page review.

- **§2.1 Alignment Faking** (lines 90–93) — TRIM. ~4 lines. Greenblatt + Hubinger summaries are useful context but Day-3 audit already softened the α+ framing. Cut to 2 sentences. Save ~0.15 page. Risk: LOW.
- **§2.2 Representation Engineering and Activation Steering** (lines 95–103) — TRIM heavily. 3 dense paragraphs covering Zou, Belitsky, Gupta-Jenner, Bailey. The Bailey 2024 paragraph at line 101 is structurally important for §3.2; keep that sentence. Zou + Belitsky can be 1 sentence each. Save ~0.3 page. Risk: LOW-MEDIUM.
- **§2.3 H-Neurons** (lines 105–108) — MOVE to §4.1 (1 inline sentence). H-Neurons are defined here but only used in §4.1 for σ_H. Save ~0.15 page. Risk: LOW.
- **§2.4 MCTS for Language Model Search** (lines 110–113) — TRIM to 1 sentence in §4.2 or MOVE entirely to §A.Related-Work. Text-space MCTS comparison is useful but currently spans ~7 lines. Save ~0.25 page. Risk: LOW.
- **§2.5 Continuous Latent Reasoning and KV-Cache Search** (lines 115–120) — MOVE to §A.Related-Work. COCONUT + Okazaki-RAG comparison is methodologically interesting but 6 lines. Save ~0.2 page. Risk: LOW.
- **§2.6 LLM-as-a-Judge and Evaluation Limitations** (lines 122–125) — **DELETE**. ~4 lines. Under Option D+ (no LLM-judge baselines in our experiments) this subsection is residue from earlier framing. The closing sentence "our work extends this principle from output-space measurements to internal-state telemetry" is the only sentence relevant to current framing — can be folded into §5.5 if desired. Save ~0.2 page. Risk: LOW.

**§2 aggregate (TRIM + MOVE + DELETE):** ~1.25 pages savings. Risk LOW. R4 cross-ref check: no labels defined in §2.3–§2.6 that are forward-referenced from body; §2.1 + §2.2 retain context cited in §3.2.

### §3 Problem Formulation (pp. 5–6)
- **§3.1 Inference-Time Latent-Space Analysis** (lines 133–137) — TRIM. 2 paragraphs setting up the central question. Could compress ~30%. Save ~0.1 page. Risk: LOW.
- **§3.2 Dimensional Escape (Def 1 + 2 Hypotheses + 3 Eqs)** (lines 139–172) — KEEP. Load-bearing per R2. HIGH RISK to cut. Closing paragraph at line 172 (Bailey 2024 unification framing) could be tightened ~25% if needed; save ~0.1 page; risk HIGH.
- **§3.3 The Discrete Text-Space Bottleneck** (lines 174–181) — MOVE to §A or TRIM to single paragraph. ~8 lines. Restates motivation that's covered in §3.1 + §4.2.1. Cut entirely is safe. Save ~0.3 page. Risk: LOW.

### §4 Proposed Method (pp. 7–8)
- **§4.1 Telemetry Matrix** (lines 188–229) — TRIM. 4 paragraphs + 3 displayed equations. The CRITIQUE NOTE comment block (lines 202–209) doesn't render but takes source space; harmless. The closing paragraph at line 229 is dense; could be 1 sentence. Save ~0.15 page. Risk: LOW-MEDIUM.
- **§4.2.1 The Computational Bottleneck** (lines 233–242) — DELETE or MERGE with §3.3 (if §3.3 is preserved) or fold into §4.2 intro. Restates motivation from §3 + introduces eq:kv_memory which can be inlined in prose. Save ~0.3 page. Risk: MEDIUM (touches algorithm framing).
- **§4.2.2 The Reversible Rollback Algorithm** (lines 245–279) — TRIM + MOVE algorithm details to §A. Currently 5 numbered paragraphs (Step 1/2/3 + FP32 accumulator + Memory Complexity). Keep eq:fp32_accumulator + the FP32 paragraph in body (load-bearing for Theorem 1). MOVE Step 1 prose + Step 2 prose + Step 3 prose + eq:reverse_rollback + Memory Complexity paragraph to §A.Algorithm. Save ~0.5 page. Risk: MEDIUM (algorithm = Contribution 1; body must retain enough for theorem to make sense).
- **§4.2.3 MCTS Node Reward (Choice of r̂)** paragraph (lines 278–279) — KEEP. Already short.

### §5 Experimental Design (pp. 8–12)
**Peer-review observation:** §5 is the longest section. Several subsections read as project-management or implementation-history rather than scientific content.

- **§5 Implementation Notes paragraph** (lines 289–301) — MOVE entire 5-item list to §A.Implementation. Currently ~12 source lines describing how the Phase 2 implementation differs from the paper math. This is honest hedging but reads as "the code doesn't do what we claim" — review-time material best deferred to §A. **CRITICAL:** Item (iv) reconciles a numerical inconsistency between §4.2.1 (M_KV ≈ 40 GB) and §6.2 (M_KV ≈ 162 GB at 20B); if this paragraph moves to §A, the reconciliation must be preserved as a footnote at §6.2 or §4.2.1 (see §7 Pass L for full discussion). Save ~0.45 page. Risk: LOW with reconciliation footnote retained.
- **§5.1 Phase 0: Architectural Decoupling** (lines 303–307) — DELETE. ~5 lines describing repo refactoring. Not a contribution. Save ~0.17 page. Risk: LOW.
- **§5.2 Phase A: Local Prototyping on Lightweight Surrogates** (lines 309–322) — DELETE or MOVE to §A. ~14 lines + bullet list describing Phase 3 gate setup. Not a contribution; §A.1 + §A.2 + §A.4 Track A already cover the relevant hardware + calibration content. Save ~0.5 page. Risk: LOW.
- **§5.3 Experiments** (lines 324–404) — KEEP. Contains the two empirical pillars (Track F Exp 1 + Track G Exp 2, tables tab:exp1-results + tab:exp2-cartography-paths + Figure 1). HIGH RISK to cut per R2.
  - Mechanism + Interpretation paragraphs after Exp 1 table (lines 355–357) — TRIM ~20% feasible. Save ~0.1 page. Risk: MEDIUM (these connect data to claim).
  - Exp 2 finding (iv) "Mechanism interpretation per class" paragraph (line 374) — TRIM ~30% feasible. Save ~0.1 page. Risk: MEDIUM.
- **§5.4 Baselines** (lines 407–460) — MOVE table to §A.Related-Work (consolidate with §2 moves). Currently the §5.4 subsection contains only a methods-reference table with 4 rows and no body prose comparing baselines to our results (compare nothing). Reads as related-work table misfiled in §5. The EDITOR NOTE comment block (lines 410–443) doesn't render but flags this as a known reviewer-confusion risk. Save ~0.5 page. Risk: LOW. R4: `\ref{tab:baselines}` not referenced anywhere in body (verified via grep this session — only the table itself has the label).
- **§5.5 Evaluation Metrics** (lines 463–484) — KEEP largely intact (Day-7 rewrite is fresh). The CRITIQUE NOTE on OEI nonlinear-redistribution false-negatives (lines 477–482) is a comment block and doesn't render. TDS bullet (line 483) could be 1 sentence shorter. Save ~0.1 page. Risk: LOW.

### §6 Theoretical Analysis (pp. 12–13)
- **§6.1 Reversibility Guarantee — Theorem statement** (lines 494–512) — KEEP. Load-bearing for Contribution 1.
- **§6.1 Proof block** (lines 514–516) — MOVE to §A. Already proof-sketch length; full proof body to §A.Proofs. Pre-planned per plan v2 §8 risk register row. Save ~0.1 page. Risk: LOW.
- **§6.2 Memory Complexity — Proposition statement** (lines 521–525) — KEEP. Load-bearing for Contribution 1.
- **§6.2 Worked example sentence + footnote** (lines 527–528) — TRIM heavily. The Day-8-evening honest-framing footnote (Pass K candidate) explicitly hedges the 20B projection. A reviewer reading the abstract claim "$\sim$60$\times$ reduction... on 20B-parameter configurations" and arriving at §6.2 encounters a sentence + footnote that walks back the abstract's directness. Two options: (a) KEEP footnote + sentence as-is (current state, ~0.15 page), (b) DELETE footnote and trim the sentence to one line "We project a ${\sim}3\times$ multiple of M_KV at the algorithm's scale-invariance-by-construction limit" + retain the comparison to standard parallel MCTS. Recommend option (b). Save ~0.15 page. Risk: LOW (Pass K candidate).

### §7 Discussion (pp. 13–17)
**Peer-review observation:** §7 is the second-longest section (4 pages). Multiple subsections have growth bloat from cumulative Day-2 through Day-8 edits. Pre-trim before submission.

- **§7.1 Implications for Latent-Space Monitor and Reward Design** (lines 536–544) — TRIM 4 paragraphs → 2. The "Monitor design" paragraph (line 540) and "Reward design" paragraph (line 542) can each be ~30% shorter; the "Unifying design lesson" paragraph (line 544) is the synthesis and should be 50% of current length. Save ~0.3 page. Risk: MEDIUM (touches Contribution 3 framing).
- **§7.2 Broader Impact** (lines 546–549) — KEEP. Already 1 paragraph, ~4 lines. Required content.
- **§7.3 Limitations** (lines 551–569) — TRIM 8 numbered items → 5. Items 3 (single steering direction), 5 (4-prompt-class scope), and 6 (200-cycle empirical bound) can be merged into a single "Methodological constraints" item. Items 7 (entropy-falsified specific) overlaps with Contribution 3 framing; consolidate. Items 1, 2, 4, 8 are distinct and reviewer-valued — keep. Save ~0.4 page. Risk: MEDIUM (Limitations are reviewer-valued; over-cutting risks losing pre-empted critiques).
- **§7.4 Empirical Measurement Prerequisites and Initial Observations** (lines 574–604, incl. tab:oei-alpha-sweep) — **MOVE entire subsection to §A.MeasurementPipeline**. ~30 source lines + table. Reads as bug-history narrative (W_K projection issue + σ_H per-layer calibration resolution). For ARR review version, this is camera-ready material. The Table tab:oei-alpha-sweep is the most informative artifact; it can live in §A and be referenced from §5.5 (which currently does reference it via `\ref{eq:oei}` placement). Save ~1.0–1.3 pages (largest single MOVE in this plan). Risk: LOW. R4: `\ref{sec:measurement-prereqs}` referenced from §5.5 line 469 ("reported in §7.4 (Empirical Measurement Prerequisites)") — rewrite to "reported in §A.X (Measurement Pipeline)"; `\ref{tab:oei-alpha-sweep}` referenced from §5.5 and §6 — preserved at new location.
- **§7.5 Future Work** (lines 606–619) — TRIM 4 paragraphs → 1. Currently has Infrastructure-and-scaling + Monitor-design research + Reward-design research + Cartography extensions paragraphs. The Reward-design research paragraph (line 617) lists 8 anti-Goodhart candidates inline — Pass K candidate, consolidate to "Anti-Goodhart reward designs are a Phase B research program; candidates surveyed include multi-channel ensembles, external PRMs, and self-consistency variants." Other 3 paragraphs each trimmed to 1 sentence. Save ~0.35 page. Risk: LOW (Pass K candidate).
- **§7.6 Ethical Considerations** (lines 621–632) — KEEP. 3 short paragraphs: general ethics + Llama license attribution (J2 required) + AI assistant disclosure (J3 / ARR Checklist E1 required). All required content.

### §A Reproducibility (pp. 17–19) — APPENDIX, unlimited pages
- KEEP entirely intact. After absorbing the moves above (§2.3–§2.6 → §A.RelatedWork, §3.3 → §A.Background, §4.2 algorithm details → §A.Algorithm, §5 Impl Notes → §A.Implementation, §5.1+§5.2 → §A.LocalPrototyping, §5.4 baselines table → §A.RelatedWork-Table, §6.1 proof → §A.Proof, §7.4 → §A.MeasurementPipeline), §A grows from ~2.5 pages to ~6–8 pages. Per ACL rules this is acceptable.

---

## §3. Ranked cut plan (risk-adjusted)

Risk-adjusted-savings = (page savings) × (1 − risk weight); LOW = 0.0, MEDIUM = 0.3, HIGH = 0.7. Ranked descending. Cumulative savings column tracks the running total.

| Rank | Cut | Action | Save (pp) | Risk | Adj. save | Cumulative | Framing-impact note |
|---:|---|---|---:|---|---:|---:|---|
| 1 | §7.4 Empirical Measurement Prerequisites + tab:oei-alpha-sweep | MOVE entire to §A.MeasurementPipeline | 1.20 | LOW | 1.20 | 1.20 | Bug-trail narrative; not contribution-load-bearing |
| 2 | §2 Background subsections (§2.3 H-Neurons, §2.4 MCTS, §2.5 Continuous Latent, §2.6 LLM-as-Judge) | §2.3 MOVE inline §4.1; §2.4 MOVE to §A; §2.5 MOVE to §A; §2.6 DELETE | 0.85 | LOW | 0.85 | 2.05 | §2.1 + §2.2 retained; covers Bailey 2024 framing |
| 3 | §5.4 Baselines table | MOVE to §A.RelatedWork-Table | 0.50 | LOW | 0.50 | 2.55 | No body prose compares; table is related-work in §5 misfiled |
| 4 | §5.2 Phase A Local Prototyping | DELETE | 0.50 | LOW | 0.50 | 3.05 | Operational scaffolding; §A.1 + §A.2 + §A.4 Track A cover hardware/calibration |
| 5 | §5 Implementation Notes (5-item list) | MOVE to §A.Implementation + retain M_KV reconciliation footnote at §6.2 | 0.45 | LOW | 0.45 | 3.50 | Camera-ready material; reads as "code doesn't match math" |
| 6 | §7.5 Future Work (4 paragraphs → 1) | TRIM | 0.35 | LOW | 0.35 | 3.85 | Anti-Goodhart 8-candidate list consolidated (Pass K candidate) |
| 7 | §4.2 Algorithm details (Step 1/2/3 prose + eq:reverse_rollback + Memory Complexity paragraph) | MOVE to §A.Algorithm; keep §4.2 intro + FP32 paragraph + eq:fp32_accumulator in body | 0.50 | MEDIUM | 0.35 | 4.20 | Algorithm = Contribution 1; body retains theorem-supporting content |
| 8 | §3.3 Discrete Text-Space Bottleneck | MOVE to §A.Background or DELETE | 0.30 | LOW | 0.30 | 4.50 | Restates §3.1 + §4.2.1 motivation |
| 9 | §2.1 + §2.2 (Alignment Faking + RepE) prose | TRIM each by ~50% | 0.45 | LOW | 0.45 | 4.95 | Day-3 audit already softened α+ framing; further compression safe |
| 10 | §7.3 Limitations 8 items → 5 | TRIM (merge items 3+5+6 into one "methodological constraints" item; merge item 7 with Contribution-3-overlap) | 0.40 | MEDIUM | 0.28 | 5.23 | Limitations are reviewer-valued; over-cutting loses pre-empted critiques |
| 11 | §7.1 Implications 4 paragraphs → 2 | TRIM | 0.30 | MEDIUM | 0.21 | 5.44 | Touches Contribution 3 framing |
| 12 | §1 Intro ¶2 (prior-application-domain hypotheses) | TRIM ~50% | 0.20 | LOW | 0.20 | 5.64 | Overlap with §2.1, §2.2 |
| 13 | §4.2.1 The Computational Bottleneck | DELETE / MERGE with §4.2 intro | 0.30 | MEDIUM | 0.21 | 5.85 | Restates §3.3 + §3.1 motivation |
| 14 | §5.1 Phase 0 Architectural Decoupling | DELETE | 0.17 | LOW | 0.17 | 6.02 | Operational scaffolding |
| 15 | §1 Intro ¶3 (algorithm contribution summary) | TRIM ~50% | 0.10 | LOW | 0.10 | 6.12 | Re-states §1.1 Contribution 1 |
| 16 | §4.1 Telemetry Matrix prose | TRIM closing paragraph at line 229 | 0.15 | MEDIUM | 0.11 | 6.23 | Touches Contribution 2 framing |
| 17 | §5.5 TDS bullet + ~3 line trim | TRIM | 0.10 | LOW | 0.10 | 6.33 | Day-7 rewrite is fresh; minimal further trim |
| 18 | §6.1 Theorem 1 proof block | MOVE to §A.Proof | 0.10 | LOW | 0.10 | 6.43 | Pre-planned plan v2 §8 cut |
| 19 | §6.2 worked example footnote (Pass K candidate) | DELETE footnote, retain sentence with hedge | 0.10 | LOW | 0.10 | 6.53 | Body retains the projection claim with abstract-hedge consistency |
| 20 | §5.3 Exp 1 + Exp 2 Mechanism/Interpretation paragraphs | TRIM each ~20% | 0.20 | MEDIUM | 0.14 | 6.67 | Connects data to claim; reviewer-valued |
| 21 | §3.1 Inference-Time Latent-Space framing | TRIM ~30% | 0.10 | LOW | 0.10 | 6.77 | Frames central question |
| 22 | §1.1 Contributions Contribution-3 prose | TRIM ~25% | 0.20 | HIGH | 0.06 | 6.97 | R2-protected; framing-critical |
| 23 | §3.2 closing paragraph (Bailey unification) | TRIM ~50% | 0.10 | HIGH | 0.03 | 7.07 | R2-protected; unification framing |

**Sign-off thresholds:**

- **Minimum compliance line (≥7 pp savings, body to ~9 pp; 1 pp over budget but tolerable for review):** Cuts #1 through #21. Cumulative: 6.77 pp.
- **Budget compliance line (≥8 pp savings, body to ~8 pp; exactly at budget):** Cuts #1 through #23 (all cuts applied, including HIGH-risk). Cumulative: 7.07 pp. **CRITICAL: this is still 1 pp short of the 8-pp target.**
- **Recommended sign-off threshold (with margin, ≥9 pp savings, body to ~7 pp):** **NOT REACHABLE** by the cuts ranked above. See §8 CRITICAL escalation.

**Recommended threshold for Day-10 application: Cuts #1 through #20 (cumulative 6.67 pp savings).** This applies all LOW-risk cuts + MEDIUM-risk cuts where the framing impact is bounded. Body lands at ~9.3 pp — **1.3 pp over budget**.

To reach the 8-page budget, additional cuts beyond Cut #20 are needed (Cuts #21–#23 are HIGH-risk or framing-protected). The §8 escalation describes options.

---

## §4. Sign-off ask

**Recommended sign-off threshold:** Cuts #1 through #20.

**Page savings:** 6.67 pp cumulative (best-effort estimate; primary-source-derived from .aux + line-count heuristic; ±0.5 pp uncertainty).

**Resulting body page count estimate:** ~9.3 pp (target ≤ 8 pp).

**Margin:** −1.3 pp (over budget).

**Critical-escalation flag:** YES — see §8 below. Applying Cuts #1 through #20 leaves the body ~1.3 pp over the 8-page review budget. Reaching 8 pp requires Cuts #21–#23 (HIGH risk per R2: touches §1 Intro, §1.1 Contributions, §3.2 closing paragraph — framing-load-bearing). Reaching 7 pp (margin) requires structural decisions beyond this cut plan (see §8 options).

**Josh approves application of Cuts #1 through #20?** [Y / N / Modify-threshold]

- If **Y**: Day-10 Session B applies Cuts #1 → #20 in order, with G-smoke-compile gate after each batch (recommended: batch by section — §2 batch, §5 batch, §7 batch). Final page-count verification post-#20. Body expected at ~9.3 pp.
- If **Y + extend to #21–#23**: Day-10 applies all 23 cuts. Body expected at ~8.9 pp. Still under target; see §8.
- If **N or Modify**: re-rank with Josh's preferred risk-weighting OR consider §8 options.

After Day-10 application: G-style + G-smoke-compile gates per plan v2 §9; cover-letter decision deferred to Day-14 final pre-submission.

---

## §5. Out-of-scope notes (considered but rejected)

- **Drop a contribution from §1.** Considered (per Phase 4 trip-wire option (a)). Rejected at audit stage — Contributions 1, 2, 3 form the paper's track #13 alignment per the 26-paper Deep Research sample (algorithm-as-interpretability-lens; negative-result-as-mechanism-diagnosis). Dropping any of them would break the track-selection thesis. If §8 escalation forces this option, Contribution 2 (telemetry matrix) is the lowest-cost drop — current evidence is descriptive (alpha-sweep on 1B), not discriminative.
- **Switch tracks.** Considered. Rejected — track-selection was locked 2026-05-13 (J6c) after 4-report Deep Research synthesis; switching now is high-cost.
- **Defer submission to ARR June 2026 cycle.** Considered (per trip-wire option (d)). Rejected at audit stage — EMNLP 2026 main track accepts only ARR-routed submissions, May 2026 cycle is the binding window. Findings track is an EMNLP-commit-time decision (2026-08-02), not an ARR-submission-time decision.
- **Move §A Reproducibility entirely to supplementary materials.** Considered (per plan v2 §3 plasticity rule). Rejected — §A is at appendix (not supplementary) per ARR convention; reviewers may consult it; supplementary materials are explicitly non-required reading per [acl_formatting.md:23](../NeurIPS/acl_formatting.md:23).
- **Reduce abstract from current ~250 words to ACL 200-word maximum.** Defer to Day-10 application checklist. Save ~50 words but <0.1 page; not in scope as a numbered cut.
- **DELETE eq:reverse_rollback (redundant with eq:forward_mutation sign-flip)** and **DELETE eq:reversibility_naive (only used as theorem comparison)**. Each saves ~0.03 pp. Considered too cosmetic to rank separately; fold into Cut #7 (§4.2 algorithm details MOVE) since both equations live in that block.

---

## §6. Day-10 application notes

**Order of application (batched for G-smoke-compile gate efficiency):**

**Batch 1 — §2 restructuring (Cuts #2 + #9 + Cut #18 §6 proof move).** §2 subsection moves + §2.1/§2.2 trims + §6 proof move. Likely largest single batch (1.45 pp savings). G-smoke-compile gate after batch: verify no undefined refs introduced, page count check.

**Batch 2 — §5 cleanup (Cuts #3 + #4 + #5 + #14 + #17 + #20).** §5.1 + §5.2 deletions, §5.4 baselines table move, §5 Implementation Notes move (with M_KV reconciliation footnote at §6.2 to be added post-move), §5.5 TDS trim, §5.3 Mechanism/Interpretation trims. G-smoke-compile gate after batch.

**Batch 3 — §7 trimming (Cuts #1 + #6 + #10 + #11 + #16 + #19).** §7.4 MOVE entire to §A (largest single move), §7.5 trim 4→1 paragraph, §7.3 trim 8→5 items, §7.1 trim 4→2 paragraphs, §4.1 closing trim, §6.2 footnote delete. G-smoke-compile gate after batch.

**Batch 4 — §1, §3, §4.2 (Cuts #7 + #8 + #12 + #13 + #15 + #21).** §4.2 algorithm details MOVE, §3.3 MOVE/DELETE, §1 Intro ¶2 trim, §4.2.1 delete/merge, §1 Intro ¶3 trim, §3.1 trim. Touches contribution-adjacent content; smoke-compile + visual inspection.

**Cross-reference updates required during application:**

- §5.5 line 469 "reported in §7.4 (Empirical Measurement Prerequisites)" → "reported in §A.X (Measurement Pipeline)" after Cut #1.
- §4.2 → §6 cross-section M_KV reconciliation: add footnote at §6.2 line 528 noting "Different assumed sequence lengths give different M_KV figures: §4.2.1 uses S=64K assumption (40 GB); §A.X Implementation uses S=128K + clone overhead (80 GB); §6.2 uses gpt-oss-20b architectural specs (162 GB)." After Cut #5.
- §4.2 internal references (eq:forward_mutation, eq:reverse_rollback, eq:fp32_accumulator): verify after Cut #7 MOVE.
- §A.0 may need a new "How this appendix is organized" opening paragraph after batch 1 + 2 + 3 + 4 since §A grows substantially. Optional polish.

**Verification gates per plan v2 §9:**

- **G-smoke-compile** after each batch (4 batches → 4 smoke-compile runs). Required clean pdflatex on both passes; zero undefined refs.
- **G-style** at end of Day 10 — page count ≤ 8 pages body (verify via .aux page anchors).
- **G-revision** at end of Day 10 — `grep -n "pending audit"` over canonical TeX.

**Estimated Day-10 wall time:** 3–4 hours including all 4 batches + smoke-compile gates + cross-reference updates + final page-count verification. (Plan v2 §2 Day-9 + Day-10 nominal allocation is ~6 hours combined; this fits.)

---

## §7. Peer-review observations — proposed additions to Day-12 parallel-review plan

Per Josh's 2026-05-15 directive that this audit be conducted from a neutral double-blind-reviewer perspective, the following observations surfaced during the §2 inventory work and should inform the Day-12 parallel adversarial review (`docs/logs/2026-05-14_parallel-review-plan.md`). Each is a candidate new Pass or strengthened sub-pass for Axis 5 (standard adversarial pass) or Axis 2 (framing defensibility).

### Proposed new Axis 5 Pass L: Cross-section numerical-consistency check

**Motivation:** Audit surfaced that M_KV at 20B appears with three different numerical values across the body:
- §4.2.1 line 242: "$M_{KV} \approx 40$ GB"
- §5 Implementation Notes line 299: "approximately $80$ GB before any accumulator overhead"
- §6.2 line 528: "$M_{KV} \approx 162$ GB per cache snapshot"

§5 Impl Notes item (iv) reconciles them (different sequence-length assumptions + clone overhead), but a reviewer reading the body sequentially encounters three different M_KV figures before reaching the reconciliation. This is a real internal-consistency finding none of the current 11 Axis-5 sub-passes would catch (Pass I "numerical-claim verification" picks 3 samples, not exhaustive cross-section).

**Prompt addition (Pass L):**
```
Pass L. Cross-section numerical-consistency check.
  Grep the paper for every numeric or quantitative claim. For each
  physical quantity that appears in multiple sections, verify the
  numerical values agree (or are reconciled via footnote/sub-section).
  Specifically scrutinize:
    (a) M_KV at 20B: §4.2.1 says "≈40 GB"; §5 Impl Notes says "≈80 GB
        before accumulator overhead"; §6.2 says "≈162 GB". Are the
        sequence-length assumptions or overhead assumptions clearly
        differentiated in the text a reviewer reads sequentially, or
        does the reader encounter conflicting numbers?
    (b) Memory bound constant: abstract says "3.04–3.06×"; §1
        Contribution 1 says "3.06× at 1B and 3.04× at 3B"; §A.4
        Track D paragraph says "3.04–3.06"; §6.2 says "3.04·M_KV
        ≈ 493 GB". Are these consistent? Is the (b, d, n) tuple
        independence claim primary-source verified?
    (c) Cartography Δ values: §5 Exp 2 (i) reports |Δ|≈0.67 for C2
        and |Δ|≈0.55 for C3; Track G report has raw -0.667 / -0.547.
        Does the rounded-then-subtracted vs subtract-then-rounded
        distinction matter (memory says it doesn't for the headline
        claim)?
  For each finding, severity: MODERATE (revision request) unless a
  number is wrong (SEVERE).
```

### Proposed new Axis 5 Pass M: Abstract-to-body claim escalation

**Motivation:** Day-8-evening §6 honest-framing refinement hedged the 20B projection in the body. The abstract still reads "$\sim$60$\times$ reduction... on 20B-parameter configurations" without the hedge a reviewer reading the body later finds in §6.2 ("is therefore a projection from smaller-scale measurements rather than a direct 20B observation"). This is asymmetric hedging — abstract reads stronger than body anchor. A reviewer doing the standard "scan abstract first, verify against §-N later" workflow would catch this.

**Prompt addition (Pass M):**
```
Pass M. Abstract-to-body claim escalation check.
  Read the abstract sentence-by-sentence. For each numeric or
  methodological claim, verify the body anchor uses the same or
  stronger hedge. Failure mode: abstract reads more confident than
  body. Specifically scrutinize:
    (a) Abstract: "$\sim$60$\times$ reduction... on 20B-parameter
        configurations" vs §6.2 body: "is therefore a projection
        from smaller-scale measurements rather than a direct 20B
        observation." Severity: MODERATE-MAJOR.
    (b) Abstract: "Croissant 1.1 metadata package... released
        alongside this submission" — verify the supplementary URL
        is live at submission time (Day-14 G-anon-B gate); if URL
        is still anonymous.4open.science placeholder, the abstract
        is overstating release status.
    (c) Abstract: "validated to machine precision over 200 apply–
        revert cycles on Llama 3.2-1B and Llama 3.2-3B" — body §6.1
        + §7.3 limitation 6 echo this. Match.
    (d) Abstract: "Dimensional Escape... satisfying the proxy
        metric without satisfying the semantic objective" — strong
        claim. Body §3.2 + §5 Exp 1 evidence H2 (reward-driven) only;
        H1 (monitor-driven) is conjectured. Does the abstract claim
        both or only the reward-driven case?
  For each finding, severity: MODERATE unless reviewer-rejection-
  likely (then MAJOR).
```

### Proposed new Axis 5 Pass N: Contributions coherence stress test

**Motivation:** A peer reviewer reading §1 Contributions list will ask: does the paper present three coherent contributions, or an algorithm (C1) and then "the algorithm with its primary use case fails" (C3)? The abstract reframes the algorithm as a "high-precision diagnostic probe... rather than a reasoning-improvement tool" — but Contribution 1's packaging (FP32 accumulator, memory bound, theorem, 60× reduction) reads as a method paper, not a diagnostic-probe paper. A hostile reviewer may flag narrative tension.

**Prompt addition (Pass N):**
```
Pass N. Contributions coherence stress test.
  Read §1 Contributions list sequentially. After each contribution,
  ask: does the next contribution build on, complement, or undercut
  the prior?
    - C1 (algorithm) → C2 (telemetry matrix): coherent if C2 is
      "the algorithm's monitoring substrate."
    - C2 (telemetry matrix) → C3 (Dimensional Escape demonstration):
      coherent if C3's evidence uses the telemetry matrix to
      diagnose the failure mode. Does §5 Exp 1 use the telemetry
      matrix (σ_H, ρ_R) to evidence the prior-amplification
      mechanism, or does it use only entropy + accuracy?
    - C1 (algorithm) → C3 (the algorithm's reward function fails):
      does this read as coherent ("algorithm + negative result") or
      as tension ("algorithm whose obvious use case doesn't work")?
  If a reviewer reads C1 and forms the prior "this is a method
  paper," will the rest of the paper meet that expectation, or
  redirect mid-stream? Severity: MAJOR if reviewer-rejection-likely
  due to narrative confusion; MODERATE if hedge-needed.
```

### Proposed new Axis 5 Pass O: Operational-scaffolding triage

**Motivation:** Audit surfaced four sections that read as project-management or development-history rather than scientific content: §5.1 Phase 0 (repo refactoring), §5.2 Phase A (local prototyping), §5 Implementation Notes (5-item Phase 2 vs Phase 3 deviation list), §7.4 Empirical Measurement Prerequisites (σ_H + W_K projection bug history). For an Interpretability-track paper, these read as camera-ready material at best. The cut plan above (§3) already recommends moving / deleting most of them, but a parallel-review Pass should explicitly scrutinize the remaining body for similar scaffolding.

**Prompt addition (Pass O):**
```
Pass O. Operational-scaffolding triage.
  Read the paper looking for sections, paragraphs, or sentences that
  describe HOW WE DID THE WORK rather than WHAT WE FOUND. Common
  patterns:
    - "Phase 0/A/B" sub-headings (development milestones)
    - "Implementation Notes" listing how the code differs from paper
      math
    - "Measurement Prerequisites" sections describing the bug-fix
      history of getting a metric to work
    - Sentences referencing internal artifacts (e.g., "Track F report"
      without source-fidelity-required hedge)
  For each, recommend: KEEP (the operational detail is methodologically
  important), TRIM (compress to one sentence), MOVE (to §A), DELETE
  (camera-ready material).
  Severity: MINOR-MODERATE; aggregate findings drive the page-budget
  trim list.
```

### Proposed strengthening of Axis 2 Q3 (Cross-experiment unification)

**Motivation:** Axis 2 Q3 currently asks whether the H1+H2 unification is empirical or theoretical. Strengthen: explicitly point out that Contribution 3 says "Empirical demonstration of Dimensional Escape" (singular) but body evidences only H2 (Track F + Track G are reward-driven; H1 is conjectured via Bailey 2024 analogy). Recommend either: rewrite Contribution 3 to "Empirical demonstration of reward-driven Dimensional Escape; structural-equivalence argument for monitor-driven case" OR hedge §3.2 closing paragraph to remove the unification claim and present as parallel cases.

### Proposed strengthening of Axis 2 Q5 (Goodhart vs Dimensional Escape)

**Motivation:** Q5 already asks for the precise distinction. Strengthen: add an actionable recommendation — if the reviewer cannot articulate the distinction in one sentence, recommend either (a) retiring "Dimensional Escape" and re-anchoring on "Goodhart's Law in high-dimensional latent spaces," or (b) sharpening §3.2 Definition to make the distinction operational (e.g., "Dimensional Escape predicts a specific geometric route through null space, whereas Goodhart's Law predicts only that the proxy will be optimized over the objective"). The paper's value-add of the new term must be defensible.

**Recommendation:** Add these 4 new Passes (L, M, N, O) and 2 Q-strengthenings to `2026-05-14_parallel-review-plan.md` Axis 5 + Axis 2 in this same Day-9 session. The additions are derived from the audit and are best applied while the observations are fresh.

---

## §8. CRITICAL escalation flag

**Status:** The ranked cut plan (§3 above) cannot reach 9 pages of cumulative savings (margin-with-buffer target) and reaches only 7.07 pages with all cuts applied including HIGH-risk. Reaching the 8-page review budget requires applying ALL 23 cuts; reaching 7-page body (1 pp margin) requires structural decisions beyond this plan.

**Per Phase 4 trip-wire, Josh's escalation options are:**

**(a) Drop a contribution from §1.** Lowest-cost candidate: Contribution 2 (telemetry matrix). Current evidence is descriptive (alpha-sweep on 1B Llama-3.2 only, Table tab:oei-alpha-sweep), not discriminative. Dropping Contribution 2 saves ~1.5 pp body (§4.1 + §7.4 portions remaining body anchor) and reframes §1 as 2 contributions (algorithm + Dimensional Escape demonstration). **Risk:** R2 protected at the framing level; track #13 alignment may shift (current alignment leans on per-layer telemetry as interpretability infrastructure).

**(b) Restructure §A to absorb more body content.** Already recommended in this cut plan (§7.4 + §2.3–§2.6 + §4.2 algorithm details + §5 Impl Notes + §5.4 baselines table + §5.1+§5.2 → §A). The cut plan applies all this restructuring and still falls short. Going further (e.g., MOVE §3.3 + §4.1 telemetry equations + §6.1 theorem statement to §A) would compress the body's contribution narrative below readable threshold. **Limit reached.**

**(c) Submit at ~9 pp body with cover-letter justification.** Acceptable if reviewers tolerate 1-pp overage at review (per [acl_formatting.md:27](../NeurIPS/acl_formatting.md:27): "Papers that do not conform to these requirements may be rejected without review"). HIGH rejection risk; not recommended.

**(d) Defer submission to ARR June 2026 cycle.** No-go for EMNLP 2026 main track (May ARR cycle is binding); Findings track may accept but is an EMNLP-commit-time decision (2026-08-02), not an ARR-submission-time decision.

**Recommendation for Josh:** Apply Cuts #1–#20 at Day-10 (most aggressive feasible without HIGH-risk framing impact). Re-measure body page count after Day-10 application. If body still >9 pp, escalate option (a) — drop Contribution 2 — as Day-11 pre-co-author-send-out decision. If body 8.0–8.9 pp, accept option (c) with cover-letter justification.

**Trip-wire fired:** YES. CRITICAL flag raised per Phase 4 condition: cut plan with all cuts applied cannot reach 9 pp cumulative savings.

---

*End of Day-9 page-budget cut plan v1. Authored 2026-05-15 Opus 4.7 Session A. Application target: Day-10 (2026-05-16 nominal calendar; could pull forward to 2026-05-15 PM/evening per Josh sign-off). Total ranked cuts: 23. Sign-off threshold recommended: Cuts #1–#20 (6.67 pp savings; body lands ~9.3 pp). CRITICAL escalation flag raised on the gap-to-budget (1.3 pp short at recommended threshold; 0.9 pp short with all cuts applied). Peer-review observations recorded in §7 for Day-12 parallel-review plan; 4 new Passes (L, M, N, O) and 2 Q-strengthenings recommended for Axis 5 + Axis 2 — to be applied to `docs/logs/2026-05-14_parallel-review-plan.md` in this same Day-9 session per Josh directive.*
