# Day-12 Parallel-Review Synthesis — 5 Axes Consolidated

**Authored:** 2026-05-17 EOD (Day 12 of 14-day execution plan, compressed from Day-13-EOD original schedule per Josh directive after Round 2 landed Day-12 PM).
**Scope:** Aggregate findings across all 5 parallel-review axes (Round 1: Axes 1+2+5 against paper HEAD `267e9cb`; Round 2: Axes 3+4 against post-M1-M5 paper HEAD `934a288`); de-duplicate cross-axis findings; emit patch plan with owners + deadlines.
**Per-axis reports (audit trail):**
- [docs/logs/2026-05-17_axis-1-report.md](2026-05-17_axis-1-report.md) — code-semantic correctness
- [docs/logs/2026-05-17_axis-2-report.md](2026-05-17_axis-2-report.md) — Dimensional Escape framing defensibility
- [docs/logs/2026-05-17_axis-3-report.md](2026-05-17_axis-3-report.md) — Reproducibility from §A
- [docs/logs/2026-05-17_axis-4-report.md](2026-05-17_axis-4-report.md) — Reviewer-prior alignment for track #13
- [docs/logs/2026-05-17_axis-5-report.md](2026-05-17_axis-5-report.md) — Standard adversarial pass (18 sub-passes A-R)

**Synthesis-target ref:** [docs/logs/2026-05-14_parallel-review-plan.md](2026-05-14_parallel-review-plan.md) §"Synthesis protocol".

---

## 1. Severity counts

### Per-axis (orchestrator-verified against rubric)

| Axis | SEVERE | MAJOR | MODERATE | MINOR | Total |
|---|---|---|---|---|---|
| 1 — Code semantics | 0 | 0 | 2 | 1 | 3 + 3 CLEAN |
| 2 — Framing | 0 | 2 | 3 | 2 | 7 |
| 3 — Reproducibility | **1** | 4 | 4 | 6 | 15 |
| 4 — Reviewer-prior | 0 | 4 | 3 | 1 | 8 |
| 5 — Standard adversarial | 0 | 4 | 8 | 6 | 18 |
| **Total (pre-dedup)** | **1** | **14** | **20** | **16** | **51** |

### Post-dedup aggregate

Cross-axis dedup at higher severity (Axis-X+Axis-Y both flagging the same finding → counted once):

| Severity | Pre-dedup | Dedup'd | Already patched Day-12 | Outstanding |
|---|---|---|---|---|
| SEVERE | 1 | 1 | 1 (commit `bb83ddb`) | **0** |
| MAJOR | 14 | 11 | 5 (commit `934a288` M1-M5) + Pass-2 hedge | **6** |
| MODERATE | 20 | 17 | 0 | 17 |
| MINOR | 16 | 13 | 0 | 13 |

### Headline: 0 SEVERE / 6 MAJOR outstanding post-Day-12

No code-vs-paper SEVERE (Axis 1 = 0 SEVERE). No advisor go/no-go escalation triggered. All 6 outstanding MAJORs are text-level patches deliverable in the Day-14 window.

---

## 2. De-duplicated finding list

### SEVERE (1 total; 1 patched Day-12)

| ID | Source axes | Finding | Status |
|---|---|---|---|
| **S1** | Axis 3 GAP-5e.3 | §A Track C paragraph declared system prompt `"You are a helpful assistant. Answer concisely."`; actual driver `scripts/diagnose_track_f_negcontrol.py:64` uses `"You are a careful, concise reasoner. Answer multiple-choice questions with a single capital letter."`. Primary-source-fidelity violation; reviewer following §A gets 3-5pp accuracy delta from reported numbers. | **PATCHED** commit `bb83ddb` |

### MAJOR (14 pre-dedup → 11 post-dedup; 5 patched Day-12; 6 outstanding)

| ID | Source axes | Finding | Day-12 status |
|---|---|---|---|
| **M1** | Axis 2 Q3 + Q6 + Axis 5 Pass M(d) | §3.2 line 119 body retained unification overclaim "we argue these are sub-cases of a single underlying pathology" after Day-11 hedges fixed abstract/§1/§1.1 | PATCHED `934a288` M1 |
| **M2** | Axis 2 Q3 | §1.1 C2 title singular: "Empirical demonstration of Dimensional Escape" — no reward-driven qualifier | PATCHED `934a288` M2 |
| **M3** | Axis 2 Q5 + Axis 5 Pass H | §3.2 Definition lacks Goodhart distinction → coined-term defense gap; missing reward-hacking citations (Gao 2023, Skalse 2022) | PATCHED `934a288` M3 |
| **M4** | Axis 5 Pass A | Abstract: "expose a fundamental geometric pathology" + "under any surrogate reward lacking exogenous grounding" — universalizing overclaim | PATCHED `934a288` M4 |
| **M5** | Axis 5 Pass D + Axis 1 Q4 | Title says "MCTS" but experiments enumerate 27 paths (no UCB1); method/experiments scope mismatch | PATCHED `934a288` M5 |
| **M2-prose** | Pass-2 framing-consistency (above-and-beyond) | §1.1 C2 prose body: "surrogate rewards lacking exogenous grounding inevitably resolve" — same universalizing claim M4 hedged | PATCHED `934a288` (Pass-2 hedge) |
| **A** | Axis 3 GAP-3.1 | §A defers HF checkpoint commit SHAs (1B + 3B) to supplementary; reviewer cannot verify weight identity → "I tried to reproduce but got different numbers because I used a different revision" | **OUTSTANDING — Day-14** |
| **B** | Axis 3 GAP-4.1 | §A omits exact reproduction CLI commands; reviewer must reverse-engineer from driver source | **OUTSTANDING — Day-14** |
| **C** | Axis 3 GAP-1.1 | §A omits HF authentication + Llama license-accept step; reviewer cannot start without it | **OUTSTANDING — Day-14** |
| **D** | Axis 3 GAP-2.1 | §A defers critical version pins (`transformers ≥ 4.45`, `accelerate ≥ 1.0`, `numpy < 2.0`); wrong version silently produces different numerics | **OUTSTANDING — Day-14** |
| **E** | Axis 4 Q1 + Axis 5 Pass F | Title front-loads Methods/Efficiency tokens ("MCTS", "Memory-Bounded") before reaching interp signal ("Probe"); ACL routing system may route to ML/Efficient Methods reviewers instead of interp | **OUTSTANDING — Day-14** |
| **F** | Axis 4 Q2 + Axis 5 Pass F | Abstract S1-S2 (~40% of word count) pure algorithm+formalism; Dimensional Escape doesn't arrive until S3; self-correction at S8 too late for first-impression routing | **OUTSTANDING — Day-14** |
| **G** | Axis 4 Q3 + Axis 5 Pass N (post-drop-C2 strengthening) | §1.1 Contributions list packages C1 as algorithm achievements (FP32 accumulator, memory bound) without infrastructure-generator framing the track values; "diagnostic probe rather than reasoning-improvement tool" reframe is in abstract but not echoed in C1 | **OUTSTANDING — Day-14** |
| **H** | Axis 4 Q4 | 0 ARR routing keywords ("interpretability", "feature attribution", "knowledge tracing") in abstract; insufficient saturation for routing to interp reviewers | **OUTSTANDING — Day-14** |
| **I** | Axis 4 Q5 + Axis 5 Pass R(d) | Cross-scale positive finding (MCQ prior-structure direction-flip 1B↔3B; mean-step α flip from 0.89 → 0.22 for C2 ARC-Easy) buried in Exp 2 results bullets, not promoted to §1 — reviewer reads paper as "two negatives, zero positives" rather than "two complementary findings (negative reward + positive cartography)" | **OUTSTANDING — Day-14** |
| **J** | Axis 5 Pass P | Page-budget format-compliance hard limit; if body ends up at 9pp at submission, desk-reject risk per `acl_formatting.md:27` | **MONITORING — body currently 8pp** post-Day-12 patches; Day-14 PDF page-count gate |

**Note on dedup**: E + F + G + H all converge on a single underlying issue ("interp-track first-impression layer needs reframing"). They're listed as separate findings because each has a distinct patch target (title / abstract sentence-order / §1.1 packaging / keyword saturation), but a single coordinated rewrite of title + abstract + §1.1 could address all four.

### MODERATE (20 pre-dedup → 17 post-dedup; all outstanding)

Listed by axis for compactness; full text in per-axis reports.

**Axis 1 (2):** Q4 (path-enumeration vs sampling clarity — partially addressed by M5 §5 search-variant scope clarifier); Q6 (drift script measures layer-0 K only).

**Axis 2 (3):** Q1 (Definition operational-signature missing); Q4 ("geometric" → "structural" framing softening); Q7 (single-reward-function limitation).

**Axis 3 (4):** GAP-5e.1 (letter-extraction description); GAP-5c.1 (steering-direction sanity-check checksum); GAP-8.2 (smoke-test invocation); GAP-1.2 (Python interpreter version).

**Axis 4 (3):** Q5 negative-result framing residue (any "we tried X" sentences); Q6 actionable-infrastructure §7 language; Q7 most-likely reject-reason coverage.

**Axis 5 (8):** Pass G (Limitations completeness — 9th limitation); Pass H related-work blind spot (mooted by M3 bibitems); Pass B single-direction steering; Pass C R-uninformed baseline; Pass E reproducibility skepticism cell-by-cell table; Pass I numerical-claim traceability; Pass N (a)+(c) narrative-tension redirection at §1.1 C1; Pass Q body §4 depth re-fold.

**Cross-axis dedup**: Axis 5 Pass H is mooted by M3's bibitem additions (commit `934a288`). Net MODERATE = 17.

### MINOR (16 pre-dedup → 13 post-dedup; all outstanding)

Per-axis: Axis 1 (1); Axis 2 (2); Axis 3 (6); Axis 4 (1); Axis 5 (6). Mostly cosmetic / clarity / version-pin completeness items. Dedup removes 3 overlapping cosmetic items between Axis 3 and Axis 5.

---

## 3. Day-12 already-landed patches (audit trail)

Day-12 work landed in 4 commits on branch `claude/sharp-lalande-f3d5de` pushed to `origin/main` (HEAD = `bb83ddb`):

| Commit | Description | Findings addressed |
|---|---|---|
| `2a87b9b` | Round 1 axis reports (audit trail) | — (review artifacts) |
| `934a288` | M1-M5 patches + prep edit + Pass-2 hedge | M1, M2, M3, M4, M5, M2-prose (6 MAJORs) |
| `aade81c` | Round 2 axis reports (audit trail) | — (review artifacts) |
| `bb83ddb` | SEVERE §A system-prompt fix | S1 (1 SEVERE) |

**Smoke-compile state post-Day-12:** PDF 14 pp / 467 KB; body 8 pp confirmed via `.aux` `sec:reproducibility` on page 9; 0 LaTeX errors; 0 undefined refs.

---

## 4. Outstanding patch plan (Day-14 owners + deadlines)

### Tier 1: MAJOR (6 outstanding) — patch by Day-14 EOD

| ID | Patch | Cost | Location | Owner |
|---|---|---|---|---|
| A | Add HF checkpoint commit SHAs (1B + 3B) to §A.2 Models paragraph | ~1 sentence / ~2 TeX lines | TeX `sec:reproducibility` §A.2 | Josh |
| B | Add exact CLI reproduction commands to §A.4 closing (Track F invocation; --seed 42 --model X --out Y) | ~3 lines / code block | TeX `sec:reproducibility` §A.4 | Josh |
| C | Add HF auth + Llama license-accept note to §A.2 | ~1 sentence | TeX `sec:reproducibility` §A.2 | Josh |
| D | Pin `transformers ≥ 4.45`, `accelerate ≥ 1.0`, `numpy < 2.0` in §A.1 | ~1 sentence | TeX `sec:reproducibility` §A.1 | Josh |
| E + F + G + H (coordinated) | Title + abstract + §1.1 interp-track reframe: (a) reverse title colon-ordering so "Probing Dimensional Escape" or similar leads; (b) reorder abstract S1-S3 to lead with geometric question + Dimensional Escape, defer algorithm details to S3-S4; (c) add infrastructure-generator framing to C1; (d) saturate ARR keywords ("probing"/"interpretability"/"knowledge tracing") in abstract | ~80 words net (rewrites, not additions) | TeX title line ~42 / abstract line ~56 / §1.1 line ~82 | Josh |
| I | Promote cross-scale MCQ direction-flip to §1 (or §1.1 contributions): add 1-sentence highlight | ~30 words | TeX §1 line ~73 or §1.1 C2 | Josh |

**Day-14 wall-time estimate**: ~2-3 hours of focused TeX editing. All patches are surgical; none require new experiments.

### Tier 2: MODERATE (17 outstanding) — patch best-effort by Day-14 EOD

Highest-leverage Tier-2 items:

1. **Axis 1 Q6**: Patch `measure_lipschitz_drift.py` to use `accumulator.residual_norm()` for full-layer coverage OR add §A appendix-impl-notes sentence noting the per-tensor algorithm symmetry makes the layer-0 K probe a faithful representative. (~1 sentence) — addresses reviewer skepticism on "0.00e+00 drift over 200 cycles" claim.
2. **Axis 2 Q1**: Append operational signatures to def:dim_escape (decoupling + argmax-prior amplification). (~50 words) — anchors term in measurable quantities.
3. **Axis 3 GAP-5c.1**: Add steering-direction sanity-check checksum (l2 norm of the seed-42 unit vector). (~10 words) — closes the "did I generate the right random vector?" reviewer question.
4. **Axis 3 GAP-8.2**: Add `--smoke` invocation as sanity-check in §A.4 Track F. (~1 sentence) — closes the "is my setup right before I launch the 33-minute sweep?" reviewer question.
5. **Axis 5 Pass G**: Add 9th Limitation acknowledging the algorithmic-framework-vs-experimental-instantiation scope gap (the algorithm supports UCB1 but the experiments enumerate). (~3 lines) — pre-empts Axis 5 Pass D's other side.

Remaining MODERATEs: defer to Phase-3 camera-ready expansion or document in cover note.

### Tier 3: MINOR (13 outstanding) — best-effort; document deferred in cover note

Cosmetic / clarity items. Aggregate cost <30 lines if all addressed. Recommend addressing during the Day-14 TeX-editing session at zero marginal cost when adjacent Tier 1 / Tier 2 patches are being applied.

### Tier 4: MONITORING — page-budget verification (Axis 5 Pass P)

Day-14 PDF page-count gate: after all Tier 1 + Tier 2 patches land, re-run smoke-compile and verify body still ≤8 pp. The 6 outstanding MAJORs add roughly: Tier-A/B/C/D ≈ +0.10 pp (appendix-only), Tier-E/F/G/H/I ≈ -0.10 to +0.05 pp (mostly rewrites in body title/abstract/§1.1). Net expected delta: near-zero. The headroom margin is small (currently 8pp exactly; the orchestrator prompt's commit `bb83ddb` smoke-compile confirmed `sec:reproducibility` on page 9). If Day-14 patches push body to 9pp: trigger contingency cut plan from [docs/logs/2026-05-15_page-budget-cut-plan-v1.md](2026-05-15_page-budget-cut-plan-v1.md) Tier 2 (additional MOVEs).

---

## 5. Co-author feedback integration plan (Day-14)

Day-11 send-out reached Tianyu + Max + Alaa with a 4-day async window. Day-14 (2026-05-19) integrates whatever feedback has arrived alongside the Tier 1 patches above. The two-cover-note pattern from [docs/logs/2026-05-14_parallel-review-plan.md](2026-05-14_parallel-review-plan.md) §"Cover-note language for Day-14 final synthesis" applies.

If a co-author flags any of the Tier 1 MAJORs from a different angle, fold their wording into the Day-14 patches (their version usually beats Round-1/Round-2-agent wording).

If a co-author flags an issue NOT surfaced by the 5-axis review, treat it as Tier-0 priority (new finding outside our adversarial-process coverage). Highest probability of this: domain-specific or upstream-citation concerns Tianyu would catch that the parallel review's outside-view bias would miss.

---

## 6. Submission-readiness assessment (honest outside-view)

**Strengths after Day-12 patches**:
- 0 code-vs-paper SEVERE (Axis 1 CLEAN on FP32 accumulator + W_K projection + entropy reward)
- Day-11 hedges (Pass M) + M1-M5 patches close the worst framing overclaims at abstract / §1 / §3.2 / §5 layer
- §A SEVERE system-prompt fix removes the load-bearing reproducibility blocker
- Body fits 8pp ACL hard limit (confirmed via .aux)
- All 5 axes' SEVERE+MAJOR findings either patched (Day-12) or queued for Day-14 (~2-3 hours of work)

**Outstanding risks for Day-15 submission**:
- **Risk 1 (HIGH if not patched)**: Title/abstract/§1.1 interp-track-fit (E/F/G/H above). If Day-14 patches don't land, ACL routing may assign the paper to ML/Efficient-Methods reviewers, who will read it as a Methods paper and reject for "not novel enough on the algorithm side" rather than evaluating the interp contribution. Mitigation: prioritize the coordinated title/abstract/§1.1 rewrite first thing Day-14.
- **Risk 2 (MEDIUM)**: §A still missing 4 reproducibility-blocking items (HF SHAs, CLI commands, auth, version pins) — Tier 1 A/B/C/D. Reviewer cannot reproduce without them; this would be a revision-request-tier critique rather than reject-tier, but it weakens the paper's reproducibility-track standing.
- **Risk 3 (LOW)**: Page-budget regression if Day-14 patches inflate body. Mitigation: monitor at smoke-compile; trim adjacent Tier-2 candidates if needed.
- **Risk 4 (LOW)**: Co-author feedback may surface a Tier-0 finding outside the 5-axis coverage. Mitigation: Day-14 schedule has buffer for one new finding.

**Acceptance probability estimate**: Given Day-11 baseline ~45-55% (per recast_action_plan), Day-12 patches probably moved this +5pp (Round-1 MAJORs addressed). Day-14 Tier-1 patches landing fully would add another +5pp. Submission-ready ~55-65% acceptance probability range if all Tier-1 patches land on Day-14. Submitting without Tier-1 E/F/G/H ≈ 40-50% (significant routing risk).

---

## 7. Decision rule outcomes (per plan-doc §"Synthesis protocol" / orchestrator escalation triggers)

Per [docs/logs/2026-05-17_day-12-session-prompt.md](2026-05-17_day-12-session-prompt.md) HARD CONSTRAINTS:
- **Axis 1 SEVERE → immediate Tianyu escalation**: NOT triggered (Axis 1 = 0 SEVERE).
- **≥1 SEVERE OR ≥3 MAJOR total → pause Round 2 + surface to Josh**: Triggered after Round 1 (6 MAJOR). Round 2 paused; Josh reviewed; Option B framing signed off; M1-M5 patches applied; Round 2 fired Day-12 PM.
- **Round 2 SEVERE finding**: §A system-prompt mismatch surfaced; NOT an Axis-1 code-vs-paper SEVERE so no advisor escalation; patched same-day commit `bb83ddb`.

**Day-12 closes with**: all SEVERE patched, 6 MAJOR outstanding (deliverable Day-14), 17 MODERATE outstanding (best-effort Day-14), 13 MINOR outstanding (cosmetic; documented as deferred in cover note if not addressed).

---

## 8. Day-13 + Day-14 + Day-15 task list (post-synthesis)

### Day-13 (2026-05-18) — buffer day

The original plan-doc Day-13 schedule was: AM fire Round 2 / PM synthesize. Both compressed into Day-12 PM. Day-13 becomes a **buffer / co-author-feedback-integration day**:
- Read any co-author replies arriving in the Day-11 async window
- Light Tier-3 MINOR patches at zero marginal cost
- Optional: re-run smoke-compile to confirm Day-12's PDF state stable

### Day-14 (2026-05-19) — synthesis-driven patch day

Wall-time estimate: **2-3 hours focused TeX editing**.

1. Coordinated title + abstract + §1.1 rewrite (E/F/G/H/I) — highest leverage; do first.
2. §A reproducibility patches (A/B/C/D) — 4 surgical paragraphs in §A.1/§A.2/§A.4.
3. Tier 2 MODERATE best-effort patches (5 highest-leverage items above).
4. Co-author feedback integration (whatever arrived in the async window).
5. PDF page-count gate; trigger contingency cuts if 9pp.
6. Final REVISION-marker audit.

### Day-15 (2026-05-20) — ARR portal submission

Click-through submission per [docs/logs/2026-05-10_14-day-execution-plan.md](2026-05-10_14-day-execution-plan.md) Day-15 plan. Responsible NLP Checklist + Croissant supplementary package + anonymized supplementary repo (built Day-12 in parallel).

ARR deadline 2026-05-25 (T-5 days from Day-15). EMNLP 2026 commitment 2026-08-02.

---

*End of 5-axis parallel-review synthesis. Day-12 EOD 2026-05-17. Total adversarial-review wall-time across both rounds: ~25 minutes (5 parallel sub-agents × ~5 min wall each, twice). Patch wall-time Day-12: ~1 hour (5 M-patches + Pass-2 hedge + SEVERE fix + 3 commits + push + memory updates). Outstanding patch work for Day-14: ~2-3 hours focused TeX editing. Net Day-12 outcome: 1 SEVERE + 6 of 14 MAJORs patched; remaining 8 MAJORs queued for Day-14 with concrete patch locations + cost estimates.*
