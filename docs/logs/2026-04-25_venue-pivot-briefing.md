# Venue Pivot Briefing — KV-Cache Inception
**Date**: 2026-04-25
**Audience**: Josh, Max, Tianyu Shi
**Author framing**: Opus, advising as if a research professor with project-level familiarity
**Status**: Decision needed; affects the next 4–20 weeks of work

> **Caveats up front.** I'm reasoning from the repo state, the paper draft, and the session log. Three things I don't actually know and you must verify before acting on this briefing:
> 1. Max's actual available hours/week over the next 4 weeks.
> 2. Tianyu's real threshold on "aim high" — does workshop count? Does TMLR count?
> 3. H100 access — is it on for this work or only for the original Phase B plan?
> Replace my placeholders with real numbers before deciding.

---

## TL;DR

EMNLP main is a stretch the math doesn't quite support given current code-paper divergence and a 3-person team. EMNLP Findings is plausible *if* Exp 1 + Exp 2 land cleanly by May 18 and the flagship efficiency claim gets honest qualification. The realistic high-quality home for this paper is **TMLR (rolling) with a NeurIPS workshop in parallel**. The Hungary motivation is a legitimate input but should determine *how hard you work*, not *which deadline you target*.

---

## 1. Honest status assessment

**Validated empirically:**
- Theorem 1: FP32-accumulator reversibility (drift = 0.00e+00 over 200 cycles, 2026-04-16).
- KV-cache mutability probe.
- 10-node MCTS smoke test; telemetry pipeline emits `T_t`, OEI, TDS.

**Claimed in the paper, not yet measured:**
- Exp 1 alpha sweep at 1B and 3B.
- Exp 2 latent signal quality vs random KV-cache baseline (baseline not yet implemented).
- The flagship efficiency claim. Per the DRAFT-HOLD blocks at tex:520–544, the proposition asserts `O(M_KV + d·K_acc)` ≈ ~40 GB; actual code-measured floor is ~2–3·M_KV due to the baseline-clone and full-shape accumulators. **The headline number 40 GB vs 9.7 TB is currently incorrect.** Reviewer who checks code will catch this.
- Procrustes 1B↔7B (Exp 5, now Future Work).

**Code-paper divergences flagged in the draft (DRAFT-HOLD blocks):**
- §4.1 says "LAT procedure" but `whitebox.py` uses difference-in-means, not PCA (GAP-C1-05).
- §4.2 implies per-layer steering vectors `d_K^(ℓ)`; code broadcasts a single middle-layer vector (P4-B note).
- §6 Memory Complexity Proposition: Phase 2 approximation, not validated (GAP-C2-03/04).

**Net assessment:** the paper is currently a *position-and-method paper* anchored by one strong theorem. To clear main-track review at any top conference it needs at minimum (a) Exp 1 results, (b) Exp 2 results, (c) honest qualification or code fix for the memory claim. Without those, a careful reviewer can desk-reject for reproducibility/code mismatch.

---

## 2. Venue options

I'll cover five real options and rule out the rest.

### A. EMNLP Main (deadline May 25, ~22–25% accept rate)
- **Fit:** mediocre. EMNLP is an NLP venue; this is a methods/systems paper that happens to be about LLMs. NLP reviewers will want to see strong empirical NLP results (e.g., red-teaming benchmarks, jailbreak rates) — not just method validation.
- **Capacity:** ~30 days from now. With Exp 1 (5 days), Exp 2 (7 days), code-paper cleanup (3–5 days), paper rewrite with results (5–7 days), polish (3 days), buffer (3 days), you are at ~26–30 days for two people. No margin.
- **Realistic accept odds if everything lands:** 12–18%. Penalty for venue-fit mismatch. Single area-chair desk-reject risk if code-paper inconsistency surfaces.

### B. EMNLP Findings (same deadline, ~30–35% accept rate)
- **Fit:** better. Findings track absorbs solid methods work that doesn't have flagship empirical results.
- **Capacity:** identical to (A). Same prep work — you don't choose the track at submission.
- **Realistic accept odds if everything lands:** 22–30%. Best Hungary play if you commit aggressively.

### C. NeurIPS 2026 workshops (deadlines typically June–August)
- **Fit:** excellent. Specific candidates:
  - SafeML / Trustworthy ML / Red Teaming workshops — telemetry + alignment-faking framing is exactly the audience.
  - Mechanistic Interpretability workshop — H-Neurons + LAT framing fits.
  - Foundation Models / Open-Source Models — KV-cache method angle.
- **Capacity:** comfortable. 4+ months of breathing room.
- **Acceptance:** ~50–60%. NeurIPS audience without main-track competition.
- **Tianyu calibration question:** does a NeurIPS workshop satisfy his prestige bar? Workshops at NeurIPS are not throwaway venues — many high-impact papers debut there — but they are not the same line item as a NeurIPS main paper on a CV.

### D. TMLR (Transactions on Machine Learning Research, rolling submissions)
- **Fit:** very strong. TMLR explicitly accepts on rigor and clarity, not on novelty/impact. Methods papers with solid theorems and clean experiments are exactly its diet.
- **Capacity:** no deadline pressure. Aim for a clean v1 by mid-June.
- **Accept odds:** ~30–50% on first submission, with reviewer feedback that often guides resubmission. Average decision time ~2–3 months.
- **Indexed, citable, peer-reviewed.** Well-recognized in ML; somewhat less recognized in NLP-specific tenure committees.

### E. Sit on it until NeurIPS 2027 (May 2027 deadline)
- **Pros:** a year of additional results — Procrustes done, H100 scaling done, possible new co-authors onboarded.
- **Cons:** opportunity cost; field moves fast; another lab may publish a similar reversible-MCTS approach in 12 months, eroding novelty.
- **Risk-adjusted:** generally not the right call when you have a reasonable shorter-term venue.

**Ruled out:** ICLR (deadline passed), AAAI (passed), ACL (passed). NeurIPS 2026 D&B (current deadline you're abandoning).

---

## 3. Capacity reality check

The binding constraint is people-time. Approximate person-days available between now (2026-04-25) and each deadline, assuming Max + Josh ~ 0.7 FTE each, Tianyu ~0.05 FTE:

| Deadline | Calendar days | Estimated person-days | Required person-days (rough) | Margin |
|---|---|---|---|---|
| EMNLP May 25 | 30 | ~45 | ~50 | **negative** |
| TMLR mid-June | 50 | ~75 | ~50 | comfortable |
| NeurIPS workshop Sept | ~150 | ~225 | ~50–80 | very comfortable |

The EMNLP timeline is doable only if (a) both Max and Josh treat this as their primary commitment for 4 weeks, (b) there are no further pivots, (c) Tianyu commits to <48-hour review turnarounds.

The implicit assumption "we'll just push hard" is the failure mode I see most often. A 4-week sprint with no margin and a real-world bug or a single failed experiment puts you in late-May limbo, with neither EMNLP submitted nor a clean v1 for elsewhere.

---

## 4. The collaborator question

**Don't recruit from the open internet for a 4-week deadline.** Onboarding cost dominates marginal contribution; IP risk is real but secondary to the failure mode of "collaborators who don't deliver."

Better moves, in order:
1. **Ask Tianyu for an RA.** Lab RAs are vetted, his reputation handles the IP question structurally, and they may already be familiar with adjacent infrastructure. This is the highest-EV move you can make in the next 48 hours.
2. **Pre-print on arXiv before any external visibility.** Timestamp = priority. Do this regardless of venue choice.
3. **If you do recruit later (e.g., for Phase B), do it with clearly scoped contributions and a written pre-authorship agreement.** "Person X handles Procrustes alignment for 7B transfer" is a contribution; "person X is interested in MCTS" is not.

The internet-collaborators-without-getting-stolen anxiety is largely solvable by (a) pre-printing first and (b) only sharing scoped sub-tasks rather than the full draft.

---

## 5. Recommendation

**Drop EMNLP main.** The math doesn't support it.

**Pick one of the following two paths.** Both are defensible; they reflect different risk appetites.

### Path A — Aggressive: EMNLP Findings, May 25
- Commit fully. No further pivots. No scope creep.
- Week 1 (Apr 26–May 2): Exp 1 alpha sweep on 1B + 3B; honest qualification on memory claim drafted.
- Week 2 (May 3–9): Exp 2 random-KV baseline implementation + run; code-paper consistency pass.
- Week 3 (May 10–16): full paper rewrite integrating results; OEI Option B reframing finalized.
- Week 4 (May 17–23): polish, citation audit, submission prep.
- May 24: hold day. May 25: submit.
- Tianyu must commit to <48-hour reviews on weeks 3–4.
- If any week slips, **abandon and switch to Path B by May 11** (don't wait until May 24 to pivot).

### Path B — Sustainable: TMLR mid-June + NeurIPS workshop September (recommended)
- Mid-June: clean TMLR submission. By "clean" I mean: experiments run, code-paper consistency established, memory claim either fixed in code or honestly qualified in paper. No DRAFT-HOLD blocks remaining.
- TMLR review window (~2–3 months) coincides with NeurIPS workshop deadline preparation.
- September: targeted workshop submission (SafeML or MechInterp) using the TMLR-quality draft, possibly with workshop-specific framing tweaks.
- If TMLR rejects, you have actionable reviewer feedback for the workshop submission and time to incorporate it.
- December: NeurIPS workshop presentation. Paper has a formal published home (TMLR or workshop proceedings) and presentation visibility.

**My personal lean: Path B.** Reasoning:
1. The paper deserves to be published with the experiments actually run, not implied. EMNLP gets you the deadline; TMLR gets you the publication you'd actually be proud of.
2. The Hungary motivation is real. Hungary will still be there next year. Your family will still be there. EMNLP 2026 vs EMNLP 2027 is not a meaningful career difference.
3. Tianyu's "aim high" preference is satisfied by NeurIPS workshop + TMLR — both are recognized venues. If he disagrees, he should put an RA on the project to make EMNLP genuinely tractable.

If Tianyu does provide an RA in the next 48 hours, Path A becomes substantially more viable; revisit then.

---

## 6. Bias check on me

I'm risk-averse on tight deadlines because I've seen the failure mode (desk-reject for reproducibility) more than the success mode (heroic last-week sprint that works). My recommendation reflects that. If you have evidence of comparable 3-person teams hitting EMNLP from a similar starting state, weight my recommendation accordingly. Tianyu's calibration on this is more useful than mine.

---

## 7. Questions you need to answer before locking the path

1. **Max's hours/week, honestly.** Not aspirational — actual. If under 25, Path A is off the table.
2. **Tianyu's prestige threshold.** Workshop yes/no. TMLR yes/no.
3. **H100 access status.** If on, the experiment story strengthens for any venue. If off, the paper is firmly a 1B/3B story and we lean into that framing.
4. **Procrustes Exp 5 cost-benefit.** Is 1B↔7B alignment a 5-day task or a 15-day task? The answer changes whether it can fit in any of the timelines.
5. **Pre-print decision.** Are you willing to put a v0 on arXiv before submission? (Recommended yes, regardless of path.)

Get Tianyu in a 30-minute meeting with these five questions. Decision should follow within 24 hours of that meeting.

---

## 8. Immediate next step (independent of path)

Resume the §5.4 metrics cull conversation that we paused this session. Whatever path you pick, the paper needs to be internally consistent before the next decision. The TDS keep/cull and §2.2 reframing decisions (preserved verbatim in `docs/logs/2026-04-25_session-log.md`) are blocking work. 30 minutes of decisions there unblocks the rest.
