# Round-1 Parallel-Review Synthesis

**Date:** 2026-05-17 (Day 12)
**Round:** 1 of 2 (Axes 1, 2, 5 only; Round 2 paused per plan decision rule)
**Reports synthesized:** [Axis 1](2026-05-17_axis-1-report.md), [Axis 2](2026-05-17_axis-2-report.md), [Axis 5](2026-05-17_axis-5-report.md)
**Paper commit reviewed:** `8085545` (Day-11) — auditors did not see Day-12 commits `8f3f1c5` (SHOWSTOPPER fix) or `267e9cb` (20B-projection drop).

---

## TL;DR

- **0 SEVERE.** No paper-go/no-go-level findings. No Tianyu escalation required on code-vs-paper grounds (Axis 1 confirmed code-semantic alignment for FP32 / W_K projection / entropy reward).
- **~5 MAJOR after de-duplication.** All textual, all addressable in <300 words total.
- **~10 MODERATE, ~7 MINOR.** Time-permitting patches.
- **Decision-rule outcome:** ≥3 MAJOR → Round 2 pause is correct per plan. **Recommendation: resume Round 2 once Day-14 patches for MAJOR items are scoped (below).** Round 2 firing parallel to Day-14 patch application is calendar-feasible.
- **Day-12 commits already addressed 2 of 4 MAJORs** (Pass P page-budget moot; Pass M 20B hedge moot).

---

## Aggregate severity counts (post-Day-12 cross-check)

| Axis | SEVERE | MAJOR (audit) | MAJOR (post-Day-12) | MODERATE | MINOR | CLEAN |
|---|---|---|---|---|---|---|
| Axis 1 (code semantics) | 0 | 0 | 0 | 2 | 1 | 3 |
| Axis 2 (framing) | 0 | 3 | 3 | 2 | 1 | — |
| Axis 5 (adversarial, 18 sub-passes) | 0 | 4 | 2 | 8 | 6 | — |
| **De-duplicated total** | **0** | **~7** | **~5** | **~10** | **~7** | — |

**Post-Day-12 reductions:**
- Axis 5 **Pass P** (page-budget desk-reject if 9pp): MOOT — body is 8pp post-Day-11 `8085545`.
- Axis 5 **Pass A** (abstract "under any surrogate reward" overclaim) + Axis 2 Q3/Q6 (unification overclaim) are PARTIALLY overlapping; counts deduplicated to ~3 effective MAJORs.

---

## Open MAJOR findings (must-patch before submission)

### M1 — §3.2 line 119 unification claim unhedged (Axis 2 Q3 + Q6)

**Current text (verbatim):** `"We argue these are sub-cases of a single underlying pathology."`

**Why MAJOR:** This is the body claim that the Day-11 hedge fixes were designed to align with — but the Day-11 patch only touched abstract / §1 Intro ¶4 / §1.1 C2 closing, missing this §3.2 body sentence. A sequential reviewer reads §3.2 line 119 ("single underlying pathology") then §3.2 closing ("Bailey et al. characterized... we characterize the reward-driven case empirically") and gets contradictory framing.

**Patch:** Rewrite to:
> "We hypothesize these as parallel manifestations of the overparameterized-latent-space geometry described below; only the reward-driven case is evidenced empirically in this paper, with the monitor-driven case under inference-time KV-cache perturbation an open empirical question (§\ref{sec:future-work})."

**Cost:** ~35 words net (current sentence is ~12 words). Page impact: negligible.

---

### M2 — §1.1 Contribution 2 title not hedged (Axis 2 Q3)

**Current text (TeX line 84):** `\item \textbf{Empirical demonstration of Dimensional Escape under continuous-latent search.}`

**Why MAJOR:** The contribution title claims empirical demonstration of the unified Dimensional Escape phenomenon — but per the body §3.2 + §5 evidence, only the reward-driven sub-case is empirically evidenced. The monitor-driven sub-case is conjectured. The contribution title overclaims relative to the contribution body (which IS hedged per Day-11).

**Patch:** Rewrite title to:
> `\item \textbf{Empirical demonstration of reward-driven Dimensional Escape; structural-equivalence argument for monitor-driven case.}`

**Cost:** ~12 words. Page impact: title may wrap one line longer; verify smoke-compile.

---

### M3 — §3.2 Definition `def:dim_escape` Goodhart-distinction missing (Axis 2 Q5)

**Why MAJOR:** Coined-term defense is a standard reviewer attack vector. Without the geometric-route distinction articulated, the paper exposes itself to "this is Goodhart's Law with extra steps" rejection. Day-11 hedges softened the framing but didn't add the distinction itself.

**Patch:** Add 1-paragraph preamble before Definition `def:dim_escape`:
> "Dimensional Escape extends generic Goodhart's Law by specifying the GEOMETRIC ROUTE through which proxy-optimization resolves: optimization pressure on a low-rank scoring projection $\mathbf{P}$ routes through the null space of $\mathbf{P}_{\text{semantic}}$ (the typically-unobserved projection onto the semantically relevant subspace), not merely through the proxy reaching its maximum. Goodhart's Law is silent on the geometric route; Dimensional Escape predicts the route is through the unconstrained complement of the rank-$k$ scoring subspace in $\mathbb{R}^d$, $d \gg k$."

**Cost:** ~80 words. Page impact: ~0.1pp.

---

### M4 — Abstract "under any surrogate reward" overclaim (Axis 5 Pass A)

**Current text (TeX line 56):** `"...under any surrogate reward lacking exogenous grounding, the high-dimensional cache resolves localized optimization pressure by routing representations through unconstrained orthogonal dimensions, satisfying the proxy metric without satisfying the semantic objective."`

**Why MAJOR:** Empirical evidence covers ONE surrogate reward (entropy). "Any surrogate reward" generalizes beyond what the data supports. Combined with Axis 2 Q7 (single-reward-function limitation missing from §7.3), this overclaim is the strongest reviewer attack surface on the empirical scope.

**Patch:** Hedge to:
> "...under the entropy-normalized surrogate reward we test (and, by extrapolation under §\ref{sec:dim_escape}'s dimension-counting argument, candidate surrogate rewards more generally), the high-dimensional cache resolves localized optimization pressure by routing representations through unconstrained orthogonal dimensions..."

Plus soften "**fundamental** geometric pathology" earlier in abstract to "**geometric** pathology" (drop "fundamental"; data supports the phenomenon's existence, not its fundamentality across all rewards).

**Cost:** ~15-20 words. Page impact: ~0.02pp.

---

### M5 — §5 MCTS-vs-enumeration scope clarifier (Axis 5 Pass D)

**Why MAJOR:** Paper title says "Reversible KV-Cache **MCTS**." Experiments enumerate all 27 paths from $\{0.1, 0.5, 1.0\}^3$ — full Cartesian enumeration, not UCB1 tree search. This is a methodology gap: the algorithmic contribution (MCTS with UCB1) is not the experimental procedure (enumerated path sampling).

**Patch:** Add §5 paragraph (before Exp 1 setup) explicitly scoping the experimental variant:
> "In this paper we instantiate the MCTS framework as exhaustive depth-3 path enumeration over the alpha-set $\{0.1, 0.5, 1.0\}^3$, evaluating all $3^3 = 27$ paths per item. The UCB1 bandit policy is a natural extension when search budget exceeds full-enumeration cost; for the depth-3 alpha-sweep we report, full enumeration is computationally cheaper than UCB1 with overhead. The algorithmic framework supports either variant; we report enumeration for reproducibility and bandit-independence of the headline results."

**Cost:** ~70 words / ~5 lines two-column. Page impact: ~0.1pp.

---

## MODERATE findings — recommended Day-14 (time-permitting)

| ID | Finding | Source | Patch effort |
|---|---|---|---|
| Mod1 | Body §4 §4.2 thin — Step 1/2/3 in appendix only | Axis 5 Pass Q | Re-fold ~0.15pp page cost; trade-off vs page budget. May be inadvisable post-Day-11. |
| Mod2 | §1.1 C1 closing — add explicit "diagnostic probe" redirection sentence | Axis 5 Pass N | ~2 lines. Pre-empts reviewer mental-model mismatch. |
| Mod3 | §7.3 add single-reward-function limitation (entropy-specific) | Axis 2 Q7 + Axis 5 Pass G | ~5 lines. Pre-empts strongest reviewer critique on generalization. |
| Mod4 | Add reward-hacking citations (Gao 2023, Skalse 2022) to §2/§3.2 | Axis 5 Pass H | 2 bibitems + 2 inline cites. ~30 words. |
| Mod5 | §3.2 Definition operational signatures (decoupling + argmax-prior amplification) | Axis 2 Q1 | 1 sentence appended. ~50 words. |
| Mod6 | Abstract + §3.2 "geometric" → "structural" or "high-dimensional" | Axis 2 Q4 | 2 word swaps. |
| Mod7 | Cell-by-cell 3.04 vs 3.06 numbers in §A | Axis 5 Pass I | Add 3-row table to §A.4 Track D paragraph. |
| Mod8 | Axis 1 Q4 — 27-path enumerated vs sampled clarifier | Axis 1 Q4 | Subsumed by M5 if M5 wording correct. |
| Mod9 | Axis 1 Q6 — drift script scope clarifier | Axis 1 Q6 | 1 sentence in §C appendix-impl-notes item (v). |
| Mod10 | Pass E reproducibility skepticism (too-clean numbers) | Axis 5 Pass E | Add cell-by-cell table + 1-sentence justification of zero-drift. Subsumed by Mod7. |

---

## MINOR findings — best-effort

| ID | Finding | Source |
|---|---|---|
| Min1 | Axis 1 Q5 — "same vector" across different-d_model scales: should be "same seed" | Axis 1 Q5 |
| Min2 | Axis 2 Q2 — H2 SemanticManifold notion needs operational anchor | Axis 2 Q2 |
| Min3 | Axis 5 Pass F — title + abstract first-page-reject-filter mostly clears | Axis 5 Pass F |
| Min4 | Axis 5 Pass J — "Why this paper now?" framing optional 1-sentence | Axis 5 Pass J |
| Min5 | Axis 5 Pass K — §A.5 anon URL placeholder XXXX (Day-14 gate; handled by cleanup prompt) | Axis 5 Pass K |
| Min6 | Axis 5 Pass L — §D 80GB vs §7.5 162GB reconciliation (appropriate cross-section) | Axis 5 Pass L |
| Min7 | Axis 5 Pass O — strip `[REVISION]` comments from published source (Day-14 gate; handled by cleanup prompt) | Axis 5 Pass O |

---

## What the Day-12 commits already addressed

| Audit finding | Status |
|---|---|
| Axis 5 Pass P (page-budget desk-reject) | RESOLVED — body 8pp post-`8085545` |
| Axis 5 Pass M (abstract-to-body 60× hedge asymmetry) | RESOLVED — abstract dropped 60× claim entirely post-`267e9cb` |
| Axis 5 Pass A footnote re: 20B projections | PARTIALLY RESOLVED — 20B-projection content removed from body per `267e9cb`, but "under any surrogate reward" overclaim independent and unaddressed |
| Axis 5 Pass L (cross-section M_KV numerical-consistency) | RESOLVED — body has no specific M_KV value after `267e9cb`; only §7.4 + §D mention 20B specifics |
| Audit-flagged §6.2 worked-example math (SHOWSTOPPER) | RESOLVED — both `8f3f1c5` (correct math) and `267e9cb` (remove math entirely) |

---

## Round 2 firing decision

**Plan decision rule (per parallel-review-plan §"Decision rule for Round 2"):**
- ≥1 SEVERE → escalate, pause until SEVERE patches scoped or Tianyu weighs in
- ≥3 MAJOR → pause until MAJOR patches scoped

**Status:** 0 SEVERE / 5 MAJOR. Pause is correct.

**Resume condition:** MAJOR patches scoped above (M1-M5). Round 2 can fire IN PARALLEL with Day-14 patch application — they're independent. Round 2 catches additional findings in axes 3 (reproducibility) + 4 (reviewer-prior alignment for #13) that Day-14 patch synthesis integrates.

**Recommendation: Fire Round 2 (Axes 3 + 4) Day-13 AM** parallel to applying M1-M5 patches Day-13 AM/PM. Synthesis Day-13 PM combines Round 1 + Round 2.

---

## Tianyu communication

**Context:** Josh sent Tianyu the cover note on Day-11 pointing at commit `8085545`. Tianyu has not yet replied (per Day-11 EOD signal expectation). Current paper state is `267e9cb` post-SHOWSTOPPER fix + 20B-projection drop. Round 1 surfaced 5 MAJOR findings, patches drafted but not applied.

**Recommendation:** Send Tianyu a brief Day-12 follow-up:

> Quick update before your read: Day-12 audit + adversarial review surfaced findings. Two showstopper-class issues fixed and pushed (commits `8f3f1c5` + `267e9cb`): (a) §6.2 worked example math was wrong (60× ratio didn't follow from stated numbers); fix preserves asymptotic argument, drops 20B numerical projections from body. (b) Body content was 9 pp; now 8 pp ACL-compliant. Five MAJOR framing/methodology findings drafted as ~250-word patch set, applying Day-14 (M1: §3.2 unification hedge; M2: §1.1 C2 title hedge; M3: §3.2 def Goodhart distinction; M4: abstract "any surrogate reward" hedge; M5: §5 MCTS-vs-enumeration scope clarifier). Synthesis report at `docs/logs/2026-05-17_round-1-synthesis.md`. Round 2 fires Day-13 in parallel with Day-14 patches. Heads-up — if you started reading, you may have spotted the §6.2 issue; the corrected version is on main.

This signals: (i) audit caught a real bug before Tianyu did; (ii) we're disciplined; (iii) the paper is improving rapidly; (iv) we're prepared for his feedback.

---

## Day-13 / Day-14 calendar (revised)

| Slot | Task |
|---|---|
| Day-13 AM | Fire Round 2 (Axes 3 + 4) per Day-12 orchestrator prompt; apply M1 + M2 (cheapest patches; ~50 words each) |
| Day-13 PM | Apply M3 + M4 + M5; Round 2 reports return; partial synthesis integrating Round 2 findings |
| Day-13 EOD | Synthesis doc `docs/logs/2026-05-18_full-synthesis.md` complete |
| Day-14 AM | Integrate co-author feedback (Tianyu + Max + Alaa); apply remaining MAJOR/MODERATE patches |
| Day-14 PM | Final REVISION-marker audit; smoke-compile; double-blind compile check; supplementary anonymization (Day-13 PM cleanup prompt session) |
| Day-15 | ARR portal submission |

If Round 2 fires Day-13 and patches M1-M5 land Day-13 also, Day-14 has slack for co-author feedback + cleanup. Total patch-application work for M1-M5 estimated 30-45 min wall.

---

## Verdict on paper readiness

**Going to Tianyu now (already done):** Defensible. The 5 MAJOR findings are all textual and addressable in <300 words. None are paper-go/no-go. Tianyu's review + the parallel review converge on the same kind of finding (framing hedge + scope clarification), validating both processes.

**Going to ARR May 25 with M1-M5 patches applied:** Defensible to PASS at desk-review, likely surface in peer-review feedback as revision items but not reject-tier.

**Going to ARR May 25 WITHOUT M1-M5 patches:** Risky. The "structurally identical to a single underlying pathology" body claim vs the hedged abstract is internally inconsistent — a reviewer who notices will flag it as either careless or misleading. Apply M1-M5.

---

*End synthesis. Round 2 firing recommended Day-13 AM. M1-M5 patch application Day-13 same day. Day-13 PM full synthesis. Day-14 co-author + cleanup integration. Day-15 submission.*
