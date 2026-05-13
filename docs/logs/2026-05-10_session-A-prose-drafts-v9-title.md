# Session A Prose Drafts v9 — Title Lock — 2026-05-10

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits) — Day 2 of ARR-cycle execution plan
**Status:** Day-1 title-lock decision per ARR-cycle plan v2 §2. Replaces the Option α+ title at canonical TeX:39.

## Title-lock decision

**Selected title (Recommended):**

> **Reversible KV-Cache MCTS: A Memory-Bounded Probe of Dimensional Escape in Frozen Language Models**

Three candidates were evaluated; this one selected. Rationale below.

## Candidate evaluation

### Candidate 1 (SELECTED): "Reversible KV-Cache MCTS: A Memory-Bounded Probe of Dimensional Escape in Frozen Language Models"

**Pros:**
- Lead with the algorithm (reviewer recognition); subhead with the unified empirical finding (Dimensional Escape) and the methodological frame (Memory-Bounded Probe).
- "Memory-Bounded" anchors Contribution 1 (Track D's 3.04-3.06× M_KV bound, 60× reduction at $(b,d)=(3,5)$).
- "Frozen Language Models" precludes confusion with gradient-trained adversarial work (Bailey 2024, RL-Obfuscation).
- "Dimensional Escape" appears in the title, abstract, §3.2 Definition — strong cross-referenced terminology trail.
- Length: 14 words / 88 characters. Within EMNLP norms (typical title 10-20 words).

**Cons:**
- "Reversible KV-Cache MCTS" puts machinery before mechanism in the headline. Some reviewers may prefer mechanism-first.
- Doesn't explicitly say "Goodhart" or "alignment" — relies on the abstract to pull in mechanistic-interpretability and alignment-evaluation readerships.

**Reviewer-routing prediction:** mechanistic interpretability + test-time compute + monitor-design tracks. Mid-strong fit for EMNLP Findings.

### Candidate 2 (NOT SELECTED): "Dimensional Escape in Continuous Latent Search: An Empirical Study with Reversible KV-Cache MCTS"

**Pros:**
- Mechanism-first headline.
- "Empirical Study" signals "this paper measures something, doesn't propose a solution" — honest framing of negative-result-as-positive-contribution.

**Cons:**
- "Empirical Study" is slightly weaker than naming the contribution; some venues prefer claim-first titles.
- Sub-headline "Reversible KV-Cache MCTS" demotes the algorithmic contribution to a method-mention.
- Length: 15 words / 100 characters. Slightly longer, harder to scan.

### Candidate 3 (NOT SELECTED): "When Search Hacks the Cache: Mapping Dimensional Escape under Reversible KV-MCTS"

**Pros:**
- "Hacks the Cache" is memorable and accurate to the Goodhart mechanism (the search "hacks" the entropy objective via cache-state engineering rather than reasoning).
- Active phrasing.

**Cons:**
- "Hacks" is colloquial and reads as informal for EMNLP main-program style. Reviewers may flag this as overclaiming or unprofessional.
- "Search Hacks the Cache" loses Track F's specific mechanism (Goodhart's Law manifesting via prior-amplification or greedy-collapse, depending on scale).
- Length: 13 words / 84 characters. Comparable to Candidate 1.

**Reviewer-routing prediction:** memorable but risks dismissal as clickbait. Lower selected-for-best-paper probability; same Findings probability.

## Final title for v9 application

```latex
\title{Reversible KV-Cache MCTS: A Memory-Bounded Probe of Dimensional Escape in Frozen Language Models}
```

## Notes for Session B

**TeX target:** `\title{...}` at canonical TeX:39.

**REVISION marker:** `[REVISION | title-recast-D+ | 2026-05-10 | Title-lock for Option D+ pivot (Session A v9 decision, applied 2026-05-11). Replaces Option α+ working title "KV-Cache Inception: Reversible Monte Carlo Tree Search in Latent Space for Detecting and Stress-Testing Alignment Faking in Large Language Models." Two-line alignment-faking-detection framing is fully retired. New title leads with the algorithm + memory bound (Contribution 1 + 2), pivots to unified empirical finding (Dimensional Escape, Contribution 3). "Frozen Language Models" precludes confusion with gradient-trained adversarial work. "Memory-Bounded Probe" anchors Track D's measured constant. Final decision among three candidates; Candidate 2 (mechanism-first) and Candidate 3 (colloquial "Hacks the Cache") considered and rejected for the reasons in `docs/logs/2026-05-10_session-A-prose-drafts-v9-title.md`. — pending audit]`

**Bibliography-side implication:** the bib entries citing this paper internally (none expected; reverse self-citation only) need no change. But if Track G Cartography reports ever reference the paper by title, Session A should ensure they use the new title.

**Cross-referenced terminology:**
- "Reversible KV-Cache MCTS" — used in abstract, §1 Contribution 1, §4.2 method name
- "Dimensional Escape" — used in abstract, §1 Contribution 3, §3.2 Definition, §7 Discussion
- "Memory-Bounded" — used in abstract, §1 Contribution 1, §6 Memory Complexity
- "Frozen Language Models" — used in abstract, §1, §3.1

All four phrases-in-title resolve to existing or v9-drafted content. No new cross-reference work needed.

**Page-budget impact:** title length comparable to Option α+ working title (~30 char shorter, in fact); no impact on first-page real estate.

## Phase 3 (camera-ready) note

This title-lock is for the ARR-submitted version. At camera-ready (Phase 3, 2026-09-20 deadline if accepted), Session A may wish to revisit the title if reviewer feedback suggests refinement. The decision matrix above is preserved here as the audit trail.

---

*Session A v9 title-lock complete. Apply Day 2 of ARR-cycle execution plan as part of the coordinated abstract → contributions → §3 → §7.3 → title quintet.*
