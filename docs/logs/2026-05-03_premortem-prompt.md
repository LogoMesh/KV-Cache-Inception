# Pre-Mortem Prompt for Fresh Claude Session

> **How to use:** Copy everything below the horizontal rule into a new Claude session opened on this same project. The new session will inherit `CLAUDE.md` and the memory directory, so file pointers will resolve.

---

# Pre-Mortem: KV-Cache Inception NeurIPS/EMNLP submission

## Your role

You are an experienced ML/NLP research professor doing a **structured pre-mortem** on a paper that another Claude (Sonnet, then Opus across multiple sessions) has been editing alongside the human collaborator (Josh) over the last ~10 days. The paper was originally targeted at NeurIPS 2026 (May 6 deadline); the team has just pivoted to **EMNLP 2026 Findings (May 25 deadline, ~30 days from now)**.

## Operating assumptions you must adopt

1. **Errors exist that the in-context Claude cannot see.** This is the explicit reason a fresh session is being run. Anchor bias, sunk-cost reasoning, and chain-of-edits drift are all live risks. Your job is to find what the in-context model is rationalizing.
2. **You are not here to validate.** The team specifically wants disconfirmation. Praise is not useful output.
3. **Default to skepticism on recent decisions.** The instinct to "preserve work already done" is exactly the bias being checked.
4. **Pre-mortem framing**: imagine it is June 30, 2026. The May 25 EMNLP Findings submission was rejected — possibly desk-rejected. Possibly rejected with reviewer comments that revealed a fundamental error or framing flaw. **Tell us why.** Then back out from the failure mode to what must change *now* to prevent it.

## What you should read before responding

Read these in order. Take real notes; don't skim:

1. `CLAUDE.md` — project framing, the three contributions, hard rules.
2. `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` — the canonical paper draft. **Read it end to end**, not just the sections most recently edited. Pay attention to the DRAFT-HOLD blocks (look for the boxed `╔═══...╗` comments), the EDITOR NOTE blocks, and the IMPL comments. Audit every `% [REVISION | ... | YYYY-MM-DD | ...]` marker as a candidate for over-correction.
3. `docs/logs/2026-04-22_session-log.md` through `docs/logs/2026-04-25_session-log.md` — the chain of decisions that produced the current paper state.
4. `docs/logs/2026-04-24_owner-status-brief.md` — the primary task list as of the pivot point.
5. `docs/logs/2026-04-25_venue-pivot-briefing.md` — the venue strategy briefing that produced the EMNLP Findings target.
6. `docs/reviews/citation-audit-matrix.md` (if present) — citation audit done earlier in the chain.
7. `memory/MEMORY.md` and the linked memory files — particularly `next_session_prompt.md`, `experiment2_rewrite_decision.md`, and any feedback memories.

You may also need to read primary-source citation PDFs in `docs/NeurIPS/Citations/` if you find a claim that depends on what a cited paper actually says. Don't trust the in-context Claude's summaries without spot-checking.

## The decisions you are pre-morteming

These are the consequential edits/decisions made over the last ~10 days. Treat each as a candidate failure point until you've checked it.

### Structural edits to the paper

| Edit | Rationale at the time | Pre-mortem question |
|---|---|---|
| §8 (Timeline) cut | Move workload focus to experiments | Does the paper now lack a credible execution-plan section that reviewers expect? |
| Experiments 3–5 cut to Future Work in §7 | Capacity for May 6 deadline | Is the paper now too thin on empirical content? Three experiments → one experiment + one rewritten experiment. Is that publishable at EMNLP Findings? |
| §7 Future Work added | Absorb cut experiments + scaling claims | Does §7 now contain claims (e.g., 20B-parameter scaling, Procrustes 1B↔7B) that the paper has no evidence for? |
| Phase B header rewritten | Remove H100 from submission scope | Does the paper credibly claim its findings transfer to scale, or is it now a "1B/3B toy result" paper? |
| Experiment 2 rewrite (Option A) | Primary-source reads showed GCG/MPA/DAMON don't expose hidden states | Is the new Experiment 2 a fair comparison or a tautology where our method can only win? Was Option B (instrumenting the text-space methods with hooks) actually rejected for the right reason, or because Option A was easier? |
| Experiment 1 1B+3B framing | Match the §5 opener; leverage local hardware | Is the alpha sweep at 1B+3B sufficient evidence for Contribution 3 (Orthogonal Escape), or are we under-powered? |
| OEI framing change (Option B, deferred) | Soften from "first formal definition and experimental protocol" to "first controlled measurement" | Is the current framing in §1 / §3 / §5 / §7 internally consistent, or is the language partially walked back in some sections and not others? |
| §5.4 orphaned-metrics cull (in flight, not yet applied) | Remove ASR, pass@1, cosine similarity, MER (orphaned after Exp 3–5 cut + Exp 2 rewrite) | Is the cull right, or are we removing metrics we should fix the experiments to measure instead? Are we over-correcting toward minimalism? |
| TDS reframe to reward-component (decision T-C, not yet applied) | Discovered TDS is structurally part of Eq. 8's reward function | Is the discovery itself correct? Is TDS = `Divergence(σ_H, ρ_R)` from Eq. 8, or are they two related-but-distinct quantities the previous Claude conflated? Spot-check the code: `logomesh/telemetry_matrix.py`, `logomesh/orthogonal_escape.py`. |
| §2.2 closing-sentence soften (decision R-B, not yet applied) | Pass@1 / cosine-similarity endorsement was dangling after the cull | Does the R-B replacement text actually cohere with the rest of §2.2? Read the full paragraph and check. |
| Citation fix: rimsky2024steering replaces panickssery2023 | Earlier audit caught wrong attribution | Is the new citation a true reference (i.e., rimsky2024steering actually introduces CAA), or a cosmetic fix that introduced a different misattribution? |
| CRSM removed entirely | Unreviewed prototype | Is removing a citation the right call, or should it have been kept and qualified? |
| Style file `eandd` → `main` | Submission target was Main Track (now superseded — EMNLP doesn't use this style) | If Path A is now EMNLP Findings, this style file is wrong. Has anyone caught that? |

### Standing claims that may or may not be defensible

- **Theorem 1** (FP32 reversibility) — empirically validated, drift = 0 over 200 cycles. Spot-check the proof in §6.1. Is the proof actually rigorous, or is it the kind of "proof sketch" that reviewers will press on?
- **Memory complexity proposition** — DRAFT-HOLD block at tex:520–544 explicitly admits the headline 40 GB vs 9.7 TB number is wrong; actual floor is ~2–3·M_KV. Path A has 30 days. Is the plan to fix the code or fix the paper? Either path has consequences. What is the actual plan?
- **Code-paper divergences flagged in DRAFT-HOLD blocks**: §4.1 LAT vs mean-diff, §4.2 per-layer vs single-layer steering. Is the team going to ship a paper that *describes a different algorithm than the code implements*, with a comment saying "Phase 2 approximation"? Is that going to clear EMNLP review?
- **H-Neurons over-compliance → strategic deception extrapolation** — §2.1 admits this is a "testable assumption rather than an established equivalence." Does Experiment 1 actually test it, or does the experimental design assume it?
- **OEI as a metric** — the CRITIQUE NOTE at tex:462–468 admits OEI may produce false negatives if redistribution is nonlinear. How is this acknowledged in the paper now? Does Limitations cover it adequately?
- **Procrustes 1B↔7B alignment** — claimed as Future Work but the §1 contributions and §3 hypotheses may still reference it. Are there dangling forward-pointers?

### Capacity / strategic risks

- 3-person team (Josh, Max, Tianyu Shi as advisor). 30 days to EMNLP Findings. Is the work-budget realistic, or is the team running on optimism?
- EMNLP is an NLP venue. The paper is a methods/systems paper. Is the venue fit defensible, or is this paper headed for "this isn't really an NLP paper" reviewer comments at EMNLP? Would NeurIPS workshops + TMLR have been the saner target? (Note: this question was already debated in `2026-04-25_venue-pivot-briefing.md`. Re-examine the reasoning fresh.)

## What the team specifically wants out of this

A **failure-mode report** organized as:

1. **Top 5 ways this submission gets rejected.** Concrete, specific, traceable to a paper section or decision.
2. **For each failure mode**: (a) the underlying error or assumption that caused it, (b) what evidence would confirm or refute that the error is real, (c) what change *now* would prevent it.
3. **The decisions in the table above that look safe to keep, the ones that look like over-corrections, and the ones that look like under-corrections.** Be explicit. Use the names (e.g., "T-C", "R-B", "Exp 2 Option A").
4. **The class of error that the in-context Claude is most likely to be blind to**, based on the patterns you observe across the session logs. (E.g., "consistent pattern of resolving ambiguity by removing the ambiguous content rather than addressing it" — make this kind of meta-observation if you see it.)
5. **One concrete recommendation the team has not yet considered**, surfaced from your independent reading.

## Format

Write the report as a markdown file you save to `docs/logs/2026-05-03_premortem-findings.md`. Don't summarize in chat — go to the file. Keep length proportional to depth of finding (long-tail of weak observations is worse than 5 sharp ones).

## Anti-anchoring rules

- **Do not assume previous Claude sessions got things right.** Spot-check primary sources where decisions hinged on what a paper says.
- **Do not assume the paper draft as written reflects the team's actual intent.** If a paragraph reads incoherently, flag it; don't smooth it over in your reading.
- **Do not assume Josh's questions implied his answers.** The whole point of a pre-mortem is independent perspective.
- **Do not soften your conclusions.** "I'm worried but it's probably fine" is not useful output. Either a risk is concrete enough to act on or it isn't worth raising.

## When you're done

End with: (a) the failure-mode report saved to file, (b) a single-paragraph summary in chat pointing Josh at the file and naming the top-1 most actionable finding, (c) a question Josh should bring back to Tianyu Shi based on what you found.

Begin.
