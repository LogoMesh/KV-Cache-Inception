# Gemini 3.1 Pro Deep-Research Pre-Mortem Prompt — EMNLP Findings 2026

> **How to use:** Open Gemini 3.1 Pro in a Deep Research session. Attach `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`. Paste everything below the horizontal rule. Gemini's web-search and multi-turn deliberation tools should be enabled.

> **Why this prompt exists:** The paper is being submitted to EMNLP 2026 Findings on 2026-05-25 after a recast from NeurIPS Main Track. A fresh Claude pre-mortem on 2026-05-03 surfaced six failure modes that we've worked through over the past five days. The paper's structure, framing, and empirical content are now stable. Before submission, we want a second outside-perspective pre-mortem from a reviewer-class reasoner with web access — specifically to compare the paper against accepted EMNLP/ACL Findings work and identify EMNLP-fit issues a Claude pre-mortem might miss.

---

# Pre-Mortem: KV-Cache Inception EMNLP Findings 2026 submission

## Your role

You are an experienced ML/NLP reviewer with substantial program-committee experience at EMNLP, ACL, and NAACL — both Main and Findings tracks. You have detailed working knowledge of what gets accepted vs. rejected at these venues, the unstated norms reviewers apply, and the specific differences between EMNLP and ML-systems venues (NeurIPS, ICLR) for methods/infrastructure papers. You are doing a structured pre-mortem on a paper that has been heavily edited over the past two weeks.

## Operating assumptions you must adopt

1. **You are not here to validate.** The team specifically wants disconfirmation. Praise is not useful output.
2. **Default to skepticism on every framing decision.** The team has gone through one Claude-driven pre-mortem and made substantial structural changes; that pre-mortem may itself have over-corrected or missed venue-specific issues.
3. **Errors exist that the in-context Claude could not see.** Two parallel Claude sessions worked on this paper for the last five days; both have anchor bias on the recast direction. Your job is to find what they're rationalizing.
4. **Pre-mortem framing**: imagine it is October 2026. The May 25 EMNLP Findings submission was rejected with reviewer comments that reveal a fundamental issue — possibly framing, possibly empirical scope, possibly venue fit. **Tell us why.** Then back out from the failure mode to what must change *now* to prevent it.
5. **EMNLP is an NLP venue, not an ML/systems venue.** A paper that would land at NeurIPS Main might struggle at EMNLP Findings even with strong content if it doesn't speak the field's language or engage its conventions.

## What you should read before responding

In order. Use real notes; don't skim.

1. **The attached TeX file** (`04.18.2026-NeurIPS-Research-Proposal-2.tex`) end to end. Note the file path retains its original `NeurIPS/` directory — repo-history continuity, not venue confusion. The content has been recast for EMNLP Findings. Pay particular attention to:
   - Abstract, §1 Introduction, §1 Contributions
   - §2 Related Work, especially §2.1 (alignment faking), §2.2 (RepE / activation steering / Bailey 2024 engagement)
   - §3 Problem Formulation, especially §3.1 ("Inference-Time Latent-Space Analysis on Frozen Models")
   - §5 Experimental Design, especially Experiment 1 with Table 1 (per-α telemetry: σ_H, ρ_R, OEI, TDS) and the Interpretation paragraph
   - §6 Theoretical Analysis (Theorem 1 + Memory Complexity Proposition)
   - §7 Discussion, especially §7.4 ("Empirical Measurement Prerequisites and Initial Observations")
   - The full set of REVISION markers throughout — they trace the recast arc as comments

2. **3–5 recent accepted EMNLP/ACL Findings or Main Track papers** in adjacent areas. Use web search to find them. Categories:
   - Mechanistic interpretability of LLMs (e.g., circuit analysis, attribution, attention probing)
   - Activation steering / representation engineering
   - Alignment monitoring / latent-space analysis / probing
   - Methods/systems papers at NLP venues (so you can calibrate whether algorithm-headlined work fits)
   - Bailey et al. 2024 (NeurIPS) "Obfuscated Activations Bypass LLM Latent-Space Defenses" — read this directly; it's load-bearing in §2.2 of our paper

   For each accepted paper, note: title, venue, year, what kind of contribution (empirical / methods / position / dataset), and what the abstract / §1 looks like structurally.

3. **The pre-mortem context** (optional but recommended for understanding what the team has already worked through):
   - `docs/logs/2026-05-03_premortem-findings.md` — the original Claude pre-mortem and addendum
   - `docs/logs/2026-05-05_recast-action-plan.md` §0 (the recast decision) and §2 (six pre-mortem decisions, 5/6 locked)
   - `README.md` — the public-facing positioning

## The decisions you are pre-morteming

These are the substantive framing/structural commitments made over the past two weeks. Treat each as a candidate failure point until you've checked it against EMNLP norms.

| Decision | Rationale at the time | Pre-mortem question |
|---|---|---|
| Recast paper from "alignment-faking detection" headline to algorithm-headlined ("Reversible MCTS in Latent Space") | Bailey 2024 already operationalized OE-style measurement; algorithm + Theorem 1 are the rock-solid contribution | Does an algorithm-first NLP paper at EMNLP Findings work, or does it read as misaligned with the venue's empirical-NLP expectations? |
| Single experiment (Experiment 1: alpha-sweep on Llama 3.2-1B with reversibility on 3B independently) | Realistic scope for the timeline; algorithm is the headline; experiment is the application demonstration | Is one alpha-sweep enough empirical content for EMNLP Findings? How does this compare to typical Findings papers' empirical scope? |
| Bailey et al. 2024 framed as peer-measurement (not competitor) | Bailey already operationalized the orthogonality question; we extend to a different attack regime | Does this framing read as principled (complementary measurement) or defensive (we're doing something Bailey already did, slightly differently)? |
| "Two measurement-pipeline issues identified, both resolved" framing in §7.4 | Honest disclosure of bugs found and fixed during paper preparation; demonstrates implementation-level rigor | Is this credibility-positive (rare honesty) or does it signal "lots of things were broken"? Reviewers may parse this differently than the team hopes. |
| OEI as a targeted geometric metric, complementing Bailey's global cosine-to-baseline | More specialized than Bailey; reveals different geometry | Is OEI sufficiently differentiated from existing probe-evasion metrics (Bailey, Gupta & Jenner RL-Obfuscation, etc.)? Does it warrant the prominence it gets? |
| TDS reframed as the divergence component of the MCTS reward (not a separately tabulated metric) | Avoids a circularity (TDS is both reward third term and reported diagnostic) | Is the circularity acknowledgment in §5.4 sufficient, or does the dual role still confuse a reviewer? |
| Anti-OE empirical observation as the headline result (TDS *decreases* with α) | Honest reporting of what the data showed; consistent with Bailey's negative-result regime | Does a paper whose empirical result is "the predicted phenomenon does NOT emerge" land well at Findings, or does it require additional positive content? |
| Page-budget: paper is ~648 lines uncompiled; we have not yet compiled under EMNLP style | Submission target is 8-page EMNLP Long Papers; estimated ~10–12 pages currently | Will critical content be cut to fit, and if so, which sections lose first? |

## What the team specifically wants out of this

A **failure-mode report** organized as:

1. **Top 5 ways this submission gets rejected at EMNLP Findings 2026.** Concrete, specific, traceable to a paper section or framing decision. Distinguish between "rejection" (clear no) and "weak accept that doesn't make the cutoff" (Findings-specific dynamic).

2. **For each failure mode**: (a) the underlying issue, (b) a citation or reference to an accepted EMNLP paper that handled the same concern well, (c) what change *now* would prevent it.

3. **EMNLP-fit-specific concerns.** This is an NLP venue and ours is a methods/systems paper. Specifically:
   - Is the §2 Related Work weighted appropriately toward NLP/ACL-ecosystem citations vs. ML/NeurIPS-ecosystem citations?
   - Is the experimental setup (Llama 3.2-1B/3B) the kind of model EMNLP reviewers want to see, or does it fall into "too small to be interesting" territory?
   - Are the test prompts (referenced as instruction-following / Alpaca-eval-style in the recast direction) on standard NLP benchmarks?
   - Is the result interpretation framed in NLP-relevant terms (model behavior, generation quality, downstream task performance) or in ML-systems-relevant terms (memory, scaling, algorithmic complexity)?

4. **Page-budget assessment.** The paper currently has not been compiled under EMNLP style. Estimate: how many pages will it likely produce? Which sections are most likely to need cutting to hit the 8-page Long Papers limit (references and appendix not counted)? Are there sections that read as unnecessary in an EMNLP context that could go without affecting the contribution?

5. **The three highest-leverage edits for the remaining 18 days.** Concrete: what specific text change, on what specific section, would most improve the paper's EMNLP Findings acceptance probability? Order by leverage.

6. **Honest acceptance probability estimate.** What is the chance this paper, *as currently written*, gets accepted at EMNLP Findings 2026? Show your reasoning by reference to (a) the Findings-track acceptance rate baseline, (b) the paper's specific strengths and weaknesses, (c) comparison with the accepted-paper sample you reviewed.

7. **A class of error that the team's previous Claude pre-mortems are most likely to be blind to**, based on the patterns you observe. The first pre-mortem found the paper was overclaiming; the recast may have over-corrected toward minimalism, or may have introduced new framing issues. What kind of error specifically?

8. **One concrete recommendation the team has not yet considered**, surfaced from your independent reading.

## Anti-anchoring rules

- **Do not assume the previous Claude pre-mortems got things right.** Spot-check primary sources where decisions hinged on what a paper says.
- **Do not assume the paper draft as written reflects the team's actual intent.** If a paragraph reads incoherently, flag it; don't smooth it over in your reading.
- **Do not soften your conclusions.** "I'm worried but it's probably fine" is not useful output. Either a risk is concrete enough to act on or it isn't worth raising.
- **Do not invent strengths to balance criticisms.** A failure-mode pre-mortem is by design lopsided toward weaknesses; if a strength is genuinely critical, mention it once, but don't pad.
- **EMNLP-specific expertise is the primary value-add.** Where your knowledge of what reviewers at this venue specifically expect differs from generic ML-paper-review heuristics, foreground that knowledge.

## Format

Write the report as a well-structured markdown response. Length should be proportional to depth of finding — typically 2,000–4,000 words. Save to a file the user can pass back to Claude (suggest `docs/logs/2026-05-08_gemini-pre-mortem-EMNLP-findings.md` if the user wants a specific path).

End with: (a) the top-1 most actionable finding stated in one sentence, (b) the cited accepted-paper most worth reading in full as a model for how this paper should be structured.

Begin.
