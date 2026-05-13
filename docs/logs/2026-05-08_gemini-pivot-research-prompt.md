# Discovery Prompt: EMNLP-Fit Pivot for Reversible KV-Cache MCTS Infrastructure

## What this prompt is

This is **not** another pre-mortem. Two prior pre-mortems (2026-05-07 and 2026-05-08) have already characterized what's wrong with the current EMNLP submission — they reproduced each other's findings with high consistency. We accept that diagnosis. The current framing yields <5% acceptance probability, and post-mitigation estimates land at 20–25% (still below the 17% Findings track baseline by margin-of-uncertainty).

This is a **synthesis prompt**. The team has eighteen days, a working codebase, a constrained empirical envelope (consumer hardware, 1B/3B Llama models), and is choosing to make one substantive pivot of the paper's framing rather than retreat to a workshop venue. We want you to help discover the strongest possible EMNLP-fit framing — and to be honest about whether such a framing exists.

The team's gut says "do the hallucination-mitigation pivot from round 2 of the pre-mortem." The team's intellect says "before we commit to that, let's see if a deep research session can surface a stronger or more novel framing — something with a future-research story attached." Your job is to satisfy both — confirm the gut answer if it's truly the best, or surface something better if one exists.

## Your role

You are simultaneously two reviewers in one head:

1. **An EMNLP/ACL program-committee veteran** with detailed working knowledge of EMNLP 2024–2026 acceptance trends, the editorial taste of senior area chairs in interpretability/representation/safety, and the unstated norms that distinguish a paper that lands at EMNLP from one that lands at NeurIPS.
2. **A mechanistic-interpretability research practitioner** who has read the activation-steering / probing / causal-mediation / sparse-autoencoder / circuit-discovery literature deeply, and can identify under-served corners — questions the field cares about but where existing infrastructure is inadequate.

The intersection of these two viewpoints is where the right framing lives.

## Operating assumptions you must adopt

1. **The hallucination-mitigation pivot (round 2 §7 of the pre-mortem at `docs/logs/2026-05-08_gemini-pre-mortem-EMNLP-findings.md`) is one candidate among others. Do not rubber-stamp it.** Evaluate it against alternatives. If it survives the comparison, fine — say so explicitly. If something better exists, say what.
2. **Future-research story is part of the assessment, but committing to a multi-paper program is not yet a team decision.** The team's academic advisor is Prof. Tianyu Shi (McGill); any research-program-scale commitment requires his sign-off on feasibility. The team has not consulted him on this yet and is not willing to commit to a 2–3-paper research program before that consultation. Therefore: **provide BOTH a research-program-seed framing AND a one-shot-safe-fallback framing.** The first is the upside case if Prof. Shi signs off; the second is the insurance case where the paper has to stand entirely on its own as a complete contribution. The two framings may be the same paper read two ways, or they may be genuinely different papers; you decide based on the analysis.
3. **The infrastructure is the asset, not the constraint.** The team has spent months building this. The question is what it's actually good for, not how to retrofit it to popular EMNLP topics.
4. **You are not here to be polite.** If the honest answer is "no framing in 18 days will get this above 30% acceptance, recommend workshop pivot," say that clearly.
5. **Errors exist that the in-context Claude could not see.** Two parallel Claude sessions plus two prior Gemini pre-mortems have anchored toward specific failure modes; you may discover that the actual bottleneck is something none of those four sessions surfaced.

## What you should read before responding

In order:

1. **The current TeX file**: `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2-premortem-v2.tex` (note: this is the post-triage variant with v7 + v8 edits applied; not the canonical, but reflects the team's current best state of the paper).

2. **The two prior Gemini pre-mortems**:
   - `docs/logs/2026-05-07_gemini-pre-mortem-EMNLP-findings.md` (round 1)
   - `docs/logs/2026-05-08_gemini-pre-mortem-EMNLP-findings.md` (round 2)
   These tell you the failure modes that have already been identified. **Do not re-derive them.** Treat them as the validated diagnosis.

3. **The recast history**:
   - `docs/logs/2026-05-05_recast-action-plan.md` §0 (the May 5 recast decision) and §2 (six locked decisions, 5/6 with rationale)
   - `docs/logs/2026-05-03_premortem-findings.md` (the original, pre-Gemini Claude pre-mortem that the recast was based on)

4. **The codebase outline**:
   - `CLAUDE.md` (top-level project guide; describes the modules, validated capabilities, and hard rules)
   - `README.md` (post-2026-05-07 rewrite reflecting the recast state)

5. **EMNLP 2026 acceptance trends — fresh research required**. Use web search to characterize what is being accepted at EMNLP 2025, what's hot at ACL 2025/2026 in interpretability/steering/probing, what's expected at EMNLP 2026 main and Findings tracks, and what areas are under-served (i.e., where a methodologically sound contribution would land cleanly because few competitors are working there). Pay particular attention to:
   - Mechanistic interpretability papers from EMNLP 2024–2025 (sparse autoencoders, circuit analysis, attribution)
   - Activation steering / representation engineering follow-up work post-Zou et al.\ 2023
   - Causal mediation / interchange interventions / counterfactual probing
   - Multi-turn dialogue interpretability (KV-cache as conversation context — note the topical adjacency)
   - In-context learning mechanism papers
   - Behavioral phenomena that 1B–3B parameter models *do* exhibit (hallucination, sycophancy, refusal-of-benign-tasks, format inconsistency, instruction-following violations, copying-vs-reasoning trade-offs, etc.)

## The asset, in concrete terms

- **Reversible Monte Carlo Tree Search in KV-cache latent space**, validated end-to-end. Theorem 1 (FP32 accumulator → zero-drift reversibility) holds at the 1B and 3B scales (residual_norm = 0 over 200 cycles).
- **Memory complexity** $O(M_{KV} + d \cdot K_{\text{acc}})$, independent of branching factor — orders of magnitude cheaper than parallel-cache MCTS.
- **Per-layer telemetry matrix** $\mathbf{T}_t \in \mathbb{R}^{2 \times L}$ fusing two channels:
  - Bottom-up: $\sigma_H$, the H-Neuron activation stress (Gao et al.\ 2025; H-Neurons are causally linked to hallucination/over-compliance). **Just verified post-merge** to vary monotonically with steering magnitude (0.228→0.330 over $\alpha \in [0.1, 5.0]$, 14–15 of 17 layers firing).
  - Top-down: $\rho_R$, RepE honesty-direction projection. Calibrated; varies substantially with steering ($-1.31 \to -0.53$ over the same sweep).
- **Working models**: Llama-3.2-1B-Instruct (22 layers, hidden dim 2048), Llama-3.2-3B-Instruct (28 layers).
- **Hardware floor**: RTX 3060 12GB. **8B is not accessible** before May 25; H100 rentals are not viable for solving the venue mismatch (analyzed and rejected).
- **What the algorithm uniquely enables** that no prior published method does, jointly: deep tree-search over continuous-space latent interventions on a frozen model with bounded memory and exact rollback, allowing systematic exploration of a perturbation manifold without gradients, without finetuning, and without the $O(b^d)$ memory blowup of parallel-cache approaches.

## The constraint

- **18 days** to EMNLP Findings 2026 deadline (2026-05-25, ARR submission portal — please verify the portal in your search).
- **8-page Long Paper limit**, excluding references + Limitations + Ethical Considerations.
- **One small team**: principal investigator + ~3 collaborators, ~half-time on the paper.
- **No frontier-scale evaluation** is possible. Whatever framing emerges must be empirically demonstrable at 1B–3B scale.

## What we want from you

A **synthesis report** organized as follows:

### Part 1: Characterize the EMNLP 2026 zeitgeist

Before recommending any framing, give a 1-page characterization of what EMNLP 2026 is actually buying. Specifically:
- Three areas of mechanistic-interpretability / activation-steering work that EMNLP 2025 accepted heavily — give specific paper titles + venues.
- Three areas where EMNLP is *under-served* (papers in the area are accepted but the area is small enough that a strong methodological contribution would stand out).
- One area that you think is *over-served* and should be avoided regardless of how well our infrastructure could speak to it.
- The unstated norms that an EMNLP reviewer applies to a methods/infrastructure paper that an ICLR or NeurIPS reviewer would not.

### Part 2: Generate 5–7 candidate framings

For each candidate framing, provide:

- **Title and one-sentence pitch**.
- **Core claim** the paper would make.
- **Profile tag** — one of: `RESEARCH-PROGRAM-SEED` (paper's value depends meaningfully on opening 2–3 follow-on papers and would feel incomplete as a one-off), `ONE-SHOT-SAFE` (paper stands on its own as a complete contribution; future work is bonus, not foundation), or `BOTH-VIABLE` (the same paper reads cleanly either way). This tag matters because the team will not commit to a research-program scale of work without their advisor (Prof. Tianyu Shi, McGill) signing off on feasibility, so candidates need to be sortable by which path they support.
- **The recognized EMNLP/ACL research thread it connects to** (cite 2–3 specific accepted papers as anchors).
- **The infrastructure use** — explicitly which components from §The Asset are required, and what would change about how they're used.
- **The experimental design** — what benchmark(s) at 1B/3B scale would validate the claim, and why this scale is sufficient (or what's lost at this scale).
- **What's genuinely novel** — what does this framing enable that wasn't possible before? Is the novelty methodological, empirical, or both?
- **The forward story** — for `RESEARCH-PROGRAM-SEED` candidates: what does this open up? (One paragraph.) For `ONE-SHOT-SAFE` candidates: why does this paper stand cleanly on its own? (One paragraph.) For `BOTH-VIABLE` candidates: address both readings.
- **18-day execution feasibility** — concrete: what's already done, what needs new code, what needs new runs, what's the riskiest step.
- **Estimated EMNLP Findings acceptance probability** — show your reasoning, anchored against the 17% baseline and against accepted-paper precedents.
- **Biggest risk to the framing** — what kills it under adversarial review?

Cover the framing space *broadly*. Include conservative candidates (hallucination mitigation, sycophancy, instruction-following diagnostics) AND adventurous candidates (multi-turn dialogue dynamics, in-context learning mechanism, counterfactual probing at depth, monitor-design red-teaming, ICL via KV-cache search, anything else you can find). At least 5; at most 7. **Aim for at least 2 candidates per profile tag** so the team has real choices in both columns.

### Part 3: Rank-order the candidates — two tables

Provide **two** ranked tables, since the team needs to be able to pick under two different scenarios:

**Table 3A — Research-program path** (assumes Prof. Tianyu Shi signs off on a multi-paper program). Rank candidates with profile tag `RESEARCH-PROGRAM-SEED` or `BOTH-VIABLE`. Score components: EMNLP acceptance probability + execution feasibility + research-program yield + true novelty. The strongest research-program seed should be unambiguous from this table.

**Table 3B — One-shot fallback path** (assumes the paper has to stand entirely on its own; no future-research commitment). Rank candidates with profile tag `ONE-SHOT-SAFE` or `BOTH-VIABLE`. Score components: EMNLP acceptance probability + execution feasibility + standalone completeness + true novelty. The strongest one-shot pick should be unambiguous from this table.

If the same candidate tops both tables (a `BOTH-VIABLE` candidate that's strongest under both scoring schemes), say so plainly — that simplifies the team's decision considerably. If different candidates top the two tables, the team will have a real choice to make. **No hedging in either table.**

### Part 4: Detailed proposals for the strongest candidates

Draft full proposals for **the top-ranked candidate from each of Tables 3A and 3B**. If the same candidate tops both tables, draft one full proposal and a short note on whether the framing emphasis would shift between the two readings (program-seed framing vs one-shot-completeness framing); both versions of the abstract may differ even if the experimental design is identical.

For each top-ranked candidate, draft:
- A new title.
- A new abstract (~150–200 words).
- A new §1.1 Contributions list (3 contributions, prose form).
- A revised §3 Problem Formulation framing (just the framing, not the full subsection).
- The specific Experiment 1 design — including benchmark, prompts, measurement protocol, expected result, and what a positive result looks like.
- A list of citations that would need to enter the bibliography (from the EMNLP 2024–2025 vintage).

### Part 5: Comparison against the round-2 hallucination pivot

Compare your #1 recommended framing directly against the hallucination-mitigation pivot from round 2 §7 of the prior pre-mortem. Be specific about:
- Where they agree.
- Where they differ in EMNLP fit.
- Where they differ in 18-day feasibility.
- Where they differ in future-research yield.
- Whether the round-2 pivot is your #1, or whether something better exists. Justify.

### Part 6: Two forward stories — research program AND standalone completeness

Write both, separately:

**Part 6A — The research-program story** (conditional on Prof. Tianyu Shi signing off). If the team executes the Table 3A #1 framing successfully and lands at EMNLP, what's the natural next paper? And the one after that? Sketch a 2–3-paper research program that this submission seeds. Be honest about the difficulty of each follow-on: which require infrastructure the team already has, which require new infrastructure, which require frontier-scale compute, which require new collaborators or domain expertise. The team will use this sketch as one input when consulting Prof. Shi on whether the program is realistic.

**Part 6B — The standalone-completeness story** (no future-research commitment assumed). If the team executes the Table 3B #1 framing successfully and lands at EMNLP, articulate why this paper is a complete scientific contribution that doesn't require any follow-up to be valuable. What does it definitively close? What question does it answer in a way that future work could *cite*, not *complete*? A reviewer reading the abstract should understand that the paper is the artifact, not a teaser.

The team's strategic question is twofold: (a) "if Prof. Shi signs off, is there a research program here?" and (b) "if he doesn't, do we still have a publishable paper that doesn't depend on that?" Help them see both answers.

### Part 7: What the four prior pre-mortem sessions are most likely missing

Two Claude pre-mortems and two Gemini pre-mortems have produced overlapping but not identical findings. There's a class of error none of them have surfaced — the "what is the deepest unique capability of this infrastructure that NLP cares about" question has not been asked directly by any of them; they have all been critiquing or salvaging an existing framing rather than discovering a new one. Identify the class of error that this pattern implies.

### Part 8: One concrete recommendation the team has not yet considered

Surface one recommendation that is neither hallucination mitigation nor any of the framings tried in the recast nor any of the failure-mode mitigations from the prior pre-mortems. This is your wildcard. It can be conservative (a small reframe that has high yield) or adventurous (a structural reconceptualization). Justify briefly.

## Anti-anchoring rules

- **Do not assume the round-2 hallucination-mitigation pivot is correct.** It might be, but assume it isn't until your own analysis confirms.
- **Do not assume any of the locked Option α+ decisions are correct.** Question them all. The team has already conceded that locking decisions early may have produced anchor bias.
- **Do not invent strengths.** If the infrastructure has a fundamental weakness for EMNLP fit (e.g., the algorithm is genuinely a systems contribution that doesn't have a natural NLP-empirical home), say so plainly. The team will respect that answer.
- **Do not avoid the future-research question.** It's a hard question and the team specifically values your judgment on it. "Is this the seed of something, or a dead end?" is a legitimate question to give a confident answer to.
- **Foreground what an EMNLP reviewer specifically wants** that an ICLR or NeurIPS reviewer doesn't. The team has been told this in the abstract but needs concrete examples (specific accepted papers, specific reviewer concerns, specific deliverable formats).
- **If the honest answer is "the strongest framing still doesn't get this above 30%, recommend workshop pivot," say that.** Confidence in a low-probability assessment is more useful than false hope.

## Output format

Markdown. No length cap, but Part 4 should be the longest section. Use tables where they help. Cite specific papers (with title + venue + year). Surface URLs where they help the team validate.

## Suggested filename for your output

`docs/logs/2026-05-08_gemini-pivot-research-findings.md`

(So it parallels the two prior pre-mortem-findings filenames and pairs in `ls`.)

---

Thanks for going deep on this. The team is making a real strategic decision based on your output, and they've earned the right to a substantive answer. Take the time you need.
