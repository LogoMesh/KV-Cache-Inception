# Next-Paper Novelty Check — Pre-Training Reward Audit

> **Why this exists:** before committing to a follow-on paper direction, check whether the framing has already been published. This file contains (a) the premise of the proposed next paper, written precisely enough that a literature search can confirm or refute novelty, and (b) a deep-research prompt for Gemini 3.1 Pro (or equivalent) to scan the relevant literature.

---

## Premise of the proposed next paper

**Working title (placeholder):** *Pre-Training Reward Audit via Inference-Time Controlled Negative Controls: Detecting Goodhart Vulnerabilities Before RLHF*

**Core claim.** Before training a language model with a proxy reward (RLHF, DPO, RLAIF, constitutional AI, RLVR, reward-model-driven fine-tuning), the proxy can be audited at inference time on a frozen base model. The audit produces a Goodhart-vulnerability score that predicts whether subsequent RL training will exploit the proxy's structure (prior amplification, length hacking, format collapse, first-letter bias, etc.) rather than driving toward the intended downstream objective.

**Methodology.** For each candidate proxy reward `R(s)`:
1. Run a gradient-free search procedure (e.g., reversible KV-cache MCTS, beam search with reward-weighted reranking, or a simpler activation-steering sweep) on a *frozen* base model with `R` as the search objective.
2. Run the same search with a controlled negative control reward — uniform random, length-only, or a perturbed-`R` — at matched compute budget.
3. Measure (a) whether the search-selected outputs move toward the intended downstream metric (correctness on held-out benchmark; human-preference agreement; safety classifier score); (b) whether the selected outputs concentrate on prior structure (token distributions, first-letter bias, position-dependent artifacts, format conventions).
4. The gap between (a) and (b) is the Goodhart-vulnerability score. Large gap = proxy is robust to prior exploitation under that base model. Small gap = proxy will likely be Goodharted under RL.

**Empirical demonstration.** Audit 5–10 RLHF/RLAIF reward functions on 1–3 base model scales. Predict which will Goodhart under subsequent RL training. Then actually train each with PPO/DPO and verify the predictions.

**Methodological contribution.** A reusable pre-training diagnostic that is:
- Cheap (no gradients, no training runs; hours on a single GPU)
- Interpretable (you see which prior is being exploited)
- Predictive (Goodhart-vulnerability score correlates with post-RL behavior)

**Why this might be new:** the components exist separately (Goodhart literature, reward-hacking work, inference-time analysis), but a *unified pre-training audit protocol with controlled negative controls* applied specifically to RLHF reward design — and validated by training the audited rewards — does not appear to have been published as of my last knowledge (January 2026). This is exactly what the literature scan needs to confirm or refute.

**Why this matters:** RLHF training runs cost tens to hundreds of thousands of dollars. A cheap pre-training audit that catches Goodhart-prone rewards before training saves real money and avoids deploying models that gamed their reward. Even a 30% catch rate on Goodhart failures would be valuable.

**What this paper does NOT claim:** it doesn't claim to fix Goodhart; it claims to *detect* it cheaply. It doesn't claim novelty in defining Goodhart or in the broad concept of reward hacking; it claims novelty in the *audit protocol* and its empirical validation.

---

## Deep research prompt for Gemini 3.1 Pro

> **How to use:** open Gemini 3.1 Pro Deep Research. Paste everything below the horizontal rule. Web-search and multi-turn deliberation should be enabled.

---

You are an experienced ML/NLP researcher with detailed working knowledge of the RLHF, reward modeling, reward hacking, and inference-time analysis literatures across NeurIPS, ICLR, ICML, EMNLP, ACL, and the major ML safety venues. You have written reviews for at least three of these venues.

A team is considering writing a follow-on paper after their current EMNLP 2026 submission. Before they commit to the direction, they need a thorough literature scan to determine novelty. Your job is to find work that already covers this premise, fully or partially, and report what's been done.

**Operating assumptions:**

1. **Default to "already done" until evidence shows otherwise.** Most plausible-sounding research directions have been at least partially explored. Find the partial coverage.
2. **You are not here to validate the premise.** If the premise is fully covered by existing work, say so plainly and cite the papers. The team needs to know.
3. **Distinguish "components exist" from "this exact framing is published."** A paper on Goodhart detection that doesn't run pre-training audits is component coverage, not full coverage.
4. **Specifically scan 2023–2026 publications.** This is a fast-moving field. A paper from 2024 may have done exactly this and not yet propagated into common knowledge.

**The premise to scan against:**

[Paste the "Core claim" + "Methodology" + "Empirical demonstration" + "Methodological contribution" sections from the premise above]

**What to search for, in order:**

1. **Direct hits.** Papers that propose a pre-training audit protocol for RLHF/DPO/RLAIF reward functions using inference-time analysis on a frozen base model. Search terms to try (combine freely):
   - "reward hacking detection" + "inference time"
   - "Goodhart" + "language model" + "audit"
   - "reward model evaluation" + "before training" / "pre-training" / "ahead of training"
   - "proxy reward" + "diagnostic"
   - "specification gaming" + "language model" + "detect"
   - "reward model debugging" / "reward function debugging"

2. **Component coverage.** Even if no paper does the full protocol, scan for:
   - Goodhart's Law characterizations in ML / LLMs (Manheim & Garrabrant 2018; Krakovna et al.; Skalse et al. "Defining and Characterizing Reward Hacking" NeurIPS 2022; Pan et al. "The Effects of Reward Misspecification" ICLR 2022; recent 2024–2026 work)
   - Reward model evaluation benchmarks (RewardBench Lambert et al.; subsequent reward-model audits)
   - Pre-RL reward function red-teaming (any "red-team your reward before training" work)
   - Inference-time analysis of reward models (using frozen base models to probe reward function behavior)
   - Reward model robustness / specification robustness (Gleave et al. on reward function distance metrics; recent work on reward model overfitting)
   - Length bias, format bias, sycophancy detection in RLHF — and any work using inference-time methods to detect these BEFORE training

3. **Adjacent methodology.** Has anyone:
   - Used gradient-free search (MCTS, beam, sampling) on frozen models specifically to audit reward functions?
   - Built controlled-negative-control protocols for reward proxy evaluation?
   - Validated pre-training predictions against post-RL outcomes (predict-then-train studies)?

4. **Find the closest 3–5 papers** to the premise. For each, summarize:
   - Title, venue, year, lead author
   - What the paper actually does (one paragraph)
   - What overlaps with the premise (specific)
   - What the premise adds that this paper doesn't (specific)
   - Whether the premise is still publishable given this paper

5. **Find the most recent (2025–2026) work that might supersede the premise.** Specifically check NeurIPS 2025, ICLR 2025 (and 2026 if submissions are public), ICML 2025, EMNLP 2025, the ML safety workshops (SafeML, ML Safety NeurIPS workshop, etc.), and arXiv preprints from late 2025 onward.

**Output format:**

A structured markdown report with these sections:

```
1. VERDICT (one paragraph)
   One of:
   - "Premise is publishable as-is — no direct prior work found"
   - "Premise is publishable with reframing — partial coverage exists; recommend [specific reframe]"
   - "Premise is largely covered by [paper X] — recommend pivot to [specific adjacent direction]"
   - "Premise is fully covered — do not pursue"

2. THE 3–5 CLOSEST PAPERS
   For each: title / venue / year / what it does / what overlaps / what it doesn't cover.

3. COMPONENT COVERAGE MAP
   Table or list mapping each component of the premise (pre-training audit; inference-time
   reward analysis; controlled negative controls; Goodhart-vulnerability score; predict-then-
   train validation) to the prior work that covers it.

4. THE GAP THE PREMISE FILLS (if any)
   In two sentences. If there is no gap, write "no gap" and explain.

5. RECOMMENDED REFRAMING (if applicable)
   If the premise is partially covered, propose the smallest reframing that makes it
   publishable. Be specific — e.g., "shift scope from RLHF generally to RLAIF specifically,
   where prior work [Y] is thinner."

6. NOTES ON VENUE FIT
   For the surviving framing (if any), suggest 2–3 candidate venues with rationale.
```

**Anti-anchoring rules:**

- Do not soften findings. If the premise is dead, say it's dead.
- Do not pad the closest-papers list with weak hits. 3 strong hits beats 5 medium ones.
- Do not assume the team has read everything; cite specific papers with year and venue.
- Cite arXiv IDs where possible.

Target length: 1,500–2,500 words. Tight is better than thorough.

Begin.

---

## What to do with the result

| Verdict | Action |
|---|---|
| Publishable as-is | Add to post-EMNLP queue; sketch experimental plan; check timing against ICLR 2027 (Sep 2026 deadline) / NeurIPS 2027 (May 2027 deadline) / EMNLP 2027 |
| Publishable with reframing | Adopt the reframing; sketch experimental plan |
| Covered by paper X — recommend pivot | Read paper X carefully; consider whether the pivot direction is interesting |
| Fully covered | Drop. Save the time. Pick a different next-paper direction. |

In all cases: save the Gemini report to `docs/logs/2026-05-DD_next-paper-novelty-check-report.md` for the record. Useful context if the question recurs.
