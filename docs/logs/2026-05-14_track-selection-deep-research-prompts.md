# EMNLP 2026 Track-Selection Deep-Research Prompts

**Authored:** 2026-05-14 EOD (Day 5 of plan v2, post-§5-Exp-2-recast)
**For execution:** Day 6 (2026-05-15), backup Day 7 morning. Hard deadline Day 10 EOD (2026-05-19) before Day-11 co-author send-out 2026-05-20.
**Purpose:** Triangulate the answer to "which of #13 Interpretability / #6 Efficient Methods / #17 ML for NLP maximizes acceptance probability for our specific paper?" via four parallel Deep Research sessions.

## How to use

Run all four prompts in parallel as separate Deep Research sessions (Gemini Deep Research, Claude Research, or equivalent). Each returns a standalone report. Synthesis at end:

- Prompts 2, 3, 4 each give paper-shape-vs-track-norms fit for one candidate track.
- Prompt 1 gives base-rate acceptance data across the three candidates.
- After all four return, compare: (a) which track has highest base rate (Prompt 1), (b) which has best paper-shape fit (Prompts 2-4), (c) where the two signals diverge if anywhere.
- Synthesis is a Session-A 30-min task; output is a one-paragraph track-pick recommendation appended to the `docs/logs/2026-05-13_session-log.md` "Track-selection decision" entry.

If synthesis still points to #13 → no action needed (current J6c row stands).
If synthesis points to #6 or #17 → update J6c row + project_context.md + co-author cover note before Day 11.

---

## Prompt 1 — Track-level acceptance rates at EMNLP/ARR 2024-2025

```
Context: We are submitting a paper to EMNLP 2026 via the ARR May 2026 cycle.
We need to choose one of 29 subject-area tracks. We are considering three:
  #6 "Efficient Methods for NLP"
  #13 "Interpretability and Analysis of Models for NLP"
  #17 "Machine Learning for NLP"
The 29-track list is at https://2026.emnlp.org/calls/main_conference_papers/.

Research question: What are the per-track acceptance rates at EMNLP 2024
(Miami), EMNLP 2025 (Suzhou), and the ARR cycles that fed those venues
(roughly: ARR cycles from Oct 2023 through April 2024 for EMNLP 2024;
ARR cycles from Oct 2024 through April 2025 for EMNLP 2025)?

Specifically:
  1. For each of the three target tracks, report the acceptance rate at
     EMNLP 2024 main + Findings, and EMNLP 2025 main + Findings. Distinguish
     main-conference rate from Findings rate. If your data only covers one,
     say so explicitly.
  2. If track-level rates are not published, report the AGGREGATE EMNLP
     acceptance rate for each cycle (main / Findings / overall) and name
     the source.
  3. If any chair editorial / closing-remarks talk / ACL blog post mentions
     per-track quality observations (e.g., "the X track was particularly
     competitive this year"), capture those quotes verbatim with citation.
  4. If ARR per-cycle reports break down soundness/excitement/overall meta
     score distributions by track for the three target tracks, include
     those.
  5. Note any visible trend: is one of the three tracks historically
     stricter, more lenient, or stable cycle-to-cycle?

Sources to prioritize (best-effort if unreachable):
  - Official EMNLP / ACL chair reports posted to conference websites
  - ARR cycle reports at https://aclrollingreview.org
  - ACL Anthology metadata (paper-track tags allow post-hoc rate estimation
    if accepted paper counts are public and total submissions are known)
  - Conference opening / closing slides shared on Twitter / X / Bluesky
    by chairs
  - Academic blog summaries (e.g., Sebastian Ruder's NLP newsletter,
    individual chair blogs)

Format: numbered list per cycle (EMNLP 2024, EMNLP 2025, plus per-ARR-cycle
data if available). For each cycle, give the three target-track rates if
available, or "not published" + best aggregate + source. Every numeric
claim must cite a source URL. No speculation about why rates differ — just
data.

Length target: 800-1200 words.
```

---

## Prompt 2 — Interpretability track paper-shape norms

```
Context: We are submitting a paper to the EMNLP 2026 "Interpretability and
Analysis of Models for NLP" track (#13 of 29 at
https://2026.emnlp.org/calls/main_conference_papers/). Our paper has a
specific structural shape and we need to characterize how the
Interpretability track at EMNLP 2024-2025 and ACL 2024-2025 received papers
shaped like ours.

Our paper's shape (one-line per facet):
  - Primary novelty: a reversible Monte Carlo Tree Search algorithm
    operating in KV-cache latent space, with a formal memory-bound
    theorem (~3.04× the KV-cache footprint vs naive parallel-cache MCTS,
    proven zero-drift over 200 cycles via FP32 accumulator construction).
  - Empirical contribution 1: a controlled negative-control on ARC-Easy
    (n=200 paired items per scale × 2 scales = 1200 records) showing that
    an entropy-minimizing MCTS reward in latent space fails to improve
    task accuracy — paired McNemar p=1.0 at both 1B and 3B Llama-3.2
    scales.
  - Empirical contribution 2: a "Latent Cartography" sweep characterizing
    class-conditional best-path selection across 4 prompt classes
    (factual recall, ARC-Easy MCQ, TruthfulQA MCQ, HellaSwag continuation)
    × 2 model scales (1B, 3B). The search exploits class-specific
    pre-training priors rather than reasoning-relevant features.
  - Probing infrastructure: a per-layer "telemetry matrix" combining
    H-Neuron stress (bottom-up activation magnitudes) and RepE
    honesty-probe alignment (top-down, Zou et al. 2023) at each
    transformer layer.
  - Frozen-weights regime throughout. Inference-time only. No training,
    no fine-tuning, no causal mediation analysis, no circuit-level
    claims, no learned probing classifiers.

Research question: How does the Interpretability and Analysis of Models
for NLP track at EMNLP/ACL/EACL 2024-2025 receive papers shaped like
the above?

Specifically:
  1. Sample 20-30 accepted Interpretability-track papers from EMNLP 2024,
     EMNLP 2025, ACL 2024, ACL 2025 (or EACL 2024 if other sources are
     thin). For each, characterize:
       (a) Primary contribution type: new algorithm / new finding about
           a model / new probing method / new dataset / formal analysis
           / position paper / survey.
       (b) Whether the empirical contribution is a positive
           characterization (e.g., "here's how X works"), a negative
           result framed as mechanism diagnosis (e.g., "X fails because
           Y"), or both.
       (c) Interpretability methods used: probing classifier / causal
           mediation / activation patching / circuit analysis / RepE
           representation engineering / steering vector / sparse
           autoencoder / attention analysis / other.
       (d) Whether the paper's primary novelty engine is the
           interpretability finding itself or an algorithm/method that
           enables interpretability work.
  2. Identify patterns from the sample:
       (a) Do algorithm-primary papers succeed at this track, or do they
           typically route to Methods / Efficient-Methods / ML-for-NLP
           tracks?
       (b) Do papers whose dominant empirical contribution is a negative
           result succeed at this track if framed as mechanism diagnosis?
           If yes, what pattern do those papers share?
       (c) Are there examples of papers whose primary novelty is an
           algorithm + memory bound (or any formal-complexity claim)
           that succeeded at the Interpretability track because of
           probing/interpretability framing in their analysis?
  3. Quote the track's CFP scope language from the EMNLP 2026 call page
     (https://2026.emnlp.org/calls/main_conference_papers/) verbatim.
     Does it explicitly include or exclude any of the paper-shape facets
     above?

Sources: ACL Anthology filtered by track (https://aclanthology.org);
sample paper abstracts and introductions from those listings; track
CFP descriptions; track-chair closing remarks if any.

Format:
  (1) ~200-word characterization of the track's accepted-paper shape
      based on the 20-30-paper sample.
  (2) A table of sampled papers with columns: title, venue/year, primary
      contribution type, empirical sign (positive / negative / both),
      interp methods used, novelty engine (interp finding vs algorithm).
  (3) A direct one-paragraph answer to "does our paper's shape fit this
      track?" citing 3-4 of the sampled papers as comparison anchors
      (specifically: papers whose shape resembles ours, papers that
      represent the track's modal shape, papers that nearly missed
      the track scope).

Length target: 2000-3000 words. No speculation about reviewer
psychology — only patterns from observed accepted papers and verbatim
CFP language.
```

---

## Prompt 3 — Efficient Methods track paper-shape norms

```
Context: We are considering the EMNLP 2026 "Efficient Methods for NLP"
track (#6 of 29 at
https://2026.emnlp.org/calls/main_conference_papers/) as either primary
or secondary track for our paper. The paper's strongest never-falsified
contribution is a memory bound (~3.04× M_KV measured, ~60× reduction
projected at 20B model scale vs naive parallel-cache MCTS) backed by a
formal reversibility theorem (FP32 accumulator construction; zero drift
over 200 mutate-then-reverse cycles on Llama-3.2 1B and 3B). The
empirical contribution is a characterization + a negative-result
diagnostic, NOT an end-task gain.

Our paper's shape: see Prompt 2 for full facets. Key point for this
track: efficiency contribution = reversibility + memory bound for
latent-space search infrastructure. Target use case = diagnostic
probing of reward-driven search failure modes. NO downstream task
accuracy gains.

Research question: What is the shape of papers accepted to the Efficient
Methods for NLP track at EMNLP/ACL 2024-2025?

Specifically:
  1. Sample 20-30 accepted Efficient-Methods papers from EMNLP 2024,
     EMNLP 2025, ACL 2024, ACL 2025. For each, characterize:
       (a) Primary efficiency claim type: memory reduction / compute
           reduction / latency reduction / parameter reduction / energy
           reduction / data-efficiency / a combination.
       (b) Whether the paper pairs the efficiency claim with an
           end-task gain (e.g., "we match accuracy at lower cost") or
           is purely an efficiency-characterization paper without
           end-task gains.
       (c) Whether the paper includes formal analysis (theorem /
           bound / complexity claim) or is purely empirical
           benchmarking.
       (d) Target model scale (small <1B / medium 1-10B / large
           >10B / mixed).
       (e) Target use case: training / inference / fine-tuning /
           search-time / a combination.
  2. Identify patterns:
       (a) Do Efficient Methods reviewers accept papers where the
           efficiency claim is paired with a characterization /
           diagnostic finding rather than an end-task gain? Or do
           reviewers consistently require "this efficient method
           enables X downstream task improvement"?
       (b) Is a paper whose efficiency contribution is for
           latent-space search infrastructure (rather than for the
           more common inference or training efficiency targets)
           within scope? Are there precedent papers about
           search-time efficiency at this track?
       (c) What proportion of accepted papers include a formal
           theorem / bound vs purely empirical efficiency claims?
  3. Quote the track's CFP scope language verbatim from the EMNLP
     2026 call page. Does the scope explicitly mention
     latent-space search, inference-time-only methods, or
     memory-bounded probing infrastructure? What's explicitly
     in and out of scope?

Sources: ACL Anthology + CFP pages + sample of accepted paper
abstracts.

Format: same structure as Prompt 2 (200-word characterization +
table + direct fit-paragraph with 3-4 comparison anchors).

Length target: 2000-3000 words.
```

---

## Prompt 4 — ML-for-NLP negative-result acceptance patterns

```
Context: We are considering the EMNLP 2026 "Machine Learning for NLP"
track (#17 of 29 at
https://2026.emnlp.org/calls/main_conference_papers/) as a candidate
track for our paper. We are currently leaning AGAINST this track on
the basis that ML-for-NLP reviewers historically expect end-task
benchmark gains, and our paper's central empirical result is a
negative control: an entropy-minimizing MCTS in latent space fails to
improve ARC-Easy accuracy (paired McNemar p=1.0 at both Llama-3.2 1B
and 3B scales; the search succeeds at minimizing entropy but the
entropy-reduction does not correspond to better answers — Goodhart's
Law manifestation in cache geometry). We need to verify whether this
"avoid ML-for-NLP because of negative result" reasoning holds against
the actual acceptance pattern.

Our paper's shape: see Prompt 2 for full facets. Key point for this
track: the empirical headline IS a negative-result mechanism
diagnosis. We do also have a positive contribution (the reversible
KV-cache MCTS algorithm + formal memory bound) but the central
empirical takeaway is "minimizing this latent-space surrogate reward
does not improve task accuracy, and here's why geometrically."

Research question: How do papers with negative empirical results fare
at the Machine Learning for NLP track at EMNLP/ACL 2024-2025?

Specifically:
  1. Sample 20-30 accepted ML-for-NLP papers from EMNLP 2024, EMNLP
     2025, ACL 2024, ACL 2025. For each, determine:
       (a) Whether the primary empirical finding is a positive task
           gain, a negative result, or a mixed contribution where the
           negative result accompanies a positive method.
       (b) If the paper contains a negative result, is it the
           PRIMARY empirical contribution (paper's headline finding)
           or is it secondary support for a positive contribution
           (e.g., "method X works AND method Y doesn't, here's why
           X works better")?
       (c) If the negative result is primary, how is the paper framed
           (failure mode characterization / surrogate-reward
           pathology / out-of-distribution analysis / mechanism
           diagnosis / theoretical impossibility result)?
       (d) Does the paper have an accompanying theoretical
           contribution (proof / bound / formal mechanism) that
           offsets the absence of task gains?
  2. Identify patterns:
       (a) What share of accepted ML-for-NLP papers have negative
           results as the PRIMARY empirical contribution?
       (b) Are there clear precedent papers shaped like ours
           (algorithm + formal bound + negative-result mechanism
           diagnosis)? If yes, sample 2-3 for direct comparison.
       (c) What share of accepted papers have at least one negative
           result section (even if not primary)? This estimates
           whether the track is hostile or merely indifferent to
           negative findings.
  3. Quote the track's CFP scope language verbatim. Does the scope
     explicitly invite, discourage, or stay silent on negative-result
     submissions? Compare to the Interpretability track's scope
     language (Prompt 2) for explicit-or-implicit shape signals.

Sources: ACL Anthology + CFP pages + sample of accepted abstracts.

Format: same structure as Prompts 2-3 (characterization + table +
direct fit-paragraph).

The CRITICAL output is a calibrated answer to: "would our
negative-result-primary paper plausibly be accepted at the
ML-for-NLP track, or would the negative result be read against us?"
Answer with one of three calibrated levels:
  - "Plausibly accepted" with 2-3 precedent papers cited
  - "Indifferent — depends on reviewer luck" with rationale
  - "Likely read against — clear pattern of expecting positive gains"
    with the pattern documented

Length target: 2000-3000 words.
```

---

## Synthesis protocol (for after all four prompts return)

When all four reports are in hand, run a synthesis pass:

1. **Base-rate diff** (Prompt 1): if any of the three tracks has a
   >5pp acceptance-rate advantage over the others at recent EMNLP
   cycles, flag it. <5pp is noise.
2. **Paper-shape fit** (Prompts 2-4): score each track as "Strong fit"
   / "Plausible fit" / "Risky fit" based on the fit-paragraph in each
   prompt's output.
3. **Decision matrix:**
   - If one track scores Strong fit + ≥0pp base-rate advantage → lock
     that track.
   - If two tracks tie at Strong fit → favor the one whose
     fit-paragraph cites more precedent papers shaped like ours.
   - If #13 scores Strong, #6 scores Plausible, #17 scores Risky → keep
     #13 primary, evaluate ARR portal for secondary-track support; if
     supported, add #6 as secondary.
   - If #13 scores Risky and #6 scores Strong → consider switching
     primary to #6 BEFORE Day-11 send-out. Reframe needed for title
     and abstract on Days 6-10.
   - If all three score Risky → escalate to Tianyu directly Day-7 or
     Day-8 (do not wait for Day-11 send-out; he has acceptance
     priors we don't).
4. **Update artifacts** (only if decision changes):
   - J6c row in `docs/logs/2026-05-10_14-day-execution-plan.md` (revise
     track string, update reasoning pointer).
   - `memory/project_context.md` venue/deadline paragraph (revise
     track string).
   - `docs/logs/2026-05-13_session-log.md` Track-selection decision
     entry: append a "REVISION 2026-05-1X" section reflecting the
     new pick and the deep-research evidence.
   - Day-11 co-author cover note: include the original reasoning,
     the deep-research evidence, and the final pick.

Synthesis wall-time target: 30 min if all four prompts return well-formed.
Up to 60 min if any prompt returns ambiguous or contradictory data and
requires a follow-up Deep Research session.
