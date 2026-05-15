# Pre-Submission Parallel Adversarial Review Plan

**Authored:** 2026-05-14 EOD (Day 5 of plan v2 execution, post-track-selection synthesis).
**Postponed 2026-05-14 EOD per Josh's directive:** original timing was Days 6-7 (paper-content half-written); revised to **Days 12-13** so adversarial review operates on the structurally-complete paper (post-Day-10 paper-content freeze + post-Day-11 co-author send-out). Adversarial review now runs IN PARALLEL with co-author review on Days 12-13, not BEFORE it.
**Purpose:** Surface implementation bugs, framing overclaims, reproducibility gaps, reviewer-prior misalignments, and structural vulnerabilities **before** Day-15 ARR portal submission. Compensates for the structural absence of institutional mentorship by running an adversarial process against the paper before reviewers do.
**Owner:** Josh, with Session A synthesis support.
**Trigger context:** Josh flagged (paraphrased) — "I have a secret fear that the experiments were run incorrectly, or the data was hallucinated. The only way I'll feel confident before submission is to run review sessions in parallel and attack the submission."
**Synthesis output target:** [`docs/logs/2026-05-22_parallel-review-synthesis.md`](2026-05-22_parallel-review-synthesis.md) (Day-13 EOD authoring).

---

## Timing (revised 2026-05-14 EOD)

| Day | Calendar | Activity |
|---|---|---|
| **Days 6-10** | 2026-05-15 → 2026-05-19 | (Not adversarial-review window.) Plan-v2 paper-content work: §5.5 Metrics rewrite (Day 7) + ACL style switch (Day 8) + §6 worked-example numbers (Day 8) + page-budget check + cuts (Day 9) + §A Reproducibility Appendix application + Responsible NLP final (Day 10). **Paper-content freezes Day 10 EOD.** |
| **Day 11** | 2026-05-20 | Co-author send-out to Tianyu + Max + Alaa (paper now in 4-day async review window). **Adversarial review NOT yet fired** — wait for paper to be ~24h into co-author review so any early Day-11 EOD feedback can inform Round 1 scoping. |
| **Day 12** | 2026-05-21 AM | Fire **Round 1**: Axes 1 (code-semantic correctness), 2 (framing defensibility), 5 (standard adversarial). Highest-leverage axes. ~3 parallel sessions, ~1-2 hr total wall (mostly waiting). Simultaneously: Day-12 supplementary anonymization continues per plan v2 §5.1. |
| **Day 12** | 2026-05-21 PM | Synthesize Round 1 results + draft patches. If Round 1 shows ≥1 SEVERE finding, pause Round 2 firing until SEVERE patches land or escalation to Tianyu happens. |
| **Day 13** | 2026-05-22 AM | Fire **Round 2**: Axes 3 (reproducibility), 4 (reviewer-prior alignment for #13). ~2 parallel sessions, ~1 hr wall. |
| **Day 13** | 2026-05-22 PM | Synthesize all 5 axes into [`docs/logs/2026-05-22_parallel-review-synthesis.md`](2026-05-22_parallel-review-synthesis.md). Combine with whatever co-author feedback has arrived (Day-11 send-out + Day-13 = ~48h, partial feedback likely). |
| **Day 14** | 2026-05-23 | Integrate co-author feedback + adversarial findings into the canonical TeX. Apply SEVERE/MAJOR patches by Day-14 EOD. Final REVISION-marker audit + double-blind compile check per plan v2 §2 Day-14 nominal task. |
| **Day 15** | 2026-05-24 | ARR portal click-through + submission. |
| **Submission** | 2026-05-25 | ARR deadline. |

**Decision rule for Round 2:** if Round 1 surfaces ≥1 SEVERE or ≥3 MAJOR findings, pause Round 2 firing on Day 13 until Round 1 patches are at least scoped (not necessarily landed). Round 2 results on an unpatched-Round-1 base produce noise instead of signal.

**Trip-wire for SEVERE on Day 12:** if a SEVERE code-vs-paper divergence surfaces in Axis 1 (logomesh/kv_mcts.py semantic-correctness), escalate to Tianyu immediately (don't wait for the 4-day co-author async window to complete). A code-vs-paper SEVERE is a paper-go/no-go-level call; Tianyu's advisor judgment is required.

## Parallelization model

5 axes, each runs as a separate fresh-context session. No cross-axis dependencies during firing — sessions are independently runnable.

| Axis | Recommended agent | Why |
|---|---|---|
| **1. Code-semantic correctness** | Fresh Claude Code session (Opus or Sonnet) | Needs filesystem access to read `.py` and `.tex` together; can run scripts if needed |
| **2. Framing defensibility** | Gemini Deep Research OR Claude Opus | Pure analytical reading; no code execution needed |
| **3. Reproducibility from §A** | Fresh Claude Code session | Needs to simulate following the appendix step-by-step from filesystem |
| **4. Reviewer-prior alignment (#13)** | Gemini Deep Research | Web access useful for additional precedent papers beyond the 26 already sampled |
| **5. Standard adversarial pass** | Claude Opus (chat session, no filesystem needed if paper PDF attached) | Senior-reviewer-shaped read; needs the whole paper in context |

Each session takes 30-60 min wall (autonomous research/analysis) + 10-15 min setup. Parallel firing means total Round 1 wall ≈ 1-2 hr, mostly idle.

## Severity rubric for findings

Each adversarial-review prompt instructs the agent to rate findings on this scale:

| Severity | Definition | Patch deadline |
|---|---|---|
| **SEVERE** | Paper-claim contradicted by primary source / code-vs-paper divergence / unreproducible result claim / illegal/unethical practice | Pre-send-out (Day 10 EOD) |
| **MAJOR** | Reviewer-likely rejection trigger / framing overclaim / methodology gap that invalidates a contribution | Pre-send-out (Day 10 EOD) |
| **MODERATE** | Reviewer-likely revision request / unclear hedging / weak baseline / missing detail | Pre-submission (Day 14 EOD) |
| **MINOR** | Cosmetic / typo / could-be-clearer / nice-to-have improvements | Best-effort; not blocking |

The pre-submission patch deadline matrix is the synthesis protocol's decision rule.

---

## Axis 1 — Code-semantic correctness on `logomesh/kv_mcts.py`

**Critical axis.** The Track G audit pass and the figure-vs-raw verification confirm result reproducibility (same JSON → same number) but do NOT verify that the algorithm computes what the equations claim at the semantic level.

```
You are reviewing a Python module that implements the algorithm described
in §4 of a NeurIPS / EMNLP paper submission. Your job is to verify that
the code computes what the equations claim at the semantic level.

REQUIRED READING (in this order, all in full):
  1. docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex §4 (Proposed
     Method) — read the algorithm description, equation definitions, and
     theorem statement. Note specifically: Theorem 1 (FP32 accumulator
     reversibility), the steering-direction projection (Eq. 6), the
     reward formulation (Eq. 8 if present + Eq. goodhart for entropy
     reward), the MCTS path-sampling protocol.
  2. logomesh/kv_mcts.py — full file. Note especially: FP32Accumulator
     class, apply/rollback methods, _project_residual_to_kv_shape,
     _extract_kv_tensors, _kv_eval_cache, MCTSConfig dataclass, ReversibleMCTS class.
  3. logomesh/local_model.py — LocalLlamaOracle.get_kv_cache and
     set_kv_cache methods (the KV-cache mutation interface the MCTS
     calls).
  4. docs/logs/2026-04-16_session-log.md — read the section on Phase 3
     gate validation (KV-cache mutability probe + 10-node MCTS smoke
     test + 200-cycle Lipschitz drift validation). Confirms what was
     measured.
  5. scripts/measure_lipschitz_drift.py — the drift-validation script
     that produced the "0.00e+00 drift over 200 cycles" claim.

For each of the following code-vs-paper mapping questions, verify the
mapping is mathematically equivalent or flag the divergence:

  Q1. Does FP32Accumulator.apply + .rollback implement what the paper
      claims in Theorem 1 (Eq. 6: "A^(ℓ) ← A^(ℓ) + α·d^(ℓ)" with FP32
      accumulator, "K_t ← K_base + cast_bf16(A^(ℓ))" cast on read)? Is
      the FP32 storage actually FP32 (not silently upcast/downcast)?
      Is the cast on read symmetric so apply+rollback returns to the
      exact starting state at bf16 precision?

  Q2. Does _project_residual_to_kv_shape implement a valid projection
      from residual-stream space to KV-cache space? Specifically:
      what is the projection matrix? Where does it come from
      (learned / W_K weight / random)? Is the projection norm-preserving
      or norm-shrinking? Does the paper claim the projection has any
      formal property that the code does or doesn't satisfy?

  Q3. Does the entropy reward in the MCTS reward function match Eq.
      goodhart in the paper: `hat{r}(h) = -H(p(.|h)) / log|V|`? Verify:
      is H the Shannon entropy over the full vocabulary softmax of the
      last-position logits? Is the normalization by log|V| applied
      correctly (and is |V| = 128256 for Llama 3.2)?

  Q4. Does the MCTS path enumeration in run_kv_mcts.py / Track F driver
      actually enumerate ALL 27 paths from the depth-3 alpha-set
      {0.1, 0.5, 1.0}, or does it use a UCB1 bandit (which the paper
      says is bypassed for the depth-3 path-sampling variant)? Confirm
      no path is silently skipped or de-duplicated.

  Q5. Does the steering direction stay invariant across items in a
      Track F run (the COCONUT defense)? Confirm the random unit vector
      is generated once at run start (--seed 42) and reused, not
      re-sampled per item.

  Q6. Lipschitz drift: confirm the measurement script measures drift in
      the way the paper claims (cycle = apply then rollback; drift =
      L2 norm of cache_after - cache_before after N cycles). If the
      cache state has multiple components (K and V for each layer,
      multiple layers), confirm the drift is measured across ALL of
      them, not a representative subset.

REPORT FORMAT:
  - 6 sections, one per Q. For each Q: (a) the paper claim verbatim,
    (b) the code location (file:line), (c) the mathematical mapping,
    (d) verdict (CLEAN / DIVERGENT / UNDERSPECIFIED-IN-PAPER), (e)
    severity if not CLEAN (SEVERE / MAJOR / MODERATE / MINOR per the
    rubric in the parallel-review plan).
  - Final summary section: total findings by severity. List of "what
    would I tell Josh to patch first."

Length target: 2000-3500 words. Do not hedge — if you see a divergence,
name it. If the paper underspecifies something, say so explicitly.
```

---

## Axis 2 — Dimensional Escape framing defensibility (§3 + §7)

**Tests whether the theoretical framing overclaims relative to what the data supports.**

```
You are reviewing the theoretical framing of a NeurIPS / EMNLP paper
submission. Your job is to test whether the "Dimensional Escape" framing
is a rigorous geometric claim supported by the data, or theory layered on
top of the data that overclaims.

The paper's framing arc:
  - §1 Contributions: "empirical demonstration of Dimensional Escape"
    (Contribution 3)
  - §3.2: definition + Hypothesis 1 (deception-driven case, monitor-led)
    + Hypothesis 2 (reward-driven case, surrogate-led) + Eq. goodhart
  - §5 Experiments: Experiment 1 (Track F) provides reward-driven
    evidence; Experiment 2 (Track G Cartography) extends across classes
  - §7 Discussion: implications for monitor design + reward design;
    Dimensional Escape as unifying framework

REQUIRED READING:
  1. docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex — §1
     (Contributions), §3 (Problem Formulation, all subsections), §5
     (Experiments — Exp 1 + Exp 2), §7 (Discussion + Limitations).
  2. docs/logs/2026-05-11_track-F-negative-control-report.md §3
     (statistics), §6 (failure cases), §7 (mechanism diagnosis), §10
     (protocol caveats).
  3. docs/logs/2026-05-11_track-G-cartography-report.md §2.1-§2.4
     (per-class findings), §3 (cross-scale prose).
  4. docs/logs/2026-05-09_Latent-Space MCTS_ Reward Design Strategy.md
     Part 5 (the original Dimensional Escape sketch from Gemini round 4).

For each of the following framing-defensibility questions, identify
overclaims, hedging gaps, and stronger alternative framings:

  Q1. Definition of Dimensional Escape (§3.2): is the definition
      operationally testable? What would falsify it? Are the conditions
      under which it manifests stated precisely enough that a reviewer
      could check them on a new model?

  Q2. Hypothesis 2 (reward-driven case): does Track F actually evidence
      H2 as stated, or evidence a weaker statement? Specifically, H2
      claims "the optimal cache state under r̂ generically lies off the
      semantic manifold." Track F shows that minimizing entropy in cache
      doesn't improve task accuracy. Is "off the semantic manifold" the
      right interpretation, or is it consistent with "the entropy
      gradient is decoupled from the answer gradient" (a weaker claim)?

  Q3. Cross-experiment unification: §3 claims H1 (deception) and H2
      (reward) are both manifestations of the same geometric pathology.
      Track F evidences H2; does anything in the paper evidence H1? If
      not, is the unification an empirical claim or a theoretical
      conjecture? Should §3 hedge or §7 expand?

  Q4. The "geometric" framing: is the failure mechanism actually
      geometric, or is it a softer claim (correlation in cache space
      doesn't transfer to correlation in answer space)? "Geometric" has
      connotations of high-dimensional manifold structure; do the Track
      F + Track G data support that level of geometric claim, or do they
      support a weaker statistical claim?

  Q5. Goodhart-vs-Dimensional Escape: §3 + §7 use both terms. Is
      Dimensional Escape genuinely distinct from Goodhart's Law, or is
      it a relabeling? If distinct, what is the precise distinction?
      (Reviewers will ask this.)

  Q6. Most-overclaim sentence in §3 + §7: pick the single sentence that
      a hostile reviewer would most likely circle as "overclaim — the
      data doesn't support this strongly." Suggest a hedged
      replacement.

  Q7. Reviewer-likely Limitations question: §7.3 has 8 numbered
      limitations. Identify the strongest critique the paper does NOT
      acknowledge — the one a reviewer will raise that isn't pre-empted
      by §7.3.

REPORT FORMAT:
  - 7 sections, one per Q. For each Q: (a) the relevant paper passage
    quoted verbatim, (b) the data the passage rests on (cite Track F /
    Track G report sections), (c) verdict (DEFENSIBLE / HEDGE-NEEDED /
    OVERCLAIM), (d) suggested rewrite if HEDGE-NEEDED or OVERCLAIM, (e)
    severity per the rubric.
  - Final summary: ranked list of framing patches needed before
    submission.

Length target: 2000-3000 words.
```

---

## Axis 3 — Reproducibility from §A

**Tests whether the Reproducibility Appendix actually enables reproduction.**

```
You are reviewing a Reproducibility Appendix for a NeurIPS / EMNLP paper
submission. Your job is to simulate following the appendix from scratch
and identify gaps that would prevent a reviewer or community researcher
from reproducing the paper's headline result (Track F: paired McNemar
p=1.0 at both Llama-3.2 1B and 3B scales for entropy-MCTS vs greedy on
ARC-Easy n=200).

REQUIRED READING:
  1. docs/logs/2026-05-13_session-A-prose-drafts-v10-reproducibility-appendix.md
     — the §A draft (Section 2 contains the actual TeX block; Section 3
     contains application notes; Section 4 contains the deferred-to-
     supplementary list).
  2. docs/logs/2026-05-11_track-F-negative-control-report.md — Appendix
     A "Reproduction notes" if present. Compare what the §A draft says
     vs what the Track F report says.
  3. scripts/diagnose_track_f_negcontrol.py — the actual driver script
     §A points reviewers to. Read it to understand what CLI args, what
     dependencies, what env vars are needed.
  4. logomesh/kv_mcts.py lines 610-631 — MCTSConfig dataclass that §A
     cites as the source of defaults.

Simulate the following user journey (you are a hypothetical researcher
with access to an H100, never touched this repo before):

  Step 1. Read §A draft top-to-bottom. List everything you would need
          before starting that §A does NOT tell you. (E.g., HuggingFace
          token, accept Llama license, specific transformers version.)
  Step 2. Try to construct the conda/uv environment from §A alone.
          List dependencies that are named vs un-named. Flag any pinned
          version conflicts.
  Step 3. Try to download the model from §A's instructions. Is the
          model name unambiguous? Is the checkpoint version specified?
  Step 4. Try to construct the CLI command to reproduce Track F at 1B.
          §A says "Driver: scripts/diagnose_track_f_negcontrol.py" but
          does it give you the exact --seed, --model, --out arguments?
          Find them by reading the driver source. Note any gap.
  Step 5. Identify if §A correctly describes:
            (a) The 200-item ARC-Easy subset (which 200? indices 0-199
                deterministic? chat-template formatting?)
            (b) The MCTS protocol (depth-3, 27 paths, alpha-set
                {0.1, 0.5, 1.0}, entropy reward formula)
            (c) The steering direction (random unit vector, seed 42,
                fixed across items)
            (d) The 3 conditions (G / R / E) and their distinguishing
                features
            (e) The output measurement (accuracy via case-insensitive
                substring match; mean perplexity definition)
  Step 6. List gaps that would make a reviewer say "I tried to
          reproduce but couldn't because §A omits X."

Also assess:
  - Are the 3 deferred metadata items (HF SHAs, library pins, per-script
    CLI args) genuinely supplementary-OK, or are some of them
    submission-blocking (i.e., a reviewer would say "without this in §A
    I can't even start")?
  - Is the wall-clock claim verifiable (18.3 min at 1B, 33.7 min at 3B)?
    Hint: it requires a specific GPU class. Is the GPU class stated?

REPORT FORMAT:
  - 6 sections per the simulated journey + 2 assessment sections.
  - For each gap: (a) what's missing, (b) what a reviewer would say,
    (c) severity per the rubric, (d) suggested §A patch (with sentence
    or paragraph to add).
  - Final summary: ranked list of §A patches needed before Day-10
    application or Day-15 submission.

Length target: 2000-2500 words.
```

---

## Axis 4 — Reviewer-prior alignment for track #13

**Tests whether the paper's framing matches what successful Interpretability-track papers look like.**

```
You are reviewing a NeurIPS / EMNLP paper submission that has been
locked to the EMNLP 2026 "Interpretability and Analysis of Models for
NLP" track. Your job is to test whether the paper's framing matches the
specific shape of papers that succeed at this track — and to surface
any reviewer-prior mismatches before submission.

REQUIRED READING:
  1. docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex — full
     paper.
  2. docs/logs/gemini research reports/Analyzing Interpretability Track
     Submissions.md — the 26-paper sample from the track-selection
     synthesis 2026-05-14. The "Strategic Synthesis" section lists 4
     pillars of success at this track: Algorithmic Legitimacy /
     Negative Results as Diagnostic Tools / Formalism as Ultimate
     Interpretability / Methodological Alignment with the Bleeding
     Edge.
  3. ACL Anthology pages for 3-5 of the closest precedent papers cited
     in that report: An L* Algorithm for Deterministic Weighted Regular
     Languages (EMNLP 2024); Finite State Automata Inside Transformers
     with Chain-of-Thought (ACL 2025); CUTE: Measuring LLMs'
     Understanding of Their Tokens (EMNLP 2024); RAVEL: Evaluating
     Interpretability Methods on Disentangling Language Model
     Representations (ACL 2024).

For each of the following reviewer-prior questions, identify mismatches
and patch recommendations:

  Q1. Title alignment: the paper title is "Reversible KV-Cache MCTS: A
      Memory-Bounded Probe of Dimensional Escape in Frozen Language
      Models." Does this title signal "Interpretability track" to a
      reviewer at first glance? Compare to L*'s title structure
      (algorithm name : research question) and Finite State
      Automata's title structure (finding : mechanistic study). Does
      ours read as a Method paper, an Interpretability paper, or
      ambiguous?

  Q2. Abstract structure: read the paper's abstract. Map sentence-by-
      sentence: does each sentence move the reviewer toward
      "Interpretability track" framing or could a Methods-track
      reviewer plausibly route this elsewhere?

  Q3. Contribution claims: §1 Contributions list 3 contributions.
      Compare against the 4 pillars from the synthesis report. Does each
      contribution map to a pillar? If yes, name which. If no, flag the
      gap. Critically: the synthesis report claims the most-influential
      interp papers are "infrastructure-and-framework-generator" papers
      (Mosbach et al. 2024). Does our Contribution 1 (algorithm +
      memory bound) read as infrastructure-generator, or does it need
      explicit reframing to land that way?

  Q4. Vocabulary and keyword saturation: the ARR track keywords are
      "probing", "feature attribution", "knowledge tracing/discovering/
      inducing", "data shortcuts/artifacts". Grep the paper for these.
      Are they saturated enough in the abstract and §1 to ensure the
      ACL routing system assigns this paper to interp reviewers, not ML
      or Efficient Methods reviewers? Compare against L* and RAVEL.

  Q5. Negative-result framing: §5 Experiment 1 (Track F) is a negative
      result. The synthesis report shows negative results succeed at
      this track when framed as mechanism diagnosis. Read the §5
      Experiment 1 prose. Is it framed as mechanism diagnosis (correct)
      or as "we tried X and it didn't work" (wrong)? Identify any
      sentences that read as the latter.

  Q6. Discussion / Implications section (§7.1, §7.5): compare against
      what the Mosbach et al. meta-analysis would expect. Does §7
      explicitly call out what other researchers can DO with the
      contributions (reuse the algorithm, reuse the telemetry matrix,
      apply to their own models)? The track values "actionable
      infrastructure" — does §7 deliver?

  Q7. Most-likely reviewer reject reason: imagine a reviewer trained on
      the Interpretability-track norms. What is the single most likely
      reject-tier critique they would write? "The paper is interesting
      but lacks X." Fill in X based on the 26-paper sample patterns.

REPORT FORMAT:
  - 7 sections, one per Q. For each: (a) the relevant paper passage
    quoted, (b) the precedent-paper comparison, (c) mismatch
    identification, (d) suggested patch, (e) severity per the rubric.
  - Final summary: ranked list of reviewer-prior alignment patches.

Length target: 2000-2500 words.
```

---

## Axis 5 — Standard adversarial pass

**Senior-reviewer-shaped read. Finds the weakest claim, the most-overclaim sentence, the biggest methodology gap.**

```
You are a senior reviewer at EMNLP 2026 with 10+ years of NLP review
experience. You have been assigned this paper for adversarial review.
Your job is to find every reviewer-likely critique a real reviewer would
raise.

REQUIRED READING:
  1. docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex — full
     paper (currently 745 lines; 20 pages NeurIPS-style, projected ~9
     pages ACL two-column after Day 8 style switch).
  2. docs/logs/2026-05-11_track-F-negative-control-report.md — the
     central empirical artifact.
  3. docs/logs/2026-05-11_track-G-cartography-report.md — the second
     empirical artifact.

Run the following adversarial passes. Be specific, name passages by
section/sentence; don't hedge.

  Pass A. Most-overclaim sentence in the paper.
    Identify the single sentence anywhere in the paper that a hostile
    reviewer would most likely flag as "your data doesn't support this
    claim that strongly." Quote it verbatim. Explain why it's an
    overclaim. Suggest a hedged replacement.

  Pass B. Weakest experimental design choice.
    Identify the single weakest experimental design decision (e.g.,
    n=200 too small / single random seed / one steering direction /
    only Llama family). Quote the relevant §4 / §5 passage. Estimate
    the cost of fixing it (which we won't do — but a reviewer will
    flag it). Suggest a Limitations-section sentence that pre-empts
    the critique.

  Pass C. Missing baseline.
    A baseline a reviewer will say "you should have run this." Name
    it. Estimate effort. Recommend whether to add it (Day 14 patch
    window is ~24h, very tight) or pre-empt in Limitations.

  Pass D. Biggest methodology gap.
    A methodology choice that's underspecified or unjustified. Quote
    the §4 / §5 / §A passage where the gap lives. Suggest how to fix
    in pre-submission time.

  Pass E. Reproducibility skepticism.
    Imagine a reviewer who can't (or won't) try to reproduce. What
    aspect of the paper would they most distrust on first read? Often
    it's a too-clean result (p=1.0!), a suspiciously-perfect theorem
    claim (0.00e+00 drift!), or unusual numbers that should be
    triangulated (3.04× M_KV measured — where's the comparison
    table?).

  Pass F. Title + abstract + first paragraph "fit" test.
    Read just title + abstract + §1 Introduction paragraph 1. Form a
    snap judgment: would this clear the first-page reject filter?
    If not, what's wrong with the first impression?

  Pass G. Limitations completeness.
    §7.3 has 8 numbered limitations. Identify the strongest critique
    a reviewer will raise that §7.3 does NOT acknowledge. Suggest a
    9th limitation to add.

  Pass H. Related-work blind spot.
    Is there a critically-relevant prior work the paper doesn't cite?
    Especially: any paper that proposed something similar enough that a
    reviewer will say "this is incremental"? Bailey 2024 is in §2.4;
    is there a peer to Bailey that's missing?

  Pass I. Numerical-claim verification.
    Pick 3 quantitative claims from the abstract or §1. For each,
    verify it traces to a primary artifact (Track F/G report, raw
    JSON). If you can't trace it in 5 minutes of reading, that's a
    reviewer-likely "where does this number come from?" critique.

  Pass J. "Why this paper now?" framing.
    Reviewers ask: why is this important for the field at this moment?
    Read §1 (Introduction) for the implicit answer. Is it convincing?
    Or is it a "we did this work, here it is" submission that lacks the
    field-context motivation reviewers want?

REPORT FORMAT:
  - 10 passes (A-J), one section each. For each: the finding, the
    severity per the rubric, the recommended patch.
  - Final ranking: list all findings sorted by severity. The "fix-first"
    item is the topmost SEVERE; the "fix-best-effort" items are the
    MINORS.

Length target: 3000-4500 words. The most comprehensive of the 5 axes;
this is the reviewer-impersonation pass that should catch what the
narrower axes miss.
```

---

## Synthesis protocol (after all 5 axes return)

When all 5 reports are in hand (Round 1: Axes 1, 2, 5; Round 2: Axes 3, 4), run a synthesis pass:

1. **Aggregate severity counts:** total SEVERE / MAJOR / MODERATE / MINOR findings across all 5 axes. Build a single ranked list.
2. **De-duplicate cross-axis findings:** if Axis 2 and Axis 5 both flag the same overclaim sentence, count it once at the higher severity. If Axis 3 and Axis 5 both flag the same reproducibility gap, count once.
3. **Patch decision per finding:**
   - SEVERE → patch before Day-10 EOD (pre-send-out). If can't patch, escalate to Tianyu Day-7 with the SEVERE finding for ARR-submission-go/no-go call.
   - MAJOR → patch before Day-10 EOD. If can't patch in time, add to Limitations + flag in cover note.
   - MODERATE → patch before Day-14 EOD. Triage if time-constrained.
   - MINOR → best-effort. Document any deferred items in cover note.
4. **Write synthesis report:** `docs/logs/2026-05-1X_parallel-review-synthesis.md`. Includes: severity counts, deduplicated finding list, patch plan with assignees and deadlines, escalation items if any SEVERE remain unpatched.

## Pre-flight checklist before firing Round 1 (Day 12 AM)

By Day-12 morning, the paper should be in the post-content-freeze state. Verify each item before firing the 3 Round-1 sessions:

- [ ] §5.5 Evaluation Metrics rewrite applied (Day 7 task).
- [ ] §6 worked-example numbers replaced with Track D 162 GB / 60× figures (Day 8 task).
- [ ] ACL style switch landed (`acl.sty` swapped for `neurips_2026.sty`; Day 8 task; risk-flagged 25% P-failure per plan v2 §8).
- [ ] Page-budget within ACL 8-page review limit (Day 9 check; cuts applied if over — most likely candidate: §6 Memory Complexity proof block to `\appendix`).
- [ ] §A Reproducibility Appendix applied to canonical TeX (Day 10 task; REVISION marker `reproducibility-appendix-v1-D+`; ZERO undefined-ref warnings expected per Day-5 Item-G prep).
- [ ] Responsible NLP Checklist final version landed (Day 10 task).
- [ ] Co-author send-out completed Day 11 (Tianyu + Max + Alaa received the paper, OpenReview accounts confirmed).
- [ ] At least one Day-11 EOD signal from a co-author received (even just "got it, will read this week") — confirms send-out worked.
- [ ] Canonical TeX smoke-compiles clean post-ACL-switch (warning inventory recalibrated for ACL natbib mode).
- [ ] `logomesh/kv_mcts.py` is at the same state as Day-5 (no code changes Days 6-11 unless Day-5 review found something — none flagged).
- [ ] All Track F/G raw JSONs are present at expected paths.
- [ ] All linked Track reports (A/C/D/F/G) exist and are readable.
- [ ] Two prior commits + Days 6-11 commits all in git history.

If any item fails, the paper isn't yet "structurally complete" — fix before firing. Adversarial reviews on a half-baked state produce noise instead of signal.

## Cover-note language for Day-11 send-out (adversarial review NOT yet run; reference forward to Day-13 synthesis)

The Day-11 send-out happens BEFORE adversarial review fires (Days 12-13). The cover note can flag the adversarial-review plan but cannot claim it's complete:

> Adversarial review plan scheduled Days 12-13 (parallel with your co-author async window): 5 axes — code-semantic correctness, framing defensibility, §A reproducibility, reviewer-prior alignment for #13, standard adversarial pass. Synthesis Day-13 EOD at `docs/logs/2026-05-22_parallel-review-synthesis.md`. Will share findings Day-14 morning so we can integrate your feedback + adversarial findings into Day-14 patches together before Day-15 submission. Please flag if you'd like to weigh in on any specific axis the adversarial review should focus on.

## Cover-note language for Day-14 final synthesis (post-co-author + post-adversarial integration)

By Day 14 the team has both co-author feedback and adversarial findings. The final pre-submission cover note (to Tianyu + Max + Alaa for last sign-off Day 14) includes:

> Day-13 adversarial review synthesis at `docs/logs/2026-05-22_parallel-review-synthesis.md`: [N] SEVERE, [N] MAJOR, [N] MODERATE, [N] MINOR findings. All SEVERE + MAJOR patched today (Day 14). MODERATE patched best-effort; remaining items added to Limitations or flagged as known-and-deferred. Co-author feedback integration: [summarize the patches incorporated]. Final pre-submission state at commit `<hash>`. Submitting Day 15 at the ARR portal.

The two-cover-note pattern (Day 11 forward-looking + Day 14 summary) shows Tianyu/Max/Alaa that the team operates with adversarial discipline as part of the standard workflow — a useful trust signal regardless of what the adversarial review surfaces.

---

## Notes for execution

**Don't fire all 5 axes at once if running solo.** Round 1's three axes (1, 2, 5) are the highest-leverage. If Round 1 surfaces SEVERE findings, the answers from Axes 3 and 4 may need re-scoping after patches. Run sequentially-with-overlap: fire Round 1, synthesize in parallel with Round 1 still running on the slowest axis, then fire Round 2 when Round 1 synthesis is mostly done.

**Axis 1 is the most likely to find load-bearing issues.** Code-vs-paper divergences are SEVERE by definition. Allocate the most attention to its findings.

**Axis 5 is the most likely to find easy wins.** Standard reviewer critiques are well-known patterns. Many SEVEREs and MAJORs will surface there with clear patches.

**Axis 2 + Axis 4 may have overlap on framing.** Cross-check when synthesizing.

**Axis 3 may surface issues that aren't actually in scope.** Some "reproducibility gaps" are legitimately supplementary-OK per ACL ARR rules. Don't over-patch §A — the appendix is unlimited in space but the body isn't.

**If a SEVERE finding lands in code-semantic correctness (Axis 1), escalate to Tianyu directly Day-7 or Day-8.** A code-vs-paper divergence may invalidate a contribution; that's a paper-go/no-go-level decision, not a patch-and-keep-going decision.

---

*End of pre-submission parallel adversarial review plan. **Fire Round 1 Day-12 AM (2026-05-21) per the revised timing table** (postponed 2026-05-14 from original Day-6/7 timing per Josh's directive — adversarial review now operates on the structurally-complete paper, in parallel with the Day-11 co-author send-out async window). Synthesis target: `docs/logs/2026-05-22_parallel-review-synthesis.md` (Day-13 EOD authoring). Estimated total Days-12+13 wall: ~3-4 hours including synthesis. The plan is the response to Josh's expressed need: an adversarial process before ARR reviewers do it, compensating for missing institutional mentorship with structural discipline.*
