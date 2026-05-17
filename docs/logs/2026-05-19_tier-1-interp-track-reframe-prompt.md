# Tier-1 Interp-Track Reframe Prompt — Day-14 First Priority

**Created:** 2026-05-17 (Day 12 EOD).
**Target run date:** 2026-05-19 (Day 14) AM — FIRST thing Day-14, before cleanup prompt fires.
**Replaces:** The retire-DE prompt at `docs/logs/2026-05-18_retire-dimensional-escape-prompt.md` (Josh reversed Option A decision after Round 2 synthesis surfaced conflicting Axis 4 Tier-1 recommendation).
**Decision basis:** Synthesis doc `docs/logs/2026-05-22_parallel-review-synthesis.md` Tier-1 patches E+F+G+H+I (the coordinated title + abstract + §1.1 interp-track reframe; per synthesis line 171, routing risk is "HIGH if not patched" which beats the Axis 2 Q5 coined-term concern). Current paper state (post-`fa2a36f` Day-12 EOD) already anchors DE to Goodhart per M2+M3 — the Option B compromise. This prompt adds Tier-1 probing framing ON TOP of the M2+M3 anchor. DE survives as the phenomenon being probed; Goodhart anchor in §3.2 defends against relabeling attack.

---

## Why this prompt over retire-DE

- **Axis 4 routing concern is higher-severity than Axis 2 coined-term concern** per synthesis's own risk-assessment ("HIGH if not patched").
- **Option B anchor (DE = instance of Goodhart's Law) already landed at M3.** The relabeling attack is already partially defended. Layering Tier-1 probing framing on top is additive, not contradictory.
- **Path of least disruption:** ~80 words net rewrite vs. ~3-5 hour Option A reframe.
- **Aligns Axis 2 + Axis 4 instead of choosing.** The current paper says DE is an instance of Goodhart's Law (Axis 2); the reframe makes DE the thing-being-probed (Axis 4). Both signals coexist.

---

## Copy-paste prompt for the session

```
TIER-1 INTERP-TRACK REFRAME — apply synthesis Tier-1 patches E+F+G+H+I
in a single coordinated rewrite of title + abstract + §1.1 + §1.

Goal: reroute the paper toward the Interpretability track at ACL routing
by leading with the "Probing X" structure that Axis 4 (reviewer-prior
alignment) showed is the dominant successful-paper pattern at this track.
Current title leads with the algorithm; reverse the colon-ordering so the
phenomenon-being-probed leads. Abstract currently spends ~40% of its
words on algorithm+formalism before introducing Dimensional Escape; reorder
so the geometric question + DE arrive first.

Per synthesis (`docs/logs/2026-05-22_parallel-review-synthesis.md` line
124 + 171): Tier-1 routing concern is HIGH if not patched. ACL routing
may otherwise assign to ML/Efficient-Methods reviewers, who will read
the paper as a Methods submission and reject for "not novel enough on
the algorithm side."

PRECONDITIONS (verify before starting):
  - M1-M5 patches applied (Day-12 EOD, commit `934a288` on origin/main)
  - SEVERE §A system-prompt mismatch fix applied (commit `bb83ddb`)
  - Synthesis doc complete (commit `fa2a36f`)
  - Current paper state: body 8pp, PDF 14pp/467KB, 0 undefined refs

INPUTS:
  - Paper TeX: `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`
  - Synthesis: `docs/logs/2026-05-22_parallel-review-synthesis.md`
    (read Tier-1 patches E/F/G/H/I before starting)

OUTPUT: edited TeX, smoke-compile clean, push to origin/main.

NET BUDGET: ~80 words net rewrite per synthesis (rewrites, not additions).
Body must stay ≤ 8pp; abstract must stay ≤ 200 words.

----------------------------------------------------------------------
PATCH E — Title colon-ordering reverse (Axis 4 Q1)
----------------------------------------------------------------------

Locate (TeX ~line 42):
  \title{Reversible KV-Cache MCTS: A Memory-Bounded Probe of Dimensional Escape in Frozen Language Models}

Replace with one of these (pick the cleanest; default to the first
unless it reads awkwardly):

  Option E.1 (DEFAULT — closest to synthesis recommendation):
    \title{Probing Dimensional Escape in Frozen Language Models via
    Reversible KV-Cache MCTS}

  Option E.2 (algorithm-name preserved as colon-suffix):
    \title{Probing Dimensional Escape in Frozen Language Models:
    A Reversible KV-Cache MCTS Approach}

  Option E.3 (memory-bound preserved as adjective):
    \title{Probing Dimensional Escape via Memory-Bounded Reversible
    KV-Cache MCTS in Frozen Language Models}

Rationale: leads with "Probing X" — the dominant successful-paper title
pattern at the Interpretability track per the 26-paper sample in
`docs/logs/gemini research reports/Analyzing Interpretability Track
Submissions.md`. The phenomenon (Dimensional Escape) is the topic
being investigated; the algorithm is the methodology.

Verify after applying: title fits on 1-2 lines in compiled PDF; doesn't
clobber the abstract / first-page layout.

----------------------------------------------------------------------
PATCH F — Abstract sentence-order rewrite (Axis 4 Q2 + Axis 5 Pass F)
----------------------------------------------------------------------

Current abstract structure (S1-S2 = ~40% algorithm/formalism):
  S1: "We introduce Reversible Monte Carlo Tree Search..." (algorithm)
  S2: "We prove FP32 accumulator..." (theorem + Proposition 1)
  S3: "Using this framework, we... expose a geometric pathology
       we term Dimensional Escape..."
  ...

Replace with this rough sequence (preserve numerical facts; rewrite
prose flow):
  S1 NEW: "We probe how inference-time tree search behaves in the
    continuous latent state of a frozen Transformer's KV cache when
    guided by a low-rank scoring projection."
  S2 NEW: "We characterize a geometric pathology — Dimensional Escape
    (an instance of Goodhart's Law in high-dimensional latent-space
    search) — under which surrogate-reward optimization in cache
    space succumbs to prior-amplification or greedy-collapse rather
    than satisfying the semantic objective."
  S3 NEW: "Our probe is Reversible KV-Cache MCTS, an inference-time
    algorithm that mutates a frozen Transformer's cached key/value
    tensors in place via FP32 accumulators and rolls back exact
    baseline state after each tree expansion."
  S4 NEW: "We prove zero-drift reversibility (Theorem~\ref{thm:reversibility};
    validated to machine precision over 200 apply-revert cycles on Llama
    3.2-1B and Llama 3.2-3B) and that the working-set memory complexity
    is O(M_KV + d · K_acc) rather than the O(b^d · M_KV) of naive
    parallel-cache MCTS---reducing the memory floor from exponential
    in b^d to additive in d (Proposition~\ref{prop:memory})."
  S5-S6: Track F + Track G empirical content (preserve current
    statistical claims verbatim — the numerical content stays).
  S7 NEW: "We argue that the reward-driven failure surfaced here is
    structurally analogous to the monitor-driven Orthogonal Escape
    previously hypothesized for alignment faking [bailey2024obfuscated];
    we conjecture both as sub-cases of the same overparameterized-
    latent-space geometry." [PRESERVE Day-11 hedge wording]
  S8 NEW: "Reversible MCTS is a high-precision interpretability probe
    of frozen-model representational geometry: a controlled-access
    diagnostic tool for studying knowledge-tracing in latent space
    rather than a reasoning-improvement method. Artifacts are
    released with a Croissant 1.1 metadata package."

KEY CHANGES from current abstract:
  - S1: NEW probing-framing opener (~30 words). Saturates "probe" /
    "guided by a low-rank scoring projection" / "continuous latent
    state" — interp keywords.
  - S2: phenomenon (Dimensional Escape) moved EARLY; Goodhart anchor
    preserved from M4. Routing signal lands in S2 not S3.
  - S3: algorithm description re-positions as "our probe is..."
    (subordinates algorithm to the probing methodology).
  - S4: Theorem + Proposition 1 + asymptotic claim, similar to
    current. Light edit to flow.
  - S5-S6: experimental content unchanged.
  - S7: M4-hedged unification claim preserved verbatim.
  - S8: NEW closing emphasizes "interpretability probe" +
    "knowledge-tracing in latent space" (saturates ARR keywords
    "interpretability" + "knowledge tracing").

Net delta: ~80 words rewrite. Abstract word count must stay ≤ 200.

Verify after applying: abstract reads as interp-track submission
opening (probing framing, phenomenon early, algorithm as methodology
for probing).

----------------------------------------------------------------------
PATCH G — §1.1 Contribution 1 infrastructure-generator framing (Axis 4 Q3)
----------------------------------------------------------------------

Current §1.1 C1 (TeX ~line 79):
  "\item \textbf{Reversible Monte Carlo Tree Search in KV-cache latent
  space.} We introduce an inference-time algorithm that mutates a frozen
  Transformer's cached key/value tensors via in-place arithmetic..."

Rewrite the SECOND sentence (after the algorithm intro) to add
infrastructure-generator framing:

Replace (after "...standard parallel-cache MCTS." which is the current
end of the asymptotic-claim sentence) — add this sentence at the end of
the C1 body, before the closing "The algorithm is gradient-free..."
sentence:

  "We position this algorithm as reusable interpretability infrastructure:
  a memory-bounded, exact-rollback probe for studying continuous-latent
  search behavior on any frozen Transformer, applicable to monitor-driven
  and reward-driven scoring projections beyond the entropy reward we test
  in §\ref{sec:exp1}."

Rationale: per the synthesis-report's 26-paper sample, the most-
influential interpretability papers are positioned as
"infrastructure-and-framework generators" (Mosbach et al. 2024 cited
in the Gemini report). The current C1 reads as a Methods paper. Adding
this sentence explicitly positions the algorithm as INFRASTRUCTURE for
the broader research community, increasing interpretability-track
routing confidence.

Net delta: ~40 words added.

----------------------------------------------------------------------
PATCH H — ARR keyword saturation in abstract (Axis 4 Q4)
----------------------------------------------------------------------

Verify the rewritten abstract (post-Patch F) contains explicit usage
of ARR Interpretability-track keywords:

  - "probing" / "probe": should appear 2-3 times (S1, S3, S8 in Patch F)
  - "interpretability": should appear 1-2 times (S8 in Patch F)
  - "knowledge tracing" / "knowledge-tracing": should appear 1 time
    (S8 in Patch F)
  - "frozen language models" / "frozen Transformer": appears multiple
    times naturally; preserve
  - "latent space" / "latent-space": appears multiple times naturally

If the rewritten abstract has fewer than 2 occurrences of "probing",
1 occurrence of "interpretability", and 1 occurrence of "knowledge
tracing", append/edit to ensure the minimum keyword density.

Per Axis 4 Q4, keyword saturation is what the ACL routing system uses
to route the paper to the correct reviewer pool. Saturation under-
specification → routing miss.

Net delta: 0 if Patch F is applied correctly with the suggested S1 +
S8 wording; else minor adjustments.

----------------------------------------------------------------------
PATCH I — Promote cross-scale MCQ direction-flip to §1 (Axis 4 finding)
----------------------------------------------------------------------

Currently the cross-scale MCQ direction-flip ("at 1B C2 heavy steering
α=0.89, at 3B C2 minimal steering α=0.22, magnitude of flip Δ≈0.67")
is reported in §5 Exp 2 finding (i). The synthesis recommends promoting
this to §1 as headline empirical content for interp-track signal
(scale-conditional mechanism = interp-relevant finding).

Locate (TeX ~line 71, §1 paragraph 4 closing):
  Current §1 paragraph 4 closes with: "...most usefully positioned not
  as a reasoning-improvement tool, but as a high-precision diagnostic
  probe of frozen-model representational geometry---one that empirically
  exposes the reward-driven sub-case of Dimensional Escape, with the
  monitor-driven sub-case left to Phase~B research."

Insert before that closing sentence:

  "Our Latent Cartography sweep (Experiment~2) further reveals a scale-
  conditional mechanism: at the 1B scale, multiple-choice classes
  exhibit heavy-steering search behavior ($\bar{\alpha} = 0.89, 0.83$)
  that amplifies the model's first-letter prior; at the 3B scale, the
  pattern inverts to minimal-steering greedy preservation ($\bar{\alpha}
  = 0.22, 0.28$), with cross-scale flip magnitude $\Delta \approx 0.67$
  on ARC-Easy and $0.55$ on TruthfulQA. Goodhart in cache geometry is
  class-conditional, with the route through latent space determined by
  the model's class-specific prior structure (§\ref{sec:exp2})."

Rationale: surfaces the most-novel positive empirical observation
(the cross-scale direction-flip) to §1 where reviewers form first
impressions. Currently this is buried in §5 finding (i). Promoting
it makes the §1 read "mechanism diagnosis at multiple scales" which
is the dominant successful-paper pattern at Interp track.

Net delta: ~80 words added to §1.

----------------------------------------------------------------------
TOTAL NET BUDGET CHECK
----------------------------------------------------------------------

Estimated combined net delta:
  - Patch E (title): 0 (rewrite)
  - Patch F (abstract): ~80 words rewrite, ~0 net (length-preserving)
  - Patch G (§1.1 C1): +40 words
  - Patch H (keywords): 0 if Patch F includes saturation
  - Patch I (§1 promotion): +80 words

Total: ~+120 words. Approx +0.15-0.20pp body. Current body is 8pp
EXACTLY. Risk: post-patch body slips to 9pp.

MITIGATION if body slips:
  - Apply Tier-2 best-effort trim from Tier-1 budget: §5 Exp 2 finding
    (i) now duplicates the §1 promotion in Patch I; trim §5 finding
    (i) by ~80 words (move the direction-flip prose to §1 entirely;
    leave §5 finding (i) as a one-sentence reference back to §1).
  - If still over: defer Patch G's infrastructure-generator sentence
    (~40 words; lowest-leverage Tier-1 patch per synthesis).

SMOKE-COMPILE GATE:
  - After Patch E only: verify title compiles + first-page layout OK
  - After Patch F only: verify abstract word count ≤ 200
  - After Patches E+F+G+I (all body-affecting): verify body ≤ 8pp
    (sec:reproducibility = page 9 in .aux)
  - If body = 9pp: apply mitigation above

----------------------------------------------------------------------
WORKFLOW (section by section, smoke-compile after each batch)
----------------------------------------------------------------------

  Batch 1: Patch E (title)
    - Apply title change
    - Smoke-compile
    - Verify first-page layout
    - Verify body ≤ 8pp

  Batch 2: Patch F (abstract) + Patch H (keyword check)
    - Apply abstract rewrite (S1-S8 sequence)
    - Apply keyword check; minor edits if saturation under-specified
    - Smoke-compile
    - Verify abstract word count ≤ 200
    - Verify body ≤ 8pp

  Batch 3: Patch G (§1.1 C1 infrastructure framing)
    - Insert sentence
    - Smoke-compile
    - Verify body ≤ 8pp

  Batch 4: Patch I (§1 promotion of cross-scale direction-flip)
    - Insert paragraph
    - Smoke-compile
    - Verify body ≤ 8pp (HIGHEST risk batch — largest word add)
    - If body slips to 9pp: apply mitigation per Total Budget Check

  Batch 5: Cross-check final state
    - Body ≤ 8pp
    - PDF compiles clean (0 undefined refs, 0 LaTeX errors)
    - Keyword density per Patch H
    - Cross-refs all resolve (especially §5 Exp 2 → §1 link if
      Patch I mitigation was applied)

----------------------------------------------------------------------
COMMIT MESSAGE TEMPLATE
----------------------------------------------------------------------

  docs: Tier-1 interp-track reframe (Day-14 first priority)

  Apply synthesis Tier-1 patches E+F+G+H+I from
  docs/logs/2026-05-22_parallel-review-synthesis.md. Highest-leverage
  Day-14 patch per synthesis line 171 ("Risk 1: HIGH if not patched");
  routes paper to Interpretability track via title + abstract + §1.1
  reframe rather than ML/Efficient-Methods reading.

  Patches:
    E — Title colon-ordering reversed
    F — Abstract reordered (S1-S8): probing leads (S1); DE moved to S2
    G — §1.1 C1 infrastructure-generator framing added
    H — ARR keyword saturation verified (probing/interpretability/knowledge tracing)
    I — Cross-scale MCQ direction-flip promoted from §5 to §1

  Body remains 8pp; mitigation applied if needed (§5 (i) trim).

  Option A (retire DE entirely) was considered Day-12 EOD but reversed
  Day-13 AM after recognizing conflict with Axis 4 Tier-1: retiring DE
  removes the "Probing X" routing signal Axis 4 requires. Current
  Option B compromise (DE = instance of Goodhart's Law per M3) defends
  against Axis 2 Q5's "relabeling" attack while preserving probing
  framing Axis 4 requires.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

Push to origin/main: git push origin HEAD:main

REPORT BACK TO JOSH:
  - Commit hash + push confirmation
  - Body page count (must be ≤ 8pp)
  - Abstract word count (must be ≤ 200)
  - Net diff stats
  - Any mitigation applied (e.g., §5 (i) trim for body-budget)
  - Recommendation: fire repo cleanup prompt next (Day-14 PM)

----------------------------------------------------------------------
HARD RULES
----------------------------------------------------------------------

1. PRESERVE M2/M3/M4 GOODHART ANCHOR. The current paper says DE is
   "an instance of Goodhart's Law in high-dimensional latent-space
   search" per M3 preamble. The Tier-1 reframe makes DE the
   thing-being-probed but DOES NOT undo the Goodhart anchor.

2. PRESERVE M1 §3.2 LINE 119 HEDGE.

3. PRESERVE M5 §5 ENUMERATION-VS-MCTS SCOPE CLARIFIER.

4. PRESERVE SCIENTIFIC CONTENT. Numbers (n=200, p=1.0, Δ=±0.5pp,
   |r|≈0.60, etc.) stay verbatim.

5. STAY UNDER 200 WORDS in abstract.

6. STAY UNDER 8PP body. Compile-gate after every batch.

7. DON'T COMMIT MID-BATCH. Apply all Tier-1 patches as one commit.

8. PUSH AFTER ALL PATCHES + COMPILE PASS.

----------------------------------------------------------------------
ANTI-OVERLOAD GUIDANCE
----------------------------------------------------------------------

Estimated total wall: ~90 min.

If Patch F (abstract) takes >30 min, you're overworking it. The S1-S8
structure provided is good enough; minor polish can wait for Day-14
final review.

----------------------------------------------------------------------
BEGIN
----------------------------------------------------------------------

Start with Patch E (title). After each batch, smoke-compile and
report progress. If page-budget slips at any batch: STOP and apply
mitigation per Total Net Budget Check above.
```

---

## What changes vs the retire-DE prompt

- **Preserves "Dimensional Escape" as a term.** DE survives as the phenomenon being probed. M3's Goodhart anchor in §3.2 stays load-bearing; DE remains framed as "an instance of Goodhart's Law in high-dimensional latent-space search." Both Axis 2 + Axis 4 attack vectors defended.
- **No wholesale §3.2 rewrite.** Section title, Definition title, Hypothesis 1/2 titles all stay. The reframe targets title + abstract + §1.1 + §1 only.
- **Estimated 90 min wall** vs the retire-DE prompt's 3-4 hours.
- **Net page-budget impact ~+120 words gross, ~0 net** after §5 (i) mitigation. Body stays 8pp.

## Calendar (updated)

```
Day 12 EOD (TODAY)      — M1-M5 + Round 2 + SEVERE fix + synthesis (DONE, commit fa2a36f)
Day 13 (2026-05-18)     — BUFFER: co-author replies + light Tier-3 patches
Day 14 AM (2026-05-19)  — Tier-1 reframe prompt fires (THIS prompt)
Day 14 PM (2026-05-19)  — Repo cleanup prompt fires
Day 14 EOD              — Final smoke-compile + REVISION-marker audit
Day 15 (2026-05-20)     — ARR portal submission
```
