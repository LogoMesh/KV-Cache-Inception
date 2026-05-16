# Option A Reframe — Retire "Dimensional Escape" Terminology

**Created:** 2026-05-17 (Day 12 PM)
**Target run date:** 2026-05-18 (Day 13) EOD or 2026-05-19 (Day 14) AM
**Prerequisites (must be complete before firing this prompt):**
1. M1-M5 patches applied (Day-12 EOD; commit hash to be assigned)
2. Round 2 (Axes 3 + 4) fired and synthesized (Day-13 AM/PM)
3. Day-14 cleanup prompt **NOT YET** fired (this reframe must land first)

**Target session:** the same Day-12 parallel-review orchestrator session in worktree `sharp-lalande-f3d5de`, OR a fresh session — both work. Fresh session is cleaner (no carryover context bias).

**Framing decision (Josh, Day-12 EOD):** retire the coined term "Dimensional Escape" entirely. Reposition the paper as evidence for an established phenomenon class — **Goodhart's Law in high-dimensional latent-space search** — with our specific contributions being (a) the reversible KV-cache MCTS algorithm; (b) the cache-space empirical setting; (c) the two-mechanism characterization (prior-amplification at small scale + greedy-collapse at larger scale).

---

## Why this design

- **Discovery before plan before execution** prevents accidental cross-reference breakage.
- **Labels stay stable** (`\label{def:dim_escape}`, `\label{hyp:orthogonal_escape}`, `\label{hyp:goodhart}`, `\label{sec:dim_escape}`, `\label{eq:goodhart}`). The LaTeX cross-reference machinery doesn't care what a label's title text is; it resolves by label name. We can change every visible string while keeping every `\ref{...}` working. **Do not rename the labels.**
- **Section-by-section execution** with smoke-compile gates means any breakage surfaces immediately rather than at the end.
- **Page-budget protection:** after each section's rewrite, verify body ≤ 8pp. Goodhart-anchored framing tends to ADD words (cites to predecessor literature) — must trim elsewhere if needed.

---

## Copy-paste prompt for the session

```
OPTION A REFRAME — retire "Dimensional Escape" coined terminology and
reposition paper as evidence for Goodhart's Law in high-dimensional
latent-space search.

Josh has decided (Day-12 EOD): the audit + adversarial review's
recommendation to anchor DE to Goodhart was too soft. Better to drop
the coined term entirely. The paper's contribution becomes:
  (a) Reversible KV-Cache MCTS algorithm (unchanged)
  (b) Empirical demonstration that Goodhart's Law manifests in
      inference-time KV-cache MCTS with surrogate-reward scoring
      (specific empirical setting, not a new phenomenon class)
  (c) Two-mechanism characterization: prior-amplification at small
      model scale + greedy-collapse at larger scale

PRECONDITIONS (verify before starting):
  - M1-M5 patches applied (commit hash should be on origin/main; check
    git log for M1-M5 commit message)
  - Round 2 (Axes 3 + 4) synthesized; any framing-related Round 2
    findings have been read
  - The Day-14 cleanup prompt has NOT yet fired
  - Body is currently 8pp (per .aux sec:reproducibility = page 9)

INPUTS:
  - Paper TeX: docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex
  - Synthesis: docs/logs/2026-05-17_round-1-synthesis.md (Round 1)
    + docs/logs/2026-05-1?_round-2-synthesis.md (Round 2, if exists)
  - This prompt: docs/logs/2026-05-18_retire-dimensional-escape-prompt.md

OUTPUT TARGETS:
  - Discovery report: docs/logs/<today>_de-reframe-discovery.md
  - Per-location action plan: docs/logs/<today>_de-reframe-plan.md
  - Execution log: docs/logs/<today>_de-reframe-execution.md

----------------------------------------------------------------------
PHASE 0 — DISCOVERY (do not skip; do not start Phase 1 until done)
----------------------------------------------------------------------

(0.1) Grep the canonical TeX for every occurrence of "Dimensional
      Escape" / "dim_escape" / "dim escape" / "dimensional escape":

  grep -n "Dimensional Escape\|dim_escape\|dim escape\|dimensional escape" \
      docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex

  Catalogue each location: line number, surrounding section, current
  rendered text. Some occurrences are in:
    - Section titles (e.g., §3.2 title "Dimensional Escape: A
      Unifying Geometric Pathology")
    - Hypothesis titles (Hypothesis 1 "Monitor-driven Dimensional
      Escape"; Hypothesis 2 "Reward-driven Dimensional Escape")
    - Definition body (def:dim_escape contents)
    - %-comment REVISION markers (NOT rendered; can ignore for
      rendered-prose work, but flag for separate cleanup-prompt cycle)
    - Body prose throughout §1 / §3 / §5 / §7

(0.2) Grep for label usages (these are pure plumbing — keep stable):

  grep -n "def:dim_escape\|hyp:orthogonal_escape\|hyp:goodhart\|sec:dim_escape\|eq:goodhart\|eq:alignment_faking\|eq:orthogonal_escape" \
      docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex

  Catalogue each \label and each \ref usage. These labels MUST remain
  in the final TeX. Their titles change; the labels do not.

(0.3) Grep the bibliography for citation keys that the reframe may
      reposition:

  grep -n "bailey2024obfuscated\|gao2023scaling\|skalse2022defining\|goodhart" \
      docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex

  - bailey2024obfuscated: currently positioned as the "Orthogonal
    Escape predecessor." Under Option A, repositions as "structurally
    parallel work on gradient-trained input-space attacks."
  - gao2023scaling, skalse2022defining: added by M3 if M3 landed.
    Under Option A, these become PRIMARY anchors (not supporting
    citations to a coined term).
  - Verify all three are present in bibliography; if not, plan to add.

(0.4) Read §3.2 in full (current state, post-M1-M5). Specifically:
  - Section title "Dimensional Escape: A Unifying Geometric
    Pathology" (current title under M1-M5 framing)
  - The two-mode-motivation paragraph (§3.2 intro)
  - Definition def:dim_escape (currently titled "Dimensional Escape")
  - Hypothesis 1 (hyp:orthogonal_escape) "Monitor-driven
    Dimensional Escape: Orthogonal Escape..."
  - Hypothesis 2 (hyp:goodhart) "Reward-driven Dimensional Escape:
    Goodhart-collapse..."
  - §3.2 closing paragraph (Bailey 2024 framing)

(0.5) Write discovery report at docs/logs/<today>_de-reframe-discovery.md:

  # Dimensional Escape Reframe — Discovery Report

  **Date:** YYYY-MM-DD
  **Paper commit:** <hash post-M1-M5 + post-Round-2-patches>

  ## DE occurrences in TeX (rendered text)
  | Line | Section | Current text (≤80 chars) | Type | Notes |
  |---|---|---|---|---|

  ## DE occurrences in %-comment markers (not rendered)
  [List for separate cleanup-prompt cycle; do not act on these now]

  ## Labels (preserve as-is)
  | Label | First defined | Referenced from |
  |---|---|---|

  ## Bibliography citations
  | Cite key | Current role | Reframe role |
  |---|---|---|

REPORT TO JOSH after Phase 0 with:
  - Total rendered DE occurrences (typically 20-40)
  - Section count affected (typically 6-8 body sections + appendix)
  - Page-budget impact estimate (will adding Goodhart-anchored prose
    bust 8pp? Pre-estimate based on word count delta)

STOP. Do not proceed to Phase 1 until Josh signs off on Phase 0.

----------------------------------------------------------------------
PHASE 1 — PER-LOCATION ACTION PLAN
----------------------------------------------------------------------

For each rendered DE occurrence from Phase 0.5, propose an action:

  - REPLACE — substitute "Dimensional Escape" with a Goodhart-anchored
    phrase (see Replacement vocabulary below). Preserves the
    surrounding sentence.
  - RELABEL — change the title of a Section / Definition / Hypothesis
    while keeping its \label intact.
  - DELETE — remove the entire claim. Use only if the claim was
    framing-dependent (e.g., "we coined this term") and has no
    Goodhart-flavored equivalent.
  - NO-OP — leave alone. Use for %-comment markers (handled by
    cleanup prompt) or other non-rendered context.

REPLACEMENT VOCABULARY (use consistently; don't mix freely):

  "Dimensional Escape" (as a phenomenon name) →
    "Goodhart's Law in high-dimensional latent-space search"
    (full form, use first mention in each major section)
  OR
    "the cache-space Goodhart failure"
    (shorter form, use after first mention in a section)
  OR
    "this Goodhart-class failure" / "the failure mode"
    (briefest, use in flow prose)

  "a fundamental geometric pathology" → "a Goodhart-class failure
    mode" or "a reward-hacking failure mode characterized below"

  "the Dimensional Escape phenomenon" → "Goodhart's Law as it
    manifests in this setting"

  "monitor-driven Dimensional Escape" → "monitor-driven Orthogonal
    Escape" (Bailey 2024's term; we re-defer to their naming for the
    gradient-trained input-attack case)

  "reward-driven Dimensional Escape" → "reward-driven
    Goodhart-collapse" / "the reward-driven Goodhart failure case"

  "Dimensional Escape extends generic Goodhart's Law by specifying
    the geometric route..." → DELETE THIS (it was the Option B
    novelty defense; Option A drops the novelty claim)

KEY SECTION RELABELINGS:

  §3.2 title:
    "Dimensional Escape: A Unifying Geometric Pathology"
    →
    "Goodhart's Law in High-Dimensional Latent-Space Search"

  Definition (label: def:dim_escape, keep label):
    Title: "Dimensional Escape" → "Goodhart-Class Failure in
    Low-Rank Projected Latent Spaces"
    Body: unchanged structure; the geometric definition stays
    (it characterizes the failure mode mechanically); drop any
    sentence claiming this is a "new phenomenon" or
    "unifying pathology" — those become "instance" / "case" /
    "manifestation" of established Goodhart's Law.

  Hypothesis 1 (label: hyp:orthogonal_escape, keep label):
    Title: "Monitor-driven Dimensional Escape: Orthogonal Escape
    under inference-time KV-cache perturbation"
    →
    "Monitor-driven Orthogonal Escape under inference-time KV-cache
    perturbation"
    Body: drop "specializes Definition def:dim_escape"; reframe as
    "specializes Bailey 2024's Orthogonal Escape construction".

  Hypothesis 2 (label: hyp:goodhart, keep label):
    Title: "Reward-driven Dimensional Escape: Goodhart-collapse
    under surrogate-reward MCTS"
    →
    "Reward-driven Goodhart-collapse under surrogate-reward MCTS"

  §1.1 Contribution 2 title (post-M2):
    Current: "Empirical demonstration of reward-driven Dimensional
    Escape---an instance of Goodhart's Law in high-dimensional
    latent-space search"
    →
    "Empirical demonstration of Goodhart's Law in high-dimensional
    cache-space MCTS, with two-mechanism characterization"

  §7 references: search for "Dimensional Escape" in §7 Discussion /
    §7.1 Implications / §7.3 Limitations / §7.4 Future Work — replace
    with appropriate Goodhart-anchored phrasing per Replacement
    Vocabulary.

Write the per-location action plan to docs/logs/<today>_de-reframe-plan.md:

  # Dimensional Escape Reframe — Per-Location Action Plan

  **Date:** YYYY-MM-DD

  ## Section-level relabelings
  [§3.2, Definition, Hyp 1, Hyp 2, §1.1 C2 title, etc.]

  ## Prose-level replacements (line-by-line)
  | Line | Old text | New text | Action |
  |---|---|---|---|

  ## Bibliography updates
  [If gao2023scaling / skalse2022defining not yet present, plan to add.]

  ## Page-budget impact estimate
  [Word count delta; expected page-budget change]

REPORT TO JOSH after Phase 1 with:
  - Action count by type (REPLACE/RELABEL/DELETE/NO-OP)
  - Top 5 highest-impact replacements (sections most affected)
  - Page-budget delta estimate
  - Ask Josh to approve the plan before Phase 2.

STOP. Do not proceed to Phase 2 until Josh signs off on the plan.

----------------------------------------------------------------------
PHASE 2 — EXECUTION (section by section, smoke-compile after each)
----------------------------------------------------------------------

Apply the approved plan in this section order (highest-risk section
first; smoke-compile after each):

  Batch 1: §3.2 (the core reframe location)
    - Section title change
    - §3.2 intro paragraph rewrite (drop "we name DE", reframe as
      "the failure mode we characterize is an instance of Goodhart's
      Law in high-D latent spaces, building on Bailey 2024")
    - Definition title change + body rewrite
    - Hypothesis 1 title change + closing rewrite
    - Hypothesis 2 title change
    - §3.2 closing paragraph (Bailey 2024 framing)
    - SMOKE-COMPILE. Verify body ≤ 8pp. Verify 0 undefined refs.
      If page slips: STOP and report to Josh.

  Batch 2: Abstract (line ~56) + §1 Intro paragraph 4 (line ~71)
    - Replace remaining "Dimensional Escape" prose
    - Verify abstract doesn't bust word limit (200 words)
    - SMOKE-COMPILE. Verify body ≤ 8pp.

  Batch 3: §1.1 Contribution 2
    - Title rewrite (per Replacement Vocabulary)
    - Body prose: replace "Dimensional Escape" references; keep
      "reward-driven failure" framing; keep Bailey 2024 reference
    - SMOKE-COMPILE.

  Batch 4: §5 Experiments
    - §5 Exp 1 mechanism + interpretation paragraphs (replace DE
      references in prose)
    - §5 Exp 2 closing paragraph (replace DE references)
    - SMOKE-COMPILE.

  Batch 5: §7 Discussion + §7.1 + §7.3 + §7.4
    - Replace remaining DE prose references
    - SMOKE-COMPILE.

  Batch 6: Appendix sections (if any DE references remain)
    - Should be few; mostly handled by %-comment cleanup (defer to
      cleanup prompt)
    - SMOKE-COMPILE.

  Batch 7: Bibliography
    - Verify gao2023scaling + skalse2022defining present (added by M3)
    - If missing, add per M3 prompt format
    - Any other citation additions needed (e.g., goodhart 1984)?
      Decide based on Phase 1 plan
    - SMOKE-COMPILE.

After all batches:
  - FULL SMOKE-COMPILE (2 passes)
  - Verify body still ≤ 8pp (sec:reproducibility = page 9)
  - Verify 0 undefined refs
  - Verify no orphaned section/eq numbers (no [??] in compiled PDF)
  - grep the TeX one more time for "Dimensional Escape" — should now
    only appear in %-comment markers (which we left alone for the
    cleanup prompt). If any rendered occurrences remain: STOP and
    surface to Josh.

Write execution log to docs/logs/<today>_de-reframe-execution.md:

  # Dimensional Escape Reframe — Execution Log

  ## Batch 1 (§3.2)
  - Title changed: <old> → <new>
  - Smoke-compile: PASS / FAIL
  - Page count after: <N>pp body
  ...
  [Per batch]

  ## Final state
  - Net diff: <insertions/deletions>
  - PDF size: <KB>
  - Body page count: <N>pp
  - 0 undefined refs: YES/NO
  - Remaining "Dimensional Escape" rendered occurrences: <count>

----------------------------------------------------------------------
PHASE 3 — COMMIT + PUSH
----------------------------------------------------------------------

Commit message template:

  docs: Option A reframe — retire "Dimensional Escape" coined terminology

  Per Josh's decision (Day-12 EOD): retire the coined term entirely.
  Reposition paper as evidence for Goodhart's Law in high-dimensional
  latent-space search — a specific empirical setting (inference-time
  KV-cache MCTS with surrogate-reward scoring) rather than a novel
  phenomenon class.

  Rationale: the audit (Axis 2 Q5) flagged "you're just relabeling X"
  as a high-likelihood reviewer attack on the coined term. Day-12 M3
  patch attempted Option B (anchor DE to Goodhart explicitly). Josh's
  Day-12 EOD framing review concluded Option B was still too soft;
  Option A (retire entirely) is more defensible because it eliminates
  the coined-term defense overhead and lets the empirical findings
  stand on their own merits.

  Changes:
  - §3.2 title: "Dimensional Escape: A Unifying Geometric Pathology"
    → "Goodhart's Law in High-Dimensional Latent-Space Search"
  - Definition def:dim_escape (label preserved): retitled
    "Goodhart-Class Failure in Low-Rank Projected Latent Spaces";
    body reframed to drop novelty-of-phenomenon claim
  - Hypothesis 1 hyp:orthogonal_escape (label preserved): retitled
    "Monitor-driven Orthogonal Escape under inference-time KV-cache
    perturbation" (defers to Bailey 2024's terminology)
  - Hypothesis 2 hyp:goodhart (label preserved): retitled
    "Reward-driven Goodhart-collapse under surrogate-reward MCTS"
  - §1.1 Contribution 2 (title + body): reframed around Goodhart's
    Law in high-dimensional cache-space search; two-mechanism
    characterization (prior-amplification + greedy-collapse) becomes
    the specific contribution
  - Abstract: drops "Dimensional Escape" terminology; reframed
    around Goodhart-class failure
  - §1 Intro paragraph 4: drops DE terminology
  - §5 + §7: prose-level replacements per Replacement Vocabulary
  - Bibliography: relies on M3's gao2023scaling +
    skalse2022defining as primary anchors

  Labels preserved: def:dim_escape, hyp:orthogonal_escape,
  hyp:goodhart, sec:dim_escape, eq:goodhart, eq:alignment_faking,
  eq:orthogonal_escape. All cross-references resolve.

  Verification: <fill in: PDF size, page count, undefined refs, etc.>

  M1-M5 (Day-12 EOD) status: M1, M4, M5 survive unchanged. M2 is
  overwritten by this reframe (C2 title rewritten). M3 is overwritten
  (Goodhart-anchor preamble retired; Goodhart is now load-bearing
  rather than supporting). gao2023scaling + skalse2022defining
  bibitems persist as primary anchors.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

Push to origin/main:
  git push origin HEAD:main

REPORT BACK TO JOSH:
  - Commit hash
  - Push confirmation
  - Body page count (must be ≤ 8pp)
  - PDF size
  - Net insertions/deletions
  - Any deviations from plan
  - Recommendation: fire the repo cleanup prompt next (no longer
    blocked on framing)

----------------------------------------------------------------------
HARD RULES
----------------------------------------------------------------------

1. DISCOVERY BEFORE PLAN BEFORE EXECUTION. No skipping. If Phase 0
   reveals more DE occurrences than expected (>40 rendered), surface
   to Josh before Phase 1.

2. LABELS STAY STABLE. Never rename a \label. Cross-references resolve
   by label name, not by title text. Renaming a label would require
   updating every \ref{...} usage and is a much larger change.

3. SMOKE-COMPILE AFTER EVERY BATCH. If page-budget slips to 9pp at
   any point: STOP. Recover by reverting the last batch and surfacing
   to Josh — page-budget is the load-bearing constraint.

4. PRESERVE eq:goodhart EQUATION INTACT. The mathematical statement
   of Hypothesis 2 (entropy reward Eq. goodhart) is the empirical
   anchor for the paper's claim. Don't reformulate the math.

5. DON'T TOUCH THE SCIENTIFIC CLAIMS. Track F's p=1.0 / 185-of-200
   prediction / mean-perplexity numbers stay as-is. Reframe is
   TERMINOLOGY-ONLY; the data + mechanism characterization are
   unchanged.

6. CHECK §7.5 BROADER IMPACT + §7.6 ETHICAL CONSIDERATIONS. These
   reference DE; replacements should preserve the safety-relevant
   framing (white-box infrastructure for diagnostic vs adversarial
   tool).

7. DON'T COMMIT %-COMMENT MARKER REWRITES. Markers contain DE
   references but they're non-rendered (LaTeX comments). Leave them
   alone for this reframe; the cleanup prompt will handle them
   separately. Reframe is RENDERED-PROSE-ONLY.

8. PUSH ONLY IF SMOKE-COMPILE PASSES. Per CLAUDE.md.

----------------------------------------------------------------------
ANTI-OVERLOAD GUIDANCE
----------------------------------------------------------------------

- Phase 0 should take ~30 min (grep + catalog).
- Phase 1 should take ~30-45 min (per-location plan).
- Phase 2 (execution) is the longest: ~2-3 hours wall (section by
  section with compile gates).
- Phase 3 commit + push: ~10 min.
- Total: 3-4 hours wall.

- If at any point you find yourself rationalizing ("this section
  doesn't really need to change..."), STOP and read the Replacement
  Vocabulary again. Be consistent — partial reframes are worse than
  no reframe.

- If you accidentally rename a label or break a cross-ref, smoke-
  compile will catch it. Revert the broken batch; surface to Josh.

----------------------------------------------------------------------
BEGIN
----------------------------------------------------------------------

Start Phase 0 (Discovery). Open
docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex. Grep for
"Dimensional Escape" and label/cite patterns. Build the discovery
report at docs/logs/<today>_de-reframe-discovery.md. After Phase 0
complete, REPORT TO JOSH and WAIT.
```

---

## What this prompt deliberately does NOT do

- **Does not touch %-comment REVISION markers.** Those contain DE references but they're invisible in compiled PDF. Cleanup prompt handles them.
- **Does not change scientific claims.** Numbers, mechanism characterization, p-values, statistical tests all preserved.
- **Does not rename labels.** Labels are plumbing — keep them stable to preserve cross-references.
- **Does not coordinate with co-author feedback.** If Tianyu/Max/Alaa send framing comments during the reframe window, defer integration until after the reframe lands.
- **Does not fire the cleanup prompt.** Cleanup is the NEXT step after this; reframe must land first.

## What to do if Round 2 (Axis 4 reviewer-prior alignment) recommends something different

If Axis 4's Day-13 report recommends a framing direction OTHER than Option A — for example, "the paper should lean harder INTO Interpretability-track infrastructure framing rather than Goodhart framing" — pause this prompt and synthesize Axis 4's recommendation with Option A before firing. Don't apply Option A blindly if Axis 4 surfaces incompatible framing direction.

## Calendar reminder

```
Day 12 EOD (TODAY) — M1-M5 fired (in progress)
Day 13 AM         — Round 2 (Axes 3+4) fires
Day 13 PM         — Round 2 synthesis
Day 13 EOD        — Option A reframe (this prompt fires)
Day 14 AM         — Repo cleanup prompt fires
Day 14 PM         — Co-author feedback integration + final smoke-compile
Day 15            — ARR portal submission
```

The reframe slots between Round 2 synthesis and the cleanup prompt. ~3-4 hours wall. Day-14 AM cleanup operates on the reframed paper.
