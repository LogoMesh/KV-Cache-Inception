# Day 9 Session Prompt — Page-Budget Cut Plan (Option A)

> **Context for Josh:** the Day-8 working session surfaced that the body is ~16-17 pages in ACL two-column rendering, against an 8-page review limit — a ~8-9 page overage rather than the ~1-2 pages plan v2 §8 anticipated. Plan v2's pre-planned single cut ("§6 Memory Complexity full proof moves to Appendix A") saves only ~1 page; we need ~8x that. This prompt invokes a fresh Opus 4.7 session to author a structured cut plan (Option A from the Day-8-evening decision menu): read full TeX, identify cuts, estimate savings per cut, rank by risk, produce a single ranked recommendation Josh signs off on at a cumulative-savings line.
>
> **How to use:** paste the block below into a fresh Opus 4.7 session. The block is self-contained; the model has no prior context. Memory auto-loads (`MEMORY.md` + indexed pointers) so the orientation phase narrows to artifacts specific to the cut-plan task.
>
> **Output:** the session produces `docs/logs/2026-05-15_page-budget-cut-plan-v1.md` with a ranked cut list + cumulative-savings column + sign-off ask. The session does NOT apply any cut — application is a separate Day-10 session after Josh sign-off.

---

## Paste everything below this line into the fresh Opus 4.7 session

```
You are starting a single-task Day-9 Session A audit-and-plan for the
LogoMesh / KV-Cache-Inception NeurIPS-via-EMNLP paper. The current
working session (Day 8 EOD) discovered the paper is ~16-17 body pages
against an 8-page ACL review limit. Plan v2's pre-planned cuts were
sized for ~1-page overage; we need ~8-9 pages of cuts. Your job is to
author a structured cut plan that gets the body to ≤8 pages while
preserving the contributions, framing, and empirical pillars. You will
NOT apply any cut — application is Day 10, a separate session, after
Josh sign-off on your plan.

This is a focused single-task session. Do NOT pull additional work
forward; do NOT apply cuts; do NOT modify the canonical TeX. Read,
audit, write the plan, stop.

PHASE 1 — REQUIRED READING (in this order; all in full unless noted)

  1. CLAUDE.md (project root). Standard project orientation including
       the hard rules + paper-to-code verification discipline + dev
       commands. Do not skip — sets the operating context.
  2. memory/MEMORY.md (index — read all line entries).
  3. memory/recast_action_plan.md — strategic state pointer. The first
       paragraph names the Day-8 EOD state explicitly with 7 commits
       cumulative.
  4. memory/feedback.md — full read. Five protocol rules R1-R5 are
       load-bearing for paper claims and primary-source fidelity. R4
       (verify URLs/paths/refs/labels from primary sources before
       action) applies to your page-count claims — every page-savings
       estimate must trace to .aux ToC data or direct TeX inspection,
       not memory.
  5. memory/project_context.md — project context including the EMNLP
       track-selection lock (track #13 "Interpretability and Analysis
       of Models for NLP") and the 8-page ACL review budget.
  6. docs/logs/2026-05-10_14-day-execution-plan.md — plan v2 §2
       (day-by-day matrix), §6 (J-item list with J6c track lock + J1
       page budget), §8 (risk register including the page-overage
       plasticity row), §9 (verification gates including G-smoke-
       compile + G-style), §10 (decision-corpus annex). The plan
       expected "irreducibly > 9 pages" as the YELLOW threshold; we
       are at ~16-17 which is well beyond that threshold's anticipated
       severity.
  7. docs/logs/2026-05-15_session-log.md — full read. Covers Day 6 (§A
       Reproducibility Appendix application) + Day 7 (§5.5 Evaluation
       Metrics rewrite per audit Finding §5-6) + Day 8 (ACL style
       switch + §6 worked-example numbers + §6 honest-framing
       refinement). The "Day 8 evening" addendum at the end records
       Pass K addition to the parallel-review plan and the §6
       projection-vs-measurement clarification. This is the most
       recent state.
  8. docs/logs/2026-05-14_parallel-review-plan.md — read Axis 5 Pass K
       in particular (Day-12 review evaluates speculation-vs-page-
       budget tradeoff). Your cut plan should align with the Pass K
       scrutiny items: §6 20B-scale numbers, §7.5 Future Work
       anti-Goodhart list, §A.5 URL placeholder. If you recommend
       cutting these, Pass K becomes partly preempted.
  9. docs/logs/2026-05-11_stale-sentence-audit.md — read the Summary
       section (line 304+) for the historical 16-finding context. Most
       findings were applied Days 3-7; the audit's "page-budget delta"
       projections may inform your cut sizing.
  10. docs/NeurIPS/acl_formatting.md — read the "Paper Length" section
       (lines 14-23). Verbatim ACL rules: "Review versions of long
       papers may have up to eight (8) pages of content plus unlimited
       pages for references." Appendices are unlimited at review
       (line 23). This is the binding constraint.
  11. docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex — full
       read (776 lines post-`76d63f1`). This is the artifact you are
       auditing.
  12. docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.aux — extract
       all `\@writefile{toc}` entries with grep. The page-number anchor
       per section/subsection is your primary source for current page
       distribution. Cross-check against the rendered PDF if needed
       (PDF at `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.pdf`,
       use Read tool with `pages` parameter for spot checks).

After reading, do NOT start audit work yet. Tell Josh in ONE short
paragraph (under 150 words):

  - That you have read the Phase-1 artifacts and confirm the page-
    overage situation: current body pages, target body pages, page-
    cut delta needed.
  - Two anchors verified: §A starts at page X (from .aux); §1 ends at
    page Y; therefore body occupies pages 1-Y, ~Y pages total.
  - Your readiness to enter Phase 2 (section-by-section audit + cut
    plan authoring).

Then wait for Josh's go-ahead before proceeding.

PHASE 2 — PROTOCOL RULES (load-bearing; not optional)

The five rules from memory/feedback.md you must operate under for the
entire session:

  R1. PRIMARY-SOURCE FIDELITY. Every page-savings estimate must trace
       to either (a) .aux ToC entries with explicit page numbers or
       (b) direct line-count → page-count derivation with stated
       assumed lines/page density. Do not estimate from memory.
       When a section spans pages P1 to P2 per .aux, page savings of
       moving that section = (P2 - P1 + 1) - (fraction of P1 above
       the section + fraction of P2 below). Round conservatively.

  R2. PROTOCOL-DEVIATION PRE-CHECK. The paper's framing decisions are
       load-bearing for track-selection (#13 Interpretability). Any
       cut that compresses §3 Dimensional Escape framing, §5 Exp 1
       (Track F) negative control, §5 Exp 2 (Track G) Cartography,
       or §1 Contributions list is a paper-affecting deviation — flag
       to Josh as HIGH-RISK in the cut plan, with a one-sentence
       framing-impact note. Acceptable cuts: anything that preserves
       the 3 contributions + 2 empirical pillars + Dimensional Escape
       unification narrative.

  R3. WORD + SECTION-LEVEL DISCIPLINE. When evaluating a section for
       cutting, read both heading + opening paragraph + closing
       paragraph. The opening paragraph sets the section's purpose;
       if the purpose is genuinely load-bearing for the contribution
       story, the section is harder to cut wholesale (but its
       internal paragraphs may still be trimmed). The closing
       paragraph often summarizes; if the summary is informative,
       deleting it loses the reader's anchor.

  R4. VERIFY EXTERNAL REFERENCES BEFORE PROPOSING ANYTHING. If you
       propose moving content to §A or supplementary, verify that the
       content's forward-references (table labels, equation labels,
       section refs) all resolve from the new location. A cut that
       breaks 7 forward references is not a cut; it's a regression.

  R5. NO APPLICATION. Do not edit the canonical TeX in this session.
       The cut plan you author goes to Josh for sign-off; application
       happens in a separate Day-10 Session B session. If you find
       yourself tempted to "fix" a typo or apply a small inline
       change, stop — file it as a Day-10 residue item instead.

PHASE 3 — TASK STEPS (after Josh's go-ahead)

Execute the following in order:

  Step A. PAGE-DISTRIBUTION TABLE.
    From the .aux ToC entries (grep for `\@writefile{toc}` then
    extract section + page number), build a table:

    | Section | Subsection | Page start | Estimated pages |
    | §1 Introduction | | 1 | ? |
    | §1.1 Contributions | | 2 | ? |
    ...
    | §A Reproducibility | | 17 | ? |

    Compute "Estimated pages" as page-of-next-section minus
    page-of-this-section (the section roughly spans those pages).
    Note any §-spanning issue and document. Total body pages =
    sum across §1 through §7.6.

  Step B. SECTION-BY-SECTION CUT-CANDIDATE INVENTORY.
    For each subsection / paragraph block in §1 through §7.6, decide
    one of three actions and document:

      KEEP — load-bearing for contribution story, cannot cut.
      TRIM — internal compression possible without losing substance.
      MOVE — wholesale move to §A appendix (which is page-unlimited)
        or to supplementary materials.

    For each TRIM and MOVE, estimate page savings (in pages, fractional
    OK). For TRIM, name the specific paragraphs that would be cut. For
    MOVE, name the destination (§A subsection, supplementary file)
    and verify all forward references resolve from the destination
    (R4).

    Specifically scrutinize (these are pre-flagged candidates):
      (a) §6 Theorem 1 proof block (lines 514-516) — already a proof
          sketch; moving the full proof body to §A is the v2 §8
          pre-planned cut.
      (b) §6 Memory Complexity Proposition + worked example (lines
          519-528) — the proposition is Contribution-1-essential;
          worked example with 20B projection (post-Day-8-evening
          refinement) is a candidate per parallel-review Pass K.
      (c) §5.1 Phase 0 Architectural Decoupling + §5.2 Phase A Local
          Prototyping (Day-3 audit-applied, but still ~1 page of
          operational scaffolding) — candidates to MOVE to §A.
      (d) §2 Background and Related Work — 6 subsections covering ~4
          pages. The most-cuttable subsections per topic relevance
          are: §2.5 Continuous Latent Reasoning (overlap with §4.2),
          §2.6 LLM-as-a-Judge (relevance to current Option D+ is
          unclear given α+/Candidate-A pivot retirement of judges as
          baselines).
      (e) §7.4 Empirical Measurement Prerequisites — substantial
          content about W_K projection + OEI alpha-sweep table. The
          table (`tab:oei-alpha-sweep`) was relocated here from §5
          Day 4. Candidate to MOVE the alpha-sweep table back into
          §A.
      (f) §7.5 Future Work — 4 paragraph blocks; the anti-Goodhart
          reward designs list is the Pass K candidate.
      (g) §3.3 Discrete Text-Space Bottleneck — verify its load-
          bearing role for §3 framing; if it's expandable historical
          context, candidate for trimming.

    Beyond these pre-flagged candidates, identify any others your
    audit surfaces. Be exhaustive in §2 (longest section) and §7
    (5 subsections).

  Step C. RANKED CUT PLAN.
    Order the cuts by (risk-adjusted-savings) descending. Risk-
    adjusted-savings = (page savings) × (1 - risk weight), where
    risk weight is 0.0 for LOW, 0.3 for MEDIUM, 0.7 for HIGH.

    Output a single ranked list:

      Cut #1: [section/subsection], action [TRIM | MOVE], destination
        [section / supplementary], page savings [X.X], risk [LOW |
        MEDIUM | HIGH], framing-impact note [one sentence], cumulative
        savings if applied through this cut: [Y.Y]
      Cut #2: ...
      ...
      Cut #N: ...

    Identify the cumulative-line where total savings ≥ 9 pages (the
    target to ensure we hit 8 pages with margin). Mark that line
    explicitly: "Sign-off threshold for budget compliance: Cuts #1
    through #K." Identify a second threshold line at savings ≥ 8
    pages (minimum compliance).

    Note: if even applying ALL cuts in your plan doesn't reach 9
    pages of savings, surface that as a CRITICAL flag — we need to
    consider additional restructuring (e.g., dropping a contribution,
    or accepting submission with cover-letter justification).

  Step D. SIGN-OFF ASK.
    The plan output file ends with a clear sign-off section:

      "Recommended sign-off threshold: Cuts #1 through #K.
       Page savings: X.X pages cumulative.
       Resulting body page count estimate: Y.Y pages (target ≤ 8).
       Margin: Z pages.

       Josh approves application of Cuts #1 through #K? [Y/N]
       If Y, Day-10 Session B applies in order #1, #2, ..., #K
       with G-smoke-compile gate after each cut + final page-count
       verification post-#K."

    Do NOT pre-approve cuts on Josh's behalf. The session ends with
    the plan file written and a brief status report to Josh.

  Step E. WRITE THE PLAN FILE.
    Output target: `docs/logs/2026-05-15_page-budget-cut-plan-v1.md`.

    File structure:
      - Header (date, owner, status, decision basis)
      - §1: Current page distribution table (from Step A)
      - §2: Section-by-section cut-candidate inventory (from Step B)
      - §3: Ranked cut plan (from Step C)
      - §4: Sign-off ask (from Step D)
      - §5: Out-of-scope notes (what was considered but rejected;
        e.g., dropping a contribution, switching tracks, etc.)
      - §6: Day-10 application notes (which cuts apply first, smoke-
        compile gate after each, verification steps).

    Length target: 2500-4000 words. Tables encouraged for the page-
    distribution + cut-plan content (clear cumulative-savings column).

  Step F. STOP.
    Report to Josh in under 150 words:
      - Cut plan written to the file path above.
      - Total estimated cuts needed (in pages).
      - Number of cuts in the ranked list.
      - Sign-off threshold line (Cut #K) for budget compliance.
      - Highest-risk cut in the plan.
      - Estimated Day-10 wall time for application (rough).
    Stop. Do not pull additional work forward.

PHASE 4 — TRIP-WIRES (no-go conditions for this session)

  - If Phase 1 reading reveals the canonical TeX has been modified
    since the Day-8-evening commit (`76d63f1`) — i.e., new commits
    exist that you weren't briefed on — STOP and report. The cut
    plan must operate on the state Josh signed off on for this
    session.
  - If your Step A page-distribution audit finds that the body is
    actually ≤ 8 pages (contradicting the Day-8 working session's
    finding), STOP and report. Re-verify before proposing cuts that
    aren't needed.
  - If your Step C ranked cut plan cannot reach 9 pages of cumulative
    savings even with all cuts applied, mark this as a CRITICAL flag
    in §4 of the plan file and escalate to Josh. He may need to
    consider:
      (a) dropping a contribution from §1 (e.g., the §6 theoretical
          analysis as Contribution 1, or per-layer telemetry as
          Contribution 2);
      (b) restructuring §A to absorb more body content;
      (c) accepting submission over budget with cover-letter
          justification (high rejection risk);
      (d) deferring submission to ARR June 2026 cycle (no-go for
          EMNLP 2026 main track but Findings track may accept).
  - If you identify a HIGH-risk cut that would break paper framing
    (e.g., compressing §3 Dimensional Escape, dropping §5 Exp 1
    Track F, or removing §5 Exp 2 Cartography), flag it as such in
    the cut plan but do NOT include it in the recommended sign-off
    threshold. The framing-breaking cuts go in a separate
    "consider-only-if-mandatory" section at the bottom of the plan.

PHASE 5 — END-OF-TASK HANDOFF

You stop after Step F. Memory updates that should land in this
session (no commit needed; memory is outside the git tree):

  - memory/recast_action_plan.md "Pending TeX application" section:
    add a new bullet "Day-9 page-budget cut plan AUTHORED 2026-05-15
    at `docs/logs/2026-05-15_page-budget-cut-plan-v1.md`. Day-10
    application target per Josh sign-off."
  - memory/MEMORY.md recast-pointer line: append "Page-budget cut
    plan drafted Day-9 at `docs/logs/2026-05-15_page-budget-cut-
    plan-v1.md`; sign-off pending."

Do NOT update memory until the plan file is written and you are at
Step F. If the session terminates before Step F (e.g., trip-wire
fires), do not update memory.

The Day-9 session log entry happens after Josh's sign-off lands — not
this session. This session's deliverable is the plan file + memory
pointer updates. The Day-9 session log (in `2026-05-15_session-log.md`
or a new `2026-05-16_session-log.md` depending on calendar) is
authored after Day-10 application has begun, summarizing both the
audit + the application together.

END OF PROMPT.
```

---

## Notes for Josh

- The prompt is structured parallel to the Day-5 resume prompt (Phase 1 reading → Phase 2 rules → Phase 3 task steps → Phase 4 trip-wires → Phase 5 handoff). The model has no context from this session; everything it needs is loaded from disk.
- Phase 1 is heavier than the Day-5 resume (12 items vs 9) because the cut plan requires understanding the full paper structure + the rendering state + plan-v2 page-budget rules + the parallel-review Pass K alignment. Cutting these reads risks the model applying ill-informed cuts.
- The Phase-3 step structure (A page-distribution → B inventory → C ranked plan → D sign-off ask → E write file → F stop) maps cleanly to deliverable structure. The plan file's §1-§6 structure is pre-specified so Opus 4.7 doesn't have to invent it.
- Risk-adjusted-savings formula (risk weight: LOW=0.0, MED=0.3, HIGH=0.7) is conservative — it discounts high-risk cuts heavily. If you want more aggressive ranking (higher savings preferred regardless of risk), modify the weights before pasting.
- The plan file ends with an explicit sign-off ask (Step D). You will respond Y/N to a specific cut-threshold; if Y, Day-10 Session B applies in order. This avoids "infinite discussion" mode and forces a decision.
- The trip-wires include the option of considering dropping a contribution if the cut plan can't reach 9 pages. This is the worst-case escape valve — surfacing it explicitly so the prompt doesn't hide it behind "be creative."
- Estimated wall time for Opus 4.7 to execute Phases 1-3: ~60-90 min including the reading. Phase 1 alone is ~30 min of reading; Phases 2-3 are the substantive work.
- Memory updates in Phase 5 are scoped to two specific lines. The Day-9 session log is deferred to after Day-10 application begins (parallel-with-application logging, similar to the Day-5/6 pattern).
