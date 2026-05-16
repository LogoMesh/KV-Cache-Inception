# Day-12 Orchestrator Session Prompt

**Calendar:** 2026-05-17 (Day 12 of 14-day execution plan; ARR submission 2026-05-25, T-8 days).
**Session role:** Orchestrator. Dispatch 5 axis sub-agents in parallel + synthesize their reports into the parallel-review-synthesis doc.
**Paper state:** Post-Day-11 commit `8085545` ("Day 11 — hedge fixes + page-budget body-to-8pp"). Body 8 pp, ACL format-compliant, hedges applied (Pass M findings pre-empted).

---

## Copy-paste prompt for a fresh Claude session

```
Day-12 of LogoMesh NeurIPS / EMNLP via ARR May 2026 cycle. You are an
orchestrator session. Your job: dispatch the 5 axes of the parallel
adversarial review per `docs/logs/2026-05-14_parallel-review-plan.md`,
collect reports, and synthesize.

READ FIRST (~10 min):
1. `memory/MEMORY.md` + `memory/recast_action_plan.md` — current state.
2. `memory/feedback.md` — Josh's collaboration preferences. CRITICAL:
   primary-source fidelity; honest outside-view; protocol-deviation
   pre-check; word + section-level audit discipline.
3. `docs/logs/2026-05-14_parallel-review-plan.md` — the full parallel-
   review plan with 5 pre-drafted axis prompts (Axis 1: code-semantic
   correctness; Axis 2: framing defensibility; Axis 3: reproducibility;
   Axis 4: reviewer-prior alignment; Axis 5: standard adversarial w/
   19 sub-passes A-S).
4. Latest commit hash: `8085545` (Day-11). Body 8 pp.

DO NOT read the full paper before firing agents. Each agent reads the
paper itself per its prompt's "REQUIRED READING" list.

ROUND 1 (AM, ~1-2 hr wall, mostly idle):
Fire Axes 1, 2, 5 as PARALLEL Agent calls in a single message:
  - Axis 1: subagent_type=general-purpose. Prompt = verbatim copy of
    `2026-05-14_parallel-review-plan.md` §"Axis 1 — Code-semantic
    correctness" code block (lines 64-140 approx).
  - Axis 2: subagent_type=general-purpose. Prompt = verbatim §"Axis 2
    — Dimensional Escape framing defensibility" block.
  - Axis 5: subagent_type=general-purpose. Prompt = verbatim §"Axis 5
    — Standard adversarial pass" block (19 sub-passes A-S; the largest
    of the 5).

For each agent return: IMMEDIATELY write the report to
`docs/logs/2026-05-17_axis-N-report.md` (where N = 1, 2, 5). Then drop
the report from in-context memory — synthesis reads files, not in-
context. Each report is 2000-8000 words; storing in context bloats
the orchestrator.

ROUND 1 PM (triage):
After all 3 agents return:
  - Read each report file. Tally SEVERE / MAJOR / MODERATE / MINOR
    findings per the rubric in `2026-05-14_parallel-review-plan.md`
    §"Severity rubric".
  - If ≥1 SEVERE finding in Axis 1 (code-vs-paper semantic divergence):
    ESCALATE TO JOSH IMMEDIATELY. Do not fire Round 2 until Josh has
    weighed in. Code-vs-paper SEVERE is paper-go/no-go territory.
  - If ≥1 SEVERE OR ≥3 MAJOR total: pause Round 2; surface to Josh.
  - Else: proceed to Round 2.

ROUND 2 (Day-13 AM, ~1 hr wall):
Fire Axes 3 + 4 as PARALLEL Agent calls:
  - Axis 3: prompt = verbatim plan doc §"Axis 3 — Reproducibility from §A".
  - Axis 4: prompt = verbatim plan doc §"Axis 4 — Reviewer-prior alignment".

Write returns to `docs/logs/2026-05-17_axis-3-report.md` and
`docs/logs/2026-05-17_axis-4-report.md`.

SYNTHESIS (Day-13 PM):
Write `docs/logs/2026-05-22_parallel-review-synthesis.md` per plan doc
§"Synthesis protocol":
  1. Aggregate severity counts across all 5 axes.
  2. De-duplicate cross-axis findings (e.g., if Axis 2 and Axis 5 both
     flag the same overclaim sentence, count once at higher severity).
  3. Patch decision per finding:
       - SEVERE: patch by Day-14 EOD; if can't patch, escalate to
         Tianyu for paper-go/no-go.
       - MAJOR: patch by Day-14 EOD; else add to Limitations + cover
         note flag.
       - MODERATE: patch best-effort by Day-14 EOD.
       - MINOR: best-effort; document deferred items in cover note.
  4. Output: severity counts, dedup'd finding list, patch plan with
     owner + deadline.

HARD CONSTRAINTS:
  - Don't synthesize axis reports in-context. Always read-from-file.
  - For Axis 1 SEVERE findings: stop, escalate, wait.
  - Don't re-fire an axis from scratch if first-pass output is shallow.
    Instead, send a follow-up SendMessage with a sharpened question.
  - The parallel-review prompts are pre-drafted; do NOT reword them
    when dispatching. The prompt design is load-bearing — non-leading
    sub-questions for Pass N + P/Q/R were carefully phrased Day-10 EOD.
  - DON'T LIFT ANONYMITY in agent prompts. Sub-agents read the
    anonymized review PDF, not author info. The supplementary
    anonymization happens Day-12 in parallel (separate from this
    parallel-review work) per plan v2 §5.1.

KNOWN OPEN RESIDUAL RISKS (Day-12 review should confirm or refute):
  - Pass Q body §4 depth: partial Day-11 fix (re-folded eq:reverse_rollback
    inline + brief FP32 mechanism note in §4.2 intro). Full §E re-fold
    deferred. Q asks whether minimal-fold is enough.
  - Pass M(d)/Axis 2 Q3 unification claim: hedged Day-11 ("argue ...
    analogous; conjecture both as sub-cases"). Stronger defense
    (retire "Dimensional Escape" coined term) was not taken — Day-12
    Axis 2 Q5 evaluates whether the hedge is sufficient.
  - Pass R 1 family + 2 negatives + 0 positives: unaddressed Day-11.
    Day-12 evaluates whether reviewer-likely rejection or revision-
    request.

PARALLEL WORK STREAM (Day-12 same day):
  - Supplementary anonymization build per plan v2 §5.1 (build curated
    copy of repo subset with URLs / identifiers anonymized; not
    affecting the live LogoMesh/KV-Cache-Inception repo). This is
    Day-12 work but SEPARATE from this parallel-review orchestration.
    Do not bundle.

REPORT BACK TO JOSH:
  - Round 1 firing summary (≤100 words).
  - Round 1 triage outcome (severity counts) and decision (proceed
    Round 2 / escalate / pause).
  - Round 2 firing summary.
  - Synthesis doc location + headline findings.
```

---

## Why this works

- **Sub-agent prompts are pre-drafted** in `2026-05-14_parallel-review-plan.md`. Orchestrator pastes them verbatim — no re-design under time pressure.
- **Reports go to files, not in-context.** 5 reports × ~4000 words = 20000 words. Storing in orchestrator context = context bloat. File-write-then-drop = orchestrator stays lean.
- **Severity-driven flow control.** If SEVERE in Axis 1 (code-vs-paper), Day-12 stops. If only MODERATE, Day-12 proceeds. Decision rule is in the plan doc.
- **Honest outside-view bias.** The plan doc's non-leading sub-questions (esp. Pass N strengthened + P/Q/R added Day-10 EOD) are designed to surface what we've missed. Trust the design; don't reword.

## What to NOT do Day-12

- Don't read the full paper before dispatching. The agents read it themselves.
- Don't try to be one of the 5 axes in addition to orchestrating. Orchestrate only.
- Don't fire all 5 at once — Round 1/2 split is per the plan's escalation logic.
- Don't synthesize until all reports are in files.
- Don't push to remote, commit code, or anonymize the live repo. Day-12 anonymization is a curated *copy*; orchestrator doesn't touch it.

## Day-13 + Day-14 (orientation only — not Day-12 scope)

- Day-13 AM: fire Round 2 (Axes 3 + 4); Day-13 PM: synthesize.
- Day-14: integrate co-author feedback (from Day-11 send-out async window) + adversarial findings; apply SEVERE/MAJOR patches; final REVISION-marker audit.
- Day-15: ARR portal submission.
- Submission: 2026-05-25.
