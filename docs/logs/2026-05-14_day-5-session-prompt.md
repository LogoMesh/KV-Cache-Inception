# Day 5 Session Prompt — 2026-05-14

> **How to use:** paste the block below into a fresh Claude session at the start of Day 5. The block is self-contained and assumes the model has no prior context beyond `CLAUDE.md` and the auto-loaded memory index.

---

## Paste everything below this line into the fresh Claude session

```
You are picking up a multi-day ARR-cycle execution sprint on Day 5 of a 14-day plan.
Before you write anything or touch any file, you have a read-and-orient phase. Do
not skip it; do not skim. Several of the protocol rules below were created in
response to specific 2026-05-11 / 2026-05-12 / 2026-05-13 failures, and they
exist because shortcuts here have already cost the project hours of recovery.

PHASE 1 — REQUIRED READING (in this order, all in full unless noted)

  1. CLAUDE.md (project-root).
  2. memory/MEMORY.md (index — read the line entries; they are the orientation map).
  3. memory/recast_action_plan.md (strategic state pointer; full read).
       Pay particular attention to the "Pending TeX application" section — the
       Day-7 §5 Exp 2 row contains a BLOCKING fix spec for v10-exp2 that
       Session B must not miss when applying to TeX.
  4. memory/project_context.md (full read).
  5. memory/feedback.md (full read — every entry is a protocol rule, not an
       opinion. The most recent additions cover (a) primary-source fidelity,
       (b) protocol-deviation pre-check, (c) word+section-level audit discipline,
       (d) verify URLs / paths / refs / labels before use, (e) verification-script
       convention discipline. All five are load-bearing.).
  6. docs/logs/2026-05-12_session-log.md (full read — Day 4 work: §5 Exp 1 recast
       applied, §A v1 drafted, Track G audit detour with false-positive trip-wire
       diagnosis).
  7. docs/logs/2026-05-13_track-G-audit-pass.md (full read — Track G audit
       findings; the Day-7 fix spec for v10-exp2 traces back to §2.3 of this doc).
  8. docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex
       (FULL READ — all 704 lines. This is the canonical paper. Read it
       end-to-end before you do anything else. Do not grep your way through it;
       do not partial-read it. The point of Day 5 is that you arrive at any
       work block already holding the full paper in your head. If you can't
       cite which section comes after §5.4 without looking it up, you haven't
       read it yet.).
  9. docs/logs/2026-05-11_session-A-prose-drafts-v10-exp2.md (full read — this
       is the Day-7 application target; the SCOPE caveat fix from Day-4 Track G
       audit applies HERE before Session B touches TeX).
 10. docs/logs/2026-05-13_session-A-prose-drafts-v10-reproducibility-appendix.md
       (full read — this is the Day-10 application target; smoke-compiled
       standalone on Day 4 but not yet in canonical TeX).
 11. docs/logs/2026-05-12_session-A-prose-drafts-v10-exp1.md (full read — the
       Day-4-applied Exp 1 recast; useful structural parallel for v10-exp2
       application planning).
 12. docs/logs/2026-05-10_14-day-execution-plan.md (full read — day-by-day
       matrix, §3 plasticity rules, §8 risk register, §9 verification gates).

After reading, do NOT start a task yet. Tell Josh in ONE paragraph:

  - What day of plan v2 it is, and the calendar offset state (whether internal
    day-numbering matches plan-calendar day, or is ahead).
  - The locked strategic state in one sentence.
  - Current canonical TeX line count + which sections are D+-aligned vs which
    still have pending work.
  - What landed in Day 4 (one clause each: Exp 1, §A draft, Track G audit,
    Day-7 fix spec).
  - The single Day-7 BLOCKING fix waiting in v10-exp2 (one sentence).
  - The plan-v2 Day 5 nominal task vs what is actually available given
    pull-forward state.

Then stop and wait for Josh's direction before picking a task.

PHASE 2 — PROTOCOL RULES (load-bearing; not optional)

Five rules from memory/feedback.md you must operate under for the entire
session. These are reproduced here so you don't drift after a long read.

  R1. PRIMARY-SOURCE FIDELITY. Every empirical number, methodology description,
       and hypothesis claim that lands in the paper must trace to a primary
       artifact (Track A/C/D/F/G report, canonical TeX, protocol document, raw
       scripts/_track_*_results_*.json) you have read in THIS session. Never
       cite from a running-log summary, never approximate from compaction
       memory, never trust a number that was already cited once in a derived
       artifact without re-deriving from the rawest available source when the
       number is going INTO the paper.

  R2. PROTOCOL-DEVIATION PRE-CHECK. Before deviating from a written protocol
       (sweep design, experiment setup, analysis convention), cross-check the
       deviation against §1 Contributions / §3 Hypotheses / §5 Experiment
       framings in the canonical TeX. If any paper-level claim invokes the
       methodology being dropped, flag to Josh BEFORE execution.

  R3. WORD + SECTION-LEVEL AUDIT DISCIPLINE. Audits that grep for keywords
       (Pass 1) catch word-level issues; they do not catch paragraph-level
       framing residue or heading-level inconsistency. Any audit that touches
       framing requires a Pass-2 framing-consistency read at section
       granularity. Pass-1-only is not an audit; it is a smoke test.

  R4. VERIFY URLS / PATHS / REFS / LABELS BEFORE USE. Memory strings are
       pointers, not ground truth. Before using a URL, file path, label
       (\label{...}), or reference (\ref{...}), verify it exists at the named
       location. Two 2026-05-13 failures grounded this rule:
       (a) "joshhickson/LogoMesh-NeurIPS" used as a remote URL → 404; correct
       origin is LogoMesh/KV-Cache-Inception. (b) "\ref{sec:memory-complexity}"
       used in §A v1 draft → label doesn't exist; correct is "\ref{prop:memory}".

  R5. VERIFICATION-SCRIPT CONVENTION DISCIPLINE. Audit scripts must declare
       and verify their indexing / rounding / scope conventions against the
       artifact being checked BEFORE producing findings. An audit that uses
       a different convention than its source generates phantom errors that
       look like source errors. Two 2026-05-13 instances grounded this rule:
       (a) top-5 indexing off-by-one (r <= 5 on 0-indexed → top-6) classified
       a clean Track G cell as a transcription error; (b) rounding-order
       convention difference classified two cells as digit errors. Both
       triggered a full Track-G report audit; both turned out to be the
       audit script's bugs, not source errors.

PHASE 3 — TASK OPTIONS FOR DAY 5

Plan v2 §2 Day 5 nominal tasks:
  - "§5 Experiment 2 polish + Table 2 + Figure 1/2 specs from Track G report
     §5 anchors" (drafts file)
  - "Begin §5 Experiment 2 draft as data lands"

Both nominal tasks ALREADY EXIST as artifacts (v10-exp2 drafted Day 3 pull-
forward; Day-7 application target). Day 5 therefore has more slack than the
plan-v2 line suggests. Available options, roughly ordered by dependency and
effort:

  A. PURE DAY-5 POLISH. Re-read v10-exp2 with fresh eyes; verify the Day-7
     BLOCKING fix spec in recast_action_plan.md is sufficient and unambiguous
     for Session B to apply mechanically; light revisions to the draft itself
     if any framing residue surfaces. ~30-45 min. Lowest-effort, on-plan.

  B. PULL FORWARD DAY 6 — FIGURE 1 GENERATION. Build a matplotlib script
     that reads scripts/_track_g_mcts_results_*.json and produces docs/NeurIPS/
     figures/exp2-mean-step-alpha.pdf (cross-scale mean-step α bar chart per
     prompt class). ~30 min. Doubles as a third-pass raw-JSON re-derivation
     of v10-exp2 Figure 1 spec values; if any number disagrees with the
     v10-exp2 draft, surface it immediately. Doesn't touch canonical TeX.

  C. PULL FORWARD DAY 8 — ACL STYLE SWITCH DRY-RUN. Swap \usepackage[main]
     {neurips_2026} → \usepackage[review]{acl}; remove the explicit-load
     natbib line; smoke-compile twice. §8 risk register flags this at 25% P
     of failure, HIGH impact — pulling forward de-risks 5 days early. ~1-2 hr.
     Higher-leverage but heavier; do this at the start of a session, not a
     tail.

  D. DAY-9 RESIDUE — zou2023repe \citet FIX. Two-line rewrite at canonical
     TeX lines 292 + 294 to follow line 214's "Zou et al.\ (2023)~\cite{...}"
     pattern (eliminates numbers-mode awkwardness). ~5 min. Concrete and
     small; touches canonical TeX so requires G-smoke-compile after.

  E. BAILEY 2024 §7.2 POLISH. ~20 min. Adds 1-2 sentence positioning of
     Bailey 2024 in §7.2 Broader Impact. Was deferred earlier per Josh
     sign-off; reopening requires Josh's explicit direction.

After your Phase-1 state summary, recommend one of A-E (or "pause" if you read
the state and conclude Day 5 should be light) with explicit dependency mapping
to plan v2 §2. Then wait for Josh's call.

PHASE 4 — TRIP-WIRES (no-go conditions for this session)

  - If during Option A you find v10-exp2 framing residue from the α+ era,
    flag immediately and stop — do not silently patch framing in a polish
    pass. Framing edits require Josh sign-off.
  - If during Option B the Figure 1 re-derivation surfaces a number that
    disagrees with v10-exp2's draft beyond rounding tolerance, stop the
    figure work and report. Do not paper over a number-discrepancy by
    "trusting the draft" — the Track G audit established that derived
    numbers in chains get verified against the rawest source.
  - If during Option C the ACL switch produces an Error: line, run the
    G-smoke-compile plasticity rule (§3 plan v2): diagnose within 30 min;
    if not resolved, roll back the day's edits and report. Failed edits
    do not promote to the next day's baseline.
  - If during Option D the rewrite at lines 292+294 conflicts with surrounding
    prose (e.g., already says "Zou et al.\ (2023)" in adjacent text), flag
    and stop — the residue list item assumes the lines are bare \citet{} in
    isolation; re-verify before patching.

PHASE 5 — END-OF-DAY HANDOFF

Whatever option you complete, write a Day-5 entry into docs/logs/2026-05-14_
session-log.md (create if absent; follow the 2026-05-12 session-log structure)
covering:

  - Day-of-plan reconciliation (Day 5 vs calendar 2026-05-14).
  - What landed (file paths, line ranges, REVISION marker tags, commit
    hashes if any).
  - Plan-v2 §2 reconciliation (which Day-N task was advanced; whether the
    pull-forward pattern continues).
  - Any memory updates made in-session.
  - Open items for Day 6 (do not pre-decide Josh's Day-6 call).

DO NOT update MEMORY.md MEMORY.md project_context line for "Day 5 complete"
without checking that the work actually landed AND the corresponding session-
log entry is committed. Memory updates have lagged work by a day twice in this
sprint; tighten the loop.

END OF PROMPT.
```

---

## Notes for Josh

- The prompt requires a full read of the canonical TeX (704 lines). That's the load-bearing change vs prior prompts.
- Phase 1 ends with a single-paragraph state summary, then waits. You can redirect the session before it picks a task.
- Phase 2's five protocol rules are reproduced inline so the session doesn't drift after a long read phase.
- Phase 3 lists A-E options with effort + dependency mapping; Option A is the strictly-on-plan choice, Option B is the recommended pull-forward, Option C is the higher-leverage risk reduction.
- Phase 4 trip-wires encode the "diagnose then roll back" plasticity rule (plan v2 §3) and the primary-source-fidelity discipline.
- Phase 5 closes the memory-lag loop the Day-4 session flagged on its own.

If you want to tighten further before Day 5, the two most likely cuts are: (a) drop Option E (Bailey polish) since it was deferred indefinitely; (b) collapse Phase 4 trip-wires into Phase 3 task descriptions to keep the prompt shorter. Both are judgment calls — current draft errs toward more guidance rather than less, on the assumption a fresh session benefits from explicit guardrails.
