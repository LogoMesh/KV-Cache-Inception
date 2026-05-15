# Day 5 Resume Session Prompt — 2026-05-14 EOD

> **Context for Josh:** the VSCode-Claude-Code session you used for Day 5 hit a `Unhandled case: [object Object]` client-side stream error after applying the §5 Experiment 2 Cartography recast Edit to the canonical TeX. The Edit landed on disk (uncommitted); G-smoke-compile + commit + Day-5 session-log-append did not run. Use this prompt to resume in a fresh Claude session (Claude Desktop app or fresh VSCode Claude Code session) and close out Day 5 cleanly.
>
> **How to use:** paste the block below into a fresh Claude session. The block is self-contained and assumes the model has no prior context beyond what it reads from disk. Memory auto-loads at session start (`MEMORY.md` + indexed pointers) so the orientation is largely handled — the reading phase narrows to the artifacts specific to the resume task.

---

## Paste everything below this line into the fresh Claude session

```
You are resuming a single-task close-out of Day 5 of a 14-day ARR-cycle
execution sprint for the LogoMesh / KV-Cache-Inception NeurIPS-via-EMNLP
paper. The Day-5 §5 Experiment 2 Cartography recast Edit was applied to
the canonical TeX in the prior session but G-smoke-compile + commit +
Day-5 session-log-append did NOT complete due to a VSCode-Claude-Code
client-side stream error ("Unhandled case: [object Object]"). The Edit
is on disk, uncommitted. Your job is to verify it, smoke-compile, commit,
append the Day-5 session log, and stop.

This is a focused resume task. Do NOT pull additional work forward; do
NOT re-derive analyses. The hard discipline of this resume is: trust the
prior session's artifact on disk, verify it as you find it, run the gate,
commit, log, stop.

PHASE 1 — REQUIRED READING (in this order, all in full unless noted)

  1. CLAUDE.md (project root). Standard project orientation.
  2. memory/MEMORY.md (index — read the line entries).
  3. memory/recast_action_plan.md — strategic state pointer. The first
       paragraph names today's Day-5 EOD state explicitly: zou2023repe
       fix already committed (`19f9eb5`); §5 Exp 2 Edit applied
       uncommitted; G-smoke-compile + commit + session-log-append pending.
       The "Pending TeX application" section's §5 Exp 2 bullet (now
       marked 🟡 EDIT APPLIED) names the BLOCKING fix wording that was
       applied inline (recommended footnote per option b).
  4. memory/feedback.md (full read — five protocol rules R1-R5 are
       load-bearing for paper claims, audit discipline, and primary-source
       fidelity; they apply equally to a resume task and must be loaded
       before you touch the TeX).
  5. docs/logs/2026-05-12_session-log.md (full read — Day 4 P1 + P2 work
       context: §5 Exp 1 recast applied + §A v1 drafted + smoke-compile
       gate landing + natbib quick patch).
  6. docs/logs/2026-05-14_session-log.md (full read — Day 5 entry
       written earlier today covering the zou2023repe fix + commit
       `19f9eb5`. This is the file you will APPEND a §5 Exp 2 entry to
       once the commit lands; do not rewrite the existing Day-5 entry).
  7. docs/logs/2026-05-13_track-G-audit-pass.md (skim §1 + §2.3 Family D
       only — primary source for the BLOCKING fix raw values: 3B C4
       HellaSwag continuation reward gain $+0.0367$ which rounds to
       $+0.037$; original Track G report "+0.005 to +0.025" range was
       across reported cells only with 3B C4 marked "data truncated").
  8. docs/logs/2026-05-11_session-A-prose-drafts-v10-exp2.md (full read
       — the draft that was applied. Pass-1+Pass-2 framing audit in the
       prior session verified NO α+ residue; you do NOT need to re-audit
       framing. Read for orientation on what the applied TeX block looks
       like).
  9. docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex — read
       lines 358–460 in particular (the new §5 Exp 2 block: REVISION
       marker + paragraph with `\label{sec:exp2}` + four-finding body +
       BLOCKING-fix closing paragraph + footnote + IMPL line + new
       Table 2 + new Figure 1 + the §5-Exp3-5-cut marker preserved
       after). Verify the marker reads `option-D+-exp2-cartography-
       2026-05-14`, the paragraph title is "Experiment 2: Latent
       Cartography MCTS across Prompt Classes", and the footnote text is
       exactly "The upper bound is driven by HellaSwag continuation at
       the 3B scale (path-mean reward gain $+0.037$); the other seven
       cells span $+0.005$ to $+0.025$." If any of these are missing or
       wrong, STOP — do not smoke-compile a wrong-state file. Report to
       Josh and wait.

After reading, do NOT start commit work yet. Tell Josh in ONE short
paragraph (under 120 words):

  - That you have read the Phase-1 artifacts and the §5 Exp 2 Edit is
    confirmed on-disk as expected.
  - The current TeX line count (verify; should be ~785-790, was 705
    post-`19f9eb5`).
  - Confirmation that the prior Day-5 commit (`19f9eb5`, zou2023repe
    fix) is in git history and the §5 Exp 2 changes are unstaged.
  - Confirmation that `docs/NeurIPS/figures/exp2-mean-step-alpha.pdf`
    exists (the figure the new \includegraphics line references).
  - Your readiness to run G-smoke-compile + commit + log-append.

Then wait for Josh's go-ahead before proceeding to Phase 3.

PHASE 2 — PROTOCOL RULES (load-bearing; not optional)

The five rules from memory/feedback.md you must operate under for the
entire session:

  R1. PRIMARY-SOURCE FIDELITY. Every empirical number in the new TeX
       block traces to a primary artifact (Track F report, Track G
       report, Track G audit pass) — do not "verify" by re-checking
       against running-log summaries. If you find a number that
       surprises you, re-derive from the rawest available source.

  R2. PROTOCOL-DEVIATION PRE-CHECK. The Edit on disk implements the
       v10-exp2 draft methodology verbatim (depth-3 MCTS, 27 paths,
       alpha-set {0.1, 0.5, 1.0}, entropy reward, 4 prompt classes × 2
       scales). If smoke-compile surfaces a methodology-divergence
       symptom, flag to Josh BEFORE patching.

  R3. WORD + SECTION-LEVEL AUDIT DISCIPLINE. The Pass-1+Pass-2 framing
       audit already ran 2026-05-14 and cleared NO α+ residue in the
       body prose; you do NOT need to re-audit. However, if Phase-1
       reading uncovers a framing inconsistency between the new §5 Exp
       2 block and §3 Hypothesis 2 / §5 Exp 1 that the earlier audit
       missed, flag to Josh BEFORE committing.

  R4. VERIFY URLS / PATHS / REFS / LABELS BEFORE USE. The Edit
       references `figures/exp2-mean-step-alpha.pdf`, `\ref{eq:goodhart}`,
       `\ref{sec:exp1}`, `\ref{def:dim_escape}`, `\ref{hyp:goodhart}`,
       `\label{sec:exp2}`, `\label{tab:exp2-cartography-paths}`,
       `\label{fig:exp2-mean-step-alpha}`. Before commit, verify (a)
       the figure PDF exists, (b) the four `\ref{}` targets resolve in
       the smoke-compile log, (c) the three `\label{}`s do not collide
       with anything pre-existing in the TeX.

  R5. VERIFICATION-SCRIPT CONVENTION DISCIPLINE. Not applicable to this
       resume task — no audit script involved. Carry the rule forward
       for awareness.

PHASE 3 — RESUME TASK STEPS (after Josh's go-ahead)

Execute the following in order. Each step has a verification criterion;
if a criterion fails, STOP and report to Josh — do not proceed to the
next step.

  Step A. SANITY CHECK on the Edit.
    - `git status` shows `docs/NeurIPS/04.18.2026-NeurIPS-Research-
      Proposal-2.tex` as Modified (not Added/Deleted/Untracked).
    - `git diff docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex
      | head -100` shows the new REVISION marker `option-D+-exp2-
      cartography-2026-05-14` on top of the deleted `Exp2-recast-
      reward-ablation` marker.
    - `wc -l docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`
      reports a value in the 780-795 range (was 705 post-`19f9eb5`,
      Edit added ~80 lines: REVISION marker + 4-finding paragraph +
      closing paragraph + footnote + IMPL + Table 2 block + Figure 1
      block).
    - `ls docs/NeurIPS/figures/exp2-mean-step-alpha.pdf` confirms the
      figure file exists.

  Step B. G-SMOKE-COMPILE (per plan v2 §9).
    Run pdflatex TWICE from inside docs/NeurIPS/. MiKTeX at
    C:\Users\Josh\AppData\Local\Programs\MiKTeX\miktex\bin\x64\
    pdflatex.exe with -interaction=nonstopmode -halt-on-error.

    Pass-criteria for BOTH passes:
      - Exit code 0.
      - PDF produced.
      - Zero `Error:` lines in the log (excluding any in `LaTeX
        Warning:` lines, which are warnings not errors).
      - Acceptable LaTeX warnings: cosmetic float-position warnings
        (`'h' float specifier changed to 'ht'`), undefined references
        ONLY for `eq:mcts_reward` (1×, unchanged from Day 4 — that
        equation was deleted in §4.2.3 deletion Day-3 audit, the stale
        reference at line ~361 is filed for Day-7 §5.5 rewrite). All 7
        previously-undefined `sec:exp2` warnings should be GONE in this
        pass — the new `\label{sec:exp2}` resolves them.
      - The 2× `Author undefined for citation 'zou2023repe'` natbib
        warnings present at Day-4 EOD: ALREADY RETIRED by today's
        commit `19f9eb5`; they should remain absent.

    Trip-wire: if either pdflatex pass exits non-zero, OR an `Error:`
    line appears, OR new undefined-ref warnings appear beyond the
    single `eq:mcts_reward`, INVOKE the plan-v2 §3 plasticity rule:
    diagnose within 30 minutes; if not resolved in that window, ROLL
    BACK the §5 Exp 2 Edit (`git checkout
    docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`),
    explain to Josh, do NOT commit. Failed edits do not promote to
    the next day's baseline.

  Step C. WARNING INVENTORY VERIFICATION.
    Grep the post-Pass-2 log for `LaTeX Warning|^!|Author undefined`.
    Expected inventory (after the §5 Exp 2 Edit lands):
      - 1× `eq:mcts_reward` undefined (Day-7 §5.5 rewrite resolves)
      - 1× `'h' float specifier changed to 'ht'` (cosmetic)
      - 1× "There were undefined references" meta-warning
      - Possibly 1× cosmetic floating-figure warning from the new
        Figure 1 (acceptable)
    Total expected: 3-4 lines. Compare against the Day-5 pre-Exp2
    inventory of 10 lines; the Exp2-Edit should NET DOWN to 3-4
    (removing 7× sec:exp2 + retaining 1× eq:mcts_reward + h-float).

  Step D. COMMIT.
    `git add docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex
    && git commit -m "docs: Day 5 — §5 Exp 2 Cartography recast +
    Table 2 + Figure 1 + BLOCKING fix"` (use a HEREDOC if the message
    contains characters the shell may interpret; the message above
    has no special chars and should commit clean).

    Scope: 1 file changed (canonical TeX only). Do NOT bundle in
    build artifacts (.aux/.log/.pdf/.out), `.claude/`, premortem-v2
    variant, submodule churn, or unrelated dirty session-log files —
    they were excluded from the Day-4 P1 commit `3c5d741` and the
    Day-5 commit `19f9eb5` for the same reasons.

  Step E. DAY-5 SESSION LOG APPEND.
    Open `docs/logs/2026-05-14_session-log.md` (created by the prior
    Day-5 session; contains the zou2023repe entry). APPEND a new
    section after the existing Day-5 entry, formatted parallel to
    the Day-4 P1 entry in `docs/logs/2026-05-12_session-log.md`,
    covering:
      - The §5 Exp 2 Edit details (source draft, REVISION marker
        name, target lines 360-363 replaced with the new block).
      - Section-level edits summary (paragraph replace + Table 2
        insert + Figure 1 insert + IMPL line + BLOCKING fix inline).
      - The BLOCKING fix wording verbatim (the recommended-option
        footnote with HellaSwag-at-3B as upper-bound driver).
      - Reference resolution table (`\label{sec:exp2}` resolves N
        forward refs; verify N from the log inventory).
      - G-smoke-compile gate results (pass 1 + pass 2 exit codes,
        PDF page count, warning inventory).
      - Plan-v2 §2 reconciliation: §5 Exp 2 was scheduled Day 7;
        applying it Day 5 is a 2-day pull-forward, the 7th of the
        sprint.
      - Day-9 residue list status (unchanged from Day-5 morning).
      - Git commit hash and message (`git log -1 --oneline`).
      - Open items for Day 6 (do NOT pre-decide Josh's Day-6 call —
        most plan-v2 Day 7 work is now done; Day 6 has substantial
        slack).

    Also update the "Day 5 wall time" cumulative table near the end
    of the existing entry to add a new row covering the resume work
    (rough estimate; Josh will correct if needed).

    Do NOT commit the session log update separately — Josh's pattern
    for Day 4/5 has been to commit session logs alongside the next
    day's commit, not as separate commits.

  Step F. STOP.
    Report to Josh:
      - Day-5 §5 Exp 2 Cartography recast COMMITTED as `<new hash>`.
      - Day-5 session log appended (uncommitted, per pattern).
      - TeX state: ~785-790 lines, D+-aligned through §7 EXCEPT
        §5.5 Evaluation Metrics (Day-7 rewrite) and §6 worked-example
        numbers (Day-8 rewrite). All §5 + §7 paper-body work for
        Phase 1 is now in the TeX.
      - Plan-v2 §2 status: Day 5 absorbed Days 5 + 6 + 7-§5-Exp-2
        scheduled work; Days 6-7 remain on the calendar but the
        scheduled tasks are mostly done.
      - Day-9 residue list: 1 item (Day-8 ACL conditional check).
    Stop. Do not pull additional work forward.

PHASE 4 — TRIP-WIRES (no-go conditions for this resume)

  - If Phase 1 reading reveals the Edit on disk does NOT match the
    expected state (paragraph title, REVISION marker tag, footnote
    text, line count, label names), STOP and report — do not commit
    a wrong-state file.
  - If G-smoke-compile fails per Step B trip-wire criteria, invoke
    plan-v2 §3 diagnose-then-rollback plasticity rule.
  - If the warning inventory grows beyond the expected 3-4 lines
    with new undefined refs or new package errors, STOP — diagnose
    the cause before committing.
  - If during Step E session-log append you discover a structural
    framing issue in the applied TeX block that the prior audit
    missed, STOP — do not silently patch. Flag to Josh.

PHASE 5 — END-OF-TASK HANDOFF

You stop after Step F. Memory updates that landed today (the recast
action plan pointer in MEMORY.md was updated to reflect Day-5 EOD; the
recast_action_plan.md "State as of" line was updated; the "Pending TeX
application" Exp 2 entry was marked 🟡 EDIT APPLIED) should be UPDATED
ONCE MORE after Step D commit lands, changing:
  - MEMORY.md recast action plan pointer line: change "(b) §5 Exp 2
    Cartography recast Edit APPLIED to canonical TeX ... UNCOMMITTED"
    to "(b) §5 Exp 2 Cartography recast COMMITTED as <new hash>".
  - recast_action_plan.md "State as of 2026-05-14 EOD" line:
    similar update.
  - recast_action_plan.md "Pending TeX application" Exp 2 bullet:
    change 🟡 EDIT APPLIED to ✅ APPLIED and add commit hash; remove
    the "UNCOMMITTED" + "pending next session" + "Resume prompt"
    references.

These memory edits do not require a commit (memory is outside the
git tree).

END OF PROMPT.
```

---

## Notes for Josh

- The prompt is shorter than the Day-5-kickoff prompt (focused single task vs full-day scope).
- Phase 1 reading list is 9 items (vs 12 in the kickoff prompt); plan-v2 + Day-4-other-artifacts trimmed because the resume is a closeout, not a re-orientation.
- Phase 3 is a 6-step linear sequence (A-F); each step has a verification criterion and a clear stop condition.
- Phase 4 trip-wires include the plan-v2 §3 diagnose-then-rollback rule explicitly.
- Phase 5 includes the memory clean-up after commit (otherwise memory stays at "uncommitted" state forever, against the durable-only-after-it-lands discipline).
- If the desktop-app session does NOT have filesystem MCP, the prompt fails at Phase 1 step 1 (it cannot read CLAUDE.md). Fallback: open the CLI Claude Code in a terminal alongside the desktop app, or run the resume in a fresh VSCode Claude Code session that does have filesystem tools.
- The commit message in Step D is a single short line ("docs: Day 5 — §5 Exp 2 Cartography recast + Table 2 + Figure 1 + BLOCKING fix"); no Co-Authored-By trailer to match the Day-4 P1 / Day-5 zou2023repe commits' style.
