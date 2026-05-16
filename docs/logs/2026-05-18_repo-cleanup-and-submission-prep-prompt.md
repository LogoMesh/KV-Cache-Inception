# Repo Cleanup + ACL/ARR/EMNLP Submission Prep — Discovery-First Prompt

**Calendar:** Day-13 or Day-14 of the 14-day execution plan (2026-05-18 / 2026-05-19). Runs AFTER the paranoid hallucination audit (Day-12 AM) and AFTER the parallel-review-plan firing (Day-12 + Day-13).
**Session role:** Discovery agent → planner → executor, in that strict order. Discovery phase MUST complete before any plan is proposed; plan MUST receive Josh's explicit approval before any file is renamed, deleted, or moved.
**Output target 1:** `docs/logs/<run-date>_submission-prep-discovery-report.md` — answers to the unknowns.
**Output target 2:** `docs/logs/<run-date>_repo-cleanup-and-supplementary-plan.md` — proposed plan, awaiting Josh sign-off.
**Output target 3 (post-approval only):** Actual file renames / deletions / package builds + a `<run-date>_repo-cleanup-execution-log.md`.

---

## Copy-paste prompt for a fresh Claude session

```
DISCOVERY-FIRST REPO CLEANUP + SUBMISSION PREP — final pre-submission
infrastructure work. Two intertwined tasks:

  (A) Repo cleanup: rename files (paper has "NeurIPS" in filename when
      it's an EMNLP submission), delete stale artifacts, reorganize
      documentation so the repo's structure reflects current project
      reality.

  (B) ACL/ARR/EMNLP submission prep: figure out how to actually submit
      the paper + supplementary materials anonymously, build the
      supplementary package, and document the submission workflow.

HARD CONSTRAINT: Phase 0 (Discovery) MUST complete and produce a
written report BEFORE Phase 1 (Planning) starts. Phase 1 plan MUST
receive Josh's explicit approval BEFORE Phase 2 (Execution) starts.
No file gets renamed, deleted, moved, copied to a supplementary
package, or uploaded anywhere until Josh signs off on the plan.

This is a DISCOVERY prompt, not a predetermined-instruction prompt.
You don't know yet:
  - exactly how ARR/EMNLP supplementary materials work
  - whether anonymity requires a new GitHub repo, anonymous.4open.science,
    direct OpenReview upload, or something else
  - what scope of files belongs in the supplementary package
  - what stale content in this repo can safely be removed
You will find out by reading primary sources, then propose, then act.

----------------------------------------------------------------------
PHASE 0 — DISCOVERY (do not skip; do not start Phase 1 until done)
----------------------------------------------------------------------

Write answers to docs/logs/<today's-date>_submission-prep-discovery-report.md
as you go. Each answer must cite a primary source (URL + verbatim
quote OR file path + line number). Do not paraphrase from memory —
the team's working understanding of ARR/EMNLP has been corrected
mid-project once already (2026-05-12: anonymization scope memory was
wrong — see memory/feedback.md).

ANSWER THESE QUESTIONS IN ORDER:

Q1. SUBMISSION PORTAL MECHANICS
  Primary sources to read:
    - https://aclrollingreview.org/cfp
    - https://aclrollingreview.org/authors
    - https://aclrollingreview.org/dates (verify ARR May 25 deadline
      from prior verification — Day-11 audit confirmed)
    - https://2026.emnlp.org/calls/main_conference_papers/
    - OpenReview ARR submission flow (search aclrollingreview.org for
      links to the OpenReview venue/portal)
  Answer:
    (a) Where does the paper PDF get submitted? (OpenReview venue ID?)
    (b) Where do supplementary materials get submitted? Same portal,
        separate field? Maximum file count? Maximum size?
    (c) What file formats are accepted for supplementary (zip, tar.gz,
        individual PDFs, code archives)?
    (d) Is there a separate "code & data" submission field distinct
        from "supplementary materials"?
    (e) Is the Responsible NLP Checklist (Josh's J3) submitted as
        a separate PDF or filled in via portal form? Where exactly?

Q2. ANONYMIZATION REQUIREMENTS — what must be anonymized vs what stays
  Primary sources to read:
    - https://aclrollingreview.org/cfp (search for "anonymity" and
      "anonymization")
    - https://aclrollingreview.org/anonymity (if exists)
    - https://aclrollingreview.org/responsibleNLPresearch/
    - acl_formatting.md in docs/NeurIPS/ — review-version anonymity
      requirements re: title page + self-citations
  Memory says (verify against primary): "The main `LogoMesh/KV-Cache-
  Inception` GitHub repo, CLAUDE.md, team briefs, `external/Logomesh`
  submodule, the `McGill` mention in `EXPERIMENTS.md`, and the
  `github.com/Logomesh` URL in `croissant_export.py:335` do NOT need
  to be anonymized — only the SUPPLEMENTARY COPY of these files (built
  Day 12) does." (memory/recast_action_plan.md preamble.)
  Answer:
    (a) Re-verify this scope claim against primary sources. Quote
        verbatim. If memory is wrong, correct it.
    (b) For the supplementary package specifically: what must be
        stripped (author names, emails, URLs containing identifying
        info, commit history with author info, README crediting
        team members, etc.)?
    (c) Are there published examples of "good" anonymized supplementary
        packages from recent ACL/EMNLP submissions we can model
        ours after?

Q3. ANONYMOUS CODE/DATA HOSTING — how does it actually work?
  Primary sources to read:
    - https://anonymous.4open.science/ (the service the paper already
      points to as placeholder at §A.5 with URL slug ...XXXX)
    - https://github.com/anonymous-academic — alternative anonymous-
      hosting service (verify if it exists/is recommended)
    - ARR CFP "Software repositories must be properly anonymized
      (e.g., Anonymous GitHub)" language — what tools does it
      recommend?
  Answer:
    (a) How does anonymous.4open.science work mechanically? Do you
        upload a zip? Do you point it at a public GitHub repo and it
        mirrors with author info stripped? Do you give it credentials
        for a private repo? What does the resulting URL look like?
    (b) Does it support arbitrary file structures (logomesh/ Python
        package + scripts/ + docs/ + Croissant JSON), or does it
        impose constraints?
    (c) Does it preserve directory structure / line-number stability
        (important if the paper references "logomesh/kv_mcts.py:78")?
    (d) Does the URL expire? Can it be updated post-submission (for
        camera-ready de-anonymization)?
    (e) Alternative path: create a NEW anonymized GitHub repo with
        fresh commit history (no author info), push, link from paper.
        Compare pros/cons vs anonymous.4open.science.
    (f) Recommended path: which option does the team adopt?

Q4. SCOPE OF SUPPLEMENTARY PACKAGE — what to include
  Primary sources:
    - acl_formatting.md §"Supplementary Material" (line 292+)
    - aclrollingreview.org/cfp (search for "supplementary" and
      "code")
    - The paper's §A Reproducibility appendix (docs/NeurIPS/
      04.18.2026-NeurIPS-Research-Proposal-2.tex lines ~466-493):
      what does the paper PROMISE the supplementary contains?
  Answer:
    (a) What does the paper promise will be in supplementary? Quote
        each promise verbatim from the TeX.
    (b) Minimum scope to honor those promises: source code? raw JSON
        outputs (scripts/_track_*.json)? Croissant metadata? README
        with reproduction commands? License? Reproducibility script
        bundle?
    (c) What about the model weights themselves (Llama 3.2-1B/3B)?
        Open weights — distributed by Meta with license. Do we
        re-distribute? Or just cite + provide HF hub link?
    (d) What about logs/track reports under docs/logs/? Are those
        supplementary (scientific documentation) or internal
        (project management)?
    (e) ACL formatting rule: "Submissions that misuse the
        supplementary material may be rejected without review."
        (acl_formatting.md:300). What constitutes "misuse"? Is
        including detailed proofs supplementary-OK? Is including
        track reports supplementary-OK? Where's the line?

Q5. RESPONSIBLE NLP CHECKLIST — final form + delivery
  Primary sources:
    - https://aclrollingreview.org/responsibleNLPresearch/
    - docs/NeurIPS/checklist.tex (if Josh has captured it locally)
    - memory says Josh's J3 captured 17 questions + first-pass answers;
      verify against current state
  Answer:
    (a) Is the checklist a separate PDF, a TeX appendix in the paper,
        or a form filled in at portal submission?
    (b) What's the final-form deliverable Josh needs Day-14/15?
    (c) Are answers anonymized too?

Q6. CITATION OF SUPPLEMENTARY IN PAPER
  Primary source: paper TeX §A.5 line ~492:
    "...available at \url{https://anonymous.4open.science/r/
     kv-mcts-dimensional-escape-XXXX}"
  Answer:
    (a) What replaces XXXX at submission? Is it a slug we choose? Is
        it assigned by the hosting service?
    (b) Timing: do we register/upload BEFORE we put the URL in the
        TeX, or do we upload first then patch the TeX?
    (c) Does the URL need to be live AT submission time, or can it be
        added during the review period?
    (d) Camera-ready (post-acceptance): does the anonymous URL get
        replaced with a public GitHub URL? If yes, can both URLs
        co-exist or is there a swap?

Q7. STALE-CONTENT INVENTORY — what's in this repo that shouldn't be?
  Run a discovery pass over the repository to catalogue:
    (a) Files with "NeurIPS" in name but no longer relevant (the
        paper is EMNLP via ARR May 2026, not NeurIPS). Notable:
        canonical TeX is named "04.18.2026-NeurIPS-Research-
        Proposal-2.tex" — that's the obvious one. List all such files.
    (b) Files referencing the (long-retired) Option α+ "alignment
        faking detection" framing. Per memory, the paper is now
        Option D+ "Dimensional Escape" framing. Internal docs in
        docs/logs/ that document the pivot are historical; what's
        the recent (post-2026-05-09) shape?
    (c) Phase-1 / Phase-2 development artifacts under scripts/,
        logomesh/, or docs/ that are no longer load-bearing for
        the submission. E.g., probe_kv_cache_mutability.py was a
        Phase-2 gate probe — is it still relevant or just
        historical?
    (d) Files referencing other LogoMesh projects (the broader
        LogoMesh codebase has competition-track history that was
        cleaned up but residue may remain — see CLAUDE.md hard
        rule 1: "No competition framing in code/comments/docs").
        Find any residue.
    (e) Multiple-version paper drafts (e.g.,
        04.18.2026-NeurIPS-Research-Proposal-2-premortem-v2.tex
        per memory). Which is canonical? Which are historical
        artifacts?
    (f) Build artifacts: .aux, .log, .out, .pdf, texput.log in
        docs/NeurIPS/. Add to .gitignore if not already?
    (g) docs/logs/ has 30+ files documenting Day 1 through Day-13
        of execution. Which are scientifically load-bearing
        (track reports, audit findings) vs project-management
        artifacts (session logs, scheduling docs)?

----------------------------------------------------------------------
PHASE 0 OUTPUT — Discovery Report
----------------------------------------------------------------------

Write a single report at:
  docs/logs/<today's-date>_submission-prep-discovery-report.md

Structure:
  # Submission Prep + Repo Cleanup — Discovery Report

  **Date:** YYYY-MM-DD
  **Phase:** 0 (Discovery only — no actions taken)

  ## TL;DR (≤200 words)
  [Bottom-line answer to each of Q1-Q7]

  ## Q1 — Submission Portal Mechanics
  [Detailed answer + primary-source citations]

  ... [Q2-Q7] ...

  ## Open questions for Josh
  [Anything that couldn't be answered from primary sources alone —
   e.g., "ARR FAQ page returned 404; can you confirm X?"]

After the report is written, REPORT TO JOSH:
  - Confirm the discovery report path
  - Surface the 3 most important unknowns / corrections to prior
    working assumptions
  - Ask Josh to read the report before you proceed to Phase 1.

STOP HERE. Do not proceed to Phase 1 until Josh explicitly says go.

----------------------------------------------------------------------
PHASE 1 — PLANNING (after Josh signs off on discovery report)
----------------------------------------------------------------------

Now that discovery is done, propose two coordinated plans:

(1) REPO CLEANUP PLAN — `docs/logs/<today's-date>_repo-cleanup-plan.md`
    Per finding from Q7, propose for each stale-content item:
      - File path
      - Action: RENAME / DELETE / MOVE / KEEP / FLAG
      - If RENAME: new name (justify per submission-context naming
        convention from Q1)
      - If DELETE: explain why no longer needed (git history
        preserves)
      - If MOVE: source → destination
      - Severity: SAFE (uncontroversial) / NEEDS-JOSH-SIGN-OFF
        (touches scientific content) / RISKY (e.g., deleting
        track-report; may need to copy to supplementary first)
    Per CLAUDE.md hard rule 5: any changes to logomesh/ or scripts/
    must run the code review agent before the change is final.

(2) SUPPLEMENTARY PACKAGE BUILD PLAN — `docs/logs/<today's-date>_
    supplementary-package-plan.md`
    Per findings from Q3-Q6, propose:
      - Curated subset of files to include
      - Anonymization steps per file (strip headers, sub names/
        emails/URLs)
      - Build script (or manual checklist)
      - Hosting target (anonymous.4open.science slug? new anon
        GitHub repo? OpenReview direct?)
      - URL patch into paper TeX §A.5 (replace XXXX)
      - Camera-ready de-anonymization plan (post-acceptance)

After both plans are written, REPORT TO JOSH:
  - Plan files paths
  - Top 5 decisions Josh needs to make (e.g., "anonymous.4open.science
    vs new anon GitHub repo — recommend X because Y")
  - Estimated wall time for execution

STOP HERE. Do not execute until Josh says go.

----------------------------------------------------------------------
PHASE 2 — EXECUTION (after Josh signs off on plans)
----------------------------------------------------------------------

Execute the cleanup plan and the supplementary-package build plan.
Write an execution log at:
  docs/logs/<today's-date>_repo-cleanup-execution-log.md

For each action:
  - Action taken (verbatim git command, mv, rm, etc.)
  - Verification (file exists? file gone? compile passes?)
  - Any deviation from plan + reason

After each batch of related actions, run:
  - Smoke-compile of paper TeX (verify cleanup didn't break references)
  - `git status` to surface unintended changes

After supplementary package is built:
  - Upload to chosen hosting (per Phase 1 plan)
  - Verify URL is live + content accessible
  - Patch paper TeX §A.5 URL placeholder XXXX with actual slug
  - Smoke-compile + verify
  - Commit + push (per usual workflow; do NOT push without Josh's
    explicit ask)

After Phase 2:
  - Report severity counts (RENAME: N, DELETE: N, MOVE: N)
  - Report supplementary package status (uploaded? URL?)
  - Report any remaining manual steps Josh must do (e.g., portal
    submission itself, Responsible NLP Checklist filing)

----------------------------------------------------------------------
HARD RULES
----------------------------------------------------------------------

1. DISCOVERY BEFORE PLAN BEFORE EXECUTION. No skipping phases. If
   you're tempted to "just rename this one obvious file" — STOP. Run
   discovery first, then plan, then execute.

2. NEVER delete or modify a file without Josh's plan approval. The
   bar for "uncontroversial" deletions (e.g., texput.log) is still
   Josh approval; surface in the plan.

3. PRIMARY-SOURCE FIDELITY for Phase 0. Quote URLs + verbatim text.
   Memory says X — verify against primary. Per memory/feedback.md
   2026-05-13 rule.

4. ANONYMITY-SCOPE PARANOIA. Per memory's 2026-05-12 correction:
   anonymization is for the SUBMISSION package, not the main repo.
   Don't accidentally start anonymizing the live LogoMesh/
   KV-Cache-Inception repo. The main-repo work is reorganization,
   not anonymization. Supplementary package is the anonymized copy.

5. CODE-REVIEW HOOK PRESERVED. Per CLAUDE.md hard rule 5, any
   modification to logomesh/ or scripts/ requires the code-review
   agent before completion. If the cleanup plan touches code, build
   that gate into the plan.

6. DON'T BREAK PAPER COMPILATION. After every file move/rename
   that affects paths the paper TeX references (e.g.,
   docs/NeurIPS/figures/exp2-mean-step-alpha.pdf), smoke-compile
   immediately. If broken, revert.

7. PUSH ONLY ON EXPLICIT REQUEST. Per CLAUDE.md, do not push to
   remote without Josh asking.

8. DON'T SUBMIT TO ARR YOURSELF. The actual portal submission
   (May 25 ARR deadline) is Josh's manual step. This session
   prepares everything FOR submission but doesn't submit.

----------------------------------------------------------------------
ANTI-OVERLOAD GUIDANCE
----------------------------------------------------------------------

- Phase 0 is the biggest chunk (~1-2 hours wall). Don't rush.
- Each Q in Phase 0 needs primary-source verification, not
  speculation. If a primary source is unreachable, say so explicitly.
- Phase 1 plans should be concrete enough that Phase 2 is mechanical
  execution.
- If you find yourself wanting to combine Phase 0/1/2, RESIST. The
  whole point of discovery-first is to avoid acting on wrong
  assumptions.

----------------------------------------------------------------------
BEGIN
----------------------------------------------------------------------

Start Phase 0. Open primary sources for Q1 first. Write to
docs/logs/<today's-date>_submission-prep-discovery-report.md as you
go. After all 7 questions answered, REPORT TO JOSH and WAIT.
```

---

## Why this prompt is structured the way it is

- **Discovery-first** is non-negotiable per Josh's directive. The questions about ARR/OpenReview/anonymous.4open.science workflow are genuinely unknown — speculating from memory has burned the team once (the 2026-05-12 anonymization-scope correction). Primary-source verification before planning is the only way to avoid repeating that.
- **Q7 (stale-content inventory)** is the long-tail discovery step. The auditor doesn't know what to delete until they've surveyed the repo. The inventory feeds the cleanup plan.
- **Plan-then-approve-then-execute** prevents "I'll just rename this one file" creep. File renames near submission time are higher-stakes than during normal development — a broken reference in the paper TeX is a re-compile + re-push at minimum, possibly a re-anonymize cycle.
- **Three artifacts** (discovery report + cleanup plan + supplementary plan) preserve the full audit trail. Even if the execution fails, the discovery + plan stay as durable documentation.
- **Hard rule 4 (anonymity-scope paranoia)** explicitly defends against the temptation to over-anonymize the live repo. Memory's 2026-05-12 correction is load-bearing.

## What this prompt does NOT do

- Does not submit the paper to ARR. That's Josh's manual portal step Day-15.
- Does not modify logomesh/ or scripts/ code logic. Code changes require the code-review-agent gate per CLAUDE.md hard rule 5; this prompt's scope is documentation + filenames + supplementary-package assembly.
- Does not decide between anonymous.4open.science vs a fresh anonymized GitHub repo unilaterally. Phase 0 surfaces the tradeoff; Phase 1 recommends; Josh decides.
- Does not de-anonymize anything. Camera-ready de-anonymization is post-acceptance (August 2026) and out of scope for this session.

## Estimated wall time

- Phase 0 (Discovery): 1-2 hours wall (primary-source reads + inventory)
- Josh review of discovery report: ~30 min
- Phase 1 (Planning): 30-45 min wall
- Josh review of plans: ~30 min
- Phase 2 (Execution): 1-2 hours wall (file renames + package build + upload + paper-URL patch + compile + push)

Total session-side: ~3-4 hours wall, plus ~1 hour of Josh's review/approval time.

## When to run this prompt

After both Day-12 sessions complete:
- Paranoid hallucination audit (Day-12 AM)
- Parallel-review-plan Round 1 (Day-12 AM/PM)
- Parallel-review-plan Round 2 (Day-13 AM)
- Synthesis (Day-13 PM)

Run this prompt Day-13 PM or Day-14 AM. By then:
- Paper content is frozen
- All adversarial findings are integrated
- Cleanup operates on stable paper state — no risk of cleaning up something that gets re-added

## Coordination with Day-14 work

Per plan v2 §2, Day-14 nominal tasks:
- Integrate co-author feedback (from Tianyu + Max + Alaa)
- Integrate adversarial findings
- Apply SEVERE/MAJOR patches
- Final REVISION-marker audit
- Double-blind compile check

This cleanup prompt is COMPATIBLE with Day-14 patches: cleanup happens to filenames + supplementary content; Day-14 patches happen to paper content. If both run same day, sequence: paper-content patches first, then cleanup (so cleanup operates on final paper TeX). If cleanup renames the canonical TeX file, all Day-14 paper-content patches use the new filename.
