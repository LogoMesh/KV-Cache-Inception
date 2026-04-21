# Session Log — 2026-04-18 (Session 2)

Session focus: PR #1 review (Max's batch dataset collector) + meeting debrief.

---

## Meeting Debrief — April 18, 2026

Source: `docs/04-18-2026 KV Cache Inception Meeting.srt`

### What happened

Max demonstrated the batch collector script live. The meeting's main substantive moment was
when a new contributor (the person being referred to as "Mark" — real name Kuan, website:
https://kuanzhou.online/) joined, said he wanted to run the experiments, and asked Josh for
a repo tour. Mark has an RTX 5090, so he can run everything locally.

Josh walked him through the repo on screen. Mid-tour, Josh said:

> "I think the naming of the scripts is a little confusing."

He then made a direct commitment to Mark before ending the meeting:

> "I'll make sure that all the scripts are very clear about how they tie to the specific
> sections of the paper and where to look in the paper for each script. And then I will
> give you the go ahead Mark — that should be done tonight."

That commitment was not yet fulfilled. This log records what needs to happen before
the go-ahead is given.

---

### The core problem confirmed by the meeting

Mark's first question after the tour was: "If I want to run the experiments, where should
I start?" Josh could not answer directly. He tried to find `run_kv_mcts.py` in the paper
and couldn't locate a clear mapping. The naming confusion is real and was confirmed live.

Two separate problems compound each other:

**Problem 1 — Naming system mismatch: repo phases vs. paper phases**

The README uses a Phase 1 / Phase 2 / Phase 3 / Phase 4 numbering system. The paper uses
Phase 0 / Phase A / Phase B. There is no bridge between them anywhere in the repo. A new
contributor reading the README and the paper simultaneously is immediately disoriented.

| README | Paper |
|---|---|
| Phase 1 (repo cleanup) | Phase 0 (architectural decoupling) |
| Phase 2 (Reversible KV-MCTS) | Phase A partial + Phase B setup |
| Phase 3 (Experiments 1-5) | Phase B (Experiments 1-5 on H100) |
| Phase 4 (dataset) | implicit in Phase B output |

These are not the same system. The README uses a dev milestone numbering; the paper uses a
scientific execution sequence. A reader of the README cannot find where they are in the paper.

**Problem 2 — Scripts have no paper anchors**

Every script in the repo calls itself something from the dev lifecycle, not from the paper:

| Script | Dev label in README | Paper section / experiment |
|---|---|---|
| `probe_kv_cache_mutability.py` | "Phase 2 gate" | Phase A: KV-cache mutability validation (Section 5.2 prerequisite) |
| `run_kv_mcts.py` | "Phase 2 runner" | Phase A smoke test + Phase B Experiments 1, 2, 5 (inputs vary) |
| `measure_lipschitz_drift.py` | "Theorem 1 validation" | Phase A: FP32 reversibility proof (Theorem 1); also feeds Exp 3 |
| `train_lat_probes.py` | — (no label in README) | Phase A prerequisite: LAT probe training for RepE (Section 5.2) |
| `run_offline_mcts.py` | "Phase A offline MCTS" | Phase A baseline; Experiment 2 comparison baseline |
| `export_kv_mcts_to_croissant.py` | "Convert runtime JSON artifacts" | Dataset packaging (Section 6) |
| `collect_dataset.py` | (new, no README entry yet) | Dataset assembly for NeurIPS submission (Section 6) |

No script has a docstring or comment pointing to a paper section. A new contributor cannot
tell whether to run `run_kv_mcts.py` for Experiment 1 or Experiment 2.

**Problem 3 — Missing scripts for paper experiments**

Two experiments named in the paper have no corresponding script:

- **Experiment 3 (Memory Efficiency):** "We measure KV-cache memory footprint as a function
  of search depth, branching factor, and model size." `measure_lipschitz_drift.py` validates
  FP32 accumulator drift, which is related but different. There is no memory benchmarking
  script.
- **Experiment 5 (Cross-Model Transfer):** "Procrustes alignment of steering vectors across
  model scales." There is no Procrustes script. CLAUDE.md mentions it as "Phase 3" work
  not yet built.

Both gaps are known (they are in the experiment-readiness doc as pending). But they need
to be clearly called out before handing over the repo.

**Problem 4 — README has teammate names**

The README `## Team` section lists teammates by name. Per the handover policy: docs going
to a new contributor must not name other teammates. This section needs to be removed or
anonymized before the repo is handed over.

**Problem 5 — README canonical paper path is stale**

README points to `docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex`. The canonical
paper has been `04.17.2026-NeurIPS-Research-Proposal.tex` since the April 17 session.

**Problem 6 — Meeting transcript must be removed before handover**

`docs/04-18-2026 KV Cache Inception Meeting.srt` is now tracked in the repo. It contains
informal team conversation including internal uncertainty about project state. This must be
removed from the repo before Mark is given access.

---

### What Mark offered / what he needs from us

Mark said he would read the docs and the paper. He offered citation validation and wants to
run experiments. He did not express frustration at the meeting — he was patient and
constructive. He is a good-faith contributor who will arrive at the repo cold, with no
context beyond what is documented.

What he needs to be productive on day one:
1. A clear map: which script runs which experiment, with paper section references
2. Know which experiments are runnable now (Phase A on his RTX 5090) vs. H100-only
3. Know what experiments are not yet implemented (Exps 3 and 5 in Phase B)
4. Know the current experimental state: "we are still at step one" — 1B model only,
   no H100 data yet, thresholds uncalibrated
5. No teammate names anywhere; no transcript

---

### Plan of action (multiple passes, not yet executing)

This log records the problems. We are not yet fixing them. Awaiting Max's briefing document
which covers the same ground from his perspective. After that, several passes in order:

**Pass 1 — Transcript removal + README names**
Remove or git-rm the .srt file. Remove `## Team` section from README. Fix canonical paper
path in README. These are the only changes that need to happen before any other pass.
Do not hand over the repo until these are done.

**Pass 2 — Script-to-paper mapping**
For each script, add a header comment block pointing to the relevant paper section and
experiment number. Alternatively, add a `## Script Index` section to the README that
provides this mapping in one place. This was the specific commitment made at the meeting.

**Pass 3 — Phase naming reconciliation**
Either: rename README phases to match paper phases (Phase 0/A/B), or add an explicit
cross-reference table that bridges the two systems. Do not silently maintain two parallel
naming conventions.

**Pass 4 — Experiment status table**
Add a section (likely to the README or experiment-readiness doc) that states plainly for
each experiment: what script runs it, what hardware it requires, whether it is implemented,
and what its current status is.

**Pass 5 — Paper-repo consistency audit**
A careful reading of the paper's experimental design sections against the current scripts
to verify that the scripts actually measure what the paper claims they measure. This was
Josh's concern at the meeting: "I just want to make sure that the scripts do everything
that the paper says the scripts will do if you catch my drift."

---

### One other thing from the meeting

Josh mentioned at the end that he is also juggling the Agent Beats Phase 2 competition
(sprint 2 and 3 deadline May 3rd, conflicts with NeurIPS May 4th abstract + May 6th
submission). Decision made: drop multi-track ambition, focus on cybersecurity track only
and the paper. This is relevant context — resources are constrained and the paper is
the priority.

---

## PR #1 Review — `feat: batch dataset collector with auto-collect and incremental append`

**Author:** max-toscano  
**Branch:** `feature/batch-collector-incremental`  
**URL:** https://github.com/LogoMesh/KV-Cache-Inception/pull/1  
**Opened:** 2026-04-19T03:33:40Z  
**Additions:** 1336 lines, 0 deletions

### What the PR does

Solves the missing link in the dataset pipeline: `croissant_export.py` could convert a
single MCTS run artifact into a Croissant package. But NeurIPS needs one unified package
across all experiment runs. This PR adds:

| File | Purpose |
|---|---|
| `scripts/collect_dataset.py` | Batch collector: N artifacts → 1 Croissant package |
| `logomesh/croissant_export.py` | +`build_batch_metadata_document()` for multi-source metadata |
| `scripts/run_kv_mcts.py` | +`--auto-collect`, `--dataset-dir` flags for post-run collection |
| `tests/test_collect_dataset.py` | 10 unit tests for the new pipeline |
| `docs/logs/batch-collector-progress.md` | Max's progress summary |

Key design choices:
- **Safe by default:** auto-detects whether to build or append based on manifest presence.
  No `--append` flag; `--full-rebuild` is the only destructive option and must be explicit.
- **SHA-256 dedup:** artifacts already in the manifest are skipped on incremental runs.
- **Rich manifest:** each `manifest.json` entry stores model_id, seed, node_count,
  timestamps, git_sha, and command — enough to distinguish TinyLlama smoke tests from
  real H100 runs at a glance.
- **Additive only:** no existing `croissant_export.py` interfaces changed.

End-to-end verified by Max on RTX 3060 / TinyLlama. 145/145 tests green on his branch.

---

### Issues Found

#### Required before merge

**Issue 1 — `full_rebuild` leaks stale source artifacts**

Location: `scripts/collect_dataset.py`, `collect_dataset()` function (~line 825–843)

When `full_rebuild=True`, `dataset_exists` is forced to `False`, the manifest resets to
`{"entries": []}`, and a fresh build proceeds. The CSV (`interventions.csv`) and
`metadata.json` are overwritten, but `sources/` is never cleaned. Old
`source_run_001.json` through `source_run_NNN.json` from the previous build persist on
disk and become unreferenced from the new manifest.

The test `test_full_rebuild_overwrites` only checks CSV row count and manifest entry
count — both pass — but doesn't verify that the sources directory was emptied. The test
gives a false green.

**Fix:**
```python
# At the start of the full-build path (else branch), before any writes:
if full_rebuild and sources_dir.exists():
    shutil.rmtree(sources_dir)
    logger.info("Full rebuild: removed existing sources directory")
```

Also update `test_full_rebuild_overwrites` to assert that only `source_run_001.json`
exists in `sources/` after a rebuild (no `source_run_002.json` etc.).

---

**Issue 2 — Branch needs rebase on `main` before merge**

Max branched before commit `b7944bc` (Items 8+10: `t_matrix_json` + `ClassifyConfig`),
which also modifies `logomesh/croissant_export.py`. There will be a conflict or merge
gap in that file.

Evidence: Max's progress doc reports "144/144 full suite green." Main currently has 139
tests (post-`b7944bc`). With 10 new tests from this PR, merged total should be 149. The
count of 144 or 145 in the progress doc reflects a branch state without `b7944bc`.

**Fix:** Max rebases `feature/batch-collector-incremental` on current main, resolves
any conflict in `croissant_export.py` (his addition goes at the bottom after the new
`t_matrix_json` code), confirms 149/149 tests green, force-pushes branch.

Alternatively, Josh/Claude can apply the fix locally and commit directly to main after
squash-merging — see Merge Plan below.

---

#### Minor (fix before merge or in follow-up — team's call)

**Issue 3 — Duplicate SHA-256 helper**

`collect_dataset.py` defines `_sha256_of_file()` locally (~line 530) AND imports
`_sha256_file` from `croissant_export`. Both are identical. The local function is used
in `filter_new_artifacts()` and `copy_source_artifacts()`; the import is used for the
CSV hash. The local function can be removed and all call sites switched to `_sha256_file`.

Note: `_sha256_file` starts with an underscore (private). Importing private functions
across module boundaries is slightly fragile but acceptable here since both are in the
same repo.

**Issue 4 — Cross-script import in `run_kv_mcts.py` (line ~317)**

```python
from scripts.collect_dataset import collect_dataset
```

`scripts/` has no `__init__.py` — this relies on Python 3.3+ namespace packages and
the repo root being in `sys.path`, which both scripts set up. Works in practice but
breaks silently if either script's `sys.path` setup changes. Not blocking.

**Issue 5 — `docs/logs/batch-collector-progress.md` location**

The `docs/logs/` directory is for date-stamped session logs. A feature progress doc
doesn't belong there. Options: move to `docs/NeurIPS/`, or delete it since the PR
description + test docstrings cover the same ground.

**Issue 6 — Test count inconsistency in progress doc**

Progress doc says "9/9 unit tests" in the status table, but `test_collect_dataset.py`
has 10 test functions. One was added late without updating the count. Trivial.

---

### What's correct and should not change

- `build_batch_metadata_document()` wrapper pattern: builds skeleton from existing
  function, replaces distribution list. Right call given the constraint of not
  modifying the existing function's contract.
- ID collision behavior: warn, don't drop. Correct default for research data.
- Auto-detect build vs append (no `--append` flag). Right UX.
- `--auto-collect` opt-in on `run_kv_mcts.py`. Preserves existing behavior.
- Manifest `run_detail` schema. Exactly what's needed for submission-time provenance.
- Label join integrated but optional (`--labels`). Correct — code is ready, data
  doesn't exist yet.

---

### Verdict

**Approve with two required fixes (Issues 1 and 2) before merge.**
Issues 3–6 are minor; 3 and 6 are worth fixing in the same pass since they're trivial.

---

## Merge Plan

All fixes can be applied locally and pushed to Max's branch (preferred — keeps git
history clean), or Josh/Claude can apply them after a squash merge to main. The local
fix approach is described below.

### Option A — Fix on Max's branch, then merge (preferred)

**Step 1 — Josh asks Max to rebase and fix Issue 1**

Request from Max:
1. Rebase `feature/batch-collector-incremental` on current `main` (resolves
   `croissant_export.py` conflict — his `build_batch_metadata_document()` goes after
   all existing code at the bottom of the file).
2. Fix `full_rebuild` sources leak: add `shutil.rmtree(sources_dir)` at the start of
   the full-build path.
3. Update `test_full_rebuild_overwrites` to assert the sources directory was cleaned.
4. Optionally: remove `_sha256_of_file()` duplicate (Issue 3) and fix test count in
   progress doc (Issue 6).
5. Run `uv run pytest tests/ -v` — confirm 149/149 green.
6. Force-push to `feature/batch-collector-incremental`.

**Step 2 — Claude applies minor fixes if Max doesn't (Issues 3, 5, 6)**

After Max pushes, Claude can clean up Issues 3, 5, 6 in a follow-up commit on the
branch or immediately after merge on main. None are blocking.

**Step 3 — Merge**

```bash
gh pr merge 1 --squash --subject "feat: batch dataset collector with auto-collect and incremental append"
```

Or rebase-merge to preserve Max's commit history.

---

### Option B — Josh/Claude applies all fixes, squash-merges

If Max is unavailable before the deadline:

1. Fetch Max's branch locally:
   ```bash
   git fetch origin feature/batch-collector-incremental
   git checkout -b fix/batch-collector origin/feature/batch-collector-incremental
   ```

2. Rebase on main:
   ```bash
   git rebase main
   # Resolve croissant_export.py conflict: keep both our t_matrix_json addition
   # and Max's build_batch_metadata_document() addition at the bottom
   ```

3. Apply Issue 1 fix (`collect_dataset.py` — add `shutil.rmtree` in full_rebuild path).

4. Update `test_full_rebuild_overwrites` to assert sources/ cleaned.

5. Apply Issue 3 fix (remove `_sha256_of_file` duplicate).

6. Fix Issue 6 (test count in progress doc).

7. Run tests: `uv run pytest tests/ -v` — confirm 149/149.

8. Push and merge.

---

### Files touched by the fix pass

| File | Change |
|---|---|
| `scripts/collect_dataset.py` | Issue 1: add `shutil.rmtree(sources_dir)` in full_rebuild path; Issue 3: remove `_sha256_of_file`, update call sites to `_sha256_file` |
| `tests/test_collect_dataset.py` | Issue 1: add sources/ dir assertion to `test_full_rebuild_overwrites`; Issue 6: update test count comment if present |
| `docs/logs/batch-collector-progress.md` | Issue 5: move to `docs/NeurIPS/` or delete |

No changes to `logomesh/croissant_export.py`, `scripts/run_kv_mcts.py`, or any tests
beyond the one assertion addition — the logic is correct.

---

## Post-Merge State

After merge, the full pipeline from experiment run to submission-ready Croissant package
will be one command:

```bash
uv run python scripts/run_kv_mcts.py \
    --model openai/gpt-oss-20b \
    --nodes 500 --depth 5 --branches 3 \
    --seed 20260501 \
    --auto-collect --dataset-dir ./docs/dataset
```

The `./docs/dataset/` directory becomes the living NeurIPS submission artifact, growing
with every H100 experiment run.

Remaining dataset pipeline items (none require code changes to `collect_dataset.py`):

| Item | Blocked on |
|---|---|
| Ground truth annotation CSV | Phase 3 experiment data |
| `mlcroissant` strict validation on batch package | One-time run after first H100 collect |
| HuggingFace upload script | Final dataset + hosting decision from meeting |
| Dataset card (README) | Final dataset contents |

---

## Checklist State at Session Open

| # | Task | Status |
|---|---|---|
| 7 | OpenReview profiles | In progress (team action) |
| 8 | H100 smoke test + recalibrate thresholds | Prep ✅; recalibration needs H100 |
| 9 | mlcroissant strict validation | ✅ Done (single-run); batch validation pending merge |
| 10 | T_t matrix-level export | ✅ Done |
| 11 | Saturday meeting: fill Priority table | **Done at meeting — debrief pending** |
| 12 | Batch dataset collector | PR open; 2 required fixes before merge |
