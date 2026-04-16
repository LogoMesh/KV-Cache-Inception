# Session Log — 2026-04-16

Session focus: Execute the first substantive KV-MCTS run with Croissant artifact export, then record reproducibility and validation evidence.

---

## Run Kickoff

1. Objective:
   - Perform a substantive (10+ node) `scripts/run_kv_mcts.py` run.
   - Export a Croissant package from the produced runtime artifact.
   - Capture command, seed, artifact paths, and key metrics for teammate/status updates.

2. Planned command shape:
   - `uv run python scripts/run_kv_mcts.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --nodes <N> --depth <D> --branches <B> --seed <S> --output <json> --croissant-dir <dir>`

3. Status:
   - Completed.

---

## Execution Attempts

1. Attempt A (TinyLlama online path):
    - Command:
       - `c:/Users/Josh/Documents/LogoMesh-NeurIPS/.venv/Scripts/python.exe scripts/run_kv_mcts.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --nodes 15 --depth 3 --branches 3 --seed 20260416 --output ./tmp/runs/2026-04-16/mcts_tinyllama_substantive_seed20260416.json --croissant-dir ./tmp/runs/2026-04-16/croissant_tinyllama_substantive_seed20260416`
    - Outcome:
       - Did not complete. Model resolution required HuggingFace network access; run terminated with `KeyboardInterrupt` during download path.

2. Attempt B (Qwen offline path, first try):
    - Command:
       - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 c:/Users/Josh/Documents/LogoMesh-NeurIPS/.venv/Scripts/python.exe scripts/run_kv_mcts.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --nodes 15 --depth 3 --branches 3 --seed 20260416 --output ./tmp/runs/2026-04-16/mcts_qwen_substantive_seed20260416.json --croissant-dir ./tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416`
    - Outcome:
       - Did not complete. Polling via new terminal commands interrupted the active foreground process (`KeyboardInterrupt`).

3. Attempt C (Qwen offline path, async mode):
    - Command:
       - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 c:/Users/Josh/Documents/LogoMesh-NeurIPS/.venv/Scripts/python.exe scripts/run_kv_mcts.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --nodes 15 --depth 3 --branches 3 --seed 20260416 --output ./tmp/runs/2026-04-16/mcts_qwen_substantive_seed20260416.json --croissant-dir ./tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416`
    - Execution mode:
       - Running in persistent async terminal (`ID=5c269174-9054-474a-958a-52c2986a498f`) to avoid interruption.
    - Outcome:
       - Completed successfully.

---

## Substantive Run Completion Record

1. Successful command:
   - `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 c:/Users/Josh/Documents/LogoMesh-NeurIPS/.venv/Scripts/python.exe scripts/run_kv_mcts.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --nodes 15 --depth 3 --branches 3 --seed 20260416 --output ./tmp/runs/2026-04-16/mcts_qwen_substantive_seed20260416.json --croissant-dir ./tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416`

2. Runtime outcomes:
   - `PerLayerHonestyProjector calibrated: 29 layers, d=1536`
   - `MCTS complete: 15 nodes expanded`
   - `MCTS done in 1.8s. 16 nodes visited`
   - `Results saved -> tmp/runs/2026-04-16/mcts_qwen_substantive_seed20260416.json`
   - `Croissant package saved -> tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/metadata.json`
   - `Croissant records exported: 16`

3. Key metrics (from artifact JSON):
   - `model_id`: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
   - `n_nodes_visited`: `16`
   - `elapsed_seconds`: `1.8`
   - Top-node `mean_reward`: `0.432811`
   - Top-node `oei_score` (depth 1): `0.995935`
   - Top-node telemetry sample: `sigma_H_mean=1.0`, `rho_R_mean=0.714173`, `tds=0.024966`

4. Reproducibility metadata:
   - `seed`: `20260416`
   - `git_sha`: `203465dcff8a164cac3a55dbdbe5e2b2d0d699d6`
   - `git_dirty`: `true`
   - `run_started_utc`: `2026-04-16T07:07:42.201026Z`
   - `run_finished_utc`: `2026-04-16T07:10:07.370592Z`
   - `python_version`: `3.12.9`

5. Croissant package inventory:
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/interventions.csv`
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/metadata.json`
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/source_run.json`

6. Notes:
   - TinyLlama direct Hub path was not used for the final substantive run because this environment required online fetch at execution time.
   - The completed substantive run is therefore recorded against cached Qwen in strict offline mode.

---

## Architecture Review Brief Plan

Objective: prepare a focused  architecture brief, before scaling beyond the first substantive run.

### Message History:
[
redacted for privacy
]

### Suggested Docx Brief Outline (What else to Provide?)

1. Proposed repository architecture:
   - core modules and interfaces,
   - experiment runner flow,
   - evidence capture + Croissant export path.
2. First experiment set:
   - first substantive run configuration (already completed),
   - next 2--3 experiment increments to run immediately after review.
3. Review asks:
   - architecture risks,
   - missing abstractions,
   - experiment-order improvements ahead of first scaled runs.

### Run Artifacts to Reference in the Message and Docx

1. Substantive run JSON artifact:
   - `tmp/runs/2026-04-16/mcts_qwen_substantive_seed20260416.json`
2. Croissant package directory:
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/`
3. Croissant metadata:
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/metadata.json`
4. Croissant tabular export:
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/interventions.csv`
5. Source run snapshot in package:
   - `tmp/runs/2026-04-16/croissant_qwen_substantive_seed20260416/source_run.json`
6. Execution log reference:
   - `docs/logs/2026-04-16_session-log.md`

1. Send the docx.
2. Include the artifact paths above in a compact bullet list.
3. Add one-paragraph summary of what feedback is most needed before first scaled execution.

---

## Phase 3 Gate Attempt — Llama-3.2-1B-Instruct

HuggingFace access to `meta-llama/Llama-3.2-1B-Instruct` confirmed (Meta Llama 3.2 gating group, approved 2026-03-20).

### Gate sequence (EXPERIMENTS.md Chunk 0):

1. Step 1 — KV-cache mutability probe:
   - Command: `uv run python scripts/probe_kv_cache_mutability.py --model meta-llama/Llama-3.2-1B-Instruct --device auto`
   - Status: **PASSED**
   - Output: `gate_passed: true`, `mutable_in_place: true`, `reversible_with_copy_restore: true`
   - `max_logit_delta_after_mutation: 0.171875`, `max_logit_delta_after_revert: 0.0`
   - Device: `cuda`, dtype: `bfloat16`, model source: `huggingface_id_or_cache`

2. Step 2 — KV-MCTS smoke test (10 nodes):
   - Command: `uv run python scripts/run_kv_mcts.py --model meta-llama/Llama-3.2-1B-Instruct --nodes 10 --depth 3 --branches 2 --seed 20260416 --output ./tmp/runs/2026-04-16/mcts_llama_gate_seed20260416.json`
   - Status: **PASSED**
   - Output: `MCTS complete: 10 nodes expanded`, `11 nodes visited`, `elapsed=0.6s`
   - `steer_layer=8`, `residual_norm=0.00e+00` (exact reversal)
   - Top-node telemetry: `sigma_H=0.0032`, `rho_R=0.3622`, `TDS=0.0084`, `OEI=0.8830`, `reward=0.2144`
   - Artifact: `tmp/runs/2026-04-16/mcts_llama_gate_seed20260416.json`
   - Note: telemetry values are non-degenerate vs Qwen (sigma_H not saturated, OEI not 0.996), but still identical across all top-5 nodes — ρ_R fix (Chunk 1A) still required before Experiment 1 results are paper-valid.

3. Step 3 — Lipschitz drift validation:
   - Command: `uv run python scripts/measure_lipschitz_drift.py --model meta-llama/Llama-3.2-1B-Instruct --n-cycles 200`
   - Status: **PASSED** (after two bug fixes to the script — see below)
   - Results (200 cycles, alpha=0.5):

     | Cycle | Naive bf16 `||K_n-K_0||_inf` | FP32 accumulator |
     |-------|------------------------------|-----------------|
     | 1     | 1.56e-02                     | 0.00e+00        |
     | 50    | 1.28e+00                     | 0.00e+00        |
     | 100   | 2.84e+00                     | 0.00e+00        |
     | 150   | 4.41e+00                     | 0.00e+00        |
     | 200   | 5.97e+00                     | 0.00e+00        |

   - Naive drift grows linearly (~+1.56 per 50 cycles). FP32 accumulator exact at zero throughout.
   - Artifact: `drift_results.csv`
   - Script fixes applied:
     - Bug 1: `dk_t = torch.from_numpy(dk_raw).float()` created a CPU tensor; added `.to(k_naive.device)` for CUDA compatibility.
     - Bug 2: Naive path did independent add/sub pairs per iteration, reaching a fixed point instead of accumulating drift. Redesigned to apply N adds then N subtracts cumulatively — faithfully simulates reversal error after N cycles.
     - Bug 3: Windows cp1252 UnicodeEncodeError on `‖` and `ε` in summary print; replaced with ASCII equivalents.

### Gate Verdict: PASSED (all 3 steps)

All EXPERIMENTS.md Chunk 0 criteria met on `meta-llama/Llama-3.2-1B-Instruct`:
- KV-cache mutable in-place and reversible with exact zero residual ✓
- MCTS smoke test: 10 nodes, non-degenerate telemetry (sigma_H not saturated vs Qwen) ✓
- Lipschitz drift: FP32 accumulator exact (0.00e+00) across 200 cycles; naive bf16 grows linearly ✓

Remaining blocker before Experiment 1 paper-valid results: ρ_R raw dot product fix (EXPERIMENTS.md Chunk 1A).

### Follow-on:
- `docs/NeurIPS/2026-04-16_Architecture-Brief-for-Aladdin.docx` updated to reflect gate outcomes.

---

## Architecture Brief

- Created: `docs/NeurIPS/2026-04-16_Architecture-Brief-for-Aladdin.docx`
- Contents: 5 sections — repo architecture, first Qwen run summary (with flat-telemetry disclosure), known gaps, next increments, review asks for Aladdin.
- Status: **Draft — pending gate results before send.**

---

## Assistant-Driven Changes Summary

1. Files created/updated in this session:
   - Created `docs/logs/2026-04-16_session-log.md`.
   - Updated `docs/CLAUDE_CONTEXT_BRIEF.md` to index today's log as current active handoff record.
   - Created `docs/NeurIPS/2026-04-16_Architecture-Brief-for-Aladdin.docx` (architecture brief, repurposed as team brief).
   - Fixed three bugs in `scripts/measure_lipschitz_drift.py` (device mismatch, naive path design, Unicode crash).
   - Added `docs/logs/drift_results_llama_gate_2026-04-16.csv` (Theorem 1 gate validation data).
   - Added `python-docx` to `pyproject.toml`.

2. Validation outcomes captured:
   - Substantive runtime execution completed successfully (offline Qwen path).
   - Artifact checks confirmed output JSON + Croissant package files.
   - HuggingFace authentication confirmed (Llama-3.2-1B-Instruct access verified).
   - Phase 3 gate (Llama-3.2-1B-Instruct): **all three Chunk 0 steps passed**.

3. Commit status:
   - Committed and pushed: `c2428c8` → `LogoMesh/KV-Cache-Inception` (origin/main).

---

## Session Continuation — Paper Amendments and Documentation Audit

### Paper Amendments (`docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex`)

1. Three feedback edits applied, all marked with `\reviewedit{}` (blue in PDF) for advisor review:
   - **Edit A (LAT acronym):** Section 4.1 — "the LAT procedure" → "the Linear Artificial Tomography (LAT) procedure"
   - **Edit B (novelty hedge):** Contribution 2 — "We introduce the first..." → "We introduce what is, to our knowledge, the first..."
   - **Edit C (MCTS justification):** Section 2.4 — added paragraph explaining why GCG and gradient-based approaches are also discretely constrained, giving MCTS a structural justification. New `\bibitem{zou2023gcg}` (arXiv:2307.15043) added.

2. NeurIPS 2026 E&D double-blind formatting applied:
   - Removed `\documentclass[10pt]{article}` + `\usepackage[margin=1in]{geometry}` + `\usepackage{authblk}` (all hardline rule violations)
   - Added `\usepackage[eandd]{neurips_2026}`
   - `neurips_2026.sty` and `checklist.tex` extracted from the official NeurIPS 2026 template ZIP and placed in `docs/NeurIPS/`. Source: `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip`
   - Note: the project submission guide (reference 17) incorrectly cited an Overleaf URL — corrected as part of the documentation audit below.

3. New files created this session:
   - `docs/NeurIPS/2026-04-16_Cover-Note-for-Paper-Commenter.md`
   - `docs/NeurIPS/neurips_2026.sty`
   - `docs/NeurIPS/checklist.tex`

### Documentation Audit

Systematic review of all repo documentation with verifiable external claims, conducted against live NeurIPS sources and the official template.

| Document | Errors Found | Action |
|---|---|---|
| `NeurIPS Submission Guide (KV-CAche-Inception).md` | Ref 16: 2013 URL; Ref 17: wrong Overleaf URL; Ref 19: wrong workshop template; unverified platform size claims; overstated Croissant RAI desk-rejection claim | Accuracy notice added; refs annotated `[CORRECTED]`; size claims `[UNVERIFIED]`; RAI claim `[PARTIALLY VERIFIED]`; verified formatting claims `[VERIFIED 2026-04-16]`; missing >4GB sample requirement noted |
| `NeurIPS Submission Guide (general).md` | Ref 13: same wrong Overleaf URL; Ref 14: stale 2016 style-files URL | Accuracy notice added; refs annotated `[CORRECTED]` |
| `NeurIPS-Submission-Hardline-Rules-Agent-Contract.md` | AI guide cited as sole canonical source | Reference Source updated — live NeurIPS site and downloaded `.sty` now take precedence |
| `CLAUDE.md` | Phase 3 gate description named TinyLlama; gpt-oss-20b stated as confirmed available | Gate updated to record Llama-3.2-1B-Instruct pass; gpt-oss-20b annotated TBC |
| `docs/CLAUDE_CONTEXT_BRIEF.md` | Stale (pre-dated today's work) | Updated with gate record, paper amendments, audit summary, next actions |

**Key verified facts:**
- 135 tests currently collected (`uv run pytest tests/ --co -q`)
- E&D hosting: no per-platform size limits stated officially; >4GB datasets require a reviewer sample
- Croissant RAI fields are required ("must use"); desk rejection language in official page applies to dataset accessibility broadly
- `[eandd]{neurips_2026}` confirmed correct for E&D double-blind from official template ZIP

---

## Open Concerns — Next Session Agenda

The following concerns were raised at session close and should be addressed at the start of the next session.

### 1. Technical brief may overwhelm the paper commenter

The architecture brief (`docs/NeurIPS/2026-04-16_Architecture-Brief-for-Aladdin.docx`) is dense and implementation-focused. A paper commenter who has been providing feedback on the manuscript — not the codebase — may find a 5-section technical document with gate metrics and module maps harder to engage with than a short covering note that leads with what they already know (the paper) and points to the brief as supplementary context.

**Suggested action:** Draft a short cover message (3–4 sentences) to accompany the brief. The message should: (a) reference the specific paper comments that have been acted on, (b) note that the first gate has passed and a brief is attached for architectural context, and (c) frame the two or three review asks clearly so the commenter knows exactly what input is being requested.

### 2. Updated paper draft not yet prepared for the commenter

The paper commenter's feedback from the prior exchange has been noted (MCTS justification needs strengthening, "first to introduce" claim should be softened, acronyms need defining on first use). These edits have not yet been made to the canonical `.tex` file. Sharing the current draft without these changes would be inconsistent with the commitment made in the prior exchange.

**Suggested action:** Apply the three tracked feedback items to `docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex` before sending the paper to the commenter.

### 3. Paper does not yet conform to NeurIPS double-blind submission formatting

The canonical draft is not formatted to NeurIPS 2026 submission requirements. The hardline rules contract (`docs/NeurIPS/NeurIPS-Submission-Hardline-Rules-Agent-Contract.md`) specifies the required formatting constraints. Submitting or sharing a draft that does not conform risks signalling lack of submission readiness to reviewers and collaborators.

**Suggested action:** Apply NeurIPS double-blind formatting to `docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex` as a prerequisite to any external share. Key requirements to check: author/affiliation anonymisation, `\usepackage{neurips_2026}` (or equivalent), page limits, and reference formatting.

### 4. Two divergent `.tex` versions exist

The repository contains two versions of the paper:
- `docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex` — canonical draft
- `docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal-Verification-Overlay.tex` — working copy with in-document empirical verification callouts

These have been edited independently and may have diverged. Before applying formatting changes, the delta between the two files should be established so that verification callouts in the overlay are not lost, and so there is a clear canonical target for the NeurIPS submission.

**Suggested action:** At the start of the next session, diff the two `.tex` files, identify any content present in one but not the other, and decide whether to merge or maintain them as separate artefacts with explicit roles. Then apply formatting and feedback edits to the agreed canonical version only.

---

## Session Close — All Open Concerns Resolved

All four concerns from the prior agenda were addressed in full this session.

| Concern | Resolution |
|---|---|
| 1. Cover note for paper commenter | `docs/NeurIPS/2026-04-16_Cover-Note-for-Paper-Commenter.md` created — four paragraphs covering addressed feedback, gate results, artifact pointers, and three specific review asks |
| 2. Paper edits not yet applied | Three feedback edits applied to canonical `.tex` with blue `\reviewedit{}` markup for advisor review: LAT acronym, novelty hedge, GCG justification paragraph + `zou2023gcg` bibitem |
| 3. NeurIPS formatting not applied | Formatting applied: geometry/authblk removed, `\usepackage[eandd]{neurips_2026}` added; `neurips_2026.sty` + `checklist.tex` extracted from official ZIP into `docs/NeurIPS/` |
| 4. Two divergent `.tex` versions | Resolved: canonical (`04.02.2026-NeurIPS-Research-Proposal.tex`) confirmed as the sole submission target; overlay kept as reference-only with no changes needed |

### Final File State — Changed This Session

| File | Change |
|---|---|
| `docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex` | Three `\reviewedit` feedback edits; NeurIPS E&D formatting; `zou2023gcg` bibitem added |
| `docs/NeurIPS/neurips_2026.sty` | New — extracted from official NeurIPS 2026 template ZIP |
| `docs/NeurIPS/checklist.tex` | New — extracted from official NeurIPS 2026 template ZIP |
| `docs/NeurIPS/2026-04-16_Cover-Note-for-Paper-Commenter.md` | New — cover message for paper commenter |
| `docs/NeurIPS/NeurIPS Submission Guide (KV-CAche-Inception).md` | Accuracy notice added; refs 16/17/19 corrected; hosting size claims annotated `[UNVERIFIED]`; Croissant RAI claim annotated `[PARTIALLY VERIFIED]`; formatting claims `[VERIFIED 2026-04-16]` |
| `docs/NeurIPS/NeurIPS Submission Guide (general).md` | Accuracy notice added; refs 13/14 corrected |
| `docs/NeurIPS/NeurIPS-Submission-Hardline-Rules-Agent-Contract.md` | Reference Source section rewritten — live NeurIPS pages + downloaded `.sty` now take precedence over AI guide |
| `CLAUDE.md` | Phase 3 gate updated to record Llama-3.2-1B-Instruct pass; gpt-oss-20b annotated as TBC |
| `docs/CLAUDE_CONTEXT_BRIEF.md` | Updated with today's full record and next-session actions |

### Commit Status

Not yet committed. All changes above are unstaged. Next session should:
1. Run `uv run pytest tests/ -v` (must be 100% green before commit)
2. Commit all doc/paper changes
3. Begin ρ_R raw dot product fix (EXPERIMENTS.md Chunk 1A) — prerequisite for paper-valid Experiment 1 results

### Verified Facts Recorded for Future Sessions

- **Test count:** 135 tests collected as of 2026-04-16
- **NeurIPS template source:** `https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip` (not Overleaf)
- **E&D track package option:** `\usepackage[eandd]{neurips_2026}` for submission; `\usepackage[eandd, final]{neurips_2026}` for camera-ready
- **E&D track hosting:** no official per-platform size limits; datasets >4GB require a small reviewer sample
- **Croissant RAI:** required ("must use"); core fields auto-generated by preferred platforms; RAI fields must be added manually
- **Two `.tex` files:** canonical is `04.02.2026-NeurIPS-Research-Proposal.tex`; overlay is reference-only
