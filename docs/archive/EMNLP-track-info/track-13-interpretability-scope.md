# EMNLP 2026 Track #13 — Interpretability and Analysis of Models for NLP

**Folder purpose:** store EMNLP 2026 track-info reference material for the submission. This file is the first entry; Josh will populate the rest of the folder in a separate session.

**Submission target:** ARR May 2026 cycle → EMNLP 2026 commitment binding at submission. Track-selection synthesis 2026-05-14 locked #13 as primary (J6c in `docs/logs/2026-05-10_14-day-execution-plan.md` §6).

---

## Primary sources

| Resource | URL | What it contains |
|---|---|---|
| EMNLP 2026 main conference call | [https://2026.emnlp.org/calls/main_conference_papers/](https://2026.emnlp.org/calls/main_conference_papers/) | Verbatim list of 29 subject-area tracks + 1 special theme. Track names only — no scope descriptions on this page. |
| ARR area keywords | [https://aclrollingreview.org/areas](https://aclrollingreview.org/areas) | The actual operational scope per track, expressed as keyword lists. **This is the authoritative scope source.** No fragment anchors — single-page format; search within page for the track name. |

## Track #13 — verbatim scope from ARR areas page

**Track heading (verbatim):** `Interpretability and Analysis of Models for NLP`

**Scope (verbatim keyword list, 2026-05-14 fetch):**

> adversarial attacks/examples/training; calibration/uncertainty; counterfactual/contrastive explanations; data influence; data shortcuts/artifacts; explanation faithfulness; feature attribution; free-text/natural language explanations; hardness of samples; hierarchical & concept explanations; human-subject application-grounded evaluations; knowledge tracing/discovering/inducing; model editing; probing; robustness; topic modeling

## How the paper's facets map to the keyword list

(Cross-walk used in Day-5 track-selection decision; see `docs/logs/2026-05-13_session-log.md` "Track-selection decision" entry for full reasoning.)

| Paper facet | ARR keyword(s) mapped |
|---|---|
| Per-layer telemetry matrix (σ_H + ρ_R) | "feature attribution"; "probing" |
| RepE honesty-probe alignment (Zou 2023) | "probing" |
| H-Neuron stress (bottom-up activation magnitudes) | "feature attribution" |
| Latent Cartography across 4 prompt classes | "knowledge tracing/discovering/inducing"; "data shortcuts/artifacts" |
| Track F entropy-MCTS negative control | "data shortcuts/artifacts" (Goodhart in cache geometry as a data-shortcut analog) |
| Memory-bound theorem (Theorem 1) | Not a direct keyword match — this is the algorithm/infrastructure half (the secondary-track candidate #6 "Efficient Methods for NLP" carries this) |

## Primary-source-fidelity discipline note

Per `memory/feedback.md` R4 (verify URLs/paths/refs/labels from primary sources before action), this content is a snapshot at 2026-05-14. The ARR areas page may update before the 2026-05-25 ARR submission deadline. Re-verify the track heading + keyword list immediately before final submission (Day 14 = 2026-05-23) and update this file if the source has drifted.

## What's NOT here (for the future-populate-this-folder session)

Josh has flagged he'll populate the rest of this folder in a separate session. Likely candidates:
- Track #6 "Efficient Methods for NLP" scope (the synthesis recommended secondary track if ARR portal supports it).
- Track #17 "Machine Learning for NLP" scope (the rejected primary, useful for context).
- Special Theme "New Missions for NLP Research" scope.
- Per-track example-paper lists from the Day-5 deep-research reports.
- ARR submission checklist + form-field map.
- OpenReview portal screenshots / mechanics notes.
- Cover-letter draft for the Day-14 final submission.
