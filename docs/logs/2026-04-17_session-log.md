# Session Log — 2026-04-17

Session focus: Honest project assessment; structural critique added to paper revision brief
ahead of Saturday April 18 team meeting.

---

## Opening Assessment — State of the Project

*This section records an honest outside-view read of the project as of April 17, 2026 —
19 days before the May 6 NeurIPS submission deadline.*

### What is genuinely strong

The core technical idea is clean: MCTS operating on continuous KV-cache tensors rather than
discrete token sequences is a meaningful departure from prior work, and the comparison to
text-space red-teaming approaches is defensible against the cited literature. The reversibility
proof (FP32 accumulators, O(M + d·K) memory) is mathematically sound and experimentally
confirmed by the gate run. The Croissant/RAI export infrastructure is production quality —
probably better than most E&D submissions will have. The writing is sophisticated and the
framing is ambitious in a way that reads well to reviewers who care about the field.

### The central problem

There is a significant mismatch between where the paper reads and where the experimental
work actually is. The paper is written as if the experiments are done or nearly done.
They are not. Five experiments are described in careful detail; none have been run at scale.
The only empirical evidence is:
- A 15-node smoke test on a 1B model (Qwen)
- A 10-node gate run on Llama 1B confirming KV-cache reversibility and non-degenerate telemetry

This is a working prototype, not a validated system. Claims like "enables deep MCTS on a
single 80GB GPU for models up to 20B parameters" are stated as present facts. They haven't
been run. The OEI signal on Qwen saturated at ~0.996 — which suggests the metric is not
discriminative in the current regime. The ρ_R implementation has a known bug (normalized
instead of raw dot product per Eq. 4), meaning even the small-scale data that exists is
technically incorrect per the paper's own equations.

The Orthogonal Escape formalization is presented as a third contribution. The paper honestly
states in Limitations that it "has not yet been empirically validated." That honesty is correct
and necessary — but it also means that one of the three headline contributions is an unvalidated
hypothesis with math. NeurIPS reviewers will notice.

### The timeline problem

May 6 is 19 days away. Between now and then, the following need to happen to make the paper's
claims land:
1. Fix ρ_R bug (Chunk 1A)
2. Get H100 access from Tianyu's lab
3. Run 5 experiments at meaningful scale
4. Process results, compute statistics
5. Write results sections in the paper
6. Integrate data into Croissant packages
7. Revise all present-tense claims to match actual results
8. OpenReview profiles for all four co-authors
9. Anonymization pass on paper and dataset
10. Final formatting and PDF assembly

That is a full project's worth of work in 19 days for a part-time team.

### Three structural options for the Saturday meeting

These are offered as framing for the discussion, not as directives. The team knows the
constraints (thesis timelines, advisor availability, H100 access windows) better than anyone.

**Option A — Focused scope: two contributions, three experiments**

Drop the Orthogonal Escape formalization to "emerging observation / future work." Keep
contributions 1 (T_t matrix) and 2 (Reversible MCTS). This means the experiments become:
- Experiment 1: alignment detection under MCTS (core result)
- Experiment 3: ablation (which telemetry channel carries the signal)
- Experiment 4: calibration overhead (practical feasibility on H100)

Experiments 2 (OE characterization) and 5 (Procrustes cross-model transfer) become
future work explicitly. The OE section of the paper says: "we observe a pattern consistent
with orthogonal redistribution; quantifying it systematically is ongoing work." This is honest
and the remaining claims are achievable in the timeline. The paper is smaller but defensible.

**Option B — Lead with the dataset, reframe method as methodology**

The E&D track explicitly wants datasets and evaluation protocols, not algorithm papers.
Reframe the contribution as: "we release a dataset of N alignment stress-test traces with
full telemetry, generated via a novel evaluation methodology (KV-MCTS), with Croissant 1.1/RAI
metadata." The MCTS method becomes the instrument that generated the dataset. The T_t matrix
and OEI/TDS are the evaluation metrics the dataset enables. This framing has the advantage of
making the Croissant infrastructure a feature rather than a footnote, and it puts the dataset
front-and-center for E&D reviewers. The risk: the dataset at current scale (27 nodes) needs to
be substantially larger before this framing is credible.

**Option C — Target a later deadline**

If honest assessment says 19 days cannot produce five experiments with statistical significance
for a 20B model, then submitting a weaker paper to NeurIPS 2026 may be worse than submitting
a strong paper to ICLR 2027 or NeurIPS 2027. This depends entirely on team circumstances
(thesis chapter deadlines, advisor expectations, whether a workshop submission counts). Not
raising this to recommend it, but raising it because it should be on the table at the meeting
rather than assumed off the table.

### The OEI signal quality issue

The Qwen gate run showed OEI saturating at ~0.996 across all non-root nodes. This is worth
thinking through carefully. Three possible explanations:

1. **α is too large** — the perturbation is so strong that the model's representation is
   completely disrupted regardless of content. This would mean the metric is responding to
   perturbation magnitude, not to meaningful alignment-faking signal.
2. **Qwen 1.5B is too small** — at this scale, KV-cache interventions may produce chaotic
   rather than structured responses, saturating any orthogonality metric.
3. **OEI is correctly identifying that all intervened states are orthogonally escaped** —
   which would be a meaningful finding, but requires comparison to a baseline where no
   alignment-faking pressure is present.

The Llama gate run showed OEI = 0.883, which is less saturated and more discriminative.
This is a better signal. The H100 experiments should prioritize finding α ranges where OEI
varies meaningfully across nodes — otherwise the core diagnostic metric is not working as
claimed.

### The ρ_R bug is more important than it looks

The ρ_R normalization bug (returning values in [0,1] instead of the raw dot product per Eq. 4)
affects not just the current data but the diagnostic state classifier, the reward function
(Eq. 8), and the OEI/TDS calculations — all of which consume ρ_R as input. This means the
telemetry matrix T_t is not currently implemented as specified in the paper. Before any
experiment that claims to validate the paper's framework, this must be fixed and the affected
downstream calculations verified. It is correctly the first item in the Chunk 1A roadmap.

---

## Actions This Session

- Created `docs/logs/2026-04-17_session-log.md` (this file)
- Updated `docs/NeurIPS/paper-revision-claims-audit.md` with a new Section 0 covering
  structural concerns and scope options for the Saturday meeting
- Fixed ρ_R raw dot product bug (Chunk 1A) — commit `4bc8fa9`
- Created `docs/NeurIPS/experiment-readiness-and-paper-revision.md` — commit `f17b365`
- Commissioned and reviewed external literature survey on prior latent-space MCTS work
  (`docs/NeurIPS/LLM Latent Space Search Literature Review.md`)
- Rewrote Contribution 2 framing and updated Related Work based on survey findings — commit `e1df154`:
  - Removed "to our knowledge, the first MCTS in latent space" (false: COCONUT, CRSM precede it)
  - Reframed as first MCTS via continuous arithmetic KV-cache mutations with FP32 rollback
  - Qualified "no prior adversarial latent search" (false: Bailey et al., RL-Obfuscation precede it)
  - Fixed GCG rhetoric ("structurally cannot" → "are not designed to navigate")
  - Added new Section 2.5 citing and distinguishing COCONUT, CRSM, Okazaki-RAG
  - Added 4 bibliography entries

---

---

## ρ_R Bug Fix — Chunk 1A (commit 4bc8fa9)

### Problem

`PerLayerHonestyProjector.project()` in `logomesh/whitebox.py` was returning a normalized
[0,1] value via `(proj + 1.0) / 2.0` clipped to [0,1]. This assumed the hidden state `h`
was unit-normalised (i.e., computing cosine similarity). Hidden states from LLaMA/Qwen are
NOT unit-normalised, so the transformation was semantically wrong and produced arbitrary values
in [0,1] that did not correspond to the paper's Eq. 4.

The T_t matrix, diagnostic state classifier, MCTS reward function (Eq. 8), OEI, and TDS were
all computed using this incorrect ρ_R signal. All gate run data (seed 20260416 and the Phase 3
gate run) was produced with the incorrect implementation.

### Fix

Three files changed:
- `logomesh/whitebox.py`: return `float(np.dot(w, h_np))` directly; fallbacks `0.5 → 0.0`
- `logomesh/kv_mcts.py`: fallbacks `0.5 → 0.0` for rho_R padding
- `logomesh/telemetry_matrix.py`: docstring updated (repe_honesty no longer [0,1]); note
  added to `classify()` that rho thresholds need re-calibration on real H100 data

### Implication

The existing gate run artifacts in `tmp/runs/2026-04-16/` were produced with the wrong
implementation. They should not be cited as paper-valid ρ_R results. The fix must be applied
before any experiment run that will appear in the paper. A re-run of the gate smoke test
with the corrected implementation should be done at the start of the H100 session.

### Tests

135 passed (no regressions). The existing tests use fake projectors with dummy [0,1] values;
they are not affected by the raw dot product change.

### Next step

Run gate smoke test on H100 with corrected implementation to establish what the actual raw
ρ_R distribution looks like, then re-calibrate classify() thresholds accordingly.

---

## Suggested Next Steps (Post-Compact)

### Combined Pre-flight + Paper Revision Document

**Problem:** Two related issues identified this session:
1. The paper reads in present tense as if experiments are already complete — they are not.
2. There is no pre-flight checklist to ensure experiments are set up correctly before H100 runs.

**Insight:** These are the same problem from different angles. Each experiment, when it passes,
unlocks specific present-tense claims in the paper. The right solution is one combined document
that chains implementation readiness → experiment execution → paper revision in a single
trackable artifact.

**Proposed document:** `docs/NeurIPS/experiment-readiness-and-paper-revision.md`

Structure for each of the 5 experiments:
1. **Pre-flight gate** — what must be true in the implementation before running (e.g., ρ_R
   corrected, α calibrated, LAT probes trained)
2. **What it produces** — the artifact, metric, and expected output shape
3. **Paper unlock** — exact quoted text from the .tex that converts from projected/future-tense
   to confirmed/past-tense once results are in hand

**How to use:**
- Saturday meeting: review structure, assign owners, confirm experiment priority order
- H100 session: open doc, work down the pre-flight column, then run the experiment
- After results: make exactly the paper edits listed in the unlock column for that experiment

**Why combined is better than separate docs:**
- Prevents pre-flight checks from being skipped ("we'll do that later")
- Prevents paper edits from being forgotten after experiments run
- Creates a direct chain: passing implementation gate → running experiment → specific .tex lines change
- Single document the team can work through end-to-end rather than context-switching between docs

**Action plan (execute after context compact):**
1. Create `docs/NeurIPS/experiment-readiness-and-paper-revision.md` with the above structure,
   populated from the claims in `paper-revision-claims-audit.md` and the implementation gaps
   in `neurips-2026-data-requirements.md`
2. Add it to the Document Index in `docs/CLAUDE_CONTEXT_BRIEF.md` at high priority
3. Save a memory pointer to it
4. Commit and push
5. Then proceed to obey the remaining suggestions in this session log in order

---

## Post-Compact Execution (commit TBD)

### Experiment Readiness + Paper Revision document created

`docs/NeurIPS/experiment-readiness-and-paper-revision.md` created as planned.

Structure for each of the 5 experiments:
- **Pre-flight gate** — what must be true before running
- **What it produces** — artifact, metric, expected output shape
- **Paper unlock** — exact quoted .tex text that converts from projected/future-tense to
  past-tense confirmed once results are in hand

Key additions beyond the plan outline:
- Global pre-flight checklist (applies before any H100 run): ρ_R fix done ✅; 8 other items pending
- α calibration note for Experiment 1: Qwen gate run showed OEI ~0.996 (saturated); must find
  discriminative α range before committing to Experiment 1 α values
- Dependency chain made explicit: Exp 1 → {Exp 2, Exp 5}; Exp 3 independent; Exp 4 gated on
  large dataset (only exists after Exp 1 or 2 at scale)
- Priority decision table for Saturday meeting ("which experiments will definitely run before May 5?")
- Claim-to-experiment mapping appendix

Document added to Document Index in `docs/CLAUDE_CONTEXT_BRIEF.md` at priority 1a.
Memory pointer saved: `reference_experiment_readiness.md`.

---

## Remaining Suggestions — Status Tracker

Items from the opening assessment and ρ_R fix section, ordered by priority:

| # | Task | Status | Notes |
|---|---|---|---|
| 1 | Fix ρ_R raw dot product bug (Chunk 1A) | ✅ Done (commit 4bc8fa9) | `whitebox.py`, `kv_mcts.py`, `telemetry_matrix.py` |
| 2 | Create experiment-readiness + paper-revision doc | ✅ Done (commit f17b365) | `docs/NeurIPS/experiment-readiness-and-paper-revision.md` |
| 3 | Apply Edit B to .tex — reframe Contribution 2; update related work with new citations | ✅ Done (commit e1df154) | Superseded Path 2: full related work rewrite based on literature survey |
| 4 | Apply Edit C to .tex ("structurally cannot" → "are not designed to navigate") | ✅ Done (commit e1df154) | Applied as part of Section 2.4 rewrite |
| 5 | Remove `\reviewedit{}` wrapper from Edit A (LAT acronym — keep text, drop color markup) | ❌ Pending | Team should confirm at meeting, but safe to pre-apply |
| 6 | Convert present-tense experiment claims to future-tense (Sections 5.1–5.4 and Contributions) | ❌ Pending | Moderate scope; use experiment-readiness doc "Paper unlock" column as guide |
| 7 | Verify OpenReview profiles for all 4 co-authors | ❌ Pending | **Urgent: 2-week moderation window from Apr 18 = May 2, only 2 days before May 4 abstract deadline** |
| 8 | Run gate smoke test on H100 with corrected ρ_R; re-calibrate `classify()` thresholds | ❌ Pending | Requires H100 access; first action of H100 session |
| 9 | Install `mlcroissant` CLI in PATH; run strict Croissant validation on gate run artifacts | ❌ Pending | Can do locally; blocked only by env setup |
| 10 | Implement T_t matrix-level export (2×L per node) in `croissant_export.py` | ❌ Pending | Design gap identified in data requirements doc |
| 11 | Saturday meeting: fill Priority Decision table in experiment-readiness doc | ❌ Pending | Requires team; scope question drives all subsequent work |

