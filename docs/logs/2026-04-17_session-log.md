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
- Committed and pushed

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
