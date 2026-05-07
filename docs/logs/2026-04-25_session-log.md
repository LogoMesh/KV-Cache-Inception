# Session Log — April 25, 2026

**Attendees:** Josh (owner)
**Purpose:** Structural paper cuts (continued from April 24); Experiment 2 rewrite blocked pending paper reads

---

## Context

Continuation of the April 24 session, which ran into the early hours of April 25. All G-section edits (G1–G4) were completed in prior sessions. This session focused on structural TeX cuts and reframing. The Experiment 2 rewrite was reached but not executed — a genuine open question was identified that requires reading cited papers before the rewrite can be done honestly.

Prior session records: `docs/logs/2026-04-24_session-log.md` (full record of G1–G4 and brief updates).

---

## Work Completed This Session

### Files Read at Session Start

- `docs/logs/2026-04-24_session-log.md` — full record confirmed
- `docs/logs/2026-04-24_owner-status-brief.md` — all 23 inline notes confirmed in place (683 lines)
- `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` — full read (686 lines at session start)

---

### G5 — Forward Pointer (Confirmed No-Op)

`\label{hyp:orthogonal_escape}` confirmed at line 136 inside the `\begin{hypothesis}` environment. The `\ref{hyp:orthogonal_escape}` on line 99 (§2.2) resolves. No action taken.

---

### Flags Raised from Full TeX Read

**Flag 1 — Style file (line 8):** `\usepackage[eandd]{neurips_2026}` still set to E&D track. Must be changed to Main Track option before submission. Not yet fixed — deferred.

**Flag 2 — Contribution 2 broken reference (line 81):** "a capability we verify empirically in Experiment~3" is a broken reference since Experiment 3 is cut. Additionally, "up to 20B parameters" is unverifiable with the 1B+3B submission scope. A CRITIQUE NOTE was added to the TeX at this location (not a text change) flagging both issues and their entanglement with DRAFT-HOLD blocks at ~lines 291 and 504.

---

### Structural Cuts Applied

| Edit | Location | Action | Revision Marker |
|---|---|---|---|
| §8 Timeline cut | tex:561–581 | Entire section removed — research proposal artifact | `[REVISION | §8-cut | 2026-04-24]` |
| §5 Experiments 3–5 cut | tex:388–398 | Three `\paragraph{}` blocks removed | `[REVISION | §5-Exp3-5-cut | 2026-04-24]` |
| §7 Future Work added | before Ethical Considerations | New `\subsection{Future Work}` covering Exp 3–5, gpt-oss-20b Phase B, sparse accumulators | `[REVISION | §7-FutureWork | 2026-04-24]` |
| Phase B header renamed | tex:376 | `Phase B: Scaling to Target Models` → `Experiments`; H100 sentence → RTX 3060 + Llama 3.2 1B+3B | `[REVISION | PhaseB-header | 2026-04-24]` |

---

### Notes and CRITIQUE NOTEs Added (No Text Changed)

- **Contribution 2 CRITIQUE NOTE:** Flags "up to 20B parameters" unverifiable at 1B+3B scope; broken Experiment 3 reference; entanglement with DRAFT-HOLD blocks at ~lines 291 and 504.
- **Future Work CRITIQUE NOTE:** Flag to confirm with Tianyu Shi whether a dedicated Future Work subsection is standard practice for NeurIPS Main Track, or whether content belongs in Limitations/Discussion.
- **Table 2 EDITOR NOTE updated:** Added UPDATE block inside existing box noting Experiment 2 is reframed to Option 2 (latent signal quality per compute unit); eventual results column will be per-node telemetry richness, not full-response ASR.
- **Evaluation Metrics CRITIQUE NOTE:** Four metrics now orphaned by experiment cuts — ASR (Experiment 2 reframed), pass@1, Cosine Similarity of Semantic Intent, Memory Efficiency Ratio (all tied to cut Experiments 3–5). Decision deferred: cull vs. qualify as Phase B. Gut preference: cull.

---

## Experiment 2 Rewrite — Open and Blocked

### Where We Got To

Context fully assembled for the Experiment 2 rewrite (Option 2: latent signal quality per compute unit). The surrounding TeX was read, the brief's B7 Option 2 description was confirmed, and the max-briefing was spot-checked for consistency (lines 1–60 — no surprises on decisions; consistent with brief).

### The Open Question

Under Option 2, the metric changes from full-response ASR to latent signal quality per node evaluation (OEI, σ_H, ρ_R per compute unit). This creates an unresolved structural choice about which methods can be legitimately included as comparison points:

**Option A — Qualitative contrast with text-space; numeric comparison with random KV-cache search only:**
Text-space methods (GCG, MPA/wu2025mcts, DAMON) are cited as the categorical alternative that produces no internal latent signal — argued qualitatively as motivation rather than tabulated. The numeric comparison is restricted to our MCTS vs. random search in KV-cache space, both of which produce OEI/σ_H/ρ_R per node evaluation.

**Option B — Instrument text-space methods with the same telemetry monitors:**
Run text-space MCTS (`run_offline_mcts.py`) with hidden-state monitoring active at each node expansion, recording OEI/σ_H/ρ_R for each text-space mutation step. Compare telemetry richness across all methods numerically. This requires confirming that `run_offline_mcts.py` captures hidden states — currently unknown.

### Why This Requires Reading the Papers

The key question for choosing between A and B is: **do MPA, DAMON, and GCG access or expose hidden states at the node level, or are they purely behavioral (text output only)?**

- If they are purely behavioral → Option A is the only honest framing. Comparing their "latent signal quality" would be undefined.
- If they do expose hidden states (unlikely but worth confirming) → Option B becomes feasible.
- Even under Option A, the exact language used to position GCG, MPA, and DAMON needs to match what those papers actually claim. Getting this wrong in a NeurIPS submission invites a reviewer who knows the system to flag the characterization.

### Papers to Read Before Rewriting

| Paper | File | What to confirm |
|---|---|---|
| wu2025mcts (MPA, COLING 2025) | `docs/NeurIPS/Citations/2025.coling-main.71.pdf` | Does MPA access hidden states per node, or is it purely text-space prompt mutation? What is the actual node evaluation mechanism? |
| damon2025 (EMNLP 2025) | `docs/NeurIPS/Citations/2025.emnlp-main.323.pdf` | Same question — hidden states or text only? Dialogue-aware MCTS node structure. |
| zou2023gcg (GCG, arXiv) | `docs/NeurIPS/Citations/arXiv-2307.15043v2/` | GCG is gradient-based in token space — confirm it produces no per-step internal telemetry. Read enough to confidently state what GCG does NOT do in the comparison framing. |

Use `reducto parse` on the PDFs if needed. The goal is not deep reading — just enough to characterise the node evaluation mechanism of each method accurately.

### What the Rewrite Should Achieve

Once papers are read, rewrite `\paragraph{Experiment 2}` (currently at tex:384–386) to:
1. Name the metric: latent signal quality per compute unit (OEI, σ_H, ρ_R per node evaluation)
2. State which methods are compared numerically vs. cited qualitatively
3. Include the honest caveat that cross-method ASR comparison requires multi-step rollout (deferred to Phase B)
4. Update the `% [IMPL: PARTIAL]` comment to reflect the reframed scope

Flag with `[REVISION | Exp2-rewrite | 2026-04-25]` when applied.

---

## What's Queued for Next Session (In Order)

1. **Read wu2025mcts, damon2025, zou2023gcg** (papers listed above) — confirm node evaluation mechanism for each
2. **Decide Option A vs. Option B** for Experiment 2 comparison structure
3. **Rewrite §5 Experiment 2** based on decision
4. **Add 1B→3B scaling framing** to §5 Experiment 1 (still pending)
5. **Fix style file option** — `\usepackage[eandd]{neurips_2026}` → Main Track option (confirm correct option from template zip)
6. **Cull orphaned metrics** from §5.4 Evaluation Metrics once Experiment 2 rewrite is settled (ASR, pass@1, cosine similarity, MER)
7. **Audit all REVISION markers**: `grep -n "REVISION" docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`

---

## Cross-Check Against April 24 Log — Three Gaps Found

Read `docs/logs/2026-04-24_session-log.md` in full at session close to verify the next_session_prompt captured everything. Three items from the April 24 plan were missing or underweighted.

**Gap 1 — Croissant sentence missing from §7 Future Work (concrete omission):**
The April 24 plan explicitly listed "Croissant/dataset content — one sentence in Future Work" as a paper cut item. The Future Work subsection was written this session but the Croissant sentence was not included. Needs to be added next session. Planned text: "A Croissant~1.1 dataset package of runtime evidence artifacts is planned for release alongside the Phase~B submission."

**Gap 2 — OEI Option B paper changes not tracked:**
The April 24 log (line 19) records that Option B requires four paper text changes not yet applied: §1 Contributions (soften OE formalization claim), §3 Hypothesis 1 (add gate-run observation framing), §5 Experiment 1 (reframe from "will confirm" to "first controlled measurement"), §7 Limitations (strengthen existing caveat). These are deferred pending Josh confirming exact reframing language and the B6 α-sweep diagnostic running first. Were not listed in the next_session_prompt — added now.

**Gap 3 — Abstract registration deadline not prominent:**
The April 24 log flags May 4 (AoE) as 🚨 IMMEDIATE — registering the abstract on OpenReview (Main Track portal) gates the full paper submission on May 6. Cannot switch tracks after registering. Status: awaiting Tianyu Shi's guidance. 9 days away as of April 25. Added as a top-level warning block in the next_session_prompt.

All three gaps added to `next_session_prompt.md`.

---

## All REVISION Markers in TeX as of This Session

```
grep -n "REVISION" docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex
```

| Marker | Location | Description |
|---|---|---|
| G1 | §7 Limitations (~line 555) | Claims 12+13+14 added |
| G2 | Table 2 cell + bibliography | panickssery2023steering → rimsky2024steering |
| G3 (×2) | §2.5 line 112 + bibliography line 667 | CRSM removed entirely |
| §8-cut | Where §8 was | Timeline section removed |
| §5-Exp3-5-cut | Where Experiments 3–5 were | Three paragraphs removed |
| §7-FutureWork | §7 Future Work | New subsection added |
| PhaseB-header | §5 subsection heading | Renamed; H100 → RTX 3060 |
| Exp2-rewrite | §5 Experiment 2 paragraph | Reframed to latent signal quality per compute unit; Option A — added 2026-04-25 |

---

## 2026-04-25 (Opus session, afternoon) — Paper Reads + Experiment 2 Rewrite Applied

### Primary-source reads (empirical verification, not spot-check)

Read the three cited methods directly to characterise their node-evaluation mechanism. All three findings are quotations or equation references from the source.

| Method | Source | Setting | Node-evaluation signal | Hidden states accessed? |
|---|---|---|---|---|
| GCG (zou2023gcg) | `arXiv-2307.15043v2/sections/3b-method.tex` | White-box gradient | Final-layer NLL on target tokens (Eq. generation-loss, line 60); gradients w.r.t. one-hot token indicators identify candidates, scored by full forward pass | **No** |
| MPA (wu2025mcts) | `2025.coling-main.71.pdf` (extracted to `_extracted/coling-71.txt`) | Black-box (paper §2.2 explicit) | Evaluator-LLM 1–10 score on text response; first-position target-token logprob via API as UCT prior `P(s,a) = 1 - exp(log P_T(t \| p_sm))` | **No** — paper: "cannot access internal information of the target LLM (e.g., parameters, logits and loss)" |
| DAMON (damon2025) | `2025.emnlp-main.323.pdf` (extracted to `_extracted/emnlp-323.txt`) | Fully black-box | Reward-model scalar `E(s)` on dialogue harmfulness | **No** — paper: "no access to any internal information of the LLMs, including parameters, logits and loss" |

Cross-check: zero matches on `hidden|representation|latent|activation|embedding` across all three papers' relevant sections.

### Decision: Option A

**Why not Option B:** Instrumenting the text-space methods with hidden-state hooks during their forward passes is technically feasible but tautological — KV-cache MCTS *targets* OEI/σ_H/ρ_R as its reward; the text-space methods do not. A "telemetry richness per compute unit" comparison would amount to "does the method that optimizes for X produce more X than methods that don't?" — a reviewer would flag it.

**Option A is the honest framing:** numeric comparison restricted to KV-cache space (our MCTS vs. random KV-cache search, both producing the same telemetry triple per node natively); GCG/MPA/DAMON cited qualitatively as the categorical alternative whose per-step compute is invested in text-output evaluation rather than internal-state probing.

**MPA caveat captured in rewrite:** MPA does use one model-output signal (first-position target-token logprob via API). The rewrite phrases this carefully: "operate exclusively on model outputs (text tokens, and in MPA's case first-position target-token log-probabilities) rather than on internal representations." Avoids the imprecise "purely behavioral" framing.

### TeX edit applied

`\paragraph{Experiment 2}` at tex:385–387 (pre-edit) replaced with new paragraph at the same location, flagged `[REVISION | Exp2-rewrite | 2026-04-25]`. The `[IMPL: PARTIAL]` comment was updated to reflect: latent-space MCTS implemented; random-KV-search baseline pending; cross-method ASR rollout deferred to Phase B.

Eq references verified against existing labels: `eq:telemetry_matrix` (line 217) and `eq:oei` (line 456). Initial draft used `eq:telemetry-matrix` (hyphen) — fixed to underscore form to match existing label convention.

### Decision artifacts saved

`memory/experiment2_rewrite_decision.md` — Option A rationale, citation table with file/line evidence, full draft paragraph text. Persists post-compaction in case rewrite is revisited.

### Still pending (to do post-compact next session)

- §5 Experiment 1: 1B→3B scaling framing (ORIGINAL PLAN — do not skip)
- §7 Future Work: Croissant sentence (ORIGINAL PLAN omission)
- Style file: `eandd` → Main Track option
- §5.4 orphaned metrics cull
- §1 Contribution 2 fix (coordinate with DRAFT-HOLD blocks)
- OEI Option B paper changes (deferred pending B6 diagnostic)


## 2026-04-25 (Opus session, post-compact) — Exp 1 1B+3B framing + Croissant Future Work sentence

Resumed after auto-compact at the explicitly-flagged top of the queue from `memory/next_session_prompt.md`. Two small surgical edits applied to the canonical TeX, both flagged with fresh REVISION markers.

### Verification before editing

- §5 opener at tex:379 already commits to "Llama~3.2-1B-Instruct and Llama~3.2-3B-Instruct on a local RTX~3060 (12GB VRAM)" (set in the PhaseB-header revision on 2026-04-24); no contradiction with adding parallel mention in Experiment 1.
- `scripts/run_kv_mcts.py` accepts any HF model via `--model` (line 257); no script hard-codes 3B, but no script forbids it either — the 3B sweep is forward-looking but the interface supports it.
- The pre-existing IMPL comment at tex:383 ("H100 scale pending") was stale: H100 was removed as a submission scope in the PhaseB-header revision. Refreshed as part of the same edit.
- §7 Ethical Considerations (tex:583) already mentions a Croissant 1.1 metadata file release; the Future Work sentence is scoped to *runtime evidence artifacts* (per-run telemetry from MCTS runs) — complementary to the metadata file, not redundant. Captured in REVISION marker.

### Edits applied

1. **§5 Experiment 1 — `[REVISION | Exp1-1B3B-framing | 2026-04-25]` at tex:381–384.** Added explicit "run independently on Llama~3.2-1B-Instruct and Llama~3.2-3B-Instruct so the sweep is observed at two model scales on the same consumer hardware" to the alpha-sweep paragraph. Refreshed IMPL comment: 1B sweep gated on Llama-3.2-1B-Instruct (Phase 3 gate, 2026-04-16); 3B sweep on Llama-3.2-3B-Instruct via same `--model` interface, pending execution.

2. **§7 Future Work — `[REVISION | §7-Croissant | 2026-04-25]` at tex:579–581.** Appended sentence: "A Croissant~1.1 dataset package of runtime evidence artifacts is planned for release alongside the Phase~B submission." Marker notes complementarity with the §7 Ethical Considerations Croissant metadata sentence.

### Read-back verification

Both edits read back cleanly at the expected line numbers; equation references and surrounding REVISION markers (Exp2-rewrite, §7-FutureWork) intact.

### Still pending (next session queue)

- Style file: `eandd` → Main Track option (confirm from template zip)
- §5.4 orphaned metrics cull (ASR, pass@1, cosine similarity, MER)
- §1 Contribution 2 fix (coordinate with DRAFT-HOLD blocks): soften "up to 20B parameters" + reframe Experiment 3 reference
- OEI Option B paper changes (deferred pending B6 diagnostic): §1 Contributions, §3 Hypothesis 1, §5 Experiment 1, §7 Limitations
- Audit all REVISION markers: `grep -n "REVISION" docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`
- 🚨 Abstract registration on OpenReview by May 4 (awaiting Tianyu Shi)


## 2026-04-25 (Opus session, post-compact, Edit 3) — Style file track switch

Applied the smallest remaining queue item: switched the NeurIPS style option from `eandd` (Evaluations and Datasets track) to `main` (Main Track), per the locked decision in `memory/next_session_prompt.md` "Decisions Locked" table.

### Verification before editing

- `docs/NeurIPS/neurips_2026.sty` declares the following track options: `main` (line 49), `position`, `eandd`, `creativeai`, `sglblindworkshop`, `dblblindworkshop`. Confirmed `main` is the canonical Main Track option name. Default is also main (line 48: `\@maintrue`).
- Editing both line 3 (header comment "Evaluations and Datasets track" → "Main Track") and line 7 (camera-ready hint `[eandd, final]` → `[main, final]`) so the file is internally consistent.

### Edit applied

`docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` lines 3, 7, 8 — `[REVISION | track-main | 2026-04-25]`. Header comment, camera-ready hint, and `\usepackage` line all updated.

### Still pending (next session queue)

- §5.4 orphaned metrics cull (ASR, pass@1, cosine similarity, MER) — see CRITIQUE NOTE block at tex:470–478
- §1 Contribution 2 fix (coordinate with DRAFT-HOLD blocks): soften "up to 20B parameters" + reframe Experiment 3 reference (CRITIQUE NOTE at tex:83–84)
- OEI Option B paper changes (deferred pending B6 diagnostic)
- Audit all REVISION markers: `grep -n "REVISION" docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`
- 🚨 Abstract registration on OpenReview by May 4 (awaiting Tianyu Shi)


## 2026-04-25 (Opus session, post-compact, Edit 3) — Style file track switch

Applied the smallest remaining queue item: switched the NeurIPS style option from `eandd` (Evaluations and Datasets track) to `main` (Main Track), per the locked decision in `memory/next_session_prompt.md` "Decisions Locked" table.

### Verification before editing

- `docs/NeurIPS/neurips_2026.sty` declares the following track options: `main` (line 49), `position`, `eandd`, `creativeai`, `sglblindworkshop`, `dblblindworkshop`. Confirmed `main` is the canonical Main Track option name. Default is also main (line 48: `\@maintrue`).
- Editing both line 3 (header comment "Evaluations and Datasets track" → "Main Track") and line 7 (camera-ready hint `[eandd, final]` → `[main, final]`) so the file is internally consistent.

### Edit applied

`docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` lines 3, 7, 8 — `[REVISION | track-main | 2026-04-25]`. Header comment, camera-ready hint, and `\usepackage` line all updated.

### Still pending (next session queue)

- §5.4 orphaned metrics cull (ASR, pass@1, cosine similarity, MER) — see CRITIQUE NOTE block at tex:470–478
- §1 Contribution 2 fix (coordinate with DRAFT-HOLD blocks): soften "up to 20B parameters" + reframe Experiment 3 reference (CRITIQUE NOTE at tex:83–84)
- OEI Option B paper changes (deferred pending B6 diagnostic)
- Audit all REVISION markers: `grep -n "REVISION" docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`
- Abstract registration on OpenReview by May 4 (awaiting Tianyu Shi)


## 2026-04-25 (Opus session, post-compact, §5.4 cull preflight) — HOLD: decisions deferred at Josh's request

Began the §5.4 orphaned metrics cull (item 7 in next_session_prompt.md, flagged "[START HERE NEXT SESSION]"). Read §5.4 (tex:440–488), grep'd the file for ASR/pass@1/Cosine/MER/TDS references, and read §2.2 in full (tex:124–126). Surfaced two non-trivial findings that change the cull scope. Josh asked to preserve the analysis verbatim for later attention; **no edits applied yet**.

### Initial proposal that was on the table (before second-pass verification)

**Edit A** — replace tex:472–488 (the CRITIQUE NOTE + 4 orphaned `\item`s) with a single revision marker followed by the closing `\end{itemize}`:

```latex
% [REVISION | §5.4-metrics-cull | 2026-04-25 | Removed orphaned metrics (ASR, pass@1, Cosine Similarity, MER) per CRITIQUE NOTE rationale: Experiments 3-5 cut and Experiment 2 reframed to latent signal quality. Surviving metric inventory (OEI, TDS) reflects what submission experiments actually measure. — pending audit]
\end{itemize}
```

**Edit B** — collateral fix at tex:126. Reframe the closing sentence so the §2.2 paragraph keeps its anti-LLM-judge punchline but pivots to the metrics we actually use:

> Grounding evaluation in **objective signals derived from internal model state—such as the telemetry triple $(\sigma_H, \rho_R, \mathrm{OEI})$ defined in §3 and §5—provides stronger anchoring than purely subjective LLM scoring.**

(Replaces "standard deterministic metrics such as pass@1 and cosine similarity of semantic intent provides...")

### Cross-cutting reference scan (tex grep)

| Line | Context | Initial decision |
|---|---|---|
| 126 (§2.2) | "Grounding evaluation in standard deterministic metrics such as pass@1 and cosine similarity of semantic intent provides stronger anchoring than purely subjective LLM scoring." | Collateral fix needed — endorses metrics we won't use, dangling claim |
| 390 | IMPL comment, "cross-method ASR rollout deferred to Phase B" | Keep as-is — internal comment about Phase B |
| 396–430 | EDITOR NOTE about not adding published ASR figures to baselines table | Keep — still relevant for baselines table at tex:432–447 |
| 537 | DRAFT-HOLD block at Memory Complexity, "measured MER will contradict the proposition" | Keep — Phase B planning context |
| 545–551 | Memory Complexity Proposition with "9.7 TB vs 40.05 GB" | Keep — theoretical claim, not a metric measurement |

### Finding 1 — TDS is also functionally orphaned from submission experiments

| Where TDS appears | Type |
|---|---|
| tex:70 | Comment: "Reviewers will ask: does an increase in OEI/TDS during KV-cache steering correspond..." |
| tex:350 | IMPL comment: "MCTS smoke test: run_kv_mcts.py --nodes 10, all T_t / OEI / TDS fields populated" |
| tex:470 | The §5.4 definition itself |

TDS is **not named in any submission Experiment paragraph**. Exp 1 measures (a) compliance / (b) σ_H / (c) ρ_R / (d) null-space norm — no TDS. Exp 2 measures the triple (σ_H, ρ_R, OEI) — no TDS. So TDS is in the same orphaned-from-experiments category as ASR/pass@1/MER, but it differs from them in two ways:

- The implementation does populate it (tex:350 IMPL comment)
- It's the conceptual divergence companion to OEI in the broader telemetry framework

**Three options:**

- **(T-A) Keep TDS in §5.4** — defensible because the run output emits it and it's part of the telemetry framework, even if no submission experiment tabulates it. Risk: a reviewer asks "where are the TDS results?"
- **(T-B) Cull TDS too** — leaves §5.4 with OEI as the sole metric. Cleanest for reviewers, but loses the divergence framing entirely.
- **(T-C) Keep TDS but downgrade phrasing** — define it in §5.4 as a "diagnostic signal emitted by the telemetry pipeline" rather than a "metric," so the reader doesn't expect a results table for it. Half-measure but honest about its current status.

Opus recommendation (NOT a decision): **(T-C)**, because the IMPL comment at tex:350 is empirical evidence that we're producing TDS data even if we don't tabulate it in the submission, and the divergence framing matters for the §3 hypothesis structure. Final call deferred to Josh.

### Finding 2 — The proposed §2.2 reframe (Edit B) doesn't actually fit the paragraph

The full §2.2 paragraph (tex:124–126) flows as: LLM judges are weak → static analysis tools can't measure intent → so deterministic metrics like pass@1 / cosine similarity provide stronger anchoring. The closing sentence is specifically about evaluating **generated outputs** (code intent, response quality). The earlier proposal (Edit B) swapped in the telemetry triple (σ_H, ρ_R, OEI) — but the telemetry triple measures **internal model state**, which is a different category of measurement. The paragraph's logic would break: "static tools can't measure code intent → so use internal hidden-state telemetry" doesn't follow.

**Three better options for tex:126:**

- **(R-A) Delete the closing sentence entirely.** The paragraph still works as a two-clause critique of LLM-judges and static analysis. Lowest risk.
- **(R-B) Soften to a landscape claim**: "Standard deterministic metrics such as pass@1 or semantic-similarity scores anchor evaluation to objective signals; our work extends this principle from output-space measurements to internal-state telemetry." Preserves the citation pivot, honest that pass@1/cosine-sim are part of the field even though we don't use them.
- **(R-C) Replace with our internal-state pivot but rewrite the paragraph's setup** so the logical flow stays intact (more invasive — would touch the SonarQube/Semgrep clause too).

Opus recommendation (NOT a decision): **(R-B)**, because it preserves §2.2's logical motion while not falsely implying we measure pass@1/cosine-sim. (R-A) is a safe fallback if you'd rather not introduce new prose.

### Combined landing point if both recommendations approved

If T-C + R-B both approved, the §5.4 surviving inventory becomes: OEI (full metric, used in Exp 1 + 2) and TDS (downgraded to "diagnostic signal emitted by the telemetry pipeline; not tabulated in submission experiments"). And §2.2 keeps a clean handoff into our approach.

### 🛑 STATUS: HELD — RETURN TO THIS

Josh paused the work after Findings 1 + 2 were surfaced and asked that the analysis be preserved verbatim "for reasons to be explained later." **No edits applied to the TeX.** All decisions still open:

- T-{A, B, C} for TDS scope
- R-{A, B, C} for §2.2 collateral fix
- The Edit A wording at tex:472–488 (REVISION marker text) is still proposed but contingent on the TDS decision (it currently asserts "OEI, TDS" as the surviving inventory; would need rewording under T-B)

Resume entry point: re-read this section + the underlying tex sections (§5.4 at tex:440–488, §2.2 at tex:124–126), then ask Josh for the T- and R- decisions before applying any edits.


## 2026-04-25 (Opus session, post-compact, §5.4 cull preflight) — DECISIONS RECORDED (logged 2026-05-03)

> Logging note: Josh approved T-C + R-B on 2026-05-03, after the venue pivot to EMNLP Findings (Path A). Recording the decisions here, on the 04-25 log, because that's where the underlying analysis lives. **Edits not yet applied to the TeX** — Josh has paused implementation pending a fresh-session pre-mortem to check for contextual debt / over-corrections in the recent edit chain. See `docs/logs/2026-05-03_premortem-prompt.md`.

### Decision: T-C (TDS reframe, not cull)

**Definition.** Keep TDS defined in §5.4, but reframe its positioning so the reader understands it as the divergence component of the MCTS reward function (Eq. 8, the `λ₃·Divergence(σ_H, ρ_R)` term at tex:324) — defined here for completeness — rather than as a metric to be tabulated separately in submission results.

**Why this and not T-B (cull entirely).** A grep on 2026-04-25 surfaced that Eq. 8 at tex:324 uses `Divergence(σ_H, ρ_R)` as the third reward term, and the explanatory text at tex:328 names this exact quantity ("measures the cross-channel inconsistency that signals Orthogonal Escape"). TDS is the JS-divergence form of that reward term. Culling the §5.4 TDS definition would leave Eq. 8's third reward term named but undefined — incoherent under reviewer scrutiny. **T-B is therefore structurally off the table**, not just a judgment call.

**Why this and not T-A (keep as-is).** T-A leaves §5.4 reading like TDS will be tabulated in results. Under Path A reviewer-defensibility, every defined "metric" should map to a results column or be honestly qualified. T-A invites a "where are the TDS results?" reviewer comment that T-C preempts.

**Concrete edit (deferred — apply only after pre-mortem):** rewrite the §5.4 TDS bullet at tex:470 from a metric definition to a reward-component definition. Draft text TBD until the surrounding §5.4 cull paragraph is settled (since the bullet's neighbors are being culled, the surviving prose needs to read as a coherent two-item list, not a metric list with one reframed orphan).

### Decision: R-B (§2.2 soften to landscape claim)

**Definition.** Rewrite the closing sentence of §2.2 at tex:126 to soften the pass@1 / cosine-similarity endorsement from a recommendation into a landscape acknowledgment that pivots to our internal-state telemetry approach. The current sentence reads:

> "Grounding evaluation in standard deterministic metrics such as pass@1 and cosine similarity of semantic intent provides stronger anchoring than purely subjective LLM scoring."

The R-B replacement reads:

> "Standard deterministic metrics such as pass@1 or semantic-similarity scores anchor evaluation to objective signals; our work extends this principle from output-space measurements to internal-state telemetry."

**Why this and not R-A (delete sentence).** R-A is the safe fallback but loses §2.2's logical motion: the paragraph builds "LLM judges are weak → static analysis can't measure intent → therefore deterministic alternatives anchor better," and the closing sentence is the punchline of that motion. Deleting it leaves the paragraph as a critique without a constructive pivot.

**Why this and not R-C (rewrite paragraph setup).** R-C would touch the SonarQube/Semgrep clause too, which is more invasive and expands surface area for new errors. R-B is the minimal fix that resolves the dangling-claim problem.

**Concrete edit (deferred — apply only after pre-mortem):** in-place replacement of the closing sentence of §2.2 at tex:126.

### Status

- T-C and R-B **decisions locked**.
- TeX edits **deferred** pending fresh-session pre-mortem to check for contextual debt and over-corrections in the recent edit chain (G1–G5 cuts, Exp 3–5 deferral, Exp 2 rewrite, OEI Option B framing, §8 cut, Phase B header rewrite, style file switch, Exp 1 1B+3B framing, Croissant Future Work sentence).
- Pre-mortem prompt: `docs/logs/2026-05-03_premortem-prompt.md`.
- Resume entry point: after pre-mortem returns findings, reconcile its critiques against T-C / R-B and the surviving paper structure before applying any TeX edits.
