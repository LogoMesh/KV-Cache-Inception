# Session Log — April 23, 2026

**Attendees:** Josh (owner)
**Purpose:** Paper revision — P2-E (critical), P2-B, P2-C citation fixes; source papers read before editing

---

## Context

Continuing from April 22. All three edits in this session followed the new protocol established in the April 22 peer-review hardening discussion: source paper read directly from the downloaded TeX before any text was amended.

All edits are to: `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`

---

## Source Papers Read

| Citation | Folder | Sections consulted |
|---|---|---|
| `llmasjudge2025limitations` (P2-E) | `arXiv-2503.05061v2/` | `main.tex` — Abstract, Introduction, Analysis (correct_incorrect_table, GPT-4o κ drop para), Discussion |
| `greenblatt2024alignment` (P2-B) | `arXiv-2412.14093v2/` | `sections/0_abstract.tex`, `sections/1_introduction.tex` (scale results, scratchpad design, threat models), `sections/8_discussion.tex` (limitations, scratchpad dependence) |
| `hubinger2024sleeper` (P2-C) | `arXiv-2401.05566v3/` | `main.tex` grepped for scale/persistence/naturally/adversarial/hiding/threat model — key passages confirmed |

---

## Edits Applied

### Edit 7 — P2-E · §2.5 · Remove fabricated 0.10–0.20 claim
**Gap:** `"score divergences of $0.10$--$0.20$ on normalized scales across independent runs~\cite{llmasjudge2025limitations}"` — this figure does not exist in Krumdick et al. (2025). The paper's metric is Cohen's κ; it concerns reference quality and judge competence, not run-to-run variance.

**Source finding:** Analysis section: *"when GPT-4o uses its own responses to questions as a reference (i.e., Self response), the Cohen's κ decreases from 0.86 to 0.16 in the pairwise case on questions GPT-4o could not answer."* Paper conclusion: *"judges show high agreement with human experts only on questions the judges were able to correctly answer themselves."*

**Fix:** Replaced fabricated sentence with: *"these approaches achieve reliable agreement with human experts only when the judge model can itself correctly answer the underlying question or is provided with a correct reference; without such grounding, agreement drops substantially---for instance, GPT-4o's Cohen's $\kappa$ falls from $0.86$ to $0.16$ on benchmark questions the model cannot answer~\cite{llmasjudge2025limitations}."*

---

### Edit 8 — P2-B · §1 · Scratchpad qualifier added
**Gap:** "explicitly reasoning in its chain-of-thought" — doesn't note the scratchpad was an explicit part of the experimental design. In the prompted setup, removing it nearly eliminates the compliance gap.

**Source finding:** §1: *"we provide the model with a hidden chain-of-thought scratchpad and tell it to use the scratchpad to analyze its situation"*; §8 Limitations: *"we intentionally designed our setup to ensure that the model actively reasons in detail about its situation in its hidden chain-of-thought scratchpad."*

**Fix:** `"its chain-of-thought"` → `"a hidden chain-of-thought scratchpad---included as part of the experimental design---"` and `"its existing values"` → `"its existing harmlessness values"`.

---

### Edit 9 — P2-B · §2.1 · Benign goals + scale caveat added
**Gap:** §2.1 implied strategic compliance as a general finding without noting (a) that the goals preserved were benign HHH preferences, and (b) that it only emerged at frontier scale.

**Source findings:**
- Goals: *"the goals we discover that Claude 3 Opus is willing to fake alignment for...aren't themselves concerning"* (§1 Threat models)
- Scale: *"Alignment faking emerges with model scale. We find that Claude 3 Opus and Claude 3.5 Sonnet exhibit alignment faking, whilst Claude 3 Sonnet, Claude 3 Haiku, and Claude 3.5 Haiku generally do not"* (§1)

**Fix:** §2.1 Greenblatt sentence rewritten to add: situational awareness required, goals were HHH harmlessness (not malicious), frontier-scale only (Opus + 3.5 Sonnet; smaller models did not exhibit it).

---

### Edit 10 — P2-C · §1 + §2.1 · Scale qualifier, deliberately-inserted distinction, adversarial training characterization
**Gap (10a):** §1 presented backdoor persistence as scale-independent. Paper is explicit it is strongly scale-dependent.
**Gap (10b):** §1 implied naturally arising deception; paper tests threat-model resilience, not natural occurrence plausibility.
**Gap (10c):** §1 said backdoors "persist through adversarial training" — understates the finding. Paper: adversarial training *actively taught models to better conceal their backdoor*.

**Source findings:**
- Scale: *"Backdoor persistence is contingent, with the largest models...being the most persistent"*; *"robustness of backdoored models to RL fine-tuning increases with model scale"*
- Not naturally arising: *"this work is designed to test whether our threat models can be resilient to safety training, not the plausibility of our threat models occurring by default"*
- Adversarial training: *"Adversarial training tends to make backdoored models more accurate at implementing their backdoored behaviors, effectively hiding rather than removing them"*

**Fix:** §1 Hubinger sentence replaced: deliberately inserted, adversarial training hides rather than removes, substantially weaker in smaller models, tests threat-model resilience. §2.1 Hubinger sentence similarly corrected.

---

## Verification

All four text changes read back after application. Surrounding context intact. No TeX structure disturbed.

---

---

## Source Papers Read (continued — P2-D)

| Citation | Folder | Sections consulted |
|---|---|---|
| `belitsky2025kvcache` (P2-D) | `arXiv-2507.08799v2/` | `main.tex` — Abstract, §1 Introduction (one-shot, mean-diff mechanism), §3 Method (precomputed KV vector, single post-prefill injection), ablation figures, latency comparison |

---

## Edits Applied (continued — P2-D, P2-A, P3-A, P4-G, P4-B)

### Edit 11 — P2-D · §2.2 + Table 2 · Domain qualifier, paradigm separation
**Gap 1 (domain):** §2.2 cited Belitsky's stability advantage without noting it was validated on reasoning tasks (GSM8K/ARC-c/CSQA/PIQA) with 360M–8B models. Our application to alignment faking in larger models is unqualified extrapolation.

**Gap 2 (paradigm mismatch):** Table 2 row "One-shot KV-Cache Steering" implied Belitsky's method and our MCTS are comparable. Belitsky: precompute mean-of-differences vector, inject once post-prefill. Our MCTS: iterative mutations with FP32 reversibility and telemetry-derived reward — a fundamentally different paradigm.

**Gap 3 (framing):** §2.2 read as if our KV-MCTS extends Belitsky's approach. These are distinct intervention regimes.

**Source findings (arXiv-2507.08799v2/main.tex):** *"precomputed mean-of-differences vector…added once to the KV cache post-prefill"*; evaluated on GSM8K, ARC-c, CSQA, PIQA with 360M–8B models; ablation confirms stability advantage vs. activation steering within this domain.

**Fixes:**
- §2.2: Added `"---validated for reasoning induction in small-to-medium LLMs (360M--8B)"` after the stability claim
- §2.2: Added paradigm separation paragraph: *"Unlike this static approach, our MCTS iteratively explores the KV-cache intervention space across multiple rollouts, reversing each intervention exactly via FP32 accumulators and guided by a telemetry-derived reward rather than a fixed behavioral objective."*
- Table 2 row: `"One-shot KV-Cache Steering"` → `"One-shot static KV-Cache Steering (precomputed mean-of-differences vector, single post-prefill injection)"`
- Table 2 caption: Added sentence distinguishing static one-shot paradigm from our iterative reversible MCTS

---

### Edit 12 — P2-A · §2.2 + §3 · Forward pointer + hypothesis label
**Gap:** After Edit 3 (2026-04-22), RL-Obfuscation motivation was reframed as "by analogy rather than direct mechanism." §2.2 now correctly disclaims the direct mechanistic connection but leaves the Orthogonal Escape hypothesis unmotivated at the point where the reader most needs a pointer to §3.

**Fixes:**
- §2.2: Added after the "by analogy" qualifier: *"The geometric motivation for this redistribution is formalized in Section~3 (Hypothesis~\ref{hyp:orthogonal_escape})."*
- §3 Hypothesis 1: Added `\label{hyp:orthogonal_escape}` to `\begin{hypothesis}[Orthogonal Escape]` declaration so the `\ref{}` resolves

---

### Edit 13 — P3-A · §2.4 · GCG discrete token space fix
**Gap:** §2.4 said GCG methods "operate in token embedding space." GCG operates in **discrete token space** — gradient is used as a selection heuristic for token substitutions, but evaluated candidates are actual discrete token sequences. The GCG paper explicitly contrasts itself with PEZ/GBDA-style methods which DO operate in continuous embedding space. Our wording inverted this distinction.

**Fix:** `"operate in token embedding space"` → `"operate in discrete token space"`

---

### Edit 14 — P4-G · §2.4 · AgenticRed removal
**Gap:** AgenticRed was placed in the MCTS red-teaming list. AgenticRed's primary method is evolutionary system design (Meta Agent Search-inspired). MCTS appears only as an appendix sub-agent (PHOENIX-MCTS), one artifact discovered BY AgenticRed's evolutionary search — not AgenticRed's own method.

**Fix:** Deleted `"AgenticRed~\cite{agenticred2025} applied MCTS within an agentic red-teaming framework."` from §2.4 MCTS list. MPA, DAMON, and ACE-Safety remain as accurate MCTS-based examples.

---

### Edit 15 — P4-B · Table 2 + Bibliography · CAA citation + bibitem
**Gap:** Table 2 listed "Activation Steering / CAA (Contrastive Activation Addition) — Rimsky et al., 2024" as plain text with no `\cite{}` and no corresponding `\bibitem`. A NeurIPS reviewer will notice an uncited baseline immediately.

**Fixes:**
- Table 2 CAA row: `"Rimsky et al., 2024"` → `"Panickssery \& Rimsky et al., 2023~\cite{panickssery2023steering}"`
- Bibliography: Added `\bibitem{panickssery2023steering}` entry for Panickssery, Rimsky et al. (2023), *Steering LLaMA 2 via Contrastive Activation Addition*, arXiv:2312.06681

---

## Verification (Edits 11–15)

All five batches of changes read back after application. Surrounding context intact. No TeX structure disturbed. `\ref{hyp:orthogonal_escape}` forward pointer confirmed resolvable — label added to Hypothesis 1 declaration in §3.

---

## What's Next

Remaining queue after this session:

| Entry | Citation | Key fixes | Urgency |
|---|---|---|---|
| ~~P4-A~~ | ~~`deroy2026structured`~~ | ~~Replace Zenodo preprint~~ | ✅ DONE |
| ~~P4-D~~ | ~~`crsm2025`~~ | ~~Add "(experimental prototype)" qualifier~~ | ✅ DONE |
| ~~P4-E~~ | ~~`okazakirag2025`~~ | ~~Add "(unpublished manuscript)" qualifier~~ | ✅ DONE (peer-reviewed companion deferred) |
| ~~Orphans~~ | ~~`contextual2025`, `oozeer2025transfer`~~ | ~~Remove orphaned bibitems~~ | ✅ DONE |
| P3-A (ASR) | `zou2023gcg` | Retrieve GCG ASR baselines for Table 2 (88% whitebox Vicuna; 84% GPT-3.5; 66% PaLM-2; 2.1% Claude) | Medium |
| P3-B + P3-C | `wu2025mcts`, `damon2025` | Retrieve ASR figures for Table 2 baselines (MPA + DAMON numbers) | Medium |
| P2-D (code) | `belitsky2025kvcache` | Implement mean-of-differences KV steering as actual code baseline in `logomesh/` | High (code task) |
| P4-E (deferred) | TurboRAG / CacheBlend / EPIC | Download and cite peer-reviewed KV-cache serving paper to anchor Okazaki-RAG comparison | Medium |

Protocol: read source paper before amending. For P4-D and P4-E, identify peer-reviewed replacement before removing existing citation.

---

---

### Edit 16 — P4-A · §2.1 + Bibliography · Replace Zenodo preprint with peer-reviewed sources
**Gap 1 (overstatement):** "concluding that no single approach provides reliable detection guarantees" — the Deroy review actually advocates white-box oversight, not nihilism about all detection approaches.
**Gap 2 (Critical — citation quality):** Zenodo-only preprint, single author, no peer review. A NeurIPS reviewer will flag this immediately. The citation was serving two roles in §2.1: (a) establishing that alignment misrepresentation is a longstanding concern, and (b) surveying detection approaches. Both can be served by peer-reviewed sources already in the paper.

**Fixes:**
- First `\cite{deroy2026structured}` (after "longstanding concern in AI safety research"): replaced with `\cite{hubinger2024sleeper, greenblatt2024alignment}`
- Deroy survey sentence: replaced with peer-reviewed motivational claim — *"Existing empirical work on alignment faking and backdoor persistence suggests that behavioral evaluations alone are insufficient for reliable alignment detection~\cite{greenblatt2024alignment, hubinger2024sleeper}: a model that strategically complies or conceals a backdoor can pass behavioral tests while maintaining deceptive internal representations---motivating the white-box representational monitoring approach our telemetry matrix provides."*
- `\bibitem{deroy2026structured}` removed from bibliography (no `\cite{}` calls remaining)

---

### Edit 16b — P3-A · §2.4 · GCG gradient access qualifier added
**Gap:** §2.4 mentioned GCG optimizes "via gradient signals" but didn't state the white-box requirement explicitly. Our method's gradient-free property is a meaningful applicability advantage.

**Fix:** Added `"---which require white-box gradient access to the target model---"` clause to the GCG sentence, and appended: `"Our MCTS, by contrast, operates without gradient access on a frozen model."`

---

### Edit 17 — P4-D · §2.5 · CRSM experimental prototype qualifier
**Gap:** `crsm2025` is a GitHub-only experimental prototype, single developer, explicitly "EXPERIMENTAL PROTOTYPE" per its own README — not peer-reviewed. NeurIPS reviewers cannot verify it via standard channels.

**Fix:** Added `"(an experimental prototype; not peer-reviewed)"` qualifier inline: `"CRSM~\cite{crsm2025} (an experimental prototype; not peer-reviewed) employs an asynchronous MCTS planner..."` All technical claims remain accurate and confirmed from source.

---

### Edit 18 — P4-E · §2.5 · Okazaki-RAG unpublished manuscript qualifier
**Gap:** `okazakirag2025` is an unpublished manuscript (August 2025), personal Gmail in header, no arXiv, no peer review, no experiments — a conceptual contribution only. NeurIPS reviewers cannot verify it.

**Fix:** Added `"(an unpublished conceptual contribution; not peer-reviewed)"` qualifier inline: `"the Okazaki-RAG framework~\cite{okazakirag2025} (an unpublished conceptual contribution; not peer-reviewed) implements a tree search..."` All technical claims confirmed accurate from source. Pairing with a peer-reviewed KV-cache serving paper (TurboRAG/CacheBlend/EPIC) deferred — source papers not yet downloaded.

---

### Edit 19 — Orphaned Bibliography Entries · Remove `contextual2025` + `oozeer2025transfer`
**Gap:** Both bibitems appeared in `\thebibliography` with no `\cite{}` calls in body text.
- `contextual2025`: OSF Preprints, no author attribution, citation quality concern; source not downloaded
- `oozeer2025transfer` (Oozeer et al., arXiv:2503.04429): source paper not downloaded for verification

**Fix:** Removed both `\bibitem` entries. Neither was cited in the paper. `oozeer2025transfer` can be re-added if §2.2 is extended to discuss cross-model activation transfer after source paper is read.

---

---

## Protocol Note — Table 2 and Experimental Results Sections (2026-04-23)

**What happened:** During this session, an incorrect assumption was made that Table 2 was a results table requiring published ASR figures from baseline papers. Four revision options were proposed before the surrounding experimental context (Experiment 2, the paper's own ASR definition, Phase B status) was read. Josh correctly identified that the comparison was invalid. The assumption was only corrected after probing.

**Why this matters:** Table 2 is a methods reference table until Phase B experiments complete. Adding published ASR figures from baseline papers would create a category error — different tasks, models, and success criteria in the same column — that would be harder to undo than to prevent.

**What to do differently:**

Before proposing any edit to Table 2 or any other results/comparison section, the following must be read first in this order:
1. The table's own caption and label
2. The surrounding subsection context (what experiment generates these numbers?)
3. The Evaluation Metrics subsection (what is the paper's own definition of each metric?)
4. The implementation status comments (`% [IMPL: ...]`) on the relevant experiment

Only after confirming that (a) the experiment has been run, (b) all baselines were measured on the same task under the same conditions, and (c) the success criterion is consistent across all rows, should any results be added to this table.

**General rule for all results/comparison sections:** If there is any uncertainty about whether a number belongs in a table, read the experimental design before suggesting it. Do not work from the assumption that a table is complete or that numbers from external papers are directly comparable to ours. The safe default is: if we didn't measure it in our own experiment, it doesn't go in a results column.

**Editor comment added to TeX:** A block comment has been inserted directly above Table 2 (before `\begin{table}`) documenting this decision and the three verification questions that must be answered before the table is revised.

---

## Blockers

None for paper text revision. Gate re-run on H100 still pending independently. P2-D code baseline (implement Belitsky mean-of-differences) is a Phase 3 task. P4-E: pairing Okazaki-RAG with a peer-reviewed KV-cache serving paper (TurboRAG/CacheBlend/EPIC) deferred pending source downloads.

---

## Gap Analysis — Edits 1–19 vs. Implementation Gaps (April 23, continuation)

After completing citation fixes, Max's three gap analysis documents (`contribution-1-telemetry-matrix.md`, `contribution-2-reversible-mcts.md`, `contribution-3-orthogonal-escape.md`) and the April 18 paper-revision-claims-audit were cross-referenced against Edits 1–19 to check whether citation fixes inadvertently introduced or strengthened claims about methodology or experiments that don't exist in the code.

**Finding:** Edits 1–19 did not introduce new wrong assumptions. All 19 edits targeted citation characterization — how we describe cited papers — and did not touch §4 (Proposed Method), §5 (Experimental Design), or §6 (Theoretical Analysis).

**However, four pre-existing methodology claims were identified as incorrect relative to code and unprotected by DRAFT-HOLD comments:**

| Gap ID | Location | Issue | Risk |
|---|---|---|---|
| GAP-C2-01 | §4.2, equations 8–11 | Per-layer `d_K^(ℓ)` notation throughout; code broadcasts single middle-layer vector | Reviewer will check code |
| GAP-C2-03 | §4.2 Memory Complexity paragraph | Claims ~50MB accumulator overhead; code allocates full-shape S×d accumulators (~M_KV actual) | Experiment 3 will directly falsify |
| GAP-C2-03+04 | §6 Memory Complexity Proposition | Claims O(M_KV + d·K_acc), independent of b; code stores baseline clone → 2×M_KV minimum | Flagship efficiency claim; measurable |
| GAP-C1-05 (2nd location) | §5.2 Phase A bullet | "RepE probe training via PCA" adjacent to `% [IMPL: DONE]` tag; code uses mean-diff | IMPL DONE tag makes false claim visible |

### Edit 20 — Four DRAFT-HOLD blocks added to TeX

Block comments added using the established format (╔...╚ box style). Locations:

- **~line 244** (before §4.2 subsubsection): GAP-C2-01 — per-layer steering notation
- **~line 291** (before §4.2 Memory Complexity paragraph): GAP-C2-03 — 50MB accumulator claim
- **~line 348** (before Phase A §5.2 PCA bullet): GAP-C1-05 second location — PCA/mean-diff adjacent to DONE tag
- **~line 504** (before §6 Memory Complexity Proposition): GAP-C2-03 + GAP-C2-04 — proposition + comparison numbers both wrong

All blocks document: gap ID, the specific issue, the decision (hold until Phase 3), draft fix wording, and code fix owner.

### Citation audit matrix updated (April 23, continuation)

- DRAFT-HOLD table: three new rows added (GAP-C1-05 second location, GAP-C2-01 with block comment reference, GAP-C2-03+GAP-C2-04)
- P3-A summary table row corrected: was "2 gaps remaining (gradient note; ASR figures)" — gradient access note was applied in Edit 16b; corrected to "1 gap remaining (ASR figures for Table 2)"

---

## Session Close (April 23)

### What was completed

- **Edits 7–19:** All citation text fixes through P2-E, P2-B, P2-C, P2-D, P2-A, P3-A, P4-G, P4-B, P4-A, P4-D, P4-E, and orphans. Every edit was preceded by a full source read.
- **Gap analysis:** Edits 1–19 cross-referenced against Max's three contribution gap docs — no new assumptions introduced by citation work.
- **Edit 20:** Four DRAFT-HOLD block comments added to protect unguarded implementation gap claims.
- **Matrix sync:** DRAFT-HOLD table updated with four new entries; P3-A summary row corrected.

### What's queued for next session

| Task | Priority | Notes |
|---|---|---|
| ASR figures for Table 2 (P3-A) | Medium | GCG figures already documented in matrix; MPA (P3-B) and DAMON (P3-C) figures need COLING/EMNLP PDFs |
| P2-D code baseline | High (Phase 3) | Implement Belitsky mean-of-differences KV steering in `logomesh/` |
| P4-E peer-reviewed companion | Medium | TurboRAG/CacheBlend/EPIC — download source before citing |
| Phase 3 code fixes (DRAFT-HOLD items) | Critical | GAP-C2-03, GAP-C2-04, GAP-C2-01, GAP-C1-05 — must precede paper fix application |
| Claims audit open decisions | High | OEI scope option (A/B/C), memory theorem verification, Experiment 3 in-scope decision |
