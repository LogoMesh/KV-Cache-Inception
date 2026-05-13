# Session A Prose Drafts v8 — Gemini Triage Mitigations: FM 2 + FM 4 — 2026-05-07

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits)
**Status:** Two pure-prose pieces continuing the triage of `docs/logs/2026-05-07_gemini-pre-mortem-EMNLP-findings.md`. Piece #10 (title pivot) was the routing-leverage edit; pieces #11 and #12 are the in-body language pieces that operationalize the partial-valid findings without absorbing the conflicting ones.

**Trigger:** §7 memo "2026-05-07 PM (later, post-Gemini-receipt) — Session A → Session B (Gemini EMNLP pre-mortem triage; piece #10 title pivot ready)" in `docs/logs/2026-05-05_recast-action-plan.md` queued FM 2 (scale-language tightening) and FM 4 (§7.4 / §5 Implementation Notes tone tightening) as subsequent drafts. This file produces them.

**Scope discipline:**
- **FM 2** ⇒ piece #11. We do NOT run on an 8B model (hardware constraint; deferred). We DO tighten language so the paper does not read as making a phenomenon claim at 1B/3B scale.
- **FM 4** ⇒ piece #12. We do NOT delete §7.4 (Option α+ explicitly preserves the honest-finding paragraph). We DO remove the "Phase 2 / Phase 3" developmental-staging language Gemini specifically flagged, in favor of "released implementation / future work" framing.
- **FM 3** (perplexity column + Degradation Control Baseline) — NOT in this batch. Gated on Josh wiring perplexity into the steered-generation script; no test execution possible this session per Josh's note.
- **FM 5** (light Bailey sharpening) — NOT in this batch. Will draft as piece #13 in a subsequent session if no countervailing input arrives from a second-pre-mortem run.

**Pieces in this file:**
1. **Piece #11 — FM 2 scale-language tightening** (two sub-pieces)
   - 11A: §5 Interpretation paragraph — add explicit "not a phenomenon claim at 1B/3B" sentence
   - 11B: §7.3 Limitations — add scale-dependent-phenomena caveat citing the relevant scale floors
2. **Piece #12 — FM 4 "Phase 2 / Phase 3" tone tightening** (four sub-pieces)
   - 12A: §5 Implementation Notes header — drop parenthetical
   - 12B: §5 Implementation Notes opening paragraph — strip developmental-staging framing
   - 12C: §5 Implementation Notes items (i)–(iv) — replace "Phase 2 implementation / deferred to Phase 3" with "released implementation / described in §7 Future Work"
   - 12D: §7.4 paragraph 3 — drop "as a Phase~2 approximation" qualifier

**Application order:** 11A → 11B → 12A → 12B → 12C → 12D. Pieces are textually independent but logically grouped (FM 2 first, then FM 4).

**Cross-reference impact:** none. All `\ref` / `\label` / `\cite` targets unchanged. The two new citations in piece #11B (`\cite{greenblatt2024alignment}`, `\cite{bailey2024obfuscated}`, `\cite{gupta2025rlobfuscation}`) are already in the bibliography (verified by grep before drafting).

**Page-budget impact:** rough estimate +50 words (piece #11A) + 70 words (piece #11B) – 30 words (piece #12 net, mostly verbatim length-equivalent rewrites). Net +90 words across both pieces. With the title pivot saving zero words, total v7 + v8 page-budget delta is approximately +90 words, or ~0.05 pages double-column. Negligible.

---

## 1. Piece #11A — §5 Interpretation paragraph: scale disclaimer

**TeX target:** Interpretation paragraph at TeX:362, immediately following Table~\ref{tab:exp1-results}.

**REVISION marker:** `[REVISION | Exp1-interp-scale-disclaimer | 2026-05-07]` (layered alongside `Exp1-reframe-with-results-α+` 2026-05-06 PM and the σ_H markers landed earlier today).

**Current text (TeX:362, replace in full):**

```latex
\textbf{Interpretation.} The results are consistent with the negative-result regime Bailey et al.~\cite{bailey2024obfuscated} report for gradient-trained input attacks: high-dimensional null-space saturation dominates, and orthogonal escape does not emerge from non-orthogonality-optimizing search at the model scales tested. We interpret this as a measurement under one specific monitor specification (rank-1 RepE honesty probe at the middle layer); whether richer monitors or hallucination-inducing input regimes produce a more discriminative signal is the natural next experimental step, and is not contradicted by these results.
```

**Replacement:**

```latex
\textbf{Interpretation.} The results are consistent with the negative-result regime Bailey et al.~\cite{bailey2024obfuscated} report for gradient-trained input attacks: high-dimensional null-space saturation dominates, and orthogonal escape does not emerge from non-orthogonality-optimizing search at the model scales tested. We do not interpret these observations as evidence about whether orthogonal escape arises in capability-dependent regimes: alignment-faking-class behaviors have so far been documented to emerge at frontier scale~\cite{greenblatt2024alignment}, and the Llama~3.2-1B and 3.2-3B experiments reported here characterize the measurement framework's behavior on consumer-hardware-accessible models rather than the underlying phenomenon at the scales where it is expected to manifest. We interpret the present numbers as a measurement under one specific monitor specification (rank-1 RepE honesty probe at the middle layer); whether richer monitors, hallucination-inducing input regimes, or frontier-scale evaluation produce a more discriminative signal is the natural next experimental step, and is not contradicted by these results.
```

**Notes for Session B:**

- Single paragraph replacement. Bailey-consistency framing preserved (first sentence unchanged). Final sentence's "richer monitors or hallucination-inducing input regimes" extended to "richer monitors, hallucination-inducing input regimes, or frontier-scale evaluation" — the natural-next-step list now includes the scale axis Gemini highlighted.
- New middle sentence is the load-bearing insertion: makes explicit that the present results do **not** make a phenomenon claim about 1B/3B alignment-faking behaviors, only a measurement claim about the framework's behavior at those scales. Cites `greenblatt2024alignment` (already in bibliography; verified at TeX ~605–610). Reviewer-blocking concern from Gemini FM 2 ("reviewers will attribute the negative result entirely to the 1B model's lack of representational capacity") is preempted by stating that we do not make the inference reviewers would object to.
- The phrase "alignment-faking-class behaviors" rather than "alignment faking" specifically — a hedge that includes related capability-dependent phenomena (sleeper-agent persistence, RL-Obfuscation generalization, etc.) without overclaiming what `greenblatt2024alignment` showed. Acceptable under Option α+ since alignment faking is one application area, not the headline claim.
- Net add: ~50 words. Does not introduce new labels or cross-references.

---

## 2. Piece #11B — §7.3 Limitations: scale-dependent-phenomena caveat

**TeX target:** §7.3 Limitations enumerated paragraph at TeX:515 (single-paragraph form). Insertion point: between the existing third sentence ("Third, the framework relies on H-Neuron activation tracking..." through "...remains an open theoretical liability that our experiments must address.") and the existing "Additionally, Zou et al.\ (2023) validated RepE on models of 7B parameters and above..." sentence.

**REVISION marker:** `[REVISION | §7.3-scale-caveat | 2026-05-07]` (new marker; sits adjacent to the existing `G1` 2026-04-24 marker at TeX:516).

**Current text — surrounding context (DO NOT REPLACE THESE LINES, just for orientation):**

```latex
... fifth, sixth ... (existing list continues) ... that our experiments must address. Additionally, Zou et al.\ (2023) validated RepE on models of 7B parameters and above; our 1B surrogate falls outside this validated range, and probe reliability at sub-7B scale is an open empirical question. Fourth, the current experimental design ...
```

**Insertion (place between "...that our experiments must address." and "Additionally, Zou et al.\ (2023) validated...") :**

```latex
The scale of empirical evaluation reported here (Llama~3.2-1B and 3.2-3B on consumer hardware) sits below the parameter range at which capability-dependent alignment-faking behaviors have been documented in the published literature: Anthropic's alignment-faking findings emerge at Claude~3 Opus and Claude~3.5 Sonnet scale~\cite{greenblatt2024alignment}; Bailey et al.\ evaluate at Llama-3-8B scale~\cite{bailey2024obfuscated}; the RL-Obfuscation evaluations span 7B--14B parameters~\cite{gupta2025rlobfuscation}. The 1B/3B experiments reported here therefore characterize the measurement framework on consumer-hardware-accessible models rather than the underlying phenomenon at the scales where it is expected to manifest, and frontier-scale characterization is deferred to Phase~B evaluation on H100-class hardware (§\ref{sec:future-work} not present; cf.\ Future Work subsection).
```

**Notes for Session B:**

- **Cross-reference verification needed before applying.** I drafted `§\ref{sec:future-work}` as the natural target, but the §7 Future Work subsection at TeX:532 has no `\label` (verified by grep on 2026-05-07 PM). Two options:
  - **Option A (recommended; minimal change):** drop the `§\ref{...}` and use a non-labeled phrase. Replace the final clause "deferred to Phase~B evaluation on H100-class hardware (§\ref{sec:future-work} not present; cf.\ Future Work subsection)" with simply "deferred to the Future Work subsection below."
  - **Option B:** add `\label{sec:future-work}` to the Future Work `\subsection` heading at TeX:532 in the same edit, and use the `\ref` form. This is a 2-line change (one label add, one ref). Marginally cleaner but introduces a label that isn't otherwise referenced.
  - Apply Option A unless you have a reason to prefer B. The Future Work subsection is structurally adjacent (next subsection); a linguistic pointer reads cleanly without the cross-ref machinery.
- The three citations (`greenblatt2024alignment`, `bailey2024obfuscated`, `gupta2025rlobfuscation`) are all verified present in the bibliography at TeX:564–622 by grep.
- The existing "Additionally, Zou et al.\ (2023) validated RepE on models of 7B parameters and above..." sentence remains exactly as-is, immediately following this insertion. The two scale caveats (RepE-7B for the probe, alignment-faking-frontier for the phenomenon) are now adjacent and read as a coherent scale-related limitation block.
- Net add: ~70 words.
- This insertion subsumes the implicit scale concern that the existing "Fourth, the current experimental design would benefit from cross-family scaling validation across architectures beyond the Llama family..." sentence touches on but does not address directly. We do not modify the "Fourth, ..." sentence — the cross-family concern is genuinely distinct (architecture diversity, not parameter-count scale).

**Final replacement form (for clarity, after applying Option A):**

```latex
... that our experiments must address. The scale of empirical evaluation reported here (Llama~3.2-1B and 3.2-3B on consumer hardware) sits below the parameter range at which capability-dependent alignment-faking behaviors have been documented in the published literature: Anthropic's alignment-faking findings emerge at Claude~3 Opus and Claude~3.5 Sonnet scale~\cite{greenblatt2024alignment}; Bailey et al.\ evaluate at Llama-3-8B scale~\cite{bailey2024obfuscated}; the RL-Obfuscation evaluations span 7B--14B parameters~\cite{gupta2025rlobfuscation}. The 1B/3B experiments reported here therefore characterize the measurement framework on consumer-hardware-accessible models rather than the underlying phenomenon at the scales where it is expected to manifest, and frontier-scale characterization is deferred to the Future Work subsection below. Additionally, Zou et al.\ (2023) validated RepE on models of 7B parameters and above; our 1B surrogate falls outside this validated range, and probe reliability at sub-7B scale is an open empirical question. ...
```

---

## 3. Piece #12A — §5 Implementation Notes header: drop developmental parenthetical

**TeX target:** `\paragraph{Implementation Notes (Phase 2 / Phase 3 Distinction).}` at TeX:298.

**REVISION marker:** `[REVISION | §5-impl-notes-header-tone-tighten | 2026-05-07]` (layered alongside `§5-impl-notes` 2026-05-05).

**Current line (TeX:298, replace in full):**

```latex
\paragraph{Implementation Notes (Phase 2 / Phase 3 Distinction).}
```

**Replacement:**

```latex
\paragraph{Implementation Notes.}
```

**Notes for Session B:** Trivial single-line change. Drops the parenthetical that signals a developmental staging diary; the heading now reads as a standard "Implementation Notes" section that any methods paper might carry.

---

## 4. Piece #12B — §5 Implementation Notes opening paragraph: strip developmental-staging framing

**TeX target:** Opening paragraph at TeX:299 (single source line, paragraph wraps in TeX).

**REVISION marker:** `[REVISION | §5-impl-notes-opening-tone-tighten | 2026-05-07]`

**Current text (TeX:299, replace in full):**

```latex
The reference implementation backing this submission is a Phase 2 prototype on Llama~3.2-1B-Instruct and Llama~3.2-3B-Instruct on consumer hardware (NVIDIA RTX~3060). Several elements of the method as written in §4 and §6 differ from this reference implementation in identifiable, bounded ways; we list them here as a single source of truth so a reviewer reading the paper alongside the released code finds one consolidated reference rather than scattered caveats. The asymptotic claims---exact reversibility (Theorem~1), memory-complexity scaling independent of the branching factor (Proposition in §6), and the telemetry matrix tensor shape (Eq.~\ref{eq:telemetry_matrix})---are unaffected by the distinctions below; what differs is constants and operational details that resolve once the corresponding Phase~3 code work lands.
```

**Replacement:**

```latex
The reference implementation backing this submission runs on Llama~3.2-1B-Instruct and Llama~3.2-3B-Instruct on consumer hardware (NVIDIA RTX~3060). Several elements of the method as written in §4 and §6 differ from this implementation in identifiable, bounded ways; we list them here as a single source of truth so that a reviewer reading the paper alongside the released code finds one consolidated reference rather than scattered caveats. The asymptotic claims---exact reversibility (Theorem~1), memory-complexity scaling independent of the branching factor (Proposition in §6), and the telemetry matrix tensor shape (Eq.~\ref{eq:telemetry_matrix})---are unaffected by the distinctions below; what differs is constants and operational details, with the corresponding extensions described in §7 Future Work.
```

**Diff summary:**

- "is a Phase 2 prototype on..." → "runs on..." (drops "Phase 2 prototype" framing).
- "differ from this reference implementation" → "differ from this implementation" (cosmetic; "reference" already implies the same thing).
- "...resolve once the corresponding Phase~3 code work lands" → "...with the corresponding extensions described in §7 Future Work" (concrete pointer to a paper section that exists, replacing the future-development promise).

**Notes for Session B:** The substantive content is unchanged — five distinctions, asymptotics-unaffected statement, future-work pointer. Only the developmental-staging connotation is removed. Net length identical to ±5 words.

---

## 5. Piece #12C — §5 Implementation Notes items (i)–(iv): replace "Phase 2 implementation / Phase 3" with "released implementation / §7 Future Work"

**TeX target:** Items (i) through (iv) at TeX:301, 303, 305, 307. Item (v) "On the reversibility theorem" at TeX:309 is already clean (does not use "Phase 2 / Phase 3" framing) — leave unchanged.

**REVISION marker:** `[REVISION | §5-impl-notes-items-tone-tighten | 2026-05-07]` (single marker for all four item replacements; landed once at the start of the items block).

### 5.1 Item (i) at TeX:301

**Current:**

```latex
\textit{(i) RepE probe extraction (§4.1, §5.2).} The text describes the Linear Artificial Tomography procedure of \citet{zou2023repe} using PCA on contrastive pairs. The Phase~2 implementation uses a difference-in-means estimator, which approximates the leading PCA component on contrastive pairs in expectation but is not equivalent to LAT as published. PCA-based extraction is deferred to Phase~3.
```

**Replacement:**

```latex
\textit{(i) RepE probe extraction (§4.1, §5.2).} The text describes the Linear Artificial Tomography procedure of \citet{zou2023repe} using PCA on contrastive pairs. The released implementation uses a difference-in-means estimator, which approximates the leading PCA component on contrastive pairs in expectation but is not equivalent to LAT as published. PCA-based extraction is described in §7 Future Work.
```

### 5.2 Item (ii) at TeX:303

**Current:**

```latex
\textit{(ii) Layer aggregation for the honesty probe (§4.1).} The Phase~2 implementation aggregates the RepE honesty signal across all $L$ layers; \citet{zou2023repe} validate the honesty probe on the middle layers only. Layer-restricted aggregation is deferred to Phase~3.
```

**Replacement:**

```latex
\textit{(ii) Layer aggregation for the honesty probe (§4.1).} The released implementation aggregates the RepE honesty signal across all $L$ layers; \citet{zou2023repe} validate the honesty probe on the middle layers only. Layer-restricted aggregation is described in §7 Future Work.
```

### 5.3 Item (iii) at TeX:305

**Current:**

```latex
\textit{(iii) Per-layer steering vectors (§4.2).} The notation in §4.2 presents per-layer steering vectors $\mathbf{d}_K^{(\ell)}, \mathbf{d}_V^{(\ell)}$. The Phase~2 implementation broadcasts a single middle-layer vector to all layers; per-layer independent steering is deferred to Phase~3.
```

**Replacement:**

```latex
\textit{(iii) Per-layer steering vectors (§4.2).} The notation in §4.2 presents per-layer steering vectors $\mathbf{d}_K^{(\ell)}, \mathbf{d}_V^{(\ell)}$. The released implementation broadcasts a single middle-layer vector to all layers; per-layer independent steering is described in §7 Future Work.
```

### 5.4 Item (iv) at TeX:307

**Current:**

```latex
\textit{(iv) Sparse accumulators and baseline-clone overhead (§4.2 + §6).} The accumulator analysis in §4.2 and the worked example in §6 treat $K_{\text{acc}}$ as a sparse $S' \times d$ allocation with $S' \ll S$, yielding an example overhead of approximately $50$~MB at $20$B scale. The Phase~2 implementation allocates full-shape $S \times d$ accumulators and additionally stores a baseline cache clone of size $M_{\text{KV}}$ to enable rollback. The asymptotic claim that memory is independent of the branching factor $b$ (§6 Memory Complexity Proposition) is unchanged. The constant factor in the Phase~2 implementation is approximately $2$--$3 \cdot M_{\text{KV}}$ rather than the $\sim 1 \cdot M_{\text{KV}}$ implied by the worked example in §6: the $40.05$~GB figure quoted there reflects sparse-accumulator and no-clone assumptions, while the actual Phase~2 floor at $20$B scale is approximately $80$~GB before any accumulator overhead. Sparse-accumulator implementation, baseline-clone elimination, and corrected example numbers backed by direct VRAM measurement are required to bring §6's worked example into agreement with the Phase~2 implementation, and are deferred to Phase~3.
```

**Replacement:**

```latex
\textit{(iv) Sparse accumulators and baseline-clone overhead (§4.2 + §6).} The accumulator analysis in §4.2 and the worked example in §6 treat $K_{\text{acc}}$ as a sparse $S' \times d$ allocation with $S' \ll S$, yielding an example overhead of approximately $50$~MB at $20$B scale. The released implementation allocates full-shape $S \times d$ accumulators and additionally stores a baseline cache clone of size $M_{\text{KV}}$ to enable rollback. The asymptotic claim that memory is independent of the branching factor $b$ (§6 Memory Complexity Proposition) is unchanged. The constant factor in the released implementation is approximately $2$--$3 \cdot M_{\text{KV}}$ rather than the $\sim 1 \cdot M_{\text{KV}}$ implied by the worked example in §6: the $40.05$~GB figure quoted there reflects sparse-accumulator and no-clone assumptions, while the actual floor at $20$B scale is approximately $80$~GB before any accumulator overhead. Sparse-accumulator implementation, baseline-clone elimination, and corrected example numbers backed by direct VRAM measurement are required to bring §6's worked example into agreement with the released implementation, and are described in §7 Future Work.
```

**Notes for Session B (12C overall):**

- Substantive content of every item is unchanged. Only the developmental-staging framing is replaced.
- The "deferred to Phase~3" suffix is consistently replaced by "described in §7 Future Work." This is verifiable: the §7 Future Work subsection at TeX:532 explicitly discusses sparse accumulators, scaling, and Phase B follow-on work — so the pointer is honest.
- Item (iv) replaces "Phase~2 implementation" four times in the original; my replacement keeps the last occurrence's "released implementation" form for consistency.
- Item (v) ("On the reversibility theorem") at TeX:309 already starts with "Theorem~1 (Exact Reversibility...) is independent of the four distinctions above" and uses "constant-factor revisions to §6" — no Phase 2/3 language, no edit needed.
- Net length: each item rewrite is ~length-equivalent (small compressions from "Phase~2 implementation" → "released implementation"). Total piece #12C net: ~−15 words.

**Coordinated-update flag:** §5 §4.6 row 168 (Memory Complexity Proposition four-place coordinated rewrite, gated on Max's simplified Exp 3) lists the same item (iv) numbers (40.05 GB, 80 GB, 2–3·M_KV). When Max's measurements land, the Memory-Complexity-Proposition coordinated-update batch will need to update these numbers in this item too — but the developmental-staging tone tightening done here is independent of the numerical-update batch and lands cleanly now.

---

## 6. Piece #12D — §7.4 paragraph 3: drop "as a Phase~2 approximation" qualifier

**TeX target:** Single sentence within §7.4 paragraph 3 at TeX:530.

**REVISION marker:** `[REVISION | §7.4-σH-resolved-tone-tighten | 2026-05-07]` (sixth layered marker on §7.4 paragraph 3; preserves the full chronological audit trail).

**Current sentence (replace in place; TeX:530, mid-paragraph):**

```latex
The constancy resolved when we adopted per-layer H-Neuron calibration: in our initial implementation a single set of top-$K$ H-Neuron indices was identified at the deepest layer and applied uniformly across the layer stack as a Phase~2 approximation, while Eq.~\ref{eq:hneuron_stress} as written specifies per-layer indices and per-layer baselines.
```

**Replacement:**

```latex
The constancy resolved when we adopted per-layer H-Neuron calibration: our initial implementation identified a single set of top-$K$ H-Neuron indices at the deepest layer and applied them uniformly across the layer stack, while Eq.~\ref{eq:hneuron_stress} as written specifies per-layer indices and per-layer baselines.
```

**Diff summary:**

- "in our initial implementation a single set of top-$K$ H-Neuron indices was identified..." → "our initial implementation identified a single set of top-$K$ H-Neuron indices..." (active voice, cleaner).
- "...applied uniformly across the layer stack **as a Phase~2 approximation**, while..." → "...applied **them** uniformly across the layer stack, while..." (drops the Phase 2 qualifier; the next clause already explains the approximation).

**Notes for Session B:**

- This is the sixth REVISION marker layered on §7.4 paragraph 3 (`§7-measurement-prereqs` 2026-05-06 AM, `§7.4-post-fix-update` 2026-05-06 PM, `§7.4-σH-verification-update` 2026-05-06 EOD, `§7.4-σH-investigation-update` 2026-05-07 PM, `§7.4-σH-resolved` 2026-05-07 PM-later, and now `§7.4-σH-resolved-tone-tighten`). Audit trail is intentional — every substantive change to this paragraph is recorded.
- The next clause "while Eq.~\ref{eq:hneuron_stress} as written specifies per-layer indices and per-layer baselines" is the explanatory weight of the original "Phase 2 approximation" qualifier. Dropping the Phase-2 phrase loses no information; it just removes the developer-log connotation that "Phase 2" carries.
- Net length: −7 words.
- Rest of §7.4 paragraph 3 is preserved verbatim from the `§7.4-σH-resolved` rewrite landed earlier today.

---

## 7. Items intentionally NOT touched in v8

For the audit trail, here is what I considered and deliberately left for later:

1. **§5.1 "Phase 0: Architectural Decoupling" / §5.2 "Phase A: Local Prototyping" subsection headers (TeX:311, TeX:316).** These use "Phase 0 / Phase A" labels that are project-roadmap rather than developmental-staging-of-this-implementation. They do read somewhat unconventional for an EMNLP paper (which prefers flatter section structure), but the cross-reference and structural risk of restructuring §5 is not justified by tone tightening alone. Defer to a possible piece #14+ if a second-pre-mortem run flags them specifically.

2. **§7 Future Work subsection (TeX:532).** Mentions "Phase~B" twice ("deferred to Phase~B", "alongside the Phase~B submission"). "Phase B" is the same project-roadmap label as Phase 0/A. Same disposition: leave for now; restructure if specifically flagged.

3. **§7.4 paragraph 3 — investigation-arc preservation.** The "we ran a controlled three-prompt comparison... we additionally tested the per-neuron baseline-subtraction-and-ReLU formulation" sentences read like a methodological narrative rather than a developer log post-`§7.4-σH-resolved`. Gemini's FM 4 critique is specifically about confessional language ("Phase 2 / Phase 3 distinction") rather than methodological process. Investigation-arc preservation is a Session A intentional choice (per Option α+ "we surface, we resolve where we can, we honestly flag what we can't"). Not touched.

4. **§5.1.5 §5 line 332 "Phase~B follow-on work" forward-reference.** Same project-roadmap label; not developmental-staging. Not touched.

5. **TeX:325 DRAFT-HOLD-block4-consolidated comment** mentions "Phase 3" in a comment line. Comments are not visible in the rendered PDF and bear no reviewer-visibility risk. Not touched.

6. **The §7.4 § header itself** ("Empirical Measurement Prerequisites and Initial Observations" at TeX:520): some prior reviewers might find "Initial Observations" hedge-y. Considered tightening to just "Empirical Measurement Prerequisites" — but the section now contains substantive observations (post-`§7.4-σH-resolved`) and the title accurately advertises that. Not touched.

---

## 8. Quick checklist for Session B before applying

1. **Verify** Future Work label state: `grep -n "^\\\\subsection{Future Work}" docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` and confirm no `\label{...}` is on that line. If absent, apply piece #11B Option A (linguistic pointer); if a label exists or you elect to add one, apply Option B.
2. **Verify** the three citations in piece #11B exist in the bibliography by `grep -n "\\\\bibitem{greenblatt2024alignment}\|\\\\bibitem{bailey2024obfuscated}\|\\\\bibitem{gupta2025rlobfuscation}"` — Session A confirmed all three present, but a second look during application is cheap insurance.
3. **Apply piece #12C as one batch** (single REVISION marker for all four item rewrites, landed once before item (i)). Each item-level edit is ~length-equivalent, so the file's overall line numbering should not shift meaningfully.
4. **After all six sub-pieces apply**, run `grep -n "Phase 2\|Phase 3\|Phase~2\|Phase~3" docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` and verify the only remaining matches are: project-roadmap labels (§5.1 Phase 0, §5.2 Phase A, §7 Phase B references) and REVISION-marker comment lines (which are not rendered). No prose-text matches should remain.

---

## 9. What v8 is NOT

For maximal clarity to anyone reading this file later:

- **v8 does NOT introduce 8B-scale results.** Gemini's recommendation to run on Llama-3.1-8B is a hardware-blocked deferral for this submission cycle. We tighten the language to make the absence honest; we do not pretend to have run it.
- **v8 does NOT delete §7.4.** Gemini's FM 4 deletion recommendation conflicts with locked Option α+. We tone-tighten only.
- **v8 does NOT add the perplexity column or Degradation Control Baseline (FM 3 mitigations).** These require code execution which Josh confirmed is unavailable this session (video rendering on the user's machine). Will draft as piece #13+ when the perplexity column data is available.
- **v8 does NOT sharpen the §2.2 Bailey paragraph (FM 5 mitigation).** Light incremental edit; deferred to piece #13+. Optional addition pending second-pre-mortem outcome.

---

End of v8 drafts file. Estimated Session B application time: ~10 minutes total across all six sub-pieces. All edits are local; none require restructuring or new label additions (assuming Option A on piece #11B).
