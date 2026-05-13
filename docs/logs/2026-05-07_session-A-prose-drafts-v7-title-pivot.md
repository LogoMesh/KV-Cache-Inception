# Session A Prose Drafts v7 — Title Pivot under Option α+ — 2026-05-07

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits)
**Status:** One single-line replacement following triage of Gemini 3.1 Pro EMNLP pre-mortem (`docs/logs/2026-05-07_gemini-pre-mortem-EMNLP-findings.md`). Closes the residual half-recast in TeX:39 — the body has been reframed under Option α+ across §1 / §2 / §3 / §5 / §7, but the title still leads with "KV-Cache Inception" and names "Detecting and Stress-Testing Alignment Faking" as the paper's purpose.

**Trigger:** Gemini Failure Mode 1 ("Systems-First Algorithmic Framing in a Semantic Venue") flagged the current title as a routing problem — systems-heavy titles attract systems reviewers who'll demand baselines against alternative tree-search heuristics, while an algorithm-with-application title routes to interpretability reviewers aligned with the paper's actual content. Gemini's own counter-proposal ("Detecting Latent Alignment Faking via Reversible KV-Cache Search") would re-headline alignment-faking detection — the framing Option α+ explicitly walked away from on 2026-05-05 PM. **We adopt neither the current title nor Gemini's. We adopt the working title already canonical in the recast plan §0 and the README rewritten 2026-05-07 ~17:00.**

**Narrative shift in the paper:** the body already reads as algorithm-headline-with-Bailey-complementing-application. The title currently mismatches the body, signaling unfinished recast to any first-pass reviewer. This is the cheapest single edit that closes the framing loop end-to-end.

**Pieces in this file:**
1. **Piece #10 — Title replacement** at TeX:39

**Application order:** N/A (single piece).

**Cross-reference impact:** none. Title is not labeled and is not referenced by any `\ref` / `\cite`. The bibtex citation block in `README.md` ~line 200 uses the same title text — no further change needed since README was rewritten with the new title in place.

---

## 1. Piece #10 — Title replacement

**TeX target:** `\title{...}` at TeX:39 (single line).

**REVISION marker:** `[REVISION | title-recast-α+ | 2026-05-07]`

**Current line (TeX:39, replace in full):**

```latex
\title{KV-Cache Inception: Reversible Monte Carlo Tree Search in Latent Space for Detecting and Stress-Testing Alignment Faking in Large Language Models}
```

**Replacement:**

```latex
% [REVISION | title-recast-α+ | 2026-05-07 | Replaced full title from "KV-Cache Inception: Reversible Monte Carlo Tree Search in Latent Space for Detecting and Stress-Testing Alignment Faking in Large Language Models" to the new working title canonical in `docs/logs/2026-05-05_recast-action-plan.md` §0 and the README rewritten 2026-05-07 ~17:00. Aligns the title with the Option α+ recast already implemented in §1 abstract (`abstract-recast-α+`), §1 Contributions (`contributions-recast-α+`), §3 (`§3-reframe-α+`), §5 Experiment 1 (`Exp1-reframe-with-results-α+`), and §7.4 (`§7-measurement-prereqs` + four layered updates). Drops "KV-Cache Inception" branding (NeurIPS-era artifact; not used anywhere in the recast body) and "Detecting and Stress-Testing Alignment Faking" framing (one application, not the paper's headline subject under Option α+). Triggered by Gemini 3.1 Pro EMNLP pre-mortem FM 1 (title routing); we adopt the Option α+ working title rather than Gemini's counter-proposal which would re-headline alignment faking. — pending audit]
\title{Reversible Monte Carlo Tree Search in Latent Space: A Memory-Bounded Algorithm for Inference-Time Latent-Space Analysis of Frozen Language Models}
```

**Notes for Session B:**

- This is a one-line change plus a marker comment line. Estimated application time: ~2 min.
- Title text is verbatim identical to the working title in `docs/logs/2026-05-05_recast-action-plan.md` line 18 ("Working title:") and in `README.md` (rewritten 2026-05-07; first-line H1). Verbatim match is intentional — README and TeX should not drift.
- The new title contains "Latent Space" twice (once in "Latent Space:" before the colon; once in "Latent-Space Analysis" after). Mild redundancy, but the colon-divided form ("Method: Property of Method on Application") is conventional for ML papers and reads cleanly. If you have a strong contrary opinion (e.g., prefer "Reversible KV-Cache MCTS: A Memory-Bounded Algorithm for Inference-Time Latent-Space Analysis on Frozen Language Models"), flag back via §7 — but I'd default to the canonical version unless you see a specific reason to break consistency with the README.
- Author block, date block, and `\maketitle` immediately below are untouched.
- **Submission caveat (no action today, just flagging):** EMNLP submissions are double-blind. The current `\author{[Authors]}` placeholder is correct for double-blind. When EMNLP style template lands and Session B does the style switch, double-blind verification is part of that pass — no anonymization concern in the title itself (the title contains no team / institution / author identifiers).

---

## What this does *not* address from the Gemini pre-mortem

This drafts file is intentionally narrow — title pivot only. The full Gemini triage outcome is captured in the §7 memo "2026-05-07 PM (later, post-Gemini-receipt) — Session A → Session B (Gemini EMNLP pre-mortem triage; piece #10 title pivot ready)" appended to `docs/logs/2026-05-05_recast-action-plan.md`. That memo lists:

- The two findings that conflict with locked Option α+ (FM 1 full reversion to OEI-headline; FM 4 §7.4 deletion) — push-back items, not edit items.
- The valid findings beyond the title (FM 2 scale-language tightening; FM 3 perplexity column + degradation control baseline; FM 4 §7.4 tone tightening; FM 5 light Bailey sharpening) — to be drafted as separate pieces in subsequent sessions.
- The deferred findings (8B run; standard NLP benchmarks) — to be logged in §4.6 standing tracker as post-EMNLP work.
- The unblocking dependency (EMNLP style template download → page-budget verification) — Josh's lane, no Session A draft needed.

Piece #10 is the highest-leverage single-pass edit because title routing is binary: a reviewer decides which area-chair queue to sit in based on the title and abstract. We've already updated the abstract; the title was the residual half-recast.
