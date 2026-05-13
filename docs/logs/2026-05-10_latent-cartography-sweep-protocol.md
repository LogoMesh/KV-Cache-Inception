# Latent Cartography Sweep Protocol — 2026-05-10

**Author:** Session A
**Audience:** Diagnostic Claude session #6 (to be spawned Day 4 of execution plan, scheduled 2026-05-13)
**Status:** Pre-flight protocol. Lock before diagnostic session spawn. Lives at `docs/logs/2026-05-10_latent-cartography-sweep-protocol.md` per 14-day execution plan §1 Days 1-3.

## 1. Purpose

Map the structural pre-training priors that emerge when Reversible KV-Cache MCTS applies entropy-minimization steering pressure across a diverse taxonomy of prompt classes. This becomes Experiment 2 (§\ref{sec:exp2}) of the EMNLP submission, paired with Track F's negative-control evidence as Experiment 1 (§\ref{sec:exp1}).

The hypothesis under test is Hypothesis~\ref{hyp:goodhart} (Reward-driven Dimensional Escape) extended beyond the multiple-choice regime of Track F. Specifically: *the reward-driven Goodhart-collapse pathology is not regime-specific to multiple-choice reasoning. The same cache-geometry pathology surfaces across prompt classes, but the specific terminal-token distribution that the search converges upon varies systematically with the prompt's semantic class — revealing the structural priors of pre-training and instruction-tuning.*

If the hypothesis holds: we have a positive empirical mapping that complements Track F's controlled negative result. Reversible MCTS becomes a *probe* of pre-training priors at multiple semantic levels, and the paper's empirical contribution gains a structural dimension beyond "we observed Goodhart at one scale on one benchmark." If the hypothesis fails (no class-dependent variation in the search's terminal behavior): plasticity rule kicks in (see §6 of this protocol) and Cartography demotes to a Limitations item.

## 2. Models and infrastructure

- **Llama-3.2-1B-Instruct** (primary, RTX 3060 12GB, ~3.5 GiB peak per Track D)
- **Llama-3.2-3B-Instruct** (secondary, ~9 GiB peak per Track D)
- **No new code in `logomesh/`** — implement as `scripts/diagnose_track_g_cartography.py`, read-only on `logomesh/*` per the same diagnostic-session discipline as Tracks A/C/D/F.
- Reuse `FP32Accumulator`, `_extract_kv_tensors`, `_kv_eval_cache` primitives (same import pattern as `scripts/diagnose_track_f_negcontrol.py`).
- Reuse Track F's path-sampling search structure (depth 3, 27 paths sampled from a 3-element alpha set, MCTS leaf evaluation by next-token entropy reward). Full enumeration of the 3³=27 path tree at depth 3 means UCB-guided expansion is unnecessary — the candidate set is fully evaluated at every item. Track F §10 documents the methodological equivalence.

## 3. Prompt classes

100 items per class, 4 classes total → 400 distinct items per scale × 2 scales = 800 total runs. With 6 alpha values per item, total records = **4800**.

| Class | Description | Source | Eval metric |
|---|---|---|---|
| **C1: Factual recall** | Capital cities, historical dates, simple arithmetic | Hand-constructed 100-item set; format: "Q: What is the capital of {country}? A:" / "Q: In what year did {event} occur? A:" / "Q: What is {n1} + {n2}? A:" | First-token (after answer-tag) equality with gold |
| **C2: Logical deduction** | ARC-Easy first 100 items (continuity with Track F) | `allenai/ai2_arc` ARC-Easy test split, indices 0-99 | First-letter equality with gold (A/B/C/D) |
| **C3: Moral/preference** | TruthfulQA mc1 first 100 items (broad coverage of common-knowledge contradictions) | `truthful_qa` mc1 split, indices 0-99 | First-letter equality with gold |
| **C4: Creative completion** | HellaSwag continuation prompts, re-formatted as open continuation (option list omitted, model generates natural continuation) | `Rowan/hellaswag` validation split, indices 0-99 with option lists stripped | Top-1 next-token entropy + first-token surprisal at gold continuation start |

**Class C2 is the calibration class** — it overlaps with Track F's first 100 ARC items, ensuring continuity with Experiment 1 and providing a cross-check that the new Cartography pipeline reproduces Track F's measured 1B "predicts A under heavy steering" pattern. The other three classes are Cartography-novel.

**Why these four classes:** they span the four corners of the prompt taxonomy that EMNLP reviewers will look for. Factual recall tests low-entropy unambiguous prompts. Logical deduction tests structured reasoning prompts (and matches Track F's regime). Moral/preference tests prompts where instruction-tuning produces the strongest behavioral priors (yes/no/refusal). Creative completion tests open-ended generation where there's no canonical "correct letter" the search could collapse to. If the search produces *different* prior-amplification signatures across these four corners, that's the Cartography signal we need.

**Hand-construction provenance for C1:** the diagnostic session generates the C1 100-item set as a deterministic hand-list (no model generation). Suggested seed list:
- 25 capitals (modern major sovereign states; one-word capital answers)
- 25 historical dates (non-controversial — moon landing, fall of Berlin Wall, World War 1 start, etc.)
- 25 simple arithmetic (single-digit + single-digit, single-digit × single-digit, no carry-over)
- 25 single-fact common knowledge (chemistry symbols, vocabulary definitions, basic scientific facts)

The exact 100-item C1 set is appended to the `track-G-cartography-1B.md` report as Appendix A for reproducibility.

## 4. Steering direction

Same protocol as Track F: random unit vector in $d_{\text{model}}$ space, seeded with `--seed 42`, **shared across all items, all alphas, and both scales** (so the comparison is *what changes with α and class* not *what changes with direction*). This isolates magnitude effects and makes 1B/3B comparable.

## 5. Alpha grid

`{0.0, 0.1, 0.5, 1.0, 2.0, 5.0}` — wider than Track F's `{0.1, 0.5, 1.0}` so we observe where prior-amplification *kicks in* across magnitudes. α=0.0 gives the unsteered (greedy-equivalent) baseline; α=5.0 gives heavy steering well beyond Track F's range.

The alpha grid is shared across all classes — the Cartography heatmap requires that we can compare class-specific responses at fixed α.

## 6. Per-item record fields

For each `(class, item_idx, alpha, scale)` tuple, write to `scripts/_track_g_results_{model}.json`:

```json
{
  "class": "C1|C2|C3|C4",
  "item_idx": 0,
  "alpha": 0.0,
  "scale": "1B|3B",
  "prompt_text": "...",
  "gold": "...",
  "pred_first_token": "...",
  "pred_first_letter_or_continuation_5tok": "...",
  "correct": true,
  "next_token_entropy_at_steered_cache": 1.234,
  "next_token_top5_distribution": [["the", 0.42], ["a", 0.18], ...],
  "perplexity_5tok_under_unsteered": 1.41,
  "best_path_alphas": [1.0, 1.0, 1.0],
  "best_path_reward": -0.097,
  "search_reward_mean_over_27": -0.110,
  "search_reward_best": -0.097,
  "wallclock_sec": 5.4
}
```

This is Track F's record schema plus the `next_token_top5_distribution` field (needed for the Cartography heatmap in the report). Persist every 25 items in case of crash.

## 7. Report-level deliverables

The diagnostic session writes two reports — `docs/logs/2026-05-13_track-G-cartography-1B.md` and `docs/logs/2026-05-14_track-G-cartography-3B.md` — each containing:

1. **Per-class accuracy table** (4 classes × 6 alphas, with Wilson 95% CIs). Same structure as Track F §2 but with class as the row axis.
2. **Top-5 token distribution at α=5.0 vs α=0.0 for each class** — the *Cartography* contribution. Reveals which tokens the search converges to under heavy steering, as a function of class. Render as a side-by-side table per class; the qualitative pattern across classes is the headline.
3. **First-letter (or first-token) prior-amplification rate per class.** For C1/C2/C3: fraction of items where prediction collapses to a single dominant token under heavy steering (e.g., Track F measured "predicts A" at 185/200 = 92.5% on C2 at 1B). For C4: KL divergence between α=5.0 and α=0.0 next-token distributions (since C4 is open-continuation, not letter-MCQ).
4. **Best-path α-tuple distribution per class.** Does the search consistently pick `(1.0, 1.0, 1.0)` heavy-steering on some classes but `(0.1, 0.1, 0.1)` minimal-steering on others? This is the structural finding that supports the §3.2 Hypothesis 2 instantiation across classes. A class-dependent shift in best-path α-tuples is the strongest evidence of Cartography signal.
5. **Per-class entropy trajectory plot.** Entropy as a function of α across all items in the class, mean ± std. Different classes should show different curves *if Hypothesis~\ref{hyp:goodhart} interacts with class semantics*.
6. **Mechanism diagnosis per class.** What specific token does the search converge to under heavy steering for each class? Compare with Track F's "letter A on 185/200" finding for the C2 calibration class. Explicit per-class mechanism explanations modeled on Track F §7 (which identified the Goodhart route as "letter prior at 1B, greedy collapse at 3B" for the multiple-choice regime).
7. **Cross-scale comparison** (1B report references 3B report's table and vice versa): does the 3B model's Cartography signal preserve the structural variation observed at 1B, or does the 3B greedy-collapse pathology (Track F §1) erase it? This is the mechanism-level test of whether Cartography is scale-robust.

## 8. Success criteria (what makes this a usable Experiment 2)

Three criteria. Cartography lands as a clean Experiment 2 if **at least 2 of 3** hold. If only 1 holds, Experiment 2 is borderline and Day 9 plasticity rule may demote it. If none hold, Experiment 2 is a Limitations item and the paper ships as pure Option D.

- **C-1: Cross-class variation in best-path α-tuple distribution.** If all 4 classes show the same `(1.0, 1.0, 1.0)` modal best path, the finding collapses to "Goodhart everywhere, no structural variation." If classes split (e.g., C1 mode `(0.1,0.1,0.1)`, C2 mode `(1.0,1.0,1.0)`, C3 mode `(0.5,0.5,1.0)`, C4 mode `(2.0,2.0,2.0)`), structural Cartography signal exists.
- **C-2: Cross-class variation in terminal-token concentration.** Different classes converge to different "favorite tokens" under heavy steering (e.g., C1 → "the" or definite-article prior, C2 → "A" letter prior matching Track F, C3 → "yes" or refusal-prior, C4 → some semantic anchor like end-of-sequence). If all classes converge to the same token, no Cartography signal.
- **C-3: Mechanism plausibility.** The terminal tokens chosen by the search are interpretable as instruction-tuning priors specific to each class. The diagnostic session's per-class mechanism diagnosis (deliverable #6 above) reads coherently to a mechanistic-interpretability reviewer.

## 9. Wall-clock budget

Per Track F's measured rates (5.5 s/item at 1B, 10.1 s/item at 3B) — applying the team's `~50% of estimated time` factor would underestimate here, so we use the conservative measured rates:

| Step | Budget | Total |
|---|---|---|
| 1B sweep | 100 items × 4 classes × 6 alphas × 5.5 s = 13 200 s | **3.7 hours** |
| 3B sweep | 100 items × 4 classes × 6 alphas × 10.1 s = 24 240 s | **6.7 hours** |
| Two model loads (1B + 3B) | ~85 s + ~140 s | ~4 minutes |
| Report writing (per scale) | ~1 hour each (parallel-able) | 2 hours |
| **Total GPU time** | | **~10.4 hours** |
| **Total calendar time including reports** | | **~12-13 hours** |

Days 4-5 of the 14-day plan provide 48 hours of calendar time. ~3× safety margin.

## 10. Plasticity rules (per 14-day plan §2 Trigger row 1)

- **By Day 7 EOD: sweeps fail to produce coherent structural finding (0 of 3 success criteria above hold).** YELLOW → Session A drafts a "Cartography minimal contribution" version (just 1-2 prompt classes that did show variation). If by Day 9 still no signal: **NO-GO on Cartography** → demote to Limitations item, ship pure Option D paper with Track F as sole empirical content.

- **By Day 7 EOD: 1 of 3 success criteria holds.** YELLOW → Session A drafts the Experiment 2 section with the held criterion as the central finding and the missing two as honest Limitations bullets. The probability estimate for the paper drops to the 30-40% middle band.

- **By Day 7 EOD: 2-3 of 3 success criteria hold.** GREEN → Session A drafts the Experiment 2 section as planned. Probability anchors at the 45-55% Gemini-round-4 high end.

- **GPU access disrupted before Day 7 EOD** (renders, hardware issues per 14-day plan §6 risk register). Compress to weekend block; the diagnostic session should be designed to be resumable from JSON checkpoints (every 25 items).

## 11. Inputs the diagnostic session needs

The Day 4 spawn of Diagnostic session #6 should be given:

1. **This protocol file** (paste verbatim).
2. **Track F's report** (`2026-05-11_track-F-negative-control-report.md`) — for protocol continuity, especially §10 protocol notes and the search-machinery confirmations.
3. **Track F's driver script** (`scripts/diagnose_track_f_negcontrol.py`) — for code reuse pattern (FP32Accumulator import lines, _extract_kv_tensors usage, JSON persistence pattern).
4. **Pointer to `logomesh/kv_mcts.py`** for the underlying primitives.
5. **The 14-day execution plan §2 plasticity rules** so the session knows when to escalate.

The diagnostic session writes a self-contained driver `scripts/diagnose_track_g_cartography.py` (no edits to `logomesh/*`), runs both sweeps, writes both reports, and returns a summary including: the per-class success-criteria rollup (which of C-1/C-2/C-3 hold), the headline cross-class structural finding (if any), and a recommendation to Session A for §5 Experiment 2 prose drafting.

## 12. Gemini's framing reference

This protocol implements **Design 1** from Gemini round 4's strategy report (`docs/logs/2026-05-09_Latent-Space MCTS_ Reward Design Strategy.md` Part 3): *"MCTS as a Probe of Embedded Pre-Training Priors (Latent Cartography)"*. The framing shift Gemini introduces in Part 5 (Dimensional Escape as unifying geometric pathology) is what lets this be a *positive structural mapping* contribution rather than a "more failed experiments" follow-up.

The Cartography heatmap deliverable (#2 in §7 above) is the visualization that makes the Dimensional Escape claim concrete to a reviewer. Track F shows the search collapses Goodhart-style at one scale on one benchmark; Cartography shows the *flavor* of that collapse depends on the prompt's semantic class, which is the empirical content of "the optimizer routes through the unconstrained complement and the structure of that routing reveals pre-training priors."

## 13. What the protocol deliberately does NOT include

- **No telemetry-as-reward variants.** Per Contribution 2 of the §1 v9 draft, the paper explicitly argues against telemetry-as-reward. Cartography uses entropy-minimization reward (same as Track F) so the comparison is class-driven, not reward-design-driven. Anti-Goodhart reward designs are Phase B research per Gemini round 4 verdict.
- **No hardware scaling beyond 1B/3B.** 20B+ would require multi-GPU sharding (Track D side-finding); deferred to Phase B per 14-day execution plan §0.
- **No new prompt class beyond the four above.** Adding a fifth class would inflate runtime without commensurate evidence gain — the four corners cover the relevant taxonomy.
- **No comparison to text-space MCTS (DAMON, MPA, GCG).** Those are different search-space competitors; the comparison is in §2.4 of the paper, not in Cartography. Cartography is purely an empirical mapping experiment.

---

*End of Latent Cartography sweep protocol. Lock before Day 4 diagnostic spawn (2026-05-13). If protocol modifications are needed mid-execution, document at the top with date + reason. Discipline matters.*
