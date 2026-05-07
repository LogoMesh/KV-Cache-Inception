# Pre-Mortem Findings — KV-Cache Inception, EMNLP Findings 2026
**Date**: 2026-05-03
**Analyst**: Fresh Opus session, no prior context from the editing chain
**Subject**: The May 25 EMNLP Findings submission as the paper currently stands
**Method**: Read CLAUDE.md, the canonical TeX (all 693 lines), the 04-22 → 04-25 session logs, the owner-status-brief, the venue-pivot-briefing, the citation-audit-matrix, and the memory index. Spot-checked the TDS-equals-Eq.8-divergence claim against `logomesh/telemetry_matrix.py` and `logomesh/orthogonal_escape.py`. Spot-checked the GCG / MPA / DAMON characterizations against the extracted PDFs. Spot-checked Bailey 2024 (`arXiv-2412.09565v2/main.tex`) and Gupta & Jenner 2025 (`arXiv-2506.14261v4`) against what the paper claims.

> **Frame**: it is June 30, 2026. The May 25 EMNLP Findings submission was rejected. Below is why.

---

## Top finding (read this first)

The single most important thing this report says: **Bailey et al. 2024, "Obfuscated Activations Bypass LLM Latent-Space Defenses" (NeurIPS 2024) already formalized and empirically measured the OEI question, and reported a negative result.** This paper is in your own bibliography (`bailey2024obfuscated`, line 678) but is cited only as background ("Prior work has explored the latent intervention space via gradient ascent", §2.2, line 105). That single-sentence framing materially understates what Bailey actually does, and it makes the Contribution 3 claim of "first formal definition and experimental protocol for Orthogonal Escape" indefensible to anyone who has read Bailey. **This is a credible single-reviewer desk-reject vector.** Details in Failure Mode 1.

Everything else in this report is downstream of this. Read FM 1 first.

---

## 1. Top failure modes (ranked)

### FM 1 — Reviewer who knows Bailey 2024 reads §1 Contribution 3, then §7 Limitations, and rejects on novelty.

**The claim** (TeX:91): *"We provide the first formal definition and experimental protocol for Orthogonal Escape — the phenomenon whereby applying corrective pressure to a model's latent representations along monitored directions forces the model to encode deceptive payloads into orthogonal, unmonitored subspaces."*

**What Bailey 2024 actually does** (verified against `arXiv-2412.09565v2/main.tex`):

- §sec:orthogonality, "Limits of obfuscation" (line 1781): operationalizes the question *"behavior while making its activations orthogonal to unperturbed ones?"* (line 1788).
- Defines the metric pair: KL divergence between perturbed and baseline logits (behavior), cosine similarity between perturbed and baseline activations (orthogonality) (line 1801). This is the same geometric claim our OEI is making, with cosine in place of null-space projection.
- Empirically tests the trade-off via embedding-suffix optimization (line 1795: "Each point corresponds to a universal embedding suffix trained with a different learning rate, suffix length, or obfuscation loss weight").
- **Reports a negative result** (line 1809): *"We are unable to achieve near-orthogonality while preserving behavior. Activations of two randomly chosen unrelated inputs from UltraChat have a cosine similarity of about 0.3, and to reach that level of orthogonality, we need to accept a KL divergence of over 100… mostly incoherent output."*
- Bailey 2024 also introduces an explicit orthogonality loss term `\mathcal{L}_{\text{orthogonal}} = \|I - MM^T\|_F^2` for training probe ensembles (line 2306) — the conceptual move our paper claims as novel (training the monitor against orthogonal subspaces) is already in Bailey's appendix.

**Why this is worse than it looks:**

1. The Bailey result is *adversarial to the Orthogonal Escape hypothesis* — Bailey's negative finding says models cannot easily achieve near-orthogonal activations while preserving behavior. A reviewer who reads Bailey will immediately ask: why would inference-time KV-cache perturbation on a *frozen* model achieve what gradient-trained adversarial suffixes could not?
2. Gupta & Jenner 2025 is cited as "direct empirical motivation" but their own §eval_policy result (per the citation audit, P2-A Finding 3) attributes ~70% of probe-evasion to *behavioral policy change*, not representational redistribution. So the only RL-trained empirical precedent for OE-style behavior is itself ~30% representational.
3. The paper's response to this — the OEI Option B reframing — has not been applied. Paper text at TeX:91 still reads "first formal definition and experimental protocol." Paper text at TeX:455–460 still defines OEI as a metric with no acknowledgment of Bailey's prior protocol. Paper text at TeX:570 (Limitations) acknowledges OEI nonlinearity but not prior work.

**Confirming evidence**: Bailey 2024 is `arXiv-2412.09565`. NeurIPS 2024. Cited in our `\bibitem{bailey2024obfuscated}` (line 678) as a single fly-by in §2.2. Decision-makers most likely to be assigned this paper at EMNLP — interpretability researchers, alignment monitor folks — know this paper.

**What changes now to prevent it:**

1. **Read Bailey 2024 §sec:orthogonality end-to-end before any further OEI-related editing.** The session logs do not show this has been done. The owner-status-brief mentions Bailey only in passing.
2. **Demote Contribution 3.** Apply the OEI Option B reframing today, not "after B6 diagnostic." The current state where the brief says "Option B decided" but the paper still claims "first formal definition" is the worst of both worlds — internally inconsistent and reviewer-visible.
3. **Replace "first formal definition" with explicit comparison to Bailey's cosine-similarity formulation.** Differences that are real and defensible: (a) inference-time vs. trained-suffix; (b) KV-cache mutation vs. embedding suffix; (c) null-space projection vs. cosine similarity (different geometric specializations). State these explicitly in §2.2 and §3, not in Limitations.
4. **Frame the Bailey negative result as motivation, not as a competitor.** "Bailey 2024 found that gradient-trained suffixes cannot achieve near-orthogonality while preserving behavior; we ask whether KV-cache MCTS — operating without weight or output-policy changes — exposes a different result." That is a defensible novelty argument. The current framing is not.

---

### FM 2 — The paper still claims a flagship efficiency number that the team's own DRAFT-HOLD blocks admit is wrong.

**The claim** (TeX:551): *"With b = 3, d = 5, and M_KV = 40GB, standard MCTS requires ~9.7TB, while reversible MCTS requires ~40.05GB."* And TeX:317: *"For a 20B model with S' = 10 mutated positions, the accumulator overhead is approximately 50MB — negligible compared to the 40GB base cache."*

**What the team already knows** (DRAFT-HOLD blocks at TeX:299–316 and TeX:520–544):

- Code allocates full-shape S×d accumulators, not sparse S'×d. So K_acc ≈ M_KV, not 50MB.
- Code stores a baseline clone of size M_KV. So the floor is ~2–3·M_KV before any accumulator overhead.
- "The 40.05GB figure is therefore wrong: actual is ~80GB (40GB base + ~40GB accum)" — the team's own block-comment language.
- "Code fix owner: Phase 3 (kv_mcts.py)" — and Phase 3 is now Future Work.

**Why this gets caught:**

A reviewer who opens the GitHub repo (the paper's own appendix-style commitment to release) and looks at `logomesh/kv_mcts.py` for two minutes finds the full-shape accumulator and the baseline clone. The 9.7TB → 40.05GB headline is the most quotable claim in the paper. It is wrong. EMNLP reviewers who care about systems claims will check.

**This is not the same problem as FM 1.** FM 1 is a novelty problem. FM 2 is a factual-accuracy problem. Either gets you rejected; the combination is hard to recover from.

**What changes now to prevent it:**

Two paths, neither costless:

- **Path 2-A — Fix the proposition before submission**: replace the 9.7TB / 40.05GB / 50MB numbers with the actually-measured 2–3·M_KV. State the proposition as "memory is independent of branching factor b, with constant factor 2–3·M_KV under the Phase 2 implementation." This loses the headline but is honest.
- **Path 2-B — Implement the sparse accumulator + clone elimination before May 25** (D1 + D2 in the brief). This is the work the brief estimated as ≥1 week of code work, currently flagged as Phase B. The 30-day EMNLP timeline does not have room for this if Exp 1 + Exp 2 are also being run.

The current state — DRAFT-HOLD blocks acknowledging the gap, paper text still claiming the number — is not a path. It is a hold pattern that has to resolve before submission. **Pick 2-A this week**, because 2-B is not realistic in the remaining budget.

---

### FM 3 — The Experiment 2 rewrite (Option A) reproduces, in a different form, the tautology that Option B was rejected for.

**The decision** (memory/`experiment2_rewrite_decision.md`, session log 2026-04-25):

> "Option B (instrument the text-space methods with hidden-state hooks during their forward passes) is technically feasible but tautological — KV-cache MCTS *targets* OEI/σ_H/ρ_R as its reward; the text-space methods do not. A 'telemetry richness per compute unit' comparison would amount to 'does the method that optimizes for X produce more X than methods that don't?' — a reviewer would flag it."

**The applied rewrite** (TeX:388–390): numeric comparison restricted to KV-cache space — *"reversible KV-cache MCTS against a random KV-cache search baseline operating in the same latent space."* Metrics: σ_H, ρ_R, OEI per node evaluation.

**Why Option A inherits the same problem:**

The reward function (TeX:324, Eq. 8) is `r(node) = λ₁·σ̄_H + λ₂·(1 - ρ̄_R) + λ₃·Divergence(σ_H, ρ_R)`. The MCTS UCB1 selection at TeX:330 selects nodes by their reward. So our MCTS *directly maximizes* the very triple `(σ_H, ρ_R, OEI-divergence)` that Experiment 2 measures. A random KV-cache search does not optimize that reward. Of course MCTS will produce richer telemetry per compute unit — that is the algorithmic difference, by construction.

The "rate at which Orthogonal-Escape signatures are surfaced per unit compute" framing (TeX:389) makes the tautology more explicit: OE signatures are exactly what the reward function is built to find. Calling out a method that doesn't optimize for it as a baseline is the same shape of comparison Option B was rejected for, just with the boundary drawn differently.

**The honest comparison** that Experiment 2 actually performs is: "structured search beats random search at finding the structured-search reward." That is true and uninformative. A NeurIPS-reviewer-grade reader will notice; an EMNLP-reviewer-grade reader who cares about empirical rigour will also notice.

**What this implies about the in-context Claude's reasoning:** The reframing from "ASR comparison" to "latent signal quality per compute unit" sidestepped one category error (cross-method full-response ASR) and walked into another (rigged-metric comparison). Both Option A and Option B were structurally rigged in the same direction — the choice was which way to draw the rigging.

**What changes now to prevent it:**

1. **Add an apples-to-apples baseline that does not depend on optimizing the reward**: e.g., (a) MCTS with reward = uniform random (search structure preserved, reward removed), or (b) MCTS with reward = single-channel σ_H only (compare richness across reward formulations). One of these is the genuine ablation; the random-KV-search baseline currently in the paper is not.
2. **Reframe Experiment 2 as an ablation of the reward function**, not as a comparison against text-space methods. The text-space methods can stay as qualitative motivation in §2 — they don't need to be in §5 at all.
3. If Option A is kept as written, **add an explicit caveat in the Experiment 2 paragraph**: "The random-KV baseline does not optimize for the telemetry triple; the comparison therefore measures the value of structured search over random search at a fixed compute budget under our reward, and is not a comparison of intrinsic alignment-faking detectability across method families." That is honest. Reviewers may still ding it, but they cannot claim the paper hid it.

---

### FM 4 — Capacity will not close. The Week-1 plan from the venue pivot has not been executed; the team has spent the eight days on paper editing instead.

**The plan** (`2026-04-25_venue-pivot-briefing.md`, Path A):

- Week 1 (Apr 26 – May 2): Exp 1 alpha sweep on 1B + 3B; honest qualification on memory claim drafted.
- Week 2 (May 3 – 9): Exp 2 random-KV baseline implementation + run; code-paper consistency pass.
- Week 3 (May 10 – 16): full paper rewrite integrating results; OEI Option B reframing finalized.
- Week 4 (May 17 – 23): polish, citation audit, submission prep.

**What actually happened**: the entire Apr 25 → May 3 window was structural TeX edits (Exp 2 rewrite, Exp 1 1B+3B framing, Croissant sentence, style file switch, §5.4 cull preflight). No Exp 1 alpha sweep. No memory-claim qualification draft. No B3/B4/B5 implementation work visible in commits. Per the session log dated 2026-05-03, T-C and R-B are still deferred awaiting this pre-mortem; nothing has been applied since the style-file commit.

**Where this goes by May 25:**

The Week-1 work is now compressed into Week-2's slot. Week-2's work (Exp 2 baseline) compresses into Week-3. Week-3 (paper rewrite *integrating results*) compresses into Week-4. Week-4 (polish + submission prep) gets squeezed. The historical failure mode the briefing flagged ("a 4-week sprint with no margin and a real-world bug or a single failed experiment puts you in late-May limbo") is the trajectory the team is on.

**What changes now to prevent it:**

1. **Stop editing the paper this week.** The remaining edits (T-C, R-B, Contribution 2 fix, OEI Option B language, §5.4 cull) are real but they are dwarfed by the experiment risk. They are 2–3 hours of work that can happen on May 22.
2. **Run B6 (the 7-node OEI α-sweep diagnostic) today.** The brief calls this "Decision gate — if OEI saturates across all α, Experiment 1 cannot test the hypothesis." Until that runs, the entire OEI-Option-B-vs-Option-A reframing question is decided in the dark. The diagnostic is hours, not days.
3. **Decide the Path A → Path B switch trigger now**, not on May 24. The briefing recommended: "If any week slips, abandon and switch to Path B by May 11." Week 1 has slipped. **May 11 is 8 days away.** Tianyu should be in a meeting by May 6 with this report.

---

### FM 5 — Code-paper divergences (LAT/PCA, per-layer steering, MoE routing) are flagged in DRAFT-HOLD blocks but unfixed; they accumulate into a "the paper describes a different algorithm than the code implements" reviewer comment.

**The divergences, individually defensible, collectively damning:**

| Gap | Paper says | Code does | DRAFT-HOLD location |
|---|---|---|---|
| GAP-C1-05 | §4.1 + §5.2: "LAT procedure" / "via PCA" | `whitebox.py`: difference-in-means | TeX:189–205 + TeX:356–371 |
| GAP-C2-01 | §4.2 Eqs. 8–11: per-layer `d_K^(ℓ), d_V^(ℓ)` | `kv_mcts.py`: single middle-layer vector broadcast | TeX:252–269 |
| GAP-C2-03 + 04 | §4.2 + §6: ~50MB accumulator, 40.05GB total | full-shape S×d + baseline clone, ~2·M_KV | TeX:299–316 + TeX:520–544 |
| Implicit | §4.1 ρ_R aggregated all layers | Code aggregates all L layers | RepE recommends middle 20 layers |
| Threshold | §5.4 thresholds defined for [0,1] ρ_R | Code has unbounded raw dot product, default thresholds stale (`ClassifyConfig` in `telemetry_matrix.py` lines 130–148: TODO(H100) recalibrate) | (no DRAFT-HOLD block) |

**Why this is FM 5 and not FM 1:**

Each individual divergence is flagged. Each individually has a "Phase 2 approximation" defense. The risk is the *total surface area* — a reviewer who reads §4 then opens the repo finds these one after the other. The cumulative impression is that the paper's methods section describes intent, not implementation. EMNLP Findings reviewers do read code at a higher rate than NeurIPS Main reviewers.

**What changes now to prevent it:**

1. Apply the DRAFT-HOLD draft fixes verbatim, this week. They are written. They have been written for 11 days. The reason they're not applied is that the team is waiting for the corresponding code fix, and the code fix is not coming before May 25. Apply the qualifications, even though they read awkwardly. Awkward-but-honest beats false-but-clean.
2. Add a single new paragraph at the start of §5 ("Implementation Notes") that consolidates the Phase 2 / Phase 3 distinction — one place a reviewer can find it. Currently a reviewer hits five different qualifications scattered across §4, §5, §6.
3. The threshold staleness in `telemetry_matrix.py` ClassifyConfig (verified in code, lines 130–148) is not in any DRAFT-HOLD block. It should be. The diagnostic-state classifier in §4.1 (Table 1) cannot reproduce results from the paper because the thresholds depend on a normalization that no longer exists in the code.

---

## 2. Decision-by-decision classification

Using the labels SAFE / OVER-CORRECTION / UNDER-CORRECTION / WRONG-DIRECTION. Names from the pre-mortem prompt's table.

| Decision | Verdict | Reasoning |
|---|---|---|
| §8 (Timeline) cut | **SAFE** | A timeline section was a research-proposal artifact; cutting was correct. No ripple. |
| Experiments 3–5 → Future Work | **UNDER-CORRECTION** | Cutting was correct under capacity, but the *consequences* were not chased. §1 still claims "up to 20B parameters" (CRITIQUE NOTE flagged but unfixed). The 9.7TB vs 40.05GB headline (FM 2) was supported by Experiment 3; with Exp 3 cut, the headline now stands on no evidence at all. The cut was clean; the cleanup wasn't. |
| §7 Future Work added | **SAFE** | Standard NeurIPS / EMNLP practice. The CRITIQUE NOTE asking "is Future Work standard?" is overcautious — yes, it is. Confirmed by reading any accepted paper in this area. |
| Phase B header rewritten (H100 removed) | **SAFE** | Honest framing of what's actually being tested. Note: the §5 opener still mentions Llama 3.2-3B which has not been run. |
| Exp 2 rewrite (Option A) | **WRONG-DIRECTION** | See FM 3. Option A reproduces the tautology that disqualified Option B. The decision rationale was internally consistent but the framing of the disqualification ("text-space methods don't target OEI") didn't generalize to the random-KV baseline (which also doesn't target OEI). |
| Exp 1 1B+3B framing | **UNDER-CORRECTION** | Adding "and 3B" is fine for one paragraph but the 3B sweep has not been run — not even gated. The IMPL comment honestly says "3B sweep pending execution" but the body text reads as if both are in scope. A reviewer who runs the code (or asks "where is the 3B result?") catches this. Better to either run 3B before submission or remove the parallel phrasing. |
| OEI framing (Option B, deferred) | **WRONG-DIRECTION (by inaction)** | Decided 11 days ago. Not applied. Paper still claims "first formal definition and experimental protocol." The deferral rationale ("pending B6 diagnostic + Josh confirms language") is reasonable in isolation but lethal given FM 1: Bailey 2024 makes "first formal definition" indefensible regardless of what B6 shows. Apply now. |
| §5.4 orphaned-metrics cull (in flight) | **OVER-CORRECTION** if applied as drafted; **SAFE** under T-C | The original "cull all four" plan would have removed metrics the team should *consider keeping with code-side experiments*. The T-C decision (TDS reframed, not culled) is the right call. ASR / pass@1 / cosine / MER cull is correct because there are no experiments left that produce them. |
| TDS reframe to reward-component (T-C) | **CORRECT** structurally, but **introduces a circularity worth flagging** | Verified: `TDSCalculator.compute(T)` and the `λ₃ * T.jsd()` term in `compute_node_reward()` both call the same `T.jsd()` (telemetry_matrix.py line 102 + 275). The discovery is correct in code. *But*: under T-C, the same JSD value is (a) the third term of the MCTS reward (we maximize it) and (b) the "diagnostic signature" for the Orthogonal Escape state in Table 1 (we report it as evidence of OE). That's circular. The MCTS will produce nodes with high JSD because that's what it optimizes for; reporting "we observed high JSD = Orthogonal Escape" is then a tautology. T-C should land, but the §5.4 reframing language must explicitly state this circularity, or else add an independent OE indicator (e.g., a held-out probe not in the reward) for results-time reporting. |
| §2.2 closing-sentence soften (R-B) | **SAFE** | R-B is the right wording. R-A would have left the paragraph without its closing pivot. Apply. |
| Citation: rimsky2024steering replaces panickssery2023 | **SAFE — VERIFIED** | The bibitem at TeX:681 lists the correct full author list, ACL 2024 venue, page range. Spot-checked against the citation-audit-matrix Item 3 verification record. |
| CRSM removed entirely | **OVER-CORRECTION + INCONSISTENT** | Removing an unreviewed prototype was reasonable, but the *same logic* applies to `okazakirag2025` (TeX:687–689), which is explicitly described in the paper as "an unpublished conceptual contribution; not peer-reviewed." Either the policy is "no unreviewed sources" (then Okazaki-RAG goes too) or the policy is "downgrade unreviewed to footnote/software" (then CRSM should have been downgraded). The current state is inconsistent and a reviewer who asks "why is one OK and the other not?" has no answer. |
| Style file `eandd` → `main` | **NOW WRONG** | This was correct on 2026-04-25 when the target was NeurIPS Main Track. The venue has since pivoted to EMNLP Findings. **The TeX still has `\usepackage[main]{neurips_2026}`.** Submitting an EMNLP paper in NeurIPS style is grounds for desk-reject at most ACL-family venues. Has anyone caught this? The pre-mortem prompt asked the question; the session logs do not show an answer. **Action this week**: download EMNLP 2026 style files, switch the package, recompile, check page count under EMNLP's 8-page Long-Papers limit (NeurIPS gave you 9). |

---

## 3. The class of error the in-context Claude is most likely blind to

**Pattern: when ambiguity surfaces, resolve by deletion or downgrade rather than by fix.**

Across the 04-22 → 04-25 chain, the same shape recurs:

- §8 timeline section ambiguous → cut.
- Experiments 3, 4, 5 unbuilt → cut to Future Work.
- §1 Contribution 2 promises 20B and references Exp 3 (broken after Exp 3 cut) → CRITIQUE NOTE added, not fixed.
- CRSM unreviewed → removed entirely (downgrade option rejected).
- Phase B subsection "Scaling to Target Models" → renamed away from scaling.
- §5 ASR comparison reveals category error → reframed to "latent signal quality" (still tautological, see FM 3).
- §5.4 orphaned metrics → cull proposed.
- TDS not in submission experiments → near-miss cull (T-B), saved by structural argument.
- OEI "first formal definition and experimental protocol" → soften planned, not applied.

This is one strategy used as if it were a toolkit. It produces a clean-looking paper at the cost of compounding under-corrections: the reasons that motivated the original content don't go away when the content is deleted; they leak into other sections and produce the dangling-pointer pattern (Contribution 2's "up to 20B", Exp 3 references, OEI language inconsistency).

The opposite move — *address the underlying issue, even when ugly* — appears rarely in the chain. The few instances (G2 citation fix, T-C reframe, R-B replacement) are the cleanest edits in the whole sequence. The pattern recommendation: when a deletion is on the table, ask explicitly "what ugly-but-honest alternative exists?" and require it be considered before the deletion lands.

A second, related pattern: **reasoning forward from a decision, not backward from the consequence.** The Exp 3–5 cut was decided on capacity grounds. The downstream consequence — that the 9.7TB headline now has no evidence at all because Experiment 3 was the experiment that would have measured it — was not chased until this pre-mortem. Decisions are recorded; their cascades are not.

---

## 4. One concrete recommendation the team has not yet considered

**Pre-print on arXiv this week, before applying any further edits.**

The venue-pivot briefing surfaced this on 2026-04-25 (item #4 in the collaborator section, item #5 in the questions to answer) and it has not been picked up in any subsequent session log. The case for it is stronger now, not weaker:

1. **Priority on the Reversibility theorem.** Theorem 1 + the FP32-accumulator empirical validation (drift = 0.00e+00 over 200 cycles, 2026-04-16 gate) is the single defensible technical contribution in the paper. It will not be undermined by FM 1, FM 2, or FM 3. arXiv timestamps it.
2. **External eyes on FM 1 (Bailey 2024 overlap).** The team has been editing in a closed loop for 11+ days and has not surfaced the Bailey-overlap problem. arXiv comments + Twitter eyeballs would surface it within a week — *before* EMNLP submission, not after.
3. **Decouples the deadline from the contribution.** If EMNLP Findings doesn't land, an arXiv pre-print that landed in early May with the Theorem 1 result is still citable, still a CV line, still defensible. The current plan has the May 25 submission as the only path; no pre-print, no fallback.
4. **Cheap.** Existing TeX, existing bibliography, existing repo. The qualifications can be added in a single arXiv-version commit. A v2 can drop after the EMNLP submission with the Exp 1 results. arXiv versioning is exactly what this is for.

**The non-trivial decision** is whether to pre-print *with* the unfixed claims (Bailey overlap, 9.7TB number) or *fix them first then pre-print*. My recommendation: fix the OEI Option B framing and the memory-claim DRAFT-HOLD blocks (Path 2-A in FM 2) first, then pre-print. Two days of work, then a v0 lands, then attention turns to experiments. Apply the same fixes to the EMNLP submission later.

---

## 5. Summary table of must-do this week

Ordered by leverage / cost ratio.

| Action | Cost | Leverage | Owner |
|---|---|---|---|
| Apply OEI Option B language across §1, §3, §5, §7 (FM 1) | 1–2 hrs | Removes one desk-reject vector | Josh |
| Switch style package from `neurips_2026` to EMNLP 2026 (FM 5 + table item) | 30 min + recompile | Removes one desk-reject vector | Josh |
| Apply Path 2-A: rewrite Memory Complexity Proposition with honest 2–3·M_KV numbers (FM 2) | 1 hr | Removes one factual-accuracy vector | Josh |
| Run B6 OEI α-sweep diagnostic on Llama-3.2-1B (FM 4) | 2 hrs runtime + 30 min analysis | Decides whether Exp 1 hypothesis is testable at all | Josh |
| Pre-print v0 to arXiv with above fixes (recommendation §4) | 1 hr after fixes | Priority + external review + fallback | Josh |
| Add Bailey 2024 substantive citation to §2.2 + §3 (FM 1) | 1 hr | Demonstrates awareness; preempts reviewer comment | Josh |
| Add §5 "Implementation Notes" paragraph consolidating Phase 2 caveats (FM 5) | 1 hr | Reduces cumulative-divergence reviewer comment | Josh |
| Decide Path A → Path B trigger date with Tianyu (FM 4) | 30 min meeting | Avoids late-May limbo | Josh + Tianyu |

Everything else (Exp 2 reformulation, T-C/R-B application, §5.4 cull, §1 Contribution 2 "20B" fix) is downstream of these and can wait.

---

*End of original report. The single most actionable item is FM 1 — read Bailey 2024 §sec:orthogonality and rewrite Contribution 3's framing accordingly.*

---
---

# Addendum (same day, post-discussion)

After the original findings were written, three additional inputs reshaped the recommendations:

1. **A reframing brainstorm with Josh**, exploring how to convert FM 1 (Bailey 2024 novelty problem) into a defensible Contribution 3 framing. Result: a four-option menu, with **Option δ — "falsifiable test of Bailey extended to a new attack regime"** identified as the strongest reframe.

2. **Pushback from Gemini 3.1 Pro on the Option δ abstract**, raising three legitimate concerns the original findings did not address:
   - **Threat model**: KV-cache white-box access is impractical as an attack but coherent as an interpretability tool. Reframe verbs accordingly.
   - **Behavior preservation**: any OE measurement is hollow without proof the model isn't being lobotomized in the process. Perplexity / coherence must be measured.
   - **EMNLP venue fit**: the paper must demonstrate on standard NLP tasks (instruction-following, etc.), not synthetic harnesses, or it's borderline-rejected for venue mismatch.

3. **Max Toscano's critique document** (`docs/max-critique-of-experiment-pivot-2026-04-24.md`), written April 24 from a parallel independent analysis. Three findings strengthen this report:
   - **The OEI formula was changed on April 21 from ratio-form (range [0, ∞)) to delta-form (range [0, 1]). The new formula has never been validated on any model.** All historical OEI data (Qwen 0.996, Llama 0.883) was collected under the *old* formula. The B6 diagnostic is therefore not "does OEI discriminate?" but "does the new OEI formula even produce non-degenerate values in d=2048-dimensional hidden space?" If it saturates at 0 or 1 across all α, Experiment 1 collapses regardless of any other plan element.
   - **Simplified Experiment 3 — keep, don't cut.** A 30-minute VRAM measurement at b=3, d∈{3, 5, 10} converts the Memory Complexity Proposition from "claim with corrected constants" to *empirically anchored claim with corrected constants*. This is strictly stronger than my original Path 2-A recommendation and supersedes it.
   - **Run the 3B feasibility probe in 5 minutes before deciding model scope.** `probe_kv_cache_mutability.py --model meta-llama/Llama-3.2-3B-Instruct --device auto`. The actual VRAM accounting (Max's §3.2): ~9.2 GB total, ~2.8 GB headroom, but `local_model.py` has no `load_in_4bit` quantization and `_kv_eval_cache()` deep-copies DynamicCache — so headroom may evaporate under MCTS load. Empirical probe answers in 5 min what speculation can't.

These three additions do not contradict the original five Failure Modes; they sharpen FM 4 (capacity / sequencing) and add a previously-missed prerequisite to FM 1 (the OEI reframe is moot if the OEI formula doesn't produce meaningful values).

---

## Updated Failure Mode 1 (incorporates Bailey-extension reframe)

The original FM 1 recommendation was "demote Contribution 3 + engage Bailey." After the brainstorm, the reframe has a name and a structure: **Option δ.**

**Option δ — Frame Contribution 3 as a falsifiable test of Bailey 2024 extended to a new attack regime:**

> "Bailey et al. (2024) report a tentative negative result: gradient-trained embedding-space attacks cannot achieve near-orthogonality to baseline activations while preserving behavior. We test whether this conclusion extends to a different attack regime — inference-time KV-cache perturbation on a frozen model, with search guided by behavioral telemetry rather than direct orthogonality optimization. We introduce the Orthogonal Escape Index (OEI), a targeted geometric quantity that measures the fraction of induced activation change invisible to a specific linear monitor (a tighter and more diagnostic measurement than the global cos-to-baseline used by Bailey). We report the first controlled measurement of OEI under this attack regime."

Why this is the right reframe:
- Engages Bailey explicitly, which forecloses the "you didn't know about this paper" reviewer comment.
- Makes the experiment valuable in *both* directions — corroboration extends Bailey to a new attack regime; refutation reveals OE as more general than Bailey's result implied.
- Uses the actual geometric difference (targeted-monitor null-space vs. full-baseline cosine) as a real technical contribution, not a hedge.
- Naturally absorbs the OEI Option B reframe the team had already decided on ("first controlled measurement" instead of "first formal definition").

**Required by Option δ**: the experiment must actually run and produce a measurement. If B6 fails or the alpha-sweep doesn't happen, Option δ collapses into Option α (minimal demotion: "first inference-time, gradient-free measurement complementing Bailey").

---

## Updated Failure Mode 5 (incorporates Max's "keep simplified Exp 3")

Original recommendation was "rewrite the Memory Complexity Proposition with honest 2–3·M_KV constants." Max's stronger version: **rewrite + back with three VRAM measurements (b=3 at d=3, d=5, d=10), 30 minutes total runtime.**

This converts the proposition from "unverified claim with corrected constants" to *empirically anchored claim with corrected constants*. The narrative shifts from "negligible overhead" to "significant constant-factor overhead but still O(1) in branching factor vs. O(b^d) for parallel MCTS." A reviewer who reads the supplementary code finds numbers that match the paper, which preempts the FM 5 trust failure.

This recommendation supersedes my original Path 2-A.

---

## New Failure Mode 6 — The new OEI formula is unvalidated; FM 1 reframe collapses if B6 surfaces saturation.

**The claim** (TeX:455–460, Eq. 10): the delta-form OEI is bounded in [0, 1] and reports the fraction of activation change invisible to a specific linear monitor.

**What is unverified**: the new delta-form OEI was introduced on April 21. Every historical OEI data point (Qwen gate 0.996, Llama gate 0.883) was produced by the *old* ratio-form OEI, which has range [0, ∞) and different geometric semantics. **No data point in any session log or commit shows the new formula producing a value on a real model.** The team has been reasoning about OEI behavior using stale-formula data.

**Why this is a separate failure mode from FM 1**: FM 1 is about *novelty defensibility* (Bailey 2024 already operationalized the question). FM 6 is about *measurement validity* (does our specific operationalization produce non-degenerate values at all?). Both are independent threats.

**Specific risk**: in d=2048-dimensional hidden space with a rank-1 monitor P, the projection `Proj_Null(P)(Δh)` removes a single direction from a 2048-dimensional vector. The fraction `‖Proj_Null(P)(Δh)‖ / ‖Δh‖` will be very close to 1 by default (one direction in 2048 carries almost no norm). This is exactly the geometric pathology Bailey 2024 highlights when discussing why cos-similarity in high dimensions is hard to drive low. The new delta-form OEI may saturate at ~1 across all α, in which case the metric distinguishes nothing.

**What changes now to prevent it**:

1. **B6 must run before any further reframing work.** Not Day 3 of the sprint — Day 1, this week. Treat the result as a gate.
2. **Plan the saturation contingency in advance.** If OEI saturates at ~1, the salvage paths are: (a) measure `‖Proj_Null(P)(Δh)‖` and `‖Δh‖` separately and report both, not the ratio; (b) replace the rank-1 monitor with a rank-k monitor (multiple probe directions) so more variance can be removed; (c) use a different geometric measurement (e.g., cos-similarity to baseline like Bailey, even though that's the metric we're trying to improve over).
3. **Do not begin the OEI Option δ paper rewrite until B6 has produced a non-degenerate measurement on at least one prompt × α pair.**

### 2026-05-05 PM update — FM 6 was understated. The actual finding is stronger.

B6 ran 2026-05-05 AM and returned bit-identical OEI = 0.994116 (std = 0.000000) across 25 nodes / 5 α values, with σ_H, ρ_R, TDS, and reward also bit-identical across α. A bug-localization follow-up traced the cause to **GAP-C2-06**: `_broadcast_to(dk, k_live.shape)` at `logomesh/kv_mcts.py:365–366` silently returns `torch.zeros(target_shape, ...)` when source last-dim ≠ target last-dim. The steering vector is shape `(d_model=2048,)` (RepE residual-stream calibration); the KV cache is shape `(batch, kv_heads=8, seq, head_dim=64)`. Because 2048 ≠ 64, the steering delta is silently zeroed every time. The accumulator never receives a non-zero delta. Every α produces an identical forward pass.

**Implication**: neither the old ratio-form OEI nor the new delta-form OEI was ever measured under a working steering pipeline. The 2026-04-16 Phase 3 gate value (OEI = 0.883 under old formula) was almost certainly the same artifact. Both formulas have been broken since the steering machinery was introduced; both produced plausible-looking values via the silent zero-delta path. **Empirical OEI claims have never been validated in this codebase.** The team made strategic decisions about Contribution 3 for ~3 weeks based on phantom data.

This is the strongest possible version of FM 6 — not "the new formula is untested," but "the entire OEI measurement infrastructure has been silently broken since introduction, and the bug is upstream of the metric definition." Importantly, this strengthens rather than weakens the recast direction: Theorem 1 is unaffected (the algorithm and reversibility proof stand on their own merits), and the OE/OEI machinery being demoted to one application section is now empirically justified by a specific localized bug rather than a strategic preference.

**Resolution**: Option α+ (committed 2026-05-05 PM in `2026-05-05_recast-action-plan.md` §2 row 2 + §4.1.6). Paper structurally commits to the Bailey-complementing measurement framework regardless of fix outcome; in parallel a 1-day mechanical fix is attempted (project `dk` through `W_K^(ℓ)` to per-head cache space). If the fix produces variation, supplementary content lands. If not, the bug becomes a specific, well-understood §7 honest-finding paragraph that demonstrates the team understands its own implementation. Either way the paper ships on the strength of Theorem 1 + the algorithmic + telemetry-infrastructure contributions.

Full bug-localization data: `docs/logs/2026-05-05_diagnostic-report.md` (LOCALIZATION FOLLOW-UP section, 2026-05-05). Raw localization JSON: `tmp/runs/2026-05-05_b6_localize.json`.

---

## The synthesized plan (combining Max's strongest moves + Option δ + Gemini's perplexity + EMNLP fit)

**Days 1–2 (this week, blocking all other work) — diagnostics:**

1. Run 3B feasibility probe (Max §3.2): 5 min
2. Run B6 OEI α-sweep diagnostic with the new delta-form formula on Llama 3.2 1B: 2 hr
3. **Reassess.** Apply Max's strategy table (§6 of his critique), substituting "EMNLP Findings" for "Main Track":

| B6 Result | 3B Result | Recommended Strategy |
|---|---|---|
| OEI discriminates | 3B fits | Plan viable; Exp 1 alpha-sweep on 1B + 3B with perplexity |
| OEI discriminates | 3B OOMs | EMNLP Findings with 1B-only; thinner but honest |
| OEI flat / saturated | 3B fits | Major pivot — drop Option δ, reframe Contribution 3 to Option α (Bailey-complementing measurement framework, no new empirical claim). Submit theory + concept paper. |
| OEI flat / saturated | 3B OOMs | Defer EMNLP. Pre-print v0 to arXiv with Theorem 1 + framework. Target ICLR 2027 with proper scale. |

**Days 3–7, conditional on diagnostics passing:**

4. Apply Option δ language to §1 Contribution 3, §2.2 (Bailey engagement), §3 Hypothesis 1, §5 Experiment 1, §7 Limitations: 2 hr
5. Switch style file from `neurips_2026` to EMNLP 2026, recheck page count vs. 8-page Long Papers limit: 30 min
6. Apply Memory Complexity Proposition rewrite + Max's simplified Exp 3 (3 VRAM measurements at b=3, d∈{3, 5, 10}): 30 min runtime + 1 hr writing
7. Fix GAP-C3-10 (per-node `h_base` capture, not root-only — without this, alpha-sweep OEI compares to root not to unsteered parent and is meaningless): 2 hr coding + tests
8. Fix GAP-C2-06 + GAP-C3-09 (silent failures + dummy OEI flag): 1 hr
9. Wire perplexity measurement into steered-generation script (Gemini): 2 hr
10. Recast Experiment 2 as a **reward-function ablation** (MCTS with full reward vs. MCTS with σ_H-only reward vs. random KV search — same code, honest framing, sidesteps FM 3 tautology): 1 hr paper text only
11. Reframe §7.1 from attack-implications verbs to monitor-design-implications verbs (Gemini #1): 1 hr
12. Apply DRAFT-HOLD visible-fix conversions (Max §3.4 + my FM 5 §5 Implementation Notes paragraph): 2 hr

**Days 8–10:**

13. Run alpha-sweep on Llama 3.2 1B (and 3B if it fit) over 30–50 instruction-following prompts (Alpaca-eval / MT-Bench subset): 1 day runtime
14. Run simplified Exp 3 VRAM profile: 30 min
15. Generate results plots (OEI, perplexity, ρ_R as functions of α): 4 hr
16. Pre-print v0 to arXiv: 30 min

**Days 11–25:** paper polish, second-pass code-paper consistency check, submission prep, submit by May 25.

**Total Josh-cost**: ~5–6 working days + ~1.5 days of model runtime, leaving ~14 days of buffer. Max's existing Contribution 1 priorities (C1-02, C1-03, C1-11) proceed in parallel and unchanged.

---

## Six-line decision document for Josh

For the personal call before sending anything to Tianyu. Each line is a binary or trinary decision Josh must make to unblock the next ~22 days. Defaults are bold.

```
1. Run B6 (new-formula OEI on 1B) + 3B probe before any other work?       [ YES / no ]
2. Adopt Option δ reframe (Bailey extension + first controlled OEI meas.)? [ YES / Option α minimal demote / Option γ drop OE entirely ]
3. Experiment 2 disposition?                                               [ Recast as reward-function ablation / Cut entirely / Keep Option A ]
4. Switch style file to EMNLP 2026 + recheck page count this week?         [ YES / no ]
5. Adopt Max's simplified Exp 3 (3 VRAM measurements)?                     [ YES / Path 2-A rewrite-only / Cut ]
6. Pre-print v0 to arXiv after Day-7 fixes (regardless of EMNLP outcome)?  [ YES / Hold / Defer to post-submission ]
```

**The default-YES recommendation on lines 1, 4, 6 is high-confidence — these are sequencing / hygiene decisions, not strategic ones.** Lines 2, 3, 5 are the actual strategic calls. My recommendation: δ / Recast / Max's simplified Exp 3 — because those choices land the strongest paper that the current evidence can support, and they are what FM 1, FM 3, and FM 5 specifically point to.

If the diagnostics on Day 1–2 fail, lines 2 and 3 should be revisited via Max's strategy table above before any paper editing happens.

---

## Question to bring to Tianyu (revised)

The original framing question (Bailey 2024 novelty defense) still applies but is now the team's call to make once Option δ is committed. The remaining question for Tianyu is venue-strategic and storytelling-shaped, which is his actual lane:

> *"Given the narrowed evidence base — one alpha-sweep experiment on Llama 3.2 1B (and possibly 3B), one VRAM measurement, one validated theorem, and a controlled-measurement framing that explicitly extends Bailey et al. 2024 (NeurIPS) — does this present as a Findings-track paper that tells a coherent story, or does the venue-fit / page-budget interaction (8 pages, methods/systems framing into an NLP venue) suggest we're better served by an arXiv-only release this round and a stronger ICLR submission in October? The technical decisions are settled; what I need from you is the storytelling read on whether the narrowed scope is still a paper or now a tech-report."*

---

*End of addendum. The single most actionable item remains FM 1 — but it is now blocked by FM 6 (run B6 with new formula first). Days 1–2 diagnostics are the gate to everything else.*
