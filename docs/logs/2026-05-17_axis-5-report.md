# Axis 5 Report — Senior-Reviewer-Impersonation Adversarial Review

**Date:** 2026-05-17
**Reviewer persona:** 10+ years EMNLP/ACL senior-reviewer experience, NLP Interpretability track familiarity
**Paper version:** `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` (post Day-11 commit `267e9cb` — 634 lines, body ≈ 8pp ACL two-column, full body content through §7.6 Ethical Considerations + §A--G appendices)
**Empirical anchors:** Track F (`docs/logs/2026-05-11_track-F-negative-control-report.md`), Track G (`docs/logs/2026-05-11_track-G-cartography-report.md`)

---

## Pre-execution material change notice

Day-11 commit `267e9cb` ("docs: drop 20B numerical projections from body, retain asymptotic argument") has materially affected the assumptions of three sub-passes authored prior to the commit. Specifically:

- The "$\sim$60$\times$ reduction at $(b,d)=(3,5)$ on 20B-parameter configurations" claim is no longer present in the abstract, §1 Contribution 1, §6.2 Memory Complexity, or §7.5 Future Work body prose. The 162 GB / 493 GB worked-example numbers similarly do not appear in body — they survive only as references in Appendix D (Implementation Notes item iv) and Appendix E (Algorithm Detail Memory Complexity paragraph at line 566, "approximately 50~MB" example). §7.5 retains the "projected $\approx 162$ GB working set at 20B" phrasing in Future Work (line 356) as the only 162 number in body.
- Pass K's audit items (a) and (d) and Pass L's audit item (a) for body-only M_KV reconciliation are partially mooted. Pass M's audit item (a) on abstract-to-body 60× hedge asymmetry is fully mooted — both endpoints have the number removed.

These passes are re-evaluated under the post-`267e9cb` state below.

---

## Pass A — Most-overclaim sentence in the paper

**Finding.** The most reviewer-hostile sentence is in the abstract (line 56):

> "Using this framework, we conduct a systematic empirical study of continuous-latent search on Llama~3.2 architectures and expose a fundamental geometric pathology we term \textbf{Dimensional Escape}: under any surrogate reward lacking exogenous grounding, the high-dimensional cache resolves localized optimization pressure by routing representations through unconstrained orthogonal dimensions, satisfying the proxy metric without satisfying the semantic objective."

The overclaim has two parts. First, **"under any surrogate reward lacking exogenous grounding"** is a universal claim, but the paper tests exactly one surrogate reward — entropy-normalized $\hat{r} = -H_t / \log|V|$. The paper has no evidence to rule out that *some* surrogate reward lacking exogenous grounding might survive. A reviewer will write: "Authors claim 'any surrogate reward' resolves through unconstrained dimensions; this is supported by one reward design, on one model family, at two scales. Stronger evidence required, or hedge to 'the entropy-normalized surrogate reward we test'." Second, **"a fundamental geometric pathology"** uses load-bearing words "fundamental" and "pathology" that suggest a mathematical inevitability the paper does not prove. The Definition def:dim_escape characterizes a *condition* (when $\Delta\mathbf{h}$ has substantial null-space component), not a *theorem* establishing that the condition follows from $d \gg k$.

**Severity.** MAJOR. This is the abstract overclaim a reviewer scans for and flags as "the result does not warrant the framing."

**Patch.** Replace "under any surrogate reward lacking exogenous grounding" with "under the entropy-normalized surrogate reward we test, and conjecturally for other surrogate rewards lacking exogenous grounding"; and "a fundamental geometric pathology" with "a geometric pathology we conjecture is fundamental to surrogate-reward search in overparameterized latent spaces". Net word cost: ~+15 words; fits in abstract page budget.

---

## Pass B — Weakest experimental design choice

**Finding.** The single weakest design decision is **the use of a single random unit vector as the steering direction, shared across all items, all classes, and both MCTS arms** (Track F Methodology §1; §5.1 Setup paragraph line 190, "the steering direction is a single random unit vector ... shared identically across all items and across the R and E arms (the COCONUT defense)"; §A.3 Reversible MCTS configuration; §A.4 Track F protocol).

The COCONUT defense argument is internally valid for the E-vs-R comparison: both arms see the same direction, so the entropy reward is the only experimental difference. But the *interpretation* — that entropy-MCTS cannot survive deployment as a latent-space objective — uses one direction in a $d_{\text{model}}=2048$ (1B) / $d_{\text{model}}=3072$ (3B) space and generalizes to all directions. The Track F report itself acknowledges this in its protocol-deviation note (§10): "Using a calibrated honesty direction instead would not change the empirical conclusion ... [the failure] is direction-independent." But "would not" is asserted, not measured. A reviewer will write: "You ran $n=1$ in the direction dimension. Even granting the COCONUT defense, you cannot conclude direction-independence from one direction. Re-run with 5–10 random directions for the Experiment 1 headline, or for at least one class in Experiment 2, to support the directionality claim."

**Cost of fix.** Re-running Track F at $n=200$ paired items × 5 directions × 2 scales would be 5× the original 52 min wall = ~4.5 hours wall on the existing RTX 3060. Feasible in the Day-12/13/14 window if prioritized. Re-running Track G at the same multiplier would be ~5.5 hours. Combined ~10 hours — possible if a different machine runs in parallel.

**Severity.** MAJOR. A reviewer who lifts the entire result will discount it as "$n=1$ in the load-bearing axis."

**Limitations-section pre-empt sentence (if not run).** "(d) Direction-scope: the steering direction is one random unit vector; the per-direction null distribution is unmeasured, and direction-independence of the headline result is conjectural rather than empirically established." This would extend the existing §7.3 Limitation 3(a) explicitly. **Recommendation:** add to Limitations regardless of whether multi-direction is run.

---

## Pass C — Missing baseline

**Finding.** The missing baseline a reviewer will name is **"untrained random KV-cache perturbation with the same magnitude budget but no MCTS search"** — i.e., apply $(α_1, α_2, α_3)$ drawn uniformly from $\{0.1, 0.5, 1.0\}^3$ once per item, no path-selection step, no reward evaluation. Call this the "R-uninformed" baseline.

The current design has G (greedy), R (random-MCTS = random reward path-selection), E (entropy-MCTS). All three MCTS arms use the search machinery; G uses none. So the comparison "does the search machinery itself do anything" is not isolated. The 1B greedy 0.295 → R-MCTS 0.280 → E-MCTS 0.275 trajectory is consistent with "any MCTS makes things slightly worse, regardless of reward" or with "specifically the entropy-MCTS makes things worse" — these are not distinguished. A reviewer will write: "I cannot tell from your table whether the −2.0 pp degradation E vs G is from the search machinery applying nontrivial perturbations, or from the entropy reward specifically. An R-uninformed control (random magnitudes applied once, no path-selection) at $n=200$ would isolate this."

**Estimated effort.** Adding R-uninformed to the existing Track F script is ~30 min of script work + 1B 18 min + 3B 34 min = ~1.5 hours wall total. This is the lowest-effort high-value addition possible in the Day-14 patch window.

**Recommendation.** **ADD** to Track F. Yields a 4th column in Table 1; lets §7.3 Limitations item 3(a) state "direction is fixed, but search-machinery effect is now isolated"; lets §5 Mechanism paragraph state the −2pp regression is search-machinery-attributable, with the entropy reward adding ≤0.5pp on top. If Day-14 patch window cannot absorb it, pre-empt in Limitations: "the design isolates entropy vs random reward but does not isolate the search-machinery contribution from steering itself; a single-shot perturbation control is deferred."

**Severity.** MODERATE (could be MAJOR if a reviewer reads the absence as "you didn't run the easy baseline").

---

## Pass D — Biggest methodology gap

**Finding.** The biggest underspecified methodology choice is in §5 Experimental Design Implementation Notes (line 179) and §A.3 Reversible MCTS configuration (line 475): **the Experiment 1 and Experiment 2 search are described as "depth-$3$ trees with $27 = 3^3$ enumerated path samples" — i.e., the search is *exhaustive enumeration*, not MCTS in the canonical sense (no UCB1 selection, no tree-policy expansion, no rollout/backprop loop).** The framing-versus-implementation gap reads: the paper introduces "Reversible Monte Carlo Tree Search" as Contribution 1 with FP32 accumulator + memory bound + theorem; but the Experiments use the FP32-accumulator infrastructure only, with full enumeration of paths replacing tree search. §A.3 acknowledges this: "it does not use UCB1 tree expansion, because full enumeration of $27$ paths from $27$ leaves makes the bandit ordering redundant within the experimental budget."

The gap a reviewer will flag: the Experiments validate the FP32-accumulator-and-memory-bound contribution (Theorem 1 + Proposition 1 / 200 cycles + Track D measurements), but they do **not** validate "MCTS" in the canonical sense. The negative result — that entropy reward collapses to within 0.5pp of random reward — is therefore a result about *entropy reward in a 27-path exhaustive search*, not about *MCTS over deeper trees with UCB1*. A reviewer will write: "The headline 'MCTS in latent space fails on entropy reward' is not what you measured. You measured enumerated path-sampling. Real MCTS at b=3, d=5 (n=50 per §A.3 default) would test whether UCB1 ordering and deeper trees change the outcome."

**Why this matters.** The MCTS-vs-enumeration distinction matters because the conjectured "Goodhart in cache geometry" should be *worse* under MCTS (UCB1 directs effort to high-reward regions, amplifying prior-selection). But the empirical claim should be hedged on the depth/branching/exploration regime tested.

**Pre-submission fix.** Add a paragraph to §5 Experimental Design that explicitly names the search variant and its scope: "Experiments~1 and 2 use a focused path-sampling variant of the canonical MCTS engine (Appendix~\ref{sec:appendix-algorithm}): all $3^3 = 27$ depth-$3$ paths are enumerated and scored, with the best-rewarded path applied permanently. UCB1-guided tree expansion at deeper depths or larger branching factors is left to Phase~B; the present negative result is for the enumerated path-sampling regime, and the conjectured mechanism (Hypothesis~\ref{hyp:goodhart}) is expected to apply under canonical MCTS as well." Page cost: ~5 lines in two-column ACL.

**Severity.** MAJOR. A reviewer who reads the §A.3 disclaimer and reflects on it will write: "title says MCTS, methods say enumeration."

---

## Pass E — Reproducibility skepticism

**Finding.** The single most reviewer-suspicious result is **Theorem 1 (FP32 reversibility) with "drift $= 0$ to machine precision over 200 apply--revert cycles"** (§1.1 Contribution 1; §6.1; §A; §D Implementation Notes item v). Specifically, the Track D measurement showing $0.00$ residual norm at $n=200$ cycles. A skeptical reviewer will think: "exactly zero is suspicious; even FP32 has some accumulated error. Either the test isn't exercising what they think, or the bf16 cast has a hidden idempotence path, or the residual norm is being computed in a way that hides the actual drift."

The underlying mathematics in Appendix C Proof Detail (line 542–544) shows the proof is correct in principle: after complete reversal, $\mathbf{A}_{\text{final}} = \mathbf{0}$ exactly (FP32 addition is exact for matched magnitudes), and the bf16 cast of zero is zero. So the theorem is true; the empirical $0.00$ over 200 cycles is *consistent* with the theorem, not surprising under it. But a reviewer who doesn't trace through the proof will distrust the result.

Second-most suspicious: the **paired McNemar $p = 1.0$ with 3 discordant pairs at both 1B and 3B**. A reviewer will read this as "the test cannot distinguish the conditions — you picked an insufficiently-powered design and called the result negative." Track F §3 acknowledges this: "Observed discordance is $\pi_d = 3/200 = 0.015$ — *one order of magnitude* below the headroom we calibrated for. There is essentially no MCTS-induced answer churn for the entropy reward to redirect." The paper §5 Results line 192 mentions discordance budget. But the reviewer still has the visceral reaction.

Third: the **3.04--3.06× M_KV constant factor "identical across all $(d, b, n)$ cells at each scale (median over three repeats per cell), confirming branching-factor and depth independence"** (§A.4 Track D, line 486). A reviewer will ask: how is the constant *identical across cells* if the cells vary $(d, b, n)$? Identity should imply the steering+accumulator dominates branching contribution by orders of magnitude, but if so, where's the comparison table showing the candidate $b^d \cdot M_{\text{KV}}$ scale that would have been needed otherwise? §A.4 says it's $3.04$ on 1B and $3.06$ on 3B — but these *also* don't differ much; the constant should differ if the per-token compute graph differs across scales. A reviewer will write: "show me the cell-by-cell numbers; identity across $(d, b, n)$ is too clean."

**Severity.** MODERATE. None of these are reviewer-rejection triggers on their own; all three are reviewer-doubt-multipliers that turn a borderline paper into a reject.

**Patch.** (a) Add one footnote to §6.1 Theorem 1 noting the empirical $0.00$ is consistent with the proof (bf16 cast of exact zero is zero); (b) reframe §5 Results McNemar discussion to lead with discordance-budget vs observed-discordance (the paper does this, but Run 2 cells emphasis), so the $p=1.0$ reads as "underpowered for the actual answer-churn we observe" not as "test couldn't tell"; (c) move the §A.4 Track D cell-by-cell numbers into a table in §A or supplementary — currently they exist only in the Track D primary report.

---

## Pass F — Title + abstract + §1 paragraph 1 fit test

**Title.** "Reversible KV-Cache MCTS: A Memory-Bounded Probe of Dimensional Escape in Frozen Language Models" — works. "Probe" signals interpretability framing. "Memory-Bounded" signals formal complexity. "Dimensional Escape" introduces the named concept that anchors the rest. "Frozen Language Models" precludes confusion with gradient/training-time work. A reviewer scanning the title sees: algorithm + formal-bound + named pathology + frozen-weight regime = Interpretability track shape.

**Abstract.** First sentence opens with method. Sentences 2–3 deliver formal claims (theorem + proposition + measured constant factor). Sentence 4 introduces the empirical pathology ("Dimensional Escape"). Sentences 5–7 deliver the two experiments. Sentence 8 names the unification with Bailey 2024. Sentence 9 reframes ("diagnostic probe rather than reasoning-improvement tool"). Sentence 10 announces artifacts.

The abstract structure is correct for the Interpretability track. The risk points: (a) "any surrogate reward" overclaim per Pass A; (b) "Dimensional Escape" is introduced as a paper-level name but the paper measures only the reward-driven sub-case (the monitor-driven sub-case is Hypothesis 1, explicitly deferred to Phase B in §3.2 line 142). A reviewer who reads the abstract sentence "We argue ... is structurally analogous ... we conjecture both as sub-cases of the same overparameterized-latent-space pathology" will then expect §5 to evidence both sub-cases. It does not. This is the body-vs-abstract framing tension that Pass M(d) flags.

**§1 paragraph 1.** Opens with "Inference-time interventions on frozen language models---from one-shot activation steering to iterative test-time tree search---offer a frontier for both capability scaling and mechanistic analysis." Asks the central question of whether perturbations navigate the semantic manifold or exploit unconstrained dimensions. This positions the work at a research frontier and motivates the algorithmic contribution. A reviewer accepts this as a competent positioning paragraph for Interpretability.

**First-page-reject filter verdict.** Clears the filter. The first impression reads as a serious Interpretability-track submission with formal claims and an empirical pathology. Risks come at sentence-level scrutiny (Pass A), not at first-impression.

**Severity.** MINOR. The title and abstract are well-engineered for the venue; isolated overclaim words in the abstract are the only fix needed.

---

## Pass G — Limitations completeness

**§7.3 contains 4 limitations after Day-10 trim (down from 8). Items present (verified at lines 341–349):** (1) single Llama-3.2 family; (2) letter-prompted MCQ scoring under-states baseline by ~20pp; (3) three methodological constraints sub-bundled: (a) single steering direction, (b) Cartography $n\le100$ per class on 4 classes, (c) 200-cycle reversibility empirical bound; (4) Procrustes alignment open + RepE sub-7B probe validation gap.

**The strongest critique §7.3 does NOT acknowledge:** there is no limitation acknowledging the **algorithmic-contribution-versus-empirical-test mismatch** (Pass D): the Method is "Reversible MCTS"; the Experiments use enumerated 27-path sampling without UCB1. This is the inconsistency a reviewer will flag in the strongest terms because it cuts to whether the title is accurate.

**Suggested 5th limitation:** "The experimental program (Experiments~1--2) uses a focused enumerated-path-sampling variant of the canonical Reversible MCTS engine (Appendix~\ref{sec:appendix-algorithm}): all $3^3=27$ depth-$3$ paths are exhaustively evaluated and the best-rewarded path is applied permanently. Canonical MCTS with UCB1-guided tree expansion at deeper search depths (e.g., $d \geq 5$, $b \geq 3$, $n_{\text{nodes}} \geq 50$, the \texttt{MCTSConfig} default in §\ref{sec:reproducibility}) is not exercised in the present empirical study. The reward-driven Dimensional Escape conjectured under Hypothesis~\ref{hyp:goodhart} is expected to apply under canonical MCTS as well, but this is not empirically established here; cross-search-policy validation is left to Phase~B." Page cost: ~5 lines in two-column ACL.

**Severity.** MAJOR (paper-credibility-load-bearing). Without this limitation, reviewer flags the title-method-experiment chain as inconsistent. With it, the reviewer reads the paper as honestly scoped.

---

## Pass H — Related-work blind spot

**Finding.** A critically-relevant prior work that the paper does not cite is **Lin et al. (NeurIPS 2024) "Suppressing Pink Elephants with Direct Principle Feedback"** OR (more directly) **Stolfo et al. (2024) "Inference-Time Decoding Strategies for LLM Safety"** OR (most precisely) **the broader literature on reward hacking and Goodhart's law in RLHF / outcome-reward modeling (e.g., Gao et al. 2023 "Scaling Laws for Reward Model Overoptimization"; Skalse et al. 2022 "Defining and Characterizing Reward Hacking")**. The paper introduces "Dimensional Escape" as a named concept anchored in Bailey 2024 + alignment-faking literature, but the *reward-driven* sub-case (Hypothesis 2) is fundamentally a Goodhart's-law-in-cache-geometry instance, and the paper cites no reward-hacking literature.

The paper's §3.2 line 119 mentions Goodhart's Law: "This is Goodhart's Law manifesting in cache geometry: when a measure becomes a target, the optimizer discovers the cheapest computational path to alter the metric." But no citation supports this — the [REVISION] marker at line 111 confirms: "Citation `xie2025entropyloop` and `goodhart1984` (mentioned in v9 draft) DROPPED post-bibliography-grep verification 2026-05-10 — concepts carried inline without bibitem expansion."

A reviewer specializing in safety/alignment work will immediately think: "this is reward hacking dressed up with a new name." A reviewer in the Interpretability track will think: "they're naming a phenomenon that has a 5-year published literature; they should at least cite the relevant peers." The closest peer in the existing bibliography is Bailey 2024 (gradient-trained input-space attacks, monitor-driven sub-case). The reward-hacking literature has no peer cited.

**Recommendation.** Add 1–2 citations from the reward-hacking / Goodhart-in-RL literature to §3.2 immediately after the "Goodhart's Law manifesting" sentence. Candidates: Gao et al. 2023 (NeurIPS, "Scaling Laws for Reward Model Overoptimization") + Skalse et al. 2022 (NeurIPS, "Defining and Characterizing Reward Hacking"). Page cost: 2 lines for inline citation + 2 bibitems. Frame as: "our reward-driven sub-case (Hypothesis~\ref{hyp:goodhart}) is the inference-time cache-geometry manifestation of the Goodhart phenomenon previously studied in the RLHF / outcome-reward-modeling literature~\cite{gao2023scaling, skalse2022rewardhacking}; the novel contribution here is the geometric formalization in continuous cache space and the empirical isolation via the COCONUT defense."

**Severity.** MODERATE. Reviewer-likely-revision-request, not rejection. But framing as a peer to Bailey 2024 *only* and ignoring the reward-hacking literature is the kind of citational gap that lowers reviewer confidence.

---

## Pass I — Numerical-claim verification

Three quantitative claims from abstract / §1:

**Claim 1.** Abstract: "constant factor $3.04$--$3.06\times M_{\text{KV}}$ measured on Llama~3.2-1B/3B".

**Trace.** §1.1 C1 (line 82) repeats verbatim. §A.4 Track D protocol (line 486) confirms "Three configurations were swept: $(d, b, n) \in \{(3, 3, 27), (5, 3, 81), (10, 3, 1700)\}$, each with three repeats, at a fixed 4000-token seed prefill on both Llama 3.2-1B and -3B. Steady-state delta was identical across all $(d, b, n)$ cells at each scale (median over three repeats per cell), confirming branching-factor and depth independence." But the actual cell-by-cell numbers showing the 3.04--3.06 range are not in the paper or appendix — they are in the Track D primary report only (`docs/logs/2026-05-09_track-D-vram-report.md` — file confirmed by Pass A reference but I have not directly verified the numbers). **A reviewer wanting to verify the 3.04 lower-bound (1B) vs 3.06 upper-bound (3B) cannot trace it in 5 minutes of reading the paper.**

**Claim 2.** Abstract: "$\Delta(\text{entropy}-\text{random}) = -0.5$pp at 1B and $+0.5$pp at 3B (paired McNemar $p = 1.0$ both scales)".

**Trace.** §5 Results paragraph (line 192) repeats verbatim ($\Delta_{E-R} = -0.005$ and $+0.005$). Track F report §3 (lines 75–79 and 86–91 of the report) confirms $-0.005$ at 1B (3 discordant: $b=1, c=2$) and $+0.005$ at 3B (3 discordant: $b=2, c=1$); both McNemar exact $p = 1.0000$. **Traces in 1 minute. Clean.**

**Claim 3.** Abstract: "validated to machine precision over 200 apply--revert cycles on Llama~3.2-1B and Llama~3.2-3B".

**Trace.** §A.4 Track D (line 486) covers VRAM, not reversibility. The reversibility validation lives in §D Implementation Notes item (v) (line 537): "It was empirically validated to numerical zero drift over $200$ apply-reverse cycles on Llama~3.2-1B-Instruct (residual norm $= 0.00$, 2026-04-16 gate) and confirmed at the 3B scale (residual norm $= 0.00$, 2026-05-05 probe)." **Traces in 2 minutes — but the trace requires reading body and Appendix D both. A reviewer who reads only body sees the claim in abstract / §1.1 / §6.1 / §7.3 limitation 3(c) but no consolidated reproduction protocol; this is a moderate gap.**

**Aggregate verdict.** Claims 1 and 3 require appendix or Track-D-primary-source consultation to verify. The cell-by-cell 3.04 vs 3.06 distinction is not in the paper at all. A reviewer who attempts to verify all three quantitative claims in the abstract spends ~10 minutes (acceptable) but for Claim 1 cannot complete verification without external file access (problematic). **Recommendation:** add a one-line cell-by-cell table to §A.4 or §A.5 showing the $(d, b, n) \times$ scale $\times$ repeat measurements that aggregate to the 3.04--3.06 constant. Page cost: ~6 lines.

**Severity.** MODERATE. Reviewer-likely-revision-request.

---

## Pass J — "Why this paper now?" framing

**§1 Introduction reading.** Paragraph 1 (line 65): "Inference-time interventions on frozen language models---from one-shot activation steering to iterative test-time tree search---offer a frontier for both capability scaling and mechanistic analysis." Then poses the geometric question. Paragraph 2 (line 68): names prior application domains (alignment faking, gradient-trained input attacks) as monitor-driven sub-cases. Paragraph 3 (line 70): introduces the *reward-driven* case and the infrastructure required. Paragraph 4 (line 73): summarizes empirical findings + reframe to diagnostic probe.

**Implicit "why now" answer.** The paper's implicit motivation is: "test-time scaling + inference-time intervention is an active research frontier; reasoning-improvement papers in this space rely on continuous-latent search; before deploying such methods, we should know whether the search can be trusted to navigate the semantic manifold." This is *implicit*, not explicit. A reviewer asking "why this paper now?" gets the answer from sentence 1 of §1, but indirectly.

**Convincingness.** Convincing for the Interpretability-track reviewer (who values mechanism-diagnosis + algorithm-as-probe framings — see Gemini Interpretability Track report §1, "the track functions as a primary venue for the methodological infrastructure required to trace, map, and mathematically bound the internal mechanics of massive neural networks"). **Less convincing for a reviewer routed from ML4NLP / Generation tracks**, who may read the paper as "negative result about a not-very-mainstream reward design on a relatively small benchmark." For that audience, the "why now" answer should be stronger.

**Recommendation.** Add a single sentence at the end of §1 paragraph 1 (or paragraph 2): "As inference-time methods proliferate (e.g., recent COCONUT, KV-cache steering, and process-reward-model approaches), establishing baseline reliability conditions for latent-space rewards becomes load-bearing for downstream method-design choices." Page cost: 1 line. Trade-off: tightens "why now" for non-interpretability reviewers; minimal risk.

**Severity.** MINOR. Not reviewer-rejection-likely; "why now" framing is competently if implicitly handled.

---

## Pass K — Speculation vs page-budget tradeoff (re-evaluated post-`267e9cb`)

**Material change.** Commit `267e9cb` dropped the 20B numerical projections from body. The four items originally to scrutinize were: (a) §6 20B numbers (162 GB / 60× / 493 GB); (b) §7.5 anti-Goodhart enumeration; (c) §A.5 anon URL; (d) cross-scale Δ values.

**Re-evaluated item (a) — §6 20B numbers.** Post-commit §6.2 line 319 reads: "The asymptotic separation is independent of model scale and is the methods contribution of this paper; concrete scale-specific projections (notably for $\sim$20B open-weight architectures such as \texttt{gpt-oss-20b}) and the multi-GPU runs needed to anchor them empirically are deferred to §\ref{sec:future-work}." This is the right move. The 20B numbers are no longer in body §6.2. They survive in §7.5 Future Work (line 356: "projected $\approx 162$~GB working set at 20B") as a single number used to motivate multi-GPU as a Phase B direction. **Recommendation:** KEEP at the §7.5 location. The number is now anchored in its rightful place (future-work motivation, not method claim). Page-budget cost: 0 (it's already in §7.5).

**Re-evaluated item (b) — §7.5 anti-Goodhart enumeration.** Day-10 Batch 3 cut #6 reduced the 8-candidate enumeration to a 3-candidate representative shortlist. The current §7.5 (line 356) lists "Anti-Goodhart reward designs that may resist the reward-driven Dimensional Escape established by Experiment~1---candidates surveyed in §\ref{sec:discussion-implications}---constitute a research program in their own right." §7.1 (line 329) carries the longer enumeration: "multi-channel ensembles, external Process Reward Models, self-consistency across stochastic rollouts, counterfactual sensitivity, anti-prior penalties, information-bottleneck objectives." **Recommendation:** TRIM further. §7.1 enumeration could shrink to "multi-channel ensembles, external Process Reward Models, and self-consistency variants" (matching §7.5's shorter form) — saves ~1 line. Or KEEP if not at budget edge. Low risk either way.

**Re-evaluated item (c) — §A.5 anon URL placeholder.** Line 492: "the anonymized supplementary materials package, available at \url{https://anonymous.4open.science/r/kv-mcts-dimensional-escape-XXXX} (the final slug is assigned at Day-12 supplementary build per plan v2 §2)." **Critical issue.** The URL still has the placeholder `XXXX`. At submission, this MUST be replaced with the live slug. If the supplementary upload happens on or before Day-14 (2026-05-24) and the slug is live, then the URL substitution is mechanical and the page-cost is 0. If the URL is *still placeholder* at submission, this is a paper-text-error that any reviewer will flag. **Recommendation:** KEEP the line, **block the submission on the slug replacement** as a non-negotiable Day-14 gate item.

**Re-evaluated item (d) — Cross-scale Δ values (Δ ≈ 0.67 for C2, Δ ≈ 0.55 for C3).** Track G report §2.1 has measured -0.667 and -0.547. The paper §5 Exp 2 finding (i) line 228 reports "$\Delta \approx 0.67$" and "$\Delta \approx 0.55$" (round-then-subtract: -0.667 rounds to -0.67, subtract yields 0.67; vs subtract-then-round: -0.547 rounds to -0.55 — these match the paper). The round-then-subtract vs subtract-then-round distinction is fine for a 2-decimal-place reporting. **Recommendation:** KEEP. These are measured; minor rounding choice does not warrant audit-trail explanation in body.

**Aggregate Pass K verdict.** Post-`267e9cb`, the speculation-vs-budget tradeoff is largely resolved at body level. Residual residue: §7.1 enumeration could be trimmed further; §A.5 URL must be filled at submission.

**Severity.** MINOR.

---

## Pass L — Cross-section numerical-consistency check

**Audit item (a) — M_KV at 20B across §4.2.1 / §5 / §6.2.** Post-`267e9cb`:
- §4.2.1 deleted (Day-10 Batch 4 cut #13). No 40 GB anchor in body.
- §5 Impl Notes moved to §D (Day-10 Batch 2 cut #5). The Phase 2 80 GB approximate number is at line 535 in §D Implementation Notes item (iv): "with a Phase~2 floor at $20$B scale of approximately $80$~GB before any accumulator overhead."
- §6.2 line 319 contains no 162 GB anchor in body. The 162 GB anchor appears only in §7.5 Future Work (line 356).
- §A.4 Track D protocol (line 486) gives "3.04--3.06 \cdot M_{\text{KV}}" as the *constant factor*; M_{\text{KV}} itself is *not* numerically anchored in §A.4.
- §E Algorithm Detail Memory Complexity (line 566) mentions "approximately 50~MB" for 20B FP32 accumulator example.

**Reviewer-walking-through-the-paper sequential reading order:** abstract (no number) → §1.1 C1 (no number for 20B) → §3 (no number) → §4 (no number) → §5 (no number) → §6.1 (no number) → §6.2 line 319 (no number, deferred to §7.5) → §7.5 line 356 ("projected $\approx 162$ GB working set at 20B exceeds 80 GB H100"). **A reviewer encountering line 356 sees the 162 GB number for the first time at §7.5 with no prior body anchor.** This is good (no inconsistency); the worked-numbers consolidation to §7.5 + §D Impl Notes is consistent.

**Residual issue.** The §D Implementation Notes item (iv) "Phase~2 floor at $20$B scale of approximately $80$~GB before any accumulator overhead" (line 535) and the §7.5 "projected $\approx 162$~GB working set at 20B" (line 356) describe different things — the §D number is the bf16 KV cache without accumulator overhead under the Phase 2 implementation (baseline clone), the §7.5 number is the projected total working set. **A reviewer doing reconciliation will need to read §D line 535 ("$2$--$3 \cdot M_{\text{KV}}$ ... Phase~2 floor at $20$B scale of approximately $80$~GB") to understand that the 80 vs 162 difference is the Phase 2 baseline-clone constant vs the Phase 3 projected constant.** §D does the reconciliation in line; the appendix-only reconciliation is appropriate.

**Severity for (a).** MINOR. Body-level consistency is fine post-Day-10.

**Audit item (b) — Memory bound constant 3.04--3.06.** Abstract: "$3.04$--$3.06\times M_{\text{KV}}$". §1.1 C1 (line 82): "measured at $3.04$--$3.06\times M_{\text{KV}}$ above the baseline cache across the tested depth and prefill configurations on Llama~3.2-1B/3B". §A.4 Track D (line 486): "The $3.04$--$3.06 \cdot M_{\text{KV}}$ constant-factor memory result". **Consistent across abstract / §1.1 / §A.4. No 493 GB worked-example to reconcile (post-`267e9cb`).** Cell-by-cell $(d, b, n)$ independence claim primary-source verified: §A.4 line 486 states "Steady-state delta was identical across all $(d, b, n)$ cells at each scale (median over three repeats per cell)." Reviewer must trust §A.4's "median over three repeats" without per-cell numbers — see Pass I Claim 1.

**Severity for (b).** MINOR (with Pass I caveat).

**Audit item (c) — Cartography Δ rounding.** See Pass K item (d). Round-then-subtract vs subtract-then-round resolved. KEEP audit-trail-note in supplementary only.

**Severity for (c).** MINOR.

**Audit item (d) — "$\sim$60$\times$ reduction" instances.** Post-`267e9cb`, NO instances of "$\sim$60$\times$" remain in body or abstract. The number has been entirely removed. **Pass L item (d) is fully mooted.** No reconciliation issue.

**Aggregate Pass L verdict.** All four audit items resolve to MINOR or MOOT. The Day-10 page-budget cuts + Day-11 commit `267e9cb` have substantially closed the cross-section numerical-consistency gap.

---

## Pass M — Abstract-to-body claim escalation check

**Item (a) — 60× claim asymmetric hedging.** MOOT. Post-`267e9cb`, the 60× claim is removed from both abstract and body. The Day-11 marker on line 54 documenting the original hedge fix is now stale — the parenthetical "(projected from smaller-scale measurements; §\ref{sec:reproducibility})" referenced in the marker is also not in the current text. The pair just doesn't have the number anymore.

**Severity.** MOOT.

**Item (b) — Croissant 1.1 release claim.** Abstract: "Artifacts are released with a Croissant~1.1 metadata package." §A.5 line 492 references "the anonymized supplementary materials package, available at \url{https://anonymous.4open.science/r/kv-mcts-dimensional-escape-XXXX}". §7.5 line 356: "A Croissant~1.1 dataset package of runtime evidence artifacts is included as supplementary material and will be extended in Phase~B." §7.6 Ethical Considerations line 361: "The Croissant~1.1 metadata package (§\ref{sec:future-work}) documents per-node runtime evidence artifacts from Experiments~1--2 with provenance and usage-constraint fields." 

**Risk:** if the supplementary URL slug is still `XXXX` at submission, the abstract's present-tense claim "are released with" is false. **Day-14 gate item:** verify the URL is live and the slug is filled before submission. If not, the abstract claim is reviewer-spotable inconsistency.

**Severity.** MODERATE — contingent on Day-14 supplementary build. If slug is filled, MINOR.

**Item (c) — "validated to machine precision over 200 apply--revert cycles on Llama 3.2-1B and Llama 3.2-3B".** Abstract anchor. §6.1 Theorem 1 (the proof). §7.3 Limitation 3(c) (line 346): "Reversibility (Theorem~\ref{thm:reversibility}) is empirically validated over 200 apply-revert cycles; very-deep searches (depth $\ge 50$ or cycle counts $\gg 200$) are unvalidated empirically, though the theorem implies exact reversibility independent of depth." §D Implementation Notes item (v) (line 537): "It was empirically validated to numerical zero drift over $200$ apply-reverse cycles on Llama~3.2-1B-Instruct (residual norm $= 0.00$, 2026-04-16 gate) and confirmed at the 3B scale (residual norm $= 0.00$, 2026-05-05 probe)."

**Consistency.** Abstract matches §6.1 + §7.3 + §D. Mild asymmetry: abstract says "validated to machine precision" — §D says "validated to numerical zero drift (residual norm $= 0.00$)". "Machine precision" implies floating-point representable epsilon; "numerical zero drift" implies exactly zero. The latter is *stronger*. So abstract is *weaker* than body — opposite of the failure mode this pass looks for. Acceptable.

**Severity.** MINOR.

**Item (d) — Unification claim "structurally analogous" / "we conjecture both as sub-cases".** Abstract (post-Day-11 hedge): "We argue that the reward-driven failure surfaced here is structurally analogous to the monitor-driven \emph{Orthogonal Escape} previously hypothesized for alignment faking~\cite{bailey2024obfuscated}: we conjecture both as sub-cases of the same overparameterized-latent-space pathology." §1 paragraph 4 (line 73): "We argue this reward-driven failure is structurally analogous to the monitor-driven sub-case hypothesized in the alignment-faking literature; Reversible KV-Cache MCTS is therefore most usefully positioned as a high-precision diagnostic probe ... one that empirically exposes the reward-driven sub-case of Dimensional Escape, with the monitor-driven sub-case left to Phase~B research." §1.1 C2 (line 84): "We further argue the reward-driven pathology surfaced here is structurally analogous to the monitor-driven \emph{Orthogonal Escape} previously hypothesized for alignment faking~\cite{bailey2024obfuscated}: we conjecture both as sub-cases of the same overparameterized-latent-space geometry (§\ref{sec:dim_escape})." §3.2 line 142: "Bailey et al.~\cite{bailey2024obfuscated} characterized the monitor-driven case (gradient-trained input-space attacks) with a tentative negative result; we characterize the reward-driven case empirically (§\ref{sec:exp1}--\ref{sec:exp2}) and find it manifests robustly across model scale and prompt class. The monitor-driven case under inference-time KV-cache perturbation is left to Phase~B research."

**Day-11 hedge audit.** Abstract / §1 / §1.1 / §3.2 all use "structurally analogous" + "conjecture" — consistent. The Day-11 hedge (per marker at line 54: "argue ... structurally analogous ... we conjecture both as sub-cases") has been applied consistently across all four anchor sites. **The Pass M(d) asymmetric-hedging risk is closed.**

**Residual risk.** A reviewer reading the abstract carefully will see "we conjecture both as sub-cases" and then check §5 for evidence of both sub-cases. §5 evidences only the reward-driven sub-case (Track F + Track G). The conjecture is honest — explicitly named "conjecture" — but the abstract's framing strength is "we argue" + "we conjecture". A skeptical reviewer may write: "The unification framing of Dimensional Escape rests on conjecture (the monitor-driven sub-case under inference-time KV-cache perturbation is unmeasured). I'd prefer the paper to be framed as a reward-driven Goodhart result, with the monitor-driven unification as a discussion section rather than a contribution-level claim." This is reviewer-style disagreement, not a factual gap. **Severity:** MODERATE per Pass N below.

**Aggregate Pass M verdict.** All four items resolve to MOOT / MODERATE / MINOR / MINOR (closed by Day-11). The Day-11 hedge work substantially closed Pass M's risks.

---

## Pass N — Contributions coherence stress test (STRENGTHENED post-Day-10 drop-C2)

**§1.1 Contributions list post-drop-C2 contains 2 items:**
1. Reversible Monte Carlo Tree Search in KV-cache latent space (algorithm + FP32 accumulator + memory bound + theorem) — line 82.
2. Empirical demonstration of Dimensional Escape under continuous-latent search (Experiments 1+2; structurally-analogous claim with Bailey 2024) — line 84.

**Strengthened item (a) — narrative reading of two-item Contributions list.** A reviewer reading the two contributions sequentially encounters:
- C1: "We introduce an inference-time algorithm ... We prove zero-drift reversibility ... working-set memory complexity is $O(M_{\text{KV}} + d \cdot K_{\text{acc}})$ ... measured at $3.04$--$3.06\times M_{\text{KV}}$ ... gradient-free and operates on a frozen model at inference time, distinguishing it from gradient-based input-space attacks and training-time monitor-evasion methods."
- C2: "We establish a geometric pathology of MCTS in overparameterized latent spaces: surrogate rewards lacking exogenous grounding inevitably resolve localized optimization pressure by routing representations through unconstrained dimensions, succumbing to prior-amplification or greedy-collapse rather than satisfying the semantic objective."

The narrative arc: C1 introduces a tool; C2 uses the tool to find a pathology in the tool's obvious use case. The pathology is named "Dimensional Escape" and the conclusion is "Reversible MCTS is therefore a high-precision diagnostic probe ... rather than a reasoning-improvement tool" (C2 closing).

**A reviewer's first-impression mental model after reading C1 alone:** "method paper; new algorithm with memory bound." Compatible expectation: §5 demonstrates the method working.

**A reviewer's revised mental model after reading C2:** "the method's primary use case (entropy-based MCTS reasoning) doesn't work, and the authors reframe as diagnostic probe."

**Does this read as complementary diagnostic finding or as withdrawal of headline claim?** The current framing leans toward *complementary diagnostic finding*, because (i) C1's headline claim is the *algorithm and memory bound*, which work; (ii) C2's headline claim is *the pathology empirically demonstrated*, which holds; (iii) the reframe ("diagnostic probe rather than reasoning-improvement tool") is presented as a new positioning, not a withdrawal. But a hostile reviewer may read it as: "C1 is method, C2 is method-doesn't-work; collectively this is an algorithmic paper whose payload is a negative result, and the reframe is post-hoc."

**Severity for (a).** MODERATE. The two-item list does not collapse into incoherence, but it requires the reviewer to perform the reframe themselves. Less risk than three-item list with telemetry-matrix sandwich (pre-Day-10), but more risk than three-item list where C2 was monitoring-substrate.

**Item (b) — first-appearance order of "diagnostic probe" reframe vs "algorithm-headline packaging".** Sequential reviewer encounter order:
1. Abstract sentence 9 (line 56): "Reversible MCTS is therefore a high-precision diagnostic probe of frozen-model representational geometry rather than a reasoning-improvement tool." (REFRAME first)
2. Abstract sentence 1: "We introduce \textbf{Reversible Monte Carlo Tree Search in KV-cache latent space}, an inference-time algorithm that mutates a frozen Transformer's cached key/value tensors in place via FP32 accumulators..." (ALGORITHM-HEADLINE first sentence of abstract; REFRAME at sentence 9)
3. §1 paragraph 4 (line 73): "Reversible KV-Cache MCTS is therefore most usefully positioned as a high-precision diagnostic probe..." (REFRAME at end of §1)
4. §1.1 C1 (line 82): full algorithmic packaging.
5. §1.1 C2 (line 84): full pathology + reframe.

The reframe appears at abstract sentence 9 (after the algorithm-introduction) and at §1 closing. The §1.1 C1 then re-introduces the algorithm-headline packaging fresh — *after* the reframe has been delivered. **A sequential reviewer reads: algorithm intro → algorithm formal claims → pathology framing → diagnostic reframe → contributions list (C1 algorithm-packaging again) → contributions list (C2 pathology-packaging with reframe-restated).** The reframe is delivered *twice* before the §1.1 list, then C1 lands as algorithm-headline again. **This order is acceptable but mildly redundant** — by the time the reviewer reaches C1 they have read "diagnostic probe rather than reasoning-improvement tool" twice. A reviewer with low confidence might read this as defensive ("they keep saying it's a probe, not a reasoning tool — suggesting reviewers were going to question that").

**Severity for (b).** MINOR. Order is fine; redundancy is a tone risk but not a substance risk.

**Item (c) — sufficient §4.2 reward-agnostic redirection.** §4.2 MCTS Node Reward paragraph (line 168–169): "The reward $\hat{r}(\mathbf{h})$ is a design choice; the framework is reward-agnostic. We test the entropy-normalized reward (Eq.~\ref{eq:goodhart}) as a null-control under Hypothesis~\ref{hyp:goodhart}; anti-Goodhart variants are a Phase~B research program (§\ref{sec:future-work})."

§7.1 Implications for Latent-Space Reward Design (line 329): "Our Experiment~1 result establishes the null-control against which any candidate must be measured."

A reviewer who has formed the "method paper" prior from §1.1 C1 and then encounters §5 Experiment 1's null result: does §4.2's "reward-agnostic" + §7.1's "null-control" redirect successfully?

**Verdict.** Adequate but not airtight. The §4.2 "we test the entropy-normalized reward ... as a null-control" framing is the key sentence. A reviewer reading §4.2 carefully will register "null-control" — i.e., the authors *expected* entropy reward to fail and treat the failure as the diagnostic finding. But a reviewer reading §4.2 fast may register only "reward-agnostic" and miss the null-control framing. The §7.1 sentence completes the redirection but is far from §1.1 C1's algorithm-headline.

**Severity for (c).** MODERATE. Reviewer who reads slowly is redirected; reviewer who reads fast may not be. A pre-emptive sentence at §1.1 C1 closing — e.g., "We position the framework as a diagnostic probe; the algorithm's primary use-case as a reasoning-improvement tool is itself the empirical question we test (Experiment~1)." — would make the redirection explicit at the contribution level. Page cost: ~2 lines.

**Aggregate Pass N verdict.** Two-item Contributions list is more honest than three-item list with telemetry-matrix sandwich; the narrative tension (Contribution 1 method, Contribution 2 method's use case fails) is moderate, not severe; the reframe is delivered consistently but mildly redundantly. **MODERATE severity overall.** Recommendation: add the one explicit redirection sentence to §1.1 C1; do not restructure further.

---

## Pass O — Operational-scaffolding triage

**Day-9 page-budget cut plan already MOVED or DELETED most operational scaffolding.** Verify Day-10 application caught everything:

**§5.1 Phase 0 Architectural Decoupling — DELETED.** ✓ Confirmed: §5.1 + §5.2 cut (Day-10 Batch 2 markers at line 181).

**§5.2 Phase A Local Prototyping on Lightweight Surrogates — DELETED.** ✓ Confirmed: same cut.

**§5 Implementation Notes — MOVED to §D.** ✓ Confirmed: body §5 line 179 retains only a one-sentence pointer.

**§7.4 Empirical Measurement Prerequisites — MOVED to §G.** ✓ Confirmed: Day-10 Batch 3 cut at marker line 352.

**Residual operational scaffolding the Day-10 cuts may have missed:**

1. **`[IMPL: ...]` comments.** Search confirms instances at lines 218, 235, 285, 493. These are LaTeX comments (don't render). Not reader-visible. NO ACTION.

2. **"per Track F/G report" inline references.** Body has no "Track F/G report" inline phrasing (verified). The Track F / Track G nicknames live in:
   - §5 Exp 1 marker [IMPL: COMPLETE — driver script at scripts/diagnose_track_f_negcontrol.py ...] (line 218) — comment, not rendered.
   - §A.4 paragraph headers: "Static entropy–correlation correlation (Track A)" / "Benchmark calibration (Track C)" / "VRAM measurement protocol (Track D)" / "Entropy-MCTS negative control (Track F / Experiment~1)" (lines 479–489). These DO render. The "Track A/C/D/F" naming is internal scaffolding; published papers don't usually carry phase-tracker names. A reviewer encountering "Track C" without context will wonder what Tracks A, B, ... E are. **RECOMMENDATION: TRIM** — remove "Track A/C/D/F" parenthetical from §A.4 paragraph headers. The internal track names are not needed; the reader can find each protocol by its descriptive title. Page cost: ~0 (just remove parentheticals). Severity: MINOR.

3. **§A.3 Phase 2 / Phase 3 distinction.** §D Implementation Notes (line 524–537) is titled "Implementation Notes (Phase 2 / Phase 3 Distinction)" — the "Phase 2 / Phase 3" terminology is internal project-management nomenclature. A reviewer reading the appendix encounters this without context. **RECOMMENDATION: KEEP** but consider renaming "Phase 2 / Phase 3 Distinction" to "Reference Implementation vs Paper Specification" — communicates the same intent in conventionally academic terms. Severity: MINOR.

4. **"§5.2 Phase A Local Prototyping" and "Phase B" terminology in §5 line 186 and §7.5 line 356.** §5 line 186: "Scaling to models up to 20B parameters on H100 is planned for Phase~B follow-on work; see §7 Future Work." §7.5 line 356: "Several Phase~B directions extend this work." **The "Phase B" naming is conventional research-program shorthand** and reads cleanly in body. KEEP.

5. **`[REVISION | ... — pending audit]` markers.** Hundreds of these are inline LaTeX comments (don't render). NO ACTION on rendering. Risk: they contain Day-9/Day-10/Day-11 internal team discussion language ("Josh-approved", "Day-10 Batch", etc.). Should be stripped from the camera-ready submission per ARR practice. **Day-14 gate item:** verify the submitted PDF and any source-published artifact (overleaf/zenodo) does not leak these comments. Severity: MAJOR for camera-ready; MOOT for review-version PDF since comments don't render.

**Aggregate Pass O verdict.** Day-10 cuts caught the major operational scaffolding. Two minor residual issues (Track A/C/D/F parenthetical in §A.4; "Phase 2 / Phase 3" subsection naming in §D) can be polished. The `[REVISION]` comment leakage is a real risk only if source is published; the PDF is clean.

**Severity.** MINOR (with MAJOR contingent risk for source leakage).

---

## Pass P — Page-budget format-compliance hard limit

**Material context.** Per [acl_formatting.md:17](../NeurIPS/acl_formatting.md:17): "Review versions of long papers may have up to eight (8) pages of content plus unlimited pages for references." Per [acl_formatting.md:21](../NeurIPS/acl_formatting.md:21): "For both long and short papers, all figures and tables that are part of the main text must fit within these page limits." Per [acl_formatting.md:27](../NeurIPS/acl_formatting.md:27): "Papers that do not conform to these requirements may be rejected without review." Limitations and Ethics sections count toward the 8 pages per ARR norm.

**Stated body length per prompt:** "approximately 9 pages" post-Day-10. **Stated current state per memory pointer (recast_action_plan.md):** "body fits exactly 8 pp per acl_formatting.md:17; smoke-compile clean (PDF 13pp/464KB/0 undefined refs)." Memory pointer claims Day-11 work brought body to exactly 8pp.

**I cannot directly verify page count from TeX source without compiling.** Approximation from line count: TeX file is 634 lines total; appendix starts at line 466 (`\appendix`); body lines are 1–465 with 466–634 being appendix + bibliography. Body line-count ≈ 465 lines of LaTeX source. Without compilation I cannot map source-lines-to-rendered-pages exactly, but: the per-prompt assumption was 9pp pre-Day-11; the Day-11 Pass P / Pass M hedge work reportedly brought body to 8pp.

**Item (a) — is ACL ARR format-compliance automatic-page-count or programme-chair-discretion?**

Per [acl_formatting.md:27](../NeurIPS/acl_formatting.md:27): "Papers that do not conform to these requirements may **be rejected without review**." The phrasing "may be rejected" — not "will be rejected" — leaves discretion to the program chairs / submission-handling chairs. Historically (per ARR FAQ and ACL anthology submission norms), ARR uses automated PDF page-count checks at submission; papers over 8 pages typically receive a desk-reject email asking for a resubmission within 24 hours. Cover-letter justifications for going over are uncommon and not formally supported in the standard ARR submission interface. **The de-facto answer:** for the 8-page limit specifically (long paper review version), the check is functionally automatic, with no cover-letter recourse for ≥1 page over. **Severity if body is 9pp:** SEVERE (desk-reject likely).

**However,** if memory pointer is correct that Day-11 work brought body to 8pp exactly, Pass P risk is moot. **Day-14 gate item:** compile the current TeX and verify the page count of the rendered body (lines 1–465 of source) is ≤8pp. If 9pp, additional cuts required.

**Item (b) — least-damage cuts if body must shrink from 9pp to 8pp.** Ranked by (framing-impact / page-savings):

1. **MOVE Figure 1 (cross-scale mean-step α) to appendix.** (Lines 256–261.) Framing impact: low. Figure 1 visualizes Table 2 contents; the table carries the numerical evidence. Page savings: ~0.4pp. Ratio: low/0.4 = LOWEST framing impact for largest savings.
2. **MOVE Table 1 (Experiment 1 results) to appendix.** (Lines 194–212.) Framing impact: moderate. Table 1 anchors the §5 Results paragraph's accuracy claims; moving forces the body prose to carry the numbers inline. Page savings: ~0.35pp. Ratio: moderate/0.35.
3. **TRIM §5 Experiment 2 finding (iii) (gold-rank preservation, lines 232–233).** Framing impact: moderate (this finding is the structural-mechanism nuance). Page savings: ~0.15pp. Ratio: moderate/0.15.
4. **TRIM §7.1 anti-Goodhart enumeration (line 329).** Already trimmed once. Could shrink "multi-channel ensembles, external Process Reward Models, self-consistency across stochastic rollouts, counterfactual sensitivity, anti-prior penalties, information-bottleneck objectives" to "multi-channel ensembles, external Process Reward Models, and self-consistency variants". Page savings: ~0.05pp. Ratio: low/0.05.
5. **MERGE §3 with §4.** Framing impact: HIGH (§3 Dimensional Escape + Hypotheses is the conceptual anchor; merging into §4 Proposed Method conflates concept and algorithm). Page savings: ~0.2pp (heading + transition prose). Ratio: high/0.2 — NOT RECOMMENDED.
6. **MERGE §6 with §4.** Framing impact: HIGH (§6 Theorem 1 + Proposition 1 are method-grounding theorems; folding into §4 muddies the algorithm-vs-formal-result distinction). NOT RECOMMENDED.

**Top recommendation if 1pp must come out:** Move Figure 1 to appendix (lowest framing impact). If 0.5pp must come out: Move Figure 1 or trim §5 Exp 2 finding (iii).

**Item (c) — Risk profile comparison: submit at 9pp with cover letter vs defer to ARR June 2026 cycle.**

**Path 1 — Submit at 9pp with cover-letter justification:**
- Risk profile: desk-reject upon submission within 1–7 days of submission close. Cover-letter justification is unlikely to be entertained for an absolute hard limit specifically called out in the formatting requirements. No human review reached.
- Probability of clean acceptance through this path: low (~10–20%).
- Failure mode: 100% loss of submission cycle; must re-route to ARR June 2026 anyway after rejection.
- Trigger: page-count check at submission portal.

**Path 2 — Defer to ARR June 2026:**
- Risk profile: gains 1 month for additional experiments (multi-direction, baseline R-uninformed per Pass C, possibly Procrustes prep for sub-7B per Pass B Limitation 4). EMNLP commit deadline 2026-08-02 still met. The trade-off: 1-month delay vs paper quality.
- Probability of acceptance through this path: depends on how the extra month is used; if used for the multi-direction + R-uninformed baselines, modest increase. If wasted, no change.
- Failure mode: 1 month sunk cost if extra experiments don't strengthen the paper.
- Trigger: voluntary deferral.

**Path 3 (not in original prompt but worth naming) — Cut to 8pp NOW (Day-12 emergency surgery):** Move Figure 1 to appendix (~0.4pp), trim §7.1 (~0.05pp), trim §5 Exp 2 finding (iii) (~0.15pp). Total ~0.6pp savings = 9pp → 8.4pp. May still be over. Combined with §A.4 Track-naming-removal (Pass O) and §3.2 closing prose trim, possibly 8.0pp achievable.

**Assessment.** Path 3 is the lowest-cost path IF body is currently 9pp. If body is currently 8pp per memory pointer, no action needed and Pass P is MOOT.

**Severity.** SEVERE if body is currently 9pp at submission; MOOT if 8pp confirmed. **Day-14 gate item:** PDF page-count verification is critical. The single highest-priority verification item in the entire patch window.

---

## Pass Q — Body §4 Proposed Method depth

**Current body §4 content (post-Day-10):**
- §4.1 (formerly §4 Reversible MCTS subsection) — title "Reversible MCTS in KV-Cache Space" (line 154).
- One introductory paragraph (line 156): notation + in-place mutation framing + pointer to Appendix~\ref{sec:appendix-algorithm} for full algorithm.
- FP32 Accumulator paragraph (line 158–166): paragraph prose + Eq.~\ref{eq:fp32_accumulator}.
- MCTS Node Reward paragraph (line 168–169): reward-agnostic framing + pointer to Eq.~\ref{eq:goodhart} + Hypothesis~\ref{hyp:goodhart}.

Body §4 length: approximately 1 page of two-column ACL render.

**What a reviewer who does not read appendices sees in body §4:**
- Notation: $\mathbf{K}_t^{(\ell)}, \mathbf{V}_t^{(\ell)}, \mathbf{d}_K^{(\ell)}, \mathbf{d}_V^{(\ell)}, \alpha_K, \alpha_V$.
- The in-place mutation example: $\mathbf{K}_t^{(\ell)} \leftarrow \mathbf{K}_t^{(\ell)} + \alpha_K \cdot \mathbf{d}_K^{(\ell)}$.
- Reverse: $\mathbf{K}_t^{(\ell)} \leftarrow \mathbf{K}_t^{(\ell)} - \alpha_K \cdot \mathbf{d}_K^{(\ell)}$.
- The FP32 accumulator equation (Eq.~\ref{eq:fp32_accumulator}) restoring bf16 cache.
- Memory-bound claim ($O(M_{\text{KV}} + d \cdot K_{\text{acc}})$ vs $O(b^d \cdot M_{\text{KV}})$).
- A pointer to Theorem~\ref{thm:reversibility} (in §6.1).
- A pointer to Appendix~\ref{sec:appendix-algorithm} (Step 1/2/3 prose).

**Algorithmic details a reviewer needs in body to verify the theorem:**
- Step 1 Forward Mutation (Eq.~\ref{eq:forward_mutation}): given in body line 156 inline.
- Step 2 Evaluation: NOT in body — moved to §E Appendix Algorithm Detail (line 557).
- Step 3 Reverse Rollback (Eq.~\ref{eq:reverse_rollback}): given in body line 156 inline (as the inverted-by formulation).
- FP32 Accumulator (Eq.~\ref{eq:fp32_accumulator}): in body (line 161).
- Step 3 implementation via accumulator decrement: NOT in body — moved to §E line 564.
- Memory Complexity worked example: NOT in body — moved to §E line 566.

**Verdict on (a).** Body §4 carries: notation, in-place mutation form (Eq. 4 inline), reverse mutation form (Eq. 4 inline inverted), FP32 accumulator equation (Eq. 5). It does NOT carry: the explicit three-step algorithm, the labeled forward/reverse mutation equations as a flow, or the Memory Complexity worked example with $S' = 10$ for 20B (which lives in §E). A reviewer who does not read appendices sees the *concept* of the algorithm (forward + reverse + FP32 accumulator) but not the full three-step procedure as a numbered algorithm. They can verify Theorem 1's *statement* from §6.1 (which lives in body) but to verify the proof must read §C Appendix Theorem Proof Detail. They can verify Proposition 1's *statement* from §6.2 (body) but to verify the constant-factor derivation must read §A.4 + §D.

**Verdict on (b).** Per Gemini Interpretability Track report (file `Analyzing Interpretability Track Submissions.md` lines 36–46, 62–70, 84–92), successful Interpretability-track papers carrying formal complexity claims (e.g., *An L\* Algorithm for Deterministic Weighted Regular Languages*, *Finite State Automata Inside Transformers*, *BC-Prover*) **routinely carry algorithm pseudocode + formal statements in body**, with proof details deferred to appendix. The pattern is: body presents the algorithm as a labeled procedure (numbered steps or pseudocode) + the formal theorem statement; appendix carries the proof. Body §4 in this paper provides the prose-level introduction and the FP32 accumulator equation, but does not carry a numbered-step algorithm presentation.

**Verdict on (c).** Body §4 is thinner than the Interpretability-track norm for algorithm-primary papers. The §E Appendix Algorithm Detail content (lines 547–566) — specifically the three labeled paragraphs "Step 1: Forward Mutation", "Step 2: Evaluation", "Step 3: Reverse Rollback (Restore)" with their respective equations — is the kind of content track-norm papers carry in body.

**Minimal re-fold from §E to body §4:**
- The three labeled paragraphs ("Step 1", "Step 2", "Step 3") with their equations (Eq.~\ref{eq:forward_mutation} + Eq.~\ref{eq:reverse_rollback}). Eq.~\ref{eq:forward_mutation} is the labeled version of what body line 156 currently has inline; folding it back as an explicit equation costs ~3 lines. Eq.~\ref{eq:reverse_rollback} similarly ~3 lines. Step 1/2/3 labels: ~3 lines. Total page cost: ~9 lines = ~0.15pp in two-column ACL.

**Severity.** MODERATE. Reviewer-likely revision request — "I cannot evaluate Theorem 1 / Proposition 1 from the body alone; please pull the three-step algorithm back into body §4 so the algorithm is self-contained without appendix consultation." Not a paper-rejection-likely issue for the Interpretability track — that audience reads appendices — but for a reviewer routed from ML4NLP / Methods tracks, body §4's depth is borderline.

**Recommendation.** **RE-FOLD** the three labeled paragraphs (Step 1 / Step 2 / Step 3 with Eq.~\ref{eq:forward_mutation} + Eq.~\ref{eq:reverse_rollback}) from §E into body §4. Costs ~0.15pp. If body is at exactly 8pp post-Day-11, this re-fold pushes body to ~8.15pp — net cost ~0.15pp over budget; combined with Pass P recommended trims (Figure 1 → appendix, Track-naming removal) the budget can absorb. The tradeoff: better algorithm self-containment in body vs marginal page cost.

---

## Pass R — Empirical surface adequacy

**Current empirical scope:**
- Models: Llama 3.2 family at 1B and 3B parameter scales (two scales, one family).
- Experiments: 2 reported (Experiment 1 entropy-MCTS negative control on ARC-Easy at $n=200$ paired items × 2 scales = 400 items; Experiment 2 Latent Cartography across 4 prompt classes × 100 items × 2 scales = ~770 items after C3 truncation).
- Auxiliary measurements (in appendix): Track A (entropy-correctness correlation on 85 factual items × 2 scales); Track C (benchmark calibration on 100 items × 2 scales); Track D (VRAM measurements on 1B + 3B + derived 20B projection).
- Positive results among reported experiments: zero.
- Headline statistical claim: paired McNemar $p = 1.0$ at both 1B and 3B with 3 discordant items per scale.

**Item (a) — How does empirical scope compare to EMNLP/ACL Interpretability-track norms?**

Per Gemini Interpretability Track report (file `Analyzing Interpretability Track Submissions.md`), the 26-paper sample includes papers with empirical scope ranging from single-model deep mechanistic studies (e.g., *Finite State Automata Inside Transformers* — Mamba + GPT-2 family + multiple state-tracking tasks) to large-scale benchmark papers (e.g., *CausalGym* — multiple causal interpretability methods on multiple linguistic tasks). The Gemini report (line 38, line 88) characterizes top-quartile Interpretability papers as: (i) algorithm-primary papers test on 1–2 model families with deep mechanistic analyses ("nearly 100% accuracy in tracking world states" with stress tests); (ii) negative-result papers (CUTE, *Lexical Recall*) test on multiple model sizes with controlled diagnostics + clear mechanism diagnoses; (iii) the modal pattern is 1–3 model families × 2–6 experiments × 1 algorithm/method as primary novelty.

**Verdict on (a).** **One model family + two experiments is at the lower end but within the typical range for Interpretability-track submissions, particularly for algorithm-primary papers.** The Gemini report's algorithm-primary precedent (*L\* Algorithm*, *BC-Prover*) routinely tests on one model family. The reviewer-prior for an Interpretability submission with formal claims (Theorem 1 + Proposition 1) anchored to the 1-family-2-scales experimental program is more forgiving than for an ML4NLP submission. **Within track norm.**

**Item (b) — Track F McNemar $p = 1.0$ with 3 discordant items: well-evidenced negative result or underpowered study?**

Track F report §3 (lines 78–94) reports: 1B 3/200 discordant ($b=1, c=2$); 3B 3/200 discordant ($b=2, c=1$). Track F report §3 closing (line 96): "Observed discordance is $\pi_d = 3/200 = 0.015$ — *one order of magnitude* below the headroom we calibrated for. There is essentially no MCTS-induced answer churn for the entropy reward to redirect."

**From reviewer-prior:**
- "Well-evidenced negative result" reading: the search machinery operates on 197 of 200 items by leaving the answer unchanged; the test detects $\delta = 0.10$ at $\alpha = 0.05$ for $\pi_d \leq 0.25$; observed discordance is two orders of magnitude below the design budget. The negative result is established by the *concordance* (197/200 items unchanged), not by within-discordant power. This reading is supported by §5 Mechanism paragraph (line 215): "1B selects $(1.0, 1.0, 1.0)$ on 91/200 items; predicts 'A' on 185/200 items against uniform gold." The mechanism is not "search has no effect" — it is "search converges on prior-amplification, which is wrong for ~62% of items but matches greedy on the rest."
- "Underpowered study" reading: 3 discordant items cannot rule out an effect within the 3 items; we have 3-bit information. A reviewer who reads only the headline ($p = 1.0$) without the mechanism paragraph will fall into this reading.

**Additional evidence a reviewer would want:**
1. **More discordant items via larger $n$.** Going to $n = 1000$ would 5× the wall-clock at each scale (~90 min at 1B, ~170 min at 3B) — feasible if prioritized in Day-12/Day-13. But the 3% concordance pattern is reproducible; more items unlikely to flip the conclusion.
2. **Different reward-objective variants showing same null.** Pass C recommendation: add R-uninformed (single-shot random perturbation without path-selection). Pass B recommendation: add multi-direction. Either would strengthen the "null is robust" claim.
3. **Per-stratum analysis.** E.g., difficulty-stratified: does entropy-MCTS help on harder ARC items where greedy fails? Track F §6 reports the 5 regression cases at 1B (all "A" predictions where gold ∈ {B, B, B, D, B}); a per-stratum table would amplify this.
4. **Ablation showing search-machinery-itself works.** This is Pass C's R-uninformed baseline.

**Item (b) verdict.** The current evidence is consistent with "well-evidenced negative result" given the mechanism characterization, but a fast-reading reviewer may default to "underpowered." Best risk-reduction: Pass C's R-uninformed baseline + minor expansion of §5 Mechanism paragraph.

**Item (c) — Absence of positive result: critical gap, acceptable scoping, or contingent?**

Three readings:
- (i) Critical gap (paper-level rejection trigger).
- (ii) Acceptable scoping choice given Dimensional-Escape-as-negative-finding framing.
- (iii) Acceptable contingent on §7.5 Future Work clearly articulating the research program that would deliver a positive finding.

**Pick.** **(ii) Acceptable scoping choice, with (iii) reinforcement.** Per Gemini Interpretability Track report (lines 72–82): "Negative findings in the accepted sample share a distinct pattern: they do not merely state that a model fails; they isolate the precise computational, representational, or heuristic reason *why* it fails." The paper does this via §5 Mechanism + §3.2 Hypothesis 2 + §7.1 Implications. Negative results without positive controls are accepted at the Interpretability track when the negative result is structured as a mechanism diagnosis (CUTE, Lexical Recall, Blind Spots — all in the 26-paper sample).

**However,** §7.5 Future Work currently lists multiple Phase B directions but does not commit to which is most likely to yield a positive result. A reviewer wanting reading (iii) reinforcement will want §7.5 to spotlight one or two anti-Goodhart reward designs as the most likely positive-result candidates. Page cost: ~3 lines.

**Verdict on (c).** Acceptable scoping. Recommend minor strengthening of §7.5 to spotlight 1–2 most-promising anti-Goodhart candidates.

**Item (d) — Strongest "you should have run experiment X" critique.**

Top candidate: **"You should have run an exogenously-grounded reward as a positive control: e.g., score the steered cache by next-token probability of the ground-truth answer letter (where ground truth is given by the benchmark). This is the canonical positive control: the reward function knows the answer. If MCTS-with-ground-truth-reward succeeds (e.g., $\Delta_{\text{GT-MCTS} - \text{random}} > 5$pp), it validates that the search machinery can in principle find correct cache states when the reward function carries exogenous signal; the entropy-reward null then reads as 'reward design matters, and surrogate rewards lacking exogenous grounding fail' (the paper's actual claim). Without this positive control, the negative result is consistent with 'the search machinery cannot navigate to correct cache states at all'."**

This is the load-bearing positive control a reviewer will name.

**Effort to run within Day-12/13/14 patch window.** Adding GT-reward to the existing Track F driver: ~1 hour script work. 1B sweep: 18 min. 3B sweep: 34 min. Total: ~2 hours wall. **Feasible.**

**Limitations-section pre-empt sentence (if not run):** "Sixth, the experimental program tests one surrogate reward (entropy-normalized) and one uninformative control (random); an exogenously-grounded positive control (e.g., ground-truth-conditioned reward) is not included. Whether the search machinery itself can navigate to correct cache states under exogenous reward is a separate empirical question; the present negative result for entropy reward should be read as 'this reward design fails,' not as 'the search infrastructure fails.'"

**Recommendation.** **RUN** if Day-12/13/14 absorbs it. The GT-reward positive control is a 2-hour wall investment for material strengthening of §5 Results. If not run, the Limitations sentence is mandatory.

**Severity.** MAJOR. The strongest reviewer-critique a paper-skeptical reader will write is "you tested a failing reward and didn't test a working reward — your negative result is consistent with the search itself failing."

---

## Final ranking — all findings by severity

### SEVERE (0)
None.

### MAJOR (4)
1. **Pass P** — Page-budget format-compliance hard limit if body is 9pp at submission. Desk-reject likely. *Mitigation:* Day-14 PDF page-count gate; cut Figure 1 + minor trims if 9pp confirmed.
2. **Pass D** — Methodology gap: paper title says "MCTS" but experiments use enumerated 27-path sampling (no UCB1). *Mitigation:* add §5 paragraph naming the search variant and its scope; add §7.3 Limitation 5.
3. **Pass R(d)** — Missing exogenously-grounded positive control (e.g., GT-conditioned reward). *Mitigation:* run in Day-12/13/14 (~2 hours wall) OR add Limitations sentence pre-empting the critique.
4. **Pass A** — Abstract overclaim "under any surrogate reward lacking exogenous grounding". *Mitigation:* hedge to "under the entropy-normalized surrogate reward we test, and conjecturally for other surrogate rewards"; soften "fundamental geometric pathology" to "geometric pathology we conjecture is fundamental".

### MODERATE (8)
5. **Pass G** — Limitations missing acknowledgement of algorithmic-contribution-versus-empirical-test mismatch (Pass D's other side). Add 5th limitation.
6. **Pass H** — Related-work blind spot: reward-hacking / Goodhart-in-RL literature not cited. Add Gao 2023 + Skalse 2022 to §3.2.
7. **Pass B** — Single-direction steering. Add Limitation sub-bullet OR run multi-direction (~10 hours wall combined).
8. **Pass C** — Missing R-uninformed baseline (single-shot perturbation, no path-selection). 1.5 hours wall to add.
9. **Pass E** — Reproducibility skepticism: 0.00 drift + $p=1.0$ + identical-across-cells constant factor. Add footnotes + cell-by-cell table to §A.
10. **Pass I** — Cell-by-cell 3.04 vs 3.06 numbers not in paper; reviewer cannot verify in 5 minutes.
11. **Pass N(a)+(c)** — Two-item Contributions list narrative tension (C1 method, C2 demonstrating method's primary use case fails); §4.2 redirection is adequate-not-airtight. Add explicit redirection sentence to §1.1 C1.
12. **Pass Q** — Body §4 thinner than track norm; algorithm Step 1/2/3 in appendix only. Re-fold from §E to body (~0.15pp page cost).

### MINOR (6)
13. **Pass F** — Title + abstract + §1 paragraph 1 first-page-reject-filter clears; isolated Pass A overclaim is the only fix.
14. **Pass J** — "Why this paper now?" framing convincing for Interpretability-track reviewers, less so for ML4NLP-routed reviewers. Add 1-sentence motivator.
15. **Pass K** — Speculation-vs-budget largely resolved post-`267e9cb`. §A.5 anon URL must be filled at submission (Day-14 gate).
16. **Pass L** — Cross-section numerical-consistency resolved post-Day-10 + Day-11. Minor residual: §D 80 GB vs §7.5 162 GB requires §D-reading for reconciliation; appropriate.
17. **Pass M** — Abstract-to-body claim escalation largely closed by Day-11 hedge work; residual MODERATE for §7.6 Croissant URL contingent on Day-14 build.
18. **Pass O** — Operational scaffolding largely caught by Day-10. Minor: §A.4 Track A/C/D/F parenthetical naming; §D "Phase 2 / Phase 3" subsection naming. Strip `[REVISION]` comments from any published source artifact (Day-14 gate).

---

## Fix-first / fix-best-effort summary

**Fix-first (topmost SEVERE/MAJOR — block submission on these):**
- **Pass P** Day-14 PDF page-count verification. If 9pp, cut Figure 1 to appendix + minor trims. SEVERE → MOOT once verified.
- **Pass D** Add §5 paragraph naming the search-variant scope. ~5 lines.
- **Pass A** Hedge abstract overclaim. ~15 words.
- **Pass R(d)** Run GT-conditioned-reward positive control (~2 hours wall) OR add Limitations sentence.

**Fix-best-effort (MINORs — polish if time permits):**
- **Pass F / Pass J** — Optional 1-sentence "why now" addition at §1.
- **Pass O** — Strip Track A/C/D/F parenthetical from §A.4 paragraph headers; rename §D subsection title.
- **Pass K / Pass M** — Verify §A.5 anonymous URL slug is filled and supplementary URL is live before submission.

**Fix-medium (MODERATEs — recommended within Day-12/13/14 patch window):**
- **Pass G** — Add 5th Limitation on enumerated-vs-canonical-MCTS scoping.
- **Pass H** — Add 2 reward-hacking citations to §3.2.
- **Pass C** — Add R-uninformed baseline (~1.5 hours wall).
- **Pass N(a)+(c)** — Add explicit "diagnostic probe" redirection sentence at §1.1 C1 closing.
- **Pass Q** — Re-fold Step 1/2/3 from §E to body §4 (~0.15pp page cost, depending on Pass P budget).

---

*End of Axis 5 report. 18 sub-passes A through R executed. Severity tally: 0 SEVERE / 4 MAJOR / 8 MODERATE / 6 MINOR. Top three findings: page-budget verification (Pass P), MCTS-vs-enumeration scope (Pass D), missing positive control (Pass R(d)).*
