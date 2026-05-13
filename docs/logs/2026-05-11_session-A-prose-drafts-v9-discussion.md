# Session A Prose Drafts v9 — §7 Discussion — 2026-05-11

**Author:** Session A (Opus, strategic)
**Audience:** Session B (Opus, tactical TeX edits) — Day 3 of ARR-cycle execution plan
**Status:** Day-2 prose draft for the Option D+ pivot. Covers two §7 subsections: §7.1 Implications (expanded to monitor + reward design) and §7.5 Future Work (split into monitor-design + reward-design Phase B research lines).

## Pivot summary

Under Option α+, §7.1 framed monitor-design implications only (single-axis pathology — null-space evasion under deception). Under Option D+, the unified Dimensional Escape framing (§3.2) covers BOTH monitor-driven and reward-driven manifestations as sub-cases of the same geometric pathology. §7.1 should reflect this symmetry: the design implications cut both ways (monitor design AND reward design).

§7.5 Future Work under Option α+ covered Phase B as a single bag (gpt-oss-20b scaling, sparse accumulators, Croissant). Under Option D+ the Phase B research program splits cleanly into two complementary tracks: (a) monitor-design research (testing whether monitor-driven OE holds under inference-time KV-cache perturbation per Hypothesis 1) and (b) reward-design research (anti-Goodhart reward designs per Hypothesis 2). Both are testable on the same Reversible MCTS infrastructure.

## Replacement TeX for §7.1

**TeX target:** Canonical TeX lines 522-527 (§7.1 subsection: REVISION marker + heading + two paragraphs).

**REVISION marker:** `[REVISION | §7.1-Implications-recast-D+ | 2026-05-11 | Retitled and expanded §7.1 from "Implications for Latent-Space Monitor Design" (Option α+, single-axis) to "Implications for Latent-Space Monitor and Reward Design" (Option D+, unified). Original two paragraphs on monitor-design preserved with slight tightening to reference Definition~\ref{def:dim_escape}; new third paragraph adds the parallel reward-design implication (Track F as evidence that surrogate-reward MCTS faces the same Dimensional Escape pathology); new fourth paragraph states the unifying design lesson (any low-rank surrogate guidance signal in high-dimensional latent space risks Dimensional Escape; the design response is the same for both — increase rank or anchor in exogenous signal). The 2026-05-05 §7.1-monitor-design-verbs marker is subsumed; the white-box / interpretability-tool framing it introduced is preserved. — pending audit]`

**Replacement body** (replaces TeX:522-527 verbatim — the REVISION marker line + `\subsection{Implications for Latent-Space Monitor Design}` + the two prose paragraphs):

```latex
% [REVISION | §7.1-Implications-recast-D+ | 2026-05-11 | Retitled and expanded §7.1 from "Implications for Latent-Space Monitor Design" (Option α+, single-axis) to "Implications for Latent-Space Monitor and Reward Design" (Option D+, unified). Original two paragraphs on monitor-design preserved with slight tightening to reference Definition~\ref{def:dim_escape}; new third paragraph adds the parallel reward-design implication (Track F as evidence that surrogate-reward MCTS faces the same Dimensional Escape pathology); new fourth paragraph states the unifying design lesson. The 2026-05-05 §7.1-monitor-design-verbs marker is subsumed. — pending audit]
\subsection{Implications for Latent-Space Monitor and Reward Design}\label{sec:discussion-implications}

The empirical evidence assembled in §\ref{sec:experiments} surfaces design implications that cut across two seemingly distinct activities: building monitors that detect deception, and building rewards that incentivize correct reasoning. Under the unified framing of Definition~\ref{def:dim_escape}, both activities involve guiding a search through high-dimensional cache geometry using a low-rank scoring projection, and both inherit the same vulnerability to Dimensional Escape.

\paragraph{Monitor design.} If alpha-sweep characterization surfaces systematic activation redistribution into the null space of a fixed linear monitor, the implication for monitor design is direct: a single static probe direction does not bound the space a model's representations may occupy under intervention. The dimensionality of continuous representation spaces provides a large reservoir of directions orthogonal to any one chosen monitor axis, and a white-box tool that operates inside the frozen model can map this reservoir directly---surfacing where a fixed monitor would produce false negatives without requiring the model to be deceptive in deployment. The correct frame for these findings is monitor-design diagnostics rather than attack capability: Reversible KV-Cache MCTS requires white-box access to a frozen model and is therefore unavailable to a real adversary at inference time. Its value is interpretability and controlled-access red-teaming. The forward-looking design implication is that monitor selection should be evaluated against the directions a frozen model's hidden states actually visit under perturbation, not only against the directions a probe was trained on. The framework presented here provides the computational infrastructure for such evaluation; the same algorithm, paired with an adaptive-probe reward, would support monitor-selection procedures that update the monitored subspace in response to observed redistribution patterns surfaced by the search.

\paragraph{Reward design.} The reward-driven manifestation of Dimensional Escape---Hypothesis~\ref{hyp:goodhart}, empirically established in §\ref{sec:exp1}---carries a parallel implication for the design of test-time scaling procedures that operate on continuous latent representations. Track F demonstrates that a reward signal validated as a static correlate of correctness ($|r| \approx 0.60$ between next-token entropy and ARC-Easy answer correctness, Track~A) does not survive deployment as an active MCTS optimization objective: at 1B the search converges on cache states that amplify the model's first-letter prior (predicting ``A'' on 185 of 200 items under heavy steering); at 3B the search collapses to greedy decoding (minimal-magnitude steering on $51\%$ of items). Neither pathway corresponds to ``find the latent state that produces the correct answer''---both are degenerate optima that satisfy the entropy proxy without traversing the semantically relevant subspace. The design lesson is symmetric to the monitor-design lesson: a single low-rank surrogate scoring function does not bound the cache states an MCTS search can reach, and the unconstrained complement provides an ample reservoir for the search to exploit. Anti-Goodhart reward designs surveyed in the Phase B research program (§\ref{sec:future-work})---multi-channel ensembles, external Process Reward Models, self-consistency across stochastic rollouts, counterfactual sensitivity mapping, anti-prior penalties, information-bottleneck objectives---all attempt to raise the effective rank of the scoring projection or to anchor it in exogenous signal that the search cannot itself rewrite. Whether any of these resists Dimensional Escape at the empirical level is an open question; our Track F result establishes the null-control (entropy minimization fails) against which any candidate must be measured.

\paragraph{The unifying design lesson.} The convergence of monitor design and reward design under Definition~\ref{def:dim_escape} suggests a single methodological prescription. Any low-rank scoring projection deployed against a high-dimensional cache---whether as a corrective monitor, a search reward, or a representational probe---creates an unconstrained complement large enough to admit degenerate optima. The design response, in either application, is the same: either raise the effective rank of the scoring projection (e.g., multi-channel ensembles, attention-aggregated probes) or ground the scoring in signal that the searched representation cannot trivially rewrite (e.g., exogenous answer verifiers, ground-truth-matched reward functions). The relative cost-effectiveness of these two response classes is itself an open empirical question that the Reversible MCTS framework is positioned to address.
```

## Replacement TeX for §7.5 Future Work

**TeX target:** Canonical TeX lines 567-576 (§7.5 subsection: heading + CRITIQUE NOTE comment + single prose paragraph + two REVISION markers).

**REVISION marker:** `[REVISION | §7.5-FutureWork-recast-D+ | 2026-05-11 | Expanded §7.5 Future Work from single-bag Phase B framing (Option α+: gpt-oss-20b scaling + sparse accumulators + Croissant) to two complementary research tracks under Option D+ (monitor-design research per Hypothesis 1; reward-design / anti-Goodhart research per Hypothesis 2). The single existing prose paragraph is retained as the "infrastructure-and-scaling" track and slightly tightened; a new paragraph adds the anti-Goodhart reward-design research program (the 9 candidates from Track F §9 + Gemini round 4 Part 2 survey); a third new paragraph adds the Latent Cartography extension program (additional prompt classes, mechanistic interpretability with Cartography heatmaps). The 2026-04-24 §7-FutureWork and 2026-04-25 §7-Croissant markers are subsumed. CRITIQUE NOTE comment is preserved (Future Work as standalone subsection is acceptable at ACL/EMNLP venues; Tianyu Shi sign-off Day 11 final verification). New `\label{sec:future-work}` enables §7.1's forward-reference. — pending audit]`

**Replacement body** (replaces TeX:567-576 verbatim — the heading + CRITIQUE NOTE comment + the existing prose paragraph + the two trailing REVISION markers):

```latex
\subsection{Future Work}\label{sec:future-work}
% [CRITIQUE NOTE | Future Work section — venue convention check]
% Future Work as a standalone subsection is acceptable at ACL/EMNLP venues; Tianyu Shi
% sign-off (Day 11 of execution plan) provides final verification.

% [REVISION | §7.5-FutureWork-recast-D+ | 2026-05-11 | Expanded §7.5 from single-bag Phase B framing to two complementary research tracks under Option D+. The 2026-04-24 §7-FutureWork and 2026-04-25 §7-Croissant markers are subsumed. — pending audit]

\paragraph{Infrastructure and scaling.} Several directions are deferred to Phase B research. Cross-model steering-vector transfer via Procrustes alignment between model families (e.g., Llama 1B $\leftrightarrow$ Llama 7B, Llama 3.2 $\leftrightarrow$ Mistral 7B) remains an open research question with uncertain guarantees. Scaling to multi-GPU configurations is required for $M_{\text{KV}} \ge 20$ GB regimes: our measured working set of $\approx 162$ GB at 20B-parameter configurations exceeds a single 80 GB H100 and necessitates tensor-parallel sharding. Full sparse FP32 accumulator implementation---reducing per-step overhead from $O(S \cdot d_{\text{model}})$ to $O(S' \cdot d_{\text{model}})$ for $S' \ll S$ mutated positions---would extend the practical search-depth envelope substantially beyond the $d \le 10$ regime we validate empirically. A Croissant~1.1 dataset package of runtime evidence artifacts is included as a supplementary release alongside this submission and will be extended in Phase B to include cross-model and cross-family sweeps.

\paragraph{Monitor-design research (Hypothesis~\ref{hyp:orthogonal_escape}).} The monitor-driven case of Dimensional Escape under inference-time KV-cache perturbation is left to Phase B for empirical characterization. The protocol is direct: extend the Latent Cartography sweep design (§\ref{sec:exp2}) to use a calibrated honesty-probe direction rather than a random direction, measure OEI and TDS across the alpha grid, and compare against Bailey et al.'s~\cite{bailey2024obfuscated} gradient-trained input-space negative result. If inference-time KV-cache perturbation can drive a model's representations into the null space of a linear honesty probe while preserving behavioral outputs in the safe set $\mathcal{S}$ (Eq.~\ref{eq:alignment_faking}), monitor-driven Dimensional Escape is empirically demonstrated; if not, the inference-time regime joins Bailey's gradient-trained input regime as a negative-result case for the geometric pathology.

\paragraph{Reward-design research (Hypothesis~\ref{hyp:goodhart}).} A second Phase B research track explores anti-Goodhart reward designs that may resist the reward-driven Dimensional Escape established empirically by Track F. Candidates surveyed in the literature and in our internal analysis include: multi-channel ensemble agreement (simultaneous satisfaction of $\sigma_H$, $\rho_R$, entropy, and output perplexity); external Process Reward Model (PRM) scoring of intermediate reasoning steps to anchor the reward in signal the searched cache cannot rewrite; self-consistency across stochastic high-temperature rollouts; counterfactual sensitivity mapping (rewarding cache states whose outputs are stable under opposite-direction perturbations); anti-prior penalty (subtracting a penalty proportional to the amplification of the model's pre-identified prior); information-bottleneck rewards (mutual information between prompt and answer modulated through the perturbed cache); hybrid Best-of-$N$ plus MCTS (combining outcome-verifier safety with continuous-latent search); and activation-norm regularization (constraining perturbation magnitudes to remain near the natural data manifold). Each candidate raises the effective rank of the scoring projection or anchors it in exogenous signal; whether any resists Dimensional Escape at the empirical level is the central question of this research program. The Reversible MCTS infrastructure presented here supports direct comparison: Track F's controlled negative-control protocol provides the null-control against which any candidate must be measured.

\paragraph{Cartography extensions.} The Latent Cartography sweeps (§\ref{sec:exp2}) characterize structural pre-training priors across four prompt classes at six steering magnitudes. Natural extensions include additional prompt classes (adversarial probes, multilingual inputs, code-generation tasks, multi-turn chain-of-thought reasoning), per-class mechanistic interpretability analysis of the terminal-token distributions the search converges upon, and cross-model Cartography to test whether the class-dependent prior amplification pattern is family-specific or architecture-universal.
```

## Notes for Session B

**Length impact:**
- §7.1 expansion: ~570 words of new prose (third + fourth paragraph; the first two paragraphs are slight tightening of existing). Net +400 words vs. existing §7.1.
- §7.5 expansion: ~480 words of new prose (paragraphs 2, 3, 4; paragraph 1 is tightening of existing). Net +320 words vs. existing §7.5.
- Combined §7 Discussion delta: ~+720 words. Together with §7.3 Limitations (+410 words) and §3 (+280 words) and §1 Contributions (+95 words), the cumulative Option D+ delta is approximately +1500 words. EMNLP long-paper budget is 8 pages; current paper is ~9-10 dense pages including bibliography. Day 10 page-count check (per ARR-cycle plan v2 §2) is critical.

**Cross-references that must resolve:**
- `\ref{def:dim_escape}` — exists at TeX:145 ✓
- `\ref{hyp:orthogonal_escape}` — exists at TeX:151 ✓
- `\ref{hyp:goodhart}` — exists at TeX:165 ✓
- `\ref{eq:alignment_faking}` — exists at TeX:162 ✓
- `\ref{sec:experiments}` — exists ✓
- `\ref{sec:exp1}`, `\ref{sec:exp2}` — Day 7 §5 rewrite adds these labels; will produce undefined-reference warnings on intermediate compilation
- `\ref{sec:future-work}` — NEW label added in this draft (§7.5 heading); §7.1 forward-references it
- `\ref{sec:discussion-implications}` — NEW label added in this draft (§7.1 heading); not yet referenced anywhere
- `\cite{bailey2024obfuscated}` — exists ✓

**Numbers cited (verbatim from track reports, all verified):**
- "$|r| \approx 0.60$ between next-token entropy and ARC-Easy answer correctness" — Track A §4 (1B hard 0.602) / §5b (3B ultra 0.614)
- "predicting ``A'' on 185 of 200 items under heavy steering" — Track F §1 + §7 mechanism diagnosis (1B)
- "minimal-magnitude steering on $51\%$ of items" — Track F §1 + §7 (3B: 101/200 = 50.5%, rounded to 51%)
- "$\approx 162$ GB at 20B-parameter configurations" — Track D §3 + §4 replacement-paragraph

**Section-cross-coordination:**
- §7.1 fourth paragraph references "raise the effective rank" — same conceptual framing used in §7.5 reward-design paragraph and in §3.2 hypothesis statements. Consistent terminology.
- §7.5 monitor-design paragraph forward-references §\ref{sec:exp2}; Cartography extensions paragraph also references §\ref{sec:exp2}. These resolve Day 7.
- §7.5 reward-design paragraph references "Track F's controlled negative-control protocol" — the §\ref{sec:exp1} resolves Day 7.

**Day-3 application order** (Session B Day 3 in ARR-cycle plan v2):
1. Apply §7.1 Implications recast (TeX:522-527 → new content)
2. Apply §7.5 Future Work recast (TeX:567-576 → new content)
3. Verify cross-references resolve (or are documented as Day-7-pending)

**License attribution paragraph (separate Day-2 Session A draft, depends on Josh's J2 response):**
The Llama 3.2 attribution string needs to land in §7.6 Ethical Considerations (TeX:580). This is NOT included in this Discussion draft because it's a single-paragraph insertion that depends on Josh's verbatim J2 response. Separate file: `docs/logs/2026-05-11_session-A-prose-drafts-v9-license-attribution.md` — DEFERRED until J2 lands.

---

*Session A v9 §7 Discussion draft complete. Apply Day 3 of ARR-cycle execution plan v2. The license-attribution paragraph (depends on J2) and Responsible NLP Checklist draft (depends on J3) are deferred until Josh's verification responses land in the session log.*
