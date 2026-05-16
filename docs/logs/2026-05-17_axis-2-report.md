# Axis 2 Report — Dimensional Escape Framing Defensibility

**Reviewer scope.** §1 + §1.1 + §3 (esp. §3.2) + §5 Exp 1 + §5 Exp 2 + §7
Discussion of the canonical TeX
(`docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`, 634 lines,
Day-11 post-hedge state). Primary corroborating sources: Track F report
(`docs/logs/2026-05-11_track-F-negative-control-report.md`), Track G
report (`docs/logs/2026-05-11_track-G-cartography-report.md`), Gemini
Round 4 Part 5 sketch (`docs/logs/2026-05-09_Latent-Space MCTS_ Reward
Design Strategy.md`).

**Headline verdict.** Day-11 hedge fixes (lines 54, 72, 80) have already
softened the most egregious overclaims at abstract/§1/§1.1 level
("structurally identical" → "structurally analogous"; "both are
sub-cases" → "we conjecture both as sub-cases"). The unification framing
is now defensible at the §1 perimeter. **The remaining attack surface is
inside §3.2 itself**: Definition `def:dim_escape` (line 121-122) and the
§3.2 closing paragraph (line 142) still present the unification as
established by the empirical data when the data evidences only H2. The
Definition is also not operationally testable as written (Q1), and the
"Dimensional Escape vs Goodhart" distinction (Q5) is not articulated in
the body at all — a reviewer who asks "what does this term add over
Goodhart's Law" cannot get an answer from the paper. Q4 ("geometric"
framing) flags a softer but real overclaim: Track F + Track G evidence
correlation-decoupling between cache-space and answer-space metrics, not
high-dimensional manifold structure. Severity tally: 0 SEVERE, 2 MAJOR
(Q3 unification gap inside §3.2; Q5 Goodhart-distinction gap), 3
MODERATE (Q1 operational testability; Q4 geometric overclaim; Q7
missing limitation), 2 MINOR (Q2 wording; Q6 sentence pick).

---

## Q1. Operational testability of the Dimensional Escape definition (§3.2)

**(a) Verbatim passage.** Lines 121-123:

> "**Definition (Dimensional Escape).** Let $\mathbf{P} \in
> \mathbb{R}^{k \times d}$ be a monitoring or scoring projection of
> rank $k \ll d$, and let $\mathcal{L}: \mathbb{R}^d \to \mathbb{R}$
> be a localized optimization objective acting on hidden state
> $\mathbf{h}$. *Dimensional Escape* occurs when the optimization
> process resolves $\mathcal{L}$ through trajectories whose induced
> perturbation $\Delta\mathbf{h}$ has substantial component in
> $\text{Null}(\mathbf{P}_{\text{semantic}})$, where
> $\mathbf{P}_{\text{semantic}}$ is the (typically unobserved)
> projection onto the semantically relevant subspace of
> $\mathbb{R}^d$. The optimizer thus satisfies $\mathcal{L}$ without
> traversing the manifold of semantically meaningful states."

**(b) Data the passage rests on.** No direct measurement: Track F report
§7 mechanism diagnosis evidences correlation-decoupling (entropy↔correctness
breaks under cache perturbation), not "trajectories with substantial
component in $\text{Null}(\mathbf{P}_{\text{semantic}})$." Track G §2.3
gold-rank preservation finding ("the model 'knows' the answer but
predicts the wrong (prior) token at argmax") is the closest experimental
signature, but it is reported as a top-5 rank statistic, not as a
projection-onto-null-space measurement.

**(c) Verdict: HEDGE-NEEDED (MODERATE).** The Definition is
mathematically precise but operationally untestable as written, because
$\mathbf{P}_{\text{semantic}}$ is explicitly labeled "(typically
unobserved)". A reviewer asking "what would falsify Dimensional Escape
on a new model?" gets no actionable answer. The Definition also does not
state a quantitative threshold for "substantial component" — is 51% in
null-space sufficient? 90%? The OEI (Eq. 10, appendix) supplies a
candidate operational metric but is no longer in body §3.2 after the
Day-10 drop-C2 surgery.

**(d) Suggested rewrite.** Append one sentence to the Definition:

> "We operationalize Dimensional Escape via two observational
> signatures: (i) decoupling — the optimizer increases its score on
> $\mathcal{L}$ while task accuracy does not improve relative to a
> reward-matched random control (Experiment 1 falsification criterion:
> $|\Delta_{E-R}| < 2$pp at $p \geq 0.05$); (ii) argmax-prior
> amplification — the steered top-1 token concentrates on a prompt-class
> prior rather than the gold answer, while gold-rank is preserved in
> top-$k$ (Experiment 2 falsification: argmax collapse $\leq$ baseline
> ratio AND median gold-rank $> k$). A direct measurement of the
> null-space component requires access to $\mathbf{P}_{\text{semantic}}$
> and is left to Phase B research."

**(e) Severity: MODERATE** — reviewer will request operational
falsifiability; current text exposes the paper to "what would convince
you you're wrong?" pushback.

---

## Q2. Track F evidence for Hypothesis 2 — strong or weak claim?

**(a) Verbatim passage.** Lines 132-138:

> "**Hypothesis (Reward-driven Dimensional Escape: Goodhart-collapse
> under surrogate-reward MCTS).** When MCTS optimizes a surrogate
> reward $\hat{r}: \mathbb{R}^d \to \mathbb{R}$ lacking exogenous
> grounding [...] the optimal cache state under $\hat{r}$ generically
> lies off the semantic manifold:
> $\arg\min_{\mathbf{h}} \hat{r}(\mathbf{h}) \cap
> \text{SemanticManifold}(x) = \emptyset$ (generically, in dimension
> $d \gg \text{rank}(\hat{r})$)"

**(b) Data the passage rests on.** Track F §3 headline statistical
claim: "Δ accuracy (E − R) = **−0.005** at 1B and **+0.005** at 3B;
McNemar p = 1.0000 at both scales." Track F §5: "1B E/G ratio of 1.35
[...] this is well inside the 2.0× threshold. At 3B the ratio is 1.06."
Track F §7 mechanism: "Track A's |Pearson r| = 0.60 between entropy and
correctness was measured on *fixed prompts that vary in difficulty*. It
was *not* measured on *fixed-difficulty items where the cache is being
perturbed in arbitrary directions*. [...] confidence becomes a function
of cache geometry rather than prompt difficulty."

**(c) Verdict: HEDGE-NEEDED (MINOR/MODERATE).** Track F evidences a
weaker statement than H2 as written. The data shows:
1. Entropy-MCTS does not beat random-MCTS (decoupling of cache-space
   metric from answer-space metric);
2. Perplexity stays bounded ($E/G \leq 1.35$ at 1B; 1.06 at 3B), meaning
   the searched states are NOT off the on-distribution manifold under
   the unsteered model.

The second point directly tensions with the H2 claim that the optimal
cache state "generically lies off the semantic manifold." If "semantic
manifold" means "on-distribution under the unsteered model" then Track F
§5 G3.3 PASS refutes H2; if it means "on the answer-correctness
sub-manifold" then H2 holds but the Definition's notion of
$\mathbf{P}_{\text{semantic}}$ collapses into "the projection onto the
correct-answer logit" which is degenerate. The paper does not
disambiguate.

The accurate framing is: **the entropy gradient is decoupled from the
answer gradient** (cache states with lower entropy under the steered
model are not states with higher correctness under the unsteered
model). This is a statistical-correlation claim, not a geometric-manifold
claim.

**(d) Suggested rewrite.** Append clarifier to Hypothesis 2:

> "Here SemanticManifold$(x)$ denotes the set of cache states for which
> the unsteered model's argmax matches the gold answer; under-the-base-
> model perplexity of the steered completion remains bounded (Experiment 1,
> $E/G \leq 1.35$), so the failure is decoupling of the surrogate's
> gradient from the correctness gradient rather than off-distribution
> drift."

**(e) Severity: MINOR** — the framing arc still holds, but a careful
reviewer will catch the manifold-vs-distribution conflation. Cheap fix.

---

## Q3. Cross-experiment unification — H1 evidence audit

**(a) Verbatim passages.**

§1.1 Contribution 2 (line 84, post-Day-11 hedge):

> "We further argue the reward-driven pathology surfaced here is
> structurally analogous to the monitor-driven *Orthogonal Escape*
> previously hypothesized for alignment faking [bailey2024obfuscated]:
> we conjecture both as sub-cases of the same overparameterized-latent-
> space geometry (§ref:sec:dim_escape)."

§3.2 closing (line 142):

> "Bailey et al. [bailey2024obfuscated] characterized the monitor-
> driven case (gradient-trained input-space attacks) with a tentative
> negative result; we characterize the reward-driven case empirically
> (§ref:sec:exp1--ref:sec:exp2) and find it manifests robustly across
> model scale and prompt class. The monitor-driven case under
> inference-time KV-cache perturbation is left to Phase~B research."

§3.2 opening (line 119):

> "We argue these are sub-cases of a single underlying pathology."

**(b) Data the passage rests on.** Track F + Track G evidence H2 only.
Bailey 2024 evidences a *negative result* in a *different experimental
regime*: gradient-trained input-space attacks against frozen models, not
inference-time KV-cache mutation. Per §3.2 line 119, Bailey is "the
subject of [bailey2024obfuscated] (gradient-trained input-space variant,
with a tentative negative result reported)". §2.2 line 102: "[Bailey's]
work is complementary along three axes: (i) the intervention is applied
to cached key/value tensors at inference time rather than to input
embeddings during gradient training; (ii) the search optimizes a
surrogate reward [...]; (iii) [...] leaving open whether the inference-
time, frozen-weight, surrogate-reward setting exhibits the same
geometric constraint."

The §2.2 framing already acknowledges that Bailey's regime is
fundamentally different from ours along three explicit axes, *and that
the question of whether the constraint generalizes is open*. Yet §3.2
line 119 still asserts "We argue these are sub-cases of a single
underlying pathology" as if it were a synthesis of established facts.

**(c) Verdict: OVERCLAIM (MAJOR), partially mitigated by Day-11 hedges
at §1 perimeter.** The Day-11 fixes have softened the abstract (line 54)
and §1.1 (line 84) and §1 closing (line 73) to "structurally analogous"
+ "we argue" + "we conjecture". But **the body §3.2 itself still
contains the unhedged "We argue these are sub-cases of a single
underlying pathology" (line 119), and the §3.2 closing (line 142) reads
as if the unification is established**. A reviewer reading §3.2 in
isolation gets the pre-hedge version of the claim.

Furthermore — and this is the strengthened-Q3 finding — **Contribution 2
in the §1.1 enumeration is titled "Empirical demonstration of
Dimensional Escape under continuous-latent search"** (line 84,
emphasis added). The use of the bare term "Dimensional Escape" (which
§3.2 defines as the *unified* H1+H2 pathology) in a contribution title
that only evidences H2 is itself a framing tension. The Day-11 hedge
softens the unification sentence but does not adjust the contribution
title.

**(d) Suggested rewrite.** Two complementary fixes.

Fix 1 (§1.1 Contribution 2 title rewrite — adopting prompt option (a)):

> "**Empirical demonstration of reward-driven Dimensional Escape;
> structural-equivalence argument for monitor-driven case.** We
> establish the reward-driven sub-case of a geometric pathology of
> MCTS in overparameterized latent spaces [...]"

Fix 2 (§3.2 line 119 + line 142 — adopting prompt option (b)):

Line 119 current: "We argue these are sub-cases of a single underlying
pathology."

Rewrite: "We hypothesize these as parallel manifestations of the
overparameterized-latent-space geometry described below; only the
reward-driven case is evidenced empirically in this paper, with the
monitor-driven case under inference-time KV-cache perturbation an open
empirical question (§ref:sec:future-work)."

Line 142 needs no further change beyond the existing Day-10 hedge; it
already concedes Phase B.

**(e) Severity: MAJOR** — this is the framing-defensibility hinge of
the paper. The Day-11 hedges fixed the abstract and §1 but left §3.2
body and the Contribution 2 title still presenting unification as
established. Reviewer will circle "We argue these are sub-cases of a
single underlying pathology" (line 119) and ask: what data evidences
H1? The honest answer is "none in this paper; we cite Bailey 2024 which
is a different regime and reports a negative result." That answer must
be in the body, not deduced from the cumulative hedge structure.

---

## Q4. The "geometric" framing — is the failure mechanism actually geometric?

**(a) Verbatim passage.** Abstract line 56:

> "[...] we [...] expose a fundamental geometric pathology we term
> **Dimensional Escape**: under any surrogate reward lacking exogenous
> grounding, the high-dimensional cache resolves localized optimization
> pressure by routing representations through unconstrained orthogonal
> dimensions, satisfying the proxy metric without satisfying the
> semantic objective."

§3.2 line 125:

> "The high dimensionality of modern Transformer hidden states
> ($d \in [2048, 8192]$ for the Llama~3 family) ensures that for any
> low-rank monitor or scoring function, the unconstrained complement
> is overwhelmingly large; localized optimization pressure routes
> through this complement, whether applied as corrective signal or as
> reward signal."

**(b) Data the passage rests on.** Track F §7 mechanism: prior
amplification at 1B (entropy-MCTS picks $(1.0,1.0,1.0)$ on 91/200 items;
predicts "A" on 185/200); greedy-collapse at 3B (entropy-MCTS picks
$(0.1,0.1,0.1)$ on 101/200; outputs match greedy ± small noise). Track G
§2.4: "search picks whichever α minimizes entropy under the model's
per-class baseline distribution." Track G §4.2 entropy trajectory: 1B
C2 ARC-Easy MCQ shows entropy minimum at α≈1.0 in the parameter sweep,
which is exactly where MCTS converges.

**(c) Verdict: HEDGE-NEEDED (MODERATE).** The Track F + Track G data
support the claim that **the cache-space entropy-minimizing trajectory
does not coincide with the answer-correctness-maximizing trajectory**.
This is fundamentally a statistical-correlation claim
(cache→entropy↔correctness decoupling), not a high-dimensional manifold
claim. The "geometric" framing — "routing through unconstrained
orthogonal dimensions" — connotes that one could measure the angle
between the steered trajectory and a putative semantic manifold; the
paper never measures such an angle. What it measures is:

1. Cache perturbation magnitude (mean-step α);
2. Steered top-1 token argmax;
3. Steered top-1 token entropy.

None of these are directly geometric measurements of trajectories in
hidden-state space. The §3.2 line 125 argument ("for any low-rank
monitor, the unconstrained complement is overwhelmingly large") is a
*combinatorial* argument about dimension counting, not a geometric
argument about manifold structure. A reviewer with a geometry background
will note that "routing through unconstrained dimensions" requires
demonstrating that $\Delta\mathbf{h}$ has measurable projection onto
$\text{Null}(\mathbf{P}_{\text{semantic}})$ — and that the paper does
not produce that measurement.

**(d) Suggested rewrite.** Soften "geometric" to "structural" or
"high-dimensional" in two places:

Abstract line 56 — "a fundamental geometric pathology" → "a structural
pathology of high-dimensional cache search". The §3.2 Definition can
keep its geometric phrasing (it is defining the formal notion), but the
empirical claim in the abstract and §1 should be downgraded.

§3.2 line 125 — current "localized optimization pressure routes through
this complement" → "the optimal cache state under any low-rank surrogate
generically lies outside the answer-correctness sub-manifold by simple
dimension-counting; the empirical manifestation is decoupling of the
surrogate's optimization gradient from the correctness gradient
(Experiment 1, §exp1)."

**(e) Severity: MODERATE** — framing-overhead question that a hostile
reviewer will hit. The Track F + Track G data are honestly described
elsewhere as correlation-decoupling; the "geometric" elevation happens
at the §1 + §3.2 framing layer.

---

## Q5. Goodhart's Law vs Dimensional Escape — distinct or relabeling?

**(a) Verbatim passages.** §3.2 opening (line 119):

> "This is Goodhart's Law manifesting in cache geometry: when a measure
> becomes a target, the optimizer discovers the cheapest computational
> path to alter the metric, even if that path circumvents the original
> objective."

Hypothesis 2 title (line 132): "**Reward-driven Dimensional Escape:
Goodhart-collapse under surrogate-reward MCTS**". The two terms are
used in synonymous-or-adjacent ways throughout §3, §5, §7.

§7.1 line 329:

> "The reward-driven manifestation of Dimensional Escape---Hypothesis~2,
> empirically established in §exp1---carries a direct implication [...]
> A reward signal validated as a static correlate of correctness
> ($|r| \approx 0.60$ between next-token entropy and ARC-Easy answer
> correctness) does not survive deployment as an active MCTS optimization
> objective [...]"

**(b) Data the passage rests on.** Gemini Round 4 Part 5 (line 175 of
strategy doc) provides the intended distinction:

> "the broader framing should be rigorously defined as **Dimensional
> Escape**: the fundamental tendency of overparameterized latent spaces
> to resolve localized optimization pressure by routing representations
> through unconstrained orthogonal dimensions. Goal-directed deception
> (evading a monitor while maintaining behavior) is simply a specialized
> sub-case of this broader mathematical pathology. Reward hacking
> (minimizing entropy by collapsing to a pre-training prior) is the
> second sub-case."

The intended distinction is therefore: **Goodhart is a statement about
proxy optimization without geometric content; Dimensional Escape adds
the geometric claim that optimization is resolved through unconstrained
dimensions of an overparameterized latent space.** This distinction is
not present in the body of the paper. §3.2 line 119 says "This is
Goodhart's Law manifesting in cache geometry" — explicitly identifying
the two — and then the Definition is given without ever asserting that
"Dimensional Escape adds something Goodhart's Law doesn't capture."

**(c) Verdict: OVERCLAIM (MAJOR).** The paper coins a term and never
defends its distinctness from a well-known prior concept. A reviewer
will ask: what does "Dimensional Escape" capture that "Goodhart's Law
in high-dimensional latent spaces" does not? Per the prompt's
strengthened Q5 instruction, the paper's value-add of introducing a new
term must be defensible; the current text fails that test.

**(d) Suggested rewrite.** Adopting prompt option (b) — sharpen §3.2
Definition `def:dim_escape`. Add one preamble sentence before the
Definition:

> "Dimensional Escape differs from generic Goodhart's Law in
> specifying the *geometric route* by which the proxy is resolved:
> Goodhart's Law states that the proxy will be over-optimized at the
> expense of the true objective; Dimensional Escape predicts that, in
> overparameterized latent spaces, the optimizer resolves the proxy
> via trajectories in the null space of the scoring projection rather
> than through trajectories that traverse the answer-relevant
> subspace. This geometric specificity makes empirical signatures
> available (per-class argmax-prior amplification, gold-rank
> preservation under argmax collapse) that pure Goodhart-Law framing
> does not predict."

This explicitly answers the "is it a relabeling?" question. If the
authors cannot defend the geometric-route claim empirically (cf. Q4),
the alternative is prompt option (a): retire the term and re-anchor on
"Goodhart's Law in high-dimensional latent spaces". This would save the
coined-term defense overhead but require global rewrite of §3.2 title
+ Hypothesis 2 title + abstract + §1.1 Contribution 2 + §7 Discussion.
Recommendation: **option (b) is lower-cost** given the post-Day-11 hedge
state of the paper.

**(e) Severity: MAJOR** — coined-term defense is a standard reviewer
attack vector. Without the distinction articulated in body, the paper
exposes itself to "this is Goodhart's Law with extra steps" rejection.

---

## Q6. Most-overclaim sentence in §3 + §7

**(a) Verbatim passage.** §3.2 line 119:

> "We argue these are sub-cases of a single underlying pathology."

(This is the unhedged claim that survived Day-11 inside the body, after
the abstract / §1 / §1.1 sentences were softened.)

Honorable mention from §7.1 line 329:

> "Anti-Goodhart reward designs surveyed in §future-work---multi-channel
> ensembles, external Process Reward Models, self-consistency across
> stochastic rollouts, counterfactual sensitivity, anti-prior penalties,
> information-bottleneck objectives---all attempt to raise the effective
> rank of the scoring projection or to anchor it in exogenous signal
> that the search cannot itself rewrite. **Whether any resists
> Dimensional Escape empirically is an open question; our Experiment~1
> result establishes the null-control against which any candidate must
> be measured.**"

The §7.1 sentence is internally hedged; the §3.2 line 119 sentence is
not.

**(b) Data the passage rests on.** None for H1; Track F + Track G for
H2 (see Q3).

**(c) Verdict: OVERCLAIM (MAJOR, but subsumed by Q3 fix).** Same finding
as Q3 in different words.

**(d) Suggested rewrite.** Per Q3 Fix 2:

> "We hypothesize these as parallel manifestations of the
> overparameterized-latent-space geometry described below; only the
> reward-driven case is evidenced empirically in this paper, with the
> monitor-driven case under inference-time KV-cache perturbation an
> open empirical question (§future-work)."

**(e) Severity: MAJOR (= Q3 severity).**

---

## Q7. Strongest critique §7.3 does not pre-empt

**(a) Verbatim passage (§7.3 limitations enumerated).** §7.3 has
exactly five limitations after the Day-10 Batch 3 + Day-10 Batch 5
trims:
1. Llama 3.2 single family (lines 341);
2. Letter-prompted MCQ scoring under-states accuracy ~20 pp (line 343);
3. Methodological scope along three dimensions: single steering
   direction, 4-prompt-class Cartography, 200-cycle reversibility bound
   (line 346);
4. Procrustes + sub-7B probe-validation (line 349). (The original 8
   limitations have been trimmed.)

**(b) Data the passage rests on.** Track F §10 protocol deviations
(steering direction, search-policy substitution); Track G §6
success-criteria evaluation; Track F §3 statistical-power calculation.

**(c) Verdict: MISSING LIMITATION (MODERATE).** The strongest critique
the paper does not acknowledge:

> **The entropy-reward formulation is uniquely susceptible to
> prior-amplification by construction.** Track F §7 (mechanism
> diagnosis) explicitly states: "an entropy-minimising objective on a
> frozen model's KV cache reinforces the model's existing priors
> regardless of input semantics." Yet the paper generalizes from this
> *single reward function's failure* to a **claim about all surrogate
> rewards** ("**any** surrogate reward lacking exogenous grounding" —
> abstract line 56; "surrogate rewards lacking exogenous grounding
> inevitably resolve localized optimization pressure" — §1.1
> Contribution 2). Track F's mechanism is mathematically tied to
> entropy: minimizing $-H/\log|V|$ trivially reduces to maximizing the
> prior's mass (this is what Track F §7 means by "a search direction
> toward whatever the model is already biased toward producing"). A
> different surrogate reward — for instance, a reward function trained
> to predict task accuracy from cache features — would not have the
> same trivial prior-collapse failure mode.

The paper's case for generalization beyond entropy rests on the §7.1
+ §future-work argument that anti-Goodhart designs (multi-channel
ensembles, external PRMs, self-consistency, counterfactual sensitivity,
etc.) are *also expected to fail* — but **the paper provides zero
empirical evidence for any reward function other than entropy**. A
reviewer will write: "the negative result is for entropy, not for
surrogate rewards in general; the unification of these into 'all
surrogate rewards lacking exogenous grounding' is a theoretical
extrapolation, not an empirical finding."

This critique is partially pre-empted by §7.3 item 3 (methodological
scope) — but only along the "single steering direction" + "4 prompt
classes" axes, **not along the "single reward function" axis**. The
single-reward-function limitation is the strongest version of the
"generalization" critique and is not currently enumerated.

**(d) Suggested rewrite.** Add a new limitation to §7.3 (or extend item
3):

> "**Single surrogate reward function.** Our empirical evidence is
> specific to the entropy-normalized reward $\hat{r}(\mathbf{h}) = -H/
> \log|V|$, which has a structural property — minimizing it reduces to
> maximizing the most-confident token's probability — that may not be
> shared by alternative surrogate rewards (e.g., learned cache-feature
> classifiers, attention-aggregated probes, or external Process Reward
> Models). We frame Dimensional Escape as a candidate-general pathology
> on the basis of the dimension-counting argument (§dim_escape) and the
> anti-Goodhart design analysis (§discussion-implications), but
> empirical generalization across reward functions is an open question.
> Experiment 1 establishes the null-control against which candidate
> reward functions can be benchmarked."

**(e) Severity: MODERATE** — reviewer-likely revision request; the fix
adds ~5 lines to §7.3 and substantively strengthens the paper's claim
to honest scope.

---

## Summary: Ranked framing patches needed before submission

**Required (MAJOR severity).** Without these, reviewer pushback on
framing is likely.

1. **§3.2 line 119 unification claim (Q3, Q6).** Rewrite "We argue
   these are sub-cases of a single underlying pathology" to "We
   hypothesize these as parallel manifestations [...] only the
   reward-driven case is evidenced empirically." This harmonizes §3.2
   body with the Day-11 abstract/§1/§1.1 hedges. Estimated cost: 1
   sentence rewrite.

2. **§1.1 Contribution 2 title (Q3).** Rewrite "Empirical demonstration
   of Dimensional Escape under continuous-latent search" to "Empirical
   demonstration of reward-driven Dimensional Escape; structural-
   equivalence argument for monitor-driven case." Aligns contribution
   title with the hedge structure already applied to the body of
   Contribution 2. Estimated cost: 1 title rewrite (~12 words).

3. **§3.2 Definition `def:dim_escape` — Goodhart distinction (Q5).**
   Add the one-paragraph preamble articulating the geometric-route
   distinction from generic Goodhart's Law. Without this, the coined
   term is undefended against "relabeling" pushback. Estimated cost:
   1 short paragraph before the Definition (~80 words).

**Recommended (MODERATE severity).** Strengthens defensibility but is
not catastrophic if skipped.

4. **§3.2 Definition — operational signatures (Q1).** Append the two-
   signature operationalization (decoupling + argmax-prior
   amplification). Anchors the term in measurable quantities.
   Estimated cost: 1 sentence appended to Definition (~50 words).

5. **Abstract + §3.2 line 125 — soften "geometric" to "structural" or
   "high-dimensional" (Q4).** Aligns framing with the
   correlation-decoupling evidence actually presented. Estimated cost:
   2 word swaps + ~1 sentence rewrite.

6. **§7.3 add single-reward-function limitation (Q7).** Strongest
   reviewer-side critique the paper currently fails to pre-empt.
   Estimated cost: 1 new item or extension to item 3 (~5 lines).

**Optional (MINOR severity).** Cheap quality-of-life.

7. **Hypothesis 2 clarifier on SemanticManifold notion (Q2).** Anchor
   the manifold concept to under-base-model perplexity to disambiguate
   from "on-distribution" reading. Estimated cost: 1 sentence appended
   to H2 (~40 words).

**Page-budget impact of the full Q1-Q7 patch set.** Net additions
~250-300 words; net deletions in unification-claim rewrites ~50 words;
net body-budget cost ~0.15-0.20 pp at ACL two-column. The paper is
currently at 8 pp post-Day-11 page-budget cuts; the framing-patch set
is feasible without further trim if Day-12 axis 4 (Pass O reproducibility
appendix length) does not also expand. If page-budget is binding,
patches 1+2+3 are the minimum required to address MAJOR-severity
findings.

**Overall framing-defensibility verdict.** Day-11 hedges substantially
reduced exposure at the abstract / §1 / §1.1 perimeter. The remaining
attack surface is concentrated in §3.2 body (line 119 unification
claim; missing Goodhart distinction; missing operational signatures)
and §7.3 (missing single-reward-function limitation). All four are
fixable with one round of edits totaling under 300 words. **Recommend
holding the Day-11 hedges as the baseline and applying patches 1-3
(MAJOR) before submission; patches 4-7 if page budget permits.**
