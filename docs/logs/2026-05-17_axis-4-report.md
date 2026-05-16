# Axis 4 Review Report — Reviewer-Prior Alignment with EMNLP 2026 Interpretability Track

**Reviewer:** Axis 4 (Round 2 parallel adversarial review)
**Date:** 2026-05-17
**Target paper:** `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex` (post-Day-12 patches M2/M3/M5; 652 lines)
**Comparator corpus:** Gemini track-selection synthesis `docs/logs/gemini research reports/Analyzing Interpretability Track Submissions.md` (26-paper sample)
**Scope:** Does the paper's framing match the specific shape of papers that succeed at the EMNLP 2026 "Interpretability and Analysis of Models for NLP" track?

---

## Q1. Title alignment

### (a) Relevant paper passage (TeX line 42)

> "Reversible KV-Cache MCTS: A Memory-Bounded Probe of Dimensional Escape in Frozen Language Models."

### (b) Precedent-paper comparison

The Gemini synthesis (§"Direct Assessment of Fit", §"The Viability of Algorithm-Primary Contributions") cites two precedent title structures as the canonical track shapes:

- *L\*-style "algorithm name : research question / interpretability application"*: "An L\* Algorithm for Deterministic Weighted Regular Languages" (EMNLP 2024). The colon-separator structure surfaces the algorithm and the (here, implicit) interpretability application together. The synthesis notes this paper "succeeded because its algorithmic novelty served explicitly to extract interpretable structures rather than solely to optimize general text generation" (Gemini §"Algorithm-Primary Contributions").
- *FSA-style "finding : mechanistic study"*: "Finite State Automata Inside Transformers with Chain-of-Thought: A Mechanistic Study on State Tracking" (ACL 2025). The colon-separator surfaces the mechanism (FSA-inside-Transformer) and the study type (mechanistic state-tracking).

### (c) Mismatch identification

Our title is colon-structured "Algorithm Name : What It Is + Object of Study", which lexically resembles the L\* form. However, the algorithm-name half ("Reversible KV-Cache MCTS") is the head noun, and the predicate ("A Memory-Bounded Probe of Dimensional Escape") leads with **memory-bound** (a Methods/Efficient-Methods-track signal) before reaching **probe** (an Interpretability signal). A speed-reading reviewer or routing-system tokenizer will hit `MCTS`, `Memory-Bounded`, `Frozen Language Models` first — three Methods/Efficiency-track tokens — before `Probe of Dimensional Escape` resolves the framing. Compared to L\*, which front-loads "Deterministic Weighted Regular Languages" (an explicitly linguistic/formal-language object), our title front-loads computational-efficiency machinery. Compared to FSA, which front-loads "Finite State Automata Inside Transformers" (an explicit interpretability finding), we front-load the search algorithm.

The token "Probe" does appear, which is load-bearing — it directly matches the ARR track keyword "probing" (Gemini §"Verbatim Scope Analysis"). But "Memory-Bounded Probe" reads as a probe-of-memory-consumption (Efficient-Methods reading) more naturally than as a representational-probing-with-bounded-memory (Interpretability reading) at first parse.

### (d) Suggested patch

Reverse the colon ordering so the interpretability framing leads. Two candidates, both Day-12-scope:

- "**Probing Dimensional Escape in Frozen Language Models: A Reversible KV-Cache MCTS with a Memory-Bounded Guarantee.**" — Interpretability finding (probing) + object (Dimensional Escape) + venue (Frozen LMs) → algorithm + formalism.
- "**Dimensional Escape in Frozen Language Models: Diagnosing Reward-Driven Latent-Space Goodhart via Reversible KV-Cache MCTS.**" — Pure FSA-form: finding : mechanistic-study verb.

The first more closely mirrors L\*; the second mirrors FSA. Either eliminates the front-loaded efficiency tokens. Day-12 cost: ~1 line, ~0 page-budget impact.

### (e) Severity

**MAJOR.** Title is the highest-leverage routing signal for the ACL submission system (Gemini §"Algorithm-Primary Contributions" warns that algorithm-leading papers "would undoubtedly be diverted to 'Machine Learning for NLP'" without interpretability framing). The current title leans Methods.

---

## Q2. Abstract structure

### (a) Relevant paper passage (TeX line 56)

Sentence-by-sentence map of the abstract (verbatim, one sentence per row):

| # | Verbatim sentence-opening | Track signal |
|---|---|---|
| S1 | "We introduce \textbf{Reversible Monte Carlo Tree Search in KV-cache latent space}, an inference-time algorithm that mutates a frozen Transformer's cached key/value tensors in place via FP32 accumulators ..." | Methods / algorithm |
| S2 | "We prove that the FP32 accumulator guarantees zero-drift reversibility (Theorem~\ref{thm:reversibility}; validated to machine precision over 200 apply--revert cycles ...) and that the working-set memory complexity is $O(M_{\text{KV}} + d \cdot K_{\text{acc}})$ ..." | Formalism / Efficient Methods |
| S3 | "Using this framework, we conduct a systematic empirical study of continuous-latent search on Llama~3.2 architectures and characterize a geometric pathology we term \textbf{Dimensional Escape} ..." | Interpretability (finally) |
| S4 | "We establish this via two complementary experiments." | Empirical structure |
| S5 | "A controlled negative-control on ARC-Easy ($n=200$ paired items) shows that an entropy-minimization reward ... collapses under MCTS optimization at both scales ..." | Negative result / mechanism diagnosis |
| S6 | "Latent Cartography sweeps across four prompt classes ... map the structural pre-training priors the search converges upon." | Interpretability ("knowledge tracing/discovering") |
| S7 | "We argue that the reward-driven failure surfaced here is structurally analogous to the monitor-driven \emph{Orthogonal Escape} previously hypothesized for alignment faking ..." | Unification / hypothesis |
| S8 | "Reversible MCTS is therefore a high-precision diagnostic probe of frozen-model representational geometry rather than a reasoning-improvement tool." | Interpretability (positioning) |
| S9 | "Artifacts are released with a Croissant~1.1 metadata package." | Reproducibility |

### (b) Precedent-paper comparison

Gemini synthesis Strategic-Synthesis §"Algorithmic Legitimacy" + §"Methodological Alignment with the Bleeding Edge" establishes the modal shape: papers must "frame the algorithm's express purpose" as serving interpretability "rather than mere performance booster." The RAVEL (ACL 2024) and FSA-Inside-Transformers (ACL 2025) abstracts both open with the interpretability question or finding, then introduce the method. Both papers' interpretability-track acceptance is attributed by the Gemini synthesis to this front-loading.

### (c) Mismatch identification

Sentences S1 and S2 — the first 2 of 9 sentences, comprising ~40% of the abstract by word count — are pure algorithm+formalism content (FP32, memory complexity, $O(M_{\text{KV}} + d \cdot K_{\text{acc}})$). A reviewer or routing-system reading the abstract top-down hits **Methods** signals for two sentences before reaching "geometric pathology" at S3. S8 explicitly self-corrects ("diagnostic probe... rather than reasoning-improvement tool") — which is precisely the Gemini-flagged risk ("a fate that nearly befell boundary-straddling papers like *TAGExplainer*"). Self-correction at S8 is too late to influence routing or the reviewer's first impression.

### (d) Suggested patch

Re-order the abstract to lead with S3+S8's framing-positioning, then introduce the algorithm as the **enabling method**:

- **S1 (new):** Open with the geometric question — adapt TeX line 65's introduction-opening question into one sentence: "When inference-time search procedures over a frozen language model's continuous latent state are guided by a low-rank scoring projection (monitor, reward, or probe), do trajectories navigate the semantic manifold or exploit unconstrained latent dimensions?"
- **S2 (new):** State the contribution as a probe: "We name this geometric pathology Dimensional Escape and characterize its reward-driven manifestation."
- **S3 (was S1):** Introduce the method as the enabling search infrastructure.
- **S4 (was S2):** State the formal guarantees as machinery the method delivers.
- **S5-S9:** Existing order preserved.

Cost: structural rewrite of 2 sentences, ~0 net word change.

### (e) Severity

**MAJOR.** First-two-sentence misalignment with track norms is the highest-leverage abstract-level fix. The closing self-correction at S8 confirms the authors already know the desired framing — they just haven't pushed it to the top.

---

## Q3. Contribution claims vs. the 4 pillars

### (a) Relevant paper passage (TeX lines 82+84)

> **C1:** "Reversible Monte Carlo Tree Search in KV-cache latent space. We introduce an inference-time algorithm that mutates a frozen Transformer's cached key/value tensors via in-place arithmetic ... We prove zero-drift reversibility (Theorem~\ref{thm:reversibility}) ... The working-set memory complexity is $O(M_{\text{KV}} + d \cdot K_{\text{acc}})$ ... reducing the memory floor from exponential in $b^d$ to additive in $d$."

> **C2:** "Empirical demonstration of reward-driven Dimensional Escape---an instance of Goodhart's Law in high-dimensional latent-space search. We establish a geometric pathology of MCTS in overparameterized latent spaces ... Two complementary experiments support the claim."

(C2 was renumbered from C3 after the Day-10 Batch 5 drop of the telemetry-matrix contribution.)

### (b) Precedent-paper comparison: the 4 pillars

From Gemini §"Strategic Synthesis" (verbatim):

1. **Algorithmic Legitimacy** — "Algorithmic interventions ... are fully supported ... provided their ultimate application is diagnostic rather than purely performance-enhancing."
2. **Negative Results as Diagnostic Tools** — "rigorous use of a controlled negative-control ... to mathematically disprove the efficacy of entropy-minimizing rewards."
3. **Formalism as the Ultimate Interpretability** — "formal memory-bound theorem linking the KV-cache footprint and FP32 accumulator zero-drift."
4. **Methodological Alignment with the Bleeding Edge** — "inference-time, frozen-weight architectures ... global representation engineering telemetry matrices."

And the Mosbach et al. infrastructure-generator criterion: "the most highly influential IA papers ... are those that propose novel methods, metrics, or frameworks based on interpretability findings, allowing external researchers to adopt and build upon those infrastructures" (Gemini §"Epistemological Function of the Interpretability Track").

### (c) Mismatch identification

Mapping:

| Contribution | Pillar match | Comment |
|---|---|---|
| C1 (algorithm + memory bound) | Pillar 1 (Algorithmic Legitimacy) + Pillar 3 (Formalism) | Hits two pillars cleanly — BUT C1 prose does **not** explicitly name itself as "infrastructure other researchers can reuse" (the Mosbach criterion). It reads as a self-contained method paper. |
| C2 (Dimensional Escape empirical) | Pillar 2 (Negative Results) | Clean match. The §3.2 Goodhart anchor (Day-12 patch M3) ties this to the Skalse/Gao prior literature, which is a strength. |
| (missing) | Pillar 4 (Methodological Bleeding Edge) | The Batch-5 drop of the telemetry-matrix contribution removed the explicit "global RepE feature attribution" framing that the Gemini synthesis cites as Pillar 4. The telemetry framework still exists in the appendix but is no longer surfaced as a §1 contribution. |

The biggest gap is the **Mosbach infrastructure-generator framing**. C1 does not say "we provide a reusable algorithm + memory budget + reversibility guarantee that the community can adopt to probe their own models." It says "we introduce an algorithm and prove its properties." Compare to RAVEL (Gemini §"Architectural Alignment", cited in the synthesis as Pillar-4 exemplar): RAVEL's contribution explicitly positions MDAS as an infrastructure for "evaluating representations globally across all layers" that other researchers can use.

### (d) Suggested patch

Two additive fixes (no structural surgery needed):

- **C1 closing sentence add:** "We release the algorithm and FP32 accumulator implementation as reusable infrastructure for probing latent-space interventions on any frozen Transformer; Croissant~1.1 artifacts (§\ref{sec:future-work}) document the protocol."
- **Optional new bullet (re-add a trimmed C3) framing the telemetry matrix as **methodological infrastructure** rather than as a contribution-with-empirical-claims:** "We provide a per-layer telemetry framework (Appendix~\ref{sec:appendix-telemetry}) that combines bottom-up activation magnitudes with top-down RepE alignment, deployable as a measurement substrate for future white-box latent-search studies." This regains the Pillar 4 signal without re-inflating page budget (the §1.1 cost is ~3 lines).

### (e) Severity

**MODERATE.** Both gaps are landable in Day-12 scope. The infrastructure-generator framing is the more important of the two — without it, C1 reads as a one-shot algorithm paper rather than as a reusable lens, which the Mosbach meta-analysis explicitly identifies as the dominant impact-correlate at this track.

---

## Q4. Vocabulary and keyword saturation

### (a) Relevant paper passage

ARR track keywords (Gemini §"Verbatim Scope Analysis", quoting ARR area keywords): "adversarial attacks/examples/training; calibration/uncertainty; counterfactual/contrastive explanations; data influence; data shortcuts/artifacts; explanation faithfulness; feature attribution; free-text/natural language explanations; hardness of samples; hierarchical & concept explanations; ... knowledge tracing/discovering/inducing; model editing; **probing**; robustness; topic modeling."

### (b) Grep over abstract + §1 (TeX lines 55–85)

- "probe" / "probing" / "probes": appears in abstract at S8 ("diagnostic **probe** of frozen-model representational geometry") and in §1 intro line 65 ("low-rank scoring projection (whether monitor, reward, or **probe**)"). 2 instances. Title also has "Probe" (1 instance).
- "feature attribution": **0 instances** in abstract or §1. Appears nowhere in the body. The telemetry framework (now appendix) would have surfaced this but is dropped from contributions.
- "knowledge tracing/discovering/inducing": **0 instances**. Latent Cartography (S6 abstract; §5.2 body) maps semantic-class priors — this **is** knowledge-discovery work, but the paper never uses that vocabulary.
- "data shortcuts/artifacts": **0 instances**. The 1B-letter-"A"-prior collapse **is** a data-shortcut phenomenon (the search exploits a pre-training artifact), but the paper calls it "prior-amplification" instead.
- "interpretability"/"interpretable": **0 instances** in abstract. Body: line 333+ uses "white-box infrastructure" but never "interpretability" as a noun. Appendix marker mentions interpretability in REVISION-comment text only (not rendered).
- "representation engineering" / "RepE": appears in §2.2 (line 100) and §3.1 (line 115), zero in abstract.
- "mechanistic": appears in S1 of §1 intro (line 65: "mechanistic analysis"). Zero in abstract.

### (c) Precedent-paper comparison

L\* and RAVEL both saturate "interpretability" and "probing" in their abstracts (per Gemini §"Algorithm-Primary Contributions" + §"Architectural Alignment" — RAVEL's MDAS is described as a "global feature attribution" lens). RAVEL's title contains "Interpretability Methods" directly. L\*'s abstract per Gemini's quotation foregrounds "extracting finite state automata (FSAs) from black-box models" as "interpretable insights" — explicit interpretability vocabulary.

### (d) Mismatch identification

Our abstract uses **0 instances** of "interpretability," **0 instances** of "feature attribution," and **0 instances** of "knowledge tracing/discovering/inducing" — three of the explicit ARR routing keywords. The single instance of "probe" at S8 carries the entire interpretability load, and the title's "Probe" is ambiguous (Q1 finding). The body fills some of this gap (RepE, probe extraction in §6.3 appendix), but the routing system reads abstract + title only. This is **highly likely to misroute to Efficient Methods or Methods**.

The Gemini synthesis §"Architectural Alignment" explicitly identifies "global feature attribution" as the modal track-acceptance signal — and we have dropped exactly this vocabulary from the abstract in the Batch-5 drop-C2 surgery.

### (e) Suggested patch

Saturate 3-4 keywords in abstract + §1 Contributions. Low-cost edits:

- Abstract S8 currently: "Reversible MCTS is therefore a high-precision diagnostic probe of frozen-model representational geometry rather than a reasoning-improvement tool." → "Reversible MCTS is therefore a high-precision **interpretability probe** of frozen-model representational geometry — a **feature-attribution lens** for diagnosing how inference-time latent-space search resolves low-rank scoring projections — rather than a reasoning-improvement tool."
- Abstract S6 currently: "Latent Cartography sweeps ... map the structural pre-training priors the search converges upon." → "Latent Cartography sweeps ... map the structural pre-training priors the search converges upon, a form of **knowledge-tracing** for inference-time interventions."
- §1.1 C2 currently opens with "Empirical demonstration of reward-driven Dimensional Escape": consider adding "(a **data-shortcut/artifact** of the entropy reward against the model's first-letter prior)" parenthetical.

Cost: ~20 words total. Page-budget impact: negligible.

### (e) Severity

**MAJOR.** This is the highest-leverage routing fix. Without keyword saturation, the ACL routing pipeline may not match this paper to the Interpretability track's keyword filter at all — making the rest of our framing work moot.

---

## Q5. Negative-result framing in §5 Experiment 1

### (a) Relevant paper passages (TeX lines 196–224)

The §5 Experiment 1 prose includes:

- **Hypothesis-anchored setup** (line 196): "We test whether the entropy-normalized surrogate reward $\hat{r}(\mathbf{h}) = -H(p(\cdot \mid \mathbf{h}))/\log|V|$ (Eq.~\ref{eq:goodhart}) carries useful optimization signal for reasoning when deployed as the objective of MCTS over the KV-cache latent space." (Mechanism-diagnostic frame: tests a specific hypothesis about reward design.)
- **Results paragraph** (line 199): "$\Delta_{E-R} = -0.005$ ... ($p = 1.0$, 3 discordant pairs); ... entropy-MCTS does not differentiate from a uniform-random reward at either scale." (Statistical framing — clean.)
- **Mechanism paragraph** (line 222): "The pathway by which the search satisfies $\hat{r}_E$ is class-conditional on the unsteered model's prior. At 1B ... the search selects heavy steering ... and entropy-MCTS predicts ``A'' on $185/200$ items against a roughly uniform gold distribution. At 3B the unperturbed cache is already low-entropy and minimal steering preserves the greedy-equivalent state ... Neither pathway selects the correct-answer cache state; both select the entropy-minimum under the model's biases---the empirical mechanism predicted by Hypothesis~\ref{hyp:goodhart}." (Mechanism diagnosis — exactly the Gemini-recommended frame.)
- **Interpretation paragraph** (line 224): "Next-token entropy correlates statically with correctness at $|r| \approx 0.60$ on fixed prompts ... but the correlation does not transfer to the latent-search regime ... The result is the reward-driven counterpart to Bailey et al.'s~\cite{bailey2024obfuscated} negative result ... both manifestations terminate in the unconstrained complement of a low-rank scoring projection rather than on the semantic manifold (Definition~\ref{def:dim_escape})." (Theory-anchored — strong.)

### (b) Precedent-paper comparison

Gemini §"Mechanism Diagnosis via Negative Results" cites *CUTE* (EMNLP 2024) and *Lexical Recall or Logical Reasoning* (ACL 2025) as the gold standard: "they do not merely state that a model fails; they isolate the precise computational, representational, or heuristic reason *why* it fails." The §5 Experiment 1 Mechanism paragraph hits this standard cleanly. The Interpretation paragraph adds the Bailey-2024 theoretical anchor.

### (c) Mismatch identification

Two minor "we tried X and it didn't work" residues, both at the **paragraph-opener** level (which carries reviewer-attention weight):

- **Line 199 Results paragraph opener:** "Per-condition accuracy and 95\% Wilson CIs are in Table~\ref{tab:exp1-results}. The headline pairwise comparison is entropy-MCTS vs random-MCTS: at 1B $\Delta_{E-R} = -0.005$..." This **leads with the null result** before the mechanism. A "we tried X and it didn't work" reading would extract "at 1B Δ = -0.005, at 3B Δ = +0.005, both p=1.0" → null finding. The mechanism doesn't arrive until the next paragraph.
- **Mean-perplexity sentence** (end of line 199): "Mean-perplexity ratios $E/G = 1.348$ at 1B and $1.058$ at 3B confirm the searched cache states remain on-distribution---the failure is from the entropy objective misaligning with correctness, not from off-manifold drift." The phrase "the failure is from the entropy objective misaligning with correctness" is a **good** mechanism-diagnostic sentence — but the broader Results paragraph still reads as "we tried X and it didn't work" because the diagnostic verb arrives only in the last sub-clause.

These are minor: the §5 Exp 1 section overall is well-framed as mechanism diagnosis (Hypothesis-anchored setup → null result → mechanism → interpretation tied to Definition def:dim_escape). The patch is to **shift one diagnostic sentence forward** to inoculate against a fast-reading reviewer skimming the Results paragraph in isolation.

### (d) Suggested patch

Reorder Results paragraph (line 199): move the diagnostic frame ahead of the numbers.

- **Current:** "Per-condition accuracy and 95\% Wilson CIs are in Table~\ref{tab:exp1-results}. The headline pairwise comparison is entropy-MCTS vs random-MCTS: at 1B $\Delta_{E-R} = -0.005$ ..."
- **Patched:** "The headline test asks whether entropy-MCTS — an active deployment of a reward that **statically correlates with correctness at $|r| \approx 0.60$** — outperforms a uniform-random reward when used as the MCTS scoring function. Per-condition accuracy ... at 1B $\Delta_{E-R} = -0.005$ ..."

Cost: ~10 words. The diagnostic frame ("an active deployment of a reward that statically correlates with correctness") now leads the Results paragraph, ensuring even a skim-read carries the mechanism-diagnostic signal.

### (e) Severity

**MINOR.** §5 Experiment 1 is already mostly well-framed. This is a polish edit; the existing prose is acceptance-quality.

---

## Q6. Discussion / Implications section: actionable infrastructure?

### (a) Relevant paper passages (TeX lines 334–363)

- **§7.1 Implications opening** (line 336): "The reward-driven manifestation of Dimensional Escape ... carries a direct implication for the design of test-time scaling procedures that operate on continuous latent representations." (Calls out **design implications** for the field.)
- **§7.1 closing** (line 338): "Reversible KV-Cache MCTS provides white-box infrastructure for evaluating which response class generalizes." (Self-positions as infrastructure — strong.)
- **§7.2 Broader Impact** (line 343): "Reversible KV-Cache MCTS provides a controlled-access white-box probe of the latent geometry of frozen language models; the Dimensional Escape pathology it surfaces ... has implications for the design of both monitoring procedures and test-time scaling schemes." (Reuse framing — present.)
- **§7.5 Future Work** (line 363): "Anti-Goodhart reward designs that may resist the reward-driven Dimensional Escape established by Experiment~1 ... constitute a research program in their own right. The Latent Cartography sweep design (§\ref{sec:exp2}) extends naturally to additional prompt classes (adversarial probes, multilingual inputs, code generation, multi-turn chain-of-thought reasoning) and to cross-model comparisons ..." (Specific reuse pathways — strong.)
- **Croissant 1.1 artifacts** (line 363 closing, line 368): "A Croissant~1.1 dataset package of runtime evidence artifacts is included as supplementary material and will be extended in Phase~B." (Tangible reusable artifact — present.)

### (b) Precedent-paper comparison

Gemini §"Epistemological Function" quotes Mosbach et al. EMNLP 2024 verbatim: "the most highly influential IA papers ... are those that propose novel methods, metrics, or frameworks based on interpretability findings, allowing external researchers to adopt and build upon those infrastructures." The Mosbach call-to-action requires three components: (i) the method/framework, (ii) explicit reuse pathways, (iii) released artifacts.

### (c) Mismatch identification

We have all three components, but the discussion is **distributed across three subsections** (§7.1, §7.2, §7.5) and the explicit "what other researchers can DO with this" framing is **not consolidated into a single read-once paragraph**. A reviewer reading §7.1 Implications gets the design-implications-for-reward-design framing but does **not** get the algorithm-reuse-for-probing-your-own-model framing until §7.2. The Latent Cartography extensibility framing (Cartography across new prompt classes / cross-model comparisons) appears in §7.5 Future Work, which a reviewer may skip.

Compare to the L\*-paper framing per Gemini synthesis: the algorithmic contribution is positioned as "a mathematical key to unlock the black box" in the body itself — not deferred to a Future Work section.

The biggest specific gap: **there is no sentence in §7.1 or §7.2 saying "researchers studying [X domain] can apply Reversible KV-Cache MCTS to their own model to probe [Y phenomenon]."** The infrastructure is positioned by what it does (white-box probe) but not by who uses it and how.

### (d) Suggested patch

Add one paragraph (or one strong sentence) to §7.1 immediately after the design-response sentence (line 338) that **enumerates explicit reuse pathways**. Candidate:

> "**Reuse pathways.** The released algorithm, FP32 accumulator implementation, and Croissant~1.1 artifacts (§\ref{sec:future-work}) are deployable as a measurement substrate for three concrete research lines: (i) anti-Goodhart reward design — any candidate reward (Process Reward Models, multi-channel ensembles, self-consistency variants) can be evaluated against the null-control protocol established by Experiment~1; (ii) cross-model representational-geometry comparisons — the Latent Cartography sweep (Experiment~2) extends to any frozen open-weight Transformer with bf16-bf16 cache compatibility; (iii) monitor-driven Dimensional Escape testing — Hypothesis~\ref{hyp:orthogonal_escape} can be operationalized by replacing the entropy reward with a learned-probe monitor on the same FP32-accumulator infrastructure (Phase~B)."

Cost: ~80 words. Page-budget impact: ~3 lines in ACL two-column. Hits Pillar 1 (algorithmic legitimacy via reuse) + Pillar 4 (methodological alignment) + the Mosbach infrastructure-generator criterion in one paragraph.

### (e) Severity

**MODERATE.** The infrastructure-as-toolkit framing is present but distributed; a consolidated paragraph in §7.1 would substantially strengthen the Mosbach-criterion alignment. Borderline MAJOR if the abstract+title fixes from Q1-Q4 are not adopted, since the discussion would be the only place that carries the infrastructure framing.

---

## Q7. Most-likely reviewer reject reason

### (a) Constructed reviewer critique

A reviewer trained on the 26-paper interpretability-track corpus would most likely write:

> "The paper is interesting but **lacks a demonstrated downstream use of the algorithm as an interpretability lens beyond the negative-control study itself.** The Reversible KV-Cache MCTS framework is well-motivated and the formal memory bound is appreciated. Experiment 1 cleanly establishes that entropy-MCTS Goodharts; Experiment 2 (Latent Cartography) extends the diagnosis across prompt classes. However, the paper does not show **a single positive interpretability finding obtained by using the algorithm as a probe** — i.e., a discovery about model representations that could not have been made without Reversible KV-Cache MCTS. Bailey et al.'s gradient-trained negative result already establishes that orthogonal escape is hard; our reward-driven analog confirms the same intuition in a different regime. What is the new knowledge about *the model* (not about the algorithm) that this paper produces?"

### (b) Why this reviewer prior

Gemini §"Mechanism Diagnosis via Negative Results" identifies the precedent papers' shared structure: each negative-result paper produces a **specific positive claim about model representations or behavior** — *CUTE* reveals "the multi-character tokenization process [as] the model's atomic processing limits"; *Lexical Recall or Logical Reasoning* proves "models are overly reliant on lexical recall heuristics"; *CausalGym* proves "the mechanism implementing negative polarity item licensing is learned in discrete stages." Each negative finding is structured as: failure mode + isolated mechanism + positive characterization of what the model *does* do.

Our paper has the first two pieces (entropy-MCTS fails, isolated mechanism = prior-amplification at 1B / greedy-collapse at 3B) and Latent Cartography produces a structural observation (the cross-scale direction-flip in α-tuple selection between MCQ and continuation classes). But the **positive characterization** is currently framed as "Goodhart confirmed in another regime" rather than as a specific finding about Llama 3.2's representational geometry.

There **is** a positive finding present: Experiment 2 Table 2 shows the MCQ direction-flip across 1B → 3B (heavy steering at 1B, minimal at 3B). This is a real positive structural claim about how MCQ-prior representations differ across scale. But the paper doesn't lead with it — Experiment 2 is currently framed as "the failure manifests across domains" rather than as "we discover a scale-dependent prior-structure inversion in MCQ classes."

### (c) Mismatch identification

The paper's headline positive contribution is the **algorithm + memory bound** (C1), with the empirical study (C2) framed as confirmation of a previously-hypothesized geometric pathology. The Gemini synthesis Pillar 1 ("Algorithmic Legitimacy") requires that "algorithmic novelty served explicitly to extract interpretable structures" — and the structures extracted need to be **specific, named, and downstream-relevant**. Our Latent Cartography findings (direction-flip, argmax-A collapse, gold-rank-preserved-despite-collapse) **are** such structures, but they are buried in Experiment 2 results bullets rather than surfaced as headline positive contributions.

### (d) Suggested patch

Add a third bulleted contribution (or upgrade C2 into a two-part contribution) that surfaces the **positive interpretability finding**:

> "**C3 (added):** Cross-scale structural finding: we discover that the MCQ prior-amplification regime inverts between Llama 3.2-1B and 3.2-3B (Table~\ref{tab:exp2-cartography-paths}) — at 1B, MCQ items drive heavy steering ($\bar\alpha = 0.83$–$0.89$) toward a letter-``A'' attractor; at 3B, the same items drive minimal steering ($\bar\alpha = 0.22$–$0.28$) because the balanced baseline distribution offers no exploitable prior. Factual recall and continuation classes show neither pattern. This direction-flip is the first observation that the Goodhart pathology's specific manifestation is jointly determined by prompt class and pre-training prior structure — a finding that extends to the design of prompt-class-conditional reward shaping (§\ref{sec:future-work})."

Cost: ~80 words. This converts the existing Experiment 2 sub-finding (already in body) into a §1-level positive contribution.

### (e) Severity

**MAJOR.** This is the single highest-leverage anti-reject fix. A reviewer asking "what does this paper teach me about the model that I didn't already know?" currently has to dig into Experiment 2 §5.2(i)-(iii) to find the answer; surfacing it at §1 transforms the contribution shape from "we confirmed Bailey in a new regime" to "we discovered a scale-dependent prior-structure regime in Llama 3.2 MCQ representations." The latter is a Pillar-1 + Pillar-2 + Pillar-4 triple-hitter; the former hits only Pillar 2.

---

## Final Summary: Ranked Reviewer-Prior Alignment Patches

Patches ranked by acceptance-leverage / severity (most important first):

1. **(MAJOR) Surface the positive cross-scale finding as a §1 contribution.** The MCQ direction-flip (Llama 3.2-1B → 3.2-3B) is currently buried in Experiment 2 results bullets. Promoting it to a §1.1 contribution converts the paper from "confirmation of Bailey in a new regime" to "discovery of a scale-dependent prior-structure regime + an algorithm that found it." Cost: ~80 words. Day-12-scope. (Q7)

2. **(MAJOR) Saturate ARR routing keywords in abstract + §1.** Add 3-4 instances of "interpretability," "feature attribution," "knowledge tracing," "data shortcut/artifact" — currently 0 instances of three of these. Cost: ~20 words. Day-12-scope. Routing is high-leverage; without this, the rest of the framing work may not reach the right reviewers. (Q4)

3. **(MAJOR) Re-order abstract S1+S2 to lead with interpretability framing, not algorithm+formalism.** Move the Dimensional Escape question to S1, position the algorithm as enabling infrastructure for the probe. Cost: structural rewrite of 2 sentences, ~0 net word change. Day-12-scope. (Q2)

4. **(MAJOR) Re-order the title.** Current title leads with "Reversible KV-Cache MCTS" + "Memory-Bounded" — three Methods/Efficiency tokens before the Interpretability signal. Reverse to lead with "Probing Dimensional Escape" or "Dimensional Escape in Frozen Language Models." Cost: 1 line, ~0 page-budget. Highest single-edit leverage. (Q1)

5. **(MODERATE) Add a "Reuse pathways" paragraph to §7.1 Implications.** Consolidate the three currently-distributed reuse signals (anti-Goodhart evaluation, cross-model Cartography, monitor-driven testing) into one paragraph immediately after the design-response sentence. Cost: ~80 words / ~3 lines two-column. Hits the Mosbach infrastructure-generator criterion explicitly. (Q6)

6. **(MODERATE) Re-add a trimmed telemetry-matrix infrastructure framing as a §1 deliverable.** The Batch-5 drop removed our Pillar-4 ("global feature attribution") explicit signal. Re-add as one-line infrastructure-bullet that points to the appendix, not as a re-inflated contribution-with-empirical-claims. Cost: ~3 lines. Day-12-scope. (Q3)

7. **(MODERATE) Add "infrastructure" framing to C1.** C1 currently reads as a self-contained method paper. Add one closing sentence pointing to the released algorithm + Croissant artifacts as deployable infrastructure for the community. Cost: ~25 words. Day-12-scope. (Q3)

8. **(MINOR) Polish §5 Experiment 1 Results paragraph opener.** Shift the diagnostic frame ("an active deployment of a reward that statically correlates with correctness") ahead of the null-result numbers, so the mechanism-diagnostic signal carries even a skim-read. Cost: ~10 words. Polish-tier. (Q5)

**Headline severity tally: 0 SEVERE / 4 MAJOR / 3 MODERATE / 1 MINOR.**

**Top-rank finding:** The paper has the right shape but its headline doesn't say so. The positive interpretability finding (MCQ prior-structure direction-flip across 1B → 3B) exists in the body but is not promoted to §1; combined with the title and abstract front-loading algorithm+formalism over interpretability+probing, the routing risk is significant. Four MAJOR patches at the title/abstract/§1 level — each ≤80 words — would convert this from a Methods-track-borderline submission to a high-confidence Interpretability-track fit. None require new experiments; all are achievable within Day-12 scope.
