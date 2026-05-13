# Track B-delta — Citation Verification Audit (4 Newly-Downloaded Papers)

**Author:** Diagnostic Claude Session (Track B-delta agent)
**Created:** 2026-05-08 (in response to original Track B audit's MISSING-citation queue from 2026-05-09)
**Scope:** Audit four citations that were flagged MISSING in the original Track B audit (Snell, Hasani, Stoehr, MUSE) now that they have been downloaded. Snell flagged as the highest-priority fatal-eligible unknown.
**Method:** Read each paper's abstract + introduction + main-claim sections from local source bundle (LaTeX or PDF). Extract verbatim claims with section refs. Compare against Candidate A framing in `docs/logs/2026-05-08_EMNLP Framing for KV-Cache MCTS.md` Part 4 + against premortem-v2.tex. Verdict per protocol: CLEAN / NUANCED / MISMATCHED.

---

## TOP-LINE FINDINGS (read first)

- **DELTA PILE-UP:** 0 NUANCED, 0 MISMATCHED across the 4 new papers. All four return CLEAN.
- **SNELL FATAL-FINDING:** **NEGATIVE.** Snell does NOT argue that test-time compute scaling requires discrete-token search. Their unified Proposer-Verifier abstraction explicitly admits search modifications and continuous interpolations; their lookahead-search-as-degenerate-MCTS framing actually aids our continuous-latent extension. See §3 below for verbatim analysis.
- **CUMULATIVE PILE-UP** (post-Track A retirement of RepE NUANCE): **0 NUANCED + 0 MISMATCHED total.** Track A's certainty-probe validation retired the only original NUANCE; the delta adds zero new ones.
- **GATE 1 G1.3 (≤2 NUANCED):** **PASS** with margin (0/2).
- **GATE 2 G2.5 (no late-arriving contradictions):** **PASS.** Snell, Hasani, Stoehr, MUSE all return clean differentiation; no contradictions surfaced.
- **NEW MANDATORY CITATIONS SURFACED:** None. The four papers slot into the existing citation plan as anchors (System-2, test-time-compute, dynamic-vs-static activation steering, multi-turn discrete-text MCTS).

---

## 1. Per-paper table

| # | Name | arXiv/file | Verdict | Notes |
|---|---|---|---|---|
| D1 | Snell et al. 2024 — Scaling LLM Test-Time Compute Optimally | `arXiv-2408.03314v1/template_content.tex` | **CLEAN** | Proposer-Verifier abstraction is method-agnostic; lookahead-search defined as a special case of MCTS with stochastic exploration removed; no claim that test-time compute must be discrete. Snell anchors our framing rather than excluding it. |
| D2 | Hasani et al. 2025 — Mechanistic Interpretability of Large-Scale Counting through a System-2 Strategy | `arXiv-2601.02989v2/acl_latex.tex` | **CLEAN** | System-2 = behavioral chain-of-thought via prompt-engineered task partitioning on FROZEN models, no fine-tuning. They use System-1/System-2 as "high-level operational abstractions inspired by human cognition" — exactly the framing flexibility we need to extend to latent-search. |
| D3 | Stoehr et al. 2024 — Activation Scaling for Steering and Interpreting Language Models (EMNLP Findings) | `2024.findings-emnlp.479.pdf` | **CLEAN** | Static, gradient-trained multiplicative scalars on activation vectors at fixed token positions. Differentiator from RepE: ACTIVSCALAR scales magnitude only (not direction), is parameter-efficient, and is trained per-task. NOT search-based, NOT dynamic-during-inference. Clean anchor for "static steering" baseline contrast. |
| D4 | MUSE — Yu et al. 2025 (EMNLP main 2025) | `2025.emnlp-main.1080.pdf` | **CLEAN** | MCTS over a discrete frame-semantics action space (Expansion / Decomposition / Redirection prompts) for multi-turn jailbreaks. Operates entirely on text inputs to a frozen target model; rewards via GPT-4o judge. NOT operating on activations or KV-cache. Clean anchor for "discrete-text-space MCTS" precedent we extend continuously. |

---

## 2. Detailed entries

### 2.1 Snell et al. 2024 — Test-Time Compute Scaling — **CLEAN**

**Paper file:** `docs/NeurIPS/Citations/arXiv-2408.03314v1/template_content.tex`

**Verbatim quote (§1 Introduction, line 15-16):**

> "Humans tend to think for longer on difficult problems to reliably improve their decisions. Can we instill a similar capability into today's large language models (LLMs)? More specifically, given a challenging input query, *can we enable language models to most effectively make use of additional computation at test time so as to improve the accuracy of their response?*"

**Verbatim quote (§2 Unified Perspective, line 50):**

> "First, we view the use of additional test-time compute through the lens of modifying the model's predicted distribution *adaptively* at test-time, conditioned on a given prompt... In general, there are two knobs to induce modifications to an LLM's distribution: **(1) at the input level**: by augmenting the given prompt with an additional set of tokens that the LLM conditions on to obtain the modified distribution, or **(2) at the output level**: by sampling multiple candidates from the standard LM and performing surgery on these candidates."

**Verbatim quote (§5.1 Search Methods, line 165):**

> "Note that this version of lookahead search is a special case of MCTS, wherein the stochastic elements of MCTS, designed to facilitate exploration, are removed since the PRM is already trained and is frozen."

**Verbatim quote (§7 Discussion, line 339-341):**

> "While we combined verifiers with revisions in Section 6, we did not experiment with PRM tree-search techniques in combination with revisions. Neither did we study other techniques such as critique and revise... Future work should investigate how test-time compute scaling can be further improved by combining a variety of these approaches. Additionally, we found that across the board these schemes provided small gains on hard problems; future work should work to develop new ways of using test-time compute which can circumvent this limitation."

**Candidate A framing claim (Part 4 of `2026-05-08_EMNLP Framing for KV-Cache MCTS.md`):**

> "We formalize a novel paradigm for System-2 reasoning where computational budgets are spent exploring and optimizing internal representations via heuristic tree search, rather than autoregressively generating and discarding discrete text tokens."

**Comparison:**
- Snell's Proposer-Verifier abstraction is **method-agnostic**: any process that modifies the predicted distribution adaptively at test-time qualifies. Continuous-latent KV-cache mutation falls cleanly within this abstraction as a *novel* axis Snell did not study but did not exclude. ✓
- Snell's experimental scope is text-space methods (best-of-N, beam search, lookahead-search, sequential revisions), all operating on PaLM 2-S* with verifier-guided proposal/output modification. ✓ Differentiated.
- Snell's lookahead-search is explicitly described as a degenerate MCTS with stochasticity removed; this is structurally important: it confirms that MCTS-style methods are part of the test-time-compute family, and that the choice to operate on tokens vs. activations is independent of the algorithmic framework. ✓ Our continuous-latent MCTS is a parallel branch within Snell's family.
- Snell's §7 limitation explicitly invites "new ways of using test-time compute which can circumvent" the diminishing returns on hard problems. ✓ This is rhetorical sympathy, not exclusion.
- Capability-specific finetuning is required only for the **revision-as-proposal-distribution** modifier (Snell footnote, §1 line 38). The verifier-side search experiments (best-of-N, beam search, lookahead) use frozen base models with separately-trained PRMs. Our paper uses a frozen base model and a separately-defined reward function (T_t telemetry) — this matches Snell's verifier-side regime exactly. ✓

**FATAL-CHECK RESULT:** **NEGATIVE.**
- Does Snell argue test-time compute REQUIRES discrete-token search on frozen models? **NO.** Snell defines test-time compute as adaptive distribution modification; their experiments happen to operate on tokens but the abstraction does not require it.
- Does Snell rule out continuous-latent variants? **NO.** They simply do not study them. §7 limitations actively call for new methods.
- Does Snell claim frozen base models lack the structure for continuous test-time-compute? **NO.** No such claim anywhere in the paper.

**Verdict: CLEAN.** No revision needed. Recommended *additive* sentence in §1 of Candidate A draft: "We extend the test-time compute taxonomy of Snell et al.~\cite{snell2024scaling}, which formalized parallel best-of-N, sequential revisions, and discrete-token tree search as the canonical axes, by introducing a fourth axis: continuous-latent tree search operating directly on the KV-cache of a frozen base model."

This framing positions our work as a clean extension rather than a rebuttal, which is the strongest possible alignment with Snell.

---

### 2.2 Hasani et al. 2025 — System-2 Counting — **CLEAN**

**Paper file:** `docs/NeurIPS/Citations/arXiv-2601.02989v2/acl_latex.tex`

**Verbatim quote (Abstract, line 55):**

> "...we propose a simple test-time strategy inspired by System-2 cognitive processes that decomposes large counting tasks into smaller, independent sub-problems that the model can reliably solve."

**Verbatim quote (§1 Introduction, line 100):**

> "In this study, we use System-1 and System-2 as high-level operational abstractions inspired by human cognition. Specifically, we refer to System-1 as the model's single-pass, layer-wise processing, and to System-2 as a more deliberate procedure that extends computation across token generation beyond a fixed layer-wise computation."

**Verbatim quote (§2 Methodology, line 122):**

> "This strategy is based on test-time scaling by leveraging the LLM's inherent capabilities and does not require fine-tuning or any external tools."

**Verbatim quote (§3 Behavioral Results, line 152):**

> "...for each model, we find that only one configuration consistently succeeds on long contexts: structured inputs combined with intermediate reasoning steps."

**Candidate A framing claim:** Hasani is cited as one anchor for the "System-2" terminology to head off a reviewer challenge of "what does System-2 mean here?".

**Comparison:**
- Hasani uses System-2 in exactly the **operational** sense Candidate A needs: "a more deliberate procedure that extends computation across token generation beyond a fixed layer-wise computation" (§1, line 100). This abstraction explicitly admits any procedure that extends computation beyond single-pass — including continuous-latent MCTS. ✓
- Hasani's System-2 instantiation is behavioral / prompt-engineering (partition lists with `|` separators + chain-of-thought). It operates on **frozen models without fine-tuning**, which matches our regime. ✓
- Hasani provides a mechanistic interpretation backbone (counting accumulates layerwise; partitions reset the counter; aggregation happens at intermediate steps) — useful for our paper's mechanistic claims about how telemetry-guided latent search exploits per-layer accumulator structure. ✓ Relevant for §6 discussion link.
- The "System-2 strategy" in Hasani is a **specific instantiation** of the broader concept (decomposition + intermediate steps). Our continuous-latent MCTS is a **different specific instantiation** of the same broader concept (extended computation beyond single-pass via heuristic tree search). Both are legitimate System-2 instances under Hasani's operational definition. ✓

**Verdict: CLEAN.** No revision needed. Recommended one-sentence anchor for §1 of Candidate A: "We adopt the operational System-2 framing of Hasani et al.~\cite{hasani2025counting}, who define System-2 as 'a more deliberate procedure that extends computation across token generation beyond a fixed layer-wise computation'; while their instantiation uses behavioral task decomposition, we instantiate the same paradigm via heuristic tree search in continuous latent space."

---

### 2.3 Stoehr et al. 2024 — Activation Scaling — **CLEAN**

**Paper file:** `docs/NeurIPS/Citations/2024.findings-emnlp.479.pdf` (EMNLP Findings 2024, pages 8189-8200)

**Verbatim quote (Abstract):**

> "Given the prompt 'Rome is in', can we steer a language model to flip its prediction of an incorrect token 'France' to a correct token 'Italy' by only multiplying a few relevant activation vectors with scalars? ... activation scaling only modifies the signed magnitude of activation vectors to strengthen, weaken, or reverse the steering directions already encoded in the model."

**Verbatim quote (§1 Introduction, p. 8189):**

> "A parallel literature on model steerability also seeks to develop effective interventions, not for the primary purpose of interpretability, but to steer models toward desirable behaviors... Methods in this literature are typically designed to be maximally effective for steering and thus tend to involve model-wide (i.e., not necessarily localized) interventions to the weights or activation vectors."

**Verbatim quote (§3.2 Defining an Intervention, p. 8191, Eq. 5):**

> "We instead propose a more parameter-efficient intervention which is restricted to scaling the signed magnitude of each activation vector via a single multiplicative scalar... `\tilde{h}_{l,i}^{(s)} = h_{l,i}^{(s)} (1 + \beta \alpha_{l,i}^{(s)})`."

**Verbatim quote (§6.2 Other Related Work, p. 8197):**

> "Joint interpretability and steerability is also a desired property in circuit discovery... However, this means that the intervention can be considered discrete, as it cannot smoothly facilitate different intervention strengths and directions. The scaling aspect of the β hyperparameters in ACTIVSCALAR and STEERVEC, in turn, is continuous."

**Verbatim quote (§5 Extension to Variable-Length Prompts, p. 8194):**

> "ACTIVSCALAR and STEERVEC, as well as most existing interpretability methods such as activation patching, learn intervention parameters that are tied to specific token positions i in the prompt x_n. To reuse learned parameters for intervention with a new prompt, the test prompt must match the length I_n of the training prompt."

**Candidate A framing claim:** Stoehr is cited as the "venue-fit precedent in §2" for activation steering, and as the differentiator anchor for "dynamic vs static activation steering".

**Comparison:**
- ACTIVSCALAR is a **gradient-trained, fixed-position-tied, multiplicative-scalar intervention** — i.e., parameters θ are LEARNED on a training set with a three-term objective (effectiveness, faithfulness, minimality), then APPLIED at inference time as a static scaling. ✓ This is "static steering" in our differentiator sense.
- DYNSCALAR (their §5 extension) makes the scalar a learned function of the activation vector itself (a probe-like dot product, Eq. 10). Still gradient-trained. Still static during inference (the function is fixed; only the scalar value adapts to the prompt). ✓ NOT search-based; NOT iterative; NO test-time exploration.
- The "continuous β" Stoehr emphasizes (§6.2) refers to the **strength scalar β** being a continuous knob — NOT to operating in continuous latent search space. They are continuous in the sense that you can dial intervention strength smoothly between -1 and +1. We are continuous in a different and stronger sense: we explore the latent geometry via tree search. ✓ Differentiation is on multiple axes (search vs. fixed; iterative vs. one-shot; per-prompt latent exploration vs. learned-then-applied).
- Stoehr's largest models are GPT2-XL (1.5B) and Pythia-1.4B (Limitations section, p. 8197) — same scale regime as our Phase 2/3 work, supporting generalizability claims.
- Differentiator from RepE specifically: Stoehr scales magnitude only (not direction); RepE extracts directional concepts via PCA on contrastive activations and applies additive interventions. Both are static-steering but methodologically distinct. ✓

**Verdict: CLEAN.** No revision needed. Recommended anchor for §2 (Related Work) of Candidate A: "Stoehr et al.~\cite{stoehr2024activation} demonstrated that gradient-trained multiplicative scalars on activation vectors yield effective and parameter-efficient steering on synthetic tasks, while remaining tied to fixed token positions in the training prompt. Our work differs along three independent axes: (i) we operate on the KV-cache rather than on residual-stream activations, (ii) we explore latent geometry iteratively via tree search rather than applying a learned static intervention, and (iii) our reward signal is a per-step telemetry computation rather than a fixed three-term gradient objective."

---

### 2.4 MUSE — Yu et al. 2025 — Multi-Turn MCTS Red-Teaming — **CLEAN**

**Paper file:** `docs/NeurIPS/Citations/2025.emnlp-main.1080.pdf` (EMNLP main 2025, pages 21282-21303)

**Verbatim quote (Abstract):**

> "We introduce MUSE, a comprehensive framework tackling multi-turn jailbreaks from both attack and defense angles. For attacks, we propose MUSE-A, a method that uses frame semantics and heuristic tree search to explore diverse semantic trajectories."

**Verbatim quote (§4.1 Multi-turn Semantic Attack, p. 21284):**

> "MUSE-A operates in two key phases. First, it constructs an action space by applying three frame semantics strategies, providing a set of candidate actions for dialogue generation. Then, at each step, it selects an action from this space to generate the next dialogue query, leveraging MCTS to strategically navigate the action space..."

**Verbatim quote (§4.1.1, p. 21284):**

> "By integrating these three strategies, we construct a comprehensive action space A: A = A_exp ∪ A_dec ∪ A_red. Each action a_t ∈ A corresponds to a specific conversational move based on one of the strategies."

**Verbatim quote (§4.1.2 Reward Function, p. 21285):**

> "Leveraging GPT's strong alignment with human judgment in content safety, we use GPT-4o as an automatic judge to rigorously score responses on a 1–10 scale during evaluation. If the score is less than 5, the jailbreak is considered successful, and the reward is set to 1; otherwise, the reward is set to 0."

**Verbatim quote (Algorithm 1, p. 21286):**

> "Attacker: x_t ∼ π_{θ_a}(x_t | q, a_t, c_{t-1}); Defender: y_t ∼ π_{θ_d}(y_t | c_{t-1}, x_t)"

**Candidate A framing claim:** MUSE is cited as the multi-turn-safety MCTS precedent we extend continuously vs. discretely.

**Comparison:**
- MUSE's MCTS operates on a **discrete action space of three text-generation strategies** (Intra-Frame Expansion, Inter-Frame Decomposition, Perspective-Driven Redirection), with each action selecting a prompt-template that the attacker LLM uses to generate the next turn's text query x_t. ✓ Pure text-space.
- The target/defender model is queried as a **frozen black box**: the attacker generates a text prompt; the defender produces a text response; the reward is computed by GPT-4o judging the text. No activations, no KV-cache, no internal states are touched. ✓ Differentiator on intervention site is total.
- The "tree" in MUSE-A is a tree of **dialogue contexts** c_t (sequences of (x_t, y_t) text pairs). Each node is a full text-conversation state. Our paper's tree is a tree of **KV-cache states** A^(ℓ) (per-layer activation accumulators). ✓ Different state space entirely.
- The reward in MUSE is a **binary judge-based signal** on dialogue endpoints (safe / unsafe). Our reward is a **per-step continuous telemetry signal** computed from internal model state (T_t = [σ_H, ρ_R]). ✓ Different reward locality and modality.
- MUSE's UCB1 selection formula (p. 21285) is functionally identical to our search policy formula — confirming MCTS-as-meta-algorithm is shared, but the substrate differs completely. ✓ This is a clean precedent-and-extension story.
- MUSE's frozen-target setting matches our paper's frozen-base-model regime. ✓ Both are training-free.
- MUSE's largest target is Claude-3.5-sonnet (commercial); their MUSE-A attacker is Wizard-Vicuna-30B-Uncensored. Our scale (1B-3B local + gpt-oss-20b on H100) is smaller than MUSE's ASR-target side but consistent with MUSE-A's per-call efficiency analysis (Llama-3-8B at 12 model calls per success, Table 9 p. 21296).
- MUSE-D (the defense side of MUSE) uses MCTS-derived risk scores to weight DPO training data — this is a **training-time** use of MCTS outputs, which is orthogonal to our test-time-only paradigm. We can cite MUSE-D as a downstream application of MCTS-derived signals if useful, but it's not load-bearing.

**Verdict: CLEAN.** No revision needed. Recommended anchor for §2 (Related Work) of Candidate A: "Yu et al.~\cite{muse2025} (MUSE) recently introduced MCTS-driven multi-turn red-teaming with a discrete action space of text-generation strategies (frame expansion, decomposition, perspective redirection) over dialogue contexts. While MUSE shares the meta-algorithm of UCB1-driven tree search on a frozen target model, its substrate is text-space dialogue trajectories with judge-based binary rewards. Our continuous-latent KV-cache MCTS extends this paradigm to the activation level, replacing the discrete action space with arithmetic mutations of K/V tensors and the binary judge with per-step telemetry signals."

---

## 3. Snell-specific fatal-finding analysis (separate subsection)

This was the highest-priority unknown going into Track B-delta. The original Track B audit explicitly flagged: "If Snell turns out to argue something like 'test-time compute scaling is fundamentally discrete because continuous latent search is intractable on frozen models without specialized training', this is a late-breaking fatal finding that escalates Candidate A from GO-with-1-nuance to PIVOT-to-Wildcard."

**Resolution: NO FATAL FINDING.** Three independent lines of evidence support the negative:

**Evidence 1 — Snell's abstraction is method-agnostic.**

§2 unifies test-time compute under "modifying the model's predicted distribution adaptively at test-time", with two knobs (input-level token augmentation; output-level candidate surgery via verifier). This is a **functional** definition. Continuous-latent KV-cache mutation modifies the predicted distribution adaptively at test-time — it instantiates Snell's framework via a third knob (intermediate-state surgery) that Snell did not enumerate but did not exclude.

**Evidence 2 — Snell explicitly frames MCTS-style search as part of the family.**

§5.1 line 165: "lookahead search is a special case of MCTS, wherein the stochastic elements of MCTS, designed to facilitate exploration, are removed since the PRM is already trained and is frozen. These stochastic elements are largely useful for learning the value function (which we've already learned with our PRM), but less useful at test-time when we want to exploit rather than explore. Therefore, lookahead search is largely representative of how MCTS-style methods would be applied at test-time."

This is structurally important. Snell's framing is: discrete-token tree search with a frozen PRM doesn't need stochastic exploration because the value function is already learned. **Our regime is different**: we have a per-step telemetry signal but no learned value function over latent geometry — we genuinely need exploration. Snell's footnote actually implies that *if* the value function is **not** pre-learned, full MCTS exploration is justified. This is rhetorically supportive of our continuous-latent MCTS, not opposed.

**Evidence 3 — Snell's §7 limitations actively invite our extension.**

"Future work should work to develop new ways of using test-time compute which can circumvent this limitation [diminishing returns on hard problems]." Their diminishing-returns result is on text-space methods. The natural reading is that token-space search saturates, motivating exploration of alternative substrates. Continuous-latent search is exactly such an alternative substrate.

**One additional note on capability-specific finetuning:** Snell footnote in §1 (line 38) says revision-as-proposal-distribution required capability-specific finetuning ("Capability-specific finetuning is necessary to induce revision and verification capabilities into the base model on MATH since these capabilities are absent even in strong proprietary LLMs"). This is a constraint on the revision-side of their study, NOT on the verifier-search side. Their best-of-N and beam-search experiments use the few-shot prompted base model directly (Appendix prompting note + §5.1 "We optimize the PRM at test time via search methods. We study three search approaches that sample outputs from a few-shot prompted base LLM"). Our paper operates on a frozen base with a separate reward — this matches Snell's verifier-search regime, not the revision regime. **No fine-tuning constraint applies to our extension.**

**Conclusion:** Snell is the strongest possible anchor for Candidate A. Far from being a fatal finding, the paper structurally invites continuous-latent search as a natural extension within its own taxonomy. The original Track B's risk hypothesis is fully discharged.

---

## 4. Updated cumulative pile-up

**Original Track B count:** 1 NUANCED (RepE/certainty under Candidate A swap) + 0 MISMATCHED + 0 FATAL.

**Track A status (per memory + recast plan):** Track A is doing per-layer H-Neuron calibration + certainty-probe validation. The original Track B audit flagged that "if Track A returns clean signal magnitude + correctness correlation, the methodological extension is justified" — i.e., the RepE NUANCE retires once Track A confirms certainty extractability. Per the user's task spec referring to RepE certainty as "now RETIRED by Track A", this NUANCE is no longer counted.

**Track B-delta (this audit):** 0 NUANCED + 0 MISMATCHED across the four newly-downloaded papers.

**Cumulative total post-delta:** **0 NUANCED + 0 MISMATCHED + 0 FATAL.**

This is below the YELLOW threshold (2) and well below NO-GO (3+). Track B's overall Gate-1 contribution is unambiguously GREEN.

| Source | NUANCED | MISMATCHED | FATAL |
|---|---|---|---|
| Original Track B (22 citations) | 1 (RepE/Candidate A) — retired by Track A | 0 | 0 |
| Track B-delta (4 newly-downloaded) | 0 | 0 | 0 |
| **Cumulative post-Track-A retirement** | **0** | **0** | **0** |

---

## 5. Verdict for Gate 1 G1.3 (≤2 NUANCED total)

**PASS WITH MARGIN.** Cumulative count is 0/2. Even if Track A's RepE retirement is contested and the original NUANCE is reinstated, the count stays at 1/2 — still PASS. The four delta papers contributed zero new nuance, so the gate is robust to any single Track A reversal.

---

## 6. Verdict for Gate 2 G2.5 (no late-arriving contradictions)

**PASS.** None of the four newly-downloaded papers contradict any load-bearing claim in either premortem-v2.tex or the Candidate A framing. Specifically:
- Snell does not contradict the test-time-compute scaling framing — they anchor it.
- Hasani does not contradict the System-2 framing — they provide the operational definition we use.
- Stoehr does not contradict the activation-steering differentiator — they provide the static-steering baseline against which our search-based approach is contrasted.
- MUSE does not contradict the discrete-vs-continuous MCTS differentiator — they are exactly the discrete-text-space precedent we extend.

No late-arriving contradictions surfaced.

---

## 7. Bibliography of files inspected (delta)

- `docs/NeurIPS/Citations/arXiv-2408.03314v1/template_content.tex` (Snell et al. 2024) — full intro through discussion read; abstract + §1-§3 (Unified Perspective + Compute-Optimal Strategy) + §5.1 (Search Methods) + §7 (Discussion) verified verbatim
- `docs/NeurIPS/Citations/arXiv-2601.02989v2/acl_latex.tex` (Hasani et al. 2025) — abstract + §1 introduction + §2 methodology + §3 behavioral results read; "System-2 as operational abstraction" + "no fine-tuning" + "frozen models" claims verified verbatim
- `docs/NeurIPS/Citations/2024.findings-emnlp.479.pdf` (Stoehr et al. 2024) — abstract + §1 + §3 (Activation-Level Interventions + Eq. 4-5) + §5 (DYNSCALAR extension) + §6.2 (continuous-vs-discrete framing) + §7 conclusion + Limitations all read verbatim
- `docs/NeurIPS/Citations/2025.emnlp-main.1080.pdf` (Yu et al. 2025 — MUSE) — abstract + §1 + §2 related work + §3 problem definition + §4.1-§4.2 methodology + Algorithm 1 + §5 experimental setup + Limitations all read verbatim

**Time on task:** ~2.5h wall-clock.
**Strongest finding:** Snell fatal-check decisively NEGATIVE; Snell actively supports our framing rather than excluding it.

---

*End of Track B-delta. Recommendation to Session A: GO on Gate 1 G1.3 and Gate 2 G2.5. The four newly-downloaded papers integrate cleanly into Candidate A's citation plan as anchors and differentiators. No revisions to existing paper text required; only additive citations and 1-2 sentence framing anchors per the suggestions in §2.1-§2.4 above.*
