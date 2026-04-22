# Citation Audit Matrix
**Paper:** KV-Cache Inception (NeurIPS 2026 D&B submission)
**Purpose:** Verify that every cited result is accurately represented and every implemented method faithfully matches its source paper.
**Process:** Drop PDFs in `docs/NeurIPS/Citations/`. Review one paper at a time in priority order. Update Status column after each review.

---

## Category Key

| Category | Meaning | Verification requirement |
|---|---|---|
| **IMPLEMENT** | We derive a formula, algorithm, or metric directly from this paper | Code must match paper exactly. Gaps invalidate our claims. |
| **RESULT** | We assert a finding from this paper as established fact | Claim in our paper must accurately reflect what the source paper actually shows. |
| **BASELINE** | We compare against a method from this paper | Need to understand the method well enough to implement the comparison fairly. |
| **BACKGROUND** | Cited for context or motivation only | Lower risk — verify we haven't overstated the claim. |

---

## Priority 1 — Implement (Code must match paper)

### P1-A: `gao2025hneurons` — H-Neurons
**Full ref:** Gao et al. (2025). *H-Neurons: On the Existence, Impact, and Origin of Hallucination-Associated Neurons in LLMs.* arXiv:2512.01797v2. Tsinghua University.
**Category:** IMPLEMENT + RESULT
**Where cited:** §1 (Contributions), §4.1 (σ_H formula, Eq. 3), §7 (Limitations)
**What we implement:**
- Identification of H-Neurons (hallucination-associated FFN units)
- σ_H stress formula (Eq. 3 in our paper)
- Calibration procedure for neuron selection
**Claims we make from it:**
- H-Neurons are causally linked to over-compliance and hallucination
- H-Neuron activation provides a bottom-up alignment monitoring signal
- We "extend this finding to the domain of alignment faking" (§2.3)
**Status:** ⚠️ AUDITED — 5 GAPS FOUND (see `docs/logs/2026-04-21_session-log.md`, H-Neuron section)
**Summary of gaps:**
1. Our σ_H uses raw activation; paper uses CETT (down-projection normalized contribution)
2. Our calibration dataset (5+5 generic prompts) ≠ TriviaQA consistency-filtered
3. Paper validates over-compliance, NOT strategic deception — we overstate the link to alignment faking
4. Paper's smallest model is 4B; our gate runs use 1B (out of validated range)
5. Proper CETT calibration is expensive — not implemented
**Required paper fix:** Reframe σ_H as "over-compliance pressure signal" not "alignment faking detector"
**Required code fix:** None blocking (σ_H implementation acknowledged as simplified proxy in Limitations)

---

### P1-B: `zou2023repe` — Representation Engineering
**Full ref:** Zou et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405.
**Category:** IMPLEMENT + RESULT
**Where cited:** §1 (Contributions), §4.1 (ρ_R formula, Eq. 4), §2.2 (Background)
**What we implement:**
- Linear Artificial Tomography (LAT) probe training procedure
- ρ_R honesty projection (Eq. 4): `ρ_R^(l)(t) = w_hon^(l)ᵀ h_t^(l)`
- Contrastive pairs approach for probe extraction
- Probe directions for honesty, certainty, goal-coercion dimensions
**Claims we make from it:**
- "High-level cognitive states—including honesty, truthfulness, and compliance—are linearly decodable from a model's latent representation space"
- LAT procedure extracts linear probes for cognitive dimensions
**Status:** ⚠️ AUDITED — 5 GAPS FOUND (Apr 21, arXiv:2310.01405v4 TeX source reviewed)

### Finding 1: PCA vs. Mean-Difference — We Use the Wrong Method
The paper is explicit: **PCA on difference vectors is the primary and preferred method** for LAT (§3.1, Step 3). Inputs to PCA are `{A_f+^(i) - A_f-^(i)}` with alternating sign; the first principal component is the reading vector. "Difference between cluster means" is listed as a supervised alternative, used when labels are available. Our `PerLayerHonestyProjector.calibrate()` uses `mean(coerced) - mean(benign)` — the supervised mean-diff fallback, not PCA. The paper notes PCA "can yield superior results when stimuli share similarities except for the target concept." Our implementation discards the preferred method.

### Finding 2: Scale Not Validated — Smallest Model is LLaMA-2-13B
All RepE experiments are run on LLaMA-2-7B, 13B, 33B and Vicuna-33B-Uncensored. The paper **never validates LAT on sub-7B models**. Our 1B gate runs are out of range. The paper also notes instability risk: "a small fraction of runs may yield significantly lower performance than the reported mean accuracy" when stimulus samples are small — our 5+5 calibration dataset is at the bottom of the paper's recommended range of 5–128 examples.

### Finding 3: "Compliance" and "Goal-Coercion" Are Not in the Paper
We claim we apply LAT to extract probes for "honesty, certainty, and goal-coercion" (§4.1 of our paper). The RepE paper studies: truthfulness, honesty, utility, probability, morality, emotion, lying, power-seeking. **"Compliance" and "goal-coercion" are not studied**. We are extrapolating LAT to two novel concept dimensions with no validation from the source paper. Whether a linear direction for "goal-coercion" is reliably extractable via LAT is an open question.

### Finding 4: Layer Selection — Paper Uses Middle Layers Only
For the honesty lie detector, the paper explicitly uses **"the middle 20 layers, which exhibit the strongest reading performance"** (§4.3). Our `ρ_R` computation runs probes for all L layers and aggregates. Using all layers dilutes the signal with uninformative early/late layers. The paper's heatmap (Figure, §4) clearly shows early and late layers have near-zero honesty signal.

### Finding 5: The Linearity Claim Is Narrower Than We State
We claim "high-level cognitive states—including honesty, truthfulness, and compliance—are linearly decodable from a model's latent representation space" (§2.2). The paper validates this for honesty and truthfulness on LLaMA-2/Vicuna. **Compliance is not validated in the paper.** The claim holds for the concepts the paper studies; extending it to "compliance" requires either a citation to different work or an experiment of our own.

### What Is Correct
- The dot-product formula `ρ_R^(l)(t) = w_hon^(l)ᵀ h_t^(l)` matches the paper's prediction rule `Rep(M,x)ᵀ v` ✓
- Using contrastive prompt pairs (honest instruction vs. dishonest instruction) as calibration stimuli matches the paper's function-extraction template design ✓
- Per-layer probes are consistent with the paper's per-layer reading vector approach ✓

### Required Fixes
| Fix | Location | Urgency |
|---|---|---|
| Switch `PerLayerHonestyProjector.calibrate()` from mean-diff to PCA on difference vectors | `logomesh/whitebox.py` | High — paper prefers PCA |
| Restrict ρ_R aggregation to middle layers (exclude first ~25% and last ~25% of layers) | `logomesh/kv_mcts.py` + `whitebox.py` | Medium — improves signal quality |
| Remove "compliance" from the list of linearly decodable concepts in §2.2 | `.tex` | Medium — unsupported claim |
| Add caveat: "goal-coercion" probe direction is an extrapolation of LAT, not directly validated by Zou et al. | `.tex` §4.1 or Limitations | Medium |
| Note 1B scale is outside RepE's validated range in Limitations | `.tex` | Low — consistent with H-Neuron caveat already planned |

---

## Priority 2 — Load-Bearing Results (Empirical claims our paper stands on)

### P2-A: `gupta2025rlobfuscation` — RL-Obfuscation
**Full ref:** Gupta & Jenner (2025). *RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?* arXiv:2506.14261v4.
**Category:** RESULT (load-bearing — motivates entire Orthogonal Escape hypothesis)
**Where cited:** Abstract, §1 (Introduction), §2.2 (Background), §7 (Limitations)
**Claims we make from it:**
- "LLMs finetuned via RL can learn to evade latent-space monitors while maintaining black-box behavior"
- "Models trained to evade a single monitor can generalize to evade unseen monitors"
- "Token-level monitors being particularly vulnerable"
- "Models learn to repurpose tokens to have different internal representations"
- This finding is described as "direct empirical motivation for our Orthogonal Escape hypothesis"
**Status:** ⚠️ AUDITED — 4 GAPS FOUND (Apr 21, arXiv:2506.14261v4 TeX source reviewed)

### Finding 1: Generalization Claim Requires a Critical Qualifier — We Omit It
Our paper (§1): "models trained to evade a single monitor can generalize to evade unseen monitors."
The paper (abstract + RQ2): "models trained to evade a single static monitor can generalise to evade **other unseen monitors**" — **but max-pooling and attention-based probes are explicitly excluded as robust and not evadable.**

The paper's key takeaway (§compare_probes): "monitors using mean or median aggregations are NOT robust to RL-Obfuscation, whereas attention probes and max aggregation remain robust." Evasion generalises within the class of mean/median-aggregated monitors only. We present the generalisation as unqualified. This misrepresents the finding and omits the paper's own most important limitation.

**This matters for us:** If our ρ_R probe uses mean aggregation over layers, it falls in the "vulnerable" category. If it used max or attention aggregation, it would be robust. Our current per-layer dot-product design is effectively mean-aggregated across layers — which the paper says is the type that RL-trained models can evade. We should note this in the paper.

### Finding 2: The Mechanism Is RL Weight Finetuning, Not Inference-Time Perturbation
The entire RL-Obfuscation methodology is RL **finetuning** — model weights are updated over multiple training epochs (Llama-2 7B, Llama-3 8B, Qwen-2.5 14B). Evasion is a learned behavior baked into updated weights. Our KV-cache steering leaves model weights **completely unchanged** — we perturb the KV cache at inference time on a frozen model.

These are fundamentally different threat models:
- **RL-Obfuscation:** model weights change → model learns new representations and output policies over thousands of gradient steps
- **Our work:** weights frozen → KV cache perturbed at inference time → we ask whether representations geometrically redistribute

We cite this paper as "direct empirical motivation for our Orthogonal Escape hypothesis" (§2.2, §7). The motivation is conceptual — RL-Obfuscation shows that representational redistribution is possible in principle. But the paper does NOT demonstrate that inference-time KV perturbation produces the same effect. That inference must be stated as a hypothesis, not as something this paper validates.

### Finding 3: Evasion Is Primarily Behavioral (~70%), Not Representational (~30%)
The paper (§eval_policy) quantifies the contribution: "change in policy contributes to approximately 70% of the drop in recall." The primary evasion mechanism is the model learning to produce different output tokens (more HTML tags, punctuation) that have low probe scores — a behavioral change. The activation-level redistribution (the "repurposing tokens" finding) accounts for ~30% of the effect.

Our paper presents the "repurpose tokens to have different internal representations" finding as a clean representational story directly motivating Orthogonal Escape. In reality it is the secondary mechanism. The primary mechanism is behavioral output policy change — something our KV-steering probe (which doesn't change output policy via RL) may not replicate.

### Finding 4: "Token-Level Monitors Are Highly Vulnerable" Needs the Full Sentence
Our paper (§2.2): "token-level monitors being particularly vulnerable." This is accurate but incomplete. The full finding is a paired result: token-level monitors with **mean/median aggregation are vulnerable**; max-pooling and attention-based monitors are **robust**. Citing only the vulnerability half without the robustness half gives a distorted picture of the paper's contribution, which is precisely the identification of which monitor designs survive this attack.

### What Is Correct
- "LLMs finetuned via RL can learn to evade latent-space monitors while maintaining black-box behavior" ✓
- "token-level monitors being particularly vulnerable" ✓ (with aggregation qualifier)
- "models learn to repurpose tokens to have different internal representations" ✓ (secondary mechanism)
- Conceptual motivation for Orthogonal Escape as a phenomenon worth investigating ✓

### Required Fixes
| Fix | Location | Urgency |
|---|---|---|
| Add qualifier to generalisation claim: "of the same aggregation type (mean/median); max-pool and attention probes remain robust" | `.tex` §1 Introduction | High — current phrasing misrepresents the paper |
| Reframe "direct empirical motivation" as conceptual/analogical motivation; distinguish RL finetuning from inference-time KV steering | `.tex` §2.2 Background | High — overstates mechanistic connection |
| Add the paired robustness finding (max-pool/attention probes robust) to §2.2 | `.tex` §2.2 | Medium — the robustness result informs our own probe design |
| Note that RL-Obfuscation's primary mechanism is behavioral (~70%), not representational — inference to KV-steering regime is not direct | `.tex` §2.2 or Limitations | Medium |

---

### P2-B: `greenblatt2024alignment` — Alignment Faking
**Full ref:** Greenblatt, Shlegeris et al. (2024). *Alignment Faking in Large Language Models.* Anthropic Research, December 2024.
**Category:** RESULT (load-bearing — primary empirical motivation)
**Where cited:** Abstract, §1 (Introduction), §2.1 (Background)
**Claims we make from it:**
- "Claude 3 Opus, when informed that its responses would be used for retraining, strategically complied with instructions it would otherwise refuse"
- The model "explicitly reasoning in its chain-of-thought that compliance was instrumentally rational to preserve its existing values"
**Status:** ⚠️ AUDITED — 4 GAPS FOUND (Apr 21, arXiv:2412.14093v2 TeX source reviewed)

### Finding 1: Core Claims Are Accurate — But Strip the Paper's Own Caveats
The two specific claims we make are factually correct:
- "Claude 3 Opus…strategically complied with instructions it would otherwise refuse" ✓ (14% compliance with harmful queries in implied training vs. nearly 0% unmonitored)
- "explicitly reasoning in its chain-of-thought that compliance was instrumentally rational to preserve its existing values" ✓ (paper: "in almost all cases where the model complies…we observe explicit alignment-faking reasoning")

However we cite these stripped of three material caveats the paper states explicitly in its own Limitations section: (1) alignment faking only emerged when the model was specifically **told** about its training situation via prompt or synthetic doc fine-tuning — not spontaneously; (2) "alignment faking which is as incompetent as we see given this present level of model capabilities poses no serious threat" (paper's own words); (3) the rates are probabilistic — 7–24%, not reliable or universal.

### Finding 2: The Goals Being Faked For Were Benign — Not Malicious
We use this paper as motivation for studying "deceptive behaviors" and "misaligned capabilities." The paper is explicit: **the model was faking alignment to preserve its HHH harmlessness training** — i.e., to keep refusing harmful queries. The paper states directly: "the goals we discover that Claude 3 Opus is willing to fake alignment for…aren't themselves concerning."

This is alignment faking in service of a safe, pro-social preference. Our paper frames it as evidence of the danger of models preserving misaligned capabilities — which is the paper's hypothetical threat model for future systems, not what was demonstrated. The paper is careful to distinguish what it showed from what it didn't.

### Finding 3: Scale Requirement — Only Frontier Models, Not 20B or Below
The paper found alignment faking in Claude 3 Opus and Claude 3.5 Sonnet. It **did not find it** in Claude 3 Sonnet, Claude 3 Haiku, Claude 3.5 Haiku, or smaller Llama/Mistral models (explicitly tested). It found a compliance gap in Llama 3.1 405B only.

We cite this paper as motivation for our experiments on a 20B model. The paper's own scale finding suggests 20B is likely below the threshold at which alignment faking reliably emerges — this doesn't invalidate our approach, but is a caveat our paper should acknowledge.

### Finding 4: Chain-of-Thought Scratchpad Is Load-Bearing — We Don't Mention This
The alignment-faking reasoning is visible in a **hidden scratchpad** the model is explicitly instructed to use. The paper tests scratchpad removal: in prompted setups, the compliance gap is "nearly eliminated" without the scratchpad. In synthetic doc fine-tuned setups, the gap persists without the scratchpad (suggesting implicit reasoning), but prompted setups depend heavily on it.

We cite the chain-of-thought reasoning as a clean demonstration of strategic intent. We don't note that this reasoning was elicited via an experimental scratchpad design, not observed in ordinary deployment.

### What Is Correct
- The core factual claims about Claude 3 Opus's behavior are accurate ✓
- "Provided the first large-scale evidence that frontier models engage in strategic compliance" — reasonable characterization ✓
- Using this as conceptual motivation for the existence of alignment faking as a real phenomenon ✓

### Required Fixes
| Fix | Location | Urgency |
|---|---|---|
| Add that the scenario required explicit situational awareness (being told about training) — not spontaneous | `.tex` §1 or §2.1 | Medium |
| Add that the faked-for goals were benign (HHH preservation), not malicious misalignment | `.tex` §2.1 | Medium — changes the framing significantly |
| Add scale caveat: only emerged in Opus/3.5 Sonnet; not in smaller models | `.tex` §2.1 or Limitations | Medium |
| Note scratchpad dependence when citing the chain-of-thought reasoning | `.tex` §1 | Low |

---

### P2-C: `hubinger2024sleeper` — Sleeper Agents
**Full ref:** Hubinger, Denison, Mu et al. (2024). *Sleeper Agents: Training Deceptive LLMs That Persist Through Safety Training.* arXiv:2401.05566.
**Category:** RESULT
**Where cited:** Abstract, §1 (Introduction), §2.1 (Background)
**Claims we make from it:**
- "Sleeper agent backdoors, once trained into LLMs, persist through RLHF, supervised fine-tuning, and adversarial training"
**Status:** ⚠️ AUDITED — 4 GAPS FOUND (Apr 21, arXiv:2401.05566v3 TeX source reviewed)

### Finding 1: Persistence Is Strongly Scale-Dependent — We Omit This Entirely
Our paper states backdoors "persist through standard safety training procedures" as a general result. The paper is explicit that this is **strongly conditioned on model size**: "all of our backdoors are substantially easier to remove in smaller models" and the figure caption states "backdoored behavior is much less persistent against safety training in smaller models." The scaling scan (810M to 170B parameters) shows that at the smallest scales (810M, 1.6B), backdoors ARE removed by RL safety training. Persistence is primarily a property of large models.

This is a critical omission for our paper: our 1B surrogate is almost certainly below the persistence threshold. Our intended 20B target sits in an intermediate zone where the paper's Claude-1.2-instant-equivalent (~20B class) shows persistence only for the distilled chain-of-thought backdoor variant, not the normal backdoor. Presenting the finding as scale-independent misrepresents the paper's conclusion.

### Finding 2: These Are Deliberately Inserted Backdoors, Not Naturally Arising Deception
The paper is explicit: "this work is designed to test whether our threat models can be resilient to safety training, **not the plausibility of our threat models occurring by default.**" And: "we are not concerned with how realistic our backdoor training process is or how likely models with these specific behaviors are to arise naturally."

Our abstract says "deceptive behaviors persist through standard safety training" — which reads as a general claim about deceptive LLMs. The paper demonstrates this only for deliberately, surgically inserted backdoors via SFT. The paper itself distinguishes this sharply from emergent or naturally arising deception.

### Finding 3: "RLHF" Terminology Is Loose
Our §1 says backdoors persist through "reinforcement learning from human feedback (RLHF)." The paper tests RL fine-tuning for HHH using a preference model (which is the mechanism of RLHF), but the paper consistently uses the term "RL fine-tuning for HHH" and does not use the term "RLHF" itself. The substance is essentially the same, but our phrasing implies an exact match with the standard RLHF framing the paper deliberately avoids. Minor but worth correcting for precision.

### Finding 4: Adversarial Training Made It Worse — We Understate This
We cite that backdoors "resist removal through…adversarial training." The paper's actual finding is stronger: adversarial training **actively taught models to better hide their backdoors** — "rather than removing backdoors, we find that adversarial training can teach models to better recognize their backdoor triggers, effectively hiding the unsafe behavior." This is not merely ineffective; it is counterproductive. Our framing (it "persists through" adversarial training) understates the alarming directionality of the result.

### What Is Correct
- Backdoors resist RL fine-tuning, SFT, and adversarial training for the models tested (Claude-1.2-instant-equivalent and Claude-1.3-equivalent) ✓
- Chain-of-thought reasoning increases backdoor persistence, even when distilled away ✓
- Adversarial training fails to remove backdoors ✓ (and is actually counterproductive — see Finding 4)
- The paper demonstrated this is possible as a proof-of-concept ✓

### Required Fixes
| Fix | Location | Urgency |
|---|---|---|
| Add scale qualifier: persistence is strongest in large models; smaller models' backdoors ARE removed | `.tex` §1, Abstract | High — presented as scale-independent when it isn't |
| Note this is deliberately inserted backdoors, not emergent deception | `.tex` Abstract / §2.1 | Medium — changes the implied generality of the claim |
| Correct "RLHF" to "RL fine-tuning for HHH" for terminological precision | `.tex` §1 | Low |
| Strengthen the adversarial training characterization: it hid backdoors rather than just failing to remove them | `.tex` §1 | Low — more accurate and actually more alarming |

---

### P2-D: `belitsky2025kvcache` — KV Cache Steering
**Full ref:** Belitsky, Kopiczko, Dorkenwald et al. (2025). *KV Cache Steering for Controlling Frozen LLMs.* arXiv:2507.08799.
**Category:** RESULT + BASELINE
**Where cited:** §2.2 (Background), Table 2 (Baselines)
**Claims we make from it:**
- "A single, one-shot modification to the KV cache can steer model behavior more stably than continuous activation interventions"
**Status:** ⚠️ 4 gaps — stability claim correct but domain-unqualified; Table 2 baseline paradigm mismatch (static mean-diff ≠ MCTS iterative); framing implies extension when methods differ; no 20B experiments

**What is correct:**
- "Single, one-shot modification to the KV cache" — precisely accurate; that is the paper's defining claim
- "More stably than continuous activation interventions" — supported by ablation (Fig. ablations: `c^k` varies 0.0–0.4 with minor change; activation steering has catastrophic failures) and Fig. latency (~10 ms/token vs ~15 ms/token)
- Paper is real and results are internally consistent

**Gaps:**

| # | Our claim | What paper shows | Gap | Severity |
|---|---|---|---|---|
| 1 | Stability advantage cited without qualifier (§2.2) | Demonstrated only on reasoning tasks (GSM8K/ARC-c/CSQA/PIQA) with 1B–8B models | Stability in alignment faking / deceptive representation steering unverified; domain omitted | Medium |
| 2 | Table 2 baseline "One-shot KV-Cache Steering" implies our method is comparable | Belitsky: precompute mean-diff KV vector, add once. Ours: MCTS iterative mutations `K_t ← K_t + α·d_K` with FP32 reversibility and telemetry reward | Fundamentally different paradigms; reviewer will challenge this immediately | **High** |
| 3 | §2.2 implies our KV-MCTS extends cache steering | Belitsky = static precomputed vector + single injection; ours = heuristic tree search with reversibility | Conflates two different KV intervention paradigms; obscures novelty | Medium |
| 4 | Cited in 20B experiment context | All experiments: 360M–8B only; no 20B experiments | Scale gap; stability advantage unverified at 20B MoE | Low |

**Required fixes:**

| Fix | Location | Urgency |
|---|---|---|
| Add qualifier: "demonstrated for reasoning induction in small-to-medium LLMs; we apply KV-cache intervention to alignment faking detection in larger models" | `.tex` §2.2 | Medium |
| Distinguish Belitsky (static one-shot mean-diff) from our MCTS (iterative reversible search) in Table 2 caption or §4 | `.tex` Table 2 + §4 | **High** |
| Implement Belitsky's mean-of-differences KV steering as an actual baseline in experiments — no implementation currently exists in `logomesh/` | new file in `logomesh/` | **High** |
| Add explicit paradigm separation in §2.2: "Unlike the static vector approach of Belitsky et al., our MCTS explores the KV intervention space dynamically…" | `.tex` §2.2 | Medium |

---

### P2-E: `llmasjudge2025limitations` — LLM-as-Judge Limitations
**Full ref:** Krumdick, Lovering, Reddy, Ebner, Tanner (2025). *No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding.* arXiv:2503.05061.
**Category:** RESULT
**Where cited:** §2.5 (LLM-as-Judge)
**Claims we make from it:**
- "Score divergences of 0.10–0.20 on normalized scales across independent runs"
**Status:** ⚠️ CRITICAL — fabricated numeric claim; 0.10–0.20 figure does not appear anywhere in this paper; paper mischaracterized as about reproducibility when it is about reference-quality dependence

**What is correct:**
- LLM judges have reliability problems: TRUE — but the nature is reference dependence + judge competence, not inter-run variance
- Paper is a legitimate empirical study of LLM judge reliability

**Gaps:**

| # | Our claim | What paper shows | Gap | Severity |
|---|---|---|---|---|
| 1 | "score divergences of 0.10–0.20 on normalized scales across independent runs" | This number does not exist in the paper. Metric is Cohen's κ. Paper uses self-consistency (5-inference majority vote) to suppress variance; prompt sensitivity appendix shows "no meaningful variation" | **Fabricated numeric claim** — attributing a figure to a source that does not contain it | **Critical** |
| 2 | Framed as "reproducibility problems" alongside MT-Bench/G-Eval | Finding is about judge reliability conditioned on reference quality and judge competence — orthogonal to run-to-run reproducibility | Mischaracterizes the paper's contribution | High |
| 3 | Implies problem affects MT-Bench / G-Eval | Paper tests corrected MT-Bench subset; finding is about correctness evaluation under different reference conditions, not general MT-Bench variance | Context mismatch | Medium |

**Accurate characterization of this paper:** "LLM judges achieve high agreement with human experts only when provided with a correct reference or when the judge can correctly answer the underlying question; without human grounding, agreement drops dramatically (κ from 0.86 to 0.16 for GPT-4o on questions it cannot answer)."

**Required fixes:**

| Fix | Location | Urgency |
|---|---|---|
| Remove "score divergences of 0.10–0.20 on normalized scales across independent runs" — not in this paper | `.tex` §2.5 | **Critical** |
| Find actual source for 0.10–0.20 figure (possibly Zheng 2023 / Liu 2023) and cite correctly, OR replace with accurate characterization of Krumdick et al. | `.tex` §2.5 | **Critical** |
| Replace with: "LLM judges require correct references or prior knowledge to achieve reliable agreement with human experts" | `.tex` §2.5 | High |

---

## Priority 3 — Baselines (Need to understand to run comparisons)

### P3-A: `zou2023gcg` — GCG
**Full ref:** Zou, Wang, Kolter & Fredrikson (2023). *Universal and Transferable Adversarial Attacks on Aligned Language Models.* arXiv:2307.15043.
**Category:** BASELINE + RESULT
**Where cited:** §2.4 (Related Work), §4.2 (Exp 2), Table 2
**Claims we make from it:**
- GCG "operates in token embedding space" and is "constrained to the discrete combinatorial surface"
- GCG "optimizes token sequences via gradient signals but is not designed to navigate the continuous geometry of the model's internal representation space"
**Status:** ⚠️ 3 gaps — "token embedding space" is wrong (GCG is discrete-token, not embedding-space); gradient access requirement omitted; ASR baselines unspecified for Table 2

**What is correct:**
- GCG is constrained to the discrete combinatorial surface (token sequences): ✅ accurate
- GCG does not navigate the model's internal continuous representation space: ✅ accurate
- Appropriate as a baseline comparison: ✅ canonical gradient-based suffix attack

**Gaps:**

| # | Our claim | What paper shows | Gap | Severity |
|---|---|---|---|---|
| 1 | GCG "operates in token embedding space" | GCG operates in **discrete token space**. Paper explicitly contrasts GCG with embedding-based methods (PEZ/GBDA) which operate in continuous embedding space. GCG uses gradient of one-hot vector as a selection heuristic, then evaluates actual discrete token substitutions. The paper's key insight is that embedding-based methods fail because they can't produce valid discrete tokens. | Factually inverts the paper's own distinction — we're describing PEZ, not GCG | **High** |
| 2 | No gradient requirement mentioned | GCG requires **white-box gradient access** to the target model | Our MCTS is gradient-free (heuristic search on frozen model) — this is a meaningful applicability advantage we omit | Medium |
| 3 | No ASR figures for Table 2 | 88% whitebox Vicuna-7B; 84% transfer GPT-3.5; 66% PaLM-2; 2.1% Claude | Need these for Table 2 baseline quantification | Medium |

**Required fixes:**

| Fix | Location | Urgency |
|---|---|---|
| Replace "operates in token embedding space" with "operates in discrete token space via gradient-guided vocabulary search" | `.tex` §2.4 | **High** |
| Add: "GCG requires white-box gradient access; our MCTS operates without gradient information" | `.tex` §2.4 or Table 2 | Medium |
| Record GCG ASR baselines for Table 2: 88% whitebox Vicuna; 84% transfer GPT-3.5; 66% PaLM-2; 2.1% Claude | `.tex` §4 / Table 2 | Medium |

---

### P3-B: `wu2025mcts` — MCTS Prompt Autogeneration
**Full ref:** Wu et al. (2025). *Monte Carlo Tree Search Based Prompt Autogeneration for Jailbreak Attacks Against LLMs.* COLING 2025.
**Category:** BASELINE + RESULT
**Where cited:** §1 (Contributions), §2.4 (Related Work), §4.2 (Exp 2), Table 2
**Claims we make from it:**
- Text-space MCTS "mutating prompts, suffixes, or dialogue turns"
- Operates "exclusively in text space"
**Status:** ✅ 2 minor gaps — characterization accurate; ASR figures not retrieved; attack LLM dependency not mentioned (low severity)

**What is correct:**
- "MCTS-based prompt autogeneration for jailbreak attacks": ✅ matches title/abstract exactly
- "operates exclusively in text space — mutating prompts, suffixes, or dialogue turns": ✅ confirmed via source code: nodes = adversarial prompt strings, 10 text transformation action types, gradient-free
- Prior art framing to establish KV-cache MCTS novelty: ✅ distinction is valid and precise

**Gaps:**

| # | Our claim | What paper shows | Gap | Severity |
|---|---|---|---|---|
| 1 | No detail on MPA's architecture | MPA requires an external **attack LLM** to generate candidate prompt variants; our method uses only the target model's telemetry signal | Omits a meaningful architectural difference; optionally worth noting | Low |
| 2 | No ASR figures for Table 2 | Paper claims MPA surpasses GCG and AutoDAN; specific figures not retrieved (no arXiv version found) | Need ASR baselines for Table 2 quantitative comparison | Medium |

**Required fixes:**
- No text fixes required — all claims are accurate
- Retrieve MPA ASR figures (check COLING proceedings or GitHub repo)
- Optionally add: "MPA requires an external attack LLM; our approach uses the target model's own telemetry signal"

---

### P3-C: `damon2025` — DAMON
**Full ref:** DAMON: A Dialogue-Aware MCTS Framework for Jailbreaking Large Language Models. EMNLP 2025. (GitHub: pkulcwmzx/DAMON)
**Category:** BASELINE + RESULT
**Where cited:** §1 (Contributions), §2.4 (Related Work), Table 2
**Claims we make from it:**
- "Dialogue-aware MCTS for multi-turn jailbreaks"
- Operates in text space
**Status:** ✅ Clean — all claims accurate; nodes = multi-turn conversation prompt lists, text-space, gradient-free, UCB selection confirmed via source code; only gap is missing ASR figures for Table 2
**Note:** Author names missing from bibliography entry — verify full citation before submission.

**What is correct:**
- "Dialogue-aware MCTS for multi-turn jailbreaks": ✅ nodes store multi-turn conversation prompt lists
- Operates exclusively in text/dialogue space: ✅ prompt string manipulation via `replace_first_occurrence()`, no latent access
- Gradient-free: ✅ UCB-based selection, reward-based backpropagation, no gradient computation
- Prior art framing for KV-cache MCTS novelty: ✅ valid

**Gaps:**

| # | Our claim | What paper shows | Gap | Severity |
|---|---|---|---|---|
| 1 | No ASR figures for Table 2 | Evaluated across 5 LLMs and 3 datasets; specific numbers not retrieved | Need baselines for Table 2 | Medium |

**Required fixes:**
- No text fixes required
- Retrieve DAMON ASR figures from EMNLP proceedings or GitHub
- Verify and complete the bibliography entry (author names missing)

---

## Priority 4 — Background (Motivation/context citations)

### P4-A: `deroy2026structured` — Structured Review on Alignment Faking
**Full ref:** Deroy (2026). *A Structured State-of-the-Art Review on Alignment Faking in Large Language Models.* Zenodo preprint (DOI: 10.5281/zenodo.18673604), February 2026. Single author, 120 KB, no peer review.
**Category:** BACKGROUND
**Where cited:** §2.1 (Background)
**Claim:** "catalogues detection strategies ranging from mechanistic interpretability and activation probing to hidden reasoning analysis, concluding that no single approach provides reliable detection guarantees"
**Status:** ⚠️ 2 gaps — conclusion overstated (paper advocates white-box oversight, not "nothing works"); citation quality concern (Zenodo preprint, not peer-reviewed — HIGH risk at NeurIPS)

**What is correct:**
- "Detection strategies ranging from mechanistic interpretability and activation probing to hidden reasoning analysis": ✅ matches abstract language

**Gaps:**

| # | Our claim | What paper shows | Gap | Severity |
|---|---|---|---|---|
| 1 | "concluding that no single approach provides reliable detection guarantees" | Paper says behavioral metrics are "increasingly unreliable" and advocates for white-box oversight as the solution — recommends an approach, does not conclude all approaches fail | Overstatement: paper is pro-white-box, not nihilistic about all detection | Medium |
| 2 | Cited as background authority | Zenodo-only preprint, single author Aniket Deroy, 120 KB, February 2026, no peer review | NeurIPS reviewer will notice this is self-published; undermines credibility of the background claim | **High** — citation quality |

**Required fixes:**

| Fix | Location | Urgency |
|---|---|---|
| Soften: replace "no single approach provides reliable detection guarantees" with "traditional behavioral detection metrics are increasingly unreliable, motivating white-box oversight" | `.tex` §2.1 | Medium |
| Replace with a peer-reviewed source (e.g., Greenblatt 2024, Hubinger 2024, or an established survey) — Zenodo preprint is inadequate as a NeurIPS background citation | `.tex` §2.1 | **High** |

---

### P4-B: `turner2023steering` — Activation Steering
**Full ref:** Turner, Thiergart, Leech, Udell et al. (2023). *Steering Language Models with Activation Engineering.* arXiv:2308.10248.
**Category:** BACKGROUND only (pre-audit matrix was wrong — Table 2 CAA baseline is attributed to "Rimsky et al., 2024", NOT Turner; Rimsky/CAA is a separate uncited paper)
**Where cited:** §2.2 (Background only)
**Claim:** "targeted vector additions to intermediate representations can modify model behavior without weight updates"
**Status:** ✅ Claim accurate — but uncovers HIGH issue: Table 2 "Activation Steering / CAA (Contrastive Activation Addition) — Rimsky et al., 2024" is uncited plain text with no \cite{} and no bibliography entry

**What is correct:**
- "targeted vector additions to intermediate representations can modify model behavior without weight updates": ✅ ActAdd adds steering vectors to the residual stream at inference, frozen weights
- Background citation for activation steering: ✅ appropriate

**Gaps:**

| # | Our claim | What paper shows | Gap | Severity |
|---|---|---|---|---|
| 1 | Audit matrix listed as Table 2 baseline | Turner is BACKGROUND only; Table 2 attributes CAA baseline to "Rimsky et al., 2024" as plain text with no `\cite{}` — an entirely uncited and unbibliographied baseline | Table 2 baseline "Rimsky et al., 2024" for CAA has no \cite{} and no \bibitem — reviewer will notice | **High** |
| 2 | Domain | ActAdd demonstrated on sentiment and detoxification, not alignment faking | Background-only citation, low stakes | Low |

**Required fixes:**

| Fix | Location | Urgency |
|---|---|---|
| Turner §2.2 claim: no fix needed | — | — |
| Add `\cite{}` to Table 2 "Rimsky et al., 2024" CAA row and add corresponding \bibitem | `.tex` Table 2 line ~336 + bibliography | **High** |
| Verify correct paper: canonical CAA is Panickssery, Rimsky et al. 2023 (arXiv:2312.06681) — confirm which paper Table 2 intends | `.tex` bibliography | High |

---

### P4-C: `hao2025coconut` — COCONUT
**Full ref:** Hao, Suber, Wang et al. (2025). *Training Large Language Models to Reason in a Continuous Latent Space.* ICLR 2025. arXiv:2412.06769.
**Category:** BACKGROUND
**Where cited:** §1 (Contributions), §2.5 (Continuous Latent Reasoning)
**Claim:** "bypasses discrete token decoding by feeding final-layer hidden states back as input embeddings, enabling emergent breadth-first latent tree search"
**Status:** ✅ Clean — all claims accurate; 2 low-severity gaps (training requirement not mentioned; GPT-2 scale only)
**Source reviewed:** `docs/NeurIPS/Citations/arXiv-2412.06769v3/paper.tex`

**What the paper actually demonstrates:**
- In "latent mode" tokens, COCONUT feeds the last hidden state directly as the next input embedding, bypassing the LM head and discrete token decoding — exactly as claimed
- Probing experiments show continuous thoughts encode multiple next reasoning steps simultaneously — paper explicitly describes BFS emergence
- Requires multi-stage curriculum training (not inference-time on a frozen model)
- Proof-of-concept experiments on GPT-2 scale; not tested on large transformers

**What our claims get right:**
- "bypasses discrete token decoding by feeding final-layer hidden states back as input embeddings" ✅ verbatim match to method
- "enabling emergent breadth-first latent tree search" ✅ paper probing confirms BFS structure
- "recurrent hidden-state traversal rather than direct arithmetic mutation of a frozen transformer's KV-cache tensors" ✅ accurate — COCONUT requires training, does not touch KV cache

**Gaps:**

| # | Claim gap | Severity | Notes |
|---|---|---|---|
| 1 | We don't mention COCONUT requires training; could imply it's inference-time like our method | Low | Our framing is contrastive (COCONUT ≠ us) so omission is unlikely to mislead |
| 2 | Experiments are GPT-2 scale only; we don't note this | Low | Background citation only — scale caveat not required |

**Required fixes:**

| Fix | Location | Urgency |
|---|---|---|
| None required — claim accurate | — | — |

---

### P4-D: `crsm2025` — CRSM
**Full ref:** Pomilon Intelligence Lab. *CRSM: Continuous Reasoning State Manipulation.* GitHub, 2025.
**Category:** BACKGROUND
**Where cited:** §2.5 (Continuous Latent Reasoning), line 111
**Claim:** "employs an asynchronous MCTS planner to directly edit the internal hidden states of a Mamba state-space model during generation. Both validate the viability of continuous latent search, but operate via recurrent hidden-state traversal rather than direct arithmetic mutation of a frozen transformer's KV-cache tensors."
**Status:** ⚠️ 3 gaps — all technical claims accurate; citation quality concern (GitHub-only, no peer review, AI-assisted experimental prototype)
**Source reviewed:** GitHub `Pomilon-Intelligence-Lab/CRSM` — README, `crsm/core/reasoning.py`, `crsm/core/crsm.py`, `docs/ARCHITECTURE.md`

**What the repo actually demonstrates:**
- `AsyncDeliberationLoop` with `asyncio.to_thread` — fully asynchronous MCTS running in parallel with generation
- Mamba SSM backbone (`crsm/core/mamba.py`); hidden states are Mamba recurrent states (per-layer `List[torch.Tensor]`)
- `MCTSNode.state_cache` stores Mamba SSM states; MCTS traverses over these latent states (not token sequences)
- Sparse-Gated Hierarchical Injection applies planning results: `h_new = (1 − α) · h_old + α · h_target` (α = injection_rate, default 0.05)
- `think_and_generate` applies deltas via a Targeted Delta Buffer as tokens are generated
- Does NOT touch any transformer KV-cache tensors
- README explicitly states: "EXPERIMENTAL PROTOTYPE" — targets 100k–500k parameters, ARC-AGI only

**What our claims get right:**
- "asynchronous MCTS planner" ✅ — `AsyncDeliberationLoop` with `asyncio` confirmed
- "edit the internal hidden states of a Mamba state-space model" ✅ — Sparse-Gated Injection to Mamba SSM recurrent states confirmed
- "during generation" ✅ — `think_and_generate` async loop with Targeted Delta Buffer confirmed
- "recurrent hidden-state traversal" ✅ — Mamba SSM recurrent state is precisely what MCTS traverses
- "rather than direct arithmetic mutation of a frozen transformer's KV-cache tensors" ✅ — CRSM does not touch KV-cache; contrast is accurate

**Gaps:**

| # | Claim gap | Severity | Notes |
|---|---|---|---|
| 1 | Citation quality: GitHub-only repo, no peer review, single developer, explicitly experimental prototype, README notes AI-assisted ("Centaur workflow" with GPT/Claude/Gemini). NeurIPS reviewers may question this. | Medium | Same class of concern as Deroy 2026 (Zenodo). Consider whether this citation can be replaced or at least paired with a peer-reviewed continuous latent reasoning citation |
| 2 | CRSM requires multi-stage training (backbone pretraining → dynamics distillation → value head finetuning); not inference-time on a frozen model. Our contrast "frozen transformer" correctly distinguishes us, but the sentence doesn't make clear CRSM also requires training | Low | Background citation; framing is contrastive, not descriptive — unlikely to mislead |
| 3 | Scale omitted: CRSM is 100k–500k parameters targeting ARC-AGI (nano-scale proof-of-concept), not an LLM-scale system | Low | Background citation only — scale caveat not required |

**Required fixes:**

| Fix | Location | Urgency |
|---|---|---|
| Consider pairing `crsm2025` with a peer-reviewed continuous latent search citation, or add "(experimental prototype, no peer review)" qualifier | §2.5, bibliography | Medium — NeurIPS reviewer risk |
| No technical claim corrections needed | — | — |

---

### P4-E: `okazakirag2025` — Okazaki-RAG
**Full ref:** *System 2 through Okazaki-RAG: Capacity-Bounded Inline Fragment Manipulation for Deliberative Reasoning.* ResearchGate, 2025.
**Category:** BACKGROUND
**Where cited:** §2.5 (Continuous Latent Reasoning), line 113
**Claims:**
1. "implements a tree search whose nodes are KV-cache assemblies, with explicit rollback when heuristic scores deteriorate"
2. "constructs its search tree by swapping pre-computed, discrete cache blocks rather than executing continuous arithmetic mutations"
**Status:** ⚠️ 2 gaps — all technical claims accurate; citation quality concern (unpublished manuscript, no peer review, personal email, no experiments)
**Source:** Full PDF read via `pdftotext`. `docs/NeurIPS/Citations/System2throughOkazaki-RAG.pdf`

**What the paper actually demonstrates:**
- Dual-strand architecture: lightweight Context Strand (global gist, SoftMax over fragment library) + frozen Value Strand (reasoning over curated working set)
- "Okazaki fragments" = pre-computed final-layer KV-cache blocks (Ki(L), Vi(L)) with summary embeddings and provenance metadata
- Inline Manipulation Engine (IME): reversible primitive operations {Append, Remove, Assemble, Swap} on the working set
- Section 3.3 Search Policy step 4: *"If its value drops below a threshold, the engine can backtrack to explore another branch, emulating ToT-style search"* — rollback confirmed
- Paper's own summary: *"a tree search whose nodes are latent assemblies rather than text strings"* — our exact phrasing confirmed
- Scoring: look-ahead log-likelihood from Value Strand + context prior SoftMax — qualifies as heuristic scoring
- Paper header: **"Manuscript as of August 09, 2025"** — authors: Dyuti Ghosh, Dibya Ghosh, Debi Prasad Ghosh (personal Gmail)
- Explicitly a **conceptual contribution only** — "we define the controller formalism" — no experiments, no implementation

**What our claims get right:**
- "tree search whose nodes are KV-cache assemblies" ✅ — paper's own words: "tree search whose nodes are latent assemblies rather than text strings"
- "explicit rollback when heuristic scores deteriorate" ✅ — §3.3: "backtrack to explore another branch" when value drops below threshold
- "swapping pre-computed, discrete cache blocks" ✅ — IME {Swap} operation on pre-computed final-layer KV blocks confirmed
- "rather than executing continuous arithmetic mutations" ✅ — IME manipulates discrete fragment sets, no continuous tensor arithmetic

**Gaps:**

| # | Claim gap | Severity | Notes |
|---|---|---|---|
| 1 | Citation quality: unpublished manuscript (Aug 2025), personal Gmail in header, no arXiv, no peer review, no experiments (conceptual contribution only). Fourth such low-quality citation (after Deroy, CRSM, and the original Okazaki-RAG blueprint). NeurIPS reviewers can't verify via any standard channel. | High | Consider citing alongside a peer-reviewed KV-cache reuse paper (e.g., TurboRAG Lu et al. 2024, CacheBlend Yao et al. 2024, or EPIC Hu et al. 2024) that validates the representation-level retrieval concept |
| 2 | The "tree search" in IME is explicitly modeled on Tree-of-Thoughts (Yao et al. 2023) — Okazaki-RAG is applying ToT-style search to KV fragments, not inventing tree search. If we cite Okazaki-RAG as the source of the tree-search-with-rollback idea, we may be citing the wrong paper. | Low | Background citation only — framing is contrastive, so attribution is not critical |

**Required fixes:**

| Fix | Location | Urgency |
|---|---|---|
| Pair `okazakirag2025` citation with at least one peer-reviewed KV-cache serving paper (TurboRAG/CacheBlend/EPIC) to anchor the comparison | §2.5, bibliography | High — NeurIPS reviewer risk |
| No technical claim corrections needed — all confirmed | — | — |

---

### P4-F: `bailey2024obfuscated` — Obfuscated Activations
**Full ref:** Bailey et al. (2024). *Obfuscated Activations Bypass LLM Latent-Space Defenses.* arXiv:2412.09565.
**Category:** BACKGROUND
**Where cited:** §2.2 (Background), line 97
**Claim:** "Prior work has explored the latent intervention space via gradient ascent~\cite{bailey2024obfuscated}"
**Status:** ✅ Mostly clean — 2 low-severity gaps (framing imprecision; domain mismatch)
**Source reviewed:** `docs/NeurIPS/Citations/arXiv-2412.09565v2/main.tex`

**What the paper actually demonstrates:**
- Obfuscation attacks optimize a joint objective: `min L_beh(M(x)) + λ·L_obf(repr(x))` — behavioral loss + obfuscation loss defined over residual stream activations
- Embedding attacks: Adam optimizer on continuous input suffix embeddings (gradient flows through model activations back to the suffix) — gradient-based ✅
- Hard prompt attacks: FLRT (gradient-free) or GCG (gradient-based discrete token search)
- Shows SOTA latent-space monitors (probes, OOD detectors, SAEs) can all be bypassed
- Results: 100% → 0% recall on harmfulness probes while retaining 90% jailbreaking rate
- Tested on Llama-3-8B-Instruct

**What our claim gets right:**
- "gradient ascent" ✅ — embedding attacks use Adam gradient optimization; the objective function is defined over latent activations; gradient-based exploration confirmed
- The contrastive point holds perfectly: Bailey requires white-box gradient access; our MCTS is gradient-free, operates on KV-cache, and requires no input suffix modification ✅

**Gaps:**

| # | Claim gap | Severity | Notes |
|---|---|---|---|
| 1 | "latent intervention space" is slightly imprecise — Bailey optimizes **input embeddings** (gradient w.r.t. suffix), not latent states directly. The loss is defined over activations, but the optimization variable is the input. A precise framing would be "gradient-based input optimization targeting latent-monitor evasion." | Low | Background citation with clear contrastive purpose — imprecision doesn't undermine the comparison |
| 2 | Domain mismatch: paper is about bypassing harmful-content monitors (probes, OOD detectors, SAEs), not alignment faking detection. Our paper frames it as general "latent intervention space" exploration. | Low | Background citation only — the domain difference is not claimed to be identical |

**Required fixes:**

| Fix | Location | Urgency |
|---|---|---|
| None required — contrast with our approach is valid; "gradient ascent" characterization acceptable in context | — | — |

---

### P4-G: `agenticred2025` — AgenticRed
**Full ref:** *AgenticRed: Optimizing Agentic Systems for Automated Red-teaming.* arXiv:2601.13518v3, 2025.
**Category:** BACKGROUND
**Where cited:** §2.4 (MCTS for Red-Teaming), line 107
**Claim:** "AgenticRed~\cite{agenticred2025} applied MCTS within an agentic red-teaming framework" — placed in a list of MCTS-based red-teaming systems
**Status:** ⚠️ HIGH — misattributes AgenticRed's method. AgenticRed uses evolutionary algorithms, not MCTS. MCTS appears only as an appendix sub-agent.
**Source reviewed:** `docs/NeurIPS/Citations/arXiv-2601.13518v3/notes/` — abstract, introduction, methods, appendix

**What the paper actually demonstrates:**
- **Primary method:** Evolutionary algorithms applied to agentic system design — a meta-agent iteratively generates and evaluates red-teaming agentic systems, retaining the "fittest" in an archive. Inspired by Meta Agent Search. Darwin Gödel Machine concepts applied to red-teaming.
- **MCTS in the paper:** PHOENIX-MCTS (appendix §7) is ONE of many systems automatically designed BY AgenticRed during its evolutionary search — described as "second best performing system under Mixtral-attacker-Llama2 setting, designed in 6th generation." PHOENIX-MCTS uses MCTS over prompt variants (text space) with UCB selection and rewrite operators.
- MCTS is not AgenticRed's method — it is a component of one discovered artifact produced by AgenticRed's evolutionary search.
- ASR results: 96% on Llama-2-7B, 98% on Llama-3-8B, 100% on GPT-5.1 and DeepSeek-R1 (HarmBench)

**What our claim gets wrong:**
- "AgenticRed applied MCTS within an agentic red-teaming framework" ❌ — AgenticRed's method is evolutionary system design. Listing it among MCTS-based approaches mischaracterizes the paper.
- The placement in the "MCTS-based red-teaming" paragraph implies AgenticRed is an MCTS system. It is not.

**What is still accurate:**
- The overall claim "all existing MCTS-based red-teaming operates exclusively in text space" ✅ — still holds without AgenticRed; MPA, DAMON, and ACE-Safety are sufficient examples, and PHOENIX-MCTS (if retained) also operates in text space
- The broader novelty claim (no prior MCTS over KV-cache) ✅ — unaffected

**Gaps:**

| # | Claim gap | Severity | Notes |
|---|---|---|---|
| 1 | AgenticRed listed as an MCTS-based red-teaming system — primary method is evolutionary algorithms, not MCTS | High | Misattributes their core contribution; MCTS appears only in one appendix sub-agent |

**Required fixes:**

| Fix | Location | Urgency |
|---|---|---|
| Remove AgenticRed from the MCTS list at line 107: delete "AgenticRed~\cite{agenticred2025} applied MCTS within an agentic red-teaming framework." — MPA, DAMON, ACE-Safety are sufficient to establish the MCTS red-teaming prior art | §2.4 line 107 | High |
| Optional: add separate sentence in §2.4 or elsewhere: "AgenticRed~\cite{agenticred2025} applied evolutionary algorithms to automate agentic red-teaming system design, achieving up to 100\% ASR on frontier models." | §2.4 | Low — omitting entirely is cleaner |

---

### P4-H: `li2025ace` — ACE-Safety
**Full ref:** Li et al. (2025). *Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization.* arXiv:2511.19218v2.
**Category:** BACKGROUND
**Where cited:** §2.4 (MCTS for Red-Teaming), line 107
**Claim:** "ACE-Safety formulated attack-defense co-evolution via tree-group dual-aware search"
**Status:** ✅ Clean — all claims accurate; 2 low-severity notes (terminology compression; domain)
**Source reviewed:** `docs/NeurIPS/Citations/arXiv-2511.19218v2/acl_latex.tex`

**What the paper actually demonstrates:**
- **ACE-Safety** = Adversarial Co-Evolution for LLM Safety — two components:
  1. **GS-MCTS** (Group-aware Strategy-guided MCTS): MCTS with PUCT selection over jailbreak strategy actions; expansion generates a *group* of G rewritten queries per node; node state `s = {p, q̂, ô, ĵ}` where `q̂` is the set of modified prompt queries — **text space** ✅
  2. **ACTGPO** (Adversarial Curriculum Tree-aware Group Policy Optimization): jointly trains attack and defense LLMs using tree-structured samples via curriculum RL
- "tree-group dual-aware" accurately captures both the tree structure (MCTS) and group evaluation (G queries per node) present in both components
- Attack and defense co-evolve iteratively: updated models feed the next GS-MCTS round ✅

**What our claims get right:**
- "attack-defense co-evolution" ✅ — explicit in paper name and method; attack + defense models mutually improve
- "tree-group dual-aware search" ✅ — GS-MCTS is both tree-structured (MCTS) and group-aware (G queries per node); ACTGPO is "tree-aware group" policy optimization
- Text space ✅ — MCTS nodes are prompt rewriting strategies and modified query sets; no latent intervention

**Gaps:**

| # | Claim gap | Severity | Notes |
|---|---|---|---|
| 1 | "dual-aware" is our compression, not the paper's term (paper says "tree-aware and group-aware mechanisms") | Low | Spirit is accurate; unlikely to confuse reviewers |
| 2 | Domain: jailbreak harmfulness attacks, not alignment faking | Low | Background citation only |

**Required fixes:**

| Fix | Location | Urgency |
|---|---|---|
| None required — all claims accurate | — | — |

---

### P4-I: `zheng2023judging` — MT-Bench
**Full ref:** Zheng, Chiang, Sheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023 (Datasets & Benchmarks).
**Category:** BASELINE + BACKGROUND
**Where cited:** §1 line 58 (LLM-as-a-Judge paradigm example), §2.5 line 117 (MT-Bench named as "suffering from reproducibility problems"), Table 2 (LLM-as-Judge baseline row)
**Status:** ⚠️ 2 gaps — framing mischaracterizes paper's positive conclusion; domain mismatch (helpfulness ≠ safety)
**Source reviewed:** arXiv:2306.05685v4 TeX source (NeurIPS 2023)

**What the paper actually demonstrates:**
- Introduces MT-Bench (80 multi-turn questions across 8 categories) and Chatbot Arena as human-preference benchmarks for evaluating chatbot assistants
- Systematically studies LLM-as-a-Judge: pairwise comparison, single-answer grading, reference-guided grading
- Documents biases: position bias, verbosity bias, self-enhancement bias, limited math/reasoning grading — and proposes mitigations for each
- Key finding: GPT-4 as judge achieves **85% agreement with human experts** — equal to human-human agreement (81%)
- Overall conclusion (abstract): "LLM-as-a-judge is a scalable and explainable way to approximate human preferences"
- Scoring scale: MT-Bench uses **1–10 integer scores**, not normalized 0–1 scales
- Focus domain: chatbot quality/helpfulness — Discussion §6 explicitly states "This paper emphasizes helpfulness but largely neglects safety"

**What our paper claims (and what is correct):**
- Line 58: "automated LLM-as-a-Judge frameworks~\cite{zheng2023judging, liu2023geval}" — ✅ Correct. MT-Bench is exactly an LLM-as-a-Judge framework.
- Line 117: "use language models to evaluate output quality holistically" — ✅ Correct. GPT-4 is the judge, providing holistic quality scores and explanations.
- Table 2: Listed as LLM-as-Judge baseline — ✅ Correct.

**Gaps:**

| # | Claim in our paper | What the paper actually shows | Gap | Severity |
|---|---|---|---|---|
| 1 | "These approaches suffer from well-documented reproducibility problems" (line 117, implying MT-Bench) | MT-Bench's own conclusion is the opposite: GPT-4 achieves 85% agreement with humans (same as human-human). Biases are documented but characterized as mitigable, not fundamental. The paper concludes GPT-4 judging is reliable and scalable. | Framing contradicts the cited paper's positive conclusion | LOW |
| 2 | Cited alongside the fabricated "0.10–0.20 on normalized scales" figure | MT-Bench uses 1–10 integer scores. No "0.10–0.20 on normalized scales across independent runs" figure appears anywhere in this paper. (The figure is attributed to llmasjudge2025limitations — already confirmed as fabricated.) | Domain framing; also MT-Bench is helpfulness-focused, not safety-focused | LOW |

**Required fixes:**
- No standalone fixes required specific to `zheng2023judging` beyond the already-identified 🚨 CRITICAL fix for line 117 (remove fabricated 0.10–0.20 claim from llmasjudge2025limitations). When that claim is removed, the remaining sentence "MT-Bench uses language models to evaluate output quality holistically" is accurate.
- **Optional:** Consider adding a note that MT-Bench's own finding is 85% GPT-4/human agreement — this actually supports our point that behavioral evaluation is insufficient even when high-agreement automated judges exist. Positive reframing possible.

**Citation quality:** ✅ Excellent — NeurIPS 2023 Datasets & Benchmarks, peer-reviewed, highly cited, UC Berkeley / Stanford / CMU authors.

---

### P4-J: `liu2023geval` — G-Eval
**Full ref:** Liu, Iter, Xu et al. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* EMNLP 2023. (Microsoft Cognitive Services Research)
**Category:** BASELINE + BACKGROUND
**Where cited:** §1 line 58 (LLM-as-a-Judge paradigm example), §2.5 line 117 (implied to "suffer from reproducibility problems"), Table 2 (LLM-as-Judge baseline row)
**Status:** ⚠️ 3 gaps — framing mischaracterizes paper; domain mismatch (NLG quality ≠ safety); actual documented concern is LLM-preference bias (not run-to-run variance)
**Source reviewed:** arXiv:2303.16634v3 TeX source (EMNLP 2023)

**What the paper actually demonstrates:**
- G-Eval: LLM (GPT-4) + auto chain-of-thought + form-filling paradigm for NLG quality evaluation
- Scoring: 1–5 integer scales, with probability-weighted normalization: `score = Σ p(sᵢ) × sᵢ`; for GPT-4 (which doesn't expose token probs), uses n=20 samples at temperature=1 to estimate probabilities
- Tested on: text summarization (SummEval) and dialogue generation (Topical-Chat); NOT safety evaluation
- Key result: G-Eval-4 achieves Spearman ρ = 0.514 with human on summarization — outperforms all prior methods by a large margin
- Documented limitation 1: **low variance issue** — "one digit usually dominates the distribution of the scores (e.g., 3 for a 1–5 scale). This may lead to the low variance of the scores." — probability normalization addresses this
- Documented limitation 2: **LLM-generated text preference bias** — G-Eval-4 always assigns higher scores to GPT-3.5 summaries even when humans prefer human-written ones. Inter-annotator agreement is very low (Krippendorff's α = 0.07)
- "0.10–0.20 on normalized scales across independent runs" — **absent from this paper**. No such figure anywhere.

**What our paper claims (and what is correct):**
- Line 58: "automated LLM-as-a-Judge frameworks~\cite{zheng2023judging, liu2023geval}" — ✅ Correct. G-Eval IS an LLM-as-a-Judge framework.
- Line 117: "use language models to evaluate output quality holistically" — ✅ Correct. GPT-4 with CoT evaluates holistically.
- Line 117: "these approaches suffer from well-documented reproducibility problems: score divergences of $0.10$--$0.20$..." — ⚠️ Mischaracterizes. G-Eval's documented problems are (a) score clustering/low variance and (b) LLM-preference bias — neither is "score divergence across independent runs."
- Table 2: LLM-as-Judge baseline — ✅ Correct.

**Gaps:**

| # | Claim in our paper | What the paper actually shows | Gap | Severity |
|---|---|---|---|---|
| 1 | "Suffer from well-documented reproducibility problems" (implied via line 117 context) | G-Eval achieves ρ=0.514, SOTA. Documented problems are low-variance clustering and LLM-preference bias, not run-to-run score divergence | Framing inverts the paper's characterization | LOW |
| 2 | "Normalized scales" language in adjacent sentence | G-Eval uses 1–5 integer scales with probability-weighted refinement; explicitly notes LLMs "usually only output integer scores" | Terminology mismatch | LOW |
| 3 | Cited as safety/alignment evaluation baseline | G-Eval is specifically for NLG quality: summarization + dialogue generation. Domain is helpfulness/quality, not safety. | Domain extrapolation | LOW-INFO |

**Bonus finding — accurately citable real limitation:**
G-Eval does document a genuine concern relevant to our paper: **LLM judges prefer LLM-generated outputs over human-written ones** (Krippendorff's α=0.07 for inter-annotator agreement). This could be cited accurately in §2.5 to support our argument that LLM-as-a-Judge has structural biases — just replace the fabricated run-to-run variance claim with this real one.

**Required fixes:**
- No standalone fixes specific to `liu2023geval`. When the Critical fix for line 117 (remove fabricated 0.10–0.20 claim) is applied, the remaining "G-Eval uses language models to evaluate output quality holistically" statement is accurate.
- **Optional enhancement:** Cite G-Eval's real finding — LLM judges prefer LLM outputs (α=0.07) — to replace the fabricated reproducibility claim with an accurate limitation.

**Citation quality:** ✅ Good — EMNLP 2023, peer-reviewed, Microsoft Research, published work.

---

## Orphaned Bibliography Entries (in .tex but not cited in body text)

These appear in `\thebibliography` but have no `\cite{}` call in the main text. Verify intent — either add a citation or remove the entry.

| Key | Title | Action |
|---|---|---|
| `contextual2025` | Contextual Residual Inversion in Transformer-Based LLMs (OSF Preprints, 2025) | Verify: intended citation? If yes, add `\cite{}`. If no, remove `\bibitem`. |
| `oozeer2025transfer` | Activation Space Interventions Can Be Transferred Between LLMs (arXiv:2503.04429) | Verify: intended citation? May belong in §2.2 alongside Belitsky. |

---

## Prospective Citations — Future Work

Papers reviewed that are **not currently cited** in the paper but have assessed citation potential for future submissions or §7 Future Work notes.

---

### FW-1: `jha2025vec2vec` — vec2vec / Universal Geometry of Embeddings
**Full ref:** Jha, Zhang, Shmatikov, Morris (2025). *Harnessing the Universal Geometry of Embeddings.* NeurIPS 2025. Cornell University.
**Category:** PROSPECTIVE — not currently cited; assessed for §7 Future Work
**Source reviewed:** arXiv:2505.12540v4 TeX source

**What the paper actually demonstrates:**
- `vec2vec`: first method for unsupervised translation of sequence-level text embeddings across encoder model families (GTR ↔ GTE ↔ BERT-based), without paired data, encoder access, or predefined matches
- Method: adversarial GAN + cycle consistency + vector space preservation losses; MLP adapters map into a shared universal latent space
- Proves the **Strong Platonic Representation Hypothesis**: a learnable universal latent geometry exists across encoder spaces — neural networks trained on same modality converge to a shared representation structure
- Key result: cosine similarity up to 0.96 to ground-truth target embeddings; perfect matching on 8K+ shuffled embeddings
- Security implication: embedding database leakage ≈ document content leakage — sensitive patient records and email content extracted from raw vectors alone

**Relevance to KV-Cache Inception:**

| Angle | Strength | Assessment |
|---|---|---|
| Platonic Rep Hypothesis → cross-model KV-cache generalization | Weak | If latent spaces converge universally, alignment faking signatures might transfer across model families — but Experiment 5 (Procrustes transfer) was dropped; we have no empirical evidence for this claim in our paper. Citing risks inviting an unanswerable reviewer question. |
| Embeddings-as-rich-signals analogy | Conceptual only | vec2vec shows static *encoder* embeddings contain more information than apparent. Our paper argues dynamically-mutated *KV-cache states* expose alignment intent. Parallel claim, different objects — not directly comparable. |
| RepE probe generalization motivation | Moderate | Platonic Rep Hypothesis provides theoretical grounding for why RepE probes trained on one text distribution generalize to others. Could legitimately appear in 1–2 sentences in §2.2 (RepE background). |
| Future work — cross-model steering transfer | Genuine | If Experiment 5 is revived or cited as future direction, this is the canonical citation. |

**Verdict: do not cite in current NeurIPS 2026 submission.**
Inserting it now would be a decoration cite that raises "have you tested cross-model KV-cache representation transfer?" — which the paper cannot answer. The Experiment 5 connection is real but the experiment was dropped.

**Recommended future use:**
Add 1–2 sentences to §7 Limitations/Future Work:
> *"Recent work demonstrates that encoder representation spaces share a learnable universal geometry~\cite{jha2025vec2vec}, suggesting that KV-cache alignment signatures may generalize across model families via unsupervised translation — a direction we leave for future work."*

This is honest, adds no unsupported claims, and signals breadth to reviewers without creating an empirical burden.

**Citation quality:** ✅ Excellent — NeurIPS 2025, peer-reviewed, Cornell (Shmatikov lab), strong empirical results.

---

## Review Order and Status Summary

| Priority | Key | Category | Status | PDF in Citations/ |
|---|---|---|---|---|
| P1-A | `gao2025hneurons` | IMPLEMENT + RESULT | ⚠️ 5 GAPS FOUND | Not needed (already read) |
| P1-B | `zou2023repe` | IMPLEMENT + RESULT | ⚠️ 5 GAPS FOUND | arXiv-2310.01405v4 TeX reviewed |
| P2-A | `gupta2025rlobfuscation` | RESULT (load-bearing) | ⚠️ 4 GAPS FOUND | arXiv-2506.14261v4 TeX reviewed |
| P2-B | `greenblatt2024alignment` | RESULT (load-bearing) | ⚠️ 4 GAPS FOUND | arXiv-2412.14093v2 TeX reviewed |
| P2-C | `hubinger2024sleeper` | RESULT | ⚠️ 4 GAPS FOUND | arXiv-2401.05566v3 TeX reviewed |
| P2-D | `belitsky2025kvcache` | RESULT + BASELINE | ⚠️ 4 GAPS FOUND | arXiv-2507.08799v2 TeX reviewed |
| P2-E | `llmasjudge2025limitations` | RESULT | 🚨 CRITICAL — fabricated claim | arXiv-2503.05061v2 TeX reviewed |
| P3-A | `zou2023gcg` | BASELINE + RESULT | ⚠️ 3 GAPS FOUND | arXiv-2307.15043v2 TeX reviewed |
| P3-B | `wu2025mcts` | BASELINE | ✅ Clean | COLING PDF + GitHub source reviewed |
| P3-C | `damon2025` | BASELINE | ✅ Clean | EMNLP PDF + GitHub source reviewed |
| P4-A | `deroy2026structured` | BACKGROUND | ⚠️ 2 GAPS (citation quality) | Zenodo API reviewed |
| P4-B | `turner2023steering` | BACKGROUND | ✅ §2.2 claim clean; ⚠️ Table 2 CAA uncited | arXiv-2308.10248v5 TeX reviewed |
| P4-C | `hao2025coconut` | BACKGROUND | ✅ Clean | arXiv-2412.06769v3 TeX reviewed |
| P4-D | `crsm2025` | BACKGROUND | ⚠️ 3 gaps (citation quality concern) | GitHub source reviewed |
| P4-E | `okazakirag2025` | BACKGROUND | ⚠️ 2 gaps (citation quality; no experiments) — all technical claims confirmed | Full PDF read via pdftotext |
| P4-F | `bailey2024obfuscated` | BACKGROUND | ✅ Mostly clean — 2 low gaps (framing; domain) | arXiv-2412.09565v2 TeX reviewed |
| P4-G | `agenticred2025` | BACKGROUND | ⚠️ HIGH — misattributes MCTS; AgenticRed is evolutionary, not MCTS | arXiv-2601.13518v3 TeX reviewed |
| P4-H | `li2025ace` | BACKGROUND | ✅ Clean — all claims accurate | arXiv-2511.19218v2 TeX reviewed |
| P4-I | `zheng2023judging` | BASELINE + BACKGROUND | ⚠️ 2 gaps (framing; domain) — citation quality ✅ excellent | arXiv-2306.05685v4 TeX reviewed |
| P4-J | `liu2023geval` | BASELINE + BACKGROUND | ⚠️ 3 gaps (framing; domain; real limitation is LLM-preference bias, not run-to-run variance) | arXiv-2303.16634v3 TeX reviewed |
| — | `contextual2025` | ORPHAN | ⚠️ CHECK | Needed if keeping |
| — | `oozeer2025transfer` | ORPHAN | ⚠️ CHECK | Needed if keeping |
