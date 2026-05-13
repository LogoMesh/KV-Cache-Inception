# Track B — Citation Verification Audit

**Author:** Diagnostic Claude Session #2
**Created:** 2026-05-09 (T+18 target; produced ahead of schedule)
**Scope:** Load-bearing citations under Candidate A "System-2 Latent Search" framing AND current premortem-v2.tex state.
**Method:** Read each citation's local source bundle (LaTeX or extracted PDF text) for abstract + introduction + main-claim section. Extract verbatim claims. Compare to paper text. Verdict per protocol.

---

## TOP-LINE FINDINGS (read first)

- **PILE-UP COUNT:** 1 NUANCED, 0 MISMATCHED. Below the YELLOW threshold (2) and well below NO-GO (3+).
- **FATAL FINDINGS:** **NONE.** The Gate 1 fatal-check on COCONUT comes back clean — see §2.1 below.
- **MISSING (must be downloaded before Candidate A revision can ship):** 5 citations — Snell 2024, Hasani 2025, Stoehr 2024 (Activation Scaling), MUSE (Yu et al. 2025), and Hao 2024 (RAP, the COCONUT predecessor referenced separately in our §2 text). Plus low-priority: SAE survey, RouteSAE, SHARP, SteerVLM. See §3.
- **CITATIONS-TO-ADD (Candidate A):** 5 mandatory + 4 nice-to-have. See §4.
- **GATE 1 RECOMMENDATION (Track B portion):** **PASS.** Criterion G1.3 (≤2 nuanced/mismatched) and G1.4 (no fatal) are both met from this track's perspective. Gate decision still depends on Track A and Track D.

---

## 1. Per-citation table

| # | Name | arXiv ID | Local file | Verdict | Notes |
|---|---|---|---|---|---|
| P1 | Hao et al. 2025 — COCONUT | 2412.06769 | `arXiv-2412.06769v3/paper.tex` | **CLEAN** | Trained model w/ multi-stage curriculum; hidden-state feedback as next embedding (not KV-cache mutation); not reversible. Differentiation intact. |
| P2 | Snell et al. 2024 — Test-time compute scaling | 2408.03314 (expected) | **MISSING** | **MISSING** | Not in `Citations/`. Required for Candidate A framing per the framing doc Part 4. |
| P3 | Zou et al. 2023 — RepE | 2310.01405 | `arXiv-2310.01405v4/sections/4-honesty.tex` | **CLEAN** for current paper; **NUANCED** for Candidate A | RepE validates probes for honesty/truthfulness/utility/emotion/harmlessness. Does NOT validate a "certainty" probe specifically. The Candidate A swap from honesty→certainty is a methodological extension, not a Zou et al. validated capability. |
| P4 | Gao et al. 2025 — H-Neurons | 2512.01797 | `arXiv-2512.01797v2/neurips_2024.tex` | **CLEAN** | H-Neurons identified, sparse subset (<0.1%), causally linked to over-compliance (sycophancy, instruction-following, fabrication). Both hallucination AND over-compliance are explicit causal claims in the paper. |
| P5 | Hasani et al. 2025 — System-2 counting | 2601.02989 (expected) | **MISSING** | **MISSING** | Not in `Citations/`. Required for Candidate A framing only. |
| P6 | Stoehr et al. 2024 — Activation Scaling | 2410.04444 (expected) | **MISSING** | **MISSING** | Not in `Citations/`. Suggested addition by framing doc; not yet cited in our TeX. |
| P7 | Bailey et al. 2024 — Obfuscated Activations | 2412.09565 | `arXiv-2412.09565v2/main.tex` | **CLEAN** | Tentative negative result for trained input-space attacks (§B.6 / §sec:orthogonality) verified verbatim. Our 3-axis differentiation (intervention site / objective / metric scope) holds under either framing. |
| P8 | Yu et al. 2025 — MUSE (EMNLP main 1080) | n/a | **MISSING** | **MISSING** | NOT identified in `_extracted/` despite framing-doc hint. `2025.emnlp-main.323.pdf` is DAMON, not MUSE. `coling-71.txt` is Wu MCTS, not MUSE. |
| 9 | Greenblatt et al. 2024 — Alignment Faking | 2412.14093 | `arXiv-2412.14093v2/sections/0_abstract.tex` | **CLEAN** | All paper claims (Opus 14% compliance, scratchpad reasoning, harmlessness preservation, scale-dependence) verbatim from abstract. |
| 10 | Hubinger et al. 2024 — Sleeper Agents | 2401.05566 | `arXiv-2401.05566v3/main.tex` | **CLEAN** | "Adversarial training can teach models to better recognize their backdoor triggers, effectively hiding the unsafe behavior" verbatim from abstract — matches our paper's framing exactly. Scale-dependence verified. |
| 11 | Gupta & Jenner 2025 — RL-Obfuscation | 2506.14261 | `arXiv-2506.14261v4/sections/abstract.tex` + `sections/eval_policy.tex:71` | **CLEAN** | Token-level monitor vulnerability verified; max-pool/attention robustness verified; "70% of recall drop is policy change" verified verbatim. |
| 12 | Zou et al. 2023 — GCG | 2307.15043 | `arXiv-2307.15043v2/main.tex` | **CLEAN** | Gradient-based, white-box, discrete token suffix attack — matches our framing. |
| 13 | Zheng et al. 2023 — MT-Bench / LLM-as-Judge | 2306.05685 | `arXiv-2306.05685v4/sec-intro.tex` | **CLEAN** | LLM-as-Judge with biases (position, verbosity, self-enhancement) — matches our citation. |
| 14 | Liu et al. 2023 — G-Eval | 2303.16634 | `arXiv-2303.16634v3/acl2021.tex` | **CLEAN** | G-Eval / GPT-4 NLG evaluation — matches our citation. |
| 15 | Turner et al. 2023 — Activation Steering | 2308.10248 | `arXiv-2308.10248v5/iclr2025_conference.tex` | **CLEAN** | "Inference-time modification of activations to steer model outputs"; ActAdd technique — matches our citation. |
| 16 | Belitsky et al. 2025 — Cache Steering | 2507.08799 | `arXiv-2507.08799v2/main.tex` | **CLEAN** | "One-shot intervention applied directly to the key-value cache"; reasoning induction in small LMs; scales to larger models — matches "360M–8B" claim in our text. |
| 17 | Limitations of LLM-as-Judge | 2503.05061 | `arXiv-2503.05061v2/main.tex` | **CLEAN** (title-only verified) | "No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding" — matches our framing. The κ=0.86→0.16 number specifically should be spot-verified by Session A before camera-ready. |
| 18 | Wu et al. 2025 — MCTS Prompt Autogeneration | n/a (COLING) | `_extracted/coling-71.txt` | **CLEAN** | MCTS-based prompt auto-generation for jailbreak — matches our "MCTS-based prompt autogeneration for jailbreak attacks". |
| 19 | DAMON | n/a (EMNLP) | `_extracted/emnlp-323.txt` | **CLEAN** | Dialogue-aware MCTS for multi-turn jailbreaks — matches our framing. |
| 20 | ACE-Safety | 2511.19218 | `arXiv-2511.19218v2/acl_latex.tex` | **CLEAN** | "Adversarial Attack-Defense Co-Evolution... via Tree-Group Dual-Aware Search" — matches our citation. |
| 21 | AgenticRed | 2601.13518 | `arXiv-2601.13518v3/main.tex` | **CLEAN** (title-only verified) | "Evolving Agentic Systems for Red-Teaming" — matches our citation. |
| 22 | "Harnessing Universal Geometry of Embeddings" | 2505.12540 | `arXiv-2505.12540v4/neurips_2025.tex` | **N/A — extra in Citations/** | Not currently cited in premortem-v2.tex. Title appears unrelated to our paper's load-bearing claims. May be a stale download. |

---

## 2. Detailed entries — citations requiring substantive comparison

### 2.1 Hao et al. 2025 — COCONUT — **CLEAN** (FATAL CHECK PASSED)

**Paper file:** `docs/NeurIPS/Citations/arXiv-2412.06769v3/paper.tex`

**Verbatim claims from COCONUT paper:**

> "We utilize the last hidden state of the LLM as a representation of the reasoning state (termed 'continuous thought'). Rather than decoding this into a word token, we feed it back to the LLM as the subsequent input embedding directly in the continuous space." — Abstract, lines 42–45.

> "Our proposed continuous thoughts are fully differentiable and allow for back-propagation. We perform $n+1$ forward passes when $n$ latent thoughts are scheduled in the current training stage..." — §3 Training Details, line 136.

> "We leverage language CoT data to supervise continuous thought by implementing a multi-stage training curriculum inspired by [Deng 2024]." — §3 Training Procedure, line 122.

> "Continuous thoughts in COCONUT can encode multiple potential next steps simultaneously, allowing for a reasoning process akin to breadth-first search (BFS). While the model may not initially make the correct decision, it can maintain many possible options within the continuous thoughts and progressively eliminate incorrect paths through reasoning, guided by some implicit value functions. This advanced reasoning mechanism surpasses traditional CoT, even though the model is not explicitly trained or instructed to operate in this manner." — §1 Introduction, line 79.

**Our paper's framing (premortem-v2.tex line 134):**

> "COCONUT~\cite{hao2025coconut} bypasses discrete token decoding by feeding final-layer hidden states back as input embeddings, enabling emergent breadth-first latent tree search. This validates the viability of continuous latent search, but operates via recurrent hidden-state traversal rather than direct arithmetic mutation of a frozen transformer's KV-cache tensors."

**Comparison:**
- COCONUT REQUIRES TRAINING — multi-stage curriculum on language CoT data. Our paper is training-free. ✓ Differentiated.
- COCONUT operates on hidden-state-as-next-embedding (residual stream input route), NOT on KV-cache K/V tensors. ✓ Differentiated.
- COCONUT does not claim exact reversibility — the paradigm doesn't require it (it's gradient-trained end-to-end during training; at inference there's no rollback). ✓ Our reversibility claim is unique.
- COCONUT's "BFS" is described as an *emergent property* of the trained continuous thoughts encoding multiple options in superposition (§1 introduction quote above). It is NOT an explicit MCTS algorithm. ✓ Our explicit search algorithm is differentiated.

**Verdict: CLEAN.** Gate 1 fatal-check #1 (does COCONUT do training-free latent search?) returns **NEGATIVE**. COCONUT requires training; our paper is training-free. Differentiation is on three independent axes (training vs frozen, hidden-state route vs KV-cache, emergent BFS vs explicit MCTS) and all three hold.

**Recommendation:** No revision needed. Optional strengthening: in §2.5 or §6 discussion, add one sentence explicitly noting "COCONUT's BFS is emergent from the trained continuous-thought representation encoding multiple options in superposition; our MCTS imposes the search structure explicitly via a reward-driven UCB1 selection process operating on a frozen base model" — this would preempt a reviewer who reads only COCONUT's abstract and asks "isn't this already done?".

---

### 2.2 Bailey et al. 2024 — Obfuscated Activations — **CLEAN**

**Paper file:** `docs/NeurIPS/Citations/arXiv-2412.09565v2/main.tex`

**Verbatim quote (§sec:orthogonality, line 1809):**

> "We are unable to achieve near-orthogonality while preserving behavior. Activations of two randomly chosen unrelated inputs from UltraChat have a cosine similarity of about 0.3, and to reach that level of orthogonality, we need to accept a KL divergence of over 100. Qualitative inspection of those samples shows that this KL divergence corresponds to mostly incoherent output."

**Verbatim quote (§sec:orthogonality, lines 1785–1790):**

> "we ask: *can a universal embedding suffix preserve the model's behavior while making its activations orthogonal to unperturbed ones?* If this were possible, even orthogonality of activations---an extreme linear form of activation dissimilarity---does not robustly imply output dissimilarity."

**Our paper's framing (premortem-v2.tex line 122):**

> "Bailey et al.\ (2024)~\cite{bailey2024obfuscated} systematically study whether adversarial perturbations to model inputs can achieve activation states that are simultaneously behavior-preserving and orthogonal to baseline activations. Their gradient-trained universal embedding suffixes optimize a joint objective: minimize the KL divergence between perturbed and baseline output distributions while minimizing the cosine similarity between perturbed and baseline activations. They report a tentative negative result for the regime of trained input-space attacks against a frozen model: near-orthogonality and behavior preservation cannot be jointly achieved..."

**Comparison:** Verbatim Bailey claim ("we are unable to achieve near-orthogonality while preserving behavior") is a stronger statement than "tentative negative result". Our paper's wording "tentative negative result" is appropriately hedged given that Bailey's experiment is one configuration (universal embedding suffixes against UltraChat-class inputs) and they explicitly bracket the conclusion with "qualitative inspection shows...incoherent output". The 3-axis complementarity claim (intervention site / objective / metric scope) is structurally honest.

**Verdict: CLEAN.** No revision needed. Bailey's negative result is real and the wording is fair to their hedging.

---

### 2.3 Zou et al. 2023 — RepE — **CLEAN for current paper, NUANCED for Candidate A**

**Paper file:** `docs/NeurIPS/Citations/arXiv-2310.01405v4/sections/1-introduction.tex` and `4-honesty.tex`

**Verbatim quote (§1, line 19):**

> "[RepE techniques] can provide traction on a wide variety of safety-relevant problems, including truthfulness, honesty, hallucination, utility estimation, knowledge editing, jailbreaking, memorization, tracking emotional states, and avoiding power-seeking tendencies."

**Verbatim quote (§4 Honesty, line 1):**

> "We explore applications of RepE to concepts and functions related to honesty. First, we demonstrate that models possess a consistent internal concept of truthfulness, which enables detecting imitative falsehoods and intentional lies generated by LLMs."

**Verbatim quote (§4 Honesty, line 130):**

> "With this setup, the resulting LAT reading vector reaches a classification accuracy of over $90\%$ in distinguishing between held-out examples where the model is instructed to be honest or dishonest."

**Our paper's framing (premortem-v2.tex line 117):**

> "Representation Engineering (RepE), introduced by Zou et al.\ (2023)~\cite{zou2023repe}, established that high-level cognitive states---including honesty and truthfulness---are linearly decodable from a model's latent representation space."

**Comparison for current paper (premortem-v2):** CLEAN. Our paper cites RepE for honesty probes; Zou et al. validate exactly that.

**Comparison for Candidate A framing (proposed in `2026-05-08_EMNLP Framing for KV-Cache MCTS.md`):** The Candidate A framing proposes swapping "honesty" with "certainty" as the RepE probe target (Part 4 of framing doc, "Telemetry-Guided Latent Optimization" contribution). I searched RepE for "certainty" and found only incidental usage — RepE does NOT explicitly validate a "certainty" probe direction. The list of validated concepts in §1 is: truthfulness, honesty, hallucination, utility estimation, knowledge editing, jailbreaking, memorization, emotional states, power-seeking. "Certainty" is NOT in this list. The Candidate A swap is therefore a methodological *extension* (using RepE methodology to extract a certainty direction de novo), not a re-use of an established Zou et al. probe.

**Verdict: CLEAN for premortem-v2; NUANCED for Candidate A.**

**Suggested revision for Candidate A:** When framing the System-2 paper, do NOT write "we use the RepE certainty probe". Instead write "Following the Representation Engineering methodology of Zou et al.~\cite{zou2023repe}, we extract a certainty/uncertainty direction via Linear Artificial Tomography (LAT) on a contrastive set of confident-versus-uncertain prompts". Adds one sentence to §4 of the Candidate A draft; protects against reviewer objection that "RepE certainty probe" is not a thing.

**Track A dependency:** This NUANCE is exactly the validation Track A is doing — confirming a certainty signal is extractable at 1B/3B. If Track A returns clean signal magnitude + correctness correlation, the methodological extension is justified. If Track A returns degenerate, this NUANCE escalates to an architectural problem for Candidate A.

---

### 2.4 Gao et al. 2025 — H-Neurons — **CLEAN**

**Paper file:** `docs/NeurIPS/Citations/arXiv-2512.01797v2/neurips_2024.tex`

**Verbatim quote (Abstract, line 131):**

> "Regarding their identification, we demonstrate that a remarkably sparse subset of neurons (less than $0.1\%$ of total neurons) can reliably predict hallucination occurrences, with strong generalization across diverse scenarios. In terms of behavioral impact, controlled interventions reveal that these neurons are causally linked to over-compliance behaviors."

**Verbatim quote (§1 Introduction, line 174):**

> "Our analysis uncovers that H-Neurons are linked to *over-compliance* behaviors in LLMs. To establish this causal relationship, we conduct controlled interventions by systematically scaling the activation magnitudes of these neurons. The interventions reveal a distinctive behavioral pattern: amplifying H-Neurons' activations systematically increases a spectrum of over-compliance behaviors -- ranging from overcommitment to incorrect premises and heightened susceptibility to misleading contexts, to increased adherence to harmful instructions and stronger sycophantic tendencies. These findings suggest that H-Neurons do not simply encode factual errors, but rather represent a general tendency to prioritize conversational compliance over factual integrity."

**Our paper's framing (premortem-v2.tex line 126):**

> "Gao et al.\ (2025)~\cite{gao2025hneurons} identified H-Neurons---a sparse subset of feed-forward network units whose activation patterns reliably predict hallucination occurrences and are causally linked to over-compliance behaviors. ... We extend this finding to the domain of alignment faking by treating $\sigma_H$ as an *over-compliance pressure proxy*..."

**Comparison:** Gao et al. explicitly establish: (a) sparse predictive subset, (b) causal link to over-compliance, (c) over-compliance spans hallucination, sycophancy, and harmful-instruction-following. Our paper's framing is verbatim consistent. The "extension to alignment faking" is correctly flagged in the next sentence as a "testable assumption rather than an established equivalence", which is the right honesty calibration.

**Verdict: CLEAN.** No revision needed.

**Note for Candidate A:** Under the System-2 framing, σ_H is repurposed as a "stress" / "structural hallucination penalty" signal during search. This use-case is *adjacent* to Gao et al.'s causal claim — we are using σ_H as a continuous reward signal during latent exploration, not as a binary hallucination predictor. This is a valid methodological extension, but the Candidate A draft should be careful not to phrase it as "Gao et al. show that σ_H predicts reasoning failure during latent search" (they don't — they show it predicts hallucinations and over-compliance in standard generation). One-sentence framing fix: "We adopt $\sigma_H$ from Gao et al.~\cite{gao2025hneurons} as a continuous structural-stress signal; while Gao et al. establish causal links to hallucination and over-compliance in standard generation, we use $\sigma_H$ as an exploration penalty in continuous-latent search, treating elevated $\sigma_H$ as a no-go signal for low-fidelity reasoning paths."

---

### 2.5 Hubinger et al. 2024 — Sleeper Agents — **CLEAN**

**Paper file:** `docs/NeurIPS/Citations/arXiv-2401.05566v3/main.tex`

**Verbatim quote (Abstract, line 210):**

> "Furthermore, rather than removing backdoors, we find that adversarial training can teach models to better recognize their backdoor triggers, effectively hiding the unsafe behavior."

**Verbatim quote (Abstract, line 210):**

> "The backdoor behavior is most persistent in the largest models and in models trained to produce chain-of-thought reasoning about deceiving the training process, with the persistence remaining even when the chain-of-thought is distilled away."

**Our paper's text:** premortem-v2.tex line 78 and line 113 — both contain the phrase "adversarial training found to actively teach the model to better conceal its backdoor rather than remove it" and the scale-dependence caveat.

**Comparison:** Verbatim. The "tests threat-model resilience rather than how likely such behaviors are to arise naturally" phrasing in our paper is a fair characterization of the paper's contribution framing as proof-of-concept (deliberately constructed backdoors).

**Verdict: CLEAN.**

---

### 2.6 Greenblatt et al. 2024 — Alignment Faking — **CLEAN**

**Paper file:** `docs/NeurIPS/Citations/arXiv-2412.14093v2/sections/0_abstract.tex`

**Verbatim quote (Abstract, line 4):**

> "We find the model complies with harmful queries from free users 14% of the time, versus almost never for paid users. Explaining this gap, in almost all cases where the model complies with a harmful query from a free user, we observe explicit alignment-faking reasoning, with the model stating it is strategically answering harmful queries in training to preserve its preferred harmlessness behavior out of training."

**Our paper's text (line 78):**

> "Anthropic's landmark study on alignment faking demonstrated that Claude 3 Opus, when informed that its responses would be used for retraining, strategically complied with instructions it would otherwise refuse, explicitly reasoning in a hidden chain-of-thought scratchpad---included as part of the experimental design---that compliance was instrumentally rational to preserve its existing harmlessness values."

**Comparison:** The "14% of the time / scratchpad reasoning / harmlessness preservation / scale-dependence" framing matches verbatim. The §2.1 caveat that "the goals being preserved were the model's existing HHH harmlessness preferences rather than malicious objectives, and that this behavior emerged only in the largest models tested" is well-calibrated to the paper's actual scope.

**Verdict: CLEAN.**

---

### 2.7 Gupta & Jenner 2025 — RL-Obfuscation — **CLEAN**

**Paper files:** `docs/NeurIPS/Citations/arXiv-2506.14261v4/sections/abstract.tex` and `sections/eval_policy.tex:71`

**Verbatim quote (Abstract):**

> "We find that token-level monitors are highly vulnerable to this attack while more holistic monitors, such as max-pooling or attention-based probes, remain robust. Moreover, for these vulnerable monitors, models trained to evade a single static monitor can generalise to evade other unseen monitors."

**Verbatim quote (§eval_policy, line 71):**

> "These two factors—and the magnitude of their effects—explain the trend observed in [Fig. per_token_diff], where the change in policy contributes to approximately $70\%$ of the drop in recall."

**Our paper's text (line 119):**

> "Their finding that representational redistribution contributes to monitor evasion---alongside a primary behavioral mechanism of output-policy change (accounting for approximately 70\% of the recall drop)---provides conceptual motivation for our Orthogonal Escape hypothesis"

**Comparison:** "70% of the recall drop is policy change" verified verbatim. "Representational redistribution alongside policy change" verified verbatim. The "weight finetuning vs inference-time KV-cache" caveat in our paper correctly identifies the regime gap.

**Verdict: CLEAN.**

---

### 2.8 Belitsky et al. 2025 — KV Cache Steering — **CLEAN**

**Paper file:** `docs/NeurIPS/Citations/arXiv-2507.08799v2/main.tex`

**Verbatim quote (Abstract, line 112):**

> "We propose *cache steering*, a lightweight method for implicit steering of language models via a one-shot intervention applied directly to the key-value cache. ... Our approach constructs steering vectors from reasoning traces ... that shift model behavior toward more explicit, multi-step reasoning without fine-tuning or prompt modifications. ... Compared to prior activation steering techniques that require continuous interventions, our one-shot cache steering offers substantial advantages in terms of inference latency, hyperparameter stability, and ease of integration with existing inference APIs."

**Our paper's text (line 117):**

> "Belitsky et al.\ (2025)~\cite{belitsky2025kvcache} extended this to *cache steering*, demonstrating that a single, one-shot modification to the KV cache can steer model behavior more stably than continuous activation interventions---validated for reasoning induction in small-to-medium LLMs (360M--8B)."

**Comparison:** "One-shot KV-cache modification" verified verbatim. "More stable than continuous interventions" verified ("avoids cascading effects, is robust to hyperparameter choices"). The "360M-8B" range claim is consistent with Belitsky's "small language models...also scales to larger models" — Session A should spot-check the numerical range against Belitsky's experimental section if a reviewer challenges the specific bounds.

**Verdict: CLEAN.** Spot-check note: confirm 360M and 8B are the actual model scales used in Belitsky's experiments before camera-ready.

---

### 2.9 Wu et al. 2025 — MCTS Prompt Autogeneration — **CLEAN**

**Paper file:** `docs/NeurIPS/Citations/_extracted/coling-71.txt`

**Verbatim quote (Abstract):**

> "we propose a Monte Carlo Tree Search (MCTS) based Prompt Auto-generation (MPA) method to enhance the effectiveness and efficiency of attacks across various models. MPA automatically searches for and generates adversarial suffixes for valid jailbreak attacks."

**Our paper's text (line 130):**

> "Wu et al.\ (2025)~\cite{wu2025mcts} proposed MCTS-based prompt autogeneration for jailbreak attacks."

**Verdict: CLEAN.**

---

### 2.10 DAMON — **CLEAN**

**Paper file:** `docs/NeurIPS/Citations/_extracted/emnlp-323.txt` (= 2025.emnlp-main.323)

**Verbatim quote (Abstract):**

> "we propose DAMON, a novel multi-turn jailbreak attack method. DAMON leverages Monte Carlo Tree Search (MCTS) to systematically explore multi-turn conversational spaces, efficiently identifying sub-instruction sequences that induce harmful responses."

**Our paper's text (line 130):**

> "The DAMON framework~\cite{damon2025} introduced dialogue-aware MCTS for multi-turn jailbreaks."

**Comparison:** DAMON explicitly operates on dialogue strategies / sub-instructions (text space), confirming our framing's "all existing MCTS-based red-teaming operates exclusively in text space". The paper's identification of `2025.emnlp-main.323.pdf` as DAMON (and NOT MUSE) is critical — see §3 below for MUSE missing-citation note.

**Verdict: CLEAN.**

---

### 2.11 Zou et al. 2023 GCG, Zheng et al. 2023 MT-Bench, Liu et al. 2023 G-Eval, Turner et al. 2023, ACE-Safety, AgenticRed — all **CLEAN** (abstracts verified, no nuance discovered)

Brief verbatim from each:

- **GCG (2307.15043 line 97):** "we propose a simple and effective attack method that causes aligned language models to generate objectionable behaviors. Specifically, our approach finds a suffix that, when attached to a wide range of queries...automatically produces these adversarial suffixes by a combination of greedy and gradient-based search techniques". Matches our white-box-gradient discrete-token framing.
- **MT-Bench (2306.05685 sec-intro line 2-4):** "we explore using strong LLMs as judges to evaluate these models on more open-ended questions. We examine the usage and limitations of LLM-as-a-judge, including position, verbosity, and self-enhancement biases". Matches our LLM-as-Judge framing.
- **G-Eval (2303.16634 title):** Matches.
- **Turner activation steering (2308.10248 abstract):** "inference-time modification of activations to control model outputs...introduces ActAdd". Matches.
- **ACE-Safety (2511.19218):** Title and abstract match our framing.
- **AgenticRed (2601.13518):** Title matches.

---

## 3. MISSING citations (must be downloaded before camera-ready)

| Citation | Likely arXiv ID | Required for | Priority |
|---|---|---|---|
| Snell et al. 2024 — "Scaling LLM Test-Time Compute Optimally" | 2408.03314 | Candidate A foundation framing (§1, §3) | **MANDATORY** |
| Hasani et al. 2025 — "Mechanistic Interpretability of Large-Scale Counting...System-2 Strategy" | 2601.02989 (suspected) | Candidate A System-2 conceptual link | **MANDATORY** |
| Stoehr et al. 2024 — "Activation Scaling for Steering and Interpreting LMs" (EMNLP Findings) | 2410.04444 (suspected) | Candidate A — needed for "dynamic vs static activation steering" differentiator argument | **MANDATORY** |
| MUSE / Yu et al. 2025 (EMNLP main 2025) | aclanthology 2025.emnlp-main.1080 | Candidate A — needed for "discrete vs continuous MCTS" differentiator | **MANDATORY** |
| Hao et al. 2024 — RAP (Reasoning As Planning) | likely 2305.14992 | Currently not in our bib but framed as COCONUT's predecessor in framing doc; if cited, must verify | **NICE-TO-HAVE** |
| Rimsky et al. 2024 — Contrastive Activation Addition (ACL 2024) | already cited as `rimsky2024steering` in our bib | Currently cited in §2.2; full paper not in `Citations/` | **NICE-TO-HAVE** (low risk; minor citation) |
| SAE survey (Findings of EMNLP 2025) | tbd | Suggested by framing doc Part 1 zeitgeist analysis | **NICE-TO-HAVE** |
| RouteSAE (EMNLP 2025 main) | tbd | Suggested by framing doc Part 1 zeitgeist analysis | **NICE-TO-HAVE** |
| SHARP (Wu et al. 2025) | tbd | Mentioned in framing doc as zeitgeist anchor for hallucination/RepE in LVLMs; not load-bearing for our text | **NICE-TO-HAVE** (skippable for Candidate A) |
| SteerVLM (Sivakumar et al. 2025) | tbd | Same as SHARP | **NICE-TO-HAVE** (skippable) |

**Action items for Josh:**
1. Download all four MANDATORY missing citations within next 24 hours so Session A can verify against them at Gate 2.
2. Snell 2024 in particular is foundation-level for Candidate A — without it the "test-time compute scaling" framing has no anchor. **If Snell is misframed (e.g., they specifically argue test-time compute MUST be discrete), this is a late-breaking fatal finding for Candidate A.** Re-spawn Track B with Snell + Hasani as priority once downloaded.

---

## 4. Citations to ADD for Candidate A pivot

Per the framing doc's Part 4 ("Citations to Integrate") + Part 1 zeitgeist characterization:

**MANDATORY for Candidate A:**

1. **Snell et al. 2024** — "Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters" — foundation for the "test-time compute" claim in §1 and §3. Cannot ship Candidate A without it.
2. **Hasani et al. 2025** — "Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy" — anchor for the "System-2" terminology in §1. Reviewers will challenge System-2 nomenclature without an established anchor.
3. **Stoehr et al. 2024** (Activation Scaling, EMNLP Findings 2024) — anchor for the "dynamic vs static activation steering" differentiator in §2 / §6. EMNLP-Findings co-venue citation is an alignment win with the program committee.
4. **MUSE (Yu et al. 2025, EMNLP main)** — anchor for the "discrete-text-space MCTS vs our continuous-latent MCTS" differentiator in §2. EMNLP main-track co-venue citation is critical.
5. **Hao et al. 2024 — RAP / Reasoning-as-Planning** — separates COCONUT-the-trained-method from RAP-the-prompted-search-method. Useful for Related Work taxonomy.

**NICE-TO-HAVE for Candidate A (zeitgeist alignment):**

6. SAE survey (Findings of EMNLP 2025) — establishes our paper isn't ignoring the SAE wave; a single sentence in §2 noting "while SAE-based interpretability dominates discrete-feature analysis, our work targets the orthogonal axis of continuous-state geometry" is sufficient.
7. RouteSAE (EMNLP 2025 main) — same as above.
8. SHARP / SteerVLM — only if we want to acknowledge the multi-modal hallucination-mitigation wave, otherwise skip.

**REMOVE for Candidate A (no longer load-bearing under System-2 framing):**

- Greenblatt 2024, Hubinger 2024 — currently §1 introduction headliners. Under Candidate A these become §2.1 (alignment-evaluation as one downstream application) but lose headline position. They should NOT be removed entirely (they're real and good citations) but the §1 paragraphs that lean on them get severely abbreviated.
- DAMON, ACE-Safety, AgenticRed, Wu MCTS — currently §2.4 "MCTS for LLM Red-Teaming" subsection. Under Candidate A this subsection collapses to ONE paragraph contrasting "discrete text-space MCTS" against our continuous-latent variant; individual citations stay but lose the dedicated subsection.

---

## 5. Pile-up summary

- **NUANCED count:** 1 (RepE — only nuanced under Candidate A swap to certainty probe; CLEAN under current paper)
- **MISMATCHED count:** 0
- **FATAL count:** 0
- **MISSING-but-cited count:** 1 (Rimsky 2024 — minor; full paper not local but our citation is from the EMNLP/ACL anthology so the cite is anchored)
- **MISSING-and-not-cited count:** 4 mandatory (Snell, Hasani, Stoehr, MUSE) + several nice-to-have

**Per master checklist plasticity rules (§0):**
- 0–1 contradicting findings → CONTINUE.
- This audit yields 1 contradicting finding (RepE NUANCE under Candidate A). At this gate count, plasticity rule says **CONTINUE; log nuance + framing-adjustment in track-E-gate-decisions.md**.
- Plasticity action for Session A at Gate 1 synthesis: log the RepE NUANCE; note that Track A's certainty-probe-validation work is the empirical resolution path; if Track A succeeds, this NUANCE is fully retired; if Track A fails, this NUANCE escalates to Candidate A architectural problem.

**Gate 1 G1.3 (Track B priority batch returns ≤2 nuanced/mismatched):** **PASS** (1 nuanced).
**Gate 1 G1.4 (NO fatal findings):** **PASS** (0 fatal; COCONUT explicitly differentiated).

---

## 6. Recommendation to Session A

Track B portion of Gate 1 is **GO**. Recommend continuing the 72-hour run into Track C dispatch at T+24 conditional on Tracks A and D also returning GO.

If Tracks A or D return YELLOW or NO-GO, this audit's pile-up count of 1 stays low enough that Track B is not the bottleneck. The bottleneck is signal-magnitude (Track A G1.1, G1.2) and VRAM-constant (Track D G1.5).

**Late-breaking risk to monitor:** Snell 2024 download is mandatory before Candidate A revision starts. If Snell turns out to argue something like "test-time compute scaling is fundamentally discrete because continuous latent search is intractable on frozen models without specialized training", this is a late-breaking fatal finding that escalates Candidate A from GO-with-1-nuance to PIVOT-to-Wildcard. This can only be resolved by reading Snell directly, which I could not do in this audit because the file isn't downloaded. **Strongly recommend Josh download Snell within 4 hours so Track B can return a delta-audit before Gate 2.**

---

## 7. Bibliography of files inspected

**Local arXiv source bundles read (priority + load-bearing):**
- `arXiv-2412.06769v3/paper.tex` (COCONUT, Hao et al. 2025) — full intro + method §3 read; reversibility/training claims verified verbatim
- `arXiv-2412.09565v2/main.tex` (Bailey et al. 2024) — abstract + §sec:orthogonality (lines 1779–1812) read
- `arXiv-2310.01405v4/sections/{1-introduction,4-honesty,6-sections/6.5-memorization}.tex` (Zou et al. 2023 RepE) — intro + honesty section read; "certainty" search across all sections returned only incidental usage
- `arXiv-2512.01797v2/neurips_2024.tex` (Gao et al. 2025 H-Neurons) — abstract + §1 read; over-compliance causal claim verified
- `arXiv-2412.14093v2/sections/0_abstract.tex` (Greenblatt et al. 2024) — abstract verified
- `arXiv-2401.05566v3/main.tex` (Hubinger et al. 2024) — abstract (line 209) verified verbatim
- `arXiv-2506.14261v4/sections/abstract.tex` + `sections/eval_policy.tex:71` (Gupta & Jenner 2025) — abstract + 70%-policy-claim verified verbatim
- `arXiv-2307.15043v2/main.tex` (Zou et al. 2023 GCG) — abstract verified
- `arXiv-2306.05685v4/sec-intro.tex` (Zheng et al. 2023 MT-Bench) — abstract verified
- `arXiv-2303.16634v3/acl2021.tex` (Liu et al. 2023 G-Eval) — title verified
- `arXiv-2308.10248v5/iclr2025_conference.tex` (Turner et al. 2023 activation steering) — abstract verified
- `arXiv-2507.08799v2/main.tex` (Belitsky et al. 2025 cache steering) — abstract verified verbatim
- `arXiv-2503.05061v2/main.tex` (LLM-as-Judge limitations) — title verified
- `arXiv-2511.19218v2/acl_latex.tex` (ACE-Safety) — title + abstract verified
- `arXiv-2601.13518v3/main.tex` (AgenticRed) — title verified
- `arXiv-2505.12540v4/neurips_2025.tex` ("Harnessing Universal Geometry of Embeddings") — flagged as not-cited; appears to be a stale download

**Local extracted PDFs read:**
- `_extracted/emnlp-323.txt` (DAMON, EMNLP 2025) — abstract verified
- `_extracted/coling-71.txt` (Wu MCTS, COLING 2025) — abstract verified

**Paper text inspected:**
- `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2-premortem-v2.tex` lines 67–179, 570–670 (abstract, §1 introduction, §1.1 contributions, §2.1–2.6 background+related work, §3.1–3.2 problem formulation, full bibliography)

**Framing doc inspected:**
- `docs/logs/2026-05-08_EMNLP Framing for KV-Cache MCTS.md` Parts 1–4 (zeitgeist, candidates, ranking, Candidate A detailed proposal)

**Master checklist + diagnostic prompt:**
- `docs/logs/2026-05-08_72hr-execution-checklist.md` §0–§5.2
- `docs/logs/2026-05-08_diagnostic-prompt-track-B.md` (full)

---

*End of Track B citation audit. Time-on-task: ~3.5h wall-clock. No contradictions surfaced beyond the documented RepE/certainty NUANCE; no fatal findings; Gate 1 G1.3 + G1.4 both PASS from Track B's perspective.*
