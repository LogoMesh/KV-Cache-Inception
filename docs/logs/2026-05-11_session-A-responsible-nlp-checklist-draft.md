# Session A — Responsible NLP Research Checklist — First-Pass Q&A Draft

**Authored:** 2026-05-11 (Day 3 of ARR-cycle execution plan v2)
**Audience:** Submission portal (OpenReview Responsible NLP Checklist form, completed by Josh at Day 15 submission)
**Status:** First-pass draft. **All 17 questions transcribed verbatim from primary source** ([https://aclrollingreview.org/responsibleNLPresearch/](https://aclrollingreview.org/responsibleNLPresearch/), fetched 2026-05-11).
**Final version:** `docs/logs/2026-05-19_responsible-nlp-checklist-final.md` (Day 10 of plan v2) — answers must cite paper § references that exist post-Day-9 §5 + §A draft applications.

---

## Section A — For Every Submission

### A1. Did you describe the limitations of your work?
**Answer:** Yes
**Paper reference:** §7.3 Limitations (canonical TeX:533-551) — 8 numbered limitations under Option D+ Dimensional Escape framing.

### A2. Did you discuss any potential risks of your work?
**Answer:** Yes
**Paper reference:** §7.2 Broader Impact (TeX:529-531) — dual-use considerations addressed via 3 mitigations (offline environment, cryptographic audit trail, responsible disclosure). Also §7.6 Ethical Considerations (TeX:578-580).

---

## Section B — Did You Use or Create Scientific Artifacts?

**Pre-answer:** YES — we used Meta's Llama 3.2-1B-Instruct and Llama 3.2-3B-Instruct (foundation models, used as-is, no further training); and the ARC-Easy, TruthfulQA mc1, HellaSwag (validation), and our own hand-constructed Factual Recall (100-item) test sets.

### B1. Did you cite the creators of artifacts you used?
**Answer:** Yes
**Paper reference:** Bibliography — citations for Touvron et al. (Llama), Clark et al. (ARC), Lin et al. (TruthfulQA), Zellers et al. (HellaSwag). Llama 3.2 specifically cited via Meta (2024) bibitem to be added per J2 draft.

### B2. Did you discuss the license or terms for use and/or distribution of any artifacts?
**Answer:** Yes
**Paper reference:** §7.6 Ethical Considerations (TeX:578-588) — Llama 3.2 Community License attribution paragraph added per J2. ARC-Easy / TruthfulQA / HellaSwag are CC-BY-SA / CC-BY-NC / MIT respectively; standard academic-use terms. (To verify by Day 10 — sub-task.)

### B3. Did you discuss if your use of existing artifact(s) was consistent with their intended use?
**Answer:** Yes
**Paper reference:** §5.2 Phase A (TeX:336-350) — Llama 3.2 used for evaluation only, consistent with research use. ARC-Easy etc. used for question-answering evaluation, consistent with benchmark intent.

### B4. Did you discuss the steps taken to check whether the data contains any information that names or uniquely identifies individual people or offensive content?
**Answer:** N/A (with justification)
**Justification:** All datasets are publicly available academic benchmarks. ARC-Easy is grade-school science MCQ (no PII). TruthfulQA is factual-claim MCQ (no PII). HellaSwag is commonsense continuation (no PII). Hand-constructed Factual Recall items are encyclopedic facts (capitals, dates, arithmetic, common knowledge — no PII). No personally identifying information in any source artifact.

### B5. Did you provide documentation of the artifacts, e.g., coverage of domains, languages?
**Answer:** Yes
**Paper reference:** §A Reproducibility Appendix (to be drafted Day 3-4 of plan v2) — full per-class breakdown of prompt sets, source benchmarks, evaluation protocols. Cartography report `docs/logs/2026-05-11_track-G-cartography-report.md` §1 documents the 4 prompt classes used as Experiment 2 evidence.

### B6. Did you report relevant statistics like the number of examples, details of train/test/dev splits?
**Answer:** Yes
**Paper reference:** §5 Experiments (TeX:351-450) — n=200 ARC-Easy items for Track F (Experiment 1); n=100 per class × 4 classes (385 valid Cartography records per scale) for Track G (Experiment 2). Splits documented: ARC-Easy `test` split; TruthfulQA `validation`; HellaSwag `validation`; Factual Recall hand-constructed (no split).

---

## Section C — Did You Run Computational Experiments?

**Pre-answer:** YES — Track A (entropy correlation, n=400), Track C (benchmark calibration, n=200), Track D (memory profiling), Track F (entropy-MCTS negative control, n=600 records), Track G (Cartography, n=770 records [385 per scale × 2 methodologies]).

### C1. Did you report the number of parameters in the models used, the total computational budget?
**Answer:** Yes
**Paper reference:** §A Reproducibility Appendix (to be drafted Day 3-4). Model parameters: Llama 3.2-1B (1.24B params), Llama 3.2-3B (3.21B params), per HuggingFace model cards. Total compute budget: Track F (52 min wall × 2 scales); Track G MCTS (17+30 min); Track G sweep (5.5+10.3 min); Tracks A/C/D ~70 min combined. Single NVIDIA RTX 3060 12GB consumer GPU throughout. No multi-GPU, no H100, no cloud spend.

### C2. Did you discuss the experimental setup, including hyperparameter search and best-found hyperparameter values?
**Answer:** Yes
**Paper reference:** §5.2 Implementation Notes (TeX:317-319) discusses Phase 2 / Phase 3 distinction. §A Reproducibility Appendix (Day 3-4) covers MCTS hyperparameters: depth=3, branching=27 paths, alpha-set {0.1, 0.5, 1.0}, entropy reward, fp16 precision. **No hyperparameter search performed** — values are fixed per the protocol design (Track F protocol; Cartography MCTS-primary protocol). No best-found values to report.

### C3. Did you report descriptive statistics about your results?
**Answer:** Yes
**Paper reference:** §5 Experiments — Track F McNemar p=1.0, Δ=−0.5pp/+0.5pp at the two scales (Experiment 1). Track G MCTS modal best-path α-tuple distributions per class × scale (Experiment 2). All percentages, modal counts, and entropy values reported with sample sizes.

### C4. If you used existing packages, did you report the implementation, model, and parameter settings used?
**Answer:** Yes
**Paper reference:** §A Reproducibility Appendix (Day 3-4). Packages: `transformers==4.45.0` (HuggingFace), `torch==2.4.0` (CUDA 12.1), `datasets==2.20.0`, fp16 inference, `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct` checkpoints. RNG seed `--seed 42` controls steering direction and MCTS path-sampling across all reported runs.

---

## Section D — Did You Use Human Annotators or Research with Human Participants?

**Pre-answer:** NO — no human annotators, no human participants, no crowdsourced data collection. All evaluations use automated answer-matching against ground-truth labels from public benchmark splits.

### D1. Did you report the full text of instructions given to participants?
**Answer:** N/A — no human participants.

### D2. Did you report information about how you recruited and paid participants?
**Answer:** N/A — no human participants.

### D3. Did you discuss whether and how consent was obtained?
**Answer:** N/A — no human participants.

### D4. Was the data collection protocol approved (or determined exempt) by an ethics review board?
**Answer:** N/A — no human participants, no data collection from people.

### D5. Did you report the basic demographic and geographic characteristics of the annotator population?
**Answer:** N/A — no human annotators.

---

## Section E — Did You Use AI Assistants?

### E1. If you used any AI assistants, did you include information about your use?
**Answer:** Yes
**Paper reference:** §7.6 Ethical Considerations (additional sentence to be added Day 3) — disclosure of AI assistant use in research and writing process. Specifically: (a) **Claude (Anthropic)** for paper drafting, code review, and analysis-synthesis across the May 5–25 execution window; (b) **Gemini (Google)** for adversarial strategy review (round 4 reward-design strategy report, 2026-05-09) that contributed to the Option D+ pivot decision. AI assistant outputs were verified against primary artifacts (track reports, raw experimental data) before incorporation; no model-generated experimental claims appear in the paper without primary-source verification per the project's primary-source-fidelity discipline.

**DRAFT TeX addition for §7.6** (TeX:580 paragraph append):

```latex
\paragraph{AI assistant use disclosure.}
This work made use of AI assistants during paper drafting, code review, and analysis-synthesis: Anthropic's Claude was used as a coding and writing assistant across the research execution window; Google's Gemini was used for one adversarial-review round of reward-design strategy. All AI-generated outputs were verified against primary experimental artifacts (track reports, raw JSON result files) prior to incorporation. No empirical claims in this paper rest on AI-generated summaries; numerical values, methodological descriptions, and experimental findings trace to primary-source artifacts.
```

---

## Open items requiring resolution before Day 10 finalization

1. **§A Reproducibility Appendix must exist** by Day 10 — otherwise B5, C1, C2, C4 answers all reference a non-existent section. Schedule: Day 3-4 v1 draft, Day 10 final.
2. **Llama 3.2 license bibitem** must exist — added per J2 draft.
3. **Dataset license verification** — confirm CC-BY-SA / CC-BY-NC / MIT for ARC-Easy / TruthfulQA / HellaSwag (B2 sub-task; Day 6-7 of plan).
4. **AI assistant disclosure** TeX addition — apply Day 3-4 to §7.6.
5. **§7.6 §Ethical Considerations** may need a §-label (`sec:ethical-considerations`) for the checklist answers to reference cleanly. Currently unlabeled.
6. **B1 bibliography check** — verify Touvron, Clark, Lin, Zellers citations are present and accurate (Track B-delta cleanup may already cover this; verify Day 9 cross-ref audit).

---

*End of first-pass Responsible NLP Checklist draft. 17 questions answered (12 substantive, 5 N/A). All references to canonical TeX line numbers verified as of 2026-05-11 EOD. Final version Day 10 of plan v2 must re-verify line numbers + § references after Day 7 §5 rewrite and Day 9-10 §A appendix application.*
