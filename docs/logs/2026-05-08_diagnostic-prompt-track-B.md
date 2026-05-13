You are a diagnostic Claude session. Your task is to verify that the load-bearing citations in our paper (under the new Candidate A "System-2 Latent Search" framing) accurately reflect what the cited papers actually claim. The user has downloaded every cited paper to docs/NeurIPS/Citations/. Your job is to read each one and produce an audit report.

CONTEXT FILES:
1. docs/logs/2026-05-08_EMNLP Framing for KV-Cache MCTS.md (the proposed framing)
2. docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2-premortem-v2.tex (current paper state)
3. docs/NeurIPS/Citations/ (PDFs and arXiv source bundles)
4. docs/NeurIPS/Citations/_extracted/ (some pre-extracted text)

PRIORITY CITATIONS (verify these first, in this order):

1. Hao et al. 2025 — "Training Large Language Models to Reason in a Continuous Latent Space" (COCONUT). Likely arXiv-2412.06769. Required verification: does COCONUT actually train the model? Does it operate in KV-cache space or residual stream? Is exact reversibility a property COCONUT claims? Our framing claims we are training-free and reversible — confirm we differentiate cleanly. **THIS IS A FATAL-FINDING CHECK.**

2. Snell et al. 2024 — "Scaling LLM Test-Time Compute Optimally..." (likely NOT in Citations folder; flag as missing if so). What's the actual claim about test-time compute? Discrete or continuous? Sample-based or search-based? Our paper would frame it as foundation for inference-time search — confirm fit.

3. Zou et al. 2023 — RepE (arXiv-2310.01405). What probes are validated? For honesty specifically. Does the paper provide a "certainty" formulation, or is that derived? Our paper claims we are using their probes — confirm whether certainty is in their framework or a stretch.

4. Gao et al. 2025 — H-Neurons (arXiv-2512.01797 likely). What's the actual causal claim about H-Neurons? Hallucination only, or broader (over-compliance, sycophancy)? Our paper now uses σ_H as a "stress" signal in a System-2 framing — confirm or flag as nuanced.

5. Hasani et al. 2025 — "Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy" (arXiv-2601.02989). What is their actual System-2 claim? Search-based, or behavioral? How do they connect mechanistic interpretability to System-2?

6. Stoehr et al. 2024 — "Activation Scaling for Steering and Interpreting Language Models" (EMNLP Findings 2024). What's the steering method? Static vector add, or something more dynamic? What's our differentiator from them?

7. Bailey et al. 2024 — "Obfuscated Activations Bypass LLM Latent-Space Defenses" (arXiv-2412.09565). Already cited extensively. Under the new System-2 framing, does the Bailey relationship still hold (i.e., "complementary measurement"), or does the framing shift change what's actually relevant?

8. Yu et al. 2025 — MUSE (EMNLP main 2025, aclanthology 2025.emnlp-main.1080). Likely 2025.emnlp-main.323.pdf or coling-71.txt — verify identification. What's the multi-turn MCTS algorithm? Does it operate on text or activations? How does our continuous-latent-search differ?

ADDITIONAL CITATIONS (if time permits, lower priority):
9. SHARP (Wu et al. 2025), SteerVLM (Sivakumar et al. 2025), Greenblatt et al. 2024 (alignment faking), Hubinger et al. 2024 (sleeper agents), Gupta & Jenner 2025 (RL-Obfuscation), Zheng et al. 2023 (LLM-as-Judge), Liu et al. 2023 (G-Eval), Hao et al. 2024 (COCONUT predecessor if different from #1), and any others present in Citations/.

PROTOCOL FOR EACH CITATION:
1. Locate the paper file. If not present, mark "MISSING — needs download".
2. Read abstract + introduction + main claim sections. ~15 min per paper.
3. Extract the precise claim our paper builds on, verbatim from the cited paper, with section reference.
4. Compare to how our paper currently frames the citation (premortem-v2.tex).
5. Verdict: CLEAN (matches cleanly), NUANCED (cited claim is more limited/different than our usage), or MISMATCHED (citation does not support our usage at all).
6. If NUANCED or MISMATCHED, draft a one-sentence revision to our paper's relevant text that would resolve the issue.

OUTPUT FILE: docs/logs/2026-05-09_track-B-citation-audit.md

REQUIRED REPORT SECTIONS:
1. Per-citation table: Name | arXiv ID | File location | Verdict | Notes
2. Detailed entry for each citation with verbatim quote + comparison + suggested revision (if NUANCED or MISMATCHED)
3. Summary of fatal findings (if any) — ESPECIALLY whether COCONUT does training-free latent search (Gate 1 fatal #1)
4. Summary of pile-up: count of NUANCED + MISMATCHED across all verified citations (per master checklist plasticity rule)
5. List of MISSING citations that need to be downloaded before the paper can be safely revised
6. List of citations we should ADD (per Gemini round-2 zeitgeist characterization: SAE survey, RouteSAE, MUSE if not already, Stoehr, Hao et al. 2025 COCONUT, Snell et al. 2024) — note which are mandatory vs nice-to-have

DO NOT:
- Edit the .tex file directly (Session A handles paper text)
- Skip a citation because "it looks fine" — read it
- Conclude CLEAN without a verbatim quote
- Defer documentation
