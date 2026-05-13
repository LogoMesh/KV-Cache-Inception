# Session A v10 Prose Draft — §A Reproducibility Appendix (v1, ARR review)

**Authored:** 2026-05-13 (Day 4 P2 of plan v2 execution).
**Audience:** Session B (TeX application target — `\appendix \section{Reproducibility}` per plan v2 Day 10).
**Status:** Day 4 P2 first draft. **All numeric values, hyperparameters, and hardware figures traceable to primary artifacts** — Track A/C/D/F reports + `logomesh/kv_mcts.py` MCTSConfig dataclass (all re-read 2026-05-13).
**Scope:** ARR review version, **target 1 page** in ACL two-column format. Phase 3 camera-ready expansion to 3–4 pages will add HuggingFace checkpoint commit SHAs, exhaustive library version pins, full per-script command-line argument tables, and individual driver-script reproduction commands. Items deferred to supplementary materials package are flagged inline as "recorded in the supplementary materials package."

---

## 1. Pivot summary (NEW appendix; not a replacement)

| Element | Status before this draft | Status after Day-10 application |
|---|---|---|
| `\appendix` block | Absent from canonical TeX | New — wraps all post-bibliography content |
| `\section{Reproducibility}` | Absent | New — 5 subsections (Hardware / Models / MCTSConfig / Per-track / Seeds-and-caveats) |
| Cross-references | Tracks A/C/D/F cited inline in §5 + §6 + §7.4 prose; no consolidated reproducibility section | All four tracks' protocols consolidated; reverse-referenced from §5 Experiment 1, §6 Memory Complexity, §7.4 Measurement Prerequisites |
| Supplementary anchor | None | Single URL placeholder for `anonymous.4open.science/r/kv-mcts-dimensional-escape-XXXX` (final slug at Day 12 supplementary build) |

The appendix is unlimited per ACL ARR rules — does not consume the 8-page review body budget. v1 targets ~50 body lines of TeX (≈ 1 column-page at ACL two-column density).

---

## 2. Replacement TeX (new content inserted at end of canonical TeX, before `\end{document}`)

**TeX target:** Insert immediately after `\end{thebibliography}` and before `\end{document}` (currently line 700 + line 701 + line 702 in canonical TeX after the natbib REVISION marker + `\usepackage` additions).

```latex
% [REVISION | reproducibility-appendix-v1-D+ | 2026-05-13 | New §A Reproducibility Appendix v1 (ARR review version, target 1 page). Consolidates per-track protocols + hardware/software stack + MCTSConfig defaults from Tracks A/C/D/F primary reports and logomesh/kv_mcts.py dataclass (commit 3c5d741). Three metadata gaps acknowledged inline and deferred to the supplementary materials package: (1) HuggingFace checkpoint commit SHAs for meta-llama/Llama-3.2-{1,3}B-Instruct; (2) exhaustive library version pins beyond PyTorch 2.6.0+cu124; (3) full per-script command-line argument tables. Phase 3 camera-ready expansion (3-4 pages) fills these. Draft source: docs/logs/2026-05-13_session-A-prose-drafts-v10-reproducibility-appendix.md. — pending audit]
\appendix
\section{Reproducibility}\label{sec:reproducibility}

\subsection{Hardware and software stack}
All experiments reported in §\ref{sec:experiments} were conducted on a single NVIDIA RTX~3060 (12~GiB VRAM, driver version 591.44) running Windows~10 Pro 10.0.19045 with CUDA~12.4 and PyTorch~2.6.0+cu124. Tokenization, model loading, and benchmark access used the HuggingFace \texttt{transformers} and \texttt{datasets} libraries; specific version pins beyond PyTorch are recorded in the supplementary materials package.

\subsection{Models and checkpoints}
We evaluate two open-weight checkpoints from Meta's Llama~3.2 family: \texttt{meta-llama/Llama-3.2-1B-Instruct} (16~transformer layers, hidden dimension $d_{\text{model}} = 2048$, 8~grouped-query KV heads, head dimension 64, vocabulary $|V| = 128{,}256$, fp16) and \texttt{meta-llama/Llama-3.2-3B-Instruct} (28~transformer layers, 8~KV heads, head dimension 128, same vocabulary, fp16). Models are loaded via \texttt{AutoModelForCausalLM.from\_pretrained(..., torch\_dtype=torch.float16)}; HuggingFace checkpoint commit SHAs are recorded in the supplementary materials package.

\subsection{Reversible MCTS configuration}
The reference implementation is at \texttt{logomesh/kv\_mcts.py}. The \texttt{MCTSConfig} dataclass defaults are \texttt{n\_nodes=50}, \texttt{branching\_factor=3}, \texttt{max\_depth=5}, \texttt{alpha\_values=(0.1, 0.5, 1.0, 2.0, 5.0)}, \texttt{exploration\_constant=1.414} (UCB1 $c = \sqrt{2}$), \texttt{steer\_layer=$-1$} (auto-middle-layer). Experiment~1 (§\ref{sec:exp1}) and Experiment~2 (§\ref{sec:exp2}) use a focused path-sampling variant of this engine: depth-$3$ trees with $27 = 3^3$ enumerated path samples drawn from a three-element alpha set $\{0.1, 0.5, 1.0\}$. The variant evaluates each path under the entropy-normalized reward (Eq.~\ref{eq:goodhart}) and re-applies the best-rewarded path permanently before generation; it does not use UCB1 tree expansion, because full enumeration of $27$ paths from $27$ leaves makes the bandit ordering redundant within the experimental budget.

\subsection{Per-track configurations}

\paragraph{Static entropy--correctness correlation (Track A).}
The static $|r| \approx 0.60$ Pearson correlation between next-token entropy and answer correctness cited in §\ref{sec:exp1} Interpretation was measured by computing the Shannon entropy $H = -\sum_i p_i \log p_i$ over $\mathrm{softmax}(\mathrm{logits}_{[-1, :]})$ from a single fp16 forward pass on the chat-templated prompt, scored against a hand-written acceptable-answer list via case-insensitive substring match. Sample: 85 factual-recall items (25 easy + 30 hard + 30 ultra), evaluated with \texttt{model.generate(do\_sample=False, max\_new\_tokens=15, output\_scores=True)}. Wall-clock approximately 13~min for both 1B and 3B scales. Driver: \texttt{scripts/diagnose\_track\_a\_entropy.py}; raw artifacts \texttt{scripts/\_track\_a\_results\_meta-llama\_Llama-3.2-\{1B,3B\}-Instruct.json}.

\paragraph{Benchmark calibration (Track C).}
The ARC-Easy 200-item subset used in Experiment~1 is the first 200 deterministic test-split items from \texttt{allenai/ai2\_arc} configuration ``ARC-Easy'' (indices $0\ldots199$, no shuffling or filtering). Prompts use the model's chat template with system message ``You are a helpful assistant. Answer concisely.'' and user-side suffix ``Answer with the single capital letter (A/B/C/D):''. The sample size was selected via the paired-McNemar power formula $n_{\text{pairs}} \approx (z_{\alpha/2}\sqrt{\pi_d} + z_\beta \sqrt{\pi_d - \delta^2})^2 / \delta^2$ at $\alpha = 0.05$, power $= 0.80$, target effect size $\delta = 0.10$ absolute; the resulting budget admits detection for discordance rates up to $\pi_d \approx 0.25$. The calibration sweep examined six candidate benchmarks (ARC-Easy, SciQ, HellaSwag, AQuA-RAT, ProofWriter, BBH-LogicalDeduction-3); SciQ qualifies as a corroboration target deferred to Phase 3. Wall-clock approximately 19~min for both scales. Driver: \texttt{scripts/diagnose\_track\_c\_benchmarks.py}.

\paragraph{VRAM measurement protocol (Track D).}
The $3.04$--$3.06 \cdot M_{\text{KV}}$ constant-factor memory result reported in §\ref{sec:memory-complexity} was measured via \texttt{torch.cuda.memory\_allocated()} for steady-state and \texttt{torch.cuda.max\_memory\_allocated()} for transient peak, cross-checked against \texttt{nvidia-smi --query-gpu=memory.used} at the largest configuration. Three configurations were swept: $(d, b, n) \in \{(3, 3, 27), (5, 3, 81), (10, 3, 1700)\}$, each with three repeats, at a fixed 4000-token seed prefill on both Llama 3.2-1B and -3B. Steady-state delta was identical across all $(d, b, n)$ cells at each scale (median over three repeats per cell), confirming branching-factor and depth independence. Wall-clock approximately 5~min for both scales. Driver: \texttt{scripts/measure\_kv\_mcts\_vram.py}.

\paragraph{Entropy-MCTS negative control (Track F / Experiment~1).}
The negative-control protocol underlying Experiment~1 evaluates three paired conditions on each of 200~ARC-Easy items at each model scale: G (greedy decoding, no MCTS); R (random-MCTS, $\hat{r}_R \sim \mathrm{Uniform}(0, 1)$); E (entropy-MCTS, $\hat{r}_E(\mathbf{h}) = -H_t / \log|V|$ from Eq.~\ref{eq:goodhart}). The MCTS arms enumerate all $27$ depth-$3$ paths from $\{0.1, 0.5, 1.0\}$; the best-rewarded path is re-applied permanently to the live KV cache before the model generates the 5-token answer continuation via \texttt{model.generate(do\_sample=False, max\_new\_tokens=5, output\_scores=True)}. The steering direction is a single random unit vector in $d_{\text{model}}$ space, seeded by \texttt{--seed 42}, shared identically across all items and across the R and E arms (the COCONUT defense; see §\ref{sec:exp1} Setup paragraph). The driver script imports \texttt{FP32Accumulator}, \texttt{\_extract\_kv\_tensors}, and \texttt{\_kv\_eval\_cache} from \texttt{logomesh/kv\_mcts.py} as read-only primitives. Wall-clock: 18.3~min (1B sweep, 5.5~s/item) and 33.7~min (3B sweep, 10.1~s/item). Driver: \texttt{scripts/diagnose\_track\_f\_negcontrol.py}; raw artifacts \texttt{scripts/\_track\_f\_results\_meta-llama\_Llama-3.2-\{1B,3B\}-Instruct.json}.

\subsection{Random seeds and reproducibility caveats}
Greedy decoding (\texttt{do\_sample=False}) is deterministic given fixed model weights, tokenizer, and hardware; Tracks A, C, and D therefore do not set a manual seed beyond the default PyTorch initialization. Experiment~1 (Track F) and Experiment~2 (Latent Cartography) seed the steering direction with \texttt{--seed 42}; the per-item random number generator for the random-MCTS arm uses \texttt{hashlib.md5(f"\{item\_idx\}:R:\{seed\}")} so per-item draws are reproducible across runs at the same \texttt{--seed}. The 5-token completion accuracies and the per-condition perplexities reported in Table~\ref{tab:exp1-results} are deterministic. VRAM measurements (Table~\ref{tab:oei-alpha-sweep}'s parent paragraph and §\ref{sec:memory-complexity}'s constant factor) exhibit ${\sim}1$--$2\%$ between-run variation traceable to PyTorch's caching-allocator block-rounding to power-of-two boundaries; this affects measured VRAM but not any computed quantity in the paper. Exact HuggingFace checkpoint commit SHAs, the full library version pin manifest (\texttt{transformers}, \texttt{datasets}, \texttt{huggingface\_hub}, \texttt{numpy}, \texttt{scipy}), and per-script command-line argument references are recorded in the anonymized supplementary materials package, available at \url{https://anonymous.4open.science/r/kv-mcts-dimensional-escape-XXXX} (the final slug is assigned at Day-12 supplementary build per plan v2 §2).
% [IMPL: COMPLETE — all driver scripts under scripts/ are read-only on logomesh/* and reproducible per their respective Appendix~A blocks in the Track A/C/D/F reports (docs/logs/2026-05-{09,10,11}_track-*-*.md).]
```

---

## 3. Notes for Session B (Day 10 application)

### Application order

1. **Insert** the entire block above (REVISION marker + `\appendix \section{Reproducibility}` + 5 subsections + IMPL line) immediately after `\end{thebibliography}` (currently at canonical TeX ~line 700) and before `\end{document}` (currently at line 702).
2. **Verify** that the new `\label{sec:reproducibility}` does not collide with any existing label (a quick `grep "sec:reproducibility"` over the canonical TeX before applying — at draft time, no such label exists).
3. **Verify** the four existing forward references the appendix relies on: `\ref{sec:experiments}` (defined at §5 header), `\ref{sec:exp1}` (defined line 327), `\ref{sec:exp2}` (defined Day 7 by Exp 2 application), `\ref{sec:memory-complexity}` (defined at §6 header). The `sec:exp2` reference will produce an undefined-reference warning if §A is applied before Day 7 Exp 2 application — expected and accepted.
4. **Smoke-compile** at end of Day 10 per G-smoke-compile gate (plan v2 §9): both pdflatex passes clean; only acceptable undefined refs are the upstream `sec:exp2` + `eq:mcts_reward` until Day 7 EOD (zero after).

### Cross-references introduced

- `\label{sec:reproducibility}` — NEW. The Day-10 audit will add cross-references from §3 / §5 / §6 / §7.4 prose pointing here, on a best-effort basis (cross-referencing to a Reproducibility appendix is conventional but not strictly required at ARR review).

### Cross-references RELIED ON (already exist)

- `\ref{sec:experiments}` ✓ (§5 header)
- `\ref{sec:exp1}` ✓ (defined Day 4 at line 327)
- `\ref{sec:exp2}` — **NOT YET DEFINED** until Day 7 v10-exp2 application
- `\ref{sec:memory-complexity}` ✓ (§6 header, defined at `\section{Theoretical Analysis}` block)
- `\ref{eq:goodhart}` ✓ (line 165)
- `\ref{tab:exp1-results}` ✓ (line 349; the new Track F results table)
- `\ref{tab:oei-alpha-sweep}` ✓ (line 555; the relocated alpha-sweep table)

### Compilation impact estimate

- Lines added to canonical TeX: 1 REVISION marker (~3 wrapped lines) + 1 `\appendix` line + 1 `\section{}\label{}` line + 5 subsection headings (5 lines) + 5 subsection paragraphs (~30–35 prose lines) + 1 IMPL line ≈ **net +45 body lines**
- Estimated rendered length: ~1 column-page at ACL two-column density (~70 body lines/page)
- Page-budget impact on review submission: ZERO. `\appendix` content is excluded from the 8-page review budget per ACL ARR rules.

---

## 4. Page-budget commentary

The appendix is unlimited at review time. Phase 3 camera-ready expansion (3–4 pages) will add:

- Full HuggingFace checkpoint commit SHA table for both Llama 3.2-1B and -3B at the moment of each Track's run.
- `requirements-frozen.txt` excerpt or equivalent pip-freeze listing (\texttt{transformers}, \texttt{datasets}, \texttt{huggingface\_hub}, \texttt{numpy}, \texttt{scipy}, \texttt{accelerate}, etc.).
- Per-script CLI argument reference table — for each of \texttt{diagnose\_track\_a\_entropy.py}, \texttt{diagnose\_track\_c\_benchmarks.py}, \texttt{measure\_kv\_mcts\_vram.py}, \texttt{diagnose\_track\_f\_negcontrol.py}, \texttt{diagnose\_track\_g\_cartography.py}, list `--seed`, `--model`, `--out`, and any track-specific switches.
- Cross-platform notes: the experiments were run on Windows 10 + CUDA 12.4; Phase 3 will add validation on Linux + CUDA 12.x with smoke-test reproduction wall-clocks.
- Full random-seed manifest: the `--seed 42` choice is single across Tracks F + G; Phase 3 will report sensitivity to seed selection by running 3–5 alternate seeds and reporting Δ accuracy distributions.

---

## 5. Sources (primary-source-fidelity discipline)

All numeric values, hyperparameter declarations, and hardware/software figures trace to primary artifacts re-read in this session:

| Claim | Primary source | Verified |
|---|---|---|
| RTX 3060 12 GiB, driver 591.44, CUDA 12.4, PyTorch 2.6.0+cu124, Windows 10 Pro 10.0.19045 | Track D report §1 (Hardware reference table) | ✓ |
| Llama 3.2-1B architecture: 16 layers, hidden 2048, 8 KV heads, head_dim 64 | Track D report §2 first footnote + tensor-byte sanity check Appendix B | ✓ |
| Llama 3.2-3B architecture: 28 layers, 8 KV heads, head_dim 128 | Track D report §2 second footnote | ✓ |
| Vocabulary $|V| = 128{,}256$ | Track F report §1 (Reward function row, "Llama-3.2 vocab"); Appendix B confirms ($\log\|V\| = 11.762$ → $|V| = e^{11.762} \approx 128{,}256$) | ✓ |
| MCTSConfig defaults (n_nodes=50, branching=3, max_depth=5, alpha=(0.1,0.5,1.0,2.0,5.0), c=1.414, steer_layer=-1) | `logomesh/kv_mcts.py` lines 610-631 (MCTSConfig dataclass) | ✓ |
| Path-sampling variant: depth-3, 27 paths, alpha={0.1, 0.5, 1.0} | Track F report §1 (Run configuration table) + §10 (deviation justification) | ✓ |
| Track A: 85 factual-recall items, |r| ≈ 0.60/0.61, ~13 min wall | Track A report §4 (1B hard set: |r| = 0.602), §5b (3B ultra set: |r| = 0.614), header (wall-clock ~13 min) | ✓ |
| Track A: entropy formula H = -Σ p_i log p_i over softmax(logits[-1,:]) | Track A report §3 + Appendix A | ✓ |
| Track A: generate hyperparameters (do_sample=False, max_new_tokens=15, output_scores=True) | Track A report §4 (Protocol paragraph) | ✓ |
| Track C: ARC-Easy 200-item first deterministic subset, indices 0..199 | Track C report §3 (ARC-Easy subset) | ✓ |
| Track C: chat template + system + user prompt structure | Track C report §3 ARC-Easy block | ✓ |
| Track C: McNemar power formula at α=0.05, power=0.8, target δ=10pp, discordance ≤25% | Track C report §5 (G2.3 verdict + power calculation table) | ✓ |
| Track C: ~19 min wall both scales | Track C report header (Wall-clock line) | ✓ |
| Track D: VRAM measurement primitives (torch.cuda.memory_allocated + max_memory_allocated + nvidia-smi cross-check) | Track D report §1 + §2 | ✓ |
| Track D: 3 cells × 3 repeats × 2 scales at 4000-token prefill | Track D report §2 tables | ✓ |
| Track D: ~5 min wall both scales | Track D report header (Wall-clock line) | ✓ |
| Track F: 200 paired ARC-Easy items × 2 scales × 3 conditions | Track F report §1 (Run configuration) | ✓ |
| Track F: --seed 42 for steering direction; per-item RNG via md5(item_idx:R:seed) | Track F report §10 (Notes on anomalies) + Appendix A (Determinism notes) | ✓ |
| Track F: 5-token completion, do_sample=False, max_new_tokens=5, output_scores=True | Track F report §1 (Greedy generate row) | ✓ |
| Track F: 18.3 min 1B wall + 33.7 min 3B wall | Track F report header (Wall-clock line) | ✓ |
| FP32Accumulator + _extract_kv_tensors + _kv_eval_cache imports read-only on logomesh/* | Track F report §10 (Protocol deviation) + §11 (Bibliography of files) | ✓ |
| Allocator block-rounding 1-2% variation explanation | Track D report §2 footnote 3 + Appendix B | ✓ |
| anonymous.4open.science as supplementary host | Plan v2 §6 J5 (decided 2026-05-11) | ✓ |
| The three deferred metadata items (SHAs / pins / CLI args) | Acknowledged in pre-§A check-in this session; consistent with Josh-approved scope | ✓ |

No number, hyperparameter, or hardware figure in this draft was sourced from a running-log summary, a compaction-memory recall, or a citation-of-a-citation chain. The single architectural fact previously contradicted by CLAUDE.md (1B layer count: 22 vs measured 16) is anchored to Track D's measurement; the CLAUDE.md error was fixed in the same session (22 → 16 at CLAUDE.md:221).

---

*End of v10 Reproducibility Appendix v1 draft (ARR review version, target 1 page). Day-10 application target per plan v2 §2 (Session B applies as `\appendix \section{Reproducibility}` insert immediately before `\end{document}`). Phase 3 camera-ready expansion to 3-4 pages fills the three metadata gaps (HF SHAs, library pins, per-script CLI args). Ready for Josh review.*
