# Croissant Build Plan

**Date:** 2026-05-17
**Phase:** 1 (Planning; no build actions taken)
**Session:** 3 (Croissant + repo cleanup)
**Discovery report:** [docs/logs/2026-05-17_croissant-discovery.md](docs/logs/2026-05-17_croissant-discovery.md)

**Status:** Josh greenlit all 7 recommendations from the discovery report (Q-J1 through Q-J7). This plan operationalizes those into a Phase 2 build spec.

---

## (1.1) Package scope (Q-J2 confirmed)

The package's `distribution[]` and `recordSet[]` declarations are derived from primary-source JSONs at [scripts/_track_*.json](scripts/). Three tiers:

### Tier A — Core flattened record sets (headline experiments)

| RecordSet @id | Source | Rows | Per-row data |
|---|---|---|---|
| `track_f_negcontrol` | [scripts/_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json](scripts/_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json) + [_3B](scripts/_track_f_results_meta-llama_Llama-3.2-3B-Instruct.json) | 1200 | 200 items × 3 conditions (G/R/E) × 2 scales |
| `track_g_cartography_mcts` | [scripts/_track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json](scripts/_track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json) + [_3B](scripts/_track_g_mcts_results_meta-llama_Llama-3.2-3B-Instruct.json) | 770 | 385 per scale (C1=100 factual_recall, C2=100 arc_easy_mcq, C3=85 truthfulqa_mcq, C4=100 hellaswag_continuation) × 2 scales [CORRECTED 2026-05-17 — original plan listed C3=hellaswag, C4=truthfulqa; swapped per Track G report 47-50] |

Each gets a flattened CSV under `docs/dataset/data/` + Croissant `recordSet` declaration with full field-level metadata pointing at the CSV columns. The original JSONs go in `docs/dataset/data/raw/` and are also declared as `FileObject`s.

### Tier B — Supplementary FileObjects (no flatten; reviewers read JSONs directly)

Declared in `distribution[]` with sha256 hashes; NOT flattened into CSV; NOT referenced by any recordSet field-source mapping.

| @id | Source path | Description in metadata |
|---|---|---|
| `track_g_cartography_initial_sweep_1b` | `scripts/_track_g_results_meta-llama_Llama-3.2-1B-Instruct.json` | Historical parameter-sweep variant (deviated from MCTS protocol per Cartography 2026-05-11 re-run; preserved for audit-trail) |
| `track_g_cartography_initial_sweep_3b` | `scripts/_track_g_results_meta-llama_Llama-3.2-3B-Instruct.json` | Same, 3B |
| `track_g_c1_factual_recall_definition` | `scripts/_track_g_c1_factual_recall.json` | Dataset definition for class C1 (100 hand-constructed factual-recall items) |
| `track_a_correlation_1b` | `scripts/_track_a_results_meta-llama_Llama-3.2-1B-Instruct.json` | Static entropy↔correctness correlation (Pearson/Spearman) on 85-item factual recall |
| `track_a_correlation_3b` | `scripts/_track_a_results_meta-llama_Llama-3.2-3B-Instruct.json` | Same, 3B |
| `track_c_calibration_1b` | `scripts/_track_c_results_meta-llama_Llama-3.2-1B-Instruct.json` | Benchmark accuracy + entropy stats across 6 benchmarks (ARC-Easy, SciQ, HellaSwag, AQuA-RAT, ProofWriter, LogiQA) |
| `track_c_calibration_3b` | `scripts/_track_c_results_meta-llama_Llama-3.2-3B-Instruct.json` | Same, 3B |
| `track_d_vram_1b` | `scripts/_track_d_vram.json` | VRAM measurement across 3 (d,b,n) configurations |
| `track_d_vram_3b` | `scripts/_track_d_vram_3b.json` | Same, 3B |

### Tier C — Reproducibility metadata

| @id | Source path | Description |
|---|---|---|
| `uv_lock` | `uv.lock` (project root, 269 KB) | Pinned dependency manifest |

### Out of scope

Explicitly NOT included (Josh confirmed Q-J2 scope):
- [mcts_output_1776800541.json](mcts_output_1776800541.json) at repo root — Phase-2 debugging output; not paper-referenced.
- [drift_results.csv](drift_results.csv) at repo root — Phase-2 debug artifact; Phase 4 cleanup verifies what this actually is.
- Old [docs/dataset/croissant_schema_stub.json](docs/dataset/croissant_schema_stub.json) — replaced by the new `croissant.json`.
- Old [docs/dataset/data/interventions.csv](docs/dataset/data/interventions.csv) + [docs/dataset/data/source_run.json](docs/dataset/data/source_run.json) — toy TinyLlama placeholders.
- Old [docs/dataset/data/sources/](docs/dataset/data/sources/) — empty directory.

**Phase 2 deletion list** (Josh sign-off via plan approval):
- `docs/dataset/croissant_schema_stub.json`
- `docs/dataset/data/interventions.csv`
- `docs/dataset/data/source_run.json`
- `docs/dataset/data/sources/` (empty dir)

(`logomesh/croissant_export.py` and `tests/test_croissant_export.py` are **kept** — they're working Phase-2 utilities; the Phase 2 build does not need to modify them. Whether to delete them as α+ residue is a Phase 4 cleanup decision, NOT a Phase 2 Croissant-build decision.)

---

## (1.2) Schema structure plan

### Top-level dataset metadata

Croissant 1.1 required + recommended properties (per spec line 347+):

```json
{
  "@context": <CROISSANT_CONTEXT_1.1_PLUS_RAI>,
  "@type": "sc:Dataset",
  "name": "KV-Cache Inception: Reversible KV-Cache MCTS Runtime Evidence",
  "description": "<2-3 sentences, Option D+ era; see §1.2.1 below>",
  "conformsTo": ["http://mlcommons.org/croissant/1.1", "http://mlcommons.org/croissant/RAI/1.0"],
  "license": "https://opensource.org/licenses/MIT",
  "url": "https://github.com/LogoMesh/KV-Cache-Inception",
  "creator": {"@type": "sc:Organization", "name": "LogoMesh Research Team"},
  "publisher": {"@type": "sc:Organization", "name": "LogoMesh Research Team"},
  "datePublished": "2026-05-17",
  "dateCreated": "2026-05-17",
  "dateModified": "2026-05-17",
  "version": "1.0.0",
  "sdVersion": "1.1",
  "inLanguage": "en",
  "keywords": [
    "dimensional escape",
    "reversible kv-cache mcts",
    "frozen language model interpretability",
    "latent geometry probing",
    "goodhart's law",
    "negative result"
  ],
  "citeAs": "<see §1.2.4 below>",
  "distribution": [...],
  "recordSet": [...],
  "rai:dataUseCases": "<see §1.2.5 below>",
  "rai:dataLimitations": "<see §1.2.5 below>",
  "rai:dataBiases": "<see §1.2.5 below>",
  "rai:personalSensitiveInformation": "<see §1.2.5 below>",
  "rai:dataReleaseMaintenancePlan": "<see §1.2.5 below>"
}
```

#### §1.2.1 Description (verbatim text to bake into the package)

> "Runtime evidence and per-item records from Reversible KV-Cache Monte Carlo Tree Search (MCTS) on frozen Llama 3.2-{1B,3B}-Instruct, used to characterize the Dimensional Escape pathology (an instance of Goodhart's Law in high-dimensional latent-space search) under an entropy-normalized surrogate reward. Includes two headline experiments: Track F (entropy-MCTS negative control vs random-MCTS vs greedy baseline on 200 ARC-Easy items; n=200 paired McNemar, p=1.0 at both scales) and Track G (Latent Cartography across four prompt classes — hand-constructed factual recall, ARC-Easy MCQ, HellaSwag continuation, TruthfulQA MCQ). Supporting tracks A, C, D provide static entropy↔correctness correlation, benchmark calibration, and VRAM-measurement records that anchor the paper's reproducibility appendix §A. Released as supplementary material for the ARR May 2026 / EMNLP 2026 paper 'Probing Dimensional Escape in Frozen Language Models via Reversible KV-Cache MCTS.'"

#### §1.2.2 FileObject declarations (one per file)

Schema per FileObject:
```json
{
  "@type": "cr:FileObject",
  "@id": "<stable_id>",
  "name": "<filename>",
  "contentUrl": "<relative_path_within_package>",
  "encodingFormat": "<text/csv | application/json | text/plain>",
  "sha256": "<computed_by_build_script>",
  "description": "<1-sentence what this file contains>"
}
```

13 FileObjects total (4 core + 9 Tier-B + 1 uv.lock + croissant.json itself is not a FileObject):

| @id | encodingFormat | contentUrl |
|---|---|---|
| `track-f-csv` | text/csv | `data/track_f_negcontrol.csv` |
| `track-g-csv` | text/csv | `data/track_g_cartography_mcts.csv` |
| `track-f-raw-1b` | application/json | `data/raw/_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json` |
| `track-f-raw-3b` | application/json | `data/raw/_track_f_results_meta-llama_Llama-3.2-3B-Instruct.json` |
| `track-g-mcts-raw-1b` | application/json | `data/raw/_track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json` |
| `track-g-mcts-raw-3b` | application/json | `data/raw/_track_g_mcts_results_meta-llama_Llama-3.2-3B-Instruct.json` |
| `track-g-init-raw-1b` | application/json | `data/raw/_track_g_results_meta-llama_Llama-3.2-1B-Instruct.json` |
| `track-g-init-raw-3b` | application/json | `data/raw/_track_g_results_meta-llama_Llama-3.2-3B-Instruct.json` |
| `track-g-c1-definition` | application/json | `data/raw/_track_g_c1_factual_recall.json` |
| `track-a-raw-1b` | application/json | `data/raw/_track_a_results_meta-llama_Llama-3.2-1B-Instruct.json` |
| `track-a-raw-3b` | application/json | `data/raw/_track_a_results_meta-llama_Llama-3.2-3B-Instruct.json` |
| `track-c-raw-1b` | application/json | `data/raw/_track_c_results_meta-llama_Llama-3.2-1B-Instruct.json` |
| `track-c-raw-3b` | application/json | `data/raw/_track_c_results_meta-llama_Llama-3.2-3B-Instruct.json` |
| `track-d-raw-1b` | application/json | `data/raw/_track_d_vram.json` |
| `track-d-raw-3b` | application/json | `data/raw/_track_d_vram_3b.json` |
| `uv-lock` | text/plain | `data/uv.lock` |

That's 16 FileObjects total. All file copies live under `docs/dataset/data/` (CSVs flat; raw JSONs in `data/raw/`; lock file in `data/uv.lock`).

#### §1.2.3 RecordSet field schemas

**RecordSet `track_f_negcontrol`** — extracted from `track-f-csv` (the flattened CSV):

| Field name | dataType | Description | Source field in raw JSON |
|---|---|---|---|
| `scale` | `sc:Text` | Model scale, "1B" or "3B" | Filename-derived |
| `item_idx` | `sc:Integer` | Item index, 0-199 | `items[i].idx` |
| `source_id` | `sc:Text` | ARC-Easy item identifier (e.g., "Mercury_417466") | `items[i].source_id` |
| `gold_letter` | `sc:Text` | Gold answer letter (A/B/C/D) | `items[i].gold_letter` |
| `n_options` | `sc:Integer` | Number of multiple-choice options | `items[i].n_options` |
| `condition` | `sc:Text` | Decoding condition: "G" (greedy), "R" (random-MCTS), "E" (entropy-MCTS) | Synthesized from items[i].{G,R,E} sub-blocks |
| `pred_letter` | `sc:Text` | Predicted letter | `items[i].<cond>.pred_letter` |
| `correct` | `sc:Integer` | 1 if pred_letter == gold_letter else 0 | `items[i].<cond>.correct` |
| `answer_text` | `sc:Text` | First 5 tokens of model output | `items[i].<cond>.answer_text` |
| `first_token_entropy_nats` | `sc:Float` | Entropy of first-token distribution (nats) | `items[i].<cond>.first_token_entropy_nats` |
| `per_token_entropy_mean_nats` | `sc:Float` | Mean per-token entropy across generated tokens (nats) | `items[i].<cond>.per_token_entropy_mean_nats` |
| `perplexity_5tok` | `sc:Float` | Perplexity over the 5-token completion (NaN coerced to empty string for greedy/random when defined as such; will surface NaN literal where present) | `items[i].<cond>.perplexity_5tok` |
| `search_status` | `sc:Text` | "ok" (MCTS ran), "n/a" (greedy, no search) | `items[i].<cond>.search_status` |
| `mean_search_reward` | `sc:Float` | Mean reward over all 27 enumerated paths (NaN if search_status="n/a") | `items[i].<cond>.mean_search_reward` |
| `best_search_reward` | `sc:Float` | Reward of the best-rewarded path (missing for G) | `items[i].<cond>.best_search_reward` |
| `mean_terminal_entropy_nats` | `sc:Float` | Mean terminal entropy across paths (missing for G) | `items[i].<cond>.mean_terminal_entropy_nats` |
| `best_path_json` | `sc:Text` | JSON-encoded best path (3-element alpha list; "null" string for G) | `items[i].<cond>.best_path` |
| `n_paths_evaluated` | `sc:Integer` | Path count (always 27 for R/E; missing for G) | `items[i].<cond>.n_paths_evaluated` |
| `n_apply_failures` | `sc:Integer` | Apply-failures count (always 0 for the 200 items at both scales empirically; missing for G) | `items[i].<cond>.n_apply_failures` |

**Key (Croissant `recordSet.key`):** Composite of `track_f_negcontrol/scale`, `track_f_negcontrol/item_idx`, `track_f_negcontrol/condition`. Per Croissant 1.1 spec line 880+, composite keys are an array of `{@id: ...}` references.

**NaN handling:** Per Croissant 1.1 spec §"Embedding data" (line 1321+) and JSON-LD spec, NaN is not a valid JSON number. The build script will coerce `NaN` → literal string `"NaN"` in the CSV (the column's `dataType` is `sc:Float` but the field carries a sentinel). Alternative: emit empty string for NaN cells; flag the field's description that empty = "not applicable". **Recommend** empty-string-for-N/A approach to keep `sc:Float` semantically valid.

**RecordSet `track_g_cartography_mcts`** — extracted from `track-g-csv`:

| Field name | dataType | Description | Source field in raw JSON |
|---|---|---|---|
| `scale` | `sc:Text` | Model scale, "1B" or "3B" | Filename-derived |
| `class_code` | `sc:Text` | Class code: "C1", "C2", "C3", "C4" | `records[i].class` |
| `class_name` | `sc:Text` | Human name: "factual_recall", "arc_easy_mcq", "hellaswag_continuation", "truthfulqa_mcq" | Synthesized via fixed mapping |
| `item_idx` | `sc:Integer` | Item index within class (0-99 except C3 at 3B = 0-84) | `records[i].item_idx` |
| `subclass` | `sc:Text` | Per-class subclass (e.g., "capital", "date", "arithmetic", "common_knowledge" for C1) | `records[i].subclass` |
| `source_id` | `sc:Text` | Stable record identifier (e.g., "C1_0_capital") | `records[i].source_id` |
| `eval_mode` | `sc:Text` | "first_token_str" | `records[i].eval_mode` |
| `prompt_preview` | `sc:Text` | First ~80 chars of the prompt template applied | `records[i].prompt_preview` |
| `gold` | `sc:Text` | Gold answer string | `records[i].gold` |
| `search_status` | `sc:Text` | "ok" or error code | `records[i].search_status` |
| `best_path_json` | `sc:Text` | JSON-encoded 3-element alpha list | `records[i].best_path` |
| `n_paths_evaluated` | `sc:Integer` | Always 27 | `records[i].n_paths_evaluated` |
| `n_apply_failures` | `sc:Integer` | Apply-failures count | `records[i].n_apply_failures` |
| `mean_search_reward` | `sc:Float` | Mean reward across all 27 paths | `records[i].mean_search_reward` |
| `best_search_reward` | `sc:Float` | Best-path reward | `records[i].best_search_reward` |
| `mean_terminal_entropy_nats` | `sc:Float` | Mean terminal entropy across paths (nats) | `records[i].mean_terminal_entropy_nats` |
| `best_terminal_entropy_nats` | `sc:Float` | Best-path terminal entropy (nats) | `records[i].best_terminal_entropy_nats` |
| `argmax_token_str` | `sc:Text` | First-token argmax decoded to string under best-path steering | `records[i].argmax_token_str` |
| `argmax_token_id` | `sc:Integer` | First-token argmax token ID | `records[i].argmax_token_id` |
| `next_token_entropy_nats` | `sc:Float` | Best-path terminal entropy at the generation step (nats) | `records[i].next_token_entropy_nats` |
| `top5_json` | `sc:Text` | JSON-encoded list of 5 [token_str, prob] pairs at best-path | `records[i].top5` |
| `gold_first_token_rank` | `sc:Integer` | Rank of gold first token among all vocab (0 = top, -1 if outside top-K returned) | `records[i].gold_first_token_rank` |
| `gold_first_token_prob` | `sc:Float` | Probability of gold first token | `records[i].gold_first_token_prob` |
| `wallclock_sec` | `sc:Float` | Wall-clock for this item's MCTS expansion (seconds) | `records[i].wallclock_sec` |

**Key:** Composite of `track_g_cartography_mcts/scale`, `track_g_cartography_mcts/class_code`, `track_g_cartography_mcts/item_idx`.

#### §1.2.4 `citeAs` field

The paper is currently anonymous-for-review with `\author{[Authors]}` placeholder at [04.18.2026-NeurIPS-Research-Proposal-2.tex](docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex). Three options for `citeAs`:

- **(α)** Placeholder bibtex matching `\author{[Authors]}`:
  ```
  @inproceedings{kvcache_inception_2026,
    title = {Probing Dimensional Escape in Frozen Language Models via Reversible KV-Cache MCTS},
    author = {[Authors]},
    booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (via ACL Rolling Review May 2026)},
    year = {2026}
  }
  ```
- **(β)** Omit `citeAs` (it's optional per Croissant 1.1 spec).
- **(γ)** Fill in roster from `memory/MEMORY.md` line 119 (Josh, Max Toscano, Alaa Elobaid, Tianyu Shi) — but then Phase 4 anonymization must rewrite.

**Recommend (α).** Matches the paper's own author placeholder; anonymization-pass-friendly (the placeholder string `[Authors]` requires no edit for review); de-anonymization at camera-ready is a single replace.

#### §1.2.5 RAI 1.0 fields (verbatim text to bake)

Refreshed from α+ vintage stub strings to Option D+ era.

| RAI field | Verbatim text |
|---|---|
| `rai:dataUseCases` | "Interpretability research on inference-time KV-cache search behavior in frozen language models. The data documents a negative-result characterization of Dimensional Escape (an instance of Goodhart's Law in high-dimensional latent-space search) under an entropy-normalized surrogate reward across two model scales. Use cases include: reproducing the paper's Track F (entropy-MCTS negative control) and Track G (Latent Cartography) headline findings; benchmarking subsequent reward designs against the dataset's pre-registered failure mode; extending the Cartography sweep to additional prompt classes or model families." |
| `rai:dataLimitations` | "Records are derived from two Llama 3.2 scales (1B and 3B Instruct variants) on a small sample of hand-curated and public-benchmark items (Track F: 200 ARC-Easy items per scale; Track G: ~100 items per prompt class per scale, with HellaSwag continuation dropping items whose gold labels are not unambiguously tokenizable). No human-annotated ground-truth labels beyond the public benchmarks' provided gold answers. Top-5 token records and best-path entries are model-internal artifacts; cross-family comparability requires re-calibration of the steering protocol. Wall-clock seconds are captured per-item where available but cover the MCTS expansion loop only, not model-loading or evaluation overhead." |
| `rai:dataBiases` | "Findings are conditional on the specific steering direction (a single random unit vector in d_model space, seeded by --seed 42, shared across items and across the random-MCTS and entropy-MCTS arms) and on the Llama 3.2 instruction-tuned models. Behavioral patterns (first-letter prior amplification at 1B, greedy-collapse at 3B) are properties of this model family under this steering protocol and should not be generalized to other model families, other steering directions, or non-instruction-tuned base models without independent verification." |
| `rai:personalSensitiveInformation` | "No personal data. Prompts are drawn from public benchmarks (ARC-Easy, HellaSwag, TruthfulQA, SciQ, AQuA-RAT, ProofWriter, LogiQA) or hand-constructed factual-recall items (world capitals, historical dates, single-digit arithmetic, common-knowledge facts). Model-generated answers are 5-token continuations against multiple-choice options or first-token completions; no PII generation paths." |
| `rai:dataReleaseMaintenancePlan` | "Versioned snapshot for ACL Rolling Review May 2026 submission (EMNLP 2026 commitment). Provenance metadata baked into each record: random seed (42), HuggingFace model identifiers (meta-llama/Llama-3.2-1B-Instruct, meta-llama/Llama-3.2-3B-Instruct), PyTorch version (2.6.0+cu124), CUDA version (12.4). HuggingFace checkpoint commit SHAs are released as a separate companion field per §A.2 of the paper (filled at submission via huggingface-cli). Phase B (post-acceptance) extensions will add additional models and prompt classes; the package version field will increment accordingly." |

#### §1.2.6 Provenance metadata captured per RecordSet

Each FileObject gets sha256. Beyond that, the build script captures global provenance into the dataset description and the RAI maintenance-plan field; per-record provenance lives in the source JSON's top-level fields and the build script preserves them by copying the raw JSON into `data/raw/` (so reviewers can `jq '.config.seed' data/raw/_track_f_results_*.json` to verify seed=42).

---

## (1.3) Build approach (Q-J1 confirmed: Option B)

### New file: `scripts/build_croissant_package.py`

**Inputs:**
- Hard-coded list of 13 source JSON paths under `scripts/_track_*.json` + 1 hand-constructed C1 definition (`scripts/_track_g_c1_factual_recall.json`) + 1 lock file (`uv.lock`).
- Hard-coded class-code → class-name mapping for Track G (C1=factual_recall, C2=arc_easy_mcq, C3=hellaswag_continuation, C4=truthfulqa_mcq).
- Optional CLI args: `--output-dir docs/dataset` (default), `--strict` (run mlcroissant validation; fail loudly).

**Output (`docs/dataset/`):**
```
docs/dataset/
├── croissant.json                                 # new; replaces croissant_schema_stub.json
└── data/
    ├── track_f_negcontrol.csv                     # new; 1200 rows
    ├── track_g_cartography_mcts.csv               # new; 770 rows
    ├── uv.lock                                    # copy from project root
    └── raw/
        ├── _track_f_results_meta-llama_Llama-3.2-1B-Instruct.json
        ├── _track_f_results_meta-llama_Llama-3.2-3B-Instruct.json
        ├── _track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json
        ├── _track_g_mcts_results_meta-llama_Llama-3.2-3B-Instruct.json
        ├── _track_g_results_meta-llama_Llama-3.2-1B-Instruct.json   # historical sweep variant
        ├── _track_g_results_meta-llama_Llama-3.2-3B-Instruct.json
        ├── _track_g_c1_factual_recall.json
        ├── _track_a_results_meta-llama_Llama-3.2-1B-Instruct.json
        ├── _track_a_results_meta-llama_Llama-3.2-3B-Instruct.json
        ├── _track_c_results_meta-llama_Llama-3.2-1B-Instruct.json
        ├── _track_c_results_meta-llama_Llama-3.2-3B-Instruct.json
        ├── _track_d_vram.json
        └── _track_d_vram_3b.json
```

**Pre-build deletions** (Josh sign-off via this plan):
- Remove `docs/dataset/croissant_schema_stub.json`
- Remove `docs/dataset/data/interventions.csv`
- Remove `docs/dataset/data/source_run.json`
- Remove `docs/dataset/data/sources/`

**Script structure (logical steps):**

1. **Load** all 13 source JSONs from `scripts/` (full content into memory).
2. **Verify** structural invariants:
   - Track F: each file has `n_items == 200`, `len(items) == 200`, each item has `idx`, `source_id`, `gold_letter`, G/R/E condition blocks.
   - Track G MCTS: each file has `len(records) ≥ 385`, `classes == ["C1","C2","C3","C4"]`, each record has expected keys.
   - If any invariant fails, raise `ValueError` with the violated invariant + path. (Per Hard Rule 1: fail loudly, never silently default.)
3. **Flatten Track F** → list of dicts → write to `docs/dataset/data/track_f_negcontrol.csv`. NaN → empty string.
4. **Flatten Track G MCTS-primary** → list of dicts → write to `docs/dataset/data/track_g_cartography_mcts.csv`.
5. **Copy raw JSONs + uv.lock** to `docs/dataset/data/raw/` + `docs/dataset/data/uv.lock`. Use `shutil.copy2` (preserves timestamps).
6. **Compute sha256** for each output file (the two CSVs + 13 raw JSONs + uv.lock).
7. **Build the Croissant JSON-LD** (the metadata) from the §1.2 spec. Use the existing `logomesh.croissant_export.CROISSANT_CONTEXT` dict for the @context block (line 35-72 of [croissant_export.py](logomesh/croissant_export.py)) — that part is already correct and reusable.
8. **Validate** the metadata:
   - Run `logomesh.croissant_export.validate_metadata_shape()` as cheap shape-check.
   - If `--strict`, attempt `subprocess.run(["mlcroissant", "validate", "--jsonld", out_path], check=False)`; raise `RuntimeError` on non-zero exit.
9. **Write** `docs/dataset/croissant.json` (pretty-printed, 2-space indent, UTF-8, ensure_ascii=True).
10. **Print summary** to stdout: record counts, sha256 sums, total output size, validation status.

**Idempotency:** If `docs/dataset/croissant.json` already exists, overwrite. If `docs/dataset/data/raw/<file>.json` already exists with matching sha256, skip copy. Logs deltas.

**Pre-build install step** (Phase 2 first action):
- `uv pip install mlcroissant` (per Q-J7 recommendation)
- Verify with `mlcroissant --version`

---

## (1.4) Validation plan

After the build runs:

### Layer 1 — Built-in shape validator
- `from logomesh.croissant_export import validate_metadata_shape`
- Returns empty list = pass; non-empty list = fail with specific reasons.
- This is fast (<100ms) and catches most structural issues (missing required keys, malformed source mappings, invalid sha256 format).

### Layer 2 — mlcroissant strict validation
- After install, run `mlcroissant validate --jsonld docs/dataset/croissant.json` (or pass `--strict` to build script).
- mlcroissant validates against the Croissant 1.1 JSON-LD schema directly.
- Failures here mean spec drift; must be fixed before Phase 3 commit.

### Layer 3 — Manual paper-claim cross-check

Build script writes a manifest file `docs/dataset/MANIFEST.txt` listing each output file + sha256. Manual check against this list:

| Paper claim | Verification |
|---|---|
| Abstract S8 line 56 "Artifacts ship with a Croissant 1.1 metadata package." | `docs/dataset/croissant.json` exists; `conformsTo` includes `http://mlcommons.org/croissant/1.1` |
| §7.5 line 363 "Croissant 1.1 dataset package of runtime evidence artifacts is included as supplementary material" | Same. Plus `recordSet[]` has track_f_negcontrol + track_g_cartography_mcts. |
| §7.6 line 368 "documents per-node runtime evidence artifacts from Experiments 1--2 with provenance and usage-constraint fields" | Per-item not per-node (paper-claim drift acknowledged Q-J3); Experiments 1 + 2 covered; provenance via `creator`/`datePublished`/`git_sha`-equivalent in each raw JSON + RAI fields. |
| §A.5 line 510 supplementary URL placeholder XXXX | Slug pending — Phase 4 cleanup assigns. |

### Layer 4 — Human-readable smoke test
- Open `docs/dataset/croissant.json` in a JSON viewer (e.g., VS Code with built-in formatter).
- Verify it's pretty-printed, ~500-1500 lines, no obviously broken UTF-8 or escape issues.
- Spot-check 3 records in `track_f_negcontrol.csv` against the corresponding `scripts/_track_f_*.json` records.
- Spot-check 3 records in `track_g_cartography_mcts.csv` against the corresponding `scripts/_track_g_mcts_*.json` records.

### Layer 5 — Test integration (optional)
- The existing [tests/test_croissant_export.py](tests/test_croissant_export.py) tests Phase-2-era functionality and should continue to pass after our build (we don't modify `croissant_export.py`).
- Run `uv run pytest tests/ -v` to confirm (per CLAUDE.md hard rule 2: pytest must be 100% green before any commit).
- If `tests/test_croissant_export.py` fails, that's a separate issue — we did not modify the module.

---

## (1.5) Anonymization plan (Phase 4 only; not Phase 2)

Per Q-J6 confirmed: Phase 2 builds the live-repo `docs/dataset/` tree with real names/URLs. Phase 4 cleanup creates an anonymized copy for upload to anonymous.4open.science.

### Live-repo (Phase 2 output)
- `name`: "LogoMesh Research Team"
- `creator.name`, `publisher.name`: "LogoMesh Research Team"
- `url`: "https://github.com/LogoMesh/KV-Cache-Inception"
- `citeAs`: `@inproceedings{...author = {[Authors]}...}` (placeholder per Q-J3 recommendation α)

### Anonymized copy (Phase 4 output, placed in supplementary upload tree)

Phase 4 will apply these sed-style swaps to a working copy of `docs/dataset/`:
| Field | Live-repo value | Anonymized value |
|---|---|---|
| `creator.name` | "LogoMesh Research Team" | "Anonymous Authors" |
| `publisher.name` | "LogoMesh Research Team" | "Anonymous Authors" |
| `url` | "https://github.com/LogoMesh/KV-Cache-Inception" | "https://anonymous.4open.science/r/kv-mcts-dimensional-escape-XXXX" (slug from Phase 4 upload) |
| `citeAs` | `@inproceedings{...author = {[Authors]}...}` | Same — `[Authors]` placeholder is already anonymity-compliant |

Phase 4 ALSO does these companion edits in the supplementary tree (not just croissant.json):
- Any README in `docs/dataset/` → strip author names + LogoMesh references.
- `uv.lock` file — review for any author-identifying URLs (uv lock files generally are clean; verify).
- Raw JSONs — they contain `model_id`, `device`, `gpu_name`, `torch_version`, `cuda_version`, `seed` only. No author info baked in (already audited; primary-source verified Phase 0.4).

The §A.5 URL patch into the paper TeX is also a Phase 4 task (the one paper-TeX edit Session 3 is allowed to make).

---

## Phase 2 execution checklist

Once Josh approves this plan:

1. ☐ Install mlcroissant: `uv pip install mlcroissant`. Verify `mlcroissant --version`.
2. ☐ Pre-build cleanup: delete `docs/dataset/croissant_schema_stub.json`, `docs/dataset/data/interventions.csv`, `docs/dataset/data/source_run.json`, `docs/dataset/data/sources/`.
3. ☐ Write `scripts/build_croissant_package.py` per §1.3 spec.
4. ☐ Run `python scripts/build_croissant_package.py --strict` from repo root.
5. ☐ Confirm 5 expected outputs land: `docs/dataset/croissant.json`, 2 CSVs, 13 raw JSON copies, `uv.lock` copy.
6. ☐ Run Layer 1 + Layer 2 validation. Capture exit codes + outputs.
7. ☐ Layer 3 manual cross-check + Layer 4 spot-checks.
8. ☐ Run `uv run pytest tests/ -v` (Layer 5).
9. ☐ Write execution log to `docs/logs/2026-05-17_croissant-execution-log.md`.

---

## Risks + escalation

### R1: `mlcroissant` install fails
- Symptom: `uv pip install mlcroissant` errors (network, dependency conflict, Python version).
- Fallback: skip Layer 2; rely on Layers 1, 3, 4.
- Impact: weaker validation; minor risk of spec drift that the shape-checker misses.
- Mitigation: run Layer 2 as a manual step later (Josh's machine, or a fresh venv).

### R2: NaN handling in Track F greedy condition
- Symptom: `items[i].G.mean_search_reward` is JSON literal `NaN` (not a number). `json.load()` parses it as Python `math.nan`.
- Approach: CSV writer coerces `nan` → empty string. The `sc:Float` dataType then has empty cells for "not applicable". Field descriptions explicitly note "empty = condition does not search (greedy decoding, no MCTS)".
- Validation impact: mlcroissant strict validation may complain about non-numeric values in `sc:Float` columns. If so, fall back to coercing NaN → literal string `"NaN"` (column dataType becomes `sc:Text` or we use `sc:Float` with the documented NaN sentinel — Croissant 1.1 spec doesn't explicitly forbid).

### R3: Track F best_path is `null` for G condition
- Symptom: `items[i].G.best_path == null` (no path searched).
- Approach: CSV writer emits empty string for null; `best_path_json` column dataType is `sc:Text`. Description notes "empty for greedy condition (no search performed)".

### R4: anonymous.4open.science file-size limit
- Symptom: total `docs/dataset/` size ~8 MB (CSVs + 13 raw JSONs + uv.lock + croissant.json). Anonymous.4open.science limits unknown until Phase 4 cleanup discovery.
- Mitigation: if over limit, Phase 4 drops `_track_g_results_*.json` (the historical sweep variant, 2.4 MB × 2 = 4.8 MB savings → drops total to ~3 MB). The historical files are Tier-B supporting, not headline.

### R5: Building during Session 2's active work
- Symptom: Session 2 is concurrently revising paper content (per task prompt's session-coordination preamble).
- Mitigation: Session 3 (this work) operates only in `docs/dataset/`, `scripts/build_croissant_package.py`, `docs/logs/2026-05-17_croissant-*.md`. Zero overlap with paper TeX files Session 2 touches.

### R6: License copyright string surprise
- Observation: [LICENSE](LICENSE) header reads "Copyright 2025 AgentBeats" — not "LogoMesh" as memory might imply.
- Approach: Croissant `license` field uses SPDX URL `https://opensource.org/licenses/MIT` (the license is structurally MIT regardless of copyright holder). Do not fabricate a "LogoMesh" copyright; preserve the AgentBeats string as-is in any reference back to LICENSE. The supplementary upload copies the live LICENSE file verbatim.
- Escalation: flag to Josh — is "AgentBeats" the expected copyright holder, or is this a leftover that should be fixed Phase 4? (Phase 4 cleanup-prompt's stale-content inventory at Q7 covers this kind of repo-hygiene question.)

### R7: pyproject.toml description string is α+ vintage
- Observation: [pyproject.toml line 8](pyproject.toml:8) reads `description = "KV-Cache Inception — Reversible MCTS for Alignment Faking Detection"`. This is α+ vintage.
- Approach: NOT a Croissant-build issue (Croissant package's description is independent and Option-D+-era). However, Phase 4 cleanup should consider updating pyproject.toml's description.
- Escalation: flag to Phase 4 cleanup inventory.

---

## Estimated Phase 2 wall time

- Pre-build setup (install mlcroissant + deletions): 10 min
- Write `scripts/build_croissant_package.py`: 60 min (one shot; ~500-700 lines including validation + invariant checks + provenance prose)
- Build + validate run: 5 min
- Layer 3 + Layer 4 manual checks: 20 min
- Write execution log: 15 min

**Total: ~110 min (about 1.5-2 hours wall).** Within the Phase 2 budget of 1-3 hours.

---

## Phase 1 verdict

Plan complete. All Phase 0 questions resolved. Ready for Phase 2 execution on Josh's go.

**Pending Josh sign-off:**
- (1) Plan approval as-written → fire Phase 2 immediately.
- (2) Plan approval with adjustments (e.g., different scope, different NaN handling, different `citeAs` option) → patch + re-confirm.
- (3) Pause for further discussion (no expected blockers).
