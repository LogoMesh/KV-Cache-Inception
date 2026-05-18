# Croissant Package Discovery Report

**Date:** 2026-05-17
**Phase:** 0 (Discovery; no actions taken)
**Session:** 3 (Croissant + repo cleanup; paper-content edits NOT in scope)
**HEAD verified:** `44d2ff3` (Day-14-PM cleanup batch landed; per `memory/recast_action_plan.md` preamble)

---

## TL;DR

**Bottom line:** The existing Croissant scaffolding cannot be reused as-is. `logomesh/croissant_export.py` is a Phase-2-era module that targets MCTS-node-level data from `scripts/run_kv_mcts.py`; it has no awareness of Track F (negative control) or Track G (Latent Cartography) — the two experiments the paper actually reports. The April-15 `croissant_schema_stub.json` + `data/{interventions.csv, source_run.json}` are toy placeholders (TinyLlama, one node, all-zero git SHA) generated from a single demo `run_kv_mcts.py` run; their schema is Croissant-1.1-valid but bakes in α+ vintage framing ("alignment-faking detection" keywords; "per-node telemetry" record set) and points at a wrong repo URL. The existing pytest suite (`tests/test_croissant_export.py`) hard-codes the deleted Phase-2 Table 1 diagnostic-state taxonomy.

**The supportable Croissant package** for ARR submission is a one-shot rebuild from the actual Track F + Track G JSONs (per-item-per-condition records, not per-MCTS-node). Track F: 200 items × 3 conditions (G/R/E) × 2 scales = 1200 records total. Track G MCTS-primary: 4 prompt classes × ~100 items × 2 scales = 770 records (one class has 85 at 3B). Optional supporting tracks (A correlation, C calibration, D VRAM) have aggregate-stats-shaped JSONs that the package can describe as separate `FileObject`s without flattening into the same `recordSet`.

**Three paper-claim discrepancies surfaced incidentally** (NOT Croissant blockers, but Josh should know — paper-content edits are Session 2's domain):
- §7.6 line 368 says "per-node runtime evidence artifacts" — α+ vintage wording, Track F/G data is per-item not per-node.
- §A.4 Track C line 501 says calibration sweep included "BBH-LogicalDeduction-3" — actual `_track_c_*.json` files have **LogiQA**, not BBH-LogicalDeduction-3.
- §A.4 Track A line 498 says "Wall-clock approximately 13~min" — actual `_track_a_*.json` `wallclock_seconds` field shows **116 sec (1B), 134 sec (3B)** ≈ 2 min each.

**Recommendation:** Phase 1 Option B (one-shot build script under `scripts/build_croissant_package.py`) — disposable, scoped to the actual track JSONs, sidesteps the surgery `croissant_export.py` would need. Hand-author option (C) is feasible but error-prone for 20+ fields per track-record-set.

---

## (0.1) Croissant export module state

**File:** [logomesh/croissant_export.py](logomesh/croissant_export.py) (703 lines)

**Exposed surface:**
- `build_records_from_run_artifact(run_artifact, run_artifact_path, experiment_id)` — flattens a single `run_kv_mcts.py` artifact (`run_metadata` + `all_nodes` + `model_id`) into a list of dicts, one row per MCTS node.
- `write_records_csv(records, output_csv_path)` — writes the records to CSV with a stable 27-column field order.
- `build_metadata_document(...)` — builds Croissant 1.1 + RAI 1.0 JSON-LD with hard-coded dataset name "KV-Cache Inception Runtime Interventions", α+ vintage keywords and RAI fields, URL `https://github.com/Logomesh/kv-cache-inception`.
- `build_batch_metadata_document(source_artifacts, ...)` — same skeleton but for multiple source JSONs.
- `validate_csv_shape()`, `validate_metadata_shape()` — shape checkers (not full Croissant validation).
- `validate_with_mlcroissant(metadata_path)` — runs the `mlcroissant` CLI if installed; returns `(127, "mlcroissant CLI not found in PATH")` otherwise.
- `export_run_artifact_to_croissant(run_artifact_path, output_dir, strict=False, ...)` — main entry point: takes a single run-artifact JSON and writes `interventions.csv`, `source_run.json`, `metadata.json`.

**Schema (Phase-2-era):**
- 27 hard-coded fields per row (`FIELD_SPECS`, [croissant_export.py:84-121](logomesh/croissant_export.py:84-121)): `artifact_id, run_artifact_path, model_id, seed, git_sha, git_dirty, run_started_utc, run_finished_utc, command, node_id, parent_id, depth, alpha, intervention_layer, visit_count, mean_reward, oei_score, sigma_h_mean, rho_r_mean, tds_score, h_neuron_json, repe_honesty_json, t_matrix_json, diagnostic_state, textual_compliance, ground_truth_label, experiment_id, hardware_context, audit_hash`.
- Most fields are MCTS-node-level (`node_id`, `parent_id`, `depth`, `visit_count`, `oei_score`) — these have **no direct analog in Track F or Track G JSONs**. Track F has `idx`, `source_id`, `gold_letter` per item + per-condition (G/R/E) blocks; Track G MCTS has `class`, `item_idx`, `subclass`, `gold` per record + per-record `best_path`, `mean_search_reward`, etc.

**Maintenance state:** Stale. Hard-coded `url`, `keywords` ("alignment-faking detection"), and RAI strings (`rai:dataUseCases`: "Safety diagnostics and stress testing of alignment-faking behavior...") are all α+/Phase-2 vintage. The `t_matrix_json` field references the telemetry-matrix definition that was **moved to Appendix §F in Day-10 batch-5 drop-C2** ([04.18.2026-NeurIPS-Research-Proposal-2.tex:586](docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex:586)); §4.1's table tab:oei-alpha-sweep was relocated, and the C2 telemetry-matrix-as-paper-contribution framing was dropped from §1.1.

**Tests:** [tests/test_croissant_export.py](tests/test_croissant_export.py) (161 lines, 4 tests, all green at time of authoring per memory). The first test (lines 61-108) asserts `child["diagnostic_state"] in {"creative_synthesis", "contextual_fabrication", "alignment_faking", "orthogonal_escape"}` — the **deleted Phase-2 4-state taxonomy** (per `% [REVISION | §4.1-Table1-DiagnosticStates-delete-D+ | 2026-05-11 | Deleted Table 1 ...]` at [04.18.2026-NeurIPS-Research-Proposal-2.tex:616](docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex:616)). Tests pass because the module's classifier (`telemetry_matrix.classify()`) still emits those state strings; the module hasn't been updated for Option D+.

**Verdict:** Reusable only at the JSON-LD context + validator-shape level (`CROISSANT_CONTEXT`, `validate_metadata_shape`). The records/CSV/metadata-builder pipeline is too scoped to Phase-2 MCTS-node data to retrofit cleanly.

---

## (0.2) Existing schema stub state

**File:** [docs/dataset/croissant_schema_stub.json](docs/dataset/croissant_schema_stub.json) (533 lines)

**Schema:** Croissant 1.1 + RAI 1.0 — `conformsTo` declares both. Required top-level keys all present (@context, @type, dct:conformsTo, description, license, name, url, creator, datePublished, distribution, recordSet). Fields all carry proper @id / dataType / source mappings.

**Content provenance:** Generated by `export_run_artifact_to_croissant()` on the toy `data/source_run.json` (TinyLlama, one node). All values are either:
- Placeholder strings (`name`: "KV-Cache Inception Runtime Interventions"; `description` Phase-2-era; `version`: "0.1.0"; `datePublished`: "2026-04-15T00:00:00Z"; `creator.name`: "LogoMesh Research Team"; `url`: `https://github.com/Logomesh/kv-cache-inception` — incorrect; per `memory/MEMORY.md` corrected scope, the actual repo is `github.com/LogoMesh/KV-Cache-Inception`).
- Verbatim copies of `FIELD_SPECS` from `croissant_export.py` lines 84-121.

**RAI block:** Carries α+ vintage framing:
- `rai:dataUseCases`: "Safety diagnostics and stress testing of alignment-faking behavior in latent representations." — paper's Option D+ framing is interpretability probe + dimensional escape characterization, NOT alignment-faking detection.
- `keywords`: `["alignment-faking detection", "kv-cache intervention", "reversible mcts", "latent telemetry"]` — first keyword is α+.

**Verdict:** Discardable. Useful only as a Croissant-1.1-valid skeleton to crib the @context block from when writing the new package.

---

## (0.3) Existing data scaffolding state

**Directory:** [docs/dataset/data/](docs/dataset/data/)

- **`interventions.csv` (757 bytes, 1 data row):** Header has the 27 columns. The single sample row has `model_id=TinyLlama/TinyLlama-1.1B-Chat-v1.0`, `seed=42`, `git_sha=0000...0000`, `command=scripts/run_kv_mcts.py --nodes 1`, `node_id=node-0001`, `depth=1`, `diagnostic_state=creative_synthesis`. **All placeholder/example data**, not real experimental output.
- **`source_run.json` (737 bytes):** Single-node TinyLlama placeholder. Schema matches what `croissant_export.py:build_records_from_run_artifact()` expects: `model_id`, `run_metadata`, `all_nodes[]`.
- **`sources/` (empty):** Created but never populated. `croissant_export.py:build_batch_metadata_document()` writes to "data/sources/source_run_NNN.json" relative paths, so this directory was preallocated for a batch build that never ran.

**Verdict:** Discardable. The CSV and JSON do not contain any of the real experimental data the paper reports.

---

## (0.4) Raw experiment output structure (per Track)

All raw artifacts live under `scripts/`. Validated by direct `python -c "json.load(...)"` inspection (Session 3, 2026-05-17).

### Track F — Entropy-MCTS negative control (Experiment 1 headline)

**Files:** [scripts/_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json](scripts/_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json) (430 KB), [scripts/_track_f_results_meta-llama_Llama-3.2-3B-Instruct.json](scripts/_track_f_results_meta-llama_Llama-3.2-3B-Instruct.json) (408 KB)

**Top-level keys:** `model_id`, `device`, `n_items` (=200), `config` (dict: `alpha_set=[0.1,0.5,1.0]`, `depth=3`, `n_nodes=27`, `max_new_tokens=5`, `seed=42`), `torch_version` (=`2.6.0+cu124`), `cuda_version` (=`12.4`), `gpu_name` (=`NVIDIA GeForce RTX 3060`), `items` (list of 200), `wallclock_seconds` (1098.6 for 1B; ~2020 for 3B est).

**Items structure:** Each item has `idx`, `source_id`, `gold_letter`, `n_options` and three condition blocks **G / R / E**:
- G (greedy): `answer_text`, `pred_letter`, `first_token_entropy_nats`, `per_token_entropy_mean_nats`, `n_generated`, `new_ids`, `perplexity_5tok`, `search_status` (="n/a"), `mean_search_reward` (=NaN), `best_path` (=null), `correct` (0/1).
- R (random-MCTS): same keys plus `best_search_reward`, `mean_terminal_entropy_nats`, `best_path` (e.g., `[0.5, 0.1, 0.1]`), `n_paths_evaluated` (=27), `n_apply_failures`.
- E (entropy-MCTS): same shape as R.

**Suitable recordSet flatten strategy:** One row per `(item_idx, condition)` tuple = 200 × 3 = 600 rows per scale. Combined 1B + 3B = 1200 rows. Fields per row: `scale, item_idx, source_id, gold_letter, n_options, condition (G|R|E), pred_letter, correct, answer_text, first_token_entropy_nats, per_token_entropy_mean_nats, perplexity_5tok, search_status, mean_search_reward, best_search_reward, mean_terminal_entropy_nats, best_path_json, n_paths_evaluated, n_apply_failures`. (NaN values for G's `mean_search_reward` / `best_search_reward` etc. need string-null handling.)

### Track G — Latent Cartography (Experiment 2; load-bearing primary)

**Files (MCTS-primary, the load-bearing variant):** [scripts/_track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json](scripts/_track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json) (518 KB), [scripts/_track_g_mcts_results_meta-llama_Llama-3.2-3B-Instruct.json](scripts/_track_g_mcts_results_meta-llama_Llama-3.2-3B-Instruct.json) (519 KB)

**Top-level keys:** `model`, `mode` (=`mcts`), `seed` (=42), `n_items_per_class` (=100), `classes` (=`["C1","C2","C3","C4"]`), `elapsed_sec`, `records[]`, `alpha_set` (=`[0.1, 0.5, 1.0]`), `n_nodes` (=27), `depth` (=3).

**Records:** 1B has 385 records (C1=100, C2=100, C3=85, C4=100); 3B has 385 records (same breakdown). [CORRECTION 2026-05-17 Phase 2 — see execution-log Correction-1: this entry originally claimed "C3 has 85 because HellaSwag continuation evaluator drops items whose gold_label can't be unambiguously tokenized — confirmed per Track G report." That was a hallucinated primary-source citation. The actual reason, per Track G report line 44 + 49 verbatim verification: C3 = TruthfulQA mc1 MCQ (NOT HellaSwag), and 15 of the first 100 validation items are dropped at the dataset-construction step for having fewer than 3 distractors. The 85-record count is identical at both scales (not a 3B-only effect).]

**Record schema:** `class, item_idx, subclass, source_id, eval_mode (=first_token_str), prompt_preview, gold, search_status, best_path, n_paths_evaluated, n_apply_failures, mean_search_reward, best_search_reward, mean_terminal_entropy_nats, best_terminal_entropy_nats, argmax_token_str, argmax_token_id, next_token_entropy_nats, top5 (list of [str, prob] pairs), gold_first_token_rank, gold_first_token_prob, wallclock_sec`.

**Class mapping** [CORRECTED 2026-05-17 Phase 2 — original entry below had C3/C4 swapped; the corrected mapping per Track G report at [docs/logs/2026-05-11_track-G-cartography-report.md:47-50](docs/logs/2026-05-11_track-G-cartography-report.md) + JSON `subclass` cross-check is]:
- C1: Factual recall (hand-constructed, 25 capitals + 25 dates + 25 arithmetic + 25 common-knowledge; definition at [scripts/_track_g_c1_factual_recall.json](scripts/_track_g_c1_factual_recall.json); subclass labels: capital/date/arithmetic/knowledge)
- C2: ARC-Easy MCQ (subclass label: logical_deduction)
- C3: **TruthfulQA mc1 MCQ** (subclass label: moral_preference; 85 records due to <3-distractor filter at dataset construction)
- C4: **HellaSwag Continuation** (subclass label: creative_completion; 100 records)

**Suitable recordSet flatten strategy:** One row per existing record. Total = 385 × 2 = 770 records.

**Files (initial-variant, historical):** [scripts/_track_g_results_*.json](scripts/_track_g_results_meta-llama_Llama-3.2-1B-Instruct.json) (2.4 MB each). 2310 records each — the parameter-sweep variant (`alpha_grid = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]` × ~385 items). Per memory's 2026-05-11 protocol-deviation correction, this variant is **historical (deviated from MCTS protocol)** and was superseded by the MCTS-primary files. **Should NOT be the primary Cartography record set** in the Croissant package. Could be included as a supplementary FileObject with description noting its status as the supplementary parameter-sweep variant cited at §5 Exp 2.

**Supporting C1 definition file:** [scripts/_track_g_c1_factual_recall.json](scripts/_track_g_c1_factual_recall.json) (12 KB) holds the 100-item C1 question/gold list. This is the **dataset definition** (not results); include as a separate FileObject in the package describing the C1 source.

### Track A — Static entropy↔correctness correlation

**Files:** [scripts/_track_a_results_meta-llama_Llama-3.2-1B-Instruct.json](scripts/_track_a_results_meta-llama_Llama-3.2-1B-Instruct.json) (32 KB), [scripts/_track_a_results_meta-llama_Llama-3.2-3B-Instruct.json](scripts/_track_a_results_meta-llama_Llama-3.2-3B-Instruct.json) (32 KB)

**Top-level keys:** `model_id`, `device`, `calibration` (dict, 5 keys), `magnitude` (dict, 9 keys), `correctness` (dict, 4 keys: `easy`, `hard`, `ultra`, `combined`), `wallclock_seconds`.

**Shape:** Aggregate analysis. `correctness.easy.records` has 25 entries (question/model_answer/correct/first_token_entropy_nats/expected_substrs); `correctness.hard.records` has 30; `correctness.ultra.records` has 30. The 85-item total matches paper §A.4 line 498 claim ("Sample: 85 factual-recall items (25 easy + 30 hard + 30 ultra)"). Each `correctness.<bucket>` block also has `pearson_r_entropy_vs_correct`, `spearman_rho_entropy_vs_correct`, `abs_pearson`, `abs_spearman`, `accuracy`, `n_correct`, `n_total`.

**Suitable flatten:** One row per (item, bucket) tuple = 85 × 2 scales = 170 records. Or describe the aggregate `calibration`/`magnitude` dicts as their own typed sub-RecordSets. Simpler approach: include as a supplementary FileObject with no flattened CSV; reviewers read the JSON directly.

### Track C — Benchmark calibration

**Files:** [scripts/_track_c_results_meta-llama_Llama-3.2-1B-Instruct.json](scripts/_track_c_results_meta-llama_Llama-3.2-1B-Instruct.json) (588 KB), [scripts/_track_c_results_meta-llama_Llama-3.2-3B-Instruct.json](scripts/_track_c_results_meta-llama_Llama-3.2-3B-Instruct.json) (584 KB)

**Top-level keys:** `model_id`, `device`, `n_per_benchmark` (=100), `benchmarks` (dict, 6 keys).

**Benchmarks (actual names from JSON):** `ARC-Easy, SciQ, HellaSwag, AQuA-RAT, ProofWriter, LogiQA`. Each has `n=100`, `n_correct`, `accuracy`, entropy stats, `headroom_*` counters, `mcnemar_n_for_5pp_p05_pwr80`, `wall_seconds`, `records[]` (100 entries with `question_preview, options, labels, gold_letter, answer_text, pred_letter, correct, first_token_entropy_nats, letter_probs, top3_letters, gold_in_top3, gold_rank_among_letters`).

**Total records:** 100 × 6 benchmarks × 2 scales = 1200.

**Suitable flatten:** One row per `(scale, benchmark, item_idx)`. Or describe as supplementary FileObject (no flatten) per Track A approach.

### Track D — VRAM measurement

**Files:** [scripts/_track_d_vram.json](scripts/_track_d_vram.json) (6.4 KB, 1B), [scripts/_track_d_vram_3b.json](scripts/_track_d_vram_3b.json) (6.5 KB, 3B)

**Top-level keys:** `model_id`, `device`, `torch_version`, `cuda_version`, `gpu_name`, `gpu_total_mib`, `n_repeats` (=3), `seed_tokens` (=4000), `configs[]` (3 entries: (d,b,n)=(3,3,27)/(5,3,81)/(10,3,1700)), `notes[]`.

**Configs:** Per (depth, branches, nodes) cell: `kv_shapes`, `kv_cache_mib_median`, `fwd_peak_mib_runs`, `baseline_steady_mib_runs`, `mid_steady_mib_runs`, `full_steady_mib_runs`, `full_step_peak_mib_runs`, `accumulator_total_mib_runs`, and median + delta versions of each. The 3.04–3.06× M_KV constant-factor result reported in §6 derives from `delta_full_in_kv_units` (= 3.06375 at 1B per [scripts/_track_d_vram.json:74](scripts/_track_d_vram.json:74)).

**Suitable flatten:** One row per `(scale, config)` = 6 records. Or as supplementary FileObject.

---

## (0.5) Paper claims that the package must support

### Direct Croissant claims (load-bearing)

| Location | Line | Verbatim |
|---|---|---|
| Abstract S8 | 56 | "Artifacts ship with a Croissant~1.1 metadata package." |
| §7.5 Future Work | 363 | "A Croissant~1.1 dataset package of runtime evidence artifacts is included as supplementary material and will be extended in Phase~B." |
| §7.6 Ethical Considerations | 368 | "The Croissant~1.1 metadata package (§\ref{sec:future-work}) documents per-node runtime evidence artifacts from Experiments~1--2 with provenance and usage-constraint fields." |

### Supplementary URL placeholder

**Location:** [04.18.2026-NeurIPS-Research-Proposal-2.tex:510](docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex:510). Closing sentence of §A.5:

> "Exact HuggingFace checkpoint commit SHAs, the full library version pin manifest (`transformers`, `datasets`, `huggingface_hub`, `numpy`, `scipy`), and per-script command-line argument references are recorded in the anonymized supplementary materials package, available at \url{https://anonymous.4open.science/r/kv-mcts-dimensional-escape-XXXX} (the final slug is assigned at Day-12 supplementary build per plan v2 §2)."

XXXX is the slug placeholder. Per the task prompt, filling §A.5's slug into the paper TeX is the **one paper-TeX edit allowed in Session 3, deferred to Phase 4**.

### Implicit content claims (the package must describe these, by reference at minimum)

- §A.2 line 490: HF checkpoint commit SHAs for both scales (one per scale), to be passed to `from_pretrained(revision=<SHA>)`. **Josh fills these via `huggingface-cli` at submission**. Package should have placeholder fields.
- §A.1 line 487: `uv.lock` export of the pinned dependency manifest.
- §A.4 paragraphs: CLI commands for reproducing each track (`scripts/diagnose_track_f_negcontrol.py --model ... --seed 42`, etc.).

### Paper-claim drift surfaced during Phase 0 (NOT Croissant blockers; informational only)

1. **§7.6 line 368 "per-node runtime evidence artifacts"** — α+ vintage wording. Track F + G actual data is per-item-per-condition (Track F) and per-item-per-class (Track G), NOT per-MCTS-node. The Croissant package will document per-item records; reviewers comparing the §7.6 claim to the package will likely accept it as "runtime evidence" charitably, but the literal "per-node" claim is inaccurate. **Session 2's domain** to consider.
2. **§A.4 line 501 "BBH-LogicalDeduction-3"** — the Track C JSON benchmarks list is `[ARC-Easy, SciQ, HellaSwag, AQuA-RAT, ProofWriter, LogiQA]`. The paper says BBH-LogicalDeduction-3 in the 6th slot; the JSON says LogiQA. **These are different datasets.** Session 2's domain.
3. **§A.4 line 498 "Wall-clock approximately 13~min for both 1B and 3B scales"** for Track A — actual `wallclock_seconds` in the JSONs is **116 sec (1B) and 134 sec (3B)** ≈ 2 min each. The 13-min figure may be the original protocol estimate (recorded in `memory/MEMORY.md` Pace observation: "Track A: estimated ~30 min, completed ~13 min"), but the run-tracked wallclock is ~2 min. Session 2's domain.

---

## (0.6) Croissant 1.1 spec requirements (primary-source verified)

**Spec:** [external/croissant/docs/croissant-spec-1.1.md](external/croissant/docs/croissant-spec-1.1.md) (2428 lines, Croissant 1.1 published 2026-01-29). RAI 1.0 spec at [external/croissant/docs/croissant-rai-spec.md](external/croissant/docs/croissant-rai-spec.md) (609 lines).

### Required dataset-level properties (verbatim from §"Dataset-level Information" → "Required" table, spec line 347)

| Property | Type | Cardinality |
|---|---|---|
| `@context` | URL | ONE |
| `@type` | Text (must be `sc:Dataset`) | ONE |
| `dct:conformsTo` | URL — must include `http://mlcommons.org/croissant/1.1` | MANY |
| `description` | Text | ONE |
| `license` | URL (SPDX recommended) or CreativeWork | MANY |
| `name` | Text | ONE |
| `url` | URL | ONE |
| `creator` | Organization or Person | MANY |
| `datePublished` | Date or DateTime | ONE |
| `distribution` | FileObject or FileSet (required per "Modified and Added Properties" §510) | MANY |

### Recommended properties

- `keywords` (DefinedTerm/Text/URL, MANY)
- `publisher` (Organization/Person, MANY)
- `version` (Number/Text, ONE)
- `dateCreated`, `dateModified` (DateTime, ONE each)
- `sameAs`, `sdLicense`, `inLanguage`
- `citeAs` (Text, ONE) — paper citation (bibtex format recommended)
- `sdVersion` (Number/Text, ONE) — metadata version, "1.1" for Croissant 1.1

### RecordSet + Field structure (spec §"RecordSets" line 870)

- `recordSet[]`: each is `{@type: cr:RecordSet, @id, name, description, key?, field[]}`
- `field[]`: each is `{@type: cr:Field, @id, name, description, dataType, source: {fileObject: {@id}, extract: {column}}}`
- `dataType` examples: `sc:Text`, `sc:Integer`, `sc:Float`, `sc:Boolean`, `sc:Date`, `sc:DateTime`, `sc:ImageObject`, `cr:Field` (for nested fields)

### RAI 1.0 extension fields (spec §"Responsible AI and Governance" line 2075; RAI spec at croissant-rai-spec.md)

RAI is OPTIONAL; recommended for safety-relevant data. Fields the existing stub uses:
- `rai:dataUseCases` — intended uses
- `rai:dataLimitations` — known limitations
- `rai:dataBiases` — known biases
- `rai:personalSensitiveInformation` — PII screening
- `rai:dataReleaseMaintenancePlan` — versioning/maintenance plan

### Validation tooling

- **Reference validator:** `mlcroissant` Python package + CLI. Source under [external/croissant/python/mlcroissant/](external/croissant/python/mlcroissant/). **NOT currently installed** in the project Python env (`pip show mlcroissant` → not found; `which mlcroissant` → not in PATH).
- **Install options:** `pip install mlcroissant` (PyPI) OR install from submodule (`pip install ./external/croissant/python/mlcroissant`).
- `logomesh/croissant_export.py:validate_with_mlcroissant()` gracefully returns `(127, "...not found...")` if absent; the shape-checker `validate_metadata_shape()` covers core structure even without the strict validator.

---

## (0.7) ACL/ARR format-acceptance status

**Source:** [docs/NeurIPS/acl_formatting.md](docs/NeurIPS/acl_formatting.md) §"Supplementary Material" (lines 292-306).

### Key acceptance rules (verbatim)

> "Submissions may include non-readable supplementary material used in the work and described in the paper." (line 294)

> "Any accompanying software and/or data should include licenses and documentation of research review as appropriate." (line 295)

> "Supplementary material may report preprocessing decisions, model parameters, and other details necessary for the replication of the experiments reported in the paper." (line 296)

> "Nonetheless, supplementary material should be supplementary (rather than central) to the paper. **Submissions that misuse the supplementary material may be rejected without review.** Supplementary material may include explanations or details of proofs or derivations that do not fit into the paper, lists of features or feature templates, sample inputs and outputs for a system, pseudo-code or source code, and data." (lines 299-301)

> "(Source code and data should be separate uploads, rather than part of the paper)." (line 302)

> "The paper should not rely on the supplementary material: while the paper may refer to and cite the supplementary material and the supplementary material will be available to the reviewers, they will not be asked to review the supplementary material." (line 304)

> "Review versions of supplementary material must follow the same anonymity guidelines as the main paper." (line 306)

### Findings

- **Croissant is not specifically required or specially accepted** by ACL/ARR. The acl_formatting.md document never mentions Croissant by name. The package is just JSON-LD data uploaded as part of supplementary materials.
- **The package must be anonymized** for review (line 306).
- **"Misuse" risk** (line 300): Including the Croissant package as a documentation-of-data file is on-spec; including the raw track JSONs as data uploads is on-spec. Including paper content (proofs, claims) that should have been in the paper proper would be misuse.
- **Paper must not rely on supplementary** (line 304): Tracks F + G headline numbers in the abstract and §5 must stand on their own from the paper alone. The Croissant package is a reproducibility add-on; reviewers cite-but-don't-review it.
- **Per Q3 of the cleanup-prompt's discovery list (`docs/logs/2026-05-18_repo-cleanup-and-submission-prep-prompt.md` lines 99-123):** Phase 0 of the cleanup prompt will verify anonymous.4open.science workflow specifics; for Croissant-package purposes, the relevant constraint is the URL slug XXXX in §A.5 must be filled at Phase 4 (cleanup).

**Verdict:** No ACL/ARR-imposed Croissant format constraints beyond standard supplementary-material rules. Validate the package against Croissant 1.1 spec; let the supplementary hosting and §A.5 URL be handled by Phase 4 cleanup.

---

## Open questions for Josh

The following Phase 1 decisions should be made before the build plan locks in. None block writing the build plan in draft form; all need Josh's sign-off before Phase 2 executes.

### Q-J1: Build approach — recommend Option B (one-shot script)

**Three options for the Phase 2 build:**
- **(A) Extend `logomesh/croissant_export.py`:** Add per-track record-builder functions (`build_records_from_track_f`, etc.), generalize `build_metadata_document()` to accept arbitrary record sets, refresh keywords/URLs/RAI fields. Pro: code reusable for Phase B. Con: existing tests assume α+ diagnostic-state taxonomy; module's design assumes one CSV per package — needs significant surgery. Estimated wall time: 3-5 hours.
- **(B) One-shot `scripts/build_croissant_package.py`:** Standalone script that loads the actual track JSONs, flattens to CSVs, generates `docs/dataset/croissant.json` + `data/track_f.csv` + `data/track_g_cartography.csv` (+ optional supporting-track files), hashes everything, validates against the Croissant 1.1 spec via `mlcroissant` (after `pip install`). Pro: simpler, scoped to the actual deliverable, disposable after submission. Con: Phase B will need a separate build pipeline. Estimated wall time: 1.5-2.5 hours.
- **(C) Hand-author `docs/dataset/croissant.json`:** Start from the existing stub's @context block, hand-write the per-track recordSets and field definitions, compute hashes manually. Pro: full control over every field. Con: error-prone for 50+ fields across Track F + G + supporting tracks; tedious. Estimated wall time: 2.5-4 hours plus higher rework probability.

**Recommendation: Option B.** Reasons: (i) `croissant_export.py` retrofit cost ≈ rewriting it; (ii) the package is a one-shot artifact for ARR submission, not a long-lived pipeline; (iii) Phase B can build its own pipeline when scope is clearer; (iv) a script under `scripts/` is in-scope for repo cleanup Phase 4 and surveys easily.

### Q-J2: Package scope — recommend Track F + Track G MCTS-primary as core, supporting tracks A/C/D as supplementary FileObjects (no flattened CSV)

**Core record sets (flattened to CSVs):**
- `track_f_negcontrol`: 1200 rows = 200 items × 3 conditions × 2 scales.
- `track_g_cartography_mcts`: 770 rows = 385 records × 2 scales (per-record from MCTS-primary variant).

**Supporting FileObjects (declared in distribution but not flattened):**
- `track_g_cartography_initial_sweep` (historical parameter-sweep variant; cited at §5 Exp 2 for completeness; description must note it's the deviated-from-MCTS-protocol variant per memory's 2026-05-11 correction).
- `track_g_c1_factual_recall_definition` (the dataset definition for class C1).
- `track_a_correlation` (aggregate JSON; reviewers read directly).
- `track_c_calibration` (aggregate + per-benchmark records; reviewers read directly).
- `track_d_vram_1b` + `track_d_vram_3b` (small aggregate JSONs).

**Out of scope:** `mcts_output_1776800541.json` at repo root (timestamp-named Phase-2 debugging output; not paper-referenced). `drift_results.csv` at repo root (need to verify what this is in Phase 4 cleanup; if Phase-2 debug, exclude). The Phase-2 `docs/dataset/data/{interventions.csv, source_run.json}` placeholders.

### Q-J3: §7.6 "per-node" wording — flag-and-proceed, or escalate?

Per **Hard Rule 7** of the session prompt: "If Croissant build surfaces a paper-claim discrepancy (e.g., paper says X is in the package but X doesn't exist), STOP and escalate to Josh — paper-content change is session 2's domain."

The Phase-2-era "per-node runtime evidence" phrasing at [04.18.2026-NeurIPS-Research-Proposal-2.tex:368](docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex:368) is a soft drift — the Croissant package will document per-item records (which is reasonable "runtime evidence"). I recommend:
- **Option α**: Build the package with per-item records; do not block on §7.6 wording. Add a flag to Session 2's queue for §7.6 to swap "per-node" → "per-item" or similar at their next pass. Risk: reviewer literal-reads §7.6 and complains. Low probability but real.
- **Option β**: Escalate immediately; pause Phase 1 until Josh decides whether §7.6 needs to swap before package build.

**Recommendation: Option α** (build per-item, flag for Session 2). The Croissant package's content is determined by the data; the §7.6 text alignment is independent.

### Q-J4: Track C benchmark drift — LogiQA vs BBH-LogicalDeduction-3

Per **Hard Rule 7**: another paper-claim drift. Track C's actual 6th benchmark is `LogiQA`, but [04.18.2026-NeurIPS-Research-Proposal-2.tex:501](docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex:501) states `BBH-LogicalDeduction-3`. LogiQA and BBH-LogicalDeduction-3 are different benchmarks (LogiQA = Liu et al. 2020 IJCAI Chinese-civil-service logical reasoning; BBH-LogicalDeduction-3 = BIG-Bench Hard 3-objects logical-deduction subset).

- The Croissant package will describe the actual benchmark (LogiQA) since that's what the JSON contains.
- Session 2's domain to consider whether to (i) update §A.4 to "LogiQA" or (ii) note the substitution explicitly.

**Recommendation:** Flag to Session 2; Croissant package proceeds with LogiQA. (Track C is a supporting track in any case, not headline.)

### Q-J5: Track A wallclock drift — ~13 min (paper) vs ~2 min (JSON)

Per **Hard Rule 7**: §A.4 line 498 says "Wall-clock approximately 13~min for both 1B and 3B scales"; JSON `wallclock_seconds` is 116 sec (1B) and 134 sec (3B) — about 2 min each.

Possible explanations: (i) 13 min was the original protocol estimate retained from earlier paper drafts (memory MEMORY.md pace section lists "Track A: estimated ~30 min, completed ~13 min" — the 13 min may be the wall-clock for an earlier longer protocol that was then trimmed); (ii) the JSON's `wallclock_seconds` may only cover the generation loop, not model-loading + setup; (iii) the paper claim may be incorrect.

- The Croissant package can faithfully describe the JSON's `wallclock_seconds` value as-is.
- Session 2's domain whether to update §A.4.

**Recommendation:** Flag to Session 2; Croissant package proceeds with actual JSON values. (Track A is a supporting track, not headline.)

### Q-J6: Anonymization timing — bake into Phase 2 build, or defer to Phase 4 cleanup?

Per the task prompt's Phase 1.5: anonymization is for the SUPPLEMENTARY PACKAGE only (memory's 2026-05-12 correction). The Croissant package being built lives in `docs/dataset/croissant.json` in the live repo — which is NOT anonymized.

Two builds possible:
- **Option α**: Build live-repo Croissant package in Phase 2 (with author info, real GitHub URLs). Phase 4 cleanup then creates an anonymized copy of the supplementary tree (including the Croissant package) for upload to anonymous.4open.science.
- **Option β**: Build directly into anonymized form (no author info, no identifying URLs) in Phase 2, then the live-repo copy IS the anonymized version. Phase 4 just copies it to the supplementary upload tree.

**Recommendation: Option α.** Reasons: (i) keeps the in-repo Croissant package useful as documentation of the live repo's data (with proper attribution, real URL); (ii) cleanly separates "Croissant build" (Phase 2) from "anonymization" (Phase 4 cleanup); (iii) Phase 4's anonymization step is mechanical (sed pass over a copy).

### Q-J7: `mlcroissant` install — yes, install for validation?

`mlcroissant` is not currently installed. Strict validation requires it. Three options:
- Install via `pip install mlcroissant` (or `uv pip install mlcroissant`) into the project venv.
- Use the submodule directly: `pip install ./external/croissant/python/mlcroissant`.
- Skip strict validation; rely on `croissant_export.py:validate_metadata_shape()` (which the build script can import).

**Recommendation:** Install via `uv pip install mlcroissant` so strict validation runs in Phase 2 (catches issues the shape-checker misses). Quick (~30 sec).

---

## Phase 0 verdict + Phase 1 readiness

**Phase 0 complete.** Discovery surfaced:
- The existing scaffolding (`croissant_export.py`, `croissant_schema_stub.json`, `data/*`) is Phase-2-era and cannot be reused as-is. The April-15 stub data is toy placeholder.
- The actual Track F + G + supporting JSONs are well-structured and contain everything the Croissant package needs to document.
- Croissant 1.1 spec requirements verified from primary source (submodule); all 10 required dataset-level properties are clear; RAI 1.0 fields are optional and the existing stub's α+ vintage strings need replacement.
- ACL/ARR has no special Croissant requirements; supplementary anonymity is required.
- 3 paper-claim drift items surfaced incidentally; all NON-blocking for Croissant but should reach Session 2.

**Phase 1 readiness:** Ready to write the build plan. Pending Josh sign-off on:
- Q-J1 build approach (recommend Option B)
- Q-J2 package scope (recommend Track F + G as core, supporting tracks as FileObjects)
- Q-J3 §7.6 wording escalation level (recommend α: flag-and-proceed)
- Q-J4 Track C drift escalation (recommend flag-and-proceed)
- Q-J5 Track A drift escalation (recommend flag-and-proceed)
- Q-J6 anonymization timing (recommend α: live-repo first, anonymize in Phase 4)
- Q-J7 mlcroissant install (recommend yes, install)
