# Croissant Build Execution Log

**Date:** 2026-05-17
**Phase:** 2 (Build executed)
**Discovery report:** [docs/logs/2026-05-17_croissant-discovery.md](docs/logs/2026-05-17_croissant-discovery.md)
**Build plan:** [docs/logs/2026-05-17_croissant-build-plan.md](docs/logs/2026-05-17_croissant-build-plan.md)
**Status:** ✅ Build complete, all validation layers passed (after Correction-1 below).

## Correction-1 — Track G C3/C4 class mapping (caught at Phase 3 final cross-check)

**What was wrong (initial Phase 2 build):** The build script's `TRACK_G_CLASS_NAMES` dict had:
- C3 → "hellaswag_continuation"
- C4 → "truthfulqa_mcq"

**What is correct (per primary-source verification at Phase 3.1):**
- C3 → **"truthfulqa_mcq"** (subclass label `moral_preference`; first prompt: "What is the smallest country in the world that is at least one square mile in area? A. ...")
- C4 → **"hellaswag_continuation"** (subclass label `creative_completion`; first prompt: "A man is sitting on a roof. he")

**Primary sources verifying the correct mapping:**
1. [docs/logs/2026-05-11_track-G-cartography-report.md:47-50](docs/logs/2026-05-11_track-G-cartography-report.md) — lists the four classes verbatim: C3 = "TruthfulQA mc1 MCQ", C4 = "HellaSwag Continuation".
2. [docs/logs/2026-05-11_track-G-cartography-report.md:44](docs/logs/2026-05-11_track-G-cartography-report.md) — "C3 drops 15 items with <3 distractors" — TruthfulQA filter at dataset construction, not HellaSwag tokenization at runtime.
3. The actual JSON records: `subclass` fields show C3 = `moral_preference` and C4 = `creative_completion`, and prompt_preview strings match TruthfulQA-style and HellaSwag-style respectively.
4. The paper itself, [04.18.2026-NeurIPS-Research-Proposal-2.tex:266](docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex:266) Table 2 caption: "MCQ classes (C2 ARC-Easy, C3 TruthfulQA)".

**Root cause:** Initial Phase 0 discovery hallucinated the C3/C4 mapping (likely influenced by the order of class names in the abstract S6 enumeration: "factual recall, ARC-Easy/TruthfulQA MCQ, HellaSwag continuation"). The discovery report claimed "per [Track G report]" but the report file was not actually opened until Phase 3 cross-check surfaced the inconsistency between paper §5 line 266 (C3=TruthfulQA) and my build's class_name (truthfulqa_mcq mapped to C4). This is a **primary-source-fidelity violation per memory's 2026-05-13 rule**: trusted a memory-style recall over the actual source.

**Fix applied (Phase 3.1):**
- Updated `TRACK_G_CLASS_NAMES` dict in [scripts/build_croissant_package.py](scripts/build_croissant_package.py): swap C3 ↔ C4 names.
- Updated `TRACK_G_FIELDS` `class_code` / `class_name` / `item_idx` / `subclass` field descriptions.
- Updated `track_g_cartography_mcts` recordSet description in `_build_metadata`.
- Updated `rai:dataLimitations` text (drop reason: TruthfulQA <3-distractor filter, not HellaSwag tokenization).
- Updated `track-g-mcts-raw-1b` / `track-g-mcts-raw-3b` FileObject descriptions (drop reason; also clarified both scales have 85 records for C3, not 3B-only).

**Re-ran build with `--strict`:** mlcroissant exit=0 (OK). CSV verification shows correct mapping per (scale, class_code) tuples (all 8 combinations spot-checked).

**Companion corrections applied to upstream docs:**
- [docs/logs/2026-05-17_croissant-discovery.md](docs/logs/2026-05-17_croissant-discovery.md): inline `[CORRECTION 2026-05-17 Phase 2]` markers on the Class mapping section + records-count entry.
- [docs/logs/2026-05-17_croissant-build-plan.md](docs/logs/2026-05-17_croissant-build-plan.md): inline `[CORRECTED 2026-05-17]` marker on the Tier-A scope table row.

---



---

## Phase 2 actions taken (in order)

### Action 1 — LICENSE fix (Josh's Day-14-PM directive)

**File:** [LICENSE](LICENSE)

Replaced line 1 verbatim:
```diff
- Copyright 2025 AgentBeats
+ Copyright (c) 2026 [Authors]
```

Rest of MIT body unchanged. Rationale per Josh's directive: "AgentBeats" was fork-template residue; `[Authors]` matches the paper TeX `\author{[Authors]}` anonymity convention required by [acl_formatting.md:306](docs/NeurIPS/acl_formatting.md:306); year updated to current submission year 2026; MIT body preserved as academic-research-code standard. Camera-ready de-anonymization (post-acceptance ~2026-08) will update both the paper `\author{...}` and the LICENSE copyright line in a single coordinated edit.

### Action 2 — Pre-build deletions (plan §1.1 approved scope)

Removed Phase-2-era toy scaffolding:
- `docs/dataset/croissant_schema_stub.json` (April-15 stub generated from `run_kv_mcts.py --nodes 1` toy run)
- `docs/dataset/data/interventions.csv` (TinyLlama single-row placeholder)
- `docs/dataset/data/source_run.json` (TinyLlama single-node placeholder)
- `docs/dataset/data/sources/` (empty directory)

Live `logomesh/croissant_export.py` and `tests/test_croissant_export.py` left untouched (out of Phase 2 scope; Phase 4 cleanup may revisit as α+ vintage candidates).

### Action 3 — `mlcroissant` install

Command:
```
uv pip install mlcroissant
```

Result: installed into the project venv at `.venv/Scripts/mlcroissant.exe`. The `mlcroissant` Python package importable; CLI accessible via `uv run mlcroissant {load|documentation|validate}`.

### Action 4 — Build script authored

**File:** [scripts/build_croissant_package.py](scripts/build_croissant_package.py) (~900 lines including docstrings)

Structure per plan §1.3:
- Imports + path constants + class-code → human-name mapping.
- `FieldSpec` dataclass + Track F (19 fields) + Track G (24 fields) schemas.
- IO helpers: `_sha256_file`, `_load_json` (handles JSON `NaN`/`Infinity` constants via `parse_constant=_parse_json_constant`), `_coerce_cell` (None→"", NaN→"", bool→"true/false", list/dict→compact JSON).
- Flatteners: `_flatten_track_f` (with invariant assertions: n_items==200, seed==42, depth==3, n_nodes==27), `_flatten_track_g_mcts` (mode=='mcts', classes==C1-C4, ≥385 records).
- File-copy helper with sha256 idempotency check.
- Inline `_validate_shape()` (Croissant 1.1 generic; replaces the Phase-2-coupled `validate_metadata_shape` from croissant_export.py — see Deviation 1 below).
- Croissant metadata builder: `_field_dict` per spec, `_build_metadata` aggregates everything per plan §1.2, `_build_file_objects` for distribution[].
- `build_package()` orchestrates 7 steps.
- `_write_manifest()` produces a human-readable MANIFEST.txt with sha256s + descriptions.
- `main()` argparse: `--output-dir`, `--strict`, `--build-date`.

### Action 5 — Build run (no strict)

Command:
```
uv run python scripts/build_croissant_package.py
```

Result: build completed successfully, all 7 stages logged, summary printed:
- Croissant package: `docs/dataset/croissant.json` (37,520 bytes, 917 lines)
- Track F CSV: 1200 rows = 200 items × 3 conditions × 2 scales ✓
- Track G CSV: 770 rows = 385 per scale × 2 scales ✓
- 13 raw JSON copies + uv.lock under `docs/dataset/data/raw/` + `docs/dataset/data/uv.lock` ✓
- MANIFEST.txt written

### Action 6 — Build run (strict validation)

Command:
```
uv run python scripts/build_croissant_package.py --strict
```

Result: `mlcroissant validate --jsonld docs/dataset/croissant.json` returned **exit=0 (OK)**. Output: `I0517 18:25:08.910680 11480 validate.py:53] Done.`

### Action 7 — Validation layers 3-5

**Layer 3 — Manual paper-claim cross-check** (plan §1.4):

| Paper claim | Verification | Status |
|---|---|---|
| Abstract S8 line 56 "Artifacts ship with a Croissant 1.1 metadata package." | `docs/dataset/croissant.json` exists; `conformsTo == ["http://mlcommons.org/croissant/1.1", "http://mlcommons.org/croissant/RAI/1.0"]` | ✅ |
| §7.5 line 363 "Croissant 1.1 dataset package of runtime evidence artifacts is included as supplementary material" | Same; plus `recordSet[]` has `track_f_negcontrol` (19 fields) + `track_g_cartography_mcts` (24 fields) | ✅ |
| §7.6 line 368 "documents per-node runtime evidence artifacts from Experiments 1--2 with provenance and usage-constraint fields" | "per-node" wording stale (data is per-item, flagged Q-J3 for Session 2); Experiments 1 + 2 covered (Track F = Exp 1, Track G = Exp 2); provenance via `creator`, `datePublished`, `version`, sha256 per FileObject, raw JSONs contain seed/model_id/torch_version/cuda_version; usage-constraint via `license` + 5 `rai:*` fields | ⚠️ (per-node wording flagged; rest verified) |
| §A.5 line 510 supplementary URL placeholder XXXX | Slug pending — Phase 4 cleanup assigns | ⏸ deferred to Phase 4 |

**Layer 4 — Spot-checks against raw JSONs:**

Track F (item idx=0, 1B, all 3 conditions):
```
CSV: scale=1B idx=0 cond=G correct=1 pred=A gold=A    <==> RAW: pred=A correct=1 (no search) ✓
CSV: scale=1B idx=0 cond=R correct=1 pred=A gold=A    <==> RAW: pred=A correct=1 best_path=[0.5,0.1,0.1] ✓
CSV: scale=1B idx=0 cond=E correct=1 pred=A gold=A    <==> RAW: pred=A correct=1 best_path=[1.0,0.5,1.0] mean_reward=-0.1257 ✓
```

Track G (C1 idx=0, 1B):
```
CSV: gold=Paris argmax=Paris gold_rank=0 prob=0.3483574092388153 best_path=[0.1,0.1,0.1] best_reward=-0.17178559218078948
RAW: gold=Paris argmax=Paris gold_rank=0 prob=0.3483574092388153 best_path=[0.1,0.1,0.1] best_reward=-0.17178559218078948
                                                                                                            (verbatim match)
```

croissant.json top-level structural check:
- `name`: "KV-Cache Inception: Reversible KV-Cache MCTS Runtime Evidence" ✓ (Option D+ era)
- `conformsTo`: [Croissant 1.1 URI, RAI 1.0 URI] ✓
- `license`: `https://opensource.org/licenses/MIT` ✓
- `url`: `https://github.com/LogoMesh/KV-Cache-Inception` ✓ (live-repo; Phase 4 anonymizes the supplementary copy)
- `version`: "1.0.0", `sdVersion`: "1.1" ✓
- `distribution`: 16 FileObjects (2 CSVs + 13 raw JSONs + 1 uv.lock) ✓
- `recordSet`: 2 (track_f_negcontrol 19 fields, track_g_cartography_mcts 24 fields) ✓
- All 5 `rai:*` fields present ✓

**Layer 5 — pytest hygiene** (CLAUDE.md hard rule 2):

Command:
```
uv run pytest tests/ -v
```

Result: **179 passed in 32.25s**, 0 failures, 0 errors. The pre-existing [tests/test_croissant_export.py](tests/test_croissant_export.py) (4 tests) still passes — Phase 2 did not modify `logomesh/croissant_export.py` so its tests remained valid.

---

## Deviations from Phase 1 plan

### Deviation 1 — Replaced `validate_metadata_shape()` with inline `_validate_shape()`

**Reason:** The `validate_metadata_shape()` function in [logomesh/croissant_export.py:430-546](logomesh/croissant_export.py:430-546) hard-codes the Phase-2 28-column schema via `REQUIRED_COLUMNS` and only validates the FIRST recordSet (line 471 `first_record_set = record_set[0]`). It rejects any package whose fields differ from `artifact_id, run_artifact_path, model_id, ..., audit_hash` — incompatible with our Track F (19 fields) + Track G (24 fields) schema.

**Fix:** Authored an inline `_validate_shape()` function in `scripts/build_croissant_package.py` that performs Croissant-1.1-generic structural checks (required top-level keys, conformsTo URI inclusion, distribution items have valid sha256, recordSet entries have @id + name + description + field[], each field has required keys + valid source.fileObject.@id reference + source.extract.column, key references resolve to known field @ids).

**Impact:** None on output correctness. The Phase-2 validator's strict column-name check was a Phase-2-only invariant; the inline validator's check is spec-aligned (Croissant 1.1 §"RecordSets" line 870+). mlcroissant strict validation (Layer 2) confirms spec conformance.

### Deviation 2 — Strict-validation subprocess call

**Reason:** Initial implementation tried `[sys.executable, "-m", "mlcroissant", ...]` which failed because the `mlcroissant` package has no `__main__.py` ("`mlcroissant` is a package and cannot be directly executed").

**Fix:** Switched to `shutil.which("mlcroissant")` to locate the CLI entry point (`mlcroissant.exe` installed in `.venv/Scripts/` by pip), invoking it directly. Same approach as [logomesh/croissant_export.py:549-563](logomesh/croissant_export.py:549-563).

**Impact:** Validation now runs successfully; exit=0 confirmed.

### Non-deviation: NaN handling
Plan §1.3 noted "NaN → empty string" as the recommended approach. The build script implements this: Python `math.nan` → empty cell; column `dataType` stays `sc:Float`. Track F greedy condition's `mean_search_reward` (raw JSON has `NaN`) and `best_path` (raw JSON has `null`) both surface as empty cells. mlcroissant accepted this (Layer 2 exit=0).

---

## Final build state (Phase 2 output)

### `docs/dataset/` tree

```
docs/dataset/
├── MANIFEST.txt                                                       (4,583 bytes)
├── croissant.json                                                    (37,520 bytes, 917 lines)
└── data/
    ├── track_f_negcontrol.csv                                       (186,912 bytes, 1200 rows)
    ├── track_g_cartography_mcts.csv                                 (391,902 bytes, 770 rows)
    ├── uv.lock                                                      (269,591 bytes)
    └── raw/
        ├── _track_a_results_meta-llama_Llama-3.2-1B-Instruct.json    (32,486 bytes)
        ├── _track_a_results_meta-llama_Llama-3.2-3B-Instruct.json    (31,935 bytes)
        ├── _track_c_results_meta-llama_Llama-3.2-1B-Instruct.json   (587,860 bytes)
        ├── _track_c_results_meta-llama_Llama-3.2-3B-Instruct.json   (583,668 bytes)
        ├── _track_d_vram.json                                         (6,424 bytes)
        ├── _track_d_vram_3b.json                                      (6,505 bytes)
        ├── _track_f_results_meta-llama_Llama-3.2-1B-Instruct.json   (430,387 bytes)
        ├── _track_f_results_meta-llama_Llama-3.2-3B-Instruct.json   (408,502 bytes)
        ├── _track_g_c1_factual_recall.json                           (12,057 bytes)
        ├── _track_g_mcts_results_meta-llama_Llama-3.2-1B-Instruct.json (518,178 bytes)
        ├── _track_g_mcts_results_meta-llama_Llama-3.2-3B-Instruct.json (519,505 bytes)
        ├── _track_g_results_meta-llama_Llama-3.2-1B-Instruct.json  (2,403,941 bytes)
        └── _track_g_results_meta-llama_Llama-3.2-3B-Instruct.json  (2,413,015 bytes)
```

**Total: 8.5 MB.** Well within the 100 MB ceiling typically imposed by anonymous-hosting services (Phase 4 cleanup will verify anonymous.4open.science specifics).

### sha256s (excerpt; full list in `docs/dataset/MANIFEST.txt`)

- `docs/dataset/croissant.json` — `7afa8adb93be3b40d416ac3d369e32e166cacceaa529b2686e2a114732f039f9`
- `docs/dataset/data/track_f_negcontrol.csv` — `52d0ad62c4cc105e09d1f7264fc5a5ea3de28d79d4d56b8787b9b826ea8465a7`
- `docs/dataset/data/track_g_cartography_mcts.csv` — `230e1c06292412a1e9994b58b8f43db17b15b03797806022f75296bf815dc56e`

---

## Open items for Josh (Phase 3 → Phase 4)

### Status reminder (no change from Phase 1)
- **§7.6 line 368 "per-node" wording**: paper-claim drift; Session 2's domain.
- **§A.4 line 501 "BBH-LogicalDeduction-3" vs JSON "LogiQA"**: paper-claim drift; Session 2's domain.
- **§A.4 line 498 "~13 min" Track A wallclock vs JSON ~2 min**: paper-claim drift; Session 2's domain.

### New from Phase 2
- **License copyright fix applied** (LICENSE line 1: `Copyright (c) 2026 [Authors]`). Camera-ready de-anonymization couples paper `\author{...}` + LICENSE copyright in a single edit.
- **[pyproject.toml:8](pyproject.toml:8) description α+ vintage** (`"KV-Cache Inception — Reversible MCTS for Alignment Faking Detection"`): NOT updated by Phase 2 (not load-bearing for Croissant). Phase 4 cleanup territory.

---

## Phase 2 wall time

- Pre-build setup (LICENSE fix + deletions + mlcroissant install): 4 min
- Script authoring: 45 min
- Two build runs + two fixes (validator + strict-subprocess): 12 min
- Layer 3/4/5 cross-checks: 8 min
- This execution log: 12 min

**Actual total: ~80 min.** Plan budget was 110 min; came in under by Josh's 50%-pace heuristic.

---

## Phase 2 verdict

✅ **Complete.** All validation layers pass:
- Layer 1 (inline shape validator): no errors.
- Layer 2 (mlcroissant strict): exit=0.
- Layer 3 (paper-claim cross-check): 3/4 verified; §A.5 URL slug deferred to Phase 4 (1 known drift item flagged for Session 2 = §7.6 "per-node" wording).
- Layer 4 (raw-JSON spot-checks): Track F item 0 1B + Track G C1 item 0 1B verbatim matches.
- Layer 5 (pytest hygiene): 179/179 passing.

Ready for Phase 3 commit (no push without Josh's explicit ask).
