# Claude Context Brief

Last updated: 2026-04-15 (Croissant deep-dive guidance + 135-test validation)
Owner: GitHub Copilot session
Purpose: Single handoff index for Claude so you only need one filepath.

## Start Here

Primary handoff filepath:
- [docs/CLAUDE_CONTEXT_BRIEF.md](CLAUDE_CONTEXT_BRIEF.md)

If Claude only reads one file first, use this one.

## Document Index

| Priority | Document | Why it matters | Status |
|---|---|---|---|
| 0 | [docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal-Verification-Overlay.tex](NeurIPS/04.02.2026-NeurIPS-Research-Proposal-Verification-Overlay.tex) | Read-along working copy with in-document empirical verification callouts mapped to concrete logs and filepaths | Active (non-canonical) |
| 1 | [docs/logs/2026-04-15_session-log.md](logs/2026-04-15_session-log.md) | Current session: experiment-prep history plus completed Croissant alignment implementation (exporter module, schema upgrade, runtime wiring, and validation outcomes) | Active |
| 2 | [docs/reviews/croissant-implementation-guidance-2026-04-15.md](reviews/croissant-implementation-guidance-2026-04-15.md) | Standards-grounded Croissant 1.1/RAI implementation contract, validator expectations, and release checklist for runtime artifact export | Active |
| 3 | [docs/logs/2026-04-14_session-log.md](logs/2026-04-14_session-log.md) | Deep runtime audit (live Qwen gate attempts, DynamicCache findings, calibration timing) | Active |
| 4 | [docs/logs/2026-04-11_session-log.md](logs/2026-04-11_session-log.md) | Chronological record of Phase 2 implementation work and earlier validation outcomes | Active |
| 5 | [docs/reviews/local-to-h100-transition-audit-2026-04-11.md](reviews/local-to-h100-transition-audit-2026-04-11.md) | Readiness verdict and migration runbook (historical baseline) | Active |
| 6 | [docs/reviews/phase-a-gaps-2026-04-11.md](reviews/phase-a-gaps-2026-04-11.md) | Deep verification of architecture vs paper-track gaps (historical baseline) | Active |

## Canonical Project Context

| Document | Role |
|---|---|
| [CLAUDE.md](../CLAUDE.md) | Session rules, phases, hard constraints, and required commands |
| [docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex](NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex) | Canonical paper draft and algorithm targets |

## Current Snapshot

- Core Phase 2 modules remain implemented and current full suite is green at **135 passed**.
- Step 1 (runner output encoding reliability) is complete and baseline-committed (`d794ad2`).
- Step 2 reproducibility controls are now implemented in `scripts/run_kv_mcts.py`:
	- `--seed` CLI support and Python/NumPy/Torch seeding
	- persisted JSON run metadata including `seed`, `git_sha`, `git_dirty`, `model`, and `config`
- Stable seeded offline verification now completed:
	- Artifact: `tmp/mcts_qwen_offline_n1_step2check.json`
	- Metadata verified (`seed=1337`, `git_sha` present, `git_dirty=False`, timestamps and command present)
- Step 3 hygiene completed by ignoring transient artifacts (`tmp/`) via commit `ee31085`.
- Croissant integration commits are pushed:
	- `ae0ee65` (submodule integration)
	- `bd48285` (export pipeline + tests)
	- `a531b29` (docs/paper updates)
- Croissant alignment implementation is now active and integrated:
	- New module: `logomesh/croissant_export.py`
	- New CLI: `scripts/export_kv_mcts_to_croissant.py`
	- Runtime integration in `scripts/run_kv_mcts.py` for optional Croissant package emission
	- Schema template upgraded in `docs/dataset/croissant_schema_stub.json`
	- Exporter unit tests added in `tests/test_croissant_export.py`
	- Standards hardening added: `datePublished`, `inLanguage`, `keywords`, `publisher`, `sdVersion`, `recordSet.key`, sha256/key mapping validation
	- Deep guidance document added: `docs/reviews/croissant-implementation-guidance-2026-04-15.md`
	- Strict external validation currently blocked in this environment until `mlcroissant` CLI is installed in PATH
- Croissant submodule is present at `external/croissant` as reference material.
- DynamicCache compatibility is validated on cached Qwen runtime:
	- Mutability gate now returns `gate_passed: true`.
	- Minimal offline `run_kv_mcts.py` run completes through calibration + MCTS and writes output JSON.
- Calibration overhead remains substantial, but it is now a performance concern rather than a hard runtime blocker in the validated path.
- Next immediate work: commit/push the standards-hardening continuation batch, then proceed to Phase 3 experiment infrastructure (5 experiment scripts, Procrustes/evaluation stack) plus larger-run Croissant export regression checks.

## Update Protocol (Keep This Stable)

When any assistant creates or updates docs:
1. Add or update the document in the Document Index table above.
2. Update Last updated at the top of this file.
3. Append a matching entry to the current date log (for example [docs/logs/2026-04-14_session-log.md](logs/2026-04-14_session-log.md)).
4. Keep this file path unchanged so handoffs remain one-click.

## Copy-Ready Handoff Prompt

Use this exact starting message with Claude:

"Start from docs/CLAUDE_CONTEXT_BRIEF.md. Use it as the canonical index for this workspace handoff, then continue implementation from the highest-priority active blocker."
Touch marker: 2026-04-15 croissant-guidance-check.
