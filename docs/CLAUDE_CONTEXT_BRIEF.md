# Claude Context Brief

Last updated: 2026-04-15 (baseline commit after step 1; step 2 reproducibility controls implemented)
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
| 1 | [docs/logs/2026-04-15_session-log.md](logs/2026-04-15_session-log.md) | Current session: experiment-prep steps tracking (step 1 reliability fix, baseline commit `d794ad2`, step 2 reproducibility implementation + validation) | Active |
| 2 | [docs/logs/2026-04-14_session-log.md](logs/2026-04-14_session-log.md) | Deep runtime audit (live Qwen gate attempts, DynamicCache findings, calibration timing) | Active |
| 3 | [docs/logs/2026-04-11_session-log.md](logs/2026-04-11_session-log.md) | Chronological record of Phase 2 implementation work and earlier validation outcomes | Active |
| 4 | [docs/reviews/local-to-h100-transition-audit-2026-04-11.md](reviews/local-to-h100-transition-audit-2026-04-11.md) | Readiness verdict and migration runbook (historical baseline) | Active |
| 5 | [docs/reviews/phase-a-gaps-2026-04-11.md](reviews/phase-a-gaps-2026-04-11.md) | Deep verification of architecture vs paper-track gaps (historical baseline) | Active |

## Canonical Project Context

| Document | Role |
|---|---|
| [CLAUDE.md](../CLAUDE.md) | Session rules, phases, hard constraints, and required commands |
| [docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex](NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex) | Canonical paper draft and algorithm targets |

## Current Snapshot

- Core Phase 2 modules are implemented and current full suite is green at **130 passed**.
- Step 1 (runner output encoding reliability) is complete and baseline-committed (`d794ad2`).
- Step 2 reproducibility controls are now implemented in `scripts/run_kv_mcts.py`:
	- `--seed` CLI support and Python/NumPy/Torch seeding
	- persisted JSON run metadata including `seed`, `git_sha`, `git_dirty`, `model`, and `config`
- DynamicCache compatibility is validated on cached Qwen runtime:
	- Mutability gate now returns `gate_passed: true`.
	- Minimal offline `run_kv_mcts.py` run completes through calibration + MCTS and writes output JSON.
- Step 2 minimal offline runtime re-checks in this session encountered terminal cancellation during calibration (exit 130), so Step 2 runtime artifact generation remains to be re-run in a stable terminal pass.
- Calibration overhead remains substantial, but it is now a performance concern rather than a hard runtime blocker in the validated path.
- Next immediate work: Phase 3 experiment infrastructure (5 experiment scripts, Procrustes/evaluation stack), plus calibration/runtime optimization.

## Update Protocol (Keep This Stable)

When any assistant creates or updates docs:
1. Add or update the document in the Document Index table above.
2. Update Last updated at the top of this file.
3. Append a matching entry to the current date log (for example [docs/logs/2026-04-14_session-log.md](logs/2026-04-14_session-log.md)).
4. Keep this file path unchanged so handoffs remain one-click.

## Copy-Ready Handoff Prompt

Use this exact starting message with Claude:

"Start from docs/CLAUDE_CONTEXT_BRIEF.md. Use it as the canonical index for this workspace handoff, then continue implementation from the highest-priority active blocker."
Touch marker: 2026-04-14 commit-date-check.
