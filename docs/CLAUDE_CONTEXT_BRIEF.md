# Claude Context Brief

Last updated: 2026-04-11 (Phase 2 complete)
Owner: GitHub Copilot session
Purpose: Single handoff index for Claude so you only need one filepath.

## Start Here

Primary handoff filepath:
- [docs/CLAUDE_CONTEXT_BRIEF.md](CLAUDE_CONTEXT_BRIEF.md)

If Claude only reads one file first, use this one.

## Document Index

| Priority | Document | Why it matters | Status |
|---|---|---|---|
| 1 | [docs/logs/2026-04-11_session-log.md](logs/2026-04-11_session-log.md) | Chronological record of all session changes and validation outcomes | Active |
| 2 | [docs/reviews/local-to-h100-transition-audit-2026-04-11.md](reviews/local-to-h100-transition-audit-2026-04-11.md) | Readiness verdict and migration runbook (C1/C2 blockers now resolved) | Active |
| 3 | [docs/reviews/phase-a-gaps-2026-04-11.md](reviews/phase-a-gaps-2026-04-11.md) | Deep verification of architecture vs paper-track gaps | Active |

## Canonical Project Context

| Document | Role |
|---|---|
| [CLAUDE.md](../CLAUDE.md) | Session rules, phases, hard constraints, and required commands |
| [docs/NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex](NeurIPS/04.02.2026-NeurIPS-Research-Proposal.tex) | Canonical paper draft and algorithm targets |

## Current Snapshot

- **Phase 2 is complete.** Core paper algorithm modules are implemented and all 117 tests pass.
- Audit blockers C1 (null plugin crash) and C2 (HF model ID + MPS device) are resolved.
- New modules: `telemetry_matrix.py`, `orthogonal_escape.py`, `kv_mcts.py`; extended: `hneuron_monitor.py`, `whitebox.py`, `local_model.py`.
- New scripts: `run_kv_mcts.py`, `measure_lipschitz_drift.py`.
- Next: Phase 3 — gate on `run_kv_mcts.py` smoke test with real model, then 5 experiment scripts.

## Update Protocol (Keep This Stable)

When any assistant creates or updates docs:
1. Add or update the document in the Document Index table above.
2. Update Last updated at the top of this file.
3. Append a matching entry to [docs/logs/2026-04-11_session-log.md](logs/2026-04-11_session-log.md) (or the current date log).
4. Keep this file path unchanged so handoffs remain one-click.

## Copy-Ready Handoff Prompt

Use this exact starting message with Claude:

"Start from docs/CLAUDE_CONTEXT_BRIEF.md. Use it as the canonical index for this workspace handoff, then continue implementation from the highest-priority active blocker."
