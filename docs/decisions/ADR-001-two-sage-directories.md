# ADR-001: Two sage/ Directories — Canonical is security_arena/sage/

**Date:** 2026-03-20
**Status:** Accepted

## Context

The repo contains two `sage/` packages:
- `scenarios/security_arena/sage/`
- `scenarios/security_arena/agents/attacker/sage/`

Both contain the same modules (`ablation.py`, `search_policy.py`, etc.) but with different defaults. Discovered when tests failed: `security_arena/sage/` is the one added to sys.path by `agent.py` at runtime and by all test files.

## Decision

Treat `scenarios/security_arena/sage/` as canonical. Apply all V3 changes there first. Mirror changes to `attacker/sage/` for consistency but do not assume it is the active copy.

The differing defaults (`enable_self_eval=False` in attacker copy, `True` in canonical) are intentional — the attacker copy is optimized for gpt-oss-20b competition use. Do not "align" them.

## Consequences

- Every new file added to `sage/` must be created in both directories.
- If tests fail with `ModuleNotFoundError: No module named 'sage.X'`, the new file was created only in `attacker/sage/`, not in the canonical `security_arena/sage/`.
- Future refactor option: symlink `attacker/sage/` → `security_arena/sage/`, but this risks breaking the intentional defaults divergence.
