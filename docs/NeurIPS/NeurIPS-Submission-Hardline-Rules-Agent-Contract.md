# NeurIPS Submission Hardline Rules (Agent Contract)

Status: ACTIVE
Version: 2026-04-15
Intended use: Persistent, agent-facing constraints for all planning and implementation work toward NeurIPS submission readiness.

## Reference Source (Must Be Used)

This contract is grounded in the following project-specific submission strategy document:

- docs/NeurIPS/NeurIPS Submission Guide (KV-CAche-Inception).md

Agents may use that file for strategic context (track selection rationale, experiment framing,
ethics posture). **However, it is an AI-generated document that contains confirmed errors
(wrong bibliography URLs, unverified platform size claims). It must NOT be treated as
authoritative for specific formatting requirements or submission mechanics.**

For authoritative requirements, the following sources take precedence in this order:

1. **Live NeurIPS 2026 website** (fetch at time of use):
   - E&D track: https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets
   - Main track / general: https://neurips.cc/Conferences/2026/CallForPapers
   - Main Track Handbook: https://neurips.cc/Conferences/2026/MainTrackHandbook

2. **Downloaded official template** (already in repo):
   - `docs/NeurIPS/neurips_2026.sty` — the actual NeurIPS 2026 LaTeX style file
   - Source ZIP: https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip

3. **Main Track Handbook (saved copy)**:
   - `docs/NeurIPS/Main Track Handbook 2026_files/`

4. **KV-CAche-Inception guide** — for strategic framing only, not for formatting facts.

## Scope

This contract applies to all work that affects:

1. Paper content and formatting.
2. Dataset and artifact packaging.
3. Croissant/RAI metadata and validation.
4. Reproducibility evidence and release readiness.
5. Submission workflow, deadlines, and OpenReview integrity.

## Hardline Rules (Non-Negotiable)

### 1. Track and Positioning Lock

1. Submission target is NeurIPS 2026 Evaluations and Datasets track.
2. Work must be framed as evaluation-method and dataset-governance contribution, not a generic algorithm-only paper.
3. Do not pivot to Position Paper framing.

### 2. Formatting and PDF Assembly Lock

1. Use official NeurIPS 2026 style package for E&D (`neurips_2026` with E&D option) for submission build.
2. Do not use custom geometry, margin hacks, or font-size hacks.
3. Maintain strict 9-page main content limit (references and appendices excluded per NeurIPS rules).
4. Final PDF order must be:
- Main content
- References
- Optional appendices
- Mandatory NeurIPS checklist

### 3. Double-Blind Integrity Lock

1. No deanonymizing author metadata in submission artifacts.
2. No identifiable private links in manuscript during anonymous phase.
3. Self-citations must preserve anonymity conventions (third-person neutral style).

### 4. Croissant and RAI Lock

1. All released evidence artifacts must support dual form:
- Raw runtime JSON
- Croissant 1.1 JSON-LD metadata (+ RAI fields)
2. Metadata must include Croissant and RAI conformance URIs.
3. Dataset-level required metadata must include `datePublished` and other required Schema.org/Croissant fields.
4. Record identity must be explicit (`recordSet.key`) and tied to stable record identifiers.
5. Distribution hashes must be valid SHA-256 hex values.
6. Field extraction mappings must be explicit and resolver-safe.

### 5. Data Hosting and Access Control Lock

1. Hosting platform must support review-time access and controlled release posture appropriate for dual-use risk.
2. Safety-sensitive artifacts (interventions, attack traces, latent steering evidence) must not be treated as unrestricted public dumps by default.
3. Release model must include explicit intended-use and non-intended-use language.

### 6. Ethics and Dual-Use Lock

1. Every dataset/package update must preserve defensible dual-use mitigation language.
2. Do not remove or weaken safeguards in ethics, social impact, or release-maintenance fields.
3. Red-team utility must stay aligned to defensive auditing and scientific evaluation.

### 7. Reproducibility and Evidence Lock

1. Runtime outputs must retain reproducibility provenance:
- seed
- git SHA
- git dirty state
- command line
- timestamps
2. Croissant export path must remain functional from run artifacts.
3. Validation must include repository tests before commit.

### 8. Validation Gate Lock

1. Minimum pre-commit gate: `uv run pytest tests/ -v` must pass.
2. Croissant export tests must remain green.
3. Internal metadata-shape validation is mandatory.
4. External strict validation should be run when `mlcroissant` CLI is available; if unavailable, blockage must be documented in logs and guidance docs.

### 9. Visualization Accessibility Lock

1. No color-only encoding for critical distinctions in figures.
2. Prefer colorblind-safe palettes and high contrast defaults.
3. Figure text must remain legible after two-column paper scaling.

### 10. Deadline and Operations Lock

1. All planning must treat AoE deadlines as hard boundaries.
2. Internal deadlines should be set earlier than AoE to avoid platform congestion risk.
3. OpenReview profile and authorship constraints must be treated as release blockers.

## Agent Execution Checklist (Run Before Major Changes)

1. Confirm alignment to E&D track framing.
2. Confirm no violation of formatting/anonymity constraints.
3. Confirm Croissant + RAI integrity for any data artifact changes.
4. Run tests and capture outcomes.
5. Update session log with what changed, what was validated, and what remains blocked.
6. Update handoff/index docs when new policy or review documents are added.

## Agent Execution Checklist (Run Before Final Submission Readiness Signoff)

1. Manuscript structure and checklist placement verified.
2. Croissant metadata validated (internal + strict external if available).
3. Dataset hosting/access controls verified against dual-use posture.
4. Ethics, limitations, and intended-use language present and consistent across paper and metadata.
5. Reproducibility fields present in released artifacts.
6. All tests green and logged.

## Change-Control Rule

If any future work conflicts with this contract, agents must:

1. Stop and flag the conflict explicitly.
2. Propose compliant alternatives.
3. Avoid implementing conflicting changes until a human maintainer approves a policy override.

## Authority Statement

This document is both:

1. A supplementary planning document for NeurIPS submission operations.
2. A persistent reminder contract for agents to maintain submission-critical discipline across coding, documentation, and release workflows.
