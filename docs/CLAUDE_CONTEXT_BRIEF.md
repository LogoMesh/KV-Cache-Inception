# Claude Context Brief

**Last updated:** 2026-04-18 (session 2)
**Owner:** Josh / Claude sessions
**Purpose:** Single handoff index. Read this file first, then go directly to Priority 1.

---

## Canonical Files

| Role | Path |
|---|---|
| Session rules, phases, hard constraints | [CLAUDE.md](../CLAUDE.md) |
| Canonical paper draft | [docs/NeurIPS/04.17.2026-NeurIPS-Research-Proposal.tex](NeurIPS/04.17.2026-NeurIPS-Research-Proposal.tex) |

---

## Document Index

| Priority | Document | Purpose |
|---|---|---|
| 1 | [experiment-readiness-and-paper-revision.md](NeurIPS/experiment-readiness-and-paper-revision.md) | Pre-flight gates + paper unlock text for all 5 experiments. Read before any H100 work. |
| 2 | [neurips-2026-submission-guidelines.md](NeurIPS/neurips-2026-submission-guidelines.md) | Verified NeurIPS 2026 E&D submission rules (deadlines, formatting, anonymization, checklist). |
| 3 | [neurips-2026-data-requirements.md](NeurIPS/neurips-2026-data-requirements.md) | Croissant 1.1/RAI compliance state + hosting decision + data roadmap. |
| 4 | [paper-revision-claims-audit.md](NeurIPS/paper-revision-claims-audit.md) | Paper claims inventory and decision record. May overlap with Priority 1 — review for consolidation. |
| 5 | [NeurIPS-Submission-Hardline-Rules-Agent-Contract.md](NeurIPS/NeurIPS-Submission-Hardline-Rules-Agent-Contract.md) | Non-negotiable submission rules binding all agents. |
| 6 | [2026-04-18_Meeting-Briefing.md](NeurIPS/2026-04-18_Meeting-Briefing.md) | Apr 18 team meeting agenda: OE scope decision, experiment priority table, operational items. |
| 7 | [docs/logs/2026-04-18_session-log.md](logs/2026-04-18_session-log.md) | Most recent session record. |

---

## Current State

- **Tests:** 139/139 green (as of commit `b7944bc`)
- **Phase 3 gate:** passed 2026-04-16 on `meta-llama/Llama-3.2-1B-Instruct`
- **ρ_R fix:** done — commit `4bc8fa9`; thresholds in `classify()` need H100 recalibration (`ClassifyConfig` ready)
- **Paper:** `04.17.2026-NeurIPS-Research-Proposal.tex` — no `\reviewedit{}` remaining; related work rewritten with 4 new citations; capability claims hedged to future tense
- **Croissant:** mlcroissant validates clean (1 non-blocking warning: `citeAs` pending OpenReview ID)

**Open blockers (nothing can proceed without these):**
1. H100 access from Tianyu's lab — required for all Phase B experiments
2. OpenReview profiles verified for all 4 co-authors — **critical, 2-week moderation window**
3. Saturday Apr 18 meeting: Priority table in `experiment-readiness-and-paper-revision.md` must be filled before scoping paper claims

---

## Update Protocol

When creating or updating docs:
1. Add or update the entry in the Document Index above.
2. Update **Last updated** at the top.
3. Append a matching entry to the current date session log.
