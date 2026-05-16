# Paranoid Hallucination Audit — Final Claim-by-Claim Verification

**Created:** 2026-05-16 (Day-11 EOD).
**Runs:** Day-11 EOD or Day-12 AM (before parallel-review-plan firing).
**Mode:** Paranoid. Trust nothing in the paper without primary-source verification.
**Output target:** `docs/logs/<run-date>_claims-audit-matrix.md` (matrix of every claim, with verdicts).
**Why this exists:** Josh's feedback-memory rule applied at full scope — "Source every paper claim from its primary artifact, not from a summary/log/compaction citation of it." Final pre-Day-12 hallucination pass before paper goes to Tianyu + ARR.

---

## Copy-paste prompt for a fresh Claude session

```
PARANOID HALLUCINATION AUDIT — final pre-flight pass on the LogoMesh
NeurIPS / EMNLP submission. Your job: verify every single claim in the
paper against primary sources. Default assumption: any claim might be
hallucinated, mis-cited, or numerically wrong. You prove each claim is
correct, or you flag it.

Do not trust your memory of the project. Do not trust track reports as
primary (they are summaries of raw JSONs). Do not trust code docstrings
or comments. Read the source itself, every time.

INPUTS (all under the worktree at .claude/worktrees/elegant-shockley-b745dc/):
- Paper TeX: docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex
  (635 lines; commit 8085545, post-Day-11)
- Downloaded papers: docs/NeurIPS/Citations/ (20 PDFs; arXiv-IDed +
  named-format)
- Repo code: logomesh/*.py (kv_mcts.py, local_model.py, hneuron_monitor.py,
  whitebox.py, telemetry_matrix.py, orthogonal_escape.py, etc.)
- Driver scripts: scripts/diagnose_track_*.py + scripts/measure_*.py
- Raw experiment outputs: scripts/_track_*_results_*.json (Tracks A, C, D,
  F, G at 1B + 3B)
- Track reports: docs/logs/2026-05-*_track-*-report.md — SECONDARY ONLY;
  use only as a map to find the raw JSON. Never quote a track report as
  primary verification.

OUTPUT: write to docs/logs/<today's-date>_claims-audit-matrix.md as you
go (write per-section batches to keep context lean). Schema below.

----------------------------------------------------------------------
CLAIM TYPES + VERIFICATION RULES
----------------------------------------------------------------------

CITE — claim about what an external paper says
  Example: "Greenblatt 2024 demonstrated frontier-scale strategic
  compliance" → \cite{greenblatt2024alignment}
  Verification: (a) map cite key to PDF filename in Citations/; (b) open
  the PDF; (c) find the section/page that supports our claim; (d) quote
  verbatim in the Notes column. If the cited paper does NOT say what we
  claim, mark DIVERGENT or SHOWSTOPPER. If the PDF isn't in Citations/,
  mark UNVERIFIABLE.

EXP-NUM — numerical claim about an experimental result
  Example: "paired McNemar p = 1.0 at both 1B and 3B" (abstract line 55)
  Primary source: scripts/_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json
  + scripts/_track_f_results_meta-llama_Llama-3.2-3B-Instruct.json
  Verification: open the JSONs, locate the relevant field, verify the
  number matches within rounding (cite the field path in Notes). Track
  reports are NEVER primary for numbers — open the JSON.

EXP-METHOD — methodological claim about how an experiment was run
  Example: "depth-3 MCTS with 27 path samples from {0.1, 0.5, 1.0}"
  Primary source: scripts/diagnose_track_*.py
  Verification: read the driver script, confirm parameter values and the
  procedure. If the script does NOT enumerate all 27 paths (e.g., uses
  UCB1 sampling that may skip some), that's DIVERGENT from the paper's
  exhaustive-enumeration claim.

CODE — claim about what the code does
  Example: "the FP32 accumulator absorbs bf16 quantization error" (§4.2)
  Primary source: logomesh/kv_mcts.py (FP32Accumulator class)
  Verification: read the class. Verify the algorithm matches the paper's
  equation. Verify any precision-management claims are actually true at
  the dtype level. Verify any "absorbs"/"prevents"/"guarantees" claims
  by reading the math.

THEOREM — theoretical claim
  Example: Theorem 1 zero-drift reversibility
  Primary source: the proof at §C (Appendix sec:appendix-proof)
  Verification: read the proof, verify it actually establishes the
  stated bound. If the proof skips steps or is hand-wavy, mark IMPRECISE.
  If the bound in the proof differs from the bound in the theorem
  statement, mark DIVERGENT.

META — internal-consistency claim about the paper itself
  Example: §7.3 Limitations item 1: "we make no claim about MoE
  architectures (e.g., gpt-oss-20b)"
  Primary source: grep the rest of the paper for contradicting claims.
  Verification: if any body claim contradicts the limitations / scope
  statement, mark DIVERGENT. Includes cross-section consistency: does
  the abstract match §1.1? Do §3 Hypotheses align with §5 Experiments?

OTHER — anything that doesn't fit the above categories
  Use sparingly. Most claims fit one of the six types. If you need OTHER,
  explain why in Notes.

----------------------------------------------------------------------
SEVERITY LEGEND
----------------------------------------------------------------------

SHOWSTOPPER — claim is FALSE in primary source. Examples:
  - Paper says "p = 1.0" but JSON has "p = 0.85"
  - Paper says Bailey 2024 reports X but the cited PDF says NOT X
  - Theorem statement bound differs from the proof's actual bound
  - Code does NOT implement what the equation claims
  ESCALATE TO JOSH IMMEDIATELY. Do not continue audit until Josh decides.

DIVERGENT — paper claim differs from primary source materially (>5%
  numerical, or methodological). MAJOR. Patch before submission.

IMPRECISE — paper claim approximately correct but rounded/simplified
  beyond honest hedging, or proof skips steps. MODERATE. Patch if time.

UNVERIFIABLE — primary source not accessible (cited paper not in
  Citations/; raw JSON missing; etc.). MAJOR. The paper makes an
  undefended claim.

CONFIRMED — claim matches primary source within rounding tolerance. OK.

----------------------------------------------------------------------
MATRIX SCHEMA
----------------------------------------------------------------------

Write to the output matrix file as you go. Schema:

  # Claims Audit Matrix — Final Hallucination Pre-Flight

  **Audit date:** <today YYYY-MM-DD>
  **Paper commit:** 8085545
  **Auditor:** Claude (paranoid mode), fresh session

  ## Showstoppers
  [List any SHOWSTOPPER findings. If empty, write "None found."]

  ## Severity counts
  [Update at end: SHOWSTOPPER: N, DIVERGENT: N, IMPRECISE: N,
   UNVERIFIABLE: N, CONFIRMED: N]

  ## Cite-to-PDF mapping (built at start of audit)
  [Build the map of \cite{key} → Citations/<filename> here, with notes
   on any cite key whose PDF is not in Citations/.]

  ## Abstract claims
  | ID | TeX line | Claim (verbatim, ≤120 chars) | Type | Primary source | Verdict | Notes |
  |---|---|---|---|---|---|---|
  | A-001 | 55 | "validated to machine precision over 200 apply--revert cycles on Llama 3.2-1B and 3B" | EXP-NUM | scripts/measure_lipschitz_drift.py + _track_d_vram.json | TBD | TBD |
  ...

  ## §1 Introduction claims
  | ID | TeX line | Claim | Type | Primary source | Verdict | Notes |
  ...

  ## §1.1 Contributions claims
  ...

  [Continue per section through §7.5 + §A through §G appendix]

  ## Cross-section findings
  [After per-section pass, run these checks:
   1. Does the abstract match §1.1 Contributions in numbers and framing?
   2. Do §3.2 Hypotheses match §5 Experiments' claimed evidence scope?
   3. Does §7.3 Limitations contradict any body claim?
   4. Are forward references resolved (\ref{} cite valid label)?
   5. Do appendix claims align with body claims?]

  ## Summary + recommendations
  [Total claims, severity breakdown, top 5 recommended patches.]

----------------------------------------------------------------------
WORKFLOW
----------------------------------------------------------------------

PHASE 0 (setup, ~15 min):
1. Build the cite-to-PDF mapping. Open the bibliography in the TeX
   (lines ~376-462). For each \bibitem{key}, identify the cited paper
   and find its PDF in Citations/. Map arXiv IDs to filenames (e.g.,
   bibitem says "arXiv:2401.05566" → file "arXiv-2401.05566v3"). For
   bibitems without arXiv IDs (greenblatt2024alignment is Anthropic
   Research only; bailey2024obfuscated is NeurIPS 2024 only; rimsky2024
   is ACL 2024; meta2024llama3_2_aup is URL-only), search Citations/
   for matching titles. List any cite keys whose PDFs are MISSING from
   Citations/ — these are immediate UNVERIFIABLE candidates.
2. Write the cite-to-PDF mapping to the matrix file.

PHASE 1 (per-section claim enumeration + verification, ~30-60 min/section):
Work section by section in this order:
  - Abstract (lines 54-56)
  - §1 Introduction (lines 61-71)
  - §1.1 Contributions (lines 73-82)
  - §2.1 Alignment Faking (lines 89-92)
  - §2.2 RepE / Activation Steering (lines 94-99)
  - §3 Problem Formulation (lines 106-141 incl. §3.1, §3.2 def + 2 hyps)
  - §4 Proposed Method (lines 146-167 incl. §4.2 + FP32 accum + Reward)
  - §5 Experimental Design (lines 171-275; abstract setup + Exp 1 +
     Table 1 + Exp 2 + Table 2 + Figure 1 + §5.5 Eval Metrics)
  - §6 Theoretical Analysis (lines 278-321 incl. Theorem 1 + Prop 1
     + worked example)
  - §7 Discussion (lines 324-371 incl. §7.1, §7.2, §7.3, §7.4, §7.5)
  - References (lines 376-462; verify each bibitem's arXiv/venue/year
     against the actual PDF metadata)
  - §A Reproducibility (lines 465-493)
  - §B-§G appendices (lines 495-633)

For each section:
  (a) Read the section in full. Note every claim — even subsidiary
      claims in parentheticals or footnotes.
  (b) For each claim: categorize, identify primary source, verify,
      record verdict + Notes (cite file:line or PDF page).
  (c) Write the section's matrix rows to the output file.
  (d) If ANY row is SHOWSTOPPER: STOP. Write to top of matrix's
      Showstoppers section. Report to Josh. Wait for Josh.
  (e) Drop the section's claims from in-context memory and move on.

PHASE 2 (cross-section consistency, ~30 min):
After all sections done, run the 5 cross-section checks in the schema
template. Write findings.

PHASE 3 (summary, ~15 min):
Severity counts, top 5 recommended patches, fitness verdict.

----------------------------------------------------------------------
HARD RULES
----------------------------------------------------------------------

1. NEVER trust memory. If your memory says "Track F reports p=1.0",
   that's not verification. Open the JSON.
2. NEVER trust track reports as primary. They are summaries; JSON is
   primary.
3. NEVER trust code comments / docstrings. "# implements Eq. X"
   does not prove anything. Read the code itself.
4. NEVER paraphrase a citation. If paper says "Bailey 2024 reports X",
   open the Bailey PDF and find the exact passage that says X. Quote it
   in Notes. If you cannot find the exact passage, flag UNVERIFIABLE
   (even if you "remember" the paper roughly says X).
5. Surface UNVERIFIABLE explicitly. If a cited paper isn't in Citations/,
   that's a MAJOR-severity gap — the paper makes an undefended claim
   that won't pass adversarial review.
6. Cross-check numerical rounding. Paper "3.04--3.06×" + JSON "3.0397"
   and "3.0571" = CONFIRMED. Paper "3.04--3.06×" + JSON "2.85" and
   "3.42" = DIVERGENT.
7. Notes column MUST cite primary source by file:line or PDF:page.
   "verified at scripts/_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json
   key 'mcnemar_p_value' = 1.0" is good. "verified" alone is not.
8. Do NOT commit anything unless Josh explicitly asks. Write the matrix
   file; leave it uncommitted.
9. Do NOT modify the paper TeX. This audit is READ-ONLY on the paper.
10. Do NOT fire the Day-12 parallel-review axes from this session. This
    is a SEPARATE pre-flight pass; the Day-12 orchestrator does that.

----------------------------------------------------------------------
WORKED VERIFICATION EXAMPLES
----------------------------------------------------------------------

EXAMPLE 1 — EXP-NUM verification:
  Paper claim (abstract, ~line 55): "$\Delta(\text{entropy}-\text{random}) = -0.5$pp at 1B and $+0.5$pp at 3B (paired McNemar $p = 1.0$ both scales)"
  Type: EXP-NUM
  Primary source: scripts/_track_f_results_meta-llama_Llama-3.2-1B-Instruct.json + scripts/_track_f_results_meta-llama_Llama-3.2-3B-Instruct.json
  Workflow:
    - Open 1B JSON. Find the McNemar p-value field. Likely path:
      results["pairwise_e_vs_r"]["mcnemar_p_value"] (verify actual path).
    - Verify the value is 1.0 (or very close).
    - Find the accuracy gap: E_accuracy - R_accuracy.
    - Verify the 1B difference is -0.005 (or -0.5pp).
    - Repeat for 3B JSON; verify +0.005 (or +0.5pp).
    - Verdict CONFIRMED if both match.
  Notes example: "1B: results.pairwise_e_vs_r.mcnemar_p_value = 1.0;
  E.accuracy - R.accuracy = 0.275 - 0.280 = -0.005. 3B: same field = 1.0;
  0.820 - 0.815 = +0.005. CONFIRMED."

EXAMPLE 2 — CITE verification:
  Paper claim (§2.2 line ~99): "Bailey et al.~\cite{bailey2024obfuscated}
  ... report a tentative negative result for that regime: near-orthogonality
  and behavior preservation cannot be jointly achieved..."
  Type: CITE
  Primary source: Citations/bailey2024obfuscated.pdf or similar (look for
  the NeurIPS 2024 paper "Obfuscated Activations Bypass LLM Latent-Space
  Defenses"). If filename is unclear, grep the cite-to-PDF mapping built
  in Phase 0.
  Workflow:
    - Open the Bailey PDF.
    - Search for "tentative negative result" or "near-orthogonality" or
      "behavior preservation". Find the section/passage that supports
      our claim.
    - If Bailey says something materially different (e.g., a POSITIVE
      result, or qualified the negative result differently), mark
      DIVERGENT or SHOWSTOPPER.
    - If the passage matches our paraphrase, mark CONFIRMED + quote it
      in Notes.
  Notes example: "Bailey 2024 §5.2 page 7: 'we find that achieving
  cosine similarity below 0.05 requires KL divergence above [threshold]
  to the model's natural output, producing incoherent generations.'
  Matches our paraphrase. CONFIRMED."

EXAMPLE 3 — CODE verification:
  Paper claim (§4.2 line ~158): Eq fp32_accumulator states "$\mathbf{A}^{(\ell)}
  \leftarrow \mathbf{A}^{(\ell)} + \alpha \cdot \mathbf{d}^{(\ell)}$ (FP32),
  $\mathbf{K}_t^{(\ell)} \leftarrow \mathbf{K}_{\text{base},t}^{(\ell)} +
  \text{cast}_{\text{bf16}}(\mathbf{A}^{(\ell)})$"
  Type: CODE
  Primary source: logomesh/kv_mcts.py — FP32Accumulator class apply() +
  rollback() methods
  Workflow:
    - Read FP32Accumulator class.
    - Verify the apply() method:
        (i) accumulator is FP32 dtype (torch.float32)
        (ii) cache reconstruction casts FROM fp32 TO bf16 (or
             whatever the base cache dtype is — fp16 per §A.2)
        (iii) the addition happens in FP32, not silently in bf16
    - If the code silently upcasts/downcasts at the wrong point, mark
      DIVERGENT or SHOWSTOPPER (this is a code-vs-paper semantic
      divergence).
  Notes example: "logomesh/kv_mcts.py lines 78-92 (FP32Accumulator):
  self.accumulator = torch.zeros(..., dtype=torch.float32) [verifies
  FP32]; apply(): self.accumulator.add_(alpha * delta) [FP32 add];
  read: cache = base + self.accumulator.to(cache.dtype) [cast on read].
  Matches Eq fp32_accumulator. CONFIRMED."

EXAMPLE 4 — THEOREM verification:
  Paper claim (§6.1 Theorem 1): "$\|\mathbf{K}_n - \mathbf{K}_0\|_\infty
  \leq \epsilon_{\text{bf16}} \cdot \|\mathbf{A}_{\text{final}}\|_\infty$"
  Type: THEOREM
  Primary source: Theorem 1 proof at §C (sec:appendix-proof)
  Workflow:
    - Read the proof.
    - Verify it actually establishes the inf-norm bound stated.
    - Check: does the proof explicitly account for the FP32 accumulator's
      cumulative-delta property? Does it bound the cast error correctly?
    - If proof is one-sentence and waves at "the FP32 accumulator
      maintains the exact cumulative delta" without showing the
      algebra, mark IMPRECISE.
  Notes example: "§C proof sketch: 'FP32 accumulator maintains exact
  cumulative delta; quantization error bounded by epsilon_bf16 times
  accumulator magnitude.' This is a sketch, not a rigorous proof — no
  algebraic derivation of why infinity-norm holds vs. L2-norm. Adequate
  for a proof SKETCH but not for a full theorem proof. IMPRECISE.
  Recommend either (a) expand proof for camera-ready, or (b) hedge
  theorem statement to 'we conjecture'."

EXAMPLE 5 — META verification:
  Paper claim (§7.3 Limitations item 1): "we make no claim about MoE
  architectures (e.g., gpt-oss-20b)"
  Type: META
  Primary source: rest of paper
  Workflow:
    - Grep paper for "MoE", "gpt-oss-20b", "Mixture of Experts".
    - Find every occurrence and check what claim is made.
    - If any body section makes a MoE-specific claim that the limitations
      says we don't make, mark DIVERGENT.
  Notes example: "Grep results: §6.2 line 318 mentions gpt-oss-20b as
  the 20B-architecture spec source; §7.4 Future Work line 356 lists
  gpt-oss-20b as a natural next target. Both are 'we will / could' framings,
  not 'we showed' framings. No body claim contradicts §7.3 item 1.
  CONFIRMED."

----------------------------------------------------------------------
ESCALATION RULES
----------------------------------------------------------------------

After completing the audit, report to Josh:
  - Total claims audited
  - Severity counts (SHOWSTOPPER, DIVERGENT, IMPRECISE, UNVERIFIABLE,
    CONFIRMED)
  - Top 5 most critical findings (lowest-severity-acceptable)
  - Matrix file path

Verdict for Day-12 readiness:
  - 0 SHOWSTOPPERS + 0-3 DIVERGENT/UNVERIFIABLE → PASS. Paper is fit
    for Day-12 parallel review.
  - 0 SHOWSTOPPERS + 4-10 DIVERGENT/UNVERIFIABLE → CAUTION. Josh decides
    whether to patch before firing Day-12 axes or accept findings as
    inputs to Day-12 review.
  - 0 SHOWSTOPPERS + 11+ DIVERGENT/UNVERIFIABLE → PAUSE. Pre-Day-12
    patch required.
  - 1+ SHOWSTOPPERS → STOP. Escalate immediately. Paper-go/no-go decision
    territory; Tianyu consultation may be needed.

----------------------------------------------------------------------
ANTI-OVERLOAD GUIDANCE
----------------------------------------------------------------------

- The paper has ~150-250 distinct claims. Work iteratively. Section by
  section. Write to file as you go.
- Time budget: ~30-45 min per body section (Abstract/§1/§2/§3/§4/§5/
  §6/§7); ~20-30 min per appendix section (§A through §G); ~30 min for
  bibliography metadata verification; ~30 min for cross-section. Total
  estimated: 4-6 hours wall.
- If you're going faster than 30 sec per claim, you're rationalizing.
  Slow down and verify properly.
- If a section has 30+ claims, consider spawning a sub-agent (Agent tool,
  subagent_type=general-purpose) with this SAME paranoid prompt scoped
  to that one section. Pass the sub-agent the section's TeX line range
  + the matrix output path + the cite-to-PDF mapping. Orchestrator
  stays lean.
- Pause and report to Josh after Phase 1 if you're past 4 hours wall.
  Don't push through into Phase 2 without Josh's awareness.

----------------------------------------------------------------------
DON'T
----------------------------------------------------------------------

- Don't fire the Day-12 parallel-review axes from this session. Separate.
- Don't modify the paper TeX. Read-only on the paper.
- Don't synthesize, rationalize, round, or interpret. Quote primary
  sources verbatim.
- Don't accept "memory says X" as verification. Memory may be stale.
- Don't commit anything unless Josh explicitly asks.

----------------------------------------------------------------------
BEGIN
----------------------------------------------------------------------

Start with Phase 0 setup. Read the bibliography (TeX lines ~376-462).
Build the cite-to-PDF mapping. Then start Phase 1 with the Abstract
(TeX line ~55). Write to docs/logs/<today's-date>_claims-audit-matrix.md
as you go.

If you hit a SHOWSTOPPER at any point, STOP. Write to the matrix's
Showstoppers section and report to Josh.

End-of-task: report severity counts + top 5 findings + matrix file path.
```

---

## Why this prompt is structured the way it is

- **Paranoia tone is set early and reinforced.** The "Mode: Paranoid" framing + hard rules + "never trust memory" rule prevents the auditor from rationalizing.
- **Per-section workflow with file-write-as-you-go** keeps context lean. 200 claims is too many to hold in context simultaneously.
- **Worked examples per claim type** show what "good" looks like — the auditor doesn't have to invent the verification pattern.
- **Severity rubric with escalation rules** makes the audit decision-shaped, not just observation-shaped.
- **Cite-to-PDF mapping is a Phase-0 step** because the bibliography's `\cite{key}` to filename mapping requires inference (some keys map to arXiv IDs, some to named PDFs, some to no PDF). This step also surfaces UNVERIFIABLE claims early.
- **Cross-section checks at Phase 2** catch internal-consistency issues that per-section passes miss (e.g., abstract overclaim vs body hedge).
- **Sub-agent option for dense sections** allows parallel scaling if needed without losing paranoid framing.
- **Showstopper escalation is immediate** — a code-vs-paper semantic divergence is paper-go/no-go territory and shouldn't wait for the full audit to complete.

## Why run this BEFORE Day-12 parallel review

- The parallel-review axes (Axes 1-5) are adversarial-reviewer-shaped — they ask "is this defensible?", not "is every number verbatim correct?"
- A claim that's hallucinated will pass Axes 2/4/5 (framing/reviewer-prior/adversarial) if it sounds plausible, but fail this audit (which checks primary source).
- If this audit surfaces SHOWSTOPPERS, Day-12 parallel review is moot — the paper needs surgery first.
- Running this first lets Day-12 axes operate on a verified claim base, making their findings more actionable.

## Estimated wall time

4-6 hours single-session. If using sub-agent parallelization (one sub-agent per body section), ~2-3 hours wall (with synthesis at end).

## What to do with the audit output

- If PASS: forward matrix file to Day-12 orchestrator as a pre-flight reference.
- If CAUTION: Josh triages findings; patch the top-N before firing Day-12 axes.
- If PAUSE / STOP: surgery required; Day-12 axes deferred until patches land.

## Caveat on UNVERIFIABLE findings

A cited paper not present in `Citations/` is a real risk for two reasons:
1. The claim can't be defended under reviewer pushback ("what page of Bailey 2024 says this?")
2. The auditor can't confirm vs deny — they just can't verify
For each UNVERIFIABLE in the matrix, the recommendation should be: download the missing PDF or remove/soften the citation. Josh decides.
