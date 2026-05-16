# Cover note draft — Tianyu Shi send-out (Day 11)

**To:** Tianyu Shi
**From:** Josh
**Subject:** KV-Cache Inception — paper for your review before ARR May 25 (T-9 days)
**Date intended:** 2026-05-16 (Day 11 of execution plan)

---

Hi Tianyu,

Sending the latest revision of "Reversible KV-Cache MCTS: A Memory-Bounded Probe of Dimensional Escape in Frozen Language Models" for your review before the ARR May 25 submission. EMNLP 2026 commit deadline is 2026-08-02 via the ARR May cycle.

**Repository:** https://github.com/LogoMesh/KV-Cache-Inception
**Latest commit:** `8085545` (Day-11: hedge fixes + page-budget body-to-8pp)
**Canonical TeX:** `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`

**Format:** ACL review-version (acl.sty), 8 pp body + unlimited references + 5 pp appendix. Body fits exactly within the ACL 8-page hard limit per [acl_formatting.md:17].

**Contributions (post Day-10 drop-C2 surgery):**
1. Reversible KV-Cache MCTS algorithm with FP32-accumulator zero-drift reversibility (Theorem 1; validated to machine precision over 200 apply-revert cycles on Llama 3.2-1B/3B) and branching-factor-independent memory bound (Proposition 1; measured 3.04--3.06× M_KV; implied ~60× projected reduction at (b,d)=(3,5) on 20B configurations).
2. Empirical demonstration of Dimensional Escape under continuous-latent search: Experiment 1 (Track F entropy-MCTS negative control on n=200 paired ARC-Easy items × 1B + 3B; paired McNemar p=1.0 both scales) + Experiment 2 (Track G Latent Cartography across 4 prompt classes × 2 scales). Reward-driven failure framed as structurally analogous to the monitor-driven Orthogonal Escape from the alignment-faking literature; conjecture both as sub-cases.

**Recent work (Days 5-11):**
- Days 5-6: §5 Experiment 2 Cartography recast + §A Reproducibility Appendix applied
- Day 7: §5.5 Evaluation Metrics rewrite per audit
- Day 8: ACL style switch + §6 worked-example numbers (Track D measurements + 20B projection footnote)
- Day 9: page-budget cut plan (23 ranked cuts) + parallel-review plan extended w/ Passes L/M/N/O
- Day 10: cut application — 23 cuts + drop-Contribution-2 (telemetry matrix → §F appendix); body 19pp → 14pp PDF / ~9pp body
- Day 11: Pass M hedge fixes (60× hedge harmonization + "structurally identical" → "structurally analogous") + Pass P page-budget cuts (body ~9pp → 8pp without moving tables/figures)

**Three places I'd most value your read:**
1. **§3.2 Dimensional Escape framing.** A reviewer-likely critique is "this is just Goodhart's Law relabeled." We've hedged the unification ("we argue ... structurally analogous; we conjecture both as sub-cases") but not retired the coined term. Is the def:dim_escape definition operationally distinct enough to defend? If not, recommend either (a) retire the coined term and re-anchor on "Goodhart in high-dimensional latent spaces," or (b) sharpen the geometric distinction.
2. **§4 Proposed Method depth.** After Day-10 we moved §4.2 Step 1/2/3 algorithm detail + Memory Complexity proof to Appendix §E to fit 8 pp. Body §4 now has: in-place mutation form (with reverse form re-folded inline Day-11), FP32 accumulator paragraph + Eq.~fp32_accumulator, MCTS Node Reward paragraph. A reviewer who skips appendices (a known subset of ACL reviewers) takes Theorem 1 + Proposition 1 on partial faith. Is this defensible or does §4 need re-folding?
3. **Empirical surface.** 1 model family (Llama 3.2 at 1B + 3B), 2 experiments, both with null/negative results. No positive result reported. The §5 Exp 1 paired McNemar p=1.0 has only 1-3 discordant items per scale. Is this scope reviewer-adequate for an Interpretability-track submission, or thin? §7.3 Limitations item 1 + item 3 acknowledge the scope; §7.4 Future Work scopes Phase B extensions. Whether the framing carries depends on the track's empirical norms.

**Parallel adversarial review (in parallel with your read, Day 12-13):**
We're firing a 5-axis pre-submission adversarial review Days 12-13 per `docs/logs/2026-05-14_parallel-review-plan.md`:
- Axis 1: code-semantic correctness on `logomesh/kv_mcts.py` vs paper §4 equations
- Axis 2: Dimensional Escape framing defensibility
- Axis 3: §A Reproducibility appendix gap audit
- Axis 4: Interpretability-track reviewer-prior alignment
- Axis 5: standard adversarial pass (19 sub-passes A-S; the most comprehensive)

Synthesis target `docs/logs/2026-05-22_parallel-review-synthesis.md` Day-13 EOD. I'll share findings Day-14 morning so we can integrate your feedback + adversarial findings into Day-14 patches before Day-15 portal submission. Let me know if you'd like to weigh in on any axis specifically.

**Standing risk profile (in priority order):**
1. Empirical scope — Pass R question above. Reviewer-likely revision-request trigger.
2. Body §4 depth — Pass Q. Reviewer-likely revision-request if you/parallel-review confirm thinness.
3. Unification claim — Pass M(d) / Axis 2 Q3. Hedge applied Day-11; reviewer-mileage varies.

**OpenReview accounts:**
- Josh ✓, Tianyu ✓, Max + Alaa pending response.

If you have a few hours this weekend, the first-priority read is §1 Contributions list + §3.2 Dimensional Escape + §5 Experiment 1 + §7.3 Limitations. Those four sections carry the paper's claim spine.

Happy to schedule a call once you've had a chance. Otherwise inline TeX comments or an email reply with the key issues works too.

Best,
Josh

---

## Notes for Josh (do not send to Tianyu)

- Origin remote confirmed: `LogoMesh/KV-Cache-Inception` (NOT `joshhickson/...`). Memory was corrected 2026-05-13.
- The "tianyushi.org" reference Josh mentioned: confirm Tianyu's preferred contact channel from his site (email vs other) before sending.
- Day-11 EOD signal target per plan v2: at least one co-author replies "got it, will read this week" so we know the send-out landed. If no Tianyu reply by 2026-05-17 morning, follow up.
- Max + Alaa send-outs: separate cover notes (similar template; Max has the v1-draft-Apr-25 mental model from before pivot, so include the Days 5-11 change log section). Alaa added 2026-05-11 — assume cold-context.
- Day-12 supplementary anonymization (plan v2 §5.1) runs in parallel; doesn't affect this send-out since main repo + canonical TeX don't need anonymization (only the supplementary package does).
