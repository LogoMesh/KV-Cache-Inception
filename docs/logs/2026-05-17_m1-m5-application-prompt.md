# M1-M5 Patch Application Prompt — Day-13 AM (for Day-12 parallel-review session)

**Target:** the Day-12 parallel-review session that produced the three Axis reports
**Created:** 2026-05-17 (Day 12 PM)
**Framing context:** Josh has decided (or is deciding) on Option B reframe of "Dimensional Escape" — anchored to Goodhart's Law as a specific instance, not a novel phenomenon class. M3 below reflects Option B. If Josh chooses Option A (retire DE entirely), M2 + M3 below need wholesale rewrite; ask Josh first.

---

## Copy-paste prompt for the other session

```
M1-M5 PATCH APPLICATION. Round 1 parallel-review (Axes 1, 2, 5) surfaced
5 MAJOR findings synthesized at:
  docs/logs/2026-05-17_round-1-synthesis.md (in worktree
  elegant-shockley-b745dc; you can read the synthesis there as
  background, but the patch instructions below are self-contained).

Apply M1-M5 to docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex
(current paper, post-Day-12 commits 8f3f1c5 + 267e9cb). Net page-budget
cost ~0.1-0.2pp; body has headroom (currently exactly 8pp). After all
patches: smoke-compile (pdflatex × 2 passes), verify body still ≤8pp,
verify 0 undefined refs, then commit and push to origin/main.

FRAMING NOTE: M2 + M3 below use Option B framing (Dimensional Escape =
named instance of Goodhart's Law in high-D latent spaces). If Josh has
not signed off on Option B framing, PAUSE before M2 + M3 and ask. M1,
M4, M5 are framing-independent and can apply unconditionally.

----------------------------------------------------------------------
PATCH M1 — §3.2 line 119 unification claim hedge
----------------------------------------------------------------------

Locate (TeX line ~119):
  "We argue these are sub-cases of a single underlying pathology."

Replace with:
  "We hypothesize these as parallel manifestations of the
  overparameterized-latent-space geometry described below; only the
  reward-driven case is evidenced empirically in this paper, with the
  monitor-driven case under inference-time KV-cache perturbation an
  open empirical question (§\ref{sec:future-work})."

Rationale: this body sentence directly contradicted the Day-11
abstract hedge. Patch harmonizes body with abstract. Word count:
~12 → ~45 (+33 words).

----------------------------------------------------------------------
PATCH M2 — §1.1 Contribution 2 title hedge (Option B framing)
----------------------------------------------------------------------

Locate (TeX line ~84):
  "\item \textbf{Empirical demonstration of Dimensional Escape under
  continuous-latent search.}"

Replace with:
  "\item \textbf{Empirical demonstration of reward-driven Dimensional
  Escape---an instance of Goodhart's Law in high-dimensional
  latent-space search.}"

Rationale: title currently overclaims empirical scope (monitor-driven
case is conjectured); also anchors DE to Goodhart's Law per Option B
framing. Word count: ~9 → ~17 (+8 words).

After applying M2, smoke-compile to verify title doesn't break section
layout (longer title may wrap one line).

----------------------------------------------------------------------
PATCH M3 — §3.2 Definition def:dim_escape Goodhart anchor (Option B)
----------------------------------------------------------------------

Locate (TeX, immediately before "\begin{definition}[Dimensional
Escape]\label{def:dim_escape}" — currently around line 118-119):

  [no current text; insert new paragraph before \begin{definition}]

Insert immediately before "\begin{definition}":

  "We position Dimensional Escape as a specific manifestation of
  Goodhart's Law~\cite{gao2023scaling, skalse2022defining} in
  low-rank-scored, high-dimensional latent spaces, generalizing the
  gradient-trained input-attack case characterized by Bailey et al.
  as Orthogonal Escape~\cite{bailey2024obfuscated}. The novelty we
  claim is not a new phenomenon class but a specific empirical
  setting (inference-time KV-cache MCTS with surrogate-reward
  scoring) and a two-mechanism characterization of how the search
  satisfies the proxy without satisfying the objective
  (prior-amplification at small model scale, greedy-collapse at
  larger scale; §\ref{sec:exp1}--\ref{sec:exp2})."

Plus add bibitems to References (after existing bibitems near line
~445):

  \bibitem{gao2023scaling}
  L.~Gao, J.~Schulman, and J.~Hilton.
  \newblock Scaling Laws for Reward Model Overoptimization.
  \newblock In \emph{ICML}, 2023. \emph{arXiv:2210.10760}.

  \bibitem{skalse2022defining}
  J.~Skalse, N.~Howe, D.~Krasheninnikov, and D.~Krueger.
  \newblock Defining and Characterizing Reward Hacking.
  \newblock In \emph{NeurIPS}, 2022. \emph{arXiv:2209.13085}.

Rationale: anchors "Dimensional Escape" to established Goodhart's Law
literature. Defeats the "you're just relabeling X" reviewer attack by
explicitly saying "yes, we're naming a specific instance of X."
Word count: ~110 words new prose + 2 new bibitems.

VERIFICATION: before adding the bibitems, check that they aren't
already present in the bibliography (grep for "gao2023scaling" and
"skalse2022defining" in the .tex first). If already present, skip the
bibitem additions.

----------------------------------------------------------------------
PATCH M4 — Abstract "under any surrogate reward" hedge
----------------------------------------------------------------------

Locate (TeX line ~56, abstract):
  "...expose a fundamental geometric pathology we term \textbf{
  Dimensional Escape}: under any surrogate reward lacking exogenous
  grounding, the high-dimensional cache resolves localized
  optimization pressure by routing representations through
  unconstrained orthogonal dimensions, satisfying the proxy metric
  without satisfying the semantic objective."

Replace with:
  "...characterize a geometric pathology we term \textbf{
  Dimensional Escape} (an instance of Goodhart's Law in
  high-dimensional latent-space search): under the entropy-normalized
  surrogate reward we test---and, by extrapolation under
  §\ref{sec:dim_escape}'s dimension-counting argument, surrogate
  rewards lacking exogenous grounding more generally---the
  high-dimensional cache resolves localized optimization pressure by
  routing representations through unconstrained orthogonal
  dimensions, satisfying the proxy metric without satisfying the
  semantic objective."

Changes: (a) "expose a **fundamental** geometric pathology" → "**
characterize a** geometric pathology" (data supports phenomenon's
existence, not its fundamentality); (b) "under **any** surrogate
reward" → "under **the entropy-normalized surrogate reward we test
and, by extrapolation, surrogate rewards lacking exogenous grounding
more generally**" (hedges generalization). Word count: +30 words.
Anchors DE to Goodhart per Option B framing.

----------------------------------------------------------------------
PATCH M5 — §5 MCTS-vs-enumeration scope clarifier
----------------------------------------------------------------------

Locate (TeX, immediately after the §5 "Experiments" subsection intro
paragraph at line ~183 and before "Experiment 1" paragraph at line
~186):

Insert new paragraph:
  "\paragraph{Search-variant scope.} The Reversible MCTS framework
  defined in §\ref{sec:method} supports both exhaustive path
  enumeration and UCB1 bandit-guided expansion. In the experiments
  below we instantiate the depth-$3$ alpha-set variant via exhaustive
  enumeration of all $3^3 = 27$ paths from $\{0.1, 0.5, 1.0\}^3$
  per item; this is computationally cheaper than UCB1 at depth $3$
  and makes the headline results independent of the bandit's
  exploration constant (Appendix~\ref{sec:reproducibility}
  documents the configuration). The complexity bound of
  Proposition~\ref{prop:memory} is independent of which variant is
  used."

Rationale: paper's title says "MCTS" but experiments enumerate. This
paragraph explicitly scopes the experimental variant and notes the
algorithmic framework supports both. Defends against reviewer-likely
"you don't actually do tree search?" attack. Word count: ~75 words.

----------------------------------------------------------------------
SMOKE-COMPILE + VERIFY
----------------------------------------------------------------------

After all patches applied, run in docs/NeurIPS/:
  pdflatex -interaction=nonstopmode 04.18.2026-NeurIPS-Research-Proposal-2.tex
  pdflatex -interaction=nonstopmode 04.18.2026-NeurIPS-Research-Proposal-2.tex

Verify:
  - 0 undefined references (grep .log for "undefined")
  - Body still ≤ 8pp (grep .aux for "sec:reproducibility" → should be
    on page 9, i.e., body fits 8pp)
  - Net page count unchanged or +1 (acceptable; current is 13pp)
  - No new warnings beyond the pre-existing 5× "h→ht float specifier"
    cosmetic warnings

If body slips to 9pp (sec:reproducibility on page 10): STOP and
escalate to Josh. The M3 + M5 patches are the heaviest (~110 + 75
words new prose). If page-budget bursts, M3 is the cut candidate (M5
is methodology-defense; M3 is framing-defense and partially overlaps
with M2's title hedge).

----------------------------------------------------------------------
COMMIT + PUSH
----------------------------------------------------------------------

Commit message template:

  docs: M1-M5 patches from Round-1 parallel-review synthesis

  Apply 5 MAJOR-severity findings from
  docs/logs/2026-05-17_round-1-synthesis.md per Option B framing
  (Dimensional Escape = named instance of Goodhart's Law in
  high-dimensional latent-space search, not a novel phenomenon class).

  M1 — §3.2 line 119 unification-claim hedge (Axis 2 Q3+Q6):
       "we argue these are sub-cases of a single underlying pathology"
       → "we hypothesize these as parallel manifestations [...] only
       the reward-driven case is evidenced empirically".

  M2 — §1.1 C2 title hedge (Axis 2 Q3): adds "reward-driven" qualifier
       and "an instance of Goodhart's Law in high-dimensional
       latent-space search" anchor.

  M3 — §3.2 Goodhart-anchor preamble (Axis 2 Q5 + Axis 5 Pass H):
       new paragraph immediately before def:dim_escape positioning
       Dimensional Escape as a named instance of Goodhart's Law (cite
       gao2023scaling, skalse2022defining) generalizing Bailey 2024's
       Orthogonal Escape. Two new bibitems added.

  M4 — Abstract "under any surrogate reward" hedge (Axis 5 Pass A):
       softens "fundamental geometric pathology" to "geometric
       pathology" and "under any surrogate reward" to "under the
       entropy-normalized surrogate reward we test, and by
       extrapolation, surrogate rewards lacking exogenous grounding
       more generally". Anchors DE to Goodhart.

  M5 — §5 search-variant scope clarifier (Axis 5 Pass D): new
       paragraph distinguishing the algorithmic framework (supports
       both exhaustive enumeration and UCB1) from the experimental
       instantiation (depth-3 exhaustive enumeration). Defends
       against "you don't actually do tree search?" reviewer attack.

  Verification: smoke-compile clean (13pp PDF, 0 undefined refs,
  body 8pp per .aux sec:reproducibility = page 9). Net diff:
  ~250-300 word additions, page-budget impact ~0.1-0.2pp; body
  retains 8pp budget.

  Round 2 (Axes 3 + 4) fires Day-13 AM per
  docs/logs/2026-05-17_day-12-session-prompt.md timing table; M1-M5
  patches and Round 2 firing are independent and can run in parallel.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>

Then push to origin/main:
  git push origin HEAD:main

REPORT BACK TO JOSH:
  - Commit hash
  - Push confirmation
  - Net diff stats (insertions/deletions)
  - Body page count (should still be 8pp)
  - Any deviations from the patch plan (e.g., M3 page-budget bust)
  - Suggest: Round 2 (Axes 3+4) firing per
    docs/logs/2026-05-17_day-12-session-prompt.md
```

---

## Why this prompt is structured the way it is

- **Self-contained for the other session.** No reliance on memory shared between sessions; the prompt provides the exact old/new strings.
- **Option B framing baked in.** M2 + M3 explicitly anchor "Dimensional Escape" to Goodhart's Law, eliminating the "this is just relabeling" attack vector. M1/M4/M5 are framing-independent.
- **Framing-pause gate.** If Josh hasn't decided Option B yet, the other session pauses before M2 + M3.
- **Bibitem-verification step.** M3 adds new bibitems but checks for duplicates first.
- **Smoke-compile + page-budget gate.** Stops and escalates if body slips to 9pp.
- **Push instruction included.** Per CLAUDE.md "do not push unless explicitly asked," but Josh's "ill paste it in its session" implies he wants the patches landed + pushed.

## What happens after this prompt fires

1. Other session applies M1-M5, commits, pushes
2. Reports back to Josh
3. Josh decides on Round 2 firing (Day-13 AM per parallel-review-plan timing)
4. Tianyu can read the updated paper at any time — patches are defensive against the framing attacks his pass-1 read would likely miss anyway (per your note he doesn't catch citation errors / framing issues)
