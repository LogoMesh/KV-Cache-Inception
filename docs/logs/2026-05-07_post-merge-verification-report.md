================================================================
POST-MERGE VERIFICATION — 2026-05-07
================================================================

BRANCH STATE:
  Branch: merge/max-fixes
  Commits ahead of main: 2
  Hash 1: ca3ec8e (feat: integrate Max's per-layer H-Neuron calibration + FP32 accumulator hardening (976fce5))
  Hash 2: 7bdb619 (fix: 3 bugs — reward formula sigmoid normalization, blocked counter, bare except)
  Working tree: clean (only untracked files: .claude/, docs/NeurIPS/Citations/_extracted/, this report's prompt)

POST-MERGE B6 RUN (full alpha sweep):
  Status: COMPLETE
  Total runtime: 2 min 59 sec wall (start 2026-05-07T06:07:44Z → end 2026-05-07T06:10:43Z;
                 calibration ~155s split across HNeuronMonitor + PerLayerHonestyProjector;
                 MCTS itself 2.2s for 25 expansions)
  residual_norm: 0.00e+00 (Theorem 1 reversibility holds end-to-end)
  oei_valid (any False?): 0 False / 25 non-root nodes (the only False is the root at depth=0,
            which has oei_score=None by design since no steering is applied)
  Calibration mode (verbatim from stderr): "Calibration complete. Mode: dense/MLP. Layers: 17. H-Neurons per layer: 50"
                                           — confirms Max's per-layer calibration is active
                                           (vs. the prior single-aggregate "H-Neurons: 50" log)

  Per-α telemetry (within-α range = [min, max] across 5 sibling nodes, all bit-identical
  per α as expected with temperature=0 + identical (α, dk, layer) per child):

    α=0.1: σ_H_mean=[0.228252, 0.228252] (n=5)  σ_H non-zero layers=[14, 14]/17  ρ_R_mean=[-1.306346, -1.306346]  TDS=[0.083559, 0.083559]  OEI mean=0.990905 std=0.000000
    α=0.5: σ_H_mean=[0.236740, 0.236740] (n=5)  σ_H non-zero layers=[14, 14]/17  ρ_R_mean=[-1.257858, -1.257858]  TDS=[0.080212, 0.080212]  OEI mean=0.991165 std=0.000000
    α=1.0: σ_H_mean=[0.247375, 0.247375] (n=5)  σ_H non-zero layers=[14, 14]/17  ρ_R_mean=[-1.198148, -1.198148]  TDS=[0.076101, 0.076101]  OEI mean=0.991547 std=0.000000
    α=2.0: σ_H_mean=[0.267263, 0.267263] (n=5)  σ_H non-zero layers=[14, 14]/17  ρ_R_mean=[-1.080120, -1.080120]  TDS=[0.068344, 0.068344]  OEI mean=0.992407 std=0.000000
    α=5.0: σ_H_mean=[0.329815, 0.329815] (n=5)  σ_H non-zero layers=[15, 15]/17  ρ_R_mean=[-0.533443, -0.533443]  TDS=[0.040312, 0.040312]  OEI mean=0.994995 std=0.000000

  Sample σ_H per-layer (one node per α; rounded to 5 decimals):
    α=0.1: [0.0, 0.05134, 0.17678, 0.26944, 0.1992, 0.32967, 0.41818, 0.4022, 0.35354, 0.4202, 0.35847, 0.08655, 0.35556, 0.26955, 0.18959, 0.0, 0.0]
    α=0.5: [0.0, 0.05219, 0.17923, 0.267,   0.19886, 0.32876, 0.4124,  0.40541, 0.35736, 0.43045, 0.37491, 0.10936, 0.38626, 0.29914, 0.22328, 0.0, 0.0]
    α=1.0: [0.0, 0.05357, 0.18258, 0.26396, 0.19861, 0.32894, 0.40694, 0.4103,  0.36315, 0.44457, 0.39566, 0.13465, 0.42404, 0.33371, 0.26469, 0.0, 0.0]
    α=2.0: [0.0, 0.05572, 0.18869, 0.25471, 0.1988,  0.33222, 0.39865, 0.4203,  0.37565, 0.47756, 0.43496, 0.17429, 0.49426, 0.39479, 0.34286, 0.0, 0.0]
    α=5.0: [0.0, 0.06162, 0.21291, 0.20721, 0.22917, 0.33949, 0.35465, 0.39759, 0.41721, 0.54948, 0.55411, 0.32989, 0.64287, 0.55965, 0.52496, 0.22606, 0.0]
    Pattern: layer 0 (embedding) and layer 16 (final) are 0.0 across all α; layer 15
    is 0.0 at α∈{0.1,0.5,1.0,2.0} and 0.22606 at α=5.0; layers 1–14 are uniformly
    non-zero at every α with magnitudes that shift coherently with α.

COMPARISON TO PAPER §5 TABLE 1 (post-W_K-fix-only run, tmp/runs/2026-05-06_b6_post-fix.json):
  OEI: paper says     {0.990905, 0.991165, 0.991547, 0.992407, 0.994995}
       merged shows   {0.990905, 0.991165, 0.991547, 0.992407, 0.994995}
       Delta: 0.000000 across all five α — bit-identical at full 6-decimal precision.

  ρ_R: paper says     {-1.306346, -1.257858, -1.198148, -1.080120, -0.533443}
       merged shows   {-1.306346, -1.257858, -1.198148, -1.080120, -0.533443}
       Delta: 0.000000 across all five α — bit-identical.

  TDS: paper says     {0.094414, 0.089815, 0.084467, 0.075068, 0.045962}
       merged shows   {0.083559, 0.080212, 0.076101, 0.068344, 0.040312}
       Delta:         {-0.010855, -0.009603, -0.008366, -0.006724, -0.005650}
                       i.e. each TDS lower by 0.006–0.011 absolute, ≈7–13% relative.
       This is the only telemetry channel that shifted. The shift is consistent with
       σ_H going from all-zero (paper run) to a 14–15-layer non-zero distribution
       (this run), which changes the JSD between the σ_H and ρ_R arrays.

  σ_H: paper says     {0.000000, 0.000000, 0.000000, 0.000000, 0.000000}
       merged shows   {0.228252, 0.236740, 0.247375, 0.267263, 0.329815}
       Delta:         entire column changed; new column rises monotonically with α.

  Agreement: PARTIAL — OEI and ρ_R bit-identical to paper §5 Table 1; σ_H column
  fundamentally different (0 → non-zero, monotonic in α); TDS column shifted as a
  downstream consequence of σ_H now being non-zero.

THREE-PROMPT σ_H VERIFICATION:
  Skipped — STEP 2 already shows σ_H_mean > 0 at every α value (range 0.228 → 0.330
  monotonic with α), so Q1 is conclusively answered without needing the deeper run.
  The brief explicitly authorizes skipping STEP 3 in this case.

ANSWERS:

Q1 — Does per-layer H-Neuron calibration fix σ_H = 0?
  YES. σ_H_mean is non-zero at every α in the post-merge run, varies monotonically
  with α (0.228252 → 0.236740 → 0.247375 → 0.267263 → 0.329815), and the per-layer
  σ_H vectors show 14–15 of 17 layers carrying non-zero values (only embedding layer 0
  and the final layer 16 are uniformly zero; layer 15 is zero at four α values and
  non-zero at α=5.0). This is the same merged code that previously reported σ_H = 0
  across all layers under three prompt regimes including a coercion-class prompt drawn
  from the calibration's selection class — so per-layer calibration is the change that
  flipped σ_H from constant-zero to varying-with-α.

Q2 — Does the merged code reproduce paper §5 Table 1?
  PARTIAL.
  - OEI: BIT-IDENTICAL at full 6-decimal precision across all five α.
  - ρ_R: BIT-IDENTICAL across all five α.
  - σ_H: MATERIAL CHANGE — now non-zero (was 0). See Q1.
  - TDS: MATERIAL CHANGE — shifted by 0.006–0.011 absolute (≈7–13% relative) lower at
    every α. Direct downstream consequence of the σ_H column changing: TDS = JSD(σ_H, ρ_R),
    so a non-zero σ_H array gives a different JSD than an all-zero σ_H array.

ERRORS / WARNINGS:
  No warnings or errors of any flagged kind. Specifically:
  - No "_broadcast_to: shape mismatch" warning fired (W_K projection path is firing
    correctly for every (layer, α) pair, as expected on the merged branch).
  - No "OEI computation failed" / "dummy OEI" / "OEI calculator fallback" messages.
  - No CUDA OOM, no deepcopy errors.
  - oei_valid=True for every non-root node (25/25); root is oei_valid=False as designed.
  - Only non-INFO line in stderr was the pre-existing `torch_dtype` is deprecated! Use
    `dtype` instead! transformers deprecation, same as every prior run.

OUTPUT JSON: ./tmp/runs/2026-05-07_b6_post-merge.json

================================================================
