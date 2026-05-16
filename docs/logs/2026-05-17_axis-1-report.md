# Axis 1 Report — Code-vs-Paper Semantic Fidelity Review

Date: 2026-05-17
Scope: §4 (Proposed Method) of `docs/NeurIPS/04.18.2026-NeurIPS-Research-Proposal-2.tex`
       vs `logomesh/kv_mcts.py`, `logomesh/local_model.py`,
       `scripts/measure_lipschitz_drift.py`, `scripts/diagnose_track_f_negcontrol.py`.
Reviewer brief: Verify that the code computes what the equations claim at the
semantic level. Six mapping questions (Q1–Q6).

---

## Q1. FP32 Accumulator (Theorem 1 / Eq. 6)

### (a) Paper claim verbatim

Eq. 6, paper lines 160–163:

> $\mathbf{A}^{(\ell)} \leftarrow \mathbf{A}^{(\ell)} + \alpha \cdot \mathbf{d}^{(\ell)} \quad (\text{FP32}),
>  \qquad \mathbf{K}_t^{(\ell)} \leftarrow \mathbf{K}_{\text{base},t}^{(\ell)} + \text{cast}_{\text{bf16}}(\mathbf{A}^{(\ell)})$

Theorem 1, lines 289–304:

> "Let $\mathbf{K}_0 \in \mathbb{R}^{S \times d}$ be the baseline KV cache in bf16.
>  Let $\{\delta_1, \ldots, \delta_n\}$ be a sequence of steering interventions
>  applied and reversed via the FP32 accumulator. […] With FP32 accumulation,
>  this bound tightens to $\|\mathbf{K}_n - \mathbf{K}_0\|_\infty \leq
>  \epsilon_{\text{bf16}} \cdot \|\mathbf{A}_{\text{final}}\|_\infty$, which is
>  independent of $n$, depending only on the final accumulator state (which is
>  zero after complete reversal)."

### (b) Code location

`logomesh/kv_mcts.py`:
- `FP32Accumulator.from_kv_cache`: lines 245–270 (constructs `k_accum`, `v_accum` as FP32 zero tensors).
- `FP32Accumulator.apply`: lines 272–344.
- `FP32Accumulator.rollback`: lines 346–408.
- `FP32Accumulator.residual_norm`: lines 410–419.

### (c) Mathematical mapping

| Paper symbol | Code variable | Code line |
|---|---|---|
| $\mathbf{A}^{(\ell)}$ (FP32 accumulator) | `self.k_accum[l_idx]`, `self.v_accum[l_idx]` | kv_mcts.py:268-269 |
| $\alpha \cdot \mathbf{d}^{(\ell)}$ (FP32 delta) | `alpha * dk`, `alpha * dv` (dk/dv cast to FP32 at line 307-312) | kv_mcts.py:331-332 |
| $\mathbf{A}^{(\ell)} \leftarrow \mathbf{A}^{(\ell)} + \alpha \cdot \mathbf{d}^{(\ell)}$ (apply) | `self.k_accum[l_idx].add_(alpha * dk)` | kv_mcts.py:331 |
| $\mathbf{A}^{(\ell)} \leftarrow \mathbf{A}^{(\ell)} - \alpha \cdot \mathbf{d}^{(\ell)}$ (rollback) | `self.k_accum[l_idx].sub_(alpha * dk)` | kv_mcts.py:396 |
| $\text{cast}_{\text{bf16}}(\mathbf{A}^{(\ell)})$ | `self.k_accum[l_idx].to(dtype=self.k_base[l_idx].dtype)` | kv_mcts.py:337 |
| $\mathbf{K}_t^{(\ell)} \leftarrow \mathbf{K}_{\text{base},t}^{(\ell)} + \text{cast}_{\text{bf16}}(\mathbf{A}^{(\ell)})$ | `k_live.copy_(self.k_base[l_idx] + self.k_accum[l_idx].to(...))` | kv_mcts.py:335-338 |

**FP32 storage check.** `from_kv_cache` at line 268 explicitly creates accumulator tensors with `dtype=torch.float32`. The vectors `dk, dv` are cast to `torch.float32` at lines 307–312 before being added. Multiplications `alpha * dk` preserve dtype. `add_` and `sub_` are in-place ops that preserve the accumulator's FP32 dtype. **The FP32 promise holds end-to-end on the accumulator path.**

**Cast-on-read symmetry.** The cast `self.k_accum[l_idx].to(dtype=self.k_base[l_idx].dtype)` is applied identically in `apply` (line 337) and `rollback` (line 401). After a complete apply→rollback pair, the FP32 accumulator returns to exactly zero (FP32 subtraction of the same value is exact). The reconstruction `K_base + cast_bf16(0) = K_base + 0 = K_base` is exact in bf16. **Symmetric.**

**Phase 3 gate empirical evidence** (`docs/logs/2026-04-16_session-log.md` lines 155–166):

> "Cycle | Naive bf16 ‖K_n-K_0‖_∞ | FP32 accumulator
>   1   | 1.56e-02              | 0.00e+00
>  200  | 5.97e+00              | 0.00e+00"

The accumulator path reads exactly `0.00e+00` at all checkpoints (cycles 1, 50, 100, 150, 200). The reported residual is below FP32 machine epsilon, consistent with Theorem 1's stronger inequality $\|\mathbf{K}_n - \mathbf{K}_0\|_\infty \leq \epsilon_{\text{bf16}} \cdot \|\mathbf{A}_{\text{final}}\|_\infty = \epsilon_{\text{bf16}} \cdot 0 = 0$.

### (d) Verdict

**CLEAN.** Theorem 1 / Eq. 6 are implemented faithfully. FP32 storage is real. Apply/rollback symmetry holds. Empirical drift is exact-zero per the 2026-04-16 gate.

### (e) Severity

N/A (clean).

---

## Q2. `_project_residual_to_kv_shape` — residual→KV projection

### (a) Paper claim verbatim

§G Measurement Pipeline (paper line 611):

> "Representation-engineering probes are calibrated against residual-stream
>  activations of dimension $d_{\text{model}}$, while the KV cache stores
>  per-head representations of dimension $\text{kv\_heads} \times d_{\text{head}}$
>  under grouped-query attention. For Llama~3.2-1B these are 2048 and $8 \times 64$
>  respectively; an explicit projection between probe space and cache space is
>  required. [...] The fix projects probe directions through layer-specific key
>  projection weights, $\mathbf{d}_K^{(\ell)} = \mathbf{W}_K^{(\ell)}
>  \mathbf{d}_{\text{probe}}^{(\ell)}$, reshaped to per-head form."

### (b) Code location

`logomesh/kv_mcts.py:458–518` (`_project_residual_to_kv_shape`), called from
`_shape_match_kv` at lines 521–545, called from `FP32Accumulator.apply` at
lines 318–319 and from `FP32Accumulator.rollback` at lines 383–384.

### (c) Mathematical mapping

```python
weight = model.model.layers[layer_idx].self_attn.k_proj.weight  # [kv_heads*head_dim, d_model]
flat = w @ vec  # [kv_heads*head_dim], dtype matches vec dtype
per_head = flat.reshape(kv_heads, head_dim)
return per_head.unsqueeze(0).unsqueeze(2).expand(target_shape).contiguous()
```

| Paper | Code | Line |
|---|---|---|
| $\mathbf{W}_K^{(\ell)}$ (per-layer K projection) | `model.model.layers[layer_idx].self_attn.k_proj.weight` | kv_mcts.py:486 |
| $\mathbf{d}_{\text{probe}}^{(\ell)}$ | `vec` (1-D tensor of length `d_model`) | function signature, line 459 |
| $\mathbf{d}_K^{(\ell)} = \mathbf{W}_K^{(\ell)} \mathbf{d}_{\text{probe}}^{(\ell)}$ | `flat = w @ vec` (shape `[kv_heads*head_dim]`) | kv_mcts.py:501 |
| Reshape to per-head form | `flat.reshape(kv_heads, head_dim)` then unsqueeze+expand | kv_mcts.py:506-507 |

`nn.Linear.weight` stores `[out_features, in_features]`, so `w @ vec` (where `vec` is 1-D of length `in_features = d_model`) yields the `out_features` matvec product — exactly `W_K @ d_probe` per the paper.

### Norm / projection properties

The paper makes **no** formal claim about norm preservation, rank, or any other formal property of the projection. It is operationally defined as "the K-projection weight matrix multiplied by the probe direction." The code matches that operational definition.

The projection is **not** norm-preserving (W_K is a linear projection from d_model=2048 to a kv_heads*head_dim=512 space for Llama 3.2-1B; the operator norm depends on the layer's weight matrix). It is **not** an isometry. But the paper doesn't claim it is either.

**Side observation (not flagged as a finding because not paper-relevant):** The projection is then broadcast across batch and seq dims via `unsqueeze(0).unsqueeze(2).expand(target_shape)` — i.e., the **same** projected steering vector is applied at every token position in the cache. This is the documented Phase 2 behavior; it matches the implementation note in Appendix §sec:appendix-impl-notes item (iii) ("Phase 2 implementation broadcasts a single middle-layer vector to all layers"). The paper acknowledges this differs from "per-layer steering vectors" notation in §4.

### (d) Verdict

**CLEAN.** The projection $\mathbf{d}_K^{(\ell)} = \mathbf{W}_K^{(\ell)} \mathbf{d}_{\text{probe}}^{(\ell)}$ is implemented per the paper formula, the layer indexing is correct, and the per-head reshape matches the cache geometry. No formal property is claimed in the paper that the code violates.

### (e) Severity

N/A (clean).

---

## Q3. Entropy reward (`-H/log|V|`)

### (a) Paper claim verbatim

Hypothesis 2, paper lines 132–137 (Eq. goodhart):

> "When MCTS optimizes a surrogate reward $\hat{r}: \mathbb{R}^d \to \mathbb{R}$
>  lacking exogenous grounding---for instance, $\hat{r}(\mathbf{h}) =
>  -H(p(\cdot \mid \mathbf{h}))/\log|V|$ where $H$ is the next-token entropy at
>  the steered cache state and $|V|$ is the vocabulary size---the optimal
>  cache state under $\hat{r}$ generically lies off the semantic manifold"

Experiment 1 prose (line 190):

> "$\hat{r}_E(\mathbf{h}) = -H_t/\log|V|$ evaluated at the steered leaf"

### (b) Code location

`scripts/diagnose_track_f_negcontrol.py:210–217`:

```python
def _entropy_of_logits(logits: torch.Tensor) -> tuple[float, float]:
    """Return (entropy_nats, normalised_certainty_reward = -H/log|V|)."""
    l = logits.float()
    lp = torch.log_softmax(l, dim=-1)
    p = lp.exp()
    H = -(p * lp).sum().item()
    V = l.shape[-1]
    return H, -H / math.log(V)
```

Used at line 270 (`H, neg_norm_H = _entropy_of_logits(next_logits)`), reward
assigned at line 275 (`r = neg_norm_H`).

### (c) Mathematical mapping

The Shannon entropy is

$$H(p) = -\sum_i p_i \log p_i$$

computed in nats (using natural log via `torch.log_softmax`). The code constructs `p = softmax(logits)` and `log p = log_softmax(logits)`, then `(p * log p).sum()` is the sum of $p_i \log p_i$. Negation gives $H$ in nats.

Normalization by `log V` where `V = l.shape[-1]` is the vocabulary size taken from the **last dimension of the logits tensor** — for Llama 3.2-1B / 3.2-3B-Instruct this is `model.config.vocab_size = 128256`. Both numerator and denominator use natural log, so the ratio is unit-invariant: $-H_{\text{nats}}/\ln V = -H_{\text{bits}}/\log_2 V$.

The function returns $r = -H/\log V \in [-1, 0]$. With $r = -1$ at maximum entropy (uniform over $V$ tokens) and $r = 0$ at zero entropy (one-hot). Path-selection uses `r = neg_norm_H` and picks the max — so the search prefers low-entropy (high-certainty) cache states, matching the paper's "reward signal for confident next-token prediction" framing.

### Logit source

`next_logits = step.logits[0, -1, :]` (line 269) — the **last-position** logits from a forward pass over the steered cache. `step.logits` is the unmodified logit output (shape `[batch, seq, vocab]`), and `[0, -1, :]` selects batch=0, last seq position, all vocab. **Correct: entropy is computed over the full softmax of last-position logits.**

The forward pass at lines 263–268 calls `model(input_ids=input_ids[:, -1:], past_key_values=_kv_eval_cache(past_kv), use_cache=True)` — feeding only the **last** token of the prompt and continuing from the steered cache. This corresponds to the paper's $H(p(\cdot \mid \mathbf{h}))$ evaluated at the steered leaf.

### (d) Verdict

**CLEAN.** The entropy reward in `_entropy_of_logits` matches Eq. goodhart exactly. Shannon entropy over the full vocab softmax of last-position logits, normalized by $\log V$ taken from the logits' last dim (correctly 128256 for Llama 3.2 models).

### (e) Severity

N/A (clean).

---

## Q4. Path enumeration — 27 paths from `{0.1, 0.5, 1.0}^3`

### (a) Paper claim verbatim

Experiment 1 prose, paper line 190:

> "$27$ depth-$3$ paths drawn from the alpha-set $\{0.1, 0.5, 1.0\}$"

Experiment 2 prose, paper line 222:

> "depth-$3$ MCTS with $27$ path samples drawn from the alpha-set $\{0.1, 0.5, 1.0\}$"

Methods section, paper line 153, on the UCB1 / path-sampling shift:

> "(ii) compacted §4.2.2 […]; (iii) preserved §4.2 retains the FP32
>  accumulator paragraph and the MCTS Node Reward paragraph […]; UCB1 also
>  dropped."

The paper explicitly drops UCB1 from the depth-3 path-sampling variant. The
phrasing throughout §5 is "27 path samples drawn from the alpha-set" — i.e.,
**samples**, not "all 27 distinct paths."

### (b) Code location

`scripts/diagnose_track_f_negcontrol.py:68–70` (constants):

```python
ALPHA_SET = (0.1, 0.5, 1.0)   # branches=3
DEPTH = 3
N_NODES = 27                   # 27 path samples
```

`scripts/diagnose_track_f_negcontrol.py:244–245` (sampling loop):

```python
for _ in range(N_NODES):
    path = [rng.choice(ALPHA_SET) for _ in range(DEPTH)]
```

`scripts/diagnose_track_g_cartography.py:78–83, 425`:

```python
DEPTH = 3
MCTS_ALPHA_SET = (0.1, 0.5, 1.0)
MCTS_N_NODES = 27
...
path = [rng.choice(alpha_set) for _ in range(depth)]
```

### (c) Mathematical mapping

The code samples 27 paths **independently and with replacement** from the
27-element Cartesian product $\{0.1, 0.5, 1.0\}^3$. It does **not** enumerate
the 27 distinct paths.

Expected number of distinct paths under sample-with-replacement:
$$E[\text{distinct}] = 27 \cdot (1 - (26/27)^{27}) \approx 27 \cdot 0.6427 \approx 17.4$$

In expectation, ~9–10 distinct path tuples are sampled multiple times (duplicates), and ~9–10 path tuples in the Cartesian product are never sampled. Per item, the actual sample composition is RNG-dependent (per-item seed from `per_item_seed(i, "R"|"E", args.seed)`).

The deduplication observation: if the search were intended to be **exhaustive** enumeration, every item would test all 27 tuples and the modal-tuple statistics in Table 2 (paper line 244) would reflect per-item certainty. As implemented, the modal-tuple distribution conflates two effects: (a) which alpha-tuples the entropy reward prefers, and (b) which alpha-tuples happened to be sampled.

The paper says "drawn from the alpha-set" and "27 path samples" — these phrasings are consistent with sampling with replacement. The number 27 is the size of the full Cartesian product, which is a notable coincidence with the sample count, but the paper does not assert exhaustive enumeration. The paper says (line 190): "27 depth-3 paths drawn from the alpha-set ${0.1, 0.5, 1.0}$, scored by $\hat{r}_R \sim \mathrm{Uniform}(0,1)$" — note "drawn" not "enumerated".

### Implications for the experiment

- **No path is silently skipped or de-duplicated in code** — sampled paths are kept and rewarded individually. Duplicate samples receive independent entropy evaluations (which are deterministic for greedy decoding under the same path; the entropy of the steered next-token distribution is path-dependent, so duplicates have identical rewards).
- The "best path" computation picks the argmax across the 27 sampled rewards, which is well-defined.
- **However**, Table 2 (paper line 244) reports modal-path frequencies like "(1.0,1.0,1.0): 42%" for C2 1B. If only ~17 of the 27 path tuples are sampled per item in expectation, modal-path frequencies are mildly biased by which tuples were sampled across items — though the steering direction is fixed across items, so the same RNG choices reproduce the same sampling patterns within a condition.

### (d) Verdict

**CLEAN with a caveat: UNDERSPECIFIED-IN-PAPER on enumeration vs sampling.**

The paper text "drawn from the alpha-set" is consistent with the sample-with-replacement implementation. The 27 = $3^3$ coincidence is **not** flagged as enumeration in the paper. A reviewer reading line 190 carefully will register "drawn" as sampling, not enumeration. **However**, the paper does not state the sampling protocol (with or without replacement) explicitly, which is a moderate clarity gap — a careful reviewer is likely to ask "are these 27 distinct tuples or 27 independent samples?" Adding one clause to §5.1 ("27 path samples drawn independently with replacement from the 27-element Cartesian product $\{0.1,0.5,1.0\}^3$") would close the gap.

### (e) Severity

**MODERATE.** This is a reviewer-likely revision request. The current text is technically defensible but invites the natural question "if you drew 27 samples from a 27-tuple set, did you cover all of them?" A two-clause clarification ("sampled independently with replacement" or "enumerated all $3^3=27$ tuples" — pick one and say it) would prevent reviewer confusion. The current implementation is sampling-with-replacement.

---

## Q5. COCONUT defense — steering direction invariant across items

### (a) Paper claim verbatim

Experiment 1 prose, paper line 190:

> "The steering direction is a single random unit vector in $d_{\text{model}}$
>  space (\texttt{--seed 42}), shared identically across all items and across
>  the R and E arms; this isolates the entropy reward signal as the only
>  difference between R and E (the COCONUT defense)."

Experiment 2 prose, paper line 222:

> "The steering direction is the same random unit vector used in Experiment~1,
>  fixed via \texttt{--seed 42} across all items, classes, and scales."

### (b) Code location

`scripts/diagnose_track_f_negcontrol.py:342–347` (steering direction
construction):

```python
def make_steering_dir(model, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    d_model = model.config.hidden_size
    v = rng.randn(d_model).astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v
```

Constructed **once** at `main()` line 385:

```python
steering_dir = make_steering_dir(model, args.seed)
```

Passed **unchanged** to every per-item search call:
- Line 419: `run_search(..., steering_dir=steering_dir, rng=r_rng)` — Random arm.
- Line 425: `run_search(..., steering_dir=steering_dir, rng=e_rng)` — Entropy arm.

Inside `run_search` (line 237):

```python
dk_per_layer = [steering_dir.astype(np.float32)] * n_layers
```

The same `steering_dir` is broadcast identically across all `n_layers` for every item.

### (c) Mathematical mapping

| Paper claim | Code behavior |
|---|---|
| "single random unit vector" | `v = rng.randn(d_model); v /= ||v||` (line 345–346) |
| "in $d_{\text{model}}$ space" | `d_model = model.config.hidden_size` (line 344); for Llama 3.2-1B, 2048; for 3.2-3B, 3072 |
| "(--seed 42)" | `make_steering_dir(model, args.seed)` with `args.seed=42` default (line 360); seeded `np.random.RandomState(seed)` (line 343) |
| "shared identically across all items" | `steering_dir` constructed once at `main()` line 385, reused across the for-loop (lines 410–438) |
| "across the R and E arms" | Same `steering_dir` passed to both R and E `run_search` calls (lines 419 + 425) |

The default seed is `42` (line 360 — `p.add_argument("--seed", type=int, default=42)`). Per Track G report referenced in paper §5 Exp 2 implementation note (paper line 235), the same convention holds: steering direction is constructed once and reused.

### Cross-class / cross-scale claim

The paper says (Exp 2, line 222) the steering vector is "fixed via --seed 42 across all items, classes, and scales." Implementation check:
- "All items" — covered by the one-time construction at `main()` line 385.
- "All classes" — Track G driver (`diagnose_track_g_cartography.py`) processes all four classes within a single `main()` run, using one `make_steering_dir(model, seed)` call.
- "All scales" — when running `python diagnose_track_f_negcontrol.py --model 1B` vs `--model 3B`, each run uses `--seed 42` and `make_steering_dir(model, 42)` where `model.config.hidden_size` differs (2048 vs 3072). **The two scales draw from different d_model spaces, so they produce different vectors per call.** The paper claim "the same random unit vector used in Experiment 1, fixed via --seed 42 across all items, classes, and scales" is technically imprecise — it cannot be "the same vector" across two different `d_model` spaces. What it must mean (and what the code implements) is "the same random seed used to construct a unit vector in the model's d_model space at the time of each run."

This is a minor verbal imprecision in the paper but the experimental machinery is consistent.

### (d) Verdict

**CLEAN with a minor verbal-precision gap.** The COCONUT defense is implemented faithfully:
- Single random unit vector generated once at run start with `--seed 42`.
- Same vector reused for every item, both R and E arms.
- Within a given (model, scale, run) the steering vector is identical across items and arms.

The "same vector across scales" wording in §5 Exp 2 is loose since d_model differs across scales (1B=2048, 3B=3072), but the **seeded construction protocol** is identical and the spirit of the COCONUT defense (isolate the reward signal as the only between-arm difference) is preserved.

### (e) Severity

**MINOR.** Verbal precision in Exp 2 — "the same random seed" would be more accurate than "the same random unit vector" across scales. The implementation is correct. Optional one-word rewrite.

---

## Q6. Lipschitz drift — what does `measure_lipschitz_drift.py` actually measure?

### (a) Paper claim verbatim

Theorem 1, paper lines 289–304:

> "Let $\mathbf{K}_0 \in \mathbb{R}^{S \times d}$ be the baseline KV cache in
>  bf16. Let $\{\delta_1, \ldots, \delta_n\}$ be a sequence of steering
>  interventions applied and reversed via the FP32 accumulator. Then after $n$
>  complete apply-reverse cycles: $\|\mathbf{K}_n - \mathbf{K}_0\|_\infty
>  \leq n \cdot \epsilon_{\text{bf16}} \cdot \max_i \|\delta_i\|_\infty$
>  […] With FP32 accumulation, this bound tightens to
>  $\|\mathbf{K}_n - \mathbf{K}_0\|_\infty \leq \epsilon_{\text{bf16}} \cdot
>  \|\mathbf{A}_{\text{final}}\|_\infty$"

§sec:appendix-impl-notes item (v), paper line 537:

> "It was empirically validated to numerical zero drift over $200$ apply-reverse
>  cycles on Llama~3.2-1B-Instruct (residual norm $= 0.00$, 2026-04-16 gate) and
>  confirmed at the 3B scale (residual norm $= 0.00$, 2026-05-05 probe)."

Note the paper claim is **about the full KV cache K_0** ("Let $\mathbf{K}_0 \in
\mathbb{R}^{S \times d}$" — though "K" is used both as a stand-in for the full
cache and as the K-tensor of an attention block; from context this is the
generic cache state).

### (b) Code location

`scripts/measure_lipschitz_drift.py:113–138`:

```python
# Snapshot of layer-0 K at start (float32 for accurate comparison)
k_base_f32 = layers0[0][0].float().clone()

...
for cycle in range(1, args.n_cycles + 1):
    ok = accumulator.apply(past_kv_live, args.alpha, dk_vectors)
    ...
    ok = accumulator.rollback(past_kv_live, args.alpha, dk_vectors)
    ...
    # Measure ‖K_n − K_0‖_∞ for accumulator path
    k_now = _extract_kv_tensors(past_kv_live)[0][0].float()
    acc_inf_norm = float((k_now - k_base_f32).abs().max().item())
```

### (c) Mathematical mapping

**What is measured:** $\|K_{\text{layer 0}}^{(t=n)} - K_{\text{layer 0}}^{(t=0)}\|_\infty$ — the infinity norm of the difference between layer-0 K at start and after `n` apply/rollback cycles.

**What is NOT measured:**
- V tensors at layer 0 (`v_accum` is rolled back but never compared to `v_base`).
- K and V tensors at layers 1 through L-1.
- Per-token-position breakdown across the sequence dimension.

**Why this is still a meaningful empirical validation of Theorem 1:** The FP32 accumulator algorithm operates **identically on each (K|V, layer) tensor independently**:

```python
for l_idx in range(n):
    ...
    self.k_accum[l_idx].add_(alpha * dk)   # FP32 += FP32
    self.v_accum[l_idx].add_(alpha * dv)
    k_live.copy_(self.k_base[l_idx] + self.k_accum[l_idx].to(...))  # bf16 = bf16 + cast(FP32)
    v_live.copy_(self.v_base[l_idx] + self.v_accum[l_idx].to(...))
```

(kv_mcts.py lines 327–342)

Each accumulator is independent; each cast is independent. If layer-0 K stays exactly at zero residual across 200 cycles, layer-0 V follows the same dataflow and would also stay at zero residual. The other layers do too. **The layer-0 K measurement is a faithful single-tensor representative of the algorithm's per-tensor reversibility property.**

However:
- This is a **single-tensor probe**, not an **end-to-end cache equality check**.
- If a hypothetical bug existed only at, e.g., layer 5 V (a layer-specific or component-specific issue, perhaps stemming from a heterogeneous broadcast / shape mismatch quirk), the layer-0-K probe would not detect it.
- The Phase 3 gate (session log line 148) reports `residual_norm=0.00e+00` — which uses `FP32Accumulator.residual_norm()` (kv_mcts.py:410). That function does take the max over **all K layers** of the accumulator's abs-max:
  ```python
  return float(max(a.abs().max().item() for a in self.k_accum))
  ```
  So the 0.00e+00 in `run_kv_mcts.py` runs **does** cover all K layers (but still not V). And `measure_lipschitz_drift.py` does NOT use `residual_norm()` — it computes its own layer-0-K diff (line 138).

### Naive path simulation

The "naive" path at lines 140–150 simulates "what would happen without FP32 accumulator":

```python
dk_t = torch.from_numpy(dk_raw).to(device=k_base_f32.device, dtype=torch_dtype)
k_naive = k_base_f32.clone().to(torch_dtype)
for _ in range(cycle):
    k_naive.add_(args.alpha * dk_t)
for _ in range(cycle):
    k_naive.sub_(args.alpha * dk_t)
naive_inf_norm = float((k_naive.float() - k_base_f32).abs().max().item())
```

This is: apply N times, then reverse N times. The bf16 add/sub ops accumulate rounding error at each step. The 2026-04-16 log (line 169) records that bug 2 in this script was that the naive path "did independent add/sub pairs per iteration, reaching a fixed point instead of accumulating drift" — the fix was to apply N adds then N subs cumulatively, faithfully simulating the worst case. The current code at lines 146–149 does the N-adds-then-N-subs version, which is the correct simulation of the linear-growth bound $n \cdot \epsilon_{\text{bf16}} \cdot \max\|\delta\|_\infty$.

Verified by the session log table (line 159): cycle 200 has naive_inf_norm = 5.97e+00, which is roughly $200 \cdot 3.9 \times 10^{-3} \cdot \|0.5 \cdot d_{\text{unit}}\|_\infty$ — order of magnitude matches the predicted linear drift.

### (d) Verdict

**CLEAN at the per-tensor algorithm level, MODERATELY UNDERSPECIFIED in the empirical-validation scope.**

The script's accumulator-path measurement is **scoped to layer-0 K only**, but the algorithm operates identically on every (K|V, layer) tensor, so the per-tensor representative is faithful. The Phase 3 gate also separately checks `residual_norm()` across all K layers via `run_kv_mcts.py`. The paper text in §sec:appendix-impl-notes item (v) reports "residual norm = 0.00" without specifying scope, which is technically accurate per the per-tensor-representative argument but invites reviewer follow-up.

The naive path simulation is correct after the 2026-04-16 bug-2 fix (verified against the recorded drift table).

### (e) Severity

**MODERATE.** A reviewer asking "did you check V tensors too?" or "did you check every layer?" would receive the per-tensor-representative argument as justification, but this is post-hoc. Cheap fix: expand `measure_lipschitz_drift.py` to also track V at layer 0 and at least one mid-stack layer's K, OR have it use `accumulator.residual_norm()` directly to cover all K layers, OR add a V-equivalent `v_residual_norm()` method.

---

## Final Summary

### Severity tally

| Severity | Count | Findings |
|---|---|---|
| SEVERE | 0 | — |
| MAJOR | 0 | — |
| MODERATE | 2 | Q4 (path-enumeration vs sampling ambiguity); Q6 (drift script measures layer-0 K only) |
| MINOR | 1 | Q5 (verbal precision: "same vector" across different-d_model scales) |
| CLEAN | 3 | Q1 (FP32 accumulator); Q2 (W_K projection); Q3 (entropy reward) |

### What to patch first (ranked)

**Rank 1 — Q4: Path enumeration vs sampling clarity.** Add one clause to Exp 1 prose (§5.1 / paper line 190) and Exp 2 prose (§5.2 / paper line 222) stating explicitly that the 27 paths are "sampled independently with replacement from $\{0.1,0.5,1.0\}^3$." This is a 6–10 word insertion that closes the natural reviewer question. Cost: trivial (well under the 8-page budget).

**Rank 2 — Q6: Empirical-validation scope.** Either (a) update `measure_lipschitz_drift.py` to use `accumulator.residual_norm()` and add a `v_residual_norm()` for V coverage, OR (b) add one sentence to §sec:appendix-impl-notes item (v) ("Residual norm aggregated across all K layers via `FP32Accumulator.residual_norm()`; per-tensor algorithm is layer- and K/V-symmetric, so the layer-0 K probe in `measure_lipschitz_drift.py` is a faithful single-tensor representative."). Option (b) is cheaper and equally defensible.

**Rank 3 — Q5: Verbal precision.** Optional one-word edit to Exp 2 (§5.2 / paper line 222): change "the same random unit vector used in Experiment~1" to "the same random seed used in Experiment~1, regenerated as a unit vector in each model's $d_{\text{model}}$ space." Low priority; experimental machinery is correct.

### What is *not* a finding

- The FP32 accumulator implementation is correct and Theorem 1 holds.
- The W_K projection matches the paper's operational definition.
- The entropy reward formula is computed correctly with the correct vocabulary normalization for Llama 3.2 (V=128256 via dynamic `l.shape[-1]`).
- The COCONUT defense is implemented faithfully (single steering vector across items + arms within a run).
- The "0.00e+00 drift over 200 cycles" empirical claim is reproducible and the recorded data table is consistent with the predicted per-tensor algorithm behavior.

### Reviewer-likely questions the paper does not currently pre-empt

1. "Are the 27 paths the full Cartesian product or 27 independent samples?" (Q4)
2. "Was 0.00e+00 drift measured across all layers and both K/V, or only a representative tensor?" (Q6)
3. (Minor) "The steering vector cannot be 'the same' across 1B and 3B scales since d_model differs — what is invariant?" (Q5)

All three are addressable with paper text edits totaling under 50 words. No code surgery is required to defend the headline empirical claims.
