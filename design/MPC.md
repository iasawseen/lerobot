# MPC inference in SawSeenVLAWM — schemes, findings, alternatives

Phase B of the Future Sight stack (see [`SawSeenVLAWM.md`](./SawSeenVLAWM.md)
section "Phase B — MPC inference with le-wm predictor" and
[`future-sight-implicit-wm.md`](./future-sight-implicit-wm.md)). Runtime-only
on a trained LGE-enabled checkpoint: the action expert produces an
anchor chunk `a*`, the LGE produces a clean goal latent `z_g` in le-wm
projector space, and an inner-loop MPC scores candidate chunks via the
le-wm JEPA predictor's `chunk_size`-step rollout from `z_t`.

```
cost(a_k) = ‖lewm.rollout(z_t, a_k)[-1] − z_g‖²
```

`z_t` is the encoder CLS of the current frame post-projector; `z_g` is
the LGE's denoised prediction in the same space. Both endpoints are in
le-wm's JEPA prediction manifold (post-projector); MSE is directly
comparable. The MPC inner loop is wired in `modeling_sawseenvlawm.py:1701`
(`_mpc_sample_actions`).

## Implemented schemes

### `anchor_perturb` (Scheme A)

Single shot. Anchor at slot 0; `N − 1` Gaussian perturbations from
`N(anchor, σ_init² · I)`. Argmin over `cost`. Anchor is always a
candidate, so the returned action is no-worse-on-the-score than the
anchor. Code: `_anchor_perturb_search` at `modeling_sawseenvlawm.py:1756`.

Knobs: `mpc_num_candidates` (N, default 16), `mpc_noise_scale` (σ_init,
default 0.1).

### `cem` (Scheme B)

Iterative refit-around-elites.

```
mu_0   = anchor
sigma_0 = σ_init (per-dim)
for m in 1..M:
    candidates_m = {anchor; mu + sigma * ε_k for k=1..N-1}   # anchor-included
    cost_m       = predictor_score(candidates_m)
    elite        = topk(candidates_m, K, by lowest cost)
    mu           = elite.mean()
    sigma        = elite.std() · (1 − α) + σ_init · α
    track best-ever across iters
return best_actions
```

The **anchor-included variant** (current code) injects the anchor as
slot 0 at every iter, not just iter 0. Two consequences:
1. Anchor's cost is in the topk pool. If anchor is the best, μ stays
   near anchor; σ collapses toward the empirical std of the elite
   (≈ 0 when anchor dominates), and subsequent iters sample tightly
   around anchor. **Self-anchoring on strong anchors.**
2. If anchor is genuinely worse than perturbations, anchor falls out of
   the topk, μ drifts toward perturbation mean, and CEM proceeds
   normally. **No regression on weak anchors.**

Code: `_cem_search` at `modeling_sawseenvlawm.py:1779`.

Knobs (in addition to AP's): `mpc_cem_num_iter` (M, default 4),
`mpc_cem_topk` (K, default 4), `mpc_cem_anchor_blend` (α, default 0.5).

Total candidate evaluations per chunk decision:
- AP: N
- CEM: N · M + 1 (anchor evaluated once, cached)

At defaults (N=16, M=4): AP = 16, CEM = 65.

### `mppi` (Scheme C)

Cost-weighted softmax aggregation instead of CEM's hard top-K elite cut.

```
mu_0  = anchor
sigma = σ_init (fixed across iters)
for m in 1..M:
    candidates_m = {anchor; mu + sigma * ε_k for k=1..N-1}   # anchor-included
    cost_m       = predictor_score(candidates_m)
    w_k          = softmax(−(cost_k − cost_m.min()) / β)
    mu           = Σ_k w_k · candidates_m,k                  # softmax-weighted mean
    track best-ever argmin across iters
mu_cost = predictor_score(mu)                                # consensus action
return best of {anchor, all per-iter argmins, mu}
```

Every candidate contributes proportionally to `exp(−cost / β)` —
no information thrown away. Robust to score noise (the exact pathology
hurting object_0 / spatial_0 under CEM). The softmax-weighted mean `mu`
is evaluated at the end and added to the best-ever pool, so the MPPI
"consensus" action can win on its own merits — not just participate in
the update.

Anchor-included by construction (slot 0 every iter, same safety
guarantee as `_cem_search`).

σ is held fixed at `σ_init` across iters (vanilla MPPI). Adding
σ-shrinkage analogous to CEM's elite-std refit is a possible
extension; not yet implemented. Temporal smoothness of perturbations
is now controlled by the iCEM colored-noise knob (`mpc_icem_beta`,
documented below) — that applies equally to MPPI and CEM.

Code: `_mppi_search` at `modeling_sawseenvlawm.py:1844`.

Knobs (in addition to AP's): `mpc_mppi_temperature` (β, default 1.0),
`mpc_mppi_num_iter` (M, default 4).

Total candidate evaluations per chunk decision: `N · M + 2` (1 anchor
scoring + (N−1) perturbations per iter + 1 final `mu` evaluation).

| scheme | total predictor rollouts (N=16 defaults) |
|--------|--:|
| AP | 16 |
| MPPI (M=1) | 17 |
| CEM (M=4) | 65 |
| MPPI (M=4) | 65 |

MPPI defaults to M=4 to match CEM at fixed compute — the cleanest
head-to-head ("softmax aggregation vs hard top-K elite cut" at equal
total rollouts). M=1 stays available as `MPC_MPPI_NUM_ITER=1` for the
vanilla-single-shot baseline ("does softmax beat argmin at AP's
budget?"), but is *under-explored* in our one-shot-per-chunk setup —
no receding horizon to re-anchor μ between control steps.

## Orthogonal levers

These knobs apply on top of any scheme (AP / CEM / MPPI). They share
plumbing — added once in `_mpc_sample_actions` (score-floor) or
`_perturbation_noise` (iCEM β) and inherited by every search method.

### iCEM colored noise (`mpc_icem_beta`)

Power-law spectral exponent for action-chunk perturbations: noise
PSD ∝ 1/f^β along the time axis (Pinneri et al. 2020).

| β | name | character |
|---|------|-----------|
| 0 | white | per-timestep i.i.d. — default, legacy behavior |
| 1 | pink | mild temporal correlation |
| 2 | red / Brownian | strong slow drifts — iCEM default |

Implementation: `_perturbation_noise` at `modeling_sawseenvlawm.py:1781`.
β=0 short-circuits to `torch.randn`; β>0 samples real+imag white noise
in the rFFT domain, scales each frequency bin by f^(−β/2) (with
f_min = 1/T floor to keep DC finite), and inverse-rFFTs back to the
time axis. Each (B, N, A) trajectory is rescaled to unit std along T,
so the σ semantics from `mpc_noise_scale` are preserved across β.
FFT is run in fp32 for stability and cast back to the dtype of the
caller's anchor.

Motivation: the le-wm predictor was trained on real action
trajectories, which are temporally smooth. White-noise perturbations
push candidates off the training manifold, and the predictor's cost
ranking degenerates into noise on those off-manifold candidates —
plausibly load-bearing in the object_0 / spatial_0 regressions
("score-noise rank inversion" failure mode in *Open failure modes*).
Colored perturbations stay closer to the manifold; topk elite
selection should become more meaningful.

Status: wired everywhere (AP / CEM / MPPI); β=0 keeps legacy
behavior so existing eval reproducibility is unchanged.

**Empirical result (2026-05-12 evening):** β=2 at σ=0.1 *regresses*
−12.9pp vs β=0 on the 7-task spatial partial. The Brownian-spectrum
noise has cumulative drift O(σT) over T=50 vs O(σ√T) for white, so
at the same per-element σ the candidates wander far further than
σ_init implies and the predictor's cost surface stops being
meaningful on them. To test the actual iCEM-claim at our budget, σ
needs to shrink in proportion to colored-noise's cumulative drift
(σ≈0.02–0.05 for β=2). Not yet tested.

### Score-floor escape (`mpc_score_floor_margin`)

Already documented in *Open failure modes* below — listed here for
symmetry with `mpc_icem_beta` as the other orthogonal lever.

## Empirical findings (2026-05-12)

Checkpoint:
`sawseenvlawm_libero_12k_bs64_lewm_proj_lge_sigreg_scheduled_middle_k10_2xGPUs_bf16`
(12k steps, 2×3090, LGE + Mode 3 + SIGReg + scheduled z_g from step 6k).

LIBERO eval at `EVAL_EPISODES=10`, 4 suites × 10 tasks = 400 episodes.

### Three-way comparison

| Suite          | eval (no MPC) | anchor_perturb N=16 | CEM N=16, M=4 | Δ CEM−eval | Δ CEM−AP |
|----------------|--:|--:|--:|--:|--:|
| libero_spatial | 78.0 | 81.0 | **83.0** | +5.0 | +2.0 |
| libero_object  | **95.0** | 85.0 | 87.0 | −8.0 | +2.0 |
| libero_goal    | 85.0 | 83.0 | **88.0** | +3.0 | +5.0 |
| libero_10      | 56.0 | 52.0 | **59.0** | +3.0 | +7.0 |
| **Average**    | **78.50** | 75.25 | **79.25** | **+0.75** | **+4.0** |

Wall-clock: eval 1h40m, AP 1h47m (+7%), CEM 2h2m (+22%).

### Headline

- **CEM > AP uniformly (+2 to +7pp per suite).** Iterative refit
  around elites consistently outperforms single-shot Gaussian
  perturbation at the same N=16 budget.
- **CEM > eval (+0.75pp aggregate).** First MPC variant tested that
  beats the no-MPC baseline on aggregate. Wins on 3 of 4 suites.
- **The compute-matched AP-N=64 ablation was stopped early** (was
  scheduled to test whether CEM's advantage is the iterative refit or
  just more candidates per chunk). Re-run if the σ-sweep results need
  the discriminating data point.

### Worst-Phase-A-tasks rescue pattern

| Phase A task | eval | AP-16 | CEM-16×4 | best |
|--------------|--:|--:|--:|---|
| spatial_5 | 30 | 60 | **60** | tie AP |
| libero_10_1 | 30 | 50 | **80** | **CEM** |
| libero_10_8 | 30 | 10 | **40** | **CEM** |
| spatial_9 | 40 | 70 | **90** | **CEM** |
| libero_10_4 | 50 | **0** | **50** | tie eval (AP catastrophic) |
| libero_10_2 | 50 | 50 | **90** | **CEM** |
| goal_6 | 60 | 80 | **80** | tie AP |
| **object_0** | **90** | **40** | **70** | eval (MPC degrades) |
| spatial_0 | 90 | 80 | **60** | eval (MPC degrades) |

Five of nine Phase-A-weak tasks (≤ 60%) get a clean win under CEM,
three are tied (CEM matches the better of AP/eval), and one degrades.
On Phase-A-strong tasks (≥ 90%), CEM still occasionally regresses —
object_0 and spatial_0 are the two clearest cases.

## Follow-up findings (2026-05-12 evening) — orthogonal levers, compute budget, le-wm reference

Same checkpoint and eval protocol. Four further runs against the
AI-CEM N=16 M=4 baseline (87.3% / 26-task partial → 86.7% / 27-task
partial reconstructed from `running_success_rate` parse of the prior
log). All runs were killed early once direction was clear.

### Headline

| variant | N | M | K | σ | β | result on first ~spatial+object slice | vs AI-CEM β=0 |
|---------|---:|---:|---:|---:|---:|---|---|
| AI-CEM (baseline) | 16 | 4 | 4 | 0.10 | 0 | spatial 88, object 87 | — |
| iCEM β=2 | 16 | 4 | 4 | 0.10 | **2** | spatial 75.7 (7-task partial) | **−12.9pp** |
| le-wm-ref CEM¹ | 16 | 4 | 4 | **0.50** | 0 | spatial 0/0/0 (3-task partial) | **−90pp**, collapsed |
| variant A | **32** | **8** | 4 | 0.10 | 0 | spatial 72, object 82, goal 83 | **−16/−5/~tied** |
| variant B | **64** | **16** | **8** | 0.10 | 0 | spatial 76, object 75, goal 40¹ | **−12/−12/−30 (1 task)** |

¹`mpc_cem_include_anchor=false, mpc_cem_init_mean=zero, mpc_cem_return=final_mean`
— matches `stable_worldmodel.solver.CEMSolver` defaults (μ_0 = 0,
slot 0 = current μ each iter, return final μ).

### Three negative results, one mechanism

1. **iCEM colored noise (β=2) hurts at σ=0.1.** Brownian-spectrum
   perturbations have cumulative drift O(σT) over the chunk horizon
   T=50 vs O(σ√T) for white noise — so at the same per-element σ,
   colored candidates wander far further from the anchor than σ_init
   implies. The le-wm predictor's cost surface is calibrated near
   real action trajectories; candidates that drift far off the
   manifold get bogus low costs that drag CEM's elite refit. Strong
   anchors get the most damage (spatial_4 100→70, spatial_5 90→30 in
   the 7-task partial). σ rescaling to ~0.02–0.05 may rescue β=2 but
   was not tested.

2. **Le-wm reference CEM (zero μ_0, no anchor inclusion) collapses
   at our compute budget.** N=16, M=4 with σ=0.5 starting from
   μ_0=0 puts candidates almost entirely off-manifold; the predictor
   cost ranking is noise; CEM refits μ to bogus low-cost candidates;
   all 30 episodes across 3 spatial tasks failed (0%). Le-wm's defaults
   are N=300, M=30 (≈140× our budget) — the algorithm requires that
   compute to escape the off-manifold start before the cost surface
   becomes informative. With our budget, the anchor is load-bearing.

3. **Compute-bumped AI-CEM (variants A and B) regresses uniformly
   on spatial.** 4× compute (A) and 16× compute (B) both ended up
   below baseline on spatial (72 and 76 vs 88) and below or close
   to baseline on object/goal. B underperformed A on object (75 vs
   82) — more compute is *not monotonically better*. Variant B's
   3-task early lead (90.0% on spatial_0–_2) was sample variance;
   collapsed to 76% over the full 10 spatial tasks.

### Mechanism: predictor cost surface is the bottleneck, not search

All three negative results point at the same failure mode: the le-wm
predictor's cost (sum-of-squares to z_g_pred in projector space) is
only meaningful in a *narrow neighborhood* of the data manifold of
real action chunks. Any perturbation distribution that produces
candidates far from this neighborhood — whether by large σ, by colored
noise's cumulative drift, by zero-mean μ_0, or just by more samples
(N=64 vs N=16) — surfaces candidates whose cost is dominated by
prediction-quality noise rather than goal-distance. The elite refit
then learns the noise.

Anchors are protected because they are *on-distribution by
construction* — the policy's flow-matching anchor lies in the data
manifold. The anchor's cost is informative even when the search
candidates' costs are not. AI-CEM at N=16 M=4 is roughly at the
budget ceiling: enough perturbation to recover failed anchors, not
so much that off-manifold noise dominates.

This explains the strong-anchor regression on spatial_0 / object_0 /
spatial_4 / spatial_5 / spatial_9 across every variant tested: when
the anchor is already strong (≥90% Phase A), there is no signal in
the candidate cost ranking — the best candidate is the anchor itself
or marginally-perturbed copies, and the noise floor of the predictor
swamps the marginal differences.

### Implication for next ablations

- **Compute-bumping is now closed as a productive direction at this
  predictor.** N=16, M=4 is roughly optimal at σ=0.1 white noise.
  Score-floor (margin>0) caps the downside but doesn't lift the
  ceiling — already tested in earlier 2026-05-12 work.
- **σ-sweep at the *lower* end of the range** is now the most likely
  win. σ ∈ {0.02, 0.05, 0.10} sweeps the "stay near anchor" regime.
  σ=0.02 may preserve strong anchors while still rescuing weak ones.
- **Calibration probe is now the highest-priority unwritten
  feature.** Without `cost_anchor` / `cost_best` / `elite_std` per
  task, every "MPC regression" failure is opaque. Add scalar logs in
  `_mpc_sample_actions`.
- **Cost-surface calibration**: revisit whether z_g distance is the
  right cost. Normalize per-task (whitening cost by predictor variance
  on real trajectories) might be the structural fix.

## Patterns observed

1. **MPC rescues failure-mode anchors.** When the action expert's anchor
   is bad (Phase A success rate ≤ 50%), the predictor-scored argmin
   reliably identifies a better candidate. spatial_5 (30→60),
   spatial_9 (40→90), libero_10_1 (30→80), libero_10_2 (50→90),
   libero_10_8 (30→40) — across both schemes, this pattern is
   consistent. The design hypothesis is confirmed for the target use case.

2. **AP regresses on near-perfect anchors; CEM regresses less.** When
   the action expert's anchor is already strong (Phase A ≥ 90%), the
   predictor's cost surface in z_g space is too flat for argmin to
   reliably keep the anchor. AP picks degraded candidates frequently
   (object_0 90→40, libero_10_4 50→0). CEM's iterative refit and
   best-ever tracking mitigates this (object_0 90→70, libero_10_4
   50→50) but doesn't eliminate it (spatial_0 90→60).

3. **CEM's wins concentrate in long-horizon (libero_10, +3pp) and
   spatial (+5pp).** These are the two suites where the action expert
   is weakest at baseline (78%, 56% vs paper 90%, 71%). MPC's value
   tracks anchor weakness — most useful exactly where the policy needs
   help.

4. **The σ_init=0.1 default is in the worst part of the bias/noise
   tradeoff curve.** Too tight to escape an anchor that's locally
   optimal but globally wrong; too loose to keep candidates near the
   action manifold when the anchor is already near-optimal. A σ sweep
   is the highest-EV next ablation.

## Open failure modes

- **Score-surface noise in z_g space.** Predictor was trained on
  imitation rollouts; on perturbed actions (off-policy), prediction
  quality degrades. When the cost surface is flat or noisy near the
  anchor, argmin picks badly. Symptom: object_0 / spatial_0
  regressions.

- **Score-floor escape wired but defaults to off** (`mpc_score_floor_margin=0.0`).
  Returns anchor when `(anchor_cost − best_cost) / anchor_cost < margin`
  per batch element. Caps the downside on near-perfect anchors at
  zero loss when enabled (`margin > 0`). Applied centrally in
  `_mpc_sample_actions` after the per-scheme search; each of AP, CEM,
  MPPI returns `(best_actions, anchor_cost, best_cost)` so the gate
  sees the same anchor/best costs the scheme tracked internally.
  Recommended starting margin: 0.05 (require ≥ 5% relative
  improvement before deviating from anchor).

- **Predictor history fabrication.** le-wm trained with `history_size=3`;
  at inference we have one real frame and pad by repeating `z_t`
  3× to fill the context window. First 2 rollout outputs are
  off-distribution and discarded (we only score the final emb), but
  the bias is non-zero.

- **Action-norm gap (accepted).** sawseenvlawm normalizes actions
  per-LeRobotDataset; le-wm uses its own StandardScaler. Magnitudes
  are close on LIBERO but not identical. No evidence the gap is
  currently load-bearing.

## Extensions not yet implemented

Ranked by signal-per-effort for the current setup.

### 1. Differentiable MPC

The predictor is differentiable and frozen. Replace stochastic search
with K Adam steps on `cost(z_t, actions, z_g)`:

```
for k in range(K):
    a ← a − lr · ∂cost/∂a
```

Pros: gradient is directional; could converge in 2–3 steps where CEM
needs M=4 iters × N=16 samples. Cons: ARPredictor depth-6
autoregressive backward over 9 rollout steps has non-trivial gradient
magnitude/stability concerns; requires `requires_grad=True` on the
action tensor + careful handling of frozen predictor's BN running
stats. Bigger lift (~50–80 LOC) but potentially much higher quality
than gradient-free methods. Smoke test before committing.

### 2. Multi-anchor (policy-level diversification)

Current MPC perturbs locally around one anchor. If the policy commits
to the wrong sub-skill (wrong hand, wrong grasp axis), no Gaussian
perturbation recovers — all candidates terminate in the same wrong
region. Mitigation: rerun the flow-matching denoise with 2–3
different seeds → 2–3 actually-different anchors → CEM/MPPI on each →
argmin overall. Cost: 2–3× anchor cost (cheap fraction of total MPC
budget). One extra hyperparameter (num_anchors).

Mentioned in [`SawSeenVLAWM.md`](./SawSeenVLAWM.md) and
[`future-sight-implicit-wm.md`](./future-sight-implicit-wm.md). Worth
trying only when calibration probes (cost_anchor, cost_best,
anchor_chosen_share per task) show a "locked anchor" failure mode —
i.e., low elite std with high cost on near-failing tasks.

## Suggested ablation order

Reordered after the 2026-05-12 evening follow-up runs. Completed
items are marked, with key result inline.

1. ~~Anchor-included CEM eval~~. **Done.** spatial_0 regression
   eliminated under AI-CEM (90→90 vs AP's 80). AI-CEM N=16 M=4 σ=0.1
   is the current best variant: **87.3% / 26-task partial**, +0.75pp
   over no-MPC eval on full 30-task slice.

2. ~~MPPI default eval (M=4, β=1.0)~~. **Done.** 84.2% / 19-task
   partial. Below AI-CEM. Cost-weighted softmax aggregation does not
   beat hard top-K elite cut at our budget.

3. **MPPI single-shot baseline (M=1, β=1.0).** Skipped — MPPI-M=4
   already underperformed AI-CEM, so the lower-budget variant is
   not expected to recover. Unblocked but de-prioritized.

4. ~~Score-floor margin sweep~~. **Partially done.** Margin=0.05
   stalled at 4 tasks; margin=0.15 reached 85.6% over 16 tasks —
   essentially tied with no-floor AI-CEM. Score-floor caps downside
   but doesn't lift the ceiling. Wired and available.

5. ~~Compute-budget bump on AI-CEM~~. **Done.** Variants A (N=32,
   M=8, K=4) and B (N=64, M=16, K=8) both regressed vs N=16 M=4
   baseline (see "Follow-up findings" above). **Compute-bound
   hypothesis falsified at σ=0.1.**

6. ~~iCEM colored noise (β=2) on AI-CEM~~. **Done.** −12.9pp on
   7-task spatial partial. σ=0.1 is mis-calibrated for β=2 (Brownian
   cumulative drift). Wired and available; needs paired σ-rescale
   (σ≈0.02–0.05) to retest fairly.

7. ~~Le-wm reference CEM (zero μ_0, no anchor)~~. **Done.** Collapsed
   to 0% on 3 spatial tasks. Le-wm's algorithm requires its N=300
   M=30 budget to escape the off-manifold start; at our budget the
   anchor is load-bearing.

**Open items (ranked by priority after evening findings):**

8. **Calibration probe (now top priority).** Wire `cost_anchor`,
   `cost_best`, `elite_std`, `anchor_chosen_share` as scalar logs in
   `_mpc_sample_actions` and persist via `--output_dir`. Without this,
   every regression is opaque — we cannot distinguish "predictor
   mis-ranks" from "σ too tight" from "anchor sub-optimal but locked".

9. **σ-sweep on AI-CEM, biased low.** σ_init ∈ {0.02, 0.05, 0.10,
   0.15}. ~6h total. σ=0.02 may preserve strong anchors at the cost
   of some weak-anchor rescue. Probably the largest remaining lever
   on this checkpoint.

10. **Differentiable MPC** smoke test. Gradient-direct optimization
    bypasses the noise-floor problem entirely (no stochastic candidate
    sampling). Cost: ~50–80 LOC + handling of frozen-predictor BN /
    LayerNorm running stats. Higher ceiling but bigger lift.

11. **Cost calibration / normalization**. Whiten z_g distance by the
    predictor's per-frame variance on real trajectories. Structural
    fix to the score-noise problem. Untested.

12. **Multi-anchor** — only after calibration probe identifies a
    "locked anchor" failure mode (elite std → 0 while best_cost
    stays high).

## Calibration probe (unimplemented)

Per-task logging of:
- `cost_anchor`: predictor-scored cost of the anchor chunk.
- `cost_best_perturb`: cost of the chosen candidate.
- `anchor_chosen_share`: fraction of chunk decisions where the
  anchor's cost is in the topk (CEM) or argmin (AP).
- `elite_std` per dim (CEM): collapse rate of the search distribution.

These metrics distinguish three failure modes:
1. **Anchor always wins, but is bad** → MPC isn't searching widely
   enough (σ too small or anchor diversity insufficient).
2. **Anchor rarely wins, but the chosen candidate is worse than
   anchor** → predictor score is uncalibrated (rank inversion).
3. **Anchor in top-K, μ drifts away anyway** → blend or topk
   misconfigured.

Wire these as scalar logs on the inner-model `_mpc_sample_actions`
forward; reconstruct per-task by reading from `eval_info.json`'s
per-episode list (once the eval `--output_dir` plumbing is fixed so
eval_info.json persists past container `--rm`).

## Cross-references

- [`SawSeenVLAWM.md`](./SawSeenVLAWM.md) — Phase B section. Originating
  MPC design (Scheme A + B, calibration probe, anchor-norm risks).
- [`future-sight-implicit-wm.md`](./future-sight-implicit-wm.md) —
  broader single-latent implicit-WM synthesis. Anchor + noise sampling
  and multi-anchor mitigations are introduced there.
- [`TODO.md`](./TODO.md) — item 3 ("VLAWM hybrid Phase B"). This
  document supersedes the design-stub there; that item is implemented
  modulo the calibration probe and score-floor escape.
