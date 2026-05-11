---
title: "Future Sight: Single-Latent Implicit World Modeling"
type: synthesis
status: draft
tags: [world-model, jepa, vla, mpc-planning, system-1-system-2, hybrid-architecture, target-prediction, flow-matching, single-latent]
created: 2026-05-09
updated: 2026-05-10
last_verified: 2026-05-10
covers: [implicit-world-modeling, le-world-model, pi0.7, smolvla]
---
# Future Sight: Single-Latent Implicit World Modeling

> **Note on naming.** "Future Sight Expert" is the architectural name used
> in this synthesis. The lerobot implementation in
> [`design/SawSeenVLAWM.md`](./SawSeenVLAWM.md) calls the same head the
> **Latent Goal Expert** — same component, more functional name.
> "Future Sight" refers to the broader *concept* (single-latent
> implicit-WM stack); Latent Goal Expert refers to the specific *head* in code.

A sister architecture to the [VLM-Async Implicit WM](./vlm-async-implicit-wm.md) synthesis. Same goal — fill the [implicit-WM contract](../entities/implicit-world-modeling.md) at deployment. Same MPC scaffolding — K trajectory candidates, WM rollout, scoring, argmin. Different bet on **how the language pathway plugs into the WM's latent space**: instead of bridging two spaces with a learned Cost transformer, predict the target latent directly with a goal-conditioned generator ("Future Sight"). The result is a *single*-latent stack — encoder, WM, target predictor, and scorer all live in one space.

Three structural moves:

1. **Failure-aware WM training** — train the LeWM-style encoder + predictor on a mixture of successful imitation data and *mined failure trajectories*, so the latent space is dynamics-complete on the bad regions of the manifold, not just the success band.
2. **One VLM, two flow-matching experts** — the Action Expert (existing in SmolVLA / π0.7) and a new Future Sight Expert sit on top of a single shared VLM backbone. One forward pass through the VLM dispatches to both heads: the Action Expert emits the anchor action chunk; the Future Sight Expert emits `z_g` directly in the WM's latent space.
3. **Anchor + noise sampling** — the Action Expert generates one anchor trajectory; K perturbations are scored against `z_g` via direct (or learned-norm) distance.

## The four-step recipe

**Step 0 — mine failure trajectories.** Source: not-fully-trained imitation policies (early checkpoints) rolled out in simulation. Augment with policies tested in OOD scenes (held-out objects, novel layouts), action-noise-perturbed rollouts of trained policies, and failed autonomous deployments where available. The goal is a dataset that *covers the failure modes the deployed policy will encounter*, not just early-training failure modes.

**Step 1 — train encoder + WM.** LeWM-style ViT-tiny encoder + JEPA predictor + SIGReg, trained on `successful_imitation ∪ mined_failures`. The encoder learns dynamics-relevant features across both regions; the predictor learns valid forward dynamics across both. Outputs: a frozen encoder and a frozen predictor in a 192-dim latent space.

**Step 2 — train the Future Sight Expert.** A second flow-matching head added to the VLA's existing VLM backbone, sitting alongside the Action Expert. One VLM forward pass conditions both heads. Input (to the VLM): `(observation, goal_text)`. Output of the Future Sight Expert: target latent `z_g` in the same space as the encoder from Step 1. **Horizon: chunk-end** — `z_g` regresses to the encoded latent of the frame at offset `chunk_size` from the anchor observation (i.e., the observation that would follow the *last* action of the chunk the Action Expert emits). This ties Future Sight's prediction horizon to the Action Expert's planning horizon, makes training and inference horizons identical, and works on any imitation dataset without episode-success annotations. Training signal: contrastive on `(goal_text, z_{t + chunk_size})` pairs — positives are real (goal, chunk-end-latent) matches, negatives are shuffled. Optionally bootstrapped online by a frozen VLM scoring observed chunk-end frames. Joint training with the Action Expert is straightforward: `L = L_action + λ · L_future_sight` over the shared backbone.

**Step 3 — runtime: shared VLM + two experts + WM in an MPC inner loop.** One VLM forward pass produces both the anchor `a*` (from the Action Expert) and the target `z_g` (from the Future Sight Expert). Sample K perturbations `a*_k = a* + ε_k`. Roll each through the WM: `ẑ_k = WM_H(z_t, a*_k)`. Score: `s_k = d(ẑ_k, z_g)`. Execute the first action of `argmin_k s_k`; replan next step.

## Architecture

```
─────────────  TRAINING  ─────────────────────────────

  successful demos ─┐
                    ├──► combined dataset ──► Step 1: encoder + WM (LeWM-style)
  mined failures   ─┘                                │
                                                     │ frozen encoder + predictor
                                                     ▼
                              Step 2: Future Sight (flow-matching VLA)
                                       (obs, goal_text, z_t) → z_g
                                       contrastive loss

─────────────  RUNTIME (each control step)  ─────────

   obs_t ──┬─► encoder (LeWM ViT-tiny) ──► z_t
           │                                 │
           │      goal_text                  │
           ▼          │                      │
        ┌──────────────────────────┐         │
        │   VLM backbone (shared)  │         │
        └────┬────────────────┬────┘         │
             │                │              │
             ▼                ▼              │
      ┌──────────────┐ ┌──────────────────┐  │
      │Action Expert │ │ Future Sight     │  │
      │(flow-match)  │ │ Expert           │  │
      │              │ │  (flow-match)    │  │
      └──────┬───────┘ └────────┬─────────┘  │
             │                  │            │
             ▼ anchor a*         ▼ z_g       │
             │                  │            │
        a*_k = a* + ε_k          │            │
        k = 1..K                │            │
             │                  │            │
             ▼                  │            │
       ┌──────────────┐          │            │
       │  WM (× K)    │ ◄────────────────────┘
       │z_t,a*_k → ẑ_k│          │
       └──────┬───────┘          │
              │                  │
              ▼ K × ẑ_k          │
              │                  │
          d(ẑ_k, z_g) ◄──────────┘
              │
              ▼ K scores
           argmin
              │
              ▼
   execute first action of a*_{k*};
   replan on next observation
```

## Components

### Encoder + WM (Step 1)

LeWM-style: ~5 M ViT-tiny encoder + ~10 M JEPA predictor + SIGReg. Trained on `successful + failure` mixture. The failure-data inclusion is what makes this distinct from a standard JEPA pretrain — without it, the predictor is only valid in the narrow band of trajectories the imitation policy actually produces, and any trajectory perturbation that wanders outside that band crashes prediction quality. With it, the predictor stays valid in the regions where MPC rollouts will actually probe the dynamics.

### VLM backbone with two flow-matching experts (Step 2)

Architecturally identical to π0.7's "VLA + same-architecture high-level policy" pattern, but collapsed onto a single backbone: one VLM, two heads. Both experts are flow-matching; they differ only in output space.

- **Action Expert** — the existing [SmolVLA](../entities/smolvla.md) / [π0.7](../entities/pi0.7.md) action expert, conditioned on the VLM's output. Produces one trajectory per inference call (the *anchor*). Drop-in with the existing flow-matching primitive.
- **Future Sight Expert** — new. Conditioned on the same VLM output. Emits `z_g` in the WM's latent space (192-dim, LeWM-shaped). Same flow-matching machinery as the Action Expert; only the output projection and the loss differ.
- **Shared VLM backbone** — one expensive forward pass per VLM tick, dispatched to both heads in parallel. The amortization is what makes the architecture cheap: the marginal cost of Future Sight on top of an Action Expert is one extra flow-matching head, not a second VLM.

Cadence: the VLM backbone runs at action-chunk rate (~1 Hz with H = 50 actions and 50 Hz control), refreshing both anchor `a*` and target `z_g` together. The inner loop (perturb + WM rollout + score) runs per control step against the cached pair.

### Scorer

`d(ẑ_k, z_g)`. Simplest: direct L2 in latent space. Better (probably): a small learned norm — an MLP or 1–2 cross-attention layers over `(ẑ_k, z_g)` outputting a scalar. Far simpler than the [VLM-async](./vlm-async-implicit-wm.md) Cost transformer because both inputs live in the *same* space; no cross-space bridging.

## Data flow per control step

The VLM backbone fires at chunk-boundary cadence (~1 Hz); the inner loop fires at control rate (≥10 Hz). Both pipelines run in the same step at chunk boundaries; only the inner loop runs in between.

**At a chunk boundary** (VLM tick):
1. Read observation `o_t`. Encode → `z_t` (LeWM encoder, parallel pipeline).
2. VLM backbone forward pass on `(o_t, goal_text)`. Dispatch to:
   - Action Expert → anchor `a*`
   - Future Sight Expert → target `z_g`
3. Cache `a*` and `z_g` for the inner loop.

**Each control step** (inner loop):
4. Read `z_t` (re-encode current observation). Let `i` = steps since the most recent VLM tick (`i = 0` at chunk start, `i < chunk_size` always).
5. Sample K perturbations of the *remaining* chunk suffix: `a*_k = a*[i:] + ε_k`, `k = 1..K`. (`a*` is cached from the most recent VLM tick.)
6. For each `k`, roll WM forward by `chunk_size − i` steps from `z_t`: `ẑ_k = WM_{chunk_size − i}(z_t, a*_k)`. The rollout horizon shrinks within a chunk so all `ẑ_k` land at the same wall-clock frame as `z_g`.
7. Score: `s_k = d(ẑ_k, z_g)`. (`z_g` is cached and points at frame `t_anchor + chunk_size`.)
8. Execute first action of `a*_{argmin s_k}`. Re-plan next step.

## Operating modes

The architecture **degrades gracefully**. Three modes, each a strict subset of the next; the system can switch between them at runtime.

1. **Mode 1 — Action Expert only** (no Future Sight, no WM, no scorer). Only the Action Expert head fires; the Future Sight Expert head is dormant. The VLM forward pass and the Action Expert head are *bit-identical* to stock [SmolVLA](../entities/smolvla.md) — same code path, same compute. Used at deployment day 1 (before Future Sight + WM are trained), as a fallback when the WM mispredicts catastrophically, and whenever edge compute is too tight for the inner loop.
2. **Mode 2 — full Future Sight loop** (both heads + WM + scorer). The full system described above. K perturbations of the anchor, K WM rollouts, distance to `z_g`, argmin. Lookahead, re-rankability, OOD recovery.
3. **Mode 3 — distilled** (Action Expert only, trained on Mode-2 winners). At inference, only the Action Expert head fires; Future Sight + WM + scorer are dropped from the runtime path. Same wall-clock as Mode 1, capability of Mode 2.

The cost of Future Sight is opt-in. Drop the second head and the system is SmolVLA. Add it and you get an MPC. Distill once everything is stable and the MPC vanishes back into the Action Expert. Each step is independent — the architecture doesn't force a commitment up front.

Compared to the [VLM-async synthesis](./vlm-async-implicit-wm.md)'s Mode 1 (which still runs the VLM at *some* cadence to produce `context_embed` for the action policy), Future Sight's Mode 1 is *literally* stock SmolVLA — no architectural delta. The shared backbone means the second head can be added or omitted at the dispatch level, not the model-loading level.

## Mapping to the implicit-WM contract

| Slot         | Component                                                |
| ------------ | -------------------------------------------------------- |
| Perception   | LeWM-style encoder (for `z_t`); VLM backbone (for semantics) |
| Policy       | Action Expert head on shared VLM (anchor)                |
| World Model  | LeWM predictor (× K rollouts)                            |
| Cost         | `d(ẑ_k, z_g)` — direct or learned norm                   |
| Optimization | Anchor + K perturbations + argmin                        |

Goal-target generator is the **Future Sight Expert head on the same VLM** — it's not a slot in the original contract, but it's what makes the Cost slot computable in the WM's space.

All slots fill components sharing one latent space (the WM's). The `(Action Expert → WM)` link uses actions; the `(WM, scorer, Future Sight Expert)` triangle is fully in `z`-space.

## Why single-latent beats cross-space

The [VLM-async synthesis](./vlm-async-implicit-wm.md) carries two latent spaces (VLM tokens + WM latent) and a learned bridge (Cost transformer with cross-attention). This proposal collapses that to one space: Future Sight emits `z_g` directly in the WM latent, and the scorer is a same-space distance.

Architectural consequences:

- **One fewer learned bridge.** The Cost transformer is replaced by a (possibly identity) distance, or a small learned norm. Smaller surface area for training and debugging.
- **No alignment problem.** Cross-space comparisons require the bridging head to align two distinct geometric spaces. Same-space comparisons inherit the geometry the WM already learned.
- **Cleaner CEM.** Score gradients and trajectory ranking are in the WM's intrinsic geometry — the same one the predictor was trained to be smooth in.

The cost: a new flow-matching head (Future Sight Expert) trained on top of the same pretrained VLM backbone the Action Expert already uses. Roughly comparable to adding π0.7's same-architecture high-level subtask policy — well-trodden in the π-family lineage. The backbone is shared and reused; the marginal addition is one head and its loss term.

The central trade-off: *training a new flow-matching head on a shared backbone* vs *training a cross-space bridge between two pretrained models*. Both are doable; the former produces a single coordinated stack, the latter produces a coupling between two pretrained models.

## Failure data sourcing (Step 0)

Not-fully-trained imitation policies are a starting point but not sufficient. They produce *training-time* failure modes (gripper-late, premature-terminate, overshoot), which are not necessarily the *deployment-time* failure modes (novel objects, occlusion, OOD textures, distractor clutter).

A robust Step-0 mix:

1. **Early-checkpoint policies** in nominal scenes — under-trained skill spectrum.
2. **Trained policies in OOD scenes** — held-out objects, novel layouts, lighting changes. Deployment-distribution failures.
3. **Action-noise-perturbed rollouts** of trained policies — perturbation-recovery dynamics.
4. **Failed autonomous deployments** where available — real failure modes from real distributions.

Each source contributes a *region* of the failure manifold; the WM is dynamics-complete only over the union. Mixing ratios are an empirical question — likely roughly proportional to how much of the deployment distribution each source covers.

## Future Sight training (Step 2)

**Horizon — committed: chunk-end.** `z_g` regresses to the encoded latent of the frame at offset `chunk_size` from the anchor observation (e.g., 50 frames out at chunk_size=50). The target is the observation that would follow the *last* action of the chunk the Action Expert just emitted. Three reasons this is the right choice over either episode-end or arbitrary fixed-horizon:

1. **Training and inference horizons match exactly.** At inference, the WM rolls forward up to `chunk_size` steps from `z_t` using the anchor's actions. Training Future Sight to predict at the same offset means the scorer compares `WM(z_t, a*_k)` to `z_g` over identical horizons — no train/test horizon shift, no smearing from variable success times.
2. **Tied to the Action Expert's planning horizon.** Future Sight predicts the consequence of executing the chunk the Action Expert just emitted. The two heads are predicting compatible quantities (action sequence vs. its outcome), which is what makes anchor-vs-perturbation scoring meaningful.
3. **No episode-success annotations needed.** Every imitation frame has a frame `chunk_size` later (modulo episode boundaries). Trains on any LeRobot dataset; no metadata gate, in contrast with episode-end variants which require success-judgment frames (π0.7 has them; LeWM and stock LeRobot do not).

Trade-off: `z_g` represents progress over one chunk, not the final goal state. For long-horizon tasks the goal-text-conditioning is what carries semantic intent across chunks; `z_g` carries the local "where am I one chunk from now" signal that the WM can actually verify. This two-tier separation (semantics in `goal_text`, local geometry in `z_g`) is the load-bearing decision behind this architecture — and it's exactly what cleanly maps onto the existing chunked-action structure of SmolVLA / π0.7.

The training signal is the next decision. Three options for how Future Sight regresses to `z_{t + chunk_size}`:

1. **BC from chunk-end frames.** Future Sight regresses directly to the encoded frame at `t + chunk_size` from successful trajectories. Simple, but Future Sight ends up learning *what this policy would achieve over one chunk*, not *what the goal demands*. Outputs are policy-shaped.
2. **Contrastive on (goal_text, z_{t+chunk_size}) pairs.** Positives are real (goal, chunk-end-latent) matches; negatives are shuffled chunk-end latents from other goals. Same approach the VLM-async synthesis proposes for its Cost transformer, lifted to Future Sight. Most likely the right starting point.
3. **VLM-supervised online correction.** A frozen VLM scores observed chunk-end frames w.r.t. goals; Future Sight regresses to whichever chunk-end latents the VLM rates highly. Bootstraps off (2); corrects for distribution shift between demonstrations and deployment.

Likely path: train (2) offline; bolt on (3) online for calibration.

## Distance metric: direct vs learned

Direct L2 in 192-dim treats every dimension equally. The LeWM latent compresses both task-relevant (object pose, gripper state) and task-irrelevant (background lighting, gripper-self-image) features. A trajectory that nails object pose but misses lighting scores worse than one that nails lighting but misses pose.

Mitigations:

- **Small learned norm**: an MLP on `(ẑ_k, z_g)` outputting a scalar. Far smaller than the VLM-async Cost transformer because there's no cross-space bridging — just a learned weighting over the existing 192 dims. ~100 K params should suffice.
- **Goal-conditional scaling**: `d(ẑ, z_g) = ‖W(z_g) ⊙ (ẑ − z_g)‖` where `W` is a small head producing a per-dimension weight from the goal latent. Captures "for *this* goal, these dimensions matter."
- **Per-task fine-tuning**: train a per-task scorer in the few-shot regime once Future Sight is stable.

The simplest sufficient scorer is probably the goal-conditional scaling — a few thousand parameters, trained jointly with Future Sight (or as a separate stage), capturing the load-bearing variation while staying same-space.

## Anchor + noise: local search and escape mechanisms

K perturbations of one anchor explore *locally*. If the anchor commits to the wrong sub-skill (wrong hand, wrong object), no perturbation recovers — every `ẑ_k` lives near the anchor's terminus. This is the cost of cheap sampling.

Mitigations:

1. **Multi-anchor**: sample 2 (or 3) anchors per step, each with K/2 perturbations. Pure CEM at small scale.
2. **Score-floor escape**: if `min_k s_k > τ` (all candidates bad), discard the batch and resample the anchor with a different noise seed or higher entropy.
3. **Hierarchical CEM**: anchor + K_1 perturbations on a short horizon → top-2 → K_2 perturbations on longer horizons. Compounds with multi-anchor.
4. **Anchor diversification via VLA noise**: increase the flow-matching sampling temperature on the anchor itself (rather than perturbing post-hoc), gaining anchor-level diversity at the cost of anchor quality.

Practical first cut: 1 anchor, K=8 perturbations, score-floor escape. If escapes are frequent, move to multi-anchor.

## VLM-async vs Future Sight

|                                  | [VLM-async](./vlm-async-implicit-wm.md)         | Future Sight (this synthesis)                       |
| -------------------------------- | ----------------------------------------------- | --------------------------------------------------- |
| Number of latent spaces          | 2 (VLM tokens, WM latent)                       | 1 (WM latent)                                       |
| Backbone topology                | Two separate models (VLM + action VLA)          | One shared VLM with two flow-matching expert heads  |
| Goal representation              | `context_embed` (slow VLM output)               | `z_g` (Future Sight Expert output)                  |
| Goal cadence                     | ~1 Hz                                           | ~1 Hz (refresh-with-anchor) or per-step             |
| Bridge (policy goal → WM)        | Cost transformer (cross-attention)              | Direct distance / small learned norm                |
| Sampling                         | K independent flow-matching draws               | 1 anchor + K perturbations                          |
| Reuses                           | SmolVLA action expert; LeWM; pretrained VLM     | SmolVLA action expert + VLM; LeWM                   |
| New components to train          | Cost transformer                                | Future Sight Expert head + (optional) learned norm  |
| Failure-data inclusion           | Implicit (encoder same)                         | **Explicit (Step 0 mining)**                        |

Both are coherent end states. Future Sight wins on *architectural simplicity* (one VLM, one latent space, one inner-loop scoring function) and *coordination* (the two experts share a representation by construction). VLM-async wins on *pretraining reuse* (no new generator head; the Cost transformer is the only new component) and *modular swap* (the VLM and the action policy can be replaced independently).

## Composition with existing entities

- **[LeWorldModel](../entities/le-world-model.md)** — provides the encoder, predictor, and JEPA training recipe. Step 1 is exactly LeWM training extended to include failure data.
- **[π0.7](../entities/pi0.7.md)** — provides both the architectural template (a VLM backbone with multiple flow-matching heads on top — π0.7 has one for actions; Future Sight adds a second for target latents) and the metadata-tagged dataset structure (success/failure annotations).
- **[SmolVLA](../entities/smolvla.md)** — alternative VLA template if compute is tighter.
- **[Implicit World Modeling](../entities/implicit-world-modeling.md)** — the contract this fills. Future Sight is the first concrete instance where the goal encoder is a *trained generator in the WM's latent space* rather than an inherited VLM.

## Open questions

- **Future Sight training signal**: contrastive (option 2) is the working assumption; how much does VLM-supervised online correction (option 3) add?
- **Distance metric**: how much does the learned norm matter? Direct L2 baseline first, then learned norm as ablation.
- **Anchor diversity**: at what K does anchor + noise hit diminishing returns vs multi-anchor or pure CEM?
- **Failure-data ratio**: what fraction of the encoder-training dataset should be failures? 50/50? 10/90? Empirical sweep needed.
- **Joint training schedule**: weight λ on `L_future_sight` relative to `L_action`; whether to alternate epochs or train heads simultaneously. Knowledge-Insulation-style separation (Action Expert gradient doesn't flow into VLM) may or may not transfer to the Future Sight head.
- **Latent expressiveness**: are there tasks where the LeWM-style latent strips information that Future Sight would need to disambiguate goals (e.g., "set the table" — many valid configurations)?

## Building and testing on LeRobot + LeWM

Mirrors the VLM-async three-phase plan, with substitutions for Future Sight components.

### Component → code mapping

| Component                | Existing code                                                   | Modification                                                                          |
| ------------------------ | --------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Failure mining (Step 0)  | LeRobot rollout harness                                         | Run early-checkpoint SmolVLA on LIBERO + OOD variants; log failures explicitly.        |
| Encoder + WM (Step 1)    | LeWM repo                                                       | Pretrain on success+fail mix; report JEPA prediction MSE on held-out failures.        |
| VLA action expert        | LeRobot SmolVLA action expert                                   | Reuse. Sample with elevated noise scale for anchor diversity.                          |
| Future Sight Expert (Step 2) | New flow-matching head added to SmolVLA's existing VLM backbone | Train contrastively on `(goal_text, z_{t+chunk_size})` pairs (chunk-end horizon); jointly with the Action Expert via `L = L_action + λ · L_future_sight`. |
| Scorer                   | New: `d(ẑ, z_g)` — start with L2; add learned norm as ablation  | ~100 K params for the learned norm.                                                    |
| Inner loop (CEM)         | LeWM CEM-over-latents code                                      | Replace LeWM's policy with anchor + K perturbations from the SmolVLA expert.           |
| Eval harness             | LeRobot — LIBERO + SO-101 wrappers                              | None.                                                                                  |

### Build in three phases

1. **Mode 1 (anchor only).** Wire `VLA → anchor → execute`. Match stock SmolVLA on LIBERO. Time: ~2 weeks.
2. **Mode 2 (full Future Sight loop).** Pretrain encoder + WM on success+failure mix; train Future Sight contrastively; wire `anchor → K perturbations → WM rollout → score → argmin`. Eval on LIBERO at delays {0, 1, 2, 3, 4} steps — same matrix as VLM-async, expecting flat SR vs SmolVLA's degradation. Time: ~6 weeks.
3. **Mode 3 (VLM-supervised correction).** Frozen VLM scores observed endpoints; Future Sight is fine-tuned online. Measure SR drift over 10-hour SO-101 deployment. Time: ~2 weeks.

### Concrete first milestones

- **Week 1**: implement Step-0 failure miner; produce a 50/50 success+failure dataset on a 100 K-frame LIBERO slice.
- **Week 2**: pretrain encoder+WM on that mix; report JEPA MSE on held-out successes vs failures (should be comparable; if predictor is *worse* on failures, mining ratio is wrong).
- **Week 3**: train Future Sight contrastively on `(goal_text, z_{t+chunk_size})` pairs. Report top-1 retrieval accuracy on held-out (goal, chunk-end-latent) pairs.
- **Week 4**: stand up the inner loop with K=8 perturbations and L2 distance. Verify ≈parity with stock SmolVLA on in-distribution LIBERO.
- **Week 5+**: enable mode 2; run delay sweep; ablate (a) failure-data ratio (10/90, 50/50, 90/10), (b) learned-norm vs L2, (c) K, (d) anchor vs multi-anchor.

Whole prototype fits a single laptop with an RTX 4090 for sim; SO-101 stage adds a Jetson Orin. No proprietary code or weights required.

## See also

- [VLM-Async Implicit World Modeling](./vlm-async-implicit-wm.md) — sister architecture using a cross-space Cost transformer instead of single-latent target prediction.
- [Implicit World Modeling via the π-family and LeWorldModel](./implicit-wm-via-pi-and-lewm.md) — earlier composition sketch.
- [World Models for Physical AI](./world-models-for-physical-ai.md) — the four-axis taxonomy this synthesis instances ((1) + (2) + (3) hybrid via failure-mining + WM + Future Sight).
- [Implicit World Modeling](../entities/implicit-world-modeling.md) — the contract.
- [LeWorldModel](../entities/le-world-model.md) — encoder + WM.
- [π0.7](../entities/pi0.7.md) / [SmolVLA](../entities/smolvla.md) — VLA templates.
