# SawSeenVLAWM — SawSeenVLA + le-wm pathways

The `sawseenvlawm` policy hosts two ways to feed le-wm features into the
training signal, controlled by independent flags:

1. **le-wm visual side-channel** — prepends encoder tokens to the action
   expert's suffix. Implemented and ablated; **parked** after the 1k-step
   runs all clustered with the no-lewm baseline (see "Empirical results
   (parked)" below). Code stays in place.
2. **Latent Goal Expert (Phase A — active)** — a second flow-matching
   expert sitting layer-by-layer next to the action expert on the shared
   VLM backbone, trained to predict `z_{t+chunk_size}` in le-wm space
   given `(prefix, language goal, z_t anchor)`. Action-blind by
   construction. See the [Latent Goal Expert](#latent-goal-expert-phase-a--active)
   section for the full architecture, and
   [`design/future-sight-implicit-wm.md`](./future-sight-implicit-wm.md)
   for the broader single-latent implicit-WM synthesis.

The two pathways are independent — you can enable either, both, or
neither via separate config flags.

## Why

SawSeenVLA's only visual pathway is the SmolVLM2 prefix: images go through the
frozen SigLIP encoder, get tokenised, and reach the action expert via
cross-attention layered through SmolVLM. The action expert never sees a vision
representation tailored to the manipulation domain.

[le-wm](https://github.com/iasawseen/le-wm) trains a small JEPA world model
(ViT-Tiny encoder + autoregressive predictor) directly on Libero rollouts. The
encoder learns features that are predictive of next-frame embeddings under the
robot's actions — a signal that's plausibly useful for action selection.

**Hypothesis:** feeding le-wm features directly into the action expert (in
addition to whatever SmolVLM passes through) improves the action expert's
ability to ground actions in scene structure, especially for in-distribution
tasks (Libero, RoboCasa).

## Design principles

1. **Vanilla SawSeenVLA stays untouched.** The lewm logic lives in a
   separate policy package (`src/lerobot/policies/sawseenvlawm/`), cloned
   from sawseenvla and registered as `--policy.type=sawseenvlawm`. The
   sawseenvla files keep zero lewm references — vanilla is your reference.
2. **Mirror the smolvla→sawseenvla pattern.** SawSeenVLA itself was a
   structural clone of SmolVLA (separate registration, separate config
   class) so iteration on it doesn't perturb upstream. SawSeenVLAWM
   continues the same convention one level deeper.
3. **Switch by Makefile.** `sawseenvla.mk` runs `--policy.type=sawseenvla`;
   `sawseenvlawm.mk` runs `--policy.type=sawseenvlawm` and sets
   `--policy.lewm_encoder_path=/lewm/<ckpt>` (host-mounted at `/lewm`).
4. **Checkpoint isolation.** A SawSeenVLAWM checkpoint can't be loaded as
   sawseenvla (and vice versa) because draccus dispatches on the policy
   `type` key. This is intentional — they are different graphs.
5. **Run-time toggle within the WM policy.** `lewm_encoder_path=None`
   short-circuits the encoder + projection construction inside
   SawSeenVLAWM, so you can also build a structurally vanilla
   SawSeenVLAWMPolicy if you want a "WM-class but no WM signal" baseline.

## Architecture

```
                 ┌──────────────────────────── prefix (frozen, KV cached) ──┐
   images ──────►│  SigLIP → SmolVLM2 layers (16) → KV cache                │
   language ────►│  state → state_proj                                      │
   state ───────►│                                                          │
                 └────────────────────────────────┬─────────────────────────┘
                                                  │ cross-attention from suffix
                                                  ▼
                 ┌─────────────────── suffix (action expert, 12 layers) ───┐
   le-wm ───►│ lewm_proj   ◄─── new tokens, prepended to suffix             │
   tokens   │     │                                                         │
            │     ▼                                                         │
            │  ┌─────────┐  ┌──────────────┐                                │
            │  │ lewm    │  │ noisy_action │                                │
            │  │ tokens  │  │ + time_emb   │                                │
            │  └────┬────┘  └──────┬───────┘                                │
            │       └──────────────┴────────► action expert ► v_t           │
            └─────────────────────────────────────────────────────────────┘
   img(s) ──────────► le-wm encoder (frozen) ────► (B, num_tokens, 192)
                                                                            
```

The le-wm encoder is **frozen ViT-Tiny** trained by JEPA on Libero. It produces
257 tokens per image (1 CLS + 256 patches at 224/14). We slice the first
`num_tokens` (192 by default), project them to the action expert's hidden
size (720 = 960 × 0.75), and prepend them to the suffix sequence.

The lewm tokens are prepended to the action expert's **suffix** (not the
VLM's prefix): the question we're answering is "what extra signal does
*the action expert* get?", not "what extra signal does the VLM get?". A
prefix-injection variant was implemented and ablated alongside suffix
injection; both clustered with the no-lewm baseline (see Empirical
results below) and only the suffix path was retained in the codebase.

### Attention mask design (within suffix)

Existing SmolVLA convention sets `att_mask=1` for every action token,
yielding a *causal* pattern over the chunk. We prepend lewm tokens with
`att_mask = [1, 0, 0, ..., 0]` so:

* All lewm tokens form a single bidirectional block among themselves.
* Every action token (causal) can attend to every lewm token (since they
  precede it).
* lewm tokens don't attend to actions (they have lower cumulative-mask
  rank).

This matches how the prefix already treats image + language tokens (one
input block) versus state/action (each its own causal block).

## Config surface

Five new fields on `SawSeenVLAConfig`:

| Field | Default | Notes |
|---|---|---|
| `lewm_encoder_path: str \| None` | `None` | When `None`, lewm is disabled. When set to a `<name>_object.ckpt`, weights are loaded once at policy construction. |
| `lewm_freeze: bool` | `True` | If `False`, encoder parameters become trainable (we still keep `lewm_proj` always trainable). |
| `lewm_num_tokens: int` | `192` | Slice of `last_hidden_state[:, :num_tokens]`. `1` ≈ CLS-only. Max = `(image/patch)² + 1 = 257`. |
| `lewm_image_size: int` | `224` | Must match training resolution; encoder bilinearly resizes inputs to this. |
| `lewm_patch_size: int` | `14` | Must match training. |

## Token math

Per camera: `num_tokens = 192` (CLS + 191 patches).
Suffix length grows from `chunk_size = 50` to `chunk_size + num_cams × num_tokens`.

| Setting | Cameras | Suffix length | vs vanilla |
|---|---|---|---|
| Vanilla | n/a | 50 | 1.0× |
| Libero (default) | 2 (agentview + wrist) | 50 + 384 = 434 | 8.7× |
| RoboCasa | 3 | 50 + 576 = 626 | 12.5× |

Action-expert self-attention is `O(suffix²)`, so naive throughput cost is
~75–150× in the action expert's attention layers. Action expert is only ~12
layers at hidden 720, so end-to-end training step time roughly **doubles**
(SmolVLM prefix + KV-cached cross-attn dominate at small chunk sizes).

If that's too slow, the cheap knobs are:
* `lewm_num_tokens=1` → CLS-only, suffix = 50 + num_cams.
* Spatial pool to `k` tokens (would require encoder forward change).

## Code touchpoints

| Path | Status | Purpose |
|---|---|---|
| `src/lerobot/policies/sawseenvlawm/__init__.py` | new (cloned) | Re-exports `SawSeenVLAWMConfig`, `SawSeenVLAWMPolicy`, `make_sawseenvlawm_pre_post_processors`. |
| `src/lerobot/policies/sawseenvlawm/configuration_sawseenvlawm.py` | new (cloned + 5 fields) | `SawSeenVLAWMConfig` registered as `"sawseenvlawm"`; identical to `SawSeenVLAConfig` plus `lewm_encoder_path`, `lewm_freeze`, `lewm_num_tokens`, `lewm_image_size`, `lewm_patch_size`. |
| `src/lerobot/policies/sawseenvlawm/modeling_sawseenvlawm.py` | new (cloned + lewm hooks) | `SawSeenVLAWMPolicy` (registered name `"sawseenvlawm"`); `VLAFlowMatching` instantiates `LeWMVisionEncoder` + `lewm_proj` when configured. New `compute_lewm_tokens(images)` runs the encoder per camera. `embed_suffix(..., lewm_tokens=None)` prepends them with the right attention mask. `forward` / `sample_actions` compute lewm tokens once and thread through. `denoise_step` accepts and passes through. |
| `src/lerobot/policies/sawseenvlawm/processor_sawseenvlawm.py` | new (cloned, no logic change) | `make_sawseenvlawm_pre_post_processors` mirrors the sawseenvla processor exactly. |
| `src/lerobot/policies/sawseenvlawm/lewm_encoder.py` | new | `LeWMVisionEncoder` wraps a stripped-down HF `ViTModel`; `from_lewm_checkpoint` loads weights from a pickled le-wm `JEPA` object. |
| `src/lerobot/policies/factory.py` | edit | Adds `"sawseenvlawm"` branches in `get_policy_class`, `make_policy_config`, `make_pre_post_processors`. |
| `src/lerobot/policies/__init__.py` | edit | Re-exports `SawSeenVLAWMConfig`. |
| `src/lerobot/policies/sawseenvla/*` | **untouched** | Vanilla reference. |
| `sawseenvlawm.mk` | new | Clone of `sawseenvla.mk` using `--policy.type=sawseenvlawm`, with `--policy.lewm_encoder_path=/lewm/<ckpt>` and a host-mounted `LEWM_HOST_DIR ?= $(HOME)/.stable-wm/libero`. |

## Switching back to vanilla

* **By Makefile:** `make -f sawseenvla.mk train` runs vanilla SawSeenVLA; `make -f sawseenvlawm.mk train` runs the WM variant. They are *different policy types*, not one with a flag.
* **At eval time:** load the vanilla checkpoint with vanilla policy type; load the WM checkpoint with WM policy type. `from_pretrained` dispatches on `type` so they cannot be cross-loaded — by design.
* **Within SawSeenVLAWM:** setting `lewm_encoder_path=None` (or omitting the flag) builds a SawSeenVLAWMPolicy whose graph matches vanilla SawSeenVLA architecturally — useful as a "same class, no signal" control. But the saved type is still `sawseenvlawm`.

## Risks and open questions

| Risk | Mitigation |
|---|---|
| **Domain mismatch:** le-wm trained on Libero, but RoboCasa scenes/objects/lighting differ. Frozen encoder may produce useless features. | Start on Libero (matches training distribution); ablate on RoboCasa later. If poor transfer, fine-tune encoder (`lewm_freeze=false`) or skip the WM path on RoboCasa. |
| **Throughput collapse:** suffix grows 8–12×. | Track step time vs vanilla in the smoke run. Have `lewm_num_tokens=1` (CLS-only) as a backup — recovers near-vanilla speed. |
| **Projector applied to CLS only:** le-wm's MLP projector was trained on CLS during JEPA training. The LGE / Mode 3 / MPC pipeline routes through `lewm_encoder.encode_cls()` which applies `projector` — same space the predictor was supervised against. The lewm side-channel path (`compute_lewm_tokens`) keeps raw patch tokens (projector on patches would be OOD); the trainable `lewm_proj` handles the dynamic-range gap there. |
| **Image-size mismatch:** SawSeenVLA's `resize_imgs_with_padding=(512,512)` defaults vs le-wm's 224. | Encoder bilinearly resizes (with antialias) to 224. May lose detail; if needed, drop SawSeenVLA's resize for the lewm path only. |
| **bf16 autocast on a frozen ViT:** keeping it in fp32 is safer for numerical stability of frozen weights. | Encoder runs under `torch.no_grad()` when frozen and casts inputs to its parameter dtype (`fp32` by default). |
| **Multi-camera order matters.** | Python dict insertion order in `present_img_keys` is stable. Document the camera ordering in the run config. |

## Validation plan

1. **Smoke test (host).** Load policy with `lewm_encoder_path=…/lewm_epoch_10_object.ckpt`, run a forward + sample_actions on a dummy batch (1 sample, 2 cameras), assert no NaN, assert output shape `(1, chunk_size, action_dim)`.
2. **Libero short run (Docker).** 5k steps, vanilla vs WM, compare loss curves and step time. Pass criterion: WM-loss within 5% of vanilla and step time ≤ 2× vanilla.
3. **Libero full run.** 50k–96k steps, compare eval success rate on libero_spatial / object / goal / 10. Pass criterion: WM ≥ vanilla on at least 2 of 4 task suites.
4. **RoboCasa365 transfer (optional).** Same recipe, target/atomic split. If domain mismatch tanks WM, repeat with `lewm_freeze=false`.

## Future work (not in v1)

* **Cross-attention adapter (Option C):** instead of suffix tokens, add a per-block FiLM / cross-attn layer in the action expert that conditions on a pooled lewm vector. Smaller token-budget cost, but new params.
* **Train le-wm on RoboCasa.** The encoder is small (~5M params); a domain-matched encoder may dominate the Libero one.
* **Use le-wm's predictor too.** Currently we only lift the encoder + (dropped) projector. The predictor's autoregressive embeddings are richer; could add as another suffix block.

---

## Empirical results (parked, 2026-05-09)

1k-step ablations on libero @ bs=24, LR=2.5e-4, sweeping lewm side-channel
configurations (token count, freeze, per-cam vs concat-cameras input, and
prefix-vs-suffix injection — the prefix injection variant has since been
removed from the codebase, see note below). **None beat the no-lewm
baseline.**

| variant                           | step rate     | GPU mem | step 950 loss |
|-----------------------------------|--------------:|--------:|--------------:|
| lewm=0 (no encoder)               | 1.62 step/s   |  8.5 GB | 0.655         |
| lewm=1 frozen, suffix, per-cam    | 1.55 step/s   |  8.5 GB | **0.640**     |
| lewm=1 frozen, suffix, concat     | 1.61 step/s   |  6.9 GB | 0.642         |
| lewm=1 unfrozen, suffix, per-cam  | 1.42 step/s   |  8.2 GB | 0.639         |
| lewm=192 frozen, suffix, per-cam  | 1.10 step/s   | 19.3 GB | 0.656         |

(A `lewm=1 frozen, prefix, concat` variant was also run and landed at
loss 0.639 — equivalent to the suffix variants. The prefix-injection
codepath was dropped after the ablation since it added complexity without
distinguishing itself from suffix injection.)

Key observations:
* All lewm variants and the no-lewm baseline cluster within 0.017 of each other at step 950 — well within seed-to-seed noise.
* lewm=192 (patch tokens) is the only consistent loser earlier in training (steps 300–500 it sits at ~1.3 vs ~1.0 for the others). The action expert spends capacity learning to filter the OOD patch features, then catches up by step 950.
* Concat-camera vs per-camera input *did not* matter despite matching le-wm's training distribution exactly — runs are within 0.005 of each other throughout.
* Unfreezing did *not* help — the encoder doesn't learn anything useful in 1k steps.

### Hypothesis for the negative result (most likely first)

1. **The action expert already has rich vision via SmolVLM/SigLIP** (960-d × 16 layers). A frozen 192-d ViT-Tiny can't add information SmolVLM doesn't already extract.
2. **JEPA next-frame objective ≠ action selection.** Features predictive of "what comes next given an action" aren't necessarily features predictive of "what action to take given a state."
3. **The trainable `lewm_proj` (192→720) likely learns to suppress an unhelpful stream**, leaving the action expert to rely on the prefix attention.

### Status

* All scaffolding (policy class, makefile, encoder wrapper, smoke test) stays in place — vanilla SawSeenVLA is untouched and remains the production policy.
* Re-launching this experiment is one command: `make -f sawseenvlawm.mk train ...`.
* Worth revisiting if/when:
  * le-wm gets retrained on RoboCasa365 (domain match)
  * an adapter-style injection (Option C) is implemented
  * a longer training horizon is run (>5k steps) to test if the encoder helps only late in training

---

## Latent Goal Expert (Phase A — active)

A second flow-matching expert that sits **next to the action expert on the
shared SmolVLM backbone**, trained to predict the chunk-end frame's le-wm
latent `z_{t+chunk_size}` from `(prefix, language goal, z_t anchor)` —
explicitly *blind* to the action chunk being committed.

The broader motivation is in
[`design/future-sight-implicit-wm.md`](./future-sight-implicit-wm.md). This
section documents the SawSeenVLAWM-specific implementation.

### Why this is the right shape

- The side-channel ablations showed that *adding one more visual stream
  to the action expert* doesn't move action loss — the SmolVLM/SigLIP
  prefix already supplies rich vision.
- The Latent Goal Expert is a different bet: instead of feeding le-wm features to the
  action expert, train a *second head* on the same backbone to predict
  the **goal state** in le-wm space. The shared cost is one extra
  ViT-Tiny encoder pass per step (cheap); the marginal gain is a
  goal-target generator that maps language to le-wm geometry.
- This unlocks Phase B (MPC inner loop): given `z_g` from Latent Goal Expert and `z_t`
  from the encoder, score K perturbations of the action chunk by
  `d(WM(z_t, a*_k), z_g)` — both endpoints anchored in the same space.

### Architecture

```
                            VLM (shared, frozen)
                                  ▲ ▲
                                  │ │  cross-attn into VLM at every layer
                       ┌──────────┘ └──────────┐
                       │                       │
              ┌────────┴────────┐    ┌─────────┴───────────┐
              │  Action Expert  │    │  Latent Goal        │
              │  (existing,     │    │  Predictor (Latent Goal Expert)    │
              │   own weights)  │    │  (new, 720-d, 16L)  │
              └────────┬────────┘    └─────────┬───────────┘
                       │                       │
                  50 noisy actions      [ z_t_anchor , noisy_z_g + time ]
                                        (2-token Latent Goal Expert suffix)
                       │                       │
                action_out_proj          latent_goal_out_proj
                       │                       │
                v_action (B, 50, 32)     v_latent_goal (B, 192)
                       │                       │
                  L_action (MSE)         L_latent_goal (MSE)
                       └──────────┬────────────┘
                                  │
                       L = L_action + λ · L_latent_goal

  Action ↔ Latent Goal Expert attention: blocked in both directions.
    - Action → Latent Goal Expert:  cumulative-mask ordering (Latent Goal Expert sits after actions)
    - Latent Goal Expert → Action:  custom 2D-mask edit (att_2d_masks[latent_goal:, suffix:] = False)
```

Each expert has its **own weights, projections, depth, and width**. They
share *only* the VLM backbone (one VLM forward per step) — exactly the
"share only VLM" coupling style. Per-layer interleaving is identical to
the existing single-expert path: half the layers do self-attn (Q/K/V
concatenated and split per stream), the other half do cross-attn (each
expert reads VLM K/V via its own re-projection).

### Latent Goal Expert suffix structure

```
Latent Goal Expert suffix = [ z_t_anchor , noisy_z_g + time ]    # 2 tokens, both 720-dim

z_t_anchor  : latent_goal_anchor_proj( lewm_encoder.encode_cls(o_t) )
              # projector(CLS) of the current frame — the JEPA prediction space
              # — then mapped into Latent Goal Expert hidden. Frozen, deterministic —
              # the Latent Goal Expert's "where am I now."

noisy_z_g+t : latent_goal_time_mlp_out( silu( latent_goal_time_mlp_in( [latent_goal_in_proj(noisy_z), sin_cos_time] ) ) )
              # Standard flow-matching denoising token.
              # Velocity is read from this position only (anchor's output
              # is discarded — it has no flow-matching role).
```

Within the Latent Goal Expert suffix the two tokens share one attention block
(`att_mask=[1, 0]`) so they're bidirectional — the denoising token reads
the anchor at every layer, and vice versa.

### Action-blindness — the 2D-mask edit

The cumulative `att_mask` scheme would let Latent Goal Expert read action tokens (Latent Goal Expert sits
after actions in the suffix → higher cumsum → Latent Goal Expert sees actions). To enforce
"Latent Goal Expert predicts the goal state independent of the action chunk", we zero out
the Latent Goal Expert-rows × suffix-columns quadrant of the 2D attention mask after
building it the standard way:

```python
att_2d_masks = make_att_2d_masks(pad_masks, att_masks)      # cumulative
latent_goal_start = prefix_len + suffix_len                          # Latent Goal Expert tokens start here
att_2d_masks[:, latent_goal_start:, prefix_len:latent_goal_start] = False    # block Latent Goal Expert → suffix
```

So Latent Goal Expert attends to: prefix (image, language, state) plus its own anchor +
denoise tokens. Nothing in the action expert's suffix.

The reverse direction (action → Latent Goal Expert) is already blocked by cumulative
ordering — actions have lower cumsum than Latent Goal Expert tokens.

This makes the action expert and the Latent Goal Expert **conditionally
independent given the prefix**. Both branches use the same multimodal
context but never see each other.

### Why "predict goal independently of actions"

If Latent Goal Expert were action-conditioned, its `z_g` would partly reflect the policy's
own choices — the MPC scorer would then rank candidates against itself,
conflating "what the policy will do" with "what it should do." Making Latent Goal Expert
read only `(language goal, scene, robot state, z_t)` keeps `z_g` fixed
across the K action perturbations at inference and gives the scorer a
clean, action-independent target.

### Twin-experts wrapper

`src/lerobot/policies/sawseenvlawm/smolvlm_with_two_experts.py` —
`SmolVLMWithTwoExpertsModel(SmolVLMWithExpertModel)`. Adds a second
`lm_expert` (the Latent Goal Expert) with its own width / depth, and generalizes
the cross-attn dispatch from one expert to N via
`_forward_cross_attn_layer_n`. Falls back to the parent's single-expert
path when called with `inputs_embeds` of length ≤ 2 (e.g., inference Mode
1, where only the action expert fires).

Per-expert weight counts at the SawSeenVLA defaults
(`expert_width_multiplier=0.75`, `num_vlm_layers=16`):

| Expert  | Hidden | Layers | Trainable params |
|---------|-------:|-------:|-----------------:|
| Action  |    720 |     16 |             ~98M |
| Latent Goal Expert      |    720 |     16 |             ~98M |

Doubling the trainable parameter count vs vanilla sawseenvlawm
(when Latent Goal Expert is on) is a known cost.

### Config surface

Five new fields on `SawSeenVLAWMConfig`, plus one new value for the
existing `lewm_inject_to`:

| Field | Default | Notes |
|---|---|---|
| `latent_goal_enabled` | `False` | Master switch. Off → bit-identical to vanilla sawseenvlawm. |
| `latent_goal_loss_weight` | `1.0` | λ in `L = L_action + λ · L_fs`. |
| `latent_goal_loss_type` | `"bc"` | Phase A only: flow-matching MSE. `"contrastive"` reserved for a later ablation. |
| `latent_goal_num_steps` | `10` | Inference-time denoising steps for Latent Goal Expert (unused in Phase A). |
| `latent_goal_expert_width_multiplier` | `0.75` | Mirrors the action expert default; keeps the two heads symmetric. |
| `latent_goal_num_expert_layers` | `-1` | -1 = match VLM depth (same default as the action expert). |
| `lewm_inject_to=`**`"none"`** | — | New value. Encoder is loaded but no tokens flow into the action expert. Used to isolate the Latent Goal Expert as the only le-wm pathway into the training signal. |

`observation_delta_indices` is now a property: returns `[0, chunk_size]`
when Latent Goal Expert is enabled (so the dataset delivers both the anchor and chunk-end
frames per sample), `[0]` otherwise.

### Code touchpoints

| Path | Status | Purpose |
|---|---|---|
| `src/lerobot/policies/sawseenvlawm/smolvlm_with_two_experts.py` | new | `SmolVLMWithTwoExpertsModel`. Adds `latent_goal_expert` + multi-stream cross-attn dispatch. |
| `src/lerobot/policies/sawseenvlawm/configuration_sawseenvlawm.py` | edit | Six new fields; `observation_delta_indices` returns `[0, chunk_size]` when Latent Goal Expert is on; `lewm_inject_to="none"` allowed. |
| `src/lerobot/policies/sawseenvlawm/modeling_sawseenvlawm.py` | edit | Picks `SmolVLMWithTwoExpertsModel` over `SmolVLMWithExpertModel` when Latent Goal Expert is on. New projections (`latent_goal_in_proj`, `latent_goal_anchor_proj`, `latent_goal_time_mlp_*`, `latent_goal_out_proj`). New methods `embed_latent_goal_suffix`, `_encode_lewm_cls`, `prepare_chunk_end_images`. Latent Goal Expert branch in `VLAFlowMatching.forward()` returns `(action_losses, latent_goal_loss)`. Outer policy `forward()` combines them and surfaces `loss_action` / `loss_latent_goal` in `loss_dict`. |
| `sawseenvlawm.mk` | edit | New `LATENT_GOAL`, `LATENT_GOAL_LOSS_WEIGHT` knobs; passes `--policy.latent_goal_enabled` and `--policy.latent_goal_loss_weight` to `lerobot-train`. |

### Loss and logging

The combined loss `L = L_action + λ · L_fs` is what the optimizer sees.
Three TB scalars surface every log step:

| TB scalar | What it is |
|---|---|
| `train/loss_action` | Action expert flow-matching MSE alone |
| `train/loss_latent_goal` | Latent Goal Expert flow-matching MSE alone |
| `train/loss` | Combined optimizer loss |

This works automatically because `loss_dict` is merged into the
`output_dict` returned from `policy.forward()`, and the training loop
already feeds `output_dict` into the TB logger.

### What Latent Goal Expert is *not* doing yet (Phase A boundary)

- **No inference path.** `sample_actions` calls the wrapper with
  `inputs_embeds=[prefix, action_suffix]` (length 2), which falls back to
  the parent's single-expert path. Latent Goal Expert is silent at inference.
- **No MPC inner loop.** K-perturbation + WM rollout + argmin is Phase B.
- **No le-wm JEPA predictor.** Only the encoder is loaded — the
  forward-dynamics predictor lands in Phase B.
- **No contrastive loss.** `latent_goal_loss_type="bc"` is the only
  path; flow-matching MSE against the encoded chunk-end.
- **No distillation.** Mode-3 in the synthesis doc is Phase D.

### Validation gate

| Run | Status | Result |
|---|---|---|
| 4-step smoke (bs=2, 1-token Latent Goal Expert suffix) | done | `loss_latent_goal` 2.41 → 1.96 → 1.88; gradients flow into both experts |
| 4-step smoke (bs=2, 2-token Latent Goal Expert suffix + action-blind mask) | pending — GPU contention | architecture verified by import test; smoke pending GPU availability |
| Long Latent Goal Expert-only ablation (`Latent Goal Expert=true LEWM_INJECT_TO=none`) | pending | clean isolation of Latent Goal Expert's effect on action loss |
| Latent Goal Expert retrieval probe at chunk-end checkpoint | pending | held-out cosine sim of Latent Goal Expert-decoded `z_g` vs actual encoded chunk-end frame should beat random other-episode chunk-end frames |

### Recommended ablation matrix

| Run | LEWM_INJECT_TO | Latent Goal Expert | What it isolates |
|---|---|---|---|
| Baseline (vanilla sawseenvla) | n/a | false | Vanilla reference |
| Side-channel only (existing parked result) | suffix (k=1) | false | The parked ablation |
| **Latent Goal Expert-only (target)** | **none** | **true** | **Pure Latent Goal Expert effect on action loss** |
| Latent Goal Expert + side-channel | suffix (k=1) | true | Stacked uplift (if any) |

The clean experiment is "Latent Goal Expert-only" vs "Baseline" — both have the same
information available to the action expert (prefix only, no le-wm
side-channel), the only difference being whether Latent Goal Expert is trained as a
joint auxiliary head.

### Run command

```bash
# Latent Goal Expert-only Phase A ablation:
make -f sawseenvlawm.mk train \
  LATENT_GOAL=true LEWM_INJECT_TO=none \
  STEPS=8000 BATCH_SIZE=64 \
  OUTPUT_DIR=outputs/train/sawseenvlawm_libero_latent_goal_only_8k
```

`BATCH_SIZE` may need to drop from 96 → 64 to fit the Latent Goal Expert's ~98M
extra params on 24 GB cards. Watch `nvidia-smi` during the first 200
steps and bump up if there's headroom.

---

## Mode 3 — Latent Goal Expert-conditioned action expert

A third pathway: feed the LGE's predicted goal latent **directly into
the action expert's suffix** as conditioning, instead of using it only
as a Phase B MPC scoring target. The action expert's suffix is
prepended with two tokens `[z_t, z_g]` (192-d le-wm latents projected
to expert hidden), with cumulative attention `[1, 0]` so they form one
bidirectional block before the causal action chunk.

### Why this is a different bet

- The parked side-channel ablation showed that adding *another visual
  stream* to the action expert doesn't move action loss. Mode 3 is a
  different hypothesis: not "more vision," but "give the action expert
  a goal cue in le-wm geometry that the language prefix doesn't
  already encode."
- Re-uses the LGE head we already trained for Phase A. No new losses
  or training schedule — just a new pathway through the suffix.
- End-to-end differentiable (no MPC inner loop needed).

### Architecture — 3-pass training, sequential by construction

Because the action expert needs LGE's *output* as input, the two
experts can no longer run in parallel through the shared backbone.
Training switches to three sequential calls into
`SmolVLMWithTwoExpertsModel`, all sharing one VLM K/V cache:

```
Pass 1: inputs_embeds=[prefix, None, None]   fill_kv_cache=True   → builds VLM K/V
Pass 2: inputs_embeds=[None, None, lge]      fill_kv_cache=False  → emits v_lge → L_lge
        z_g_pred = lge_x_t - t · v_lge     (clean prediction reconstructed via FM identity)
Pass 3: inputs_embeds=[None, action+inject, None]  fill_kv_cache=False  → action loss
        action_suffix = [proj(z_t), proj(z_g_pred).detach(), lewm?, action_chunk]
```

The cache is read-only after Pass 1 (`fill_kv_cache=False` in 2/3) so
the action expert and LGE never see each other's K/V. The
`.detach()` on `z_g` (and `z_t`) into the action expert is the
KI-style barrier: action loss cannot reshape LGE weights through the
conditioning path; LGE remains supervised purely by `L_lge`. Both
losses still flow into the VLM via their independent cross-attentions.

### Inference — K-step LGE denoise + K-step action denoise

```
1.  prefix forward → cache  (one VLM pass, identical to Phase A)
2.  K-step LGE flow-matching denoising on top of the cache → clean z_g
3.  K-step action denoising loop, with [z_t, z_g] tokens fixed across steps
```

The marginal cost on top of Phase A is `K · (LGE forward)`. With
`latent_goal_num_steps=10` and the LGE expert at width 720 / 16 layers,
this roughly matches the action expert's cost — wall-clock latency
~2× Phase A.

### Config surface (Mode 3)

| Field | Default | Notes |
|---|---|---|
| `latent_goal_inject_to_action` | `False` | Master switch. Off → bit-identical to Phase A. |
| `latent_goal_inject_z_g_source` | `"encoded"` | `"encoded"` = train on dataset's chunk-end CLS (clean target, but train≠eval since eval uses LGE's denoised output). `"predicted"` = train on LGE's reconstructed clean prediction `z_g_pred = x_t - t·v` (matches inference distribution; noisier early). `"scheduled"` = per-sample Bernoulli mix that ramps from 100% encoded (teacher) at step 0 to 100% predicted (student) at `latent_goal_inject_schedule_end_step` — closes the train/eval gap gradually. |
| `latent_goal_inject_schedule_end_step` | `0` | Required when source=`"scheduled"`. Step at which the schedule reaches 100% predicted (linear ramp from step 0). Typical: equal to `scheduler_decay_steps`. |
| `latent_goal_inject_detach` | `True` | Detach `z_g` (and `z_t`) before the action expert reads them. False makes the conditioning path differentiable so action loss also reshapes LGE — collapses goal latent toward "whatever helps the policy." |

Requires `latent_goal_enabled=True` (validated in `__post_init__`).

### Scheduled-sampling mode (`source="scheduled"`)

Mixes the two sources via per-sample Bernoulli — at step `s`:

```
p          = clamp(s / latent_goal_inject_schedule_end_step, 0, 1)
mask_i     ~ Bernoulli(p)          (i.i.d. per sample in the batch)
z_g_i      = z_g_predicted_i  if mask_i else z_g_target_i
```

Each sample sees one *real* source (no soft interpolation that the
action expert wasn't trained to handle). Standard scheduled-sampling
recipe from seq2seq, applied to z_g.

Why it helps:
- Early in training the LGE head is undertrained — its `z_g_predicted`
  is noise. `encoded` gives the action expert a clean teacher signal
  so it can learn the conditioning shape.
- Late in training LGE is well-fit and `encoded` is unavailable at
  eval. `predicted` matches the eval z_g distribution exactly.
- The linear ramp gives the action expert a curriculum to gradually
  tolerate the LGE's noise — never a sharp distribution shift.

Plumbing: a `_train_step` buffer on the inner model is incremented by
`SawSeenVLAWMPolicy.update()` (called by the training loop after each
optimizer step). The buffer is persistent, so resuming training picks
the schedule up where it left off. The current fraction is logged each
step as `latent_goal_schedule_p`.

### Run command

```bash
# Mode 3 training: predicted z_g, paper-faithful detach, isolated LGE
# (no le-wm side-channel into the action expert):
make -f sawseenvlawm.mk train \
  LATENT_GOAL=true LATENT_GOAL_INJECT_TO_ACTION=true \
  LATENT_GOAL_INJECT_Z_G_SOURCE=predicted LATENT_GOAL_INJECT_DETACH=true \
  LEWM_INJECT_TO=none \
  STEPS=8000 BATCH_SIZE=64 \
  OUTPUT_DIR=outputs/train/sawseenvlawm_libero_mode3_predicted_8k
```

### What can go wrong

1. **Anchor token unused under pure cross-attn.** If
   `self_attn_every_n_layers=-1` is forced, the LGE's anchor and
   denoise tokens stop seeing each other (cross-attn-only means each
   expert token reads only the VLM prefix, not its own siblings).
   Mode 3 keeps the default mixed-attention regime, so LGE's
   bidirectional anchor↔denoise still works in the self-attn-every-2
   layers.
2. **Train/eval z_g mismatch with `source="encoded"`.** Training sees
   the dataset's exact chunk-end CLS; eval sees LGE's K-step denoised
   output. If LGE is undertrained, eval z_g is off-distribution from
   what the action expert learned to condition on.
3. **`detach=False` collapses the goal.** Action loss gradient through
   `z_g` reshapes LGE to output "whatever helps the policy," and the
   chunk-end-target supervision can't compete. The Phase A
   action-blindness rationale assumed detach (or no path at all); only
   flip to `False` deliberately and watch `loss_lge` for collapse.

## Phase B — MPC inference with le-wm predictor

A runtime-only addition (no training change): rescore the action expert's
output against the LGE goal using le-wm's forward dynamics. The policy
already produces a clean *anchor* chunk; MPC samples perturbations
around the anchor, rolls each through the le-wm predictor in latent
space, and picks the candidate whose terminal latent is closest to the
LGE-predicted goal.

### Why now / what it gives us

- All three required components are already on disk: encoder + projector
  + action_encoder + predictor + pred_proj live in the same le-wm
  `<name>_object.ckpt` we use for the encoder (the file is a pickled
  `JEPA` module, not just a state-dict).
- We already have a goal in LGE (`_latent_goal_denoise` returns a clean
  `z_g`). MPC turns LGE from "extra training signal" into a runtime
  filter on the policy's own samples.
- No retraining, no new losses. Failure mode is graceful: if MPC scoring
  is uncalibrated, candidate 0 = anchor wins, behavior equals current
  policy.

### Two schemes (both anchor-based)

Both schemes perturb around the policy's clean anchor `a*`, NOT around
random noise. The anchor is the action expert's deterministic
flow-matching output (same code path as today). This gives MPC a strong
prior — perturbations explore the policy's neighborhood, not the whole
action space.

```
Scheme A (single-shot, recommended for v1):
   a*       ← anchor from one full denoising of the action expert (B, T, A)
   ε_k      ← N candidates of additive Gaussian noise, k=1..N-1
   a_k      = a* + σ ⊙ ε_k       (candidate 0 = a* itself, σ from config)
   ẑ_k      ← rollout(z_t, a_k) via le-wm predictor → (B, N, 192)
   k*       = argmin_k ‖ẑ_k − z_g‖² in post-projector space
   return a_{k*}

Scheme B (CEM, comparison knob):
   μ_0      ← a*    (anchor as initial mean)
   σ_0      ← σ_init (config; per-dim or scalar)
   for m in 1..M:
       a_k  ← μ_{m-1} + σ_{m-1} ⊙ ε_k,  k=1..N
       cost ← ‖rollout(z_t, a_k) − z_g‖²
       top  ← top-K_elite candidates by cost
       μ_m  = top.mean();  σ_m = top.std()   (optional EMA toward prior)
   return arg-min cost from final iter
```

Scheme A is one CEM iter (M=1) with no Gaussian update. Implementation
shares the same inner kernel:

```
def _mpc_score(z_t_emb, z_g_emb, candidates_actions):
    """candidates_actions: (B*N, T, action_dim). Returns (B, N) cost."""
    return rollout_and_compare(...)
```

### Anchor + perturbations (Scheme A) — step-by-step

```
1. Build VLM prefix once: prefix_embs, cache ← vlm_with_expert(prefix, fill_kv=True)
2. z_t_emb  ← lewm_encoder.encode_cls(o_t)                            (B, 192) — projector(CLS)
3. z_g_emb  ← _latent_goal_denoise(z_t_emb, prefix_pad, cache)        (B, 192) — LGE trained
                                                                       # in projector space, so
                                                                       # this is already aligned.
4. Mode 3 inject_tokens = [proj_zt(z_t_emb), proj_zg(z_g_emb)]        (B, 2, H_act)
5. ── anchor: standard flow-matching denoising (10 steps), single batch B
       a* = sample_actions_core(...)   shape (B, T, A_raw_padded)
       a*_raw = a*[:, :, :action_dim]    shape (B, T, A_raw)
6. ── perturbations
       ε     ~ N(0, I) shape (B, N-1, T, A_raw)
       a_k   = a*_raw[:, None] + σ * ε  shape (B, N, T, A_raw)   (k=0 is a* itself)
7. ── le-wm rollout, batch (B*N) — both z_t_emb and z_g_emb already in the
       predictor's training space (post-projector); no extra projection needed.
       hist_emb = z_t_emb[:, None, :].expand(-1, HS=3, -1)            (B*N, 3, 192)
       hist_act = a_k[:, :3, :]                                        (B*N, 3, A_raw)
       for t in 0..T-2:
            act_emb = lewm_world.action_encoder(hist_act[:, -3:])      (B*N, 3, 192)
            pred    = lewm_world.predict(hist_emb[:, -3:], act_emb)[:, -1:]
            hist_emb = cat([hist_emb, pred], dim=1)
            hist_act = cat([hist_act, a_k[:, t+3:t+4, :]], dim=1)
       ẑ = hist_emb[:, -1]                                              (B*N, 192)
9. ── score and pick
       cost = ((ẑ - z_g_emb_expanded)**2).sum(-1).view(B, N)
       best = cost.argmin(dim=1)                                        (B,)
       return torch.stack([a_k[b, best[b]] for b in range(B)])           (B, T, A_raw)
```

The key engineering moves:
- **Anchor reuse**: the existing `sample_actions` body produces `a*`
  with **no replication** (batch B), so VLM prefix and action denoising
  stay at their current cost.
- **Predictor rollout is cheap**: ARPredictor is depth-6 / 192-d. Even
  at B*N=64, 9 rollout steps take << 1 action-expert step on the same
  batch. The dominant marginal cost is the rollout, not the policy.
- **History init**: we have one frame (`o_t`). le-wm trained with
  history_size=3, so we repeat `z_t_emb` 3× to fill the context window.
  This makes the predictor see `(z_t, z_t, z_t)` initially — the
  "no-motion" prior. After 3 rollout steps the window is fully
  predicted; the first 2 outputs are slightly OOD but discarded (we
  only use the final emb).

### CEM scheme (Scheme B) — what changes

Same kernel, wrapped in a Gaussian-fit outer loop:

```
μ, σ = a*_raw, σ_init
for m in range(num_iter):
    a_k  = μ[:, None] + σ * randn(B, N, T, A_raw)
    cost = _mpc_score(z_t_emb, z_g_emb, a_k)         # (B, N)
    elite_idx = cost.topk(K_elite, dim=1, largest=False).indices
    elite_a   = gather(a_k, elite_idx)                # (B, K_elite, T, A_raw)
    μ = elite_a.mean(dim=1)
    σ = elite_a.std(dim=1) * (1 - α) + σ_init * α     # optional anchoring
return _mpc_score on final samples, pick argmin
```

For comparable wall clock, set `num_iter * N_per_iter ≈ N` of Scheme A
(e.g. Scheme A: N=32; Scheme B: 4 iters × 8 candidates).

### Config surface (Phase B / MPC)

| Field | Default | Notes |
|---|---|---|
| `mpc_enabled` | `False` | Master switch. Off → `sample_actions` returns the anchor directly (current behavior). |
| `mpc_scheme` | `"anchor_perturb"` | `"anchor_perturb"` (Scheme A, single-shot) or `"cem"` (Scheme B). |
| `mpc_num_candidates` | `16` | N. Includes the anchor itself as candidate 0 in Scheme A. |
| `mpc_noise_scale` | `0.1` | σ on perturbations. In normalized-action units; 0.1 ≈ 1/10 of action std. |
| `mpc_cem_num_iter` | `4` | Scheme B only. Outer CEM iterations. |
| `mpc_cem_topk` | `4` | Scheme B only. Elite set per iter. |
| `mpc_cem_anchor_blend` | `0.5` | Scheme B only. σ-anchoring weight toward `σ_init` (1.0 = pure init, 0.0 = drift freely). |
| `mpc_predictor_path` | inherits `lewm_encoder_path` | Path to the le-wm `<name>_object.ckpt`. Same pickle holds encoder + projector + predictor; we load the full module at policy construction when `mpc_enabled=True`. |

The relevant validations in `__post_init__`:

```python
if self.mpc_enabled:
    if not self.latent_goal_enabled:
        raise ValueError("mpc_enabled=True requires latent_goal_enabled=True (z_g supplier)")
    if not self.lewm_encoder_path and not self.mpc_predictor_path:
        raise ValueError("mpc_enabled=True requires a le-wm checkpoint (lewm_encoder_path or mpc_predictor_path)")
    if self.mpc_num_candidates < 2:
        raise ValueError("mpc_num_candidates must be >= 2 (anchor + at least one perturbation)")
    if self.mpc_scheme not in ("anchor_perturb", "cem"):
        raise ValueError(...)
```

### Code touchpoints

1. **`lewm_encoder.py`** — add `LeWMWorldModel` class that wraps the
   pickled `JEPA` (encoder + projector + action_encoder + predictor +
   pred_proj). Expose `cls(images) → (B, 192)`,
   `project(cls) → (B, 192)`, `predict_emb(emb, act_emb) → (B, T, 192)`.
   Frozen by default. The existing `LeWMVisionEncoder` becomes a thin
   wrapper around `LeWMWorldModel.encoder` when `mpc_enabled=False`, so
   only one ViT lives in memory either way.

2. **`modeling_sawseenvlawm.py`** — add:
   - `self.lewm_world: LeWMWorldModel | None = None` constructed when
     `mpc_enabled=True`.
   - `mpc_sample_actions(images, img_masks, lang_tokens, lang_masks, state, ...)`
     orchestrating the anchor → perturb → rollout → score → argmin
     flow described above.
   - Branch in `_get_action_chunk`:
     `if self.config.mpc_enabled: actions = self.model.mpc_sample_actions(...)`
     else fall through to `sample_actions` as today.
   - `_lewm_rollout_score(...)` private helper, shared between Scheme A
     and Scheme B.

3. **`configuration_sawseenvlawm.py`** — add the fields above with the
   validations.

4. **`sawseenvlawm.mk`** — add the new MPC knobs as `?=`-overridable
   variables; new eval target `eval-mpc` that flips `mpc_enabled=true`
   for an existing trained checkpoint (no retraining).

### Compute budget vs Phase A inference

| Component | Phase A (current) | MPC v1 (Scheme A, N=16) |
|---|---|---|
| VLM prefix | 1× | 1× |
| LGE denoise (K=10) | 10× LGE forward | 10× LGE forward |
| Action denoise (10 steps) | 10× action forward, batch B | 10× action forward, batch B (anchor only) |
| le-wm rollout | — | 9× ARPredictor forward, batch B·N |
| le-wm encode (z_t) | reused — `lewm_encoder.encode_cls` already applies the projector | reused, no new cost |

ARPredictor is 5 orders of magnitude smaller than the action expert
forward per token. Expected wall-clock overhead at N=16: **~10–20%**
beyond Phase A. At N=64, **~30–50%**. Memory: O(N) on the rollout
embeddings (192-d × T × N) only — negligible compared to action-expert
activations.

### Validation plan

1. **Smoke test** (bs=2, N=4, num_iter=1) — verify shapes flow through
   the new path, no NaN, output `(2, T, action_dim)`.
2. **Calibration probe** — on 50 libero episodes: for each step, log
   `(cost_anchor, cost_best_perturbation, anchor_chosen_share)`. If MPC
   chooses anchor >90% of the time, σ is too small or LGE is too
   uncertain to discriminate; if <10%, anchor is bad and we should
   sanity-check the predictor calibration.
3. **A/B vs Mode 3** — same checkpoint, eval Phase A vs MPC-Scheme-A vs
   MPC-Scheme-B on libero_10. Pass criterion: at least one MPC variant
   ≥ Phase A on overall success rate; full table of per-suite deltas.
4. **σ sweep** — N=16, σ ∈ {0.05, 0.1, 0.2, 0.4}. Plot success vs σ.

### Risks and open questions

| Risk | Notes / mitigation |
|---|---|
| **Action-norm gap (accepted).** sawseenvlawm normalizes actions per LeRobotDataset; le-wm uses its own per-column StandardScaler. Magnitudes likely close on libero (both near zero-mean unit-var) but not identical. | v1 measures the gap empirically via the calibration probe. If `cost_anchor` drifts with action magnitude, add a renormalizer in v2. |
| **Predictor / projector mode drift.** `projector` and `pred_proj` are BatchNorm-bearing MLPs. If the frozen LeWM submodules slip into `.train()` mode during SawSeenVLAWM fine-tuning, running stats get corrupted from LIBERO batches and the LGE / MPC feature distribution drifts. | `LeWMVisionEncoder.train()` and `LeWMWorldModel.train()` override propagation: when `freeze=True`, force the ViT + projector (+ predictor + pred_proj on the world model) back to `.eval()` regardless of parent mode. |
| **History-init (z_t, z_t, z_t) is OOD for predictor.** Predictor trained on real 3-frame sequences, never on repeated frames. | First 2 rollout outputs are discarded (we score only the final emb), so the impact is bounded but non-zero. If costs are noisy, switch to history_size=1 or pad with a learned token. |
| **Candidates collapse around anchor.** If σ is too small or the policy is over-confident, all candidates produce nearly the same `ẑ` and MPC is a no-op. | σ sweep in validation. Also: include a "stochastic flow" mode that re-runs partial denoising from a higher t to broaden the distribution (v2). |
| **Anchor 'always wins'.** Calibration may favor the anchor's trajectory because the predictor sees in-distribution actions only for the anchor. Perturbations push actions slightly off-policy, which might *increase* predicted-state error without changing real-world outcome. | Use σ small enough that perturbations stay near the action manifold. The argmin-against-z_g objective should still favor genuinely-better candidates when they exist. |
| **Throughput hit on libero eval (1024 envs).** N=16 expands the action-expert batch implicitly via rollout only, not via VLM/action-expert. Should be fine, but worth a libero throughput probe before launching a full eval. | Profile MPC at bs=8 and bs=32 before launching a 1024-env eval. |

### v1 → v2 roadmap

- v1 (this design): Scheme A + Scheme B selectable, anchor-based, post-projector scoring (LGE + z_t + predictor all live in the same space — see "LeWM projector wiring" below), accepted action-norm gap.
- v2 candidates (only if v1 shows promise):
  - Load le-wm action stats and re-normalize.
  - Receding-horizon execute (re-plan every k<chunk_size steps).
  - Stochastic flow for broader sampling (SDE flow / partial denoising).
  - Multi-step LGE goal (predict z at multiple horizons, score with weighted MSE).

### LeWM projector wiring (canonical embedding space)

LeWM's encoder produces a raw 192-d CLS token; its `projector` MLP
(192 → 2048 → 192 with BatchNorm) maps CLS into the JEPA *prediction
space* — the space where LeWM's predictor was supervised to land
(via `pred_proj`). The sawseenvlawm pipeline routes every "scalar
latent for an image" call through `lewm_encoder.encode_cls()`, which
applies the projector. That means:

- **LGE training target** `z_g_target = encode_cls(o_{t+chunk_size})`
  is in projector space.
- **LGE anchor** `z_t_anchor = encode_cls(o_t)` is in projector
  space.
- **Mode 3 inject tokens** `[z_t, z_g]` are both in projector space.
- **MPC scoring** compares the predictor's post-`pred_proj` rollout
  output against the LGE-produced (projector-space) `z_g`. Both
  supervised to land in the same target manifold during JEPA training,
  so they're directly comparable in MSE.

The lewm side-channel (`compute_lewm_tokens`, prepending raw ViT
patches + CLS to the action expert's suffix) is the **only** path
that keeps unprojected features — the projector was trained on CLS
only, so applying it to patch tokens would be OOD. The trainable
`lewm_proj` in `embed_suffix` handles the dynamic-range gap on that
path.

**Checkpoint compatibility.** Checkpoints trained before this change
have LGE weights tuned to pre-projector CLS targets; loading them
with the new pipeline produces semantically wrong z_g (predicted in
the wrong space). There is no auto-migration — those checkpoints
should be retrained.

