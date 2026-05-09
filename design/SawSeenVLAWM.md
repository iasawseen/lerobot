# SawSeenVLAWM — SawSeenVLA augmented with a le-wm visual side-channel

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

### Why the suffix (Option B) and not the prefix (Option A)

| Concern | Prefix injection | Suffix injection (chosen) |
|---|---|---|
| Direct path to action expert | No — features pass through 16 frozen SmolVLM layers | Yes — action expert reads them directly |
| KV cache efficiency | Cacheable across denoising steps | Re-runs per step, but encoder runs once |
| Risk of being filtered out | High (frozen VLM may not preserve novel features) | Low (trainable cross-attn from action tokens) |
| Code-change footprint | Modify `embed_prefix`, change prefix length | Modify `embed_suffix`, change suffix length |
| Compatibility with KV-cache LRU | Have to invalidate/re-warm | Untouched |

The suffix is the right fit because the question is "what extra signal does
*the action expert* get?", not "what extra signal does the VLM get?".

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
| **Projector dropped:** le-wm's MLP projector saw only CLS during training; applying it to patch tokens would be OOD. We use raw ViT hidden states. | Trade-off: features have higher dynamic range than the post-projector ones. Mitigated by the trainable `lewm_proj`. |
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

Six 1k-step ablations on libero @ bs=24, LR=2.5e-4, with all six configurations of the lewm side-channel. **None beat the no-lewm baseline.**

| variant                          | step rate | GPU mem | step 950 loss |
|----------------------------------|----------:|--------:|--------------:|
| lewm=0 (no encoder)              |   1.62 step/s | 8.5 GB | 0.655 |
| lewm=1 frozen, suffix, per-cam   |   1.55 step/s | 8.5 GB | **0.640** |
| lewm=1 frozen, suffix, concat    |   1.61 step/s | 6.9 GB | 0.642 |
| lewm=1 unfrozen, suffix, per-cam |   1.42 step/s | 8.2 GB | 0.639 |
| lewm=1 frozen, **prefix**, concat |  1.63 step/s | 6.9 GB | 0.639 |
| lewm=192 frozen, suffix, per-cam |   1.10 step/s | 19.3 GB | 0.656 |

Key observations:
* All five lewm variants and the no-lewm baseline cluster within 0.017 of each other at step 950 — well within seed-to-seed noise.
* lewm=192 (patch tokens) is the only consistent loser earlier in training (steps 300–500 it sits at ~1.3 vs ~1.0 for the others). The action expert spends capacity learning to filter the OOD patch features, then catches up by step 950.
* Concat-camera vs per-camera input *did not* matter despite matching le-wm's training distribution exactly — runs are within 0.005 of each other throughout.
* Unfreezing did *not* help — the encoder doesn't learn anything useful in 1k steps.
* Prefix injection vs suffix injection did *not* matter.

### Hypothesis for the negative result (most likely first)

1. **The action expert already has rich vision via SmolVLM/SigLIP** (960-d × 16 layers). A frozen 192-d ViT-Tiny can't add information SmolVLM doesn't already extract.
2. **JEPA next-frame objective ≠ action selection.** Features predictive of "what comes next given an action" aren't necessarily features predictive of "what action to take given a state."
3. **The trainable `lewm_proj` (192→720 or 192→960) likely learns to suppress an unhelpful stream**, leaving the action expert to rely on the prefix attention.

### Status

* All scaffolding (policy class, makefile, encoder wrapper, smoke test) stays in place — vanilla SawSeenVLA is untouched and remains the production policy.
* Re-launching this experiment is one command: `make -f sawseenvlawm.mk train ...`.
* Worth revisiting if/when:
  * le-wm gets retrained on RoboCasa365 (domain match)
  * an adapter-style injection (Option C) is implemented
  * a longer training horizon is run (>5k steps) to test if the encoder helps only late in training
