# SawSeenVLA — SmolVLA clone, LIBERO recipe

`sawseenvla` is a structural clone of `smolvla` registered as a separate policy
(`src/lerobot/policies/sawseenvla/`). Same SmolVLM2-500M backbone, same
flow-matching action expert, identical defaults — but its own choice key,
config class, and policy class so it can be fine-tuned and modified
independently of the upstream SmolVLA.

End-to-end recipe for fine-tuning on LIBERO and matching the SmolVLA paper
numbers ([arxiv 2506.01844](https://arxiv.org/abs/2506.01844), Section 4.3,
Table 2). Because SawSeenVLA is architecturally identical to SmolVLA, the paper
recipe applies verbatim — its target numbers are the SmolVLA paper's numbers.

All commands run inside the LIBERO benchmark Docker image.
[`sawseenvla.mk`](./sawseenvla.mk) is the orchestration entrypoint — it's a
clone of [`smolvla.mk`](./smolvla.mk) with `--policy.type=sawseenvla` baked in
and sawseenvla-namespaced output paths. Both Makefiles use the same
`lerobot-benchmark-libero` Docker image, so `make -f sawseenvla.mk build` is
interchangeable with `make -f smolvla.mk build`.

## What's different from SmolVLA

| | SmolVLA | SawSeenVLA |
| --- | --- | --- |
| Choice key (`--policy.type`) | `smolvla` | `sawseenvla` |
| Config class | `SmolVLAConfig` | `SawSeenVLAConfig` |
| Policy class | `SmolVLAPolicy` | `SawSeenVLAPolicy` |
| Backbone wrapper | `SmolVLMWithExpertModel` | imported from smolvla (shared) |
| Defaults / hparams | — | identical |
| Hub checkpoints | `lerobot/smolvla_base`, etc. | none — train your own |

> Hub checkpoints saved as `type: smolvla` will **not** load via
> `--policy.type=sawseenvla` — the choice keys differ. Bootstrap SawSeenVLA
> from the SmolVLM2 backbone (`--policy.load_vlm_weights=true`) to produce
> SawSeenVLA-native checkpoints.

## Paper recipe (Sec. 4.3)

| | paper | `sawseenvla.mk` variable |
| --- | --- | --- |
| Steps | 100 000 | `STEPS` |
| Global batch | 64 | `BATCH_SIZE` |
| Optimizer | AdamW, β=(0.9, 0.95) | (sawseenvla default — same as smolvla) |
| LR schedule | cosine, 1e-4 → 2.5e-6, 1k warmup | (sawseenvla default; `LR` overrides peak) |
| Precision | bf16 | `MIXED_PRECISION=bf16` (default in `sawseenvla.mk`) |
| Image resize | 512×512 | (sawseenvla default) |
| Chunk size | 50 | (sawseenvla default) |
| `n_action_steps` (inference) | 10 | `EVAL_N_ACTION_STEPS=10` |
| VLM backbone | frozen | `freeze_vision_encoder=True` (default) |
| Action expert | trained | `train_expert_only=True` (default) |
| Dataset | `HuggingFaceVLA/libero` (1693 ep.) | `DATASET_REPO` |
| Eval protocol | 10 trials × 40 tasks (4 suites × 10) | `EVAL_TASKS` + `EVAL_EPISODES=10` |

## Target numbers (SmolVLA Table 2, 0.45B `smolvla_base` row)

Architecture is identical, so these are the same numbers SawSeenVLA aims for.

| Spatial | Object | Goal | Long | Average |
| --- | --- | --- | --- | --- |
| **90** | **96** | **92** | **71** | **87.3** |

## Run it

### Paper-exact (2× 24 GB GPUs, ~4 h)

```bash
make -f sawseenvla.mk train \
  OUTPUT_DIR=outputs/train/sawseenvla_libero_paper \
  STEPS=100000 BATCH_SIZE=64 LR=1e-4 \
  NUM_GPUS=2 MIXED_PRECISION=bf16

make -f sawseenvla.mk eval \
  EVAL_POLICY=outputs/train/sawseenvla_libero_paper/checkpoints/last/pretrained_model \
  EVAL_N_ACTION_STEPS=10

make -f sawseenvla.mk table TABLE_LABEL="SawSeenVLA (paper recipe)"
```

### Single 24 GB GPU fallback

Halve the global batch and scale LR by ~√2:

```bash
make -f sawseenvla.mk train \
  OUTPUT_DIR=outputs/train/sawseenvla_libero_paper_1gpu \
  STEPS=100000 BATCH_SIZE=32 LR=0.7e-4 \
  NUM_GPUS=1 MIXED_PRECISION=bf16
```

Sees half the samples per step; either accept the gap or double `STEPS` to 200k.

### Defaults shipped in `sawseenvla.mk`

The committed defaults aim at a longer 384k-step run, not the paper-exact 100k.
Override on the command line as shown above to reproduce the paper recipe.

| Variable | Default | Paper recipe |
| --- | --- | --- |
| `STEPS` | `384000` | `100000` |
| `BATCH_SIZE` | `16` | `64` |
| `LR` | `2.0e-4` | `1.0e-4` |
| `MIXED_PRECISION` | `bf16` | `bf16` |
| `SAVE_FREQ` | `1000` | (any) |
| `LOG_FREQ` | `200` | (any) |
| `OUTPUT_DIR` | `outputs/train/sawseenvla_libero_test` | choose your own |
| `JOB_NAME` | `sawseenvla_libero` | choose your own |

## Notes

- **Architectural parity.** `sawseenvla` and `smolvla` share the SmolVLM2-500M
  backbone wrapper (`SmolVLMWithExpertModel`) — `modeling_sawseenvla.py`
  imports it from `..smolvla.smolvlm_with_expert`. Same shapes, same defaults,
  same optimizer/scheduler presets. To diverge the backbone too, copy that
  file into `sawseenvla/` and update the import.
- **Cannot load `lerobot/smolvla_base` directly.** That checkpoint serializes
  `"type": "smolvla"`, so `from_pretrained` instantiates `SmolVLAPolicy`, not
  `SawSeenVLAPolicy`. Bootstrap SawSeenVLA with
  `--policy.type=sawseenvla --policy.load_vlm_weights=true` — this loads the
  SmolVLM2 backbone fresh from HuggingFace and trains a fresh action expert.
- **Dataset version matters.** The on-Hub `lerobot/smolvla_libero` checkpoint
  was trained on the older `lerobot/libero` dataset (3 cameras, 6-D state) and
  is **incompatible** with today's `--env.type=libero` (2 cameras, 8-D state).
  Always train against `HuggingFaceVLA/libero`.
- **`n_action_steps=10` is an inference-time override**, not a training-time
  setting. It tells the env to consume only the first 10 actions from each
  50-step chunk before re-querying the policy. Pi0.5's recipe in
  `docs/source/libero.mdx:165` uses the same value.
- **Image rendering is on GPU** via EGL (`MUJOCO_GL=egl`). The eval bottleneck
  is CPU-bound MuJoCo physics, not rendering — see `EVAL_BATCH` ×
  `EVAL_PARALLEL` in `sawseenvla.mk` for parallelism knobs.
- **Result row.** `make -f sawseenvla.mk table` reads the most recent
  `outputs/eval/<date>/<run>/eval_info.json` and prints a markdown row matching
  the column order from `docs/source/libero.mdx`. `TABLE_LATEX=1` swaps to a
  `&`-delimited LaTeX row.
