# SawSeenVLA — training-speedup notes (current GPU budget)

Notes on getting more training throughput out of the existing 2× consumer GPU
setup (RTX 3090 Ti + RTX 3090, PHB topology, no NVLink, 24 GB each). No new
hardware required.

## Baseline

Run config that produced the numbers below: `STEPS=96000 BATCH_SIZE=64
NUM_GPUS=2 NUM_WORKERS=4 MIXED_PRECISION=bf16 LR=4e-4` (effective batch 128),
post-dataloader-patch (see [`design/dataloader_patch.md`](#) — the
`_build_query_views` change in `dataset_reader.py` that skips PNG decode on
action-delta-timestamp lookups).

| Metric | Value |
| --- | --- |
| Step rate | **~0.83 step/s** |
| Throughput | ~106 samples/s (eff bs 128) |
| GPU utilisation | both ≥99% |
| Host load avg | ~3.3 |
| VRAM used / card | ~12.5 GB / 24 GB (~52%) |
| Wall for 96k steps | ~32 hours |

Bottleneck is now **GPU compute**, not the dataloader. ~50% VRAM and most CPU
cores are idle, so there is room to push the GPUs harder.

## High-impact levers

### 1. Bigger per-GPU batch — **easy win, ~1.3–1.5×**

We're using ~52% of VRAM per card. Push to `BATCH_SIZE=96` (~17 GB) or
`BATCH_SIZE=128` (~22 GB, tight). Fewer kernel launches per sample → better
tensor-core utilisation; smaller DDP overhead per sample.

```bash
make -f sawseenvla.mk train BATCH_SIZE=96  NUM_GPUS=2 LR=5.0e-4 ...
make -f sawseenvla.mk train BATCH_SIZE=128 NUM_GPUS=2 LR=5.7e-4 ...
```

LR scaling: square-root rule with global batch.
`LR ≈ 4e-4 × sqrt(global_batch / 128)`.
- 128 → 192 → ~4.9e-4
- 128 → 256 → ~5.7e-4

Validate with the loss curve in the first ~1 k steps. If loss explodes, drop
LR by 20%.

Tradeoff: tight VRAM at bs128. If any optimiser-state or activation spike
OOMs, fall back to bs96.

### 2. `torch.compile` — **biggest single knob, ~1.3–1.7×**

Sawseenvla's config exposes `compile_model: bool = False` and
`compile_mode: str = "max-autotune"`. Flip it on:

```bash
make -f sawseenvla.mk train ... \
  --policy.compile_model=true \
  --policy.compile_mode=max-autotune
```

Costs 3–5 min compile at startup (negligible for a multi-hour run). Stays
compiled as long as batch shape is fixed and dtype doesn't switch.

Risk: `max-autotune` occasionally hits a misbehaving Triton kernel. If
anything looks wrong, fall back to `--policy.compile_mode=reduce-overhead`,
which also enables CUDA-graph capture (small additional win).

### 3. Confirm Flash-Attention / SDPA is on — **~1.2–1.5× if currently eager**

HF SmolVLM2 should default to `attn_implementation="sdpa"` (Flash-Attention-2
on Ampere with bf16). Verify by grepping the training log for `Using sdpa` or
`Using flash_attention_2`. If you see `eager`, force-set when constructing the
VLM:

```python
# in src/lerobot/policies/sawseenvla/smolvlm_with_expert.py (when wrapping the VLM)
AutoModel.from_pretrained(model_id, attn_implementation="sdpa")
```

Without this, attention layers run the slow vanilla path; switching can save
20–40% on forward at 512×512 image features.

## Medium-impact levers

### 4. CUDA graph capture (via `compile_mode=reduce-overhead`)
Eliminates per-step Python overhead. ~5–10% on top of `compile=True`. Worth
trying if `max-autotune` doesn't compile cleanly.

### 5. Channels-last memory format
`model.to(memory_format=torch.channels_last)` for convs/attention on Ampere
typically gives 5–10%. Requires a 1-line patch in the training script — not
exposed via CLI today. Reserve for after #1–3 are exhausted.

### 6. Reduce inference denoising steps (eval only)
SmolVLA flow-matching uses `num_steps=10` at inference. Dropping to 5 halves
eval wall-clock with usually 1–2 pt accuracy cost. Doesn't affect training
time — only useful if you're iterating on dev/eval cycles.

## Probably not worth it on this rig

- **More dataloader workers** — was needed pre-patch; now `NUM_WORKERS=4` is
  plenty. Going higher steals CPU from launch threads.
- **Pre-decoded memmap dataset** — would have helped pre-patch; current
  dataloader is ~5 ms/sample. Diminishing returns.
- **Smaller `chunk_size`** — speeds training but changes policy semantics and
  breaks comparison to the SmolVLA paper recipe. Reserve for a known-trade
  experiment, not a generic speedup.
- **Gradient accumulation** — helps memory-bound runs; we're not.
- **fp8 / int8 weights** — needs sm89+ (Hopper). 3090 is sm86. Not available.

## Concrete plan for the next run

Stack levers 1 + 2 + (verify 3). Predicted wall-clock: 32 h → **18–20 h** for
the same 96 k steps.

```bash
make -f sawseenvla.mk train \
  STEPS=96000 BATCH_SIZE=96 NUM_GPUS=2 \
  LR=5e-4 MIXED_PRECISION=bf16 \
  --policy.compile_model=true \
  --policy.compile_mode=max-autotune \
  OUTPUT_DIR=outputs/train/sawseenvla_libero_96k_bs96_compile
```

The single number to validate after launch: **the tqdm step rate at ~step
200**. If it's ≥ 1.5 step/s, the speedup is landing as expected. If it's
< 1.0 step/s, something stacked badly — the most common cause is a
torch.compile recompile loop on shape changes; check the log for
`recompiling` messages.

## Validated results (2026-05-09)

Ran 1000-step benches on the actual rig.

**Headline: 1.35× throughput vs bs64 baseline at bs96 + max-autotune +
pad_language_to=max_length**, less than the predicted 1.7–2× but still
meaningful. Lever #1 (bigger batch) carries almost none of the speedup on this
rig — we were already 100% GPU-compute-bound at bs64. Lever #2 (torch.compile)
carries it all, but only after fixing a non-obvious shape-stability
prerequisite that the original analysis missed.

### Results table

| Config | Step rate | Throughput | vs bs64 | Status |
| --- | --- | --- | --- | --- |
| bs64 (ref) | 0.83 step/s | 106 samples/s | 1.00× | ✅ |
| bs96 no-compile | 0.59 step/s | 113 samples/s | 1.07× | ✅ |
| bs96 + max-autotune | — | — | — | ❌ illegal CUDA mem |
| bs96 + default | (recompile loop) | — | — | ❌ ~5–15× slower steady |
| bs96 + reduce-overhead | (recompile loop) | — | — | ❌ same |
| bs96 + reduce-overhead + padmax | 0.74 step/s | 141 samples/s | 1.33× | ✅ |
| **bs96 + max-autotune + padmax** | **0.75 step/s** | **143 samples/s** | **1.35×** | ✅ winner |
| bs128 + max-autotune + padmax | 0.59 step/s | 150 samples/s | 1.42× | ⚠ 23/24 GB tight |

(`padmax` = `pad_language_to=max_length`)

### Key finding: `compile_model` requires `pad_language_to=max_length`

Not flagged in the original analysis. The default `pad_language_to="longest"`
pads each batch's language tokens to the longest in that batch — the shape is
data-dependent. Dynamo treats every new max-length as a fresh shape, breaking
compile in two distinct ways:

- **`max-autotune`**: the autotuner picks a fused mask-construction kernel
  (`triton_spl_fused__to_copy_cat_cumsum_ones_sub_unsqueeze_4` — the
  `make_att_2d_masks` cumsum in `modeling_sawseenvla.py:120`) that crashes on
  its second invocation with `RuntimeError: CUDA driver error: an illegal
  memory access` deep inside `cudagraph_trees.py`. Nothing trains.
- **`default` / `reduce-overhead`**: compile completes cleanly, then a
  recompile spike at step ~25 (~22 s/step for ~10 steps), accompanied by
  `pow_by_natural` sympy guard warnings flooding stderr. If recompiles keep
  firing, average step rate drops 5–15×.

Pinning `pad_language_to=max_length` (fixed pad to `tokenizer_max_length=48`)
freezes the shape: zero recompile spikes, no warnings, clean steady state from
step 25 onward. **This is the single most important knob** — without it, no
compile mode is usable on this model.

### Lever #1 (bigger batch) is essentially decorative here

bs96 alone delivers **+7%** (113 vs 106 samples/s), not the predicted 1.3–1.5×.
At bs64 we were already 100% compute-bound (≥99% GPU util both ranks,
dataloader at ~0.005 s/sample), so bigger batch barely amortizes kernel-launch
overhead.

### Lever #2 (torch.compile) does the heavy lifting (with stable shapes)

bs96 + max-autotune + padmax over bs96 no-compile: **+27%** (143 vs 113
samples/s). Compile warmup: ~10 min for max-autotune, ~3 min for
`reduce-overhead`/`default` — all amortized over a multi-hour run.
`max-autotune` beats `reduce-overhead` by ~1.4% once shapes are stable (143
vs 141 samples/s) — small but real, and worth the extra ~7 min of warmup.

### bs128 is a dead end

+5% throughput over bs96 (150 vs 143), but VRAM hits **23/24 GB (96%)**. Any
activation spike or longer-than-typical episode in batch OOMs. Not worth the
margin lost.

### Eval-time compile is a separate gotcha

The saved policy config carries `compile_model=true` after training.
`from_pretrained` re-triggers compile at eval load, costing ~10 min warmup that
doesn't amortize over libero rollouts (and would recompile on the eval-time
batch shape anyway). Pass `--policy.compile_model=false` at eval — the
`sawseenvla.mk` `eval` target now does this.

### Wall-clock projection for 96k full run

| Run | Wall for 96k steps | Wall to see 12.3 M samples |
| --- | --- | --- |
| bs64 baseline | 32 h | 32 h |
| **bs96 + max-autotune + padmax** | **~36 h** (1.50× more samples) | **~24 h** (1.35× faster) |

### Working config

All optimization flags are now defaults in `sawseenvla.mk`:

```bash
make -f sawseenvla.mk train STEPS=96000
```

(`BATCH_SIZE=96`, `LR=5.0e-4`, `COMPILE_MODEL=true`,
`COMPILE_MODE=max-autotune`, `PAD_LANGUAGE_TO=max_length`,
`MIXED_PRECISION=bf16`, `NUM_GPUS=2`.)

## Don't kill the current 96 k run

The current run will finish in ~hours; let it land so you have a baseline
checkpoint. Start the optimised config from scratch in a separate `OUTPUT_DIR`
so the wall-clock comparison is clean (no resume effects, fresh torch.compile
warmup).

## What "GPU compute bound" really means here

After the dataloader patch, the per-step time is dominated by:

1. SmolVLM2 forward through 16 VLM layers + the action-expert MLP
   (~700 ms/step at bs64 on RTX 3090 Ti).
2. Bf16 backward + fp32 optimiser-state update (~150 ms).
3. NCCL all-reduce of ~400 MB grads through PHB (no NVLink) — ~30 ms.
4. Python / scheduler overhead — ~20 ms.

Bigger batch and `torch.compile` attack #1, the dominant cost. The other
items are too small to matter until #1 shrinks substantially.

---

# Frontier techniques — VLA literature survey

What the recent VLA literature does to train faster, and which ideas
plausibly transfer to sawseenvla on this rig. Sources at the bottom.

## Memory-side levers (free room → bigger effective batch)

### FSDP / ZeRO-3 sharded data parallel

Shards parameters, gradients, and optimiser states across the 2 GPUs. ZeRO
stages: ZeRO-1 (optimiser states only) saves up to 4× memory; ZeRO-2 (+grads)
up to 8×; ZeRO-3 (+params) memory reduction proportional to N GPUs.

lerobot uses HF `accelerate`, which has FSDP support, but it's not enabled
by default. Toggling it via `--fsdp` flags amplifies high-impact lever #1
(bigger per-GPU batch) — sawseenvla currently uses 12 GB / 24 GB, so we
already had headroom; FSDP would let `BATCH_SIZE` go well past 128/proc
without OOM.

**Verdict for this rig**: try after the simple bs96/bs128 + `compile_model`
stack, only if you OOM at higher batches.

### Activation / gradient checkpointing

Discard intermediate activations; recompute them in backward. Trades roughly
+25–30 % compute for ~70 % activation memory savings. Combined with FSDP,
unlocks even larger batches.

**Verdict**: knob to enable only if pure batch growth (lever #1) hits OOM
before throughput peaks.

## Compute-side levers (faster forward/backward at same shape)

### Token pruning — FocusVLA, ADP

At 512×512, the SmolVLM2 backbone produces hundreds of vision tokens per
camera per step; many are background. Two published approaches:

- **FocusVLA** (`arXiv 2603.28740`) — Modality-Cascaded Attention plus
  Focus Attention. Reports **1.5× average training speedup, 5× on
  LIBERO-Spatial** vs VLA-Adapter.
- **ADP — Action-aware Dynamic Pruning** (`github.com/chen7086/VLA-ADP`) —
  training-free, plug-and-play; prunes redundant visual tokens during
  manipulation stages.

**Verdict**: ADP is the cheaper experiment because it's training-free.
FocusVLA needs a fork of `smolvlm_with_expert.py`. Only worth it after
levers #1–3 in the original section are exhausted.

### OFT (Optimised Fine-Tuning) — parallel decoding + L1 regression

OpenVLA-OFT (`arXiv 2502.19645`) reports **25–50× inference speedup** and
**+20 pts success rate** on LIBERO via parallel decoding, action chunking,
and an L1 regression objective (vs the autoregressive token-by-token
default).

Two caveats for sawseenvla:
1. The headline 25–50× is **inference**. Sawseenvla's flow-matching forward
   already produces all action tokens in parallel, so the inference gap is
   smaller.
2. The L1-regression head vs flow-matching MSE is a **training-time**
   choice. Reportedly converges faster in some settings — would need a
   separate experiment to validate on LIBERO + sawseenvla.

**Verdict**: skip unless you're chasing inference latency or willing to
fork the action head for an experiment.

### Smaller image resolution

`smolvla_config.resize_imgs_with_padding` defaults to 512×512. Source images
are 256×256 (the dataset shape). Going from 512 back to 384 or 256 cuts
vision token count by ~44 % / ~75 %, with proportional forward/backward
speedup on the VLM stack.

```bash
make -f sawseenvla.mk train ... --policy.resize_imgs_with_padding="(384, 384)"
```

**Verdict**: easy ablation to run — likely 1.3–1.5× wall-clock speedup, with
a small accuracy cost on LIBERO (low single-digit pts based on related VLA
literature). Worth a 5 k-step head-to-head against the bs96 + compile
baseline.

### Reduced flow-matching denoising steps (inference only)

Sawseenvla's flow-matching uses `num_steps = 10` for sampling at inference.
Drops to 5 halve eval wall-clock for 1–2 pt accuracy cost. Doesn't affect
training time — included here for completeness.

## Training-procedure levers

### LoRA / PEFT

lerobot already supports PEFT (`use_peft=True`, `--peft.method_type=lora`),
and sawseenvla's `_get_default_peft_targets` returns the right regex for
the action expert plus projections. OpenVLA reports **LoRA matches full
fine-tune with 1.4 % of parameters** and 3–4× memory reduction (10–15 hr
on a single A100 for OpenVLA).

For sawseenvla on 2× 3090: LoRA buys you 3–4× memory headroom that you
spend on bigger batches. The artifact is an adapter (not a merged
checkpoint), which matters if you want to share weights or eval against
the SmolVLA paper number on equal terms.

```bash
make -f sawseenvla.mk train BATCH_SIZE=128 NUM_GPUS=2 \
  --policy.use_peft=true \
  --policy.peft.method_type=lora \
  --policy.peft.r=16 \
  ...
```

**Verdict**: run this as a parallel experiment to the full-fine-tune
baseline. It's the highest expected wall-clock speedup of any single
change, with a known artifact tradeoff.

### Data-efficient training (ActionX et al.)

`ActionX` (Frontiers, 2026) reports **+16 % success vs SOTA with
< 100 expert demonstrations**. If your downstream task is narrow,
curating a smaller, high-quality dataset and shortening the run can
dominate any per-step optimisation.

**Verdict**: doesn't apply to LIBERO benchmark runs (fixed dataset),
applies to your own SO-101 work.

## Cross-references — what sawseenvla / lerobot already does (don't reinvent)

- `bf16` (default in `sawseenvla.mk`)
- HF accelerate with mixed precision
- Action chunking via flow matching (parallel by construction)
- Frozen VLM, trainable action expert (`train_expert_only=True`)
- `torch.compile` available behind a flag (`compile_model: bool = False`,
  see lever #2 above)
- PEFT / LoRA support behind `use_peft=true`

## Recommended experiment order

| Order | Experiment | Effort | Expected gain | Notes |
| --- | --- | --- | --- | --- |
| 1 | bs96 + `compile_model=true` (already in main section) | trivial | 1.7–2× | Stack first |
| 2 | LoRA at bs128 or bs256 | low | 1.5–2× extra | Different artifact |
| 3 | Resolution 384×384 ablation | trivial | 1.3–1.5× | Small accuracy cost |
| 4 | FSDP + activation checkpoint at bs256 | medium | Lets #1 push further | Only if #1 OOMs |
| 5 | ADP token pruning | medium-high | 1.3× | Code patch in `smolvlm_with_expert.py` |
| 6 | FocusVLA-style attention | high | up to 5× on LIBERO-Spatial | Significant fork |

Stack #1 + #2 + #3 sequentially: each is independent of the others. Run #4
only as remediation if you hit OOM. #5 / #6 are research experiments, not
straight engineering wins.

## Sources

- [OpenVLA-OFT — Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://openvla-oft.github.io/) — parallel decoding, action chunking, L1 regression, 25–50× inference speedup
- [OpenVLA-OFT arXiv 2502.19645](https://arxiv.org/abs/2502.19645)
- [OpenVLA — arXiv 2406.09246](https://arxiv.org/abs/2406.09246) — base OpenVLA, LoRA training notes
- [Towards Accessible Physical AI: LoRA-Based Fine-Tuning of VLA Models — arXiv 2512.11921](https://arxiv.org/html/2512.11921v1)
- [SmolVLA paper — arXiv 2506.01844](https://arxiv.org/html/2506.01844v1) — bf16 + torch.compile + accelerate
- [HF blog — SmolVLA](https://huggingface.co/blog/smolvla)
- [Pi0 / Pi0.5 fine-tuning guide (EmbodiFlow)](https://io-ai.tech/platform/en/guides/Pipeline/LeRobot/Pi0/) — FSDP, EMA, memory knobs
- [Scaling VLA Model Training on a Budget](https://www.roboticscenter.ai/blog/scaling-vla-training-on-a-budget) — practitioner notes from 100+ fine-tunes
- [PyTorch FSDP intro](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)
- [HF blog — Accelerate with PyTorch FSDP](https://huggingface.co/blog/pytorch-fsdp)
- [Distributed and Efficient Fine-tuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/) — comprehensive ZeRO stages overview
- [FocusVLA — arXiv 2603.28740](https://arxiv.org/html/2603.28740) — token pruning for VLA
- [VLA-ADP (Action-aware Dynamic Pruning)](https://github.com/chen7086/VLA-ADP) — training-free token pruning
- [FASTER — arXiv 2603.19199](https://arxiv.org/html/2603.19199) — flow VLA inference acceleration
- [AsyncVLA — arXiv 2511.14148](https://arxiv.org/abs/2511.14148) — asynchronous flow matching
- [Real-Time Execution of Action Chunking Flow Policies — arXiv 2506.07339](https://arxiv.org/pdf/2506.07339)
- [Activation Checkpointing Mechanics (apxml)](https://apxml.com/courses/distributed-training-pytorch-fsdp/chapter-3-mixed-precision-memory-optimization/activation-checkpointing-mechanics)
- [Pure VLA Models: A Comprehensive Survey — arXiv 2509.19012](https://arxiv.org/html/2509.19012v1)
- [ActionX (Frontiers Neurorobotics, 2026)](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2026.1806605/full) — data-efficient training
