# SawSeenVLA-Qwen — Qwen3.5-0.8B as VL encoder + independent action expert

`sawseenvla_qwen` is a structural rewrite of SawSeenVLA where the VLM
backbone changes from SmolVLM2-500M (vivisected, with the expert as a
clone of its text decoder) to Qwen3.5-0.8B (treated as a black-box
encoder) plus a from-scratch independent action expert.

The architectural shift is the deeper change. SawSeenVLA deeply
interleaves the action expert with the VLM via per-layer shared K/V
cache. SawSeenVLA-Qwen runs Qwen as published and has the expert
cross-attend to selected per-layer hidden states — **Scheme B** below.
No internal surgery on the VLM, no manual layer walks, no kernel
modifications.

This document captures the design decisions and the rationale.

## Why Qwen3.5-0.8B (and not dense Qwen3)

Considered three paths:

- **A. Hybrid Qwen3.5-0.8B (Qwen3-Next derived)** — multimodal native
  (ViT + DeepStack + M-RoPE), 24 layers in `[L,L,L,F]×6` pattern
  (18 Gated DeltaNet + 6 Gated full-attention), partial RoPE 25%, vocab
  248320. Picked: yes.
- **B. Mixed per-layer integration on the same backbone** — expert
  cross-attends only at the 6 full-attention layers, skipping DeltaNet
  layers; required to access internal K/V. Skipped: more surgery for
  marginal gain over Scheme B.
- **C. Dense Qwen3-0.6B + bolted-on SigLIP** — predictable transformer
  wiring, but the SigLIP → Qwen connector starts from random init.
  Vision-language alignment has to be relearned, ~2-3× more compute to
  first useful checkpoint. Skipped after realising the connector
  cold-start cost.

The hidden lesson: the SmolVLA-style "vivisect the VLM" approach was a
*specific optimization* for the SmolVLM2 backbone, not a requirement.
Treating Qwen as a black box plus an independent expert is **less**
code, **less** risk, and leverages Qwen3-VL's pretrained vision
alignment for free.

## Scheme B: per-layer hidden-state cross-attention

```
Qwen3.5-0.8B (frozen, as published)         Action Expert (from scratch)
  [image_1 + image_2 + text] →                [noisy_actions + time]
       Qwen processor + forward                       │ K=10 denoise steps
                  │ output_hidden_states=True         │
                  ▼                                   ▼
  hidden_states tuple (25 elements,            6-layer transformer:
  one per layer + embedding)                   - own self-attn (1D RoPE)
                  │                            - own cross-attn k_proj/v_proj
                  │ pick indices                  per layer (re-projects
                  │ [4, 8, 12, 16, 20, 24]        Qwen hidden into expert
                  ▼                                space; no shared K/V)
  anchors: list[(B, L_prefix, 1024)]          - SwiGLU MLP
                                                      │
                                                      ▼
                                          velocity prediction → flow matching
```

Six anchor layers, one per expert layer. The anchor indices
`[4, 8, 12, 16, 20, 24]` match the output positions of Qwen's
**6 full-attention (GatedAttention) layers** under the `[L,L,L,F]×6`
pattern — the layers whose output is most cleanly composable.

(Index `i` = hidden_states[i] = output of layer i-1; index 24 = the
final layer output, which is what `last_hidden_state` returns.)

## Components

| file | purpose | LOC |
|---|---|---:|
| `configuration_sawseenvla_qwen.py` | dataclass config, registered `"sawseenvla_qwen"` | 120 |
| `qwen_encoder.py` | `QwenEncoder`: loads Qwen3.5-0.8B, runs processor + forward, returns selected hidden states + attention mask | 175 |
| `action_expert.py` | `ActionExpertDecoder` + `ActionExpertLayer` + RMSNorm + SwiGLU + 1D RoPE | 200 |
| `modeling_sawseenvla_qwen.py` | `SawSeenVLAQwenModel` (flow matching), `SawSeenVLAQwenPolicy` (outer wrapper) | 230 |
| `processor_sawseenvla_qwen.py` | pre/post pipeline — skips tokenization (model handles it) | 70 |

Total: ~800 LOC across 5 files. Compared to SawSeenVLA's
`smolvlm_with_expert.py` alone at 575 LOC of vivisected transformer
plumbing.

## Diff vs SawSeenVLA (SmolVLM2)

| dimension | SawSeenVLA | SawSeenVLA-Qwen |
|---|---|---|
| VLM treatment | Cloned + truncated to 16 layers + manually walked | Run as-published, no surgery |
| Expert genealogy | Deep copy of VLM text decoder, scaled width | Independent transformer, designed from scratch |
| Layer count match | Expert mirrors VLM truncation | Expert layer count fully decoupled (default 6) |
| Per-layer K/V sharing | yes — re-projection per cross-attn layer | no — expert reads per-layer **hidden states**, re-projects via its own `k_proj`/`v_proj` |
| RoPE | manual `apply_rope` in wrapper | Qwen handles its own M-RoPE; expert has plain 1D RoPE on action chunk only |
| Attention masks | custom `make_att_2d_masks` block-causal | Qwen's mask passthrough; expert uses MHA `key_padding_mask` |
| Vision tower | SigLIP + connector | Qwen3-VL native ViT + DeepStack |
| State injection | embedded into prefix BEFORE VLM forward | projected and **appended** to each anchor hidden state (post-Qwen) |
| Tokenizer | SmolVLM2 BPE (~49k vocab) | Qwen BPE (248k vocab) |
| Inference cache | per-layer K/V dict | single `hidden_states` tuple cached externally |
| Code in wrapper | ~575 LOC manual per-layer forward | ~80 LOC `qwen()` call |
| Trainable | ~120M (expert + projections) | ~35M (smaller expert) on 888M total |

## Training defaults

- Qwen3.5-0.8B **frozen** (no LoRA in v1; LoRA on the 6 full-attn
  layers is the obvious upgrade)
- Action expert: 6 layers × hidden=512 × n_heads=8 × intermediate=2048
  → 35M params trainable
- Flow matching: Beta(1.5, 1.0) timestep, MSE on velocity (`u_t = noise
  - action`), K=10 denoising steps at inference
- LR 1e-4 default; sqrt-scaled for batch size. With bs=96 / lr=3.46e-4 on a
  single GPU, peak VRAM ~12 GB, throughput ~2.3 s/step

## Known throughput limitation (v1)

The Qwen processor is called **per-sample in Python loops** inside
`QwenEncoder._processor_inputs`. At bs=4 → 0.7 s/step; at bs=96
→ 2.3 s/step (sub-linear scaling). The CPU-bound processor dominates
at large batch.

Fix for v2: precompute pixel_values + image_grid_thw at dataset hydration
time, batch-call processor with shared chat template. Expected → 0.9
s/step at bs=96 (3× speedup).

## Proposed upgrades (ranked)

### Tier 1 — high leverage, cheap

1. **LoRA r=16 on Qwen's 6 full-attn layers** (q_proj, v_proj). Adds
   ~3M params, ~1 GB VRAM during backward. Lets the VLM itself shift
   toward action-context-aware embeddings. Mirrors what sawseenvla does
   for SmolVLM2.
2. **State as a token inside Qwen** (not appended after). Reserve an
   unused token id (or use `inputs_embeds=`) to substitute the
   projected state embedding inline. State participates in Qwen's
   self-attention with images and language. Adds ~60 LOC.
3. **Self-conditioning in flow matching**. Expert takes
   `(x_t, prev_v_t)` where `prev_v_t` is the previous denoising
   step's velocity (or zero on the first step). Free quality boost,
   zero inference overhead. Adds ~40 LOC.

### Tier 2 — medium leverage

4. **Vision token compression via perceiver resampler.** 64 vision
   tokens/cam × 2 cams = 128 visual tokens dominate the cross-attn.
   Learned 32-query resampler → 4× cheaper cross-attn. Adds ~100 LOC.
5. **Cross-attn before self-attn** in expert layer (OpenVLA-OFT,
   Helix style). 5-LOC change. Anecdotally faster convergence on
   conditioned generation.
6. **Bigger expert.** 8 layers × hidden=768 = ~80M trainable params,
   still <10% of Qwen. Config flip + ~0.5 GB VRAM.

### Tier 3 — research-y, defer

7. **K=1 distillation** (consistency model) for 10× inference
   speedup after we have a usable checkpoint.
8. **Multi-anchor cross-attn** with learned mixing weights. More
   flexible than fixed 1:1 layer mapping; adaLN-style.
9. **Discrete action tokens** (FAST / Pi₀.₅-style) — replaces flow
   matching with cross-entropy on quantized actions. Major rewrite.

## What this run is for

The first SawSeenVLA-Qwen run (8k × bs=96 × lr=3.46e-4 spatial-only)
exists to answer:

> **Does Qwen3.5-0.8B as a black-box encoder beat SmolVLM2-vivisected
> on LIBERO spatial, with the cleanest possible architecture?**

If pc_success ≥ 75% (the SawSeenVLA reference), this becomes the new
backbone substrate and all Tier 1 upgrades target Qwen, not SmolVLM2.

If pc_success < 70%, two diagnostics first:
- Is the issue language grounding? Run `<EXPLORE>` token (see
  `INSTRUCTION_LABELS.md`) on the easyx4 mixed set with Qwen.
- Is the issue capacity at small batch? Bump expert to 8L × 768d.

Don't add features before the baseline number lands.

## Related design docs

- [`INSTRUCTION_LABELS.md`](./INSTRUCTION_LABELS.md) — orthogonal
  labelling strategies for mixed-instruction datasets; composes with
  any VLM choice.
- [`SawSeenVLAWM.md`](./SawSeenVLAWM.md) — the le-wm side-channel
  policy; eventually a `sawseenwam_qwen` variant becomes the obvious
  composition.
- [`future-sight-implicit-wm.md`](./future-sight-implicit-wm.md) —
  the LGE / latent-goal-expert design that sits alongside the action
  expert; would slot into Qwen-based policies the same way it slots
  into SawSeenVLA's wrapper.
